/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};
use std::cmp;
use std::io::{Read, Write};

use crate::consts::*;
use crate::helpers::{calc_sign_index, err_exit_code, here, u16_bit_length, u32_bit_length};
use crate::lepton_error::ExitCode;
use crate::metrics::{ModelComponent, ModelSubComponent};
use crate::structs::branch::Branch;
use default_boxed::DefaultBoxed;

use super::probability_tables::ProbabilityTables;
use super::quantization_tables::QuantizationTables;
use super::vpx_bool_reader::VPXBoolReader;
use super::vpx_bool_writer::VPXBoolWriter;

const BLOCK_TYPES: usize = 2; // setting this to 3 gives us ~1% savings.. 2/3 from BLOCK_TYPES=2

const NUMERIC_LENGTH_MAX: usize = 12;
pub const MAX_EXPONENT: usize = 11; // range from 0 to 1023 requires 11 bins to describe
const COEF_BITS: usize = MAX_EXPONENT - 1; // the MSB of the value is always 1

const NON_ZERO_7X7_COUNT_BITS: usize = 49_usize.ilog2() as usize + 1;
const NON_ZERO_EDGE_COUNT_BITS: usize = 7_usize.ilog2() as usize + 1;
// 0th bin corresponds to 0 non-zeros and therefore is not used for encoding/decoding.
const NUM_NON_ZERO_7X7_BINS: usize = 9;
const NUM_NON_ZERO_EDGE_BINS: usize = 7;

type NumNonZerosCountsT = [[[Branch; 1 << NON_ZERO_EDGE_COUNT_BITS]; 8]; 8];

const RESIDUAL_THRESHOLD_COUNTS_D1: usize = 1 << (1 + RESIDUAL_NOISE_FLOOR);
const RESIDUAL_THRESHOLD_COUNTS_D2: usize = 1 + RESIDUAL_NOISE_FLOOR;
const RESIDUAL_THRESHOLD_COUNTS_D3: usize = 1 << RESIDUAL_NOISE_FLOOR;

#[derive(DefaultBoxed)]
pub struct Model {
    per_color: [ModelPerColor; BLOCK_TYPES],

    counts_dc: [CountsDC; NUMERIC_LENGTH_MAX],
}

impl Model {
    /// Walks through the model and applies the walker function to each branch
    /// This is used by testing to randomize the model so we can detect
    /// any mismatches in the way that updates are handled.
    ///
    /// This is not used in the normal operation of the codec.
    ///
    /// Note: the order of the branch walking must be maintained between the model and the walker,
    /// otherwise you will break the unit tests.
    #[cfg(test)]
    pub fn walk(&mut self, mut walker: impl FnMut(&mut Branch)) {
        for x in self.per_color.iter_mut() {
            for y in x.num_non_zeros_counts7x7.iter_mut() {
                for z in y.iter_mut() {
                    walker(z);
                }
            }

            for y in x.counts.iter_mut() {
                for z in y.iter_mut() {
                    for w in z.exponent_counts.iter_mut() {
                        for q in w.iter_mut() {
                            walker(q);
                        }
                    }

                    for w in z.residual_noise_counts.iter_mut() {
                        walker(w);
                    }
                }
            }

            for y in x.num_non_zeros_counts1x8.iter_mut() {
                for z in y.iter_mut() {
                    for w in z.iter_mut() {
                        walker(w);
                    }
                }
            }

            for y in x.num_non_zeros_counts8x1.iter_mut() {
                for z in y.iter_mut() {
                    for w in z.iter_mut() {
                        walker(w);
                    }
                }
            }

            for y in x.counts_x.iter_mut() {
                for z in y.iter_mut() {
                    for w in z.exponent_counts.iter_mut() {
                        for q in w.iter_mut() {
                            walker(q);
                        }
                    }

                    for w in z.residual_noise_counts.iter_mut() {
                        walker(w);
                    }
                }
            }

            for y in x.residual_threshold_counts.iter_mut() {
                for z in y.iter_mut() {
                    for w in z.iter_mut() {
                        walker(w);
                    }
                }
            }

            for y in x.sign_counts.iter_mut() {
                for z in y.iter_mut() {
                    walker(z);
                }
            }
        }
    }

    /// calculates a checksum of the model so we can compare two models for equality
    #[cfg(test)]
    pub fn model_checksum(&mut self) -> u64 {
        use siphasher::sip::SipHasher13;
        use std::hash::Hasher;

        let mut h = SipHasher13::new();
        self.walk(|x| {
            h.write_u16(x.get_count());
        });

        h.finish()
    }
}

// Arrays are more or less in the order of access.
// Array `residual_noise_counts` is split into 7x7 and edge parts to save memory.
// Some dimensions are exchanged to get lower changing rate outer, lowering cache misses frequency.

#[derive(DefaultBoxed)]
pub struct ModelPerColor {
    // `num_non_zeros_context` cannot exceed 25, see `calc_non_zero_counts_context_7x7`
    num_non_zeros_counts7x7:
        [[Branch; 1 << NON_ZERO_7X7_COUNT_BITS]; 1 + NON_ZERO_TO_BIN[25] as usize],

    counts: [[Counts7x7; 49]; NUM_NON_ZERO_7X7_BINS],

    num_non_zeros_counts1x8: NumNonZerosCountsT,
    num_non_zeros_counts8x1: NumNonZerosCountsT,

    counts_x: [[CountsEdge; 14]; NUM_NON_ZERO_EDGE_BINS],

    residual_threshold_counts: [[[Branch; RESIDUAL_THRESHOLD_COUNTS_D3];
        RESIDUAL_THRESHOLD_COUNTS_D2]; RESIDUAL_THRESHOLD_COUNTS_D1],

    sign_counts: [[Branch; NUMERIC_LENGTH_MAX]; 3],
}

#[derive(DefaultBoxed)]
struct Counts7x7 {
    exponent_counts: [[Branch; MAX_EXPONENT]; MAX_EXPONENT],
    residual_noise_counts: [Branch; COEF_BITS],
}

#[derive(DefaultBoxed)]
struct CountsEdge {
    // predictors for exponents are max 11 bits wide, not 12
    exponent_counts: [[Branch; MAX_EXPONENT]; MAX_EXPONENT],
    // size by possible range of `min_threshold - 1`
    // that is from 0 up to `bit_width(max(freq_max)) - RESIDUAL_NOISE_FLOOR - 1`
    residual_noise_counts: [Branch; 3],
}

#[derive(DefaultBoxed)]
struct CountsDC {
    exponent_counts: [[Branch; MAX_EXPONENT]; 17],
    residual_noise_counts: [Branch; COEF_BITS],
}

impl ModelPerColor {
    #[inline(never)]
    pub fn read_coef<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        zig49: usize,
        num_non_zeros_bin: usize,
        best_prior_bit_len: usize,
    ) -> std::io::Result<i16> {
        let (exp, sign, bits) =
            self.get_coef_branches(num_non_zeros_bin, zig49, best_prior_bit_len);

        return Model::read_length_sign_coef(
            bool_reader,
            exp,
            sign,
            bits,
            ModelComponent::Coef(ModelSubComponent::Exp),
            ModelComponent::Coef(ModelSubComponent::Sign),
            ModelComponent::Coef(ModelSubComponent::Noise),
        );
    }

    #[inline(never)]
    pub fn write_coef<W: Write>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        coef: i16,
        zig49: usize,
        num_non_zeros_bin: usize,
        best_prior_bit_len: usize,
    ) -> Result<()> {
        let (exp, sign, bits) =
            self.get_coef_branches(num_non_zeros_bin, zig49, best_prior_bit_len);

        return Model::write_length_sign_coef(
            bool_writer,
            coef,
            exp,
            sign,
            bits,
            ModelComponent::Coef(ModelSubComponent::Exp),
            ModelComponent::Coef(ModelSubComponent::Sign),
            ModelComponent::Coef(ModelSubComponent::Noise),
        )
        .context(here!());
    }

    #[inline(always)]
    fn get_coef_branches(
        &mut self,
        num_non_zeros_bin: usize,
        zig49: usize,
        best_prior_bit_len: usize,
    ) -> (
        &mut [Branch; MAX_EXPONENT],
        &mut Branch,
        &mut [Branch; COEF_BITS],
    ) {
        // these bounds checks happen anyway, but we can provide more helpful error messages
        // and it also means that the compiler can move the actual array references around
        // if it helps with performance
        assert!(
            num_non_zeros_bin < NUM_NON_ZERO_7X7_BINS,
            "num_non_zeros_bin {0} too high",
            num_non_zeros_bin
        );
        assert!(zig49 < 49, "zig49 {0} too high", num_non_zeros_bin);
        assert!(
            best_prior_bit_len < MAX_EXPONENT,
            "best_prior_bit_len {0} too high",
            best_prior_bit_len
        );

        let exp = &mut self.counts[num_non_zeros_bin][zig49].exponent_counts[best_prior_bit_len];
        let sign = &mut self.sign_counts[0][0];
        let bits = &mut self.counts[num_non_zeros_bin][zig49].residual_noise_counts;

        (exp, sign, bits)
    }

    pub fn write_non_zero_7x7_count<W: Write>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        num_non_zeros_7x7_context_bin: u8,
        num_non_zeros_7x7: u8,
    ) -> Result<()> {
        let num_non_zeros_prob =
            &mut self.num_non_zeros_counts7x7[usize::from(num_non_zeros_7x7_context_bin)];

        return bool_writer
            .put_grid(
                num_non_zeros_7x7,
                num_non_zeros_prob,
                ModelComponent::NonZero7x7Count,
            )
            .context(here!());
    }

    pub fn write_non_zero_edge_count<W: Write, const HORIZONTAL: bool>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        est_eob: u8,
        num_non_zeros_bin: u8,
        num_non_zeros_edge: u8,
    ) -> Result<()> {
        let prob_edge_eob =
            self.get_non_zero_counts_edge_mut::<HORIZONTAL>(est_eob, num_non_zeros_bin);

        return bool_writer
            .put_grid(
                num_non_zeros_edge,
                prob_edge_eob,
                ModelComponent::NonZeroEdgeCount,
            )
            .context(here!());
    }

    pub fn read_non_zero_7x7_count<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        num_non_zeros_7x7_context_bin: u8,
    ) -> Result<u8> {
        let num_non_zeros_prob =
            &mut self.num_non_zeros_counts7x7[usize::from(num_non_zeros_7x7_context_bin)];

        return Ok(bool_reader
            .get_grid(num_non_zeros_prob, ModelComponent::NonZero7x7Count)
            .context(here!())? as u8);
    }

    pub fn read_non_zero_edge_count<R: Read, const HORIZONTAL: bool>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        est_eob: u8,
        num_non_zeros_bin: u8,
    ) -> Result<u8> {
        let prob_edge_eob =
            self.get_non_zero_counts_edge_mut::<HORIZONTAL>(est_eob, num_non_zeros_bin);

        return Ok(bool_reader
            .get_grid(prob_edge_eob, ModelComponent::NonZeroEdgeCount)
            .context(here!())? as u8);
    }

    pub fn read_edge_coefficient<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        qt: &QuantizationTables,
        zig15offset: usize,
        num_non_zeros_edge: u8,
        best_prior: i32,
    ) -> Result<i16> {
        let num_non_zeros_edge_bin = usize::from(num_non_zeros_edge) - 1;

        // bounds checks will test these anyway, so check here for better
        // error messages and also gives the optimizer more freedom to move code around
        assert!(
            num_non_zeros_edge_bin < NUM_NON_ZERO_EDGE_BINS,
            "num_non_zeros_edge_bin {0} too high",
            num_non_zeros_edge_bin
        );

        assert!(zig15offset < 14, "zig15offset {0} too high", zig15offset);

        // we cap the bit length since the prior prediction can be wonky
        let best_prior_abs = best_prior.unsigned_abs();
        let best_prior_bit_len =
            cmp::min(MAX_EXPONENT - 1, u32_bit_length(best_prior_abs) as usize);

        let length_branches = &mut self.counts_x[num_non_zeros_edge_bin][zig15offset]
            .exponent_counts[best_prior_bit_len];

        let length = bool_reader
            .get_unary_encoded(
                length_branches,
                ModelComponent::Edge(ModelSubComponent::Exp),
            )
            .context(here!())? as i32;

        let mut coef = 0;
        if length != 0 {
            let sign = &mut self.sign_counts[calc_sign_index(best_prior)][best_prior_bit_len];

            let neg = !bool_reader
                .get(sign, ModelComponent::Edge(ModelSubComponent::Sign))
                .context(here!())?;

            coef = 1;

            if length > 1 {
                let min_threshold: i32 = qt.get_min_noise_threshold(zig15offset).into();
                let mut i: i32 = length - 2;

                if i >= min_threshold {
                    let thresh_prob = self.get_residual_threshold_counts_mut(
                        best_prior_abs,
                        min_threshold,
                        length,
                    );

                    let mut decoded_so_far = 1;
                    while i >= min_threshold {
                        let cur_bit = bool_reader.get(
                            &mut thresh_prob[decoded_so_far],
                            ModelComponent::Edge(ModelSubComponent::Residual),
                        )? as i16;

                        coef <<= 1;
                        coef |= cur_bit;

                        // since we are not strict about rejecting jpegs with out of range coefs
                        // we just make those less efficient by reusing the same probability bucket
                        decoded_so_far = cmp::min(coef as usize, thresh_prob.len() - 1);

                        i -= 1;
                    }
                }

                if i >= 0 {
                    let res_prob = &mut self.counts_x[num_non_zeros_edge_bin][zig15offset]
                        .residual_noise_counts;

                    coef <<= i + 1;
                    coef |= bool_reader.get_n_bits(
                        i as usize + 1,
                        res_prob,
                        ModelComponent::Edge(ModelSubComponent::Noise),
                    )? as i16;
                }
            }

            if neg {
                coef = -coef;
            }
        }
        Ok(coef)
    }

    pub fn write_edge_coefficient<W: Write>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        qt: &QuantizationTables,
        coef: i16,
        zig15offset: usize,
        num_non_zeros_edge: u8,
        best_prior: i32,
    ) -> Result<()> {
        let num_non_zeros_edge_bin = usize::from(num_non_zeros_edge) - 1;

        // bounds checks will test these anyway, so check here for better
        // error messages and also gives the optimizer more freedom to move code around
        assert!(
            num_non_zeros_edge_bin < NUM_NON_ZERO_EDGE_BINS,
            "num_non_zeros_edge_bin {0} too high",
            num_non_zeros_edge_bin
        );

        assert!(zig15offset < 14, "zig15offset {0} too high", zig15offset);

        // we cap the bit length since the prior prediction can be wonky
        let best_prior_abs = best_prior.unsigned_abs();
        let best_prior_bit_len =
            cmp::min(MAX_EXPONENT - 1, u32_bit_length(best_prior_abs) as usize);

        let abs_coef = coef.unsigned_abs();
        let length = u16_bit_length(abs_coef) as usize;

        let exp_array = &mut self.counts_x[num_non_zeros_edge_bin][zig15offset].exponent_counts
            [best_prior_bit_len];

        if length > MAX_EXPONENT {
            return err_exit_code(ExitCode::CoefficientOutOfRange, "CoefficientOutOfRange");
        }

        bool_writer.put_unary_encoded(
            length,
            exp_array,
            ModelComponent::Edge(ModelSubComponent::Exp),
        )?;

        if coef != 0 {
            let sign = &mut self.sign_counts[calc_sign_index(best_prior)][best_prior_bit_len];

            bool_writer.put(
                coef >= 0,
                sign,
                ModelComponent::Edge(ModelSubComponent::Sign),
            )?;

            if length > 1 {
                let min_threshold = i32::from(qt.get_min_noise_threshold(zig15offset));
                let mut i: i32 = length as i32 - 2;

                if i >= min_threshold {
                    let thresh_prob = self.get_residual_threshold_counts_mut(
                        best_prior_abs,
                        min_threshold,
                        length as i32,
                    );

                    let mut encoded_so_far = 1;
                    while i >= min_threshold {
                        let cur_bit = (abs_coef & (1 << i)) != 0;
                        bool_writer.put(
                            cur_bit,
                            &mut thresh_prob[encoded_so_far],
                            ModelComponent::Edge(ModelSubComponent::Residual),
                        )?;

                        encoded_so_far <<= 1;
                        if cur_bit {
                            encoded_so_far |= 1;
                        }

                        // since we are not strict about rejecting jpegs with out of range coefs
                        // we just make those less efficient by reusing the same probability bucket
                        encoded_so_far = cmp::min(encoded_so_far, thresh_prob.len() - 1);

                        i -= 1;
                    }
                }

                if i >= 0 {
                    let res_prob = &mut self.counts_x[num_non_zeros_edge_bin][zig15offset]
                        .residual_noise_counts;

                    bool_writer
                        .put_n_bits(
                            abs_coef as usize,
                            i as usize + 1,
                            res_prob,
                            ModelComponent::Edge(ModelSubComponent::Noise),
                        )
                        .context(here!())?;
                }
            }
        }

        Ok(())
    }

    fn get_residual_threshold_counts_mut(
        &mut self,
        best_prior_abs: u32,
        min_threshold: i32,
        length: i32,
    ) -> &mut [Branch; RESIDUAL_THRESHOLD_COUNTS_D3] {
        // need to & 0xffff since C++ version casts to a uint16_t in the array lookup
        // and we need to match that behavior. It's unlikely that this will be a problem
        // since it would require an extremely large best_prior, which is difficult
        // due to the range limits of 2047 of the coefficients but still in the
        // interest of correctness we should match the C++ behavior.
        return &mut self.residual_threshold_counts[cmp::min(
            ((best_prior_abs & 0xffff) >> min_threshold) as usize,
            self.residual_threshold_counts.len() - 1,
        )][cmp::min(
            (length - min_threshold) as usize,
            self.residual_threshold_counts[0].len() - 1,
        )];
    }

    fn get_non_zero_counts_edge_mut<const HORIZONTAL: bool>(
        &mut self,
        est_eob: u8,
        num_nonzeros_bin: u8,
    ) -> &mut [Branch; 8] {
        if HORIZONTAL {
            return &mut self.num_non_zeros_counts8x1[est_eob as usize][num_nonzeros_bin as usize];
        } else {
            return &mut self.num_non_zeros_counts1x8[est_eob as usize][num_nonzeros_bin as usize];
        }
    }
}

impl Model {
    pub fn get_per_color(&mut self, pt: &ProbabilityTables) -> &mut ModelPerColor {
        &mut self.per_color[pt.get_color_index()]
    }

    pub fn read_dc<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        color_index: usize,
        uncertainty: i16,
        uncertainty2: i16,
    ) -> Result<i16> {
        let (exp, sign, bits) = self.get_dc_branches(uncertainty, uncertainty2, color_index);

        return Model::read_length_sign_coef(
            bool_reader,
            exp,
            sign,
            bits,
            ModelComponent::DC(ModelSubComponent::Exp),
            ModelComponent::DC(ModelSubComponent::Sign),
            ModelComponent::DC(ModelSubComponent::Noise),
        )
        .context(here!());
    }

    pub fn write_dc<W: Write>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        color_index: usize,
        coef: i16,
        uncertainty: i16,
        uncertainty2: i16,
    ) -> Result<()> {
        let (exp, sign, bits) = self.get_dc_branches(uncertainty, uncertainty2, color_index);

        return Model::write_length_sign_coef(
            bool_writer,
            coef,
            exp,
            sign,
            bits,
            ModelComponent::DC(ModelSubComponent::Exp),
            ModelComponent::DC(ModelSubComponent::Sign),
            ModelComponent::DC(ModelSubComponent::Noise),
        )
        .context(here!());
    }

    fn get_dc_branches(
        &mut self,
        uncertainty: i16,
        uncertainty2: i16,
        color_index: usize,
    ) -> (
        &mut [Branch; MAX_EXPONENT],
        &mut Branch,
        &mut [Branch; COEF_BITS],
    ) {
        let len_abs_mxm = u16_bit_length(uncertainty.unsigned_abs());
        let len_abs_offset_to_closest_edge = u16_bit_length(uncertainty2.unsigned_abs());
        let len_abs_mxm_clamp = cmp::min(len_abs_mxm as usize, self.counts_dc.len() - 1);

        let exp = &mut self.counts_dc[len_abs_mxm_clamp].exponent_counts
            [len_abs_offset_to_closest_edge as usize];
        let sign = &mut self.per_color[color_index].sign_counts[0]
            [calc_sign_index(uncertainty2 as i32) + 1]; // +1 to separate from sign_counts[0][0]
        let bits = &mut self.counts_dc[len_abs_mxm_clamp].residual_noise_counts;

        (exp, sign, bits)
    }

    #[inline(always)]
    fn read_length_sign_coef<const A: usize, const B: usize, R: Read>(
        bool_reader: &mut VPXBoolReader<R>,
        magnitude_branches: &mut [Branch; A],
        sign_branch: &mut Branch,
        bits_branch: &mut [Branch; B],
        mag_cmp: ModelComponent,
        sign_cmp: ModelComponent,
        bits_cmp: ModelComponent,
    ) -> std::io::Result<i16> {
        debug_assert!(
            A - 1 <= B,
            "A (max mag) should be not more than B+1 (max bits). A={0} B={1} from {2:?}",
            A,
            B,
            mag_cmp
        );

        let length = bool_reader.get_unary_encoded(magnitude_branches, mag_cmp)?;

        let mut coef: i16 = 0;
        if length != 0 {
            let neg = !bool_reader.get(sign_branch, sign_cmp)?;
            if length > 1 {
                coef = bool_reader.get_n_bits(length - 1, bits_branch, bits_cmp)? as i16;
            }

            coef |= (1 << (length - 1)) as i16;

            if neg {
                coef = -coef;
            }
        }

        return Ok(coef);
    }

    fn write_length_sign_coef<const A: usize, const B: usize, W: Write>(
        bool_writer: &mut VPXBoolWriter<W>,
        coef: i16,
        magnitude_branches: &mut [Branch; A],
        sign_branch: &mut Branch,
        bits_branch: &mut [Branch; B],
        mag_cmp: ModelComponent,
        sign_cmp: ModelComponent,
        bits_cmp: ModelComponent,
    ) -> Result<()> {
        debug_assert!(
            A - 1 <= B,
            "A (max mag) should be not more than B+1 (max bits). A={0} B={1} from {2:?}",
            A,
            B,
            mag_cmp,
        );

        let abs_coef = coef.unsigned_abs();
        let coef_bit_len = u16_bit_length(abs_coef);

        if coef_bit_len > A as u8 {
            return err_exit_code(
                ExitCode::CoefficientOutOfRange,
                "coefficient > MAX_EXPONENT",
            );
        }

        bool_writer.put_unary_encoded(coef_bit_len as usize, magnitude_branches, mag_cmp)?;
        if coef != 0 {
            bool_writer.put(coef > 0, sign_branch, sign_cmp)?;
        }

        if coef_bit_len > 1 {
            debug_assert!(
                (abs_coef & (1 << (coef_bit_len - 1))) != 0,
                "Biggest bit must be set"
            );
            debug_assert!(
                (abs_coef & (1 << coef_bit_len)) == 0,
                "Beyond Biggest bit must be zero"
            );

            bool_writer.put_n_bits(
                abs_coef as usize,
                coef_bit_len as usize - 1,
                bits_branch,
                bits_cmp,
            )?;
        }

        Ok(())
    }
}
