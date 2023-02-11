/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow::{Context, Result};
use std::cmp;
use std::io::{Read, Write};

use crate::consts::*;
use crate::helpers::{calc_sign_index, err_exit_code, here, u16_bit_length};
use crate::lepton_error::ExitCode;
use crate::metrics::{ModelComponent, ModelSubComponent};
use crate::structs::branch::Branch;
use default_boxed::DefaultBoxed;

use super::probability_tables::ProbabilityTables;
use super::probability_tables_coefficient_context::ProbabilityTablesCoefficientContext;
use super::quantization_tables::QuantizationTables;
use super::vpx_bool_reader::VPXBoolReader;
use super::vpx_bool_writer::VPXBoolWriter;

pub const MAX_EXPONENT: usize = 11;
const BLOCK_TYPES: usize = 2; // setting this to 3 gives us ~1% savings.. 2/3 from BLOCK_TYPES=2
pub const NUM_NON_ZERO_BINS: usize = 10;
//const BsrBestPriorMax : usize = 11; // 1023 requires 11 bits to describe
//const EntropyNodes : usize = 15;
//const NunNonZerosEofPiors : usize = 66;
//const ZeroOrEob : usize = 3;
const COEF_BITS: usize = MAX_EXPONENT - 1; // the last item of the length is always 1

const NUMERIC_LENGTH_MAX: usize = 12;
//const NumericLengthBits : usize = 4;

const EXPONENT_COUNT_DC_BINS: usize = if NUM_NON_ZERO_BINS > NUMERIC_LENGTH_MAX {
    NUM_NON_ZERO_BINS
} else {
    NUMERIC_LENGTH_MAX
};

type NumNonZerosCounts7x7T = [[[Branch; 32]; 6]; 26];

type NumNonZerosCountsT = [[[[Branch; 4]; 3]; 8]; 8];

pub const RESIDUAL_THRESHOLD_COUNTS_D1: usize = 1 << (1 + RESIDUAL_NOISE_FLOOR);
pub const RESIDUAL_THRESHOLD_COUNTS_D2: usize = 1 + RESIDUAL_NOISE_FLOOR;
pub const RESIDUAL_THRESHOLD_COUNTS_D3: usize = 1 << RESIDUAL_NOISE_FLOOR;

/// model used to track probabilities of the bins
#[derive(DefaultBoxed)]
pub struct Model {
    cmp: [ModelPerColor; BLOCK_TYPES],

    /// non color dependent bins. Not sure why, but changing this will brake the file format
    exponent_counts_dc: [[[Branch; MAX_EXPONENT]; 17]; EXPONENT_COUNT_DC_BINS],

    residual_noise_counts_dc: [[Branch; COEF_BITS]; NUMERIC_LENGTH_MAX],
}

/// this is the model per component or color. Currently there are only two that are used,
/// 0 for the first component and 1 for the rest of all components
#[derive(DefaultBoxed)]
pub struct ModelPerColor {
    num_non_zeros_counts7x7: NumNonZerosCounts7x7T,

    num_non_zeros_counts1x8: NumNonZerosCountsT,

    num_non_zeros_counts8x1: NumNonZerosCountsT,

    residual_threshold_counts: [[[Branch; RESIDUAL_THRESHOLD_COUNTS_D3];
        RESIDUAL_THRESHOLD_COUNTS_D2]; RESIDUAL_THRESHOLD_COUNTS_D1],

    coef_bins: [[AcCoefficientBin; NUM_NON_ZERO_BINS]; 64],

    sign_counts: [[Branch; NUMERIC_LENGTH_MAX]; 4],
}

/// bins for a single AC coefficent
#[derive(Default)]
struct AcCoefficientBin {
    exp: [[Branch; 11]; NUMERIC_LENGTH_MAX],
    bits: [Branch; 10],
}

impl Model {
    pub fn get_per_color(&mut self, pt: &ProbabilityTables) -> &mut ModelPerColor {
        return &mut self.cmp[pt.get_color_index()];
    }

    pub fn read_dc<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        color_index: usize,
        uncertainty: i16,
        uncertainty2: i16,
    ) -> Result<i16> {
        let (exp, sign, bits) = self.get_dc_branches(uncertainty, uncertainty2, color_index);

        return read_length_sign_coef(
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

        return write_length_sign_coef(
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
    ) -> (&mut [Branch; MAX_EXPONENT], &mut Branch, &mut [Branch; 10]) {
        let len_abs_mxm = u16_bit_length(uncertainty.unsigned_abs());
        let len_abs_offset_to_closest_edge = u16_bit_length(uncertainty2.unsigned_abs());

        let exp = &mut self.exponent_counts_dc
            [cmp::min(len_abs_mxm as usize, self.exponent_counts_dc.len() - 1)][cmp::min(
            len_abs_offset_to_closest_edge as usize,
            self.exponent_counts_dc[0].len() - 1,
        )];

        let sign = &mut self.cmp[color_index].sign_counts[0][if uncertainty2 >= 0 {
            if uncertainty2 == 0 {
                3
            } else {
                2
            }
        } else {
            1
        }];

        let bits = &mut self.residual_noise_counts_dc[cmp::min(
            self.residual_noise_counts_dc.len() - 1,
            len_abs_mxm as usize,
        )];
        (exp, sign, bits)
    }
}

impl ModelPerColor {
    #[inline(never)]
    pub fn read_coef<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        zig49: usize,
        num_non_zeros_bin: usize,
        best_prior_bit_len: usize,
    ) -> Result<i16> {
        let (exp, sign, bits) =
            self.get_coef_branches(num_non_zeros_bin, zig49, best_prior_bit_len);

        return read_length_sign_coef(
            bool_reader,
            exp,
            sign,
            bits,
            ModelComponent::Coef(ModelSubComponent::Exp),
            ModelComponent::Coef(ModelSubComponent::Sign),
            ModelComponent::Coef(ModelSubComponent::Noise),
        )
        .context(here!());
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

        return write_length_sign_coef(
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

    fn get_coef_branches(
        &mut self,
        num_non_zeros_bin: usize,
        zig49: usize,
        best_prior_bit_len: usize,
    ) -> (
        &mut [Branch; MAX_EXPONENT],
        &mut Branch,
        &mut [Branch; MAX_EXPONENT - 1],
    ) {
        debug_assert!(
            num_non_zeros_bin < NUM_NON_ZERO_BINS,
            "num_non_zeros_bin {0} too high",
            num_non_zeros_bin
        );

        let exp = &mut self.coef_bins[zig49][num_non_zeros_bin];
        let sign = &mut self.sign_counts[0][0];
        (&mut exp.exp[best_prior_bit_len], sign, &mut exp.bits)
    }

    pub fn write_non_zero_7x7_count<W: Write>(
        &mut self,
        bool_writer: &mut VPXBoolWriter<W>,
        num_non_zeros_context: u8,
        num_non_zeros_7x7: u8,
    ) -> Result<()> {
        let num_non_zeros_prob = &mut self.num_non_zeros_counts7x7
            [ProbabilityTables::num_non_zeros_to_bin(num_non_zeros_context) as usize];

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
        num_non_zeros_7x7: u8,
        num_non_zeros_edge: u8,
    ) -> Result<()> {
        let prob_edge_eob =
            self.get_non_zero_counts_edge_mut::<HORIZONTAL>(est_eob, num_non_zeros_7x7);

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
        num_non_zeros_context: u8,
    ) -> Result<u8> {
        let num_non_zeros_prob = &mut self.num_non_zeros_counts7x7
            [ProbabilityTables::num_non_zeros_to_bin(num_non_zeros_context) as usize];

        return Ok(bool_reader
            .get_grid(num_non_zeros_prob, ModelComponent::NonZero7x7Count)
            .context(here!())? as u8);
    }

    pub fn read_non_zero_edge_count<R: Read, const HORIZONTAL: bool>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        est_eob: u8,
        num_non_zeros_7x7: u8,
    ) -> Result<u8> {
        let prob_edge_eob =
            self.get_non_zero_counts_edge_mut::<HORIZONTAL>(est_eob, num_non_zeros_7x7);

        return Ok(bool_reader
            .get_grid(prob_edge_eob, ModelComponent::NonZeroEdgeCount)
            .context(here!())? as u8);
    }

    pub fn read_edge_coefficient<R: Read>(
        &mut self,
        bool_reader: &mut VPXBoolReader<R>,
        qt: &QuantizationTables,
        coord: usize,
        zig15offset: usize,
        ptcc8: &ProbabilityTablesCoefficientContext,
    ) -> Result<i16> {
        let length_branches = &mut self.coef_bins[49 + zig15offset]
            [ptcc8.num_non_zeros_bin as usize]
            .exp[usize::from(ptcc8.best_prior_bit_len)];

        let length = bool_reader
            .get_unary_encoded(
                length_branches,
                ModelComponent::Edge(ModelSubComponent::Exp),
            )
            .context(here!())? as i32;

        let mut coef = 0;
        if length != 0 {
            let sign = self.get_sign_counts_mut(ptcc8);

            let neg = !bool_reader
                .get(sign, ModelComponent::Edge(ModelSubComponent::Sign))
                .context(here!())?;

            coef = 1 << (length - 1);

            if length > 1 {
                let min_threshold: i32 = qt.get_min_noise_threshold(coord).into();
                let mut i: i32 = length - 2;

                if i >= min_threshold {
                    let thresh_prob =
                        self.get_residual_threshold_counts_mut(ptcc8, min_threshold, length);

                    let mut decoded_so_far = 1;
                    while i >= min_threshold {
                        let cur_bit = bool_reader.get(
                            &mut thresh_prob[decoded_so_far],
                            ModelComponent::Edge(ModelSubComponent::Residual),
                        )? as i16;

                        coef |= cur_bit << i;
                        decoded_so_far <<= 1;

                        if cur_bit != 0 {
                            decoded_so_far |= 1;
                        }

                        // since we are not strict about rejecting jpegs with out of range coefs
                        // we just make those less efficient by reusing the same probability bucket
                        decoded_so_far = cmp::min(decoded_so_far, thresh_prob.len() - 1);

                        i -= 1;
                    }
                }

                if i >= 0 {
                    let res_prob = &mut self.coef_bins[49 + zig15offset]
                        [ptcc8.num_non_zeros_bin as usize]
                        .bits;

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
        coord: usize,
        zig15offset: usize,
        ptcc8: &ProbabilityTablesCoefficientContext,
    ) -> Result<()> {
        let exp_array = &mut self.coef_bins[49 + zig15offset][ptcc8.num_non_zeros_bin as usize].exp
            [ptcc8.best_prior_bit_len as usize];

        let abs_coef = coef.unsigned_abs();
        let length = u16_bit_length(abs_coef) as usize;

        if length > MAX_EXPONENT {
            return err_exit_code(ExitCode::CoefficientOutOfRange, "CoefficientOutOfRange");
        }

        bool_writer.put_unary_encoded(
            length,
            exp_array,
            ModelComponent::Edge(ModelSubComponent::Exp),
        )?;

        if coef != 0 {
            let min_threshold = i32::from(qt.get_min_noise_threshold(coord));
            let sign = self.get_sign_counts_mut(ptcc8);

            bool_writer.put(
                coef >= 0,
                sign,
                ModelComponent::Edge(ModelSubComponent::Sign),
            )?;

            if length > 1 {
                let mut i: i32 = length as i32 - 2;
                if i >= min_threshold {
                    let thresh_prob =
                        self.get_residual_threshold_counts_mut(ptcc8, min_threshold, length as i32);

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
                    let res_prob = &mut self.coef_bins[49 + zig15offset]
                        [ptcc8.num_non_zeros_bin as usize]
                        .bits;

                    bool_writer
                        .put_n_bits(
                            abs_coef as usize,
                            (i + 1) as usize,
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
        ptcc8: &ProbabilityTablesCoefficientContext,
        min_threshold: i32,
        length: i32,
    ) -> &mut [Branch; RESIDUAL_THRESHOLD_COUNTS_D3] {
        return &mut self.residual_threshold_counts[cmp::min(
            (ptcc8.best_prior.abs() >> min_threshold) as usize,
            self.residual_threshold_counts.len() - 1,
        )][cmp::min(
            (length - min_threshold) as usize,
            self.residual_threshold_counts[0].len() - 1,
        )];
    }

    fn get_non_zero_counts_edge_mut<const HORIZONTAL: bool>(
        &mut self,
        est_eob: u8,
        num_nonzeros: u8,
    ) -> &mut [[Branch; 4]; 3] {
        if HORIZONTAL {
            return &mut self.num_non_zeros_counts8x1[est_eob as usize]
                [(num_nonzeros as usize + 3) / 7];
        } else {
            return &mut self.num_non_zeros_counts1x8[est_eob as usize]
                [(num_nonzeros as usize + 3) / 7];
        }
    }

    fn get_sign_counts_mut(&mut self, ptcc8: &ProbabilityTablesCoefficientContext) -> &mut Branch {
        &mut self.sign_counts[calc_sign_index(ptcc8.best_prior)][ptcc8.best_prior_bit_len as usize]
    }
}

fn read_length_sign_coef<const A: usize, const B: usize, R: Read>(
    bool_reader: &mut VPXBoolReader<R>,
    magnitude_branches: &mut [Branch; A],
    sign_branch: &mut Branch,
    bits_branch: &mut [Branch; B],
    mag_cmp: ModelComponent,
    sign_cmp: ModelComponent,
    bits_cmp: ModelComponent,
) -> Result<i16> {
    assert!(
        A - 1 <= B,
        "A (max mag) should be not more than B+1 (max bits). A={0} B={1} from {2:?}",
        A,
        B,
        mag_cmp
    );

    let length = bool_reader
        .get_unary_encoded(magnitude_branches, mag_cmp)
        .context(here!())?;

    let mut coef: i16 = 0;
    if length != 0 {
        let neg = !bool_reader.get(sign_branch, sign_cmp)?;
        if length > 1 {
            coef = bool_reader
                .get_n_bits(length - 1, bits_branch, bits_cmp)
                .context(here!())? as i16;
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
    assert!(
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
        bool_writer.put(coef >= 0, sign_branch, sign_cmp)?;
    }

    if coef_bit_len > 1 {
        assert!(
            (abs_coef & (1 << (coef_bit_len - 1))) != 0,
            "Biggest bit must be set"
        );
        assert!(
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
