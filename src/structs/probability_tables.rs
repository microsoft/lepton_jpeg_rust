/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::cmp;

use crate::consts::*;
use crate::helpers::*;
use crate::structs::idct::*;
use crate::structs::model::*;
use crate::structs::quantization_tables::*;

use super::block_based_image::BlockBasedImage;
use super::block_context::BlockContext;
use super::neighbor_summary::NeighborSummary;
use super::probability_tables_coefficient_context::ProbabilityTablesCoefficientContext;

pub struct ProbabilityTables {
    left_present: bool,
    above_present: bool,
    all_present: bool,
    color: usize,
}

pub struct PredictDCResult {
    pub predicted_dc: i32,
    pub uncertainty: i16,
    pub uncertainty2: i16,
    pub advanced_predict_dc_pixels_sans_dc: [i16; 64],
}

impl ProbabilityTables {
    pub fn new(kcolor: usize, in_left_present: bool, in_above_present: bool) -> ProbabilityTables {
        return ProbabilityTables {
            left_present: in_left_present,
            above_present: in_above_present,
            all_present: in_left_present && in_above_present,
            color: kcolor,
        };
    }

    pub fn is_all_present(&self) -> bool {
        self.all_present
    }

    pub fn adv_predict_or_unpredict_dc(
        saved_dc: i16,
        recover_original: bool,
        predicted_val: i32,
    ) -> i32 {
        let max_value = 1 << (MAX_EXPONENT - 1);
        let min_value = -max_value;
        let adjustment_factor = (2 * max_value) + 1;
        let mut retval = predicted_val;
        retval = saved_dc as i32 + if recover_original { retval } else { -retval };

        if retval < min_value {
            retval += adjustment_factor;
        }

        if retval > max_value {
            retval -= adjustment_factor;
        }

        return retval;
    }

    pub fn get_color_index(&self) -> usize {
        return if self.color == 0 { 0 } else { 1 };
    }

    pub fn num_non_zeros_to_bin(num_non_zeros: u8) -> u8 {
        return NON_ZERO_TO_BIN[NUM_NON_ZERO_BINS - 1][num_non_zeros as usize];
    }

    pub fn calc_non_zero_counts_context_7x7<const ALL_PRESENT: bool>(
        &self,
        block: &BlockContext,
        num_non_zeros: &[NeighborSummary],
    ) -> u8 {
        let mut num_non_zeros_above = 0;
        let mut num_non_zeros_left = 0;
        if ALL_PRESENT || self.above_present {
            num_non_zeros_above = block.get_non_zeros_above(num_non_zeros);
        }

        if ALL_PRESENT || self.left_present {
            num_non_zeros_left = block.get_non_zeros_left(num_non_zeros);
        }

        let num_non_zeros_context;
        if (!ALL_PRESENT) && self.above_present && !self.left_present {
            num_non_zeros_context = (num_non_zeros_above + 1) / 2;
        } else if (!ALL_PRESENT) && self.left_present && !self.above_present {
            num_non_zeros_context = (num_non_zeros_left + 1) / 2;
        } else if ALL_PRESENT || (self.left_present && self.above_present) {
            num_non_zeros_context = (num_non_zeros_above + num_non_zeros_left + 2) / 4;
        } else {
            num_non_zeros_context = 0;
        }

        return num_non_zeros_context;
    }

    pub fn calc_coefficient_context_7x7_aavg<const ALL_PRESENT: bool>(
        &self,
        image_data: &BlockBasedImage,
        coord: usize,
        block_context: &BlockContext,
        num_nonzeros_left: u8,
    ) -> ProbabilityTablesCoefficientContext {
        // from compute_aavg
        let mut best_prior: u32 = 0;

        let aligned = RASTER_TO_ALIGNED[coord] as usize;
        assert!(aligned < 64); // assert here to avoid range checks later

        let log_weight = 5;

        if ALL_PRESENT || self.left_present {
            best_prior += (block_context.left(image_data).get_coefficient(aligned)).abs() as u32;
        }

        if ALL_PRESENT || self.above_present {
            best_prior += (block_context.above(image_data).get_coefficient(aligned)).abs() as u32;
        }

        if ALL_PRESENT || (self.left_present && self.above_present) {
            best_prior *= 13;
            best_prior += 6
                * (block_context
                    .above_left(image_data)
                    .get_coefficient(aligned))
                .abs() as u32;
            best_prior >>= log_weight;
        }

        return ProbabilityTablesCoefficientContext {
            best_prior: best_prior as i32,
            num_non_zeros_bin: ProbabilityTables::num_non_zeros_to_bin(num_nonzeros_left),
            best_prior_bit_len: u32_bit_length(cmp::min(
                best_prior, /* no need for abs here as in original source */
                1023,
            ))
            .into(),
        };
    }

    pub fn calc_coefficient_context8_lak<const ALL_PRESENT: bool, const HORIZONTAL: bool>(
        &self,
        image_data: &BlockBasedImage,
        qt: &QuantizationTables,
        coefficient: usize,
        block_context: &BlockContext,
        num_non_zeros_x: u8,
    ) -> ProbabilityTablesCoefficientContext {
        let coef_idct;

        let mut compute_lak_coeffs_x: [i32; 8] = [0; 8];
        let mut compute_lak_coeffs_a: [i32; 8] = [0; 8];

        debug_assert_eq!(HORIZONTAL, (coefficient & 7) != 0);

        let coef_idct_offset;
        if HORIZONTAL && (ALL_PRESENT || self.above_present) {
            assert!(coefficient < 8); // avoid bounds check later

            // y == 0: we're the x
            let here = block_context.here(image_data);
            let above = block_context.above(image_data);

            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 0..8 {
                let cur_coef = RASTER_TO_ALIGNED[(coefficient + (i * 8)) as usize] as usize;

                compute_lak_coeffs_x[i] = if i != 0 {
                    here.get_coefficient(cur_coef).into()
                } else {
                    0
                };
                compute_lak_coeffs_a[i] = above.get_coefficient(cur_coef).into();
            }

            coef_idct = qt.get_icos_idct_edge8192_dequantized_x();
            coef_idct_offset = coefficient * 8;
        } else if !HORIZONTAL && (ALL_PRESENT || self.left_present) {
            assert!(coefficient <= 56); // avoid bounds check later

            // x == 0: we're the y
            let here = block_context.here(image_data);
            let left = block_context.left(image_data);

            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 0..8 {
                let cur_coef = RASTER_TO_ALIGNED[(coefficient + i) as usize] as usize;

                compute_lak_coeffs_x[i] = if i != 0 {
                    here.get_coefficient(cur_coef).into()
                } else {
                    0
                };
                compute_lak_coeffs_a[i] = left.get_coefficient(cur_coef).into();
            }

            coef_idct = qt.get_icos_idct_edge8192_dequantized_y();
            coef_idct_offset = coefficient;
        } else {
            return ProbabilityTablesCoefficientContext {
                best_prior: 0,
                num_non_zeros_bin: num_non_zeros_x,
                best_prior_bit_len: 0,
            };
        }

        let mut best_prior = compute_lak_coeffs_a[0] * coef_idct[coef_idct_offset]; // rounding towards zero before adding coeffs_a[0] helps ratio slightly, but this is cheaper
        for i in 1..8 {
            let sign = if (i & 1) != 0 { 1 } else { -1 };
            best_prior -= coef_idct[coef_idct_offset + i]
                * (compute_lak_coeffs_x[i] + (sign * compute_lak_coeffs_a[i]));
        }

        best_prior /= coef_idct[coef_idct_offset];

        return ProbabilityTablesCoefficientContext {
            best_prior,
            num_non_zeros_bin: num_non_zeros_x,
            best_prior_bit_len: u32_bit_length(cmp::min(best_prior.abs() as u32, 1023)),
        };
    }

    pub fn adv_predict_dc_pix<const ALL_PRESENT: bool>(
        &self,
        image_data: &BlockBasedImage,
        qt: &QuantizationTables,
        block_context: &BlockContext,
        num_non_zeros: &[NeighborSummary],
    ) -> PredictDCResult {
        let mut uncertainty_val: i16 = 0;
        let mut uncertainty2_val: i16 = 0;

        let mut pixels_sans_dc = [0i16; 64];
        let q = qt.get_quantization_table();

        run_idct::<true /*IGNORE_DC*/>(block_context.here(image_data), q, &mut pixels_sans_dc);

        let mut avgmed = 0;
        let mut dc_estimates = [0i16; 16];

        if ALL_PRESENT || self.left_present || self.above_present {
            if ALL_PRESENT || self.left_present {
                let left_context = block_context.neighbor_context_left(num_non_zeros);
                for i in 0..8 {
                    let i_mult8 = i << 3;
                    let a = i32::from(pixels_sans_dc[i_mult8]) + 1024;
                    let pixel_delta =
                        i32::from(pixels_sans_dc[i_mult8]) - i32::from(pixels_sans_dc[i_mult8 + 1]);
                    let b = i32::from(left_context.get_vertical(i)) - (pixel_delta / 2); // round to zero
                    dc_estimates[i] = (b - a) as i16;
                }
            }

            if ALL_PRESENT || self.above_present {
                let above_context = block_context.neighbor_context_above(num_non_zeros);
                for i in 0..8 {
                    let a = i32::from(pixels_sans_dc[i]) + 1024;
                    let pixel_delta =
                        i32::from(pixels_sans_dc[i]) - i32::from(pixels_sans_dc[i + 8]);
                    let b = i32::from(above_context.get_horizontal(i)) - (pixel_delta / 2); // round to zero
                    dc_estimates[i
                        + (if ALL_PRESENT || self.left_present {
                            8
                        } else {
                            0
                        })] = (b - a) as i16;
                }
            }

            let mut avg_vertical;
            let mut min_dc = dc_estimates[0];
            let mut max_dc = dc_estimates[0];
            let mut avg_horizontal =
                ProbabilityTables::estimate_dir_average(&dc_estimates, 0, &mut min_dc, &mut max_dc);
            if (!ALL_PRESENT) && (self.above_present == false || self.left_present == false) {
                avg_vertical = avg_horizontal;
            } else {
                avg_vertical = ProbabilityTables::estimate_dir_average(
                    &dc_estimates,
                    8,
                    &mut min_dc,
                    &mut max_dc,
                );
            }

            let overall_avg: i32 = (avg_vertical + avg_horizontal) >> 1;
            avgmed = overall_avg;
            uncertainty_val = ((i32::from(max_dc) - i32::from(min_dc)) >> 3) as i16;
            avg_horizontal -= avgmed;
            avg_vertical -= avgmed;

            let mut far_afield_value = avg_vertical;
            if avg_horizontal.abs() < avg_vertical.abs() {
                far_afield_value = avg_horizontal;
            }

            uncertainty2_val = (far_afield_value >> 3) as i16;
        }

        return PredictDCResult {
            predicted_dc: ((avgmed / i32::from(q[0])) + 4) >> 3,
            uncertainty: uncertainty_val,
            uncertainty2: uncertainty2_val,
            advanced_predict_dc_pixels_sans_dc: pixels_sans_dc,
        };
    }

    fn estimate_dir_average(
        dc_estimates: &[i16; 16],
        dc_start_index: usize,
        min_dc: &mut i16,
        max_dc: &mut i16,
    ) -> i32 {
        let mut dir_average: i32 = 0;
        for i in 0..8 {
            let cur_est = dc_estimates[dc_start_index + i];
            dir_average += i32::from(cur_est);

            if *min_dc > cur_est {
                *min_dc = cur_est;
            }

            if *max_dc < cur_est {
                *max_dc = cur_est;
            }
        }

        return dir_average;
    }
}
