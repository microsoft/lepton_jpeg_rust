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
use std::cmp::{max, min};

use super::block_based_image::AlignedBlock;
use super::block_context::BlockContext;
use super::neighbor_summary::NeighborSummary;
use super::probability_tables_coefficient_context::ProbabilityTablesCoefficientContext;

use wide::i16x8;

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

    pub fn is_left_present(&self) -> bool {
        self.left_present
    }
    pub fn is_above_present(&self) -> bool {
        self.above_present
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

    // calculates the average of the prior values from their corresponding value in the left, above and above/left block
    // the C++ version does one coefficient at a time, but if we do it all at the same time, the compiler vectorizes everything
    #[inline(never)]
    pub fn calc_coefficient_context_7x7_aavg_block<const ALL_PRESENT: bool>(
        &self,
        left: &AlignedBlock,
        above: &AlignedBlock,
        above_left: &AlignedBlock,
    ) -> [i16; 49] {
        let mut best_prior = [0; 49];

        if ALL_PRESENT {
            // compiler does a pretty amazing job with SSE/AVX2 here
            for i in 0..49 {
                // approximate average of 3 without a divide with double the weight for left/top vs diagonal
                best_prior[i] = (((left.get_coefficient(i).abs() as u32
                    + above.get_coefficient(i).abs() as u32)
                    * 13
                    + 6 * above_left.get_coefficient(i).abs() as u32)
                    >> 5) as i16;
            }
        } else {
            // handle edge case :) where we are on the top or left edge

            if self.left_present {
                for i in 0..49 {
                    best_prior[i] = left.get_coefficient(i).abs();
                }
            } else if self.above_present {
                for i in 0..49 {
                    best_prior[i] = above.get_coefficient(i).abs();
                }
            }
        }

        return best_prior;
    }

    #[inline(always)]
    pub fn calc_coefficient_context8_lak<const ALL_PRESENT: bool, const HORIZONTAL: bool>(
        &self,
        qt: &QuantizationTables,
        coefficient: usize,
        here: &AlignedBlock,
        above: &AlignedBlock,
        left: &AlignedBlock,
        num_non_zeros_x: u8,
    ) -> ProbabilityTablesCoefficientContext {
        let mut compute_lak_coeffs_x: [i32; 8] = [0; 8];
        let mut compute_lak_coeffs_a: [i32; 8] = [0; 8];

        debug_assert_eq!(HORIZONTAL, (coefficient & 7) != 0);

        let coef_idct;
        if HORIZONTAL && (ALL_PRESENT || self.above_present) {
            assert!(coefficient < 8); // avoid bounds check later

            // y == 0: we're the x
            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 0..8 {
                let cur_coef = usize::from(RASTER_TO_ALIGNED[usize::from(coefficient + (i * 8))]);

                let sign = if (i & 1) != 0 { -1 } else { 1 };

                compute_lak_coeffs_x[i] = if i != 0 {
                    here.get_coefficient(cur_coef).into()
                } else {
                    0
                };
                compute_lak_coeffs_a[i] = (sign * above.get_coefficient(cur_coef)).into();
            }

            coef_idct =
                &qt.get_icos_idct_edge8192_dequantized_x()[coefficient * 8..(coefficient + 1) * 8];
        } else if !HORIZONTAL && (ALL_PRESENT || self.left_present) {
            assert!(coefficient <= 56); // avoid bounds check later

            // x == 0: we're the y

            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 0..8 {
                let cur_coef = usize::from(RASTER_TO_ALIGNED[usize::from(coefficient + i)]);

                let sign = if (i & 1) != 0 { -1 } else { 1 };

                compute_lak_coeffs_x[i] = if i != 0 {
                    here.get_coefficient(cur_coef).into()
                } else {
                    0
                };
                compute_lak_coeffs_a[i] = (sign * left.get_coefficient(cur_coef)).into();
            }

            coef_idct = &qt.get_icos_idct_edge8192_dequantized_y()[coefficient..coefficient + 8];
        } else {
            return ProbabilityTablesCoefficientContext {
                best_prior: 0,
                num_non_zeros_bin: num_non_zeros_x,
                best_prior_bit_len: 0,
            };
        }

        let mut best_prior: i32 = 0;
        for i in 0..8 {
            // some extreme coefficents can cause this to overflow, but since this is just a predictor, no need to panic
            best_prior = best_prior.wrapping_add(
                coef_idct[i]
                    .wrapping_mul(compute_lak_coeffs_a[i].wrapping_sub(compute_lak_coeffs_x[i])),
            );
            // rounding towards zero before adding coeffs_a[0] helps ratio slightly, but this is cheaper
        }

        best_prior /= coef_idct[0];

        return ProbabilityTablesCoefficientContext {
            best_prior,
            num_non_zeros_bin: num_non_zeros_x,
            best_prior_bit_len: u32_bit_length(cmp::min(best_prior.unsigned_abs(), 1023)),
        };
    }

    fn from_stride(block: &[i16; 64], offset: usize, stride: usize) -> i16x8 {
        return i16x8::new([
            block[offset],
            block[offset + (1 * stride)],
            block[offset + (2 * stride)],
            block[offset + (3 * stride)],
            block[offset + (4 * stride)],
            block[offset + (5 * stride)],
            block[offset + (6 * stride)],
            block[offset + (7 * stride)],
        ]);
    }

    pub fn adv_predict_dc_pix<const ALL_PRESENT: bool>(
        &self,
        here: &AlignedBlock,
        qt: &QuantizationTables,
        block_context: &BlockContext,
        num_non_zeros: &[NeighborSummary],
    ) -> PredictDCResult {
        let mut uncertainty_val: i16 = 0;
        let mut uncertainty2_val: i16 = 0;

        let mut pixels_sans_dc = [0i16; 64];
        let q = qt.get_quantization_table();

        let mut avgmed = 0;

        run_idct::<true>(here, q, &mut pixels_sans_dc);

        if ALL_PRESENT || self.left_present || self.above_present {
            let mut min_dc = i16::MAX;
            let mut max_dc = i16::MIN;

            let avg_horizontal_option;
            if ALL_PRESENT || self.left_present {
                let left_context = block_context.neighbor_context_left(num_non_zeros);

                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc, 0, 8);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc, 1, 8);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*left_context.get_vertical()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                let dc_estimates = (b - a).to_array();

                avg_horizontal_option = Some(ProbabilityTables::estimate_dir_average(
                    &dc_estimates,
                    &mut min_dc,
                    &mut max_dc,
                ));
            } else {
                avg_horizontal_option = None;
            }

            let avg_vertical_option;
            if ALL_PRESENT || self.above_present {
                let above_context = block_context.neighbor_context_above(num_non_zeros);

                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc, 0, 1);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc, 8, 1);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*above_context.get_horizontal()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                let dc_estimates = (b - a).to_array();

                avg_vertical_option = Some(ProbabilityTables::estimate_dir_average(
                    &dc_estimates,
                    &mut min_dc,
                    &mut max_dc,
                ));
            } else {
                avg_vertical_option = None;
            }

            let mut avg_vertical = avg_vertical_option.or(avg_horizontal_option).unwrap();
            let mut avg_horizontal = avg_horizontal_option.or(avg_vertical_option).unwrap();

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

    fn estimate_dir_average(dc_estimates: &[i16; 8], min_dc: &mut i16, max_dc: &mut i16) -> i32 {
        let mut dir_average: i32 = 0;
        for i in 0..8 {
            let cur_est = dc_estimates[i];
            dir_average += cur_est as i32;
        }

        // compiler vectorizes this using pminsw and pmaxsw, so no need to optimize further
        *min_dc = min(
            *min_dc,
            min(
                min(
                    min(dc_estimates[0], dc_estimates[1]),
                    min(dc_estimates[2], dc_estimates[3]),
                ),
                min(
                    min(dc_estimates[4], dc_estimates[5]),
                    min(dc_estimates[6], dc_estimates[7]),
                ),
            ),
        );
        *max_dc = max(
            *max_dc,
            max(
                max(
                    max(dc_estimates[0], dc_estimates[1]),
                    max(dc_estimates[2], dc_estimates[3]),
                ),
                max(
                    max(dc_estimates[4], dc_estimates[5]),
                    max(dc_estimates[6], dc_estimates[7]),
                ),
            ),
        );

        return dir_average;
    }
}
