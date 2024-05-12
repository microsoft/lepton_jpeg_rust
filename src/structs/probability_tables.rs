/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::cmp;

use crate::consts::*;
use crate::enabled_features;
use crate::helpers::*;
use crate::structs::idct::*;
use crate::structs::model::*;
use crate::structs::quantization_tables::*;

use super::block_based_image::AlignedBlock;
use super::block_context::NeighborData;
use super::neighbor_summary::NeighborSummary;
use super::probability_tables_coefficient_context::ProbabilityTablesCoefficientContext;

use wide::i16x8;
use wide::i32x8;

use bytemuck::cast;

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
    pub advanced_predict_dc_pixels_sans_dc: AlignedBlock,
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

    pub fn num_non_zeros_to_bin_7x7(num_non_zeros: usize) -> usize {
        return usize::from(NON_ZERO_TO_BIN_7X7[num_non_zeros]);
    }

    pub fn calc_num_non_zeros_7x7_context_bin<const ALL_PRESENT: bool>(
        &self,
        neighbor_data: &NeighborData,
    ) -> u8 {
        let mut num_non_zeros_above = 0;
        let mut num_non_zeros_left = 0;
        if ALL_PRESENT || self.above_present {
            num_non_zeros_above = neighbor_data.neighbor_context_above.get_num_non_zeros();
        }

        if ALL_PRESENT || self.left_present {
            num_non_zeros_left = neighbor_data.neighbor_context_left.get_num_non_zeros();
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

        return NON_ZERO_TO_BIN[usize::from(num_non_zeros_context)];
    }

    // calculates the average of the prior values from their corresponding value in the left, above and above/left block
    // the C++ version does one coefficient at a time, but if we do it all at the same time, the compiler vectorizes everything
    #[inline(never)]
    pub fn calc_coefficient_context_7x7_aavg_block<const ALL_PRESENT: bool>(
        &self,
        left: &AlignedBlock,
        above: &AlignedBlock,
        above_left: &AlignedBlock,
    ) -> [i16; 64] {
        let mut best_prior = [0; 64];

        if ALL_PRESENT {
            // compiler does a pretty amazing job with SSE/AVX2 here
            for i in 8..64 {
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
                for i in 8..64 {
                    best_prior[i] = left.get_coefficient(i).abs();
                }
            } else if self.above_present {
                for i in 8..64 {
                    best_prior[i] = above.get_coefficient(i).abs();
                }
            }
        }

        return best_prior;
    }

    #[inline(always)]
    pub fn calc_coefficient_context8_decode_lak<const ALL_PRESENT: bool, const HORIZONTAL: bool>(
        &self,
        qt: &QuantizationTables,
        coefficient_tr: usize,
        pred: &[i32; 8],
        num_non_zeros_x: u8,
    ) -> ProbabilityTablesCoefficientContext {
        if !ALL_PRESENT
            && ((HORIZONTAL && !self.above_present) || (!HORIZONTAL && !self.left_present))
        {
            return ProbabilityTablesCoefficientContext {
                best_prior: 0,
                num_non_zeros_bin: num_non_zeros_x - 1,
                best_prior_bit_len: 0,
            };
        }

        let mut best_prior: i32 = pred[if HORIZONTAL {
            coefficient_tr >> 3
        } else {
            coefficient_tr
        }];
        best_prior /= (qt.get_quantization_table_transposed()[coefficient_tr] as i32) << 13;

        return ProbabilityTablesCoefficientContext {
            best_prior,
            num_non_zeros_bin: num_non_zeros_x - 1,
            best_prior_bit_len: u32_bit_length(cmp::min(best_prior.unsigned_abs(), 1023)),
        };
    }

    #[inline(always)]
    pub fn calc_coefficient_context8_lak<const ALL_PRESENT: bool, const HORIZONTAL: bool>(
        &self,
        qt: &QuantizationTables,
        coefficient: usize,
        here: &AlignedBlock,
        neighbors_data: &NeighborData,
        num_non_zeros_edge: u8,
    ) -> ProbabilityTablesCoefficientContext {
        let mut compute_lak_coeffs_x: [i32; 8] = [0; 8];

        let coef_idct; // have 0 in 0th element
        if HORIZONTAL && (ALL_PRESENT || self.above_present) {
            assert!(coefficient <= 56); // avoid bounds check later

            // y == 0: we're the x

            coef_idct =
                &qt.get_icos_idct_edge8192_dequantized_x()[coefficient..coefficient + 8];
            //coef_idct = i32x8::new(qt.get_icos_idct_edge8192_dequantized_x()[coefficient * 8..(coefficient + 1) * 8].try_into().unwrap());

            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 1..8 {
                let cur_coef = coefficient + i;

                compute_lak_coeffs_x[i] = here.get_coefficient(cur_coef).into();
            }
        } else if !HORIZONTAL && (ALL_PRESENT || self.left_present) {
            assert!(coefficient < 8); // avoid bounds check later

            // x == 0: we're the y

            coef_idct = &qt.get_icos_idct_edge8192_dequantized_y()[coefficient * 8..(coefficient + 1) * 8];
            //coef_idct = i32x8::new(qt.get_icos_idct_edge8192_dequantized_y()[coefficient..coefficient + 8].try_into().unwrap());

            // the compiler is smart enough to unroll this loop and merge it with the subsequent loop
            // so no need to complicate the code by doing anything manual

            for i in 1..8 {
                let cur_coef = coefficient + (i * 8);

                compute_lak_coeffs_x[i] = here.get_coefficient(cur_coef).into();
            }
        } else {
            return ProbabilityTablesCoefficientContext {
                best_prior: 0,
                num_non_zeros_bin: num_non_zeros_edge - 1,
                best_prior_bit_len: 0,
            };
        }

        let mut best_prior: i32 = if HORIZONTAL {
            neighbors_data.neighbor_context_above.get_horizontal_coef()[coefficient >> 3]
        } else {
            neighbors_data.neighbor_context_left.get_vertical_coef()[coefficient]
        };
        // // some extreme coefficents can cause this to overflow, but since this is just a predictor, no need to panic
        // best_prior = best_prior.wrapping_sub((coef_idct * i32x8::new(compute_lak_coeffs_x)).reduce_add());
        // // rounding towards zero before adding coeffs_a[0] helps ratio slightly, but this is cheaper
        // //best_prior /= (qt.get_quantization_table()[coefficient] as i32) << 13;
        // best_prior /= coef_idct.to_array()[0];

        for i in 1..8 {
            // some extreme coefficents can cause this to overflow, but since this is just a predictor, no need to panic
            best_prior =
                best_prior.wrapping_sub(coef_idct[i].wrapping_mul(compute_lak_coeffs_x[i]));
            // rounding towards zero before adding coeffs_a[0] helps ratio slightly, but this is cheaper
        }
        best_prior /= coef_idct[0];

        return ProbabilityTablesCoefficientContext {
            best_prior,
            num_non_zeros_bin: num_non_zeros_edge - 1,
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

    pub fn adv_predict_dc_pix_decode<const ALL_PRESENT: bool>(
        &self,
        raster_cols: &[i32x8; 8],
        q0: i32,
        neighbor_data: &NeighborData,
        enabled_features: &enabled_features::EnabledFeatures,
    ) -> PredictDCResult {
        let pixels_sans_dc = run_idct_decode(raster_cols);

        // helper functions to avoid code duplication that calculate the left and above prediction values

        let calc_left = || {
            let left_context = neighbor_data.neighbor_context_left;

            if enabled_features.use_16bit_adv_predict {
                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 0, 8);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 1, 8);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*left_context.get_vertical()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                b - a
            } else {
                let mut dc_estimates = [0i16; 8];
                for i in 0..8 {
                    let a = i32::from(pixels_sans_dc.get_block()[i << 3]) + 1024;
                    let pixel_delta = i32::from(pixels_sans_dc.get_block()[i << 3])
                        - i32::from(pixels_sans_dc.get_block()[(i << 3) + 1]);
                    let b = i32::from(left_context.get_vertical()[i]) - (pixel_delta / 2); //round to zero
                    dc_estimates[i] = (b - a) as i16;
                }

                i16x8::from(dc_estimates)
            }
        };

        let calc_above = || {
            let above_context = neighbor_data.neighbor_context_above;

            if enabled_features.use_16bit_adv_predict {
                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 0, 1);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 8, 1);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*above_context.get_horizontal()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                b - a
            } else {
                let mut dc_estimates = [0i16; 8];
                for i in 0..8 {
                    let a = i32::from(pixels_sans_dc.get_block()[i]) + 1024;
                    let pixel_delta = i32::from(pixels_sans_dc.get_block()[i])
                        - i32::from(pixels_sans_dc.get_block()[i + 8]);
                    let b = i32::from(above_context.get_horizontal()[i]) - (pixel_delta / 2); //round to zero
                    dc_estimates[i] = (b - a) as i16;
                }

                i16x8::from(dc_estimates)
            }
        };

        let min_dc;
        let max_dc;
        let mut avg_horizontal: i32;
        let mut avg_vertical: i32;

        if ALL_PRESENT || self.left_present {
            if ALL_PRESENT || self.above_present {
                // most common case where we have both left and above
                let horiz = calc_left();
                let vert = calc_above();

                min_dc = horiz.min(vert).reduce_min();
                max_dc = horiz.max(vert).reduce_max();

                avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
                avg_vertical = i32x8::from_i16x8(vert).reduce_add();
            } else {
                let horiz = calc_left();
                min_dc = horiz.reduce_min();
                max_dc = horiz.reduce_max();

                avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
                avg_vertical = avg_horizontal;
            }
        } else if self.above_present {
            let vert = calc_above();
            min_dc = vert.reduce_min();
            max_dc = vert.reduce_max();

            avg_vertical = i32x8::from_i16x8(vert).reduce_add();
            avg_horizontal = avg_vertical;
        } else {
            return PredictDCResult {
                predicted_dc: 0,
                uncertainty: 0,
                uncertainty2: 0,
                advanced_predict_dc_pixels_sans_dc: pixels_sans_dc,
            };
        }

        let avgmed: i32 = (avg_vertical + avg_horizontal) >> 1;
        let uncertainty_val = ((i32::from(max_dc) - i32::from(min_dc)) >> 3) as i16;
        avg_horizontal -= avgmed;
        avg_vertical -= avgmed;

        let mut far_afield_value = avg_vertical;
        if avg_horizontal.abs() < avg_vertical.abs() {
            far_afield_value = avg_horizontal;
        }

        let uncertainty2_val = (far_afield_value >> 3) as i16;

        return PredictDCResult {
            predicted_dc: (avgmed / q0 + 4) >> 3,
            uncertainty: uncertainty_val,
            uncertainty2: uncertainty2_val,
            advanced_predict_dc_pixels_sans_dc: pixels_sans_dc,
        };
    }

    pub fn adv_predict_dc_pix<const ALL_PRESENT: bool>(
        &self,
        here: &AlignedBlock,
        qt: &QuantizationTables,
        neighbor_data: &NeighborData,
        enabled_features: &enabled_features::EnabledFeatures,
    ) -> (PredictDCResult, NeighborSummary) {
        let q_tr: AlignedBlock = AlignedBlock::new(cast(*qt.get_quantization_table_transposed()));

        let (pixels_sans_dc, summary) = run_idct::<true>(here, &q_tr);

        // helper functions to avoid code duplication that calculate the left and above prediction values

        let calc_left = || {
            let left_context = neighbor_data.neighbor_context_left;

            if enabled_features.use_16bit_adv_predict {
                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 0, 8);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 1, 8);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*left_context.get_vertical()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                b - a
            } else {
                let mut dc_estimates = [0i16; 8];
                for i in 0..8 {
                    let a = i32::from(pixels_sans_dc.get_block()[i << 3]) + 1024;
                    let pixel_delta = i32::from(pixels_sans_dc.get_block()[i << 3])
                        - i32::from(pixels_sans_dc.get_block()[(i << 3) + 1]);
                    let b = i32::from(left_context.get_vertical()[i]) - (pixel_delta / 2); //round to zero
                    dc_estimates[i] = (b - a) as i16;
                }

                i16x8::from(dc_estimates)
            }
        };

        let calc_above = || {
            let above_context = neighbor_data.neighbor_context_above;

            if enabled_features.use_16bit_adv_predict {
                let a1 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 0, 1);
                let a2 = ProbabilityTables::from_stride(&pixels_sans_dc.get_block(), 8, 1);
                let pixel_delta = a1 - a2;
                let a: i16x8 = a1 + 1024;
                let b : i16x8 = i16x8::new(*above_context.get_horizontal()) - (pixel_delta - (pixel_delta>>15) >> 1) /* divide pixel_delta by 2 rounding towards 0 */;

                b - a
            } else {
                let mut dc_estimates = [0i16; 8];
                for i in 0..8 {
                    let a = i32::from(pixels_sans_dc.get_block()[i]) + 1024;
                    let pixel_delta = i32::from(pixels_sans_dc.get_block()[i])
                        - i32::from(pixels_sans_dc.get_block()[i + 8]);
                    let b = i32::from(above_context.get_horizontal()[i]) - (pixel_delta / 2); //round to zero
                    dc_estimates[i] = (b - a) as i16;
                }

                i16x8::from(dc_estimates)
            }
        };

        let min_dc;
        let max_dc;
        let mut avg_horizontal: i32;
        let mut avg_vertical: i32;

        if ALL_PRESENT || self.left_present {
            if ALL_PRESENT || self.above_present {
                // most common case where we have both left and above
                let horiz = calc_left();
                let vert = calc_above();

                min_dc = horiz.min(vert).reduce_min();
                max_dc = horiz.max(vert).reduce_max();

                avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
                avg_vertical = i32x8::from_i16x8(vert).reduce_add();
            } else {
                let horiz = calc_left();
                min_dc = horiz.reduce_min();
                max_dc = horiz.reduce_max();

                avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
                avg_vertical = avg_horizontal;
            }
        } else if self.above_present {
            let vert = calc_above();
            min_dc = vert.reduce_min();
            max_dc = vert.reduce_max();

            avg_vertical = i32x8::from_i16x8(vert).reduce_add();
            avg_horizontal = avg_vertical;
        } else {
            return (
                PredictDCResult {
                    predicted_dc: 0,
                    uncertainty: 0,
                    uncertainty2: 0,
                    advanced_predict_dc_pixels_sans_dc: pixels_sans_dc,
                },
                summary,
            );
        }

        let avgmed: i32 = (avg_vertical + avg_horizontal) >> 1;
        let uncertainty_val = ((i32::from(max_dc) - i32::from(min_dc)) >> 3) as i16;
        avg_horizontal -= avgmed;
        avg_vertical -= avgmed;

        let mut far_afield_value = avg_vertical;
        if avg_horizontal.abs() < avg_vertical.abs() {
            far_afield_value = avg_horizontal;
        }

        let uncertainty2_val = (far_afield_value >> 3) as i16;

        return (
            PredictDCResult {
                predicted_dc: ((avgmed / i32::from(q_tr.get_coefficient(0))) + 4) >> 3,
                uncertainty: uncertainty_val,
                uncertainty2: uncertainty2_val,
                advanced_predict_dc_pixels_sans_dc: pixels_sans_dc,
            },
            summary,
        );
    }
}
