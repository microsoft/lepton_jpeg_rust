/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::*;
use crate::enabled_features;
use crate::structs::idct::*;
use crate::structs::model::*;
use crate::structs::quantization_tables::*;

use super::block_based_image::AlignedBlock;
use super::block_context::NeighborData;

use wide::i16x8;
use wide::i32x8;

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
    ) -> [u16; 64] {
        let mut best_prior = [0; 64];

        if ALL_PRESENT {
            // compiler does a pretty amazing job with SSE/AVX2 here
            for i in 8..64 {
                // approximate average of 3 without a divide with double the weight for left/top vs diagonal
                //
                // No need to go to 32 bits since max exponent is 11, ie 2047, so
                // (2047 + 2047) * 13 + 2047 * 6 = 65504 which still fits in 16 bits.
                // In addition, if we ever returned anything higher that 2047, it would
                // assert in the array lookup in the model.
                best_prior[i] = ((left.get_coefficient(i).unsigned_abs()
                    + above.get_coefficient(i).unsigned_abs())
                    * 13
                    + 6 * above_left.get_coefficient(i).unsigned_abs())
                    >> 5;
            }
        } else {
            // handle edge case :) where we are on the top or left edge

            if self.left_present {
                for i in 8..64 {
                    best_prior[i] = left.get_coefficient(i).unsigned_abs();
                }
            } else if self.above_present {
                for i in 8..64 {
                    best_prior[i] = above.get_coefficient(i).unsigned_abs();
                }
            }
        }

        best_prior
    }

    // Predictor calculations in `compute_lak` are made using partial IDCT along only one dimension
    // on neighbor and current blocks row/column and finding predictor that makes current block edge
    // "almost-pixel" equal to that of neighbor block (see https://arxiv.org/abs/1704.06192, section A.2.2).
    // These 1D IDCT can be conveniently done separately for current block and neighbor one
    // storing components of predictor formula - dot products of dequantized DCT coefficients columns/rows
    // with `ICOS_BASED_8192_SCALED/_PM` (equivalent to former dot products of quantized DCT coefficients
    // with `icos_idct_edge_8192_dequantized_x/y`) - inside `NeighborSummary` of corresponding block.
    // Instead of non-continuous memory accesses to blocks we can use dequantized raster DCT coefficients
    // needed for DC prediction and apply horizontal SIMD instructions for direction along the raster order.

    // Produce current block predictors for edge DCT coefficients
    #[inline(always)]
    pub fn predict_current_edges(
        neighbors_data: &NeighborData,
        raster: &[i32x8; 8],
    ) -> (i32x8, i32x8) {
        // don't bother about DC in encoding - 0th component of ICOS_BASED_8192_SCALED is 0
        let mult: i32x8 = i32x8::from(ICOS_BASED_8192_SCALED);

        // load initial predictors data from neighborhood blocks
        let mut horiz_pred: [i32; 8] = neighbors_data
            .neighbor_context_above
            .get_horizontal_coef()
            .to_array();
        let mut vert_pred: i32x8 = neighbors_data.neighbor_context_left.get_vertical_coef();

        for col in 1..8 {
            // some extreme coefficents can cause overflows, but since this is just predictors, no need to panic
            vert_pred -= raster[col] * ICOS_BASED_8192_SCALED[col];
            horiz_pred[col] = horiz_pred[col].wrapping_sub((raster[col] * mult).reduce_add());
        }

        (i32x8::from(horiz_pred), vert_pred)
    }

    // Produce first part of edge DCT coefficients predictions for neighborhood blocks
    #[inline(always)]
    pub fn predict_next_edges(raster: &[i32x8; 8]) -> (i32x8, i32x8) {
        let mult = i32x8::from(ICOS_BASED_8192_SCALED_PM);

        let mut horiz_pred: [i32; 8] = [0; 8];
        let mut vert_pred = ICOS_BASED_8192_SCALED_PM[0] * raster[0];
        for col in 1..8 {
            // produce predictions for edge DCT coefficientss for the block below
            horiz_pred[col] = (mult * raster[col]).reduce_add();
            // and for the block to the right
            vert_pred += ICOS_BASED_8192_SCALED_PM[col] * raster[col];
        }

        (i32x8::from(horiz_pred), vert_pred)
    }

    #[inline(always)]
    pub fn calc_coefficient_context8_lak<const ALL_PRESENT: bool, const HORIZONTAL: bool>(
        &self,
        qt: &QuantizationTables,
        coefficient_tr: usize,
        pred: &[i32; 8],
    ) -> i32 {
        if !ALL_PRESENT
            && ((HORIZONTAL && !self.above_present) || (!HORIZONTAL && !self.left_present))
        {
            return 0;
        }

        let mut best_prior: i32 = pred[if HORIZONTAL {
            coefficient_tr >> 3
        } else {
            coefficient_tr
        }];
        best_prior /= (qt.get_quantization_table_transposed()[coefficient_tr] as i32) << 13;

        best_prior
    }

    pub fn adv_predict_dc_pix<const ALL_PRESENT: bool>(
        &self,
        raster_cols: &[i32x8; 8],
        q0: i32,
        neighbor_data: &NeighborData,
        enabled_features: &enabled_features::EnabledFeatures,
    ) -> PredictDCResult {
        // here DC in raster_cols should be 0
        let pixels_sans_dc = run_idct(raster_cols);

        // helper functions to avoid code duplication that calculate the left and above prediction values

        let calc_pred = |init_pred: i16x8, a1: i16x8, a2: i16x8| {
            if enabled_features.use_16bit_adv_predict {
                let pixel_delta = a1 - a2;
                let half_delta = (pixel_delta - (pixel_delta >> 15)) >> 1; /* divide pixel_delta by 2 rounding towards 0 */

                init_pred - a1 - 128 * X_IDCT_SCALE as i16 - half_delta
            } else {
                let a1 = i32x8::from_i16x8(a1);
                let a2 = i32x8::from_i16x8(a2);
                let pixel_delta = a1 - a2;
                let half_delta = (pixel_delta - (pixel_delta >> 31)) >> 1; /* divide pixel_delta by 2 rounding towards 0 */
                let result = i32x8::from_i16x8(init_pred) - a1 - 128 * X_IDCT_SCALE - half_delta;

                i16x8::from_i32x8_truncate(result)
            }
        };

        let calc_left = || {
            let left_pred = neighbor_data.neighbor_context_left.get_vertical_pix();
            let a1 = pixels_sans_dc.from_stride(0, 8);
            let a2 = pixels_sans_dc.from_stride(1, 8);

            calc_pred(left_pred, a1, a2)
        };

        let calc_above = || {
            let above_pred = neighbor_data.neighbor_context_above.get_horizontal_pix();
            let a1 = pixels_sans_dc.from_stride(0, 1);
            let a2 = pixels_sans_dc.from_stride(8, 1);

            calc_pred(above_pred, a1, a2)
        };

        let min_dc;
        let max_dc;
        let mut avg_horizontal: i32;
        let mut avg_vertical: i32;

        if ALL_PRESENT {
            // most common case where we have both left and above
            let horiz = calc_left();
            let vert = calc_above();

            min_dc = horiz.min(vert).reduce_min();
            max_dc = horiz.max(vert).reduce_max();

            avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
            avg_vertical = i32x8::from_i16x8(vert).reduce_add();
        } else if self.left_present {
            let horiz = calc_left();
            min_dc = horiz.reduce_min();
            max_dc = horiz.reduce_max();

            avg_horizontal = i32x8::from_i16x8(horiz).reduce_add();
            avg_vertical = avg_horizontal;
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
}
