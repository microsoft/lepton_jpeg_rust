/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::num::Wrapping;

use wide::{i16x8, i32x8};

use super::block_based_image::AlignedBlock;
use crate::consts::X_IDCT_SCALE;
use crate::enabled_features::EnabledFeatures;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NeighborSummary {
    edge_pixels_h: i16x8,
    edge_pixels_v: i16x8,

    edge_coefs_h: i32x8,
    edge_coefs_v: i32x8,

    num_non_zeros: u8,
}

pub static NEIGHBOR_DATA_EMPTY: NeighborSummary = NeighborSummary {
    edge_pixels_h: i16x8::ZERO,
    edge_pixels_v: i16x8::ZERO,
    edge_coefs_h: i32x8::ZERO,
    edge_coefs_v: i32x8::ZERO,
    num_non_zeros: 0,
};

impl Default for NeighborSummary {
    fn default() -> Self {
        NEIGHBOR_DATA_EMPTY
    }
}

impl NeighborSummary {
    pub fn new(
        here_idct: &AlignedBlock,
        dc_deq: i32,
        num_non_zeros_7x7: u8,
        horiz_pred: i32x8,
        vert_pred: i32x8,
        features: &EnabledFeatures,
    ) -> Self {
        NeighborSummary {
            edge_pixels_h: Self::set_horizontal(here_idct, dc_deq, features),
            edge_pixels_v: Self::set_vertical(here_idct, dc_deq, features),
            edge_coefs_h: horiz_pred,
            edge_coefs_v: vert_pred,
            num_non_zeros: num_non_zeros_7x7,
        }
    }

    pub fn get_num_non_zeros(&self) -> u8 {
        self.num_non_zeros
    }

    pub fn get_vertical_pix(&self) -> i16x8 {
        return self.edge_pixels_v;
    }

    pub fn get_horizontal_pix(&self) -> i16x8 {
        return self.edge_pixels_h;
    }

    fn set_pixel_pred(curr: i16x8, prev: i16x8, dc_deq: i32, features: &EnabledFeatures) -> i16x8 {
        // Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
        if features.use_16bit_dc_estimate {
            let delta = curr - prev;
            // ((delta - (delta >> 15)) >> 1) = delta / 2
            curr + (dc_deq + 128 * X_IDCT_SCALE) as i16 + ((delta - (delta >> 15)) >> 1)
        } else {
            let curr = i32x8::from_i16x8(curr);
            let prev = i32x8::from_i16x8(prev);
            let delta = curr - prev;
            // ((delta - (delta >> 31)) >> 1) = delta / 2
            i16x8::from_i32x8_truncate(
                curr + (dc_deq + 128 * X_IDCT_SCALE) + ((delta - (delta >> 31)) >> 1),
            )
        }
    }

    fn set_horizontal(here_idct: &AlignedBlock, dc_deq: i32, features: &EnabledFeatures) -> i16x8 {
        let curr = here_idct.from_stride(56, 1);
        let prev = here_idct.from_stride(48, 1);

        Self::set_pixel_pred(curr, prev, dc_deq, features)
    }

    fn set_vertical(here_idct: &AlignedBlock, dc_deq: i32, features: &EnabledFeatures) -> i16x8 {
        let curr = here_idct.from_stride(7, 8);
        let prev = here_idct.from_stride(6, 8);

        Self::set_pixel_pred(curr, prev, dc_deq, features)
    }

    pub fn get_vertical_coef(&self) -> i32x8 {
        return self.edge_coefs_v;
    }

    pub fn get_horizontal_coef(&self) -> i32x8 {
        return self.edge_coefs_h;
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn checksum(&self) -> u32 {
        let mut sum: Wrapping<u32> =
            Wrapping(i32x8::from_i16x8(self.edge_pixels_h).reduce_add() as u32);
        sum += Wrapping(i32x8::from_i16x8(self.edge_pixels_v).reduce_add() as u32);
        sum += Wrapping(self.num_non_zeros as u32);
        return sum.0;
    }
}
