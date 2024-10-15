/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::num::Wrapping;

use wide::{i16x8, i32x8};

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
        edge_pixels_h: i16x8,
        edge_pixels_v: i16x8,
        dc_deq: i32,
        num_non_zeros_7x7: u8,
        horiz_pred: i32x8,
        vert_pred: i32x8,
    ) -> Self {
        NeighborSummary {
            edge_pixels_h: edge_pixels_h + (dc_deq as i16),
            edge_pixels_v: edge_pixels_v + (dc_deq as i16),
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
