/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::num::Wrapping;

use crate::enabled_features::EnabledFeatures;

use super::{block_based_image::AlignedBlock, quantization_tables::QuantizationTables};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NeighborSummary {
    edge_pixels_h: [i16; 8],

    edge_pixels_v: [i16; 8],

    num_non_zeros: u8,
}

pub static NEIGHBOR_DATA_EMPTY: NeighborSummary = NeighborSummary {
    edge_pixels_h: [0; 8],
    edge_pixels_v: [0; 8],
    num_non_zeros: 0,
};

const X_IDCT_SCALE: i32 = 8;

impl NeighborSummary {
    pub fn new() -> Self {
        return NeighborSummary {
            edge_pixels_h: [0; 8],
            edge_pixels_v: [0; 8],
            num_non_zeros: 0,
        };
    }

    pub fn get_num_non_zeros(&self) -> u8 {
        self.num_non_zeros
    }

    pub fn get_vertical(&self) -> &[i16; 8] {
        return &self.edge_pixels_v;
    }

    pub fn get_horizontal(&self) -> &[i16; 8] {
        return &self.edge_pixels_h;
    }

    pub fn calculate_neighbor_summary(
        here_idct: &AlignedBlock,
        qt: &QuantizationTables,
        here: &AlignedBlock,
        num_non_zeros_7x7: u8,
        features: &EnabledFeatures,
    ) -> NeighborSummary {
        let mut neighbor_summary = NeighborSummary::new();

        neighbor_summary.set_horizontal(
            here_idct.get_block(),
            qt.get_quantization_table(),
            here.get_dc(),
            features,
        );

        neighbor_summary.set_vertical(
            here_idct.get_block(),
            qt.get_quantization_table(),
            here.get_dc(),
            features,
        );

        neighbor_summary.num_non_zeros = num_non_zeros_7x7;

        neighbor_summary
    }

    fn set_horizontal(
        &mut self,
        data: &[i16; 64],
        qt: &[u16; 64],
        dc: i16,
        enabled_features: &EnabledFeatures,
    ) {
        if enabled_features.use_16bit_dc_estimate {
            // Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
            for i in 0..8 {
                let delta = data[i + 56].wrapping_sub(data[i + 48]);
                self.edge_pixels_h[i] = ((dc as i32 * qt[0] as i32) as i16)
                    .wrapping_add(data[i + 56])
                    .wrapping_add(128 * X_IDCT_SCALE as i16)
                    .wrapping_add(delta / 2);
            }
        } else {
            for i in 0..8 {
                let delta = data[i + 56] as i32 - data[i + 48] as i32;
                self.edge_pixels_h[i] = ((dc as i32 * qt[0] as i32)
                    + data[i + 56] as i32
                    + (128 * X_IDCT_SCALE)
                    + (delta / 2)) as i16;
            }
        }
    }

    fn set_vertical(
        &mut self,
        data: &[i16; 64],
        qt: &[u16; 64],
        dc: i16,
        features: &EnabledFeatures,
    ) {
        if features.use_16bit_dc_estimate {
            // Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
            for i in 0..8 {
                let delta: i16 = data[(i * 8) + 7].wrapping_sub(data[(i * 8) + 6]);
                self.edge_pixels_v[i] = ((dc as i32 * qt[0] as i32) as i16)
                    .wrapping_add(data[(i * 8) + 7])
                    .wrapping_add((128 * X_IDCT_SCALE) as i16)
                    .wrapping_add(delta / 2);
            }
        } else {
            for i in 0..8 {
                let delta = data[(i * 8) + 7] as i32 - data[(i * 8) + 6] as i32;
                self.edge_pixels_v[i] = ((dc as i32 * qt[0] as i32)
                    + data[(i * 8) + 7] as i32
                    + (128 * X_IDCT_SCALE)
                    + (delta / 2)) as i16;
            }
        }
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn checksum(&self) -> u32 {
        let mut sum: Wrapping<u32> = Wrapping(0u32);
        for i in 0..self.edge_pixels_h.len() {
            sum += Wrapping(self.edge_pixels_h[i] as u32);
        }
        for i in 0..self.edge_pixels_v.len() {
            sum += Wrapping(self.edge_pixels_v[i] as u32);
        }
        sum += Wrapping(self.num_non_zeros as u32);
        return sum.0;
    }
}
