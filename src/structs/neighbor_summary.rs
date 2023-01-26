/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

use std::num::Wrapping;

#[derive(Copy, Clone)]
pub struct NeighborSummary {
    edge_pixels: [i16; 16],

    num_non_zeros: u8,
}

const X_IDCT_SCALE: i32 = 8;

impl NeighborSummary {
    pub fn new() -> Self {
        return NeighborSummary {
            edge_pixels: [0; 16],
            num_non_zeros: 0,
        };
    }

    pub fn get_num_non_zeros(&self) -> u8 {
        self.num_non_zeros
    }

    pub fn set_num_non_zeros(&mut self, v: u8) {
        self.num_non_zeros = v;
    }

    pub fn get_horizontal(&self, index: usize) -> i16 {
        return self.edge_pixels[index + 8];
    }

    pub fn get_vertical(&self, index: usize) -> i16 {
        return self.edge_pixels[index];
    }

    pub fn set_horizontal(&mut self, data: &[i16; 64], qt: &[u16; 64], dc: i16) {
        for i in 0..8 {
            let delta = data[i + 56] as i32 - data[i + 48] as i32;
            self.edge_pixels[i + 8] = ((dc as i32 * qt[0] as i32)
                + data[i + 56] as i32
                + (128 * X_IDCT_SCALE)
                + (delta / 2)) as i16;
        }
    }

    pub fn set_vertical(&mut self, data: &[i16; 64], qt: &[u16; 64], dc: i16) {
        for i in 0..8 {
            let delta = data[(i * 8) + 7] as i32 - data[(i * 8) + 6] as i32;
            self.edge_pixels[i] = ((dc as i32 * qt[0] as i32)
                + data[(i * 8) + 7] as i32
                + (128 * X_IDCT_SCALE)
                + (delta / 2)) as i16;
        }
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn checksum(&self) -> u32 {
        let mut sum: Wrapping<u32> = Wrapping(0u32);
        for i in 0..self.edge_pixels.len() {
            sum += Wrapping(self.edge_pixels[i] as u32);
        }
        sum += Wrapping(self.num_non_zeros as u32);
        return sum.0;
    }
}
