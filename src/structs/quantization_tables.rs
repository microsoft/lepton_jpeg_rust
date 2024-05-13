/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::*;
use crate::helpers::*;

use super::jpeg_header::JPegHeader;

pub struct QuantizationTables {
    quantization_table: [u16; 64],
    quantization_table_transposed: [u16; 64],
    min_noise_threshold: [u8; 14],
}

impl QuantizationTables {
    pub fn new(jpeg_header: &JPegHeader, component: usize) -> Self {
        Self::new_from_table(
            &jpeg_header.q_tables[usize::from(jpeg_header.cmp_info[component].q_table_index)],
        )
    }

    pub fn new_from_table(quantization_table: &[u16; 64]) -> Self {
        let mut retval = QuantizationTables {
            quantization_table: [0; 64],
            quantization_table_transposed: [0; 64],
            min_noise_threshold: [0; 14],
        };

        retval.set_quantization_table(quantization_table);

        return retval;
    }

    fn set_quantization_table(&mut self, quantization_table: &[u16; 64]) {
        for i in 0..64 {
            let q = quantization_table[RASTER_TO_ZIGZAG[i] as usize];
            self.quantization_table[i] = q;
        }

        for pixel_row in 0..8 {
            for i in 0..8 {
                let coord = (pixel_row * 8) + i;
                let coord_tr = (i * 8) + pixel_row;
                self.quantization_table_transposed[coord] = self.quantization_table[coord_tr];
            }
        }

        for i in 0..14 {
            let coord = if i < 7 { i + 1 } else { (i - 6) * 8 };
            let mut freq_max = FREQ_MAX[coord] + self.quantization_table[coord] - 1;
            if self.quantization_table[coord] != 0 {
                freq_max /= self.quantization_table[coord];
            }

            let max_len = u16_bit_length(freq_max) as u8;
            if max_len > RESIDUAL_NOISE_FLOOR as u8 {
                self.min_noise_threshold[i] = max_len - RESIDUAL_NOISE_FLOOR as u8;
            }
        }
    }

    pub fn get_quantization_table(&self) -> &[u16; 64] {
        &self.quantization_table
    }

    pub fn get_quantization_table_transposed(&self) -> &[u16; 64] {
        &self.quantization_table_transposed
    }

    pub fn get_min_noise_threshold(&self, coef: usize) -> u8 {
        self.min_noise_threshold[coef]
    }
}
