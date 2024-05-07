/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::*;
use crate::helpers::*;

use super::jpeg_header::JPegHeader;

pub struct QuantizationTables {
    icos_idct_edge8192_dequantized_x: [i32; 64],
    icos_idct_edge8192_dequantized_y: [i32; 64],
    quantization_table: [u16; 64],
    freq_max: [u16; 64],
    min_noise_threshold: [u8; 64],
}

impl QuantizationTables {
    pub fn new(jpeg_header: &JPegHeader, component: usize) -> Self {
        Self::new_from_table(
            &jpeg_header.q_tables[usize::from(jpeg_header.cmp_info[component].q_table_index)],
        )
    }

    pub fn new_from_table(quantization_table: &[u16; 64]) -> Self {
        let mut retval = QuantizationTables {
            icos_idct_edge8192_dequantized_x: [0; 64],
            icos_idct_edge8192_dequantized_y: [0; 64],
            quantization_table: [0; 64],
            freq_max: [0; 64],
            min_noise_threshold: [0; 64],
        };

        retval.set_quantization_table(quantization_table);

        return retval;
    }

    fn set_quantization_table(&mut self, quantization_table: &[u16; 64]) {
        for i in 0..64 {
            self.quantization_table[i] = quantization_table[RASTER_TO_ZIGZAG[i] as usize];
        }

        for pixel_row in 0..8 {
            for i in 0..8 {
                self.icos_idct_edge8192_dequantized_x[(pixel_row * 8) + i] = ICOS_BASED_8192_SCALED
                    [i]
                    * (self.quantization_table[(i * 8) + pixel_row] as i32);
                self.icos_idct_edge8192_dequantized_y[(pixel_row * 8) + i] = ICOS_BASED_8192_SCALED
                    [i]
                    * (self.quantization_table[(pixel_row * 8) + i] as i32);
            }
        }

        for coord in 0..64 {
            self.freq_max[coord] = FREQ_MAX[coord] + self.quantization_table[coord] - 1;
            if self.quantization_table[coord] != 0 {
                self.freq_max[coord] /= self.quantization_table[coord];
            }

            let max_len = u16_bit_length(self.freq_max[coord]) as u8;
            if max_len > RESIDUAL_NOISE_FLOOR as u8 {
                self.min_noise_threshold[coord] = max_len - RESIDUAL_NOISE_FLOOR as u8;
            }
        }
    }

    pub fn get_icos_idct_edge8192_dequantized_x(&self) -> &[i32] {
        &self.icos_idct_edge8192_dequantized_x
    }

    pub fn get_icos_idct_edge8192_dequantized_y(&self) -> &[i32] {
        &self.icos_idct_edge8192_dequantized_y
    }

    pub fn get_quantization_table(&self) -> &[u16; 64] {
        &self.quantization_table
    }

    pub fn get_min_noise_threshold(&self, coef: usize) -> u8 {
        self.min_noise_threshold[coef]
    }
}
