/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::cmp;

use crate::structs::component_info::*;
use crate::structs::jpeg_header::JPegHeader;

#[derive(Debug, Clone)]
struct TrucateComponentsInfo {
    trunc_bcv: i32, // the number of vertical components in this (truncated) image

    trunc_bc: i32,
}

#[derive(Debug, Clone)]
pub struct TruncateComponents {
    trunc_info: Vec<TrucateComponentsInfo>,

    pub components_count: usize,

    pub mcu_count_horizontal: i32,

    pub mcu_count_vertical: i32,
}

impl Default for TruncateComponents {
    fn default() -> Self {
        return TruncateComponents {
            trunc_info: Vec::new(),
            components_count: 0,
            mcu_count_horizontal: 0,
            mcu_count_vertical: 0,
        };
    }
}

impl TruncateComponents {
    pub fn init(&mut self, jpeg_header: &JPegHeader) {
        self.mcu_count_horizontal = jpeg_header.mcuh;
        self.mcu_count_vertical = jpeg_header.mcuv;
        self.components_count = jpeg_header.cmpc;

        for i in 0..jpeg_header.cmpc {
            self.trunc_info.push(TrucateComponentsInfo {
                trunc_bcv: jpeg_header.cmp_info[i].bcv,
                trunc_bc: jpeg_header.cmp_info[i].bc,
            });
        }
    }

    pub fn get_max_coded_heights(&self) -> Vec<u32> {
        let mut retval = Vec::<u32>::new();

        for i in 0..self.components_count {
            retval.push(self.trunc_info[i].trunc_bcv as u32);
        }
        return retval;
    }

    pub fn set_truncation_bounds(&mut self, jpeg_header: &JPegHeader, max_d_pos: [i32; 4]) {
        for i in 0..self.components_count {
            TruncateComponents::set_block_count_d_pos(
                &mut self.trunc_info[i],
                &jpeg_header.cmp_info[i],
                max_d_pos[i] + 1,
                self.mcu_count_vertical,
            );
        }
    }

    pub fn get_block_height(&self, cmp: usize) -> i32 {
        return self.trunc_info[cmp].trunc_bcv;
    }

    fn set_block_count_d_pos(
        ti: &mut TrucateComponentsInfo,
        ci: &ComponentInfo,
        trunc_bc: i32,
        mcu_count_vertical: i32,
    ) {
        assert!(
            ci.bcv == (ci.bc / ci.bch) + (if ci.bc % ci.bch != 0 { 1 } else { 0 }),
            "SetBlockCountDpos"
        );

        let mut vertical_scan_lines = cmp::min(
            (trunc_bc / ci.bch) + (if trunc_bc % ci.bch != 0 { 1 } else { 0 }),
            ci.bcv,
        );
        let ratio = TruncateComponents::get_min_vertical_extcmp_multiple(&ci, mcu_count_vertical);

        while vertical_scan_lines % ratio != 0 && vertical_scan_lines + 1 <= ci.bcv {
            vertical_scan_lines += 1;
        }

        assert!(
            vertical_scan_lines <= ci.bcv,
            "verticalScanLines <= ci.Info.bcv"
        );
        ti.trunc_bcv = vertical_scan_lines;
        ti.trunc_bc = trunc_bc;
    }

    fn get_min_vertical_extcmp_multiple(cmp_info: &ComponentInfo, mcu_count_vertical: i32) -> i32 {
        let luma_height = cmp_info.bcv;
        return luma_height / mcu_count_vertical;
    }

    pub fn get_component_sizes_in_blocks(&self) -> Vec<i32> {
        let mut retval = Vec::new();
        for i in 0..self.components_count {
            retval.push(self.trunc_info[i].trunc_bc);
        }
        return retval;
    }
}
