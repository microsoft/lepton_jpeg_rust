/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::COLOR_CHANNEL_NUM_BLOCK_TYPES;
use crate::structs::block_based_image::BlockBasedImage;

pub struct RowSpec {
    pub min_row_luma_y: i32,
    pub next_row_luma_y: i32,
    pub luma_y: i32,
    pub component: usize,
    pub curr_y: i32,
    pub mcu_row_index: i32,
    pub last_row_to_complete_mcu: bool,
    pub skip: bool,
    pub done: bool,
}

impl RowSpec {
    pub fn get_row_spec_from_index(
        decode_index: u32,
        image_data: &[BlockBasedImage],
        mcuv: i32, // number of mcus
        max_coded_heights: &[u32],
    ) -> RowSpec {
        assert!(
            image_data.len() <= COLOR_CHANNEL_NUM_BLOCK_TYPES,
            "image_data should match components count"
        );

        let num_cmp = image_data.len();

        let mut heights: Vec<u32> = Vec::with_capacity(num_cmp);
        let mut component_multiple: Vec<u32> = Vec::with_capacity(num_cmp);
        let mut mcu_multiple = 0;

        for i in 0..num_cmp {
            heights.push(image_data[i].get_original_height() as u32);
            component_multiple.push(heights[i] / mcuv as u32);
            mcu_multiple += component_multiple[i];
        }

        let mcu_row = decode_index / mcu_multiple;
        let min_row_luma_y = (mcu_row * component_multiple[0]) as i32;
        let mut retval = RowSpec {
            skip: false,
            done: false,
            mcu_row_index: mcu_row as i32,
            component: num_cmp,
            min_row_luma_y,
            next_row_luma_y: min_row_luma_y + component_multiple[0] as i32,
            luma_y: min_row_luma_y,
            curr_y: 0,
            last_row_to_complete_mcu: false,
        };

        let mut place_within_scan = decode_index - (mcu_row * mcu_multiple);

        let mut i = num_cmp - 1;
        loop {
            if place_within_scan < component_multiple[i] {
                retval.component = i;
                retval.curr_y = ((mcu_row * component_multiple[i]) + place_within_scan) as i32;
                retval.last_row_to_complete_mcu =
                    (place_within_scan + 1 == component_multiple[i]) && (i == 0);

                if retval.curr_y >= max_coded_heights[i] as i32 {
                    retval.skip = true;
                    retval.done = true; // assume true, but if we find something that needs coding, set false
                    for j in 0..num_cmp - 1 {
                        if mcu_row * component_multiple[j] < max_coded_heights[j] {
                            // we want to make sure to write out any partial rows,
                            // so set done only when all items in this mcu are really skips
                            // i.e. round down
                            retval.done = false;
                        }
                    }
                }

                if i == 0 {
                    retval.luma_y = retval.curr_y;
                }

                break;
            } else {
                place_within_scan -= component_multiple[i];
            }

            if i == 0 {
                retval.skip = true;
                retval.done = true;
                break;
            }

            i -= 1;
        }

        return retval;
    }
}
