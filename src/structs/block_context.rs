/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use super::block_based_image::{AlignedBlock, BlockBasedImage, EMPTY_BLOCK};
use super::neighbor_summary::{NeighborSummary, NEIGHBOR_DATA_EMPTY};
use super::probability_tables::ProbabilityTables;

pub struct BlockContext {
    cur_block_index: i32,
    above_block_index: i32,

    cur_neighbor_summary_index: i32,
    above_neighbor_summary_index: i32,
}
pub struct NeighborData<'a> {
    pub above: &'a AlignedBlock,
    pub left: &'a AlignedBlock,
    pub above_left: &'a AlignedBlock,
    pub neighbor_context_above: &'a NeighborSummary,
    pub neighbor_context_left: &'a NeighborSummary,
}

impl BlockContext {
    // for debugging
    #[allow(dead_code)]
    pub fn get_here_index(&self) -> i32 {
        self.cur_block_index
    }

    // as each new line BlockContext is set by `off_y`, no edge cases with dereferencing
    // out of bounds indices is possilbe, therefore no special treatment is needed
    pub fn next(&mut self) -> i32 {
        self.cur_block_index += 1;
        self.above_block_index += 1;
        self.cur_neighbor_summary_index += 1;
        self.above_neighbor_summary_index += 1;

        self.cur_block_index
    }

    pub fn new(
        cur_block_index: i32,
        above_block_index: i32,
        cur_neighbor_summary_index: i32,
        above_neighbor_summary_index: i32,
    ) -> Self {
        return BlockContext {
            cur_block_index,
            above_block_index,
            cur_neighbor_summary_index,
            above_neighbor_summary_index,
        };
    }

    pub fn here<'a>(&self, image_data: &'a BlockBasedImage) -> &'a AlignedBlock {
        let retval = image_data.get_block(self.cur_block_index);
        return retval;
    }

    pub fn get_neighbor_data<'a, const ALL_PRESENT: bool>(
        &self,
        image_data: &'a BlockBasedImage,
        neighbor_summary: &'a [NeighborSummary],
        pt: &ProbabilityTables,
    ) -> NeighborData<'a> {
        NeighborData::<'a> {
            above_left: if ALL_PRESENT {
                image_data.get_block(self.above_block_index - 1)
            } else {
                &EMPTY_BLOCK
            },
            above: if ALL_PRESENT || pt.is_above_present() {
                image_data.get_block(self.above_block_index)
            } else {
                &EMPTY_BLOCK
            },
            left: if ALL_PRESENT || pt.is_left_present() {
                image_data.get_block(self.cur_block_index - 1)
            } else {
                &EMPTY_BLOCK
            },
            neighbor_context_above: if ALL_PRESENT || pt.is_above_present() {
                &neighbor_summary[self.above_neighbor_summary_index as usize]
            } else {
                &NEIGHBOR_DATA_EMPTY
            },
            neighbor_context_left: if ALL_PRESENT || pt.is_left_present() {
                &neighbor_summary[(self.cur_neighbor_summary_index - 1) as usize]
            } else {
                &NEIGHBOR_DATA_EMPTY
            },
        }
    }

    pub fn set_neighbor_summary_here(
        &mut self,
        neighbor_summary_cache: &mut [NeighborSummary],
        neighbor_summary: NeighborSummary,
    ) {
        neighbor_summary_cache[self.cur_neighbor_summary_index as usize] = neighbor_summary;
    }
}
