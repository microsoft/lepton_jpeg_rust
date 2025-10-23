/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::jpeg::block_based_image::{AlignedBlock, BlockBasedImage, EMPTY_BLOCK};
use crate::structs::neighbor_summary::{NEIGHBOR_DATA_EMPTY, NeighborSummary};
use crate::structs::probability_tables::ProbabilityTables;
pub struct BlockContext {
    block_width: u32,
    cur_block_index: u32,
    cur_neighbor_summary_index: u32,
    above_neighbor_summary_index: u32,
}
pub struct NeighborData<'a> {
    pub above: &'a AlignedBlock,
    pub left: &'a AlignedBlock,
    pub above_left: &'a AlignedBlock,
    pub neighbor_context_above: &'a NeighborSummary,
    pub neighbor_context_left: &'a NeighborSummary,
}

impl BlockContext {
    /// Create a new BlockContext for the first line of the image at a given y-coordinate.
    pub fn off_y(y: u32, image_data: &BlockBasedImage) -> BlockContext {
        let block_width = image_data.get_block_width();

        let cur_block_index = block_width * y;

        // blocks above the first line are never dereferenced
        let cur_neighbor_summary_index = if (y & 1) != 0 { block_width } else { 0 };

        let above_neighbor_summary_index = if (y & 1) != 0 { 0 } else { block_width };

        BlockContext {
            cur_block_index,
            block_width,
            cur_neighbor_summary_index,
            above_neighbor_summary_index,
        }
    }

    // for debugging
    #[allow(dead_code)]
    pub fn get_here_index(&self) -> u32 {
        self.cur_block_index
    }

    // as each new line BlockContext is set by `off_y`, no edge cases with dereferencing
    // out of bounds indices is possible, therefore no special treatment is needed
    pub fn next(&mut self) -> u32 {
        self.cur_block_index += 1;
        self.cur_neighbor_summary_index += 1;
        self.above_neighbor_summary_index += 1;

        self.cur_block_index
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
                image_data.get_block(self.cur_block_index - self.block_width - 1)
            } else {
                &EMPTY_BLOCK
            },
            above: if ALL_PRESENT || pt.is_above_present() {
                image_data.get_block(self.cur_block_index - self.block_width)
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
        &self,
        neighbor_summary_cache: &mut [NeighborSummary],
        neighbor_summary: NeighborSummary,
    ) {
        neighbor_summary_cache[self.cur_neighbor_summary_index as usize] = neighbor_summary;
    }
}
