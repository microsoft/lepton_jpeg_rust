/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use super::block_based_image::{AlignedBlock, BlockBasedImage, EMPTY_BLOCK};
use super::neighbor_summary::NeighborSummary;
use super::probability_tables::ProbabilityTables;

pub struct BlockContext {
    block_width: i32,

    cur_block_index: i32,
    above_block_index: i32,

    cur_neighbor_summary_index: i32,
    above_neighbor_summary_index: i32,
}

impl BlockContext {
    // for debugging
    #[allow(dead_code)]
    pub fn get_here_index(&self) -> i32 {
        self.cur_block_index
    }

    pub fn next(&mut self, has_more: bool) -> i32 {
        self.cur_block_index += 1;

        let retval = self.cur_block_index;

        if retval < self.block_width {
            self.above_block_index = self.cur_block_index + self.block_width;
        } else {
            self.above_block_index = self.cur_block_index - self.block_width;
        }

        self.cur_neighbor_summary_index += 1;
        self.above_neighbor_summary_index += 1;

        if !has_more {
            let cur_row_first = self.cur_neighbor_summary_index < self.above_neighbor_summary_index;
            if cur_row_first {
                self.above_neighbor_summary_index -= self.block_width * 2;
            } else {
                self.cur_neighbor_summary_index -= self.block_width * 2;
            }
        }

        return retval;
    }

    pub fn new(
        cur_block_index: i32,
        above_block_index: i32,
        cur_neighbor_summary_index: i32,
        above_neighbor_summary_index: i32,
        image_data: &BlockBasedImage,
    ) -> Self {
        return BlockContext {
            block_width: image_data.get_block_width(),
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

    pub fn get_neighbors<'a, const ALL_PRESENT: bool>(
        &self,
        image_data: &'a BlockBasedImage,
        pt: &ProbabilityTables,
    ) -> (&'a AlignedBlock, &'a AlignedBlock, &'a AlignedBlock) {
        (
            if ALL_PRESENT {
                image_data.get_block(self.above_block_index - 1)
            } else {
                &EMPTY_BLOCK
            },
            if ALL_PRESENT || pt.is_above_present() {
                image_data.get_block(self.above_block_index)
            } else {
                &EMPTY_BLOCK
            },
            if ALL_PRESENT || pt.is_left_present() {
                image_data.get_block(self.cur_block_index - 1)
            } else {
                &EMPTY_BLOCK
            },
        )
    }

    pub fn non_zeros_here(&self, neighbor_summary: &[NeighborSummary]) -> u8 {
        return neighbor_summary[self.cur_neighbor_summary_index as usize].get_num_non_zeros();
    }

    pub fn get_non_zeros_above(&self, neighbor_summary: &[NeighborSummary]) -> u8 {
        return neighbor_summary[self.above_neighbor_summary_index as usize].get_num_non_zeros();
    }

    pub fn get_non_zeros_left(&self, neighbor_summary: &[NeighborSummary]) -> u8 {
        return neighbor_summary[(self.cur_neighbor_summary_index - 1) as usize]
            .get_num_non_zeros();
    }

    pub fn neighbor_context_here<'a>(
        &mut self,
        neighbor_summary: &'a mut [NeighborSummary],
    ) -> &'a mut NeighborSummary {
        return &mut neighbor_summary[self.cur_neighbor_summary_index as usize];
    }

    pub fn neighbor_context_above<'a>(
        &self,
        neighbor_summary: &'a [NeighborSummary],
    ) -> &'a NeighborSummary {
        return &neighbor_summary[self.above_neighbor_summary_index as usize];
    }

    pub fn neighbor_context_left<'a>(
        &self,
        neighbor_summary: &'a [NeighborSummary],
    ) -> &'a NeighborSummary {
        return &neighbor_summary[std::cmp::max(0, self.cur_neighbor_summary_index - 1) as usize];
    }
}
