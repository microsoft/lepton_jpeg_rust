/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::consts::COLOR_CHANNEL_NUM_BLOCK_TYPES;

use super::probability_tables::ProbabilityTables;

pub struct ProbabilityTablesSet {
    pub corner: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
    pub top: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
    pub mid_left: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
    pub middle: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
    pub mid_right: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
    pub width_one: [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES],
}

const fn make_probability_tables_tuple(
    left: bool,
    above: bool,
) -> [ProbabilityTables; COLOR_CHANNEL_NUM_BLOCK_TYPES] {
    return [
        ProbabilityTables::new(0, left, above),
        ProbabilityTables::new(1, left, above),
        ProbabilityTables::new(2, left, above),
    ];
}

pub static PTS: ProbabilityTablesSet = ProbabilityTablesSet {
    corner: make_probability_tables_tuple(false, false),
    top: make_probability_tables_tuple(true, false),
    mid_left: make_probability_tables_tuple(false, true),
    middle: make_probability_tables_tuple(true, true),
    mid_right: make_probability_tables_tuple(true, true),
    width_one: make_probability_tables_tuple(false, true),
};
