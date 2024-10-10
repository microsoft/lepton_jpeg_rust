/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// quantization table
    pub q_table_index: u8,

    /// no of huffman table (DC)
    pub huff_dc: u8,

    /// no of huffman table (AC)
    pub huff_ac: u8,

    /// sample factor vertical
    pub sfv: i32,

    /// sample factor horizontal
    pub sfh: i32,

    /// blocks in mcu
    pub mbs: i32,

    /// block count vertical (interleaved)
    pub bcv: i32,

    /// block count horizontal (interleaved)
    pub bch: i32,

    /// block count (all) (interleaved)
    pub bc: i32,

    /// block count vertical (non interleaved)
    pub ncv: i32,

    /// block count horizontal (non interleaved)
    pub nch: i32,

    /// block count (all) (non interleaved)
    pub nc: i32,

    /// statistical identity
    pub sid: i32,

    /// jpeg internal id
    pub jid: u8,
}

impl Default for ComponentInfo {
    fn default() -> ComponentInfo {
        return ComponentInfo {
            q_table_index: 0xff,
            sfv: -1,
            sfh: -1,
            mbs: -1,
            bcv: -1,
            bch: -1,
            bc: -1,
            ncv: -1,
            nch: -1,
            nc: -1,
            sid: -1,
            jid: 0xff,
            huff_dc: 0xff,
            huff_ac: 0xff,
        };
    }
}
