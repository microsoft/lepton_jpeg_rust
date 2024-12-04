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
    pub sfv: u32,

    /// sample factor horizontal
    pub sfh: u32,

    /// blocks in mcu
    pub mbs: u32,

    /// block count vertical (interleaved)
    pub bcv: u32,

    /// block count horizontal (interleaved)
    pub bch: u32,

    /// block count (all) (interleaved)
    pub bc: u32,

    /// block count vertical (non interleaved)
    pub ncv: u32,

    /// block count horizontal (non interleaved)
    pub nch: u32,

    /// block count (all) (non interleaved)
    pub nc: u32,

    /// statistical identity
    pub sid: u32,

    /// jpeg internal id
    pub jid: u8,
}

impl Default for ComponentInfo {
    fn default() -> ComponentInfo {
        return ComponentInfo {
            q_table_index: 0xff,
            sfv: u32::MAX,
            sfh: u32::MAX,
            mbs: u32::MAX,
            bcv: u32::MAX,
            bch: u32::MAX,
            bc: u32::MAX,
            ncv: u32::MAX,
            nch: u32::MAX,
            nc: u32::MAX,
            sid: u32::MAX,
            jid: 0xff,
            huff_dc: 0xff,
            huff_ac: 0xff,
        };
    }
}
