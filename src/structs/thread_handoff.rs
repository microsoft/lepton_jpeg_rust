/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::io::{Read, Result, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::consts::COLOR_CHANNEL_NUM_BLOCK_TYPES;

#[derive(Debug, Clone, PartialEq)]
pub struct ThreadHandoff {
    pub luma_y_start: i32,
    pub luma_y_end: i32,
    pub segment_offset_in_file: i32,
    pub segment_size: i32,
    pub overhang_byte: u8,
    pub num_overhang_bits: u8,
    pub last_dc: [i16; 4],
}

impl ThreadHandoff {
    pub fn deserialize<R: Read>(num_threads: u8, data: &mut R) -> Result<Vec<ThreadHandoff>> {
        let mut retval: Vec<ThreadHandoff> = Vec::with_capacity(num_threads as usize);

        for _i in 0..num_threads {
            let mut th = ThreadHandoff {
                luma_y_start: data.read_u16::<LittleEndian>()? as i32,
                luma_y_end: 0,             // filled in later
                segment_offset_in_file: 0, // not serialized
                segment_size: data.read_i32::<LittleEndian>()?,
                overhang_byte: data.read_u8()?,
                num_overhang_bits: data.read_u8()?,
                last_dc: [0; 4],
            };

            for j in 0..COLOR_CHANNEL_NUM_BLOCK_TYPES {
                th.last_dc[j] = data.read_i16::<LittleEndian>()?
            }
            for _j in COLOR_CHANNEL_NUM_BLOCK_TYPES..4 {
                data.read_u16::<LittleEndian>()?;
            }

            retval.push(th);
        }

        for i in 1..retval.len() {
            retval[i - 1].luma_y_end = retval[i].luma_y_start;
        }

        // last LumaYEnd is not serialzed, filled in later
        return Ok(retval);
    }

    pub fn serialize<W: Write>(data: &Vec<ThreadHandoff>, retval: &mut W) -> Result<()> {
        retval.write_u8(data.len() as u8)?;

        for th in data {
            retval.write_u16::<LittleEndian>(th.luma_y_start as u16)?;
            // SegmentOffsetInFile is not serialized to preserve compatibility with original Lepton format
            retval.write_i32::<LittleEndian>(th.segment_size as i32)?;
            retval.write_u8(th.overhang_byte)?;
            retval.write_u8(th.num_overhang_bits)?;

            for i in 0..COLOR_CHANNEL_NUM_BLOCK_TYPES {
                retval.write_i16::<LittleEndian>(th.last_dc[i])?;
            }
            for _i in COLOR_CHANNEL_NUM_BLOCK_TYPES..4 {
                retval.write_u16::<LittleEndian>(0)?;
            }
        }

        return Ok(());
    }

    // Combine two ThreadHandoff objects into a range, starting with the "from" segment, and
    // continuing until the end of the "to" segment [from, to]
    pub fn get_combine_thread_range_segment_size(
        from: &ThreadHandoff,
        to: &ThreadHandoff,
    ) -> usize {
        return (to.segment_offset_in_file - from.segment_offset_in_file + to.segment_size)
            as usize;
    }

    pub fn combine_thread_ranges(from: &ThreadHandoff, to: &ThreadHandoff) -> ThreadHandoff {
        let ret = ThreadHandoff {
            segment_offset_in_file: from.segment_offset_in_file,
            luma_y_start: from.luma_y_start,
            overhang_byte: from.overhang_byte,
            num_overhang_bits: from.num_overhang_bits,
            luma_y_end: to.luma_y_end,
            segment_size: ThreadHandoff::get_combine_thread_range_segment_size(from, to) as i32,
            last_dc: from.last_dc,
        };

        return ret;
    }
}
