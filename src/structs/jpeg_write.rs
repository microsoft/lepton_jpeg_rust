/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

/*
Copyright (c) 2006...2016, Matthias Stirner and HTW Aalen University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use anyhow::{Context, Result};
use bytemuck::cast_ref;
use byteorder::WriteBytesExt;
use wide::{i16x16, CmpEq};

use crate::{
    consts::{JPegDecodeStatus, JPegType},
    helpers::{err_exit_code, here, u16_bit_length},
    jpeg_code,
    lepton_error::ExitCode,
    structs::block_based_image::AlignedBlock,
};

use std::io::Write;

use super::{
    bit_writer::BitWriter, block_based_image::BlockBasedImage, jpeg_header::HuffCodes,
    jpeg_position_state::JpegPositionState, lepton_format::LeptonHeader, row_spec::RowSpec,
    thread_handoff::ThreadHandoff,
};

// write a range of rows corresponding to the thread_handoff structure into the writer. Only works with baseline non-progressive images.
pub fn jpeg_write_row_range<W: Write>(
    writer: &mut W,
    framebuffer: &[BlockBasedImage],
    mcuv: i32,
    thread_handoff: &ThreadHandoff,
    max_coded_heights: &[u32],
    huffw: &mut BitWriter,
    lh: &LeptonHeader,
) -> Result<()> {
    huffw.reset_from_overhang_byte_and_num_bits(
        thread_handoff.overhang_byte,
        thread_handoff.num_overhang_bits.into(),
    );
    let mut last_dc = thread_handoff.last_dc.clone();

    let mut decode_index = 0;
    loop {
        let cur_row =
            RowSpec::get_row_spec_from_index(decode_index, framebuffer, mcuv, max_coded_heights);

        decode_index += 1;

        if cur_row.done {
            break;
        }

        if cur_row.skip {
            continue;
        }

        if cur_row.min_row_luma_y < thread_handoff.luma_y_start {
            continue;
        }

        if cur_row.next_row_luma_y > thread_handoff.luma_y_end {
            break; // we're done here
        }

        if cur_row.last_row_to_complete_mcu {
            recode_one_mcu_row(
                huffw,
                cur_row.mcu_row_index * lh.jpeg_header.mcuh,
                writer,
                &mut last_dc,
                framebuffer,
                lh,
            )
            .context(here!())?;

            huffw.flush_with_escape(writer).context(here!())?;
        }
    }

    Ok(())
}

// writes an entire scan vs only a range of rows as above.
// supports progressive encoding whereas the row range version does not
pub fn jpeg_write_entire_scan<W: Write>(
    writer: &mut W,
    framebuffer: &[BlockBasedImage],
    lh: &LeptonHeader,
) -> Result<()> {
    let mut last_dc = [0i16; 4];

    let mut huffw = BitWriter::new();
    let max_coded_heights = lh.truncate_components.get_max_coded_heights();

    let mut decode_index = 0;
    loop {
        let cur_row = RowSpec::get_row_spec_from_index(
            decode_index,
            framebuffer,
            lh.truncate_components.mcu_count_vertical,
            &max_coded_heights[..],
        );

        decode_index += 1;

        if cur_row.done {
            break;
        }

        if cur_row.skip {
            continue;
        }

        if cur_row.last_row_to_complete_mcu {
            let r = recode_one_mcu_row(
                &mut huffw,
                cur_row.mcu_row_index * lh.jpeg_header.mcuh,
                writer,
                &mut last_dc,
                framebuffer,
                lh,
            )
            .context(here!())?;

            huffw.flush_with_escape(writer).context(here!())?;

            if r {
                break;
            }
        }
    }

    huffw.flush_with_escape(writer).context(here!())?;

    Ok(())
}

#[inline(never)]
fn recode_one_mcu_row<W: Write>(
    huffw: &mut BitWriter,
    mcu: i32,
    writer: &mut W,
    lastdc: &mut [i16],
    framebuffer: &[BlockBasedImage],
    ch: &LeptonHeader,
) -> Result<bool> {
    let jf = &ch.jpeg_header;

    let mut state = JpegPositionState::new(jf, mcu);

    let mut cumulative_reset_markers = state.get_cumulative_reset_markers(jf);

    let mut end_of_row = false;
    let mut correction_bits = Vec::new();

    // JPEG imagedata encoding routines
    while !end_of_row {
        // (re)set status
        let mut sta = JPegDecodeStatus::DecodeInProgress;

        // ---> sequential interleaved encoding <---
        while sta == JPegDecodeStatus::DecodeInProgress {
            let current_block = framebuffer[state.get_cmp()].get_block(state.get_dpos());

            let old_mcu = state.get_mcu();

            if jf.jpeg_type == JPegType::Sequential {
                // unzigzag
                let mut block = current_block.zigzag();

                // diff coding for dc
                let dc = block.get_block()[0];
                block.get_block_mut()[0] -= lastdc[state.get_cmp()];
                lastdc[state.get_cmp()] = dc;

                // encode block
                encode_block_seq(
                    huffw,
                    jf.get_huff_dc_codes(state.get_cmp()),
                    jf.get_huff_ac_codes(state.get_cmp()),
                    &block,
                );

                huffw.flush_with_escape(writer).context(here!())?;
                sta = state.next_mcu_pos(&jf);
            } else if jf.cs_to == 0 {
                // ---> progressive DC encoding <---
                if jf.cs_sah == 0 {
                    // ---> succesive approximation first stage <---

                    // diff coding & bitshifting for dc
                    let tmp = current_block.get_coefficient_zigzag(0) >> jf.cs_sal;
                    let v = tmp - lastdc[state.get_cmp()];
                    lastdc[state.get_cmp()] = tmp;

                    // encode dc
                    write_coef(huffw, v, 0, jf.get_huff_dc_codes(state.get_cmp()));
                } else {
                    // ---> succesive approximation later stage <---

                    // fetch bit from current bitplane
                    huffw.write(
                        ((current_block.get_coefficient_zigzag(0) >> jf.cs_sal) & 1) as u64,
                        1,
                    );
                }

                huffw.flush_with_escape(writer).context(here!())?;
                sta = state.next_mcu_pos(jf);
            } else {
                // ---> progressive AC encoding <---

                // copy from coefficients we need and shift right by cs_sal
                let mut block = [0i16; 64];
                for bpos in jf.cs_from..jf.cs_to + 1 {
                    block[usize::from(bpos)] = div_pow2(
                        current_block.get_coefficient_zigzag(usize::from(bpos)),
                        jf.cs_sal,
                    );
                }

                if jf.cs_sah == 0 {
                    // ---> succesive approximation first stage <---

                    // encode block
                    encode_ac_prg_fs(
                        huffw,
                        jf.get_huff_ac_codes(state.get_cmp()),
                        &block,
                        &mut state,
                        jf.cs_from,
                        jf.cs_to,
                    )
                    .context(here!())?;

                    sta = state.next_mcu_pos(jf);

                    // encode remaining eobrun (iff end of mcu or scan)
                    if sta != JPegDecodeStatus::DecodeInProgress {
                        encode_eobrun(huffw, jf.get_huff_ac_codes(state.get_cmp()), &mut state);
                    }
                    huffw.flush_with_escape(writer).context(here!())?;
                } else {
                    // ---> succesive approximation later stage <---

                    // encode block
                    encode_ac_prg_sa(
                        huffw,
                        jf.get_huff_ac_codes(state.get_cmp()),
                        &block,
                        &mut state,
                        jf.cs_from,
                        jf.cs_to,
                        &mut correction_bits,
                    )
                    .context(here!())?;

                    sta = state.next_mcu_pos(jf);

                    // encode remaining eobrun and correction bits (iff end of mcu or scan)
                    if sta != JPegDecodeStatus::DecodeInProgress {
                        encode_eobrun(huffw, jf.get_huff_ac_codes(state.get_cmp()), &mut state);

                        // encode remaining correction bits
                        encode_crbits(huffw, &mut correction_bits);
                    }
                    huffw.flush_with_escape(writer).context(here!())?;
                }
            }

            if old_mcu != state.get_mcu() && state.get_mcu() % jf.mcuh == 0 {
                end_of_row = true;
                if sta == JPegDecodeStatus::DecodeInProgress {
                    // completed only MCU aligned row, not reset interval so don't emit anything special
                    huffw.flush_with_escape(writer).context(here!())?;
                    return Ok(false);
                }
            }

            huffw.flush_with_escape(writer).context(here!())?;
        }

        // pad huffman writer
        huffw.pad(ch.pad_bit.unwrap_or(0));

        assert!(
            huffw.has_no_remainder(),
            "shouldnt have a remainder after padding"
        );

        huffw.flush_with_escape(writer).context(here!())?;

        // evaluate status
        if sta == JPegDecodeStatus::ScanCompleted {
            return Ok(true); // leave decoding loop, everything is done here
        } else {
            assert!(sta == JPegDecodeStatus::RestartIntervalExpired);

            // status 1 means restart
            if jf.rsti > 0 {
                if ch.rst_cnt.len() == 0
                    || (!ch.rst_cnt_set)
                    || cumulative_reset_markers < ch.rst_cnt[ch.scnc]
                {
                    let rst = jpeg_code::RST0 + (cumulative_reset_markers & 7) as u8;
                    writer.write_u8(0xFF)?;
                    writer.write_u8(rst)?;
                    cumulative_reset_markers += 1;
                }

                // (re)set rst wait counter
                state.reset_rstw(jf);

                // (re)set last DCs for diff coding
                for i in 0..lastdc.len() {
                    lastdc[i] = 0;
                }
            }
        }
    }

    Ok(false)
}

#[inline(never)]
fn encode_block_seq(
    huffw: &mut BitWriter,
    dctbl: &HuffCodes,
    actbl: &HuffCodes,
    block: &AlignedBlock,
) {
    // process the array of coefficients as a 4 x 16 = 64 bit integer
    let block64: &[i16x16; 4] = cast_ref(block.get_block());

    // write the DC coefficent (which is the first one in the zigzag order)
    write_coef(huffw, block.get_coefficient(0), 0, dctbl);

    // process the AC coefficients, keeping track of the number of bits left in the current 64 bit block
    // we used up the first 16 bits for the DC coefficient, so start shift right and keep track of the number of bits left

    let zero = i16x16::splat(0);

    let mask_array = [
        block64[0].cmp_eq(zero).move_mask() | 1,
        block64[1].cmp_eq(zero).move_mask(),
        block64[2].cmp_eq(zero).move_mask(),
        block64[3].cmp_eq(zero).move_mask(),
    ];

    let mut z: i32 = -16; // number of zeros in a row * 16 (shifted because this is that way the coefficients are encoded later on)

    for i in 0..4 {
        let mask = mask_array[i];

        if mask == 0xffff {
            z += 16 * 16;
            continue;
        }

        let current_value = block64[i];

        if z <= 0 {
            simpleiter(mask, &mut z, current_value, huffw, actbl);
        } else {
            complexiter(mask, &mut z, current_value, huffw, actbl);
        }
    }

    // if there were trailing zeros, then write end-of-block code, otherwise unnecessary since we wrote 64 coefficients
    if z != 0 {
        huffw.write(actbl.c_val[0x00].into(), actbl.c_len[0x00].into());
    }
}

fn complexiter(
    mask: i32,
    z: &mut i32,
    current_value: i16x16,
    huffw: &mut BitWriter,
    actbl: &HuffCodes,
) {
    for j in 0..16 {
        if mask & (1 << j) != 0 {
            *z += 16;
            continue;
        }

        let tmp = current_value.as_array_ref()[j];

        // if we have 16 or more zero, we need to write them in blocks of 16
        while *z >= 256 {
            huffw.write(actbl.c_val[0xF0].into(), actbl.c_len[0xF0].into());
            *z -= 256;
        }

        // write the non-zero coefficient we found
        write_coef(huffw, tmp, *z as u32, actbl);
        *z = 0;
    }
}

#[unroll::unroll_for_loops]
fn simpleiter(
    mask: i32,
    z: &mut i32,
    current_value: i16x16,
    huffw: &mut BitWriter,
    actbl: &HuffCodes,
) {
    let vmask: i16x16 = current_value >> 15;
    let vtemp = current_value + vmask;
    let xor: i16x16 = vmask ^ vtemp;
    let lz = xor.as_array_ref().map(|x| 16 - x.leading_zeros() as i16);
    let vsh = i16x16::new(lz.map(|x| (1 << x) - 1));
    let vn = vtemp & vsh;
    let mut tz = *z;

    for j in 0..16 {
        if mask & (1 << j) != 0 {
            tz += 16;
        } else {
            // vli encode
            let s = lz[j];
            let n = vn.as_array_ref()[j];

            let hc = ((tz as u32 | s as u32) & 0xff) as usize;

            // combine into single write
            // c_val_shift is already shifted left by s
            let val = actbl.c_val_shift[hc] | (n as u32);
            let new_bits = u32::from(actbl.c_len[hc]) + s as u32;

            // write everything to bitwriter
            huffw.write(val as u64, new_bits);
            tz = 0;
        }
    }

    *z = tz;
}

/// encodes a coefficient which is a huffman code specifying the size followed
/// by the coefficient itself
#[inline(always)]
fn write_coef(huffw: &mut BitWriter, coef: i16, z: u32, tbl: &HuffCodes) {
    // vli encode
    let (n, s) = envli(coef);
    let hc = (z | s) as usize;

    // combine into single write
    // c_val_shift is already shifted left by s
    let val = tbl.c_val_shift[hc] | u32::from(n);
    let new_bits = u32::from(tbl.c_len[hc]) + u32::from(s);

    // write everything to bitwriter
    huffw.write(val as u64, new_bits);
}

/// progressive AC encoding (first pass)
fn encode_ac_prg_fs(
    huffw: &mut BitWriter,
    actbl: &HuffCodes,
    block: &[i16; 64],
    state: &mut JpegPositionState,
    from: u8,
    to: u8,
) -> Result<()> {
    // encode AC
    let mut z = 0;
    for bpos in from..to + 1 {
        // if nonzero is encountered
        let tmp = block[usize::from(bpos)];
        if tmp != 0 {
            // encode eobrun
            encode_eobrun(huffw, actbl, state);
            // write remaining zeroes
            while z >= 16 {
                huffw.write(actbl.c_val[0xF0].into(), actbl.c_len[0xF0].into());
                z -= 16;
            }

            // vli encode
            write_coef(huffw, tmp, z << 4, actbl);

            // reset zeroes
            z = 0;
        } else {
            // increment zero counter
            z += 1;
        }
    }

    // check eob, increment eobrun if needed
    if z > 0 {
        if actbl.max_eob_run == 0 {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "there must be at least one EOB symbol run in the huffman table to encode EOBs",
            )
            .context(here!());
        }

        state.eobrun += 1;

        // check eobrun, encode if needed
        if state.eobrun == actbl.max_eob_run {
            encode_eobrun(huffw, actbl, state);
        }
    }

    Ok(())
}

/// progressive AC SA encoding subsequent pass
fn encode_ac_prg_sa(
    huffw: &mut BitWriter,
    actbl: &HuffCodes,
    block: &[i16; 64],
    state: &mut JpegPositionState,
    from: u8,
    to: u8,
    correction_bits: &mut Vec<u8>,
) -> Result<()> {
    // check if block contains any newly nonzero coefficients and find out position of eob
    let mut eob = from;

    {
        let mut bpos = to;
        while bpos >= from {
            if (block[usize::from(bpos)] == 1) || (block[usize::from(bpos)] == -1) {
                eob = bpos + 1;
                break;
            }
            bpos -= 1;
        }
    }

    // encode eobrun if needed
    if (eob > from) && state.eobrun > 0 {
        encode_eobrun(huffw, actbl, state);

        encode_crbits(huffw, correction_bits);
    }

    // encode AC
    let mut z = 0;
    for bpos in from..eob {
        let tmp = block[usize::from(bpos)];
        // if zero is encountered
        if tmp == 0 {
            z += 1; // increment zero counter
            if z == 16 {
                // write zeroes if needed
                huffw.write(actbl.c_val[0xF0].into(), actbl.c_len[0xF0].into());

                encode_crbits(huffw, correction_bits);
                z = 0;
            }
        }
        // if nonzero is encountered
        else if (tmp == 1) || (tmp == -1) {
            // vli encode
            write_coef(huffw, tmp, z << 4, actbl);

            // write correction bits
            encode_crbits(huffw, correction_bits);
            // reset zeroes
            z = 0;
        } else {
            // store correction bits
            let n = (block[usize::from(bpos)] & 0x1) as u8;
            correction_bits.push(n);
        }
    }

    // fast processing after eob
    for bpos in eob..to + 1 {
        if block[usize::from(bpos)] != 0 {
            // store correction bits
            let n = (block[usize::from(bpos)] & 0x1) as u8;
            correction_bits.push(n);
        }
    }

    // check eob, increment eobrun if needed
    if eob <= to {
        if actbl.max_eob_run == 0 {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "there must be at least one EOB symbol run in the huffman table to encode EOBs",
            )
            .context(here!());
        }

        state.eobrun += 1;

        // check eobrun, encode if needed
        if state.eobrun == actbl.max_eob_run {
            encode_eobrun(huffw, actbl, state);

            encode_crbits(huffw, correction_bits);
        }
    }

    Ok(())
}

/// encodes the eob run which consists of a huffman code the high 4 bits specifying the log2 of the run
/// followed by the number number encoded into the minimum number of bits
fn encode_eobrun(huffw: &mut BitWriter, actbl: &HuffCodes, state: &mut JpegPositionState) {
    if (state.eobrun) > 0 {
        debug_assert!((state.eobrun) <= actbl.max_eob_run);

        let mut s = u16_bit_length(state.eobrun);
        s -= 1;

        let n = encode_eobrun_bits(s, state.eobrun);
        let hc = s << 4;
        huffw.write(
            actbl.c_val[usize::from(hc)].into(),
            actbl.c_len[usize::from(hc)].into(),
        );
        huffw.write(u64::from(n), u32::from(s));
        state.eobrun = 0;
    }
}

/// encodes the correction bits, which are simply encoded as a vector of single bit values
fn encode_crbits(huffw: &mut BitWriter, correction_bits: &mut Vec<u8>) {
    for x in correction_bits.drain(..) {
        huffw.write(u64::from(x), 1);
    }
}

/// divide power of 2 rounding towards zero
fn div_pow2(v: i16, p: u8) -> i16 {
    (if v < 0 { v + ((1 << p) - 1) } else { v }) >> p
}

/// prepares a coefficient for encoding. Calculates the bitlength s makes v positive by adding 1 << s  - 1 if the number is negative or zero
#[inline(always)]
fn envli(vs: i16) -> (u32, u32) {
    let v = i32::from(vs);
    let mask = v >> 31;
    let temp = v + mask;
    let s = 32 - (mask ^ temp).leading_zeros();
    let n = temp & ((1 << s) - 1);
    return (n as u32, s);
}

/// encoding for eobrun length. Chop off highest bit since we know it is always 1.
fn encode_eobrun_bits(s: u8, v: u16) -> u16 {
    v - (1 << s)
}

#[test]
fn test_envli() {
    for i in -1023..=1023 {
        let (n, s) = envli(i);

        assert_eq!(s, u16_bit_length(i.unsigned_abs()) as u32);

        let n2 = if i > 0 { i } else { i - 1 + (1 << s) } as u32;
        assert_eq!(n, n2, "i={} s={}", i, s);
    }
}

#[test]
fn test_encode_block_seq() {
    let mut buf = Vec::new();

    let mut b = BitWriter::new();
    let mut block = AlignedBlock::default();
    for i in 0..64 {
        block.get_block_mut()[i] = i as i16;
    }

    encode_block_seq(
        &mut b,
        &HuffCodes::construct_default_code(),
        &HuffCodes::construct_default_code(),
        &block,
    );

    b.flush_with_escape(&mut buf).unwrap();

    let expected = [
        0, 1, 129, 64, 88, 28, 3, 160, 120, 15, 130, 64, 36, 130, 80, 37, 130, 96, 38, 130, 112,
        39, 130, 192, 22, 32, 178, 5, 152, 45, 1, 106, 11, 96, 91, 130, 224, 23, 32, 186, 5, 216,
        47, 1, 122, 11, 224, 95, 131, 64, 13, 8, 52, 64, 209, 131, 72, 13, 40, 52, 192, 211, 131,
        80, 13, 72, 53, 64, 213, 131, 88, 13, 104, 53, 192, 215, 131, 96, 13, 136, 54, 64, 217,
        131, 104, 13, 168, 54, 192, 219, 131, 112, 13, 200, 55, 64, 221, 131, 120, 13, 232, 55,
        192, 223,
    ];
    assert_eq!(buf, expected);
}

#[test]
fn test_encode_block_zero_runs() {
    let mut buf = Vec::new();

    let mut b = BitWriter::new();
    let mut block = AlignedBlock::default();

    for i in 0..10 {
        block.get_block_mut()[i] = i as i16;
    }
    for i in 30..50 {
        block.get_block_mut()[i] = i as i16;
    }
    for i in 50..52 {
        block.get_block_mut()[i] = i as i16;
    }

    encode_block_seq(
        &mut b,
        &HuffCodes::construct_default_code(),
        &HuffCodes::construct_default_code(),
        &block,
    );

    b.flush_with_escape(&mut buf).unwrap();

    let expected = [
        0, 1, 129, 64, 88, 28, 3, 160, 120, 15, 130, 64, 36, 248, 34, 248, 23, 224, 208, 3, 66, 13,
        16, 52, 96, 210, 3, 74, 13, 48, 52, 224, 212, 3, 82, 13, 80, 53, 96, 214, 3, 90, 13, 112,
        53, 224, 216, 3, 98, 13, 144, 54, 96,
    ];
    assert_eq!(buf, expected);
}

#[test]
fn test_encode_block_seq_zero() {
    let mut buf = Vec::new();

    let mut b: BitWriter = BitWriter::new();
    let block = AlignedBlock::default();

    encode_block_seq(
        &mut b,
        &HuffCodes::construct_default_code(),
        &HuffCodes::construct_default_code(),
        &block,
    );

    b.flush_with_escape(&mut buf).unwrap();

    let expected = [0, 0];
    assert_eq!(buf, expected);
}
