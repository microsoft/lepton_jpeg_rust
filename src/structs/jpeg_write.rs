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
use byteorder::WriteBytesExt;

use crate::{
    consts::{JPegDecodeStatus, JPegType},
    helpers::{err_exit_code, here, u16_bit_length},
    jpeg_code,
    lepton_error::ExitCode,
};

use std::{io::Write, num::NonZeroI16};

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
                let mut block = [0i16; 64]; // store block for coeffs
                for bpos in 0..64 {
                    block[bpos] = current_block.get_coefficient_zigzag(bpos);
                }

                // diff coding for dc
                let dc = block[0];
                block[0] -= lastdc[state.get_cmp()];
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
                        ((current_block.get_coefficient_zigzag(0) >> jf.cs_sal) & 1) as u32,
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
    block: &[i16; 64],
) {
    // encode DC
    write_coef(huffw, block[0], 0, dctbl);

    let mut z = 0;

    // encode AC
    for bpos in 1..64 {
        // if nonzero is encountered
        let tmp = block[bpos];
        if tmp == 0 {
            z += 1;
            continue;
        }

        // if we have 16 or more zero, we need to write them in blocks of 16
        while z >= 16 {
            huffw.write(actbl.c_val[0xF0].into(), actbl.c_len[0xF0].into());
            z -= 16;
        }

        write_coef(huffw, tmp, z, actbl);

        // reset zeroes
        z = 0;
    }

    // write eob if needed
    if z != 0 {
        huffw.write(actbl.c_val[0x00].into(), actbl.c_len[0x00].into());
    }
}

/// encodes a coefficient which is a huffman code specifying the size followed
/// by the coefficient itself
#[inline(always)]
fn write_coef(huffw: &mut BitWriter, coef: i16, z: u8, tbl: &HuffCodes) {
    // vli encode
    let (n, s) = envli(coef);
    let hc = ((z & 0xf) << 4) + s;

    // write to huffman writer (combine into single write)
    let val = (u32::from(tbl.c_val[usize::from(hc)]) << s) | u32::from(n);
    let new_bits = u32::from(tbl.c_len[usize::from(hc)]) + u32::from(s);
    huffw.write(val, new_bits);
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
            write_coef(huffw, tmp, z, actbl);

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
            write_coef(huffw, tmp, z, actbl);

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
        huffw.write(u32::from(n), u32::from(s));
        state.eobrun = 0;
    }
}

/// encodes the correction bits, which are simply encoded as a vector of single bit values
fn encode_crbits(huffw: &mut BitWriter, correction_bits: &mut Vec<u8>) {
    for x in correction_bits.drain(..) {
        huffw.write(u32::from(x), 1);
    }
}

/// divide power of 2 rounding towards zero
fn div_pow2(v: i16, p: u8) -> i16 {
    (if v < 0 { v + ((1 << p) - 1) } else { v }) >> p
}

/// prepares a coefficient for encoding. Calculates the bitlength s makes v positive by adding 1 << s  - 1 if the number is negative or zero
#[inline(always)]
fn envli(v: i16) -> (u16, u8) {
    // since this is inlined, in the main case the compiler figures out that v cannot be zero
    if let Some(nz) = NonZeroI16::new(v) {
        let s = 16 - nz.unsigned_abs().leading_zeros();
        let mask = nz.get() >> 15; // -1 if tmp is negative and all 1

        let n = (nz.get() + (((1 << s) - 1) & mask)) as u16; // turn v into a 2s complement of s bits (avoids BitWriter from having to zero out the unused top bits indiscriminately)

        // make sure that calculating the old way is the same
        debug_assert_eq!(n, if v > 0 { v } else { v - 1 + (1 << s) } as u16);
        return (n, s as u8);
    } else {
        return (0, 0);
    }
}

/// encoding for eobrun length. Chop off highest bit since we know it is always 1.
fn encode_eobrun_bits(s: u8, v: u16) -> u16 {
    v - (1 << s)
}
