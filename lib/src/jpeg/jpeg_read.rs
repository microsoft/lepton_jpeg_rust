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

use std::cmp::{self, max};
use std::io::{BufRead, Read, Seek, SeekFrom};

use crate::helpers::*;
use crate::lepton_error::{AddContext, ExitCode, Result, err_exit_code};
use crate::{EnabledFeatures, consts::*};

use super::bit_reader::BitReader;
use super::block_based_image::{AlignedBlock, BlockBasedImage};
use super::jpeg_code;
use super::jpeg_header::{
    HuffTree, JpegHeader, ReconstructionInfo, RestartSegmentCodingInfo, parse_jpeg_header,
};
use super::jpeg_position_state::JpegPositionState;

/// Reads a JPEG file from the provided reader and returns the image data. This function is
/// designed to return all the information needed to reconstruct a bit-level identical
/// JPEG file.
///
/// In some cases this will not be possible, for example if a JPEG contains certain coding errors
/// that are non-standard, in which case the function will return an error. This doesn't mean the JPEG
/// is corrupt, just that it is not supported for identical reconstruction.
///
/// The function returns the image data as a vector of `BlockBasedImage`, which contain the
/// DCT coefficients for each block in the image (we do not perform inverse DCT, this would be lossy).
/// In addition, we return a vector of `RestartSegmentCodingInfo` which contains the information
/// needed to reconstruct a portion of the JPEG file starting at the given offset. This is useful
/// for baseline images where we can split the image into sections and decode them in parallel.
///
/// The callback function is called with the JPEG header information after it has been parsed, and
/// is useful for debugging or logging purposes. Progressive images will contain multiple scans and
/// call the callback multiple times.
///
/// Non-progressive images support the idea of truncating the image (since this happens frequently)
/// where the bitstream is cut off at an arbitrary point. We assume that all subsequent data is zero,
/// but remember enough to reconstruct the bitstream until there.
///
/// There is also the concept of "garbage data" which is what comes after the scan data but is not
/// recognized as a header. This garbage data should be appeneded to the end of the file.
pub fn read_jpeg_file<R: BufRead + Seek, FN: FnMut(&JpegHeader, &[u8])>(
    reader: &mut R,
    jpeg_header: &mut JpegHeader,
    rinfo: &mut ReconstructionInfo,
    enabled_features: &EnabledFeatures,
    mut on_header_callback: FN,
) -> Result<(
    Vec<BlockBasedImage>,
    Vec<(u64, RestartSegmentCodingInfo)>,
    u64,
)> {
    let mut startheader = [0u8; 2];
    reader.read(&mut startheader)?;
    if startheader[0] != 0xFF || startheader[1] != jpeg_code::SOI {
        return err_exit_code(
            ExitCode::UnsupportedJpeg,
            "jpeg must start with with 0xff 0xd8",
        );
    }

    if !prepare_to_decode_next_scan(jpeg_header, rinfo, reader, enabled_features).context()? {
        return err_exit_code(ExitCode::UnsupportedJpeg, "Jpeg does not contain scans");
    }

    on_header_callback(jpeg_header, &rinfo.raw_jpeg_header);

    if !enabled_features.progressive && jpeg_header.jpeg_type == JpegType::Progressive {
        return err_exit_code(
            ExitCode::ProgressiveUnsupported,
            "file is progressive, but this is disabled",
        )
        .context();
    }

    if jpeg_header.cmpc > COLOR_CHANNEL_NUM_BLOCK_TYPES {
        return err_exit_code(
            ExitCode::Unsupported4Colors,
            "doesn't support 4 color channels",
        )
        .context();
    }

    rinfo.truncate_components.init(jpeg_header);
    let mut image_data = Vec::<BlockBasedImage>::new();
    for i in 0..jpeg_header.cmpc {
        // constructor takes height in proportion to the component[0]
        image_data.push(BlockBasedImage::new(
            &jpeg_header,
            i,
            0,
            jpeg_header.cmp_info[0].bcv,
        )?);
    }

    let start_scan_position = reader.stream_position()?;

    let mut partitions = Vec::new();
    read_first_scan(
        &jpeg_header,
        reader,
        &mut partitions,
        &mut image_data[..],
        rinfo,
    )
    .context()?;
    let mut end_scan_position = reader.stream_position()?;

    if start_scan_position + 2 > end_scan_position {
        return err_exit_code(ExitCode::UnsupportedJpeg, "no scan data found in JPEG file")
            .context();
    }

    if partitions.len() == 0 {
        return err_exit_code(
            ExitCode::UnsupportedJpeg,
            "no scan information found in JPEG file",
        )
        .context();
    }

    if jpeg_header.jpeg_type == JpegType::Sequential {
        if rinfo.early_eof_encountered {
            if enabled_features.stop_reading_at_eoi {
                return err_exit_code(ExitCode::ShortRead, "early EOF encountered");
            }

            rinfo
                .truncate_components
                .set_truncation_bounds(&jpeg_header, rinfo.max_dpos);

            // If we got an early EOF, then seek backwards and capture the last two bytes and store them as garbage.
            // This is necessary since the decoder will assume that zero garbage always means a properly terminated JPEG
            // even if early EOF was set to true.
            end_scan_position = reader.seek(SeekFrom::Current(-2))?.try_into().unwrap();

            // make sure we don't return any partitions that are beyond the
            // adjusted end position
            for i in 0..partitions.len() {
                if partitions[i].0 >= end_scan_position {
                    return err_exit_code(
                        ExitCode::UnsupportedJpeg,
                        "Partition conflicts with garbage data",
                    );
                }
            }

            rinfo.garbage_data.resize(2, 0);
            reader.read_exact(&mut rinfo.garbage_data)?;
        }

        if enabled_features.stop_reading_at_eoi {
            // ensure there is an actual EOI marker since we haven't consumed it yet
            let mut end_of_file = [0u8; 2];
            reader.read_exact(&mut end_of_file).context()?;

            if end_of_file != EOI {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "JPEG file does not end with EOI marker",
                )
                .context();
            }

            rinfo.garbage_data = end_of_file.to_vec();
        } else {
            // read the rest of the file to garbage data
            reader.read_to_end(&mut rinfo.garbage_data).context()?;
        }
    } else {
        assert!(jpeg_header.jpeg_type == JpegType::Progressive);

        if rinfo.early_eof_encountered {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "truncation is only supported for baseline images",
            )
            .context();
        }

        // for progressive images, loop around reading headers and decoding until we a complete image_data
        let mut prev_raw_jpeg_header_len = rinfo.raw_jpeg_header.len();

        while prepare_to_decode_next_scan(jpeg_header, rinfo, reader, enabled_features).context()? {
            on_header_callback(
                jpeg_header,
                &&rinfo.raw_jpeg_header[prev_raw_jpeg_header_len..],
            );
            prev_raw_jpeg_header_len = rinfo.raw_jpeg_header.len();

            read_progressive_scan(&jpeg_header, reader, &mut image_data[..], rinfo).context()?;

            if rinfo.early_eof_encountered {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "truncation is only supported for baseline images",
                )
                .context();
            }
        }

        end_scan_position = reader.stream_position()?;

        // Since prepare_to_decode_next_scan consumed the EOI,
        // we need to add EOI to the beginning of the garbage data (if there is any).
        //
        // If there was actually no garbage data, this is still ok since
        // the marker will be appended, then removed when file gets truncated by the
        // overall file limit.
        rinfo.garbage_data = Vec::from(EOI);

        if !enabled_features.stop_reading_at_eoi {
            // append the rest of the file to the buffer
            reader.read_to_end(&mut rinfo.garbage_data).context()?;
        }
    }

    Ok((image_data, partitions, end_scan_position))
}

// false means we hit the end of file marker
fn prepare_to_decode_next_scan<R: Read>(
    jpeg_header: &mut JpegHeader,
    rinfo: &mut ReconstructionInfo,
    reader: &mut R,
    enabled_features: &EnabledFeatures,
) -> Result<bool> {
    // parse the header and store it in the raw_jpeg_header
    if !parse_jpeg_header(reader, enabled_features, jpeg_header, rinfo).context()? {
        return Ok(false);
    }

    rinfo.max_bpos = cmp::max(rinfo.max_bpos, u32::from(jpeg_header.cs_to));

    // FIXME: not sure why only first bit of csSah is examined but 4 bits of it are stored
    rinfo.max_sah = cmp::max(
        rinfo.max_sah,
        cmp::max(jpeg_header.cs_sal, jpeg_header.cs_sah),
    );

    for i in 0..jpeg_header.cs_cmpc {
        rinfo.max_cmp = cmp::max(rinfo.max_cmp, jpeg_header.cs_cmp[i] as u32);
    }

    return Ok(true);
}

/// Reads the scan from the JPEG file, writes the image data to the image_data array and
/// partitions it into restart segments using the partition callback.
///
/// This only works for sequential JPEGs or the first scan in a progressive image.
/// For subsequent scans, use the `read_progressive_scan`.
fn read_first_scan<R: BufRead + Seek>(
    jf: &JpegHeader,
    reader: &mut R,
    partitions: &mut Vec<(u64, RestartSegmentCodingInfo)>,
    image_data: &mut [BlockBasedImage],
    reconstruct_info: &mut ReconstructionInfo,
) -> Result<()> {
    let mut bit_reader = BitReader::new(reader);

    // init variables for decoding
    let mut state = JpegPositionState::new(jf, 0);

    let mut do_handoff = true;

    // JPEG imagedata decoding routines
    let mut sta = JpegDecodeStatus::DecodeInProgress;
    while sta != JpegDecodeStatus::ScanCompleted {
        // decoding for interleaved data
        state.reset_rstw(jf); // restart wait counter

        if jf.jpeg_type == JpegType::Sequential {
            sta = decode_baseline_rst(
                &mut state,
                &mut bit_reader,
                image_data,
                &mut do_handoff,
                jf,
                reconstruct_info,
                partitions,
            )
            .context()?;
        } else if jf.cs_to == 0 && jf.cs_sah == 0 {
            // only need DC
            jf.verify_huffman_table(true, false).context()?;

            let mut last_dc = [0i16; 4];

            while sta == JpegDecodeStatus::DecodeInProgress {
                let current_block = image_data[state.get_cmp()].get_block_mut(state.get_dpos());

                // collect the handoffs although for progressive images
                // we still split the scan into sections, but we don't partition the actual JPEG
                // writes since they have to be done on a single thread in a loop for a progressive file.
                //
                // TODO: get rid of this and just chop up the scan into sections in Lepton code
                if do_handoff {
                    partitions.push((
                        0,
                        RestartSegmentCodingInfo::new(0, 0, [0; 4], state.get_mcu(), jf),
                    ));

                    do_handoff = false;
                }

                // ---> succesive approximation first stage <---

                // diff coding & bitshifting for dc
                let coef = read_dc(&mut bit_reader, jf.get_huff_dc_tree(state.get_cmp()))?;

                let v = coef.wrapping_add(last_dc[state.get_cmp()]);
                last_dc[state.get_cmp()] = v;

                current_block.set_transposed_from_zigzag(0, v << jf.cs_sal);

                let old_mcu = state.get_mcu();
                sta = state.next_mcu_pos(jf);

                if state.get_mcu() % jf.mcuh == 0 && old_mcu != state.get_mcu() {
                    do_handoff = true;
                }
            }
        } else {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "progress must start with DC stage",
            )
            .context();
        }

        // if we saw a pad bit at the end of the block, then remember whether they were 1s or 0s. This
        // will be used later on to reconstruct the padding
        bit_reader
            .read_and_verify_fill_bits(&mut reconstruct_info.pad_bit)
            .context()?;

        // verify that we got the right RST code here since the above should do 1 mcu.
        // If we didn't then we won't re-encode the file binary identical so there's no point in continuing
        if sta == JpegDecodeStatus::RestartIntervalExpired {
            bit_reader.verify_reset_code().context()?;

            sta = JpegDecodeStatus::DecodeInProgress;
        }
    }

    Ok(())
}

/// Reads a scan for progressive images where the image is encoded in multiple passes.
/// Between each scan are a bunch of header than need to be parsed containing information
/// like updated Huffman tables and quantization tables.
fn read_progressive_scan<R: BufRead + Seek>(
    jf: &JpegHeader,
    reader: &mut R,
    image_data: &mut [BlockBasedImage],
    reconstruct_info: &mut ReconstructionInfo,
) -> Result<()> {
    // track to see how far we got in progressive encoding in case of truncated images, however this
    // was never actually implemented in the original C++ code
    reconstruct_info.max_sah = max(reconstruct_info.max_sah, max(jf.cs_sal, jf.cs_sah));

    let mut bit_reader = BitReader::new(reader);

    // init variables for decoding
    let mut state = JpegPositionState::new(jf, 0);

    // JPEG imagedata decoding routines
    let mut sta = JpegDecodeStatus::DecodeInProgress;
    while sta != JpegDecodeStatus::ScanCompleted {
        // decoding for interleaved data
        state.reset_rstw(&jf); // restart wait counter

        if jf.cs_to == 0 {
            if jf.cs_sah == 0 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "progress can't have two DC first stages",
                )
                .context();
            }

            // only need DC
            jf.verify_huffman_table(true, false).context()?;

            while sta == JpegDecodeStatus::DecodeInProgress {
                let current_block = image_data[state.get_cmp()].get_block_mut(state.get_dpos());

                // ---> progressive DC encoding <---

                // ---> succesive approximation later stage <---
                let value = bit_reader.read(1)? as i16;

                current_block.set_transposed_from_zigzag(
                    0,
                    current_block
                        .get_transposed_from_zigzag(0)
                        .wrapping_add(value << jf.cs_sal),
                );

                sta = state.next_mcu_pos(jf);
            }
        } else {
            // ---> progressive AC encoding <---

            if jf.cs_from == 0 || jf.cs_to >= 64 || jf.cs_from >= jf.cs_to {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    format!(
                        "progressive encoding range was invalid {0} to {1}",
                        jf.cs_from, jf.cs_to
                    ),
                );
            }

            // only need AC
            jf.verify_huffman_table(false, true).context()?;

            if jf.cs_sah == 0 {
                if jf.cs_cmpc != 1 {
                    return err_exit_code(
                        ExitCode::UnsupportedJpeg,
                        "Progressive AC encoding cannot be interleaved",
                    );
                }

                // ---> succesive approximation first stage <---
                let mut block = [0; 64];

                while sta == JpegDecodeStatus::DecodeInProgress {
                    let current_block = image_data[state.get_cmp()].get_block_mut(state.get_dpos());

                    if state.eobrun == 0 {
                        // only need to do something if we are not in a zero-block run
                        let eob = decode_ac_prg_fs(
                            &mut bit_reader,
                            jf.get_huff_ac_tree(state.get_cmp()),
                            &mut block,
                            &mut state,
                            jf.cs_from,
                            jf.cs_to,
                        )
                        .context()?;

                        state
                            .check_optimal_eobrun(
                                eob == jf.cs_from,
                                jf.get_huff_ac_codes(state.get_cmp()),
                            )
                            .context()?;

                        for bpos in jf.cs_from..eob {
                            current_block.set_transposed_from_zigzag(
                                usize::from(bpos),
                                block[usize::from(bpos)] << jf.cs_sal,
                            );
                        }
                    }

                    sta = state.skip_eobrun(&jf).context()?;

                    // proceed only if no error encountered
                    if sta == JpegDecodeStatus::DecodeInProgress {
                        sta = state.next_mcu_pos(jf);
                    }
                }
            } else {
                // ---> succesive approximation later stage <---

                let mut block = [0; 64];

                while sta == JpegDecodeStatus::DecodeInProgress {
                    let current_block = image_data[state.get_cmp()].get_block_mut(state.get_dpos());

                    for bpos in jf.cs_from..jf.cs_to + 1 {
                        block[usize::from(bpos)] =
                            current_block.get_transposed_from_zigzag(usize::from(bpos));
                    }

                    if state.eobrun == 0 {
                        // decode block (long routine)
                        let eob = decode_ac_prg_sa(
                            &mut bit_reader,
                            jf.get_huff_ac_tree(state.get_cmp()),
                            &mut block,
                            &mut state,
                            jf.cs_from,
                            jf.cs_to,
                        )
                        .context()?;

                        state
                            .check_optimal_eobrun(
                                eob == jf.cs_from,
                                jf.get_huff_ac_codes(state.get_cmp()),
                            )
                            .context()?;
                    } else {
                        // decode zero run block (short routine)
                        decode_eobrun_sa(
                            &mut bit_reader,
                            &mut block,
                            &mut state,
                            jf.cs_from,
                            jf.cs_to,
                        )
                        .context()?;
                    }

                    // copy back to colldata
                    for bpos in jf.cs_from..jf.cs_to + 1 {
                        current_block.set_transposed_from_zigzag(
                            usize::from(bpos),
                            current_block
                                .get_transposed_from_zigzag(usize::from(bpos))
                                .wrapping_add(block[usize::from(bpos)] << jf.cs_sal),
                        );
                    }

                    sta = state.next_mcu_pos(jf);
                }
            }
        }

        // if we saw a pad bit at the end of the block, then remember whether they were 1s or 0s. This
        // will be used later on to reconstruct the padding
        bit_reader
            .read_and_verify_fill_bits(&mut reconstruct_info.pad_bit)
            .context()?;

        // verify that we got the right RST code here since the above should do 1 mcu.
        // If we didn't then we won't re-encode the file binary identical so there's no point in continuing
        if sta == JpegDecodeStatus::RestartIntervalExpired {
            bit_reader.verify_reset_code().context()?;

            sta = JpegDecodeStatus::DecodeInProgress;
        }
    }

    Ok(())
}

/// reads an entire interval until the RST code
fn decode_baseline_rst<R: BufRead + Seek>(
    state: &mut JpegPositionState,
    bit_reader: &mut BitReader<R>,
    image_data: &mut [BlockBasedImage],
    do_handoff: &mut bool,
    jpeg_header: &JpegHeader,
    reconstruct_info: &mut ReconstructionInfo,
    partitions: &mut Vec<(u64, RestartSegmentCodingInfo)>,
) -> Result<JpegDecodeStatus> {
    // should have both AC and DC components
    jpeg_header.verify_huffman_table(true, true).context()?;

    let mut sta = JpegDecodeStatus::DecodeInProgress;
    let mut lastdc = [0i16; 4]; // (re)set last DCs for diff coding

    while sta == JpegDecodeStatus::DecodeInProgress {
        if *do_handoff {
            let (bits_already_read, byte_being_read) = bit_reader.overhang();

            partitions.push((
                bit_reader.stream_position(),
                RestartSegmentCodingInfo::new(
                    byte_being_read,
                    bits_already_read,
                    lastdc,
                    state.get_mcu(),
                    &jpeg_header,
                ),
            ));

            *do_handoff = false;
        }

        if !bit_reader.is_eof() {
            // record the max block read
            reconstruct_info.max_dpos[state.get_cmp()] =
                cmp::max(state.get_dpos(), reconstruct_info.max_dpos[state.get_cmp()]);
        }

        // decode block (throws on error)
        let mut block = [0i16; 64];
        let eob = decode_block_seq(
            bit_reader,
            &jpeg_header.get_huff_dc_tree(state.get_cmp()),
            &jpeg_header.get_huff_ac_tree(state.get_cmp()),
            &mut block,
        )?;

        if eob > 1 && (block[eob - 1] == 0) {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "cannot encode image with eob after last 0",
            );
        }

        // fix dc
        block[0] = block[0].wrapping_add(lastdc[state.get_cmp()]);
        lastdc[state.get_cmp()] = block[0];

        // prepare and set transposed raster block from zigzagged
        let block_tr = AlignedBlock::zigzag_to_transposed(block);

        image_data[state.get_cmp()].set_block_data(state.get_dpos(), block_tr);

        // see if here is a good position to do a handoff (has to be aligned between MCU rows since we can't split any finer)
        let old_mcu = state.get_mcu();
        sta = state.next_mcu_pos(&jpeg_header);

        if state.get_mcu() % jpeg_header.mcuh == 0 && old_mcu != state.get_mcu() {
            *do_handoff = true;
        }

        if bit_reader.is_eof() {
            sta = JpegDecodeStatus::ScanCompleted;
            reconstruct_info.early_eof_encountered = true;
        }
    }

    return Ok(sta);
}

/// <summary>
/// sequential block decoding routine
/// </summary>
#[inline(never)]
pub(crate) fn decode_block_seq<R: BufRead>(
    bit_reader: &mut BitReader<R>,
    dctree: &HuffTree,
    actree: &HuffTree,
    block: &mut [i16; 64],
) -> Result<usize> {
    let mut eob = 64;

    // decode dc
    block[0] = read_dc(bit_reader, dctree)?;

    let mut eof_fixup = false;

    // decode ac
    let mut bpos: usize = 1;
    while bpos < 64 {
        // decode next
        if let Some((z, coef)) = read_coef(bit_reader, actree)? {
            if (z + bpos) >= 64 {
                eof_fixup = true;
                break;
            }

            // no need to write the zeros since we are already zero initialized
            bpos += z;

            block[bpos] = coef;
            bpos += 1;
        } else {
            // EOB
            eob = bpos;
            break;
        }
    }

    // if we hit EOF then the bitreader will just start returning long strings of 0s, so handle that. If this happenes
    // outside of that case, then it's a JPEG that we cannot recode successfully
    if eof_fixup {
        if !bit_reader.is_eof() {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "If 0run is longer than the block must be truncated",
            );
        }

        while bpos < eob {
            block[bpos] = 0;
            bpos += 1;
        }

        if eob > 0 {
            block[eob - 1] = 1; // set the value to something matching the EOB
        }
    }

    // return position of eob
    return Ok(eob);
}

/// Reads and decodes next Huffman code from BitReader using the provided tree
#[inline(always)]
fn next_huff_code<R: BufRead>(bit_reader: &mut BitReader<R>, ctree: &HuffTree) -> Result<u8> {
    let mut node: u16 = 0;

    while node < 256 {
        node = ctree.node[usize::from(node)][usize::from(bit_reader.read(1)?)];
    }

    if node == 0xffff {
        err_exit_code(ExitCode::UnsupportedJpeg, "illegal Huffman code detected")
    } else {
        Ok((node - 256) as u8)
    }
}

fn read_dc<R: BufRead>(bit_reader: &mut BitReader<R>, tree: &HuffTree) -> Result<i16> {
    let (z, coef) = read_coef(bit_reader, tree)?.unwrap_or((0, 0));
    if z != 0 {
        err_exit_code(
            ExitCode::UnsupportedJpeg,
            "not expecting non-zero run in DC coefficient",
        )
    } else {
        Ok(coef)
    }
}

#[inline(always)]
fn read_coef<R: BufRead>(
    bit_reader: &mut BitReader<R>,
    tree: &HuffTree,
) -> Result<Option<(usize, i16)>> {
    // if the code we found is smaller or equal to the number of bits left, take the shortcut
    let hc;

    loop {
        // peek ahead to see if we can decode the symbol immediately
        // given what has already been read into the bitreader
        let (peek_value, peek_len) = bit_reader.peek();

        // use lookup table to figure out the first code in this byte and how long it is
        let (code, code_len) = tree.peek_code[peek_value as usize];

        if u32::from(code_len) <= peek_len {
            // found code directly, so advance by the number of bits immediately
            hc = code;
            bit_reader.advance(u32::from(code_len));
            break;
        } else if peek_len < 8 {
            // peek code works with up to 8 bits at a time. If we had less
            // than this, then we need to read more bits into the bitreader
            bit_reader.fill_register(8)?;
        } else {
            // take slow path since we have a code that is bigger than 8 bits (but pretty rare)
            hc = next_huff_code(bit_reader, tree)?;
            break;
        }
    }

    // analyse code
    if hc != 0 {
        let z = usize::from(lbits(hc, 4));
        let literal_bits = rbits(hc, 4);

        let value = bit_reader.read(u32::from(literal_bits))?;
        Ok(Some((z, devli(literal_bits, value))))
    } else {
        Ok(None)
    }
}

/// progressive AC decoding (first pass)
fn decode_ac_prg_fs<R: BufRead>(
    bit_reader: &mut BitReader<R>,
    actree: &HuffTree,
    block: &mut [i16; 64],
    state: &mut JpegPositionState,
    from: u8,
    to: u8,
) -> Result<u8> {
    debug_assert!(state.eobrun == 0);

    // decode ac
    let mut bpos = from;
    while bpos <= to {
        // decode next
        let hc = next_huff_code(bit_reader, actree)?;

        let l = lbits(hc, 4);
        let r = rbits(hc, 4);

        // check if code is not an EOB or EOB run
        if (l == 15) || (r > 0) {
            // decode run/level combination
            let mut z = l;
            let s = r;
            let n = bit_reader.read(u32::from(s))?;
            if (z + bpos) > to {
                return err_exit_code(ExitCode::UnsupportedJpeg, "run is too long");
            }

            while z > 0 {
                // write zeroes
                block[usize::from(bpos)] = 0;
                z -= 1;
                bpos += 1;
            }
            block[usize::from(bpos)] = devli(s, n); // decode cvli
            bpos += 1;
        } else {
            // decode eobrun
            let s = l;
            let n = bit_reader.read(u32::from(s))?;
            state.eobrun = decode_eobrun_bits(s, n);

            state.eobrun -= 1; // decrement eobrun ( for this one )

            break;
        }
    }

    // return position of eob
    return Ok(bpos);
}

/// progressive AC SA decoding routine
fn decode_ac_prg_sa<R: BufRead>(
    bit_reader: &mut BitReader<R>,
    actree: &HuffTree,
    block: &mut [i16; 64],
    state: &mut JpegPositionState,
    from: u8,
    to: u8,
) -> Result<u8> {
    debug_assert!(state.eobrun == 0);

    let mut bpos = from;
    let mut eob = to;

    // decode AC succesive approximation bits
    while bpos <= to {
        // decode next
        let hc = next_huff_code(bit_reader, actree)?;

        let l = lbits(hc, 4);
        let r = rbits(hc, 4);

        // check if code is not an EOB or EOB run
        if (l == 15) || (r > 0) {
            // decode run/level combination
            let mut z = l;
            let s = r;
            let v;

            if s == 0 {
                v = 0;
            } else if s == 1 {
                let n = bit_reader.read(1)?;
                v = if n == 0 { -1 } else { 1 }; // fast decode vli
            } else {
                return err_exit_code(ExitCode::UnsupportedJpeg, "decoding error").context();
            }

            // write zeroes / write correction bits
            loop {
                if block[usize::from(bpos)] == 0 {
                    // skip zeroes / write value
                    if z > 0 {
                        z -= 1;
                    } else {
                        block[usize::from(bpos)] = v;
                        bpos += 1;
                        break;
                    }
                } else {
                    // read correction bit
                    let n = bit_reader.read(1)? as i16;
                    block[usize::from(bpos)] = if block[usize::from(bpos)] > 0 { n } else { -n };
                }

                if bpos >= to {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "decoding error").context();
                }

                bpos += 1;
            }
        } else {
            // decode eobrun
            eob = bpos;
            let s = l;
            let n = bit_reader.read(u32::from(s))?;
            state.eobrun = decode_eobrun_bits(s, n);

            // since we hit EOB, the rest can be done with the zero block decoder
            decode_eobrun_sa(bit_reader, block, state, bpos, to)?;

            break;
        }
    }

    return Ok(eob);
}

/// fast eobrun decoding routine for succesive approximation when the entire block is zero
fn decode_eobrun_sa<R: BufRead>(
    bit_reader: &mut BitReader<R>,
    block: &mut [i16; 64],
    state: &mut JpegPositionState,
    from: u8,
    to: u8,
) -> Result<()> {
    debug_assert!(state.eobrun > 0);

    for bpos in usize::from(from)..usize::from(to + 1) {
        if block[bpos] != 0 {
            let n = bit_reader.read(1)? as i16;
            block[bpos] = if block[bpos] > 0 { n } else { -n };
        }
    }

    // decrement eobrun
    state.eobrun -= 1;

    Ok(())
}

/// decoding for decoding eobrun lengths. The encoding chops off the most significant
/// bit since it is always 1, so we need to add it back.
fn decode_eobrun_bits(s: u8, n: u16) -> u16 {
    n + (1 << s)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        EnabledFeatures,
        jpeg::jpeg_header::{JpegHeader, ReconstructionInfo},
    };
    use std::io::{BufRead, Seek};

    #[test]
    fn read_garbage_behavior_progressive() {
        read_garbage_behavior("iphoneprogressive");
    }

    #[test]
    fn read_garbage_behavior_baseline() {
        read_garbage_behavior("iphone");
    }

    /// reads a JPEG file and verifies that the garbage data is handled correctly.
    fn read_garbage_behavior(filename: &str) {
        let mut file = read_file(filename, ".jpg");
        let mut enabled_features = crate::EnabledFeatures::compat_lepton_scalar_read();

        let mut cursor = std::io::Cursor::new(&file);
        let (rinfo, _jh) = read_jpeg(&mut cursor, &enabled_features);

        assert_eq!(
            &rinfo.garbage_data[..],
            [0xff, 0xd9],
            "Expected garbage data to match what was written"
        );

        // now add some garbage data to the end of the file
        file.extend_from_slice(b"hi"); // EOI + some garbage

        let mut cursor = std::io::Cursor::new(&file);
        let (rinfo, _jh) = read_jpeg(&mut cursor, &enabled_features);

        assert_eq!(
            &rinfo.garbage_data[..],
            [0xff, 0xd9, b'h', b'i'],
            "Expected garbage data to match what was written"
        );

        enabled_features.stop_reading_at_eoi = true;
        let mut cursor = std::io::Cursor::new(&file);
        let (rinfo, _jh) = read_jpeg(&mut cursor, &enabled_features);

        assert_eq!(cursor.position(), file.len() as u64 - 2);

        assert_eq!(
            &rinfo.garbage_data[..],
            [0xff, 0xd9],
            "Expected garbage data to match what was written when stop_reading_at_eoi is true"
        );
    }

    /// test function to read a JPEG file and returns the reconstruction info and JPEG header
    fn read_jpeg<R: BufRead + Seek>(
        reader: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> (ReconstructionInfo, JpegHeader) {
        let mut jpeg_header = JpegHeader::default();
        let mut rinfo = ReconstructionInfo::default();

        let mut headers = Vec::new();

        let (_image_data, _partitions, _end_scan_position) = read_jpeg_file(
            reader,
            &mut jpeg_header,
            &mut rinfo,
            &enabled_features,
            |header, raw_header| {
                headers.push((header.clone(), raw_header.to_vec()));
            },
        )
        .unwrap();

        (rinfo, jpeg_header)
    }

    #[test]
    fn test_benchmark_read_block() {
        let mut f = benchmarks::benchmark_read_block();
        for _ in 0..10 {
            f();
        }
    }

    #[test]
    fn test_benchmark_read_jpeg() {
        let mut f = benchmarks::benchmark_read_jpeg();
        for _ in 0..10 {
            f();
        }
    }
}

#[cfg(any(test, feature = "micro_benchmark"))]
pub mod benchmarks {
    use std::io::Cursor;

    use crate::{
        EnabledFeatures,
        helpers::read_file,
        jpeg::{
            bit_reader::BitReader,
            bit_writer::BitWriter,
            block_based_image::AlignedBlock,
            jpeg_header::{
                HuffTree, JpegHeader, ReconstructionInfo, generate_huff_table_from_distribution,
            },
            jpeg_read::{decode_block_seq, read_jpeg_file},
            jpeg_write::encode_block_seq,
        },
    };

    /// reads the jpeg file from the test data and returns a closure that reads
    /// the jpeg header from it. Used for micro-benchmarking the jpeg header read performance.
    #[inline(never)]
    pub fn benchmark_read_jpeg() -> Box<dyn FnMut()> {
        let file = read_file("android", ".jpg");

        Box::new(move || {
            use std::hint::black_box;

            let mut reader = std::io::Cursor::new(&file);
            let enabled_features = EnabledFeatures::compat_lepton_vector_write();

            let mut jpeg_header = JpegHeader::default();
            let mut rinfo = ReconstructionInfo::default();

            let (image_data, partitions, end_scan) = read_jpeg_file(
                &mut reader,
                &mut jpeg_header,
                &mut rinfo,
                &enabled_features,
                |_, _| {},
            )
            .unwrap();

            black_box((image_data, partitions, end_scan));
        })
    }

    /// tests performance of decoding a single block
    #[inline(never)]
    pub fn benchmark_read_block() -> Box<dyn FnMut()> {
        // create a weird distribution to test the huffman encoding for corner cases
        let mut dcdistribution = [0; 256];
        for i in 0..256 {
            dcdistribution[i] = 256 - i;
        }
        let dctbl = generate_huff_table_from_distribution(&dcdistribution);

        let mut acdistribution = [0; 256];
        for i in 0..256 {
            acdistribution[i] = 1 + 256;
        }
        let actbl = generate_huff_table_from_distribution(&acdistribution);

        let mut bitwriter = BitWriter::new(Vec::with_capacity(1024));

        let mut block = AlignedBlock::default();
        for i in 0..10 {
            block.get_block_mut()[i] = i as i16 * 13;
        }
        for i in 30..50 {
            block.get_block_mut()[i] = -(i as i16) * 7;
        }
        for i in 50..52 {
            block.get_block_mut()[i] = i as i16 * 3;
        }

        encode_block_seq(&mut bitwriter, &dctbl, &actbl, &block);

        let buffer = bitwriter.detach_buffer();

        let dctree = HuffTree::construct_hufftree(&dctbl, true).unwrap();
        let actree = HuffTree::construct_hufftree(&actbl, false).unwrap();

        Box::new(move || {
            use std::hint::black_box;

            let mut bitreader = BitReader::new(Cursor::new(&buffer));

            let mut outblock = AlignedBlock::default();
            decode_block_seq(
                &mut bitreader,
                &dctree,
                &actree,
                &mut outblock.get_block_mut(),
            )
            .unwrap();

            black_box(outblock);
        })
    }
}
