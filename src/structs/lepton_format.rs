/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cpu_time::ThreadTime;
use log::{info, warn};
use std::cmp;
use std::io::{Cursor, ErrorKind, Read, Seek, SeekFrom, Write};
use std::mem::swap;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::thread::ScopedJoinHandle;
use std::time::Instant;

use anyhow::{Context, Result};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::consts::*;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::jpeg_code;
use crate::lepton_error::ExitCode;
use crate::metrics::Metrics;
use crate::structs::bit_writer::BitWriter;
use crate::structs::block_based_image::BlockBasedImage;
use crate::structs::jpeg_header::JPegHeader;
use crate::structs::jpeg_write::jpeg_write_row_range;
use crate::structs::lepton_decoder::lepton_decode_row_range;
use crate::structs::lepton_encoder::lepton_encode_row_range;
use crate::structs::probability_tables_set::ProbabilityTablesSet;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::thread_handoff::ThreadHandoff;
use crate::structs::truncate_components::TruncateComponents;

use super::jpeg_read::{read_progressive_scan, read_scan};
use super::jpeg_write::jpeg_write_entire_scan;

/// reads a lepton file and writes it out as a jpeg
pub fn decode_lepton_wrapper<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    num_threads: usize,
) -> Result<Metrics> {
    // figure out how long the input is
    let orig_pos = reader.stream_position()?;
    let size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(orig_pos))?;

    let mut lh = LeptonHeader::new();

    lh.read_lepton_header(reader).context(here!())?;

    let metrics = lh
        .recode_jpeg(writer, reader, size, num_threads)
        .context(here!())?;

    return Ok(metrics);
}

/// reads a jpeg and writes it out as a lepton file
pub fn encode_lepton_wrapper<R: Read + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    max_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    let (lp, image_data) = read_jpeg(reader, enabled_features, max_threads, |_jh| {})?;

    lp.write_lepton_header(writer).context(here!())?;

    let metrics = run_lepton_encoder_threads(
        &lp.jpeg_header,
        &lp.truncate_components,
        writer,
        &lp.thread_handoff[..],
        &image_data[..],
    )
    .context(here!())?;

    let final_file_size = writer.stream_position()? + 4;

    writer
        .write_u32::<LittleEndian>(final_file_size as u32)
        .context(here!())?;

    Ok(metrics)
}

/// Encodes JPEG as compressed Lepton format, verifies roundtrip in buffer. Requires everything to be buffered
/// since we need to pass through the data multiple times
pub fn encode_lepton_wrapper_verify(
    input_data: &[u8],
    max_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<(Vec<u8>, Metrics)> {
    let mut output_data = Vec::with_capacity(input_data.len());

    info!("compressing to Lepton format");

    let mut reader = Cursor::new(&input_data);
    let mut writer = Cursor::new(&mut output_data);

    let mut metrics = encode_lepton_wrapper(
        &mut reader,
        &mut writer,
        max_threads as usize,
        &enabled_features,
    )
    .context(here!())?;

    // decode and compare to original in order to enure we encoded correctly

    let mut verify_buffer = Vec::with_capacity(input_data.len());
    let mut verifyreader = Cursor::new(&output_data[..]);

    info!("decompressing to verify contents");

    metrics.merge_from(
        decode_lepton_wrapper(&mut verifyreader, &mut verify_buffer, max_threads)
            .context(here!())?,
    );

    if input_data.len() != verify_buffer.len() {
        return err_exit_code(
            ExitCode::VerificationLengthMismatch,
            format!(
                "ERROR mismatch input_len = {0}, decoded_len = {1}",
                input_data.len(),
                verify_buffer.len()
            )
            .as_str(),
        );
    }

    if input_data[..] != verify_buffer[..] {
        return err_exit_code(
            ExitCode::VerificationContentMismatch,
            "ERROR mismatching data (but same size)",
        );
    }

    Ok((output_data, metrics))
}

/// reads JPEG and returns corresponding header and image vector. This encapsulate all
/// JPEG reading code, including baseline and progressive images.
///
/// The callback is called for each jpeg header that is parsed, which
/// is currently only used by the dump utility for debugging purposes.
pub fn read_jpeg<R: Read + Seek>(
    reader: &mut R,
    enabled_features: &EnabledFeatures,
    max_threads: usize,
    callback: fn(&JPegHeader),
) -> Result<(LeptonHeader, Vec<BlockBasedImage>)> {
    let mut startheader = [0u8; 2];
    reader.read_exact(&mut startheader)?;
    if startheader[0] != 0xFF || startheader[1] != jpeg_code::SOI {
        return err_exit_code(ExitCode::UnsupportedJpeg, "header invalid");
    }

    let mut lp = LeptonHeader::new();
    if !prepare_to_decode_next_scan(&mut lp, reader, enabled_features).context(here!())? {
        return err_exit_code(ExitCode::UnsupportedJpeg, "JPeg does not contain scans");
    }

    callback(&lp.jpeg_header);

    if !enabled_features.progressive && lp.jpeg_header.jpeg_type == JPegType::Progressive {
        return err_exit_code(
            ExitCode::ProgressiveUnsupported,
            "file is progressive, but this is disabled",
        )
        .context(here!());
    }

    if lp.jpeg_header.cmpc > COLOR_CHANNEL_NUM_BLOCK_TYPES {
        return err_exit_code(
            ExitCode::Unsupported4Colors,
            " can't support this kind of image",
        )
        .context(here!());
    }

    lp.truncate_components.init(&lp.jpeg_header);
    let mut image_data = Vec::<BlockBasedImage>::new();
    for i in 0..lp.jpeg_header.cmpc {
        // constructor takes height in proportion to the component[0]
        image_data.push(BlockBasedImage::new(
            &lp.jpeg_header,
            i,
            0,
            lp.jpeg_header.cmp_info[0].bcv,
        ));
    }

    let mut thread_handoff = Vec::<ThreadHandoff>::new();
    let start_scan = reader.stream_position()? as i32;
    read_scan(&mut lp, reader, &mut thread_handoff, &mut image_data[..]).context(here!())?;
    lp.scnc += 1;

    let mut end_scan = reader.stream_position()? as i32;

    // need at least two bytes of scan data
    if start_scan + 2 > end_scan || thread_handoff.len() == 0 {
        return err_exit_code(
            ExitCode::UnsupportedJpeg,
            "couldnt find any sections to encode",
        )
        .context(here!());
    }

    for i in 0..thread_handoff.len() {
        thread_handoff[i].segment_offset_in_file += start_scan;

        #[cfg(feature = "detailed_tracing")]
        info!(
            "Crystalize: s:{0} ls: {1} le: {2} o: {3} nb: {4}",
            thread_handoff[i].segment_offset_in_file,
            thread_handoff[i].luma_y_start,
            thread_handoff[i].luma_y_end,
            thread_handoff[i].overhang_byte,
            thread_handoff[i].num_overhang_bits
        );
    }

    if lp.jpeg_header.jpeg_type == JPegType::Sequential {
        if lp.early_eof_encountered {
            lp.truncate_components
                .set_truncation_bounds(&lp.jpeg_header, lp.max_dpos);

            // If we got an early EOF, then seek backwards and capture the last two bytes and store them as garbage.
            // This is necessary since the decoder will assume that zero garbage always means a properly terminated JPEG
            // even if early EOF was set to true.
            reader.seek(SeekFrom::Current(-2))?;
            lp.garbage_data.resize(2, 0);
            reader.read_exact(&mut lp.garbage_data)?;

            // take these two last bytes off the last segment. For some reason the C++/CS version only chop of one byte
            // and then fix up the broken file later in the decoder. The following logic will create a valid file
            // that the C++ and CS version will still decode properly without the fixup logic.
            let len = thread_handoff.len();
            thread_handoff[len - 1].segment_size -= 2;
        }

        // rest of data is garbage data if it is a sequential jpeg (including EOI marker)
        reader.read_to_end(&mut lp.garbage_data).context(here!())?;
    } else {
        assert!(lp.jpeg_header.jpeg_type == JPegType::Progressive);

        if lp.early_eof_encountered {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "truncation is only supported for baseline images",
            )
            .context(here!());
        }

        // for progressive images, loop around reading headers and decoding until we a complete image_data
        while prepare_to_decode_next_scan(&mut lp, reader, enabled_features).context(here!())? {
            callback(&lp.jpeg_header);

            read_progressive_scan(&mut lp, reader, &mut image_data[..]).context(here!())?;
            lp.scnc += 1;

            if lp.early_eof_encountered {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "truncation is only supported for baseline images",
                )
                .context(here!());
            }
        }

        end_scan = reader.stream_position()? as i32;

        // since prepare_to_decode_next_scan consumes the EOI,
        // we need to add it to the beginning of the garbage data (if there is any)
        lp.garbage_data = Vec::from(EOI);

        // append the rest of the file to the buffer
        if reader.read_to_end(&mut lp.garbage_data).context(here!())? == 0 {
            // no need to record EOI garbage data if there wasn't anything read
            lp.garbage_data.clear();
        }
    }

    set_segment_size_in_row_thread_handoffs(&mut thread_handoff[..], end_scan as i32);
    let merged_handoffs = split_row_handoffs_to_threads(&thread_handoff[..], max_threads);
    lp.thread_handoff = merged_handoffs;
    lp.jpeg_file_size = reader.stream_position().context(here!())? as u32;
    Ok((lp, image_data))
}

fn run_lepton_decoder_threads<R: Read + Seek, P: Send>(
    lh: &LeptonHeader,
    reader: &mut R,
    last_data_position: u64,
    max_threads_to_use: usize,
    process: fn(
        thread_handoff: &ThreadHandoff,
        image_data: Vec<BlockBasedImage>,
        lh: &LeptonHeader,
    ) -> Result<P>,
) -> Result<(Metrics, Vec<P>)> {
    let wall_time = Instant::now();

    let pts = ProbabilityTablesSet::new();
    let mut qt = Vec::new();
    for i in 0..lh.jpeg_header.cmpc {
        let qtables = QuantizationTables::new(&lh.jpeg_header, i);

        // check to see if quantitization table was properly initialized
        // (table contains divisors for coefficients so it never should have a zero)
        if qtables.get_quantization_table()[0] == 0 {
            return err_exit_code(ExitCode::UnsupportedJpeg, "Quantization table is missing");
        }
        qt.push(qtables);
    }

    let r = thread::scope(|s| -> Result<(Metrics, Vec<P>)> {
        let mut running_threads: Vec<ScopedJoinHandle<Result<(P, Metrics)>>> = Vec::new();
        let mut channel_to_sender = Vec::new();

        let pts_ref = &pts;
        let q_ref = &qt[..];

        // don't use more threads than we need
        let m = cmp::min(max_threads_to_use, lh.thread_handoff.len());

        info!(
            "decoding {0} multipexed streams with {1} threads",
            lh.thread_handoff.len(),
            m
        );

        // ratio of threads to work items
        let ratio = lh.thread_handoff.len() as f32 / m as f32;

        for t in 0..m {
            let start = (t as f32 * ratio) as usize;
            let end = ((t + 1) as f32 * ratio) as usize;

            let iter_per_thread = end - start;

            let mut rx_channels = Vec::new();
            for _k in 0..iter_per_thread {
                let (tx, rx) = channel();
                channel_to_sender.push(tx);
                rx_channels.push(Some(rx));
            }

            running_threads.push(s.spawn(move || -> Result<(P, Metrics)> {
                let cpu_time = ThreadTime::now();

                // determine how much we are going to write in total to presize the buffer
                let mut decoded_size = 0;
                for thread_id in start..end {
                    decoded_size += lh.thread_handoff[thread_id].segment_size as usize;
                }

                // create a combined handoff that merges all the sections we have read so we process them in one go
                let combined_thread_handoff = ThreadHandoff {
                    luma_y_start: lh.thread_handoff[start].luma_y_start,
                    luma_y_end: lh.thread_handoff[end - 1].luma_y_end,
                    segment_offset_in_file: lh.thread_handoff[start].segment_offset_in_file,
                    segment_size: decoded_size as i32,
                    overhang_byte: lh.thread_handoff[start].overhang_byte,
                    num_overhang_bits: lh.thread_handoff[start].num_overhang_bits,
                    last_dc: lh.thread_handoff[start].last_dc.clone(),
                };

                let mut image_data = Vec::new();
                for i in 0..lh.jpeg_header.cmpc {
                    image_data.push(BlockBasedImage::new(
                        &lh.jpeg_header,
                        i,
                        combined_thread_handoff.luma_y_start,
                        if t == m - 1 {
                            // if this is the last thead, then the image should extend all the way to the bottom
                            lh.jpeg_header.cmp_info[0].bcv
                        } else {
                            combined_thread_handoff.luma_y_end
                        },
                    ));
                }

                let mut metrics = Metrics::default();

                // now run the range of thread handoffs in the file that this thread is supposed to handle
                for thread_id in start..end {
                    // get the appropriate receiver so we can read out data from it
                    let rx = rx_channels[thread_id - start].take().context(here!())?;
                    let mut reader = MessageReceiver {
                        thread_id: thread_id as u8,
                        current_buffer: Cursor::new(Vec::new()),
                        receiver: rx,
                        end_of_file: false,
                    };

                    metrics.merge_from(
                        lepton_decode_row_range(
                            pts_ref,
                            q_ref,
                            &lh.truncate_components,
                            &mut image_data,
                            &mut reader,
                            lh.thread_handoff[thread_id].luma_y_start,
                            lh.thread_handoff[thread_id].luma_y_end,
                            thread_id == lh.thread_handoff.len() - 1,
                            true,
                        )
                        .context(here!())?,
                    );
                }

                let process_result = process(&combined_thread_handoff, image_data, lh)?;

                metrics.record_cpu_worker_time(cpu_time.elapsed());

                Ok((process_result, metrics))
            }));
        }

        // now that the threads are waiting for inptut, read the stream and send all the buffers to their respective readers
        while reader.stream_position().context(here!())? < last_data_position - 4 {
            let thread_marker = reader.read_u8().context(here!())?;
            let thread_id = (thread_marker & 0xf) as u8;

            if thread_id >= channel_to_sender.len() as u8 {
                return err_exit_code(
                    ExitCode::BadLeptonFile,
                    format!(
                        "invalid thread_id at {0} of {1} at {2}",
                        reader.stream_position().unwrap(),
                        last_data_position,
                        here!()
                    )
                    .as_str(),
                );
            }

            let data_length = if thread_marker < 16 {
                let b0 = reader.read_u8().context(here!())?;
                let b1 = reader.read_u8().context(here!())?;

                ((b1 as usize) << 8) + b0 as usize + 1
            } else {
                // This format is used by Lepton C++ to write encoded chunks with length of 4096, 16384 or 65536 bytes
                let flags = (thread_marker >> 4) & 3;

                1024 << (2 * flags)
            };

            //info!("offset {0} len {1}", reader.stream_position()?-2, data_length);

            let mut buffer = Vec::<u8>::new();
            buffer.resize(data_length as usize, 0);
            reader.read_exact(&mut buffer).with_context(|| {
                format!(
                    "reading {0} bytes at {1} of {2} at {3}",
                    buffer.len(),
                    reader.stream_position().unwrap(),
                    last_data_position,
                    here!()
                )
            })?;

            channel_to_sender[thread_id as usize]
                .send(Message::WriteBlock(thread_id, buffer))
                .context(here!())?;
        }
        //info!("done sending!");

        for c in channel_to_sender {
            // ignore the result of send, since a thread may have already blown up with an error and we will get it when we join (rather than exiting with a useless channel broken message)
            let _ = c.send(Message::Eof);
        }

        let mut metrics = Metrics::default();

        let mut result = Vec::new();
        for i in running_threads.drain(..) {
            let thread_result = i.join().unwrap().context(here!())?;

            metrics.merge_from(thread_result.1);

            result.push(thread_result.0);
        }

        info!(
            "worker threads {0}ms of CPU time in {1}ms of wall time",
            metrics.get_cpu_time_worker_time().as_millis(),
            wall_time.elapsed().as_millis()
        );

        return Ok((metrics, result));
    })
    .context(here!())?;

    Ok(r)
}

/// runs the encoding threads and returns the total amount of CPU time consumed (including worker threads)
fn run_lepton_encoder_threads<W: Write + Seek>(
    jpeg_header: &JPegHeader,
    colldata: &TruncateComponents,
    writer: &mut W,
    thread_handoffs: &[ThreadHandoff],
    image_data: &[BlockBasedImage],
) -> Result<Metrics> {
    let wall_time = Instant::now();

    // Get number of threads. Verify that it is at most MAX_THREADS and fits in 4 bits for serialization.
    let num_threads = thread_handoffs.len();
    assert!(
        num_threads <= MAX_THREADS && num_threads <= MAX_THREADS_SUPPORTED_BY_LEPTON_FORMAT,
        "Too many thread handoffs"
    );

    // Prepare quantization tables
    let pts = ProbabilityTablesSet::new();
    let mut quantization_tables = Vec::new();
    for i in 0..image_data.len() {
        let qtables = QuantizationTables::new(jpeg_header, i);

        // check to see if quantitization table was properly initialized
        // (table contains divisors for coefficients so it never should have a zero)
        if qtables.get_quantization_table()[0] == 0 {
            return err_exit_code(ExitCode::UnsupportedJpeg, "Quantization table is missing");
        }
        quantization_tables.push(qtables);
    }

    let pts_ref = &pts;
    let q_ref = &quantization_tables[..];

    let mut sizes = Vec::<u64>::new();
    sizes.resize(thread_handoffs.len(), 0);

    let mut merged_metrics = Metrics::default();

    thread::scope(|s| -> Result<()> {
        let (tx, rx) = channel();

        let mut running_threads = Vec::new();

        for i in 0..thread_handoffs.len() {
            let cloned_sender = tx.clone();

            running_threads.push(s.spawn(move || -> Result<Metrics> {
                let cpu_time = ThreadTime::now();

                let thread_id = i;
                let mut thread_writer = MessageSender {
                    thread_id: thread_id as u8,
                    sender: cloned_sender,
                    buffer: Vec::with_capacity(WRITE_BUFFER_SIZE),
                };

                let mut range_metrics = lepton_encode_row_range(
                    pts_ref,
                    q_ref,
                    image_data,
                    &mut thread_writer,
                    thread_id as i32,
                    colldata,
                    thread_handoffs[thread_id].luma_y_start,
                    thread_handoffs[thread_id].luma_y_end,
                    thread_id == thread_handoffs.len() - 1,
                    true,
                )
                .context(here!())?;

                thread_writer.flush().context(here!())?;

                thread_writer.sender.send(Message::Eof).context(here!())?;

                range_metrics.record_cpu_worker_time(cpu_time.elapsed());

                Ok(range_metrics)
            }));
        }

        // drop the sender so that the channel breaks when all the threads exit
        drop(tx);

        // wait to collect work and done messages from all the threads
        let mut threads_left = thread_handoffs.len();

        while threads_left > 0 {
            let value = rx.recv().context(here!());
            match value {
                Ok(Message::Eof) => {
                    threads_left -= 1;
                }
                Ok(Message::WriteBlock(thread_id, b)) => {
                    let l = b.len() - 1;

                    writer.write_u8(thread_id).context(here!())?;
                    writer.write_u8((l & 0xff) as u8).context(here!())?;
                    writer.write_u8(((l >> 8) & 0xff) as u8).context(here!())?;
                    writer.write_all(&b[..]).context(here!())?;

                    sizes[thread_id as usize] += b.len() as u64;
                }
                Err(x) => {
                    // get the actual error that cause the channel to
                    // prematurely close
                    for result in running_threads.drain(..) {
                        let r = result.join().unwrap();
                        if let Err(e) = r {
                            return Err(e.context(here!()));
                        }
                    }

                    return Err(x);
                }
            }
        }

        for result in running_threads.drain(..) {
            merged_metrics.merge_from(result.join().unwrap().unwrap());
        }

        return Ok(());
    })
    .context(here!())?;

    info!(
        "scan portion of JPEG uncompressed size = {0}",
        sizes.iter().sum::<u64>()
    );

    info!(
        "worker threads {0}ms of CPU time in {1}ms of wall time",
        merged_metrics.get_cpu_time_worker_time().as_millis(),
        wall_time.elapsed().as_millis()
    );

    Ok(merged_metrics)
}

#[derive(Debug)]
pub struct LeptonHeader {
    /// raw jpeg header to be written back to the file when it is recreated
    pub raw_jpeg_header: Vec<u8>,

    /// how far we have read into the raw header, since the header is divided
    /// into multiple chucks for each scan. For example, a progressive image
    /// would start with the jpeg image segments, followed by a SOS (start of scan)
    /// after which comes the encoded jpeg coefficients, and once thats over
    /// we get another header segment until the next SOS, etc
    pub raw_jpeg_header_read_index: usize,

    pub thread_handoff: Vec<ThreadHandoff>,

    pub jpeg_header: JPegHeader,

    /// information about how to truncate the image if it was partially written
    pub truncate_components: TruncateComponents,

    pub rst_err: Vec<u8>,

    /// A list containing one entry for each scan segment.  Each entry contains the number of restart intervals
    /// within the corresponding scan segment.
    pub rst_cnt: Vec<i32>,

    /// the mask for padding out the bitstream when we get to the end of a reset block
    pub pad_bit: Option<u8>,

    pub rst_cnt_set: bool,

    /// garbage data (default value - empty segment - means no garbage data)
    pub garbage_data: Vec<u8>,

    /// count of scans encountered so far
    pub scnc: usize,

    pub early_eof_encountered: bool,

    /// the maximum dpos in a truncated image
    pub max_dpos: [i32; 4],

    /// the maximum component in a truncated image
    pub max_cmp: i32,

    /// the maximum band in a truncated image
    pub max_bpos: i32,

    /// the maximum bit in a truncated image
    pub max_sah: u8,

    pub jpeg_file_size: u32,

    /// on decompression, plain-text size
    pub plain_text_size: u32,

    /// on decompression, uncompressed lepton header size
    pub uncompressed_lepton_header_size: u32,
}

impl LeptonHeader {
    pub fn new() -> Self {
        return LeptonHeader {
            max_dpos: [0; 4],
            raw_jpeg_header: Vec::new(),
            raw_jpeg_header_read_index: 0,
            thread_handoff: Vec::new(),
            jpeg_header: JPegHeader::new(),
            truncate_components: TruncateComponents::new(),
            rst_err: Vec::new(),
            rst_cnt: Vec::new(),
            pad_bit: None,
            rst_cnt_set: false,
            garbage_data: Vec::new(),
            scnc: 0,
            early_eof_encountered: false,
            max_cmp: 0,
            max_bpos: 0,
            max_sah: 0,
            jpeg_file_size: 0,
            plain_text_size: 0,
            uncompressed_lepton_header_size: 0,
        };
    }

    fn recode_jpeg<R: Read + Seek, W: Write>(
        &mut self,
        writer: &mut W,
        reader: &mut R,
        last_data_position: u64,
        num_threads: usize,
    ) -> Result<Metrics, anyhow::Error> {
        writer.write_all(&SOI)?;

        // write the raw header as far as we've decoded it
        writer
            .write_all(&self.raw_jpeg_header[0..self.raw_jpeg_header_read_index])
            .context(here!())?;

        let metrics = if self.jpeg_header.jpeg_type == JPegType::Progressive {
            self.recode_progressive_jpeg(reader, last_data_position, writer, num_threads)
                .context(here!())?
        } else {
            self.recode_baseline_jpeg(reader, last_data_position, writer, num_threads)
                .context(here!())?
        };

        if !self.early_eof_encountered {
            /* step 3: blit any trailing header data */
            writer
                .write_all(&self.raw_jpeg_header[self.raw_jpeg_header_read_index..])
                .context(here!())?;
        }

        writer.write_all(&self.garbage_data).context(here!())?;
        Ok(metrics)
    }

    /// decodes the entire image and merges the results into a single set of BlockBaseImage per component
    pub fn decode_as_single_image<R: Read + Seek>(
        &mut self,
        reader: &mut R,
        last_data_position: u64,
        num_threads: usize,
    ) -> Result<(Vec<BlockBasedImage>, Metrics)> {
        // run the threads first, since we need everything before we can start decoding
        let (metrics, mut results) = run_lepton_decoder_threads(
            self,
            reader,
            last_data_position,
            num_threads,
            |_thread_handoff, image_data, _lh| {
                // just return the image data directly to be merged together
                return Ok(image_data);
            },
        )
        .context(here!())?;

        // merge the corresponding components so that we get a single set of coefficient maps (since each thread did a piece of the work)
        let mut merged = Vec::new();

        let num_components = results[0].len();
        for i in 0..num_components {
            merged.push(BlockBasedImage::merge(&mut results, i));
        }

        Ok((merged, metrics))
    }

    /// parses and advances to the next header segment out of raw_jpeg_header into the jpeg header
    pub fn advance_next_header_segment(
        &mut self,
        enabled_features: &EnabledFeatures,
    ) -> Result<bool> {
        let mut header_cursor =
            Cursor::new(&self.raw_jpeg_header[self.raw_jpeg_header_read_index..]);

        let result = self
            .jpeg_header
            .parse(&mut header_cursor, enabled_features)
            .context(here!())?;

        self.raw_jpeg_header_read_index += header_cursor.stream_position()? as usize;

        Ok(result)
    }

    /// progressive decoder, requires that the entire lepton file is processed first
    fn recode_progressive_jpeg<R: Read + Seek, W: Write>(
        &mut self,
        reader: &mut R,
        last_data_position: u64,
        writer: &mut W,
        num_threads: usize,
    ) -> Result<Metrics> {
        // run the threads first, since we need everything before we can start decoding
        let (merged, metrics) = self
            .decode_as_single_image(reader, last_data_position, num_threads)
            .context(here!())?;

        loop {
            // code another scan
            jpeg_write_entire_scan(writer, &merged[..], self).context(here!())?;

            // read the next headers (DHT, etc) while mirroring it back to the writer
            let old_pos = self.raw_jpeg_header_read_index;
            let result = self
                .advance_next_header_segment(&EnabledFeatures::all())
                .context(here!())?;

            writer
                .write_all(&self.raw_jpeg_header[old_pos..self.raw_jpeg_header_read_index])
                .context(here!())?;

            if !result {
                break;
            }

            // advance to next scan
            self.scnc += 1;
        }

        Ok(metrics)
    }

    // baseline decoder can run the jpeg encoder inside the worker thread vs progressive encoding which needs to get the entire set of coefficients first
    // since it runs throught it multiple times.
    fn recode_baseline_jpeg<R: Read + Seek, W: Write>(
        &mut self,
        reader: &mut R,
        last_data_position: u64,
        writer: &mut W,
        num_threads: usize,
    ) -> Result<Metrics> {
        // step 2: recode image data
        let (metrics, results) = run_lepton_decoder_threads(
            self,
            reader,
            last_data_position,
            num_threads,
            |thread_handoff, image_data, lh| {
                let mut result_buffer = Vec::with_capacity(thread_handoff.segment_size as usize);
                let mut cursor = Cursor::new(&mut result_buffer);

                let mut huffw = BitWriter::new();

                let _start_size = cursor.position();

                let max_coded_heights = lh.truncate_components.get_max_coded_heights();

                jpeg_write_row_range(
                    &mut cursor,
                    &image_data,
                    lh.truncate_components.mcu_count_vertical,
                    &thread_handoff,
                    &max_coded_heights[..],
                    &mut huffw,
                    lh,
                )
                .context(here!())?;

                #[cfg(detailed_tracing)]
                info!("ystart = {0}, segment_size = {1}, amount = {2}, offset = {3}, ob = {4}, nb = {5}", 
                combined_thread_handoff.luma_y_start,
                combined_thread_handoff.segment_size,
                cursor.position() - _start_size,
                combined_thread_handoff.segment_offset_in_file,
                combined_thread_handoff.overhang_byte,
                combined_thread_handoff.num_overhang_bits
            );

                if result_buffer.len() > thread_handoff.segment_size as usize {
                    warn!("warning: truncating segment");
                    result_buffer.resize(thread_handoff.segment_size as usize, 0);
                }

                return Ok(result_buffer);
            },
        )?;

        // write all the buffers that we collected
        for r in results {
            writer.write_all(&r[..]).context(here!())?;
        }

        // Injection of restart codes for RST errors supports JPEGs with trailing RSTs.
        // Run this logic even if early_eof_encountered to be compatible with C++ version.
        if self.rst_err.len() > 0 {
            let cumulative_reset_markers = if self.jpeg_header.rsti != 0 {
                ((self.jpeg_header.mcuh * self.jpeg_header.mcuv) - 1) / self.jpeg_header.rsti
            } else {
                0
            } as u8;
            for i in 0..self.rst_err[0] as u8 {
                let rst = (jpeg_code::RST0 + ((cumulative_reset_markers + i) & 7)) as u8;
                writer.write_u8(0xFF)?;
                writer.write_u8(rst)?;
            }
        }

        Ok(metrics)
    }

    /// reads the start of the lepton file and parses the compressed header. Returns the raw JPEG header contents.
    pub fn read_lepton_header<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        let mut header = [0 as u8; LEPTON_FILE_HEADER.len()];

        reader.read_exact(&mut header).context(here!())?;

        if !buffer_prefix_matches_marker(header, LEPTON_FILE_HEADER) {
            return err_exit_code(ExitCode::BadLeptonFile, "header doesn't match");
        }

        // Complicated logic of version compatibility should be verified by the caller.
        // Currently just matching the version version.
        let version = reader.read_u8().context(here!())?;
        if version != LEPTON_VERSION {
            return err_exit_code(
                ExitCode::VersionUnsupported,
                format!("incompatible file with version {0}", version).as_str(),
            );
        }

        let mut header = [0 as u8; 21];
        reader.read_exact(&mut header).context(here!())?;

        // Z = baseline non-progressive
        // Y = chunked encoding of a slice of a JPEG (not supported yet)
        // X = progressive
        if header[0] != LEPTON_HEADER_BASELINE_JPEG_TYPE[0]
            && header[0] != LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE[0]
        {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!("Unknown filetype in header {0}", header[0]).as_str(),
            );
        }

        let mut c = Cursor::new(header);

        // We use 12 bytes of git revision for our needs - mark that it's C# implementation and a not-compressed header size.
        self.uncompressed_lepton_header_size = 0;
        if header[5] == 'M' as u8 && header[6] == 'S' as u8 {
            c.set_position(7);
            self.uncompressed_lepton_header_size = c.read_u32::<LittleEndian>()?;
        }

        // full size of the original file
        c.set_position(17);
        self.plain_text_size = c.read_u32::<LittleEndian>()?;

        // now read the compressed header
        let compressed_header_size = reader.read_u32::<LittleEndian>()? as usize;

        if compressed_header_size > MAX_FILE_SIZE_BYTES as usize {
            return err_exit_code(ExitCode::BadLeptonFile, "Too big compressed header");
        }
        if self.plain_text_size > MAX_FILE_SIZE_BYTES as u32 {
            return err_exit_code(ExitCode::BadLeptonFile, "Only support images < 128 megs");
        }

        // limit reading to the compressed header
        let mut compressed_reader = reader.take(compressed_header_size as u64);

        self.raw_jpeg_header = self
            .read_lepton_compressed_header(&mut compressed_reader)
            .context(here!())?;

        // CMP marker
        let mut current_lepton_marker = [0 as u8; 3];
        reader.read_exact(&mut current_lepton_marker)?;
        if !buffer_prefix_matches_marker(current_lepton_marker, LEPTON_HEADER_COMPLETION_MARKER) {
            return err_exit_code(ExitCode::BadLeptonFile, "CMP marker not found");
        }

        self.raw_jpeg_header_read_index = 0;

        {
            let mut header_data_cursor = Cursor::new(&self.raw_jpeg_header[..]);
            self.jpeg_header
                .parse(&mut header_data_cursor, &EnabledFeatures::all())
                .context(here!())?;
            self.raw_jpeg_header_read_index = header_data_cursor.position() as usize;
        }

        self.truncate_components.init(&self.jpeg_header);

        if self.early_eof_encountered {
            self.truncate_components
                .set_truncation_bounds(&self.jpeg_header, self.max_dpos);
        }

        let num_threads = self.thread_handoff.len();

        // luma_y_end of the last thread is not serialized/deserialized, fill it here
        self.thread_handoff[num_threads - 1].luma_y_end =
            self.truncate_components.get_block_height(0);

        // if the last segment was too big to fit with the garbage data taken into account, shorten it
        // (a bit of broken logic in the encoder, but can't change it without breaking the file format)
        if self.early_eof_encountered {
            let mut max_last_segment_size = i32::try_from(self.plain_text_size)?
                - i32::try_from(self.garbage_data.len())?
                - i32::try_from(self.raw_jpeg_header_read_index)?
                - 2;

            // subtract the segment sizes of all the previous segments (except for the last)
            for i in 0..num_threads - 1 {
                max_last_segment_size -= self.thread_handoff[i].segment_size;
            }

            let last = &mut self.thread_handoff[num_threads - 1];

            if last.segment_size > max_last_segment_size {
                // re-adjust the last segment size
                last.segment_size = max_last_segment_size;
            }
        }

        Ok(())
    }

    /// helper for read_lepton_header. uncompresses and parses the contents of the compressed header. Returns the raw JPEG header.
    fn read_lepton_compressed_header<R: Read>(&mut self, src: &mut R) -> Result<Vec<u8>> {
        let mut header_reader = ZlibDecoder::new(src);

        let mut hdr_buf: [u8; 3] = [0; 3];
        header_reader.read_exact(&mut hdr_buf)?;

        if !buffer_prefix_matches_marker(hdr_buf, LEPTON_HEADER_MARKER) {
            return err_exit_code(ExitCode::BadLeptonFile, "HDR marker not found");
        }

        let hdrs = header_reader.read_u32::<LittleEndian>()? as usize;

        let mut hdr_data = Vec::new();
        hdr_data.resize(hdrs, 0);
        header_reader.read_exact(&mut hdr_data)?;

        if self.garbage_data.len() == 0 {
            // if we don't have any garbage, assume FFD9 EOI

            // kind of broken logic since this assumes a EOF even if there was a 0 byte garbage header
            // in the file, but this is what the file format is.
            self.garbage_data.extend(EOI);
        }

        // beginning here: recovery information (needed for exact JPEG recovery)
        // read further recovery information if any
        loop {
            let mut current_lepton_marker = [0 as u8; 3];
            match header_reader.read_exact(&mut current_lepton_marker) {
                Ok(_) => {}
                Err(e) => {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        break;
                    } else {
                        return Err(anyhow::Error::new(e));
                    }
                }
            }

            if buffer_prefix_matches_marker(current_lepton_marker, LEPTON_HEADER_PAD_MARKER) {
                self.pad_bit = Some(header_reader.read_u8()?);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTARTS_MARKER,
            ) {
                // CRS marker
                self.rst_cnt_set = true;
                let rst_count = header_reader.read_u32::<LittleEndian>()?;

                for _i in 0..rst_count {
                    self.rst_cnt.push(header_reader.read_i32::<LittleEndian>()?);
                }
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_LUMA_SPLIT_MARKER,
            ) {
                // HH markup
                let mut thread_handoffs =
                    ThreadHandoff::deserialize(current_lepton_marker[2], &mut header_reader)?;

                self.thread_handoff.append(&mut thread_handoffs);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER,
            ) {
                // Marker FRS
                // read number of false set RST markers per scan from file
                let rst_err_count = header_reader.read_u32::<LittleEndian>()? as usize;

                let mut rst_err_data = Vec::<u8>::new();
                rst_err_data.resize(rst_err_count, 0);

                header_reader.read_exact(&mut rst_err_data)?;

                self.rst_err.append(&mut rst_err_data);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_GARBAGE_MARKER,
            ) {
                // GRB marker
                // read garbage (data after end of JPG) from file
                let garbage_size = header_reader.read_u32::<LittleEndian>()? as usize;

                let mut garbage_data_array = Vec::<u8>::new();
                garbage_data_array.resize(garbage_size, 0);

                header_reader.read_exact(&mut garbage_data_array)?;
                self.garbage_data = garbage_data_array;
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_EARLY_EOF_MARKER,
            ) {
                self.max_cmp = header_reader.read_i32::<LittleEndian>()?;
                self.max_bpos = header_reader.read_i32::<LittleEndian>()?;
                self.max_sah = u8::try_from(header_reader.read_i32::<LittleEndian>()?)?;
                self.max_dpos[0] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[1] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[2] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[3] = header_reader.read_i32::<LittleEndian>()?;
                self.early_eof_encountered = true;
            } else {
                return err_exit_code(ExitCode::BadLeptonFile, "unknown data found");
            }
        }

        // shouldn't be any more data
        let mut remaining_buf = Vec::new();
        let remaining = header_reader.read_to_end(&mut remaining_buf)?;
        assert!(remaining == 0);

        return Ok(hdr_data);
    }

    pub fn write_lepton_header<W: Write>(&self, writer: &mut W) -> Result<()> {
        let mut lepton_header = Vec::<u8>::new();

        {
            // Most of the Lepton header data that is compressed before storage
            // The data contains recovery information (needed for exact JPEG recovery)
            let mut mrw = Cursor::new(&mut lepton_header);

            self.write_lepton_jpeg_header(&mut mrw)?;
            self.write_lepton_pad_bit(&mut mrw)?;
            self.write_lepton_luma_splits(&mut mrw)?;
            self.write_lepton_jpeg_restarts_if_needed(&mut mrw)?;
            self.write_lepton_jpeg_restart_errors_if_needed(&mut mrw)?;
            self.write_lepton_early_eof_truncation_data_if_needed(&mut mrw)?;
            self.write_lepton_jpeg_garbage_if_needed(&mut mrw, false)?;
        }

        let mut compressed_header = Vec::<u8>::new(); // we collect a zlib compressed version of the header here
        {
            let mut c = Cursor::new(&mut compressed_header);
            let mut encoder = ZlibEncoder::new(&mut c, Compression::default());

            encoder.write_all(&lepton_header[..]).context(here!())?;
            encoder.finish().context(here!())?;
        }

        writer.write_all(&LEPTON_FILE_HEADER)?;
        writer.write_u8(LEPTON_VERSION)?;

        if self.jpeg_header.jpeg_type == JPegType::Progressive {
            writer.write_all(&LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE)?;
        } else {
            writer.write_all(&LEPTON_HEADER_BASELINE_JPEG_TYPE)?;
        }

        writer.write_u8(self.thread_handoff.len() as u8)?;
        writer.write_all(&[0; 3])?;

        // Original lepton format reserves 12 bytes for git revision. We use this space for additional info
        // that our implementation needs - mark that it's MS implementation and a not-compressed header size.
        writer.write_u8('M' as u8)?;
        writer.write_u8('S' as u8)?;
        writer.write_u32::<LittleEndian>(lepton_header.len() as u32)?;
        writer.write_all(&[0; 6])?;

        writer.write_u32::<LittleEndian>(self.jpeg_file_size)?;
        writer.write_u32::<LittleEndian>(compressed_header.len() as u32)?;
        writer.write_all(&compressed_header[..])?;

        writer.write_all(&LEPTON_HEADER_COMPLETION_MARKER)?;

        Ok(())
    }

    fn write_lepton_jpeg_header<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write header to file
        // marker: "HDR" + [size of header]
        mrw.write_all(&LEPTON_HEADER_MARKER)?;

        mrw.write_u32::<LittleEndian>(self.raw_jpeg_header.len() as u32)?;

        // data: data from header
        mrw.write_all(&self.raw_jpeg_header[..])?;

        Ok(())
    }

    fn write_lepton_pad_bit<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // marker: P0D
        mrw.write_all(&LEPTON_HEADER_PAD_MARKER)?;

        // data: this.padBit
        mrw.write_u8(self.pad_bit.unwrap_or(0))?;

        Ok(())
    }

    fn write_lepton_luma_splits<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write luma splits markup HH
        mrw.write_all(&LEPTON_HEADER_LUMA_SPLIT_MARKER)?;

        // data: serialized luma splits
        ThreadHandoff::serialize(&self.thread_handoff, mrw)?;

        Ok(())
    }

    fn write_lepton_jpeg_restarts_if_needed<W: Write>(&self, mrw: &mut W) -> Result<()> {
        if self.rst_cnt.len() > 0 {
            // marker: CRS
            mrw.write_all(&LEPTON_HEADER_JPG_RESTARTS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rst_cnt.len() as u32)?;

            for i in 0..self.rst_cnt.len() {
                mrw.write_u32::<LittleEndian>(self.rst_cnt[i] as u32)?;
            }
        }

        Ok(())
    }

    fn write_lepton_jpeg_restart_errors_if_needed<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write number of false set RST markers per scan (if available) to file
        if self.rst_err.len() > 0 {
            // marker: "FRS" + [number of scans]
            mrw.write_all(&LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rst_err.len() as u32)?;

            mrw.write_all(&self.rst_err[..])?;
        }

        Ok(())
    }

    fn write_lepton_early_eof_truncation_data_if_needed<W: Write>(
        &self,
        mrw: &mut W,
    ) -> Result<()> {
        if self.early_eof_encountered {
            // EEE marker
            mrw.write_all(&LEPTON_HEADER_EARLY_EOF_MARKER)?;

            mrw.write_i32::<LittleEndian>(self.max_cmp)?;
            mrw.write_i32::<LittleEndian>(self.max_bpos)?;
            mrw.write_i32::<LittleEndian>(i32::from(self.max_sah))?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[0])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[1])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[2])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[3])?;
        }

        Ok(())
    }

    fn write_lepton_jpeg_garbage_if_needed<W: Write>(
        &self,
        mrw: &mut W,
        prefix_garbage: bool,
    ) -> Result<()> {
        // write garbage (if any) to file
        if self.garbage_data.len() > 0 {
            // marker: "PGR/GRB" + [size of garbage]
            if prefix_garbage {
                mrw.write_all(&LEPTON_HEADER_PREFIX_GARBAGE_MARKER)?;
            } else {
                mrw.write_all(&LEPTON_HEADER_GARBAGE_MARKER)?;
            }

            mrw.write_u32::<LittleEndian>(self.garbage_data.len() as u32)?;
            mrw.write_all(&self.garbage_data[..])?;
        }

        Ok(())
    }

    fn parse_jpeg_header<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> Result<bool> {
        // the raw header in the lepton file can actually be spread across different sections
        // seperated by the Start-of-Scan marker. We use the mirror to write out whatever
        // data we parse until we hit the SOS

        let mut output = Vec::new();
        let mut output_cursor = Cursor::new(&mut output);

        let mut mirror = Mirror::new(reader, &mut output_cursor);

        if self
            .jpeg_header
            .parse(&mut mirror, enabled_features)
            .context(here!())?
        {
            // append the header if it was not the end of file marker
            self.raw_jpeg_header.append(&mut output);
            return Ok(true);
        } else {
            // if the output was more than 2 bytes then was a trailing header, so keep that around as well,
            // but we don't want the EOI since that goes into the garbage data.
            if output.len() > 2 {
                self.raw_jpeg_header.extend(&output[0..output.len() - 2]);
            }

            return Ok(false);
        }
    }
}

fn split_row_handoffs_to_threads(
    thread_handoffs: &[ThreadHandoff],
    max_threads_to_use: usize,
) -> Vec<ThreadHandoff> {
    let last = thread_handoffs.last().unwrap();

    let framebuffer_byte_size = ThreadHandoff::get_combine_thread_range_segment_size(
        thread_handoffs.first().unwrap(),
        last,
    );

    // determine how many threads we need for compression
    let num_rows = thread_handoffs.len();
    let num_threads =
        get_number_of_threads_for_encoding(num_rows, framebuffer_byte_size, max_threads_to_use);

    info!("Number of threads: {0}", num_threads);

    let mut selected_splits = Vec::with_capacity(num_threads as usize);

    if num_threads == 1 {
        // Single thread execution - no split, run on the whole range
        selected_splits.push(ThreadHandoff::combine_thread_ranges(
            thread_handoffs.first().unwrap(),
            last,
        ));
    } else {
        // gbrovman: simplified split logic
        // Note: rowsPerThread is a floating point value to ensure equal splits
        let rows_per_thread = num_rows as f32 / num_threads as f32;

        assert!(rows_per_thread >= 1f32, "rowsPerThread >= 1");

        let mut split_indices = Vec::new();
        for i in 0..num_threads - 1 {
            split_indices.push((rows_per_thread * (i as f32 + 1f32)) as usize);
        }

        for i in 0..num_threads {
            let beginning_of_range = if i == 0 { 0 } else { split_indices[i - 1] + 1 };
            let end_of_range = if i == num_threads - 1 {
                num_rows - 1
            } else {
                split_indices[i]
            };
            assert!(end_of_range < num_rows, "endOfRange < numRows");
            selected_splits.push(ThreadHandoff::combine_thread_ranges(
                &thread_handoffs[beginning_of_range],
                &thread_handoffs[end_of_range],
            ));
        }
    }

    return selected_splits;
}

fn get_number_of_threads_for_encoding(
    num_rows: usize,
    framebuffer_byte_size: usize,
    max_threads_to_use: usize,
) -> usize {
    let mut num_threads = cmp::min(max_threads_to_use, MAX_THREADS);

    if num_rows / 2 < num_threads {
        num_threads = cmp::max(num_rows / 2, 1);
    }

    if framebuffer_byte_size < SMALL_FILE_BYTES_PER_ENCDOING_THREAD {
        num_threads = 1;
    } else if framebuffer_byte_size < SMALL_FILE_BYTES_PER_ENCDOING_THREAD * 2 {
        num_threads = cmp::min(2, num_threads);
    } else if framebuffer_byte_size < SMALL_FILE_BYTES_PER_ENCDOING_THREAD * 4 {
        num_threads = cmp::min(4, num_threads);
    }

    return num_threads;
}

enum Message {
    Eof,
    WriteBlock(u8, Vec<u8>),
}

struct MessageSender {
    thread_id: u8,
    sender: Sender<Message>,
    buffer: Vec<u8>,
}

const WRITE_BUFFER_SIZE: usize = 65536;

impl Write for MessageSender {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut copy_start = 0;
        while copy_start < buf.len() {
            let amount_to_copy = cmp::min(
                WRITE_BUFFER_SIZE - self.buffer.len(),
                buf.len() - copy_start,
            );
            self.buffer
                .extend_from_slice(&buf[copy_start..copy_start + amount_to_copy]);

            if self.buffer.len() == WRITE_BUFFER_SIZE {
                self.flush()?;
            }

            copy_start += amount_to_copy;
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.len() > 0 {
            let mut new_buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
            swap(&mut new_buffer, &mut self.buffer);

            self.sender
                .send(Message::WriteBlock(self.thread_id, new_buffer))
                .unwrap();
        }
        Ok(())
    }
}

/// used by the worker thread to read data for the given thread from the
/// receiver. The thread_id is used only to assert that we are only
/// getting the data that we are expecting
struct MessageReceiver {
    /// the multiplexed thread stream we are processing
    thread_id: u8,

    /// the receiver part of the channel to get more buffers
    receiver: Receiver<Message>,

    /// what we are reading. When this returns zero, we try to
    /// refill the buffer if we haven't reached the end of the stream
    current_buffer: Cursor<Vec<u8>>,

    /// once we get told we are at the end of the stream, we just
    /// always return 0 bytes
    end_of_file: bool,
}

impl Read for MessageReceiver {
    /// fast path for reads. If we get zero bytes, take the slow path
    #[inline(always)]
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let amount_read = self.current_buffer.read(buf)?;
        if amount_read > 0 {
            return Ok(amount_read);
        }

        self.read_slow(buf)
    }
}

impl MessageReceiver {
    /// slow path for reads, try to get a new buffer or
    /// return zero if at the end of the stream
    #[cold]
    #[inline(never)]
    fn read_slow(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while !self.end_of_file {
            let amount_read = self.current_buffer.read(buf)?;
            if amount_read > 0 {
                return Ok(amount_read);
            }

            match self.receiver.recv() {
                Ok(r) => match r {
                    Message::Eof => {
                        self.end_of_file = true;
                    }
                    Message::WriteBlock(tid, block) => {
                        debug_assert_eq!(
                            tid, self.thread_id,
                            "incoming thread must be equal to processing thread"
                        );
                        self.current_buffer = Cursor::new(block);
                    }
                },
                Err(e) => {
                    return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, e));
                }
            }
        }

        // nothing if we reached the end of file
        return Ok(0);
    }
}

// internal utility we use to collect the header that we read for later
struct Mirror<'a, R, W> {
    read: &'a mut R,
    output: &'a mut W,
    amount_written: usize,
}

impl<'a, R, W> Mirror<'a, R, W> {
    pub fn new(read: &'a mut R, output: &'a mut W) -> Self {
        Mirror {
            read,
            output,
            amount_written: 0,
        }
    }
}

impl<R: Read, W: Write> Read for Mirror<'_, R, W> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.read.read(buf)?;
        self.output.write_all(&buf[..n])?;
        self.amount_written += n;
        Ok(n)
    }
}

// false means we hit the end of file marker
fn prepare_to_decode_next_scan<R: Read>(
    lp: &mut LeptonHeader,
    reader: &mut R,
    enabled_features: &EnabledFeatures,
) -> Result<bool> {
    // parse the header and store it in the raw_jpeg_header
    if !lp
        .parse_jpeg_header(reader, enabled_features)
        .context(here!())?
    {
        return Ok(false);
    }

    lp.max_bpos = cmp::max(lp.max_bpos, lp.jpeg_header.cs_to as i32);

    // FIXME: not sure why only first bit of csSah is examined but 4 bits of it are stored
    lp.max_sah = cmp::max(
        lp.max_sah,
        cmp::max(lp.jpeg_header.cs_sal, lp.jpeg_header.cs_sah),
    );

    for i in 0..lp.jpeg_header.cs_cmpc {
        lp.max_cmp = cmp::max(lp.max_cmp, lp.jpeg_header.cs_cmp[i] as i32);
    }

    return Ok(true);
}

fn set_segment_size_in_row_thread_handoffs(
    thread_handoffs: &mut [ThreadHandoff],
    entropy_data_end_offset_in_file: i32,
) {
    if thread_handoffs.len() != 0 {
        for i in 0..thread_handoffs.len() - 1 {
            thread_handoffs[i].segment_size = thread_handoffs[i + 1].segment_offset_in_file
                - thread_handoffs[i].segment_offset_in_file;
        }

        thread_handoffs[thread_handoffs.len() - 1].segment_size = entropy_data_end_offset_in_file
            - thread_handoffs[thread_handoffs.len() - 1].segment_offset_in_file;
    }
}

// test serializing and deserializing header
#[test]
fn parse_and_write_header() {
    // minimal jpeg that will pass the validity read tests
    let min_jpeg = [
        0xffu8, 0xe0, // APP0
        0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00,
        0x00, 0xff, 0xdb, // DQT
        0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x02, 0x02, 0x02, 0x03, 0x03,
        0x03, 0x03, 0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x06, 0x06, 0x05, 0x06, 0x09,
        0x08, 0x0a, 0x0a, 0x09, 0x08, 0x09, 0x09, 0x0a, 0x0c, 0x0f, 0x0c, 0x0a, 0x0b, 0x0e, 0x0b,
        0x09, 0x09, 0x0d, 0x11, 0x0d, 0x0e, 0x0f, 0x10, 0x10, 0x11, 0x10, 0x0a, 0x0c, 0x12, 0x13,
        0x12, 0x10, 0x13, 0x0f, 0x10, 0x10, 0x10, 0xff, 0xC1, 0x00, 0x0b, 0x08, 0x00,
        0x10, // width
        0x00, 0x10, // height
        0x01, // cmpc
        0x01, // Jid
        0x11, // sfv / sfh
        0x00, 0xff, 0xda, // SOS
        0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0xd2, 0xcf, 0x20, 0xff, 0xd9, // EOI
    ];

    let mut lh = LeptonHeader::new();
    lh.jpeg_file_size = 123;

    lh.parse_jpeg_header(&mut Cursor::new(min_jpeg), &EnabledFeatures::all())
        .unwrap();
    lh.thread_handoff.push(ThreadHandoff {
        luma_y_start: 0,
        luma_y_end: 1,
        segment_offset_in_file: 0,
        segment_size: 1000,
        overhang_byte: 0,
        num_overhang_bits: 1,
        last_dc: [1, 2, 3, 4],
    });

    let mut serialized = Vec::new();
    lh.write_lepton_header(&mut Cursor::new(&mut serialized))
        .unwrap();

    let mut other = LeptonHeader::new();
    let mut other_reader = Cursor::new(&serialized);
    other.read_lepton_header(&mut other_reader).unwrap();
}
