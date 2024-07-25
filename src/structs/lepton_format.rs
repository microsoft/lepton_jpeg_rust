/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use log::{info, warn};
use std::cmp;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::consts::*;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::jpeg_code;
use crate::lepton_error::ExitCode;
use crate::metrics::{CpuTimeMeasure, Metrics};
use crate::structs::bit_writer::BitWriter;
use crate::structs::block_based_image::BlockBasedImage;
use crate::structs::jpeg_header::JPegHeader;
use crate::structs::jpeg_write::jpeg_write_row_range;
use crate::structs::lepton_decoder::lepton_decode_row_range;
use crate::structs::lepton_encoder::lepton_encode_row_range;
use crate::structs::multiplexer::{multiplex_read, multiplex_write};
use crate::structs::probability_tables_set::ProbabilityTablesSet;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::thread_handoff::ThreadHandoff;
use crate::structs::truncate_components::TruncateComponents;

use super::jpeg_read::{read_progressive_scan, read_scan};
use super::jpeg_write::jpeg_write_entire_scan;
use super::lepton_header::LeptonHeader;

/// reads a lepton file and writes it out as a jpeg
pub fn decode_lepton_wrapper<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    num_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    // figure out how long the input is
    let orig_pos = reader.stream_position()?;
    let size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(orig_pos))?;

    // last four bytes specify the file size
    let mut reader_minus_trailer = reader.take(size - 4);

    let mut lh = LeptonHeader::new();

    let mut features_mut = enabled_features.clone();

    lh.read_lepton_header(&mut reader_minus_trailer, &mut features_mut)
        .context(here!())?;

    let metrics = recode_jpeg(
        &mut lh,
        writer,
        &mut reader_minus_trailer,
        num_threads,
        &features_mut,
    )
    .context(here!())?;

    let expected_size = reader.read_u32::<LittleEndian>()?;
    if expected_size != size as u32 {
        return err_exit_code(
            ExitCode::VerificationLengthMismatch,
            format!(
                "ERROR mismatch expected_size = {0}, actual_size = {1}",
                expected_size, size
            )
            .as_str(),
        );
    }

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

    lp.write_lepton_header(writer, enabled_features)
        .context(here!())?;

    let metrics = run_lepton_encoder_threads(
        &lp.jpeg_header,
        &lp.truncate_components,
        writer,
        &lp.thread_handoff[..],
        &image_data[..],
        enabled_features,
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

    let mut c = enabled_features.clone();

    metrics.merge_from(
        decode_lepton_wrapper(&mut verifyreader, &mut verify_buffer, max_threads, &mut c)
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

fn run_lepton_decoder_threads<R: Read, P: Send>(
    lh: &LeptonHeader,
    reader: &mut R,
    _max_threads_to_use: usize,
    features: &EnabledFeatures,
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
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56] {
            if qtables.get_quantization_table()[i] == 0 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "Quantization table contains zero",
                );
            }
        }
        qt.push(qtables);
    }

    let pts_ref = &pts;
    let q_ref = &qt[..];

    let mut thread_results = multiplex_read(
        reader,
        lh.thread_handoff.len(),
        |thread_id, reader| -> Result<(Metrics, P)> {
            let cpu_time = CpuTimeMeasure::new();

            let mut image_data = Vec::new();
            for i in 0..lh.jpeg_header.cmpc {
                image_data.push(BlockBasedImage::new(
                    &lh.jpeg_header,
                    i,
                    lh.thread_handoff[thread_id].luma_y_start,
                    if thread_id == lh.thread_handoff.len() - 1 {
                        // if this is the last thread, then the image should extend all the way to the bottom
                        lh.jpeg_header.cmp_info[0].bcv
                    } else {
                        lh.thread_handoff[thread_id].luma_y_end
                    },
                ));
            }

            let mut metrics = Metrics::default();

            metrics.merge_from(
                lepton_decode_row_range(
                    pts_ref,
                    q_ref,
                    &lh.truncate_components,
                    &mut image_data,
                    reader,
                    lh.thread_handoff[thread_id].luma_y_start,
                    lh.thread_handoff[thread_id].luma_y_end,
                    thread_id == lh.thread_handoff.len() - 1,
                    true,
                    features,
                )
                .context(here!())?,
            );

            let process_result = process(&lh.thread_handoff[thread_id], image_data, lh)?;

            metrics.record_cpu_worker_time(cpu_time.elapsed());

            Ok((metrics, process_result))
        },
    )?;

    let mut metrics = Metrics::default();

    let mut result = Vec::new();
    for (m, r) in thread_results.drain(..) {
        metrics.merge_from(m);
        result.push(r);
    }

    info!(
        "worker threads {0}ms of CPU time in {1}ms of wall time",
        metrics.get_cpu_time_worker_time().as_millis(),
        wall_time.elapsed().as_millis()
    );

    Ok((metrics, result))
}

/// runs the encoding threads and returns the total amount of CPU time consumed (including worker threads)
fn run_lepton_encoder_threads<W: Write + Seek>(
    jpeg_header: &JPegHeader,
    colldata: &TruncateComponents,
    writer: &mut W,
    thread_handoffs: &[ThreadHandoff],
    image_data: &[BlockBasedImage],
    features: &EnabledFeatures,
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
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56] {
            if qtables.get_quantization_table()[i] == 0 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "Quantization table contains zero",
                );
            }
        }
        quantization_tables.push(qtables);
    }

    let pts_ref = &pts;
    let q_ref = &quantization_tables[..];

    let mut thread_results =
        multiplex_write(writer, thread_handoffs.len(), |thread_writer, thread_id| {
            let cpu_time = CpuTimeMeasure::new();

            let mut range_metrics = lepton_encode_row_range(
                pts_ref,
                q_ref,
                image_data,
                thread_writer,
                thread_id as i32,
                colldata,
                thread_handoffs[thread_id].luma_y_start,
                thread_handoffs[thread_id].luma_y_end,
                thread_id == thread_handoffs.len() - 1,
                true,
                features,
            )
            .context(here!())?;

            range_metrics.record_cpu_worker_time(cpu_time.elapsed());

            Ok(range_metrics)
        })?;

    let mut merged_metrics = Metrics::default();

    for result in thread_results.drain(..) {
        merged_metrics.merge_from(result);
    }

    info!(
        "worker threads {0}ms of CPU time in {1}ms of wall time",
        merged_metrics.get_cpu_time_worker_time().as_millis(),
        wall_time.elapsed().as_millis()
    );

    Ok(merged_metrics)
}

fn recode_jpeg<R: Read, W: Write>(
    lh: &mut LeptonHeader,
    writer: &mut W,
    reader: &mut R,
    num_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics, anyhow::Error> {
    writer.write_all(&SOI)?;

    // write the raw header as far as we've decoded it
    writer
        .write_all(&lh.raw_jpeg_header[0..lh.raw_jpeg_header_read_index])
        .context(here!())?;

    let metrics = if lh.jpeg_header.jpeg_type == JPegType::Progressive {
        recode_progressive_jpeg(lh, reader, writer, num_threads, enabled_features)
            .context(here!())?
    } else {
        recode_baseline_jpeg(
            lh,
            reader,
            writer,
            lh.plain_text_size as u64
                - lh.garbage_data.len() as u64
                - lh.raw_jpeg_header_read_index as u64
                - SOI.len() as u64,
            num_threads,
            enabled_features,
        )
        .context(here!())?
    };

    // Blit any trailing header data.
    // Run this logic even if early_eof_encountered to be compatible with C++ version.
    writer
        .write_all(&lh.raw_jpeg_header[lh.raw_jpeg_header_read_index..])
        .context(here!())?;

    writer.write_all(&lh.garbage_data).context(here!())?;
    Ok(metrics)
}

/// decodes the entire image and merges the results into a single set of BlockBaseImage per component
pub fn decode_as_single_image<R: Read>(
    lh: &mut LeptonHeader,
    reader: &mut R,
    num_threads: usize,
    features: &EnabledFeatures,
) -> Result<(Vec<BlockBasedImage>, Metrics)> {
    // run the threads first, since we need everything before we can start decoding
    let (metrics, mut results) = run_lepton_decoder_threads(
        lh,
        reader,
        num_threads,
        features,
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

/// progressive decoder, requires that the entire lepton file is processed first
fn recode_progressive_jpeg<R: Read, W: Write>(
    lh: &mut LeptonHeader,
    reader: &mut R,
    writer: &mut W,
    num_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    // run the threads first, since we need everything before we can start decoding
    let (merged, metrics) =
        decode_as_single_image(lh, reader, num_threads, enabled_features).context(here!())?;

    loop {
        // code another scan
        jpeg_write_entire_scan(writer, &merged[..], lh).context(here!())?;

        // read the next headers (DHT, etc) while mirroring it back to the writer
        let old_pos = lh.raw_jpeg_header_read_index;
        let result = lh
            .advance_next_header_segment(enabled_features)
            .context(here!())?;

        writer
            .write_all(&lh.raw_jpeg_header[old_pos..lh.raw_jpeg_header_read_index])
            .context(here!())?;

        if !result {
            break;
        }

        // advance to next scan
        lh.scnc += 1;
    }

    Ok(metrics)
}

// baseline decoder can run the jpeg encoder inside the worker thread vs progressive encoding which needs to get the entire set of coefficients first
// since it runs throught it multiple times.
fn recode_baseline_jpeg<R: Read, W: Write>(
    lh: &mut LeptonHeader,
    reader: &mut R,
    writer: &mut W,
    size_limit: u64,
    num_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    // step 2: recode image data
    let (metrics, results) = run_lepton_decoder_threads(
        lh,
        reader,
        num_threads,
        enabled_features,
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
            info!(
                "ystart = {0}, segment_size = {1}, amount = {2}, offset = {3}, ob = {4}, nb = {5}",
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

    let mut amount_written: u64 = 0;

    // write all the buffers that we collected
    for r in results {
        amount_written += r.len() as u64;
        writer.write_all(&r[..]).context(here!())?;
    }

    // Injection of restart codes for RST errors supports JPEGs with trailing RSTs.
    // Run this logic even if early_eof_encountered to be compatible with C++ version.
    //
    // This logic is no longer needed for Rust generated Lepton files, since we just use the garbage
    // data to store any extra RST codes or whatever else might be at the end of the file.
    if lh.rst_err.len() > 0 {
        let mut markers = Vec::new();

        let cumulative_reset_markers = if lh.jpeg_header.rsti != 0 {
            ((lh.jpeg_header.mcuh * lh.jpeg_header.mcuv) - 1) / lh.jpeg_header.rsti
        } else {
            0
        } as u8;
        for i in 0..lh.rst_err[0] as u8 {
            let rst = (jpeg_code::RST0 + ((cumulative_reset_markers + i) & 7)) as u8;
            markers.push(0xFF);
            markers.push(rst);
        }

        // the C++ version will strangely sometimes ask for extra rst codes even if we are at the end of the file and shouldn't
        // be emitting anything more, so if we are over or at the size limit then don't emit the RST code
        if amount_written < size_limit {
            writer.write_all(
                &markers[0..cmp::min(markers.len(), (size_limit - amount_written) as usize)],
            )?;
        }
    }

    Ok(metrics)
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

    let mut enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let mut lh = LeptonHeader::new();
    lh.jpeg_file_size = 123;

    lh.parse_jpeg_header(&mut Cursor::new(min_jpeg), &enabled_features)
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
    lh.write_lepton_header(&mut Cursor::new(&mut serialized), &enabled_features)
        .unwrap();

    let mut other = LeptonHeader::new();
    let mut other_reader = Cursor::new(&serialized);
    other
        .read_lepton_header(&mut other_reader, &mut enabled_features)
        .unwrap();
}
