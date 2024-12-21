/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::cmp;
use std::io::{BufRead, Cursor, Seek, Write};
use std::time::Instant;

use byteorder::{LittleEndian, WriteBytesExt};
use default_boxed::DefaultBoxed;
use log::info;

use crate::consts::*;
use crate::enabled_features::EnabledFeatures;
use crate::jpeg::block_based_image::BlockBasedImage;
use crate::jpeg::jpeg_header::JpegHeader;
use crate::jpeg::jpeg_read::read_jpeg_file;
use crate::jpeg::truncate_components::TruncateComponents;
use crate::lepton_error::{err_exit_code, AddContext, ExitCode, Result};
use crate::metrics::{CpuTimeMeasure, Metrics};
use crate::structs::lepton_encoder::lepton_encode_row_range;
use crate::structs::lepton_file_reader::decode_lepton_file;
use crate::structs::lepton_header::LeptonHeader;
use crate::structs::multiplexer::multiplex_write;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::thread_handoff::ThreadHandoff;

/// reads a jpeg and writes it out as a lepton file
pub fn encode_lepton_wrapper<R: BufRead + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    let (lp, image_data) = read_jpeg(reader, enabled_features, |_jh| {})?;

    lp.write_lepton_header(writer, enabled_features).context()?;

    let metrics = run_lepton_encoder_threads(
        &lp.jpeg_header,
        &lp.rinfo.truncate_components,
        writer,
        &lp.thread_handoff[..],
        image_data,
        enabled_features,
    )
    .context()?;

    let final_file_size = writer.stream_position()? + 4;

    writer
        .write_u32::<LittleEndian>(final_file_size as u32)
        .context()?;

    Ok(metrics)
}

/// Encodes JPEG as compressed Lepton format, verifies roundtrip in buffer. Requires everything to be buffered
/// since we need to pass through the data multiple times
pub fn encode_lepton_wrapper_verify(
    input_data: &[u8],
    enabled_features: &EnabledFeatures,
) -> Result<(Vec<u8>, Metrics)> {
    let mut output_data = Vec::with_capacity(input_data.len());

    info!("compressing to Lepton format");

    let mut reader = Cursor::new(&input_data);
    let mut writer = Cursor::new(&mut output_data);

    let mut metrics =
        encode_lepton_wrapper(&mut reader, &mut writer, &enabled_features).context()?;

    // decode and compare to original in order to enure we encoded correctly

    let mut verify_buffer = Vec::with_capacity(input_data.len());
    let mut verifyreader = Cursor::new(&output_data[..]);

    info!("decompressing to verify contents");

    let mut c = enabled_features.clone();

    metrics
        .merge_from(decode_lepton_file(&mut verifyreader, &mut verify_buffer, &mut c).context()?);

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
pub fn read_jpeg<R: BufRead + Seek>(
    reader: &mut R,
    enabled_features: &EnabledFeatures,
    callback: fn(&JpegHeader),
) -> Result<(Box<LeptonHeader>, Vec<BlockBasedImage>)> {
    let mut lp = LeptonHeader::default_boxed();

    get_git_revision(&mut lp);

    let (image_data, partitions, end_scan) = read_jpeg_file(
        reader,
        &mut lp.jpeg_header,
        &mut lp.rinfo,
        enabled_features,
        callback,
    )?;

    let mut thread_handoff = Vec::<ThreadHandoff>::new();

    for i in 0..partitions.len() {
        let (segment_offset, r) = &partitions[i];

        let segment_size = if i == partitions.len() - 1 {
            end_scan - segment_offset
        } else {
            partitions[i + 1].0 - segment_offset
        };

        thread_handoff.push(ThreadHandoff {
            segment_offset_in_file: (*segment_offset).try_into().unwrap(),
            luma_y_start: r.luma_y_start,
            luma_y_end: r.luma_y_end,
            overhang_byte: r.overhang_byte,
            num_overhang_bits: r.num_overhang_bits,
            last_dc: r.last_dc,
            segment_size: segment_size.try_into().unwrap(),
        });

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

    let merged_handoffs =
        split_row_handoffs_to_threads(&thread_handoff[..], enabled_features.max_threads as usize);
    lp.thread_handoff = merged_handoffs;
    lp.jpeg_file_size = reader.stream_position().context()? as u32;

    if lp.jpeg_file_size > enabled_features.max_jpeg_file_size {
        return err_exit_code(
            ExitCode::UnsupportedJpeg,
            "file is too large to encode, increase max_jpeg_file_size",
        );
    }

    Ok((lp, image_data))
}

const fn string_to_int(s: &str) -> u8 {
    let mut result = 0;
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let c = b[i];
        result = result * 10 + c - b'0';
        i += 1;
    }
    result
}

fn get_git_revision(lp: &mut LeptonHeader) {
    let hex_str = git_version::git_version!(args = ["--abbrev=8", "--always", "--dirty=M"]);
    if let Ok(v) = u32::from_str_radix(hex_str, 16) {
        // place the warning if we got a git revision. The --dirty=M suffix means that some files
        // were modified so the version is not a clean git version, so we don't write it.
        lp.git_revision_prefix = v.to_be_bytes();
    }

    const ENCODER_VERSION: u8 = string_to_int(env!("CARGO_PKG_VERSION_MAJOR")) * 100
        + string_to_int(env!("CARGO_PKG_VERSION_MINOR")) * 10
        + string_to_int(env!("CARGO_PKG_VERSION_PATCH"));

    lp.encoder_version = ENCODER_VERSION;
}

/// runs the encoding threads and returns the total amount of CPU time consumed (including worker threads)
fn run_lepton_encoder_threads<W: Write + Seek>(
    jpeg_header: &JpegHeader,
    colldata: &TruncateComponents,
    writer: &mut W,
    thread_handoffs: &[ThreadHandoff],
    image_data: Vec<BlockBasedImage>,
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
    let quantization_tables = QuantizationTables::construct_quantization_tables(jpeg_header)?;

    let colldata = colldata.clone();
    let thread_handoffs = thread_handoffs.to_vec();
    let features = features.clone();

    let mut thread_results = multiplex_write(
        writer,
        thread_handoffs.len(),
        move |thread_writer, thread_id| {
            let cpu_time = CpuTimeMeasure::new();

            let mut range_metrics = lepton_encode_row_range(
                &quantization_tables,
                &image_data,
                thread_writer,
                thread_id as i32,
                &colldata,
                thread_handoffs[thread_id].luma_y_start,
                thread_handoffs[thread_id].luma_y_end,
                thread_id == thread_handoffs.len() - 1,
                true,
                &features,
            )
            .context()?;

            range_metrics.record_cpu_worker_time(cpu_time.elapsed());

            Ok(range_metrics)
        },
    )?;

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

    let mut selected_splits = Vec::with_capacity(num_threads);

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

#[test]
fn test_get_git_revision() {
    let mut lh = LeptonHeader::default_boxed();
    get_git_revision(&mut lh);

    println!("{:x?}", lh.git_revision_prefix);
}

#[test]
fn test_slrcity() {
    test_file("slrcity")
}

#[cfg(test)]
fn test_file(filename: &str) {
    use crate::structs::lepton_file_reader::read_file;

    let original = read_file(filename, ".jpg");

    let mut enabled_features = EnabledFeatures::compat_lepton_vector_write();
    enabled_features.max_threads = 2;

    let mut output = Vec::new();

    let _ = encode_lepton_wrapper(
        &mut Cursor::new(&original),
        &mut Cursor::new(&mut output),
        &enabled_features,
    )
    .unwrap();

    println!(
        "Original size: {0}, compressed size: {1}",
        original.len(),
        output.len()
    );

    let mut recreate = Vec::new();

    decode_lepton_file(&mut Cursor::new(&output), &mut recreate, &enabled_features).unwrap();

    assert_eq!(original.len(), recreate.len());
    assert!(original == recreate);
}
