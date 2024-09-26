/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use log::warn;
use std::cmp::min;
use std::io::{BufRead, Cursor, Write};
use std::mem;

use anyhow::{Context, Result};

use crate::consts::*;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::jpeg_code;
use crate::lepton_error::ExitCode;
use crate::metrics::{CpuTimeMeasure, Metrics};
use crate::structs::block_based_image::BlockBasedImage;
use crate::structs::jpeg_header::JPegEncodingInfo;
use crate::structs::jpeg_write::{jpeg_write_baseline_row_range, jpeg_write_entire_scan};
use crate::structs::lepton_decoder::lepton_decode_row_range;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::thread_handoff::ThreadHandoff;

use super::lepton_header::{LeptonHeader, FIXED_HEADER_SIZE};
use super::multiplexer::{MultiplexReader, MultiplexReaderState};
use super::partial_buffer::PartialBuffer;

/// reads a lepton file and writes it out as a jpeg
pub fn decode_lepton_wrapper<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    let mut decoder = LeptonFileRead::new(enabled_features.clone());

    let mut done = false;
    while !done {
        let buffer = reader.fill_buf().context(here!())?;

        done = decoder
            .process_buffer(buffer, buffer.len() == 0, writer, usize::MAX)
            .context(here!())?;

        let amt = buffer.len();
        reader.consume(amt);
    }

    return Ok(decoder.read_metrics());
}

enum DecoderState {
    FixedHeader(),
    CompressedHeader(usize),
    CMP(),
    ScanProgressive(MultiplexReaderState<(Metrics, Vec<BlockBasedImage>)>),
    ScanBaseline(MultiplexReaderState<(Metrics, Vec<u8>)>),
    AppendTrailer(Vec<Vec<u8>>),
    ReturnResults(usize, Vec<Vec<u8>>),
    EOI,
}

/// This is the state machine for the decoder for reading lepton files. The
/// data is pushed into the state machine and processed in chuncks. Once
/// the calculations are done the data is retrieved from the output buffers.
pub struct LeptonFileRead {
    state: DecoderState,
    lh: LeptonHeader,
    enabled_features: EnabledFeatures,
    extra_buffer: Vec<u8>,
    metrics: Metrics,
    total_read_size: u64,
}

impl LeptonFileRead {
    pub fn new(features: EnabledFeatures) -> Self {
        LeptonFileRead {
            state: DecoderState::FixedHeader(),
            lh: LeptonHeader::new(),
            enabled_features: features,
            extra_buffer: Vec::new(),
            metrics: Metrics::default(),
            total_read_size: 0,
        }
    }

    fn read_metrics(&mut self) -> Metrics {
        mem::take(&mut self.metrics)
    }

    pub fn process_buffer(
        &mut self,
        in_buffer: &[u8],
        input_complete: bool,
        output: &mut impl Write,
        mut output_max_size: usize,
    ) -> Result<bool> {
        self.total_read_size += in_buffer.len() as u64;

        let mut in_buffer = PartialBuffer::new(in_buffer, &mut self.extra_buffer);
        while in_buffer.continue_processing() {
            match &mut self.state {
                DecoderState::FixedHeader() => {
                    if let Some(v) = in_buffer.take(FIXED_HEADER_SIZE, 0) {
                        let compressed_header_size = self
                            .lh
                            .read_lepton_fixed_header(
                                &v.try_into().unwrap(),
                                &mut self.enabled_features,
                            )
                            .context(here!())?;
                        self.state = DecoderState::CompressedHeader(compressed_header_size);
                    }
                }
                DecoderState::CompressedHeader(compressed_length) => {
                    if let Some(v) = in_buffer.take(*compressed_length, 0) {
                        self.lh
                            .read_compressed_lepton_header(
                                &mut Cursor::new(v),
                                &mut self.enabled_features,
                                *compressed_length,
                            )
                            .context(here!())?;

                        self.state = DecoderState::CMP();
                    }
                }
                DecoderState::CMP() => {
                    if let Some(v) = in_buffer.take(3, 0) {
                        self.state = Self::process_cmp(v, &self.lh, &self.enabled_features)?;
                    }
                }

                DecoderState::ScanProgressive(state) => {
                    state.process_buffer(&mut in_buffer)?;

                    if input_complete {
                        Self::verify_eof_file_size(self.total_read_size, &mut in_buffer)?;

                        // complete the operation and merge the metrics
                        let results = Self::merge_metrics(
                            &mut self.metrics,
                            state.complete().context(here!())?,
                        );

                        self.state = Self::process_progressive(
                            &mut self.lh,
                            &self.enabled_features,
                            results,
                        )?;
                    }
                }
                DecoderState::ScanBaseline(state) => {
                    state.process_buffer(&mut in_buffer)?;

                    if input_complete {
                        Self::verify_eof_file_size(self.total_read_size, &mut in_buffer)?;

                        // complete the operation and merge the metrics
                        let results = Self::merge_metrics(
                            &mut self.metrics,
                            state.complete().context(here!())?,
                        );

                        self.state = Self::process_baseline(&self.lh, results)?;
                    }
                }
                DecoderState::AppendTrailer(results) => {
                    // Blit any trailing header data.
                    // Run this logic even if early_eof_encountered to be compatible with C++ version.
                    results.push(
                        self.lh.raw_jpeg_header[self.lh.raw_jpeg_header_read_index..].to_vec(),
                    );
                    results.push(mem::take(&mut self.lh.garbage_data));

                    self.state = DecoderState::ReturnResults(0, mem::take(results));
                }
                DecoderState::ReturnResults(offset, leftover) => {
                    while output_max_size > 0 {
                        let bytes_to_write = min(output_max_size, leftover[0].len() - *offset);
                        output.write_all(&leftover[0][*offset..*offset + bytes_to_write])?;
                        *offset += bytes_to_write;
                        output_max_size -= bytes_to_write;

                        if *offset == leftover[0].len() {
                            leftover.remove(0);
                            *offset = 0;

                            if leftover.len() == 0 {
                                self.state = DecoderState::EOI;
                                return Ok(true);
                            }
                        }
                    }
                    return Ok(false);
                }
                DecoderState::EOI => {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn process_baseline(lh: &LeptonHeader, mut results: Vec<Vec<u8>>) -> Result<DecoderState> {
        let mut header = Vec::new();
        header.write_all(&SOI)?;
        header
            .write_all(&lh.raw_jpeg_header[0..lh.raw_jpeg_header_read_index])
            .context(here!())?;

        results.insert(0, header);

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

            let expected_total_length = results.iter().map(|x| x.len()).sum::<usize>()
                + lh.garbage_data.len()
                + (lh.raw_jpeg_header.len() - lh.raw_jpeg_header_read_index);

            if expected_total_length < lh.plain_text_size as usize {
                // figure out how much extra space we have, since C++ files can have
                // more restart markers than there is space to fit them
                let space_for_markers = min(
                    markers.len(),
                    lh.plain_text_size as usize - expected_total_length,
                );

                markers.resize(space_for_markers, 0);

                results.push(markers);
            }
        }

        Ok(DecoderState::AppendTrailer(results))
    }

    fn process_progressive(
        lh: &mut LeptonHeader,
        enabled_features: &EnabledFeatures,
        mut results: Vec<Vec<BlockBasedImage>>,
    ) -> Result<DecoderState> {
        let num_components = results[0].len();
        let mut merged = Vec::new();
        for i in 0..num_components {
            merged.push(BlockBasedImage::merge(&mut results, i));
        }

        let mut result = Vec::new();
        result.write_all(&SOI)?;
        result
            .write_all(&lh.raw_jpeg_header[0..lh.raw_jpeg_header_read_index])
            .context(here!())?;

        recode_progressive_jpeg(lh, merged, &mut result, &enabled_features).context(here!())?;

        let mut results = Vec::new();
        results.push(result);

        Ok(DecoderState::AppendTrailer(results))
    }

    fn process_cmp(
        v: Vec<u8>,
        lh: &LeptonHeader,
        enabled_features: &EnabledFeatures,
    ) -> Result<DecoderState> {
        if v[..] != LEPTON_HEADER_COMPLETION_MARKER {
            return err_exit_code(ExitCode::BadLeptonFile, "CMP marker not found");
        }
        Ok(if lh.jpeg_header.jpeg_type == JPegType::Progressive {
            let mux = run_lepton_decoder_threads(
                lh,
                enabled_features,
                4, /* reserve 4 bytes for the very end */
                |_thread_handoff, image_data, _lh| {
                    // just return the image data directly to be merged together
                    return Ok(image_data);
                },
            )
            .context(here!())?;

            DecoderState::ScanProgressive(mux)
        } else {
            let mux = run_lepton_decoder_threads(
                &lh,
                &enabled_features,
                4, /*reserve 4 bytes for the end */
                |thread_handoff, image_data, jenc| {
                    let mut result_buffer =
                        Vec::with_capacity(thread_handoff.segment_size as usize);

                    jpeg_write_baseline_row_range(
                        &mut result_buffer,
                        &image_data,
                        &thread_handoff,
                        jenc,
                    )
                    .context(here!())?;

                    #[cfg(feature = "detailed_tracing")]
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
            DecoderState::ScanBaseline(mux)
        })
    }

    fn merge_metrics<T>(metrics: &mut Metrics, r: Vec<(Metrics, Vec<T>)>) -> Vec<Vec<T>> {
        let mut results = Vec::new();
        for (metric, vec) in r {
            metrics.merge_from(metric);
            results.push(vec);
        }
        results
    }

    fn verify_eof_file_size(total_read_size: u64, in_buffer: &mut PartialBuffer<'_>) -> Result<()> {
        if let Some(bytes) = in_buffer.take_n::<4>(0) {
            let size = u32::from_le_bytes(bytes);
            if u64::from(size) != total_read_size {
                return err_exit_code(
                    ExitCode::VerificationLengthMismatch,
                    format!(
                        "ERROR mismatch input_len = {0}, decoded_len = {1}",
                        size, total_read_size
                    )
                    .as_str(),
                );
            }
            Ok(())
        } else {
            err_exit_code(
                ExitCode::VerificationLengthMismatch,
                "Missing EOF file size",
            )
        }
    }
}

pub fn run_lepton_decoder_threads<P: Send + 'static>(
    lh: &LeptonHeader,
    features: &EnabledFeatures,
    reserve: usize,
    process: fn(
        thread_handoff: &ThreadHandoff,
        image_data: Vec<BlockBasedImage>,
        jenc: &JPegEncodingInfo,
    ) -> Result<P>,
) -> Result<MultiplexReaderState<(Metrics, P)>> {
    let qt = lh.jpeg_header.construct_quantization_table()?;

    let features = features.clone();

    let jenc = JPegEncodingInfo::new(lh);

    let thread_handoff = lh.thread_handoff.clone();

    let multiplex_reader_state = MultiplexReaderState::new(
        thread_handoff.len(),
        reserve,
        move |thread_id, reader| -> Result<(Metrics, P)> {
            run_lepton_decoder_processor(
                &jenc,
                &thread_handoff[thread_id],
                thread_id == thread_handoff.len() - 1,
                &qt,
                reader,
                &features,
                process,
            )
        },
    );

    Ok(multiplex_reader_state)
}

/// the logic of a decoder thread. Takes a range of rows
fn run_lepton_decoder_processor<P>(
    jenc: &JPegEncodingInfo,
    thread_handoff: &ThreadHandoff,
    is_last_thread: bool,
    qt: &[QuantizationTables],
    reader: &mut MultiplexReader,
    features: &EnabledFeatures,
    process: fn(&ThreadHandoff, Vec<BlockBasedImage>, &JPegEncodingInfo) -> Result<P>,
) -> Result<(Metrics, P)> {
    let cpu_time = CpuTimeMeasure::new();

    let mut image_data = Vec::new();
    for i in 0..jenc.jpeg_header.cmpc {
        image_data.push(BlockBasedImage::new(
            &jenc.jpeg_header,
            i,
            thread_handoff.luma_y_start,
            if is_last_thread {
                // if this is the last thread, then the image should extend all the way to the bottom
                jenc.jpeg_header.cmp_info[0].bcv
            } else {
                thread_handoff.luma_y_end
            },
        ));
    }

    let mut metrics = Metrics::default();

    metrics.merge_from(
        lepton_decode_row_range(
            &qt,
            &jenc.truncate_components,
            &mut image_data,
            reader,
            thread_handoff.luma_y_start,
            thread_handoff.luma_y_end,
            is_last_thread,
            true,
            &features,
        )
        .context(here!())?,
    );

    let process_result = process(thread_handoff, image_data, &jenc)?;

    metrics.record_cpu_worker_time(cpu_time.elapsed());

    Ok((metrics, process_result))
}

/// progressive decoder, requires that the entire lepton file is processed first
fn recode_progressive_jpeg<W: Write>(
    lh: &mut LeptonHeader,
    merged: Vec<BlockBasedImage>,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
) -> Result<()> {
    loop {
        // code another scan
        jpeg_write_entire_scan(writer, &merged[..], &JPegEncodingInfo::new(lh)).context(here!())?;

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

    Ok(())
}

// test serializing and deserializing header
#[test]
fn parse_and_write_header() {
    use std::io::Read;

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

    let enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let mut lh = LeptonHeader::new();
    lh.jpeg_file_size = 123;
    lh.uncompressed_lepton_header_size = 140;

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

    let mut fixed_buffer = [0; FIXED_HEADER_SIZE];
    other_reader.read_exact(&mut fixed_buffer).unwrap();

    let mut other_enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let compressed_header_size = other
        .read_lepton_fixed_header(&fixed_buffer, &mut other_enabled_features)
        .unwrap();
    other
        .read_compressed_lepton_header(
            &mut other_reader,
            &mut other_enabled_features,
            compressed_header_size,
        )
        .unwrap();

    assert_eq!(
        lh.uncompressed_lepton_header_size,
        other.uncompressed_lepton_header_size
    );
}

#[cfg(test)]
fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    use std::io::Read;

    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("images")
        .join(filename.to_owned() + ext);
    println!("reading {0}", filename.to_str().unwrap());
    let mut f = std::fs::File::open(filename).unwrap();

    let mut content = Vec::new();
    f.read_to_end(&mut content).unwrap();

    content
}

#[test]
fn test_simple_parse_progressive() {
    test_file("androidprogressive")
}

#[test]
fn test_simple_parse_baseline() {
    test_file("android")
}

#[test]
fn test_simple_parse_trailing() {
    test_file("androidtrail")
}

#[test]
fn test_zero_dqt() {
    test_file("zeros_in_dqt_tables")
}

#[cfg(test)]
fn test_file(filename: &str) {
    let file = read_file(filename, ".lep");
    let original = read_file(filename, ".jpg");

    let enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let mut output = Vec::new();

    decode_lepton_wrapper(&mut Cursor::new(&file), &mut output, &enabled_features).unwrap();

    assert_eq!(output.len(), original.len());
    assert!(output == original);
}
