/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::io::{BufRead, Cursor, Write};
use std::mem;
use std::sync::mpsc::Sender;

use default_boxed::DefaultBoxed;
#[cfg(feature = "detailed_tracing")]
use log::info;
use log::warn;

use crate::enabled_features::EnabledFeatures;
use crate::jpeg::block_based_image::BlockBasedImage;
use crate::jpeg::jpeg_code;
use crate::jpeg::jpeg_header::{JpegHeader, ReconstructionInfo, RestartSegmentCodingInfo};
use crate::jpeg::jpeg_write::{jpeg_write_baseline_row_range, jpeg_write_entire_scan};
use crate::lepton_error::{AddContext, ExitCode, Result, err_exit_code};
use crate::metrics::{CpuTimeMeasure, Metrics};
use crate::structs::lepton_decoder::lepton_decode_row_range;
use crate::structs::lepton_header::{FIXED_HEADER_SIZE, LeptonHeader};
use crate::structs::multiplexer::{MultiplexReader, MultiplexReaderState, multiplex_read};
use crate::structs::partial_buffer::PartialBuffer;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::simple_threadpool::ThreadPoolHolder;
use crate::structs::thread_handoff::ThreadHandoff;
use crate::{LeptonThreadPool, consts::*};

/// Reads an entire lepton file and writes it out as a JPEG
///
/// # Parameters
///
/// - `reader`: A buffered reader from which the Lepton-encoded data is read.
/// - `writer`: A writer to which the decoded JPEG image is written.
/// - `enabled_features`: A set of toggles for enabling/disabling decoding features/restrictions.
/// - `thread_pool`: A reference to a thread pool used for parallel processing. Must be a static reference and
/// can point to `DEFAULT_THREAD_POOL`.
pub fn decode_lepton<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
    thread_pool: &dyn LeptonThreadPool,
) -> Result<Metrics> {
    let mut decoder =
        LeptonFileReader::new(enabled_features.clone(), ThreadPoolHolder::Dyn(thread_pool));

    let mut done = false;
    while !done {
        let buffer = reader.fill_buf().context()?;

        done = decoder
            .process_buffer(buffer, buffer.len() == 0, writer)
            .context()?;

        let amt = buffer.len();
        reader.consume(amt);
    }

    return Ok(decoder.take_metrics());
}

/// this is a debug function only called by the utility EXE code
/// used to dump the contents of the file
#[allow(dead_code)]
pub fn decode_lepton_file_image<R: BufRead>(
    reader: &mut R,
    enabled_features: &EnabledFeatures,
    thread_pool: &'static dyn LeptonThreadPool,
) -> Result<(Box<LeptonHeader>, Vec<BlockBasedImage>)> {
    let mut lh = LeptonHeader::default_boxed();
    let mut enabled_features = enabled_features.clone();

    let mut fixed_header_buffer = [0; FIXED_HEADER_SIZE];
    reader.read_exact(&mut fixed_header_buffer).context()?;

    let compressed_header_size = lh
        .read_lepton_fixed_header(&fixed_header_buffer, &mut enabled_features)
        .context()?;

    lh.read_compressed_lepton_header(reader, &mut enabled_features, compressed_header_size)
        .context()?;

    let mut buf = [0; 3];
    reader.read_exact(&mut buf).context()?;

    if buf != LEPTON_HEADER_COMPLETION_MARKER {
        return err_exit_code(ExitCode::BadLeptonFile, "CMP marker not found");
    }

    let mut state = LeptonFileReader::run_lepton_decoder_threads(
        &lh,
        &enabled_features,
        4,
        thread_pool,
        |_thread_handoff, image_data, _, _| {
            // just return the image data directly to be merged together
            return Ok(image_data);
        },
    )
    .context()?;

    let mut results = Vec::new();

    // process the rest of the file (except for the 4 byte EOF marker)
    let mut extra_buffer = Vec::new();
    loop {
        let b = reader.fill_buf().context()?;
        let b_len = b.len();
        if b_len == 0 {
            break;
        }
        state.process_buffer(&mut PartialBuffer::new(b, &mut extra_buffer))?;
        reader.consume(b_len);

        if let Some((_m, r)) = state.retrieve_result(false)? {
            results.push(r);
        }
    }

    while let Some((_m, r)) = state.retrieve_result(true)? {
        results.push(r);
    }

    // merge the corresponding components so that we get a single set of coefficient maps (since each thread did a piece of the work)
    let num_components = results[0].len();

    let mut block_image = Vec::new();
    for i in 0..num_components {
        block_image.push(BlockBasedImage::merge(&mut results, i).context()?);
    }

    Ok((lh, block_image))
}

enum DecoderState {
    FixedHeader(),
    CompressedHeader(usize),
    CMP(),
    ScanProgressive(MultiplexReaderState<(Metrics, Vec<BlockBasedImage>)>),
    ScanBaseline(MultiplexReaderState<(Metrics, Vec<u8>)>),
    EOI,
}

/// A writer that limits the amount of data written to a specified amount, silently truncating any excess data.
///
/// This is used to ensure that we do not write more data than the expected JPEG file size during decoding.
struct LimitedOutputWriter<'a, W: Write> {
    inner: &'a mut W,
    amount_left: &'a mut u64,
}

impl<W: Write> Write for LimitedOutputWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let to_write = std::cmp::min(buf.len() as u64, *self.amount_left) as usize;
        let written = self.inner.write(&buf[0..to_write])?;
        *self.amount_left -= written as u64;

        // always say we wrote everything, the goal here is to silently truncate
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// This is the state machine for the decoder for reading lepton files. The
/// data is pushed into the state machine and processed in chuncks. Once
/// the calculations are done the data is retrieved from the output buffers.
pub struct LeptonFileReader<'a> {
    state: DecoderState,
    lh: Box<LeptonHeader>,
    enabled_features: EnabledFeatures,
    extra_buffer: Vec<u8>,
    metrics: Metrics,
    total_read_size: u64,
    jpeg_file_size_left: u64,
    input_complete: bool,
    thread_pool: ThreadPoolHolder<'a>,
}

impl<'a> LeptonFileReader<'a> {
    /// Creates a new LeptonFileReader.
    pub fn new(features: EnabledFeatures, thread_pool: ThreadPoolHolder<'a>) -> Self {
        LeptonFileReader {
            state: DecoderState::FixedHeader(),
            lh: LeptonHeader::default_boxed(),
            enabled_features: features,
            extra_buffer: Vec::new(),
            metrics: Metrics::default(),
            total_read_size: 0,
            input_complete: false,
            jpeg_file_size_left: 0,
            thread_pool,
        }
    }

    /// Processes a buffer of data of the file, which can be a slice of 0 or more characters.
    /// If the input is complete, then input_complete should be set to true.
    ///
    /// Any available output is written to the output buffer, which can be zero if the
    /// input is not yet complete. Once the input has been marked as complete, then the
    /// call will always return some data until the end of the file is reached, at which
    /// it will return true.
    ///
    /// # Arguments
    /// * `input` - The input buffer to process.
    /// * `input_complete` - True if the input is complete and no more data will be provided.
    /// * `writer` - The writer to write the output to.
    /// * `output_buffer_size` - The maximum amount of output to write to the writer before returning.
    ///
    /// # Returns
    ///
    /// Returns true if the end of the file has been reached, otherwise false. If an error occurs
    /// then an error code is returned and no further calls should be made.
    pub fn process_buffer(
        &mut self,
        in_buffer: &[u8],
        input_complete: bool,
        output: &mut impl Write,
    ) -> Result<bool> {
        if self.input_complete && in_buffer.len() > 0 {
            return err_exit_code(
                ExitCode::SyntaxError,
                "ERROR: input was marked as complete but more data was provided",
            );
        }

        let mut limited_output = LimitedOutputWriter {
            inner: output,
            amount_left: &mut self.jpeg_file_size_left,
        };

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
                            .context()?;

                        self.state = DecoderState::CompressedHeader(compressed_header_size);
                        *limited_output.amount_left = self.lh.jpeg_file_size.into();
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
                            .context()?;

                        self.state = DecoderState::CMP();
                    }
                }
                DecoderState::CMP() => {
                    if let Some(v) = in_buffer.take(3, 0) {
                        self.state = Self::process_cmp(
                            v,
                            &self.lh,
                            &self.enabled_features,
                            &self.thread_pool,
                            &mut limited_output,
                        )?;
                    }
                }

                DecoderState::ScanProgressive(state) => {
                    state.process_buffer(&mut in_buffer)?;

                    if input_complete {
                        Self::verify_eof_file_size(self.total_read_size, &mut in_buffer)?;

                        // complete the operation and merge the metrics
                        // progressive JPEGs cannot return partial results
                        let mut results = Vec::new();
                        while let Some((m, r)) = state.retrieve_result(true)? {
                            self.metrics.merge_from(m);

                            results.push(r);
                        }

                        Self::process_progressive(
                            &mut self.lh,
                            &self.enabled_features,
                            results,
                            &mut limited_output,
                        )?;

                        self.state = DecoderState::EOI;
                    }
                }
                DecoderState::ScanBaseline(state) => {
                    state.process_buffer(&mut in_buffer)?;

                    // baseline images can return partial results
                    if let Some((m, r)) = state.retrieve_result(false)? {
                        self.metrics.merge_from(m);
                        limited_output.write_all(&r)?;
                    }

                    if input_complete {
                        Self::verify_eof_file_size(self.total_read_size, &mut in_buffer)?;

                        // once we've complete the input, block for all remaining results
                        while let Some((m, r)) = state.retrieve_result(true)? {
                            self.metrics.merge_from(m);
                            limited_output.write_all(&r)?;
                        }

                        // Injection of restart codes for RST errors supports JPEGs with trailing RSTs.
                        // Run this logic even if early_eof_encountered to be compatible with C++ version.
                        //
                        // This logic is no longer needed for Rust generated Lepton files, since we just use the garbage
                        // data to store any extra RST codes or whatever else might be at the end of the file.
                        if self.lh.rinfo.rst_err.len() > 0 {
                            let cumulative_reset_markers = if self.lh.jpeg_header.rsti != 0 {
                                (self.lh.jpeg_header.mcuc - 1) / self.lh.jpeg_header.rsti
                            } else {
                                0
                            } as u8;

                            for i in 0..self.lh.rinfo.rst_err[0] {
                                let rst = jpeg_code::RST0 + ((cumulative_reset_markers + i) & 7);

                                limited_output.write_all(&[0xff, rst])?;
                            }
                        }

                        write_tail(&mut self.lh, &mut limited_output)?;

                        self.state = DecoderState::EOI;
                    }
                }
                DecoderState::EOI => {
                    break;
                }
            }
        }

        if input_complete {
            self.input_complete = true;
            match self.state {
                DecoderState::EOI => {
                    // all good, we don't need any more data to continue decoding
                }
                _ => {
                    return err_exit_code(ExitCode::SyntaxError,
                    format!("ERROR: input was marked as complete, but the decoder in state {:?} still needs more data",
                    std::mem::discriminant(&self.state)).as_str());
                }
            }
        }

        Ok(match self.state {
            DecoderState::EOI => true,
            _ => false,
        })
    }

    /// destructively reads the metrics
    pub fn take_metrics(&mut self) -> Metrics {
        mem::take(&mut self.metrics)
    }

    /// return metrics on decoder
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    fn process_progressive(
        lh: &mut LeptonHeader,
        enabled_features: &EnabledFeatures,
        mut image_segments: Vec<Vec<BlockBasedImage>>,
        output: &mut impl Write,
    ) -> Result<()> {
        let num_components = image_segments[0].len();
        let mut merged = Vec::new();
        for i in 0..num_components {
            merged.push(BlockBasedImage::merge(&mut image_segments, i).context()?);
        }

        output.write_all(&SOI)?;
        output
            .write_all(&lh.rinfo.raw_jpeg_header[0..lh.raw_jpeg_header_read_index])
            .context()?;

        let mut scnc = 0;

        loop {
            // progressive JPEG consists of scans followed by headers
            let scan =
                jpeg_write_entire_scan(&merged[..], &lh.jpeg_header, &lh.rinfo, scnc).context()?;

            output.write_all(&scan).context()?;

            // read the next headers (DHT, etc) while mirroring it back to the writer
            let old_pos = lh.raw_jpeg_header_read_index;
            let result = lh.advance_next_header_segment(enabled_features).context()?;

            output
                .write_all(&lh.rinfo.raw_jpeg_header[old_pos..lh.raw_jpeg_header_read_index])
                .context()?;

            if !result {
                break;
            }

            // advance to next scan
            scnc += 1;
        }

        write_tail(lh, output)?;

        Ok(())
    }

    fn process_cmp(
        v: Vec<u8>,
        lh: &LeptonHeader,
        enabled_features: &EnabledFeatures,
        thread_pool: &dyn LeptonThreadPool,
        output: &mut impl Write,
    ) -> Result<DecoderState> {
        if v[..] != LEPTON_HEADER_COMPLETION_MARKER {
            return err_exit_code(ExitCode::BadLeptonFile, "CMP marker not found");
        }
        Ok(if lh.jpeg_header.jpeg_type == JpegType::Progressive {
            let mux = Self::run_lepton_decoder_threads(
                lh,
                enabled_features,
                4, /* retain the last 4 bytes for the very end, since that is the file size, and shouldn't be parsed */
                thread_pool,
                |_thread_handoff, image_data, _, _| {
                    // just return the image data directly to be merged together
                    return Ok(image_data);
                },
            )
            .context()?;

            DecoderState::ScanProgressive(mux)
        } else {
            output.write_all(&SOI)?;
            output
                .write_all(&lh.rinfo.raw_jpeg_header[0..lh.raw_jpeg_header_read_index])
                .context()?;

            let mux = Self::run_lepton_decoder_threads(
                &lh,
                &enabled_features,
                4, /*retain 4 bytes for the end for the file size that is appended */
                thread_pool,
                |thread_handoff, image_data, jpeg_header, rinfo| {
                    let restart_info = RestartSegmentCodingInfo {
                        overhang_byte: thread_handoff.overhang_byte,
                        num_overhang_bits: thread_handoff.num_overhang_bits,
                        luma_y_start: thread_handoff.luma_y_start,
                        luma_y_end: thread_handoff.luma_y_end,
                        last_dc: thread_handoff.last_dc,
                    };

                    let mut result_buffer = jpeg_write_baseline_row_range(
                        thread_handoff.segment_size as usize,
                        &restart_info,
                        &image_data,
                        &jpeg_header,
                        &rinfo,
                    )
                    .context()?;

                    #[cfg(feature = "detailed_tracing")]
                    info!(
                        "ystart = {0}, segment_size = {1}, amount = {2}, offset = {3}, ob = {4}, nb = {5}",
                        thread_handoff.luma_y_start,
                        thread_handoff.segment_size,
                        result_buffer.len(),
                        thread_handoff.segment_offset_in_file,
                        thread_handoff.overhang_byte,
                        thread_handoff.num_overhang_bits
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

    fn verify_eof_file_size(total_read_size: u64, in_buffer: &mut PartialBuffer<'_>) -> Result<()> {
        if let Some(bytes) = in_buffer.take_n::<4>(0) {
            let size = u32::from_le_bytes(bytes);
            if u64::from(size) != total_read_size {
                return err_exit_code(
                    ExitCode::VerificationLengthMismatch,
                    format!(
                        "ERROR mismatch input_len = {0}, decoded_len = {1}",
                        size, total_read_size
                    ),
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

    /// starts the decoder threads
    fn run_lepton_decoder_threads<P: Send + 'static>(
        lh: &LeptonHeader,
        features: &EnabledFeatures,
        retention_bytes: usize,
        thread_pool: &dyn LeptonThreadPool,
        process: fn(
            thread_handoff: &ThreadHandoff,
            image_data: Vec<BlockBasedImage>,
            jpeg_header: &JpegHeader,
            rinfo: &ReconstructionInfo,
        ) -> Result<P>,
    ) -> Result<MultiplexReaderState<(Metrics, P)>> {
        let qt = QuantizationTables::construct_quantization_tables(&lh.jpeg_header)?;

        let features = features.clone();

        let thread_handoff = lh.thread_handoff.clone();

        let jpeg_header = lh.jpeg_header.clone();
        let rinfo = lh.rinfo.clone();

        let multiplex_reader_state = multiplex_read(
            thread_handoff.len(),
            thread_pool,
            retention_bytes,
            features.max_threads as usize,
            move |thread_id, reader, result_tx| {
                Self::run_lepton_decoder_processor(
                    &jpeg_header,
                    &rinfo,
                    &thread_handoff[thread_id],
                    thread_id == thread_handoff.len() - 1,
                    &qt,
                    reader,
                    &features,
                    process,
                    result_tx,
                )
            },
        );

        Ok(multiplex_reader_state)
    }

    /// the logic of a decoder thread. Takes a range of rows
    fn run_lepton_decoder_processor<P>(
        jpeg_header: &JpegHeader,
        rinfo: &ReconstructionInfo,
        thread_handoff: &ThreadHandoff,
        is_last_thread: bool,
        qt: &[QuantizationTables],
        reader: &mut MultiplexReader,
        features: &EnabledFeatures,
        process: fn(
            &ThreadHandoff,
            Vec<BlockBasedImage>,
            &JpegHeader,
            &ReconstructionInfo,
        ) -> Result<P>,
        result_send: &Sender<Result<(Metrics, P)>>,
    ) -> Result<()> {
        let cpu_time = CpuTimeMeasure::new();

        let mut image_data = Vec::new();
        for i in 0..jpeg_header.cmpc {
            image_data.push(BlockBasedImage::new(
                &jpeg_header,
                i,
                thread_handoff.luma_y_start,
                if is_last_thread {
                    // if this is the last thread, then the image should extend all the way to the bottom
                    jpeg_header.cmp_info[0].bcv
                } else {
                    thread_handoff.luma_y_end
                },
            )?);
        }

        let mut metrics = Metrics::default();

        metrics.merge_from(
            lepton_decode_row_range(
                &qt,
                &rinfo.truncate_components,
                &mut image_data,
                reader,
                thread_handoff.luma_y_start,
                thread_handoff.luma_y_end,
                is_last_thread,
                true,
                &features,
            )
            .context()?,
        );

        let process_result = process(thread_handoff, image_data, jpeg_header, rinfo)?;

        metrics.record_cpu_worker_time(cpu_time.elapsed());

        result_send.send(Ok((metrics, process_result)))?;

        Ok(())
    }
}

fn write_tail(lh: &mut LeptonHeader, output: &mut impl Write) -> Result<()> {
    output
        .write_all(&lh.rinfo.raw_jpeg_header[lh.raw_jpeg_header_read_index..])
        .context()?;
    output.write_all(&mut lh.rinfo.garbage_data).context()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use default_boxed::DefaultBoxed;

    use crate::{
        DEFAULT_THREAD_POOL, EnabledFeatures, decode_lepton,
        helpers::read_file,
        structs::{
            lepton_header::{FIXED_HEADER_SIZE, LeptonHeader},
            thread_handoff::ThreadHandoff,
        },
    };

    // test serializing and deserializing header
    #[test]
    fn parse_and_write_header() {
        use crate::jpeg::jpeg_read::read_jpeg_file;
        use std::io::Read;

        let min_jpeg = read_file("tiny", ".jpg");

        let mut lh = LeptonHeader::default_boxed();
        let enabled_features = EnabledFeatures::compat_lepton_vector_read();

        lh.jpeg_file_size = min_jpeg.len() as u32;
        lh.uncompressed_lepton_header_size = Some(752);

        let (_image_data, _partitions, _end_scan) = read_jpeg_file(
            &mut Cursor::new(min_jpeg),
            &mut lh.jpeg_header,
            &mut lh.rinfo,
            &enabled_features,
            |_, _| {},
        )
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

        let mut other = LeptonHeader::default_boxed();
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

    /// truncated progessive JPEG. We don't support creating these, but we can read them
    #[test]
    fn test_pixelated() {
        test_file("pixelated")
    }

    /// requires that the last segment be truncated by 1 byte.
    /// This is for compatibility with the C++ version
    #[test]
    fn test_truncate4() {
        test_file("truncate4")
    }

    fn test_file(filename: &str) {
        let file = read_file(filename, ".lep");
        let original = read_file(filename, ".jpg");

        let enabled_features = EnabledFeatures::compat_lepton_vector_read();

        let _ = decode_lepton_file_image(
            &mut Cursor::new(&file),
            &enabled_features,
            &DEFAULT_THREAD_POOL,
        )
        .unwrap();

        let mut output = Vec::new();

        decode_lepton(
            &mut Cursor::new(&file),
            &mut output,
            &enabled_features,
            &DEFAULT_THREAD_POOL,
        )
        .unwrap();

        assert_eq!(output.len(), original.len());
        assert!(output == original);
    }
}
