/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

// Don't allow any unsafe code by default. Since this code has to potentially deal with
// badly/maliciously formatted images, we want this extra level of safety.
#![forbid(unsafe_code)]
#![forbid(trivial_numeric_casts)]
#![forbid(unused_crate_dependencies)]

mod consts;
mod helpers;
mod jpeg;
mod metrics;
mod structs;

mod enabled_features;
mod lepton_error;

use std::io::{BufRead, Cursor, Seek, Write};

pub use enabled_features::EnabledFeatures;
pub use helpers::catch_unwind_result;
pub use lepton_error::{ExitCode, LeptonError};
pub use metrics::{CpuTimeMeasure, Metrics};
pub use structs::lepton_file_writer::get_git_version;
pub use structs::simple_threadpool::{LeptonThreadPool, LeptonThreadPriority, DEFAULT_THREAD_POOL};

use crate::lepton_error::{AddContext, Result};

/// Decodes Lepton container and recreates the original JPEG file
pub fn decode_lepton<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
    thread_pool: &dyn LeptonThreadPool,
) -> Result<Metrics> {
    structs::lepton_file_reader::decode_lepton_file(reader, writer, enabled_features, thread_pool)
}

/// Encodes JPEG as compressed Lepton format.
pub fn encode_lepton<R: BufRead + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
    thread_pool: &dyn LeptonThreadPool,
) -> Result<Metrics> {
    structs::lepton_file_writer::encode_lepton_wrapper(
        reader,
        writer,
        enabled_features,
        thread_pool,
    )
}

/// Compresses JPEG into Lepton format and compares input to output to verify that compression roundtrip is OK
pub fn encode_lepton_verify(
    input_data: &[u8],
    enabled_features: &EnabledFeatures,
    thread_pool: &dyn LeptonThreadPool,
) -> Result<(Vec<u8>, Metrics)> {
    structs::lepton_file_writer::encode_lepton_wrapper_verify(
        input_data,
        enabled_features,
        thread_pool,
    )
}

static PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn get_version_string() -> String {
    format!("{}-{}", PACKAGE_VERSION, get_git_version())
}

/// Holds context and buffers while decompressing a Lepton encoded file.
///
/// Dropping the object will abort any threads or decoding in progress.
pub struct LeptonFileReaderContext {
    reader: structs::lepton_file_reader::LeptonFileReader,
}

impl LeptonFileReaderContext {
    /// Creates a new context for decompressing Lepton encoded files,
    /// features parameter can be used to enable or disable certain behaviors.
    pub fn new(features: EnabledFeatures) -> LeptonFileReaderContext {
        LeptonFileReaderContext {
            reader: structs::lepton_file_reader::LeptonFileReader::new(features),
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
        input: &[u8],
        input_complete: bool,
        writer: &mut impl Write,
        output_buffer_size: usize,
        thread_pool: &dyn LeptonThreadPool,
    ) -> Result<bool> {
        self.reader.process_buffer(
            input,
            input_complete,
            writer,
            output_buffer_size,
            thread_pool,
        )
    }
}

/// used by utility to dump out the contents of a jpeg file or lepton file for debugging purposes
#[allow(dead_code)]
pub fn dump_jpeg(input_data: &[u8], all: bool, enabled_features: &EnabledFeatures) -> Result<()> {
    use structs::lepton_file_reader::decode_lepton_file_image;
    use structs::lepton_file_writer::read_jpeg;

    let mut lh;
    let block_image;

    if input_data[0] == 0xff && input_data[1] == 0xd8 {
        let mut reader = Cursor::new(input_data);

        (lh, block_image) = read_jpeg(&mut reader, enabled_features, |jh| {
            println!("parsed header:");
            let s = format!("{jh:?}");
            println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));
        })?;
    } else {
        let mut reader = Cursor::new(input_data);

        (lh, block_image) =
            decode_lepton_file_image(&mut reader, enabled_features, DEFAULT_THREAD_POOL)
                .context()?;

        loop {
            println!("parsed header:");
            let s = format!("{0:?}", lh.jpeg_header);
            println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));

            if !lh
                .advance_next_header_segment(&enabled_features)
                .context()?
            {
                break;
            }
        }
    }

    let s = format!("{lh:?}");
    println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));

    if all {
        for i in 0..block_image.len() {
            println!("Component {0}", i);
            let image = &block_image[i];
            for dpos in 0..image.get_block_width() * image.get_original_height() {
                print!("dpos={0} ", dpos);
                let block = image.get_block(dpos);

                print!("{0}", block.get_transposed_from_zigzag(0));
                for i in 1..64 {
                    print!(",{0}", block.get_transposed_from_zigzag(i));
                }
                println!();
            }
        }
    }

    return Ok(());
}
