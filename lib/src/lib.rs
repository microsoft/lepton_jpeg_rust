/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

// Don't allow any unsafe code by default. Since this code has to potentially deal with
// badly/maliciously formatted images, we want this extra level of safety.
#![forbid(unsafe_code)]
#![forbid(trivial_casts)]
#![forbid(trivial_numeric_casts)]
#![forbid(non_ascii_idents)]
#![forbid(unused_extern_crates)]
#![forbid(unused_import_braces)]
#![forbid(redundant_lifetimes)]
#![forbid(single_use_lifetimes)]
#![forbid(unused_extern_crates)]
#![forbid(unused_lifetimes)]
#![forbid(unused_macro_rules)]
#![forbid(macro_use_extern_crate)]
#![forbid(missing_unsafe_on_extern)]

mod consts;
mod helpers;
mod jpeg;
mod metrics;
mod structs;

mod enabled_features;
mod lepton_error;

use std::io::{Cursor, Write};

pub use enabled_features::EnabledFeatures;
pub use helpers::catch_unwind_result;
pub use lepton_error::{ExitCode, LeptonError};
pub use metrics::{CpuTimeMeasure, Metrics};
pub use structs::lepton_file_writer::get_git_version;

use crate::lepton_error::{AddContext, Result};

pub use structs::simple_threadpool::set_thread_pool_priority;

pub use structs::lepton_file_reader::decode_lepton;

pub use structs::lepton_file_writer::{encode_lepton, encode_lepton_verify};

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
    ) -> Result<bool> {
        self.reader
            .process_buffer(input, input_complete, writer, output_buffer_size)
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

        (lh, block_image) = decode_lepton_file_image(&mut reader, enabled_features).context()?;

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
