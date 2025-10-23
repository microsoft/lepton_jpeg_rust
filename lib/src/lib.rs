/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

//! A lossless JPEG compressor with precise bit-for-bit recovery, supporting both baseline and progressive JPEGs.
//! Achieves compression savings of around 22%, making it suitable for cold cloud storage use cases.
//!
//! This crate is a Rust port of Dropbox’s original [lepton](https://github.com/dropbox/lepton) JPEG compression tool.
//! It retains the performance characteristics of the C++ version while benefiting from Rust’s memory safety guarantees.
//! All JPEG content—including metadata and even malformed segments—is preserved accurately.
//!
//! The original C++ codebase has been deprecated by Dropbox. This Rust implementation incorporates
//! an exhaustive security review of the original, making it a safer and more maintainable alternative.

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
#![deny(missing_docs)]

mod consts;
mod helpers;
mod jpeg;
mod metrics;
mod structs;

mod enabled_features;
mod lepton_error;

pub use enabled_features::EnabledFeatures;
pub use helpers::catch_unwind_result;
pub use lepton_error::{ExitCode, LeptonError};
pub use metrics::{CpuTimeMeasure, Metrics};
pub use structs::lepton_file_writer::get_git_version;

use crate::lepton_error::{AddContext, Result};
pub use crate::structs::simple_threadpool::{
    DEFAULT_THREAD_POOL, LeptonThreadPool, LeptonThreadPriority, SimpleThreadPool,
    SingleThreadPool, ThreadPoolHolder,
};

#[cfg(feature = "micro_benchmark")]
/// Module that exposes internal functions for micro benchmarking
pub mod micro_benchmark;

/// Trait for types that can provide the current position in a stream. This
/// is intentionally a subset of the Seek trait, as it only requires remembering
/// the current position without allowing seeking to arbitrary positions.
///
/// This is useful for callers for which it would be complex to provide seek capabilities, but can
/// count the number of bytes read or written so far.
///
/// We provide a blanket implementation for any type that implements `std::io::Seek`.
pub trait StreamPosition {
    /// Returns the current position in the stream.
    fn position(&mut self) -> u64;
}

impl<T: std::io::Seek> StreamPosition for T {
    fn position(&mut self) -> u64 {
        self.stream_position().unwrap()
    }
}

pub use structs::lepton_file_reader::decode_lepton;

pub use structs::lepton_file_writer::{encode_lepton, encode_lepton_verify};

static PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub use structs::lepton_file_reader::LeptonFileReader;

/// Returns the version string of the library, which includes the package version and the git version.
/// This is useful for debugging and logging purposes to know the exact version of the library is being used
pub fn get_version_string() -> String {
    format!("{}-{}", PACKAGE_VERSION, get_git_version())
}

/// used by utility to dump out the contents of a jpeg file or lepton file for debugging purposes
#[allow(dead_code)]
pub fn dump_jpeg(input_data: &[u8], all: bool, enabled_features: &EnabledFeatures) -> Result<()> {
    use std::io::Cursor;
    use structs::lepton_file_reader::decode_lepton_file_image;
    use structs::lepton_file_writer::read_jpeg;

    let mut lh;
    let block_image;

    if input_data[0] == 0xff && input_data[1] == 0xd8 {
        let mut reader = Cursor::new(input_data);

        (lh, block_image) = read_jpeg(&mut reader, enabled_features, |jh, _ri| {
            println!("parsed header:");
            let s = format!("{jh:?}");
            println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));
        })?;
    } else {
        let mut reader = Cursor::new(input_data);

        (lh, block_image) =
            decode_lepton_file_image(&mut reader, enabled_features, &DEFAULT_THREAD_POOL)
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
