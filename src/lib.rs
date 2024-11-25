/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

mod consts;
mod helpers;
mod jpeg_code;
pub mod metrics;
mod structs;

pub mod enabled_features;
pub mod lepton_error;

use std::io::{BufRead, Cursor, Seek, Write};

pub use enabled_features::EnabledFeatures;
use helpers::{catch_unwind_result, copy_cstring_utf8_to_buffer};
pub use lepton_error::{ExitCode, LeptonError};
pub use metrics::Metrics;

use crate::lepton_error::{AddContext, Result};

#[cfg(not(feature = "use_rayon"))]
pub fn set_thread_priority(priority: i32) {
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    {
        let p = match priority {
            100 => thread_priority::ThreadPriority::Max,
            0 => thread_priority::ThreadPriority::Min,
            _ => panic!("Unsupported thread priority value: {}", priority),
        };

        thread_priority::set_current_thread_priority(p).unwrap();
        crate::structs::simple_threadpool::set_thread_priority(p);
    }
}

/// Decodes Lepton container and recreates the original JPEG file
pub fn decode_lepton<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    structs::lepton_file_reader::decode_lepton_file(reader, writer, enabled_features)
}

/// Encodes JPEG as compressed Lepton format.
pub fn encode_lepton<R: BufRead + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics> {
    structs::lepton_file_writer::encode_lepton_wrapper(reader, writer, enabled_features)
}

/// Compresses JPEG into Lepton format and compares input to output to verify that compression roundtrip is OK
pub fn encode_lepton_verify(
    input_data: &[u8],
    enabled_features: &EnabledFeatures,
) -> Result<(Vec<u8>, Metrics)> {
    structs::lepton_file_writer::encode_lepton_wrapper_verify(input_data, enabled_features)
}

/// C ABI interface for compressing image, exposed from DLL
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn WrapperCompressImage(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: i32,
    result_size: *mut u64,
) -> i32 {
    match catch_unwind_result(|| {
        let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);

        let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

        let mut reader = Cursor::new(input);
        let mut writer = Cursor::new(output);

        let mut features = EnabledFeatures::compat_lepton_vector_write();
        if number_of_threads > 0 {
            features.max_threads = number_of_threads as u32;
        }

        let _metrics = encode_lepton(&mut reader, &mut writer, &features)?;

        *result_size = writer.position().into();

        Ok(())
    }) {
        Ok(()) => {
            return 0;
        }
        Err(e) => {
            return e.exit_code().as_integer_error_code();
        }
    }
}

/// C ABI interface for decompressing image, exposed from DLL
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn WrapperDecompressImage(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: i32,
    result_size: *mut u64,
) -> i32 {
    return WrapperDecompressImageEx(
        input_buffer,
        input_buffer_size,
        output_buffer,
        output_buffer_size,
        number_of_threads,
        result_size,
        false, // use_16bit_dc_estimate
    );
}

/// C ABI interface for decompressing image, exposed from DLL.
/// use_16bit_dc_estimate argument should be set to true only for images
/// that were compressed by C++ version of Leptron (see comments below).
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn WrapperDecompressImageEx(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: i32,
    result_size: *mut u64,
    use_16bit_dc_estimate: bool,
) -> i32 {
    match catch_unwind_result(|| {
        // For back-compat with C++ version we allow decompression of images with zeros in DQT tables

        // C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
        // depending on the compiler options. If use_16bit_dc_estimate=true, the decompression uses a back-compat
        // mode that considers it. The caller should set use_16bit_dc_estimate to true only for images that were
        // compressed by C++ version with relevant compiler options.

        // this is a bit of a mess since for a while we were encoded a mix of 16 and 32 bit math
        // (hence the two parameters in features).

        let mut enabled_features = EnabledFeatures {
            use_16bit_dc_estimate: use_16bit_dc_estimate,
            ..EnabledFeatures::compat_lepton_vector_read()
        };

        if number_of_threads > 0 {
            enabled_features.max_threads = number_of_threads as u32;
        }

        loop {
            let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);
            let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

            let mut reader = Cursor::new(input);
            let mut writer = Cursor::new(output);

            match decode_lepton(&mut reader, &mut writer, &mut enabled_features) {
                Ok(_) => {
                    *result_size = writer.position().into();
                    return Ok(());
                }
                Err(e) => {
                    // The retry logic below runs if the caller did not pass use_16bit_dc_estimate=true, but the decompression
                    // encountered StreamInconsistent failure which is commonly caused by the the C++ 16 bit bug. In this case
                    // we retry the decompression with use_16bit_dc_estimate=true.
                    // Note that it's prefferable for the caller to pass use_16bit_dc_estimate properly and not to rely on this
                    // retry logic, that may miss some cases leading to bad (corrupted) decompression results.
                    if e.exit_code() == ExitCode::StreamInconsistent
                        && !enabled_features.use_16bit_dc_estimate
                    {
                        enabled_features.use_16bit_dc_estimate = true;
                        continue;
                    }

                    return Err(e.into());
                }
            }
        }
    }) {
        Ok(()) => {
            return 0;
        }
        Err(e) => {
            return e.exit_code().as_integer_error_code();
        }
    }
}

static GIT_VERSION: &str =
    git_version::git_version!(args = ["--abbrev=40", "--always", "--dirty=-modified"]);

static PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[no_mangle]
pub unsafe extern "C" fn get_version(
    package: &mut *const std::os::raw::c_char,
    git: &mut *const std::os::raw::c_char,
) {
    *git = GIT_VERSION.as_ptr() as *const std::os::raw::c_char;
    *package = PACKAGE_VERSION.as_ptr() as *const std::os::raw::c_char;
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

const DECOMPRESS_USE_16BIT_DC_ESTIMATE: u32 = 1;

#[no_mangle]
pub unsafe extern "C" fn create_decompression_context(features: u32) -> *mut std::ffi::c_void {
    let enabled_features = if features & DECOMPRESS_USE_16BIT_DC_ESTIMATE != 0 {
        EnabledFeatures::compat_lepton_vector_read()
    } else {
        EnabledFeatures::compat_lepton_scalar_read()
    };

    let context = Box::new(LeptonFileReaderContext::new(enabled_features));
    Box::into_raw(context) as *mut std::ffi::c_void
}

#[no_mangle]
pub unsafe extern "C" fn free_decompression_context(context: *mut std::ffi::c_void) {
    let _ = Box::from_raw(context as *mut LeptonFileReaderContext);
    // let Box destroy the object
}

/// partially decompresses an image from a Lepton file.
///
/// Returns -1 if more data is needed or if there is more data available, or 0 if done successfully.
/// Returns > 0 if there is an error
#[no_mangle]
pub unsafe extern "C" fn decompress_image(
    context: *mut std::ffi::c_void,
    input_buffer: *const u8,
    input_buffer_size: u64,
    input_complete: bool,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    result_size: *mut u64,
    error_string: *mut std::os::raw::c_uchar,
    error_string_buffer_len: u64,
) -> i32 {
    match catch_unwind_result(|| {
        let context = context as *mut LeptonFileReaderContext;
        let context = &mut *context;

        let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);
        let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

        let mut writer = Cursor::new(output);
        let done = context.process_buffer(
            input,
            input_complete,
            &mut writer,
            output_buffer_size as usize,
        )?;

        *result_size = writer.position().into();
        Ok(done)
    }) {
        Ok(done) => {
            if done {
                0
            } else {
                -1
            }
        }
        Err(e) => {
            copy_cstring_utf8_to_buffer(
                e.message(),
                std::slice::from_raw_parts_mut(error_string, error_string_buffer_len as usize),
            );
            e.exit_code().as_integer_error_code()
        }
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
