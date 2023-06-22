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

pub use crate::enabled_features::EnabledFeatures;
pub use crate::lepton_error::{ExitCode, LeptonError};
pub use metrics::Metrics;

use core::result::Result;
use std::panic::catch_unwind;

use std::io::{Cursor, Read, Seek, Write};

use crate::structs::lepton_format::{
    decode_lepton_wrapper, encode_lepton_wrapper, encode_lepton_wrapper_verify,
};

/// translates internal anyhow based exception into externally visible exception
fn translate_error(e: anyhow::Error) -> LeptonError {
    match e.root_cause().downcast_ref::<LeptonError>() {
        // try to extract the exit code if it was a well known error
        Some(x) => {
            return LeptonError {
                exit_code: x.exit_code,
                message: x.message.to_owned(),
            };
        }
        None => {
            return LeptonError {
                exit_code: ExitCode::GeneralFailure,
                message: format!("unexpected error {0:?}", e),
            };
        }
    }
}

/// Decodes Lepton container and recreates the original JPEG file
pub fn decode_lepton<R: Read + Seek, W: Write>(
    reader: &mut R,
    writer: &mut W,
    num_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics, LeptonError> {
    decode_lepton_wrapper(reader, writer, num_threads, enabled_features).map_err(translate_error)
}

/// Encodes JPEG as compressed Lepton format.
pub fn encode_lepton<R: Read + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    max_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<Metrics, LeptonError> {
    encode_lepton_wrapper(reader, writer, max_threads, enabled_features).map_err(translate_error)
}

/// Compresses JPEG into Lepton format and compares input to output to verify that compression roundtrip is OK
pub fn encode_lepton_verify(
    input_data: &[u8],
    max_threads: usize,
    enabled_features: &EnabledFeatures,
) -> Result<(Vec<u8>, Metrics), LeptonError> {
    encode_lepton_wrapper_verify(input_data, max_threads, enabled_features).map_err(translate_error)
}

/// C ABI interface for compressing image, exposed from DLL
#[no_mangle]
pub unsafe extern "C" fn WrapperCompressImage(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: i32,
    result_size: *mut u64,
) -> i32 {
    match catch_unwind(|| {
        let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);

        let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

        let mut reader = Cursor::new(input);
        let mut writer = Cursor::new(output);

        match encode_lepton_wrapper(
            &mut reader,
            &mut writer,
            number_of_threads as usize,
            &EnabledFeatures::default(),
        ) {
            Ok(_) => {}
            Err(e) => match e.root_cause().downcast_ref::<LeptonError>() {
                // try to extract the exit code if it was a well known error
                Some(x) => {
                    return x.exit_code as i32;
                }
                None => {
                    return -1 as i32;
                }
            },
        }

        *result_size = writer.position().into();

        return 0;
    }) {
        Ok(code) => {
            return code;
        }
        Err(_) => {
            return -2;
        }
    }
}

/// C ABI interface for decompressing image, exposed from DLL
#[no_mangle]
pub unsafe extern "C" fn WrapperDecompressImage(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: i32,
    result_size: *mut u64,
) -> i32 {
    match catch_unwind(|| {
        let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);

        let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

        let mut reader = Cursor::new(input);
        let mut writer = Cursor::new(output);

        // For back-compat with C++ version we allow decompression of images with zeros in DQT tables
        let mut enabled_features = EnabledFeatures::all();
        enabled_features.reject_dqts_with_zeros = false;

        match decode_lepton_wrapper(
            &mut reader,
            &mut writer,
            number_of_threads as usize,
            &enabled_features,
        ) {
            Ok(_) => {}
            Err(e) => {
                return translate_error(e).exit_code as i32;
            }
        }

        *result_size = writer.position().into();

        return 0;
    }) {
        Ok(code) => {
            return code;
        }
        Err(_) => {
            return -2;
        }
    }
}
