/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

mod consts;
mod helpers;
mod jpeg_code;
mod structs;

pub mod lepton_error;

use crate::lepton_error::{ExitCode, LeptonError};

use core::result::Result;
use std::panic::catch_unwind;

use std::io::{Cursor, Read, Seek, Write};
use std::time::Duration;

use crate::structs::lepton_format::{decode_lepton_wrapper, encode_lepton_wrapper};

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
) -> Result<(), LeptonError> {
    let mut total_cpu_time = Duration::ZERO;
    decode_lepton_wrapper(reader, writer, num_threads, &mut total_cpu_time).map_err(translate_error)
}

/// Encodes JPEG as compressed Lepton format.
pub fn encode_lepton<R: Read + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    max_threads: usize,
    no_progressive: bool,
) -> Result<(), LeptonError> {
    let mut total_cpu_time = Duration::ZERO;
    encode_lepton_wrapper(
        reader,
        writer,
        max_threads,
        no_progressive,
        &mut total_cpu_time,
    )
    .map_err(translate_error)
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

        let mut total_cpu_time = Duration::ZERO;

        match encode_lepton_wrapper(
            &mut reader,
            &mut writer,
            number_of_threads as usize,
            false,
            &mut total_cpu_time,
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

        let mut total_cpu_time = Duration::ZERO;

        match decode_lepton_wrapper(
            &mut reader,
            &mut writer,
            number_of_threads as usize,
            &mut total_cpu_time,
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
