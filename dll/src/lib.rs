/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

#![forbid(trivial_numeric_casts)]
#![forbid(unused_crate_dependencies)]

use std::{io::Cursor, sync::LazyLock};

use lepton_jpeg::{
    catch_unwind_result, decode_lepton, encode_lepton, get_git_version, EnabledFeatures, ExitCode,
    LeptonFileReaderContext, LeptonThreadPool, DEFAULT_THREAD_POOL,
};
use rstest::rstest;

/// copies a string into a limited length zero terminated utf8 buffer
fn copy_cstring_utf8_to_buffer(str: &str, target_error_string: &mut [u8]) {
    if target_error_string.len() == 0 {
        return;
    }

    // copy error string into the buffer as utf8
    let b = std::ffi::CString::new(str).unwrap();
    let b = b.as_bytes();

    let copy_len = std::cmp::min(b.len(), target_error_string.len() - 1);

    // copy string into buffer as much as fits
    target_error_string[0..copy_len].copy_from_slice(&b[0..copy_len]);

    // always null terminated
    target_error_string[copy_len] = 0;
}

struct RayonThreadPool {
    pool: LazyLock<rayon::ThreadPool>,
}

impl LeptonThreadPool for RayonThreadPool {
    fn run(&'static self, f: Box<dyn FnOnce() + Send + 'static>) {
        self.pool.spawn(f);
    }
}

static RAYON_THREAD_POOL: RayonThreadPool = RayonThreadPool {
    pool: LazyLock::new(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(8) // default to 8 threads, can be adjusted
            .build()
            .unwrap()
    }),
};

#[test]
fn test_copy_cstring_utf8_to_buffer() {
    // test utf8
    let mut buffer = [0u8; 10];
    copy_cstring_utf8_to_buffer("h\u{00E1}llo", &mut buffer);
    assert_eq!(buffer, [b'h', 0xc3, 0xa1, b'l', b'l', b'o', 0, 0, 0, 0]);

    // test null termination
    let mut buffer = [0u8; 10];
    copy_cstring_utf8_to_buffer("helloeveryone", &mut buffer);
    assert_eq!(
        buffer,
        [b'h', b'e', b'l', b'l', b'o', b'e', b'v', b'e', b'r', 0]
    );
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
    WrapperCompressImage2(
        input_buffer,
        input_buffer_size,
        output_buffer,
        output_buffer_size,
        number_of_threads as u32,
        result_size,
        0,
    )
}

/// C ABI interface for compressing image, exposed from DLL
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn WrapperCompressImage2(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: u32,
    result_size: *mut u64,
    flags: u32,
) -> i32 {
    match catch_unwind_result(|| {
        let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);

        let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

        let mut reader = Cursor::new(input);
        let mut writer = Cursor::new(output);

        let mut features = EnabledFeatures::compat_lepton_vector_write();
        if number_of_threads > 0 {
            features.max_threads = number_of_threads;
        }

        let thread_pool: &'static dyn LeptonThreadPool = if flags & USE_RAYON_THREAD_POOL != 0 {
            &RAYON_THREAD_POOL
        } else {
            &DEFAULT_THREAD_POOL
        };

        let _metrics = encode_lepton(&mut reader, &mut writer, &features, thread_pool)?;

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
    return WrapperDecompressImage3(
        input_buffer,
        input_buffer_size,
        output_buffer,
        output_buffer_size,
        number_of_threads as u32,
        result_size,
        0,
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
    WrapperDecompressImage3(
        input_buffer,
        input_buffer_size,
        output_buffer,
        output_buffer_size,
        number_of_threads as u32,
        result_size,
        if use_16bit_dc_estimate {
            DECOMPRESS_USE_16BIT_DC_ESTIMATE
        } else {
            0
        },
    )
}

/// C ABI interface for decompressing image, exposed from DLL.
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "C" fn WrapperDecompressImage3(
    input_buffer: *const u8,
    input_buffer_size: u64,
    output_buffer: *mut u8,
    output_buffer_size: u64,
    number_of_threads: u32,
    result_size: *mut u64,
    flags: u32,
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
            use_16bit_dc_estimate: (flags & DECOMPRESS_USE_16BIT_DC_ESTIMATE != 0),
            ..EnabledFeatures::compat_lepton_vector_read()
        };

        if number_of_threads > 0 {
            enabled_features.max_threads = number_of_threads;
        }

        let thread_pool: &'static dyn LeptonThreadPool = if flags & USE_RAYON_THREAD_POOL != 0 {
            &RAYON_THREAD_POOL
        } else {
            &DEFAULT_THREAD_POOL
        };

        loop {
            let input = std::slice::from_raw_parts(input_buffer, input_buffer_size as usize);
            let output = std::slice::from_raw_parts_mut(output_buffer, output_buffer_size as usize);

            let mut reader = Cursor::new(input);
            let mut writer = Cursor::new(output);

            match decode_lepton(&mut reader, &mut writer, &mut enabled_features, thread_pool) {
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

static PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn get_version_string() -> String {
    format!("{}-{}", PACKAGE_VERSION, get_git_version())
}

#[no_mangle]
pub unsafe extern "C" fn get_version(
    package: &mut *const std::os::raw::c_char,
    git: &mut *const std::os::raw::c_char,
) {
    *git = get_git_version().as_ptr() as *const std::os::raw::c_char;
    *package = PACKAGE_VERSION.as_ptr() as *const std::os::raw::c_char;
}

const DECOMPRESS_USE_16BIT_DC_ESTIMATE: u32 = 1;
const USE_RAYON_THREAD_POOL: u32 = 2;

#[no_mangle]
pub unsafe extern "C" fn create_decompression_context(features: u32) -> *mut std::ffi::c_void {
    let enabled_features = if features & DECOMPRESS_USE_16BIT_DC_ESTIMATE != 0 {
        EnabledFeatures::compat_lepton_vector_read()
    } else {
        EnabledFeatures::compat_lepton_scalar_read()
    };

    let thread_pool: &'static dyn LeptonThreadPool = if features & USE_RAYON_THREAD_POOL != 0 {
        &RAYON_THREAD_POOL
    } else {
        &DEFAULT_THREAD_POOL
    };

    let context = Box::new(LeptonFileReaderContext::new(enabled_features, thread_pool));
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

#[cfg(test)]
fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    let filename = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("images")
        .join(filename.to_owned() + ext);
    println!("reading {0}", filename.to_str().unwrap());
    let mut f = std::fs::File::open(filename).unwrap();

    let mut content = Vec::new();
    std::io::Read::read_to_end(&mut f, &mut content).unwrap();

    content
}

#[test]
fn extern_interface() {
    let input = read_file("slrcity", ".jpg");

    let mut compressed = Vec::new();

    compressed.resize(input.len() + 10000, 0);

    let mut result_size: u64 = 0;

    unsafe {
        let retval = WrapperCompressImage2(
            input[..].as_ptr(),
            input.len() as u64,
            compressed[..].as_mut_ptr(),
            compressed.len() as u64,
            8,
            (&mut result_size) as *mut u64,
            0,
        );

        assert_eq!(retval, 0);
    }

    let mut original = Vec::new();
    original.resize(input.len() + 10000, 0);

    let mut original_size: u64 = 0;
    unsafe {
        let retval = WrapperDecompressImage3(
            compressed[..].as_ptr(),
            result_size,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
            0,
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}

/// tests the chunked decompression interface
#[rstest]
fn extern_interface_decompress_chunked(
    #[values(DECOMPRESS_USE_16BIT_DC_ESTIMATE,DECOMPRESS_USE_16BIT_DC_ESTIMATE|USE_RAYON_THREAD_POOL)]
    flags: u32,
) {
    use std::io::Read;

    let input = read_file("slrcity", ".lep");

    let mut output = Vec::new();

    unsafe {
        let context = create_decompression_context(flags);

        let mut file_read = Cursor::new(input);
        let mut input_buffer = [0u8; 7];
        let mut output_buffer = [0u8; 13];

        let mut error_string = [0u8; 1024];

        loop {
            let amount_read = file_read.read(&mut input_buffer).unwrap();

            let mut result_size = 0;
            let result = decompress_image(
                context,
                input_buffer.as_ptr(),
                amount_read as u64,
                amount_read == 0,
                output_buffer.as_mut_ptr(),
                output_buffer.len() as u64,
                &mut result_size,
                error_string.as_mut_ptr(),
                error_string.len() as u64,
            );

            output.extend_from_slice(&output_buffer[..result_size as usize]);

            match result {
                -1 => {
                    // need more data
                }
                0 => {
                    break;
                }
                _ => {
                    panic!("unexpected error {0}", result);
                }
            }
        }
        free_decompression_context(context);
    }

    let test_result = read_file("slrcity", ".jpg");
    assert_eq!(test_result.len(), output.len());
    assert!(test_result[..] == output[..]);
}

#[rstest]
fn verify_extern_interface_rejects_compression_of_unsupported_jpegs(
    #[values(
        ("zeros_in_dqt_tables", ExitCode::UnsupportedJpegWithZeroIdct0), 
        ("nonoptimalprogressive", ExitCode::UnsupportedJpeg))]
    file: (&str, ExitCode),
) {
    let input = read_file(file.0, ".jpg");

    let mut compressed = Vec::new();
    compressed.resize(input.len() + 10000, 0);
    let mut result_size: u64 = 0;

    unsafe {
        let retval = WrapperCompressImage2(
            input[..].as_ptr(),
            input.len() as u64,
            compressed[..].as_mut_ptr(),
            compressed.len() as u64,
            8,
            (&mut result_size) as *mut u64,
            0,
        );

        assert_eq!(retval, file.1.as_integer_error_code());
    }
}

/// While we prevent compression of images with zeros in DQT tables, since it may lead to divide-by-zero, we support decompression of
/// previously compressed images with this characteristics for back-compat.
#[rstest]
fn verify_extern_interface_supports_decompression_with_zeros_in_dqt_tables(
    #[values("zeros_in_dqt_tables")] file: &str,
) {
    let compressed = read_file(file, ".lep");
    let original = read_file(file, ".jpg");

    let mut decompressed = Vec::new();
    decompressed.resize(original.len() + 10000, 0);

    let mut decompressed_size: u64 = 0;
    unsafe {
        let retval = WrapperDecompressImage3(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            decompressed[..].as_mut_ptr(),
            decompressed.len() as u64,
            8,
            (&mut decompressed_size) as *mut u64,
            0,
        );

        assert_eq!(retval, 0);
    }

    assert_eq!(original.len() as u64, decompressed_size);
    assert_eq!(original[..], decompressed[..(decompressed_size as usize)]);
}

/// Verifies that the decode will accept existing Lepton files and generate
/// exactly the same jpeg from them when called by an external interface
/// with use_16bit_dc_estimate=true for C++ backward compatibility.
/// Used to detect unexpected divergences in coding format.
#[rstest]
fn verify_decode_external_interface_with_use_16bit_dc_estimate(
    #[values(
        "mathoverflow_16",
        "android",
        "androidcrop",
        "androidcropoptions",
        "androidprogressive",
        "androidprogressive_garbage",
        "androidtrail",
        "colorswap",
        "gray2sf",
        "grayscale",
        "hq",
        "iphone",
        "iphonecity",
        "iphonecity_with_16KGarbage",
        "iphonecity_with_1MGarbage",
        "iphonecrop",
        "iphonecrop2",
        "iphoneprogressive",
        "iphoneprogressive2",
        "progressive_late_dht", // image has huffman tables that come very late which causes a verification failure 
        "out_of_order_dqt",     // image with quanatization table dqt that comes after image definition SOF
        "narrowrst",
        "nofsync",
        "slrcity",
        "slrhills",
        "slrindoor",
        "tiny",
        "trailingrst",
        "trailingrst2",
        "trunc",
        "eof_and_trailingrst",    // the lepton format has a wrongly set unexpected eof and trailing rst
        "eof_and_trailinghdrdata" // the lepton format has a wrongly set unexpected eof and trailing header data
    )]
    file: &str,
) {
    println!("decoding {0:?}", file);

    let compressed = read_file(file, ".lep");
    let jpg_file_name = match file {
        "mathoverflow_16" => "mathoverflow",
        _ => file,
    };
    let input = read_file(jpg_file_name, ".jpg");

    let mut original = Vec::new();
    original.resize(input.len() + 10000, 0);

    let mut original_size: u64 = 0;
    unsafe {
        let retval = WrapperDecompressImage3(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
            DECOMPRESS_USE_16BIT_DC_ESTIMATE,
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}

#[test]
fn verify_extern_16bit_math_retry() {
    // verify retry logic for 16 bit math encoded image
    let compressed = read_file("mathoverflow_16", ".lep");

    let input = read_file("mathoverflow", ".jpg");

    let mut original = Vec::new();
    original.resize(input.len() + 10000, 0);

    let mut original_size: u64 = 0;
    unsafe {
        let retval = WrapperDecompressImage3(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
            0,
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}
