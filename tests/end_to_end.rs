/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use core::result::Result;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use lepton_jpeg::lepton_error::{ExitCode, LeptonError};
use lepton_jpeg::{
    create_decompression_context, decode_lepton, decompress_image, encode_lepton,
    encode_lepton_verify, free_decompression_context, EnabledFeatures, WrapperCompressImage,
    WrapperDecompressImage, WrapperDecompressImageEx,
};
use rstest::rstest;

fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    let filename = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("images")
        .join(filename.to_owned() + ext);
    println!("reading {0}", filename.to_str().unwrap());
    let mut f = File::open(filename).unwrap();

    let mut content = Vec::new();
    f.read_to_end(&mut content).unwrap();

    content
}

/// verifies that the decode will accept existing Lepton files and generate
/// exactly the same jpeg from them. Used to detect unexpected divergences in coding format.
#[rstest]
fn verify_decode(
    #[values(
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

    let input = read_file(file, ".lep");
    let expected = read_file(file, ".jpg");

    let mut output = Vec::new();

    decode_lepton(
        &mut Cursor::new(input),
        &mut output,
        &EnabledFeatures::compat_lepton_vector_read(),
    )
    .unwrap();

    assert_eq!(
        output.len(),
        expected.len(),
        "length mismatch {} {}",
        output.len(),
        expected.len()
    );
    assert!(output[..] == expected[..]);
}

/// verifies that the decode will accept existing Lepton files and generate
/// exactly the same jpeg from them. Used to detect unexpected divergences in coding format.
#[test]
fn verify_decode_scalar_overflow() {
    let file = "mathoverflow_scalar";

    println!("decoding {0:?}", file);

    let input = read_file(file, ".lep");
    let expected = read_file(file, ".jpg");

    let mut output = Vec::new();

    let features = EnabledFeatures::compat_lepton_scalar_read();

    decode_lepton(&mut Cursor::new(input), &mut output, &features).unwrap();

    assert!(output[..] == expected[..]);
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
        let retval = WrapperDecompressImageEx(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
            true, // use_16bit_dc_estimate
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}

/// encodes as LEP and codes back to JPG to mostly test the encoder. Can't check against
/// the original LEP file since there's no guarantee they are binary identical (especially the zlib encoded part)
#[rstest]
fn verify_encode(
    #[values(
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
            "progressive_late_dht", // image has huffman tables that come very late which caused a verification failure 
            "out_of_order_dqt",
            //"narrowrst",
            //"nofsync",
            "slrcity",
            "slrhills",
            "slrindoor",
            "tiny",
            "trailingrst",
            "trailingrst2",
            "trunc",
        )]
    file: &str,
) {
    let input = read_file(file, ".jpg");

    let mut lepton = Vec::new();
    let mut output = Vec::new();

    encode_lepton(
        &mut Cursor::new(&input),
        &mut Cursor::new(&mut lepton),
        &EnabledFeatures::compat_lepton_vector_write(),
    )
    .unwrap();

    decode_lepton(
        &mut Cursor::new(lepton),
        &mut output,
        &EnabledFeatures::compat_lepton_vector_read(),
    )
    .unwrap();

    assert!(input[..] == output[..]);
}

#[test]
fn verify_16bitmath() {
    // verifies that we can decode 16 bit encoded images from the C++ version
    {
        let input = read_file("mathoverflow_16", ".lep");
        let expected = read_file("mathoverflow", ".jpg");

        let mut output = Vec::new();

        let features = EnabledFeatures::compat_lepton_vector_read();

        decode_lepton(&mut Cursor::new(input), &mut output, &features).unwrap();

        assert!(output[..] == expected[..]);
    }

    // verify that we can decode the one generated by the Rust version
    {
        let input = read_file("mathoverflow_32", ".lep");
        let expected = read_file("mathoverflow", ".jpg");

        let mut output = Vec::new();

        let mut features = EnabledFeatures::compat_lepton_vector_read();
        features.use_16bit_dc_estimate = false;

        decode_lepton(&mut Cursor::new(input), &mut output, &features).unwrap();

        assert!(output[..] == expected[..]);
    }
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
        let retval = WrapperDecompressImage(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}

/// encodes as LEP and codes back to JPG to mostly test the encoder. Can't check against
/// the original LEP file since there's no guarantee they are binary identical (especially the zlib encoded part)
#[rstest]
fn verify_encode_verify(#[values("slrcity")] file: &str) {
    let input = read_file(file, ".jpg");

    encode_lepton_verify(&input[..], &EnabledFeatures::compat_lepton_vector_write()).unwrap();
}

fn assert_exception<T>(expected_error: ExitCode, result: Result<T, LeptonError>) {
    match result {
        Ok(_) => panic!("failure was expected"),
        Err(e) => {
            assert_eq!(expected_error, e.exit_code(), "unexpected error {0:?}", e);
        }
    }
}

#[rstest]
fn verify_encode_verify_fail(#[values("mismatch_encode")] file: &str) {
    let input = read_file(file, ".jpg");

    assert_exception(
        ExitCode::VerificationContentMismatch,
        encode_lepton_verify(&input[..], &EnabledFeatures::compat_lepton_vector_write()),
    );
}

/// ensures we error out if we have the progressive flag disabled
#[rstest]
fn verify_encode_progressive_false(
    #[values("androidprogressive", "iphoneprogressive", "iphoneprogressive2")] file: &str,
) {
    let input = read_file(file, ".jpg");
    let mut lepton = Vec::new();
    assert_exception(
        ExitCode::ProgressiveUnsupported,
        encode_lepton(
            &mut Cursor::new(&input),
            &mut Cursor::new(&mut lepton),
            &EnabledFeatures {
                progressive: false,
                ..EnabledFeatures::compat_lepton_vector_write()
            },
        ),
    );
}

/// non-optimally zero length encoding progressive JPEGs cannot be recreated properly since the encoder always tries to create the longest zero runs
/// legally allowed given the available huffman codes.
#[test]
fn verify_nonoptimal() {
    let input = read_file("nonoptimalprogressive", ".jpg");
    let mut lepton = Vec::new();
    assert_exception(
        ExitCode::UnsupportedJpeg,
        encode_lepton(
            &mut Cursor::new(&input),
            &mut Cursor::new(&mut lepton),
            &EnabledFeatures::compat_lepton_vector_write(),
        ),
    );
}

/// processing of images with zeros in DQT tables may lead to divide-by-zero, therefore these images are not supported
#[test]
fn verify_encode_image_with_zeros_in_dqt_tables() {
    let input = read_file("zeros_in_dqt_tables", ".jpg");
    let mut lepton = Vec::new();

    assert_exception(
        ExitCode::UnsupportedJpegWithZeroIdct0,
        encode_lepton(
            &mut Cursor::new(&input),
            &mut Cursor::new(&mut lepton),
            &EnabledFeatures::compat_lepton_vector_write(),
        ),
    );
}

#[test]
fn extern_interface() {
    let input = read_file("slrcity", ".jpg");

    let mut compressed = Vec::new();

    compressed.resize(input.len() + 10000, 0);

    let mut result_size: u64 = 0;

    unsafe {
        let retval = WrapperCompressImage(
            input[..].as_ptr(),
            input.len() as u64,
            compressed[..].as_mut_ptr(),
            compressed.len() as u64,
            8,
            (&mut result_size) as *mut u64,
        );

        assert_eq!(retval, 0);
    }

    let mut original = Vec::new();
    original.resize(input.len() + 10000, 0);

    let mut original_size: u64 = 0;
    unsafe {
        let retval = WrapperDecompressImage(
            compressed[..].as_ptr(),
            result_size,
            original[..].as_mut_ptr(),
            original.len() as u64,
            8,
            (&mut original_size) as *mut u64,
        );

        assert_eq!(retval, 0);
    }
    assert_eq!(input.len() as u64, original_size);
    assert_eq!(input[..], original[..(original_size as usize)]);
}

/// tests the chunked decompression interface
#[test]
fn extern_interface_decompress_chunked() {
    let input = read_file("slrcity", ".lep");

    let mut output = Vec::new();

    unsafe {
        let context = create_decompression_context(1);

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
        let retval = WrapperCompressImage(
            input[..].as_ptr(),
            input.len() as u64,
            compressed[..].as_mut_ptr(),
            compressed.len() as u64,
            8,
            (&mut result_size) as *mut u64,
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
        let retval = WrapperDecompressImage(
            compressed[..].as_ptr(),
            compressed.len() as u64,
            decompressed[..].as_mut_ptr(),
            decompressed.len() as u64,
            8,
            (&mut decompressed_size) as *mut u64,
        );

        assert_eq!(retval, 0);
    }

    assert_eq!(original.len() as u64, decompressed_size);
    assert_eq!(original[..], decompressed[..(decompressed_size as usize)]);
}
