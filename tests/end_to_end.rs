/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use core::result::Result;
use std::{io::Cursor, path::Path};

use std::fs::File;
use std::io::Read;

use lepton_jpeg::metrics::Metrics;
use lepton_jpeg::{
    decode_lepton, encode_lepton, encode_lepton_verify,
    lepton_error::{ExitCode, LeptonError},
    EnabledFeatures,
};
use lepton_jpeg::{WrapperCompressImage, WrapperDecompressImage};

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
        "eof_and_trailingrst",          // the lepton format has a wrongly set unexpected eof and trailing rst
        "eof_and_trailinghdrdata",      // the lepton format has a wrongly set unexpected eof and trailing header data
        "trailingrst_missing_in_jpg"    // the lepton format has trailing rsts but they are missing in the JPG
    )]
    file: &str,
) {
    println!("decoding {0:?}", file);

    let input = read_file(file, ".lep");
    let expected = read_file(file, ".jpg");

    let mut output = Vec::new();

    decode_lepton(
        &mut Cursor::new(input),
        &mut Cursor::new(&mut output),
        8,
        &EnabledFeatures::all(),
    )
    .unwrap();

    assert!(output[..] == expected[..]);
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
        8,
        &EnabledFeatures::all(),
    )
    .unwrap();

    decode_lepton(
        &mut Cursor::new(lepton),
        &mut Cursor::new(&mut output),
        8,
        &EnabledFeatures::all(),
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

        let mut features = EnabledFeatures::all();
        features.use_16bit_dc_estimate = true;

        decode_lepton(&mut Cursor::new(input), &mut Cursor::new(&mut output), 8, &features).unwrap();

        assert!(output[..] == expected[..]);
    }

    // verify that we can decode the one generated by the Rust version
    {
        let input = read_file("mathoverflow_32", ".lep");
        let expected = read_file("mathoverflow", ".jpg");

        let mut output = Vec::new();

        let mut features = EnabledFeatures::all();
        features.use_16bit_dc_estimate = false;

        decode_lepton(&mut Cursor::new(input), &mut Cursor::new(&mut output), 8, &features).unwrap();

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

    encode_lepton_verify(&input[..], 8, &EnabledFeatures::all()).unwrap();
}

fn assert_exception(expected_error: ExitCode, result: Result<Metrics, LeptonError>) {
    match result {
        Ok(_) => panic!("failure was expected"),
        Err(e) => {
            assert_eq!(expected_error, e.exit_code, "unexpected error {0:?}", e);
        }
    }
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
            8,
            &EnabledFeatures {
                progressive: false,
                ..Default::default()
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
            8,
            &EnabledFeatures::all(),
        ),
    );
}

/// processing of images with zeros in DQT tables may lead to divide-by-zero, therefore these images are not supported
#[test]
fn verify_encode_image_with_zeros_in_dqt_tables() {
    let input = read_file("zeros_in_dqt_tables", ".jpg");
    let mut lepton = Vec::new();
    assert_exception(
        ExitCode::UnsupportedJpeg,
        encode_lepton(
            &mut Cursor::new(&input),
            &mut Cursor::new(&mut lepton),
            8,
            &EnabledFeatures::all(),
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

#[rstest]
fn verify_extern_interface_rejects_compression_of_unsupported_jpegs(
    #[values("zeros_in_dqt_tables", "nonoptimalprogressive")] file: &str,
) {
    let input = read_file(file, ".jpg");

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

        assert_eq!(retval, ExitCode::UnsupportedJpeg as i32);
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
