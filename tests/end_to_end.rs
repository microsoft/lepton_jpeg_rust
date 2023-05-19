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
        "eof_and_trailingrst"   // the lepton format has a wrongly set unexpected eof and trailing rst
    )]
    file: &str,
) {
    println!("decoding {0:?}", file);

    let input = read_file(file, ".lep");
    let expected = read_file(file, ".jpg");

    let mut output = Vec::new();

    decode_lepton(&mut Cursor::new(input), &mut output, 8).unwrap();

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

    decode_lepton(&mut Cursor::new(lepton), &mut output, 8).unwrap();

    assert!(input[..] == output[..]);
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
