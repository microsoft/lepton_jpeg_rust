/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use core::result::Result;
use std::fs::read_dir;
use std::io::Cursor;
use std::path::Path;

use lepton_jpeg::{
    DEFAULT_THREAD_POOL, EnabledFeatures, decode_lepton, encode_lepton, encode_lepton_verify,
};
use lepton_jpeg::{ExitCode, LeptonError};
use rstest::rstest;

/// handy function to compare two arrays, and print the first mismatch. Useful for debugging.
#[track_caller]
pub fn assert_eq_array<T: PartialEq + std::fmt::Debug>(a: &[T], b: &[T]) {
    use core::panic;

    if a.len() != b.len() {
        for i in 0..std::cmp::min(a.len(), b.len()) {
            assert_eq!(
                a[i],
                b[i],
                "length mismatch {},{} and first mismatch at offset {}",
                a.len(),
                b.len(),
                i
            );
        }
        panic!(
            "length mismatch {} and {}, but common prefix identical",
            a.len(),
            b.len()
        );
    } else {
        for i in 0..a.len() {
            assert_eq!(
                a[i],
                b[i],
                "length identical {}, but first mismatch at offset {}",
                a.len(),
                i
            );
        }
    }
}

/// reads a file from the images directory for testing or benchmarking purposes
pub fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    use std::io::Read;

    let filename = std::path::Path::new(env!("WORKSPACE_ROOT"))
        .join("images")
        .join(filename.to_owned() + ext);
    let mut f = std::fs::File::open(filename).unwrap();

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
        "cathedral_db_non_int",
        "cathedral_db_non_int_rustold",
        "gray2sf",
        "grayscale",
        "hq",
        "half_scan",
        "half_scan_rust53",
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
        "truncbad",          // the lepton format is truncated and invalid
        "eof_and_trailingrst",    // the lepton format has a wrongly set unexpected eof and trailing rst
        "eof_and_trailinghdrdata" // the lepton format has a wrongly set unexpected eof and trailing header data
    )]
    file: &str,
) {
    use lepton_jpeg::DEFAULT_THREAD_POOL;

    println!("decoding {0:?}", file);

    let input = read_file(file, ".lep");
    let expected = read_file(file, ".jpg");

    let mut output = Vec::new();

    decode_lepton(
        &mut Cursor::new(input),
        &mut output,
        &EnabledFeatures::compat_lepton_vector_read(),
        &DEFAULT_THREAD_POOL,
    )
    .unwrap();

    assert_eq_array(&output, &expected);
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

    decode_lepton(
        &mut Cursor::new(input),
        &mut output,
        &features,
        &DEFAULT_THREAD_POOL,
    )
    .unwrap();

    assert_eq_array(&output, &expected);
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
            //"half_scan",
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
        &DEFAULT_THREAD_POOL,
    )
    .unwrap();

    decode_lepton(
        &mut Cursor::new(lepton),
        &mut output,
        &EnabledFeatures::compat_lepton_vector_read(),
        &DEFAULT_THREAD_POOL,
    )
    .unwrap();

    assert_eq_array(&input, &output);
}

/// these files are expected to fail encoding due to unsupported features or roundtrip errors
#[rstest]
fn verify_fail_encode(#[values("half_scan", "narrowrst", "nofsync")] file: &str) {
    let input = read_file(file, ".jpg");

    let result = encode_lepton_verify(
        &input,
        &EnabledFeatures::compat_lepton_vector_write(),
        &DEFAULT_THREAD_POOL,
    );

    assert!(result.is_err(), "encoding was expected to fail");
}

#[test]
fn verify_16bitmath() {
    // verifies that we can decode 16 bit encoded images from the C++ version
    {
        let input = read_file("mathoverflow_16", ".lep");
        let expected = read_file("mathoverflow", ".jpg");

        let mut output = Vec::new();

        let features = EnabledFeatures::compat_lepton_vector_read();

        decode_lepton(
            &mut Cursor::new(input),
            &mut output,
            &features,
            &DEFAULT_THREAD_POOL,
        )
        .unwrap();

        assert_eq_array(&output, &expected);
    }

    // verify that we can decode the one generated by the Rust version
    {
        let input = read_file("mathoverflow_32", ".lep");
        let expected = read_file("mathoverflow", ".jpg");

        let mut output = Vec::new();

        let mut features = EnabledFeatures::compat_lepton_vector_read();
        features.use_16bit_dc_estimate = false;

        decode_lepton(
            &mut Cursor::new(input),
            &mut output,
            &features,
            &DEFAULT_THREAD_POOL,
        )
        .unwrap();

        assert_eq_array(&output, &expected);
    }
}

/// encodes as LEP and codes back to JPG to mostly test the encoder. Can't check against
/// the original LEP file since there's no guarantee they are binary identical (especially the zlib encoded part)
#[rstest]
fn verify_encode_verify(#[values("slrcity")] file: &str) {
    let input = read_file(file, ".jpg");

    encode_lepton_verify(
        &input[..],
        &EnabledFeatures::compat_lepton_vector_write(),
        &DEFAULT_THREAD_POOL,
    )
    .unwrap();
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
        encode_lepton_verify(
            &input[..],
            &EnabledFeatures::compat_lepton_vector_write(),
            &DEFAULT_THREAD_POOL,
        ),
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
            &DEFAULT_THREAD_POOL,
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
            &DEFAULT_THREAD_POOL,
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
            &DEFAULT_THREAD_POOL,
        ),
    );
}

/// tests all previous fuzzing failures to ensure they remain fixed. This requires them to be
/// checked into the repository under the fuzz/artifacts/fuzz_target_1 directory as crash-xxxx files
#[test]
fn test_previous_fuzz_failures() {
    for entry in read_dir(
        Path::new(env!("WORKSPACE_ROOT"))
            .join("fuzz")
            .join("artifacts")
            .join("fuzz_target_1"),
    )
    .unwrap()
    {
        let entry = entry.unwrap();
        let path = entry.path();

        // see if it starts with crash-
        let filename = path.file_name().unwrap().to_str().unwrap();
        if !filename.starts_with("crash-") {
            continue;
        }

        println!(
            "testing fuzz failure reproduction for file {}",
            path.display()
        );

        let data = std::fs::read(path).unwrap();
        test_fuzz_failure(&data);
    }

    /// mirrors what we do for fuzz testing so that we can reproduce failures found by the fuzzer
    /// and ensure that they remain fixed
    fn test_fuzz_failure(data: &[u8]) {
        let mut output = Vec::new();

        let use_16bit = match data.len() % 2 {
            0 => false,
            _ => true,
        };
        let accept_invalid_dht = match (data.len() / 2) % 2 {
            0 => false,
            _ => true,
        };

        // keep the jpeg dimensions small otherwise the fuzzer gets really slow
        let features = EnabledFeatures {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_height: 1024,
            max_jpeg_width: 1024,
            use_16bit_dc_estimate: use_16bit,
            use_16bit_adv_predict: use_16bit,
            accept_invalid_dht: accept_invalid_dht,
            ..EnabledFeatures::compat_lepton_vector_write()
        };

        let r;
        {
            let mut writer = Cursor::new(&mut output);

            r = encode_lepton(
                &mut Cursor::new(&data),
                &mut writer,
                &features,
                &DEFAULT_THREAD_POOL,
            );
        }

        let mut original = Vec::new();

        match r {
            Ok(_) => {
                let _ = decode_lepton(
                    &mut Cursor::new(&output),
                    &mut original,
                    &features,
                    &DEFAULT_THREAD_POOL,
                );
            }
            Err(_) => {}
        }
    }
}
