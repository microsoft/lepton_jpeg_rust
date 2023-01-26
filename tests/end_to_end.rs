/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

 use core::result::Result;
use std::io::Cursor;

use std::fs::File;
use std::io::Read;

use lepton_jpeg::{
    decode_lepton, encode_lepton,
    lepton_error::{ExitCode, LeptonError},
};

use rstest::rstest;

fn read_file(filename: &str, ext: &str) -> Vec<u8> {
    let filename = env!("CARGO_MANIFEST_DIR").to_owned() + "\\images\\" + filename + ext;
    println!("reading {0}", filename);
    let mut f = File::open(filename).unwrap();

    let mut content = Vec::new();
    f.read_to_end(&mut content).unwrap();

    content
}

// verifies that the decode will accept existing Lepton files and generate
// exactly the same jpeg from them. Used to detect unexpected divergences in coding format.
#[rstest]
fn verify_decode(
    #[values(
        "android",
        "androidcrop",
        "androidcropoptions",
        "androidprogressive",
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
        "narrowrst",
        "nofsync",
        "slrcity",
        "slrhills",
        "slrindoor",
        "tiny",
        "trailingrst",
        "trailingrst2",
        "trunc"
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

// encodes as LEP and codes back to JPG to mostly test the encoder. Can't check against
// the original LEP file since there's no guarantee they are binary identical (especially the zlib encoded part)
#[rstest]
fn verify_encode(
    #[values(
            "android",
            "androidcrop",
            "androidcropoptions",
            "androidprogressive",
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
        false,
    )
    .unwrap();
    decode_lepton(&mut Cursor::new(lepton), &mut output, 8).unwrap();

    assert!(input[..] == output[..]);
}

fn assert_exception(expected_error: ExitCode, result: Result<(), LeptonError>) {
    match result {
        Ok(()) => panic!("failure was expected"),
        Err(e) => {
            assert_eq!(expected_error, e.exit_code, "unexpected error {0:?}", e);
        }
    }
}

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
            true,
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
            false,
        ),
    );
}
