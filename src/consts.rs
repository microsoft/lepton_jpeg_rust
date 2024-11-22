/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

#[derive(PartialEq, Debug)]
pub enum JPegDecodeStatus {
    DecodeInProgress,
    RestartIntervalExpired,
    ScanCompleted,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum JPegType {
    Unknown,
    Sequential,
    Progressive,
}

pub const COLOR_CHANNEL_NUM_BLOCK_TYPES: usize = 3;

pub const RASTER_TO_ZIGZAG: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

// pub const ZIGZAG_TO_RASTER: [u8; 64] = [
//     0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
//     13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
//     52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
// ];

pub const ZIGZAG_TO_TRANSPOSED: [u8; 64] = [
    0, 8, 1, 2, 9, 16, 24, 17, 10, 3, 4, 11, 18, 25, 32, 40, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34,
    41, 48, 56, 49, 42, 35, 28, 21, 14, 7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31,
    38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,
];

// pub const UNZIGZAG_49: [u8; 49] = [
//     9, 10, 17, 25, 18, 11, 12, 19, 26, 33, 41, 34, 27, 20, 13, 14, 21, 28, 35, 42, 49, 57, 50, 43,
//     36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62,
//     63,
// ];

pub const UNZIGZAG_49_TR: [u8; 49] = [
    9, 17, 10, 11, 18, 25, 33, 26, 19, 12, 13, 20, 27, 34, 41, 49, 42, 35, 28, 21, 14, 15, 22, 29,
    36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55,
    63,
];

// precalculated int base values for 8x8 IDCT scaled by 8192
// DC coef is zeroed intentionally
pub const ICOS_BASED_8192_SCALED: [i32; 8] = [0, 11363, 10703, 9633, 8192, 6436, 4433, 2260];

pub const ICOS_BASED_8192_SCALED_PM: [i32; 8] =
    [8192, -11363, 10703, -9633, 8192, -6436, 4433, -2260];

pub const FREQ_MAX: [u16; 14] = [
    931, 985, 968, 1020, 968, 1020, 1020, 932, 985, 967, 1020, 969, 1020, 1020,
];

// used to get prediction branches basing on nonzero-number predictor `num_non_zeros_context`
pub const NON_ZERO_TO_BIN: [u8; 26] = [
    0, 1, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
];

// used to get prediction branches basing on current `num_non_zeros_left_7x7`, 0th element is not used
pub const NON_ZERO_TO_BIN_7X7: [u8; 50] = [
    0, 0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
];

//pub const MAX_FILE_SIZE_BYTES : i32 = 128 * 1024 * 1024;
pub const MAX_THREADS: usize = 8;

pub const RESIDUAL_NOISE_FLOOR: usize = 7;

pub const LEPTON_VERSION: u8 = 1; // Lepton version, same as used by Lepton C++ since we support the same format

//pub const LogMaxNumerator : i32 = 18;
//pub const DefaultEncodingThreads : usize = 8;
pub const SMALL_FILE_BYTES_PER_ENCDOING_THREAD: usize = 125000;
//pub const TailGarbageBufferLength : i32 = 1024;
pub const MAX_THREADS_SUPPORTED_BY_LEPTON_FORMAT: usize = 16; // Number of threads minus 1 should fit in 4 bits

//pub const SingleFFByte : [u8;1] = [ 0xFF ];
pub const EOI: [u8; 2] = [0xFF, crate::jpeg_code::EOI]; // EOI segment
pub const SOI: [u8; 2] = [0xFF, crate::jpeg_code::SOI]; // SOI segment
pub const LEPTON_FILE_HEADER: [u8; 2] = [0xcf, 0x84]; // the tau symbol for a tau lepton in utf-8
pub const LEPTON_HEADER_BASELINE_JPEG_TYPE: [u8; 1] = [b'Z'];
pub const LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE: [u8; 1] = [b'X'];
pub const LEPTON_HEADER_MARKER: [u8; 3] = *b"HDR";
pub const LEPTON_HEADER_PAD_MARKER: [u8; 3] = *b"P0D";
pub const LEPTON_HEADER_JPG_RESTARTS_MARKER: [u8; 3] = *b"CRS";
pub const LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER: [u8; 3] = *b"FRS";
pub const LEPTON_HEADER_LUMA_SPLIT_MARKER: [u8; 2] = *b"HH";
pub const LEPTON_HEADER_EARLY_EOF_MARKER: [u8; 3] = *b"EEE";
pub const LEPTON_HEADER_PREFIX_GARBAGE_MARKER: [u8; 3] = *b"PGR";
pub const LEPTON_HEADER_GARBAGE_MARKER: [u8; 3] = *b"GRB";
pub const LEPTON_HEADER_COMPLETION_MARKER: [u8; 3] = *b"CMP";
//pub const ChunkedLeptonHeaderSizeMarker : [u8;3] = *b"SIZ" ;
//pub const ChunkedLeptonHeaderJpgHeaderDataRangeMarker : [u8;3] = *b"JHR";
