/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::lepton_error::{ExitCode, LeptonError};

/// Helper function to catch panics and convert them into the appropriate LeptonError
pub fn catch_unwind_result<R>(
    f: impl FnOnce() -> Result<R, LeptonError>,
) -> Result<R, LeptonError> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r.map_err(|e| e.into()),
        Err(err) => {
            if let Some(message) = err.downcast_ref::<&str>() {
                Err(LeptonError::new(ExitCode::AssertionFailure, *message))
            } else if let Some(message) = err.downcast_ref::<String>() {
                Err(LeptonError::new(
                    ExitCode::AssertionFailure,
                    message.as_str(),
                ))
            } else {
                Err(LeptonError::new(
                    ExitCode::AssertionFailure,
                    "unknown panic",
                ))
            }
        }
    }
}

/// copies a string into a limited length zero terminated utf8 buffer
pub fn copy_cstring_utf8_to_buffer(str: &str, target_error_string: &mut [u8]) {
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

#[inline(always)]
pub const fn u16_bit_length(v: u16) -> u8 {
    return 16 - v.leading_zeros() as u8;
}

#[inline(always)]
pub const fn u32_bit_length(v: u32) -> u8 {
    return 32 - v.leading_zeros() as u8;
}

pub fn buffer_prefix_matches_marker<const BS: usize, const MS: usize>(
    buffer: [u8; BS],
    marker: [u8; MS],
) -> bool {
    // Helper method, skipping checks of parameters nulls/lengths
    for i in 0..marker.len() {
        if buffer[i] != marker[i] {
            return false;
        }
    }

    return true;
}

#[inline(always)]
pub const fn devli(s: u8, value: u16) -> i16 {
    if s == 0 {
        value as i16
    } else if value < (1 << (s as u16 - 1)) {
        value as i16 + (-1 << s as i16) + 1
    } else {
        value as i16
    }
}

#[inline(always)]
pub const fn b_short(v1: u8, v2: u8) -> u16 {
    ((v1 as u16) << 8) + v2 as u16
}

#[inline(always)]
pub const fn rbits(c: u8, n: usize) -> u8 {
    return c & (0xFF >> (8 - n));
}

#[inline(always)]
pub const fn lbits(c: u8, n: usize) -> u8 {
    return c >> (8 - n);
}

#[inline(always)]
pub const fn bitn(c: u16, n: u16) -> u8 {
    return ((c >> n) & 0x1) as u8;
}

#[inline(always)]
pub fn calc_sign_index(val: i16) -> usize {
    if val == 0 {
        0
    } else {
        if val > 0 {
            1
        } else {
            2
        }
    }
}

#[cfg(test)]
pub fn get_rand_from_seed(seed: [u8; 32]) -> rand_chacha::ChaCha12Rng {
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    ChaCha12Rng::from_seed(seed)
}

/*
better way to update aritmetic encoding without using special division

const fn k16bit_length(v : u32) -> u32
{
    const LEN_TABLE256 : [i8;256] =
    [
            0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    ];

    return if (v & 0xff00) != 0 { 8 + LEN_TABLE256[(v >> 8) as usize] } else { LEN_TABLE256[v as usize] } as u32;
}

const LOG_MAX_NUMERATOR : i32= 18;

const fn calc_divisors() -> [u32;1026]
{
    let mut intermed = [0u32;1026];

    let mut d : u32  = 1;

    while d < 1026
    {
        intermed[d as usize] = ((((1 << k16bit_length(d)) - d) << LOG_MAX_NUMERATOR) / d) + 1;
        d += 1;
    }

    return intermed;
}

const DIVISORS : [u32;1026] = calc_divisors();

#[inline(always)]
pub fn fast_divide18bit_by_10bit(num : u32, denom : u16) -> u32
{
    //debug_assert_eq!(LOG2_LENGTHS[denom as usize], (16 - denom.leading_zeros() - 1) as u8, "log2{0}", denom);

    let tmp = ((DIVISORS[denom as usize] as u64 * num as u64) >> LOG_MAX_NUMERATOR) as u32;
    let r = (tmp + ((num - tmp) >> 1)) >> (16 - denom.leading_zeros() - 1);

    debug_assert_eq!(r, num/(denom as u32));
    return r;
}

*/
