/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::panic::{AssertUnwindSafe, catch_unwind};

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
                Err(LeptonError::new(ExitCode::AssertionFailure, message))
            } else {
                Err(LeptonError::new(
                    ExitCode::AssertionFailure,
                    "unknown panic",
                ))
            }
        }
    }
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

/// returns true if the 64 bit value contains an 0xff byte.
/// Uses fancy bit manipulation to avoid branches.
#[inline(always)]
pub fn has_ff(v: u64) -> bool {
    (v & 0x8080808080808080 & !v.wrapping_add(0x0101010101010101)) != 0
}

#[inline(always)]
pub const fn devli(s: u8, value: u16) -> i16 {
    let shifted = 1 << s;

    if value & (shifted >> 1) != 0 {
        value as i16
    } else {
        value.wrapping_add(2).wrapping_add(!shifted) as i16
    }
}

/// check to make sure the behavior hasn't changed even with the optimization
#[test]
fn devli_test() {
    for s in 0u8..15 {
        for value in 0..(1 << s) {
            assert_eq!(
                devli(s, value),
                if s == 0 {
                    value as i16
                } else if value < (1 << (s as u16 - 1)) {
                    value as i16 + (-1 << s as i16) + 1
                } else {
                    value as i16
                }
            );
        }
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
        if val > 0 { 1 } else { 2 }
    }
}

/// This checks to see if a vector can fit additional elements without growing,
/// but does it in such a way that the optimizer understands that a subsequent
/// push or extend will not need to grow the vector.
#[inline(always)]
pub fn needs_to_grow<T>(v: &Vec<T>, additional: usize) -> bool {
    additional > v.capacity().wrapping_sub(v.len())
}

#[cfg(test)]
pub fn get_rand_from_seed(seed: [u8; 32]) -> rand_chacha::ChaCha12Rng {
    use rand_chacha::ChaCha12Rng;
    use rand_chacha::rand_core::SeedableRng;

    ChaCha12Rng::from_seed(seed)
}

/// reads a file from the images directory for testing or benchmarking purposes
#[cfg(any(test, feature = "micro_benchmark"))]
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
