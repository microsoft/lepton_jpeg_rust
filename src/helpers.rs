/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use crate::lepton_error::{ExitCode, LeptonError};

macro_rules! here {
    () => {
        concat!("at ", file!(), " line ", line!())
    };
}

pub(crate) use here;

#[inline(always)]
pub const fn u16_bit_length(v: u16) -> u8 {
    return 16 - v.leading_zeros() as u8;
}

#[inline(always)]
pub const fn u32_bit_length(v: u32) -> u8 {
    return 32 - v.leading_zeros() as u8;
}

#[cold]
pub fn err_exit_code<T>(_error_code: ExitCode, message: &str) -> anyhow::Result<T> {
    return Err(anyhow::Error::new(LeptonError {
        exit_code: _error_code,
        message: message.to_string(),
    }));
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
