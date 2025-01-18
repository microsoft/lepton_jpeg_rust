use std::cmp::max;

/// Fast division of |divident| < (1 << precision) by i32 divisor D.
// We need for edge DCT coefs prediction precision of 32 - 13 = 19 bits
// for quantization table values 0 < D < 65536.

/// Usage for n / D:
// let mult: i64 = recip(D);
// let shift = mult & 0xFF;
// let mut res = ((n as i64) * (mult & !0xFF)) >> shift;
// res += if res < 0 { 1 } else { 0 };
/// or variant lower through absolute values.

// We use some excessive precision here, effectively precision for division
// of i32 by (q << 13) is 32 - 13 = 19 bit, but we use 20 instead
const DIV_PRECISION: i32 = 20;

// Modified from https://web.archive.org/web/20160927064638/https://raw.githubusercontent.com/ridiculousfish/libdivide/master/divide_by_constants_codegen_reference.c
// Comments from original author ridiculousfish
// See also https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
/*
  Reference implementations of computing and using the "magic number" approach to dividing
  by constants, including codegen instructions. The unsigned division incorporates the
  "round down" optimization per ridiculous_fish.

  This is free and unencumbered software. Any copyright is dedicated to the Public Domain.
*/
pub fn recip(d: i32) -> i32 {
    if d == 0 || d == 1 {
        let mult = (1 << DIV_PRECISION) + 1;
        let shift = DIV_PRECISION + 8;
        return (mult << 8) | shift;
    }

    // Absolute value of D
    let abs_d = d.unsigned_abs() as u64;

    // The initial power of 2 is one less than the first one that can possibly work
    // "two31" in Warren
    let mut exponent = DIV_PRECISION + 1; //precision - 1;
    let initial_power_of_2 = 1u64 << exponent;

    // Compute the absolute value of our "test numerator,"
    // which is the largest dividend whose remainder with d is d-1.
    // This is called anc in Warren.
    let tmp: u64 = (1u64 << DIV_PRECISION) + if d < 0 { 1 } else { 0 };
    let abs_test_numer: u64 = max(tmp - 1 - tmp % abs_d, abs_d - 1);

    // Initialize our quotients and remainders (q1, r1, q2, r2 in Warren)
    let mut quotient1: u64 = initial_power_of_2 / abs_test_numer;
    let mut remainder1: u64 = initial_power_of_2 % abs_test_numer;
    let mut quotient2: u64 = initial_power_of_2 / abs_d;
    let mut remainder2: u64 = initial_power_of_2 % abs_d;
    let mut delta: u64;

    // Begin our loop
    loop {
        // Update the exponent
        exponent += 1;

        // Update quotient1 and remainder1
        quotient1 *= 2;
        remainder1 *= 2;
        if remainder1 >= abs_test_numer {
            quotient1 += 1;
            remainder1 -= abs_test_numer;
        }

        // Update quotient2 and remainder2
        quotient2 *= 2;
        remainder2 *= 2;
        if remainder2 >= abs_d {
            quotient2 += 1;
            remainder2 -= abs_d;
        }

        // Keep going as long as (2**exponent) / abs_d <= delta
        delta = abs_d - remainder2;
        // !(quotient1 < delta || (quotient1 == delta && remainder1 == 0))
        if quotient1 >= delta && (quotient1 != delta || remainder1 != 0) {
            break;
        }
    }

    let mult = (quotient2 + 1) as i32;
    debug_assert!((mult > 0) && (mult < (1 << 23)));
    let shift = exponent + 8;

    return (mult << 8) | shift;
}

/// Precalculated arrays of reciprocals
//let recip_array: [i32; 1 << 16] = core::array::from_fn(|i| recip(i));

#[inline(always)]
pub fn div(dividend: i32, divisor_recip: i32) -> i32 {
    let abs_d = dividend.unsigned_abs() as i64;
    let shift = divisor_recip & 0xFF;
    let mult = (divisor_recip & !0xFF) as i64;
    let res = (abs_d * mult >> shift) as i32;
    if dividend < 0 { -res } else { res }
}

// pub fn div(dividend: i32x8, divisor_recip: i32x8)
// {
//     let shift = preliminary_shift + (divisor_recip & 0xFF);
//     let res = (dividend as i64) * ((divisor_recip & !0xFF) as i64) >> shift;
//     return (res + if res < 0 { 1 } else { 0 }) as i32;
// }
