/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use bytemuck::cast;
use wide::{i16x8, i32x8};

use crate::structs::block_based_image::AlignedBlock;

const _W1: i32 = 2841; // 2048*sqrt(2)*cos(1*pi/16)
const _W2: i32 = 2676; // 2048*sqrt(2)*cos(2*pi/16)
const _W3: i32 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const _W5: i32 = 1609; // 2048*sqrt(2)*cos(5*pi/16)
const _W6: i32 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const _W7: i32 = 565; // 2048*sqrt(2)*cos(7*pi/16)

const W3: i32 = 2408; // 2048*sqrt(2)*cos(3*pi/16)
const W6: i32 = 1108; // 2048*sqrt(2)*cos(6*pi/16)
const W7: i32 = 565; // 2048*sqrt(2)*cos(7*pi/16)

const W1PW7: i32 = _W1 + _W7;
const W1MW7: i32 = _W1 - _W7;
const W2PW6: i32 = _W2 + _W6;
const W2MW6: i32 = _W2 - _W6;
const W3PW5: i32 = _W3 + _W5;
const W3MW5: i32 = _W3 - _W5;

const R2: i32 = 181; // 256/sqrt(2)

#[inline(always)]
pub fn run_idct(block: &[i32x8; 8]) -> AlignedBlock {
    let t = *block;

    let mut xv0 = (t[0] << 11) + 128;
    let mut xv1 = t[1];
    let mut xv2 = t[2];
    let mut xv3 = t[3];
    let mut xv4 = t[4] << 11;
    let mut xv5 = t[5];
    let mut xv6 = t[6];
    let mut xv7 = t[7];

    // Stage 1.
    let mut xv8 = _W7 * (xv1 + xv7);
    xv1 = xv8 + (W1MW7 * xv1);
    xv7 = xv8 - (W1PW7 * xv7);
    xv8 = _W3 * (xv5 + xv3);
    xv5 = xv8 - (W3MW5 * xv5);
    xv3 = xv8 - (W3PW5 * xv3);

    // Stage 2.
    xv8 = xv0 + xv4;
    xv0 -= xv4;
    xv4 = W6 * (xv2 + xv6);
    xv6 = xv4 - (W2PW6 * xv6);
    xv2 = xv4 + (W2MW6 * xv2);
    xv4 = xv1 + xv5;
    xv1 -= xv5;
    xv5 = xv7 + xv3;
    xv7 -= xv3;

    // Stage 3.
    xv3 = xv8 + xv2;
    xv8 -= xv2;
    xv2 = xv0 + xv6;
    xv0 -= xv6;
    xv6 = ((R2 * (xv1 + xv7)) + 128) >> 8;
    xv1 = ((R2 * (xv1 - xv7)) + 128) >> 8;

    // Stage 4.
    let row = [
        (xv3 + xv4) >> 8,
        (xv2 + xv6) >> 8,
        (xv0 + xv1) >> 8,
        (xv8 + xv5) >> 8,
        (xv8 - xv5) >> 8,
        (xv0 - xv1) >> 8,
        (xv2 - xv6) >> 8,
        (xv3 - xv4) >> 8,
    ];

    // transpose and now do vertical
    let [mut yv0, mut yv1, mut yv2, mut yv3, mut yv4, mut yv5, mut yv6, mut yv7] =
        i32x8::transpose(row);

    yv0 = (yv0 << 8) + 8192;
    yv4 = yv4 << 8;

    // Stage 1.
    let mut yv8 = (W7 * (yv1 + yv7)) + 4;
    yv1 = (yv8 + (W1MW7 * yv1)) >> 3;
    yv7 = (yv8 - (W1PW7 * yv7)) >> 3;
    yv8 = (W3 * (yv5 + yv3)) + 4;
    yv5 = (yv8 - (W3MW5 * yv5)) >> 3;
    yv3 = (yv8 - (W3PW5 * yv3)) >> 3;

    // Stage 2.
    yv8 = yv0 + yv4;
    yv0 -= yv4;
    yv4 = ((W6) * (yv2 + yv6)) + 4;
    yv6 = (yv4 - (W2PW6 * yv6)) >> 3;
    yv2 = (yv4 + (W2MW6 * yv2)) >> 3;
    yv4 = yv1 + yv5;
    yv1 -= yv5;
    yv5 = yv7 + yv3;
    yv7 -= yv3;

    // Stage 3.
    yv3 = yv8 + yv2;
    yv8 -= yv2;
    yv2 = yv0 + yv6;
    yv0 -= yv6;
    yv6 = ((R2 * (yv1 + yv7)) + 128) >> 8;
    yv1 = ((R2 * (yv1 - yv7)) + 128) >> 8;

    // Stage 4.
    AlignedBlock::new(cast([
        i16x8::from_i32x8_truncate((yv3 + yv4) >> 11),
        i16x8::from_i32x8_truncate((yv2 + yv6) >> 11),
        i16x8::from_i32x8_truncate((yv0 + yv1) >> 11),
        i16x8::from_i32x8_truncate((yv8 + yv5) >> 11),
        i16x8::from_i32x8_truncate((yv8 - yv5) >> 11),
        i16x8::from_i32x8_truncate((yv0 - yv1) >> 11),
        i16x8::from_i32x8_truncate((yv2 - yv6) >> 11),
        i16x8::from_i32x8_truncate((yv3 - yv4) >> 11),
    ]))
}

#[cfg(test)]
use bytemuck::cast_ref;

#[cfg(test)]
#[inline(always)]
fn get_q(offset: usize, q_transposed: &AlignedBlock) -> i32x8 {
    use wide::u16x8;

    let rows: &[u16x8; 8] = cast_ref(q_transposed.get_block());
    i32x8::from_u16x8(rows[offset])
}

#[cfg(test)]
#[inline(always)]
fn get_c(offset: usize, q_transposed: &AlignedBlock) -> i32x8 {
    let rows: &[i16x8; 8] = cast_ref(q_transposed.get_block());
    i32x8::from_i16x8(rows[offset])
}

#[cfg(test)]
fn test_idct(test_data: &AlignedBlock, test_q: &[u16; 64]) {
    use std::num::Wrapping;

    fn mul(a: i16, b: u16) -> Wrapping<i32> {
        return Wrapping(a as i32) * Wrapping(b as i32);
    }

    pub fn run_idct_old(
        block: &AlignedBlock,
        q: &[u16; 64],
        outp: &mut [i16; 64],
        ignore_dc: bool,
    ) {
        let mut intermed = [Wrapping(0i32); 64];

        // Horizontal 1-D IDCT.
        for y in 0..8 {
            let y8: usize = y * 8;

            let mut x0 = if ignore_dc && y == 0 {
                Wrapping(0)
            } else {
                mul(block.get_coefficient(y8 + 0), q[y8 + 0]) << 11
            } + Wrapping(128);
            let mut x1 = mul(block.get_coefficient(y8 + 4), q[y8 + 4]) << 11;
            let mut x2 = mul(block.get_coefficient(y8 + 6), q[y8 + 6]);
            let mut x3 = mul(block.get_coefficient(y8 + 2), q[y8 + 2]);
            let mut x4 = mul(block.get_coefficient(y8 + 1), q[y8 + 1]);
            let mut x5 = mul(block.get_coefficient(y8 + 7), q[y8 + 7]);
            let mut x6 = mul(block.get_coefficient(y8 + 5), q[y8 + 5]);
            let mut x7 = mul(block.get_coefficient(y8 + 3), q[y8 + 3]);

            // If all the AC components are zero, then the IDCT is trivial.
            if x1 == Wrapping(0)
                && x2 == Wrapping(0)
                && x3 == Wrapping(0)
                && x4 == Wrapping(0)
                && x5 == Wrapping(0)
                && x6 == Wrapping(0)
                && x7 == Wrapping(0)
            {
                let dc = (x0 - Wrapping(128)) >> 8;
                intermed[y8 + 0] = dc;
                intermed[y8 + 1] = dc;
                intermed[y8 + 2] = dc;
                intermed[y8 + 3] = dc;
                intermed[y8 + 4] = dc;
                intermed[y8 + 5] = dc;
                intermed[y8 + 6] = dc;
                intermed[y8 + 7] = dc;
                continue;
            }

            // Prescale.

            // Stage 1.
            let mut x8 = Wrapping(W7) * (x4 + x5);
            x4 = x8 + (Wrapping(W1MW7) * x4);
            x5 = x8 - (Wrapping(W1PW7) * x5);
            x8 = Wrapping(W3) * (x6 + x7);
            x6 = x8 - (Wrapping(W3MW5) * x6);
            x7 = x8 - (Wrapping(W3PW5) * x7);

            // Stage 2.
            x8 = x0 + x1;
            x0 -= x1;
            x1 = Wrapping(W6) * (x3 + x2);
            x2 = x1 - (Wrapping(W2PW6) * x2);
            x3 = x1 + (Wrapping(W2MW6) * x3);
            x1 = x4 + x6;
            x4 -= x6;
            x6 = x5 + x7;
            x5 -= x7;

            // Stage 3.
            x7 = x8 + x3;
            x8 -= x3;
            x3 = x0 + x2;
            x0 -= x2;
            x2 = ((Wrapping(R2) * (x4 + x5)) + Wrapping(128)) >> 8;
            x4 = ((Wrapping(R2) * (x4 - x5)) + Wrapping(128)) >> 8;

            // Stage 4.
            intermed[y8 + 0] = (x7 + x1) >> 8;
            intermed[y8 + 1] = (x3 + x2) >> 8;
            intermed[y8 + 2] = (x0 + x4) >> 8;
            intermed[y8 + 3] = (x8 + x6) >> 8;
            intermed[y8 + 4] = (x8 - x6) >> 8;
            intermed[y8 + 5] = (x0 - x4) >> 8;
            intermed[y8 + 6] = (x3 - x2) >> 8;
            intermed[y8 + 7] = (x7 - x1) >> 8;
        }

        // Vertical 1-D IDCT.
        for x in 0..8 {
            // Similar to the horizontal 1-D IDCT case, if all the AC components are zero, then the IDCT is trivial.
            // However, after performing the horizontal 1-D IDCT, there are typically non-zero AC components, so
            // we do not bother to check for the all-zero case.

            // Prescale.
            let mut y0 = (intermed[(8 * 0) + x] << 8) + Wrapping(8192);
            let mut y1 = intermed[(8 * 4) + x] << 8;
            let mut y2 = intermed[(8 * 6) + x];
            let mut y3 = intermed[(8 * 2) + x];
            let mut y4 = intermed[(8 * 1) + x];
            let mut y5 = intermed[(8 * 7) + x];
            let mut y6 = intermed[(8 * 5) + x];
            let mut y7 = intermed[(8 * 3) + x];

            // Stage 1.
            let mut y8 = (Wrapping(W7) * (y4 + y5)) + Wrapping(4);
            y4 = (y8 + (Wrapping(W1MW7) * y4)) >> 3;
            y5 = (y8 - (Wrapping(W1PW7) * y5)) >> 3;
            y8 = (Wrapping(W3) * (y6 + y7)) + Wrapping(4);
            y6 = (y8 - (Wrapping(W3MW5) * y6)) >> 3;
            y7 = (y8 - (Wrapping(W3PW5) * y7)) >> 3;

            // Stage 2.
            y8 = y0 + y1;
            y0 -= y1;
            y1 = (Wrapping(W6) * (y3 + y2)) + Wrapping(4);
            y2 = (y1 - (Wrapping(W2PW6) * y2)) >> 3;
            y3 = (y1 + (Wrapping(W2MW6) * y3)) >> 3;
            y1 = y4 + y6;
            y4 -= y6;
            y6 = y5 + y7;
            y5 -= y7;

            // Stage 3.
            y7 = y8 + y3;
            y8 -= y3;
            y3 = y0 + y2;
            y0 -= y2;
            y2 = ((Wrapping(R2) * (y4 + y5)) + Wrapping(128)) >> 8;
            y4 = ((Wrapping(R2) * (y4 - y5)) + Wrapping(128)) >> 8;

            // Stage 4.
            outp[(8 * 0) + x] = ((y7 + y1) >> 11).0 as i16;
            outp[(8 * 1) + x] = ((y3 + y2) >> 11).0 as i16;
            outp[(8 * 2) + x] = ((y0 + y4) >> 11).0 as i16;
            outp[(8 * 3) + x] = ((y8 + y6) >> 11).0 as i16;
            outp[(8 * 4) + x] = ((y8 - y6) >> 11).0 as i16;
            outp[(8 * 5) + x] = ((y0 - y4) >> 11).0 as i16;
            outp[(8 * 6) + x] = ((y3 - y2) >> 11).0 as i16;
            outp[(8 * 7) + x] = ((y7 - y1) >> 11).0 as i16;
        }
    }

    let q = AlignedBlock::new(cast(*test_q));
    let data_tr = test_data.transpose();
    let q_tr = q.transpose();

    let mut raster: [i32x8; 8] = [0.into(); 8]; // transposed
    for col in 0..8 {
        raster[col] = get_c(col, &data_tr) * get_q(col, &q_tr);
    }

    let outp = run_idct(&raster);

    let mut outp2 = [0; 64];
    run_idct_old(test_data, test_q, &mut outp2, false);

    assert_eq!(*outp.get_block(), outp2);
}

/// test with a simple block to catch obvious mistakes
#[test]
pub fn test_idct_with_simple_block() {
    let mut test_data = AlignedBlock::default();
    let mut test_q = [1u16; 64];

    test_q[0] = 2;
    test_data.set_coefficient(0, 1000);
    test_data.set_coefficient(1, -1000);

    test_idct(&test_data, &test_q);
}

/// test with random permutations to verify that the current implementation matches the legacy
/// implemenation from the original scalar C++ code
#[test]
pub fn test_idct_with_random_blocks() {
    use rand::Rng;

    let mut rng = crate::helpers::get_rand_from_seed([0u8; 32]);
    let mut test_data = AlignedBlock::default();
    let mut test_q = [0u16; 64];

    for _ in 0..16 {
        for i in 0..64 {
            test_data.get_block_mut()[i] = rng.gen_range(i16::MIN..=i16::MAX);
            test_q[i] = rng.gen_range(0..=u8::MAX as u16);
        }

        test_idct(&test_data, &test_q);
    }
}
