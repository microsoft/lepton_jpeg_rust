/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use super::block_based_image::AlignedBlock;

use wide::i32x8;

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
fn get_raster<const IGNORE_DC: bool>(offset: usize, stride: usize, block: &AlignedBlock) -> i32x8 {
    return i32x8::new([
        block.get_coefficient_raster(7 * stride + offset) as i32,
        block.get_coefficient_raster(6 * stride + offset) as i32,
        block.get_coefficient_raster(5 * stride + offset) as i32,
        block.get_coefficient_raster(4 * stride + offset) as i32,
        block.get_coefficient_raster(3 * stride + offset) as i32,
        block.get_coefficient_raster(2 * stride + offset) as i32,
        block.get_coefficient_raster(1 * stride + offset) as i32,
        if IGNORE_DC && offset == 0 {
            0
        } else {
            block.get_coefficient_raster(offset) as i32
        },
    ]);
}

#[inline(always)]
pub fn get_q(offset: usize, stride: usize, q: &[u16; 64]) -> i32x8 {
    return i32x8::new([
        q[7 * stride + offset] as i32,
        q[6 * stride + offset] as i32,
        q[5 * stride + offset] as i32,
        q[4 * stride + offset] as i32,
        q[3 * stride + offset] as i32,
        q[2 * stride + offset] as i32,
        q[1 * stride + offset] as i32,
        q[offset] as i32,
    ]);
}

#[inline(always)]
fn copy_to_output(row: i32x8, offset: usize, outp: &mut [i16; 64]) {
    outp[offset] = row.as_array_ref()[0] as i16;
    outp[offset + 1] = row.as_array_ref()[1] as i16;
    outp[offset + 2] = row.as_array_ref()[2] as i16;
    outp[offset + 3] = row.as_array_ref()[3] as i16;
    outp[offset + 4] = row.as_array_ref()[4] as i16;
    outp[offset + 5] = row.as_array_ref()[5] as i16;
    outp[offset + 6] = row.as_array_ref()[6] as i16;
    outp[offset + 7] = row.as_array_ref()[7] as i16;
}

#[inline(never)]
pub fn run_idct<const IGNORE_DC: bool>(block: &AlignedBlock, q: &[u16; 64], outp: &mut [i16; 64]) {
    // horizontal
    let mut xv0 = get_raster::<IGNORE_DC>(0, 8, block);
    let mut xv1 = get_raster::<IGNORE_DC>(1, 8, block);
    let mut xv2 = get_raster::<IGNORE_DC>(2, 8, block);
    let mut xv3 = get_raster::<IGNORE_DC>(3, 8, block);
    let mut xv4 = get_raster::<IGNORE_DC>(4, 8, block);
    let mut xv5 = get_raster::<IGNORE_DC>(5, 8, block);
    let mut xv6 = get_raster::<IGNORE_DC>(6, 8, block);
    let mut xv7 = get_raster::<IGNORE_DC>(7, 8, block);

    xv0 = ((xv0 * get_q(0, 8, q)) << 11) + 128;
    xv1 *= get_q(1, 8, q);
    xv2 *= get_q(2, 8, q);
    xv3 *= get_q(3, 8, q);
    xv4 = (xv4 * get_q(4, 8, q)) << 11;
    xv5 *= get_q(5, 8, q);
    xv6 *= get_q(6, 8, q);
    xv7 *= get_q(7, 8, q);

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

    yv7 = (yv7 << 8) + 8192;
    yv3 = yv3 << 8;

    // Stage 1.
    let mut yv8 = (W7 * (yv6 + yv0)) + 4;
    yv6 = (yv8 + (W1MW7 * yv6)) >> 3;
    yv0 = (yv8 - (W1PW7 * yv0)) >> 3;
    yv8 = (W3 * (yv2 + yv4)) + 4;
    yv2 = (yv8 - (W3MW5 * yv2)) >> 3;
    yv4 = (yv8 - (W3PW5 * yv4)) >> 3;

    // Stage 2.
    yv8 = yv7 + yv3;
    yv7 -= yv3;
    yv3 = ((W6) * (yv5 + yv1)) + 4;
    yv1 = (yv3 - (W2PW6 * yv1)) >> 3;
    yv5 = (yv3 + (W2MW6 * yv5)) >> 3;
    yv3 = yv6 + yv2;
    yv6 -= yv2;
    yv2 = yv0 + yv4;
    yv0 -= yv4;

    // Stage 3.
    yv4 = yv8 + yv5;
    yv8 -= yv5;
    yv5 = yv7 + yv1;
    yv7 -= yv1;
    yv1 = ((R2 * (yv6 + yv0)) + 128) >> 8;
    yv6 = ((R2 * (yv6 - yv0)) + 128) >> 8;

    // Stage 4.
    copy_to_output((yv4 + yv3) >> 11, 0, outp);
    copy_to_output((yv5 + yv1) >> 11, 8, outp);
    copy_to_output((yv7 + yv6) >> 11, 2 * 8, outp);
    copy_to_output((yv8 + yv2) >> 11, 3 * 8, outp);
    copy_to_output((yv8 - yv2) >> 11, 4 * 8, outp);
    copy_to_output((yv7 - yv6) >> 11, 5 * 8, outp);
    copy_to_output((yv5 - yv1) >> 11, 6 * 8, outp);
    copy_to_output((yv4 - yv3) >> 11, 7 * 8, outp);
}

/// test with random permutations to verify that the current implementation matches the legacy
/// implemenation from the original scalar C++ code
#[test]
pub fn test_idct_with_existing_behavior() {
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
                mul(block.get_coefficient_raster(y8 + 0), q[y8 + 0]) << 11
            } + Wrapping(128);
            let mut x1 = mul(block.get_coefficient_raster(y8 + 4), q[y8 + 4]) << 11;
            let mut x2 = mul(block.get_coefficient_raster(y8 + 6), q[y8 + 6]);
            let mut x3 = mul(block.get_coefficient_raster(y8 + 2), q[y8 + 2]);
            let mut x4 = mul(block.get_coefficient_raster(y8 + 1), q[y8 + 1]);
            let mut x5 = mul(block.get_coefficient_raster(y8 + 7), q[y8 + 7]);
            let mut x6 = mul(block.get_coefficient_raster(y8 + 5), q[y8 + 5]);
            let mut x7 = mul(block.get_coefficient_raster(y8 + 3), q[y8 + 3]);

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

    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut rng = StdRng::from_seed([0u8; 32]);
    let mut test_data = AlignedBlock::default();
    let mut test_q = [0u16; 64];

    for _ in 0..16 {
        for i in 0..64 {
            test_data.get_block_mut()[i] = rng.gen_range(i16::MIN..=i16::MAX);
            test_q[i] = rng.gen_range(0..=u16::MAX);
        }

        {
            let mut outp = [0; 64];
            run_idct::<true>(&test_data, &test_q, &mut outp);

            let mut outp2 = [0; 64];
            run_idct_old(&test_data, &test_q, &mut outp2, true);

            assert_eq!(outp, outp2);
        }

        {
            let mut outp = [0; 64];
            run_idct::<false>(&test_data, &test_q, &mut outp);

            let mut outp2 = [0; 64];
            run_idct_old(&test_data, &test_q, &mut outp2, false);

            assert_eq!(outp, outp2);
        }
    }
}
