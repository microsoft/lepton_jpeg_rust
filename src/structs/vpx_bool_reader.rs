/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE banner below
 *  An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the VPX_AUTHORS file in this directory
 */
/*
Copyright (c) 2010, Google Inc. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of Google nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use std::io::{Read, Result};

use crate::metrics::{Metrics, ModelComponent};

use super::{branch::Branch, simple_hash::SimpleHash};

const BITS_IN_BYTE: i32 = 8;
const BITS_IN_VALUE: i32 = 32;
const BITS_IN_VALUE_MINUS_LAST_BYTE: i32 = BITS_IN_VALUE - BITS_IN_BYTE;

pub struct VPXBoolReader<R> {
    value: u32,
    range: u32, // 128 << BITS_IN_VALUE_MINUS_LAST_BYTE <= range <= 255 << BITS_IN_VALUE_MINUS_LAST_BYTE
    count: i32,
    upstream_reader: R,
    model_statistics: Metrics,
    #[allow(dead_code)]
    pub hash: SimpleHash,
}

impl<R: Read> VPXBoolReader<R> {
    pub fn new(reader: R) -> Result<Self> {
        let mut r = VPXBoolReader {
            upstream_reader: reader,
            value: 0,
            count: -8,
            range: 255 << BITS_IN_VALUE_MINUS_LAST_BYTE,
            model_statistics: Metrics::default(),
            hash: SimpleHash::new(),
        };

        Self::vpx_reader_fill(&mut r.value, &mut r.count, &mut r.upstream_reader)?;

        let mut dummy_branch = Branch::new();
        r.get(&mut dummy_branch, ModelComponent::Dummy)?; // marker bit

        return Ok(r);
    }

    pub fn drain_stats(&mut self) -> Metrics {
        self.model_statistics.drain()
    }

    #[inline(never)]
    pub fn get_grid<const A: usize>(
        &mut self,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<usize> {
        // check if A is a power of 2
        assert!((A & (A - 1)) == 0);

        let mut decoded_so_far = 1;

        for _index in 0..A.ilog2() {
            let cur_bit = self.get(&mut branches[decoded_so_far], cmp)? as usize;
            decoded_so_far <<= 1;
            decoded_so_far |= cur_bit;
        }

        // remove set leading bit
        let value = decoded_so_far ^ A;

        Ok(value)
    }

    #[inline(never)]
    pub fn get_unary_encoded<const A: usize>(
        &mut self,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<usize> {
        let mut value = 0;

        while value != A {
            let cur_bit = self.get(&mut branches[value], cmp)?;
            if !cur_bit {
                break;
            }

            value += 1;
        }

        return Ok(value);
    }

    #[inline(never)]
    pub fn get_n_bits<const A: usize>(
        &mut self,
        n: usize,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<usize> {
        assert!(n <= branches.len());

        let mut coef = 0;
        for i in (0..n).rev() {
            coef |= (self.get(&mut branches[i], cmp)? as usize) << i;
        }

        return Ok(coef);
    }

    // Lepton uses VP8 adaptive arithmetic coding scheme, where bits are extracted from file stream
    // by division of current 8-bit stream `value` by adaptive 8-bit `split`. Adaptation is achieved by
    // combination of predicted probability to get false bit (`1 <= probability <= 255`, in 1/256 units),
    // and `range` that represents maximum possible value of yet-not-decoded stream part (so that
    // `range > value`, `128 <= range <= 256` in units of $2^{-n-8}$ for the `n` bits already decoded)
    // by forming predictor `split = 1 + (((range - 1) * probability) >> BITS_IN_BYTE)`,
    // `1 <= split <= range - 1`. Comparison of predictor with stream gives the next decoded bit:
    // true for `value >= split` and false otherwise - this is effectively division step.
    // After this we shrink `value` and `range` by `split` for true or shrink `range` to `split`
    // for false and update `probability`. Now `range` can get out of allowable range and we restore it
    // by shifting left both `range` and `value` with corresponding filling of `value` by further
    // stream bits (it corresponds to bring down new digit in division). Repeat until stream ends.
    //
    // Reference: https://datatracker.ietf.org/doc/html/rfc6386#section-7.
    //
    // Here some imrovements to the basic scheme are implemented. First, we store more stream bits
    // in `value` to reduce refill rate, so that 8 MSBs of `value` represent `value` of the scheme
    // (it was already implemented in DropBox version, however, with shorter 16-bit `value`).
    // Second, `range` and `split` are also stored in 8 MSBs of the same size variables (it is new
    // and it allows to reduce number of operations to compute `split` - previously `big_split` -
    // and to update `range` and `shift`).
    #[inline(always)]
    pub fn get(&mut self, branch: &mut Branch, _cmp: ModelComponent) -> Result<bool> {
        let mut tmp_value = self.value;
        let mut tmp_range = self.range;
        let mut tmp_count = self.count;

        if tmp_count < 0 {
            Self::vpx_reader_fill(&mut tmp_value, &mut tmp_count, &mut self.upstream_reader)?;
        }

        let probability = branch.get_probability() as u32;

        let split = ((((tmp_range - (1 << BITS_IN_VALUE_MINUS_LAST_BYTE)) >> 8) * probability)
            & (0xFF << BITS_IN_VALUE_MINUS_LAST_BYTE))
            + (1 << BITS_IN_VALUE_MINUS_LAST_BYTE);

        // So optimizer understands that 0 should never happen and uses a cold jump
        // if we don't have LZCNT on x86 CPUs (older BSR instruction requires check for zero).
        // This is better since the branch prediction figures quickly this never happens and can run
        // the code sequentially.
        #[cfg(all(
            not(target_feature = "lzcnt"),
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        assert!(tmp_range - split > 0);

        let bit = tmp_value >= split;

        branch.record_and_update_bit(bit);

        if bit {
            tmp_range -= split;
            tmp_value -= split;
        } else {
            tmp_range = split;
        }

        let shift = tmp_range.leading_zeros() as i32;

        self.value = tmp_value << shift;
        self.range = tmp_range << shift;
        self.count = tmp_count - shift;

        #[cfg(feature = "compression_stats")]
        {
            self.model_statistics
                .record_compression_stats(_cmp, 1, i64::from(shift));
        }

        #[cfg(feature = "detailed_tracing")]
        {
            self.hash.hash(branch.get_u64());
            self.hash.hash(self.value);
            self.hash.hash(self.count);
            self.hash.hash(self.range);

            //if hash == 0x88f9c945
            {
                let hash = self.hash.get();

                print!("({0}:{1:x})", bit as u8, hash);
                if hash % 8 == 0 {
                    println!();
                }
            }
        }

        return Ok(bit);
    }

    #[cold]
    #[inline(always)]
    fn vpx_reader_fill(
        tmp_value: &mut u32,
        tmp_count: &mut i32,
        upstream_reader: &mut R,
    ) -> Result<()> {
        let mut shift = BITS_IN_VALUE_MINUS_LAST_BYTE - (*tmp_count + BITS_IN_BYTE);

        while shift >= 0 {
            // BufReader is already pretty efficient handling small reads, so optimization doesn't help that much
            let mut v = [0u8; 1];
            let bytes_read = upstream_reader.read(&mut v)?;
            if bytes_read == 0 {
                break;
            }

            *tmp_value |= (v[0] as u32) << shift;
            shift -= BITS_IN_BYTE;
            *tmp_count += BITS_IN_BYTE;
        }

        return Ok(());
    }
}
