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

use crate::lepton_error;
use crate::lepton_error::{err_exit_code, ExitCode};
use crate::metrics::{Metrics, ModelComponent};
use crate::structs::branch::Branch;
use crate::structs::simple_hash::SimpleHash;

const BITS_IN_BYTE: u32 = 8;
const BITS_IN_VALUE: u32 = 64;
const BITS_IN_VALUE_MINUS_LAST_BYTE: u32 = BITS_IN_VALUE - BITS_IN_BYTE;
const VALUE_MASK: u64 = (1 << BITS_IN_VALUE_MINUS_LAST_BYTE) - 1;

pub struct VPXBoolReader<R> {
    value: u64,
    range: u64, // 128 << BITS_IN_VALUE_MINUS_LAST_BYTE <= range <= 255 << BITS_IN_VALUE_MINUS_LAST_BYTE
    upstream_reader: R,
    model_statistics: Metrics,
    #[allow(dead_code)]
    pub hash: SimpleHash,
}

impl<R: Read> VPXBoolReader<R> {
    pub fn new(reader: R) -> lepton_error::Result<Self> {
        let mut r = VPXBoolReader {
            upstream_reader: reader,
            value: 1 << (BITS_IN_VALUE - 1), // guard bit
            range: 255 << BITS_IN_VALUE_MINUS_LAST_BYTE,
            model_statistics: Metrics::default(),
            hash: SimpleHash::new(),
        };

        let mut dummy_branch = Branch::new();
        let bit = r.get_bit(&mut dummy_branch, ModelComponent::Dummy)?; // marker false bit
        if bit {
            return err_exit_code(ExitCode::StreamInconsistent, "StreamInconsistent");
        }

        return Ok(r);
    }

    pub fn drain_stats(&mut self) -> Metrics {
        self.model_statistics.drain()
    }

    // Lepton uses VP8 adaptive arithmetic coding scheme, where bits are extracted from file stream
    // by "division" of current 8-bit stream `value` by adaptive 8-bit `split`. Adaptation is achieved by
    // combination of predicted probability to get false bit (`1 <= probability <= 255`, in 1/256 units),
    // and `range` that represents maximum possible value of yet-not-decoded stream part (so that
    // `range > value`, `128 <= range <= 256` in units of $2^{-n-8}$ for the `n` bits already consumed)
    // by forming predictor `split = 1 + (((range - 1) * probability) >> BITS_IN_BYTE)`,
    // `1 <= split <= range - 1`. Comparison of predictor with stream gives the next decoded bit:
    // true for `value >= split` and false otherwise - this is effectively division step.
    // After this we shrink `value` and `range` by `split` for true or shrink `range` to `split`
    // for false and update `probability`. Now `range` can get out of allowable range and we restore it
    // by shifting left both `range` and `value` with corresponding filling of `value` by further
    // stream bits (it corresponds to bring down new digit in division, and since `range > value` is invariant
    // of the operations, shifted out `value` bits are guaranteed to be 0). Repeat until stream ends.
    //
    // Reference: https://datatracker.ietf.org/doc/html/rfc6386#section-7.
    //
    // Here some improvements to the basic scheme are implemented. First, we store more stream bits
    // in `value` to reduce refill rate, so that 8 MSBs of `value` represent `value` of the scheme
    // (it was already implemented in DropBox version, however, with shorter 16-bit `value`).
    // Second, `range` and `split` are also stored in 8 MSBs of the same size variables (it is new
    // and it allows to reduce number of operations to compute `split` - previously `big_split` -
    // and to update `range` and `shift`). Third, we use local values for all stream state variables
    // to reduce number of memory load/store operations in decoding of many-bit values. Fourth,
    // we use in `value` a set bit after the stream bits as a guard - completely getting rid
    // of bit counter and not changing comparison result `value >= split`.
    #[inline(always)]
    pub fn get(
        &mut self,
        branch: &mut Branch,
        tmp_value: &mut u64,
        tmp_range: &mut u64,
        _cmp: ModelComponent,
    ) -> bool {
        let probability = branch.get_probability() as u64;

        let split = mul_prob(*tmp_range, probability);

        // So optimizer understands that 0 should never happen and uses a cold jump
        // if we don't have LZCNT on x86 CPUs (older BSR instruction requires check for zero).
        // This is better since the branch prediction figures quickly this never happens and can run
        // the code sequentially.
        #[cfg(all(
            not(target_feature = "lzcnt"),
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        assert!(*tmp_range - split > 0);

        let bit = *tmp_value >= split;

        branch.record_and_update_bit(bit);

        if bit {
            *tmp_range -= split;
            *tmp_value -= split;
        } else {
            *tmp_range = split;
        }

        let shift = (*tmp_range).leading_zeros();

        *tmp_value <<= shift;
        *tmp_range <<= shift;

        #[cfg(feature = "compression_stats")]
        {
            self.model_statistics
                .record_compression_stats(_cmp, 1, i64::from(shift));
        }

        #[cfg(feature = "detailed_tracing")]
        {
            self.hash.hash(branch.get_u64());
            self.hash.hash(*tmp_value);
            self.hash.hash(*tmp_range);

            let hash = self.hash.get();
            //if hash == 0x88f9c945
            {
                print!("({0}:{1:x})", bit as u8, hash);
                if hash % 8 == 0 {
                    println!();
                }
            }
        }

        bit
    }

    #[inline(always)]
    pub fn get_grid<const A: usize>(
        &mut self,
        branches: &mut [Branch; A],
        _cmp: ModelComponent,
    ) -> Result<usize> {
        // check if A is a power of 2
        debug_assert!((A & (A - 1)) == 0);

        let mut tmp_value = self.value;
        let mut tmp_range = self.range;

        let mut decoded_so_far = 1;
        // We can read only each 7-th iteration: minimum 56 bits are in `value` after `vpx_reader_fill`,
        // and one `get` needs 8 bits but consumes at most 7 bits (with `range` coming from >127 to 1).
        // As Lepton uses only 3 and 6 iterations, we can read only once.
        debug_assert!(A <= 128);
        tmp_value = Self::vpx_reader_fill(tmp_value, &mut self.upstream_reader)?;

        for _index in 0..A.ilog2() {
            let cur_bit = self.get(
                &mut branches[decoded_so_far],
                &mut tmp_value,
                &mut tmp_range,
                _cmp,
            ) as usize;
            decoded_so_far <<= 1;
            decoded_so_far |= cur_bit;
        }

        // remove set leading bit
        let value = decoded_so_far ^ A;

        self.value = tmp_value;
        self.range = tmp_range;

        Ok(value)
    }

    #[inline(always)]
    pub fn get_unary_encoded<const A: usize>(
        &mut self,
        branches: &mut [Branch; A],
        _cmp: ModelComponent,
    ) -> Result<usize> {
        let mut tmp_value = self.value;
        let mut tmp_range = self.range;

        for value in 0..A {
            let split = mul_prob(tmp_range, branches[value].get_probability() as u64);

            // We know that after this we have min 56 stream bits in `tmp_value`,
            // and can have at least 7 iterations, so we can decode 7 bits at once.
            // Each iteration needs at least 8 bits of stream in `tmp_value` and
            // consumes max 7 of them.
            debug_assert!(A <= 14);
            if value == 0 || value == 7 {
                tmp_value = Self::vpx_reader_fill(tmp_value, &mut self.upstream_reader)?;
            }

            if tmp_value >= split {
                branches[value].record_and_update_bit(true);

                tmp_range -= split;
                tmp_value -= split;

                let shift = tmp_range.leading_zeros();

                tmp_value <<= shift;
                tmp_range <<= shift;

                #[cfg(feature = "compression_stats")]
                {
                    self.model_statistics
                        .record_compression_stats(_cmp, 1, i64::from(shift));
                }

                #[cfg(feature = "detailed_tracing")]
                {
                    self.hash.hash(branches[value].get_u64());
                    self.hash.hash(tmp_value);
                    self.hash.hash(tmp_range);

                    let hash = self.hash.get();
                    //if hash == 0x88f9c945
                    {
                        print!("({0}:{1:x})", true as u8, hash);
                        if hash % 8 == 0 {
                            println!();
                        }
                    }
                }
            } else {
                branches[value].record_and_update_bit(false);

                tmp_range = split;

                let shift = tmp_range.leading_zeros();

                tmp_value <<= shift;
                tmp_range <<= shift;

                #[cfg(feature = "compression_stats")]
                {
                    self.model_statistics
                        .record_compression_stats(_cmp, 1, i64::from(shift));
                }

                #[cfg(feature = "detailed_tracing")]
                {
                    self.hash.hash(branches[value].get_u64());
                    self.hash.hash(tmp_value);
                    self.hash.hash(tmp_range);

                    let hash = self.hash.get();
                    //if hash == 0x88f9c945
                    {
                        print!("({0}:{1:x})", false as u8, hash);
                        if hash % 8 == 0 {
                            println!();
                        }
                    }
                }

                self.value = tmp_value;
                self.range = tmp_range;

                return Ok(value);
            }
        }

        self.value = tmp_value;
        self.range = tmp_range;

        Ok(A)
    }

    #[inline(always)]
    pub fn get_n_bits<const A: usize>(
        &mut self,
        n: usize,
        branches: &mut [Branch; A],
        _cmp: ModelComponent,
    ) -> Result<usize> {
        assert!(n <= branches.len());

        let mut tmp_value = self.value;
        let mut tmp_range = self.range;

        let mut coef = 0;
        for i in (0..n).rev() {
            // Here the fastest way is to use condition of `get_bit`, presumably as
            // this loop cannot be unrolled due to vaiable iterations number.
            // Moreover, this condition holds very rarely as `value` is usually already filled
            // by previous `get_bit` sign reading.
            if tmp_value & VALUE_MASK == 0 {
                tmp_value = Self::vpx_reader_fill(tmp_value, &mut self.upstream_reader)?;
            }

            coef |=
                (self.get(&mut branches[i], &mut tmp_value, &mut tmp_range, _cmp) as usize) << i;
        }

        self.value = tmp_value;
        self.range = tmp_range;

        return Ok(coef);
    }

    #[inline(always)]
    pub fn get_bit(&mut self, branch: &mut Branch, _cmp: ModelComponent) -> Result<bool> {
        let mut tmp_value = self.value;
        let mut tmp_range = self.range;

        // We ensure that the guard bit never comes into the first byte,
        // thus having in `value` at least 8 stream bits.
        if tmp_value & VALUE_MASK == 0 {
            tmp_value = Self::vpx_reader_fill(tmp_value, &mut self.upstream_reader)?;
        }

        let bit = self.get(branch, &mut tmp_value, &mut tmp_range, _cmp);

        self.value = tmp_value;
        self.range = tmp_range;

        return Ok(bit);
    }

    // Fill `tmp_value` maximally still preserving space for the guard bit,
    // after this returned value has `56 | (63 - shift)` stream bits
    #[inline(always)]
    fn vpx_reader_fill(mut tmp_value: u64, upstream_reader: &mut R) -> Result<u64> {
        // This `if` does not change performance but drops down instructions count by 3 %
        if tmp_value & 0xFF == 0 {
            let mut shift: i32 = tmp_value.trailing_zeros() as i32;
            // Unset the last guard bit and set a new one
            tmp_value &= tmp_value - 1;
            tmp_value |= 1 << (shift & 7);

            // BufReader is already pretty efficient handling small reads, so optimization doesn't help that much
            let mut v = [0u8; 1];
            shift -= 7;

            while shift > 0 {
                let bytes_read = upstream_reader.read(&mut v)?;
                if bytes_read == 0 {
                    break;
                }

                tmp_value |= (v[0] as u64) << shift;
                shift -= 8;
            }
        }

        return Ok(tmp_value);
    }
}

fn mul_prob(tmp_range: u64, probability: u64) -> u64 {
    ((((tmp_range - (1 << BITS_IN_VALUE_MINUS_LAST_BYTE)) >> 8) * probability)
        & (0xFF << BITS_IN_VALUE_MINUS_LAST_BYTE))
        + (1 << BITS_IN_VALUE_MINUS_LAST_BYTE)
}
