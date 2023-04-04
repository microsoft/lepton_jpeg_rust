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

#[cfg(feature = "compression_stats")]
use crate::metrics::ModelStatsCollector;

use super::{branch::Branch, simple_hash::SimpleHash};

const BITS_IN_BYTE: i32 = 8;
const BITS_IN_LONG: i32 = 64;
const BITS_IN_LONG_MINUS_LAST_BYTE: i32 = BITS_IN_LONG - BITS_IN_BYTE;

pub struct VPXBoolReader<R> {
    value: u64,
    range: u8,
    count: i32,
    upstream_reader: R,
    model_statistics: Metrics,
    pub hash: SimpleHash,
}

impl<R: Read> VPXBoolReader<R> {
    pub fn new(reader: R) -> Result<Self> {
        let mut r = VPXBoolReader {
            upstream_reader: reader,
            value: 0,
            count: -8,
            range: 255,
            model_statistics: Metrics::default(),
            hash: SimpleHash::new(),
        };

        vpx_reader_fill(&mut r.value, &mut r.count, &mut r.upstream_reader)?;

        let mut dummy_branch = Branch::new();
        r.get(&mut dummy_branch, ModelComponent::Dummy)?; // marker bit

        return Ok(r);
    }

    pub fn drain_stats(&mut self) -> Metrics {
        self.model_statistics.drain()
    }

    #[inline(never)]
    pub fn get_grid<const A: usize, const B: usize>(
        &mut self,
        branches: &mut [[Branch; B]; A],
        cmp: ModelComponent,
    ) -> Result<usize> {
        assert!(1 << (A - 1) == B);

        let mut index = A - 1;
        let mut value = 0;
        let mut decoded_so_far = 0;

        loop {
            let cur_bit = self.get(&mut branches[index as usize][decoded_so_far], cmp)? as usize;
            value |= cur_bit << index;
            decoded_so_far <<= 1;
            decoded_so_far |= cur_bit as usize;

            if index == 0 {
                break;
            }

            index -= 1;
        }

        Ok(value)
    }

    /// this is one of the most expensive functions since it is called for every exponent value,
    /// hence the amount of optimization.
    #[inline(always)]
    pub fn get_unary_encoded<const A: usize>(
        &mut self,
        branches: &mut [Branch; A],
        _cmp: ModelComponent,
    ) -> Result<usize> {
        let mut value = 0;

        let mut probability = branches[0].get_probability();

        let mut tmp_range = self.range;
        let mut tmp_value = self.value;
        let mut tmp_count = self.count;

        loop {
            if tmp_count < 0 {
                vpx_reader_fill(&mut tmp_value, &mut tmp_count, &mut self.upstream_reader)?;
            }

            let split = 1 + ((((tmp_range - 1) as u32) * (probability as u32)) >> 8) as u8;
            let big_split = (split as u64) << BITS_IN_LONG_MINUS_LAST_BYTE;
            let bit = tmp_value >= big_split;

            let shift;
            if bit {
                value += 1;
                if value < A {
                    // fetch next probability as soon as we know we will need it
                    // allows CPU significantly more parallelism when executing instructions
                    probability = branches[value].get_probability();

                    branches[value - 1].record_and_update_true_obs();
                    tmp_range -= split;
                    tmp_value -= big_split;
                    shift = tmp_range.leading_zeros();

                    tmp_value <<= shift;
                    tmp_count -= shift as i32;
                    tmp_range <<= shift;
                    continue;
                } else {
                    branches[value - 1].record_and_update_true_obs();

                    tmp_range -= split;
                    tmp_value -= big_split;
                }
            } else {
                branches[value].record_and_update_false_obs();
                tmp_range = split;
            }

            shift = tmp_range.leading_zeros();
            tmp_value <<= shift;
            tmp_count -= shift as i32;
            tmp_range <<= shift;
            break;
        }

        self.value = tmp_value;
        self.count = tmp_count;
        self.range = tmp_range;

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

        let mut tmp_range = self.range;
        let mut tmp_value = self.value;
        let mut tmp_count = self.count;

        for i in (0..n).rev() {
            coef |= (vpx_reader_get(
                &mut branches[i],
                &mut tmp_value,
                &mut tmp_range,
                &mut tmp_count,
                &mut self.upstream_reader,
                cmp,
            )? as usize)
                << i;
        }

        self.value = tmp_value;
        self.count = tmp_count;
        self.range = tmp_range;

        return Ok(coef);
    }

    #[inline(always)]
    pub fn get(&mut self, branch: &mut Branch, cmp: ModelComponent) -> Result<bool> {
        let mut tmp_range = self.range;
        let mut tmp_value = self.value;
        let mut tmp_count = self.count;

        let r = vpx_reader_get(
            branch,
            &mut tmp_value,
            &mut tmp_range,
            &mut tmp_count,
            &mut self.upstream_reader,
            cmp,
        );

        self.value = tmp_value;
        self.count = tmp_count;
        self.range = tmp_range;

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

        return r;
    }
}

#[inline(always)]
fn vpx_reader_get<R: Read>(
    branch: &mut Branch,
    tmp_value: &mut u64,
    tmp_range: &mut u8,
    tmp_count: &mut i32,
    upstream_reader: &mut R,
    _cmp: ModelComponent,
) -> Result<bool> {
    let probability = branch.get_probability() as u32;

    if *tmp_count < 0 {
        vpx_reader_fill(tmp_value, tmp_count, upstream_reader)?;
    }

    let split = 1 + ((((*tmp_range - 1) as u32) * (probability as u32)) >> 8) as u8;
    let big_split = (split as u64) << BITS_IN_LONG_MINUS_LAST_BYTE;
    let bit = *tmp_value >= big_split;

    let shift;
    if bit {
        branch.record_and_update_true_obs();
        *tmp_range -= split;
        *tmp_value -= big_split;
        shift = tmp_range.leading_zeros();
    } else {
        branch.record_and_update_false_obs();
        *tmp_range = split;
        shift = split.leading_zeros();
    }

    *tmp_value <<= shift;
    *tmp_count -= shift as i32;
    *tmp_range <<= shift;

    return Ok(bit);
}

#[inline(always)]
fn vpx_reader_fill<R: Read>(
    tmp_value: &mut u64,
    tmp_count: &mut i32,
    upstream_reader: &mut R,
) -> Result<()> {
    let mut shift = BITS_IN_LONG_MINUS_LAST_BYTE - (*tmp_count + BITS_IN_BYTE);

    while shift >= 0 {
        // BufReader is already pretty efficient handling small reads, so optimization doesn't help that much
        let mut v = [0u8; 1];
        let bytes_read = upstream_reader.read(&mut v[..])?;
        if bytes_read == 0 {
            break;
        }

        *tmp_value |= (v[0] as u64) << shift;
        shift -= BITS_IN_BYTE;
        *tmp_count += BITS_IN_BYTE;
    }

    return Ok(());
}

#[test]
fn run() {
    for a in 1..=255 {
        for b in 1..=255 {
            let prob = a * 256 / (a + b);

            let recip = (65536i64 * 65536) / (a + b);

            let prob2 = ((a * recip) + 32768) / (65536 * 256);

            if prob != prob2 {
                println!("diff={} a={}  b={}", prob - prob2, a, b);
            }
        }
    }
}
