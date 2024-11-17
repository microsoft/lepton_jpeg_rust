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

use std::io::{Result, Write};

use crate::metrics::{Metrics, ModelComponent};
use crate::structs::branch::Branch;
use crate::structs::simple_hash::SimpleHash;

// MAX_STREAM_BITS should be a multiple of 8 larger than 8,
// and (MAX_STREAM_BITS + 1 bit of carry + 1 bit of divider)
// should fit into 64 bits of `low_value`
const MAX_STREAM_BITS: i32 = 56; //48; //40;// 32;// 24;// 16;//

pub struct VPXBoolWriter<W> {
    low_value: u64,
    range: u32,
    writer: W,
    buffer: Vec<u8>,
    model_statistics: Metrics,
    #[allow(dead_code)]
    pub hash: SimpleHash,
}

impl<W: Write> VPXBoolWriter<W> {
    pub fn new(writer: W) -> Result<Self> {
        let mut retval = VPXBoolWriter {
            low_value: 1 << 9, // this divider bit keeps track of stream bits number
            range: 255,
            buffer: Vec::new(),
            writer: writer,
            model_statistics: Metrics::default(),
            hash: SimpleHash::new(),
        };

        let mut dummy_branch = Branch::new();
        // initial false bit is put to not get carry out of stream bits
        retval.put_bit(false, &mut dummy_branch, ModelComponent::Dummy)?;

        Ok(retval)
    }

    pub fn drain_stats(&mut self) -> Metrics {
        self.model_statistics.drain()
    }

    #[inline(always)]
    pub fn put(
        &mut self,
        value: bool,
        branch: &mut Branch,
        tmp_value: &mut u64,
        tmp_range: &mut u32,
        _cmp: ModelComponent,
    ) -> Result<()> {
        #[cfg(feature = "detailed_tracing")]
        {
            // used to detect divergences between the C++ and rust versions
            self.hash.hash(branch.get_u64());
            self.hash.hash(*tmp_value);
            self.hash.hash(*tmp_range);

            let hashed_value = self.hash.get();
            //if hashedValue == 0xe35c28fd
            {
                print!("({0}:{1:x})", value as u8, hashed_value);
                if hashed_value % 8 == 0 {
                    println!();
                }
            }
        }

        let probability = branch.get_probability() as u32;

        let split = 1 + (((*tmp_range - 1) * probability) >> 8);

        let mut shift;
        branch.record_and_update_bit(value);

        if value {
            *tmp_value += split as u64;
            *tmp_range -= split;
        } else {
            *tmp_range = split;
        }

        shift = (*tmp_range as u8).leading_zeros() as i32;

        #[cfg(feature = "compression_stats")]
        {
            self.model_statistics
                .record_compression_stats(_cmp, 1, i64::from(shift));
        }

        *tmp_range <<= shift;

        // check whether we have more than MAX_STREAM_BITS stream bits after shift
        let stream_bits = 64 - (*tmp_value).leading_zeros() as i32 - 2;
        let count = shift + stream_bits - MAX_STREAM_BITS;
        if count >= 0 {
            // check carry
            *tmp_value <<= MAX_STREAM_BITS - stream_bits;
            if (*tmp_value & (1 << MAX_STREAM_BITS)) != 0 {
                self.carry();
            }
            // write all full bytes
            let mut sh = MAX_STREAM_BITS - 8;
            while sh > 0 {
                self.buffer.push((*tmp_value >> sh) as u8);
                sh -= 8;
            }
            *tmp_value &= (1 << 8) - 1; // exclude written bytes
            *tmp_value |= 1 << 9; // restore divider bit

            shift = count;
        }

        *tmp_value <<= shift;

        Ok(())
    }

    /// Safe as: at the stream beginning initially put `false` ensure that carry cannot get out
    /// of the first stream byte - then `carry` cannot be invoked on empty `buffer`,
    /// and after the stream beginning `flush_non_final_data` keeps carry-terminating
    /// byte sequence (one non-255-byte before any number of 255-bytes) inside the `buffer`.
    #[inline(always)]
    fn carry(&mut self) {
        let mut x = self.buffer.len() - 1;

        while self.buffer[x] == 0xFF {
            self.buffer[x] = 0;

            assert!(x > 0);
            x -= 1;
        }

        self.buffer[x] += 1;
    }

    #[inline(always)]
    pub fn put_grid<const A: usize>(
        &mut self,
        v: u8,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<()> {
        // check if A is a power of 2
        assert!((A & (A - 1)) == 0);
        let mut tmp_value = self.low_value;
        let mut tmp_range = self.range;

        let mut index = A.ilog2() - 1;
        let mut serialized_so_far = 1;

        loop {
            let cur_bit = (v & (1 << index)) != 0;
            self.put(
                cur_bit,
                &mut branches[serialized_so_far],
                &mut tmp_value,
                &mut tmp_range,
                cmp,
            )?;

            if index == 0 {
                break;
            }

            serialized_so_far <<= 1;
            serialized_so_far |= cur_bit as usize;

            index -= 1;
        }

        self.low_value = tmp_value;
        self.range = tmp_range;

        Ok(())
    }

    #[inline(always)]
    pub fn put_n_bits<const A: usize>(
        &mut self,
        bits: usize,
        num_bits: usize,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<()> {
        let mut tmp_value = self.low_value;
        let mut tmp_range = self.range;

        let mut i: i32 = (num_bits - 1) as i32;
        while i >= 0 {
            self.put(
                (bits & (1 << i)) != 0,
                &mut branches[i as usize],
                &mut tmp_value,
                &mut tmp_range,
                cmp,
            )?;
            i -= 1;
        }

        self.low_value = tmp_value;
        self.range = tmp_range;

        Ok(())
    }

    #[inline(always)]
    pub fn put_unary_encoded<const A: usize>(
        &mut self,
        v: usize,
        branches: &mut [Branch; A],
        cmp: ModelComponent,
    ) -> Result<()> {
        assert!(v <= A);

        let mut tmp_value = self.low_value;
        let mut tmp_range = self.range;

        for i in 0..A {
            let cur_bit = v != i;

            self.put(
                cur_bit,
                &mut branches[i],
                &mut tmp_value,
                &mut tmp_range,
                cmp,
            )?;
            if !cur_bit {
                break;
            }
        }

        self.low_value = tmp_value;
        self.range = tmp_range;

        Ok(())
    }

    #[inline(always)]
    pub fn put_bit(
        &mut self,
        value: bool,
        branch: &mut Branch,
        _cmp: ModelComponent,
    ) -> Result<()> {
        let mut tmp_value = self.low_value;
        let mut tmp_range = self.range;

        self.put(value, branch, &mut tmp_value, &mut tmp_range, _cmp)?;

        self.low_value = tmp_value;
        self.range = tmp_range;

        Ok(())
    }

    // Here we write down only bytes of the stream necessary for decoding -
    // opposite to initial Lepton implementation that writes down all the buffer.
    pub fn finish(&mut self) -> Result<()> {
        let mut tmp_value = self.low_value;
        let stream_bits = 64 - tmp_value.leading_zeros() as i32 - 2;

        tmp_value <<= MAX_STREAM_BITS - stream_bits;
        if (tmp_value & (1 << MAX_STREAM_BITS)) != 0 {
            self.carry();
        }

        let mut shift = MAX_STREAM_BITS - 8;
        let mut stream_bytes = (stream_bits + 7) >> 3;
        while stream_bytes > 0 {
            self.buffer.push((tmp_value >> shift) as u8);
            shift -= 8;
            stream_bytes -= 1;
        }

        self.writer.write_all(&self.buffer[..])?;
        Ok(())
    }

    /// When buffer is full and is going to be sent to output, preserve buffer data that
    /// is not final and should be carried over to the next buffer. At least one byte
    /// will remain in `buffer` if it is non-empty.
    pub fn flush_non_final_data(&mut self) -> Result<()> {
        // carry over buffer data that might be not final
        let mut i = self.buffer.len();
        if i > 1 {
            i -= 1;
            while self.buffer[i] == 0xFF {
                assert!(i > 0);
                i -= 1;
            }

            self.writer.write_all(&self.buffer[..i])?;
            self.buffer.drain(..i);
        }

        Ok(())
    }
}

#[cfg(test)]
use crate::structs::vpx_bool_reader::VPXBoolReader;

#[test]
fn test_roundtrip_vpxboolwriter_n_bits() {
    const MAX_N: usize = 8;

    #[derive(Default)]
    struct BranchData {
        branches: [Branch; MAX_N],
    }

    let mut buffer = Vec::new();
    let mut writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let mut branches = BranchData::default();

    for i in 0..1024 {
        writer
            .put_n_bits(
                i as usize % 256,
                MAX_N,
                &mut branches.branches,
                ModelComponent::Dummy,
            )
            .unwrap();
    }

    writer.finish().unwrap();

    let mut branches = BranchData::default();

    let mut reader = VPXBoolReader::new(&buffer[..]).unwrap();
    for i in 0..1024 {
        let read_value = reader
            .get_n_bits(MAX_N, &mut branches.branches, ModelComponent::Dummy)
            .unwrap();
        assert_eq!(read_value, i as usize % 256);
    }
}

#[test]
fn test_roundtrip_vpxboolwriter_unary() {
    const MAX_UNARY: usize = 11; // the size used in Lepton

    #[derive(Default)]
    struct BranchData {
        branches: [Branch; MAX_UNARY],
    }

    let mut buffer = Vec::new();
    let mut writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let mut branches = BranchData::default();

    for i in 0..1024 {
        writer
            .put_unary_encoded(
                i as usize % (MAX_UNARY + 1),
                &mut branches.branches,
                ModelComponent::Dummy,
            )
            .unwrap();
    }

    writer.finish().unwrap();

    let mut branches = BranchData::default();

    let mut reader = VPXBoolReader::new(&buffer[..]).unwrap();
    for i in 0..1024 {
        let read_value = reader
            .get_unary_encoded(&mut branches.branches, ModelComponent::Dummy)
            .unwrap();
        assert_eq!(read_value, i as usize % (MAX_UNARY + 1));
    }
}

#[test]
fn test_roundtrip_vpxboolwriter_grid() {
    #[derive(Default)]
    struct BranchData {
        branches: [Branch; 8],
    }

    let mut buffer = Vec::new();
    let mut writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let mut branches = BranchData::default();

    for i in 0..1024 {
        writer
            .put_grid(i as u8 % 8, &mut branches.branches, ModelComponent::Dummy)
            .unwrap();
    }

    writer.finish().unwrap();

    let mut branches = BranchData::default();

    let mut reader = VPXBoolReader::new(&buffer[..]).unwrap();
    for i in 0..1024 {
        let read_value = reader
            .get_grid(&mut branches.branches, ModelComponent::Dummy)
            .unwrap();
        assert_eq!(read_value, i as usize % 8);
    }
}

#[test]
fn test_roundtrip_vpxboolwriter_single_bit() {
    let mut buffer = Vec::new();
    let mut writer = VPXBoolWriter::new(&mut buffer).unwrap();

    let mut branch = Branch::default();

    for i in 0..1024 {
        writer
            .put_bit(i % 10 == 0, &mut branch, ModelComponent::Dummy)
            .unwrap();
    }

    writer.finish().unwrap();

    let mut branch = Branch::default();

    let mut reader = VPXBoolReader::new(&buffer[..]).unwrap();
    for i in 0..1024 {
        let read_value = reader.get_bit(&mut branch, ModelComponent::Dummy).unwrap();
        assert_eq!(read_value, i % 10 == 0);
    }
}
