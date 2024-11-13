/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::mem;

pub struct BitWriter {
    data_buffer: Vec<u8>,
    fill_register: u64,
    current_bit: u32,
}

// use to write varying sized bits for coding JPEG. Escapes 0xff -> [0xff,0]
impl BitWriter {
    pub fn new(capacity: usize) -> Self {
        return BitWriter {
            current_bit: 64,
            fill_register: 0,
            data_buffer: Vec::<u8>::with_capacity(capacity),
        };
    }

    /// flushes whole bytes from the register into the data buffer
    fn flush_whole_bytes(&mut self) {
        while self.current_bit <= 56 {
            let b = (self.fill_register >> 56) as u8;
            if b != 0xff {
                self.data_buffer.push(b);
            } else {
                // escape 0xff here to avoid multiple scans of the same data
                self.data_buffer.extend_from_slice(&[0xff, 0]);
            }

            self.fill_register <<= 8;
            self.current_bit += 8;
        }
    }

    /// write data
    pub fn write_byte_unescaped(&mut self, b: u8) {
        assert!(self.current_bit == 64);
        self.data_buffer.push(b);
    }

    #[inline(always)]
    pub fn write(&mut self, val: u32, new_bits: u32) {
        /// this is the slow path that is rarely called but generates a lot of code inlined
        /// so we move it out of the main function to keep the main function small with few branches.
        ///
        /// We also call this path when we are about to overflow the buffer to avoid having
        /// to inline the buffer growing logic, which is also much bigger than a simple insert.
        #[inline(never)]
        #[cold]
        fn write_ff_encoded(data_buffer: &mut Vec<u8>, fill_register: u64) {
            for i in 0..8 {
                let b = (fill_register >> (56 - (i * 8))) as u8;
                if b != 0xff {
                    data_buffer.push(b);
                } else {
                    // escape 0xff here to avoid multiple scans of the same data
                    data_buffer.extend_from_slice(&[0xff, 0]);
                }
            }
        }

        debug_assert!(
            val < (1 << new_bits),
            "value {0} should fit into the number of {1} bits provided",
            val,
            new_bits
        );

        // first see if everything fits in the current register
        if new_bits <= self.current_bit {
            self.fill_register |= (val as u64).wrapping_shl(self.current_bit - new_bits); // support corner case where new_bits is zero, we don't want to panic
            self.current_bit = self.current_bit - new_bits;
        } else {
            // if not, fill up the register so to the 64 bit boundary we can flush it hopefully without any 0xff bytes
            let fill = self.fill_register | (val as u64).wrapping_shr(new_bits - self.current_bit);

            let leftover_new_bits = new_bits - self.current_bit;
            let leftover_val = val & (1 << leftover_new_bits) - 1;

            // flush bytes slowly if we have any 0xff bytes or if we are about to overflow the buffer
            // (overflow check matches implementation in RawVec so that the optimizer can remove the buffer growing code)
            if (fill & 0x8080808080808080 & !fill.wrapping_add(0x0101010101010101)) != 0
                || self
                    .data_buffer
                    .capacity()
                    .wrapping_sub(self.data_buffer.len())
                    < 8
            {
                write_ff_encoded(&mut self.data_buffer, fill);
            } else {
                self.data_buffer.extend_from_slice(&fill.to_be_bytes());
            }

            self.fill_register = (leftover_val as u64).wrapping_shl(64 - leftover_new_bits); // support corner case where new_bits is zero, we don't want to panic
            self.current_bit = 64 - leftover_new_bits;
        }
    }

    pub fn pad(&mut self, fillbit: u8) {
        let mut offset = 1;
        while (self.current_bit & 7) != 0 {
            self.write(if (fillbit & offset) != 0 { 1 } else { 0 }, 1);
            offset <<= 1;
        }

        self.flush_whole_bytes();

        debug_assert!(
            self.current_bit == 64,
            "there should be no remainder after padding"
        );
    }

    // flushes the data buffer while escaping all 0xff characters
    pub fn detach_buffer(&mut self) -> Vec<u8> {
        // flush any remaining whole bytes
        self.flush_whole_bytes();

        mem::take(&mut self.data_buffer)
    }

    pub fn reset_from_overhang_byte_and_num_bits(&mut self, overhang_byte: u8, num_bits: u32) {
        self.data_buffer.clear();

        self.fill_register = 0;
        self.fill_register = overhang_byte as u64;
        self.fill_register <<= 56;
        self.current_bit = 64 - num_bits;
    }

    pub fn has_no_remainder(&self) -> bool {
        return self.current_bit == 64;
    }
}

#[cfg(test)]
use std::io::Cursor;

#[cfg(test)]
use crate::helpers::u32_bit_length;
#[cfg(test)]
use crate::structs::bit_reader::BitReader;

// write a test pattern with an escape and see if it matches
#[test]
fn write_simple() {
    let arr = [0x12 as u8, 0x34, 0x45, 0x67, 0x89, 0xff, 00, 0xee];

    let mut b = BitWriter::new(1024);

    b.write(1, 4);
    b.write(2, 4);
    b.write(3, 4);
    b.write(4, 4);
    b.write(4, 4);
    b.write(0x56, 8);
    b.write(0x78, 8);
    b.write(0x9f, 8);
    b.write(0xfe, 8);
    b.write(0xe, 4);

    let w = b.detach_buffer();

    assert_eq!(w[..], arr);
}

// verify the the bits roundtrip correctly in a fairly simple scenario
#[test]
fn roundtrip_bits() {
    let buf;
    {
        let mut b = BitWriter::new(1024);
        for i in 1..2048 {
            b.write(i, u32_bit_length(i as u32) as u32);
        }

        b.pad(0xff);

        buf = b.detach_buffer();
    }

    {
        let mut r = BitReader::new(Cursor::new(&buf));

        for i in 1..2048 {
            assert_eq!(i, r.read(u32_bit_length(i as u32)).unwrap());
        }

        let mut pad = Some(0xff);
        r.read_and_verify_fill_bits(&mut pad).unwrap();
    }
}

/// verify the the bits roundtrip correctly with random bits
#[test]
fn roundtrip_randombits() {
    use rand::Rng;

    let buf;

    const ITERATIONS: usize = 10000;

    let mut rng = crate::helpers::get_rand_from_seed([0u8; 32]);
    let mut test_data = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let bits = rng.gen_range(0..=16);
        let v = rng.gen_range(0..=65535) & ((1 << bits) - 1);
        test_data.push((v as u16, bits as u8));
    }

    {
        let mut b = BitWriter::new(1024);
        for i in &test_data {
            b.write(i.0 as u32, i.1 as u32);
        }

        b.pad(0xff);

        buf = b.detach_buffer();
    }

    {
        let mut r = BitReader::new(Cursor::new(&buf));

        for i in &test_data {
            assert_eq!(i.0, r.read(i.1).unwrap());
        }

        let mut pad = Some(0xff);
        r.read_and_verify_fill_bits(&mut pad).unwrap();
    }
}
