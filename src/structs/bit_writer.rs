/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::io::Write;

pub struct BitWriter {
    data_buffer: Vec<u8>,
    fill_register: u64,
    current_bit: u32,
}

// use to write varying sized bits for coding JPEG. Escapes 0xff -> [0xff,0]
impl BitWriter {
    pub fn new() -> Self {
        return BitWriter {
            current_bit: 64,
            fill_register: 0,
            data_buffer: Vec::<u8>::with_capacity(65536),
        };
    }

    #[inline(never)]
    fn flush_bytes_slowly(&mut self) {
        let mut tmp_current_bit = self.current_bit;
        let mut tmp_fill_register = self.fill_register;

        while tmp_current_bit <= 56 {
            let b = (tmp_fill_register >> 56) as u8;
            if b != 0xff {
                self.data_buffer.push(b);
            } else {
                // escape 0xff here to avoid multiple scans of the same data
                self.data_buffer.extend_from_slice(&[0xff, 0]);
            }

            tmp_fill_register <<= 8;
            tmp_current_bit += 8;
        }

        self.fill_register = tmp_fill_register;
        self.current_bit = tmp_current_bit;
    }

    #[inline(always)]
    pub fn write(&mut self, mut val: u64, mut new_bits: u32) {
        debug_assert!(new_bits <= 64, "new_bits {0} should be <= 64", new_bits);
        debug_assert!(
            val < (1 << new_bits),
            "value {0} should fit into the number of {1} bits provided",
            val,
            new_bits
        );

        // first see if everything fits in the current register
        if new_bits <= self.current_bit {
            self.fill_register |= val.wrapping_shl(self.current_bit - new_bits); // support corner case where new_bits is zero, we don't want to panic
            self.current_bit = self.current_bit - new_bits;
        } else {
            // if not, fill up the register so to the 64 bit boundary we can flush it hopefully without any 0xff bytes
            let fill = self.fill_register | val.wrapping_shr(new_bits - self.current_bit);

            new_bits -= self.current_bit;
            val &= (1 << new_bits) - 1;

            // flush bytes slowly if we have any 0xff bytes or if we are about to overflow the buffer
            // (overflow check matches implementation in RawVec so that the optimizer can remove the buffer growing code)
            if (fill & 0x8080808080808080 & !fill.wrapping_add(0x0101010101010101)) != 0
                || self
                    .data_buffer
                    .capacity()
                    .wrapping_sub(self.data_buffer.len())
                    < 8
            {
                self.fill_register = fill;
                self.current_bit = 0;
                self.flush_bytes_slowly();
            } else {
                self.data_buffer.extend_from_slice(&fill.to_be_bytes());
            }
            self.fill_register = (val as u64).wrapping_shl(64 - new_bits); // support corner case where new_bits is zero, we don't want to panic
            self.current_bit = 64 - new_bits;
        }
    }

    pub fn pad(&mut self, fillbit: u8) {
        let mut offset = 1;
        while (self.current_bit & 7) != 0 {
            self.write(if (fillbit & offset) != 0 { 1 } else { 0 }, 1);
            offset <<= 1;
        }

        self.flush_bytes_slowly();

        debug_assert!(
            self.current_bit == 64,
            "there should be no remainder after padding"
        );
    }

    // flushes the data buffer while escaping all 0xff characters
    pub fn flush_with_escape<W: Write>(&mut self, w: &mut W) -> anyhow::Result<()> {
        // flush any remaining whole bytes
        self.flush_bytes_slowly();

        w.write_all(&self.data_buffer[..])?;

        self.data_buffer.drain(..);

        Ok(())
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
use super::bit_reader::BitReader;
#[cfg(test)]
use crate::helpers::u32_bit_length;
#[cfg(test)]
use std::io::Cursor;

// write a test pattern with an escape and see if it matches
#[test]
fn write_simple() {
    let arr = [0x12 as u8, 0x34, 0x45, 0x67, 0x89, 0xff, 00, 0xee];

    let mut b = BitWriter::new();

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

    let mut w = Vec::new();
    b.flush_with_escape(&mut Cursor::new(&mut w)).unwrap();

    assert_eq!(w[..], arr);
}

// verify the the bits roundtrip correctly in a fairly simple scenario
#[test]
fn roundtrip_bits() {
    let mut buf = Vec::new();

    {
        let mut b = BitWriter::new();
        for i in 1..2048 {
            b.write(i, u32_bit_length(i as u32) as u32);
        }

        b.pad(0xff);

        let mut writer = Cursor::new(&mut buf);
        b.flush_with_escape(&mut writer).unwrap();
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
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    let mut buf = Vec::new();

    const ITERATIONS: usize = 10000;

    let mut rng = StdRng::from_seed([0u8; 32]);
    let mut test_data = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let bits = rng.gen_range(0..=16);
        let v = rng.gen_range(0..=65535) & ((1 << bits) - 1);
        test_data.push((v as u16, bits as u8));
    }

    {
        let mut writer = Cursor::new(&mut buf);

        let mut b = BitWriter::new();
        for i in &test_data {
            b.write(i.0 as u64, i.1 as u32);

            // randomly flush the buffer
            if rng.gen_range(0..50) == 0 {
                b.flush_with_escape(&mut writer).unwrap();
            }
        }

        b.pad(0xff);

        b.flush_with_escape(&mut writer).unwrap();
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
