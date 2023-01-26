/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

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

    fn flush_bytes(&mut self) {
        let mut tmp_current_bit = self.current_bit;
        let mut tmp_fill_register = self.fill_register;

        while tmp_current_bit <= 56 {
            let b = (tmp_fill_register >> 56) as u8;
            if b != 0xff {
                self.data_buffer.push(b);
            } else {
                // escape 0xff here to avoid multiple scans of the same data
                self.data_buffer.push(0xff);
                self.data_buffer.push(0);
            }

            tmp_fill_register <<= 8;
            tmp_current_bit += 8;
        }

        self.current_bit = tmp_current_bit;
        self.fill_register = tmp_fill_register;
    }

    #[inline(always)]
    pub fn write(&mut self, val: u32, new_bits: u32) {
        debug_assert!(
            val < (1 << new_bits),
            "value {0} should fit into the number of {1} bits provided",
            val,
            new_bits
        );

        loop {
            // first see if everything fits in the current register
            let tmp_current_bit = self.current_bit;
            if new_bits <= tmp_current_bit {
                self.fill_register |= (val as u64) << (tmp_current_bit - new_bits);
                self.current_bit = tmp_current_bit - new_bits;
                return;
            }

            // if not, flush a byte off the top of the register and try again
            // there will always be room eventually since we have 64 bits and only allow
            // 32 bits to be written at a time.
            self.flush_bytes();
        }
    }

    pub fn pad(&mut self, fillbit: u8) {
        let mut offset = 1;
        while (self.current_bit & 7) != 0 {
            self.write(if (fillbit & offset) != 0 { 1 } else { 0 }, 1);
            offset <<= 1;
        }

        self.flush_bytes();

        debug_assert!(
            self.current_bit == 64,
            "there should be no remainder after padding"
        );
    }

    // flushes the data buffer while escaping all 0xff characters
    pub fn flush_with_escape<W: Write>(&mut self, w: &mut W) -> anyhow::Result<()> {
        // flush any remaining whole bytes
        self.flush_bytes();

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
