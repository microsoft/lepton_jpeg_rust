/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::mem;

use crate::helpers::has_ff;

pub struct BitWriter {
    data_buffer: Vec<u8>,
    fill_register: u64,
    current_bit: u32,
}

// use to write varying sized bits for coding JPEG. Escapes 0xff -> [0xff,0]
impl BitWriter {
    pub fn new(data_buffer: Vec<u8>) -> Self {
        return BitWriter {
            current_bit: 64,
            fill_register: 0,
            data_buffer,
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
            if has_ff(fill)
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

    pub fn ensure_space(&mut self, amount: usize) {
        if self.data_buffer.capacity() < amount {
            let len = self.data_buffer.len();
            self.data_buffer.reserve(amount - len);
        }
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

    pub fn amount_buffered(&self) -> usize {
        self.data_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Cursor;

    use crate::helpers::u32_bit_length;
    use crate::jpeg::bit_reader::BitReader;

    // write a test pattern with an escape and see if it matches
    #[test]
    fn write_simple() {
        let arr = [0x12, 0x34, 0x45, 0x67, 0x89, 0xff, 00, 0xee];

        let mut b = BitWriter::new(Vec::with_capacity(1024));

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
            let mut b = BitWriter::new(Vec::with_capacity(1024));
            for i in 1..2048 {
                b.write(i, u32_bit_length(i) as u32);
            }

            b.pad(0xff);

            buf = b.detach_buffer();
        }

        {
            let mut r = BitReader::new(Cursor::new(&buf));

            for i in 1..2048 {
                assert_eq!(i, r.read(u32_bit_length(i as u32) as u32).unwrap());
            }

            let mut pad = Some(0xff);
            r.read_and_verify_fill_bits(&mut pad).unwrap();
        }
    }

    /// verify the the bits roundtrip correctly with random bits
    #[test]
    fn roundtrip_randombits() {
        #[derive(Copy, Clone)]
        enum Action {
            Write(u16, u8),
            Pad(u8),
        }

        use rand::Rng;

        const ITERATIONS: usize = 10000;

        let mut rng = crate::helpers::get_rand_from_seed([0u8; 32]);
        let mut test_data = Vec::with_capacity(ITERATIONS);

        for _ in 0..ITERATIONS {
            let bits = rng.gen_range(0..=16);

            let t = rng.gen_range(0..=3);
            let v = match t {
                0 => 0,
                1 => 0xffff,
                _ => rng.gen_range(0..=65535),
            };

            let v = v & ((1 << bits) - 1);

            if rng.gen_range(0..100) == 0 {
                test_data.push(Action::Pad(0xff));
            } else {
                test_data.push(Action::Write(v as u16, bits as u8));
            }
        }
        test_data.push(Action::Pad(0xff));

        let buf;
        {
            let mut b = BitWriter::new(Vec::with_capacity(1024));
            for &i in &test_data {
                match i {
                    Action::Write(v, bits) => b.write(v as u32, bits as u32),
                    Action::Pad(fill) => b.pad(fill),
                }
            }

            buf = b.detach_buffer();
        }

        {
            let mut r = BitReader::new(Cursor::new(&buf));

            for a in test_data {
                match a {
                    Action::Write(code, numbits) => {
                        let expected_peek_byte = if numbits < 8 {
                            (code << (8 - numbits)) as u8
                        } else {
                            (code >> (numbits - 8)) as u8
                        };

                        let (peekcode, peekbits) = r.peek();
                        let num_valid_bits = peekbits.min(8).min(u32::from(numbits));

                        let mask = (0xff00 >> num_valid_bits) as u8;

                        assert_eq!(
                            expected_peek_byte & mask,
                            peekcode & mask,
                            "peek unexpected result"
                        );

                        assert_eq!(
                            code,
                            r.read(numbits as u32).unwrap(),
                            "read unexpected result"
                        );
                    }
                    Action::Pad(fill) => {
                        let mut pad = Some(fill);
                        r.read_and_verify_fill_bits(&mut pad).unwrap();
                    }
                }
            }
        }
    }
}
