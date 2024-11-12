/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::io::Read;

use crate::lepton_error::{err_exit_code, ExitCode};
use crate::{jpeg_code, LeptonError};

// Implemenation of bit reader on top of JPEG data stream as read by a reader
pub struct BitReader<R> {
    inner: R,
    bits: u64,
    num_bits: u8,
    cpos: u32,
    offset: i32, // offset of next bit that we will read in the file
    eof: bool,
    prev_offset: i32, // position of last escape. used to adjust the current position.
    last_byte_read: u8,
}

impl<R: Read> BitReader<R> {
    pub fn new(inner: R) -> Self {
        BitReader {
            inner: inner,
            bits: 0,
            num_bits: 0,
            cpos: 0,
            offset: 0,
            eof: false,
            prev_offset: 0,
            last_byte_read: 0,
        }
    }

    #[inline(always)]
    pub fn read(&mut self, bits_to_read: u8) -> std::io::Result<u16> {
        if bits_to_read == 0 {
            return Ok(0);
        }

        if self.num_bits < bits_to_read {
            self.fill_register(bits_to_read)?;
        }

        let retval = (self.bits >> (64 - bits_to_read)) as u16;
        self.bits <<= bits_to_read as usize;
        self.num_bits -= bits_to_read;
        return Ok(retval);
    }

    #[inline(always)]
    pub fn peek(&self) -> (u8, u8) {
        return ((self.bits >> 56) as u8, self.num_bits);
    }

    #[inline(always)]
    pub fn advance(&mut self, bits: u8) {
        self.num_bits -= bits;
        self.bits <<= bits;
    }

    #[inline(always)]
    pub fn fill_register(&mut self, bits_to_read: u8) -> Result<(), std::io::Error> {
        while self.num_bits < bits_to_read {
            let mut buffer = [0u8];
            if self.inner.read(&mut buffer)? == 0 {
                return self.fill_register_slow(None, bits_to_read);
            } else if buffer[0] == 0xff {
                return self.fill_register_slow(Some(buffer[0]), bits_to_read);
            } else {
                self.prev_offset = self.offset;
                self.offset += 1;
                self.bits |= (buffer[0] as u64) << (56 - self.num_bits);
                self.num_bits += 8;
                self.last_byte_read = buffer[0];
            }
        }
        return Ok(());
    }

    #[cold]
    fn fill_register_slow(
        &mut self,
        mut byte_read: Option<u8>,
        bits_to_read: u8,
    ) -> Result<(), std::io::Error> {
        loop {
            if let Some(b) = byte_read {
                // 0xff is an escape code, if the next by is zero, then it is just a normal 0
                // otherwise it is a reset code, which should also be skipped
                if b == 0xff {
                    let mut buffer = [0u8];

                    if self.inner.read(&mut buffer)? == 0 {
                        // Handle case of truncation: Since we assume that everything passed the end
                        // is a 0, if the file ends with 0xFF, then we have to assume that this was
                        // an escaped 0xff. Don't mark as eof yet, since there are still the 8 bits to read.
                        self.prev_offset = self.offset;
                        self.offset += 1; // we only have 1 byte to advance in the stream and don't want to go past EOF.
                        self.bits |= (0xff as u64) << (56 - self.num_bits);
                        self.num_bits += 8;
                        self.last_byte_read = 0xff;

                        // continue since we still might need to read more 0 bits
                    } else if buffer[0] == 0 {
                        // this was an escaped FF
                        self.prev_offset = self.offset;
                        self.offset += 2;
                        self.bits |= (0xff as u64) << (56 - self.num_bits);
                        self.num_bits += 8;
                        self.last_byte_read = 0xff;
                    } else {
                        // verify_reset_code should get called in all instances where there should be a reset code. If we find one that
                        // is not where it is supposed to be, then we would fail to roundtrip the reset code, so just fail.
                        return Err(LeptonError::new(
                            ExitCode::InvalidResetCode,
                            format!(
                                "invalid reset {0:x} {1:x} code found in stream at offset {2}",
                                0xff, buffer[0], self.offset
                            )
                            .as_str(),
                        )
                        .into());
                    }
                } else {
                    self.prev_offset = self.offset;
                    self.offset += 1;
                    self.bits |= (b as u64) << (56 - self.num_bits);
                    self.num_bits += 8;
                    self.last_byte_read = b;
                }
            } else {
                // in case of a truncated file, we treat the rest of the file as zeros, but the
                // bits that were ok still get returned so that we get the partial last byte right
                // the caller periodically checks for EOF to see if it should stop encoding
                self.eof = true;
                self.num_bits += 8;
                self.prev_offset = self.offset;
                self.last_byte_read = 0;

                // continue since we still might need to read more 0 bits
            }

            if self.num_bits >= bits_to_read {
                break;
            }

            let mut buffer = [0u8];
            if self.inner.read(&mut buffer)? == 0 {
                byte_read = None;
            } else {
                byte_read = Some(buffer[0]);
            }
        }
        Ok(())
    }

    pub fn get_stream_position(&self) -> i32 {
        // if there are still bits left, then we should be referring to the previous offset
        if self.num_bits > 0 {
            // if we still have bits, we need to go back to the last offset
            return self.prev_offset;
        } else {
            return self.offset;
        }
    }

    pub fn is_eof(&mut self) -> bool {
        return self.eof;
    }

    /// used to verify whether this image is using 1s or 0s as fill bits.
    /// Returns whether the fill bit was 1 or so or unknown (None)
    pub fn read_and_verify_fill_bits(
        &mut self,
        pad_bit: &mut Option<u8>,
    ) -> Result<(), LeptonError> {
        // if there are bits left, we need to see whether they
        // are 1s or zeros.

        if self.num_bits > 0 && !self.eof {
            let num_bits_to_read = self.num_bits;
            let actual = self.read(num_bits_to_read)?;
            let all_one = (1 << num_bits_to_read) - 1;

            match *pad_bit {
                None => {
                    if actual == 0 {
                        *pad_bit = Some(0);
                    } else if actual == all_one {
                        *pad_bit = Some(0xff);
                    } else {
                        return err_exit_code(
                            ExitCode::InvalidPadding,
                            format!(
                                "inconsistent pad bits num_bits={0} pattern={1:b}",
                                num_bits_to_read, actual
                            )
                            .as_str(),
                        );
                    }
                }
                Some(x) => {
                    // if we already saw a padding, then it should match
                    let expected = u16::from(x) & all_one;
                    if actual != expected {
                        return err_exit_code(ExitCode::InvalidPadding, format!("padding of {0} bits should be set to 1 actual={1:b} expected={2:b}", num_bits_to_read, actual, expected).as_str());
                    }
                }
            }
        }

        return Ok(());
    }

    pub fn verify_reset_code(&mut self) -> Result<(), LeptonError> {
        // we reached the end of a MCU, so we need to find a reset code and the huffman codes start get padded out, but the spec
        // doesn't specify whether the padding should be 1s or 0s, so we ensure that at least the file is consistant so that we
        // can recode it again just by remembering the pad bit.

        let mut h = [0u8; 2];
        self.inner.read_exact(&mut h)?;
        if h[0] != 0xff || h[1] != (jpeg_code::RST0 + (self.cpos as u8 & 7)) {
            return err_exit_code(
                ExitCode::InvalidResetCode,
                format!(
                    "invalid reset code {0:x} {1:x} found in stream at offset {2}",
                    h[0], h[1], self.offset
                )
                .as_str(),
            );
        }

        // start from scratch after RST
        self.cpos += 1;
        self.offset += 2;
        self.prev_offset = self.offset;
        self.bits = 0;
        self.num_bits = 0;

        Ok(())
    }

    /// Retrieves the byte containing the next bit to be read in the stream, with only
    /// the bits that have already been read in it possibly set, and all the rest of the
    /// bits cleared.
    ///
    /// bitsAlreadyRead: the number of bits already read from the current byte
    /// byteBeingRead: the byte currently being read, with any bits not read from it yet cleared (0'ed)
    pub fn overhang(&self) -> (u8, u8) {
        let bits_already_read = ((64 - self.num_bits) & 7) as u8; // already read bits in the current byte

        let mask = (((1 << bits_already_read) - 1) << (8 - bits_already_read)) as u8;

        return (bits_already_read, self.last_byte_read & mask);
    }
}

#[cfg(test)]
use std::io::Cursor;

// test reading a simple bit pattern with an escaped 0xff inside it.
#[test]
fn read_simple() {
    let arr = [0x12 as u8, 0x34, 0x45, 0x67, 0x89, 0xff, 00, 0xee];

    let mut b = BitReader::new(Cursor::new(&arr));

    assert_eq!(1, b.read(4).unwrap());
    assert_eq!((4, 0x10), b.overhang());
    assert_eq!(0, b.get_stream_position());

    assert_eq!(2, b.read(4).unwrap());
    assert_eq!((0, 0), b.overhang()); // byte is aligned should be no overhang
    assert_eq!(1, b.get_stream_position());

    assert_eq!(3, b.read(4).unwrap());
    assert_eq!(4, b.read(4).unwrap());
    assert_eq!(4, b.read(4).unwrap());
    assert_eq!(0x56, b.read(8).unwrap()); // 8 bits between 0x45 and 0x67
    assert_eq!(0x78, b.read(8).unwrap());

    assert_eq!(0x9f, b.read(8).unwrap());
    assert_eq!((4, 0xf0), b.overhang());
    assert_eq!(5, b.get_stream_position()); // should be at the beginning of the escape code

    assert_eq!(0xfe, b.read(8).unwrap());
    assert_eq!((4, 0xe0), b.overhang());
    assert_eq!(7, b.get_stream_position()); // now we are after the escape code

    assert_eq!(0xe, b.read(4).unwrap());
    assert_eq!((0, 0), b.overhang());
    assert_eq!(8, b.get_stream_position()); // now we read everything and should be at the end of the stream

    // read an empty byte passed the end of the stream.. should be zero and trigger EOF
    assert_eq!(0, b.read(8).unwrap());
    assert_eq!(true, b.is_eof());
    assert_eq!(8, b.get_stream_position()); // still at the same position
}

// what happens when a file has 0xff as the last character (assume that it is an escaped 0xff)
#[test]
fn read_truncate_ff() {
    let arr = [0x12 as u8, 0xff];

    let mut b = BitReader::new(Cursor::new(&arr));

    assert_eq!(0, b.get_stream_position());

    assert_eq!(0x1, b.read(4).unwrap());
    assert_eq!(0, b.get_stream_position());

    assert_eq!(0x2f, b.read(8).unwrap());
    assert_eq!((4, 0xf0), b.overhang());
    assert_eq!(1, b.get_stream_position());

    // 4 bits left, not EOF yet
    assert_eq!(false, b.is_eof());

    assert_eq!(0xf, b.read(4).unwrap());
    assert_eq!(false, b.is_eof()); // now we are at the end really
    assert_eq!(2, b.get_stream_position());

    assert_eq!(0, b.read(4).unwrap());
    assert_eq!(true, b.is_eof());
    assert_eq!(2, b.get_stream_position());
}
