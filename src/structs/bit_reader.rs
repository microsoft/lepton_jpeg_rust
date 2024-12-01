/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::io::{BufRead, Seek};

use likely_stable::unlikely;

use crate::helpers::has_ff;
use crate::lepton_error::{err_exit_code, ExitCode};
use crate::{jpeg_code, LeptonError};

// Implemenation of bit reader on top of JPEG data stream as read by a reader
pub struct BitReader<R> {
    inner: R,
    bits: u64,
    bits_left: u32,
    cpos: u32,
    eof: bool,
    start_offset: u64,
    truncated_ff: bool,
    read_ahead_bytes: u32,
}

impl<R: BufRead + Seek> BitReader<R> {
    pub fn get_stream_position(&mut self) -> i32 {
        self.undo_read_ahead();

        let pos: i32 = (self.inner.stream_position().unwrap() - self.start_offset)
            .try_into()
            .unwrap();

        if self.bits_left > 0 && !self.eof {
            if self.bits as u8 == 0xff && !self.truncated_ff {
                return pos - 2;
            } else {
                return pos - 1;
            }
        } else {
            return pos;
        }
    }

    pub fn new(mut inner: R) -> Self {
        let start_offset = inner.stream_position().unwrap();

        BitReader {
            inner: inner,
            bits: 0,
            bits_left: 0,
            cpos: 0,
            eof: false,
            start_offset,
            truncated_ff: false,
            read_ahead_bytes: 0,
        }
    }
}

impl<R: BufRead> BitReader<R> {
    #[inline(always)]
    pub fn read(&mut self, bits_to_read: u32) -> std::io::Result<u32> {
        let (mut diff, overflow) = self.bits_left.overflowing_sub(bits_to_read);

        if unlikely(overflow) {
            self.fill_register(bits_to_read)?;
            diff = self.bits_left - bits_to_read;
        }

        let mask = (1u32 << bits_to_read) - 1;
        self.bits_left = diff;
        return Ok(((self.bits >> diff) as u32) & mask);
    }

    #[inline(always)]
    pub fn advance(&mut self, bits: u32) {
        self.bits_left -= bits;
    }

    #[inline(always)]
    pub fn peek_byte(&mut self) -> Option<u8> {
        if unlikely(self.bits_left < 8) {
            let rh = self.read_ahead_bytes as usize;

            // no peeking if we are near EOF to avoid special casing
            let buffer = match self.inner.fill_buf() {
                Err(_) => return None,
                Ok(fb) => {
                    if fb.len() < rh + 8 {
                        return None;
                    } else {
                        // this is what we want to read
                        fb[rh..rh + 8].try_into().unwrap()
                    }
                }
            };

            let rh_new;
            (rh_new, self.bits, self.bits_left) =
                parse_8_byte_block(buffer, self.bits, self.bits_left);

            self.read_ahead_bytes += rh_new;
            if unlikely(self.bits_left < 8) {
                return None;
            }
        }

        return Some((self.bits >> (self.bits_left - 8)) as u8);
    }

    #[inline(always)]
    pub fn fill_register(&mut self, bits_to_read: u32) -> Result<(), std::io::Error> {
        // first consume the read_ahead bytes that we have now consumed
        // (otherwise we wouldn't have been called)
        self.inner.consume(self.read_ahead_bytes as usize);

        let fb = self.inner.fill_buf()?;

        // if we have 8 bytes and there is no 0xff in them, then we can just read the bits directly as big endian
        if fb.len() < 8 {
            self.read_ahead_bytes = 0;
            return self.fill_register_slow(bits_to_read);
        }

        let buffer = fb[..8].try_into().unwrap();
        (self.read_ahead_bytes, self.bits, self.bits_left) =
            parse_8_byte_block(buffer, self.bits, self.bits_left);

        // if we don't have enough bits here, it means that the parser choked on a 0xff marker,
        // let the slow path handle this since it is extremely rare and aborts the encoding
        if bits_to_read > self.bits_left {
            return self.fill_register_slow(bits_to_read);
        }

        return Ok(());
    }

    #[cold]
    fn bad_escape_code(b: u8) -> std::io::Error {
        // this was not an escaped 0xff which is the only thing we accept at this part of the decoding.
        //
        // verify_reset_code should have gotten called in all instances where there should be a reset code,
        // or at the end of the file we should have stopped decoding before we hit the end of file marker.
        //
        // Since we have no way of encoding these cases in our bitstream, we exit.
        return LeptonError::new(
            ExitCode::InvalidResetCode,
            format!("invalid reset {0:x} {1:x} code found in stream", 0xff, b).as_str(),
        )
        .into();
    }

    #[cold]
    fn fill_register_slow(&mut self, bits_to_read: u32) -> Result<(), std::io::Error> {
        loop {
            let fb = self.inner.fill_buf()?;
            if let &[b, ..] = fb {
                self.inner.consume(1);

                // 0xff is an escape code, if the next by is zero, then it is just a normal 0
                // otherwise it is a reset code, which should also be skipped
                if b == 0xff {
                    let mut buffer = [0u8];

                    if self.inner.read(&mut buffer)? == 0 {
                        // Handle case of truncation in the middle of an escape: Since we assume that everything passed the end
                        // is a 0, if the file ends with 0xFF, then we have to assume that this was
                        // an escaped 0xff. Don't mark as eof yet, since there are still the 8 bits to read.
                        self.bits = (self.bits << 8) | 0xff;
                        self.bits_left += 8;
                        self.truncated_ff = true;

                        // continue since we still might need to read more 0 bits
                    } else if buffer[0] == 0 {
                        // this was an escaped FF
                        self.bits = (self.bits << 8) | 0xff;
                        self.bits_left += 8;
                    } else {
                        return Err(Self::bad_escape_code(buffer[0]));
                    }
                } else {
                    self.bits = (self.bits << 8) | (b as u64);
                    self.bits_left += 8;
                }
            } else {
                // in case of a truncated file, we treat the rest of the file as zeros, but the
                // bits that were ok still get returned so that we get the partial last byte right
                // the caller periodically checks for EOF to see if it should stop encoding
                self.eof = true;
                self.bits_left += 8;
                self.bits <<= 8;

                // continue since we still might need to read more 0 bits
            }

            if self.bits_left >= bits_to_read {
                break;
            }
        }
        Ok(())
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
        self.undo_read_ahead();

        // if there are bits left, we need to see whether they
        // are 1s or zeros.

        if (self.bits_left) > 0 && !self.eof {
            let num_bits_to_read = self.bits_left;
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
                    let expected = u32::from(x) & all_one;
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
        self.undo_read_ahead();

        let mut h = [0u8; 2];
        self.inner.read_exact(&mut h)?;
        if h[0] != 0xff || h[1] != (jpeg_code::RST0 + (self.cpos as u8 & 7)) {
            return err_exit_code(
                ExitCode::InvalidResetCode,
                format!("invalid reset code {0:x} {1:x} found in stream", h[0], h[1]).as_str(),
            );
        }

        // start from scratch after RST
        self.cpos += 1;
        self.bits = 0;
        self.bits_left = 0;

        Ok(())
    }

    /// Retrieves the byte containing the next bit to be read in the stream, with only
    /// the bits that have already been read in it possibly set, and all the rest of the
    /// bits cleared.
    ///
    /// bitsAlreadyRead: the number of bits already read from the current byte
    /// byteBeingRead: the byte currently being read, with any bits not read from it yet cleared (0'ed)
    pub fn overhang(&mut self) -> (u8, u8) {
        self.undo_read_ahead();
        let bits_already_read = ((64 - self.bits_left) & 7) as u8; // already read bits in the current byte

        let mask = (((1 << bits_already_read) - 1) << (8 - bits_already_read)) as u8;

        return (bits_already_read, (self.bits as u8) & mask);
    }

    /// "puts back" read_ahead bits that were read ahead from the buffer but not consumed.
    ///
    /// This avoids special for many of the other non-speed-sensitive operations.
    ///
    /// After calling this method, we can be guaranteed that read_ahead_bytes is 0 and that
    /// the only bits that are left are part of the current byte.
    pub fn undo_read_ahead(&mut self) {
        while self.bits_left >= 8 && self.read_ahead_bytes > 0 {
            self.bits_left -= 8;
            if (self.bits & 0xff) == 0xff {
                self.read_ahead_bytes -= 2;
            } else {
                self.read_ahead_bytes -= 1;
            }
            self.bits >>= 8;
        }

        if self.read_ahead_bytes > 0 {
            self.inner.consume(self.read_ahead_bytes as usize);
            self.read_ahead_bytes = 0;
        }
    }
}

#[inline(always)]
fn parse_8_byte_block(block: [u8; 8], mut bits: u64, mut bits_left: u32) -> (u32, u64, u32) {
    #[cold]
    fn parse_cold(bits_left: &mut u32, v: &mut u64, bits: &mut u64) -> i32 {
        let mut i = 0;

        while *bits_left < 56 && i <= 6 {
            let masked = *v & 0xff;
            if masked == 0xff {
                if *v & 0xff00 != 0 {
                    // this is not an escaped 0xff, let the slow path handle it
                    break;
                }

                *bits = (*bits << 8) | 0xff;
                i += 2;
                *v >>= 16;
            } else {
                *bits = (*bits << 8) | masked;
                *v >>= 8;
                i += 1;
            }

            *bits_left += 8;
        }
        i
    }

    let mut v: u64 = u64::from_le_bytes(block);
    if has_ff(v) {
        let i = parse_cold(&mut bits_left, &mut v, &mut bits);

        (i as u32, bits, bits_left)
    } else {
        v = v.to_be();

        // only fill 63 bits not 64 to avoid having to special case
        // of self.bits << 64 which is a nop
        let bytes_to_read = (63 - bits_left) / 8;

        bits = bits << (bytes_to_read * 8) | v >> (64 - bytes_to_read * 8);
        bits_left += bytes_to_read * 8;

        (bytes_to_read, bits, bits_left)
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
