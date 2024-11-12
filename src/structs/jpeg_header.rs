/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

/*
Copyright (c) 2006...2016, Matthias Stirner and HTW Aalen University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use std::io::Read;

use crate::consts::JPegType;
use crate::enabled_features::EnabledFeatures;
use crate::helpers::*;
use crate::jpeg_code;
use crate::lepton_error::{err_exit_code, AddContext, ExitCode, Result};
use crate::structs::component_info::ComponentInfo;
use crate::structs::lepton_header::LeptonHeader;
use crate::structs::quantization_tables::QuantizationTables;
use crate::structs::truncate_components::TruncateComponents;

#[derive(Copy, Clone, Debug)]
pub struct HuffCodes {
    pub c_val: [u16; 256],
    pub c_len: [u16; 256],
    pub c_len_plus_s: [u8; 256],
    pub c_val_shift_s: [u32; 512],
    pub max_eob_run: u16,
}

impl Default for HuffCodes {
    fn default() -> Self {
        HuffCodes {
            c_val: [0; 256],
            c_len: [0; 256],
            c_len_plus_s: [0; 256],
            c_val_shift_s: [0; 512],
            max_eob_run: 0,
        }
    }
}

impl HuffCodes {
    /// Constructs from the format encoded by JPEG
    ///
    /// Tree consists of a 16 byte table with the number of codes for each bit length,
    /// followed by the actual codes for that length appended together.
    pub fn construct_from_segment(segment: &[u8]) -> Result<Self> {
        let clen_offset = 0;
        let cval_offset = 16;

        let mut hc = HuffCodes::default();

        // creating huffman-codes
        let mut k = 0;
        let mut code = 0;

        // symbol-value of code is its position in the table
        for i in 0..16 {
            ensure_space(segment, clen_offset, i + 1).context()?;

            let mut j = 0;
            while j < segment[clen_offset + (i & 0xff)] {
                ensure_space(segment, cval_offset, k + 1).context()?;

                let len = (1 + i) as u16;

                if u32::from(code) >= (1u32 << len) {
                    return err_exit_code(
                        ExitCode::UnsupportedJpeg,
                        "invalid huffman code layout, too many codes for a given length",
                    );
                }

                hc.c_len[usize::from(segment[cval_offset + (k & 0xff)])] = len;
                hc.c_val[usize::from(segment[cval_offset + (k & 0xff)])] = code;

                if code == 65535 {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "huffman code too large");
                }

                k += 1;
                code += 1;
                j += 1;
            }

            code = code << 1;
        }

        hc.post_initialize();

        Ok(hc)
    }

    /// Code to run after initializing c_len and c_val
    /// Lookup tables used for fast encoding since we already
    /// know the length of the code and the value when we write
    /// the code + bits to the bitstream
    fn post_initialize(&mut self) {
        for i in 0..256 {
            let s = i & 0xf;
            self.c_len_plus_s[i] = (self.c_len[i] + (s as u16)) as u8;
            self.c_val_shift_s[i] = (self.c_val[i] as u32) << s;

            // calculate the value for negative coefficients, which compensates for the sign bit
            self.c_val_shift_s[i + 256] = ((self.c_val[i] as u32) << s) | ((1u32 << s) - 1);
        }

        // find out eobrun (runs of all zero blocks) max value. This is used encoding/decoding progressive files.
        //
        // G.1.2.2 of the spec specifies that there are 15 huffman codes
        // reserved for encoding long runs of up to 32767 empty blocks.
        // Here we figure out what the largest code that could possibly
        // be encoded by this table is so that we don't exceed it when
        // we reencode the file.
        self.max_eob_run = 0;

        let mut i: i32 = 14;
        while i >= 0 {
            if self.c_len[((i << 4) & 0xff) as usize] > 0 {
                self.max_eob_run = ((2 << i) - 1) as u16;
                break;
            }

            i -= 1;
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct HuffTree {
    pub node: [[u16; 2]; 256],
    pub peek_code: [(u8, u8); 256],
}

impl Default for HuffTree {
    fn default() -> Self {
        HuffTree {
            node: [[0; 2]; 256],
            peek_code: [(0, 0); 256],
        }
    }
}

impl HuffTree {
    /// construct the huffman tree codes from the HuffCodes as a source
    pub fn construct_hufftree(hc: &HuffCodes, accept_invalid_dht: bool) -> Result<Self> {
        let mut ht = HuffTree::default();

        let mut nextfree = 1;
        for i in 0..256 {
            // reset current node
            let mut node = 0;

            // go through each code & store path
            if hc.c_len[i] > 0 {
                let mut j = hc.c_len[i] - 1;
                while j > 0 {
                    if node <= 0xff {
                        if bitn(hc.c_val[i], j) == 1 {
                            if ht.node[node][1] == 0 {
                                ht.node[node][1] = nextfree;
                                nextfree += 1;
                            }

                            node = usize::from(ht.node[node][1]);
                        } else {
                            if ht.node[node][0] == 0 {
                                ht.node[node][0] = nextfree;
                                nextfree += 1;
                            }

                            node = usize::from(ht.node[node][0]);
                        }
                    } else {
                        // we accept any .lep file that was encoded this way
                        if !accept_invalid_dht {
                            return err_exit_code(
                                ExitCode::UnsupportedJpeg,
                                "Huffman table out of space",
                            );
                        }
                    }

                    j -= 1;
                }
            }

            if node <= 0xff {
                // last link is number of targetvalue + 256
                if hc.c_len[i] > 0 {
                    if bitn(hc.c_val[i], 0) == 1 {
                        ht.node[node][1] = (i + 256) as u16;
                    } else {
                        ht.node[node][0] = (i + 256) as u16;
                    }
                }
            } else {
                // we accept any .lep file that was encoded this way
                if !accept_invalid_dht {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "Huffman table out of space");
                }
            }
        }
        for x in &mut ht.node {
            if x[0] == 0 {
                x[0] = 0xffff;
            }
            if x[1] == 0 {
                x[1] = 0xffff;
            }
        }
        // initial value for next free place

        // work through every code creating links between the nodes (represented through ints)

        // for every illegal code node, store 0xffff we should never get here, but it will avoid an infinite loop in the case of a bug

        // precalculate decoding peeking into the stream. This lets us quickly decode
        // small code without jumping through the node table
        for peekbyte in 0..256 {
            let mut node = 0;
            let mut len: u8 = 0;

            while node < 256 && len <= 7 {
                node = ht.node[usize::from(node)][(peekbyte >> (7 - len)) & 0x1];

                len += 1;
            }

            if node == 0xffff || node < 256 {
                // invalid code or code was too long to fit, so just say it requireds 256 bits
                // so we will take the long path to decode it
                ht.peek_code[peekbyte] = (0, 0xff);
            } else {
                ht.peek_code[peekbyte] = ((node - 256) as u8, len);
            }
        }
        Ok(ht)
    }
}

#[derive(Debug, Clone)]
pub struct JPegHeader {
    pub q_tables: [[u16; 64]; 4],     // quantization tables 4 x 64
    h_codes: [[HuffCodes; 4]; 2],     // huffman codes (access via get_huff_xx_codes)
    h_trees: [[HuffTree; 4]; 2],      // huffman decoding trees (access via get_huff_xx_tree)
    pub ht_set: [[u8; 4]; 2],         // 1 if huffman table is set
    pub cmp_info: [ComponentInfo; 4], // components
    pub cmpc: usize,                  // component count
    pub img_width: i32,               // width of image
    pub img_height: i32,              // height of image

    pub jpeg_type: JPegType,
    pub sfhm: i32, // max horizontal sample factor
    pub sfvm: i32, // max verical sample factor
    pub mcuv: i32, // mcus per line
    pub mcuh: i32, // mcus per collumn
    pub mcuc: i32, // count of mcus

    pub rsti: i32,          // restart interval
    pub cs_cmpc: usize,     // component count in current scan
    pub cs_cmp: [usize; 4], // component numbers in current scan

    // variables: info about current scan
    pub cs_from: u8, // begin - band of current scan ( inclusive )
    pub cs_to: u8,   // end - band of current scan ( inclusive )
    pub cs_sah: u8,  // successive approximation bit pos high
    pub cs_sal: u8,  // successive approximation bit pos low
}

pub struct JPegEncodingInfo {
    pub jpeg_header: JPegHeader,
    pub truncate_components: TruncateComponents,

    /// A list containing one entry for each scan segment.  Each entry contains the number of restart intervals
    /// within the corresponding scan segment.
    pub rst_cnt: Vec<i32>,

    /// the mask for padding out the bitstream when we get to the end of a reset block
    pub pad_bit: Option<u8>,

    pub rst_cnt_set: bool,

    /// count of scans encountered so far
    pub scnc: usize,
}

impl JPegEncodingInfo {
    pub fn new(lh: &LeptonHeader) -> Self {
        JPegEncodingInfo {
            jpeg_header: lh.jpeg_header.clone(),
            truncate_components: lh.truncate_components.clone(),
            rst_cnt: lh.rst_cnt.clone(),
            pad_bit: lh.pad_bit,
            rst_cnt_set: lh.rst_cnt_set,
            scnc: lh.scnc,
        }
    }
}

enum ParseSegmentResult {
    Continue,
    EOI,
    SOS,
}

impl Default for JPegHeader {
    fn default() -> Self {
        return JPegHeader {
            q_tables: [[0; 64]; 4],
            h_codes: [[HuffCodes::default(); 4]; 2],
            h_trees: [[HuffTree::default(); 4]; 2],
            ht_set: [[0; 4]; 2],
            cmp_info: [
                ComponentInfo::default(),
                ComponentInfo::default(),
                ComponentInfo::default(),
                ComponentInfo::default(),
            ],
            cmpc: 0,
            img_width: 0,
            img_height: 0,
            jpeg_type: JPegType::Unknown,
            sfhm: 0,
            sfvm: 0,
            mcuv: 0,
            mcuh: 0,
            mcuc: 0,
            rsti: 0,
            cs_cmpc: 0,
            cs_from: 0,
            cs_to: 0,
            cs_sah: 0,
            cs_sal: 0,
            cs_cmp: [0; 4],
        };
    }
}

impl JPegHeader {
    pub fn get_huff_dc_codes(&self, cmp: usize) -> &HuffCodes {
        &self.h_codes[0][usize::from(self.cmp_info[cmp].huff_dc)]
    }

    pub fn get_huff_dc_tree(&self, cmp: usize) -> &HuffTree {
        &self.h_trees[0][usize::from(self.cmp_info[cmp].huff_dc)]
    }

    pub fn get_huff_ac_codes(&self, cmp: usize) -> &HuffCodes {
        &self.h_codes[1][usize::from(self.cmp_info[cmp].huff_ac)]
    }

    pub fn get_huff_ac_tree(&self, cmp: usize) -> &HuffTree {
        &self.h_trees[1][usize::from(self.cmp_info[cmp].huff_ac)]
    }

    /// Parses header for imageinfo
    pub fn parse<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> Result<bool> {
        // header parser loop
        loop {
            match self
                .parse_next_segment(reader, enabled_features)
                .context()?
            {
                ParseSegmentResult::EOI => {
                    return Ok(false);
                }
                ParseSegmentResult::SOS => {
                    break;
                }
                _ => {}
            }
        }

        // check if information is complete
        if self.cmpc == 0 {
            return err_exit_code(
                ExitCode::UnsupportedJpeg,
                "header contains incomplete information",
            );
        }

        for cmp in 0..self.cmpc {
            if (self.cmp_info[cmp].sfv == 0)
                || (self.cmp_info[cmp].sfh == 0)
                || (self.q_tables[usize::from(self.cmp_info[cmp].q_table_index)][0] == 0)
                || (self.jpeg_type == JPegType::Unknown)
            {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    "header contains incomplete information (components)",
                );
            }
        }

        // do all remaining component info calculations
        for cmp in 0..self.cmpc {
            if self.cmp_info[cmp].sfh > self.sfhm {
                self.sfhm = self.cmp_info[cmp].sfh;
            }

            if self.cmp_info[cmp].sfv > self.sfvm {
                self.sfvm = self.cmp_info[cmp].sfv;
            }
        }

        self.mcuv = (1.0 * self.img_height as f64 / (8.0 * self.sfhm as f64)).ceil() as i32;
        self.mcuh = (1.0 * self.img_width as f64 / (8.0 * self.sfvm as f64)).ceil() as i32;
        self.mcuc = self.mcuv * self.mcuh;

        for cmp in 0..self.cmpc {
            self.cmp_info[cmp].mbs = self.cmp_info[cmp].sfv * self.cmp_info[cmp].sfh;
            self.cmp_info[cmp].bcv = self.mcuv * self.cmp_info[cmp].sfh;
            self.cmp_info[cmp].bch = self.mcuh * self.cmp_info[cmp].sfv;
            self.cmp_info[cmp].bc = self.cmp_info[cmp].bcv * self.cmp_info[cmp].bch;
            self.cmp_info[cmp].ncv = (1.0
                * self.img_height as f64
                * (self.cmp_info[cmp].sfh as f64 / (8.0 * self.sfhm as f64)))
                .ceil() as i32;
            self.cmp_info[cmp].nch = (1.0
                * self.img_width as f64
                * (self.cmp_info[cmp].sfv as f64 / (8.0 * self.sfvm as f64)))
                .ceil() as i32;
            self.cmp_info[cmp].nc = self.cmp_info[cmp].ncv * self.cmp_info[cmp].nch;
        }

        // decide components' statistical ids
        if self.cmpc <= 3 {
            for cmp in 0..self.cmpc {
                self.cmp_info[cmp].sid = cmp as i32;
            }
        } else {
            for cmp in 0..self.cmpc {
                self.cmp_info[cmp].sid = 0;
            }
        }

        return Ok(true);
    }

    /// verifies that the huffman tables for the given types are present for the current scan, and if not, return an error
    pub fn verify_huffman_table(&self, dc_present: bool, ac_present: bool) -> Result<()> {
        for icsc in 0..self.cs_cmpc {
            let icmp = self.cs_cmp[icsc];

            if dc_present && self.ht_set[0][self.cmp_info[icmp].huff_dc as usize] == 0 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    format!("DC huffman table missing for component {0}", icmp).as_str(),
                );
            } else if ac_present && self.ht_set[1][self.cmp_info[icmp].huff_ac as usize] == 0 {
                return err_exit_code(
                    ExitCode::UnsupportedJpeg,
                    format!("AC huffman table missing for component {0}", icmp).as_str(),
                );
            }
        }

        Ok(())
    }

    // returns true we should continue parsing headers or false if we hit SOS and should stop
    fn parse_next_segment<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> Result<ParseSegmentResult> {
        let mut header = [0u8; 4];

        if reader.read(&mut header[0..1]).context()? == 0 {
            // didn't get an EOI
            return Ok(ParseSegmentResult::EOI);
        }

        if header[0] != 0xff {
            return err_exit_code(ExitCode::UnsupportedJpeg, "invalid header encountered");
        }

        reader.read_exact(&mut header[1..2]).context()?;
        if header[1] == jpeg_code::EOI {
            return Ok(ParseSegmentResult::EOI);
        }

        // now read the second two bytes so we can get the size of the segment
        reader.read_exact(&mut header[2..]).context()?;

        let mut segment_data = Vec::new();

        let segment_size = b_short(header[2], header[3]);
        if segment_size < 2 {
            return err_exit_code(ExitCode::UnsupportedJpeg, "segment is too short");
        }

        segment_data.resize(usize::from(segment_size) - 2, 0);

        reader.read_exact(&mut segment_data).context()?;

        let mut hpos = 0;
        let len = segment_data.len();

        let segment = &segment_data[..];

        let btype = header[1];
        match btype
        {
            jpeg_code::DHT => // DHT segment
            {
                // build huffman trees & codes
                while hpos < len
                {
                    let lval = usize::from(lbits(segment[hpos], 4));
                    let rval = usize::from(rbits(segment[hpos], 4));
                    if (lval >= 2) || (rval >= 4)
                    {
                        break;
                    }

                    hpos+=1;

                    // build huffman codes & trees
                    self.h_codes[lval][rval] = HuffCodes::construct_from_segment(&segment[hpos..]).context()?;
                    self.h_trees[lval][rval] = HuffTree::construct_hufftree(&self.h_codes[lval][rval], enabled_features.accept_invalid_dht).context()?;
                    self.ht_set[lval][rval] = 1;

                    let mut skip = 16;

                    ensure_space(segment,hpos, 16)?;

                    for i in 0..16
                    {
                        skip += usize::from(segment[hpos + i]);
                    }

                    hpos += skip;
                }

                if hpos != len
                {
                    // if we get here, something went wrong
                    return err_exit_code(ExitCode::UnsupportedJpeg,"size mismatch in dht marker");
                }
            }

            jpeg_code::DQT => // DQT segment
            {
                // copy quantization tables to internal memory
                while hpos < len
                {
                    let lval = usize::from(lbits(segment[hpos], 4));
                    let rval = usize::from(rbits(segment[hpos], 4));
                    if lval >= 2 || rval >= 4
                    {
                        return err_exit_code(ExitCode::UnsupportedJpeg,"DQT has invalid index");
                    }

                    hpos+=1;
                    if lval == 0
                    {
                        ensure_space(segment,hpos, 64).context()?;

                        // 8 bit precision
                        for i in 0..64
                        {
                            self.q_tables[rval][i] = segment[hpos + i] as u16;
                            if self.q_tables[rval][i] == 0
                            {
                                if enabled_features.reject_dqts_with_zeros
                                {
                                    return err_exit_code(ExitCode::UnsupportedJpegWithZeroIdct0,"DQT has zero value");
                                }
                                else {
                                    break;
                                }
                            }
                        }

                        hpos += 64;
                    }
                    else
                    {
                        ensure_space(segment,hpos, 128).context()?;

                        // 16 bit precision
                        for i in 0..64
                        {
                            self.q_tables[rval][i] = b_short(segment[hpos + (2 * i)], segment[hpos + (2 * i) + 1]);
                            if self.q_tables[rval][i] == 0
                            {
                                if enabled_features.reject_dqts_with_zeros
                                {
                                    return err_exit_code(ExitCode::UnsupportedJpegWithZeroIdct0,"DQT has zero value");
                                }
                                else {
                                    break;
                                }
                            }
                        }

                        hpos += 128;
                    }
                }

                if hpos != len
                {
                    // if we get here, something went wrong
                    return err_exit_code(ExitCode::UnsupportedJpeg, "size mismatch in dqt marker");
                }

            }

            jpeg_code::DRI =>
            {  // DRI segment
                // define restart interval
                ensure_space(segment,hpos, 2).context()?;
                self.rsti = b_short(segment[hpos], segment[hpos + 1]) as i32;
            }

            jpeg_code::SOS => // SOS segment
            {
                // prepare next scan
                ensure_space(segment,hpos, 1).context()?;

                self.cs_cmpc = usize::from(segment[hpos]);

                if self.cs_cmpc == 0
                {
                    return err_exit_code( ExitCode::UnsupportedJpeg, "zero components in scan");
                }

                if self.cs_cmpc > self.cmpc
                {
                    return err_exit_code( ExitCode::UnsupportedJpeg, format!("{0} components in scan, only {1} are allowed", self.cs_cmpc, self.cmpc).as_str());
                }

                hpos+=1;
                for i in 0..self.cs_cmpc
                {
                    ensure_space(segment,hpos, 2).context()?;

                    let mut cmp = 0;
                    while cmp < self.cmpc && segment[hpos] != self.cmp_info[cmp].jid
                    {
                        cmp+=1;
                    }

                    if cmp == self.cmpc
                    {
                        return err_exit_code(ExitCode::UnsupportedJpeg, "component id mismatch in start-of-scan");
                    }

                    self.cs_cmp[i] = cmp;
                    self.cmp_info[cmp].huff_dc = lbits(segment[hpos + 1], 4);
                    self.cmp_info[cmp].huff_ac = rbits(segment[hpos + 1], 4);

                    if (self.cmp_info[cmp].huff_dc == 0xff) || (self.cmp_info[cmp].huff_dc >= 4) ||
                        (self.cmp_info[cmp].huff_ac == 0xff) || (self.cmp_info[cmp].huff_ac >= 4)
                    {
                        return err_exit_code(ExitCode::UnsupportedJpeg,"huffman table number mismatch");
                    }

                    hpos += 2;
                }

                ensure_space(segment,hpos, 3).context()?;

                self.cs_from = segment[hpos + 0];
                self.cs_to = segment[hpos + 1];
                self.cs_sah = lbits(segment[hpos + 2], 4);
                self.cs_sal = rbits(segment[hpos + 2], 4);

                // check for errors
                if (self.cs_from > self.cs_to) || (self.cs_from > 63) || (self.cs_to > 63)
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg,"spectral selection parameter out of range");
                }

                if (self.cs_sah >= 12) || (self.cs_sal >= 12)
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "successive approximation parameter out of range");
                }

                return Ok(ParseSegmentResult::SOS);
            }

            jpeg_code::SOF0| // SOF0 segment, coding process: baseline DCT
            jpeg_code::SOF1| // SOF1 segment, coding process: extended sequential DCT
            jpeg_code::SOF2 =>  // SOF2 segment, coding process: progressive DCT
            {
                if self.jpeg_type != JPegType::Unknown
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "image cannot have multiple SOF blocks");
                }

                // set JPEG coding type
                if btype == jpeg_code::SOF2
                {
                    self.jpeg_type = JPegType::Progressive;
                }
                else
                {
                    self.jpeg_type = JPegType::Sequential;
                }

                ensure_space(segment,hpos, 6).context()?;

                // check data precision, only 8 bit is allowed
                let lval = segment[hpos];
                if lval != 8
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, format!("{0} bit data precision is not supported", lval).as_str());
                }

                // image size, height & component count
                self.img_height = i32::from(b_short(segment[hpos + 1], segment[hpos + 2]));
                self.img_width = i32::from(b_short(segment[hpos + 3], segment[hpos + 4]));

                if self.img_height == 0 || self.img_width == 0
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, "image dimensions can't be zero");
                }

                if self.img_height > enabled_features.max_jpeg_height || self.img_width > enabled_features.max_jpeg_width
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, format!("image dimensions larger than {0}x{1}", enabled_features.max_jpeg_width, enabled_features.max_jpeg_height).as_str());
                }

                self.cmpc = usize::from(segment[hpos + 5]);

                if self.cmpc > 4
                {
                    return err_exit_code(ExitCode::UnsupportedJpeg, format!("image has {0} components, max 4 are supported", self.cmpc).as_str());
                }

                hpos += 6;

                // components contained in image
                for cmp in  0..self.cmpc
                {
                    ensure_space(segment,hpos, 3).context()?;

                    self.cmp_info[cmp].jid = segment[hpos];
                    self.cmp_info[cmp].sfv = lbits(segment[hpos + 1], 4) as i32;
                    self.cmp_info[cmp].sfh = rbits(segment[hpos + 1], 4) as i32;

                    if self.cmp_info[cmp].sfv > 2 || self.cmp_info[cmp].sfh > 2
                    {
                        return err_exit_code(ExitCode::SamplingBeyondTwoUnsupported, "Sampling type beyond to not supported");
                    }

                    let quantization_table_value = segment[hpos + 2];
                    if usize::from(quantization_table_value) >= self.q_tables.len()
                    {
                        return err_exit_code(ExitCode::UnsupportedJpeg,"quantizationTableValue too big");
                    }

                    self.cmp_info[cmp].q_table_index = quantization_table_value;
                    hpos += 3;
                }

            }

            0xC3 => // SOF3 segment
                {
                    // coding process: lossless sequential
                    return err_exit_code(ExitCode::UnsupportedJpeg,"sof3 marker found, image is coded lossless");
                }

            0xC5 => // SOF5 segment
                {
                    // coding process: differential sequential DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg,"sof5 marker found, image is coded diff. sequential");
                }

            0xC6 => // SOF6 segment
                {
                    // coding process: differential progressive DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg,"sof6 marker found, image is coded diff. progressive");
                }

            0xC7 => // SOF7 segment
                {
                    // coding process: differential lossless
                    return err_exit_code(ExitCode::UnsupportedJpeg,"sof7 marker found, image is coded diff. lossless");
                }

            0xC9 => // SOF9 segment
                {
                    // coding process: arithmetic extended sequential DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof9 marker found, image is coded arithm. sequential");
                }

            0xCA => // SOF10 segment
                {
                    // coding process: arithmetic extended sequential DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof10 marker found, image is coded arithm. progressive");
                }

            0xCB => // SOF11 segment
                {
                    // coding process: arithmetic extended sequential DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof11 marker found, image is coded arithm. lossless");
                }

            0xCD => // SOF13 segment
                {
                    // coding process: arithmetic differntial sequential DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof13 marker found, image is coded arithm. diff. sequential");
                }

            0xCE => // SOF14 segment
                {
                    // coding process: arithmetic differential progressive DCT
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof14 marker found, image is coded arithm. diff. progressive");
                }

            0xCF => // SOF15 segment
                {
                    // coding process: arithmetic differntial lossless
                    return err_exit_code(ExitCode::UnsupportedJpeg, "sof15 marker found, image is coded arithm. diff. lossless");
                }

            0xE0| // APP0 segment
            0xE1| // APP1 segment
            0xE2| // APP2 segment
            0xE3| // APP3 segment
            0xE4| // APP4 segment
            0xE5| // APP5 segment
            0xE6| // APP6 segment
            0xE7| // APP7 segment
            0xE8| // APP8 segment
            0xE9| // APP9 segment
            0xEA| // APP10 segment
            0xEB| // APP11 segment
            0xEC| // APP12segment
            0xED| // APP13 segment
            0xEE| // APP14 segment
            0xEF| // APP15 segment
            0xFE // COM segment
                // do nothing - return
                => {}

            jpeg_code::RST0| // RST0 segment
            0xD1| // RST1 segment
            0xD2| // RST2 segment
            0xD3| // RST3 segment
            0xD4| // RST4 segment
            0xD5| // RST5 segment
            0xD6| // RST6 segment
            0xD7 => // RST7 segment
                {
                    // return errormessage - RST is out of place here
                    return err_exit_code(ExitCode::UnsupportedJpeg, "rst marker found out of place");
                }

            jpeg_code::SOI => // SOI segment
                {
                    // return errormessage - start-of-image is out of place here
                    return err_exit_code(ExitCode::UnsupportedJpeg, "soi marker found out of place");
                }

            jpeg_code::EOI => // EOI segment
                {
                    // return errormessage - end-of-image is out of place here
                    return err_exit_code(ExitCode::UnsupportedJpeg,"eoi marker found out of place");
                }

            _ => // unknown marker segment
                {
                    // return errormessage - unknown marker
                    return err_exit_code(ExitCode::UnsupportedJpeg, format!("unknown marker found: FF {0:X}", btype).as_str());
                }
        }
        return Ok(ParseSegmentResult::Continue);
    }

    /// constructs the quantization table based on the jpeg header
    pub fn construct_quantization_tables(&self) -> Result<Vec<QuantizationTables>> {
        let mut quantization_tables = Vec::new();
        for i in 0..self.cmpc {
            let qtables = QuantizationTables::new(self, i);

            // check to see if quantitization table was properly initialized
            // (table contains divisors for edge coefficients so it never should have a zero)
            for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56] {
                if qtables.get_quantization_table()[i] == 0 {
                    return err_exit_code(
                    ExitCode::UnsupportedJpegWithZeroIdct0,
                    "Quantization table contains zero for edge which would cause a divide by zero",
                );
                }
            }
            quantization_tables.push(qtables);
        }
        Ok(quantization_tables)
    }
}

fn ensure_space(segment: &[u8], hpos: usize, amount: usize) -> Result<()> {
    if hpos + amount > segment.len() {
        return err_exit_code(ExitCode::UnsupportedJpeg, "SOF too small");
    }

    Ok(())
}

/// constructs a huffman table for testing purposes from a given distribution
#[cfg(test)]
pub fn generate_huff_table_from_distribution(freq: &[usize; 256]) -> HuffCodes {
    use std::collections::{BinaryHeap, HashMap};

    struct Node {
        symbol: Option<u8>,
        freq: usize,
        left: Option<Box<Node>>,
        right: Option<Box<Node>>,
    }

    impl PartialOrd for Node {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.freq.cmp(&other.freq).reverse())
        }
    }

    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            self.freq == other.freq
        }
    }

    impl Eq for Node {}

    impl Ord for Node {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.freq.cmp(&other.freq).reverse()
        }
    }

    fn build_tree(freq: &[usize]) -> Box<Node> {
        let mut pq = BinaryHeap::new();

        for (symbol, &freq) in freq.iter().enumerate() {
            if freq > 0 {
                pq.push(Box::new(Node {
                    symbol: Some(symbol as u8),
                    freq,
                    left: None,
                    right: None,
                }));
            }
        }

        while pq.len() > 1 {
            let left = pq.pop().unwrap();
            let right = pq.pop().unwrap();
            let new_node = Node {
                symbol: None,
                freq: left.freq + right.freq,
                left: Some(left),
                right: Some(right),
            };
            pq.push(Box::new(new_node));
        }

        pq.pop().unwrap()
    }

    fn generate_codes(
        root: &Node,
        codes: &mut std::collections::HashMap<u8, (u16, u8)>,
        prefix: u16,
        length: u8,
    ) {
        if let Some(symbol) = root.symbol {
            codes.insert(symbol, (prefix, length));
        } else {
            if let Some(ref left) = root.left {
                generate_codes(left, codes, prefix << 1, length + 1);
            }
            if let Some(ref right) = root.right {
                generate_codes(right, codes, (prefix << 1) | 1, length + 1);
            }
        }
    }

    let root = build_tree(freq);

    let mut codes = HashMap::new();
    generate_codes(&root, &mut codes, 0, 0);

    let mut retval = HuffCodes::default();

    for (&symbol, &(code, length)) in &codes {
        retval.c_len[symbol as usize] = length.into();
        retval.c_val[symbol as usize] = code;
    }

    retval.post_initialize();

    retval
}
