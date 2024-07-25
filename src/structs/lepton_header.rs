use std::io::{Cursor, ErrorKind, Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::helpers::{buffer_prefix_matches_marker, err_exit_code, here};
use crate::EnabledFeatures;
use crate::{consts::*, ExitCode};

use super::{
    jpeg_header::JPegHeader, thread_handoff::ThreadHandoff, truncate_components::TruncateComponents,
};

use anyhow::{Context, Result};

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

#[derive(Debug)]
pub struct LeptonHeader {
    /// raw jpeg header to be written back to the file when it is recreated
    pub raw_jpeg_header: Vec<u8>,

    /// how far we have read into the raw header, since the header is divided
    /// into multiple chucks for each scan. For example, a progressive image
    /// would start with the jpeg image segments, followed by a SOS (start of scan)
    /// after which comes the encoded jpeg coefficients, and once thats over
    /// we get another header segment until the next SOS, etc
    pub raw_jpeg_header_read_index: usize,

    pub thread_handoff: Vec<ThreadHandoff>,

    pub jpeg_header: JPegHeader,

    /// information about how to truncate the image if it was partially written
    pub truncate_components: TruncateComponents,

    pub rst_err: Vec<u8>,

    /// A list containing one entry for each scan segment.  Each entry contains the number of restart intervals
    /// within the corresponding scan segment.
    pub rst_cnt: Vec<i32>,

    /// the mask for padding out the bitstream when we get to the end of a reset block
    pub pad_bit: Option<u8>,

    pub rst_cnt_set: bool,

    /// garbage data (default value - empty segment - means no garbage data)
    pub garbage_data: Vec<u8>,

    /// count of scans encountered so far
    pub scnc: usize,

    pub early_eof_encountered: bool,

    /// the maximum dpos in a truncated image
    pub max_dpos: [i32; 4],

    /// the maximum component in a truncated image
    pub max_cmp: i32,

    /// the maximum band in a truncated image
    pub max_bpos: i32,

    /// the maximum bit in a truncated image
    pub max_sah: u8,

    pub jpeg_file_size: u32,

    /// on decompression, plain-text size
    pub plain_text_size: u32,

    /// on decompression, uncompressed lepton header size
    pub uncompressed_lepton_header_size: u32,
}

impl LeptonHeader {
    pub fn new() -> Self {
        return LeptonHeader {
            max_dpos: [0; 4],
            raw_jpeg_header: Vec::new(),
            raw_jpeg_header_read_index: 0,
            thread_handoff: Vec::new(),
            jpeg_header: JPegHeader::new(),
            truncate_components: TruncateComponents::new(),
            rst_err: Vec::new(),
            rst_cnt: Vec::new(),
            pad_bit: None,
            rst_cnt_set: false,
            garbage_data: Vec::new(),
            scnc: 0,
            early_eof_encountered: false,
            max_cmp: 0,
            max_bpos: 0,
            max_sah: 0,
            jpeg_file_size: 0,
            plain_text_size: 0,
            uncompressed_lepton_header_size: 0,
        };
    }

    /// reads the start of the lepton file and parses the compressed header. Returns the raw JPEG header contents.
    pub fn read_lepton_header<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &mut EnabledFeatures,
    ) -> Result<()> {
        let mut header = [0 as u8; LEPTON_FILE_HEADER.len()];

        reader.read_exact(&mut header).context(here!())?;

        if !buffer_prefix_matches_marker(header, LEPTON_FILE_HEADER) {
            return err_exit_code(ExitCode::BadLeptonFile, "header doesn't match");
        }

        // Complicated logic of version compatibility should be verified by the caller.
        // Currently just matching the version version.
        let version = reader.read_u8().context(here!())?;
        if version != LEPTON_VERSION {
            return err_exit_code(
                ExitCode::VersionUnsupported,
                format!("incompatible file with version {0}", version).as_str(),
            );
        }

        let mut header = [0 as u8; 21];
        reader.read_exact(&mut header).context(here!())?;

        // Z = baseline non-progressive
        // Y = chunked encoding of a slice of a JPEG (not supported yet)
        // X = progressive
        if header[0] != LEPTON_HEADER_BASELINE_JPEG_TYPE[0]
            && header[0] != LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE[0]
        {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!("Unknown filetype in header {0}", header[0]).as_str(),
            );
        }

        let mut c = Cursor::new(header);

        // We use 12 bytes of git revision for our needs - mark that it's C# implementation and a not-compressed header size.
        self.uncompressed_lepton_header_size = 0;
        if header[5] == 'M' as u8 && header[6] == 'S' as u8 {
            c.set_position(7);
            self.uncompressed_lepton_header_size = c.read_u32::<LittleEndian>()?;

            // read the flag bits to know how we should decode this file
            let flags = c.read_u8()?;
            if (flags & 0x80) != 0 {
                enabled_features.use_16bit_dc_estimate = (flags & 0x01) != 0;
                enabled_features.use_16bit_adv_predict = (flags & 0x02) != 0;
            }
        }

        // full size of the original file
        c.set_position(17);
        self.plain_text_size = c.read_u32::<LittleEndian>()?;

        // now read the compressed header
        let compressed_header_size = reader.read_u32::<LittleEndian>()? as usize;

        if compressed_header_size > MAX_FILE_SIZE_BYTES as usize {
            return err_exit_code(ExitCode::BadLeptonFile, "Too big compressed header");
        }
        if self.plain_text_size > MAX_FILE_SIZE_BYTES as u32 {
            return err_exit_code(ExitCode::BadLeptonFile, "Only support images < 128 megs");
        }

        // limit reading to the compressed header
        let mut compressed_reader = reader.take(compressed_header_size as u64);

        self.raw_jpeg_header = self
            .read_lepton_compressed_header(&mut compressed_reader)
            .context(here!())?;

        // CMP marker
        let mut current_lepton_marker = [0 as u8; 3];
        reader.read_exact(&mut current_lepton_marker)?;
        if !buffer_prefix_matches_marker(current_lepton_marker, LEPTON_HEADER_COMPLETION_MARKER) {
            return err_exit_code(ExitCode::BadLeptonFile, "CMP marker not found");
        }

        self.raw_jpeg_header_read_index = 0;

        {
            let mut header_data_cursor = Cursor::new(&self.raw_jpeg_header[..]);
            self.jpeg_header
                .parse(&mut header_data_cursor, &enabled_features)
                .context(here!())?;
            self.raw_jpeg_header_read_index = header_data_cursor.position() as usize;
        }

        self.truncate_components.init(&self.jpeg_header);

        if self.early_eof_encountered {
            self.truncate_components
                .set_truncation_bounds(&self.jpeg_header, self.max_dpos);
        }

        let num_threads = self.thread_handoff.len();

        // luma_y_end of the last thread is not serialized/deserialized, fill it here
        self.thread_handoff[num_threads - 1].luma_y_end =
            self.truncate_components.get_block_height(0);

        // if the last segment was too big to fit with the garbage data taken into account, shorten it
        // (a bit of broken logic in the encoder, but can't change it without breaking the file format)
        if self.early_eof_encountered {
            let mut max_last_segment_size = i32::try_from(self.plain_text_size)?
                - i32::try_from(self.garbage_data.len())?
                - i32::try_from(self.raw_jpeg_header_read_index)?
                - SOI.len() as i32;

            // subtract the segment sizes of all the previous segments (except for the last)
            for i in 0..num_threads - 1 {
                max_last_segment_size -= self.thread_handoff[i].segment_size;
            }

            let last = &mut self.thread_handoff[num_threads - 1];

            if last.segment_size > max_last_segment_size {
                // re-adjust the last segment size
                last.segment_size = max_last_segment_size;
            }
        }

        Ok(())
    }

    /// parses and advances to the next header segment out of raw_jpeg_header into the jpeg header
    pub fn advance_next_header_segment(
        &mut self,
        enabled_features: &EnabledFeatures,
    ) -> Result<bool> {
        let mut header_cursor =
            Cursor::new(&self.raw_jpeg_header[self.raw_jpeg_header_read_index..]);

        let result = self
            .jpeg_header
            .parse(&mut header_cursor, enabled_features)
            .context(here!())?;

        self.raw_jpeg_header_read_index += header_cursor.stream_position()? as usize;

        Ok(result)
    }

    /// helper for read_lepton_header. uncompresses and parses the contents of the compressed header. Returns the raw JPEG header.
    fn read_lepton_compressed_header<R: Read>(&mut self, src: &mut R) -> Result<Vec<u8>> {
        let mut header_reader = ZlibDecoder::new(src);

        let mut hdr_buf: [u8; 3] = [0; 3];
        header_reader.read_exact(&mut hdr_buf)?;

        if !buffer_prefix_matches_marker(hdr_buf, LEPTON_HEADER_MARKER) {
            return err_exit_code(ExitCode::BadLeptonFile, "HDR marker not found");
        }

        let hdrs = header_reader.read_u32::<LittleEndian>()? as usize;

        let mut hdr_data = Vec::new();
        hdr_data.resize(hdrs, 0);
        header_reader.read_exact(&mut hdr_data)?;

        if self.garbage_data.len() == 0 {
            // if we don't have any garbage, assume FFD9 EOI

            // kind of broken logic since this assumes a EOF even if there was a 0 byte garbage header
            // in the file, but this is what the file format is.
            self.garbage_data.extend(EOI);
        }

        // beginning here: recovery information (needed for exact JPEG recovery)
        // read further recovery information if any
        loop {
            let mut current_lepton_marker = [0 as u8; 3];
            match header_reader.read_exact(&mut current_lepton_marker) {
                Ok(_) => {}
                Err(e) => {
                    if e.kind() == ErrorKind::UnexpectedEof {
                        break;
                    } else {
                        return Err(anyhow::Error::new(e));
                    }
                }
            }

            if buffer_prefix_matches_marker(current_lepton_marker, LEPTON_HEADER_PAD_MARKER) {
                self.pad_bit = Some(header_reader.read_u8()?);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTARTS_MARKER,
            ) {
                // CRS marker
                self.rst_cnt_set = true;
                let rst_count = header_reader.read_u32::<LittleEndian>()?;

                for _i in 0..rst_count {
                    self.rst_cnt.push(header_reader.read_i32::<LittleEndian>()?);
                }
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_LUMA_SPLIT_MARKER,
            ) {
                // HH markup
                let mut thread_handoffs =
                    ThreadHandoff::deserialize(current_lepton_marker[2], &mut header_reader)?;

                self.thread_handoff.append(&mut thread_handoffs);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER,
            ) {
                // Marker FRS
                // read number of false set RST markers per scan from file
                let rst_err_count = header_reader.read_u32::<LittleEndian>()? as usize;

                let mut rst_err_data = Vec::<u8>::new();
                rst_err_data.resize(rst_err_count, 0);

                header_reader.read_exact(&mut rst_err_data)?;

                self.rst_err.append(&mut rst_err_data);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_GARBAGE_MARKER,
            ) {
                // GRB marker
                // read garbage (data after end of JPG) from file
                let garbage_size = header_reader.read_u32::<LittleEndian>()? as usize;

                let mut garbage_data_array = Vec::<u8>::new();
                garbage_data_array.resize(garbage_size, 0);

                header_reader.read_exact(&mut garbage_data_array)?;
                self.garbage_data = garbage_data_array;
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_EARLY_EOF_MARKER,
            ) {
                self.max_cmp = header_reader.read_i32::<LittleEndian>()?;
                self.max_bpos = header_reader.read_i32::<LittleEndian>()?;
                self.max_sah = u8::try_from(header_reader.read_i32::<LittleEndian>()?)?;
                self.max_dpos[0] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[1] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[2] = header_reader.read_i32::<LittleEndian>()?;
                self.max_dpos[3] = header_reader.read_i32::<LittleEndian>()?;
                self.early_eof_encountered = true;
            } else {
                return err_exit_code(ExitCode::BadLeptonFile, "unknown data found");
            }
        }

        // shouldn't be any more data
        let mut remaining_buf = Vec::new();
        let remaining = header_reader.read_to_end(&mut remaining_buf)?;
        assert!(remaining == 0);

        return Ok(hdr_data);
    }

    pub fn write_lepton_header<W: Write>(
        &self,
        writer: &mut W,
        enabled_features: &EnabledFeatures,
    ) -> Result<()> {
        let mut lepton_header = Vec::<u8>::new();

        {
            // Most of the Lepton header data that is compressed before storage
            // The data contains recovery information (needed for exact JPEG recovery)
            let mut mrw = Cursor::new(&mut lepton_header);

            self.write_lepton_jpeg_header(&mut mrw)?;
            self.write_lepton_pad_bit(&mut mrw)?;
            self.write_lepton_luma_splits(&mut mrw)?;
            self.write_lepton_jpeg_restarts_if_needed(&mut mrw)?;
            self.write_lepton_jpeg_restart_errors_if_needed(&mut mrw)?;
            self.write_lepton_early_eof_truncation_data_if_needed(&mut mrw)?;
            self.write_lepton_jpeg_garbage_if_needed(&mut mrw, false)?;
        }

        let mut compressed_header = Vec::<u8>::new(); // we collect a zlib compressed version of the header here
        {
            let mut c = Cursor::new(&mut compressed_header);
            let mut encoder = ZlibEncoder::new(&mut c, Compression::default());

            encoder.write_all(&lepton_header[..]).context(here!())?;
            encoder.finish().context(here!())?;
        }

        writer.write_all(&LEPTON_FILE_HEADER)?;
        writer.write_u8(LEPTON_VERSION)?;

        if self.jpeg_header.jpeg_type == JPegType::Progressive {
            writer.write_all(&LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE)?;
        } else {
            writer.write_all(&LEPTON_HEADER_BASELINE_JPEG_TYPE)?;
        }

        writer.write_u8(self.thread_handoff.len() as u8)?;
        writer.write_all(&[0; 3])?;

        // Original lepton format reserves 12 bytes for git revision. We use this space for additional info
        // that our implementation needs - mark that it's MS implementation and a not-compressed header size.
        writer.write_u8('M' as u8)?;
        writer.write_u8('S' as u8)?;
        writer.write_u32::<LittleEndian>(lepton_header.len() as u32)?;

        // write the flags that were used to encode this file
        writer.write_u8(
            0x80 | if enabled_features.use_16bit_dc_estimate {
                1
            } else {
                0
            } | if enabled_features.use_16bit_adv_predict {
                2
            } else {
                0
            },
        )?;

        writer.write_all(&[0; 5])?;

        writer.write_u32::<LittleEndian>(self.jpeg_file_size)?;
        writer.write_u32::<LittleEndian>(compressed_header.len() as u32)?;
        writer.write_all(&compressed_header[..])?;

        writer.write_all(&LEPTON_HEADER_COMPLETION_MARKER)?;

        Ok(())
    }

    fn write_lepton_jpeg_header<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write header to file
        // marker: "HDR" + [size of header]
        mrw.write_all(&LEPTON_HEADER_MARKER)?;

        mrw.write_u32::<LittleEndian>(self.raw_jpeg_header.len() as u32)?;

        // data: data from header
        mrw.write_all(&self.raw_jpeg_header[..])?;

        Ok(())
    }

    fn write_lepton_pad_bit<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // marker: P0D
        mrw.write_all(&LEPTON_HEADER_PAD_MARKER)?;

        // data: this.padBit
        mrw.write_u8(self.pad_bit.unwrap_or(0))?;

        Ok(())
    }

    fn write_lepton_luma_splits<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write luma splits markup HH
        mrw.write_all(&LEPTON_HEADER_LUMA_SPLIT_MARKER)?;

        // data: serialized luma splits
        ThreadHandoff::serialize(&self.thread_handoff, mrw)?;

        Ok(())
    }

    fn write_lepton_jpeg_restarts_if_needed<W: Write>(&self, mrw: &mut W) -> Result<()> {
        if self.rst_cnt.len() > 0 {
            // marker: CRS
            mrw.write_all(&LEPTON_HEADER_JPG_RESTARTS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rst_cnt.len() as u32)?;

            for i in 0..self.rst_cnt.len() {
                mrw.write_u32::<LittleEndian>(self.rst_cnt[i] as u32)?;
            }
        }

        Ok(())
    }

    fn write_lepton_jpeg_restart_errors_if_needed<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write number of false set RST markers per scan (if available) to file
        if self.rst_err.len() > 0 {
            // marker: "FRS" + [number of scans]
            mrw.write_all(&LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rst_err.len() as u32)?;

            mrw.write_all(&self.rst_err[..])?;
        }

        Ok(())
    }

    fn write_lepton_early_eof_truncation_data_if_needed<W: Write>(
        &self,
        mrw: &mut W,
    ) -> Result<()> {
        if self.early_eof_encountered {
            // EEE marker
            mrw.write_all(&LEPTON_HEADER_EARLY_EOF_MARKER)?;

            mrw.write_i32::<LittleEndian>(self.max_cmp)?;
            mrw.write_i32::<LittleEndian>(self.max_bpos)?;
            mrw.write_i32::<LittleEndian>(i32::from(self.max_sah))?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[0])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[1])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[2])?;
            mrw.write_i32::<LittleEndian>(self.max_dpos[3])?;
        }

        Ok(())
    }

    fn write_lepton_jpeg_garbage_if_needed<W: Write>(
        &self,
        mrw: &mut W,
        prefix_garbage: bool,
    ) -> Result<()> {
        // write garbage (if any) to file
        if self.garbage_data.len() > 0 {
            // marker: "PGR/GRB" + [size of garbage]
            if prefix_garbage {
                mrw.write_all(&LEPTON_HEADER_PREFIX_GARBAGE_MARKER)?;
            } else {
                mrw.write_all(&LEPTON_HEADER_GARBAGE_MARKER)?;
            }

            mrw.write_u32::<LittleEndian>(self.garbage_data.len() as u32)?;
            mrw.write_all(&self.garbage_data[..])?;
        }

        Ok(())
    }

    pub fn parse_jpeg_header<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> Result<bool> {
        // the raw header in the lepton file can actually be spread across different sections
        // seperated by the Start-of-Scan marker. We use the mirror to write out whatever
        // data we parse until we hit the SOS

        let mut output = Vec::new();
        let mut output_cursor = Cursor::new(&mut output);

        let mut mirror = Mirror::new(reader, &mut output_cursor);

        if self
            .jpeg_header
            .parse(&mut mirror, enabled_features)
            .context(here!())?
        {
            // append the header if it was not the end of file marker
            self.raw_jpeg_header.append(&mut output);
            return Ok(true);
        } else {
            // if the output was more than 2 bytes then was a trailing header, so keep that around as well,
            // but we don't want the EOI since that goes into the garbage data.
            if output.len() > 2 {
                self.raw_jpeg_header.extend(&output[0..output.len() - 2]);
            }

            return Ok(false);
        }
    }
}

// internal utility we use to collect the header that we read for later
struct Mirror<'a, R, W> {
    read: &'a mut R,
    output: &'a mut W,
    amount_written: usize,
}

impl<'a, R, W> Mirror<'a, R, W> {
    pub fn new(read: &'a mut R, output: &'a mut W) -> Self {
        Mirror {
            read,
            output,
            amount_written: 0,
        }
    }
}

impl<R: Read, W: Write> Read for Mirror<'_, R, W> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.read.read(buf)?;
        self.output.write_all(&buf[..n])?;
        self.amount_written += n;
        Ok(n)
    }
}
