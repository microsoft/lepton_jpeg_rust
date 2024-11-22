use std::io::{Cursor, ErrorKind, Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use default_boxed::DefaultBoxed;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::consts::*;
use crate::helpers::buffer_prefix_matches_marker;
use crate::lepton_error::{err_exit_code, AddContext, ExitCode, Result};
use crate::structs::jpeg_header::JPegHeader;
use crate::structs::thread_handoff::ThreadHandoff;
use crate::structs::truncate_components::TruncateComponents;
use crate::EnabledFeatures;

pub const FIXED_HEADER_SIZE: usize = 28;

#[derive(Debug, DefaultBoxed)]
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

    /// on decompression, uncompressed lepton header size. This is only
    /// saved by this encoder for historical reasons. It is not used by
    /// the decoder.
    pub uncompressed_lepton_header_size: Option<u32>,

    /// the git revision of the encoder that created this file (first 8 hex characters)
    pub git_revision_prefix: [u8; 4],

    /// writer version
    pub encoder_version: u8,
}

impl LeptonHeader {
    pub fn read_lepton_fixed_header(
        &mut self,
        header: &[u8; FIXED_HEADER_SIZE],
        enabled_features: &mut EnabledFeatures,
    ) -> Result<usize> {
        if header[0..2] != LEPTON_FILE_HEADER[0..2] {
            return err_exit_code(ExitCode::BadLeptonFile, "header doesn't match");
        }
        if header[2] != LEPTON_VERSION {
            return err_exit_code(
                ExitCode::VersionUnsupported,
                format!("incompatible file with version {0}", header[3]).as_str(),
            );
        }
        if header[3] != LEPTON_HEADER_BASELINE_JPEG_TYPE[0]
            && header[3] != LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE[0]
        {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!("Unknown filetype in header {0}", header[4]).as_str(),
            );
        }

        // header[4] is the number of streams/threads, but we don't care about that
        // header[5..8] is reserved

        // header[8..20] 12 bytes were the GIT revision, but for historical reasons we
        // also use this space to store the uncompressed lepton header size plus some
        // flags to detect the SIMD flavor that was used to encode, since
        // previously the encoder would generate different incompatible files depending on
        // whether SIMD or scalar was selected by the build options.
        if header[8] == 'M' as u8 && header[9] == 'S' as u8 {
            self.uncompressed_lepton_header_size =
                Some(u32::from_le_bytes(header[10..14].try_into().unwrap()));

            // read the flag bits to know how we should decode this file
            let flags = header[14];
            if (flags & 0x80) != 0 {
                enabled_features.use_16bit_dc_estimate = (flags & 0x01) != 0;
                enabled_features.use_16bit_adv_predict = (flags & 0x02) != 0;
            }

            self.encoder_version = header[15];
            self.git_revision_prefix = header[16..20].try_into().unwrap();
        } else {
            // take first bytes for git revision prefix
            self.git_revision_prefix = header[8..12].try_into().unwrap();
        }

        // total size of original JPEG
        self.jpeg_file_size = u32::from_le_bytes(header[20..24].try_into().unwrap());

        let compressed_header_size =
            u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;

        Ok(compressed_header_size)
    }

    /// reads the start of the lepton file and parses the compressed header. Returns the raw JPEG header contents.
    pub fn read_compressed_lepton_header<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &mut EnabledFeatures,
        compressed_header_size: usize,
    ) -> Result<()> {
        if compressed_header_size > enabled_features.max_jpeg_file_size as usize {
            return err_exit_code(ExitCode::BadLeptonFile, "Too big compressed header");
        }
        if self.jpeg_file_size > enabled_features.max_jpeg_file_size {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!(
                    "Only support images < {} megs",
                    enabled_features.max_jpeg_file_size / (1024 * 1024)
                )
                .as_str(),
            );
        }

        // limit reading to the compressed header
        let mut compressed_reader = reader.take(compressed_header_size as u64);

        self.raw_jpeg_header = self
            .read_lepton_compressed_header(&mut compressed_reader)
            .context()?;

        self.raw_jpeg_header_read_index = 0;

        {
            let mut header_data_cursor = Cursor::new(&self.raw_jpeg_header[..]);
            self.jpeg_header
                .parse(&mut header_data_cursor, &enabled_features)
                .context()?;
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
            let mut max_last_segment_size = i32::try_from(self.jpeg_file_size)?
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
            .context()?;

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
                        return Err(e.into());
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

            encoder.write_all(&lepton_header[..]).context()?;
            encoder.finish().context()?;
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
        // to store information about the version that wrote this.
        writer.write_u8('M' as u8)?;
        writer.write_u8('S' as u8)?;

        // write the uncompressed lepton header size
        // (historical, used by a previous version of the decoder)
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

        // version of the encoder
        writer.write_u8(self.encoder_version)?;

        // write the git revision prefix that was used to write this
        writer.write_all(&self.git_revision_prefix)?;

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
            .context()?
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

#[test]
fn test_roundtrip_fixed_header() {
    let test_data = [
        (0, true, true),
        (128, false, false),
        (129, true, false),
        (130, false, true),
        (131, true, true),
    ];
    for (v, dc_16_bit, adv_16_bit) in test_data {
        // test known good version of the header so we can detect breaks
        let fixed_buffer = [
            207, 132, 1, 90, 1, 0, 0, 0, 77, 83, 140, 0, 0, 0, v, 187, 18, 52, 86, 120, 123, 0, 0,
            0, 122, 0, 0, 0,
        ];

        let mut other_enabled_features = EnabledFeatures::compat_lepton_vector_read();

        let mut other = LeptonHeader::default_boxed();
        let compressed_header_size = other
            .read_lepton_fixed_header(&fixed_buffer, &mut other_enabled_features)
            .unwrap();
        assert_eq!(compressed_header_size, 122);
        assert_eq!(other_enabled_features.use_16bit_dc_estimate, dc_16_bit);
        assert_eq!(other_enabled_features.use_16bit_adv_predict, adv_16_bit);
    }

    // test read/write all combinations of the flags
    for (dc_16_bit, adv_16_bit) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut header = make_minimal_lepton_header();
        header.git_revision_prefix = [0x12, 0x34, 0x56, 0x78];
        header.encoder_version = 0xBB;

        let mut enabled_features = EnabledFeatures::compat_lepton_vector_write();
        enabled_features.use_16bit_dc_estimate = dc_16_bit;
        enabled_features.use_16bit_adv_predict = adv_16_bit;

        let (result_header, result_features) = verify_roundtrip(&header, &enabled_features);

        assert_eq!(result_features.use_16bit_dc_estimate, dc_16_bit);
        assert_eq!(result_features.use_16bit_adv_predict, adv_16_bit);
        assert_eq!(
            result_header.git_revision_prefix,
            header.git_revision_prefix
        );
        assert_eq!(result_header.encoder_version, header.encoder_version);
    }
}

// test serializing and deserializing header
#[test]
fn parse_and_write_header() {
    use crate::structs::lepton_header::FIXED_HEADER_SIZE;

    let lh = make_minimal_lepton_header();

    let enabled_features = EnabledFeatures::compat_lepton_vector_write();
    let mut serialized = Vec::new();
    lh.write_lepton_header(&mut Cursor::new(&mut serialized), &enabled_features)
        .unwrap();

    let mut other = LeptonHeader::default_boxed();
    let mut other_reader = Cursor::new(&serialized);

    let mut fixed_buffer = [0; FIXED_HEADER_SIZE];
    other_reader.read_exact(&mut fixed_buffer).unwrap();

    let mut other_enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let compressed_header_size = other
        .read_lepton_fixed_header(&fixed_buffer, &mut other_enabled_features)
        .unwrap();

    other
        .read_compressed_lepton_header(
            &mut other_reader,
            &mut other_enabled_features,
            compressed_header_size,
        )
        .unwrap();

    assert_eq!(
        lh.uncompressed_lepton_header_size,
        other.uncompressed_lepton_header_size
    );

    assert_eq!(lh.git_revision_prefix, other.git_revision_prefix);
    assert_eq!(lh.encoder_version, other.encoder_version);

    assert_eq!(lh.jpeg_file_size, other.jpeg_file_size);
    assert_eq!(lh.raw_jpeg_header, other.raw_jpeg_header);
    assert_eq!(lh.thread_handoff, other.thread_handoff);
}

#[cfg(test)]
fn make_minimal_lepton_header() -> Box<LeptonHeader> {
    // minimal jpeg that will pass the validity read tests
    let min_jpeg = [
        0xffu8, 0xe0, // APP0
        0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00,
        0x00, 0xff, 0xdb, // DQT
        0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x02, 0x02, 0x02, 0x03, 0x03,
        0x03, 0x03, 0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08, 0x06, 0x06, 0x05, 0x06, 0x09,
        0x08, 0x0a, 0x0a, 0x09, 0x08, 0x09, 0x09, 0x0a, 0x0c, 0x0f, 0x0c, 0x0a, 0x0b, 0x0e, 0x0b,
        0x09, 0x09, 0x0d, 0x11, 0x0d, 0x0e, 0x0f, 0x10, 0x10, 0x11, 0x10, 0x0a, 0x0c, 0x12, 0x13,
        0x12, 0x10, 0x13, 0x0f, 0x10, 0x10, 0x10, 0xff, 0xC1, 0x00, 0x0b, 0x08, 0x00,
        0x10, // width
        0x00, 0x10, // height
        0x01, // cmpc
        0x01, // Jid
        0x11, // sfv / sfh
        0x00, 0xff, 0xda, // SOS
        0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0xd2, 0xcf, 0x20, 0xff, 0xd9, // EOI
    ];

    let enabled_features = EnabledFeatures::compat_lepton_vector_read();

    let mut lh = LeptonHeader::default_boxed();
    lh.jpeg_file_size = 123;
    lh.uncompressed_lepton_header_size = Some(156);

    lh.parse_jpeg_header(&mut Cursor::new(min_jpeg), &enabled_features)
        .unwrap();
    lh.thread_handoff.push(ThreadHandoff {
        luma_y_start: 0,
        luma_y_end: 1,
        segment_offset_in_file: 0, // not serialized (computed based on segment size)
        segment_size: 500,
        overhang_byte: 0,
        num_overhang_bits: 1,
        last_dc: [1, 2, 3, 0],
    });
    lh.thread_handoff.push(ThreadHandoff {
        luma_y_start: 1,
        luma_y_end: 2,
        segment_offset_in_file: 0,
        segment_size: 600,
        overhang_byte: 1,
        num_overhang_bits: 2,
        last_dc: [2, 3, 4, 0],
    });

    lh
}

#[cfg(test)]
fn verify_roundtrip(
    header: &LeptonHeader,
    enabled_features: &EnabledFeatures,
) -> (Box<LeptonHeader>, EnabledFeatures) {
    let mut output = Vec::new();
    header
        .write_lepton_header(&mut output, &enabled_features)
        .unwrap();

    let mut read_header = LeptonHeader::default_boxed();
    let mut read_enabled_features = EnabledFeatures::compat_lepton_vector_read();

    println!("output: {:?}", &output[0..FIXED_HEADER_SIZE]);

    read_header
        .read_lepton_fixed_header(
            &output[..FIXED_HEADER_SIZE].try_into().unwrap(),
            &mut read_enabled_features,
        )
        .unwrap();
    read_header
        .read_compressed_lepton_header(
            &mut Cursor::new(&output[FIXED_HEADER_SIZE..]),
            &mut read_enabled_features,
            output.len() - FIXED_HEADER_SIZE,
        )
        .unwrap();

    (read_header, read_enabled_features)
}
