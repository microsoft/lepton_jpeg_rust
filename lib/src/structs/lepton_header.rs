use std::cmp::min;
use std::io::{Cursor, ErrorKind, Read, Seek, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use default_boxed::DefaultBoxed;
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::EnabledFeatures;
use crate::consts::*;
use crate::helpers::buffer_prefix_matches_marker;
use crate::jpeg::jpeg_header::{JpegHeader, ReconstructionInfo};
use crate::lepton_error::{AddContext, ExitCode, Result, err_exit_code};
use crate::size_value::{SizeValue, check_size};
use crate::structs::thread_handoff::ThreadHandoff;

pub const FIXED_HEADER_SIZE: usize = 28;

/// Reads a length-prefixed field without trusting the declared length for allocation:
/// the length is checked against `max_len` and the data is read incrementally.
fn read_bounded_vec<R: Read>(
    reader: &mut R,
    len: u32,
    max_len: u32,
    name: &str,
) -> Result<Vec<u8>> {
    let len = check_size(len, max_len.to_usize()?, name)?;

    let mut data = Vec::new();
    reader
        .take(u64::from(len.to_u32()?))
        .read_to_end(&mut data)?;
    if data.len() != len {
        return err_exit_code(ExitCode::ShortRead, "lepton header field is truncated");
    }
    Ok(data)
}

#[derive(Debug, DefaultBoxed)]
pub struct LeptonHeader {
    /// how far we have read into the raw header, since the header is divided
    /// into multiple chucks for each scan. For example, a progressive image
    /// would start with the jpeg image segments, followed by a SOS (start of scan)
    /// after which comes the encoded jpeg coefficients, and once thats over
    /// we get another header segment until the next SOS, etc
    pub raw_jpeg_header_read_index: usize,

    pub thread_handoff: Vec<ThreadHandoff>,

    pub jpeg_header: JpegHeader,

    pub rinfo: ReconstructionInfo,

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
    /// For certain versions of the rust encoder, we didn't handle truncation and corruption correctly.
    /// The correct behavior is to truncate the JPEG generated data up to the file size minus the garbage data,
    /// then write out the garbage data.
    ///
    /// The incorrect behavior was to write out the JPEG data, append the garbage data, and then truncate.
    pub fn bad_truncation_version(&self) -> bool {
        self.encoder_version == 55
    }

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
                format!("incompatible file with version {0}", header[3]),
            );
        }
        if header[3] != LEPTON_HEADER_BASELINE_JPEG_TYPE[0]
            && header[3] != LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE[0]
        {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!("Unknown filetype in header {0}", header[4]),
            );
        }

        // header[4] is the number of streams/threads, but we don't care about that
        // header[5..8] is reserved

        // header[8..20] 12 bytes were the GIT revision, but for historical reasons we
        // also use this space to store the uncompressed lepton header size plus some
        // flags to detect the SIMD flavor that was used to encode, since
        // previously the encoder would generate different incompatible files depending on
        // whether SIMD or scalar was selected by the build options.
        if header[8] == b'M' && header[9] == b'S' {
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

        // check limits here, before the streaming reader buffers the compressed header
        let max_jpeg_file_size = enabled_features.max_jpeg_file_size.to_usize()?;
        check_size(self.jpeg_file_size, max_jpeg_file_size, "JPEG file")?;

        let compressed_header_size = check_size(
            u32::from_le_bytes(header[24..28].try_into().unwrap()),
            max_jpeg_file_size,
            "Compressed header",
        )?;

        Ok(compressed_header_size)
    }

    /// reads the start of the lepton file and parses the compressed header. Returns the raw JPEG header contents.
    pub fn read_compressed_lepton_header<R: Read>(
        &mut self,
        reader: &mut R,
        enabled_features: &mut EnabledFeatures,
        compressed_header_size: usize,
    ) -> Result<()> {
        // limit reading to the compressed header
        let mut compressed_reader = reader.take(compressed_header_size as u64);

        self.rinfo.raw_jpeg_header = self
            .read_lepton_compressed_header(&mut compressed_reader, enabled_features)
            .context()?;

        self.raw_jpeg_header_read_index = 0;

        {
            let mut header_data_cursor = Cursor::new(&self.rinfo.raw_jpeg_header[..]);
            self.jpeg_header
                .parse(&mut header_data_cursor, enabled_features)
                .context()?;
            self.raw_jpeg_header_read_index = header_data_cursor.position() as usize;
        }

        self.rinfo.truncate_components.init(&self.jpeg_header);

        if self.rinfo.early_eof_encountered {
            self.rinfo
                .truncate_components
                .set_truncation_bounds(&self.jpeg_header, self.rinfo.max_dpos)?;
        }

        let num_threads = self.thread_handoff.len();

        // partition ids are stored in 4 bits in the multiplexed stream
        if num_threads == 0 || num_threads > 16 {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                format!("invalid number of partitions {num_threads}"),
            );
        }

        // luma_y_end of the last thread is not serialized/deserialized, fill it here
        let max_luma = self.rinfo.truncate_components.get_block_height(0);

        for i in 0..num_threads {
            self.thread_handoff[i].luma_y_start =
                min(self.thread_handoff[i].luma_y_start, max_luma);
            self.thread_handoff[i].luma_y_end = min(self.thread_handoff[i].luma_y_end, max_luma);
        }
        self.thread_handoff[num_threads - 1].luma_y_end = max_luma;

        // ranges must not run backwards, otherwise the per-thread image sizes underflow
        if self
            .thread_handoff
            .iter()
            .any(|t| t.luma_y_start > t.luma_y_end)
        {
            return err_exit_code(ExitCode::BadLeptonFile, "invalid partition row ranges");
        }

        // if the last segment was too big to fit with the garbage data taken into account, shorten it
        // (a bit of broken logic in the encoder, but can't change it without breaking the file format)
        if self.rinfo.early_eof_encountered {
            let mut max_last_segment_size = Some(self.jpeg_file_size)
                .and_then(|s| s.checked_sub(u32::try_from(self.rinfo.garbage_data.len()).ok()?))
                .and_then(|s| s.checked_sub(u32::try_from(self.raw_jpeg_header_read_index).ok()?))
                .and_then(|s| s.checked_sub(u32::try_from(SOI.len()).ok()?));

            // subtract the segment sizes of all the previous segments (except for the last)
            for i in 0..num_threads - 1 {
                max_last_segment_size = max_last_segment_size
                    .and_then(|s| s.checked_sub(self.thread_handoff[i].segment_size));
            }

            let Some(max_last_segment_size) = max_last_segment_size else {
                return err_exit_code(
                    ExitCode::BadLeptonFile,
                    "segment sizes exceed the jpeg file size",
                );
            };

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
            Cursor::new(&self.rinfo.raw_jpeg_header[self.raw_jpeg_header_read_index..]);

        let result = self
            .jpeg_header
            .parse(&mut header_cursor, enabled_features)
            .context()?;

        self.raw_jpeg_header_read_index += header_cursor.stream_position()? as usize;

        Ok(result)
    }

    /// helper for read_lepton_header. uncompresses and parses the contents of the compressed header. Returns the raw JPEG header.
    fn read_lepton_compressed_header<R: Read>(
        &mut self,
        src: &mut R,
        enabled_features: &EnabledFeatures,
    ) -> Result<Vec<u8>> {
        let max_jpeg_file_size = enabled_features.max_jpeg_file_size;
        let max_scan_entries = enabled_features.max_progressive_scans.max(1);

        // The header holds the raw JPEG headers and garbage data (both parts of the original
        // file) plus small per-scan/partition tables, so twice the file limit plus slack
        // accepts anything the encoder could write while bounding the zlib output.
        let max_header_size = 2 * u64::from(max_jpeg_file_size) + 1024 * 1024;
        let mut header_reader = ZlibDecoder::new(src).take(max_header_size + 1);

        let mut hdr_buf: [u8; 3] = [0; 3];
        header_reader.read_exact(&mut hdr_buf)?;

        if !buffer_prefix_matches_marker(hdr_buf, LEPTON_HEADER_MARKER) {
            return err_exit_code(ExitCode::BadLeptonFile, "HDR marker not found");
        }

        let hdrs = header_reader.read_u32::<LittleEndian>()?;
        let hdr_data =
            read_bounded_vec(&mut header_reader, hdrs, max_jpeg_file_size, "JPEG header")?;

        if self.rinfo.garbage_data.is_empty() {
            // if we don't have any garbage, assume 0xFF 0xD9 EOI (end of image marker)

            // Kind of broken logic since this assumes a EOI even if the file was
            // truncated at the EOI, but this is what the file format is.
            // In this case, this marker will be chopped off later by the
            // overall JPEG file size limit, so this is not a correctness problem.
            self.rinfo.garbage_data.extend(EOI);
        }

        // beginning here: recovery information (needed for exact JPEG recovery)
        // read further recovery information if any
        loop {
            let mut current_lepton_marker = [0u8; 3];
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
                self.rinfo.pad_bit = Some(header_reader.read_u8()?);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTARTS_MARKER,
            ) {
                // CRS marker
                self.rinfo.rst_cnt_set = true;
                let rst_count = header_reader.read_u32::<LittleEndian>()?;
                let new_rst_count = self
                    .rinfo
                    .rst_cnt
                    .len()
                    .checked_add(rst_count.to_usize()?)
                    .ok_or_else(|| {
                        crate::LeptonError::new(
                            ExitCode::BadLeptonFile,
                            "restart count table is too large",
                        )
                    })?;
                check_size(
                    new_rst_count,
                    max_scan_entries.to_usize()?,
                    "Restart count entries",
                )?;
                if let Err(e) = self.rinfo.rst_cnt.try_reserve(rst_count.to_usize()?) {
                    return err_exit_code(
                        ExitCode::OutOfMemory,
                        format!("failed to allocate restart count table: {e}"),
                    );
                }

                for _i in 0..rst_count {
                    self.rinfo
                        .rst_cnt
                        .push(header_reader.read_u32::<LittleEndian>()?);
                }
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_LUMA_SPLIT_MARKER,
            ) {
                // HH markup
                let new_handoff_count = self
                    .thread_handoff
                    .len()
                    .checked_add(usize::from(current_lepton_marker[2]))
                    .ok_or_else(|| {
                        crate::LeptonError::new(
                            ExitCode::BadLeptonFile,
                            "partition table is too large",
                        )
                    })?;
                check_size(
                    new_handoff_count,
                    MAX_THREADS_SUPPORTED_BY_LEPTON_FORMAT,
                    "Partitions",
                )?;

                let mut thread_handoffs =
                    ThreadHandoff::deserialize(current_lepton_marker[2], &mut header_reader)?;

                self.thread_handoff.append(&mut thread_handoffs);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER,
            ) {
                // Marker FRS
                // read number of false set RST markers per scan from file
                let rst_err_count = header_reader.read_u32::<LittleEndian>()?;
                let remaining_rst_errors = max_scan_entries
                    .checked_sub(self.rinfo.rst_err.len().to_u32()?)
                    .ok_or_else(|| {
                        crate::LeptonError::new(
                            ExitCode::BadLeptonFile,
                            "restart error table is too large",
                        )
                    })?;

                let mut rst_err_data = read_bounded_vec(
                    &mut header_reader,
                    rst_err_count,
                    remaining_rst_errors,
                    "Restart errors",
                )?;

                self.rinfo.rst_err.append(&mut rst_err_data);
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_GARBAGE_MARKER,
            ) {
                // GRB marker
                // read garbage (data after end of JPG) from file
                let garbage_size = header_reader.read_u32::<LittleEndian>()?;

                self.rinfo.garbage_data = read_bounded_vec(
                    &mut header_reader,
                    garbage_size,
                    max_jpeg_file_size,
                    "Garbage data",
                )?;
            } else if buffer_prefix_matches_marker(
                current_lepton_marker,
                LEPTON_HEADER_EARLY_EOF_MARKER,
            ) {
                self.rinfo.max_cmp = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.max_bpos = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.max_sah = u8::try_from(header_reader.read_u32::<LittleEndian>()?)?;
                self.rinfo.max_dpos[0] = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.max_dpos[1] = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.max_dpos[2] = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.max_dpos[3] = header_reader.read_u32::<LittleEndian>()?;
                self.rinfo.early_eof_encountered = true;
            } else {
                return err_exit_code(ExitCode::BadLeptonFile, "unknown data found");
            }
        }

        // shouldn't be any more data
        let mut remaining_buf = [0u8; 1];
        if header_reader.read(&mut remaining_buf)? != 0 {
            return err_exit_code(ExitCode::BadLeptonFile, "unexpected data after header");
        }

        // hitting the take() limit looks like EOF, so treat it as an error
        if header_reader.limit() == 0 {
            return err_exit_code(ExitCode::BadLeptonFile, "lepton header is too large");
        }

        Ok(hdr_data)
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

        if self.jpeg_header.jpeg_type == JpegType::Progressive {
            writer.write_all(&LEPTON_HEADER_PROGRESSIVE_JPEG_TYPE)?;
        } else {
            writer.write_all(&LEPTON_HEADER_BASELINE_JPEG_TYPE)?;
        }

        writer.write_u8(u8::try_from(self.thread_handoff.len())?)?;
        writer.write_all(&[0; 3])?;

        // Original lepton format reserves 12 bytes for git revision. We use this space for additional info
        // to store information about the version that wrote this.
        writer.write_u8(b'M')?;
        writer.write_u8(b'S')?;

        // write the uncompressed lepton header size
        // (historical, used by a previous version of the decoder)
        writer.write_u32::<LittleEndian>(lepton_header.len().to_u32()?)?;

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
        writer.write_u32::<LittleEndian>(compressed_header.len().to_u32()?)?;
        writer.write_all(&compressed_header[..])?;

        writer.write_all(&LEPTON_HEADER_COMPLETION_MARKER)?;

        Ok(())
    }

    fn write_lepton_jpeg_header<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write header to file
        // marker: "HDR" + [size of header]
        mrw.write_all(&LEPTON_HEADER_MARKER)?;

        mrw.write_u32::<LittleEndian>(self.rinfo.raw_jpeg_header.len().to_u32()?)?;

        // data: data from header
        mrw.write_all(&self.rinfo.raw_jpeg_header[..])?;

        Ok(())
    }

    fn write_lepton_pad_bit<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // marker: P0D
        mrw.write_all(&LEPTON_HEADER_PAD_MARKER)?;

        // data: this.padBit
        mrw.write_u8(self.rinfo.pad_bit.unwrap_or(0))?;

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
        if !self.rinfo.rst_cnt.is_empty() {
            // marker: CRS
            mrw.write_all(&LEPTON_HEADER_JPG_RESTARTS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rinfo.rst_cnt.len().to_u32()?)?;

            for i in 0..self.rinfo.rst_cnt.len() {
                mrw.write_u32::<LittleEndian>(self.rinfo.rst_cnt[i])?;
            }
        }

        Ok(())
    }

    fn write_lepton_jpeg_restart_errors_if_needed<W: Write>(&self, mrw: &mut W) -> Result<()> {
        // write number of false set RST markers per scan (if available) to file
        if !self.rinfo.rst_err.is_empty() {
            // marker: "FRS" + [number of scans]
            mrw.write_all(&LEPTON_HEADER_JPG_RESTART_ERRORS_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rinfo.rst_err.len().to_u32()?)?;

            mrw.write_all(&self.rinfo.rst_err[..])?;
        }

        Ok(())
    }

    fn write_lepton_early_eof_truncation_data_if_needed<W: Write>(
        &self,
        mrw: &mut W,
    ) -> Result<()> {
        if self.rinfo.early_eof_encountered {
            // EEE marker
            mrw.write_all(&LEPTON_HEADER_EARLY_EOF_MARKER)?;

            mrw.write_u32::<LittleEndian>(self.rinfo.max_cmp)?;
            mrw.write_u32::<LittleEndian>(self.rinfo.max_bpos)?;
            mrw.write_u32::<LittleEndian>(u32::from(self.rinfo.max_sah))?;
            mrw.write_u32::<LittleEndian>(self.rinfo.max_dpos[0])?;
            mrw.write_u32::<LittleEndian>(self.rinfo.max_dpos[1])?;
            mrw.write_u32::<LittleEndian>(self.rinfo.max_dpos[2])?;
            mrw.write_u32::<LittleEndian>(self.rinfo.max_dpos[3])?;
        }

        Ok(())
    }

    fn write_lepton_jpeg_garbage_if_needed<W: Write>(
        &self,
        mrw: &mut W,
        prefix_garbage: bool,
    ) -> Result<()> {
        // write garbage (if any) to file
        if !self.rinfo.garbage_data.is_empty() {
            // marker: "PGR/GRB" + [size of garbage]
            if prefix_garbage {
                mrw.write_all(&LEPTON_HEADER_PREFIX_GARBAGE_MARKER)?;
            } else {
                mrw.write_all(&LEPTON_HEADER_GARBAGE_MARKER)?;
            }

            mrw.write_u32::<LittleEndian>(self.rinfo.garbage_data.len().to_u32()?)?;
            mrw.write_all(&self.rinfo.garbage_data[..])?;
        }

        Ok(())
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
    assert_eq!(lh.rinfo.raw_jpeg_header, other.rinfo.raw_jpeg_header);
    assert_eq!(lh.thread_handoff, other.thread_handoff);
}

#[cfg(test)]
fn compressed_test_header(plain: &[u8], jpeg_file_size: u32) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(plain).unwrap();
    let compressed = encoder.finish().unwrap();

    let mut file = vec![
        207, 132, 1, 90, 1, 0, 0, 0, 77, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    file.extend(jpeg_file_size.to_le_bytes());
    file.extend((compressed.len() as u32).to_le_bytes());
    file.extend(compressed);
    file
}

#[cfg(test)]
fn read_test_header(file: &[u8], enabled_features: &EnabledFeatures) -> Result<()> {
    let mut features = enabled_features.clone();
    let mut lh = LeptonHeader::default_boxed();
    let size =
        lh.read_lepton_fixed_header(file[..FIXED_HEADER_SIZE].try_into().unwrap(), &mut features)?;
    lh.read_compressed_lepton_header(
        &mut Cursor::new(&file[FIXED_HEADER_SIZE..]),
        &mut features,
        size,
    )
}

/// sizes declared in the header must be checked before allocating or buffering anything
#[test]
fn reject_oversized_declared_lengths() {
    let features = EnabledFeatures {
        max_jpeg_file_size: 1024,
        ..EnabledFeatures::compat_lepton_vector_read()
    };

    // fixed header declares a compressed header bigger than the limit
    let mut file = compressed_test_header(b"HDR", 100);
    file[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        read_test_header(&file, &features).unwrap_err().exit_code(),
        ExitCode::BadLeptonFile
    );

    // jpeg file size bigger than the limit
    let file = compressed_test_header(b"HDR", u32::MAX);
    assert_eq!(
        read_test_header(&file, &features).unwrap_err().exit_code(),
        ExitCode::BadLeptonFile
    );

    // HDR, FRS and GRB fields declare 4 GiB without containing the data
    for prefix in [&b""[..], b"FRS", b"GRB"] {
        let mut plain = b"HDR".to_vec();
        if prefix.is_empty() {
            plain.extend(u32::MAX.to_le_bytes());
        } else {
            plain.extend(0u32.to_le_bytes());
            plain.extend(prefix);
            plain.extend(u32::MAX.to_le_bytes());
        }
        let file = compressed_test_header(&plain, 100);
        assert_eq!(
            read_test_header(&file, &features).unwrap_err().exit_code(),
            ExitCode::BadLeptonFile
        );
    }
}

/// recovery tables must be bounded before their attacker-declared contents are allocated or read
#[test]
fn reject_oversized_recovery_tables_before_reading_entries() {
    let features = EnabledFeatures {
        max_jpeg_file_size: 1024,
        max_progressive_scans: 1,
        ..EnabledFeatures::compat_lepton_vector_read()
    };

    for marker in [b"CRS", b"FRS"] {
        let mut plain = b"HDR".to_vec();
        plain.extend(0u32.to_le_bytes());
        plain.extend(marker);
        plain.extend(2u32.to_le_bytes());

        let file = compressed_test_header(&plain, 100);
        assert_eq!(
            read_test_header(&file, &features).unwrap_err().exit_code(),
            ExitCode::BadLeptonFile
        );
    }

    let mut plain = b"HDR".to_vec();
    plain.extend(0u32.to_le_bytes());
    plain.extend(b"HH\x11");
    let file = compressed_test_header(&plain, 100);
    assert_eq!(
        read_test_header(&file, &features).unwrap_err().exit_code(),
        ExitCode::BadLeptonFile
    );
}

/// malformed partition and garbage tables must return errors instead of panicking
#[test]
fn reject_inconsistent_header_tables() {
    let lh = make_minimal_lepton_header();
    let features = EnabledFeatures::compat_lepton_vector_read();

    let mut jpeg_header = b"HDR".to_vec();
    jpeg_header.extend((lh.rinfo.raw_jpeg_header.len() as u32).to_le_bytes());
    jpeg_header.extend(&lh.rinfo.raw_jpeg_header);

    // no partitions at all
    let file = compressed_test_header(&jpeg_header, 1000);
    assert_eq!(
        read_test_header(&file, &features).unwrap_err().exit_code(),
        ExitCode::BadLeptonFile
    );

    // truncated file where segment sizes exceed the declared jpeg size
    let mut plain = jpeg_header.clone();
    plain.extend(b"HH\x02");
    for _ in 0..2 {
        plain.extend([0u8; 2]);
        plain.extend(u32::MAX.to_le_bytes());
        plain.extend([0u8; 10]);
    }
    plain.extend(b"EEE");
    plain.extend([0u8; 28]);
    let file = compressed_test_header(&plain, 1000);
    assert_eq!(
        read_test_header(&file, &features).unwrap_err().exit_code(),
        ExitCode::BadLeptonFile
    );

    // partition row ranges that run backwards
    let mut plain = jpeg_header.clone();
    plain.extend(b"HH\x02");
    for luma_y_start in [1u16, 0] {
        plain.extend(luma_y_start.to_le_bytes());
        plain.extend([0u8; 14]);
    }
    let file = compressed_test_header(&plain, 1000);
    let err = read_test_header(&file, &features).unwrap_err();
    assert!(err.message().contains("partition row ranges"), "{err}");
}

#[cfg(test)]
fn make_minimal_lepton_header() -> Box<LeptonHeader> {
    // minimal jpeg that will pass the validity read tests

    use crate::jpeg::jpeg_header::parse_jpeg_header;
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

    parse_jpeg_header(
        &mut Cursor::new(min_jpeg),
        &enabled_features,
        &mut lh.jpeg_header,
        &mut lh.rinfo,
    )
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
        .write_lepton_header(&mut output, enabled_features)
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
