/// Features that are enabled in the encoder. Turn off for potential backward compat issues.
#[derive(Debug, Clone)]
pub struct EnabledFeatures {
    /// enables/disables reading of progressive images
    pub progressive: bool,

    /// reject/accept images with DQTs with zeros (may cause divide-by-zero)
    pub reject_dqts_with_zeros: bool,

    /// maximum jpeg width
    pub max_jpeg_width: u32,

    /// maximum jpeg height
    pub max_jpeg_height: u32,

    /// Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
    pub use_16bit_dc_estimate: bool,

    /// Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
    pub use_16bit_adv_predict: bool,

    /// Accept JPEG files that have invalid DHT tables
    pub accept_invalid_dht: bool,

    /// number of partitions used for encoding
    pub max_partitions: u32,

    /// maximum number of threads to use for encoding/decoding
    pub max_processor_threads: u32,

    /// maximum size of a jpeg file
    pub max_jpeg_file_size: u32,

    /// stop reading at the end of the valid JPEG file. This is useful if
    /// the stream contains other data after the EOI marker.
    ///
    /// This also disallows handling truncated JPEG files since by definition
    /// they don't have an EOI marker, instead you will get a ShortRead error.
    pub stop_reading_at_eoi: bool,
}

impl EnabledFeatures {
    /// parameters that allow everything for encoding that is compatible with c++ lepton compiled with SIMD
    #[allow(dead_code)]
    pub fn compat_lepton_vector_write() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_height: 16386,
            max_jpeg_width: 16386,
            use_16bit_dc_estimate: true,
            use_16bit_adv_predict: true,
            accept_invalid_dht: false,
            max_partitions: 8,
            max_processor_threads: 8,
            max_jpeg_file_size: 128 * 1024 * 1024,
            stop_reading_at_eoi: false,
        }
    }

    /// parameters that allow everything for decoding c++ lepton images encoded
    /// with the scalar compile options
    #[allow(dead_code)]
    pub fn compat_lepton_scalar_read() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: false,
            max_jpeg_height: u32::MAX,
            max_jpeg_width: u32::MAX,
            use_16bit_dc_estimate: false,
            use_16bit_adv_predict: false,
            accept_invalid_dht: true,
            max_partitions: 8,
            max_processor_threads: 8,
            max_jpeg_file_size: 128 * 1024 * 1024,
            stop_reading_at_eoi: false,
        }
    }

    /// parameters that allow everything for decoding c++ lepton images encoded
    /// with the vector (SSE2/AVX2) compile options
    #[allow(dead_code)]
    pub fn compat_lepton_vector_read() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: false,
            max_jpeg_height: u32::MAX,
            max_jpeg_width: u32::MAX,
            use_16bit_dc_estimate: true,
            use_16bit_adv_predict: true,
            accept_invalid_dht: true,
            max_partitions: 8,
            max_processor_threads: 8,
            max_jpeg_file_size: 128 * 1024 * 1024,
            stop_reading_at_eoi: false,
        }
    }
}
