// features that are enabled in the encoder. Turn off for potential backward compat issues.
pub struct EnabledFeatures {
    /// enables/disables reading of progressive images
    pub progressive: bool,

    // reject/accept images with DQTs with zeros (may cause divide-by-zero)
    pub reject_dqts_with_zeros: bool,

    /// maximum jpeg width
    pub max_jpeg_width: i32,

    // maximum jpeg height
    pub max_jpeg_height: i32,

    /// Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
    pub use_16bit_dc_estimate: bool,

    /// Sadly C++ version has a bug where it uses 16 bit math in the SIMD path and 32 bit math in the scalar path
    pub use_16bit_adv_predict: bool,
}

impl Default for EnabledFeatures {
    fn default() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_width: 16386,
            max_jpeg_height: 16386,
            use_16bit_dc_estimate: false,
            use_16bit_adv_predict: true,
        }
    }
}

impl EnabledFeatures {
    /// parameters that allow everything
    #[allow(dead_code)]
    pub fn all() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_height: i32::MAX,
            max_jpeg_width: i32::MAX,
            use_16bit_dc_estimate: false,
            use_16bit_adv_predict: true,
        }
    }
}
