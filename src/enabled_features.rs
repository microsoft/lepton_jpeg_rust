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
}

impl Default for EnabledFeatures {
    fn default() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_width: 16386,
            max_jpeg_height: 16386,
        }
    }
}

impl EnabledFeatures {
    /// parameters that allow everything
    pub fn all() -> Self {
        Self {
            progressive: true,
            reject_dqts_with_zeros: true,
            max_jpeg_height: i32::MAX,
            max_jpeg_width: i32::MAX,
        }
    }
}
