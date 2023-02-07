// features that are enabled in the encoder. Turn off for potential backward compat issues.
pub struct EnabledFeatures {
    /// disables reading of progressive images
    pub progressive: bool,

    /// maximum jpeg width
    pub max_jpeg_width: i32,

    // maxmimum jpeg height
    pub max_jpeg_height: i32,
}

impl Default for EnabledFeatures {
    fn default() -> Self {
        Self {
            progressive: true,
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
            max_jpeg_height: i32::MAX,
            max_jpeg_width: i32::MAX,
        }
    }
}
