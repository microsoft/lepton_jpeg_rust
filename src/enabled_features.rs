// features that are enabled in the encoder. Turn off for potential backward compat issues.
pub struct EnabledFeatures {
    /// disables reading of progressive images
    pub progressive: bool,
}

impl EnabledFeatures {
    pub fn all() -> Self {
        EnabledFeatures { progressive: true }
    }
}
