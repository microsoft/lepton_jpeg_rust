#![no_main]

//! Feeds arbitrary bytes directly to the Lepton decoder, which is the main untrusted-input path.
//!
//! Run with memory limits so oversized allocations are reported as crashes, e.g.:
//! cargo fuzz run fuzz_decode -- -malloc_limit_mb=512 -rss_limit_mb=2048

use std::io::Cursor;

use lepton_jpeg::{EnabledFeatures, SingleThreadPool, decode_lepton};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // keep the limits small so that the fuzzer stays fast and allocations stay bounded
    let features = EnabledFeatures {
        max_jpeg_height: 1024,
        max_jpeg_width: 1024,
        max_jpeg_pixels: 1024 * 1024,
        max_jpeg_file_size: 4 * 1024 * 1024,
        max_progressive_scans: 64,
        use_16bit_dc_estimate: data.len() % 2 == 1,
        ..EnabledFeatures::compat_lepton_vector_read()
    };

    let mut output = Vec::new();
    let _ = decode_lepton(
        &mut Cursor::new(data),
        &mut output,
        &features,
        &SingleThreadPool::default(),
    );
});
