#![no_main]

use std::io::Cursor;

use lepton_jpeg::{decode_lepton, encode_lepton, EnabledFeatures};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let r;

    let mut output = Vec::new();

    // keep the jpeg dimensions small otherwise the fuzzer gets really slow
    let features = EnabledFeatures {
        progressive: true,
        reject_dqts_with_zeros: true,
        max_jpeg_height: 1024,
        max_jpeg_width: 1024,
        use_16bit_dc_estimate: false,
    };

    {
        let mut writer = Cursor::new(&mut output);

        r = encode_lepton(&mut Cursor::new(&data), &mut writer, 8, &features);
    }

    let mut original = Vec::new();

    match r {
        Ok(_) => {
            let _ = decode_lepton(&mut Cursor::new(&output), &mut original, 8, &features);
        }
        Err(_) => {}
    }
});
