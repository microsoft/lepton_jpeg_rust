#![no_main]

use std::io::Cursor;

use lepton_jpeg::{decode_lepton, encode_lepton, EnabledFeatures};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let r;

    let mut output = Vec::new();

    {
        let mut writer = Cursor::new(&mut output);

        // keep the jpeg dimensions small otherwise the fuzzer gets really slow
        let features = EnabledFeatures {
            progressive: true,
            max_jpeg_height: 1024,
            max_jpeg_width: 1024,
        };

        r = encode_lepton(&mut Cursor::new(&data), &mut writer, 8, &features);
    }

    let mut original = Vec::new();

    match r {
        Ok(_) => {
            let _ = decode_lepton(&mut Cursor::new(&output), &mut original, 8);
        }
        Err(_) => {}
    }
});
