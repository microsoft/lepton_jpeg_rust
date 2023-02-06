#![no_main]

use std::io::{Cursor, Read, Seek, Write};

use lepton_jpeg::{
    decode_lepton, encode_lepton,
    lepton_error::{ExitCode, LeptonError},
};

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {

    let mut output = Vec::new();
    let mut writer = Cursor::new(output);

    let _ = encode_lepton(
        &mut Cursor::new(&data),
        &mut writer,
        8,
        false,
    );

});
