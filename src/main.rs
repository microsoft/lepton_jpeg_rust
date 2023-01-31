/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

mod consts;
mod enabled_features;
mod helpers;
mod jpeg_code;
mod lepton_error;
mod structs;

use anyhow;
use anyhow::Context;
use helpers::err_exit_code;
use lepton_error::{ExitCode, LeptonError};
use structs::lepton_format::read_jpeg;

use std::io::{Seek, Write};
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Cursor, Read},
    time::Instant,
};

use crate::enabled_features::EnabledFeatures;
use crate::helpers::here;
use crate::structs::lepton_format::{decode_lepton_wrapper, encode_lepton_wrapper, LeptonHeader};

fn parse_numeric_parameter(arg: &str, name: &str) -> Option<i32> {
    if arg.starts_with(name) {
        Some(arg[name.len()..].parse::<i32>().unwrap())
    } else {
        None
    }
}

// wrap main so that errors get printed nicely without a panic
fn main_with_result() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    let mut filenames = Vec::new();
    let mut num_threads = 8;
    let mut iterations = 1;
    let mut verify = false;
    let mut dump = false;
    let mut all = false;
    let mut enabled_features = EnabledFeatures::all();

    for i in 1..args.len() {
        if args[i].starts_with("-") {
            if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-threads:") {
                num_threads = x;
            } else if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-iter:") {
                iterations = x;
            } else if args[i] == "-dump" {
                dump = true;
            } else if args[i] == "-all" {
                all = true;
            } else if args[i] == "-verify" {
                verify = true;
            } else if args[i] == "-noprogressive" {
                enabled_features.progressive = false;
            } else {
                return err_exit_code(
                    ExitCode::SyntaxError,
                    format!("unknown switch {0}", args[i]).as_str(),
                );
            }
        } else {
            filenames.push(args[i].as_str());
        }
    }

    if dump {
        let file_in = File::open(filenames[0]).unwrap();
        let filelen = file_in.metadata()?.len() as u64;

        let mut reader = BufReader::new(file_in);

        let mut lh;
        let block_image;

        if filenames[0].to_lowercase().ends_with(".jpg") {
            (lh, block_image) = read_jpeg(
                &mut reader,
                &EnabledFeatures::all(),
                num_threads as usize,
                |jh| {
                    println!("parsed header:");
                    let s = format!("{jh:?}");
                    println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));
                },
            )
            .context(here!())?;
        } else {
            lh = LeptonHeader::new();
            lh.read_lepton_header(&mut reader).context(here!())?;

            block_image = lh
                .decode_as_single_image(&mut reader, filelen, num_threads as usize)
                .context(here!())?;

            loop {
                println!("parsed header:");
                let s = format!("{0:?}", lh.jpeg_header);
                println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));

                if !lh.advance_next_header_segment().context(here!())? {
                    break;
                }
            }
        }

        let s = format!("{lh:?}");
        println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));

        if all {
            for i in 0..block_image.len() {
                println!("Component {0}", i);
                let image = &block_image[i];
                for dpos in 0..image.get_block_width() * image.get_original_height() {
                    print!("dpos={0} ", dpos);
                    let block = image.get_block(dpos);

                    print!("{0}", block.get_coefficient_zigzag(0));
                    for i in 1..64 {
                        print!(",{0}", block.get_coefficient_zigzag(i));
                    }
                    println!();
                }
            }
        }

        return Ok(());
    }

    // this does a roundtrip verification of a single file and compares the output
    if verify {
        if filenames.len() != 1 {
            println!("requires one filename for verification");
            std::process::exit(100);
        }

        let mut file_in = File::open(filenames[0])
            .map_err(|e| LeptonError {
                exit_code: ExitCode::FileNotFound,
                message: e.to_string(),
            })
            .context(here!())?;

        // read the entire file into a buffer so we can compare it afterwards
        let mut input = Vec::new();
        file_in.read_to_end(&mut input).context(here!())?;

        for i in 0..iterations {
            let mut output = Vec::new();
            let now = Instant::now();

            {
                println!("encoding...");
                let mut input_cursor = Cursor::new(&input);
                let mut output_cursor = Cursor::new(&mut output);
                encode_lepton_wrapper(
                    &mut input_cursor,
                    &mut output_cursor,
                    num_threads as usize,
                    &enabled_features,
                )
                .context(here!())?;
            }

            let mut verify = Vec::new();

            {
                println!("decoding...");
                let mut input_cursor = Cursor::new(&output);
                let mut output_cursor = Cursor::new(&mut verify);

                decode_lepton_wrapper(&mut input_cursor, &mut output_cursor, num_threads as usize)
                    .context(here!())?;
            }

            if verify.len() != input.len() {
                return err_exit_code(
                    ExitCode::VerificationLengthMismatch,
                    format!(
                        "ERROR input_len = {0}, output_len = {1}",
                        input.len(),
                        output.len()
                    )
                    .as_str(),
                );
            }
            if input[..] != verify[..] {
                return err_exit_code(
                    ExitCode::VerificationContentMismatch,
                    "ERROR mismatching data (but same size)",
                );
            }

            println!("OK! itr {0} - {1}ms elapsed", i, now.elapsed().as_millis());
        }

        return Ok(());
    }

    if filenames.len() != 2 {
        return err_exit_code(
            ExitCode::SyntaxError,
            "source and destination filename are needed",
        );
    }

    for i in 0..iterations {
        let file_in = File::open(filenames[0])
            .map_err(|e| LeptonError {
                exit_code: ExitCode::FileNotFound,
                message: e.to_string(),
            })
            .context(here!())?;
        let mut reader = BufReader::new(file_in);

        let output_file: String = filenames[1].to_owned();
        //output_file.push_str("output");

        let fileout = File::create(output_file.as_str()).context(here!())?;
        let mut writer = BufWriter::new(fileout);
        //let mut writer = VerifyWriter::new( BufWriter::new(fileout), File::open("C:\\temp\\tgood.jpg")? );

        let now = Instant::now();
        if filenames[0].to_lowercase().ends_with(".jpg") {
            encode_lepton_wrapper(
                &mut reader,
                &mut writer,
                num_threads as usize,
                &enabled_features,
            )
            .context(here!())?;
        } else {
            decode_lepton_wrapper(&mut reader, &mut writer, num_threads as usize)
                .context(here!())?;
        }
        println!("itr {0} - {1}ms elapsed", i, now.elapsed().as_millis());
    }

    Ok(())
}

/// internal debug utility used to figure out where in the output the JPG diverged if there was a coding error writing out the JPG
struct VerifyWriter<W> {
    output: W,
    good_data: Vec<u8>,
    offset: usize,
}

impl<W> VerifyWriter<W> {
    // used for debugging
    #[allow(dead_code)]
    pub fn new<R: Read>(output: W, mut reader: R) -> Self {
        let mut r = VerifyWriter {
            output,
            offset: 0,
            good_data: Vec::new(),
        };
        reader.read_to_end(&mut r.good_data).unwrap();
        r
    }
}
impl<W: Write + Seek> Seek for VerifyWriter<W> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.output.seek(pos)
    }
}

impl<W: Write + Seek> Write for VerifyWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let goodslice = &self.good_data[self.offset..self.offset + buf.len()];

        if goodslice[..] != buf[..] {
            for i in 0..goodslice.len() {
                if goodslice[i] != buf[i] {
                    println!("at position {0}", self.output.stream_position()? + i as u64);

                    self.output.write_all(buf)?;
                    self.output.flush()?;
                    panic!("mismatched file!");
                }
            }
        }

        self.offset += buf.len();
        self.output.write_all(buf)?;
        return Ok(buf.len());
    }

    fn flush(&mut self) -> std::io::Result<()> {
        return self.output.flush();
    }
}

fn main() {
    match main_with_result() {
        Ok(_) => {}
        Err(e) => match e.root_cause().downcast_ref::<LeptonError>() {
            // try to extract the exit code if it was a well known error
            Some(x) => {
                println!(
                    "error code: {0} {1} {2}",
                    x.exit_code, x.exit_code as i32, x.message
                );
                std::process::exit(x.exit_code as i32);
            }
            None => {
                println!("unknown error {0:?}", e);
                std::process::exit(ExitCode::GeneralFailure as i32);
            }
        },
    }
}
