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
mod metrics;
mod structs;

use anyhow;
use anyhow::Context;
use helpers::err_exit_code;
use lepton_error::{ExitCode, LeptonError};
use lepton_jpeg::metrics::CpuTimeMeasure;
use log::info;
use simple_logger::SimpleLogger;
use structs::lepton_format::read_jpeg;
#[cfg(target_os = "windows")]
use thread_priority::{set_current_thread_priority, ThreadPriority, WinAPIThreadPriority};

use std::{
    env,
    fs::{File, OpenOptions},
    io::{stdin, stdout, BufReader, Cursor, IsTerminal, Read, Seek, Write},
    time::Duration,
};

use crate::enabled_features::EnabledFeatures;
use crate::helpers::here;
use crate::structs::lepton_format::{
    decode_lepton_wrapper, encode_lepton_wrapper_verify, LeptonHeader,
};

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
    let mut dump = false;
    let mut all = false;
    let mut overwrite = false;
    let mut enabled_features = EnabledFeatures::compat_lepton_vector_read();

    // only output the log if we are connected to a console (otherwise if there is redirection we would corrupt the file)
    if stdout().is_terminal() {
        SimpleLogger::new().init().unwrap();
    }

    for i in 1..args.len() {
        if args[i].starts_with("-") {
            if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-threads:") {
                num_threads = x;
            } else if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-iter:") {
                iterations = x;
            } else if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-max-width:") {
                enabled_features.max_jpeg_width = x;
            } else if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-max-height:") {
                enabled_features.max_jpeg_height = x;
            } else if args[i] == "-dump" {
                dump = true;
            } else if args[i] == "-all" {
                all = true;
            } else if args[i] == "-highpriority" {
                // used to force to run on p-cores, make sure this and
                // any threadpool threads are set to the high priority

                #[cfg(target_os = "windows")]
                {
                    let priority = ThreadPriority::Os(WinAPIThreadPriority::TimeCritical.into());

                    set_current_thread_priority(priority).unwrap();

                    let b = rayon::ThreadPoolBuilder::new();
                    b.start_handler(move |_| {
                        set_current_thread_priority(priority).unwrap();
                    })
                    .build_global()
                    .unwrap();
                }
            } else if args[i] == "-lowpriority" {
                // used to force to run on e-cores, make sure this and
                // any threadpool threads are set to the high priority

                #[cfg(target_os = "windows")]
                {
                    let priority = ThreadPriority::Os(WinAPIThreadPriority::Idle.into());

                    set_current_thread_priority(priority).unwrap();

                    let b = rayon::ThreadPoolBuilder::new();
                    b.start_handler(move |_| {
                        set_current_thread_priority(priority).unwrap();
                    })
                    .build_global()
                    .unwrap();
                }
            } else if args[i] == "-overwrite" {
                overwrite = true;
            } else if args[i] == "-noprogressive" {
                enabled_features.progressive = false;
            } else if args[i] == "-acceptdqtswithzeros" {
                enabled_features.reject_dqts_with_zeros = false;
            } else if args[i] == "-use16bitdc" {
                enabled_features.use_16bit_dc_estimate = true;
            } else if args[i] == "-useleptonscalar" {
                // lepton files that were encoded by the dropbox c++ version compiled in scalar mode
                enabled_features.use_16bit_adv_predict = false;
                enabled_features.use_16bit_dc_estimate = false;
            } else if args[i] == "-useleptonvector" {
                // lepton files that were encoded by the dropbox c++ version compiled in AVX2/SSE2 mode
                enabled_features.use_16bit_adv_predict = true;
                enabled_features.use_16bit_dc_estimate = true;
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
            (lh, block_image) =
                read_jpeg(&mut reader, &enabled_features, num_threads as usize, |jh| {
                    println!("parsed header:");
                    let s = format!("{jh:?}");
                    println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));
                })
                .context(here!())?;
        } else {
            lh = LeptonHeader::new();
            lh.read_lepton_header(&mut reader, &mut enabled_features)
                .context(here!())?;

            let _metrics;

            (block_image, _metrics) = lh
                .decode_as_single_image(
                    &mut reader.take(filelen - 4), // last 4 bytes are the length of the file
                    num_threads as usize,
                    &enabled_features,
                )
                .context(here!())?;

            loop {
                println!("parsed header:");
                let s = format!("{0:?}", lh.jpeg_header);
                println!("{0}", s.replace("},", "},\r\n").replace("],", "],\r\n"));

                if !lh
                    .advance_next_header_segment(&enabled_features)
                    .context(here!())?
                {
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

                    print!("{0}", block.get_transposed_from_zigzag(0));
                    for i in 1..64 {
                        print!(",{0}", block.get_transposed_from_zigzag(i));
                    }
                    println!();
                }
            }
        }

        return Ok(());
    }

    let mut input_data = Vec::new();
    if filenames.len() != 2 {
        if stdout().is_terminal() || stdin().is_terminal() {
            return err_exit_code(
                ExitCode::SyntaxError,
                "source and destination filename are needed or input needs to be redirected",
            );
        }

        std::io::stdin()
            .read_to_end(&mut input_data)
            .context(here!())?;
    } else {
        let mut file_in = File::open(filenames[0])
            .map_err(|e| LeptonError {
                exit_code: ExitCode::FileNotFound,
                message: e.to_string(),
            })
            .context(here!())?;

        file_in.read_to_end(&mut input_data).context(here!())?;
    }

    if input_data.len() < 2 {
        return err_exit_code(ExitCode::BadLeptonFile, "ERROR input file too small");
    }

    let mut metrics;
    let mut output_data;

    let mut overall_cpu = Duration::ZERO;

    let mut current_iteration = 0;
    loop {
        let thread_cpu = CpuTimeMeasure::new();

        if input_data[0] == 0xff && input_data[1] == 0xd8 {
            // the source is a JPEG file, so run the encoder and verify the results
            (output_data, metrics) = encode_lepton_wrapper_verify(
                &input_data[..],
                num_threads as usize,
                &enabled_features,
            )
            .context(here!())?;

            info!(
                "compressed input {0}, output {1} bytes (ratio = {2:.1}%)",
                input_data.len(),
                output_data.len(),
                ((input_data.len() as f64) / (output_data.len() as f64) - 1.0) * 100.0
            );
        } else if input_data[0] == 0xcf && input_data[1] == 0x84 {
            // the source is a lepton file, so run the decoder
            let mut reader = Cursor::new(&input_data);

            output_data = Vec::with_capacity(input_data.len());

            metrics = decode_lepton_wrapper(
                &mut reader,
                &mut output_data,
                num_threads as usize,
                &enabled_features,
            )
            .context(here!())?;
        } else {
            return err_exit_code(
                ExitCode::BadLeptonFile,
                "ERROR input file is not a valid JPEG or Lepton file",
            );
        }

        let iter_duration = thread_cpu.elapsed() + metrics.get_cpu_time_worker_time();

        info!("Total CPU time consumed:{0}ms", iter_duration.as_millis());

        overall_cpu += iter_duration;

        current_iteration += 1;
        if current_iteration >= iterations {
            break;
        }
    }

    if filenames.len() != 2 {
        std::io::stdout()
            .write_all(&output_data[..])
            .context(here!())?
    } else {
        let output_file: String = filenames[1].to_owned();
        let mut fileout = OpenOptions::new()
            .write(true)
            .create(overwrite)
            .create_new(!overwrite)
            .open(output_file.as_str())
            .context(here!())?;

        fileout.write_all(&output_data[..]).context(here!())?
    }

    if iterations > 1 {
        info!(
            "Overall average CPU consumed per iteration {0}ms ",
            overall_cpu.as_millis() / (iterations as u128)
        );
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
                    eprintln!("at position {0}", self.output.stream_position()? + i as u64);

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
                eprintln!(
                    "error code: {0} {1} {2}",
                    x.exit_code, x.exit_code as i32, x.message
                );
                std::process::exit(x.exit_code as i32);
            }
            None => {
                eprintln!("unknown error {0:?}", e);
                std::process::exit(ExitCode::GeneralFailure as i32);
            }
        },
    }
}
