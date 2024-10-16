/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use anyhow;
use lepton_jpeg::metrics::CpuTimeMeasure;
use lepton_jpeg::{
    decode_lepton, dump_jpeg, encode_lepton, encode_lepton_verify, EnabledFeatures, ExitCode,
    LeptonError, Metrics,
};
use log::{error, info};
use simple_logger::SimpleLogger;
#[cfg(all(target_os = "windows", feature = "use_rayon"))]
use thread_priority::{set_current_thread_priority, ThreadPriority, WinAPIThreadPriority};

use std::fs::OpenOptions;
use std::time::Instant;
use std::{
    env,
    fs::File,
    io::{stdin, stdout, Cursor, IsTerminal, Read, Seek, Write},
    time::Duration,
};

fn parse_numeric_parameter(arg: &str, name: &str) -> Option<i32> {
    if arg.starts_with(name) {
        Some(arg[name.len()..].parse::<i32>().unwrap())
    } else {
        None
    }
}

// wrap main so that errors get printed nicely without a panic
// wrap main so that errors get printed nicely without a panic
fn main_with_result() -> Result<(), anyhow::Error> {
    let args: Vec<String> = env::args().collect();

    let mut filenames = Vec::new();
    let mut iterations = 1;
    let mut dump = false;
    let mut all = false;
    let mut verify = true;
    let mut overwrite = false;
    let mut enabled_features = EnabledFeatures::compat_lepton_vector_read();
    let mut corrupt = false;
    let mut filter_level = log::LevelFilter::Info;

    for i in 1..args.len() {
        if args[i].starts_with("-") {
            if let Some(x) = parse_numeric_parameter(args[i].as_str(), "-threads:") {
                enabled_features.max_threads = x as u32;
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
            } else if args[i] == "-corrupt" {
                // randomly corrupt the files for testing
                corrupt = true;
            } else if args[i] == "-noverify" {
                verify = false;
            } else if args[i] == "-quiet" {
                filter_level = log::LevelFilter::Warn;
            } else if args[i] == "-version" {
                println!(
                    "compiled library Lepton version {}, git revision: {}",
                    env!("CARGO_PKG_VERSION"),
                    git_version::git_version!(
                        args = ["--abbrev=40", "--always", "--dirty=-modified"]
                    )
                );
            } else if args[i] == "-highpriority" {
                // used to force to run on p-cores, make sure this and
                // any threadpool threads are set to the high priority

                #[cfg(all(target_os = "windows", feature = "use_rayon"))]
                {
                    let priority = ThreadPriority::Os(WinAPIThreadPriority::TimeCritical.into());

                    set_current_thread_priority(priority).unwrap();

                    let b = rayon_core::ThreadPoolBuilder::new();
                    b.start_handler(move |_| {
                        set_current_thread_priority(priority).unwrap();
                    })
                    .build_global()
                    .unwrap();
                }
            } else if args[i] == "-lowpriority" {
                // used to force to run on e-cores, make sure this and
                // any threadpool threads are set to the high priority

                #[cfg(all(target_os = "windows", feature = "use_rayon"))]
                {
                    let priority = ThreadPriority::Os(WinAPIThreadPriority::Idle.into());

                    set_current_thread_priority(priority).unwrap();

                    let b = rayon_core::ThreadPoolBuilder::new();
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
                return Err(LeptonError::new(
                    ExitCode::SyntaxError,
                    format!("unknown switch {0}", args[i]).as_str(),
                )
                .into());
            }
        } else {
            filenames.push(args[i].as_str());
        }
    }

    // only output the log if we are connected to a console (otherwise if there is redirection we would corrupt the file)
    if stdout().is_terminal() {
        SimpleLogger::new().with_level(filter_level).init().unwrap();
    }

    if dump {
        let mut file_in = File::open(filenames[0]).unwrap();

        let mut contents = Vec::new();
        file_in.read_to_end(&mut contents).unwrap();
        dump_jpeg(&contents, all, &enabled_features).unwrap();
        return Ok(());
    }

    let mut input_data = Vec::new();
    if filenames.len() != 2 {
        if stdout().is_terminal() || stdin().is_terminal() {
            return Err(LeptonError::new(
                ExitCode::SyntaxError,
                "source and destination filename are needed or input needs to be redirected",
            )
            .into());
        }

        std::io::stdin().read_to_end(&mut input_data)?;
    } else {
        let mut file_in = File::open(filenames[0])
            .map_err(|e| LeptonError::new(ExitCode::FileNotFound, e.to_string().as_str()))?;

        file_in.read_to_end(&mut input_data)?;
    }

    if input_data.len() < 2 {
        return Err(LeptonError::new(ExitCode::BadLeptonFile, "ERROR input file too small").into());
    }

    let mut metrics;
    let mut output_data;
    let mut original_data = Vec::new();

    // save the data if we are going to corrupt it
    if corrupt {
        original_data = input_data.clone();
    }

    let mut overall_cpu = Duration::ZERO;

    let mut current_iteration = 0;

    let mut seed = 0x123456789abcdef0;
    fn simple_lcg(seed: &mut u64) -> u64 {
        let r = seed.wrapping_mul(6364136223846793005) + 1;
        *seed = r;
        r
    }

    let is_jpeg = input_data[0] == 0xff && input_data[1] == 0xd8;
    let is_lepton = input_data[0] == 0xcf && input_data[1] == 0x84;

    loop {
        let thread_cpu = CpuTimeMeasure::new();
        let walltime = Instant::now();

        if corrupt {
            let r = simple_lcg(&mut seed) as usize % input_data.len();

            let bitnumber = simple_lcg(&mut seed) as usize % 8;

            input_data[r] ^= 1 << bitnumber;
        }

        if is_jpeg {
            // the source is a JPEG file, so run the encoder and verify the results

            let r = if verify {
                encode_lepton_verify(&input_data, &enabled_features)
            } else {
                let mut reader = Cursor::new(&input_data);
                output_data = Vec::with_capacity(input_data.len());
                let mut writer = Cursor::new(&mut output_data);
                match encode_lepton(&mut reader, &mut writer, &enabled_features) {
                    Ok(m) => Ok((output_data, m)),
                    Err(e) => Err(e),
                }
            };

            match r {
                Err(e) => {
                    error!("error {0}", e);

                    // if we corrupted the image, then restore and continue running
                    if corrupt {
                        input_data = original_data.clone();
                        output_data = Vec::new();
                        metrics = Metrics::default();
                    } else {
                        return Err(e.into());
                    }
                }

                Ok((data, m)) => {
                    output_data = data;
                    metrics = m;

                    info!(
                        "compressed input {0}, output {1} bytes (ratio = {2:.1}%)",
                        input_data.len(),
                        output_data.len(),
                        ((input_data.len() as f64) / (output_data.len() as f64) - 1.0) * 100.0
                    );
                }
            }
        } else if is_lepton {
            // the source is a lepton file, so run the decoder
            let mut reader = Cursor::new(&input_data);

            output_data = Vec::with_capacity(input_data.len());

            match decode_lepton(&mut reader, &mut output_data, &enabled_features) {
                Err(e) => {
                    error!("error {0}", e);

                    // if we corrupted the image, then restore and continue running
                    if corrupt {
                        input_data = original_data.clone();
                        metrics = Metrics::default();
                    } else {
                        return Err(e.into());
                    }
                }
                Ok(m) => {
                    metrics = m;
                }
            }
        } else {
            return Err(LeptonError::new(
                ExitCode::BadLeptonFile,
                "ERROR input file is not a valid JPEG or Lepton file",
            )
            .into());
        }

        let localthread = thread_cpu.elapsed();
        let workers = metrics.get_cpu_time_worker_time();

        info!(
            "Main thread CPU: {}ms, Worker thread CPU: {} ms, walltime: {} ms",
            localthread.as_millis(),
            workers.as_millis(),
            walltime.elapsed().as_millis()
        );

        overall_cpu += localthread + workers;

        current_iteration += 1;
        if current_iteration >= iterations {
            break;
        }
    }

    if filenames.len() != 2 {
        std::io::stdout().write_all(&output_data[..])?
    } else {
        let output_file: String = filenames[1].to_owned();
        let mut fileout = OpenOptions::new()
            .write(true)
            .create(overwrite)
            .create_new(!overwrite)
            .open(output_file.as_str())?;

        fileout.write_all(&output_data[..])?
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
                    x.exit_code(),
                    x.exit_code() as i32,
                    x.message()
                );
                std::process::exit(x.exit_code() as i32);
            }
            None => {
                eprintln!("unknown error {0:?}", e);
                std::process::exit(ExitCode::GeneralFailure as i32);
            }
        },
    }
}
