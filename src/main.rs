/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

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
use std::process::{Command, Stdio};
use std::time::Instant;
use std::{
    env,
    fs::File,
    io::{stdin, stdout, Cursor, IsTerminal, Read, Seek, Write},
    time::Duration,
};

#[derive(Copy, Clone, Debug)]
enum FileType {
    Jpeg,
    Lepton,
}

// wrap main so that errors get printed nicely without a panic
fn main_with_result() -> Result<(), LeptonError> {
    let args: Vec<String> = env::args().collect();

    let mut verify_cpp = None;
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
        match *args[i].split(':').collect::<Vec<&str>>().as_slice() {
            ["-verifycpp", cpp] => verify_cpp = Some(cpp.to_string()),
            ["-threads", threads] => enabled_features.max_threads = threads.parse::<u32>().unwrap(),
            ["-iter", iter] => iterations = iter.parse::<i32>().unwrap(),
            ["-max-width", width] => {
                enabled_features.max_jpeg_width = width.parse::<i32>().unwrap()
            }
            ["-max-height", height] => {
                enabled_features.max_jpeg_height = height.parse::<i32>().unwrap()
            }
            ["-dump"] => dump = true,
            ["-all"] => all = true,
            ["-corrupt"] => {
                // randomly corrupt the files for testing
                corrupt = true;
            }
            ["-noverify"] => {
                verify = false;
            }
            ["-quiet"] => {
                filter_level = log::LevelFilter::Warn;
            }
            ["-version"] => {
                println!(
                    "compiled library Lepton version {}, git revision: {}",
                    env!("CARGO_PKG_VERSION"),
                    git_version::git_version!(
                        args = ["--abbrev=40", "--always", "--dirty=-modified"]
                    )
                );
            }
            ["-highpriority"] => {
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
            }
            ["-lowpriority"] => {
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
            }
            ["-overwrite"] => {
                overwrite = true;
            }
            ["-noprogressive"] => {
                enabled_features.progressive = false;
            }
            ["-acceptdqtswithzeros"] => {
                enabled_features.reject_dqts_with_zeros = false;
            }
            ["-use16bitdc"] => {
                enabled_features.use_16bit_dc_estimate = true;
            }
            ["-useleptonscalar"] => {
                // lepton files that were encoded by the dropbox c++ version compiled in scalar mode
                enabled_features.use_16bit_adv_predict = false;
                enabled_features.use_16bit_dc_estimate = false;
            }
            ["-useleptonvector"] => {
                // lepton files that were encoded by the dropbox c++ version compiled in AVX2/SSE2 mode
                enabled_features.use_16bit_adv_predict = true;
                enabled_features.use_16bit_dc_estimate = true;
            }
            _ => {
                if args[i].starts_with("-") {
                    return Err(LeptonError::new(
                        ExitCode::SyntaxError,
                        format!("unknown switch {0}", args[i]).as_str(),
                    )
                    .into());
                } else {
                    filenames.push(args[i].as_str());
                }
            }
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

    // see what file type we have
    let file_type = if input_data[0] == 0xff && input_data[1] == 0xd8 {
        FileType::Jpeg
    } else if input_data[0] == 0xcf && input_data[1] == 0x84 {
        FileType::Lepton
    } else {
        return Err(LeptonError::new(
            ExitCode::BadLeptonFile,
            "ERROR input file is not a valid JPEG or Lepton file",
        )
        .into());
    };

    loop {
        let thread_cpu = CpuTimeMeasure::new();
        let walltime = Instant::now();

        if corrupt {
            let r = simple_lcg(&mut seed) as usize % input_data.len();

            let bitnumber = simple_lcg(&mut seed) as usize % 8;

            input_data[r] ^= 1 << bitnumber;
        }

        // do the encoding/decoding, if we got an error and were corrupting the file, then restore the
        // original data and continue so we can try corrupting the file in different ways
        // per iteration
        match do_work(file_type, verify, &input_data, &enabled_features) {
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
            }
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

        let o = output_file.as_str();
        let mut fileout = OpenOptions::new()
            .write(true)
            .create(overwrite)
            .create_new(!overwrite)
            .open(o)?;

        fileout.set_len(output_data.len() as u64)?;
        fileout.write_all(&output_data[..])?;
        drop(fileout);

        // what we do is take the lepton output, and see if it recreates the input using the
        // CPP version of the encoder/decoder
        if let Some(v) = verify_cpp {
            let (output, exit_code, stderr) = call_executable_with_input(v.as_str(), o)?;
            if exit_code != 0 {
                log::error!("cpp exit code: {}", exit_code);

                return Err(LeptonError::new(
                    ExitCode::ExternalVerificationFailed,
                    format!(
                        "verify failed with exit code {0} stderr: {1}",
                        exit_code, stderr
                    )
                    .as_str(),
                ))?;
            }
            if output[..].len() != input_data.len() {
                return Err(LeptonError::new(
                    ExitCode::ExternalVerificationFailed,
                    format!(
                        "verify failed with different length {0} != {1}",
                        output[..].len(),
                        input_data.len()
                    )
                    .as_str(),
                ));
            }
            if output[..] != input_data[..] {
                return Err(LeptonError::new(
                    ExitCode::ExternalVerificationFailed,
                    "verify failed with different data",
                )
                .into());
            }
            log::info!("verify succeeded with cpp version");
        }
    }

    if iterations > 1 {
        info!(
            "Overall average CPU consumed per iteration {0}ms ",
            overall_cpu.as_millis() / (iterations as u128)
        );
    }

    Ok(())
}

/// does the actual encoding/decoding work
fn do_work(
    file_type: FileType,
    verify: bool,
    input_data: &Vec<u8>,
    enabled_features: &EnabledFeatures,
) -> Result<(Vec<u8>, Metrics), LeptonError> {
    let metrics;
    let mut output;

    match file_type {
        FileType::Jpeg => {
            if verify {
                (output, metrics) = encode_lepton_verify(input_data, enabled_features)?;
            } else {
                let mut reader = Cursor::new(input_data);
                output = Vec::with_capacity(input_data.len());
                let mut writer = Cursor::new(&mut output);

                metrics = encode_lepton(&mut reader, &mut writer, enabled_features)?
            }

            info!(
                "compressed input {0}, output {1} bytes (compression = {2:.1}%)",
                input_data.len(),
                output.len(),
                ((input_data.len() as f64) / (output.len() as f64) - 1.0) * 100.0
            );
        }
        FileType::Lepton => {
            let mut reader = Cursor::new(&input_data);

            output = Vec::with_capacity(input_data.len());

            metrics = decode_lepton(&mut reader, &mut output, &enabled_features)?;
        }
    }

    Ok((output, metrics))
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
        Err(e) => {
            eprintln!(
                "error code: {0} {1} {2}",
                e.exit_code(),
                e.exit_code().as_integer_error_code(),
                e.message()
            );
            std::process::exit(e.exit_code().as_integer_error_code());
        }
    }
}

pub fn call_executable_with_input(
    executable: &str,
    input_filename: &str,
) -> Result<(Vec<u8>, i32, String), LeptonError> {
    // temporary file to store the output of the cpp version so we can
    // compare it with the rust version
    let v = format!("{0}.verify", input_filename);
    let verify_filename = v.as_str();

    // delete if already exists
    let _ = std::fs::remove_file(verify_filename);

    log::info!(
        "verifying input filename {} with {}",
        input_filename,
        executable
    );

    // Spawn the command
    let child = Command::new(executable)
        .arg(input_filename)
        .arg(verify_filename)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Wait for the child process to exit and collect output
    let output = child.wait_with_output()?;

    let mut file_in = File::open(verify_filename).unwrap();
    let mut contents = Vec::new();
    file_in.read_to_end(&mut contents).unwrap();

    // remove the temporary file
    let _ = std::fs::remove_file(verify_filename);

    // Extract the stdout, stderr, and exit status
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-10000); // Handle the case where exit code is None

    Ok((contents, exit_code, stderr))
}
