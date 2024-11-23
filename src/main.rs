/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::borrow::Cow;
use std::env;
use std::ffi::OsStr;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{stdin, stdout, Cursor, IsTerminal, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use lepton_jpeg::metrics::CpuTimeMeasure;
use lepton_jpeg::{
    decode_lepton, dump_jpeg, encode_lepton, encode_lepton_verify, EnabledFeatures, ExitCode,
    LeptonError, Metrics,
};
use log::{error, info};
use simple_logger::SimpleLogger;

#[derive(Copy, Clone, Debug)]
enum FileType {
    Jpeg,
    Lepton,
}

fn parse_i32(s: &str) -> Result<i32, &'static str> {
    s.parse().map_err(|_| "not a number")
}

fn parse_u32(s: &str) -> Result<u32, &'static str> {
    s.parse().map_err(|_| "not a number")
}

fn parse_u64(s: &str) -> Result<u64, &'static str> {
    s.parse().map_err(|_| "not a number")
}

fn parse_path(s: &OsStr) -> Result<PathBuf, &'static str> {
    Ok(PathBuf::from(s))
}

fn override_if<T>(
    pargs: &mut pico_args::Arguments,
    name: &'static str,
    parse: fn(&str) -> Result<T, &'static str>,
    value: &mut T,
) -> Result<(), pico_args::Error> {
    if let Some(v) = pargs.opt_value_from_fn(name, parse)? {
        *value = v;
    }
    Ok(())
}

// wrap main so that errors get printed nicely without a panic
fn main_with_result() -> Result<(), LeptonError> {
    let mut pargs = pico_args::Arguments::from_env();

    let mut enabled_features = EnabledFeatures::compat_lepton_vector_read();
    let mut filter_level = log::LevelFilter::Info;

    if pargs.contains(["-h", "--help"]) {
        println!(
"lepton_jpeg_util - a fast JPEG compressor

Usage: lepton_jpeg_util [options] inputfile [outputfile]

Options:
    --iter <n>              number of iterations to run
    --dump                  dump the JPEG file
    --all                   dump includes the scan lines
    --cppverify <exe path>  verify the output with the C++ decoder
    --overwrite             overwrite the output file
    --corrupt <seed>        randomly corrupt the input file (for testing)
    --quiet                 suppress all output
    --noverify              do not verify the output
    --max-width <n>         maximum width of the JPEG file
    --max-height <n>        maximum height of the JPEG file
    --max-jpeg-file-size <n> maximum size of the JPEG file
    --threads <n>           maximum number of threads to use
    --rejectprogressive     reject progressive JPEG files
    --rejectdqtswithzeros   reject DQT tables with zeros
    --rejectinvalidhuffman  reject invalid Huffman tables
    --use32bitdc            use 32 bit DC estimate
    --use32bitadv           use 32 bit advanced prediction
    --useleptonscalar       use the scalar version of the encoder
    --highpriority          run on p-cores
    --lowpriority           run on e-cores
    --version               print the version
    --help                  print this help message
    --verifydir             Recursively verify all files in a directory can be compressed and decompressed
");
        return Ok(());
    }

    let cppverify: Option<PathBuf> = pargs.opt_value_from_os_str("--cppverify", parse_path)?;
    let verify_dir = pargs.opt_value_from_os_str("--verifydir", parse_path)?;

    let iterations = pargs.opt_value_from_fn("--iter", parse_i32)?.unwrap_or(1);
    let dump = pargs.contains("--dump");
    let dumpall = pargs.contains("--dumpall");
    let verify = !pargs.contains("--noverify");
    let overwrite = pargs.contains("--overwrite");
    let mut corrupt = pargs.opt_value_from_fn("--corrupt", parse_u64)?;

    if pargs.contains("--quiet") {
        filter_level = log::LevelFilter::Warn;
    }

    override_if(
        &mut pargs,
        "--max-width",
        parse_i32,
        &mut enabled_features.max_jpeg_width,
    )?;

    override_if(
        &mut pargs,
        "--max-height",
        parse_i32,
        &mut enabled_features.max_jpeg_height,
    )?;

    override_if(
        &mut pargs,
        "--threads",
        parse_u32,
        &mut enabled_features.max_threads,
    )?;

    override_if(
        &mut pargs,
        "--rejectprogressive",
        |_| Ok(false),
        &mut enabled_features.progressive,
    )?;

    override_if(
        &mut pargs,
        "--rejectdqtswithzeros",
        |_| Ok(true),
        &mut enabled_features.reject_dqts_with_zeros,
    )?;

    override_if(
        &mut pargs,
        "--rejectinvalidhuffman",
        |_| Ok(false),
        &mut enabled_features.accept_invalid_dht,
    )?;

    override_if(
        &mut pargs,
        "--max-jpeg-file-size",
        parse_u32,
        &mut enabled_features.max_jpeg_file_size,
    )?;

    if pargs.contains("--version") {
        println!(
            "compiled library Lepton version {}, git revision: {}",
            env!("CARGO_PKG_VERSION"),
            git_version::git_version!(args = ["--abbrev=40", "--always", "--dirty=-modified"])
        );
    }

    if pargs.contains("--use32bitdc") {
        enabled_features.use_16bit_dc_estimate = false;
    }
    if pargs.contains("--use32bitadv") {
        enabled_features.use_16bit_adv_predict = false;
    }
    if pargs.contains("--useleptonscalar") {
        // use both these options if you are trying to read a file that was encoded with the scalar version of the C++ encoder
        // sadly one old version of the Rust encoder used use_16bit_dc_estimate=false, use_16bit_adv_predict=true
        // the latest version of the encoder put these options in the header so we ignore this if the file specifies it
        enabled_features.use_16bit_adv_predict = false;
        enabled_features.use_16bit_dc_estimate = false;
    }

    #[cfg(not(feature = "use_rayon"))]
    if pargs.contains("--highpriority") {
        // used to force to run on p-cores, make sure this and
        // any threadpool threads are set to the highest priority
        lepton_jpeg::set_thread_priority(100);
    }

    #[cfg(not(feature = "use_rayon"))]
    if pargs.contains("--lowpriority") {
        // used to force to run on e-cores, make sure this and
        // any threadpool threads are set to the lowest priority
        lepton_jpeg::set_thread_priority(0);
    }

    let filenames = pargs.finish();

    for i in filenames.iter() {
        // no other options should be specified only the free standing filenames
        if i.to_string_lossy().starts_with("-") {
            return Err(LeptonError::new(
                ExitCode::SyntaxError,
                format!("unknown option {:?}", i).as_str(),
            )
            .into());
        }
    }

    // if we are verifying a directory, then we need to recursively verify all files in the directory
    if let Some(verify_dir) = verify_dir {
        execute_verify_dir(
            &cppverify,
            &verify_dir.as_path(),
            &enabled_features,
            verify,
            &mut corrupt,
        )?;
        return Ok(());
    }

    // only output the log if we are connected to a console (otherwise if there is redirection we would corrupt the file)
    if stdout().is_terminal() {
        SimpleLogger::new().with_level(filter_level).init().unwrap();
    }

    if dump {
        let mut file_in = File::open(filenames[0].as_os_str()).unwrap();

        let mut contents = Vec::new();
        file_in.read_to_end(&mut contents).unwrap();
        dump_jpeg(&contents, dumpall, &enabled_features).unwrap();
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
        let mut file_in = File::open(filenames[0].as_os_str())
            .map_err(|e| LeptonError::new(ExitCode::FileNotFound, e.to_string().as_str()))?;

        file_in.read_to_end(&mut input_data)?;
    }

    if input_data.len() < 2 {
        return Err(LeptonError::new(ExitCode::BadLeptonFile, "ERROR input file too small").into());
    }

    let mut metrics;
    let mut output_data;

    let mut overall_cpu = Duration::ZERO;

    let mut current_iteration = 0;

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

    // get a writable version of the input data so we can corrupt it if the user wants to
    let mut writable_input_data = Cow::from(&input_data);

    loop {
        let thread_cpu = CpuTimeMeasure::new();
        let walltime = Instant::now();

        corrupt_data_if_enabled(&mut corrupt, &mut writable_input_data.to_mut());

        // do the encoding/decoding, if we got an error and were corrupting the file, then restore the
        // original data and continue so we can try corrupting the file in different ways
        // per iteration
        match do_work(file_type, verify, &writable_input_data, &enabled_features) {
            Err(e) => {
                error!("error {0}", e);

                // if we corrupted the image, then restore and continue running
                if corrupt.is_some() {
                    // reset the input data not be be corrupt anymore
                    writable_input_data = Cow::from(&input_data);
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
        let output_filename = filenames[1].as_os_str();

        let mut fileout = OpenOptions::new()
            .write(true)
            .create(overwrite)
            .create_new(!overwrite)
            .open(output_filename)?;

        // ignore if this failed (etc on a pipe)
        let _ = fileout.set_len(output_data.len() as u64);
        fileout.write_all(&output_data[..])?;
        drop(fileout);

        // what we do is take the lepton output, and see if it recreates the input using the
        // CPP version of the encoder/decoder
        if let Some(cpp_path) = cppverify {
            execute_cpp_verify(cpp_path.as_path(), output_filename, &writable_input_data)?;
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

/// randomly corrupts data if there is a seed
fn corrupt_data_if_enabled(seed: &mut Option<u64>, input_data: &mut Vec<u8>) {
    fn simple_lcg(seed: &mut u64) -> u64 {
        let r = seed.wrapping_mul(6364136223846793005) + 1;
        *seed = r;
        r
    }

    if let Some(seed) = seed {
        if input_data.len() > 0 {
            let r = simple_lcg(seed) as usize % input_data.len();

            let bitnumber = simple_lcg(seed) as usize % 8;

            input_data[r] ^= 1 << bitnumber;
        }
    }
}

/// recursively verify all files in a directory, including potentially verifying it with the CPP version of the decoder
/// to make sure that we didn't break the format in some unexpected way
fn execute_verify_dir(
    cpp_executable: &Option<PathBuf>,
    dir: &Path,
    enabled_features: &EnabledFeatures,
    verify: bool,
    corrupt_data_seed: &mut Option<u64>,
) -> Result<(), LeptonError> {
    let entries;
    match std::fs::read_dir(dir) {
        Ok(e) => entries = e,
        Err(e) => {
            eprintln!("error reading directory {:?} {:?}", dir, e);
            return Ok(());
        }
    }

    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            execute_verify_dir(
                cpp_executable,
                &path,
                enabled_features,
                verify,
                corrupt_data_seed,
            )?;
            continue;
        }

        if let Some(x) = path.extension() {
            if x != "jpg" && x != "jpeg" {
                continue;
            }

            let mut file_in;
            match File::open(&path) {
                Ok(f) => file_in = f,
                Err(e) => {
                    eprintln!("error reading file {:?} {:?}", path, e);
                    continue;
                }
            }

            let mut original_contents = Vec::new();
            file_in.read_to_end(&mut original_contents).unwrap();

            corrupt_data_if_enabled(corrupt_data_seed, &mut original_contents);

            match do_work(
                FileType::Jpeg,
                verify,
                &original_contents,
                &enabled_features,
            ) {
                Err(e) => {
                    eprintln!("{:?} error {}", path, e);
                }
                Ok((output, _)) => {
                    if let Some(cpp_executable) = cpp_executable {
                        // create the input file for the lepton C++ decoder
                        let verify_output = std::env::temp_dir().join("lepton_jpeg_util_cpp.lep");

                        let mut file_out = File::create(&verify_output).unwrap();
                        file_out.write_all(&output[..]).unwrap();
                        drop(file_out);

                        let r = execute_cpp_verify(
                            cpp_executable,
                            verify_output.as_os_str(),
                            &original_contents,
                        );
                        let _ = remove_file(verify_output);
                        if r.is_err() {
                            eprintln!("{:?} CPP_VERIFY error {:?}", path, r);
                            if let Some(s) = corrupt_data_seed {
                                eprintln!("corruption seed was {0}", s);
                            }
                            // abort here, this is bad
                            return r;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn execute_cpp_verify(
    cpp_executable: &Path,
    compressed_file: &OsStr,
    original_contents: &[u8],
) -> Result<(), LeptonError> {
    let (output, exit_code, stderr) =
        call_executable_with_input(cpp_executable, compressed_file).unwrap();

    if exit_code != 0 {
        log::error!("cpp exit code: {}", exit_code);

        return Err(LeptonError::new(
            ExitCode::ExternalVerificationFailed,
            format!(
                "cpp verify failed with exit code {0} stderr: {1}",
                exit_code, stderr
            )
            .as_str(),
        ))?;
    }
    if output[..].len() != original_contents.len() {
        return Err(LeptonError::new(
            ExitCode::ExternalVerificationFailed,
            format!(
                "cpp verify failed with different length {0} != {1}",
                output[..].len(),
                original_contents.len()
            )
            .as_str(),
        ));
    }
    if output[..] != original_contents[..] {
        return Err(LeptonError::new(
            ExitCode::ExternalVerificationFailed,
            "verify failed with different data",
        )
        .into());
    }
    log::info!("verify succeeded with cpp version");
    Ok(())
}

/// does the actual encoding/decoding work
fn do_work(
    file_type: FileType,
    verify: bool,
    input_data: &[u8],
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

/// calls the CPP version of the encoder/decoder to verify the output of the Rust version
pub fn call_executable_with_input(
    cpp_executable: &Path,
    input_filename: &OsStr,
) -> Result<(Vec<u8>, i32, String), LeptonError> {
    // temporary file to store the output of the cpp version so we can
    // compare it with the rust version

    let temp_filename_buf = std::env::temp_dir().join("lepton_jpeg_util_cpp_recreate.jpg");
    let temp_filename = temp_filename_buf.as_os_str();

    // delete if already exists
    let _ = std::fs::remove_file(temp_filename);

    log::info!(
        "verifying input filename with CPP {:?} with {:?}",
        temp_filename,
        cpp_executable
    );

    // Spawn the command
    let child = Command::new(cpp_executable)
        .arg(input_filename)
        .arg(temp_filename)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Wait for the child process to exit and collect output
    let output = child.wait_with_output()?;

    let mut file_in = File::open(&temp_filename).unwrap();
    let mut contents = Vec::new();
    file_in.read_to_end(&mut contents).unwrap();

    // remove the temporary file
    let _ = std::fs::remove_file(temp_filename);

    // Extract the stdout, stderr, and exit status
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-10000); // Handle the case where exit code is None

    Ok((contents, exit_code, stderr))
}
