use std::ffi::OsStr;
use std::fs::{self, ReadDir};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use uuid::Uuid;

use crate::{read_bounded_to_end, wait_with_bounded_stderr};
use lepton_jpeg::{EnabledFeatures, LeptonError};

pub struct RecursiveFiles {
    stack: Vec<ReadDir>,
}

impl RecursiveFiles {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            stack: vec![fs::read_dir(root)?],
        })
    }
}

impl Iterator for RecursiveFiles {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(dir) = self.stack.last_mut() {
            match dir.next() {
                Some(Ok(entry)) => {
                    let path = entry.path();
                    match entry.file_type() {
                        Ok(ft) if ft.is_dir() => {
                            if let Ok(rd) = fs::read_dir(&path) {
                                self.stack.push(rd);
                            }
                        }
                        Ok(ft) if ft.is_file() => {
                            return Some(path);
                        }
                        _ => {}
                    }
                }
                Some(Err(_)) => continue,
                None => {
                    self.stack.pop();
                }
            }
        }
        None
    }
}

struct RayonPool {}

impl lepton_jpeg::LeptonThreadPool for RayonPool {
    fn max_parallelism(&self) -> usize {
        rayon::current_num_threads()
    }

    fn run(&self, f: Box<dyn FnOnce() + Send + 'static>) {
        rayon::spawn(f);
    }
}

/// randomly corrupts data if there is a seed
pub fn corrupt_data_if_enabled(seed: &mut Option<u64>, input_data: &mut Vec<u8>) {
    fn simple_lcg(seed: &mut u64) -> u64 {
        let r = seed.wrapping_mul(6364136223846793005) + 1;
        *seed = r;
        r
    }

    if let Some(seed) = seed
        && !input_data.is_empty()
    {
        let op = simple_lcg(seed);
        let r = simple_lcg(seed) as usize % input_data.len();

        match op % 5 {
                0 => {
                    // flip bit
                    let bitnumber = simple_lcg(seed) as usize % 8;
                    input_data[r] ^= 1 << bitnumber;
                }
                1 => {
                    // truncate file
                    input_data.truncate(r);
                }
                2 => {
                    // insert random byte
                    let random_byte = (simple_lcg(seed) & 0xFF) as u8;
                    input_data.insert(r, random_byte);
                }
                3 => {
                    // delete byte
                    if input_data.len() > 1 {
                        input_data.remove(r);
                    }
                }
                4
                    // truncate by 1
                    if input_data.len() > 1 => {
                        input_data.truncate(input_data.len() - 1);
                    }
                _ => {
                    // do nothing
                }
            }
    }
}

pub fn verify_dir(
    root_path: &Path,
    cpp_executable: &Path,
    corruption_seed: &mut Option<u64>,
    max_jpeg_file_size: u32,
) -> Result<(), LeptonError> {
    let iter = RecursiveFiles::new(root_path)?;

    for file_path in iter {
        if file_path.extension().and_then(|s| s.to_str()) != Some("jpg") {
            continue;
        }
        call_executable_with_input(
            cpp_executable,
            file_path.as_os_str(),
            corruption_seed,
            max_jpeg_file_size,
        )?;
    }
    Ok(())
}

/// calls the CPP version of the encoder/decoder to verify the output of the Rust version
pub fn call_executable_with_input(
    cpp_executable: &Path,
    input_filename: &OsStr,
    corruption_seed: &mut Option<u64>,
    max_jpeg_file_size: u32,
) -> Result<(), LeptonError> {
    let mut input_file = std::fs::File::open(input_filename)?;
    let mut input_data = read_bounded_to_end(
        &mut input_file,
        u64::from(max_jpeg_file_size),
        "verification input",
    )?;
    corrupt_data_if_enabled(corruption_seed, &mut input_data);

    // write to temporary file with potential corruption
    let temp_filename_input =
        std::env::temp_dir().join(format!("lepton_jpeg_util_cpp_{}.jpg", Uuid::new_v4()));
    std::fs::write(&temp_filename_input, &input_data)?;

    let temp_filename_output =
        std::env::temp_dir().join(format!("lepton_jpeg_util_cpp_{}.lep", Uuid::new_v4()));

    // delete output if already exists
    let _ = std::fs::remove_file(&temp_filename_output);

    // Spawn the command
    let child = Command::new(cpp_executable)
        .arg(&temp_filename_input)
        .arg(&temp_filename_output)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;

    let (exit_code, stderr) = wait_with_bounded_stderr(child)?;

    if exit_code != 0 {
        log::info!(
            "CPP executable failed for file {:?} with exit code {}: {}",
            input_filename,
            exit_code,
            stderr
        );
    } else {
        let mut cpp_lepton_file = fs::File::open(&temp_filename_output)?;
        let max_lepton_size = 2 * u64::from(max_jpeg_file_size) + 1024 * 1024;
        let cpp_lepton_data = read_bounded_to_end(
            &mut cpp_lepton_file,
            max_lepton_size,
            "external encoder output",
        )?;

        let mut writer = Vec::new();

        if let Err(e) = lepton_jpeg::decode_lepton(
            &mut Cursor::new(&cpp_lepton_data),
            &mut writer,
            &EnabledFeatures::compat_lepton_vector_read(),
            &RayonPool {},
        ) {
            panic!(
                "Error decoding CPP output for file {}: {} {} seed:{:?}",
                input_filename.to_string_lossy(),
                e,
                stderr,
                corruption_seed
            );
        }

        if writer != input_data {
            println!(
                "Original size: {}, Re-coded: {}",
                input_data.len(),
                writer.len()
            );

            fs::write("r_corrupted_input.jpg", &input_data).unwrap();
            fs::write("r_cpp_lepton.lep", &cpp_lepton_data).unwrap();
            fs::write("r_rust_recorded.jpg", &writer).unwrap();

            panic!(
                "Verification failed for file {}: output does not match original {} {} seed:{:?}",
                input_filename.to_string_lossy(),
                exit_code,
                stderr,
                corruption_seed
            );
        }

        log::info!("Verified file: {}", input_filename.to_string_lossy());
    }

    _ = std::fs::remove_file(&temp_filename_input);
    _ = std::fs::remove_file(&temp_filename_output);
    Ok(())
}
