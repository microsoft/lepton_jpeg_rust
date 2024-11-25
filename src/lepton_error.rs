/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::fmt::Display;
use std::io::ErrorKind;
use std::num::TryFromIntError;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
#[non_exhaustive]
/// Well-defined errors for bad things that are expected to happen as part of compression/decompression
pub enum ExitCode {
    AssertionFailure = 1,
    //CodingError = 2,
    ShortRead = 3,
    Unsupported4Colors = 4,
    CoefficientOutOfRange = 6,
    StreamInconsistent = 7,
    ProgressiveUnsupported = 8,
    SamplingBeyondTwoUnsupported = 10,
    //SamplingBeyondFourUnsupported = 11,
    //ThreadingPartialMcu = 12,
    VersionUnsupported = 13,
    //OnlyGarbageNoJpeg = 14,
    OsError = 33,
    //HeaderTooLarge = 34,
    //BlockOffsetOOM = 37,
    UnsupportedJpeg = 42,
    UnsupportedJpegWithZeroIdct0 = 43,
    InvalidResetCode = 44,
    InvalidPadding = 45,
    //WrapperOutputWriteFailed = 101,
    BadLeptonFile = 102,
    ChannelFailure = 103,

    // Add new failures here
    GeneralFailure = 1000,
    //CompressionFailedForAllChunks = 1001,
    //CompressedDataLargerThanPlainText = 1002,
    //HeaderChecksumMismatch = 1003,
    VerificationLengthMismatch = 1004,
    VerificationContentMismatch = 1005,
    SyntaxError = 1006,
    FileNotFound = 1007,
    ExternalVerificationFailed = 1008,
}

impl Display for ExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ExitCode {
    /// Converts the error code into an integer for use as an error code when
    /// returning from a C API.
    pub fn as_integer_error_code(self) -> i32 {
        self as i32
    }
}

/// Since errors are rare and stop everything, we want them to be as lightweight as possible.
#[derive(Debug, Clone)]
struct LeptonErrorInternal {
    exit_code: ExitCode,
    message: String,
}

/// Standard error returned by Lepton library
#[derive(Debug, Clone)]
pub struct LeptonError {
    i: Box<LeptonErrorInternal>,
}

pub type Result<T> = std::result::Result<T, LeptonError>;

impl Display for LeptonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}: {1}", self.i.exit_code, self.i.message)
    }
}

impl LeptonError {
    pub fn new(exit_code: ExitCode, message: &str) -> LeptonError {
        LeptonError {
            i: Box::new(LeptonErrorInternal {
                exit_code,
                message: message.to_owned(),
            }),
        }
    }

    pub fn exit_code(&self) -> ExitCode {
        self.i.exit_code
    }

    pub fn message(&self) -> &str {
        &self.i.message
    }

    #[cold]
    #[inline(never)]
    #[track_caller]
    pub fn add_context(&mut self) {
        self.i
            .message
            .push_str(&format!("\n at {}", std::panic::Location::caller()));
    }
}

#[cold]
#[track_caller]
pub fn err_exit_code<T>(error_code: ExitCode, message: &str) -> Result<T> {
    let mut e = LeptonError::new(error_code, message);
    e.add_context();
    return Err(e);
}

pub trait AddContext<T> {
    #[track_caller]
    fn context(self) -> Result<T>;
}

impl<T, E: Into<LeptonError>> AddContext<T> for core::result::Result<T, E> {
    #[track_caller]
    fn context(self) -> Result<T> {
        match self {
            Ok(x) => Ok(x),
            Err(e) => {
                let mut e = e.into();
                e.add_context();
                Err(e)
            }
        }
    }
}

impl std::error::Error for LeptonError {}

fn get_io_error_exit_code(e: &std::io::Error) -> ExitCode {
    if e.kind() == ErrorKind::UnexpectedEof {
        ExitCode::ShortRead
    } else {
        ExitCode::OsError
    }
}

impl From<TryFromIntError> for LeptonError {
    #[track_caller]
    fn from(e: TryFromIntError) -> Self {
        let mut e = LeptonError::new(ExitCode::GeneralFailure, e.to_string().as_str());
        e.add_context();
        e
    }
}

impl From<pico_args::Error> for LeptonError {
    #[track_caller]
    fn from(e: pico_args::Error) -> Self {
        let mut e = LeptonError::new(ExitCode::SyntaxError, e.to_string().as_str());
        e.add_context();
        e
    }
}

impl<T> From<std::sync::mpsc::SendError<T>> for LeptonError {
    #[track_caller]
    fn from(e: std::sync::mpsc::SendError<T>) -> Self {
        let mut e = LeptonError::new(ExitCode::ChannelFailure, e.to_string().as_str());
        e.add_context();
        e
    }
}
impl From<std::sync::mpsc::RecvError> for LeptonError {
    #[track_caller]
    fn from(e: std::sync::mpsc::RecvError) -> Self {
        let mut e = LeptonError::new(ExitCode::ChannelFailure, e.to_string().as_str());
        e.add_context();
        e
    }
}

/// translates std::io::Error into LeptonError
impl From<std::io::Error> for LeptonError {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        match e.downcast::<LeptonError>() {
            Ok(le) => {
                return le;
            }
            Err(e) => {
                let mut e = LeptonError::new(get_io_error_exit_code(&e), e.to_string().as_str());
                e.add_context();
                e
            }
        }
    }
}

/// translates LeptonError into std::io::Error, which involves putting into a Box and using Other
impl From<LeptonError> for std::io::Error {
    fn from(e: LeptonError) -> Self {
        return std::io::Error::new(std::io::ErrorKind::Other, e);
    }
}

#[test]
fn test_error_translation() {
    // test wrapping inside an io error
    fn my_std_error() -> core::result::Result<(), std::io::Error> {
        Err(LeptonError::new(ExitCode::SyntaxError, "test error").into())
    }

    let e: LeptonError = my_std_error().unwrap_err().into();
    assert_eq!(e.exit_code(), ExitCode::SyntaxError);
    assert_eq!(e.message(), "test error");

    // an IO error should be translated into an OsError
    let e: LeptonError = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into();
    assert_eq!(e.exit_code(), ExitCode::OsError);
}
