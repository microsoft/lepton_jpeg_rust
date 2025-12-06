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
    /// Assertion failure, which indicates probably indicated a bug in the library.
    AssertionFailure = 1,
    //CodingError = 2
    /// The JPEG file is too short to be a valid JPEG file.
    ShortRead = 3,

    /// We don't support 4-color JPEGs.
    Unsupported4Colors = 4,

    /// The coefficients in the JPEG file are out of range specified by the JPEG standard.
    CoefficientOutOfRange = 6,

    /// The lepton file has a coding error in the arithmetic coding part.
    StreamInconsistent = 7,

    /// The JPEG file is progressive, and progressive support is not enabled.
    ProgressiveUnsupported = 8,

    /// The JPEG file has a sampling factor that is not supported by the library.
    SamplingBeyondTwoUnsupported = 10,
    //SamplingBeyondFourUnsupported = 11,
    //ThreadingPartialMcu = 12,
    /// The lepton file is a version that is not supported by the library.
    VersionUnsupported = 13,
    //OnlyGarbageNoJpeg = 14,
    /// An error was returned by an IO operation, for example if a BufRead
    /// passed in retrned an error.
    OsError = 33,
    //HeaderTooLarge = 34,
    //BlockOffsetOOM = 37,
    /// The JPEG cannot be encoded due to a non-standard feature that is not supported by the library.
    UnsupportedJpeg = 42,

    /// The JPEG file has a zero IDCT, which is not supported by the library.
    /// Although the C++ library doesn't explicitly disallow this, it will lead to
    /// undefined behavior depending on C++, since it can lead to a division-by-zero.
    UnsupportedJpegWithZeroIdct0 = 43,

    /// The JPEG file has invalid reset codes in the stream
    InvalidResetCode = 44,

    /// The JPEG uses inconsistent padding, which is not supported by the library.
    InvalidPadding = 45,
    //WrapperOutputWriteFailed = 101,
    /// The Lepton file is not a valid Lepton file.
    BadLeptonFile = 102,

    /// An error occurred while sending a message to a thread in the thread pool.
    ChannelFailure = 103,

    /// error occured while casting an integer to a smaller type, most likely
    /// means that the JPEG contains invalid data
    IntegerCastOverflow = 1000,
    //CompressionFailedForAllChunks = 1001,
    //CompressedDataLargerThanPlainText = 1002,
    //HeaderChecksumMismatch = 1003,
    /// We verified against the original JPEG file but the regenerated length was different
    VerificationLengthMismatch = 1004,

    /// We verified against the original JPEG file but the content was different (but same length)
    VerificationContentMismatch = 1005,

    /// Caller passed in invalid parameters
    SyntaxError = 1006,

    /// The file to be read was not found (only used by utility exe)
    FileNotFound = 1007,

    /// An external verification failed (only used by utility exe when verifying
    /// against C++ Lepton implementation)
    ExternalVerificationFailed = 1008,

    /// ran out of memory trying to allocate a buffer
    OutOfMemory = 2000,
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
    /// Creates a new LeptonError with the specified exit code and message.
    pub fn new(exit_code: ExitCode, message: impl AsRef<str>) -> LeptonError {
        LeptonError {
            i: Box::new(LeptonErrorInternal {
                exit_code,
                message: message.as_ref().to_owned(),
            }),
        }
    }

    /// Returns the numeric exit code of the error to clasify the error
    pub fn exit_code(&self) -> ExitCode {
        self.i.exit_code
    }

    /// Returns the message of the error, which is a human-readable description of the error.
    pub fn message(&self) -> &str {
        &self.i.message
    }

    /// Adds context to the error by appending the current location in the code. This
    /// allows for building a callstack of where the error occurred.
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
pub fn err_exit_code<T>(error_code: ExitCode, message: impl AsRef<str>) -> Result<T> {
    let mut e = LeptonError::new(error_code, message.as_ref());
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
        let mut e = LeptonError::new(ExitCode::IntegerCastOverflow, e.to_string());
        e.add_context();
        e
    }
}

impl<T> From<std::sync::mpsc::SendError<T>> for LeptonError {
    #[track_caller]
    fn from(e: std::sync::mpsc::SendError<T>) -> Self {
        let mut e = LeptonError::new(ExitCode::ChannelFailure, e.to_string());
        e.add_context();
        e
    }
}
impl From<std::sync::mpsc::RecvError> for LeptonError {
    #[track_caller]
    fn from(e: std::sync::mpsc::RecvError) -> Self {
        let mut e = LeptonError::new(ExitCode::ChannelFailure, e.to_string());
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
                let mut e = LeptonError::new(get_io_error_exit_code(&e), e.to_string());
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
