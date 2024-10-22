/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

use std::{fmt::Display, io::ErrorKind};

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

    // Add new failures here
    GeneralFailure = 1000,
    //CompressionFailedForAllChunks = 1001,
    //CompressedDataLargerThanPlainText = 1002,
    //HeaderChecksumMismatch = 1003,
    VerificationLengthMismatch = 1004,
    VerificationContentMismatch = 1005,
    SyntaxError = 1006,
    FileNotFound = 1007,
}

impl Display for ExitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Standard error returned by Lepton library
#[derive(Debug, Clone)]
pub struct LeptonError {
    /// standard error code
    exit_code: ExitCode,

    /// diagnostic message including location. Content should not be relied on.
    message: String,
}

impl Display for LeptonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}: {1}", self.exit_code, self.message)
    }
}

impl LeptonError {
    pub fn new(exit_code: ExitCode, message: &str) -> LeptonError {
        LeptonError {
            exit_code,
            message: message.to_owned(),
        }
    }

    pub fn exit_code(&self) -> ExitCode {
        self.exit_code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::error::Error for LeptonError {}

impl From<anyhow::Error> for LeptonError {
    fn from(mut error: anyhow::Error) -> Self {
        // first see if there is a LeptonError already inside
        match error.downcast::<LeptonError>() {
            Ok(le) => {
                return le;
            }
            Err(old_error) => {
                error = old_error;
            }
        }

        // capture the original error string before we lose it due
        // to downcasting to look for stashed LeptonErrors
        let original_string = error.to_string();

        // see if there is a LeptonError hiding inside an io error
        // which happens if we cross an API boundary that returns an std::io:Error
        // like Read or Write
        match error.downcast::<std::io::Error>() {
            Ok(ioe) => match ioe.downcast::<LeptonError>() {
                Ok(le) => {
                    return le;
                }
                Err(e) => {
                    return LeptonError {
                        exit_code: get_io_error_exit_code(&e),
                        message: format!("{} {}", e, original_string),
                    };
                }
            },
            Err(_) => {}
        }

        // don't know what we got, so treat it as a general failure
        return LeptonError {
            exit_code: ExitCode::GeneralFailure,
            message: original_string,
        };
    }
}

fn get_io_error_exit_code(e: &std::io::Error) -> ExitCode {
    if e.kind() == ErrorKind::UnexpectedEof {
        ExitCode::ShortRead
    } else {
        ExitCode::OsError
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
                let caller = std::panic::Location::caller();
                return LeptonError {
                    exit_code: get_io_error_exit_code(&e),
                    message: format!("error {} at {}", e.to_string(), caller.to_string()),
                };
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
    fn my_std_error() -> Result<(), std::io::Error> {
        Err(LeptonError::new(ExitCode::SyntaxError, "test error").into())
    }

    let e: LeptonError = my_std_error().unwrap_err().into();
    assert_eq!(e.exit_code, ExitCode::SyntaxError);
    assert_eq!(e.message, "test error");

    // wrapping inside anyhow
    fn my_anyhow() -> Result<(), anyhow::Error> {
        Err(LeptonError::new(ExitCode::SyntaxError, "test error").into())
    }

    let e: LeptonError = my_anyhow().unwrap_err().into();
    assert_eq!(e.exit_code, ExitCode::SyntaxError);
    assert_eq!(e.message, "test error");

    // an IO error should be translated into an OsError
    let e: LeptonError = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into();
    assert_eq!(e.exit_code, ExitCode::OsError);
}
