/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See License.txt in the project root for license information.
 *  This software incorporates material from third parties. See Notices.txt for details.
 *----------------------------------------/----------------------------------------------------*/

use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum ExitCode {
    //AssertionFailure = 1,
    //CodingError = 2,
    //ShortRead = 3,
    Unsupported4Colors = 4,
    CoefficientOutOfRange = 6,
    StreamInconsistent = 7,
    ProgressiveUnsupported = 8,
    SamplingBeyondTwoUnsupported = 10,
    //SamplingBeyondFourUnsupported = 11,
    //ThreadingPartialMcu = 12,
    VersionUnsupported = 13,
    //OnlyGarbageNoJpeg = 14,
    //OsError = 33,
    //HeaderTooLarge = 34,
    //BlockOffsetOOM = 37,
    UnsupportedJpeg = 42,
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

#[derive(Debug)]
pub struct LeptonError {
    pub exit_code: ExitCode,
    pub message: String,
}

impl Display for LeptonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{0}: {1}", self.exit_code, self.message)
    }
}

impl std::error::Error for LeptonError {}
