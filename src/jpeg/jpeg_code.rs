/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

/// Start of Frame (size information), coding process: baseline DCT
pub const SOF0: u8 = 0xC0;

/// Start of Frame (size information), coding process: extended sequential DCT
pub const SOF1: u8 = 0xC1;

/// Start of Frame (size information), coding process: progressive DCT
pub const SOF2: u8 = 0xC2;

/// Huffman Table
pub const DHT: u8 = 0xC4;

/// Restart 0 segment
pub const RST0: u8 = 0xD0;

/// Start of Image
pub const SOI: u8 = 0xD8;

/// End of Image, or End of File
pub const EOI: u8 = 0xD9;

/// Start of Scan
pub const SOS: u8 = 0xDA;

/// Define Quantization Table
pub const DQT: u8 = 0xDB;

/// Define restart interval
pub const DRI: u8 = 0xDD;
