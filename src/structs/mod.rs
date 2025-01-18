/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt in the project root for license information.
 *  This software incorporates material from third parties. See NOTICE.txt for details.
 *--------------------------------------------------------------------------------------------*/

// Don't allow any unsafe code by default. Since this code has to potentially deal with
// badly/maliciously formatted images, we want this extra level of safety.
#![forbid(unsafe_code)]
#![forbid(trivial_numeric_casts)]

mod block_context;
mod branch;
//mod component_info;
mod div;
mod idct;
mod lepton_decoder;
mod lepton_encoder;
pub mod lepton_file_reader;
pub mod lepton_file_writer;
pub mod lepton_header;
mod model;
mod multiplexer;
mod neighbor_summary;
mod partial_buffer;
mod probability_tables;
mod quantization_tables;
mod simple_hash;

#[cfg(not(feature = "use_rayon"))]
pub mod simple_threadpool;

mod thread_handoff;
mod vpx_bool_reader;
mod vpx_bool_writer;
