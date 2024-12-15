//! Module for reading and recreation of JPEGs without the loss of any information.
//!
//! This means that it should be possible to reconstruct bit-by-bit an exactly identical
//! JPEG file from the input.
//!
//! Note that we never actually decode the JPEG into pixels, since the DCT is lossy, so
//! processing needs to be done at the DCT coefficient level and keep the coefficients in
//! the BlockBasedImage identical.

mod bit_reader;
mod bit_writer;
mod component_info;
pub mod jpeg_code;
mod jpeg_position_state;

pub mod block_based_image;
pub mod jpeg_header;
pub mod jpeg_read;
pub mod jpeg_write;
pub mod row_spec;
pub mod truncate_components;
