# Design

## Overall approach

This library is designed to encode/decode JPEGs in a compressed file format that typically compresses files by about 20%. The overall approach is as follows:

- Split JPEG into metadata/headers (stored as an binary array) and scan data (which is stored as a set of arrays of 8x8 16 bit coefficients per color channel)
- The headers/metadata are compressed via Zlib and stored at the beginning of the lepton compressed files
- The scan data is Huffman decoded, while verifying that it was encoded canonically (this is important since we canonically encode so that it is binary identical)
- The scan data is encoded using the VP8 CABAC, with coefficients binarized using [Exponential-Golomb coding](https://en.wikipedia.org/wiki/Exponential-Golomb_coding). The bins for the CABAC encoder are determined by a fairly complex predictor model for:
  - DC (the top left corner coefficient)
  - The top and left edges (which are correlated to the previous blocks)
  - The 7x7  block (the remaining 49 coefficients that are not the DC or the edges)
  - It is vital that the model is identically and deterministically updated during encoding and decoding, since any discrepancy will rapidly cause the encoder and decoder to get out of sync and fail to decode the image
- In order to increase response time, the scan data is partitioned by up to 8 into horizontal sections, each of which can be encoded/decode on a separate thread. 
- Progressive JPEGs are handled slightly differently since they cannot be partitioned during the JPEG encoding step, since each progressive scan requires access to the entire image data.
- As a last verification, the entire process is run in reverse to ensure that we can recreate the binary-identical JPEG

## Layers

The main layers of the library are as follows:

- `lepton_format.rs` implements reading and writing Lepton format files and launching partitioned decoder/encoder threads
- `lepton_encoder.rs` / `lepton_decoder.rs` perform the actual scan encoding/decoding using `model.rs` to track the bin probabilities in `branch.rs`
- JPEGs are read and written by `jpeg_header.rs, jpeg_read.rs, jpeg_write.rs`, `jpeg_position_state.rs` which are invoked by `lepton_format.rs`
- `bit_reader.rs / bit_writer.rs` are used by the Huffman encoding/decoding for reading writing JPEG format scan data
- `vpx_bool_reader.rs / vpx_bool_writer.rs` are the CABAC encoder/decoder that using an arithmetic encoded binary stream with the probability of each bin is calculated in `brach.rs`. 
- `idct.rs` performs an inverse DCT of the JPEG coefficients as part of predicting the pixel values of neighbouring blocks
- `thread_handoff.rs` used to partition the JPEG scan data so that multiple threads can process the same image. 