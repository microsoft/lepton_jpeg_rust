# Lepton JPEG compression Rust port

The Lepton compression library is designed for lossless compressION baseline and progressive JPEGs up to 22%, with exact bit-by-bit recovery of the original JPEG. The primary use case is for storing JPEGs in a cloud-storage system. Metadata headers, and even invalid content is preserved as-is.

This is a port of the C++ Lepton JPEG compression tool that was released by DropBox in this location: [dropbox/lepton: Lepton is a tool and file format for losslessly compressing JPEGs by an average of 22%. (github.com)](https://github.com/dropbox/lepton)

Due to the work involved in doing a complete security audit on the C++ code, and the fact that DropBox has deprecated the codebase, we created a port of the library to Rust, which has almost identical performance characteristics with the advantage of all the safety features the Rust offers.

## Lepton Compression Library
The source of the library itself is under the src directory, with integration tests in the test directory. There are various test images under the images folder.

#### Building

Building the project is fairly straightforward if you have **Rust 1.65 or later** installed (older version will warn about unstable features such as scoped threads). `cargo build` and `cargo test` do what you would expect, and `cargo build --release` creates the optimized release version.

Some operations are vectorized such as the IDCT using the [Wide](https://crates.io/crates/wide) crate, so you can get a significant boost if you enable +AVX2.

#### Running

There is an `lepton_jpeg_util.exe` wrapper that is built as part of the project. It can be used to compress/decompress and also to verify the test end-to-end on a given JPEG. If the input file has a `.jpg` extension, it will encode. If the input file has a `.lep` extension, it will decode back to the original`.jpg`. 

It supports the following options:

`lepton_jpeg_util.exe [options] <inputfile> [<outputfile>]`

| Option           | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `-threads:n`     | Runs with a maximum of n threads. For encoding, this limits the amount of parallelism that can be gotten out of the decoder. |
| `-dump`          | Dumps the contents of a JPG or LEP file, with the -all option, it will also dump the cooefficient image blocks |
| `-noprogressive` | Will cause an error if we encounter a progressive file rather than trying to encode it |
| `-verify`        | Reads, encodes and unencodes verifying that there is an exact match. No output file is specified. |
| `-iter:n`        | Runs N iterations of the operation. Useful when we are running inside a profiler. |

## Design

[Link to overall design of library](DESIGN.md)

## Contributing

There are many ways in which you can participate in this project, for example:

* [Submit bugs and feature requests](https://github.com/microsoft/lepton_jpeg_rust/issues), and help us verify as they are checked in
* Review [source code changes](https://github.com/microsoft/lepton_jpeg_rust/pulls) or submit your own features as pull requests.
* The library uses only **stable features**, so if you want to take advantage of SIMD features such as AVX2, use the Wide crate (see the idct.rs as an example) rather than intrinsics. 

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [Apache 2.0](LICENSE.txt) license.

