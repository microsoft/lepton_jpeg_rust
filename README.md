# Lepton JPEG Compression in Rust 
[![Rust Community](https://img.shields.io/badge/Rust_Community%20-Join_us-brightgreen?style=plastic&logo=rust)](https://www.rust-lang.org/community)

This is a port of the C++ Lepton JPEG compression tool that was released by DropBox [dropbox/lepton](https://github.com/dropbox/lepton). We developed a port of the library to Rust, which has basically the same performance characteristics with the advantage of all the safety features that Rust has to offer, due to the work involved in performing an exhaustive security check on the C++ code and the fact that DropBox has deprecated the codebase.

With precise bit-by-bit recovery of the original JPEG, the Lepton compression library is designed for lossless compression of baseline and progressive JPEGs up to 22%. JPEG storage in a cloud storage system is the main application case. Even metadata headers and invalid content are kept in good condition.


## How to Use This Library
Some operations of this library are vectorized such as the IDCT using the [Wide](https://crates.io/crates/wide) crate, so you can get a significant boost if you enable +AVX2.

#### Building From Source

- [Rust 1.65 or Above](https://www.rust-lang.org/tools/install)

```
git clone https://github.com/microsoft/lepton_jpeg_rust
cd lepton_jpeg_rust
cargo build
cargo test
cargo build --release
```

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

