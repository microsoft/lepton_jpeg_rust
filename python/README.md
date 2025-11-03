# Lepton JPEG Compression 

This is a port of the C++ Lepton JPEG compression tool that was released by DropBox [dropbox/lepton](https://github.com/dropbox/lepton). We developed a port of the library to Rust, which has basically the same performance characteristics with the advantage of all the safety features that Rust has to offer, due to the work involved in performing an exhaustive security check on the C++ code and the fact that DropBox has deprecated the codebase.

With precise bit-by-bit recovery of the original JPEG, the Lepton compression library is designed for lossless compression of baseline and progressive JPEGs up to 22%. JPEG storage in a cloud storage system is the main application case. Even metadata headers and invalid content are kept in good condition.


## How to Use This Library

The library exposes two methods, compress and decompress, which can be invoked as follows:

``` python

    with open("my image", "rb") as f:
        jpg_data = f.read()

    config = {"max_jpeg_width": 4096 }
    compressed = lepton_jpeg_python.compress_bytes(jpg_data, config)
    decompressed = lepton_jpeg_python.decompress_bytes(compressed)

    assert jpg_data == decompressed
```    

The following config options are supported:
- max_jpeg_width: reject compressing images wider than this
- max_jpeg_height: reject compressioning images taller than this
- progressive: false to forbid compressing progressive JPEGs
- reject_dqts_with_zeros: true if we should reject JPEGs with 0 in their quantitization table
- max_partitions: maximum number of partitions to split JPEG into in order to allow for parallel compression/decompression
- max_jpeg_file_size: reject JPEGs larger than this

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

