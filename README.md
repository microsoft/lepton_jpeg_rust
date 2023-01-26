# Lepton JPEG compression Rust port

This is a port of the C++ Lepton JPEG compression tool that was released by DropBox in this location: [dropbox/lepton: Lepton is a tool and file format for losslessly compressing JPEGs by an average of 22%. (github.com)](https://github.com/dropbox/lepton)

Due to the work involved in doing a complete security audit on the C++ code, and the fact that DropBox has deprecated the codebase, we created a port of the library to Rust, which has almost identical performance characteristics with the advantage of all the safety features the Rust offers.

## Lepton Compression Library
The source of the library itself is under the src directory, with integration tests in the test directory. There are various test images under the images folder.

Building the project is fairly straightforward if you have rust installed. cargo build and cargo test do what you would expect, and cargo build --release creates the optimized release version.

There is an Lepton_Rust.exe wrapper that is built as part of the project. It can be used to compress;/decompress and also to verify the test end-to-end on a given JPEG.

## Contributing

There are many ways in which you can participate in this project, for example:

* [Submit bugs and feature requests](https://github.com/microsoft/lepton_jpeg_rust/issues), and help us verify as they are checked in
* Review [source code changes](https://github.com/microsoft/lepton_jpeg_rust/pulls) or submit your own features as pull requests.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [Apache 2.0](LICENSE.txt) license.