import lepton_jpeg_python

def test_compress_decompress():
    # load slr city from images directory
    with open("../images/slrcity.jpg", "rb") as f:
        jpg_data = f.read()

    compressed = lepton_jpeg_python.compress_bytes(jpg_data)
    decompressed = lepton_jpeg_python.decompress_bytes(compressed)

    assert jpg_data == decompressed
    print("Compression and decompression successful!")