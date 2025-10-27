import lepton_jpeg_python

def test_compress_decompress():
    # load slr city from images directory
    with open("../images/slrcity.jpg", "rb") as f:
        jpg_data = f.read()

    config = {
        "max_jpeg_width": 4096,
        "max_jpeg_height": 4096,
        "progressive": False,
        "reject_dqts_with_zeros": True,
        "max_partitions": 8,
        "max_jpeg_file_size": 128 * 1024 * 1024 }

    compressed = lepton_jpeg_python.compress_bytes(jpg_data, config)
    decompressed = lepton_jpeg_python.decompress_bytes(compressed)

    assert jpg_data == decompressed
    print("Compression and decompression successful!")