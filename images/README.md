# Images

- `dc_init_non_interleaved` is a progressive JPEG with DC initial scans that do not contain all components. Associated Lepton file produced by the original implementation.
- `eof_and_trailinghdrdata` the lepton format has a wrongly set unexpected EOF and trailing header data.
- `eof_and_trailingrst` the Lepton format has a wrongly set unexpected EOF and trailing RST.
- `out_of_order_dqt` has a quantization table that comes after the image definition SOF.
- `progressive_late_dht`  has Huffman tables that come very late which causes a verification failure.
- `scan_order_reversed` is a sequential, non-interleaved JPEG where the first scan does not contain the first component. Associated Lepton file produced by the original implementation.
- `truncbad` the Lepton format is truncated and invalid.
