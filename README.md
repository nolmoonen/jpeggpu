# TODO

- Deal with restart intervals. Since the restart marker must be byte-aligned, padding may be inserted after the encoded segment. This causes issues incorrect decoding. Figure out if restart segements should be handled specially (not favourable) or some bug exists. After decoding a MCU could check if next full byte is a restart marker. But that would require removing byte-destuffing beforehand.
- Add support for non-interleaved scans.
- Add support for other SOFs?
- Add optional logging, enable logging in example_tool.
- Remove CPU decoder (by removing assumptions in GPU kernel) and replace CPU IDCT by a GPU version.
- Implement full baseline.
