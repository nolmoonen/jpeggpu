# TODO

- Deal with restart intervals. Since the restart marker must be byte-aligned, padding may be inserted after the encoded segment. This causes issues incorrect decoding. Figure out if restart segements should be handled specially (not favourable) or some bug exists. After decoding a MCU could check if next full byte is a restart marker. But that would require removing byte-destuffing beforehand.
- Add support for non-interleaved scans. Full baseline mode.
- Add support for other SOFs?
- Remove CPU decoder (by removing assumptions in GPU kernel) and replace CPU IDCT by a GPU version.
- Implement full baseline. With the restriction of only supporting the popular subsamplings and 8 bits data.
- Switch to host-based segment detection.
- Implement GPU transpose.
- Fix temp memory calculation.
