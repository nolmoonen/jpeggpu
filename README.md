# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences.

## Features and aims

- Implements DCT-based baseline JPEGs: 8-bits samples within each component, sequential, two DC and two AC Huffman tables, up to four components.
- Thread safe (except logging, for the moment).
- No implicit synchronization between host and device in the decoding process.

## Benchmarks

AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070. 12MP 4:2:0 image with restart intervals.

```shell
 nvJPEG 1661 GB/s 3.333 device tmp 0.157 pinned tmp
jpeggpu 996.6 GB/s 7.537 device tmp 0 pinned tmp
```

## TODO

- Address JPEG applications headers/extensions, like color profiles and EXIF metadata. Currently, this is mostly ignored. It would probably also be good to optionally disable this interpretation.
- Add some test.
- Add thread-safe logging.
- Some optimization

## Other

- The code has no support for non-standard JPEGs (no EOF, table idx OOB, etc.). This might be added later after the implementation is stable.
