# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences.

## Features and aims

- Implements DCT-based baseline JPEGs: 8-bits samples within each component, sequential, two DC and two AC Huffman tables, up to four components.
- Flexible API: no implicit synchronization between host and device in the decoding process, explicit device memory management, allowing flexible reuse, and thread safe.

## Benchmarks

AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070. 12MP 4:2:0 image with restart intervals, (`chunk_size = 16`):

```shell
jpeggpu singlethreaded, throughput: 80.78 image/s, avg latency: 12.38ms, max latency: 15.00ms
 nvJPEG singlethreaded, throughput: 66.01 image/s, avg latency: 15.15ms, max latency: 19.00ms
```

## To do

- Address JPEG applications headers/extensions, like color profiles and EXIF metadata. The goal is to ignore this and interpret one channel as gray, (not support two channels), three channels as YCbCr, and four channels CMYK. No interpretation of application headers will happen. It should be possible to output the data unchanged so that a user themselves can interpret.
- Issue in support of baseline standard implementation: handling a mix of non-interleaved and interleaved scans.

## Other

- No attempt is made to support non-standard JPEGs (no EOF marker, table index out of bounds, etc.).
