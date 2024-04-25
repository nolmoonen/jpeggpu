# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences.

## Features and aims

- Implements DCT-based baseline JPEGs: 8-bits samples within each component, sequential, two DC and two AC Huffman tables, up to four components.
- Thread safe (except logging, for the moment).
- No implicit synchronization between host and device in the decoding process.

## Benchmarks

AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070. 12MP 4:2:0 image with restart intervals, (`chunk_size = 16`):

```shell
jpeggpu singlethreaded, throughput: 80.78 image/s, avg latency: 12.38ms, max latency: 15.00ms
 nvJPEG singlethreaded, throughput: 66.01 image/s, avg latency: 15.15ms, max latency: 19.00ms
```

## To do

- Address JPEG applications headers/extensions, like color profiles and EXIF metadata. Currently, this is mostly ignored. It would probably also be good to optionally disable this interpretation.
- Add some test.
- Some optimization (try loading 32 at a time or buffer in shared memory)
- Handling a mix of non-interleaved and interleaved scans.

## Other

- No attempt is made to support non-standard JPEGs (no EOF marker, table index out of bounds, etc.).
