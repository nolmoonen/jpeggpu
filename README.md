# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences.

## Features and aims

- Implements DCT-based baseline JPEGs: 8-bits samples within each component, sequential, two DC and two AC Huffman tables, up to four components. Limitation: it is assumed that the first component is not subsampled, that the remaining components are subsampled with the same rate, and that the factor is either 1, 2, or 4. This is all the case for many popular subsampling factors like 4:4:4. 4:2:0, etc.
- Thread safe (except logging, for the moment).
- No implicit synchronization between host and device in the decoding process. One exception currently is the inter-sequence synchronization process.

## Benchmarks

AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070. 12MP 4:2:0 image with restart intervals.

```shell
 nvJPEG 1661 GB/s 3.333 device tmp 0.157 pinned tmp
jpeggpu 996.6 GB/s 7.537 device tmp 0 pinned tmp
```

## TODO

- Clarify where assumptions are made about only supporting 444, 420, etc. to see if the assumption can be lifted. Add check if CSS is one of "popular subsamplings", reject otherwise.
- Add subsamplings and color formats to convert function and interface
- Add some test.
- Add thread-safe logging.
- Clarify how applications headers are handled. Add API function to disable color interpretation and output unchanged.
- Some optimization
