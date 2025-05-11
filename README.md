# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences. This process is based on the paper _Accelerating JPEG Decompression on GPUs_[<sup>1</sup>](#references).

## Features and aims

- Implements DCT-based baseline JPEGs, see [JPEG Support](#jpeg-support) below for more details.
- Flexible API: no implicit synchronization between host and device in the decoding process, explicit device memory management allowing reuse, thread safety, C99 compatible, and OS independent.
- Simple library design: JPEG application segments are not interpreted, i.e. no attempt is made to interpret color space. No attempt is made to support non-standard JPEGs (no EOF marker, table index out of bounds, etc.).

## Building

Build with CMake, for example:

```shell
cmake -S . -B build
cmake --build build
```

## Example

`example/example_tool.c` is built as `jpeggpu_example`. It demonstrates basic usage of jpeggpu and outputs some information about the file.

```shell
./build/jpeggpu_example in.jpg
marker Start of image
marker Define quantization table(s)
marker Define restart interval
        restart_interval: 252
marker Baseline DCT
        size_x: 4032, size_y: 3024, num_components: 3
        c_id: 1, ssx: 2, ssy: 2, qi: 0
        c_id: 2, ssx: 1, ssy: 1, qi: 1
        c_id: 3, ssx: 1, ssy: 1, qi: 1
...
marker End of image
intra sync of 89 blocks of 256 subsequences
intra sync of 1 blocks of 131072 subsequences
gpu decode done
decoded image at: out.png
```

## Benchmark

`benchmark/benchmark.cpp` builds `jpeggpu_benchmark` that compares performance with nvJPEG, decoding a single image at a time.

Possible output with AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070, on images of [jpeg-test-images](https://github.com/nolmoonen/jpeg-test-images):

```shell
         throughput (image/s) | avg latency (ms) | max latency (ms)
006mp-cathedral.jpg
 jpeggpu               440.92               2.27               2.50
  nvJPEG               143.89               6.95               8.30
012mp-bus.jpg
 jpeggpu               164.36               6.08               7.34
  nvJPEG                66.97              14.93              15.30
026mp-temple.jpg
 jpeggpu                58.64              17.05              21.00
  nvJPEG                15.81              63.24              64.86
028mp-tree.jpg
 jpeggpu               133.61               7.48              10.15
  nvJPEG                33.63              29.74              30.66
039mp-building.jpg
 jpeggpu               129.50               7.72              10.21
  nvJPEG                34.33              29.13              46.60
```

Note that nvJPEG uses a hybrid (CPU+GPU) decoding, so nvJPEG has a throughput advantage when decoding multiple images in parallel.

## Test

`test/test.cpp` builds `jpeggpu_test` that compares output against nvJPEG. Helper script `test.sh` uses ImageMagick to convert an input image to a few different JPEG variations.

```shell
./build/jpeggpu_test test.jpg --write_out # writing out is optional
component 0 MSE: 0.23201 component 1 MSE: 0.198817 component 2 MSE: 0.199355
writing out to "test.jpg.nvjpeg.png" and "test.jpg.jpeggpu.png"

./build/test.sh test.jpg # can also optionally pass --write_out
creating tmp file test.jpg.1x1.jpg..
component 0 MSE: 0.202032 component 1 MSE: 0.155791 component 2 MSE: 0.155672 
creating tmp file test.jpg.2x1.jpg..
...
```

## JPEG support

JPEGGPU implements the full baseline process (see Table 1[<sup>2</sup>](#references)), with the extension of allowing up to four Huffman tables of each type:

- DCT-based process
- 8-bit samples within each component
- Sequential
- Huffman coding: 4 AC and 4 DC tables
- 1, 2, 3, or 4 components
- Interleaved and non-interleaved scans

Compared to nvJPEG, JPEGGPU does not support progressive JPEGs but has no restrictions on chroma subsampling. One estimate suggests 30% of JPEGs used in websites are progressive and 10% of JPEG photographs are progressive[<sup>3</sup>](#references). The parallel decoding method used in JPEGGPU is fundamentally incompatible with progressive JPEGs, specifically because of the AC refinement scan.

## References

1. [Accelerating JPEG Decompression on GPUs](https://arxiv.org/abs/2111.09219)
2. [T.81 - DIGITAL COMPRESSION AND CODING OF CONTINUOUS-TONE STILL IMAGES - REQUIREMENTS AND GUIDELINES (JPEG specification)](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
3. [Progressive JPEGs in the Wild: Implications for Information Hiding and Forensics](https://informationsecurity.uibk.ac.at/pdfs/HB2023_IHMMSEC.pdf)
