# JPEGGPU

JPEGGPU is an experimental JPEG decoder implemented in CUDA. It works by decoding many sequences of the encoded stream in parallel, and then synchronizing these decoded sequences. This process is based on the paper [Accelerating JPEG Decompression on GPUs](https://arxiv.org/abs/2111.09219).

## Features and aims

- Implements DCT-based baseline JPEGs: 8-bits samples within each component, sequential, two DC and two AC Huffman tables, and up to four components. No progressive JPEGs or extensions are handled.
- Flexible API: no implicit synchronization between host and device in the decoding process, explicit device memory management allowing reuse, thread safety, C99 compatible, and OS independent.
- Simple library design: JPEG application segments are not interpreted, i.e. no attempt is made to interpret color space. No attempt is made to support non-standard JPEGs (no EOF marker, table index out of bounds, etc.).

## Building

Build with CMake, for example:

```shell
cmake -S . -B build
cmake --build build
```

## Example

`example_tool.c` is built as `jpeggpu_example`. It demonstrates basic usage of jpeggpu and outputs some information about the file.

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

`jpeggpu_benchmark` compares performance with nvJPEG.

Possible output with AMD Ryzen 5 2600 and NVIDIA GeForce RTX 2070, 12MP 4:2:0 image with restart intervals, (`chunk_size = 32`):

```shell
./build/jpeggpu_benchmark in.jpg
                     throughput (image/s) | avg latency (ms) | max latency (ms)
jpeggpu singlethread               176.18               5.68               6.72
 nvJPEG singlethread                71.82              13.92              15.51
 nvJPEG batch 25                    22.31              44.82            1123.72
 nvJPEG batch 50                    22.69              44.07            2214.65
 nvJPEG batch 75                    22.89              43.68            3303.78
 nvJPEG  4 threads                 268.95              14.69              39.53
 nvJPEG  5 threads                 332.87              14.91              39.72
 nvJPEG  6 threads                 386.19              15.19              35.15
 nvJPEG  7 threads                 389.12              15.92              33.67
 nvJPEG  8 threads                 447.42              16.44              33.00
 nvJPEG  9 threads                 493.39              17.16              33.90
 nvJPEG 10 threads                 531.90              17.85              35.43
 nvJPEG 11 threads                 568.07              18.78              49.18
 nvJPEG 12 threads                 620.45              19.21              45.74
 nvJPEG 13 threads                 603.87              20.59              55.99
 nvJPEG 14 threads                 609.44              22.21              66.65
```

## Test

`jpeggpu_test` compares output against nvJPEG. Helper script `test.sh` uses ImageMagick to convert an input image to a few different JPEG variations.

```shell
./build/jpeggpu_example test.jpg --write_out # writing out is optional
component 0 MSE: 0.23201 component 1 MSE: 0.198817 component 2 MSE: 0.199355
writing out to "test.jpg.nvjpeg.png" and "test.jpg.jpeggpu.png"

./build/test.sh test.jpg # can also optionally pass --write_out
creating tmp file test.jpg.1x1.jpg..
component 0 MSE: 0.202032 component 1 MSE: 0.155791 component 2 MSE: 0.155672 
creating tmp file test.jpg.2x1.jpg..
...
```
