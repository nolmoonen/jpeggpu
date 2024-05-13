#include "benchmark_jpeggpu.hpp"
#include "benchmark_nvjpeg.hpp"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: jpeggpu_benchmark <jpeg file>\n";
        return EXIT_FAILURE;
    }

    std::ifstream file(argv[1]);
    file.seekg(0, std::ios_base::end);
    const std::streampos file_size = file.tellg();
    file.seekg(0);
    char* file_data = nullptr;
    CHECK_CUDA(cudaMallocHost(&file_data, file_size));
    file.read(file_data, file_size);
    file.close();

    printf("                     throughput (image/s) | avg latency (ms) | max latency (ms)\n");

    bench_jpeggpu(file_data, file_size);

    bench_nvjpeg(file_data, file_size);

    CHECK_CUDA(cudaFreeHost(file_data));
}

// TODO
//   - command line option to disable output copy?
