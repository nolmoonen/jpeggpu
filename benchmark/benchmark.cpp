// Copyright (c) 2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "benchmark_jpeggpu.hpp"
#include "benchmark_nvjpeg.hpp"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: jpeggpu_benchmark <jpeg file_0> <jpeg_file_1>\n";
        return EXIT_FAILURE;
    }

    printf("                     throughput (image/s) | avg latency (ms) | max latency (ms)\n");
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path file_path(argv[i]);
        std::cout << file_path << "\n";

        std::ifstream file(file_path);
        file.seekg(0, std::ios_base::end);
        const std::streampos file_size = file.tellg();
        file.seekg(0);
        uint8_t* file_data = nullptr;
        CHECK_CUDA(cudaMallocHost(&file_data, file_size));
        file.read(reinterpret_cast<char*>(file_data), file_size);
        file.close();

        bench_jpeggpu(file_data, file_size);

        if (true) {
            bench_nvjpeg(file_data, file_size);
        }

        CHECK_CUDA(cudaFreeHost(file_data));
    }
}

// TODO
//   - command line option to disable output copy?
