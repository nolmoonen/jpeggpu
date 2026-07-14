// Copyright (c) 2024-2026 Nol Moonen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

    printf("         throughput (image/s) | avg latency (ms) | max latency (ms)\n");
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path file_path(argv[i]);
        std::cout << file_path.filename().string() << "\n";

        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "cannot open \"" << file_path << "\"\n";
            return EXIT_FAILURE;
        }

        file.seekg(0, std::ios_base::end);
        const std::streampos file_size = file.tellg();
        file.seekg(0);
        uint8_t* file_data = nullptr;
        CHECK_CUDA(cudaMallocHost(&file_data, file_size));
        file.read(reinterpret_cast<char*>(file_data), file_size);
        file.close();

        bench_jpeggpu(file_data, file_size);
        bench_nvjpeg(file_data, file_size);

        CHECK_CUDA(cudaFreeHost(file_data));
    }
}
