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

#ifndef JPEGGPU_BENCHMARK_BENCHMARK_COMMON_HPP_
#define JPEGGPU_BENCHMARK_BENCHMARK_COMMON_HPP_

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error \"" << cudaGetErrorString(err) << "\" at: " __FILE__ ":"      \
                      << __LINE__ << "\n";                                                         \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

constexpr int num_iter = 200;

constexpr double us_in_s  = 1e6;
constexpr double us_in_ms = 1e3;

inline void print_measurement(double throughput, double avg_latency, double max_latency)
{
    printf(
        "               %6.2f              %5.2f            %7.2f\n",
        throughput,
        avg_latency,
        max_latency);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_COMMON_HPP_
