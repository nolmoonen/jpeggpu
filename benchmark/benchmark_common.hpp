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
