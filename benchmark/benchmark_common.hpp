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

constexpr int num_iter = 100;

#endif // JPEGGPU_BENCHMARK_BENCHMARK_COMMON_HPP_
