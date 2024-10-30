#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <stdint.h>

inline __device__ uint32_t swap_endian(uint32_t x)
{
    return __byte_perm(x, uint32_t{0}, uint32_t{0x0123});
}

// __device__ void load(uint4* buf, uint8_t* data)
// {
//     assert(reinterpret_cast<std::uintptr_t>(data) % sizeof(uint4) == 0);

//     *buf   = *reinterpret_cast<uint4*>(data);
//     buf->x = swap_endian(buf->x);
//     buf->y = swap_endian(buf->y);
//     buf->z = swap_endian(buf->z);
//     buf->w = swap_endian(buf->w);
// }

template <int block_size>
__global__ void kernel(uint8_t* data)
{
    __shared__ ulonglong4 buffer[block_size];

    // const int num_bytes_per_thread = sizeof(uint4);

    // const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // load(&(reinterpret_cast<uint4*>(&(buffer[threadIdx.x]))[0]), data + tid * num_bytes_per_thread);
    uint4* buf = &(reinterpret_cast<uint4*>(&(buffer[threadIdx.x]))[0]);
    *buf       = *reinterpret_cast<uint4*>(data);
    buf->x     = swap_endian(buf->x);
    buf->y     = swap_endian(buf->y);
    buf->z     = swap_endian(buf->z);
    buf->w     = swap_endian(buf->w);

    // uint4* buf = &(buffer[2 * threadIdx.x]);
    // *buf       = *reinterpret_cast<uint4*>(data + tid * num_bytes_per_thread);
    // buf->x     = swap_endian(buf->x);
    // buf->y     = swap_endian(buf->y);
    // buf->z     = swap_endian(buf->z);
    // buf->w     = swap_endian(buf->w);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < sizeof(uint4) / sizeof(uint32_t); ++i) {
            printf("0x%08x\n", reinterpret_cast<uint32_t*>(buffer)[i]);
        }
    }

    // __syncthreads();

    // for (int i = 0; i < num_bytes_per_thread; ++i) {
    //     data[tid * num_bytes_per_thread + i] = reinterpret_cast<uint8_t*>(buf)[i];
    // }
}

int main()
{
    constexpr int block_size   = 256;
    constexpr int num_blocks   = 128;
    constexpr size_t num_bytes = block_size * num_blocks * sizeof(uint4);

    uint8_t* h_data = reinterpret_cast<uint8_t*>(std::malloc(num_bytes));
    for (size_t i = 0; i < num_bytes; ++i) {
        h_data[i] = i;
    }

    uint8_t* d_data = nullptr;
    cudaMalloc(&d_data, num_bytes);
    cudaMemcpy(d_data, h_data, num_bytes, cudaMemcpyHostToDevice);

    kernel<block_size><<<num_blocks, block_size, 0, nullptr>>>(d_data);

    cudaMemcpy(h_data, d_data, num_bytes, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < num_bytes / 4; ++i) {
    //     for (int j = 0; j < 4; ++j) {
    //         size_t idx             = 4 * i + 4 - j - 1;
    //         const uint8_t expected = static_cast<uint8_t>(idx);
    //         if (h_data[4 * i + j] != expected) {
    //             std::printf("error: %zu %" PRIu8 " %" PRIu8 " \n", i, h_data[i], expected);
    //         }
    //     }
    // }
    std::printf("done\n");

    cudaFree(d_data);

    std::free(h_data);
}
