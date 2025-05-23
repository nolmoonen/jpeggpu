// Copyright (c) 2023-2024 Nol Moonen
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

#ifndef JPEGGPU_UTIL_HPP_
#define JPEGGPU_UTIL_HPP_

#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>

template <
    typename T,
    typename U,
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__device__ __host__ inline constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

namespace jpeggpu {

constexpr size_t gpu_alignment = 256;
/// \brief Mimic CUDA alignment when allocating `bytes` bytes.
inline size_t gpu_alloc_size(size_t bytes)
{
    return ceiling_div(bytes, gpu_alignment) * gpu_alignment;
}

struct allocation {
    char* data;
    size_t size;
};

/// \brief Does not actually do allocations.
///   Stack as to not have to deal with fragmentation.
struct stack_allocator {
    void reset(void* data_alloc = nullptr, size_t size_alloc = 0)
    {
        alloc.data = reinterpret_cast<char*>(data_alloc);
        alloc.size = size_alloc;
        size       = 0;
    }

    template <bool do_it, typename T>
    jpeggpu_status reserve(T** d_ptr_alloc, size_t size_alloc)
    {
        const size_t alloc_bytes = gpu_alloc_size(size_alloc);
        if (do_it && alloc_bytes > alloc.size) {
            return JPEGGPU_INTERNAL_ERROR;
        }

        if (do_it) {
            *d_ptr_alloc = reinterpret_cast<T*>(alloc.data + size);
        } else {
            *d_ptr_alloc = nullptr;
        }

        size += alloc_bytes;

        return JPEGGPU_SUCCESS;
    }

    allocation alloc; /// Device allocation

    size_t size; /// Current size of stack.
};

struct ivec2 {
    int x;
    int y;
};

/// \brief Allocates pinned/page-locked memory using CUDA.
template <class T>
struct pinned_allocator {
    using value_type = T;

    [[nodiscard]] T* allocate(size_t n)
    {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        T* ptr          = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, n * sizeof(T));
        if (err != cudaSuccess || ptr == nullptr) {
            if (ptr != nullptr) cudaFreeHost(ptr);
            throw std::bad_alloc();
        }

        return ptr;
    }

    void deallocate(T* ptr, [[maybe_unused]] size_t n) noexcept { cudaFreeHost(ptr); }
};

template <typename T, typename Allocator>
[[nodiscard]] jpeggpu_status nothrow_resize(
    std::vector<T, Allocator>& c, typename std::vector<T, Allocator>::size_type n)
{
    try {
        c.resize(n);
    } catch (const std::exception& e) {
        return JPEGGPU_OUT_OF_HOST_MEMORY;
    }
    return JPEGGPU_SUCCESS;
}

template <typename T, typename Allocator>
[[nodiscard]] jpeggpu_status nothrow_reserve(
    std::vector<T, Allocator>& c, typename std::vector<T, Allocator>::size_type n)
{
    try {
        c.reserve(n);
    } catch (const std::exception& e) {
        return JPEGGPU_OUT_OF_HOST_MEMORY;
    }
    return JPEGGPU_SUCCESS;
}

template <typename T, typename Allocator>
[[nodiscard]] jpeggpu_status nothrow_push_back(std::vector<T, Allocator>& c, const T& value)
{
    try {
        c.push_back(value);
    } catch (const std::exception& e) {
        return JPEGGPU_OUT_OF_HOST_MEMORY;
    }
    return JPEGGPU_SUCCESS;
}

template <typename T, typename Allocator>
[[nodiscard]] jpeggpu_status nothrow_push_back(std::vector<T, Allocator>& c, T&& value)
{
    try {
        c.push_back(std::forward<T>(value));
    } catch (const std::exception& e) {
        return JPEGGPU_OUT_OF_HOST_MEMORY;
    }
    return JPEGGPU_SUCCESS;
}

} // namespace jpeggpu

#endif // JPEGGPU_UTIL_HPP_
