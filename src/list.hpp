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

#ifndef JPEGGPU_LIST_HPP_
#define JPEGGPU_LIST_HPP_

#include <jpeggpu/jpeggpu.h>

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace jpeggpu {

template <typename T>
struct default_allocator {
    static T* alloc(int n) { return reinterpret_cast<T*>(std::malloc(n * sizeof(T))); }

    static void free(T* ptr) { std::free(ptr); }
};

template <typename T>
struct pinned_allocator {
    static T* alloc(int n)
    {
        T* ptr          = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, n * sizeof(T));
        if (err != cudaSuccess || ptr == nullptr) {
            if (ptr != nullptr) cudaFreeHost(ptr);
            return nullptr;
        }

        return ptr;
    }

    static void free(T* ptr) { cudaFreeHost(ptr); }
};

/// \brief Non-initializing vector, `T` should be trivially constructible and trivially copyable.
template <typename T, typename Allocator = default_allocator<T>>
struct list {
    /// \brief
    ///   If `JPEGGPU_SUCCESS` is returned, `cleanup` must be called before this instance is removed.
    [[nodiscard]] jpeggpu_status startup(int n = 0)
    {
        assert(ptr == nullptr && size == 0 && capacity == 0);
        return reserve(n);
    }

    void cleanup()
    {
        assert((ptr == nullptr && capacity == 0) || (ptr != nullptr && capacity > 0));
        assert(size <= capacity);
        if (ptr != nullptr) {
            Allocator::free(ptr);
            ptr = nullptr;
        }
        size     = 0;
        capacity = 0;
    }

    [[nodiscard]] jpeggpu_status resize(int n)
    {
        assert((ptr == nullptr && capacity == 0) || (ptr != nullptr && capacity > 0));
        const jpeggpu_status status = reserve(n);
        if (status != JPEGGPU_SUCCESS) {
            return status;
        }
        size = n;
        return JPEGGPU_SUCCESS;
    }

    void clear() { size = 0; }

    [[nodiscard]] jpeggpu_status reserve(int n)
    {
        assert((ptr == nullptr && capacity == 0) || (ptr != nullptr && capacity > 0));
        assert(size <= capacity);
        if (n <= capacity) {
            return JPEGGPU_SUCCESS;
        }

        T* ptr_new = Allocator::alloc(n);
        if (ptr_new == nullptr) {
            return JPEGGPU_OUT_OF_HOST_MEMORY;
        }

        if (size > 0) {
            std::memcpy(ptr_new, ptr, size * sizeof(T));
        }

        if (capacity > 0) {
            Allocator::free(ptr);
        }

        ptr      = ptr_new;
        capacity = n;

        return JPEGGPU_SUCCESS;
    }

    T& operator[](int n)
    {
        assert(0 <= n && n < size);
        return ptr[n];
    }

    const T& operator[](int n) const
    {
        assert(0 <= n && n < size);
        return ptr[n];
    }

    int get_size() const { return size; }

    // Using member initialization to prevent e.g. double `startup` call
    //   without needing an additional "is_initialized" variable. (Using asserts instead.)

    T* ptr       = nullptr;
    int size     = 0; /// Number of elements.
    int capacity = 0; // Number of elements that can be held in `ptr`.
};

template <typename T>
using pinned_list = list<T, pinned_allocator<T>>;

} // namespace jpeggpu

#endif // JPEGGPU_LIST_HPP_
