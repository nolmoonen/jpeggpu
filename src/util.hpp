#ifndef JPEGGPU_UTIL_HPP_
#define JPEGGPU_UTIL_HPP_

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
    void reset()
    {
        alloc.data = nullptr;
        alloc.size = 0;
        stack.clear();
        size = 0;
    }

    template <bool do_it, typename T>
    jpeggpu_status reserve(T** d_ptr_alloc, size_t size_alloc)
    {
        const size_t alloc_bytes = gpu_alloc_size(size_alloc);
        if (do_it && alloc_bytes > alloc.size) {
            // TODO log
            return JPEGGPU_INTERNAL_ERROR;
        }

        if (do_it) {
            *d_ptr_alloc = reinterpret_cast<T*>(alloc.data);
        } else {
            *d_ptr_alloc = nullptr;
        }

        allocation this_alloc;
        this_alloc.data = reinterpret_cast<char*>(*d_ptr_alloc);
        this_alloc.size = alloc_bytes;
        if (do_it) {
            alloc.data += alloc_bytes;
        }
        stack.push_back(this_alloc);
        size += alloc_bytes;

        return JPEGGPU_SUCCESS;
    }

    allocation alloc; /// Device allocation

    std::vector<allocation> stack;
    size_t size; /// Current size of stack.
};

} // namespace jpeggpu

#endif // JPEGGPU_UTIL_HPP_
