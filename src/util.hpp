#ifndef JPEGGPU_UTIL_HPP_
#define JPEGGPU_UTIL_HPP_

#include <type_traits>

template <
    typename T,
    typename U,
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
inline constexpr auto ceiling_div(const T a, const U b)
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

/// \brief
///
/// Mimic cudaMalloc:
/// \param[out] d_ptr_alloc Pointer to new allocation.
/// \param[in] size_alloc Amount of bytes that need to be allocated.
/// Extra info:
/// \param[inout] d_ptr Pointer to device memory.
/// \param[inout] size_ptr Size of `d_ptr`.
inline jpeggpu_status gpu_alloc_reserve(
    void** d_ptr_alloc, size_t size_alloc, void*& d_ptr, size_t& size_ptr)
{
    // padding before the allocation
    const size_t padding     = gpu_alignment - (reinterpret_cast<uintptr_t>(d_ptr) % gpu_alignment);
    const size_t alloc_bytes = gpu_alloc_size(size_alloc);
    if (size_ptr < padding + alloc_bytes) {
        return JPEGGPU_INVALID_ARGUMENT;
    }
    *d_ptr_alloc = static_cast<char*>(d_ptr) + padding;
    d_ptr        = static_cast<char*>(d_ptr) + padding + alloc_bytes;
    size_ptr -= padding + alloc_bytes;
    return JPEGGPU_SUCCESS;
}

} // namespace jpeggpu

#endif // JPEGGPU_UTIL_HPP_
