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

#endif // JPEGGPU_UTIL_HPP_