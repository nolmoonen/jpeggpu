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

#ifndef JPEGGPU_DECODER_DEFS_HPP_
#define JPEGGPU_DECODER_DEFS_HPP_

namespace jpeggpu {
// TODO dynamically use different values based on input

/// \brief "s", subsequence size in number of 32 bits words.
///   Configurable, paper uses 4 or 32 depending on the quality of the encoded image
constexpr int chunk_size             = 32;
constexpr int subsequence_size_bytes = chunk_size * 4;
constexpr int subsequence_size       = chunk_size * 32; ///< size in bits
static_assert(subsequence_size % 32 == 0); // is always multiple of 32 bits
} // namespace jpeggpu

#endif // JPEGGPU_DECODER_DEFS_HPP_
