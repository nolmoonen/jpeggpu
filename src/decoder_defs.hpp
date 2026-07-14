// Copyright (c) 2024 Nol Moonen
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

#ifndef JPEGGPU_DECODER_DEFS_HPP_
#define JPEGGPU_DECODER_DEFS_HPP_

// TODO https://github.com/chromium/chromium/blob/5e3a68156ec558f36b977bfae79f18e668fca1d5/third_party/blink/web_tests/images/resources/2-dht.jpg
// TODO https://github.com/chromium/chromium/blob/5e3a68156ec558f36b977bfae79f18e668fca1d5/third_party/blink/web_tests/images/resources/motion-jpeg-single-frame.jpg

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
