// Copyright (c) 2023-2024 Nol Moonen
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

#ifndef JPEGGPU_DECODER_HPP_
#define JPEGGPU_DECODER_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cstdarg>
#include <stddef.h>
#include <vector>
#include <stdint.h>

namespace jpeggpu {

struct decoder {

    jpeggpu_status init();

    void cleanup();

    jpeggpu_status parse_header(jpeggpu_img_info& img_info, const uint8_t* data, size_t size);

    jpeggpu_status transfer(void* d_tmp, size_t tmp_size, cudaStream_t stream);

    template <bool do_it>
    jpeggpu_status decode_impl(jpeggpu_img* img, cudaStream_t stream);

    jpeggpu_status decode_get_size(size_t& tmp_size);

    jpeggpu_status decode(jpeggpu_img* img, void* d_tmp, size_t tmp_size, cudaStream_t stream);

    struct reader reader;

    /// \brief Keeps track of allocations for the current decoded image.
    stack_allocator allocator;

    struct logger logger;
};

} // namespace jpeggpu

struct jpeggpu_decoder {
    jpeggpu::decoder decoder;
};

#endif // JPEGGPU_DECODER_HPP_
