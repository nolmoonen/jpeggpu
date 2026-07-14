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

#ifndef JPEGGPU_IDCT_HPP_
#define JPEGGPU_IDCT_HPP_

#include "reader.hpp"

#include <stdint.h>

namespace jpeggpu {

/// \brief
///
/// \param[in] d_image_qdct Planar data, each 64 consecutive elements form one data unit.
jpeggpu_status idct(
    const jpeg_stream& info,
    int16_t* (&d_image_qdct)[max_comp_count],
    uint8_t* (&d_image)[max_comp_count],
    int (&pitch)[max_comp_count],
    qtable* (&d_qtable)[max_comp_count],
    cudaStream_t stream,
    logger& logger);

} // namespace jpeggpu

#endif // JPEGGPU_IDCT_HPP_
