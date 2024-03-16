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
    uint8_t* (&d_qtable)[max_comp_count],
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_IDCT_HPP_
