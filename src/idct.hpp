#ifndef JPEGGPU_IDCT_HPP_
#define JPEGGPU_IDCT_HPP_

#include "reader.hpp"

namespace jpeggpu {

void idct(
    reader& reader, int16_t* (&d_image_qdct)[max_comp_count], uint8_t* (&d_image)[max_comp_count]);
}

#endif // JPEGGPU_IDCT_HPP_
