#ifndef JPEGGPU_CONVERT_HPP_
#define JPEGGPU_CONVERT_HPP_

#include <jpeggpu/jpeggpu.h>

namespace jpeggpu {

struct image_desc {
    uint8_t* channel_0;
    int pitch_0;
    uint8_t* channel_1;
    int pitch_1;
    uint8_t* channel_2;
    int pitch_2;
    uint8_t* channel_3;
    int pitch_3;
};

jpeggpu_status convert(
    int size_x,
    int size_y,
    image_desc in_image,
    jpeggpu_color_format in_color_fmt,
    jpeggpu_pixel_format in_pixel_fmt, // will always be planar
    jpeggpu_subsampling in_subsampling,
    image_desc out_image,
    jpeggpu_color_format out_color_fmt,
    jpeggpu_pixel_format out_pixel_fmt,
    jpeggpu_subsampling out_subsampling,
    cudaStream_t stream);

} // namespace jpeggpu
#endif // JPEGGPU_CONVERT_HPP_
