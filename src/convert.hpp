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
    int size_x_0,
    int size_x_1,
    int size_x_2,
    int size_x_3,
    int size_y_0,
    int size_y_1,
    int size_y_2,
    int size_y_3,
    image_desc in_image,
    jpeggpu_color_format_jpeg in_color_fmt,
    jpeggpu_subsampling in_subsampling,
    int in_num_components,
    image_desc out_image,
    jpeggpu_color_format_out out_color_fmt,
    jpeggpu_subsampling out_subsampling,
    int out_num_components,
    bool is_interleaved,
    cudaStream_t stream);

} // namespace jpeggpu

#endif // JPEGGPU_CONVERT_HPP_
