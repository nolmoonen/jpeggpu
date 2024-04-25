#include "convert.hpp"
#include "defs.hpp"
#include "util.hpp"

#include <cassert>

using namespace jpeggpu;

namespace {

__device__ __forceinline__ float clamp_pix(float c) { return fmaxf(0.f, fminf(roundf(c), 255.f)); }

__device__ void convert_ycbcr_rgb(uint8_t (&data)[4])
{
    // https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
    const uint8_t cy = data[0];
    const uint8_t cb = data[1];
    const uint8_t cr = data[2];

    // clang-format off
    const float r = cy                            + 1.402f    * (cr - 128.f);
    const float g = cy -  .344136f * (cb - 128.f) -  .714136f * (cr - 128.f);
    const float b = cy + 1.772f    * (cb - 128.f);
    // clang-format on

    data[0] = clamp_pix(r);
    data[1] = clamp_pix(g);
    data[2] = clamp_pix(b);
}

__device__ void convert_y_rgb(uint8_t (&data)[4])
{
    data[1] = data[0];
    data[2] = data[0];
}

__device__ void convert_cmyk_rgb(uint8_t (&data)[4])
{
    // based on OpenCV
    const uint8_t c = data[0];
    const uint8_t m = data[1];
    const uint8_t y = data[2];
    const uint8_t k = data[3];

    data[0] = k - ((255 - c) * k >> 8);
    data[1] = k - ((255 - m) * k >> 8);
    data[2] = k - ((255 - y) * k >> 8);
}

__device__ void convert(
    uint8_t (&data)[4],
    jpeggpu_color_format_jpeg in_color_fmt,
    jpeggpu_color_format_out out_color_fmt)
{
    if (out_color_fmt == JPEGGPU_OUT_NO_CONVERSION) {
        return;
    }

    if (in_color_fmt == JPEGGPU_JPEG_YCBCR && out_color_fmt == JPEGGPU_OUT_SRGB) {
        return convert_ycbcr_rgb(data);
    }

    if (in_color_fmt == JPEGGPU_JPEG_GRAY && out_color_fmt == JPEGGPU_OUT_SRGB) {
        return convert_y_rgb(data);
    }

    if (in_color_fmt == JPEGGPU_JPEG_CMYK && out_color_fmt == JPEGGPU_OUT_SRGB) {
        return convert_cmyk_rgb(data);
    }

    assert("color conversion not yet implemented" && false);
}

// needed for kernel argument copy
struct subsampling {
    int x_0;
    int y_0;
    int x_1;
    int y_1;
    int x_2;
    int y_2;
    int x_3;
    int y_3;
};

template <int cubes_per_block_x, int cubes_per_block_y>
__global__ void kernel_convert(
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
    jpeggpu::image_desc in_image,
    jpeggpu_color_format_jpeg in_color_fmt, // TODO make template parameter
    // e.g. 4:2:0 will get (12, 12), (6, 6), (6, 6), 4:4:4 will get (12, 12), (12, 12), (12, 12)
    subsampling in_css_inv,
    int in_num_components, // TODO make template parameter
    jpeggpu::image_desc out_image,
    jpeggpu_color_format_out out_color_fmt, // TODO make template parameter
    subsampling out_css_inv,
    int out_num_components, // TODO make template parameter
    bool is_interleaved)
{
    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;

    constexpr int num_pixels_per_block_x = cubes_per_block_x * 12;
    constexpr int num_pixels_per_block_y = cubes_per_block_y * 12;
    constexpr int num_pixels_per_block   = num_pixels_per_block_x * num_pixels_per_block_y;
    __shared__ uint8_t data[num_pixels_per_block][max_comp_count];

    const int i = threadIdx.y * num_pixels_per_block_x + threadIdx.x;

    // TODO optimize such that each thread loads at most one word at a time
    // load data, every thread loads one pixel
    // in_image will always be planar
    for (int c = 0; c < in_num_components; ++c) {
        const uint8_t* channel;
        int pitch;
        int ssx;
        int ssy;
        switch (c) {
        case 0:
            channel = in_image.channel_0;
            pitch   = in_image.pitch_0;
            ssx     = in_css_inv.x_0;
            ssy     = in_css_inv.y_0;
            break;
        case 1:
            channel = in_image.channel_1;
            pitch   = in_image.pitch_1;
            ssx     = in_css_inv.x_1;
            ssy     = in_css_inv.y_1;
            break;
        case 2:
            channel = in_image.channel_2;
            pitch   = in_image.pitch_2;
            ssx     = in_css_inv.x_2;
            ssy     = in_css_inv.y_2;
            break;
        case 3:
            channel = in_image.channel_3;
            pitch   = in_image.pitch_3;
            ssx     = in_css_inv.x_3;
            ssy     = in_css_inv.y_3;
            break;
        default:
            __builtin_unreachable();
        }

        // TODO passing subsampling as template param optimizes these divisions
        const int x_ss = x * ssx / 12;
        const int y_ss = y * ssy / 12;

        if (x_ss < ceiling_div(size_x * 12, static_cast<unsigned int>(ssx)) &&
            y_ss < ceiling_div(size_y * 12, static_cast<unsigned int>(ssy))) {
            // if image is subsampled, some redundant reads occur
            data[i][c] = channel[y_ss * pitch + x_ss];
        }
    }

    convert(data[i], in_color_fmt, out_color_fmt);
    __syncthreads();

    if (is_interleaved) {
        // if out format is interleaved, no subsampling is allowed
        if (x >= size_x || y >= size_y) {
            return;
        }

        for (int c = 0; c < out_num_components; ++c) {
            const size_t idx             = y * out_image.pitch_0 + x * out_num_components;
            out_image.channel_0[idx + c] = data[i][c];
        }
    } else {
        // offset of the cube in shared memory
        const int shared_off_x = threadIdx.x / 12 * 12;
        const int shared_off_y = threadIdx.y / 12 * 12;

        // offset in the cube in shared memory
        const int off_shared_x = threadIdx.x - shared_off_x;
        const int off_shared_y = threadIdx.y - shared_off_y;

        for (int c = 0; c < out_num_components; ++c) {
            uint8_t* channel;
            int pitch;
            int ssx_inv;
            int ssy_inv;
            int size_x_c;
            int size_y_c;
            switch (c) {
            case 0:
                channel  = out_image.channel_0;
                pitch    = out_image.pitch_0;
                ssx_inv  = out_css_inv.x_0;
                ssy_inv  = out_css_inv.y_0;
                size_x_c = size_x_0;
                size_y_c = size_y_0;
                break;
            case 1:
                channel  = out_image.channel_1;
                pitch    = out_image.pitch_1;
                ssx_inv  = out_css_inv.x_1;
                ssy_inv  = out_css_inv.y_1;
                size_x_c = size_x_1;
                size_y_c = size_y_1;
                break;
            case 2:
                channel  = out_image.channel_2;
                pitch    = out_image.pitch_2;
                ssx_inv  = out_css_inv.x_2;
                ssy_inv  = out_css_inv.y_2;
                size_x_c = size_x_2;
                size_y_c = size_y_2;
                break;
            case 3:
                channel  = out_image.channel_3;
                pitch    = out_image.pitch_3;
                ssx_inv  = out_css_inv.x_3;
                ssy_inv  = out_css_inv.y_3;
                size_x_c = size_x_3;
                size_y_c = size_y_3;
                break;
            default:
                __builtin_unreachable();
            }

            // output one pixel, possibly an aggregate due to subsampling
            if (x >= size_x_c || y >= size_y_c) {
                continue;
            }

            const int xx = 12 / ssx_inv;
            const int yy = 12 / ssy_inv;

            // aggregation in int to prevent data loss (rounding)
            //   when subsampling factor is the same
            int sum = 0;
            for (int yyy = 0; yyy < xx; ++yyy) {
                for (int xxx = 0; xxx < yy; ++xxx) {
                    const int shared_x = shared_off_x + off_shared_x / xx * xx + xxx;
                    const int shared_y = shared_off_y + off_shared_y / yy * yy + yyy;
                    sum += data[shared_y * num_pixels_per_block_x + shared_x][c];
                }
            }
            // integer division to prevent rounding
            const uint8_t val      = sum / (ssx_inv * ssy_inv);
            channel[y * pitch + x] = val;
        }
    }
}

} // namespace

jpeggpu_status jpeggpu::convert(
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
    jpeggpu::image_desc in_image,
    jpeggpu_color_format_jpeg in_color_fmt,
    jpeggpu_subsampling in_subsampling,
    int in_num_components,
    jpeggpu::image_desc out_image,
    jpeggpu_color_format_out out_color_fmt,
    jpeggpu_subsampling out_subsampling,
    int out_num_components,
    // TODO change naming, the same term is used to denote data unit interleaving
    bool is_interleaved,
    cudaStream_t stream,
    logger& logger)
{
    // lcm of 1, 2, 3, and 4 is 12. pixels are processed in "cubes" of 12x12 to make subsampling
    //   conversion easier: no inter-block communication is needed at most size_x * size_y * num_components
    //   numer of reads and writes are done

    constexpr int cubes_per_block_x = 2; // configurable for performance
    constexpr int cubes_per_block_y = 2; // configurable for performance

    constexpr int pixels_per_block_x = 12 * cubes_per_block_x;
    constexpr int pixels_per_block_y = 12 * cubes_per_block_y;

    const dim3 block_size(pixels_per_block_x, pixels_per_block_y);
    const dim3 grid_size(
        ceiling_div(size_x, static_cast<unsigned int>(pixels_per_block_x)),
        ceiling_div(size_y, static_cast<unsigned int>(pixels_per_block_y)));

    const auto calc_max = [](jpeggpu_subsampling css) -> int2 {
        int2 ret = make_int2(0, 0);
        for (int c = 0; c < jpeggpu::max_comp_count; ++c) {
            ret.x = std::max(ret.x, css.x[c]);
            ret.y = std::max(ret.y, css.y[c]);
        }
        return ret;
    };

    const int2 in_css_max  = calc_max(in_subsampling);
    const int2 out_css_max = calc_max(out_subsampling);

    const auto conv_css = [](jpeggpu_subsampling css, int2 css_max) -> subsampling {
        const auto conv_x = [=](int x) -> int {
            return ceiling_div(12 * x, static_cast<unsigned int>(css_max.x));
        };
        const auto conv_y = [=](int y) -> int {
            return ceiling_div(12 * y, static_cast<unsigned int>(css_max.y));
        };
        return {
            conv_x(css.x[0]),
            conv_y(css.y[0]),
            conv_x(css.x[1]),
            conv_y(css.y[1]),
            conv_x(css.x[2]),
            conv_y(css.y[2]),
            conv_x(css.x[3]),
            conv_y(css.y[3])};
    };

    const subsampling in_css_inv  = conv_css(in_subsampling, in_css_max);
    const subsampling out_css_inv = conv_css(out_subsampling, out_css_max);
    kernel_convert<cubes_per_block_x, cubes_per_block_y><<<grid_size, block_size, 0, stream>>>(
        size_x,
        size_y,
        size_x_0,
        size_x_1,
        size_x_2,
        size_x_3,
        size_y_0,
        size_y_1,
        size_y_2,
        size_y_3,
        in_image,
        in_color_fmt,
        in_css_inv,
        in_num_components,
        out_image,
        out_color_fmt,
        out_css_inv,
        out_num_components,
        is_interleaved);
    JPEGGPU_CHECK_CUDA(cudaGetLastError());

    return JPEGGPU_SUCCESS;
}
