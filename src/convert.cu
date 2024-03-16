#include "convert.hpp"
#include "defs.hpp"
#include "util.hpp"

__device__ __forceinline__ float clamp_pix(float c) { return fmaxf(0.f, fminf(roundf(c), 255.f)); }

__device__ void convert(uint8_t (&data)[4])
{
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

// TODO place many parameters as template parameters
__global__ void kernel_convert(
    int size_x,
    int size_y,
    jpeggpu::image_desc in_image,
    jpeggpu_color_format in_color_fmt,
    jpeggpu_pixel_format in_pixel_fmt,
    subsampling in_css, // e.g. 4:2:0 will get (1, 1), (2, 2), (2, 2)
    jpeggpu::image_desc out_image,
    jpeggpu_color_format out_color_fmt,
    jpeggpu_pixel_format out_pixel_fmt,
    subsampling out_css)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x_off = x * 4;
    const int y_off = y * 4;

    if (x_off >= size_x || y_off >= size_y) {
        return;
    }

    const int in_num_components = 3;

    // in_image will always be planar
    uint8_t data[4 * 4][4]; // 16 pix in block, 4 components
    for (int c = 0; c < in_num_components; ++c) {
        // hope this gets unrolled if `in_num_components` is constexpr
        const uint8_t* channel;
        int pitch;
        int ssx;
        int ssy;
        switch (c) {
        case 0:
            channel = in_image.channel_0;
            pitch   = in_image.pitch_0;
            ssx     = in_css.x_0;
            ssy     = in_css.y_0;
            break;
        case 1:
            channel = in_image.channel_1;
            pitch   = in_image.pitch_1;
            ssx     = in_css.x_1;
            ssy     = in_css.y_1;
            break;
        case 2:
            channel = in_image.channel_2;
            pitch   = in_image.pitch_2;
            ssx     = in_css.x_2;
            ssy     = in_css.y_2;
            break;
        case 3:
            channel = in_image.channel_3;
            pitch   = in_image.pitch_3;
            ssx     = in_css.x_3;
            ssy     = in_css.y_3;
            break;
        default:
            __builtin_unreachable();
        }

        for (int yy = 0; yy < ssy; ++yy) {
            for (int xx = 0; xx < ssx; ++xx) {
                const uint8_t val = channel[(y * ssy + yy) * pitch + x * ssx + xx];
                for (int yyy = 0; yyy < 4 / ssy; ++yyy) {
                    for (int xxx = 0; xxx < 4 / ssx; ++xxx) {
                        // no bounds check needed, input data is always a multiple of four
                        data[(yy * (4 / ssy) + yyy) * 4 + (xx * (4 / ssx) + xxx)][c] = val;
                    }
                }
            }
        }
    }

    for (int yy = 0; yy < 4; ++yy) {
        for (int xx = 0; xx < 4; ++xx) {
            convert(data[yy * 4 + xx]);
        }
    }

    const bool interleaved = true;
    if (interleaved) {
        const int out_num_components = 3;
        for (int c = 0; c < out_num_components; ++c) {
            // if out format is interleaved, no subsampling is allowed
            for (int yy = 0; yy < 4; ++yy) {
                for (int xx = 0; xx < 4; ++xx) {
                    const int out_x = x_off + xx;
                    const int out_y = y_off + yy;
                    if (out_x >= size_x || out_y >= size_y) {
                        continue;
                    }

                    const size_t idx = out_y * out_image.pitch_0 + out_x * out_num_components;
                    out_image.channel_0[idx + c] = data[yy * 4 + xx][c];
                }
            }
        }
    } else {
        const int out_num_components = 3;
        for (int c = 0; c < out_num_components; ++c) {
            // hope this gets unrolled if `out_num_components` is constexpr
            uint8_t* channel;
            int pitch;
            int ssx;
            int ssy;
            switch (c) {
            case 0:
                channel = out_image.channel_0;
                pitch   = out_image.pitch_0;
                ssx     = out_css.x_0;
                ssy     = out_css.y_0;
                break;
            case 1:
                channel = out_image.channel_1;
                pitch   = out_image.pitch_1;
                ssx     = out_css.x_1;
                ssy     = out_css.y_1;
                break;
            case 2:
                channel = out_image.channel_2;
                pitch   = out_image.pitch_2;
                ssx     = out_css.x_2;
                ssy     = out_css.y_2;
                break;
            case 3:
                channel = out_image.channel_3;
                pitch   = out_image.pitch_3;
                ssx     = out_css.x_3;
                ssy     = out_css.y_3;
                break;
            default:
                __builtin_unreachable();
            }

            // FIXME: probably, css-related is broken
            for (int yy = 0; yy < ssy; ++yy) {
                for (int xx = 0; xx < ssx; ++xx) {
                    const int out_x = x_off + xx;
                    const int out_y = y_off + yy;
                    if (out_x >= size_x || out_y >= size_y) {
                        continue;
                    }

                    int sum = 0;
                    for (int yyy = 0; yyy < 4 / ssy; ++yyy) {
                        for (int xxx = 0; xxx < 4 / ssx; ++xxx) {
                            sum += data[(yy * ssy + yyy) * 4 + (xx * ssx + xxx)][c];
                        }
                    }
                    const uint8_t val              = sum * (1.f / (ssx * ssy));
                    channel[out_y * pitch + out_x] = val;
                }
            }
        }
    }
}

jpeggpu_status jpeggpu::convert(
    int size_x,
    int size_y,
    jpeggpu::image_desc in_image,
    jpeggpu_color_format in_color_fmt,
    jpeggpu_pixel_format in_pixel_fmt,
    jpeggpu_subsampling in_subsampling,
    jpeggpu::image_desc out_image,
    jpeggpu_color_format out_color_fmt,
    jpeggpu_pixel_format out_pixel_fmt,
    jpeggpu_subsampling out_subsampling,
    cudaStream_t stream)
{
    constexpr int block_size_x = 32;
    constexpr int block_size_y = 16;
    const dim3 block_size(block_size_x, block_size_y);
    const dim3 grid_size(
        ceiling_div(ceiling_div(size_x, 4u), static_cast<unsigned int>(block_size_x)),
        ceiling_div(ceiling_div(size_y, 4u), static_cast<unsigned int>(block_size_y)));

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
            return ceiling_div(4 * x, static_cast<unsigned int>(css_max.x));
        };
        const auto conv_y = [=](int y) -> int {
            return ceiling_div(4 * y, static_cast<unsigned int>(css_max.y));
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

    const subsampling in_css  = conv_css(in_subsampling, in_css_max);
    const subsampling out_css = conv_css(out_subsampling, out_css_max);
    kernel_convert<<<grid_size, block_size, 0, stream>>>(
        size_x,
        size_y,
        in_image,
        in_color_fmt,
        in_pixel_fmt,
        in_css,
        out_image,
        out_color_fmt,
        out_pixel_fmt,
        out_css);
    JPEGGPU_CHECK_CUDA(cudaGetLastError());

    return JPEGGPU_SUCCESS;
}
