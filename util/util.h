#ifndef JPEGGPU_UTIL_H_
#define JPEGGPU_UTIL_H_

#include <jpeggpu/jpeggpu.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/// \brief Return EXIT_SUCCESS on success, EXIT_FAILURE otherwise.
///   Failing happens only if the function is not set up to support the subsampling or number of components.
///   Most commonly occurring JPEG files are supported.
int conv_to_rgbi(
    int* sizes_x,
    int* sizes_y,
    int num_components,
    struct jpeggpu_subsampling subsampling,
    const uint8_t* image_0,
    const uint8_t* image_1,
    const uint8_t* image_2,
    uint8_t* image_interleaved)
{
    const int is_grayscale_or_ycbcr = num_components == 1 || num_components == 3;
    if (!is_grayscale_or_ycbcr) {
        return EXIT_FAILURE;
    }

    if (num_components == 1) {
        for (int y = 0; y < sizes_y[0]; ++y) {
            for (int x = 0; x < sizes_x[0]; ++x) {
                const int idx                  = y * sizes_x[0] + x;
                const uint8_t cy               = image_0[idx];
                image_interleaved[3 * idx + 0] = cy;
                image_interleaved[3 * idx + 1] = cy;
                image_interleaved[3 * idx + 2] = cy;
            }
        }

        return EXIT_SUCCESS;
    }

    const int is_luminance_subsampled = subsampling.x[0] == 1 && subsampling.y[0] == 1 &&
                                        (subsampling.x[1] != 1 || subsampling.y[1] != 1) &&
                                        (subsampling.x[2] != 1 || subsampling.y[2] != 1);

    int are_chrominance_subsampled_differently =
        subsampling.x[1] != subsampling.x[2] || subsampling.y[1] != subsampling.y[2];

    if (is_luminance_subsampled || are_chrominance_subsampled_differently) {
        return EXIT_FAILURE;
    }

    const int ssx     = subsampling.x[0];
    const int ssy     = subsampling.y[0];
    const int ssize_x = (sizes_x[0] + ssx - 1) / ssx;
    const int ssize_y = (sizes_y[0] + ssy - 1) / ssy;
    for (int y = 0; y < ssize_y; ++y) {
        for (int x = 0; x < ssize_x; ++x) {
            const int idx_chroma = y * sizes_x[1] + x;
            const uint8_t cb     = num_components >= 2 ? image_1[idx_chroma] : 0;
            const uint8_t cr     = num_components >= 3 ? image_2[idx_chroma] : 0;
            for (int yy = 0; yy < ssy; ++yy) {
                const int yyy = y * ssy + yy;
                if (yyy >= sizes_y[0]) continue;
                for (int xx = 0; xx < ssx; ++xx) {
                    const int xxx = x * ssx + xx;
                    if (xxx >= sizes_x[0]) continue;
                    const int idx_luma = yyy * sizes_x[0] + xxx;
                    const uint8_t cy   = image_0[idx_luma];

                    // https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
                    // clang-format off
                    const float r = cy                            + 1.402f    * (cr - 128.f);
                    const float g = cy -  .344136f * (cb - 128.f) -  .714136f * (cr - 128.f);
                    const float b = cy + 1.772f    * (cb - 128.f);
                    // clang-format on

                    image_interleaved[3 * idx_luma + 0] = fmaxf(0.f, fminf(roundf(r), 255.f));
                    image_interleaved[3 * idx_luma + 1] = fmaxf(0.f, fminf(roundf(g), 255.f));
                    image_interleaved[3 * idx_luma + 2] = fmaxf(0.f, fminf(roundf(b), 255.f));
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

#endif // JPEGGPU_UTIL_H_
