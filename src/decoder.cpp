#include "decoder.hpp"
#include "decode_gpu.hpp"
#include "decode_cpu_legacy.hpp"
#include "defs.hpp"
#include "idct.hpp"
#include "marker.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <vector>

jpeggpu_status jpeggpu::decoder::parse_header(
    jpeggpu_img_info* img_info, const uint8_t* data, size_t size)
{
    std::memset(&reader, 0, sizeof(reader));
    reader.image     = data;
    reader.image_end = data + size;

    jpeggpu_status stat = reader.read();
    if (stat != JPEGGPU_SUCCESS) {
        return stat;
    }

    // TODO check reader consistency

    for (int c = 0; c < reader.num_components; ++c) {
        img_info->size_x[c] = reader.sizes_x[c];
        img_info->size_y[c] = reader.sizes_y[c];
        img_info->ss_x[c]   = reader.ss_x_max / reader.ss_x[c];
        img_info->ss_y[c]   = reader.ss_y_max / reader.ss_y[c];
    }
    img_info->num_components = reader.num_components;

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::decoder::decode(jpeggpu_img* img)
{
    (void)img; // FIXME use

    // FIXME: deallocation and use user buffer
    for (int i = 0; i < reader.num_components; ++i) {
        reader.data[i]      = new int16_t[reader.data_sizes_x[i] * reader.data_sizes_y[i]];
        // FIXME: for now, just output rounded up size
        reader.image_out[i] = new uint8_t[reader.data_sizes_x[i] * reader.data_sizes_y[i]];
    }

    process_scan_legacy(reader);

    // TODO check that the number of scans seen is equal to the number of
    //   components
    jpeggpu::idct(&reader);

    // FIXME find better place
    std::vector<uint8_t> rgb(reader.size_x * reader.size_y * 3);
    const int size_xx = get_size(reader.size_x, 1, reader.ss_x_max);
    const int size_yy = get_size(reader.size_y, 1, reader.ss_y_max);
    for (int yy = 0; yy < size_yy; ++yy) {
        for (int yyy = 0; yyy < reader.ss_y_max; ++yyy) {
            const int y = yy * reader.ss_y_max + yyy;
            if (y >= reader.size_y) {
                break;
            }
            for (int xx = 0; xx < size_xx; ++xx) {
                for (int xxx = 0; xxx < reader.ss_x_max; ++xxx) {
                    const int x = xx * reader.ss_x_max + xxx;
                    if (x >= reader.size_x) {
                        break;
                    }

                    // https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
                    // FIXME data_size!
                    const uint8_t cy = reader.image_out[0][y * reader.data_sizes_x[0] + x];
                    const uint8_t cb = reader.image_out[1][yy * reader.data_sizes_x[1] + xx];
                    const uint8_t cr = reader.image_out[2][yy * reader.data_sizes_x[2] + xx];

                    // clang-format off
                    const float r = cy                            + 1.402f    * (cr - 128.f);
                    const float g = cy -  .344136f * (cb - 128.f) -  .714136f * (cr - 128.f);
                    const float b = cy + 1.772f    * (cb - 128.f);
                    // clang-format on

                    const auto clamp = [](float c) {
                        return std::max(0.f, std::min(std::round(c), 255.f));
                    };

                    const size_t idx = y * reader.size_x + x;
                    rgb[3 * idx + 0] = clamp(r);
                    rgb[3 * idx + 1] = clamp(g);
                    rgb[3 * idx + 2] = clamp(b);
                }
            }
        }
    }
    std::ofstream file;
    file.open("out.ppm");
    file << "P3\n" << reader.size_x << " " << reader.size_y << "\n255\n";
    for (size_t i = 0; i < rgb.size() / 3; ++i) {
        file << static_cast<int>(rgb[3 * i + 0]) << " " << static_cast<int>(rgb[3 * i + 1]) << " "
             << static_cast<int>(rgb[3 * i + 2]) << "\n";
    }
    file.close();
    // FIXME end

    return JPEGGPU_SUCCESS;
}
