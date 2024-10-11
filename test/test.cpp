// Copyright (c) 2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "decoder.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>
// clang-format off
#include <stdio.h>
#include <jpeglib.h>
// clang-format on

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            std::cerr << "CUDA error \"" << cudaGetErrorString(err) << "\" at: " __FILE__ ":" \
                      << __LINE__ << "\n";                                                    \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

#define CHECK_JPEGGPU(call)                                                    \
    do {                                                                       \
        jpeggpu_status stat = call;                                            \
        if (stat != JPEGGPU_SUCCESS) {                                         \
            std::cerr << "jpeggpu error \"" << jpeggpu_get_status_string(stat) \
                      << "\" at: " __FILE__ ":" << __LINE__ << "\n";           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define ASSERT_EQ(val1, val2)                                               \
    do {                                                                    \
        if ((val1) != (val2)) {                                             \
            std::cerr << #val1 "(" << (val1) << ") != " #val2 "(" << (val2) \
                      << ") at: " __FILE__ ":" << __LINE__ << "\n";         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: jpeggpu_test <jpeg file>\n";
        return EXIT_FAILURE;
    }

    const std::filesystem::path filepath(argv[1]);

    std::ifstream ifstream(filepath);
    ifstream.seekg(0, std::ios_base::end);
    const std::streampos file_size = ifstream.tellg();
    ifstream.seekg(0);
    std::vector<uint8_t> data(file_size);
    ifstream.read(reinterpret_cast<char*>(data.data()), file_size);
    ifstream.close();

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, data.data(), data.size());
    const bool require_image = true;
    if (jpeg_read_header(&cinfo, require_image) != JPEG_HEADER_OK) {
        return EXIT_FAILURE;
    }

    cudaStream_t stream = 0;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(decoder, &img_info, data.data(), data.size()));

    ASSERT_EQ(cinfo.num_components, img_info.num_components);
    for (int c = 0; c < img_info.num_components; ++c) {
        const jpeggpu::jpeg_stream& info = decoder->decoder.reader.jpeg_stream;
        ASSERT_EQ(
            cinfo.comp_info[c].width_in_blocks * 8,
            static_cast<unsigned int>(info.components[c].data_size.x));
        ASSERT_EQ(
            cinfo.comp_info[c].height_in_blocks * 8,
            static_cast<unsigned int>(info.components[c].data_size.y));
    }

    size_t tmp_size = 0;
    CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(decoder, &tmp_size));
    void* d_tmp = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));

    CHECK_JPEGGPU(jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream));

    CHECK_JPEGGPU(decoder->decoder.decode_no_idct(d_tmp, tmp_size, stream));
    std::vector<std::vector<int16_t>> jpeggpu_coefs(img_info.num_components);
    for (int c = 0; c < img_info.num_components; ++c) {
        const jpeggpu::jpeg_stream& info = decoder->decoder.reader.jpeg_stream;
        const size_t num_elements = info.components[c].data_size.x * info.components[c].data_size.y;
        jpeggpu_coefs[c].resize(num_elements);
        CHECK_CUDA(cudaMemcpyAsync( // non-pinned causes implicit sync
            jpeggpu_coefs[c].data(),
            decoder->decoder.d_image_qdct[c],
            num_elements * sizeof(int16_t),
            cudaMemcpyDeviceToHost,
            stream));
    }

    jvirt_barray_ptr* coeffs_array = jpeg_read_coefficients(&cinfo);

    for (int c = 0; c < img_info.num_components; ++c) {
        const jpeggpu::jpeg_stream& info = decoder->decoder.reader.jpeg_stream;
        const int num_blocks_x           = info.components[c].data_size.x / 8;
        const int num_blocks_y           = info.components[c].data_size.y / 8;
        for (int y = 0; y < num_blocks_y; ++y) {

            JBLOCKARRAY buffer_one = (cinfo.mem->access_virt_barray)(
                (j_common_ptr)&cinfo, coeffs_array[c], y, (JDIMENSION)1, FALSE);

            for (int x = 0; x < num_blocks_x; ++x) {
                JCOEFPTR blockptr_one = buffer_one[0][x];
                // blockptr_one[bi];
            }
        }
    }

    CHECK_CUDA(cudaFree(d_tmp));

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
}
