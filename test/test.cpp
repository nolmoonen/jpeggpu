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

#include <util.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <jpeggpu/jpeggpu.h>
#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error \"" << cudaGetErrorString(err) << "\" at: " __FILE__ ":"      \
                      << __LINE__ << "\n";                                                         \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#define CHECK_JPEGGPU(call)                                                                        \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            std::cerr << "jpeggpu error \"" << jpeggpu_get_status_string(stat)                     \
                      << "\" at: " __FILE__ ":" << __LINE__ << "\n";                               \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

#define CHECK_NVJPEG(call)                                                                         \
    do {                                                                                           \
        nvjpegStatus_t stat = call;                                                                \
        if (stat != NVJPEG_STATUS_SUCCESS) {                                                       \
            std::cerr << "nvJPEG error \"" << static_cast<int>(stat) << "\" at: " __FILE__ ":"     \
                      << __LINE__ << "\n";                                                         \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

constexpr int max_num_comp = 4;
static_assert(max_num_comp == NVJPEG_MAX_COMPONENT && max_num_comp == JPEGGPU_MAX_COMP);

jpeggpu_subsampling nv_css_to_jpeggpu_css(nvjpegChromaSubsampling_t subsampling)
{
    switch (subsampling) {
    case NVJPEG_CSS_444:
        return {{1, 1, 1, 1}, {1, 1, 1, 1}};
    case NVJPEG_CSS_422:
        return {{2, 1, 1, 1}, {1, 1, 1, 1}};
    case NVJPEG_CSS_420:
        return {{2, 1, 1, 1}, {2, 1, 1, 1}};
    case NVJPEG_CSS_440:
        return {{1, 1, 1, 1}, {2, 1, 1, 1}};
    case NVJPEG_CSS_411:
        return {{4, 1, 1, 1}, {1, 1, 1, 1}};
    case NVJPEG_CSS_410:
        return {{4, 1, 1, 1}, {2, 1, 1, 1}};
    case NVJPEG_CSS_GRAY:
        return {{1, 1, 1, 1}, {1, 1, 1, 1}};
    case NVJPEG_CSS_410V:
        return {{2, 1, 1, 1}, {4, 1, 1, 1}}; // TODO correct?
    case NVJPEG_CSS_UNKNOWN:
        return {{0, 0, 0, 0}, {0, 0, 0, 0}};
    }

    return {{0, 0, 0, 0}, {0, 0, 0, 0}};
}

void decode_nvjpeg(
    const uint8_t* file_data,
    size_t file_size,
    int (&sizes_x)[max_num_comp],
    int (&sizes_y)[max_num_comp],
    int& num_comp,
    jpeggpu_subsampling& subsampling,
    uint8_t* h_img[max_num_comp])
{
    cudaStream_t stream = 0;

    nvjpegHandle_t nvjpeg_handle;
    CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));

    nvjpegJpegState_t nvjpeg_state;
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    nvjpegChromaSubsampling_t nv_subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &num_comp,
        &nv_subsampling,
        sizes_x,
        sizes_y));
    subsampling = nv_css_to_jpeggpu_css(nv_subsampling);

    nvjpegImage_t d_img;
    for (int c = 0; c < num_comp; ++c) {
        CHECK_CUDA(cudaMalloc(&(d_img.channel[c]), sizes_x[c] * sizes_y[c]));
        d_img.pitch[c] = sizes_x[c];
    }

    CHECK_NVJPEG(nvjpegDecode(
        nvjpeg_handle,
        nvjpeg_state,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        NVJPEG_OUTPUT_UNCHANGED,
        &d_img,
        stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));

    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));

    for (int c = 0; c < num_comp; ++c) {
        const size_t comp_size = sizes_x[c] * sizes_y[c];
        h_img[c]               = static_cast<uint8_t*>(malloc(comp_size));
        CHECK_CUDA(cudaMemcpy(h_img[c], d_img.channel[c], comp_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_img.channel[c]));
    }
}

void decode_jpeggpu(
    const uint8_t* file_data,
    size_t file_size,
    int (&sizes_x)[max_num_comp],
    int (&sizes_y)[max_num_comp],
    int& num_comp,
    jpeggpu_subsampling& subsampling,
    uint8_t* h_img[max_num_comp])
{
    cudaStream_t stream = 0;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    CHECK_JPEGGPU(jpeggpu_set_logging(decoder, true));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

    std::copy(img_info.sizes_x, img_info.sizes_x + max_num_comp, sizes_x);
    std::copy(img_info.sizes_y, img_info.sizes_y + max_num_comp, sizes_y);
    num_comp    = img_info.num_components;
    subsampling = img_info.subsampling;

    size_t tmp_size{};
    CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(decoder, &tmp_size));
    void* d_tmp{};
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));

    CHECK_JPEGGPU(jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream));

    jpeggpu_img d_img;
    for (int c = 0; c < num_comp; ++c) {
        CHECK_CUDA(cudaMalloc(&(d_img.image[c]), sizes_x[c] * sizes_y[c]));
        d_img.pitch[c] = sizes_x[c];
    }

    CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &d_img, d_tmp, tmp_size, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaFree(d_tmp));

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    for (int c = 0; c < num_comp; ++c) {
        const size_t comp_size = sizes_x[c] * sizes_y[c];
        h_img[c]               = static_cast<uint8_t*>(malloc(comp_size));
        CHECK_CUDA(cudaMemcpy(h_img[c], d_img.image[c], comp_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_img.image[c]));
    }
}

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        std::cerr << "usage: jpeggpu_test <jpeg file> (optional: --write_out)\n";
        return EXIT_FAILURE;
    }

    const std::filesystem::path filepath(argv[1]);
    const bool write_out = argc >= 3;

    std::ifstream file(filepath);
    file.seekg(0, std::ios_base::end);
    const std::streampos file_size = file.tellg();
    file.seekg(0);
    uint8_t* file_data = nullptr;
    CHECK_CUDA(cudaMallocHost(&file_data, file_size));
    file.read(reinterpret_cast<char*>(file_data), file_size);
    file.close();

    int sizes_x_nvjpeg[max_num_comp] = {};
    int sizes_y_nvjpeg[max_num_comp] = {};
    int num_comp_nvjpeg{};
    jpeggpu_subsampling subsampling_nvjpeg = {};
    uint8_t* img_nvjpeg[max_num_comp]      = {};
    decode_nvjpeg(
        file_data,
        file_size,
        sizes_x_nvjpeg,
        sizes_y_nvjpeg,
        num_comp_nvjpeg,
        subsampling_nvjpeg,
        img_nvjpeg);

    int sizes_x_jpeggpu[max_num_comp] = {};
    int sizes_y_jpeggpu[max_num_comp] = {};
    int num_comp_jpeggpu{};
    jpeggpu_subsampling subsampling_jpeggpu = {};
    uint8_t* img_jpeggpu[max_num_comp]      = {};
    decode_jpeggpu(
        file_data,
        file_size,
        sizes_x_jpeggpu,
        sizes_y_jpeggpu,
        num_comp_jpeggpu,
        subsampling_jpeggpu,
        img_jpeggpu);

    if (num_comp_nvjpeg != num_comp_jpeggpu) {
        std::cout << "component mismatch: " << num_comp_nvjpeg << " (nvJPEG) to "
                  << num_comp_jpeggpu << " (jpeggpu)\n";
        return EXIT_FAILURE;
    }

    for (int c = 0; c < num_comp_jpeggpu; ++c) {
        if (sizes_x_nvjpeg[c] != sizes_x_jpeggpu[c]) {
            std::cout << "component " << c << " width mismatch: " << sizes_x_nvjpeg[c]
                      << " (nvJPEG) to " << sizes_x_jpeggpu[c] << " (jpeggpu)\n";
            return EXIT_FAILURE;
        }
        if (sizes_y_nvjpeg[c] != sizes_y_jpeggpu[c]) {
            std::cout << "component " << c << " height mismatch: " << sizes_y_nvjpeg[c]
                      << " (nvJPEG) to " << sizes_y_jpeggpu[c] << " (jpeggpu)\n";
            return EXIT_FAILURE;
        }
    }

    for (int c = 0; c < num_comp_jpeggpu; ++c) {
        if (subsampling_nvjpeg.x[c] != subsampling_jpeggpu.x[c]) {
            std::cout << "component " << c << " css x mismatch: " << subsampling_nvjpeg.x[c]
                      << " (nvJPEG) to " << subsampling_jpeggpu.x[c] << " (jpeggpu)\n";
            return EXIT_FAILURE;
        }
        if (subsampling_nvjpeg.y[c] != subsampling_jpeggpu.y[c]) {
            std::cout << "component " << c << " css y mismatch: " << subsampling_nvjpeg.y[c]
                      << " (nvJPEG) to " << subsampling_jpeggpu.y[c] << " (jpeggpu)\n";
            return EXIT_FAILURE;
        }
    }

    // TODO comparison can be done in kernel

    // TODO MSE is a flawed comparison since differences in the individual components do
    //   do not represent the same difference in the perceived color
    for (int c = 0; c < num_comp_jpeggpu; ++c) {
        size_t squared_error{};
        for (int y = 0; y < sizes_y_jpeggpu[c]; ++y) {
            for (int x = 0; x < sizes_x_jpeggpu[c]; ++x) {
                const size_t idx = y * sizes_x_jpeggpu[c] + x;
                const int16_t diff =
                    int16_t{img_nvjpeg[c][idx]} - int16_t{img_jpeggpu[c][idx]}; // [-255, 255]
                const uint16_t prod = diff * diff; // [0, 65025]
                squared_error += prod;
            }
        }
        const double mse =
            static_cast<double>(squared_error) / (sizes_x_jpeggpu[c] * sizes_y_jpeggpu[c]);
        std::cout << "component " << c << " MSE: " << mse << " ";
    }
    std::cout << "\n";

    if (write_out) {
        std::filesystem::path filepath_nvjpeg = filepath;
        filepath_nvjpeg.replace_extension(std::string(filepath.extension()) + ".nvjpeg.png");
        std::filesystem::path filepath_jpeggpu = filepath;
        filepath_jpeggpu.replace_extension(std::string(filepath.extension()) + ".jpeggpu.png");

        std::cout << "writing out to " << filepath_nvjpeg << " and " << filepath_jpeggpu << "\n";

        uint8_t* img_interleaved =
            static_cast<uint8_t*>(malloc(sizes_x_jpeggpu[0] * sizes_y_jpeggpu[0] * 3));

        conv_to_rgbi(
            sizes_x_nvjpeg,
            sizes_y_nvjpeg,
            num_comp_nvjpeg,
            subsampling_nvjpeg,
            img_nvjpeg[0],
            img_nvjpeg[1],
            img_nvjpeg[2],
            img_interleaved);

        stbi_write_png(
            filepath_nvjpeg.c_str(),
            sizes_x_nvjpeg[0],
            sizes_y_nvjpeg[0],
            3,
            img_interleaved,
            sizes_x_nvjpeg[0] * 3);

        conv_to_rgbi(
            sizes_x_jpeggpu,
            sizes_y_jpeggpu,
            num_comp_jpeggpu,
            subsampling_jpeggpu,
            img_jpeggpu[0],
            img_jpeggpu[1],
            img_jpeggpu[2],
            img_interleaved);

        stbi_write_png(
            filepath_jpeggpu.c_str(),
            sizes_x_jpeggpu[0],
            sizes_y_jpeggpu[0],
            3,
            img_interleaved,
            sizes_x_jpeggpu[0] * 3);

        free(img_interleaved);
    }

    for (int c = 0; c < num_comp_jpeggpu; ++c) {
        free(img_jpeggpu[c]);
        free(img_nvjpeg[c]);
    }

    CHECK_CUDA(cudaFreeHost(file_data));
}
