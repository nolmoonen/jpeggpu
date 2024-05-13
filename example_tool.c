// Copyright (c) 2023-2024 Nol Moonen
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

#include "util/util.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image_write.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "failed: " #cond " at: " __FILE__ ":%d\n", __LINE__);                  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CHECK_JPEGGPU(call)                                                                        \
    do {                                                                                           \
        enum jpeggpu_status stat = call;                                                           \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            fprintf(                                                                               \
                stderr,                                                                            \
                "jpeggpu error \"%s\" at: " __FILE__ ":%d\n",                                      \
                jpeggpu_get_status_string(stat),                                                   \
                __LINE__);                                                                         \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            fprintf(                                                                               \
                stderr,                                                                            \
                "CUDA error \"%s\" at: " __FILE__ ":%d\n",                                         \
                cudaGetErrorString(err),                                                           \
                __LINE__);                                                                         \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "usage: example <in_jpeg_file> (optional: <out_png_file>)\n");
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];

    FILE* fp = NULL;
    CHECK((fp = fopen(filename, "r")) != NULL);

    CHECK(fseek(fp, 0, SEEK_END) != -1);
    long int off = 0;
    CHECK((off = ftell(fp)) != -1);
    CHECK(fseek(fp, 0, SEEK_SET) != -1);

    // allocate file data in pinned memory to allow jpeggpu to async copy
    uint8_t* data = NULL;
    CHECK_CUDA(cudaMallocHost((void**)&data, off));
    CHECK(fread(data, 1, off, fp) == off);
    CHECK(fclose(fp) == 0);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));
    CHECK_JPEGGPU(jpeggpu_set_logging(decoder, 1)); // enable logging on stdout

    struct jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(decoder, &img_info, data, off));

    size_t tmp_size = 0;
    CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(decoder, &tmp_size));

    void* d_tmp = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_tmp, tmp_size));

    CHECK_JPEGGPU(jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream));

    struct jpeggpu_img h_img;
    struct jpeggpu_img d_img;
    for (int c = 0; c < img_info.num_components; ++c) {
        const size_t comp_size = img_info.sizes_x[c] * img_info.sizes_y[c];
        h_img.image[c]         = malloc(comp_size);
        CHECK_CUDA(cudaMalloc((void**)&(d_img.image[c]), comp_size));
        h_img.pitch[c] = img_info.sizes_x[c];
        d_img.pitch[c] = img_info.sizes_x[c];
    }

    CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &d_img, d_tmp, tmp_size, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("gpu decode done\n");

    CHECK_CUDA(cudaFree(d_tmp));

    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaMemcpy(
            h_img.image[c],
            d_img.image[c],
            img_info.sizes_x[c] * img_info.sizes_y[c],
            cudaMemcpyDeviceToHost));
    }

    uint8_t* h_img_interleaved = malloc(img_info.sizes_x[0] * img_info.sizes_y[0] * 3);

    if (conv_to_rgbi(
            img_info.sizes_x,
            img_info.sizes_y,
            img_info.num_components,
            img_info.subsampling,
            h_img.image[0],
            h_img.image[1],
            h_img.image[2],
            h_img_interleaved) != EXIT_SUCCESS) {
        printf("simple conversion code cannot handle image\n");
        goto cleanup;
    }

    const char* out_filename = "out.png";
    if (argc >= 3) {
        out_filename = argv[2];
    }

    const size_t byte_stride = img_info.sizes_x[0] * 3;
    stbi_write_png(
        out_filename, img_info.sizes_x[0], img_info.sizes_y[0], 3, h_img_interleaved, byte_stride);

    printf("decoded image at: %s\n", out_filename);

cleanup:
    free(h_img_interleaved);

    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaFree(d_img.image[c]));
        free(h_img.image[c]);
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFreeHost(data));
}
