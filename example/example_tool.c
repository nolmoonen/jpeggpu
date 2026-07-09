// Copyright (c) 2023-2026 Nol Moonen
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            fprintf(stderr, "failed: " #cond " at: " __FILE__ ":%d\n", __LINE__); \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CHECK_JPEGGPU(call)                                   \
    do {                                                      \
        enum jpeggpu_status stat = call;                      \
        if (stat != JPEGGPU_SUCCESS) {                        \
            fprintf(                                          \
                stderr,                                       \
                "jpeggpu error \"%s\" at: " __FILE__ ":%d\n", \
                jpeggpu_get_status_string(stat),              \
                __LINE__);                                    \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#define CHECK_CUDA(call)                                   \
    do {                                                   \
        cudaError_t err = call;                            \
        if (err != cudaSuccess) {                          \
            fprintf(                                       \
                stderr,                                    \
                "CUDA error \"%s\" at: " __FILE__ ":%d\n", \
                cudaGetErrorString(err),                   \
                __LINE__);                                 \
            exit(EXIT_FAILURE);                            \
        }                                                  \
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

    uint8_t* h_image[JPEGGPU_MAX_COMP];
    struct jpeggpu_img d_img;
    for (int c = 0; c < img_info.num_components; ++c) {
        // Round up to a multiple of eight.
        const int pitch = (img_info.sizes_x[c] + 8 - 1) / 8 * 8;
        h_image[c]      = malloc(img_info.sizes_y[c] * img_info.sizes_x[c]);
        CHECK_CUDA(cudaMalloc((void**)&(d_img.image[c]), img_info.sizes_y[c] * pitch));
        d_img.pitch[c] = pitch;
    }

    CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &d_img, d_tmp, tmp_size, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("gpu decode done\n");

    CHECK_CUDA(cudaFree(d_tmp));

    for (int c = 0; c < img_info.num_components; ++c) {
        CHECK_CUDA(cudaMemcpy2D(
            h_image[c],
            img_info.sizes_x[c], // dpitch
            d_img.image[c],
            d_img.pitch[c], // spitch
            img_info.sizes_x[c],
            img_info.sizes_y[c],
            cudaMemcpyDeviceToHost));
    }

    uint8_t* h_img_interleaved = malloc(img_info.sizes_x[0] * img_info.sizes_y[0] * 3);

    if (conv_to_rgbi(
            img_info.sizes_x,
            img_info.sizes_y,
            img_info.num_components,
            img_info.subsampling,
            h_image[0],
            h_image[1],
            h_image[2],
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
        free(h_image[c]);
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFreeHost(data));
}
