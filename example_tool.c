#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

    const size_t image_size = img_info.sizes_x[0] * img_info.sizes_y[0];
    struct jpeggpu_img_interleaved img;
    CHECK_CUDA(cudaMalloc((void**)&img.image, image_size * 3));
    img.pitch     = img_info.sizes_x[0] * 3;
    img.color_fmt = JPEGGPU_OUT_SRGB;

    CHECK_JPEGGPU(jpeggpu_decoder_decode_interleaved(decoder, &img, d_tmp, tmp_size, stream));

    CHECK_CUDA(cudaFree(d_tmp));

    printf("gpu decode done\n");

    uint8_t* h_img = malloc(image_size * 3);
    CHECK_CUDA(cudaMemcpy(h_img, img.image, image_size * 3, cudaMemcpyDeviceToHost));

    const char* out_filename = "out.png";
    if (argc >= 3) {
        out_filename = argv[2];
    }

    const size_t byte_stride = img.pitch;
    stbi_write_png(out_filename, img_info.sizes_x[0], img_info.sizes_y[0], 3, h_img, byte_stride);

    printf("decoded image at: %s\n", out_filename);

    free(h_img);

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFreeHost(data));
}
