#include <jpeggpu/jpeggpu.h>
#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <cassert>
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

enum class color_fmt {
    RGBI,
    RGB, // FIXME seems to have an issue
    YCBCR
};

bool is_interleaved(color_fmt fmt)
{
    switch (fmt) {
    case color_fmt::RGBI:
        return true;
    case color_fmt::RGB:
        return false;
    case color_fmt::YCBCR:
        return false;
    }
    assert(false);
    return false;
};

struct image_buffer {
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int num_components; // logical color components
    int num_planes; // allocated memory planes (implied from num_components and mult)
    int mult; // multiplex factor of components in planes
    // pitch is equal to width
    std::vector<void*> h_img;
    std::vector<void*> d_img;
};

image_buffer buffer_startup(
    int (&widths)[NVJPEG_MAX_COMPONENT], int (&heights)[NVJPEG_MAX_COMPONENT], color_fmt fmt)
{
    image_buffer buffer;
    switch (fmt) {
    case color_fmt::RGBI:
        buffer.widths[0] = buffer.widths[1] = buffer.widths[2] = widths[0];
        buffer.heights[0] = buffer.heights[1] = buffer.heights[2] = heights[0];
        buffer.num_components                                     = 3;
        buffer.num_planes                                         = 1;
        buffer.mult                                               = 3;
        break;
    case color_fmt::RGB:
        buffer.widths[0] = buffer.widths[1] = buffer.widths[2] = widths[0];
        buffer.heights[0] = buffer.heights[1] = buffer.heights[2] = heights[0];
        buffer.num_components                                     = 3;
        buffer.num_planes                                         = 3;
        buffer.mult                                               = 1;
        break;
    case color_fmt::YCBCR:
        buffer.widths[0]      = widths[0];
        buffer.widths[1]      = widths[1];
        buffer.widths[2]      = widths[2];
        buffer.heights[0]     = heights[0];
        buffer.heights[1]     = heights[1];
        buffer.heights[2]     = heights[2];
        buffer.num_components = 3;
        buffer.num_planes     = 3;
        buffer.mult           = 1;
        break;
    }

    buffer.h_img.resize(buffer.num_planes);
    buffer.d_img.resize(buffer.num_planes);
    for (int i = 0; i < buffer.num_planes; ++i) {
        const size_t plane_bytes = buffer.widths[i] * buffer.heights[i] * buffer.mult;
        CHECK_CUDA(cudaMallocHost(&(buffer.h_img[i]), plane_bytes));
        CHECK_CUDA(cudaMalloc(&(buffer.d_img[i]), plane_bytes));
    }
    return buffer;
}

void buffer_cleanup(image_buffer buffer)
{
    for (int i = 0; i < buffer.num_planes; ++i) {
        CHECK_CUDA(cudaFree(buffer.d_img[i]));
        CHECK_CUDA(cudaFreeHost(buffer.h_img[i]));
    }
}

void buffer_transfer(image_buffer buffer)
{
    for (int i = 0; i < buffer.num_planes; ++i) {
        const size_t plane_bytes = buffer.widths[i] * buffer.heights[i] * buffer.mult;
        CHECK_CUDA(
            cudaMemcpy(buffer.h_img[i], buffer.d_img[i], plane_bytes, cudaMemcpyHostToDevice));
    }
}

void buffer_write(image_buffer buffer, color_fmt fmt, const std::string& file_name)
{
    std::ofstream file(file_name, std::ios::binary);
    const int size_x = buffer.widths[0];
    const int size_y = buffer.heights[0];
    file << "P6\n" << size_x << " " << size_y << "\n255\n";
    // TODO assumes 3-channel data
    if (is_interleaved(fmt)) {
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                const int i      = y * size_x + x;
                const char* data = static_cast<const char*>(buffer.h_img[0]);
                file << data[3 * i + 0] << data[3 * i + 1] << data[3 * i + 2];
            }
        }
    } else {
        // TODO does not correctly deal with subsampling, as it is not needed for MSE calculation
        //   and outputting the image is only useful for a visual comparison
        // assumes first component is not subsampled
        for (int y = 0; y < size_y; ++y) {
            for (int x = 0; x < size_x; ++x) {
                file << static_cast<const char*>(buffer.h_img[0])[y * size_x + x];
                if (x < buffer.widths[1] && y < buffer.heights[1]) {
                    file << static_cast<const char*>(buffer.h_img[1])[y * buffer.widths[1] + x];
                } else {
                    file << char{0};
                }
                if (x < buffer.widths[2] && y < buffer.heights[2]) {
                    file << static_cast<const char*>(buffer.h_img[2])[y * buffer.widths[2] + x];
                } else {
                    file << char{0};
                }
            }
        }
    }
}

image_buffer decode_nvjpeg(const uint8_t* file_data, size_t file_size, color_fmt fmt)
{
    cudaStream_t stream = 0;

    nvjpegHandle_t nvjpeg_handle;
    CHECK_NVJPEG(nvjpegCreateSimple(&nvjpeg_handle));

    nvjpegJpegState_t nvjpeg_state;
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &channels,
        &subsampling,
        widths,
        heights));

    image_buffer buffer = buffer_startup(widths, heights, fmt);

    const nvjpegOutputFormat_t fmt_nvjpeg = [](color_fmt fmt) -> nvjpegOutputFormat_t {
        switch (fmt) {
        case color_fmt::RGBI:
            return NVJPEG_OUTPUT_RGBI;
        case color_fmt::RGB:
            return NVJPEG_OUTPUT_RGB;
        case color_fmt::YCBCR:
            return NVJPEG_OUTPUT_YUV;
        }
        assert(false);
        return NVJPEG_OUTPUT_UNCHANGED;
    }(fmt);

    nvjpegImage_t d_img;
    for (int i = 0; i < buffer.num_planes; ++i) {
        d_img.channel[i] = static_cast<unsigned char*>(buffer.d_img[i]);
        d_img.pitch[i]   = buffer.widths[i] * buffer.mult;
    }

    CHECK_NVJPEG(nvjpegDecode(
        nvjpeg_handle,
        nvjpeg_state,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        fmt_nvjpeg,
        &d_img,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));

    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));

    buffer_transfer(buffer);
    return buffer;
}

image_buffer decode_jpeggpu(const uint8_t* file_data, size_t file_size, color_fmt fmt)
{
    cudaStream_t stream = 0;

    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

    image_buffer buffer = buffer_startup(img_info.sizes_x, img_info.sizes_y, fmt);

    size_t tmp_size = 0;
    CHECK_JPEGGPU(jpeggpu_decoder_get_buffer_size(decoder, &tmp_size));
    void* d_tmp{};
    CHECK_CUDA(cudaMalloc(&d_tmp, tmp_size));

    CHECK_JPEGGPU(jpeggpu_decoder_transfer(decoder, d_tmp, tmp_size, stream));

    const jpeggpu_color_format_out fmt_jpeggpu = [](color_fmt fmt) -> jpeggpu_color_format_out {
        switch (fmt) {
        case color_fmt::RGBI:
            return JPEGGPU_OUT_SRGB;
        case color_fmt::RGB:
            return JPEGGPU_OUT_SRGB;
        case color_fmt::YCBCR:
            return JPEGGPU_OUT_YCBCR;
        }
        assert(false);
        return JPEGGPU_OUT_NO_CONVERSION;
    }(fmt);

    if (is_interleaved(fmt)) {
        jpeggpu_img_interleaved d_img;
        d_img.image     = static_cast<uint8_t*>(buffer.d_img[0]);
        d_img.pitch     = buffer.widths[0] * buffer.mult;
        d_img.color_fmt = fmt_jpeggpu;

        CHECK_JPEGGPU(jpeggpu_decoder_decode_interleaved(decoder, &d_img, d_tmp, tmp_size, stream));
    } else {
        jpeggpu_img d_img;
        for (int i = 0; i < buffer.num_planes; ++i) {
            d_img.image[i] = static_cast<uint8_t*>(buffer.d_img[i]);
            d_img.pitch[i] = buffer.widths[i] * buffer.mult;
        }
        d_img.color_fmt   = fmt_jpeggpu;
        d_img.subsampling = img_info.subsampling; // no subsampling

        CHECK_JPEGGPU(jpeggpu_decoder_decode(decoder, &d_img, d_tmp, tmp_size, stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaFree(d_tmp));

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));

    buffer_transfer(buffer);
    return buffer;
}

void test(const uint8_t* file_data, size_t file_size, color_fmt fmt)
{
    image_buffer buffer_nvjpeg = decode_nvjpeg(file_data, file_size, fmt);
    buffer_write(
        buffer_nvjpeg,
        fmt,
        std::string("test_nvjpeg_") + std::to_string(static_cast<int>(fmt)) + std::string(".ppm"));
    buffer_cleanup(buffer_nvjpeg);

    image_buffer buffer_jpeggpu = decode_jpeggpu(file_data, file_size, fmt);
    buffer_write(
        buffer_jpeggpu,
        fmt,
        std::string("test_jpeggpu_") + std::to_string(static_cast<int>(fmt)) + std::string(".ppm"));
    buffer_cleanup(buffer_jpeggpu);

    // TODO do MSE comparison
}

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: jpeggpu_test <jpeg file>\n";
        return EXIT_FAILURE;
    }

    std::ifstream file(argv[1]);
    file.seekg(0, std::ios_base::end);
    const std::streampos file_size = file.tellg();
    file.seekg(0);
    char* file_data = nullptr;
    CHECK_CUDA(cudaMallocHost(&file_data, file_size));
    file.read(file_data, file_size);
    file.close();

    // TODO include gray
    // TODO include subsampling
    for (const color_fmt& fmt : {color_fmt::RGBI, color_fmt::RGB, color_fmt::YCBCR}) {
        test(reinterpret_cast<const uint8_t*>(file_data), file_size, fmt);
    }

    CHECK_CUDA(cudaFreeHost(file_data));
}
