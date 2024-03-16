#include <jpeggpu/jpeggpu.h>
#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <vector>

#define CHECK_NVJPEG(call)                                                                         \
    do {                                                                                           \
        nvjpegStatus_t stat = call;                                                                \
        if (stat != NVJPEG_STATUS_SUCCESS) {                                                       \
            std::cerr << "nvJPEG error \"" << static_cast<int>(stat) << "\" at: " __FILE__ ":"     \
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

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error \"" << cudaGetErrorString(err) << "\" at: " __FILE__ ":"      \
                      << __LINE__ << "\n";                                                         \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

constexpr int num_warmup_iter = 2;
constexpr int num_iter        = 100;

int device_malloc(void* ctx, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMalloc(ptr, size));
    *reinterpret_cast<size_t*>(ctx) += size;
    return 0;
}

int device_free(void* ctx, void* ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaFree(ptr));
    *reinterpret_cast<size_t*>(ctx) -= size;
    return 0;
}

int pinned_malloc(void* ctx, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMallocHost(ptr, size));
    *reinterpret_cast<size_t*>(ctx) += size;
    return 0;
}

int pinned_free(void* ctx, void* ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaFreeHost(ptr));
    *reinterpret_cast<size_t*>(ctx) -= size;
    return 0;
}

void bench_nvjpeg(char* file_data, size_t file_size, cudaStream_t stream)
{
    size_t mem_device = 0;
    size_t mem_pinned = 0;

    nvjpegDevAllocatorV2_t device_allocator    = {&device_malloc, &device_free, &mem_device};
    nvjpegPinnedAllocatorV2_t pinned_allocator = {&pinned_malloc, &pinned_free, &mem_pinned};

    const nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID;
    nvjpegHandle_t nvjpeg_handle;
    int flags = 0;
    CHECK_NVJPEG(
        nvjpegCreateExV2(backend, &device_allocator, &pinned_allocator, flags, &nvjpeg_handle));

    nvjpegJpegState_t nvjpeg_state;
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));

    nvjpegJpegDecoder_t nvjpeg_decoder;
    CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));

    nvjpegJpegState_t nvjpeg_decoupled_state;
    CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));

    nvjpegBufferDevice_t device_buffer;
    CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(nvjpeg_handle, &device_allocator, &device_buffer));
    nvjpegBufferPinned_t pinned_buffer;
    CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(nvjpeg_handle, &pinned_allocator, &pinned_buffer));

    nvjpegJpegStream_t jpeg_stream;
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));

    nvjpegDecodeParams_t nvjpeg_decode_params;
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

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

    size_t image_bytes = 0;
    nvjpegImage_t img;
    for (int c = 0; c < channels; ++c) {
        const size_t plane_bytes = widths[c] * heights[c];
        CHECK_CUDA(cudaMalloc(&img.channel[c], plane_bytes));
        img.pitch[c] = widths[c];
        image_bytes += plane_bytes;
    }

    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));

    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params, NVJPEG_OUTPUT_YUV));

    const auto run_iter = [&]() {
        const int save_metadata = 0;
        const int save_stream   = 0;
        CHECK_NVJPEG(nvjpegJpegStreamParse(
            nvjpeg_handle,
            reinterpret_cast<const unsigned char*>(file_data),
            file_size,
            save_metadata,
            save_stream,
            jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(
            nvjpeg_handle,
            nvjpeg_decoder,
            nvjpeg_decoupled_state,
            nvjpeg_decode_params,
            jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
            nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, jpeg_stream, stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(
            nvjpeg_handle, nvjpeg_decoder, nvjpeg_decoupled_state, &img, stream));
    };

    for (int i = 0; i < num_warmup_iter; ++i) {
        run_iter();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const std::chrono::high_resolution_clock::time_point t0 =
        std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) {
        run_iter();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const size_t total_bytes = num_iter * (image_bytes + file_size);
    std::cout << std::setprecision(4) << total_bytes / 1e3 / elapsed << " GB/s "
              << mem_device / static_cast<double>(image_bytes) << " device tmp "
              << mem_pinned / static_cast<double>(image_bytes) << " pinned tmp\n";

    // TODO optionally output image for checking purposes

    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaFree(img.channel[c]));
    }

    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_stream));
    CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffer));
    CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
    CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));

    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
}

void bench_jpeggpu(char* file_data, size_t file_size, cudaStream_t stream)
{
    jpeggpu_decoder_t decoder;
    CHECK_JPEGGPU(jpeggpu_decoder_startup(&decoder));

    jpeggpu_img_info img_info;
    CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
        decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

    size_t image_bytes = 0;
    jpeggpu_img img;
    for (int c = 0; c < 3; ++c) {
        int ss_x   = img_info.subsampling.x[c];
        int size_x = (img_info.size_x + ss_x - 1) / ss_x;
        int ss_y   = img_info.subsampling.y[c];
        int size_y = (img_info.size_y + ss_y - 1) / ss_y;

        const size_t plane_bytes = size_x * size_y;
        CHECK_CUDA(cudaMalloc((void**)&img.image[c], plane_bytes));
        img.pitch[0] = size_x;
        image_bytes += plane_bytes;
    }

    void* d_tmp     = nullptr;
    size_t tmp_size = 0;

    const auto run_iter = [&]() {
        CHECK_JPEGGPU(jpeggpu_decoder_parse_header(
            decoder, &img_info, reinterpret_cast<const uint8_t*>(file_data), file_size));

        size_t this_tmp_size = 0;
        CHECK_JPEGGPU(jpeggpu_decoder_decode(
            decoder,
            &img,
            JPEGGPU_YCBCR,
            JPEGGPU_P0P1P2,
            img_info.subsampling,
            nullptr,
            &this_tmp_size,
            stream));

        if (this_tmp_size > tmp_size) {
            if (d_tmp) {
                CHECK_CUDA(cudaFree(d_tmp));
            }
            d_tmp = nullptr;
            CHECK_CUDA(cudaMalloc(&d_tmp, this_tmp_size));
            tmp_size = this_tmp_size;
        }

        CHECK_JPEGGPU(jpeggpu_decoder_decode(
            decoder,
            &img,
            JPEGGPU_YCBCR,
            JPEGGPU_P0P1P2,
            img_info.subsampling,
            d_tmp,
            &this_tmp_size,
            stream));
    };

    for (int i = 0; i < num_warmup_iter; ++i) {
        run_iter();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    const std::chrono::high_resolution_clock::time_point t0 =
        std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) {
        run_iter();
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const size_t total_bytes = num_iter * (image_bytes + file_size);
    std::cout << std::setprecision(4) << total_bytes / 1e3 / elapsed << " GB/s "
              << tmp_size / static_cast<double>(image_bytes) << " device tmp "
              << 0 / static_cast<double>(image_bytes) << " pinned tmp\n"; // TODO

    // TODO optionally output image for checking purposes

    if (d_tmp) CHECK_CUDA(cudaFree(d_tmp));
    for (int c = 0; c < 3; ++c) {
        CHECK_CUDA(cudaFree(img.image[c]));
    }

    CHECK_JPEGGPU(jpeggpu_decoder_cleanup(decoder));
}

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: benchmark <jpeg file>\n";
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

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    std::cout << " nvJPEG ";
    bench_nvjpeg(file_data, file_size, stream);
    std::cout << "jpeggpu ";
    bench_jpeggpu(file_data, file_size, stream);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(file_data));
}
