#ifndef JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_
#define JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_

#include "benchmark_common.hpp"

#include <nvjpeg.h>

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <thread>
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

int device_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMalloc(ptr, size));
    return 0;
}

int device_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFree(ptr));
    return 0;
}

int pinned_malloc(void*, void** ptr, size_t size, cudaStream_t)
{
    CHECK_CUDA(cudaMallocHost(ptr, size));
    return 0;
}

int pinned_free(void*, void* ptr, size_t, cudaStream_t)
{
    CHECK_CUDA(cudaFreeHost(ptr));
    return 0;
}

struct nvjpeg_state {
    void startup()
    {
        device_allocator = {&device_malloc, &device_free, nullptr};
        pinned_allocator = {&pinned_malloc, &pinned_free, nullptr};

        const nvjpegBackend_t backend = NVJPEG_BACKEND_GPU_HYBRID;
        const int flags               = 0;
        CHECK_NVJPEG(nvjpegCreateEx(backend, nullptr, nullptr, flags, &nvjpeg_handle));
        CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state));
        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle, backend, &nvjpeg_decoder));
        CHECK_NVJPEG(
            nvjpegDecoderStateCreate(nvjpeg_handle, nvjpeg_decoder, &nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(nvjpeg_handle, &device_allocator, &device_buffer));
        CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(nvjpeg_handle, &pinned_allocator, &pinned_buffer));
        CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &jpeg_stream));
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));

        CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
    }

    void cleanup()
    {
        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
        CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_stream));
        CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffer));
        CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
        CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder));
        CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_state));
        CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    }

    nvjpegDevAllocatorV2_t device_allocator;
    nvjpegPinnedAllocatorV2_t pinned_allocator;

    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegJpegDecoder_t nvjpeg_decoder;
    nvjpegJpegState_t nvjpeg_decoupled_state;

    nvjpegBufferDevice_t device_buffer;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegJpegStream_t jpeg_stream;
    nvjpegDecodeParams_t nvjpeg_decode_params;
};

void bench_nvjpeg_st(const uint8_t* file_data, size_t file_size)
{
    cudaStream_t stream = 0;

    nvjpeg_state state;
    state.startup();

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &channels,
        &subsampling,
        widths,
        heights));

    nvjpegImage_t d_img;
    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaMalloc(&d_img.channel[c], widths[c] * heights[c]));
        d_img.pitch[c] = widths[c];
    }

    CHECK_NVJPEG(
        nvjpegDecodeParamsSetOutputFormat(state.nvjpeg_decode_params, NVJPEG_OUTPUT_UNCHANGED));

    const auto run_iter = [&]() {
        const int save_metadata = 0;
        const int save_stream   = 0;
        CHECK_NVJPEG(nvjpegJpegStreamParse(
            state.nvjpeg_handle,
            reinterpret_cast<const unsigned char*>(file_data),
            file_size,
            save_metadata,
            save_stream,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.nvjpeg_decode_params,
            state.jpeg_stream));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            state.jpeg_stream,
            stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(
            state.nvjpeg_handle,
            state.nvjpeg_decoder,
            state.nvjpeg_decoupled_state,
            &d_img,
            stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter(); // warmup; force allocation

    double sum_latency{};
    double max_latency{std::numeric_limits<double>::lowest()};
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        sum_latency += elapsed_us;
        max_latency = std::max(max_latency, elapsed_us);
    }
    const double avg_latency = sum_latency / num_iter / us_in_ms;
    max_latency /= us_in_ms;

    const double total_seconds = sum_latency / us_in_s;
    const double throughput    = num_iter / total_seconds;

    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaFree(d_img.channel[c]));
    }

    state.cleanup();

    printf(" nvJPEG singlethread");
    print_measurement(throughput, avg_latency, max_latency);
}

void bench_nvjpeg_batch(const uint8_t* file_data, size_t file_size, int batch_size)
{
    cudaStream_t stream = 0;

    nvjpeg_state state;
    state.startup();

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    CHECK_NVJPEG(nvjpegGetImageInfo(
        state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(file_data),
        file_size,
        &channels,
        &subsampling,
        widths,
        heights));

    std::vector<nvjpegImage_t> d_img(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        for (int c = 0; c < channels; ++c) {
            CHECK_CUDA(cudaMalloc(&(d_img[i].channel[c]), widths[c] * heights[c]));
            d_img[i].pitch[c] = widths[c];
        }
    }

    CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(
        state.nvjpeg_handle,
        state.nvjpeg_state,
        batch_size,
        std::thread::hardware_concurrency(),
        NVJPEG_OUTPUT_UNCHANGED));

    std::vector<const unsigned char*> file_datas(
        batch_size, reinterpret_cast<const unsigned char*>(file_data));
    std::vector<size_t> file_sizes(batch_size, file_size);

    const auto run_iter = [&]() {
        CHECK_NVJPEG(nvjpegDecodeBatched(
            state.nvjpeg_handle,
            state.nvjpeg_state,
            file_datas.data(),
            file_sizes.data(),
            d_img.data(),
            stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter(); // warmup; force allocation

    const int num_iter_batch = (num_iter - 1 + batch_size) / batch_size;
    double sum_latency{};
    double max_latency{std::numeric_limits<double>::lowest()};
    for (int i = 0; i < num_iter_batch; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        sum_latency += elapsed_us;
        max_latency = std::max(max_latency, elapsed_us);
    }
    const double avg_latency = sum_latency / (num_iter_batch * batch_size) / us_in_ms;
    max_latency /= us_in_ms;

    const double total_seconds = sum_latency / us_in_s;
    const double throughput    = (num_iter_batch * batch_size) / total_seconds;

    for (int i = 0; i < batch_size; ++i) {
        for (int c = 0; c < channels; ++c) {
            CHECK_CUDA(cudaFree(d_img[i].channel[c]));
        }
    }

    state.cleanup();

    printf(" nvJPEG batch %2d    ", batch_size);
    print_measurement(throughput, avg_latency, max_latency);
}

struct bench_nvjpeg_state {
    const uint8_t* file_data;
    size_t file_size;
};

struct bench_nvjpeg_thread_state {
    nvjpeg_state nv_state;
    cudaStream_t stream;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegImage_t d_img;
    double sum_latency; // in milliseconds
    double max_latency; // in milliseconds
};

// runs one iteration of the decoder and copies the result to host
void run_iter(bench_nvjpeg_state& state, bench_nvjpeg_thread_state& thread_state)
{
    const int save_metadata = 0;
    const int save_stream   = 0;
    CHECK_NVJPEG(nvjpegJpegStreamParse(
        thread_state.nv_state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(state.file_data),
        state.file_size,
        save_metadata,
        save_stream,
        thread_state.nv_state.jpeg_stream));

    CHECK_NVJPEG(nvjpegDecodeJpegHost(
        thread_state.nv_state.nvjpeg_handle,
        thread_state.nv_state.nvjpeg_decoder,
        thread_state.nv_state.nvjpeg_decoupled_state,
        thread_state.nv_state.nvjpeg_decode_params,
        thread_state.nv_state.jpeg_stream));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
        thread_state.nv_state.nvjpeg_handle,
        thread_state.nv_state.nvjpeg_decoder,
        thread_state.nv_state.nvjpeg_decoupled_state,
        thread_state.nv_state.jpeg_stream,
        thread_state.stream));

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(
        thread_state.nv_state.nvjpeg_handle,
        thread_state.nv_state.nvjpeg_decoder,
        thread_state.nv_state.nvjpeg_decoupled_state,
        &thread_state.d_img,
        thread_state.stream));

    CHECK_CUDA(cudaStreamSynchronize(thread_state.stream));
};

void bench_nvjpeg_mt_thread_startup(
    bench_nvjpeg_state& state, bench_nvjpeg_thread_state& thread_state)
{
    CHECK_CUDA(cudaStreamCreateWithFlags(&thread_state.stream, cudaStreamNonBlocking));

    thread_state.nv_state.startup();

    CHECK_NVJPEG(nvjpegGetImageInfo(
        thread_state.nv_state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(state.file_data),
        state.file_size,
        &thread_state.channels,
        &thread_state.subsampling,
        thread_state.widths,
        thread_state.heights));

    for (int c = 0; c < thread_state.channels; ++c) {
        const size_t plane_bytes = thread_state.widths[c] * thread_state.heights[c];
        CHECK_CUDA(cudaMalloc(&thread_state.d_img.channel[c], plane_bytes));
        thread_state.d_img.pitch[c] = thread_state.widths[c];
    }

    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(
        thread_state.nv_state.nvjpeg_decode_params, NVJPEG_OUTPUT_UNCHANGED));

    run_iter(state, thread_state); // warmup; force allocation
}

void bench_nvjpeg_mt_thread_perform(
    bench_nvjpeg_state& state, bench_nvjpeg_thread_state& thread_state)
{
    thread_state.sum_latency = 0.0;
    thread_state.max_latency = -std::numeric_limits<double>::max();
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter(state, thread_state);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        thread_state.sum_latency += elapsed_us;
        thread_state.max_latency = std::max(thread_state.max_latency, elapsed_us);
    }
}

void bench_nvjpeg_mt_thread_cleanup(bench_nvjpeg_thread_state& thread_state)
{
    for (int c = 0; c < thread_state.channels; ++c) {
        CHECK_CUDA(cudaFree(thread_state.d_img.channel[c]));
    }

    thread_state.nv_state.cleanup();

    CHECK_CUDA(cudaStreamDestroy(thread_state.stream));
}

template <typename Function>
bool bench_nvjpeg_launch(Function function, std::vector<std::thread>& threads, int num_threads)
{
    std::vector<bool> has_finished(num_threads, false);
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread([&, i]() {
            function(i);
            // this thread has successfully completed its task
            has_finished[i] = true;
        });
    }
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
    const bool all_are_finished =
        std::reduce(has_finished.cbegin(), has_finished.cend(), true, std::logical_and{});
    return all_are_finished;
}

void bench_nvjpeg_mt(const uint8_t* file_data, size_t file_size)
{
    // Initialization, benchmark, and cleanup is done in separate threads, because the potential
    //   for OOM errors is large with high thread counts. Either these threads finish successfully and
    //   set the `has_finished` flags, or they exit due to an error and do not set this flag.

    const int min_num_threads =
        std::max(std::thread::hardware_concurrency() / 2 - 2, (unsigned int){1});
    const int max_num_threads = std::thread::hardware_concurrency() + 2;

    bench_nvjpeg_state state;
    state.file_data = file_data;
    state.file_size = file_size;

    std::vector<bench_nvjpeg_thread_state> thread_states(max_num_threads);
    std::vector<std::thread> threads(max_num_threads);
    for (int num_threads = min_num_threads; num_threads <= max_num_threads; ++num_threads) {
        thread_states.resize(num_threads);

        // initialize thread states
        if (!bench_nvjpeg_launch(
                [&](int thread_id) {
                    bench_nvjpeg_mt_thread_startup(state, thread_states[thread_id]);
                },
                threads,
                num_threads)) {
            break;
        }

        // perform benchmark
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!bench_nvjpeg_launch(
                [&](int thread_id) {
                    bench_nvjpeg_mt_thread_perform(state, thread_states[thread_id]);
                },
                threads,
                num_threads)) {
            break;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        // clean up
        if (!bench_nvjpeg_launch(
                [&](int thread_id) { bench_nvjpeg_mt_thread_cleanup(thread_states[thread_id]); },
                threads,
                num_threads)) {
            break;
        }

        double sum_latency{};
        double max_latency{std::numeric_limits<double>::lowest()};
        for (int i = 0; i < num_threads; ++i) {
            sum_latency += thread_states[i].sum_latency;
            max_latency = std::max(max_latency, thread_states[i].max_latency);
        }
        const double avg_latency = sum_latency / (num_threads * num_iter) / us_in_ms;
        max_latency /= us_in_ms;

        const double total_seconds =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / us_in_s;
        const double throughput = (num_iter * num_threads) / total_seconds;

        printf(" nvJPEG %2d threads  ", num_threads);
        print_measurement(throughput, avg_latency, max_latency);
    }
}

// TODO specify iterations and override number of threads on command line
void bench_nvjpeg(const uint8_t* file_data, size_t file_size)
{
    bench_nvjpeg_st(file_data, file_size);
    bench_nvjpeg_batch(file_data, file_size, 25);
    bench_nvjpeg_batch(file_data, file_size, 50);
    bench_nvjpeg_batch(file_data, file_size, 75);
    bench_nvjpeg_mt(file_data, file_size);
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_
