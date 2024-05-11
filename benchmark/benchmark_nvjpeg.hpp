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

void bench_nvjpeg_st(const char* file_data, size_t file_size)
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

    std::vector<char*> h_img(channels);

    nvjpegImage_t d_img;
    size_t image_bytes = 0;
    for (int c = 0; c < channels; ++c) {
        const size_t plane_bytes = widths[c] * heights[c];
        CHECK_CUDA(cudaMalloc(&d_img.channel[c], plane_bytes));
        d_img.pitch[c] = widths[c];
        image_bytes += plane_bytes;
        CHECK_CUDA(cudaMallocHost(&h_img[c], plane_bytes));
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
        for (int c = 0; c < channels; ++c) {
            CHECK_CUDA(cudaMemcpyAsync(
                h_img[c],
                d_img.channel[c],
                widths[c] * heights[c],
                cudaMemcpyDeviceToHost,
                stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    run_iter();

    double sum_latency{};
    double max_latency{std::numeric_limits<double>::lowest()};
    for (int i = 0; i < num_iter; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_iter();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        sum_latency += elapsed_ms;
        max_latency = std::max(max_latency, elapsed_ms);
    }
    const double avg_latency = sum_latency / num_iter;

    const double total_seconds = sum_latency / 1e3;
    const double throughput    = num_iter / total_seconds;

    for (int c = 0; c < channels; ++c) {
        CHECK_CUDA(cudaFreeHost(h_img[c]));
        CHECK_CUDA(cudaFree(d_img.channel[c]));
    }

    state.cleanup();

    std::cout << " nvJPEG singlethreaded, throughput: " << std::fixed << std::setw(5)
              << std::setprecision(2) << throughput << " image/s, avg latency: " << avg_latency
              << "ms, max latency: " << max_latency << "ms\n";
}

struct bench_nvjpeg_state {
    const char* file_data;
    size_t file_size;
    bool is_alternative;
    cudaStream_t stream_kernel; // only used in alternative version
    std::mutex mutex; // only used in alternative version
};

struct bench_nvjpeg_thread_state {
    nvjpeg_state nv_state;
    cudaStream_t stream;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;
    std::vector<char*> h_img;
    nvjpegImage_t d_img;
    double sum_latency; // in milliseconds
    double max_latency; // in milliseconds
    cudaEvent_t event_h2d; // only used in alternative version
    cudaEvent_t event_d2h; // only used in alternative version
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

    if (state.is_alternative) {
        // Manually serialize kernels and copies to minimize latency. This is achieved
        //   by performing the scheduling in a critical section. Two streams are used,
        //   to allow overlap in memcpy and kernel execution. This works well for
        //   particular image sizes. TODO which?
        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
            thread_state.nv_state.nvjpeg_handle,
            thread_state.nv_state.nvjpeg_decoder,
            thread_state.nv_state.nvjpeg_decoupled_state,
            thread_state.nv_state.jpeg_stream,
            thread_state.stream));
        // as a consequence of using two streams, explicit synchronization is inserted
        CHECK_CUDA(cudaEventRecord(thread_state.event_h2d, thread_state.stream));
        {
            std::lock_guard<std::mutex>(state.mutex);
            CHECK_CUDA(cudaStreamWaitEvent(state.stream_kernel, thread_state.event_h2d));
            CHECK_NVJPEG(nvjpegDecodeJpegDevice(
                thread_state.nv_state.nvjpeg_handle,
                thread_state.nv_state.nvjpeg_decoder,
                thread_state.nv_state.nvjpeg_decoupled_state,
                &thread_state.d_img,
                state.stream_kernel));
            CHECK_CUDA(cudaEventRecord(thread_state.event_d2h, state.stream_kernel));
        }
        CHECK_CUDA(cudaStreamWaitEvent(thread_state.stream, thread_state.event_d2h));
        for (int c = 0; c < thread_state.channels; ++c) {
            CHECK_CUDA(cudaMemcpyAsync(
                thread_state.h_img[c],
                thread_state.d_img.channel[c],
                thread_state.widths[c] * thread_state.heights[c],
                cudaMemcpyDeviceToHost,
                thread_state.stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(thread_state.stream));
    } else {
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
        for (int c = 0; c < thread_state.channels; ++c) {
            CHECK_CUDA(cudaMemcpyAsync(
                thread_state.h_img[c],
                thread_state.d_img.channel[c],
                thread_state.widths[c] * thread_state.heights[c],
                cudaMemcpyDeviceToHost,
                thread_state.stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(thread_state.stream));
    }
};

void bench_nvjpeg_mt_thread_startup(
    bench_nvjpeg_state& state, bench_nvjpeg_thread_state& thread_state)
{
    CHECK_CUDA(cudaStreamCreateWithFlags(&thread_state.stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&thread_state.event_h2d));
    CHECK_CUDA(cudaEventCreate(&thread_state.event_d2h));

    thread_state.nv_state.startup();

    CHECK_NVJPEG(nvjpegGetImageInfo(
        thread_state.nv_state.nvjpeg_handle,
        reinterpret_cast<const unsigned char*>(state.file_data),
        state.file_size,
        &thread_state.channels,
        &thread_state.subsampling,
        thread_state.widths,
        thread_state.heights));

    thread_state.h_img.resize(thread_state.channels);

    size_t image_bytes = 0;
    for (int c = 0; c < thread_state.channels; ++c) {
        const size_t plane_bytes = thread_state.widths[c] * thread_state.heights[c];
        CHECK_CUDA(cudaMalloc(&thread_state.d_img.channel[c], plane_bytes));
        thread_state.d_img.pitch[c] = thread_state.widths[c];
        image_bytes += plane_bytes;
        CHECK_CUDA(cudaMallocHost(&thread_state.h_img[c], plane_bytes));
    }

    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(
        thread_state.nv_state.nvjpeg_decode_params, NVJPEG_OUTPUT_UNCHANGED));

    // run one warmup
    run_iter(state, thread_state);
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
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        thread_state.sum_latency += elapsed_ms;
        thread_state.max_latency = std::max(thread_state.max_latency, elapsed_ms);
    }
}

void bench_nvjpeg_mt_thread_cleanup(bench_nvjpeg_thread_state& thread_state)
{
    for (int c = 0; c < thread_state.channels; ++c) {
        CHECK_CUDA(cudaFreeHost(thread_state.h_img[c]));
        CHECK_CUDA(cudaFree(thread_state.d_img.channel[c]));
    }

    thread_state.nv_state.cleanup();

    CHECK_CUDA(cudaStreamDestroy(thread_state.stream));
    CHECK_CUDA(cudaEventDestroy(thread_state.event_d2h));
    CHECK_CUDA(cudaEventDestroy(thread_state.event_h2d));
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

void bench_nvjpeg_mt(const char* file_data, size_t file_size, bool is_alternative)
{
    // Initialization, benchmark, and cleanup is done in separate threads, because the potential
    //   for OOM errors is large with high thread counts. Either these threads finish successfully and
    //   set the `has_finished` flags, or they exit due to an error and do not set this flag.

    const int min_num_threads =
        std::max(std::thread::hardware_concurrency() / 2 - 2, (unsigned int){1});
    const int max_num_threads = std::thread::hardware_concurrency() + 2;

    bench_nvjpeg_state state;
    state.file_data      = file_data;
    state.file_size      = file_size;
    state.is_alternative = is_alternative;
    CHECK_CUDA(cudaStreamCreateWithFlags(&state.stream_kernel, cudaStreamNonBlocking));

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
        const double avg_latency = sum_latency / (num_threads * num_iter);

        const double total_seconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1e3;
        const double throughput = (num_iter * num_threads) / total_seconds;

        std::cout << "#threads: " << std::setw(2) << num_threads << ", throughput: " << std::fixed
                  << std::setw(5) << std::setprecision(2) << throughput
                  << " image/s, avg latency: " << avg_latency << "ms, max latency: " << max_latency
                  << "ms\n";
    }

    CHECK_CUDA(cudaStreamDestroy(state.stream_kernel));
}

// TODO specify iterations and override number of threads on command line
void bench_nvjpeg(const char* file_data, size_t file_size)
{
    bench_nvjpeg_st(file_data, file_size);
    std::cout << "nvJPEG\n";
    bench_nvjpeg_mt(file_data, file_size, false);
    bool do_alternative = false;
    if (do_alternative) {
        std::cout << "nvJPEG (alternative)\n";
        bench_nvjpeg_mt(file_data, file_size, true);
    }
}

#endif // JPEGGPU_BENCHMARK_BENCHMARK_NVJPEG_HPP_
