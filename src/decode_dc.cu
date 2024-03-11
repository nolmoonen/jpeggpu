#include "decode_dc.hpp"
#include "defs.hpp"
#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

using namespace jpeggpu;

namespace {

struct non_interleaved_functor {
    __device__ __host__ int operator()(int i)
    {
        const int data_idx = i * data_unit_size;
        return data_idx;
    }
};

struct interleaved_functor {
    interleaved_functor(int restart_interval, int data_units_in_mcu_component)
        : restart_interval(restart_interval),
          data_units_in_mcu_component(data_units_in_mcu_component)
    {
    }

    __device__ __host__ int operator()(int i)
    {
        const int num_data_units_in_segment = restart_interval * data_units_in_mcu_component;
        const int segment_idx               = i / num_data_units_in_segment;
        return segment_idx;
    }

    int restart_interval;
    int data_units_in_mcu_component;
};

} // namespace

#define CHECK_STAT(call)                                                                           \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return JPEGGPU_INTERNAL_ERROR;                                                         \
        }                                                                                          \
    } while (0)

jpeggpu_status jpeggpu::decode_dc(
    logger& logger,
    jpeggpu::reader& reader, // TODO pass only jpeg_stream?
    int16_t* d_out,
    void*& d_tmp,
    size_t& tmp_size,
    cudaStream_t stream)
{
    // TODO this calculation only works for 4:4:4, 4:2:0, etc.
    int off_in_mcu              = 0; // number of data units, only used for interleaved
    int off_in_data             = 0; // number of data elements, only used for non-interleaved
    const int data_units_in_mcu = reader.jpeg_stream.css.x[0] * reader.jpeg_stream.css.y[0] +
                                  reader.jpeg_stream.num_components - 1;

    for (int c = 0; c < reader.jpeg_stream.num_components; ++c) {
        const int data_units_in_mcu_component =
            reader.jpeg_stream.css.x[c] * reader.jpeg_stream.css.y[c];

        const auto interleaved_transform = [=] __device__ __host__(int i) -> int {
            const int mcu_idx    = i / data_units_in_mcu_component;
            const int idx_in_mcu = off_in_mcu + i % data_units_in_mcu_component;

            const int data_unit_idx = mcu_idx * data_units_in_mcu + idx_in_mcu;
            const int data_idx      = data_unit_idx * data_unit_size;
            return data_idx;
        };

        auto counting_iter = thrust::make_counting_iterator(int{0});
        auto interleaved_index_iter =
            thrust::make_transform_iterator(counting_iter, interleaved_transform);
        auto iter_interleaved = thrust::make_permutation_iterator(d_out, interleaved_index_iter);

        auto non_interleaved_index_iter =
            thrust::make_transform_iterator(counting_iter, non_interleaved_functor{});
        auto iter_non_interleaved =
            thrust::make_permutation_iterator(d_out + off_in_data, non_interleaved_index_iter);

        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;

        const int num_data_units_component = reader.jpeg_stream.data_sizes_x[c] *
                                             reader.jpeg_stream.data_sizes_y[c] / data_unit_size;

        if (reader.jpeg_stream.restart_interval != 0) {
            auto counting_iter_key     = thrust::make_counting_iterator(int{0});
            const int restart_interval = reader.jpeg_stream.restart_interval;
            auto iter_key              = thrust::make_transform_iterator(
                counting_iter_key,
                interleaved_functor(restart_interval, data_units_in_mcu_component));

            const auto dispatch = [&]() {
                if (reader.jpeg_stream.is_interleaved) {
                    CHECK_CUDA(cub::DeviceScan::InclusiveSumByKey(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_key,
                        iter_interleaved,
                        iter_interleaved,
                        num_data_units_component,
                        cub::Equality{},
                        stream));
                } else {
                    CHECK_CUDA(cub::DeviceScan::InclusiveSumByKey(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_key,
                        iter_non_interleaved,
                        iter_non_interleaved,
                        num_data_units_component,
                        cub::Equality{},
                        stream));
                }
            };

            dispatch();

            CHECK_STAT(jpeggpu::gpu_alloc_reserve(
                reinterpret_cast<void**>(&d_tmp_storage), tmp_storage_bytes, d_tmp, tmp_size));

            dispatch();
        } else {
            const auto dispatch = [&]() {
                if (reader.jpeg_stream.is_interleaved) {
                    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_interleaved,
                        iter_interleaved,
                        num_data_units_component,
                        stream));
                } else {
                    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_non_interleaved,
                        iter_non_interleaved,
                        num_data_units_component,
                        stream));
                }
            };

            dispatch();

            CHECK_STAT(jpeggpu::gpu_alloc_reserve(
                reinterpret_cast<void**>(&d_tmp_storage), tmp_storage_bytes, d_tmp, tmp_size));

            dispatch();
        }

        off_in_mcu += data_units_in_mcu_component;
        off_in_data += reader.jpeg_stream.data_sizes_x[c] * reader.jpeg_stream.data_sizes_y[c];
    }

    return JPEGGPU_SUCCESS;
}

#undef CHECK_STAT
