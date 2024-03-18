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
    /// \brief For non-interleaved scans, returns pixel offset of data unit `i`.
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

    /// \brief For interleaved scans, returns segment index of data unit `i`.
    __device__ __host__ int operator()(int i)
    {
        const int num_data_units_in_segment = restart_interval * data_units_in_mcu_component;
        const int segment_idx               = i / num_data_units_in_segment;
        return segment_idx;
    }

    int restart_interval;
    int data_units_in_mcu_component;
};

struct interleaved_transform_functor {
    interleaved_transform_functor(
        int data_units_in_mcu_component, int off_in_mcu, int data_units_in_mcu)
        : data_units_in_mcu_component(data_units_in_mcu_component),
          off_in_mcu(off_in_mcu),
          data_units_in_mcu(data_units_in_mcu)
    {
    }

    /// \brief For interleaved scan, returns the pixel index of data unit `i`.
    __device__ __host__ int operator()(int i)
    {
        const int mcu_idx    = i / data_units_in_mcu_component;
        const int idx_in_mcu = off_in_mcu + i % data_units_in_mcu_component;

        const int data_unit_idx = mcu_idx * data_units_in_mcu + idx_in_mcu;
        const int data_idx      = data_unit_idx * data_unit_size;
        return data_idx;
    }

    int data_units_in_mcu_component;
    int off_in_mcu;
    int data_units_in_mcu;
};

} // namespace

template <bool do_it>
jpeggpu_status jpeggpu::decode_dc(
    const jpeg_stream& info, int16_t* d_out, stack_allocator& allocator, cudaStream_t stream)
{
    // TODO assumption 3: this calculation only works for 4:4:4, 4:2:0, etc.?
    int off_in_mcu  = 0; // number of data units, only used for interleaved
    int off_in_data = 0; // number of data elements, only used for non-interleaved

    for (int c = 0; c < info.num_components; ++c) {
        const int data_units_in_mcu_component = info.components[c].ss_x * info.components[c].ss_y;

        auto counting_iter = thrust::make_counting_iterator(int{0});

        // iterates over the DC values for the current component in interleaved scan
        auto interleaved_index_iter = thrust::make_transform_iterator(
            counting_iter,
            interleaved_transform_functor(
                data_units_in_mcu_component, off_in_mcu, info.num_data_units_in_mcu));
        auto iter_interleaved = thrust::make_permutation_iterator(d_out, interleaved_index_iter);

        // iterates over DC values in non-interleaved scan
        auto non_interleaved_index_iter =
            thrust::make_transform_iterator(counting_iter, non_interleaved_functor{});
        auto iter_non_interleaved =
            thrust::make_permutation_iterator(d_out + off_in_data, non_interleaved_index_iter);

        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;

        const int num_data_units_component =
            info.components[c].data_size_x * info.components[c].data_size_y / data_unit_size;

        if (info.restart_interval != 0) {
            // if restart interval is defined, scan by key where key is segment index

            auto counting_iter_key     = thrust::make_counting_iterator(int{0});
            const int restart_interval = info.restart_interval;
            auto iter_key              = thrust::make_transform_iterator(
                counting_iter_key,
                interleaved_functor(restart_interval, data_units_in_mcu_component));

            const auto dispatch = [&]() -> cudaError_t {
                if (info.is_interleaved) {
                    return cub::DeviceScan::InclusiveSumByKey(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_key,
                        iter_interleaved,
                        iter_interleaved,
                        num_data_units_component,
                        cub::Equality{},
                        stream);
                } else {
                    return cub::DeviceScan::InclusiveSumByKey(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_key,
                        iter_non_interleaved,
                        iter_non_interleaved,
                        num_data_units_component,
                        cub::Equality{},
                        stream);
                }
            };

            JPEGGPU_CHECK_CUDA(dispatch());

            allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes);

            if (do_it) JPEGGPU_CHECK_CUDA(dispatch());
        } else {
            // if no restart interval is defined, simply perform a single scan

            const auto dispatch = [&]() -> cudaError_t {
                if (info.is_interleaved) {
                    return cub::DeviceScan::InclusiveSum(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_interleaved,
                        iter_interleaved,
                        num_data_units_component,
                        stream);
                } else {
                    return cub::DeviceScan::InclusiveSum(
                        d_tmp_storage,
                        tmp_storage_bytes,
                        iter_non_interleaved,
                        iter_non_interleaved,
                        num_data_units_component,
                        stream);
                }
            };

            JPEGGPU_CHECK_CUDA(dispatch());

            allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes);

            if (do_it) JPEGGPU_CHECK_CUDA(dispatch());
        }

        off_in_mcu += data_units_in_mcu_component;
        off_in_data += info.components[c].data_size_x * info.components[c].data_size_y;
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_dc<false>(
    const jpeg_stream&, int16_t*, stack_allocator&, cudaStream_t);
template jpeggpu_status jpeggpu::decode_dc<true>(
    const jpeg_stream&, int16_t*, stack_allocator&, cudaStream_t);
