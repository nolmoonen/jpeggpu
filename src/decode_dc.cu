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
    const jpeg_stream& info,
    const scan& scan,
    int16_t* d_out,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    int off_in_mcu = 0; // number of data units, only used for interleaved

    for (int sc = 0; sc < scan.num_scan_components; ++sc) {
        const scan_component& scan_comp       = scan.scan_components[sc];
        const component& comp                 = info.components[scan_comp.component_idx];
        const int data_units_in_mcu_component = comp.ss.x * comp.ss.y;

        auto counting_iter = thrust::make_counting_iterator(int{0});

        // iterates over the DC values for the current component in interleaved scan
        auto interleaved_index_iter = thrust::make_transform_iterator(
            counting_iter,
            interleaved_transform_functor(
                data_units_in_mcu_component, off_in_mcu, scan.num_data_units_in_mcu));
        auto iter_interleaved = thrust::make_permutation_iterator(d_out, interleaved_index_iter);

        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;

        const int num_data_units_component =
            scan_comp.data_size.x * scan_comp.data_size.y / data_unit_size;

        if (info.restart_interval != 0) {
            // if restart interval is defined, scan by key where key is segment index

            auto counting_iter_key     = thrust::make_counting_iterator(int{0});
            const int restart_interval = info.restart_interval;
            auto iter_key              = thrust::make_transform_iterator(
                counting_iter_key,
                interleaved_functor(restart_interval, data_units_in_mcu_component));

            const auto dispatch = [&]() -> cudaError_t {
                return cub::DeviceScan::InclusiveSumByKey(
                    d_tmp_storage,
                    tmp_storage_bytes,
                    iter_key,
                    iter_interleaved,
                    iter_interleaved,
                    num_data_units_component,
                    cub::Equality{},
                    stream);
            };

            JPEGGPU_CHECK_CUDA(dispatch());

            allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes);

            if (do_it) JPEGGPU_CHECK_CUDA(dispatch());
        } else {
            // if no restart interval is defined, simply perform a single scan

            const auto dispatch = [&]() -> cudaError_t {
                return cub::DeviceScan::InclusiveSum(
                    d_tmp_storage,
                    tmp_storage_bytes,
                    iter_interleaved,
                    iter_interleaved,
                    num_data_units_component,
                    stream);
            };

            JPEGGPU_CHECK_CUDA(dispatch());

            allocator.reserve<do_it>(&d_tmp_storage, tmp_storage_bytes);

            if (do_it) JPEGGPU_CHECK_CUDA(dispatch());
        }

        off_in_mcu += data_units_in_mcu_component;
    }

    return JPEGGPU_SUCCESS;
}

template jpeggpu_status jpeggpu::decode_dc<false>(
    const jpeg_stream&, const scan&, int16_t*, stack_allocator&, cudaStream_t, logger&);
template jpeggpu_status jpeggpu::decode_dc<true>(
    const jpeg_stream&, const scan&, int16_t*, stack_allocator&, cudaStream_t, logger&);
