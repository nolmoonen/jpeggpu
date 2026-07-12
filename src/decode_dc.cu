// Copyright (c) 2024-2026 Nol Moonen
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

#include "decode_dc.hpp"
#include "defs.hpp"
#include "logger.hpp"
#include "reader.hpp"
#include "util.cuh"

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
        const int data_idx      = data_unit_idx;
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
    int16_t* d_dcs,
    stack_allocator& allocator,
    cudaStream_t stream,
    logger& logger)
{
    int off_in_mcu = 0; // number of data units, only used for interleaved

    // TODO could decode all components in a single scan using int[4] though
    // this function already takes only about 50 us for a big image.

    for (int sc = 0; sc < scan.num_scan_components; ++sc) {
        const scan_component& scan_comp = scan.scan_components[sc];
        const int data_units_in_mcu_component =
            scan_comp.num_blocks_in_mcu.x * scan_comp.num_blocks_in_mcu.y;

        auto counting_iter = thrust::make_counting_iterator(int{0});

        // iterates over the DC values for the current component in interleaved scan
        auto interleaved_index_iter = thrust::make_transform_iterator(
            counting_iter,
            interleaved_transform_functor(
                data_units_in_mcu_component, off_in_mcu, scan.num_data_units_in_mcu));
        auto iter_interleaved = thrust::make_permutation_iterator(d_dcs, interleaved_index_iter);

        void* d_tmp_storage      = nullptr;
        size_t tmp_storage_bytes = 0;

        const int num_data_units_component =
            scan_comp.data_size.x * scan_comp.data_size.y / data_unit_size;

        if (scan.restart_interval != 0) {
            // if restart interval is defined, scan by key where key is segment index

            auto counting_iter_key = thrust::make_counting_iterator(int{0});
            auto iter_key          = thrust::make_transform_iterator(
                counting_iter_key,
                interleaved_functor(scan.restart_interval, data_units_in_mcu_component));

            const auto dispatch = [&]() -> cudaError_t {
                return cub::DeviceScan::InclusiveSumByKey(
                    d_tmp_storage,
                    tmp_storage_bytes,
                    iter_key,
                    iter_interleaved,
                    iter_interleaved,
                    num_data_units_component,
                    dev_eq{},
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
