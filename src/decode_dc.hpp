#ifndef JPEGGPU_DECODE_DC_HPP_
#define JPEGGPU_DECODE_DC_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

template <bool do_it>
jpeggpu_status decode_dc(
    jpeggpu::reader& reader, int16_t* d_out, stack_allocator& allocator, cudaStream_t stream);

}

#endif // JPEGGPU_DECODE_DC_HPP_
