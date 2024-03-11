#ifndef JPEGGPU_DECODE_DC_HPP_
#define JPEGGPU_DECODE_DC_HPP_

#include "logger.hpp"
#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cuda_runtime.h>

#include <stdint.h>

namespace jpeggpu {

jpeggpu_status decode_dc(
    logger& logger,
    jpeggpu::reader& reader,
    int16_t* d_out,
    void*& d_tmp,
    size_t& tmp_size,
    cudaStream_t stream);

}

#endif // JPEGGPU_DECODE_DC_HPP_
