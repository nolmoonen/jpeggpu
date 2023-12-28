#ifndef JPEGGPU_DECODE_GPU_HPP_
#define JPEGGPU_DECODE_GPU_HPP_

#include "reader.hpp"

#include <jpeggpu/jpeggpu.h>

jpeggpu_status process_scan(jpeggpu::reader& reader, cudaStream_t stream);

#endif // JPEGGPU_DECODE_GPU_HPP_