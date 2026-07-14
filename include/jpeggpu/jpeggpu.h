// Copyright (c) 2023-2026 Nol Moonen
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

#ifndef JPEGGPU_JPEGGPU_H_
#define JPEGGPU_JPEGGPU_H_

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JPEGGPU_MAX_COMP 4

struct jpeggpu_decoder;
typedef struct jpeggpu_decoder* jpeggpu_decoder_t;

enum jpeggpu_status {
    JPEGGPU_SUCCESS,
    /// \brief The user provided an illegal argument to a function.
    JPEGGPU_INVALID_ARGUMENT,
    /// \brief The JPEG stream is not compatible with the specification.
    JPEGGPU_INVALID_JPEG,
    /// \brief An error inside the library occurred.
    JPEGGPU_INTERNAL_ERROR,
    /// \brief The JPEG stream is compatible with the specification, but not supported.
    JPEGGPU_NOT_SUPPORTED,
    /// \brief The system is out of host memory.
    JPEGGPU_OUT_OF_HOST_MEMORY,
    /// \brief The JPEG stream is invalid, likely due to being incomplete.
    JPEGGPU_INCOMPLETE_BITSTREAM
};

/// \brief Return a description of the status code.
const char* jpeggpu_get_status_string(enum jpeggpu_status stat);

/// \brief If `JPEGGPU_SUCCESS` is returned, `jpeggpu_decoder_cleanup` must be called before the program
///   ends, regardless of what status code intermediate functions return.
enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder);

/// \brief Set whether to do logging. It is off by default.
enum jpeggpu_status jpeggpu_set_logging(jpeggpu_decoder_t decoder, int do_logging);

/// \brief Subsampling factors as specified in the JPEG header, in [1, 4].
struct jpeggpu_subsampling {
    int x[JPEGGPU_MAX_COMP];
    int y[JPEGGPU_MAX_COMP];
};

int is_css_444(struct jpeggpu_subsampling css, int num_components);

struct jpeggpu_img_info {
    /// Horizontal size of the image planes.
    int sizes_x[JPEGGPU_MAX_COMP];
    /// Vertical size of the image planes.
    int sizes_y[JPEGGPU_MAX_COMP];
    int num_components;
    struct jpeggpu_subsampling subsampling;
};

/// \brief Parse the JPEG header, no GPU work is performed.
///
/// For best performance, the file data should be allocated in pinned memory (e.g. with `cudaMallocHost`).
enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size);

/// \brief Returns the size of the temporary GPU memory required.
enum jpeggpu_status jpeggpu_decoder_get_buffer_size(jpeggpu_decoder_t decoder, size_t* tmp_size);

/// \brief Performs the host to device copies required by GPU decoding.
/// \param[in] d_tmp Temporary device memory, should be aligned to 256 byte boundary.
enum jpeggpu_status jpeggpu_decoder_transfer(
    jpeggpu_decoder_t decoder, void* d_tmp, size_t tmp_size, cudaStream_t stream);

/// \brief Specifies a device output image.
///   Every component is a separate plane, possibly subsampled.
struct jpeggpu_img {
    uint8_t* image[JPEGGPU_MAX_COMP];
    int pitch[JPEGGPU_MAX_COMP];
};

/// \brief Performs the GPU decode.
/// \param[in] d_tmp Temporary device memory, should be aligned to 256 byte boundary.
enum jpeggpu_status jpeggpu_decoder_decode(
    jpeggpu_decoder_t decoder,
    struct jpeggpu_img* img,
    void* d_tmp,
    size_t tmp_size,
    cudaStream_t stream);

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPEGGPU_JPEGGPU_H_
