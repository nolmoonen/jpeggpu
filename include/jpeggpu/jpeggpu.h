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
    JPEGGPU_INVALID_ARGUMENT,
    JPEGGPU_INVALID_JPEG, /// the jpeg stream is not compatible with the specification
    JPEGGPU_INTERNAL_ERROR,
    /// the jpeg stream is compatible with the specification, but not supported by jpeggpu
    JPEGGPU_NOT_SUPPORTED,
    JPEGGPU_OUT_OF_HOST_MEMORY,
};

const char* jpeggpu_get_status_string(enum jpeggpu_status stat);

enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder);

/// \brief Set whether to do logging. It is off by default.
enum jpeggpu_status jpeggpu_set_logging(jpeggpu_decoder_t decoder, int do_logging);

/// \brief Color format of JPEG file.
enum jpeggpu_color_format_jpeg {
    JPEGGPU_JPEG_GRAY = 0, /// single-component grayscale
    JPEGGPU_JPEG_YCBCR /// three-component YCbCr BT.601
};

/// \brief Color format of output buffer.
enum jpeggpu_color_format_out {
    JPEGGPU_OUT_GRAY = 0, /// grayscale, one component
    JPEGGPU_OUT_SRGB, /// standard RGB (sRGB), three components
    JPEGGPU_OUT_YCBCR, /// YCbCr BT.601, three components
    /// keep whatever is in the original image. caller should make sure same number of components are available
    ///   subsampling may be different
    JPEGGPU_OUT_NO_CONVERSION
};

/// \brief Subsampling factors as specified in the JPEG header, in [1, 4].
struct jpeggpu_subsampling {
    int x[JPEGGPU_MAX_COMP];
    int y[JPEGGPU_MAX_COMP];
};

// TODO add other help functions
int is_css_444(struct jpeggpu_subsampling css, int num_components);

struct jpeggpu_img_info {
    /// Horizontal size of the image planes.
    int sizes_x[JPEGGPU_MAX_COMP];
    int sizes_y[JPEGGPU_MAX_COMP];
    int num_components;
    enum jpeggpu_color_format_jpeg color_fmt;
    struct jpeggpu_subsampling subsampling;
};

/// \brief Caller should already allocate file data in pinned memory
///   (`cudaMallocHost`).
enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size);

enum jpeggpu_status jpeggpu_decoder_get_buffer_size(jpeggpu_decoder_t decoder, size_t* tmp_size);

/// \param[in] d_tmp Temporary device memory, should be aligned to 256 byte boundary.
enum jpeggpu_status jpeggpu_decoder_transfer(
    jpeggpu_decoder_t decoder, void* d_tmp, size_t tmp_size, cudaStream_t stream);

/// \brief Specifies a device output image.
///   Every component is a separate plane, possibly subsampled.
struct jpeggpu_img {
    uint8_t* image[JPEGGPU_MAX_COMP];
    int pitch[JPEGGPU_MAX_COMP];
    enum jpeggpu_color_format_out color_fmt;
    struct jpeggpu_subsampling subsampling;
};

/// \brief Specifies a device output image.
///   There is one plane, which contains the interleaved components.
struct jpeggpu_img_interleaved {
    uint8_t* image;
    int pitch;
    enum jpeggpu_color_format_out color_fmt;
};

/// \param[in] d_tmp Temporary device memory, should be aligned to 256 byte boundary.
enum jpeggpu_status jpeggpu_decoder_decode(
    jpeggpu_decoder_t decoder,
    struct jpeggpu_img* img,
    void* d_tmp,
    size_t tmp_size,
    cudaStream_t stream);

/// \param[in] d_tmp Temporary device memory, should be aligned to 256 byte boundary.
enum jpeggpu_status jpeggpu_decoder_decode_interleaved(
    jpeggpu_decoder_t decoder,
    struct jpeggpu_img_interleaved* img,
    void* d_tmp,
    size_t tmp_size,
    cudaStream_t stream);

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPEGGPU_JPEGGPU_H_
