#ifndef JPEGGPU_JPEGGPU_H_
#define JPEGGPU_JPEGGPU_H_

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
};

const char* jpeggpu_get_status_string(enum jpeggpu_status stat);

enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder);

enum jpeggpu_color_format {
    JPEGGPU_GRAY, /// grayscale
    JPEGGPU_SRGB, /// standard RGB (sRGB)
    JPEGGPU_YCBCR, /// YCbCr BT.601
    JPEGGPU_CMYK, /// CYMK
    JPEGGPU_YCCK,
};

enum jpeggpu_pixel_format {
    JPEGGPU_P0,
    JPEGGPU_P0P1P2,
    JPEGGPU_P0P1P2P3,
    JPEGGPU_P012,
    JPEGGPU_P0123
};

/// \brief Subsampling for planes beyond the first plane.
struct jpeggpu_subsampling {
    // ss / ss_max, values in [1, 4]
    int x[JPEGGPU_MAX_COMP];
    int y[JPEGGPU_MAX_COMP];
};

struct jpeggpu_img_info {
    /// Horizontal size of the image planes.
    int size_x;
    int size_y;
    enum jpeggpu_color_format color_fmt;
    enum jpeggpu_pixel_format pixel_fmt;
    struct jpeggpu_subsampling subsampling;
};

/// \brief Caller should already allocate file data in pinned memory
///   (`cudaMallocHost`).
enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size);

struct jpeggpu_img {
    uint8_t* image[JPEGGPU_MAX_COMP];
    int pitch[JPEGGPU_MAX_COMP];
};

// TODO split out copy and kernel. maybe separate function for tmp size
enum jpeggpu_status jpeggpu_decoder_decode(
    jpeggpu_decoder_t decoder,
    struct jpeggpu_img* img,
    enum jpeggpu_color_format color_fmt,
    enum jpeggpu_pixel_format pixel_fmt,
    struct jpeggpu_subsampling subsampling,
    void* d_tmp,
    size_t* tmp_size,
    cudaStream_t stream);

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPEGGPU_JPEGGPU_H_
