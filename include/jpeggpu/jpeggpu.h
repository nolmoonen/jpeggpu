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
    JPEGGPU_INVALID_JPEG,
    JPEGGPU_INTERNAL_ERROR,
    JPEGGPU_NOT_SUPPORTED
};

const char* jpeggpu_get_status_string(enum jpeggpu_status stat);

enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder);

struct jpeggpu_img_info {
    int size_x[JPEGGPU_MAX_COMP];
    int size_y[JPEGGPU_MAX_COMP];
    int ss_x[JPEGGPU_MAX_COMP];
    int ss_y[JPEGGPU_MAX_COMP];
    int num_components;
};

/// \brief Caller should already allocate file data in pinned memory
///   (`cudaMallocHost`).
enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size);

struct jpeggpu_img {
    uint8_t image[JPEGGPU_MAX_COMP];
    int pitch[JPEGGPU_MAX_COMP];
};

enum jpeggpu_status jpeggpu_decoder_decode(jpeggpu_decoder_t decoder, struct jpeggpu_img* img);

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JPEGGPU_JPEGGPU_H_