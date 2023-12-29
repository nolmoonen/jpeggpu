#include "decoder.hpp"
#include "marker.hpp"
#include <jpeggpu/jpeggpu.h>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h> // only for DBG_PRINT
#include <type_traits>
#include <vector>

extern "C" {

const char* jpeggpu_get_status_string(enum jpeggpu_status stat)
{
    switch (stat) {
    case JPEGGPU_SUCCESS:
        return "success";
    case JPEGGPU_INVALID_ARGUMENT:
        return "invalid argument";
    case JPEGGPU_INVALID_JPEG:
        return "invalid jpeg";
    case JPEGGPU_INTERNAL_ERROR:
        return "internal jpeggpu error";
    case JPEGGPU_NOT_SUPPORTED:
        return "jpeg is not supported";
    }
    return "unknown status";
}

enum jpeggpu_status jpeggpu_decoder_startup(jpeggpu_decoder_t* decoder)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    *decoder = new jpeggpu_decoder;

    return JPEGGPU_SUCCESS;
}

enum jpeggpu_status jpeggpu_decoder_parse_header(
    jpeggpu_decoder_t decoder, struct jpeggpu_img_info* img_info, const uint8_t* data, size_t size)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.parse_header(img_info, data, size);
}

enum jpeggpu_status jpeggpu_decoder_decode(jpeggpu_decoder_t decoder, struct jpeggpu_img* img)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    return decoder->decoder.decode(img);
}

enum jpeggpu_status jpeggpu_decoder_cleanup(jpeggpu_decoder_t decoder)
{
    if (!decoder) {
        return JPEGGPU_INVALID_ARGUMENT;
    }

    delete decoder;

    return JPEGGPU_SUCCESS;
}
}