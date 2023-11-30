#ifndef JPEGGPU_MARKER_HPP_
#define JPEGGPU_MARKER_HPP_

#include <stdint.h>

namespace jpeggpu {

// Table B.1 – Marker code assignments
enum marker_code {
    MARKER_SOF0 = 0xc0,
    MARKER_SOF1 = 0xc1,
    MARKER_SOF2 = 0xc2,
    MARKER_SOF3 = 0xc3,

    MARKER_SOF5 = 0xc5,
    MARKER_SOF6 = 0xc6,
    MARKER_SOF7 = 0xc7,

    MARKER_JPG   = 0xc8,
    MARKER_SOF9  = 0xc9,
    MARKER_SOF10 = 0xca,
    MARKER_SOF11 = 0xcb,

    MARKER_SOF13 = 0xcd,
    MARKER_SOF14 = 0xce,
    MARKER_SOF15 = 0xcf,

    MARKER_DHT = 0xc4,

    MARKER_DAC = 0xcc,

    MARKER_RST0 = 0xd0,
    MARKER_RST1 = 0xd1,
    MARKER_RST2 = 0xd2,
    MARKER_RST3 = 0xd3,
    MARKER_RST4 = 0xd4,
    MARKER_RST5 = 0xd5,
    MARKER_RST6 = 0xd6,
    MARKER_RST7 = 0xd7,

    MARKER_SOI   = 0xd8,
    MARKER_EOI   = 0xd9,
    MARKER_SOS   = 0xda,
    MARKER_DQT   = 0xdb,
    MARKER_DNL   = 0xdc,
    MARKER_DRI   = 0xdd,
    MARKER_DHP   = 0xde,
    MARKER_EXP   = 0xdf,
    MARKER_APP0  = 0xe0,
    MARKER_APP1  = 0xe1,
    MARKER_APP2  = 0xe2,
    MARKER_APP3  = 0xe3,
    MARKER_APP4  = 0xe4,
    MARKER_APP5  = 0xe5,
    MARKER_APP6  = 0xe6,
    MARKER_APP7  = 0xe7,
    MARKER_APP8  = 0xe8,
    MARKER_APP9  = 0xe9,
    MARKER_APP10 = 0xea,
    MARKER_APP11 = 0xeb,
    MARKER_APP12 = 0xec,
    MARKER_APP13 = 0xed,
    MARKER_APP14 = 0xee,
    MARKER_APP15 = 0xef,
    MARKER_JPG0  = 0xf0,
    MARKER_JPG1  = 0xf1,
    MARKER_JPG2  = 0xf2,
    MARKER_JPG3  = 0xf3,
    MARKER_JPG4  = 0xf4,
    MARKER_JPG5  = 0xf5,
    MARKER_JPG6  = 0xf6,
    MARKER_JPG7  = 0xf7,
    MARKER_JPG8  = 0xf8,
    MARKER_JPG9  = 0xf9,
    MARKER_JPG10 = 0xfa,
    MARKER_JPG11 = 0xfb,
    MARKER_JPG12 = 0xfc,
    MARKER_JPG13 = 0xfd,
    MARKER_COM   = 0xfe,

    MARKER_TEM = 0x01,
};

inline const char* get_marker_string(uint8_t code)
{
    switch (code) {
    case MARKER_SOF0:
        return "Baseline DCT";
    case MARKER_SOF1:
        return "Extended sequential DCT";
    case MARKER_SOF2:
        return "Progressive DCT";
    case MARKER_SOF3:
        return "Lossless (sequential)";
    case MARKER_SOF5:
        return "Differential sequential DCT";
    case MARKER_SOF6:
        return "Differential progressive DCT";
    case MARKER_SOF7:
        return "Differential lossless (sequential)";
    case MARKER_JPG:
        return "Reserved for JPEG extensions";
    case MARKER_SOF9:
        return "Extended sequential DCT";
    case MARKER_SOF10:
        return "Progressive DCT";
    case MARKER_SOF11:
        return "Lossless (sequential)";
    case MARKER_SOF13:
        return "Differential sequential DCT";
    case MARKER_SOF14:
        return "Differential progressive DCT";
    case MARKER_SOF15:
        return "Differential lossless (sequential)";
    case MARKER_DHT:
        return "Define Huffman table(s)";
    case MARKER_DAC:
        return "Define arithmetic coding conditioning(s)";
    case MARKER_RST0:
        return "Restart with modulo 8 count 0";
    case MARKER_RST1:
        return "Restart with modulo 8 count 1";
    case MARKER_RST2:
        return "Restart with modulo 8 count 2";
    case MARKER_RST3:
        return "Restart with modulo 8 count 3";
    case MARKER_RST4:
        return "Restart with modulo 8 count 4";
    case MARKER_RST5:
        return "Restart with modulo 8 count 5";
    case MARKER_RST6:
        return "Restart with modulo 8 count 6";
    case MARKER_RST7:
        return "Restart with modulo 8 count 7";
    case MARKER_SOI:
        return "Start of image";
    case MARKER_EOI:
        return "End of image";
    case MARKER_SOS:
        return "Start of scan";
    case MARKER_DQT:
        return "Define quantization table(s)";
    case MARKER_DNL:
        return "Define number of lines";
    case MARKER_DRI:
        return "Define restart interval";
    case MARKER_DHP:
        return "Define hierarchical progression";
    case MARKER_EXP:
        return "Expand reference component(s)";
    case MARKER_APP0:
        return "Application segment 0";
    case MARKER_APP1:
        return "Application segment 1";
    case MARKER_APP2:
        return "Application segment 2";
    case MARKER_APP3:
        return "Application segment 3";
    case MARKER_APP4:
        return "Application segment 4";
    case MARKER_APP5:
        return "Application segment 5";
    case MARKER_APP6:
        return "Application segment 6";
    case MARKER_APP7:
        return "Application segment 7";
    case MARKER_APP8:
        return "Application segment 8";
    case MARKER_APP9:
        return "Application segment 9";
    case MARKER_APP10:
        return "Application segment 10";
    case MARKER_APP11:
        return "Application segment 11";
    case MARKER_APP12:
        return "Application segment 12";
    case MARKER_APP13:
        return "Application segment 13";
    case MARKER_APP14:
        return "Application segment 14";
    case MARKER_APP15:
        return "Application segment 15";
    case MARKER_JPG0:
        return "JPEG extension 0";
    case MARKER_JPG1:
        return "JPEG extension 1";
    case MARKER_JPG2:
        return "JPEG extension 2";
    case MARKER_JPG3:
        return "JPEG extension 3";
    case MARKER_JPG4:
        return "JPEG extension 4";
    case MARKER_JPG5:
        return "JPEG extension 5";
    case MARKER_JPG6:
        return "JPEG extension 6";
    case MARKER_JPG7:
        return "JPEG extension 7";
    case MARKER_JPG8:
        return "JPEG extension 8";
    case MARKER_JPG9:
        return "JPEG extension 9";
    case MARKER_JPG10:
        return "JPEG extension 10";
    case MARKER_JPG11:
        return "JPEG extension 11";
    case MARKER_JPG12:
        return "JPEG extension 12";
    case MARKER_JPG13:
        return "JPEG extension 13";
    case MARKER_COM:
        return "Comment";
    case MARKER_TEM:
        return "Temporary private use";
    }
    return "unknown marker";
}

} // namespace jpeggpu

#endif // JPEGGPU_MARKER_HPP_