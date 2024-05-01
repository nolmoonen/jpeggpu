#ifndef JPEGGPU_READER_HPP_
#define JPEGGPU_READER_HPP_

#include "defs.hpp"
#include "logger.hpp"
#include "util.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stdint.h>

namespace jpeggpu {

/// \brief Holds info describing a segment. If a restart marker should be placed
///   but the number of encoded bits is not a multiple of eight, the encoded stream
///   is padded with zeroes. To be able to deal with this in the decoding process,
///   these segments must be decoded independently.
struct segment {
    /// Number of subsequences before this segment.
    int subseq_offset;
    /// Number of subsequences in this segment.
    int subseq_count;
};

struct huffman_table {
    struct entry {
        int32_t maxcode; /// maxcode[k] is largest code of length k+1, -1 if no codes
        uint16_t mincode; /// mincode[k] is smallest code of length k+1
        uint8_t valptr; /// Huffval[] index of 1st symbol of length k+1
    }  entries[16];
    /// Values associated with each Huffman code, in order of increasing code length
    uint8_t huffval[256];
};

struct scan {
    uint8_t ids[max_comp_count];

    int begin; ///< index, relative to image data, of first byte in scan
    int end; ///< index, relative to image data, of first byte not in scan

    int num_subsequences;
    int num_segments;
};

// TODO make vector class, e.g. int2?
struct component {
    uint8_t id; /// Id as defined in the start of frame header.
    uint8_t qtable_idx; /// Index of quantization table.
    uint8_t dc_idx; /// Index of DC Huffman table.
    uint8_t ac_idx; /// Index of AC Huffman table.
    int size_x; /// Actual image size in pixels.
    int size_y;
    // TODO add some variable for number of data units in MCU?
    int mcu_size_x; /// Number of pixels in one MCU.
    int mcu_size_y;
    int data_size_x; /// Image size in pixels, rounded up to the MCU.
    int data_size_y;
    /// \brief Subsampling factor as defined in the start of frame header,
    ///   i.e. The number of data units in the MCU (if scan is interleaved).
    int ss_x;
    int ss_y;
};

/// cleared with memset
struct jpeg_stream {
    scan scans[max_comp_count];

    int size_x; /// Actual image size in pixels.
    int size_y;
    int num_components; ///< Number of image components.

    // max of header-defined ss for each component, to calculate MCU size. 1, 2, or 4
    int ss_x_max;
    int ss_y_max;

    int num_data_units_in_mcu;
    size_t total_data_size;

    component components[max_comp_count];

    jpeggpu_color_format_jpeg color_fmt;

    int num_mcus_x; /// Image x size in number of MCUs.
    int num_mcus_y; /// Image y size in number of MCUs.

    bool is_interleaved;

    /// \brief Restart interval for differential DC encoding, in number of MCUs.
    ///   Zero if no restart interval is defined.
    int restart_interval;
    int num_scans;
};

struct reader {
    [[nodiscard]] jpeggpu_status startup();

    void cleanup();

    uint8_t read_uint8();
    uint16_t read_uint16();

    bool has_remaining(int size);
    bool has_remaining();

    jpeggpu_status read_marker(uint8_t& marker, logger& logger);

    jpeggpu_status read_sof0(logger& logger);

    jpeggpu_status read_dht(logger& logger);

    jpeggpu_status read_sos(logger& logger);

    jpeggpu_status read_dqt(logger& logger);

    jpeggpu_status read_dri(logger& logger);

    jpeggpu_status skip_segment(logger& logger);

    jpeggpu_status read(logger& logger);

    void reset(const uint8_t* image, const uint8_t* image_end);

    /// cleared with memset
    struct jpeg_stream jpeg_stream; // TODO name it info?

    struct reader_state {
        int scan_idx; /// Index of next scan.
        const uint8_t* image; /// Modifiable pointer to parsing head.
        const uint8_t* image_begin; /// Non-modifiable pointer to start of image.
        const uint8_t* image_end; /// Non-modifiable pointer to end of image.
    } reader_state;

    size_t get_file_size() const { return reader_state.image_end - reader_state.image_begin; }

    // TODO include a `seen_dht` etc and check if exactly one such segment is seen, else return error

    // pinned. in raster order
    qtable* h_qtables[max_comp_count];
    // TODO every scan can have at `max_huffman_count` tables
    // pinned.
    huffman_table* h_huff_tables[max_huffman_count][HUFF_COUNT];

    // TODO better manage this and keep allocation around
    segment* h_segments[max_scan_count];
};

inline int get_size(int size, int ss, int ss_max)
{
    return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

}; // namespace jpeggpu

#endif // JPEGGPU_READER_HPP_
