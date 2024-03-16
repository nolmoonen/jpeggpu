#ifndef JPEGGPU_READER_HPP_
#define JPEGGPU_READER_HPP_

#include "defs.hpp"
#include "logger.hpp"
#include "util.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stdint.h>

namespace jpeggpu {

struct huffman_table {
    /// Smallest code of length k
    int mincode[17];
    /// Largest code of length k (-1 if none)
    int maxcode[18];
    /// Huffval[] index of 1st symbol of length k
    int valptr[17];

    /// These two fields directly represent the contents of a JPEG DHT marker
    /// bits[k] = # of symbols with codes of
    uint8_t bits[17];
    /// The symbols, in order of incr code length
    uint8_t huffval[256];
};

struct scan {
    uint8_t ids[max_comp_count];

    int begin; ///< index, relative to image data, of first byte in scan
    int end; ///< index, relative to image data, of first byte not in scan

    int num_subsequences;
    int num_segments;
};

struct component {
    uint8_t id;
    uint8_t qtable_idx;
    uint8_t dc_idx;
    uint8_t ac_idx;
};

struct reader {
    [[nodiscard]] jpeggpu_status startup();

    void cleanup();

    uint8_t read_uint8();
    uint16_t read_uint16();

    bool has_remaining(int size);

    jpeggpu_status read_marker(uint8_t& marker);

    jpeggpu_status read_sof0();

    jpeggpu_status read_dht();

    jpeggpu_status read_sos();

    jpeggpu_status read_dqt();

    jpeggpu_status read_dri();

    jpeggpu_status skip_segment();

    jpeggpu_status read();

    void reset(const uint8_t* image, const uint8_t* image_end);

    /// cleared with memset
    struct jpeg_stream {
        scan scans[max_comp_count];

        int size_x; ///< Image width.
        int size_y; ///< Image height.
        int num_components; ///< Number of image components.

        // subsampling as defined in the header. 1, 2, or 4
        jpeggpu_subsampling css;
        // max of header-defined ss for each component, to calculate MCU size. 1, 2, or 4
        int ss_x_max;
        int ss_y_max;

        component components[max_comp_count];

        jpeggpu_color_format color_fmt;
        jpeggpu_pixel_format pixel_fmt;

        // TODO place in `components`?
        int sizes_x[max_comp_count];
        int sizes_y[max_comp_count];
        int mcu_sizes_x[max_comp_count]; // TODO make this in number of data units
        int mcu_sizes_y[max_comp_count];
        int data_sizes_x[max_comp_count];
        int data_sizes_y[max_comp_count];
        int num_mcus_x;
        int num_mcus_y;

        bool is_interleaved;

        /// \brief Restart interval for differential DC encoding, in number of MCUs.
        ///   Zero if no restart interval is defined.
        int restart_interval;
        int num_scans;
    } jpeg_stream; // TODO name it info?

    struct reader_state {
        int scan_idx; ///< Index of next scan.
    } reader_state;

    // TODO move into reader_state
    const uint8_t* image; ///< Modifiable pointer to parsing head.
    const uint8_t* image_begin;
    const uint8_t* image_end;

    // TODO include a `seen_dht` etc and check if exactly one such segment is
    //   seen, else return error

    // pinned. in natural order
    qtable* h_qtables[max_comp_count];
    // pinned.
    huffman_table* h_huff_tables[max_comp_count][HUFF_COUNT];
};

inline int get_size(int size, int ss, int ss_max)
{
    return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

}; // namespace jpeggpu

#endif // JPEGGPU_READER_HPP_
