#ifndef JPEGPUG_READER_HPP_
#define JPEGPUG_READER_HPP_

#include "defs.hpp"
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
    /// # bits, or 0 if too long
    int look_nbits[256];
    /// Symbol, or unused
    unsigned char look_sym[256];

    /// These two fields directly represent the contents of a JPEG DHT marker
    /// bits[k] = # of symbols with codes of
    uint8_t bits[17];
    /// The symbols, in order of incr code length
    uint8_t huffval[256];
};

struct reader {
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

    const uint8_t* image;
    const uint8_t* image_end;

    // TODO include a `seen_dht` etc and check if exactly one such segment is
    //   seen, else return error

    int size_x;
    int size_y;
    int num_components;

    // subsampling as defined in the header. 1, 2, or 4
    jpeggpu_subsampling css;
    // max header-defined ss, to calculate MCU size. 1, 2, or 4
    int ss_x_max;
    int ss_y_max;

    int qtable_idx[max_comp_count]; // the qtable for each comp
    // in natural order
    qtable qtables[max_comp_count];
    huffman_table huff_tables[max_comp_count][HUFF_COUNT];
    int huff_map[max_comp_count][HUFF_COUNT];

    jpeggpu_color_format color_fmt;
    jpeggpu_pixel_format pixel_fmt;

    // TODO AoS
    int sizes_x[max_comp_count];
    int sizes_y[max_comp_count];
    int mcu_sizes_x[max_comp_count];
    int mcu_sizes_y[max_comp_count];
    int data_sizes_x[max_comp_count];
    int data_sizes_y[max_comp_count];
    int num_mcus_x;
    int num_mcus_y;
    int num_segments;

    bool is_interleaved;

    // TODO what if non-interleaved?
    const uint8_t* scan_start;
    size_t scan_size;

    /// \brief Whether a restart interval has been defined.
    bool seen_dri;
    /// \brief Restart interval for differential DC encoding, in number of MCUs.
    int restart_interval;
};

inline int get_size(int size, int ss, int ss_max)
{
    return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

}; // namespace jpeggpu

#endif // JPEGPUG_READER_HPP_
