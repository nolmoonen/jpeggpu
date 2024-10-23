// Copyright (c) 2023-2024 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    } entries[16];
    /// Values associated with each Huffman code, in order of increasing code length
    uint8_t huffval[256];
};

/// \brief Describes a component as part of a scan, which is separate from the logical components
///   as defined in the start of frame segment.
///
///   Huffman tables may be redefined between scans, so the indices are properties of the scan components.
///   The MCU size and data sizes depend on whether the scan is interleaved.
struct scan_component {
    uint8_t dc_idx; /// Index of DC Huffman table, relative to those used by the scan.
    uint8_t ac_idx; /// Index of AC Huffman table, relative to those used by the scan.
    int component_idx; /// Index in the logical component array.

    ivec2 mcu_size; /// Number of pixels in one MCU.
    /// \brief Image size in pixels that are present in the scan, so rounded up to the MCU.
    ivec2 data_size;
};

struct scan {
    int num_scan_components; /// [0, max_comp_count]
    scan_component scan_components[max_comp_count];

    int begin; ///< index, relative to image data, of first byte in scan
    int end; ///< index, relative to image data, of first byte not in scan

    int num_data_units_in_mcu;

    int num_subsequences;
    int num_segments;

    ivec2 num_mcus;

    /// \brief Number of Huffman tables used by all the components in this scan.
    int num_huff_tables;
    /// \brief The global indices of the Huffman tables used in the scan.
    int huff_tables[max_baseline_huff_per_scan];
};

inline bool is_interleaved(const scan& scan) { return scan.num_scan_components > 1; }

/// \brief Logical image component.
struct component {
    uint8_t id; /// Id as defined in the start of frame header.
    uint8_t qtable_idx; /// Index of quantization table.
    /// \brief Component size taking into account subsampling.
    ivec2 size;
    /// \brief Subsampling factor as defined in the start of frame header,
    ///   i.e. The number of data units in the MCU (if scan is interleaved).
    ivec2 ss;
};

/// cleared with memset
struct jpeg_stream {
    /// \brief Do not allow scans to override components of previous scans.
    int num_scans;
    scan scans[max_baseline_scan_count];

    ivec2 size; /// Actual image size in pixels.

    // max of header-defined ss for each component, to calculate MCU size. 1, 2, or 4
    ivec2 ss_max;

    int num_components; ///< Number of image components.
    component components[max_comp_count];

    /// \brief Restart interval for differential DC encoding, in number of MCUs.
    ///   Zero if no restart interval is defined.
    int restart_interval;
};

struct reader {
    [[nodiscard]] jpeggpu_status startup();

    void cleanup();

    uint8_t read_uint8();
    uint16_t read_uint16();

    bool has_remaining(int size);
    bool has_remaining();

    jpeggpu_status read_marker(uint8_t& marker, logger& logger);

    /// \brief Read SOF0 or SOF1.
    jpeggpu_status read_sof(logger& logger);

    jpeggpu_status read_dht(logger& logger);

    jpeggpu_status read_sos(logger& logger);

    jpeggpu_status read_dqt(logger& logger);

    jpeggpu_status read_dri(logger& logger);

    jpeggpu_status skip_segment(logger& logger);

    jpeggpu_status read(logger& logger);

    void reset(const uint8_t* image, const uint8_t* image_end);

    struct jpeg_stream jpeg_stream; // TODO name it info?

    struct reader_state {
        const uint8_t* image; /// Modifiable pointer to parsing head.
        const uint8_t* image_begin; /// Non-modifiable pointer to start of image.
        const uint8_t* image_end; /// Non-modifiable pointer to end of image.
        bool found_sof; /// Whether SOF has been found, may only appear once for baseline JPEGs.

        bool qtable_defined[max_comp_count];

        static constexpr int idx_not_defined = -1;
        /// \brief For each DC Huffman table slot, the index of the last defined global table.
        int curr_huff_dc[max_comp_count];
        /// \brief For each AC Huffman table slot, the index of the last defined global table.
        int curr_huff_ac[max_comp_count];
    } reader_state;

    size_t get_file_size() const { return reader_state.image_end - reader_state.image_begin; }

    // TODO the quantization table may be redefined between scans
    /// \brief Quantization tables in pinned host memory in order of appearance in the JPEG stream.
    ///   Always `max_comp_count` elements since only one frame is allowed.
    std::vector<qtable, pinned_allocator<qtable>> h_qtables;
    /// \brief DC and AC Huffman tables in pinned host memory in order of appearance in the JPEG stream.
    std::vector<huffman_table, pinned_allocator<huffman_table>> h_huff_tables;
    /// \brief Segment info in pinned host memory. One entry per scan, multiple segment infos per scan.
    std::vector<segment, pinned_allocator<segment>> h_scan_segments[max_baseline_scan_count];
};

inline int get_size(int size, int ss, int ss_max)
{
    return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

}; // namespace jpeggpu

#endif // JPEGGPU_READER_HPP_
