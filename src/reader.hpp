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
#include "marker.hpp"
#include "util.hpp"

#include <jpeggpu/jpeggpu.h>

#include <stdint.h>
#include <vector>

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

/// \brief Describes a component as part of a scan. Separate from logical component since
///   tables may be redefined. Furthermore, MCU size etc. depends on whether the scan
///   is interleaved, but with progressive components may appear in multiple scans.
struct scan_component {
    uint8_t dc_idx; /// Global index of DC Huffman table.
    uint8_t ac_idx; /// Global index of AC Huffman table.
    int component_idx; /// Index in component array.

    ivec2 mcu_size; /// Number of pixels in one MCU.
    ivec2 data_size; /// Image size in pixels, rounded up to the MCU.
};

enum class scan_type {
    sequential,
    progressive_dc_initial,
    progressive_dc_refinement,
    progressive_ac_initial,
    progressive_ac_refinement,
};

struct scan {
    int num_components; /// [0, max_comp_count]
    scan_component scan_components[max_comp_count];

    int begin; ///< index, relative to image data, of first byte in scan
    int end; ///< index, relative to image data, of first byte not in scan

    int num_data_units_in_mcu;

    int num_subsequences;
    int num_segments;

    int spectral_start;
    int spectral_end;
    int successive_approx_hi;
    int successive_approx_lo;

    ivec2 num_mcus;
    scan_type type;
};

inline bool is_interleaved(scan& scan) { return scan.num_components > 1; }

/// \brief Logical image component.
struct component {
    uint8_t id; /// Id as defined in the start of frame header.
    uint8_t qtable_idx; /// Index of quantization table.
    ivec2 size; /// Actual image size in pixels.
    /// \brief Subsampling factor as defined in the start of frame header,
    ///   i.e. The number of data units in the MCU (if scan is interleaved).
    ivec2 ss;
    /// \brief Sum of all `data_size` of components before this one in frame header.
    // int offset;

    // TODO this message
    // some max_data_size since data_size is a property of the scan
    //   we need a data_size since it reduces the bounds checking in the decode kernel (for MCU size and data_unit_size)
    //   furthermore, allows aligned loads and stores (data_unit_size)
    ivec2 max_data_size;
};

// TODO an opaque pointer to this struct should be the result of `jpeggpu_decoder_parse_header`
//   and passed to `jpeggpu_decoder_get_buffer_size`, `jpeggpu_decoder_transfer`, and `jpeggpu_decoder_decode`
struct jpeg_stream {
    std::vector<scan> scans;

    ivec2 size; /// Actual image size in pixels.
    int num_components; ///< Number of image components.

    // max of header-defined ss for each component, to calculate MCU size. 1, 2, or 4
    ivec2 ss_max;

    component components[max_comp_count];

    /// \brief Restart interval for differential DC encoding, in number of MCUs.
    ///   Zero if no restart interval is defined.
    int restart_interval;
    int num_scans; /// Number of scans. FIXME double defined with `scans.get_size()`
    int cnt_huff; /// Number of DC and AC Huffman tables. // FIXME double defined
};

/// \brief Whether JPEG stream is sequential. Alternative is progressive.
inline bool is_sequential(uint8_t sof_marker)
{
    return sof_marker == MARKER_SOF0 || sof_marker == MARKER_SOF1;
}

struct reader {
    [[nodiscard]] jpeggpu_status startup();

    void cleanup();

    uint8_t read_uint8();
    uint16_t read_uint16();

    bool has_remaining(int size);
    bool has_remaining();

    jpeggpu_status read_marker(uint8_t& marker, logger& logger);

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
        bool found_sof;
        uint8_t sof_marker;

        static constexpr int idx_not_defined = -1;
        /// \brief For each DC Huffman table slot, the index of the last defined global table.
        int curr_huff_dc[max_comp_count];
        /// \brief For each AC Huffman table slot, the index of the last defined global table.
        int curr_huff_ac[max_comp_count];
    } reader_state;

    size_t get_file_size() const { return reader_state.image_end - reader_state.image_begin; }

    /// \brief Quantization tables in pinned host memory in order of appearance in the JPEG stream.
    ///   Always `max_comp_count` elements.
    std::vector<qtable, pinned_allocator<qtable>> h_qtables;
    /// \brief DC and AC Huffman tables in pinned host memory in order of appearance in the JPEG stream.
    std::vector<huffman_table, pinned_allocator<huffman_table>> h_huff_tables;
    /// \brief Segment info in pinned host memory. One entry per scan, multiple segment infos per scan.
    std::vector<std::vector<segment, pinned_allocator<segment>>> h_scan_segments;
};

inline int get_size(int size, int ss, int ss_max)
{
    return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

}; // namespace jpeggpu

#endif // JPEGGPU_READER_HPP_
