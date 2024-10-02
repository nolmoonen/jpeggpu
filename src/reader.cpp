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

#include "reader.hpp"
#include "decoder_defs.hpp"
#include "defs.hpp"
#include "marker.hpp"
#include "util.hpp"

#include <jpeggpu/jpeggpu.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <stdint.h>
#include <vector>

using namespace jpeggpu;

jpeggpu_status reader::startup()
{
    JPEGGPU_CHECK_STAT(nothrow_resize(h_qtables, 4));
    JPEGGPU_CHECK_STAT(nothrow_resize(h_huff_tables, 2 * max_comp_count));
    JPEGGPU_CHECK_STAT(nothrow_resize(h_scan_segments, 4));

    return JPEGGPU_SUCCESS;
}

void reader::cleanup() {}

uint8_t jpeggpu::reader::read_uint8() { return *(reader_state.image++); }

uint16_t jpeggpu::reader::read_uint16()
{
    const uint8_t high = read_uint8();
    return static_cast<uint16_t>(high) << 8 | read_uint8();
}

bool jpeggpu::reader::has_remaining(int size)
{
    return reader_state.image_end - reader_state.image >= size;
}

bool jpeggpu::reader::has_remaining() { return has_remaining(1); }

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_marker(uint8_t& marker, logger& logger)
{
    if (!has_remaining(2)) {
        logger.log("\ttoo few bytes for marker\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t ff = read_uint8();
    if (ff != 0xff) {
        logger.log("\tinvalid marker byte 0x%02x\n", ff);
        return JPEGGPU_INVALID_JPEG;
    }
    marker = read_uint8();
    return JPEGGPU_SUCCESS;
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_sof(logger& logger)
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16();
    if (length < 2 || !has_remaining(length - 2)) {
        logger.log("\tincomplete bitstream\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t precision = read_uint8();
    if (precision != 8) {
        logger.log("\tunsupported sample precision %d, only 8 is supported\n", int{precision});
        return JPEGGPU_NOT_SUPPORTED;
    }
    const uint16_t num_lines            = read_uint16();
    jpeg_stream.size.y                  = num_lines;
    const uint16_t num_samples_per_line = read_uint16();
    jpeg_stream.size.x                  = num_samples_per_line;
    if (jpeg_stream.size.x == 0 || jpeg_stream.size.y == 0) {
        logger.log("\tinvalid size x=%d, y=%d\n", jpeg_stream.size.x, jpeg_stream.size.y);
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t num_img_components = read_uint8();
    jpeg_stream.num_components       = num_img_components;
    if (jpeg_stream.num_components == 0) {
        logger.log("\tzero components\n");
        return JPEGGPU_INVALID_JPEG;
    }
    if (jpeg_stream.num_components > max_comp_count) {
        logger.log("\ttoo many components %d\n", jpeg_stream.num_components);
        return JPEGGPU_NOT_SUPPORTED;
    }

    logger.log(
        "\tsize_x: %" PRIu16 ", size_y: %" PRIu16 ", num_components: %" PRIu8 "\n",
        jpeg_stream.size.x,
        jpeg_stream.size.y,
        jpeg_stream.num_components);

    jpeg_stream.ss_max = {0, 0};
    for (int c = 0; c < jpeg_stream.num_components; ++c) {
        component& comp                = jpeg_stream.components[c];
        const uint8_t component_id     = read_uint8();
        comp.id                        = component_id;
        const uint8_t sampling_factors = read_uint8();
        const int ss_x_c               = sampling_factors >> 4;
        if (ss_x_c < 1 || ss_x_c > 4) {
            return JPEGGPU_INVALID_JPEG;
        }
        const int ss_y_c = sampling_factors & 0xf;
        if (ss_y_c < 1 || ss_y_c > 4) {
            return JPEGGPU_INVALID_JPEG;
        }

        if (jpeg_stream.num_components == 1) {
            // Specification allows the subsampling factor to not be 1 when there is only
            //   one component. However, in this case it is effectively ignored and we set
            //   it to 1x1.
            comp.ss.x = 1;
            comp.ss.y = 1;
        } else {
            comp.ss.x = ss_x_c;
            comp.ss.y = ss_y_c;
        }

        const uint8_t qi = read_uint8();
        comp.qtable_idx  = qi;
        logger.log(
            "\tc_id: %d, ssx: %d, ssy: %d, qi: %d\n",
            comp.id,
            comp.ss.x,
            comp.ss.y,
            comp.qtable_idx);
        jpeg_stream.ss_max = {
            std::max(jpeg_stream.ss_max.x, comp.ss.x), std::max(jpeg_stream.ss_max.y, comp.ss.y)};
    }

    for (int c = 0; c < jpeg_stream.num_components; ++c) {
        component& comp = jpeg_stream.components[c];
        // A.1.1
        comp.size = {
            get_size(jpeg_stream.size.x, comp.ss.x, jpeg_stream.ss_max.x),
            get_size(jpeg_stream.size.y, comp.ss.y, jpeg_stream.ss_max.y)};
    }

    return JPEGGPU_SUCCESS;
}

void compute_huffman_table(jpeggpu::huffman_table& table, const uint8_t (&num_codes)[16])
{
    uint16_t huffcode[256];
    int code_idx  = 0;
    uint16_t code = 0;
    for (int l = 0; l < 16; ++l) {
        for (uint8_t i = 0; i < num_codes[l]; i++) {
            assert(code_idx < 256);
            huffcode[code_idx++] = code;
            ++code;
        }
        code <<= 1;
    }

    // generate decoding tables for bit-sequential decoding
    code_idx = 0;
    for (int l = 0; l < 16; ++l) {
        if (num_codes[l]) {
            table.entries[l].valptr  = code_idx; // huffval[] index of 1st symbol of code length l
            table.entries[l].mincode = huffcode[code_idx]; // minimum code of length l
            code_idx += num_codes[l];
            table.entries[l].maxcode = huffcode[code_idx - 1]; // maximum code of length l
        } else {
            table.entries[l].maxcode = -1; // -1 if no codes of this length
        }
    }
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_dht(logger& logger)
{
    if (!has_remaining(2)) {
        logger.log("\ttoo few bytes in DHT segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        logger.log("\ttoo few bytes in DHT segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    int remaining = length;
    while (remaining > 0) {
        const uint8_t index = read_uint8();
        --remaining;
        const int table_class = index >> 4;
        const int th          = index & 0xf;
        if (table_class != 0 && table_class != 1) {
            logger.log("\tinvalid Huffman table class\n");
            return JPEGGPU_INVALID_JPEG;
        }
        const bool is_dc = table_class == 0;
        if (th > 3) {
            logger.log("\tHuffman table index must be 0, 1, 2, or 3\n");
            return JPEGGPU_NOT_SUPPORTED;
        }

        if (!has_remaining(16)) {
            logger.log("\ttoo few bytes in DHT segment\n");
            return JPEGGPU_INVALID_JPEG;
        }

        logger.log("\t%s Huffman table index %d\n", is_dc ? "DC" : "AC", th);

        const int idx = jpeg_stream.cnt_huff++;
        (is_dc ? reader_state.curr_huff_dc : reader_state.curr_huff_ac)[th] = idx;
        JPEGGPU_CHECK_STAT(nothrow_resize(h_huff_tables, jpeg_stream.cnt_huff));
        huffman_table& table = h_huff_tables[idx];

        /// num_codes[i] is # of symbols with codes of i + 1 bits
        uint8_t num_codes[16];
        int count = 0;
        for (int i = 0; i < 16; ++i) {
            num_codes[i] = read_uint8();
            count += num_codes[i];
        }
        remaining -= 16;

        // TODO reject if all same length, since no synchronization will happen

        if (!has_remaining(count)) {
            logger.log("\ttoo few bytes in DHT segment\n");
            return JPEGGPU_INVALID_JPEG;
        }

        if (static_cast<size_t>(count) > sizeof(table.huffval) / sizeof(table.huffval[0])) {
            logger.log("\ttoo many values\n");
            return JPEGGPU_INVALID_JPEG;
        }

        // read huffval
        for (int i = 0; i < count; ++i) {
            table.huffval[i] = read_uint8();
        }
        remaining -= count;

        compute_huffman_table(table, num_codes);
    }

    return JPEGGPU_SUCCESS;
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_sos(logger& logger)
{
    if (!reader_state.found_sof) {
        return JPEGGPU_INVALID_JPEG;
    }

    if (!has_remaining(3)) {
        logger.log("\ttoo few bytes in SOS segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const int scan_idx = jpeg_stream.num_scans++;

    JPEGGPU_CHECK_STAT(nothrow_resize(jpeg_stream.scans, jpeg_stream.num_scans));
    scan& scan = jpeg_stream.scans[scan_idx];

    const uint16_t length        = read_uint16();
    const uint8_t num_components = read_uint8();
    scan.num_components          = num_components;

    if (num_components < 1 || num_components > 4) {
        logger.log("\tinvalid number of components in scan %d\n", scan.num_components);
        return JPEGGPU_INVALID_JPEG;
    }

    if (length != 2 + 1 + 2 * num_components + 3) {
        return JPEGGPU_INVALID_JPEG;
    }

    scan.num_data_units_in_mcu = 0;
    for (int c = 0; c < num_components; ++c) {
        scan_component& scan_component = scan.scan_components[c];
        if (!has_remaining(2)) {
            return JPEGGPU_INVALID_JPEG;
        }

        const uint8_t selector      = read_uint8();
        const uint8_t acdc_selector = read_uint8();
        const uint8_t id_dc         = acdc_selector >> 4;
        const uint8_t id_ac         = acdc_selector & 0xf;
        logger.log("\tc_id: %" PRIu8 ", dc: %d, ac: %d\n", selector, id_dc, id_ac);

        int component_idx = -1;
        for (int i = 0; i < jpeg_stream.num_components; ++i) {
            if (jpeg_stream.components[i].id == selector) {
                component_idx = i;
                break;
            }
        }

        if (component_idx == -1) {
            logger.log("\tinvalid component selector\n");
            return JPEGGPU_INVALID_JPEG;
        }

        scan_component.component_idx = component_idx;

        // A.2 Order of source image data encoding:
        //   The order of the components in a scan is equal to the order of the components in the frame header.
        if (c > 0 && scan_component.component_idx <= scan.scan_components[c - 1].component_idx) {
            logger.log("\tinvalid component order in scan\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (id_dc > 3 || id_ac > 3) {
            logger.log("\tHuffman table id out of bounds\n");
            return JPEGGPU_INVALID_JPEG;
        }

        // Determine global Huffman table indices
        scan_component.dc_idx = reader_state.curr_huff_dc[id_dc];
        scan_component.ac_idx = reader_state.curr_huff_ac[id_ac];

        component& comp = jpeg_stream.components[component_idx];

        // Calculate size properties

        scan_component.mcu_size = {
            is_interleaved(scan) ? data_unit_vector_size * comp.ss.x : data_unit_vector_size,
            is_interleaved(scan) ? data_unit_vector_size * comp.ss.y : data_unit_vector_size};

        assert(scan_component.mcu_size.x > 0 && scan_component.mcu_size.y > 0);
        scan_component.data_size.x =
            ceiling_div(comp.size.x, static_cast<unsigned int>(scan_component.mcu_size.x)) *
            scan_component.mcu_size.x;
        scan_component.data_size.y =
            ceiling_div(comp.size.y, static_cast<unsigned int>(scan_component.mcu_size.y)) *
            scan_component.mcu_size.y;

        // A.2.4 Completion of partial MCU
        assert(scan_component.data_size.x % data_unit_vector_size == 0);
        assert(scan_component.data_size.y % data_unit_vector_size == 0);

        if (is_interleaved(scan)) {
            assert(scan_component.data_size.x % (data_unit_vector_size * comp.ss.x) == 0);
            assert(scan_component.data_size.y % (data_unit_vector_size * comp.ss.y) == 0);
        }

        scan.num_mcus.x = ceiling_div(
            scan_component.data_size.x, static_cast<unsigned int>(scan_component.mcu_size.x));
        scan.num_mcus.y = ceiling_div(
            scan_component.data_size.y, static_cast<unsigned int>(scan_component.mcu_size.y));

        scan.num_data_units_in_mcu += comp.ss.x * comp.ss.y;
        comp.max_data_size.x = std::max(comp.max_data_size.x, scan_component.data_size.x);
        comp.max_data_size.y = std::max(comp.max_data_size.y, scan_component.data_size.y);
    }

    if (10 < scan.num_data_units_in_mcu) {
        // B.2.3 Sum of product of horizontal and vertical sampling factors shall be leq 10.
        logger.log("\ttoo many data units in mcu\n");
        return JPEGGPU_INVALID_JPEG;
    }

    if (!has_remaining(3)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const bool scan_is_sequential = is_sequential(jpeg_stream.sof_marker);

    const uint8_t spectral_start = read_uint8();
    const uint8_t spectral_end   = read_uint8();
    if (scan_is_sequential) {
        if (spectral_start != 0 || spectral_end != 63) return JPEGGPU_INVALID_JPEG;
    } else {
        if (spectral_start > 63 || spectral_end < spectral_start) {
            return JPEGGPU_INVALID_JPEG; // values out of bounds
        }
        if (spectral_start == 0 && spectral_end != 0) {
            return JPEGGPU_INVALID_JPEG; // mixing DC and AC
        }
    }
    scan.spectral_start = spectral_start;
    scan.spectral_end   = spectral_end;

    const uint8_t successive_approximation = read_uint8();
    scan.successive_approx_hi              = successive_approximation >> 4;
    scan.successive_approx_lo              = successive_approximation & 0xf;
    if (scan_is_sequential) {
        if (scan.successive_approx_hi != 0 || scan.successive_approx_lo != 0) {
            return JPEGGPU_INVALID_JPEG;
        }
    } else {
        if (scan.successive_approx_hi > 13 || scan.successive_approx_lo > 13) {
            return JPEGGPU_INVALID_JPEG;
        }
    }
    logger.log(
        "\tss: %d, se: %d, ah: %d, al: %d\n",
        scan.spectral_start,
        scan.spectral_end,
        scan.successive_approx_hi,
        scan.successive_approx_lo);

    if (scan_is_sequential) {
        scan.type = scan_type::sequential;
    } else {
        const bool is_dc = spectral_start == 0 && spectral_end == 0;
        if (is_dc) {
            if (scan.successive_approx_hi == 0) {
                scan.type = scan_type::progressive_dc_initial;
            } else {
                scan.type = scan_type::progressive_dc_refinement;
            }
        } else {
            if (scan.successive_approx_hi == 0) {
                scan.type = scan_type::progressive_ac_initial;
            } else {
                scan.type = scan_type::progressive_ac_refinement;
            }
        }
    }

    if ((scan.type == scan_type::progressive_ac_initial ||
         scan.type == scan_type::progressive_ac_refinement) &&
        num_components != 1) {
        // G.1.1.1.1
        logger.log("\tIn progressive DCT, only DC scans may have interleaved components\n");
        return JPEGGPU_INVALID_JPEG;
    }

    for (int c = 0; c < num_components; ++c) {
        const scan_component& scan_component = scan.scan_components[c];
        if (scan.spectral_start == 0 && scan_component.dc_idx == reader_state::idx_not_defined) {
            logger.log("\tDC Huffman table not defined\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (scan.spectral_end > 1 && scan_component.ac_idx == reader_state::idx_not_defined) {
            logger.log("\tAC Huffman table not defined\n");
            return JPEGGPU_INVALID_JPEG;
        }
    }

    // Now comes the encoded data: skip through and keep track of segments.

    assert(scan_idx + 1 == jpeg_stream.num_scans);
    JPEGGPU_CHECK_STAT(nothrow_resize(h_scan_segments, jpeg_stream.num_scans));
    std::vector<segment, pinned_allocator<segment>>& h_segments = h_scan_segments[scan_idx];

    // TODO catch exception or calculate size beforehand and not use vector
    std::vector<segment> segments;
    const int scan_begin     = reader_state.image - reader_state.image_begin;
    int num_bytes_in_segment = 0;
    do {
        const uint8_t* ret = reinterpret_cast<const uint8_t*>(std::memchr(
            reinterpret_cast<const void*>(reader_state.image),
            0xff,
            reader_state.image_end - reader_state.image));
        if (ret == nullptr) {
            // file does not have an end of image marker
            return JPEGGPU_INVALID_JPEG;
        }

        // count bytes without marker
        num_bytes_in_segment += ret - reader_state.image;

        reader_state.image   = ret + 1; // skip to after 0xff
        const uint8_t marker = read_uint8();

        // `image` now points to after marker
        if (marker == 0) {
            // stuffed byte: 0xff00 is replaced by 0x00, so one additional byte
            ++num_bytes_in_segment;
            continue;
        }

        const bool is_rst = jpeggpu::MARKER_RST0 <= marker && marker <= jpeggpu::MARKER_RST7;
        // Not a restart marker, so end of scan. Rewind 0xff and marker byte
        if (!is_rst) reader_state.image -= 2;

        const int num_subsequences =
            ceiling_div(num_bytes_in_segment, static_cast<unsigned int>(subsequence_size_bytes));
        segments.push_back({scan.num_subsequences, num_subsequences});
        scan.num_subsequences += num_subsequences;
        num_bytes_in_segment = 0;
        ++scan.num_segments;

        if (is_rst) continue;
        break;
    } while (reader_state.image < reader_state.image_end);
    scan.begin = scan_begin;
    scan.end   = reader_state.image - reader_state.image_begin;

    // FIXME this is not yet implemented
    assert(scan.num_segments == 1 || scan.type != scan_type::progressive_ac_refinement);

    JPEGGPU_CHECK_STAT(nothrow_resize(h_segments, scan.num_segments));
    std::memcpy(h_segments.data(), segments.data(), scan.num_segments * sizeof(segment));

    return JPEGGPU_SUCCESS;
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_dqt(logger& logger)
{
    if (!has_remaining(2)) {
        logger.log("\ttoo few bytes in DQT segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        logger.log("\ttoo few bytes in DQT segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    int remaining = length;
    while (remaining > 0) {
        const uint8_t info = read_uint8();
        --remaining;
        const int precision = info >> 4;
        const int id        = info & 0xf;

        if ((precision != 0 && precision != 1) || id > 3) {
            logger.log("\tinvalid precision or id value\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (precision != 0) {
            logger.log("\t16-bit quantization table is not supported\n");
            return JPEGGPU_NOT_SUPPORTED;
        }

        qtable& table = h_qtables[id];
        for (int j = 0; j < 64; ++j) {
            // element in zigzag order
            const uint8_t element = read_uint8();

            // store in natural order
            table.data[order_natural[j]] = element;
        }
        remaining -= 64;
    }

    return JPEGGPU_SUCCESS;
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_dri(logger& logger)
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t rsti         = read_uint16();
    const bool seen_rsti_before = jpeg_stream.restart_interval != 0;
    if (seen_rsti_before && jpeg_stream.restart_interval != rsti) {
        // TODO is this even a problem?
        // do not support redefinining restart interval
        logger.log("\tredefined restart interval\n");
        return JPEGGPU_NOT_SUPPORTED;
    }
    jpeg_stream.restart_interval = rsti;
    logger.log("\trestart_interval: %" PRIu16 "\n", jpeg_stream.restart_interval);

    return JPEGGPU_SUCCESS;
}

[[nodiscard]] jpeggpu_status jpeggpu::reader::skip_segment(logger& logger)
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    logger.log("\twarning: skipping this segment\n");

    reader_state.image += length;
    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read(logger& logger)
{
    uint8_t marker_soi{};
    JPEGGPU_CHECK_STAT(read_marker(marker_soi, logger));
    logger.log("%s\n", jpeggpu::get_marker_string(marker_soi));
    if (marker_soi != jpeggpu::MARKER_SOI) {
        return JPEGGPU_INVALID_JPEG;
    }

    uint8_t marker{};
    do {
        JPEGGPU_CHECK_STAT(read_marker(marker, logger));
        logger.log("%s\n", get_marker_string(marker));
        switch (marker) {
        case jpeggpu::MARKER_SOF0:
        case jpeggpu::MARKER_SOF1:
        case jpeggpu::MARKER_SOF2:
            if (reader_state.found_sof) {
                return JPEGGPU_INVALID_JPEG;
            }
            reader_state.found_sof = true;
            jpeg_stream.sof_marker = marker;
            JPEGGPU_CHECK_STAT(read_sof(logger));
            continue;
        case jpeggpu::MARKER_SOF3:
        case jpeggpu::MARKER_SOF5:
        case jpeggpu::MARKER_SOF6:
        case jpeggpu::MARKER_SOF7:
        case jpeggpu::MARKER_SOF9:
        case jpeggpu::MARKER_SOF10:
        case jpeggpu::MARKER_SOF11:
        case jpeggpu::MARKER_SOF13:
        case jpeggpu::MARKER_SOF14:
        case jpeggpu::MARKER_SOF15:
            logger.log("\tunsupported JPEG type: %s\n", get_marker_string(marker));
            return JPEGGPU_NOT_SUPPORTED;
        case jpeggpu::MARKER_DHT:
            JPEGGPU_CHECK_STAT(read_dht(logger));
            continue;
        case jpeggpu::MARKER_EOI:
            break; // nothing to skip
        case jpeggpu::MARKER_SOS:
            JPEGGPU_CHECK_STAT(read_sos(logger));
            continue;
        case jpeggpu::MARKER_DQT:
            JPEGGPU_CHECK_STAT(read_dqt(logger));
            continue;
        case jpeggpu::MARKER_DRI:
            JPEGGPU_CHECK_STAT(read_dri(logger));
            continue;
        default: // FIXME not all segments may be skippable, does not account for unknown markers
            JPEGGPU_CHECK_STAT(skip_segment(logger));
            continue;
        }
    } while (marker != jpeggpu::MARKER_EOI);

    // TODO check that all qtables are found

    // Check that all components are found
    if (!reader_state.found_sof) return JPEGGPU_INVALID_JPEG;
    if (is_sequential(jpeg_stream.sof_marker)) {
        // Each component can only be defined once
        bool comp_found[max_comp_count] = {};
        for (int s = 0; s < jpeg_stream.num_scans; ++s) {
            const scan& scan = jpeg_stream.scans[s];
            for (int c = 0; c < scan.num_components; ++c) {
                const int comp_idx = scan.scan_components[c].component_idx;
                if (comp_found[comp_idx]) {
                    logger.log("\tredefined component with index %d in scan\n", comp_idx);
                    return JPEGGPU_INVALID_JPEG;
                }
                comp_found[comp_idx] = true;
            }
        }
        for (int c = 0; c < jpeg_stream.num_components; ++c) {
            if (!comp_found[c]) {
                logger.log("\tcomponent with index %d not defined in scan\n", c);
                return JPEGGPU_INVALID_JPEG;
            }
        }
    } else {
        // check that each component at least has an initial value
        uint64_t have_initial[max_comp_count] = {};
        for (int s = 0; s < static_cast<int>(jpeg_stream.scans.size()); ++s) {
            const scan& scan         = jpeg_stream.scans[s];
            const int num_coefs      = scan.spectral_end + 1 - scan.spectral_start;
            const uint64_t scan_mask = ((uint64_t{1} << num_coefs) - 1) << scan.spectral_start;
            for (int c = 0; c < scan.num_components; ++c) {
                const scan_component& scan_comp = scan.scan_components[c];
                have_initial[scan_comp.component_idx] |= scan_mask;
            }
        }
        for (int c = 0; c < jpeg_stream.num_components; ++c) {
            if (have_initial[c] != std::numeric_limits<uint64_t>::max()) {
                // TODO is it ever specified that this must be the case?
                logger.log("\tcomponent %d does not have a definition for each coefficient\n", c);
                return JPEGGPU_INVALID_JPEG;
            }
        }
        // group ac refinement scans into passes
        for (int s = 0; s < static_cast<int>(jpeg_stream.scans.size()); ++s) {
            const scan& scan = jpeg_stream.scans[s];
            if (scan.type != scan_type::progressive_ac_initial &&
                scan.type != scan_type::progressive_ac_refinement) {
                continue;
            }
            assert(scan.num_components == 1);
            const int comp_idx       = scan.scan_components[0].component_idx;
            const int num_coefs      = scan.spectral_end + 1 - scan.spectral_start;
            const uint64_t scan_mask = ((uint64_t{1} << num_coefs) - 1) << scan.spectral_start;
            // find a scan pass it can fit in
            bool found_pass = false;
            for (int i = 0; i < static_cast<int>(jpeg_stream.ac_scan_passes.size()); ++i) {
                ac_scan_pass& scan_pass       = jpeg_stream.ac_scan_passes[i];
                const bool scan_overlaps_pass = scan_pass.mask[comp_idx] & scan_mask;
                if (scan.type == scan_pass.type && !scan_overlaps_pass) {
                    scan_pass.scan_indices[scan_pass.num_scans++] = s;
                    scan_pass.mask[comp_idx] |= scan_mask;
                    found_pass = true;
                    break;
                }
            }
            if (!found_pass) {
                JPEGGPU_CHECK_STAT(nothrow_resize(
                    jpeg_stream.ac_scan_passes, jpeg_stream.ac_scan_passes.size() + 1));
                ac_scan_pass& scan_pass = jpeg_stream.ac_scan_passes.back();
                std::memset(scan_pass.scan_indices, 0, sizeof(ac_scan_pass));
                scan_pass.scan_indices[scan_pass.num_scans++] = s;
                scan_pass.mask[comp_idx]                      = scan_mask;
                scan_pass.type                                = scan.type;
            }
        }
        for (int i = 0; i < static_cast<int>(jpeg_stream.ac_scan_passes.size()); ++i) {
            {
                const ac_scan_pass& scan_pass = jpeg_stream.ac_scan_passes[i];
                logger.log("scan pass: ");
                for (int c = 0; c < jpeg_stream.num_components; ++c) {
                    logger.log("%" PRIx64, scan_pass.mask[c]);
                    if (c + 1 < jpeg_stream.num_components) logger.log(" ");
                    else logger.log("\n");
                }
            }
        }
    }

    return JPEGGPU_SUCCESS;
}

void jpeggpu::reader::reset(const uint8_t* image, const uint8_t* image_end)
{
    // TODO make these separate functions?

    // clear and reset reader state
    std::memset(&reader_state, 0, sizeof(reader_state));
    reader_state.image       = image;
    reader_state.image_begin = image;
    reader_state.image_end   = image_end;
    for (int i = 0; i < max_comp_count; ++i) {
        reader_state.curr_huff_dc[i] = reader_state::idx_not_defined;
        reader_state.curr_huff_ac[i] = reader_state::idx_not_defined;
    }

    // clear jpeg stream
    jpeg_stream.scans.clear();
    jpeg_stream.ac_scan_passes.clear();
    jpeg_stream.size           = ivec2{0, 0};
    jpeg_stream.num_components = 0;
    jpeg_stream.ss_max         = ivec2{0, 0};
    for (int c = 0; c < max_comp_count; ++c) {
        jpeg_stream.components[c].id            = 0;
        jpeg_stream.components[c].qtable_idx    = 0;
        jpeg_stream.components[c].size          = ivec2{0, 0};
        jpeg_stream.components[c].ss            = ivec2{0, 0};
        jpeg_stream.components[c].max_data_size = ivec2{0, 0};
    }
    jpeg_stream.restart_interval = 0;
    jpeg_stream.num_scans        = 0;
    jpeg_stream.cnt_huff         = 0;
    jpeg_stream.sof_marker       = 0;

    // clear remaining state
    std::memset(h_qtables.data(), 0, h_qtables.size() * sizeof(decltype(h_qtables)::value_type));
    h_huff_tables.clear();
    for (int i = 0; i < static_cast<int>(h_scan_segments.size()); ++i) {
        h_scan_segments[i].clear();
    }
    h_scan_segments.clear();
}
