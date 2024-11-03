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

#include <jpeggpu/jpeggpu.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <stdint.h>
#include <vector>

using namespace jpeggpu;

jpeggpu_status reader::startup()
{
    JPEGGPU_CHECK_STAT(nothrow_resize(h_qtables, max_comp_count));
    for (int s = 0; s < max_baseline_scan_count; ++s) {
        JPEGGPU_CHECK_STAT(nothrow_reserve(h_huff_tables[s], max_baseline_huff_per_scan));
        JPEGGPU_CHECK_STAT(nothrow_reserve(h_scan_segments[s], 1024));
    }

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
    if (length < 2) {
        return JPEGGPU_INVALID_JPEG;
    }
    if (!has_remaining(length - 2)) {
        return JPEGGPU_INCOMPLETE_BITSTREAM;
    }

    const uint8_t precision = read_uint8();
    if (precision != 8) {
        logger.log("\tunsupported sample precision %d, only 8 is supported\n", int{precision});
        return JPEGGPU_NOT_SUPPORTED;
    }
    const uint16_t num_lines            = read_uint16();
    const uint16_t num_samples_per_line = read_uint16();
    if (num_lines == 0 || num_samples_per_line == 0) { // TODO spec allows y=0, why?
        logger.log("\tinvalid size x=%d, y=%d\n", jpeg_stream.size.x, jpeg_stream.size.y);
        return JPEGGPU_INVALID_JPEG;
    }
    jpeg_stream.size.x = num_samples_per_line;
    jpeg_stream.size.y = num_lines;

    const uint8_t num_components = read_uint8();
    if (num_components == 0) {
        logger.log("\tzero components\n");
        return JPEGGPU_INVALID_JPEG;
    }
    if (num_components > max_comp_count) {
        logger.log("\ttoo many components %d\n", jpeg_stream.num_components);
        return JPEGGPU_NOT_SUPPORTED;
    }
    jpeg_stream.num_components = num_components;

    if (!has_remaining(3 * num_components)) {
        return JPEGGPU_INCOMPLETE_BITSTREAM;
    }

    logger.log(
        "\tsize_x: %" PRIu16 ", size_y: %" PRIu16 ", num_components: %" PRIu8 "\n",
        jpeg_stream.size.x,
        jpeg_stream.size.y,
        jpeg_stream.num_components);

    jpeg_stream.ss_max = {0, 0};
    for (int c = 0; c < num_components; ++c) {
        component& comp                = jpeg_stream.components[c];
        const uint8_t component_id     = read_uint8();
        comp.id                        = component_id;
        const uint8_t sampling_factors = read_uint8();
        const int ss_x_c               = sampling_factors >> 4;
        if (ss_x_c < 1 || ss_x_c > 4) {
            logger.log("\tinvalid horizontal subsampling factor (%d)\n", ss_x_c);
            return JPEGGPU_INVALID_JPEG;
        }
        const int ss_y_c = sampling_factors & 0xf;
        if (ss_y_c < 1 || ss_y_c > 4) {
            logger.log("\tinvalid vertical subsampling factor (%d)\n", ss_y_c);
            return JPEGGPU_INVALID_JPEG;
        }

        if (num_components == 1) {
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
        if (qi > 3) {
            logger.log("\tinvalid quantization table index (%d)\n", qi);
            return JPEGGPU_INVALID_JPEG;
        }
        comp.qtable_idx = qi;
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

        const int scan_idx                  = jpeg_stream.num_scans;
        const int huff_idx                  = th * HUFF_COUNT + table_class;
        huffman_table& table                = h_huff_tables[scan_idx][huff_idx];
        reader_state.huff_defined[huff_idx] = true;

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

    const uint16_t length = read_uint16();
    if (length < 3) {
        return JPEGGPU_INVALID_JPEG;
    }
    const uint8_t num_scan_components = read_uint8();
    if (num_scan_components < 1 || num_scan_components > 4) {
        logger.log("\tinvalid number of components in scan %d\n", num_scan_components);
        return JPEGGPU_INVALID_JPEG;
    }

    if (jpeg_stream.num_scans >= max_baseline_scan_count) {
        // means that at least one component is defined in two scans
        return JPEGGPU_INVALID_JPEG;
    }
    const int scan_idx       = jpeg_stream.num_scans++;
    scan& scan               = jpeg_stream.scans[scan_idx];
    scan.num_scan_components = num_scan_components;

    const uint16_t length_remaining = 2 * num_scan_components + 3;
    if (length - 2 - 1 != length_remaining) {
        return JPEGGPU_INVALID_JPEG;
    }
    if (!has_remaining(length_remaining)) {
        return JPEGGPU_INCOMPLETE_BITSTREAM;
    }

    scan.num_data_units_in_mcu = 0;
    for (int sc = 0; sc < num_scan_components; ++sc) {
        scan_component& scan_component = scan.scan_components[sc];

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
        if (sc > 0 && scan_component.component_idx <= scan.scan_components[sc - 1].component_idx) {
            logger.log("\tinvalid component order in scan\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (id_dc > 3 || id_ac > 3) {
            logger.log("\tHuffman table id out of bounds\n");
        }

        if (!reader_state.huff_defined[id_dc * HUFF_COUNT + HUFF_DC]) {
            return JPEGGPU_INVALID_JPEG;
        }
        scan_component.dc_idx = id_dc;

        if (!reader_state.huff_defined[id_ac * HUFF_COUNT + HUFF_AC]) {
            return JPEGGPU_INVALID_JPEG;
        }
        scan_component.ac_idx = id_ac;

        const component& comp = jpeg_stream.components[component_idx];

        if (!reader_state.qtable_defined[comp.qtable_idx]) {
            logger.log("\tquantization table at index %d not defined\n", comp.qtable_idx);
            return JPEGGPU_INVALID_JPEG;
        }

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
    }

    if (10 < scan.num_data_units_in_mcu) {
        // B.2.3 Sum of product of horizontal and vertical sampling factors shall be leq 10.
        logger.log("\ttoo many data units in mcu\n");
        return JPEGGPU_INVALID_JPEG;
    }

    [[maybe_unused]] const uint8_t spectral_start           = read_uint8();
    [[maybe_unused]] const uint8_t spectral_end             = read_uint8();
    [[maybe_unused]] const uint8_t successive_approximation = read_uint8();

    // If this scan is not the last, copy over the tables. This is to achieve fixed memory allocations
    const bool is_final_scan = scan_idx + 1 >= max_baseline_scan_count;
    if (!is_final_scan) {
        std::copy(
            h_huff_tables[scan_idx].begin(),
            h_huff_tables[scan_idx].end(),
            h_huff_tables[scan_idx + 1].begin());
    }

    // Now comes the encoded data: skip through and keep track of segments.

    // TODO reserve based on calculated size

    std::vector<segment, pinned_allocator<segment>>& h_segments = h_scan_segments[scan_idx];
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

        JPEGGPU_CHECK_STAT(
            nothrow_push_back(h_segments, {scan.num_subsequences, num_subsequences}));
        scan.num_subsequences += num_subsequences;
        num_bytes_in_segment = 0;
        ++scan.num_segments;

        if (is_rst) continue;
        break;
    } while (reader_state.image < reader_state.image_end);
    scan.begin = scan_begin;
    scan.end   = reader_state.image - reader_state.image_begin;

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

        reader_state.qtable_defined[id] = true;

        const bool comp_seen_in_scan = [](const struct jpeg_stream& jpeg_stream, int id) {
            for (int s = 0; s < jpeg_stream.num_scans; ++s) {
                const scan& scan = jpeg_stream.scans[s];
                for (int sc = 0; sc < scan.num_scan_components; ++sc) {
                    if (scan.scan_components[sc].component_idx == id) return true;
                }
            }
            return false;
        }(jpeg_stream, id);

        for (int j = 0; j < 64; ++j) {
            // element in zigzag order
            const uint8_t element = read_uint8();

            // store only if the component has not been seen in a scan to prevent
            //   overwriting tables needed later, and to use only a constant amount of memory
            if (!comp_seen_in_scan) {
                // store in natural order
                h_qtables[id].data[order_natural[j]] = element;
            }
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

    const uint16_t length = read_uint16();
    if (length < 2) {
        return JPEGGPU_INVALID_JPEG;
    }
    if (!has_remaining(length - 2)) {
        return JPEGGPU_INCOMPLETE_BITSTREAM;
    }

    logger.log("\twarning: skipping this segment\n");

    reader_state.image += length - 2;
    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read(logger& logger)
{
    uint8_t marker_soi{};
    JPEGGPU_CHECK_STAT(read_marker(marker_soi, logger));
    logger.log("marker %s\n", jpeggpu::get_marker_string(marker_soi));
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
            if (reader_state.found_sof) {
                return JPEGGPU_INVALID_JPEG;
            }
            reader_state.found_sof = true;
            JPEGGPU_CHECK_STAT(read_sof(logger));
            continue;
        case jpeggpu::MARKER_SOF2:
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

    // Check that all components are found
    if (!reader_state.found_sof) return JPEGGPU_INVALID_JPEG;
    // Each component can only be defined once
    bool comp_found[max_comp_count] = {};
    for (int s = 0; s < jpeg_stream.num_scans; ++s) {
        const scan& scan = jpeg_stream.scans[s];
        for (int a = 0; a < scan.num_scan_components; ++a) {
            const scan_component& scan_component = scan.scan_components[a];
            const int comp_idx                   = scan_component.component_idx;
            assert(!comp_found[comp_idx]); // error is returned in read_sos
            comp_found[comp_idx] = true;
        }
    }
    for (int c = 0; c < jpeg_stream.num_components; ++c) {
        if (!comp_found[c]) {
            logger.log("\tcomponent with index %d not defined in scan\n", c);
            return JPEGGPU_INVALID_JPEG;
        }
    }

    return JPEGGPU_SUCCESS;
}

void jpeggpu::reader::reset(const uint8_t* image, const uint8_t* image_end)
{
    // TODO make these separate functions?

    // clear jpeg stream
    jpeg_stream.num_scans = 0;
    for (int s = 0; s < max_baseline_scan_count; ++s) {
        scan& scan               = jpeg_stream.scans[s];
        scan.num_scan_components = 0;
        for (int a = 0; a < max_comp_count; ++a) {
            scan_component& scan_component = scan.scan_components[a];
            scan_component.dc_idx          = 0;
            scan_component.ac_idx          = 0;
            scan_component.component_idx   = 0;
            scan_component.mcu_size        = {0, 0};
            scan_component.data_size       = {0, 0};
        }
        scan.begin                 = 0;
        scan.end                   = 0;
        scan.num_data_units_in_mcu = 0;
        scan.num_subsequences      = 0;
        scan.num_segments          = 0;
        scan.num_mcus              = {0, 0};
        scan.num_huff_tables       = 0;
        std::memset(scan.huff_tables, 0, sizeof(scan.huff_tables));
    }
    jpeg_stream.size           = ivec2{0, 0};
    jpeg_stream.ss_max         = ivec2{0, 0};
    jpeg_stream.num_components = 0;
    for (int c = 0; c < max_comp_count; ++c) {
        jpeg_stream.components[c].id         = 0;
        jpeg_stream.components[c].qtable_idx = 0;
        jpeg_stream.components[c].size       = ivec2{0, 0};
        jpeg_stream.components[c].ss         = ivec2{0, 0};
    }
    jpeg_stream.restart_interval = 0;

    // clear and reset reader state
    reader_state.image       = image;
    reader_state.image_begin = image;
    reader_state.image_end   = image_end;
    reader_state.found_sof   = false;
    for (int i = 0; i < max_comp_count; ++i) {
        reader_state.qtable_defined[i] = false;
        reader_state.huff_defined[i]   = false;
    }

    // clear remaining state
    // clearing tables is not necessary but eases debugging if no previous data remains
    std::memset(h_qtables.data(), 0, h_qtables.size() * sizeof(qtable));
    for (int s = 0; s < max_baseline_scan_count; ++s) {
        std::memset(h_huff_tables[s].data(), 0, h_huff_tables[s].size() * sizeof(huffman_table));

        h_scan_segments[s].clear();
    }
}
