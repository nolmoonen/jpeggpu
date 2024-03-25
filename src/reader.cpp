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

using namespace jpeggpu;

jpeggpu_status reader::startup()
{
    // TODO free earlier allocations if one fails

    for (int c = 0; c < max_comp_count; ++c) {
        if (cudaMallocHost(&h_qtables[c], data_unit_size * sizeof(*(h_qtables[c]))) !=
            cudaSuccess) {
            return JPEGGPU_OUT_OF_HOST_MEMORY;
        }
    }
    for (int i = 0; i < max_huffman_count; ++i) {
        for (int j = 0; j < HUFF_COUNT; ++j) {
            if (cudaMallocHost(&h_huff_tables[i][j], sizeof(*(h_huff_tables[i][j]))) !=
                cudaSuccess) {
                return JPEGGPU_OUT_OF_HOST_MEMORY;
            }
        }
    }

    return JPEGGPU_SUCCESS;
}

void reader::cleanup()
{
    for (int i = 0; i < max_huffman_count; ++i) {
        for (int j = 0; j < HUFF_COUNT; ++j) {
            cudaFreeHost(h_huff_tables[i][j]);
            h_huff_tables[i][j] = nullptr;
        }
    }
    for (int c = 0; c < max_comp_count; ++c) {
        cudaFreeHost(h_qtables[c]);
        h_qtables[c] = nullptr;
    }
}

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

jpeggpu_status jpeggpu::reader::read_marker(uint8_t& marker)
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t ff = read_uint8();
    if (ff != 0xff) {
        return JPEGGPU_INVALID_JPEG;
    }
    marker = read_uint8();
    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read_sof0()
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t precision = read_uint8();
    (void)precision;
    const uint16_t num_lines            = read_uint16();
    jpeg_stream.size_y                  = num_lines;
    const uint16_t num_samples_per_line = read_uint16();
    jpeg_stream.size_x                  = num_samples_per_line;
    const uint8_t num_img_components    = read_uint8();
    if (num_img_components > max_comp_count) {
        return JPEGGPU_INVALID_JPEG;
    }
    jpeg_stream.num_components = num_img_components;

    log("\tsize_x: %" PRIu16 ", size_y: %" PRIu16 ", num_components: %" PRIu8 "\n",
        jpeg_stream.size_x,
        jpeg_stream.size_y,
        jpeg_stream.num_components);

    jpeg_stream.ss_x_max = 0;
    jpeg_stream.ss_y_max = 0;
    for (uint8_t c = 0; c < jpeg_stream.num_components; ++c) {
        component& comp                = jpeg_stream.components[c];
        const uint8_t component_id     = read_uint8();
        comp.id                        = component_id;
        const uint8_t sampling_factors = read_uint8();
        const int ss_x_c               = sampling_factors >> 4;
        if (ss_x_c < 1 && ss_x_c > 4) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (ss_x_c == 3) {
            // fairly annoying to handle, and extremely uncommon
            return JPEGGPU_NOT_SUPPORTED;
        }
        comp.ss_x        = ss_x_c;
        const int ss_y_c = sampling_factors & 0xf;
        if (ss_y_c < 1 && ss_y_c > 4) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (ss_y_c == 3) {
            // fairly annoying to handle, and extremely uncommon
            return JPEGGPU_NOT_SUPPORTED;
        }
        comp.ss_y        = ss_y_c;
        const uint8_t qi = read_uint8();
        comp.qtable_idx  = qi;
        log("\tc_id: %" PRIu8 ", ssx: %d, ssy: %d, qi: %" PRIu8 "\n",
            component_id,
            comp.ss_x,
            comp.ss_y,
            qi);
        jpeg_stream.ss_x_max = std::max(jpeg_stream.ss_x_max, comp.ss_x);
        jpeg_stream.ss_y_max = std::max(jpeg_stream.ss_y_max, comp.ss_y);
    }

    return JPEGGPU_SUCCESS;
}

void compute_huffman_table(jpeggpu::huffman_table& table)
{
    // Figure C.1: make table of Huffman code length for each symbol
    // Note that this is in code-length order.
    char huffsize[257];
    int p = 0;
    for (int l = 1; l <= 16; l++) {
        for (int i = 1; i <= (int)table.bits[l]; i++)
            huffsize[p++] = (char)l;
    }
    huffsize[p] = 0;

    // Figure C.2: generate the codes themselves
    // Note that this is in code-length order.
    unsigned int huffcode[257];
    unsigned int code = 0;
    int si            = huffsize[0];
    p                 = 0;
    while (huffsize[p]) {
        while (((int)huffsize[p]) == si) {
            huffcode[p++] = code;
            code++;
        }
        code <<= 1;
        si++;
    }

    // Figure F.15: generate decoding tables for bit-sequential decoding
    p = 0;
    for (int l = 1; l <= 16; l++) {
        if (table.bits[l]) {
            table.valptr[l]  = p; // huffval[] index of 1st symbol of code length l
            table.mincode[l] = huffcode[p]; // minimum code of length l
            p += table.bits[l];
            table.maxcode[l] = huffcode[p - 1]; // maximum code of length l
        } else {
            table.maxcode[l] = -1; // -1 if no codes of this length
        }
    }
    // Ensures huff_decode terminates
    table.maxcode[17] = 0xFFFFFL;
}

jpeggpu_status jpeggpu::reader::read_dht()
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    int remaining = length;
    while (remaining > 0) {
        const uint8_t index = read_uint8();
        --remaining;
        const int tc = index >> 4;
        const int th = index & 0xf;
        // TODO check if these are already defined. if so, throw warning
        if (tc != 0 && tc != 1) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (th != 0 && th != 1) {
            return JPEGGPU_NOT_SUPPORTED;
        }

        if (!has_remaining(16)) {
            return JPEGGPU_INVALID_JPEG;
        }

        jpeggpu::huffman_table& table = *(h_huff_tables[th][tc]);

        // read bits
        table.bits[0] = 0;
        int count     = 0;
        for (int i = 0; i < 16; ++i) {
            const int idx   = i + 1;
            table.bits[idx] = read_uint8();
            count += table.bits[idx];
        }
        remaining -= 16;

        if (!has_remaining(count)) {
            return JPEGGPU_INVALID_JPEG;
        }

        if (static_cast<size_t>(count) > sizeof(table.huffval)) {
            return JPEGGPU_INVALID_JPEG;
        }

        // read huffval
        for (int i = 0; i < count; ++i) {
            table.huffval[i] = read_uint8();
        }
        remaining -= count;

        // TODO is this the correct place?
        compute_huffman_table(table);
    }

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read_sos()
{
    if (!has_remaining(3)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length        = read_uint16();
    const uint8_t num_components = read_uint8();
    // there should only be one scan if the scan is interleaved (1 < num_components)
    if (num_components > 4 || (1 < num_components && num_components < jpeg_stream.num_components)) {
        return JPEGGPU_INVALID_JPEG;
    }
    jpeg_stream.is_interleaved = num_components > 1;

    if (length != 2 + 1 + 2 * num_components + 3) {
        return JPEGGPU_INVALID_JPEG;
    }

    scan& scan = jpeg_stream.scans[reader_state.scan_idx++];

    for (uint8_t c = 0; c < num_components; ++c) {
        if (!has_remaining(2)) {
            return JPEGGPU_INVALID_JPEG;
        }

        const uint8_t selector = read_uint8();
        scan.ids[c]            = selector;
        // check if selector matches the component index, since the frame header must have been seen
        //   once we encounter start of scan
        int comp_idx           = -1;
        for (int d = 0; d < jpeg_stream.num_components; ++d) {
            if (jpeg_stream.components[d].id == selector) {
                comp_idx = d;
                break;
            }
        }
        if (comp_idx == -1) {
            log("scan component %" PRIu8 " does not match any frame components (" PRIu8 " " PRIu8
                " " PRIu8 " " PRIu8 ")\n",
                selector,
                jpeg_stream.components[0].id,
                jpeg_stream.components[1].id,
                jpeg_stream.components[2].id,
                jpeg_stream.components[3].id);
            return JPEGGPU_INVALID_JPEG;
        }

        const uint8_t acdc_selector = read_uint8();
        const uint8_t id_dc         = acdc_selector >> 4;
        const uint8_t id_ac         = acdc_selector & 0xf;
        if (id_dc > 3 || id_ac > 3) {
            return JPEGGPU_INVALID_JPEG;
        }
        log("\tc_id: %" PRIu8 ", dc: %d, ac: %d\n", selector, id_dc, id_ac);
        // TODO check if these Huffman indices are found
        jpeg_stream.components[comp_idx].dc_idx = id_dc;
        jpeg_stream.components[comp_idx].ac_idx = id_ac;
    }

    if (!has_remaining(3)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t spectral_start           = read_uint8();
    const uint8_t spectral_end             = read_uint8();
    const uint8_t successive_approximation = read_uint8();
    (void)spectral_start, (void)spectral_end, (void)successive_approximation;

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

        num_bytes_in_segment += ret - reader_state.image;

        reader_state.image   = ret + 1;
        const uint8_t marker = read_uint8();
        // `image` now points to after marker
        if (marker == 0) {
            // stuffed byte
            ++num_bytes_in_segment; // 0xff00 is replaced by 0x00
            continue;
        }

        const bool is_rst      = jpeggpu::MARKER_RST0 <= marker && marker <= jpeggpu::MARKER_RST7;
        const bool is_scan_end = marker == jpeggpu::MARKER_EOI || marker == jpeggpu::MARKER_SOS;

        if (is_rst || is_scan_end) {
            scan.num_subsequences += ceiling_div(
                num_bytes_in_segment, static_cast<unsigned int>(subsequence_size_bytes));
            num_bytes_in_segment = 0;
            ++scan.num_segments;
        }

        if (is_rst) {
            continue;
        }

        if (is_scan_end) {
            // rewind 0xff and marker byte
            reader_state.image -= 2;
            break;
        }

        log("marker %s\n", jpeggpu::get_marker_string(marker));
        log("unexpected\n");
        return JPEGGPU_INVALID_JPEG;
    } while (reader_state.image < reader_state.image_end);
    scan.begin = scan_begin;
    scan.end   = reader_state.image - reader_state.image_begin;

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read_dqt()
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    if (length % 65 != 0) {
        return JPEGGPU_INVALID_JPEG;
    }

    const int qtable_count = length / 65;

    for (int i = 0; i < qtable_count; ++i) {
        const uint8_t info  = read_uint8();
        const int precision = info >> 4;
        const int id        = info & 0xf;
        // TODO warning if redefined
        if ((precision != 0 && precision != 1) || id > 3) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (precision != 0) {
            return JPEGGPU_NOT_SUPPORTED;
        }

        for (int j = 0; j < 64; ++j) {
            // element in zigzag order
            const uint8_t element = read_uint8();

            // store in natural order
            (*h_qtables[id])[jpeggpu::order_natural[j]] = element;
        }
    }

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::read_dri()
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
        return JPEGGPU_NOT_SUPPORTED;
    }
    jpeg_stream.restart_interval = rsti;
    log("\trestart_interval: %" PRIu16 "\n", jpeg_stream.restart_interval);

    return JPEGGPU_SUCCESS;
}

jpeggpu_status jpeggpu::reader::skip_segment()
{
    if (!has_remaining(2)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length = read_uint16() - 2;
    if (!has_remaining(length)) {
        return JPEGGPU_INVALID_JPEG;
    }

    log("\twarning: skipping this segment\n");

    reader_state.image += length;
    return JPEGGPU_SUCCESS;
}

#define JPEGGPU_CHECK_STATUS(call)                                                                 \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return stat;                                                                           \
        }                                                                                          \
    } while (0)

jpeggpu_status jpeggpu::reader::read()
{
    uint8_t marker_soi{};
    JPEGGPU_CHECK_STATUS(read_marker(marker_soi));
    log("marker %s\n", jpeggpu::get_marker_string(marker_soi));
    if (marker_soi != jpeggpu::MARKER_SOI) {
        return JPEGGPU_INVALID_JPEG;
    }

    uint8_t marker{};
    do {
        JPEGGPU_CHECK_STATUS(read_marker(marker));
        log("marker %s\n", get_marker_string(marker));
        switch (marker) {
        case jpeggpu::MARKER_SOF0:
            read_sof0();
            continue;
        case jpeggpu::MARKER_SOF1:
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
            log("unsupported JPEG type %s\n", get_marker_string(marker));
            return JPEGGPU_NOT_SUPPORTED;
        case jpeggpu::MARKER_DHT:
            read_dht();
            continue;
        case jpeggpu::MARKER_EOI:
            break; // nothing to skip
        case jpeggpu::MARKER_SOS:
            read_sos();
            continue;
        case jpeggpu::MARKER_DQT:
            read_dqt();
            continue;
        case jpeggpu::MARKER_DRI:
            read_dri();
            continue;
        default:
            JPEGGPU_CHECK_STATUS(skip_segment());
            continue;
        }
    } while (marker != jpeggpu::MARKER_EOI);

    // TODO check that all found scans have all components

    // TODO check that all qtables are found

    // TODO check that all huffman tables are found

    jpeg_stream.num_scans = reader_state.scan_idx;

    jpeg_stream.num_data_units_in_mcu = 0;
    jpeg_stream.total_data_size       = 0;
    for (int c = 0; c < jpeg_stream.num_components; ++c) {
        component& comp = jpeg_stream.components[c];

        comp.size_x = get_size(jpeg_stream.size_x, comp.ss_x, jpeg_stream.ss_x_max);
        comp.size_y = get_size(jpeg_stream.size_y, comp.ss_y, jpeg_stream.ss_y_max);

        comp.mcu_size_x =
            jpeg_stream.is_interleaved ? data_unit_vector_size * comp.ss_x : data_unit_vector_size;
        comp.mcu_size_y =
            jpeg_stream.is_interleaved ? data_unit_vector_size * comp.ss_y : data_unit_vector_size;

        comp.data_size_x =
            ceiling_div(comp.size_x, static_cast<unsigned int>(comp.mcu_size_x)) * comp.mcu_size_x;
        comp.data_size_y =
            ceiling_div(comp.size_y, static_cast<unsigned int>(comp.mcu_size_y)) * comp.mcu_size_y;

        // TODO assumption 3: assumes first component is not subsampled
        if (c == 0) {
            jpeg_stream.num_mcus_x =
                ceiling_div(comp.data_size_x, static_cast<unsigned int>(comp.mcu_size_x));
            jpeg_stream.num_mcus_y =
                ceiling_div(comp.data_size_y, static_cast<unsigned int>(comp.mcu_size_y));
        } else {
            assert(
                ceiling_div(comp.data_size_x, static_cast<unsigned int>(comp.mcu_size_x)) ==
                static_cast<unsigned int>(jpeg_stream.num_mcus_x));
            assert(
                ceiling_div(comp.data_size_y, static_cast<unsigned int>(comp.mcu_size_y)) ==
                static_cast<unsigned int>(jpeg_stream.num_mcus_y));
        }

        jpeg_stream.num_data_units_in_mcu += comp.ss_x * comp.ss_y;
        jpeg_stream.total_data_size += comp.data_size_x * comp.data_size_y;
    }

    // TODO read metadata to determine color formats
    switch (jpeg_stream.num_components) {
    case 1:
        jpeg_stream.color_fmt = JPEGGPU_GRAY;
        jpeg_stream.pixel_fmt = JPEGGPU_P0;
        break;
    case 3:
        jpeg_stream.color_fmt = JPEGGPU_YCBCR;
        jpeg_stream.pixel_fmt = JPEGGPU_P0P1P2;
        break;
    case 4:
        jpeg_stream.color_fmt = JPEGGPU_CMYK;
        jpeg_stream.pixel_fmt = JPEGGPU_P0P1P2P3;
        break;
    default:
        return JPEGGPU_NOT_SUPPORTED;
    }

    return JPEGGPU_SUCCESS;
}

void jpeggpu::reader::reset(const uint8_t* image, const uint8_t* image_end)
{
    // clear and reset reader state
    std::memset(&reader_state, 0, sizeof(reader_state));
    reader_state.image       = image;
    reader_state.image_begin = image;
    reader_state.image_end   = image_end;

    // clear remaining state
    std::memset(&jpeg_stream, 0, sizeof(jpeg_stream));
    for (int c = 0; c < max_comp_count; ++c) {
        std::memset(h_qtables[c], 0, data_unit_size * sizeof(*(h_qtables[c])));
    }
    for (int i = 0; i < max_huffman_count; ++i) {
        for (int j = 0; j < HUFF_COUNT; ++j) {
            std::memset(h_huff_tables[i][j], 0, sizeof(*(h_huff_tables[i][j])));
        }
    }
}
