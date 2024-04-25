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

    for (int s = 0; s < max_scan_count; ++s) {
        h_segments[s] = nullptr;
    }

    return JPEGGPU_SUCCESS;
}

void reader::cleanup()
{
    for (int s = 0; s < max_scan_count; ++s) {
        if (h_segments[s]) {
            cudaFreeHost(h_segments[s]);
            h_segments[s] = nullptr;
        }
    }

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

[[nodiscard]] jpeggpu_status jpeggpu::reader::read_sof0(logger& logger)
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

    logger.log(
        "\tsize_x: %" PRIu16 ", size_y: %" PRIu16 ", num_components: %" PRIu8 "\n",
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
        comp.ss_x        = ss_x_c;
        const int ss_y_c = sampling_factors & 0xf;
        if (ss_y_c < 1 && ss_y_c > 4) {
            return JPEGGPU_INVALID_JPEG;
        }
        comp.ss_y        = ss_y_c;
        const uint8_t qi = read_uint8();
        comp.qtable_idx  = qi;
        logger.log(
            "\tc_id: %" PRIu8 ", ssx: %d, ssy: %d, qi: %" PRIu8 "\n",
            component_id,
            comp.ss_x,
            comp.ss_y,
            qi);
        jpeg_stream.ss_x_max = std::max(jpeg_stream.ss_x_max, comp.ss_x);
        jpeg_stream.ss_y_max = std::max(jpeg_stream.ss_y_max, comp.ss_y);
    }

    return JPEGGPU_SUCCESS;
}

void compute_huffman_table(jpeggpu::huffman_table& table, const uint8_t (&num_codes)[16])
{
    // generate Huffman
    int huffcode[257]; // [1, 16]
    int code_idx  = 0; // [0, 256]
    uint16_t code = 0;
    for (int l = 0; l < 16; ++l) {
        for (uint8_t i = 0; i < num_codes[l]; i++) {
            huffcode[code_idx++] = code;
            ++code;
        }
        code <<= 1;
    }
    huffcode[code_idx] = 0;

    // generate decoding tables for bit-sequential decoding
    code_idx = 0;
    for (int l = 0; l < 16; ++l) {
        if (num_codes[l]) {
            table.valptr[l]  = code_idx; // huffval[] index of 1st symbol of code length l
            table.mincode[l] = huffcode[code_idx]; // minimum code of length l
            code_idx += num_codes[l];
            table.maxcode[l] = huffcode[code_idx - 1]; // maximum code of length l
        } else {
            table.maxcode[l] = -1; // -1 if no codes of this length
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
        const int tc = index >> 4;
        const int th = index & 0xf;
        // TODO check if these are already defined. if so, throw warning
        if (tc != 0 && tc != 1) {
            logger.log("\tinvalid Huffman table index\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (th != 0 && th != 1) {
            logger.log("\tmore than two Huffman tables are not supported\n");
            return JPEGGPU_NOT_SUPPORTED;
        }

        if (!has_remaining(16)) {
            logger.log("\ttoo few bytes in DHT segment\n");
            return JPEGGPU_INVALID_JPEG;
        }

        logger.log("\t%s Huffman table index %d\n", tc == 0 ? "DC" : "AC", th);
        jpeggpu::huffman_table& table = *(h_huff_tables[th][tc]);

        /// num_codes[i] is # of symbols with codes of i + 1 bits
        uint8_t num_codes[16];
        int count = 0;
        for (int i = 0; i < 16; ++i) {
            num_codes[i] = read_uint8();
            count += num_codes[i];
        }
        remaining -= 16;

        if (!has_remaining(count)) {
            logger.log("\ttoo few bytes in DHT segment\n");
            return JPEGGPU_INVALID_JPEG;
        }

        if (static_cast<size_t>(count) > sizeof(table.huffval)) {
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
    if (!has_remaining(3)) {
        logger.log("\ttoo few bytes in SOS segment\n");
        return JPEGGPU_INVALID_JPEG;
    }

    const uint16_t length        = read_uint16();
    const uint8_t num_components = read_uint8();
    if (num_components < 1 || num_components > 4) {
        logger.log("\tinvalid number of components in scan\n");
        return JPEGGPU_INVALID_JPEG;
    }
    jpeg_stream.is_interleaved = num_components > 1;
    if (jpeg_stream.is_interleaved && num_components != jpeg_stream.num_components) {
        // FIXME this is a wrong assumption, this is allowed! see ISO/IEC 10918-1 4.10
        logger.log(
            "\tinvalid number of components in interleaved scan (%d) compared to stream (%d)\n",
            num_components,
            jpeg_stream.num_components);
        return JPEGGPU_INVALID_JPEG;
    }

    if (length != 2 + 1 + 2 * num_components + 3) {
        return JPEGGPU_INVALID_JPEG;
    }

    const int scan_idx = reader_state.scan_idx++;
    scan& scan         = jpeg_stream.scans[scan_idx];

    for (uint8_t c = 0; c < num_components; ++c) {
        if (!has_remaining(2)) {
            return JPEGGPU_INVALID_JPEG;
        }

        const uint8_t selector = read_uint8();
        scan.ids[c]            = selector;
        // check if selector matches the component index, since the frame header must have been seen
        //   once we encounter start of scan
        int comp_idx = -1;
        for (int d = 0; d < jpeg_stream.num_components; ++d) {
            if (jpeg_stream.components[d].id == selector) {
                comp_idx = d;
                break;
            }
        }
        if (comp_idx == -1) {
            logger.log(
                "scan component %" PRIu8 " does not match any frame components (" PRIu8 " " PRIu8
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
        logger.log("\tc_id: %" PRIu8 ", dc: %d, ac: %d\n", selector, id_dc, id_ac);
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
            const int num_subsequences = ceiling_div(
                num_bytes_in_segment, static_cast<unsigned int>(subsequence_size_bytes));
            segments.push_back({scan.num_subsequences, num_subsequences});
            scan.num_subsequences += num_subsequences;
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

        logger.log("unexpected marker \"%s\"\n", jpeggpu::get_marker_string(marker));
        return JPEGGPU_INVALID_JPEG;
    } while (reader_state.image < reader_state.image_end);
    scan.begin = scan_begin;
    scan.end   = reader_state.image - reader_state.image_begin;

    if (h_segments[scan_idx]) {
        return JPEGGPU_INTERNAL_ERROR;
    }
    const size_t segment_bytes = scan.num_segments * sizeof(segment);
    JPEGGPU_CHECK_CUDA(cudaMallocHost(&h_segments[scan_idx], segment_bytes));
    std::memcpy(h_segments[scan_idx], segments.data(), segment_bytes);

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
        // TODO warning if redefined
        if ((precision != 0 && precision != 1) || id > 3) {
            logger.log("\tinvalid precision or id value\n");
            return JPEGGPU_INVALID_JPEG;
        }
        if (precision != 0) {
            logger.log("\t16-bit quantization table is not supported\n");
            return JPEGGPU_NOT_SUPPORTED;
        }

        for (int j = 0; j < 64; ++j) {
            // element in zigzag order
            const uint8_t element = read_uint8();

            // store in natural order
            (*h_qtables[id])[jpeggpu::order_natural[j]] = element;
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

#define JPEGGPU_CHECK_STATUS(call)                                                                 \
    do {                                                                                           \
        jpeggpu_status stat = call;                                                                \
        if (stat != JPEGGPU_SUCCESS) {                                                             \
            return stat;                                                                           \
        }                                                                                          \
    } while (0)

jpeggpu_status jpeggpu::reader::read(logger& logger)
{
    uint8_t marker_soi{};
    JPEGGPU_CHECK_STATUS(read_marker(marker_soi, logger));
    logger.log("marker %s\n", jpeggpu::get_marker_string(marker_soi));
    if (marker_soi != jpeggpu::MARKER_SOI) {
        return JPEGGPU_INVALID_JPEG;
    }

    uint8_t marker{};
    do {
        JPEGGPU_CHECK_STATUS(read_marker(marker, logger));
        logger.log("marker %s\n", get_marker_string(marker));
        switch (marker) {
        case jpeggpu::MARKER_SOF0:
            JPEGGPU_CHECK_STATUS(read_sof0(logger));
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
            logger.log("\tunsupported JPEG type: %s\n", get_marker_string(marker));
            return JPEGGPU_NOT_SUPPORTED;
        case jpeggpu::MARKER_DHT:
            JPEGGPU_CHECK_STATUS(read_dht(logger));
            continue;
        case jpeggpu::MARKER_EOI:
            break; // nothing to skip
        case jpeggpu::MARKER_SOS:
            JPEGGPU_CHECK_STATUS(read_sos(logger));
            continue;
        case jpeggpu::MARKER_DQT:
            JPEGGPU_CHECK_STATUS(read_dqt(logger));
            continue;
        case jpeggpu::MARKER_DRI:
            JPEGGPU_CHECK_STATUS(read_dri(logger));
            continue;
        default:
            JPEGGPU_CHECK_STATUS(skip_segment(logger));
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

        // A.2.4 Completion of partial MCU
        assert(comp.size_x % data_unit_vector_size == 0);
        assert(comp.size_y % data_unit_vector_size == 0);
        assert(comp.size_x % (data_unit_vector_size * comp.ss_x) == 0);
        assert(comp.size_y % (data_unit_vector_size * comp.ss_y) == 0);

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

        const int num_mcus_x =
            ceiling_div(comp.data_size_x, static_cast<unsigned int>(comp.mcu_size_x));
        const int num_mcus_y =
            ceiling_div(comp.data_size_y, static_cast<unsigned int>(comp.mcu_size_y));
        if (c == 0) {
            jpeg_stream.num_mcus_x = num_mcus_x;
            jpeg_stream.num_mcus_y = num_mcus_y;
        } else {
            assert(jpeg_stream.num_mcus_x == num_mcus_x);
            assert(jpeg_stream.num_mcus_y == num_mcus_y);
        }

        jpeg_stream.num_data_units_in_mcu += comp.ss_x * comp.ss_y;
        jpeg_stream.total_data_size += comp.data_size_x * comp.data_size_y;
    }

    // TODO read metadata to determine color formats
    switch (jpeg_stream.num_components) {
    case 1:
        jpeg_stream.color_fmt = JPEGGPU_JPEG_GRAY;
        break;
    case 2:
        logger.log("\tcomponent count of two is not supported\n");
        return JPEGGPU_NOT_SUPPORTED;
    case 3:
        jpeg_stream.color_fmt = JPEGGPU_JPEG_YCBCR;
        break;
    case 4:
        jpeg_stream.color_fmt = JPEGGPU_JPEG_CMYK;
        break;
    default:
        return JPEGGPU_INTERNAL_ERROR;
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

    for (int s = 0; s < max_scan_count; ++s) {
        if (h_segments[s]) {
            cudaFreeHost(h_segments[s]);
            h_segments[s] = nullptr;
        }
    }
}
