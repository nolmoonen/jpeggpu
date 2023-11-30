#include "reader.hpp"
#include "defs.hpp"
#include "marker.hpp"

#include <jpeggpu/jpeggpu.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <stdint.h>

uint8_t jpeggpu::reader::read_uint8() { return *(image++); }

uint16_t jpeggpu::reader::read_uint16()
{
    const uint8_t high = read_uint8();
    return static_cast<uint16_t>(high) << 8 | read_uint8();
}

bool jpeggpu::reader::has_remaining(int size) { return image_end - image >= size; }

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
    size_y                              = num_lines;
    const uint16_t num_samples_per_line = read_uint16();
    size_x                              = num_samples_per_line;
    const uint8_t num_img_components    = read_uint8();
    num_components                      = num_img_components;

    DBG_PRINT(
        "\tsize_x: %" PRIu16 ", size_y: %" PRIu16 ", num_components: %" PRIu8 "\n",
        size_x,
        size_y,
        num_components);

    ss_x_max = 0;
    ss_y_max = 0;
    for (uint8_t c = 0; c < num_components; ++c) {
        const uint8_t component_id     = read_uint8();
        const uint8_t sampling_factors = read_uint8();
        const int ss_x_c               = sampling_factors >> 4;
        ss_x[c]                        = ss_x_c;
        const int ss_y_c               = sampling_factors & 0xf;
        ss_y[c]                        = ss_y_c;
        const uint8_t qi               = read_uint8();
        qtable_idx[c]                  = qi;
        DBG_PRINT(
            "\tc_id: %" PRIu8 ", ssx: %d, ssy: %d, qi: %" PRIu8 "\n",
            component_id,
            ss_x[c],
            ss_y[c],
            qi);
        ss_x_max = std::max(ss_x_max, ss_x[c]);
        ss_y_max = std::max(ss_y_max, ss_y[c]);
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
            table.valptr[l]  = p;           // huffval[] index of 1st symbol of code length l
            table.mincode[l] = huffcode[p]; // minimum code of length l
            p += table.bits[l];
            table.maxcode[l] = huffcode[p - 1]; // maximum code of length l
        } else {
            table.maxcode[l] = -1; // -1 if no codes of this length
        }
    }
    // Ensures huff_decode terminates
    table.maxcode[17] = 0xFFFFFL;

    // Compute lookahead tables to speed up decoding.
    // First we set all the table entries to 0, indicating "too long";
    // then we iterate through the Huffman codes that are short enough and
    // fill in all the entries that correspond to bit sequences starting
    // with that code.
    std::memset(table.look_nbits, 0, sizeof(int) * 256);

    int HUFF_LOOKAHEAD = 8;
    p                  = 0;
    for (int l = 1; l <= HUFF_LOOKAHEAD; l++) {
        for (int i = 1; i <= (int)table.bits[l]; i++, p++) {
            // l = current code's length,
            // p = its index in huffcode[] & huffval[]. Generate left-justified
            // code followed by all possible bit sequences
            int lookbits = huffcode[p] << (HUFF_LOOKAHEAD - l);
            for (int ctr = 1 << (HUFF_LOOKAHEAD - l); ctr > 0; ctr--) {
                table.look_nbits[lookbits] = l;
                table.look_sym[lookbits]   = table.huffval[p];
                lookbits++;
            }
        }
    }
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
        if (tc != 0 && tc != 1) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (th != 0 && th != 1) {
            return JPEGGPU_NOT_SUPPORTED;
        }

        if (!has_remaining(16)) {
            return JPEGGPU_INVALID_JPEG;
        }

        jpeggpu::huffman_table& table = huff_tables[th][tc];

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
    if (length != 2 + 1 + 2 * num_components + 3) {
        return JPEGGPU_INVALID_JPEG;
    }

    for (uint8_t i = 0; i < num_components; ++i) {
        if (!has_remaining(2)) {
            return JPEGGPU_INVALID_JPEG;
        }

        const uint8_t selector      = read_uint8();
        const uint8_t acdc_selector = read_uint8();
        const int id_dc             = acdc_selector >> 4;
        const int id_ac             = acdc_selector & 0xf;
        DBG_PRINT("\tc_id: %" PRIu8 ", dc: %d, ac: %d\n", selector, id_dc, id_ac);
        huff_map[i][jpeggpu::HUFF_DC] = id_dc;
        huff_map[i][jpeggpu::HUFF_AC] = id_ac;
    }

    if (!has_remaining(3)) {
        return JPEGGPU_INVALID_JPEG;
    }

    const uint8_t spectral_start           = read_uint8();
    const uint8_t spectral_end             = read_uint8();
    const uint8_t successive_approximation = read_uint8();
    (void)spectral_start, (void)spectral_end, (void)successive_approximation;

    is_interleaved = num_components > 1;

    // TODO this is not compatible with non-interleaved, since there are multple scans
    assert(is_interleaved);
    for (int i = 0; i < num_components; ++i) {
        sizes_x[i] = get_size(size_x, ss_x[i], ss_x_max);
        sizes_y[i] = get_size(size_y, ss_y[i], ss_y_max);

        mcu_sizes_x[i] = is_interleaved ? block_size * ss_x[i] : block_size;
        mcu_sizes_y[i] = is_interleaved ? block_size * ss_y[i] : block_size;

        data_sizes_x[i] =
            ceiling_div(sizes_x[i], static_cast<unsigned int>(mcu_sizes_x[i])) * mcu_sizes_x[i];
        data_sizes_y[i] =
            ceiling_div(sizes_y[i], static_cast<unsigned int>(mcu_sizes_y[i])) * mcu_sizes_y[i];

        num_mcus_x[i] = ceiling_div(data_sizes_x[i], static_cast<unsigned int>(mcu_sizes_x[i]));
        num_mcus_y[i] = ceiling_div(data_sizes_y[i], static_cast<unsigned int>(mcu_sizes_y[i]));
    }

    // const size_t consumed = image_remaining - state.image_size;
    // *image += consumed;

    scan_start = image;
    do {
        const uint8_t* ret = reinterpret_cast<const uint8_t*>(
            std::memchr(reinterpret_cast<const void*>(image), 0xff, image_end - image));
        // ff as final char is valid since it's valid encoder symbol if it's escaped
        if (ret == nullptr || ret == image_end - 1) {
            // file is fully processed
            image = image_end;
            break;
        }
        image                = ret + 1;
        const uint8_t marker = read_uint8();
        if (marker == 0) {
            // escaped encoded 0, continue
            continue;
        }

        if (jpeggpu::MARKER_RST0 <= marker && marker <= jpeggpu::MARKER_RST7) {
            // restart marker is okay and part of scan
            DBG_PRINT("\trst marker\n");
        } else {
            // rewind the marker
            image -= 2;
            break;
        }
    } while (image < image_end);
    scan_size = image - scan_start;

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
        if ((precision != 0 && precision != 1) || id >= 4) {
            return JPEGGPU_INVALID_JPEG;
        }
        if (precision != 0) {
            return JPEGGPU_NOT_SUPPORTED;
        }

        for (int j = 0; j < 64; ++j) {
            // element in zigzag order
            const uint8_t element                  = read_uint8();
            // store in natural order
            qtables[id][jpeggpu::order_natural[j]] = element;
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

    const uint16_t rsti = read_uint16();
    restart_interval    = rsti;
    DBG_PRINT("\trestart_interval: %" PRIu16 "\n", restart_interval);

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

    DBG_PRINT("\twarning: skipping this segment\n");

    image += length;
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
    DBG_PRINT("marker %s\n", jpeggpu::get_marker_string(marker_soi));
    if (marker_soi != jpeggpu::MARKER_SOI) {
        return JPEGGPU_INVALID_JPEG;
    }

    uint8_t marker{};
    do {
        JPEGGPU_CHECK_STATUS(read_marker(marker));
        DBG_PRINT("marker %s\n", get_marker_string(marker));
        switch (marker) {
        case jpeggpu::MARKER_SOF0:
            read_sof0();
            continue;
        case jpeggpu::MARKER_DHT:
            read_dht();
            continue;
        case jpeggpu::MARKER_EOI:
            break;
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
        JPEGGPU_CHECK_STATUS(skip_segment());
    } while (marker != jpeggpu::MARKER_EOI);

    return JPEGGPU_SUCCESS;
}

#undef JPEGGPU_CHECK_STATUS