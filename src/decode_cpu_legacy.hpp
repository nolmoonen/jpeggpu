#ifndef JPEGGPU_DECODE_CPU_LEGACY_HPP_
#define JPEGGPU_DECODE_CPU_LEGACY_HPP_

#include "defs.hpp"
#include "marker.hpp"

#include <jpeggpu/jpeggpu.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdint.h>

namespace {

struct coder_state {
    int buff;
    int bits;
    int bits_until_byte;
    const uint8_t* image;
    size_t image_size;
    int dc[jpeggpu::max_comp_count];
};

/**
 * Fill more bit to current get buffer
 *
 * @param coder
 * @return void
 */
void huffman_cpu_decoder_decode_fill_bit_buffer(coder_state& state)
{
    while (state.bits < 25) {
        // Are there some data?
        if (state.image_size > 0) {
            // Attempt to read a byte
            // printf("read byte %X 0x%X\n", (int)coder->data, (unsigned
            // char)*coder->data);
            unsigned char uc = *state.image++;
            state.image_size--;

            // If it's 0xFF, check and discard stuffed zero byte
            if (uc == 0xFF) {
                do {
                    // printf("read byte %X 0x%X\n", (int)coder->data, (unsigned
                    // char)*coder->data);
                    assert(state.image_size > 0); // TODO safety
                    uc = *state.image++;
                    state.image_size--;
                } while (uc == 0xFF);

                if (uc == 0) {
                    // Found FF/00, which represents an FF data byte
                    uc = 0xFF;
                } else if (jpeggpu::MARKER_RST0 <= uc && uc <= jpeggpu::MARKER_RST7) {
                    DBG_PRINT("marker?? %s\n", jpeggpu::get_marker_string(uc));

                    // just get the next byte
                    // TODO there may not be a next byte if mcu_count % rsti == 0
                    assert(state.image_size > 0); // TODO safety
                    uc = *state.image++;
                    state.image_size--;
                } else {
                    // TODO oops, too far
                    state.image_size += 2;

                    // FIXME enforce that this function is not called again
                    //   after encountering any unexpected marker (this case)

                    // There should be enough bits still left in the data segment;
                    // if so, just break out of the outer while loop.
                    // if (m_nGetBits >= nbits)
                    if (state.bits >= 0) break;
                }
            }

            state.buff = (state.buff << 8) | ((int)uc);
            state.bits += 8;
        } else {
            break;
        }
    }
}

/**
 * Get bits
 *
 * @param coder  Decoder structure
 * @param nbits  Number of bits to get
 * @return bits
 */
static inline int huffman_cpu_decoder_get_bits(coder_state& state, int nbits)
{
    // we should read nbits bits to get next data
    if (state.bits < nbits) {
        huffman_cpu_decoder_decode_fill_bit_buffer(state);
    }
    state.bits -= nbits;
    state.bits_until_byte -= nbits;
    if (state.bits_until_byte < 0) {
        state.bits_until_byte += 8;
    }
    return (int)(state.buff >> state.bits) & ((1 << nbits) - 1);
}

/**
 * Special Huffman decode:
 * (1) For codes with length > 8
 * (2) For codes with length < 8 while data is finished
 *
 * @return int
 */
int huffman_cpu_decoder_decode_special_decode(
    coder_state& state, const jpeggpu::huffman_table& table, int min_bits)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = huffman_cpu_decoder_get_bits(state, min_bits);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while (code > table.maxcode[l]) {
        code <<= 1;
        code |= huffman_cpu_decoder_get_bits(state, 1);
        l++;
    }

    // With garbage input we may reach the sentinel value l = 17.
    if (l > 16) {
        // Fake a zero as the safest result
        return 0;
    }

    return table.huffval[table.valptr[l] + (int)(code - table.mincode[l])];
}

/**
 * To find dc or ac value according to category and category offset
 *
 * @return int
 */
static inline int huffman_cpu_decoder_value_from_category(int category, int offset)
{
    // Method 1:
    // On some machines, a shift and add will be faster than a table lookup.
    // #define HUFF_EXTEND(x,s) ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) :
    // (x))

    // Method 2: Table lookup
    // If (offset < half[category]), then value is below zero
    // Otherwise, value is above zero, and just the offset
    // entry n is 2**(n-1)
    static const int half[16] = {
        0x0000,
        0x0001,
        0x0002,
        0x0004,
        0x0008,
        0x0010,
        0x0020,
        0x0040,
        0x0080,
        0x0100,
        0x0200,
        0x0400,
        0x0800,
        0x1000,
        0x2000,
        0x4000};

#if defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-negative-value"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif // defined __GNUC_
    // start[i] is the starting value in this category; surely it is below zero
    //  entry n is (-1 << n) + 1
    static const int start[16] = {
        0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767};
#if defined __GNUC__
#pragma GCC diagnostic pop
#endif // defined __GNUC_

    return (offset < half[category]) ? (offset + start[category]) : offset;
}

/**
 * Get category number for dc, or (0 run length, ac category) for ac.
 * The max length for Huffman codes is 15 bits; so we use 32 bits buffer
 * m_nGetBuff, with the validated length is m_nGetBits.
 * Usually, more than 95% of the Huffman codes will be 8 or fewer bits long
 * To speed up, we should pay more attention on the codes whose length <= 8
 *
 * @return int
 */
static inline int huffman_cpu_decoder_get_category(
    coder_state& state, const jpeggpu::huffman_table& table)
{
    // If left bits < 8, we should get more data
    if (state.bits < 8) {
        huffman_cpu_decoder_decode_fill_bit_buffer(state);
    }

    // Call special process if data finished; min bits is 1
    if (state.bits < 8) {
        return huffman_cpu_decoder_decode_special_decode(state, table, 1);
    }

    // Peek the first valid byte
    int look = ((state.buff >> (state.bits - 8)) & 0xFF);
    int nb   = table.look_nbits[look];

    if (nb) {
        state.bits -= nb;
        state.bits_until_byte -= nb;
        if (state.bits_until_byte < 0) {
            state.bits_until_byte += 8;
        }
        return table.look_sym[look];
    } else {
        // Decode long codes with length >= 9
        return huffman_cpu_decoder_decode_special_decode(state, table, 9);
    }
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int decode_block(
    int16_t* dst,
    const jpeggpu::huffman_table& table_dc,
    const jpeggpu::huffman_table& table_ac,
    coder_state& state,
    int& dc)
{
    // Zero block output
    std::memset(dst, 0, sizeof(int16_t) * jpeggpu::block_size * jpeggpu::block_size);

    // Section F.2.2.1: decode the DC coefficient difference
    // get dc category number, s
    int s = huffman_cpu_decoder_get_category(state, table_dc);
    if (s) {
        // Get offset in this dc category
        int r = huffman_cpu_decoder_get_bits(state, s);
        // Get dc difference value
        s     = huffman_cpu_decoder_value_from_category(s, r);
    }

    // Convert DC difference to actual value, update last_dc_val
    s += dc;
    dc = s;

    // Output the DC coefficient (assumes natural_order[0] = 0)
    dst[0] = s;

    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    for (int k = 1; k < 64; k++) {
        // s: (run, category)
        int s = huffman_cpu_decoder_get_category(state, table_ac);
        // r: run length for ac zero, 0 <= r < 16
        int r = s >> 4;
        // s: category for this non-zero ac
        s &= 15;
        if (s) {
            //    k: position for next non-zero ac
            k += r;
            //    r: offset in this ac category
            r = huffman_cpu_decoder_get_bits(state, s);
            //    s: ac value
            s = huffman_cpu_decoder_value_from_category(s, r);

            dst[jpeggpu::order_natural[k]] = s;
        } else {
            // s = 0, means ac value is 0 ? Only if r = 15.
            // means all the left ac are zero
            if (r != 15) break;
            k += 15;
        }
    }

    // printf("CPU Decode Block ");
    // for (int y = 0; y < 8; y++) {
    //   for (int x = 0; x < 8; x++) {
    //     printf("%4d ", dst[y * 8 + x]);
    //   }
    // }
    // printf("\n");

    return 0;
}

} // namespace

inline jpeggpu_status process_scan_legacy(jpeggpu::reader& reader)
{
    coder_state state            = {};
    state.image                  = reader.scan_start;
    const size_t image_remaining = reader.scan_size;
    state.image_size             = image_remaining;

    if (reader.is_interleaved) {
        int mcu_count = 0;
        for (int y_mcu = 0; y_mcu < reader.num_mcus_y[0]; ++y_mcu) {
            for (int x_mcu = 0; x_mcu < reader.num_mcus_x[0]; ++x_mcu) {
                // one MCU
                for (int i = 0; i < reader.num_components; ++i) {
                    const jpeggpu::huffman_table& table_dc =
                        reader.huff_tables[reader.huff_map[i][jpeggpu::HUFF_DC]][jpeggpu::HUFF_DC];
                    const jpeggpu::huffman_table& table_ac =
                        reader.huff_tables[reader.huff_map[i][jpeggpu::HUFF_AC]][jpeggpu::HUFF_AC];
                    for (int y_ss = 0; y_ss < reader.ss_y[i]; ++y_ss) {
                        for (int x_ss = 0; x_ss < reader.ss_x[i]; ++x_ss) {
                            const int y_block = y_mcu * reader.ss_y[i] + y_ss;
                            const int x_block = x_mcu * reader.ss_x[i] + x_ss;
                            const size_t idx  = y_block * jpeggpu::block_size *
                                                   reader.mcu_sizes_x[i] * reader.num_mcus_x[i] +
                                               x_block * jpeggpu::block_size * jpeggpu::block_size;
                            int16_t* dst = &reader.data[i][idx];
                            decode_block(dst, table_dc, table_ac, state, state.dc[i]);
                        }
                    }
                }
                mcu_count++;
                // FIXME what if restart_interval is not set?
                if (mcu_count % reader.restart_interval == 0) {
                    for (int c = 0; c < jpeggpu::max_comp_count; ++c) {
                        state.dc[c] = 0;
                    }
                    // discard bits until a byte is reached
                    assert(state.bits >= state.bits_until_byte);
                    huffman_cpu_decoder_get_bits(state, state.bits_until_byte);
                    assert(state.bits_until_byte == 0);
                }
            }
        }
    } else {
        return JPEGGPU_NOT_SUPPORTED; // TODO
    }

    return JPEGGPU_SUCCESS;
}

#endif // JPEGGPU_DECODE_CPU_LEGACY_HPP_