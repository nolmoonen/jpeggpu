#include "marker.hpp"
#include <jpeggpu/jpeggpu.hpp>

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdio.h> // only for DBG_PRINT
#include <type_traits>
#include <vector>

// https://arxiv.org/abs/2111.09219
// https://www.w3.org/Graphics/JPEG/itu-t81.pdf

#define DBG_PRINT(...) printf(__VA_ARGS__);

constexpr int block_size = 8;
/// components
constexpr int max_comp_count = 4;
/// huffman types
enum huff { HUFF_DC = 0, HUFF_AC = 1, HUFF_COUNT = 2 };

using qtable = uint8_t[64];

// clang-format off
constexpr std::array<int, 64 + 16> order_natural = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, // Extra entries for safety in decoder
    63, 63, 63, 63, 63, 63, 63, 63};
// clang-format on

template <
    typename T, typename U,
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value,
                     int> = 0>
inline constexpr auto ceiling_div(const T a, const U b) {
  return a / b + (a % b > 0 ? 1 : 0);
}

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

int get_size(int size, int ss, int ss_max) {
  return ceiling_div(size * ss, static_cast<unsigned int>(ss_max));
}

struct jpeggpu::decoder {
  int size_x;
  int size_y;
  int num_components;
  int ss_x[max_comp_count]; // ss as defined in spec
  int ss_y[max_comp_count];
  int ss_x_max;
  int ss_y_max;
  int qtable_idx[max_comp_count]; // the qtable for each comp
  // in natural order
  qtable qtables[max_comp_count];
  huffman_table huff_tables[max_comp_count][HUFF_COUNT];
  int huff_map[max_comp_count][HUFF_COUNT];

  // TODO AoS
  int sizes_x[max_comp_count];
  int sizes_y[max_comp_count];
  int mcu_sizes_x[max_comp_count];
  int mcu_sizes_y[max_comp_count];
  int data_sizes_x[max_comp_count];
  int data_sizes_y[max_comp_count];
  int num_mcus_x[max_comp_count];
  int num_mcus_y[max_comp_count];

  int16_t *data[max_comp_count];
  uint8_t *image[max_comp_count];
  bool is_interleaved;

  int restart_interval; // TODO "has_restart interval"
};

const char *jpeggpu::get_status_string(jpeggpu::status stat) {
  switch (stat) {
  case jpeggpu::status::success:
    return "success";
  case jpeggpu::status::error_marker_eof:
    return "error_marker_eof";
  case jpeggpu::status::error_marker_no_ff:
    return "error_marker_no_ff";
  case jpeggpu::status::error:
    return "error";
  case jpeggpu::status::not_supported:
    return "not supported";
  }
  return "unknown status";
}

namespace {

uint8_t read_uint8(const uint8_t **image) { return *((*image)++); }

uint16_t read_uint16(const uint8_t **image) {
  const uint8_t high = read_uint8(image);
  return static_cast<uint16_t>(high) << 8 | read_uint8(image);
}

jpeggpu::status read_marker(const uint8_t **image, const uint8_t *image_end,
                            uint8_t &marker) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error_marker_eof;
  }

  const uint8_t ff = read_uint8(image);
  if (ff != 0xff) {
    return jpeggpu::status::error_marker_no_ff;
  }
  marker = read_uint8(image);
  return jpeggpu::status::success;
}

jpeggpu::status read_sof0(const uint8_t **image, const uint8_t *image_end,
                          jpeggpu::decoder_t decoder) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image) - 2;
  if (image_end - *image < length) {
    return jpeggpu::status::error;
  }

  const uint8_t precision = read_uint8(image);
  (void)precision;
  const uint16_t size_y = read_uint16(image);
  decoder->size_y = size_y;
  const uint16_t size_x = read_uint16(image);
  decoder->size_x = size_x;
  const uint8_t num_components = read_uint8(image);
  decoder->num_components = num_components;

  DBG_PRINT("\tsize_x: %" PRIu16 ", size_y: %" PRIu16
            ", num_components: %" PRIu8 "\n",
            size_x, size_y, num_components);

  int ss_x_max = 0;
  int ss_y_max = 0;
  for (uint8_t i = 0; i < num_components; ++i) {
    const uint8_t component_id = read_uint8(image);
    const uint8_t sampling_factors = read_uint8(image);
    const int ss_x = sampling_factors >> 4;
    decoder->ss_x[i] = ss_x;
    const int ss_y = sampling_factors & 0xf;
    decoder->ss_y[i] = ss_y;
    const uint8_t qi = read_uint8(image);
    decoder->qtable_idx[i] = qi;
    DBG_PRINT("\tc_id: %" PRIu8 ", ssx: %d, ssy: %d, qi: %" PRIu8 "\n",
              component_id, ss_x, ss_y, qi);
    ss_x_max = std::max(ss_x_max, decoder->ss_x[i]);
    ss_y_max = std::max(ss_y_max, decoder->ss_y[i]);
  }
  decoder->ss_x_max = ss_x_max;
  decoder->ss_y_max = ss_y_max;

  return jpeggpu::status::success;
}

void compute_huffman_table(huffman_table &table) {
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
  int si = huffsize[0];
  p = 0;
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
      table.valptr[l] = p; // huffval[] index of 1st symbol of code length l
      table.mincode[l] = huffcode[p]; // minimum code of length l
      p += table.bits[l];
      table.maxcode[l] = huffcode[p - 1]; // maximum code of length l
    } else {
      table.maxcode[l] = -1; // -1 if no codes of this length
    }
  }
  // Ensures gpujpeg_huff_decode terminates
  table.maxcode[17] = 0xFFFFFL;

  // Compute lookahead tables to speed up decoding.
  // First we set all the table entries to 0, indicating "too long";
  // then we iterate through the Huffman codes that are short enough and
  // fill in all the entries that correspond to bit sequences starting
  // with that code.
  memset(table.look_nbits, 0, sizeof(int) * 256);

  int HUFF_LOOKAHEAD = 8;
  p = 0;
  for (int l = 1; l <= HUFF_LOOKAHEAD; l++) {
    for (int i = 1; i <= (int)table.bits[l]; i++, p++) {
      // l = current code's length,
      // p = its index in huffcode[] & huffval[]. Generate left-justified
      // code followed by all possible bit sequences
      int lookbits = huffcode[p] << (HUFF_LOOKAHEAD - l);
      for (int ctr = 1 << (HUFF_LOOKAHEAD - l); ctr > 0; ctr--) {
        table.look_nbits[lookbits] = l;
        table.look_sym[lookbits] = table.huffval[p];
        lookbits++;
      }
    }
  }
}

jpeggpu::status read_dht(const uint8_t **image, const uint8_t *image_end,
                         jpeggpu::decoder_t decoder) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image) - 2;
  if (image_end - *image < length) {
    return jpeggpu::status::error;
  }

  int remaining = length;
  while (remaining > 0) {
    const uint8_t index = read_uint8(image);
    --remaining;
    const int tc = index >> 4;
    const int th = index & 0xf;
    if (tc != 0 && tc != 1) {
      return jpeggpu::status::error;
    }
    if (th != 0 && th != 1) {
      return jpeggpu::status::not_supported;
    }

    if (image_end - *image < 16) {
      return jpeggpu::status::error;
    }

    huffman_table &table = decoder->huff_tables[th][tc];

    // read bits
    table.bits[0] = 0;
    int count = 0;
    for (int i = 0; i < 16; ++i) {
      const int idx = i + 1;
      table.bits[idx] = read_uint8(image);
      count += table.bits[idx];
    }
    remaining -= 16;

    if (image_end - *image < count) {
      return jpeggpu::status::error;
    }

    if (static_cast<size_t>(count) > sizeof(table.huffval)) {
      return jpeggpu::status::error;
    }

    // read huffval
    for (int i = 0; i < count; ++i) {
      table.huffval[i] = read_uint8(image);
    }
    remaining -= count;

    compute_huffman_table(table);
  }

  return jpeggpu::status::success;
}

struct coder_state {
  int buff;
  int bits;
  int bits_until_byte;
  const uint8_t *image;
  size_t image_size;
  int dc[max_comp_count];
};

/**
 * Fill more bit to current get buffer
 *
 * @param coder
 * @return void
 */
void gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(coder_state &state) {
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
        } else if (MARKER_RST0 <= uc && uc <= MARKER_RST7) {
          DBG_PRINT("marker?? %s\n", get_marker_string(uc));

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
          if (state.bits >= 0)
            break;
        }
      }

      state.buff = (state.buff << 8) | ((int)uc);
      state.bits += 8;
    } else
      break;
  }
}

/**
 * Get bits
 *
 * @param coder  Decoder structure
 * @param nbits  Number of bits to get
 * @return bits
 */
static inline int gpujpeg_huffman_cpu_decoder_get_bits(coder_state &state,
                                                       int nbits) {
  // we should read nbits bits to get next data
  if (state.bits < nbits) {
    gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(state);
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
int gpujpeg_huffman_cpu_decoder_decode_special_decode(
    coder_state &state, const huffman_table &table, int min_bits) {
  // HUFF_DECODE has determined that the code is at least min_bits
  // bits long, so fetch that many bits in one swoop.
  int code = gpujpeg_huffman_cpu_decoder_get_bits(state, min_bits);

  // Collect the rest of the Huffman code one bit at a time.
  // This is per Figure F.16 in the JPEG spec.
  int l = min_bits;
  while (code > table.maxcode[l]) {
    code <<= 1;
    code |= gpujpeg_huffman_cpu_decoder_get_bits(state, 1);
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
static inline int gpujpeg_huffman_cpu_decoder_value_from_category(int category,
                                                                  int offset) {
  // Method 1:
  // On some machines, a shift and add will be faster than a table lookup.
  // #define HUFF_EXTEND(x,s) ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) :
  // (x))

  // Method 2: Table lookup
  // If (offset < half[category]), then value is below zero
  // Otherwise, value is above zero, and just the offset
  // entry n is 2**(n-1)
  static const int half[16] = {0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010,
                               0x0020, 0x0040, 0x0080, 0x0100, 0x0200, 0x0400,
                               0x0800, 0x1000, 0x2000, 0x4000};

#if defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-negative-value"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif // defined __GNUC_
  // start[i] is the starting value in this category; surely it is below zero
  //  entry n is (-1 << n) + 1
  static const int start[16] = {0,     -1,    -3,     -7,    -15,   -31,
                                -63,   -127,  -255,   -511,  -1023, -2047,
                                -4095, -8191, -16383, -32767};
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
static inline int
gpujpeg_huffman_cpu_decoder_get_category(coder_state &state,
                                         const huffman_table &table) {
  // If left bits < 8, we should get more data
  if (state.bits < 8) {
    gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(state);
  }

  // Call special process if data finished; min bits is 1
  if (state.bits < 8) {
    return gpujpeg_huffman_cpu_decoder_decode_special_decode(state, table, 1);
  }

  // Peek the first valid byte
  int look = ((state.buff >> (state.bits - 8)) & 0xFF);
  int nb = table.look_nbits[look];

  if (nb) {
    state.bits -= nb;
    state.bits_until_byte -= nb;
    if (state.bits_until_byte < 0) {
      state.bits_until_byte += 8;
    }
    return table.look_sym[look];
  } else {
    // Decode long codes with length >= 9
    return gpujpeg_huffman_cpu_decoder_decode_special_decode(state, table, 9);
  }
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int decode_block(int16_t *dst, const huffman_table &table_dc,
                 const huffman_table &table_ac, coder_state &state, int &dc) {
  // Zero block output
  memset(dst, 0, sizeof(int16_t) * block_size * block_size);

  // Section F.2.2.1: decode the DC coefficient difference
  // get dc category number, s
  int s = gpujpeg_huffman_cpu_decoder_get_category(state, table_dc);
  if (s) {
    // Get offset in this dc category
    int r = gpujpeg_huffman_cpu_decoder_get_bits(state, s);
    // Get dc difference value
    s = gpujpeg_huffman_cpu_decoder_value_from_category(s, r);
  }

  // Convert DC difference to actual value, update last_dc_val
  s += dc;
  dc = s;

  // Output the DC coefficient (assumes gpujpeg_natural_order[0] = 0)
  dst[0] = s;

  // Section F.2.2.2: decode the AC coefficients
  // Since zeroes are skipped, output area must be cleared beforehand
  for (int k = 1; k < 64; k++) {
    // s: (run, category)
    int s = gpujpeg_huffman_cpu_decoder_get_category(state, table_ac);
    // r: run length for ac zero, 0 <= r < 16
    int r = s >> 4;
    // s: category for this non-zero ac
    s &= 15;
    if (s) {
      //    k: position for next non-zero ac
      k += r;
      //    r: offset in this ac category
      r = gpujpeg_huffman_cpu_decoder_get_bits(state, s);
      //    s: ac value
      s = gpujpeg_huffman_cpu_decoder_value_from_category(s, r);

      dst[order_natural[k]] = s;
    } else {
      // s = 0, means ac value is 0 ? Only if r = 15.
      // means all the left ac are zero
      if (r != 15)
        break;
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

jpeggpu::status read_sos(const uint8_t **image, const uint8_t *image_end,
                         jpeggpu::decoder_t decoder) {
  if (image_end - *image < 3) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image);
  const uint8_t num_components = read_uint8(image);
  if (length != 2 + 1 + 2 * num_components + 3) {
    return jpeggpu::status::error;
  }

  for (uint8_t i = 0; i < num_components; ++i) {
    if (image_end - *image < 2) {
      return jpeggpu::status::error;
    }

    const uint8_t selector = read_uint8(image);
    const uint8_t acdc_selector = read_uint8(image);
    const int id_dc = acdc_selector >> 4;
    const int id_ac = acdc_selector & 0xf;
    DBG_PRINT("\tc_id: %" PRIu8 ", dc: %d, ac: %d\n", selector, id_dc, id_ac);
    decoder->huff_map[i][HUFF_DC] = id_dc;
    decoder->huff_map[i][HUFF_AC] = id_ac;
  }

  if (image_end - *image < 3) {
    return jpeggpu::status::error;
  }

  const uint8_t spectral_start = read_uint8(image);
  const uint8_t spectral_end = read_uint8(image);
  const uint8_t successive_approximation = read_uint8(image);
  (void)spectral_start, (void)spectral_end, (void)successive_approximation;

  decoder->is_interleaved = num_components > 1;

  // FIXME should be done in a better place
  for (int i = 0; i < decoder->num_components; ++i) {
    decoder->sizes_x[i] =
        get_size(decoder->size_x, decoder->ss_x[i], decoder->ss_x_max);
    decoder->sizes_y[i] =
        get_size(decoder->size_y, decoder->ss_y[i], decoder->ss_y_max);

    decoder->mcu_sizes_x[i] =
        decoder->is_interleaved ? block_size * decoder->ss_x[i] : block_size;
    decoder->mcu_sizes_y[i] =
        decoder->is_interleaved ? block_size * decoder->ss_y[i] : block_size;

    decoder->data_sizes_x[i] =
        ceiling_div(decoder->sizes_x[i],
                    static_cast<unsigned int>(decoder->mcu_sizes_x[i])) *
        decoder->mcu_sizes_x[i];
    decoder->data_sizes_y[i] =
        ceiling_div(decoder->sizes_y[i],
                    static_cast<unsigned int>(decoder->mcu_sizes_y[i])) *
        decoder->mcu_sizes_y[i];

    decoder->num_mcus_x[i] =
        ceiling_div(decoder->data_sizes_x[i],
                    static_cast<unsigned int>(decoder->mcu_sizes_x[i]));
    decoder->num_mcus_y[i] =
        ceiling_div(decoder->data_sizes_y[i],
                    static_cast<unsigned int>(decoder->mcu_sizes_y[i]));

    decoder->data[i] =
        new int16_t[decoder->data_sizes_x[i] * decoder->data_sizes_y[i]];
    // FIXME: for now, just output rounded up size
    decoder->image[i] =
        new uint8_t[decoder->data_sizes_x[i] * decoder->data_sizes_y[i]];
  }

  coder_state state = {};
  state.image = *image;
  const size_t image_remaining = image_end - *image;
  state.image_size = image_remaining;

  if (decoder->is_interleaved) {
    int mcu_count = 0;
    for (int y_mcu = 0; y_mcu < decoder->num_mcus_y[0]; ++y_mcu) {
      for (int x_mcu = 0; x_mcu < decoder->num_mcus_x[0]; ++x_mcu) {
        // one MCU
        for (int i = 0; i < decoder->num_components; ++i) {
          const huffman_table &table_dc =
              decoder->huff_tables[decoder->huff_map[i][HUFF_DC]][HUFF_DC];
          const huffman_table &table_ac =
              decoder->huff_tables[decoder->huff_map[i][HUFF_AC]][HUFF_AC];
          for (int y_ss = 0; y_ss < decoder->ss_y[i]; ++y_ss) {
            for (int x_ss = 0; x_ss < decoder->ss_x[i]; ++x_ss) {
              const int y_block = y_mcu * decoder->ss_y[i] + y_ss;
              const int x_block = x_mcu * decoder->ss_x[i] + x_ss;
              const size_t idx = y_block * block_size *
                                     decoder->mcu_sizes_x[i] *
                                     decoder->num_mcus_x[i] +
                                 x_block * block_size * block_size;
              int16_t *dst = &decoder->data[i][idx];
              decode_block(dst, table_dc, table_ac, state, state.dc[i]);
            }
          }
        }
        mcu_count++;
        // FIXME what if restart_interval is not set?
        if (mcu_count % decoder->restart_interval == 0) {
          for (int c = 0; c < max_comp_count; ++c) {
            state.dc[c] = 0;
          }
          // discard bits until a byte is reached
          assert(state.bits >= state.bits_until_byte);
          gpujpeg_huffman_cpu_decoder_get_bits(state, state.bits_until_byte);
          assert(state.bits_until_byte == 0);
        }
      }
    }
  } else {
    return jpeggpu::status::not_supported; // TODO
  }

  const size_t consumed = image_remaining - state.image_size;
  *image += consumed;

  // const uint8_t *segment_start = *image;
  // do {
  //   const uint8_t *ret = reinterpret_cast<const uint8_t *>(std::memchr(
  //       reinterpret_cast<const void *>(*image), 0xff, image_end - *image));
  //   // ff as final char is valid since it's valid encoder symbol if it's
  //   escaped if (ret == nullptr || ret == image_end - 1) {
  //     // file is fully processed
  //     *image = image_end;
  //     break;
  //   }
  //   *image = ret + 1;
  //   const uint8_t marker = read_uint8(image);
  //   if (marker == 0) {
  //     // escaped encoded 0, continue
  //     continue;
  //   }

  //   parse_segment(decoder, segment_start, ret - segment_start);
  //   segment_start = ret;

  //   if (MARKER_RST0 <= marker && marker <= MARKER_RST7) {
  //     // restart marker is okay and part of scan
  //     DBG_PRINT("\trst marker\n");
  //   } else {
  //     // rewind the marker
  //     *image -= 2;
  //     break;
  //   }
  // } while (*image < image_end);

  return jpeggpu::status::success;
}

jpeggpu::status read_dqt(const uint8_t **image, const uint8_t *image_end,
                         jpeggpu::decoder_t decoder) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image) - 2;
  if (image_end - *image < length) {
    return jpeggpu::status::error;
  }

  if (length % 65 != 0) {
    return jpeggpu::status::error;
  }

  const int qtable_count = length / 65;

  for (int i = 0; i < qtable_count; ++i) {
    const uint8_t info = read_uint8(image);
    const int precision = info >> 4;
    const int id = info & 0xf;
    if ((precision != 0 && precision != 1) || id >= 4) {
      return jpeggpu::status::error;
    }
    if (precision != 0) {
      return jpeggpu::status::not_supported;
    }

    for (int j = 0; j < 64; ++j) {
      // element in zigzag order
      const uint8_t element = read_uint8(image);
      // store in natural order
      decoder->qtables[id][order_natural[j]] = element;
    }
  }

  return jpeggpu::status::success;
}

jpeggpu::status read_dri(const uint8_t **image, const uint8_t *image_end,
                         jpeggpu::decoder_t decoder) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image) - 2;
  if (image_end - *image < length) {
    return jpeggpu::status::error;
  }

  const uint16_t restart_interval = read_uint16(image);
  decoder->restart_interval = restart_interval;
  DBG_PRINT("\trestart_interval: %" PRIu16 "\n", restart_interval);

  return jpeggpu::status::success;
}

jpeggpu::status skip_segment(const uint8_t **image, const uint8_t *image_end) {
  if (image_end - *image < 2) {
    return jpeggpu::status::error;
  }

  const uint16_t length = read_uint16(image) - 2;
  if (image_end - *image < length) {
    return jpeggpu::status::error;
  }

  *image += length;
  return jpeggpu::status::success;
}

} // namespace

#define JPEGGPU_CHECK_STATUS(call)                                             \
  do {                                                                         \
    jpeggpu::status stat = call;                                               \
    if (stat != jpeggpu::status::success) {                                    \
      return stat;                                                             \
    }                                                                          \
  } while (0)

jpeggpu::status jpeggpu::decoder_startup(jpeggpu::decoder_t *decoder) {
  if (!decoder) {
    return jpeggpu::status::error;
  }

  *decoder = new jpeggpu::decoder;

  return jpeggpu::status::success;
}

#define W1 2841 // 2048*sqrt(2)*cos(1*pi/16)
#define W2 2676 // 2048*sqrt(2)*cos(2*pi/16)
#define W3 2408 // 2048*sqrt(2)*cos(3*pi/16)
#define W5 1609 // 2048*sqrt(2)*cos(5*pi/16)
#define W6 1108 // 2048*sqrt(2)*cos(6*pi/16)
#define W7 565  // 2048*sqrt(2)*cos(7*pi/16)

// clipping table
static int16_t iclip[1024];
static int16_t *iclp;

void gpujpeg_idct_cpu_perform_row(int16_t *blk) {
  int x0, x1, x2, x3, x4, x5, x6, x7, x8;

  // shortcut
  if (!((x1 = blk[4] << 11) | (x2 = blk[6]) | (x3 = blk[2]) | (x4 = blk[1]) |
        (x5 = blk[7]) | (x6 = blk[5]) | (x7 = blk[3]))) {
    blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] =
        blk[0] << 3;
    return;
  }

  // for proper rounding in the fourth stage
  x0 = (blk[0] << 11) + 128;

  // first stage
  x8 = W7 * (x4 + x5);
  x4 = x8 + (W1 - W7) * x4;
  x5 = x8 - (W1 + W7) * x5;
  x8 = W3 * (x6 + x7);
  x6 = x8 - (W3 - W5) * x6;
  x7 = x8 - (W3 + W5) * x7;

  // second stage
  x8 = x0 + x1;
  x0 -= x1;
  x1 = W6 * (x3 + x2);
  x2 = x1 - (W2 + W6) * x2;
  x3 = x1 + (W2 - W6) * x3;
  x1 = x4 + x6;
  x4 -= x6;
  x6 = x5 + x7;
  x5 -= x7;

  // third stage
  x7 = x8 + x3;
  x8 -= x3;
  x3 = x0 + x2;
  x0 -= x2;
  x2 = (181 * (x4 + x5) + 128) >> 8;
  x4 = (181 * (x4 - x5) + 128) >> 8;

  // fourth stage
  blk[0] = (x7 + x1) >> 8;
  blk[1] = (x3 + x2) >> 8;
  blk[2] = (x0 + x4) >> 8;
  blk[3] = (x8 + x6) >> 8;
  blk[4] = (x8 - x6) >> 8;
  blk[5] = (x0 - x4) >> 8;
  blk[6] = (x3 - x2) >> 8;
  blk[7] = (x7 - x1) >> 8;
}

/**
 * Column (vertical) IDCT
 *
 *             7                         pi         1
 * dst[8*k] = sum c[l] * src[8*l] * cos( -- * ( k + - ) * l )
 *            l=0                        8          2
 *
 * where: c[0]    = 1/1024
 *        c[1..7] = (1/1024)*sqrt(2)
 */
void gpujpeg_idct_cpu_perform_column(int16_t *blk) {
  int x0, x1, x2, x3, x4, x5, x6, x7, x8;

  // shortcut
  if (!((x1 = (blk[8 * 4] << 8)) | (x2 = blk[8 * 6]) | (x3 = blk[8 * 2]) |
        (x4 = blk[8 * 1]) | (x5 = blk[8 * 7]) | (x6 = blk[8 * 5]) |
        (x7 = blk[8 * 3]))) {
    blk[8 * 0] = blk[8 * 1] = blk[8 * 2] = blk[8 * 3] = blk[8 * 4] =
        blk[8 * 5] = blk[8 * 6] = blk[8 * 7] = iclp[(blk[8 * 0] + 32) >> 6];
    return;
  }

  x0 = (blk[8 * 0] << 8) + 8192;

  // first stage
  x8 = W7 * (x4 + x5) + 4;
  x4 = (x8 + (W1 - W7) * x4) >> 3;
  x5 = (x8 - (W1 + W7) * x5) >> 3;
  x8 = W3 * (x6 + x7) + 4;
  x6 = (x8 - (W3 - W5) * x6) >> 3;
  x7 = (x8 - (W3 + W5) * x7) >> 3;

  // second stage
  x8 = x0 + x1;
  x0 -= x1;
  x1 = W6 * (x3 + x2) + 4;
  x2 = (x1 - (W2 + W6) * x2) >> 3;
  x3 = (x1 + (W2 - W6) * x3) >> 3;
  x1 = x4 + x6;
  x4 -= x6;
  x6 = x5 + x7;
  x5 -= x7;

  // third stage
  x7 = x8 + x3;
  x8 -= x3;
  x3 = x0 + x2;
  x0 -= x2;
  x2 = (181 * (x4 + x5) + 128) >> 8;
  x4 = (181 * (x4 - x5) + 128) >> 8;

  // fourth stage
  blk[8 * 0] = iclp[(x7 + x1) >> 14];
  blk[8 * 1] = iclp[(x3 + x2) >> 14];
  blk[8 * 2] = iclp[(x0 + x4) >> 14];
  blk[8 * 3] = iclp[(x8 + x6) >> 14];
  blk[8 * 4] = iclp[(x8 - x6) >> 14];
  blk[8 * 5] = iclp[(x0 - x4) >> 14];
  blk[8 * 6] = iclp[(x3 - x2) >> 14];
  blk[8 * 7] = iclp[(x7 - x1) >> 14];
}

/**
 * Init inverse DCT
 */
void gpujpeg_idct_cpu_init(void) {
  iclp = iclip + 512;
  for (int i = -512; i < 512; i++)
    iclp[i] = (i < -256) ? -256 : ((i > 255) ? 255 : i);
}

void idct(int16_t *data, const qtable &table) {
  for (int i = 0; i < 64; i++) {
    data[i] = (int)data[i] * (int)table[i];
  }

  for (int i = 0; i < 8; i++)
    gpujpeg_idct_cpu_perform_row(data + 8 * i);

  for (int i = 0; i < 8; i++)
    gpujpeg_idct_cpu_perform_column(data + i);
}

jpeggpu::status jpeggpu::parse_header(jpeggpu::decoder_t decoder,
                                      const std::vector<uint8_t> &file_data) {
  const uint8_t *image = file_data.data();
  const uint8_t *const image_end = file_data.data() + file_data.size();

  uint8_t marker_soi{};
  JPEGGPU_CHECK_STATUS(read_marker(&image, image_end, marker_soi));
  DBG_PRINT("marker %s\n", get_marker_string(marker_soi));
  if (marker_soi != MARKER_SOI) {
    return jpeggpu::status::error;
  }

  uint8_t marker{};
  do {
    JPEGGPU_CHECK_STATUS(read_marker(&image, image_end, marker));
    DBG_PRINT("marker %s\n", get_marker_string(marker));
    switch (marker) {
    case MARKER_SOF0:
      read_sof0(&image, image_end, decoder);
      continue;
    case MARKER_DHT:
      read_dht(&image, image_end, decoder);
      continue;
    case MARKER_SOS:
      read_sos(&image, image_end, decoder);
      continue;
    case MARKER_DQT:
      read_dqt(&image, image_end, decoder);
      continue;
    case MARKER_DRI:
      read_dri(&image, image_end, decoder);
      continue;
    default:
      JPEGGPU_CHECK_STATUS(skip_segment(&image, image_end));
      continue;
    }
    JPEGGPU_CHECK_STATUS(skip_segment(&image, image_end));
  } while (marker != MARKER_EOI);

  // TODO check that the number of scans seen is equal to the number of
  //   components

  gpujpeg_idct_cpu_init();
  for (int c = 0; c < decoder->num_components; ++c) {
    const qtable &table = decoder->qtables[decoder->qtable_idx[c]];
    const int num_blocks_x = decoder->data_sizes_x[c] / block_size;
    const int num_blocks_y = decoder->data_sizes_y[c] / block_size;
    for (int y = 0; y < num_blocks_y; ++y) {
      for (int x = 0; x < num_blocks_x; ++x) {
        const int idx = y * num_blocks_x + x;
        idct(&decoder->data[c][64 * idx], table);
      }
    }

    for (int y = 0; y < num_blocks_y; ++y) {
      for (int x = 0; x < num_blocks_x; ++x) {
        for (int z = 0; z < block_size * block_size; ++z) {
          int coefficient_index =
              (y * num_blocks_x + x) * (block_size * block_size) + z;
          int16_t coefficient = decoder->data[c][coefficient_index];
          coefficient += 128;
          if (coefficient > 255)
            coefficient = 255;
          if (coefficient < 0)
            coefficient = 0;
          int index =
              ((y * block_size) + (z / block_size)) * decoder->data_sizes_x[c] +
              ((x * block_size) + (z % block_size));
          decoder->image[c][index] = coefficient;
        }
      }
    }
  }

  // FIXME find better place
  std::vector<uint8_t> rgb(decoder->size_x * decoder->size_y * 3);
  const int size_xx = get_size(decoder->size_x, 1, decoder->ss_x_max);
  const int size_yy = get_size(decoder->size_y, 1, decoder->ss_y_max);
  for (int yy = 0; yy < size_yy; ++yy) {
    for (int yyy = 0; yyy < decoder->ss_y_max; ++yyy) {
      const int y = yy * decoder->ss_y_max + yyy;
      if (y >= decoder->size_y) {
        break;
      }
      for (int xx = 0; xx < size_xx; ++xx) {
        for (int xxx = 0; xxx < decoder->ss_x_max; ++xxx) {
          const int x = xx * decoder->ss_x_max + xxx;
          if (x >= decoder->size_x) {
            break;
          }

          // https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
          // FIXME data_size!
          const uint8_t cy =
              decoder->image[0][y * decoder->data_sizes_x[0] + x];
          const uint8_t cb =
              decoder->image[1][yy * decoder->data_sizes_x[1] + xx];
          const uint8_t cr =
              decoder->image[2][yy * decoder->data_sizes_x[2] + xx];

          // clang-format off
          const float r = cy                            + 1.402f    * (cr - 128.f);
          const float g = cy -  .344136f * (cb - 128.f) -  .714136f * (cr - 128.f);
          const float b = cy + 1.772f    * (cb - 128.f);
          // clang-format on

          const auto clamp = [](float c) {
            return std::max(0.f, std::min(std::round(c), 255.f));
          };

          const size_t idx = y * decoder->size_x + x;
          rgb[3 * idx + 0] = clamp(r);
          rgb[3 * idx + 1] = clamp(g);
          rgb[3 * idx + 2] = clamp(b);
        }
      }
    }
  }
  std::ofstream file;
  file.open("out.ppm");
  file << "P3\n" << decoder->size_x << " " << decoder->size_y << "\n255\n";
  for (size_t i = 0; i < rgb.size() / 3; ++i) {
    file << static_cast<int>(rgb[3 * i + 0]) << " "
         << static_cast<int>(rgb[3 * i + 1]) << " "
         << static_cast<int>(rgb[3 * i + 2]) << "\n";
  }
  file.close();
  // FIXME end

  return jpeggpu::status::success;
}

jpeggpu::status jpeggpu::decoder_cleanup(jpeggpu::decoder_t decoder) {
  if (!decoder) {
    return jpeggpu::status::error;
  }

  delete decoder;

  return jpeggpu::status::success;
}