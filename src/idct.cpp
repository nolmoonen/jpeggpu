#include "idct.hpp"

#define W1 2841 // 2048*sqrt(2)*cos(1*pi/16)
#define W2 2676 // 2048*sqrt(2)*cos(2*pi/16)
#define W3 2408 // 2048*sqrt(2)*cos(3*pi/16)
#define W5 1609 // 2048*sqrt(2)*cos(5*pi/16)
#define W6 1108 // 2048*sqrt(2)*cos(6*pi/16)
#define W7 565  // 2048*sqrt(2)*cos(7*pi/16)

// clipping table
static int16_t iclip[1024];
static int16_t* iclp;

void idct_cpu_perform_row(int16_t* blk)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    // shortcut
    if (!((x1 = blk[4] << 11) | (x2 = blk[6]) | (x3 = blk[2]) | (x4 = blk[1]) | (x5 = blk[7]) |
          (x6 = blk[5]) | (x7 = blk[3]))) {
        blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] = blk[0] << 3;
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
void idct_cpu_perform_column(int16_t* blk)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    // shortcut
    if (!((x1 = (blk[8 * 4] << 8)) | (x2 = blk[8 * 6]) | (x3 = blk[8 * 2]) | (x4 = blk[8 * 1]) |
          (x5 = blk[8 * 7]) | (x6 = blk[8 * 5]) | (x7 = blk[8 * 3]))) {
        blk[8 * 0] = blk[8 * 1] = blk[8 * 2] = blk[8 * 3] = blk[8 * 4] = blk[8 * 5] = blk[8 * 6] =
            blk[8 * 7] = iclp[(blk[8 * 0] + 32) >> 6];
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
void idct_cpu_init()
{
    iclp = iclip + 512;
    for (int i = -512; i < 512; i++)
        iclp[i] = (i < -256) ? -256 : ((i > 255) ? 255 : i);
}

void idct_data_unit(int16_t* data, const jpeggpu::qtable& table)
{
    for (int i = 0; i < 64; i++) {
        data[i] = (int)data[i] * (int)table[i];
    }

    for (int i = 0; i < 8; i++)
        idct_cpu_perform_row(data + 8 * i);

    for (int i = 0; i < 8; i++)
        idct_cpu_perform_column(data + i);
}

void jpeggpu::idct(jpeggpu::reader* decoder)
{
    idct_cpu_init();
    for (int c = 0; c < decoder->num_components; ++c) {
        const jpeggpu::qtable& table = decoder->qtables[decoder->qtable_idx[c]];
        const int num_blocks_x       = decoder->data_sizes_x[c] / jpeggpu::block_size;
        const int num_blocks_y       = decoder->data_sizes_y[c] / jpeggpu::block_size;
        for (int y = 0; y < num_blocks_y; ++y) {
            for (int x = 0; x < num_blocks_x; ++x) {
                const int idx = y * num_blocks_x + x;
                idct_data_unit(&decoder->data[c][64 * idx], table);
            }
        }

        for (int y = 0; y < num_blocks_y; ++y) {
            for (int x = 0; x < num_blocks_x; ++x) {
                for (int z = 0; z < jpeggpu::block_size * jpeggpu::block_size; ++z) {
                    int coefficient_index =
                        (y * num_blocks_x + x) * (jpeggpu::block_size * jpeggpu::block_size) + z;
                    int16_t coefficient = decoder->data[c][coefficient_index];
                    coefficient += 128;
                    if (coefficient > 255) coefficient = 255;
                    if (coefficient < 0) coefficient = 0;
                    int index = ((y * jpeggpu::block_size) + (z / jpeggpu::block_size)) *
                                    decoder->data_sizes_x[c] +
                                ((x * jpeggpu::block_size) + (z % jpeggpu::block_size));
                    decoder->image_out[c][index] = coefficient;
                }
            }
        }
    }
}