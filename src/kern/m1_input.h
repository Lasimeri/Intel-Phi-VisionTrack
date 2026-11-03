/* m1_input.h: the deterministic M1 input image: a smooth ramp, low-frequency blocks and noise, every byte from
 * integers, so host and card hold the same frame. */
#ifndef VT_M1_INPUT_H
#define VT_M1_INPUT_H

#include <stdint.h>

#include "vt_util.h"

static inline void m1_make_image(uint8_t *img, int w, int h, int stride)
{
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const int ramp = (x * 96) / w + (y * 64) / h;
            const int block = (((x >> 5) ^ (y >> 5)) & 7) * 8;
            const int noise = (int)(vt_rand(&s) & 31);
            img[(size_t)y * (size_t)stride + (size_t)x] = (uint8_t)(ramp + block + noise);
        }
    }
}

#endif
