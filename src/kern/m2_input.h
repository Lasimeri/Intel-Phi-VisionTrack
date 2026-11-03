/* m2_input.h: an image pair with a KNOWN sub-pixel displacement, from integers alone, so host and card hold the same bytes and the truth is exact.
 *
 * A continuous-looking pattern is sampled on a grid 4 x finer than the pixels (bilinear integer value noise at two scales); each pixel is the
 * box average of its 4 x 4 fine cells. Rendering the second image with the fine grid offset by (sx, sy) quarter pixels moves the whole pattern by
 * exactly (sx / 4, sy / 4) pixels, under the same pixel-integration model as a real sensor. */
#ifndef VT_M2_INPUT_H
#define VT_M2_INPUT_H

#include <stdint.h>

#define M2_FINE 4

static inline uint32_t m2_hash(int x, int y)
{
    uint32_t h = (uint32_t)x * 0x9E3779B1u ^ (uint32_t)y * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0xC2B2AE3Du;
    return h ^ (h >> 13);
}

/* Value noise at spacing 1 << bits fine cells, result 0 .. 255 x 2^(2 bits). */
static inline uint32_t m2_noise(int x, int y, int bits)
{
    const int xi = x >> bits, yi = y >> bits, m = (1 << bits) - 1;
    const uint32_t fx = (uint32_t)(x & m), fy = (uint32_t)(y & m), one = 1u << bits;
    const uint32_t a = m2_hash(xi, yi) & 255, b = m2_hash(xi + 1, yi) & 255, c = m2_hash(xi, yi + 1) & 255, d = m2_hash(xi + 1, yi + 1) & 255;
    return (a * (one - fx) + b * fx) * (one - fy) + (c * (one - fx) + d * fx) * fy;
}

/* A fine cell: 0 .. 255 in Q4. */
static inline uint32_t m2_fine(int x, int y)
{
    const uint32_t n3 = m2_noise(x, y, 5) >> 6;  /* spacing 8 pixels: 0 .. 255 in Q4 */
    const uint32_t n2 = m2_noise(x, y, 4) >> 4;  /* spacing 4 pixels: 0 .. 255 in Q4 */
    return (n3 * 3 + n2) >> 2;
}

/* img: h rows of stride bytes. Offset (ox, oy) in fine cells, ox and oy >= 0. */
static inline void m2_render(uint8_t *img, int w, int h, int stride, int ox, int oy)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint32_t sum = 0;
            for (int j = 0; j < M2_FINE; j++)
                for (int i = 0; i < M2_FINE; i++)
                    sum += m2_fine(M2_FINE * x + i + ox, M2_FINE * y + j + oy);
            img[(long)y * stride + x] = (uint8_t)((sum + 128) >> 8);
        }
    }
}

/* The fixed displacement tests, in quarter pixels: the pattern moves by (dx / 4, dy / 4) from the first image to the second. */
struct m2_case { int dx, dy; };
static const struct m2_case m2_cases[] = {
    {0, 0}, {1, 0}, {0, 1}, {2, 2}, {-3, 1}, {5, -6}, {13, 7}, {-22, 17}, {49, -30}, {-101, 66},
};
#define M2_NCASES ((int)(sizeof m2_cases / sizeof m2_cases[0]))
#define M2_OFFSET 2048 /* fine cells, keeps every offset positive */

#endif
