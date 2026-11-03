/* lk_common.h: what both LK instantiations share: the tile's hand-placed layout, and the small x87 helpers. */
#ifndef VT_LK_COMMON_H
#define VT_LK_COMMON_H

#include <math.h>

#include "lk.h"

/* The tile: T, Tx, Ty, then the sampled rows. Rows are whole 64 B lines. At the largest patch it is 29 KB and sits in the 32 KiB L1, with
 * J's rows streaming past it. */
#define LK_ROW (LK_MAX_V * 16)
#define LK_S_COLS ((LK_MAX_V + 2) * 16)
struct lk_tile {
    float t[LK_MAX_H][LK_ROW];
    float tx[LK_MAX_H][LK_ROW];
    float ty[LK_MAX_H][LK_ROW];
    float s[LK_MAX_H + 2][LK_S_COLS]; /* the interpolated window around the patch */
};

static const long double lk_pow2m[8] = {1.0L, 0.5L, 0.25L, 0.125L, 0.0625L, 0.03125L, 0.015625L, 0.0078125L};

/* floor by fistp in the default rounding mode, no control-word rewrite. */
static inline long lk_floor(long double v)
{
    const long i = __builtin_lrintl(v);
    return i - ((long double)i > v);
}

/* 8-bit grey per pixel from a Sobel sum over Q6: 1 / (8 x 64). Exact. */
#define LK_GRAD_SCALE (1.0f / 512.0f)
#define LK_Q6_INV (1.0f / 64.0f)

#endif
