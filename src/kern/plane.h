/* plane.h: the image layout every kernel uses: a padded F32 plane.
 *
 * Pixels are INTEGER-VALUED F32 in Q6 (an 8-bit pixel x 64, at most 16,320). A 5 x 5 binomial sums to at most
 * 16,320 x 256 = 4,177,920 < 2^24, so every blur and gradient is EXACT in F32: no rounding happens until the
 * explicit (h + 128) >> 8. So MVEX, x87 and SSE must agree bit for bit, and `make check` holds them to it. LK
 * then reads the same plane directly, with no conversion step.
 *
 * Row 0 column 0 is 64 B aligned; a row is a whole number of 64 B lines. VT_PAD_X lanes each side and
 * VT_PAD_Y rows above and below hold replicated edge pixels, so kernels never branch on an edge. */
#ifndef VT_PLANE_H
#define VT_PLANE_H

#include <stddef.h>
#include <stdint.h>

#define VT_PAD_X 32
#define VT_PAD_Y 4
#define VT_Q6 64

struct vt_plane {
    float *p; /* pixel (0, 0) */
    int w, h;
    int stride; /* in lanes, a multiple of 16 */
};

static inline int plane_stride(int w) { return ((w + 15) & ~15) + 2 * VT_PAD_X; }
static inline size_t plane_bytes(int w, int h) { return (size_t)plane_stride(w) * (size_t)(h + 2 * VT_PAD_Y) * sizeof(float); }
/* A plane over raw memory of plane_bytes(w, h), 64 B aligned. */
static inline struct vt_plane plane_at(void *mem, int w, int h)
{
    const int s = plane_stride(w);
    struct vt_plane pl = {(float *)mem + (size_t)s * VT_PAD_Y + VT_PAD_X, w, h, s};
    return pl;
}
static inline float *plane_row(const struct vt_plane *pl, int y) { return pl->p + (ptrdiff_t)y * pl->stride; }

#endif
