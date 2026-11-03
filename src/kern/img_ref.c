/* img_ref.c: the plain C reference of every image primitive, and the padding code. x87 on the card, SSE on the host,
 * integer arithmetic where the kernel's own F32 sums are exact: this is an independent statement of what the
 * vector kernels must produce, bit for bit. */
#include <math.h>
#include <string.h>

#include "img.h"

void img_pad_rows(const struct vt_plane *pl, int y0, int y1)
{
    for (int y = y0; y < y1; y++) {
        uint32_t *r = (uint32_t *)plane_row(pl, y);
        const uint32_t l = r[0], e = r[pl->w - 1];
        for (int x = -VT_PAD_X; x < 0; x++)
            r[x] = l;
        const int end = ((pl->w + 15) & ~15) + VT_PAD_X;
        for (int x = pl->w; x < end; x++)
            r[x] = e;
    }
}

void img_pad_tb(const struct vt_plane *pl)
{
    const size_t bytes = (size_t)pl->stride * sizeof(float);
    const float *top = plane_row(pl, 0) - VT_PAD_X, *bot = plane_row(pl, pl->h - 1) - VT_PAD_X;
    for (int k = 1; k <= VT_PAD_Y; k++) {
        memcpy(plane_row(pl, -k) - VT_PAD_X, top, bytes);
        memcpy(plane_row(pl, pl->h - 1 + k) - VT_PAD_X, bot, bytes);
    }
}

void img_u8_to_q6_ref(const struct vt_plane *dst, const uint8_t *src, int src_stride, int y0, int y1)
{
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = src + (ptrdiff_t)y * src_stride;
        float *d = plane_row(dst, y);
        for (int x = 0; x < dst->w; x++)
            d[x] = (float)(s[x] * VT_Q6);
    }
}

void img_norm_ref(const struct vt_plane *dst, const struct vt_plane *src, const float *gain_offset, int y0, int y1)
{
    const float g = gain_offset[0], o = gain_offset[1];
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(src, y);
        float *d = plane_row(dst, y);
        for (int x = 0; x < dst->w; x++) {
            /* x87 and host alike: an exact 48-bit product, one 64-bit sum, then rounded to an integer. */
            float v = rintf((float)((long double)s[x] * g + o));
            v = v < 0.0f ? 0.0f : v > 16320.0f ? 16320.0f : v;
            d[x] = v;
        }
    }
}

void img_minmax_ref(const struct vt_plane *src, float *out, int y0, int y1)
{
    float mn = plane_row(src, y0)[0], mx = mn;
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(src, y);
        for (int x = 0; x < src->w; x++) {
            if (s[x] < mn)
                mn = s[x];
            if (s[x] > mx)
                mx = s[x];
        }
    }
    out[0] = mn;
    out[1] = mx;
}

/* The vertical binomial, exact in F32 (at most 261,120). */
static inline float vert5(const float *r0, const float *r1, const float *r2, const float *r3, const float *r4, int x)
{
    return r0[x] + r4[x] + 4.0f * (r1[x] + r3[x]) + 6.0f * r2[x];
}

/* (h + 128) >> 8 for an exact integer-valued h: fistp in the default rounding mode. A C (int) cast on x87 rewrites the
 * control word twice per pixel (fnstcw, fldcw, fistp, fldcw) and would make this reference a rigged baseline. */
static inline float shift8(float h) { return (float)((lrintf(h) + 128) >> 8); }

void img_gauss5_ref(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1)
{
    enum { MAXW = 4096 + 64 };
    static __thread float line[MAXW];
    for (int y = y0; y < y1; y++) {
        const float *r0 = plane_row(src, y - 2), *r1 = plane_row(src, y - 1), *r2 = plane_row(src, y), *r3 = plane_row(src, y + 1), *r4 = plane_row(src, y + 2);
        float *d = plane_row(dst, y);
        float *v = line + 32;
        for (int x = -2; x < dst->w + 2; x++)
            v[x] = vert5(r0, r1, r2, r3, r4, x);
        for (int x = 0; x < dst->w; x++)
            d[x] = shift8(v[x - 2] + v[x + 2] + 4.0f * (v[x - 1] + v[x + 1]) + 6.0f * v[x]);
    }
}

void img_reduce_ref(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1)
{
    enum { MAXW = 4096 + 64 };
    static __thread float line[MAXW];
    for (int y = y0; y < y1; y++) {
        const int sy = 2 * y;
        const float *r0 = plane_row(src, sy - 2), *r1 = plane_row(src, sy - 1), *r2 = plane_row(src, sy), *r3 = plane_row(src, sy + 1), *r4 = plane_row(src, sy + 2);
        float *d = plane_row(dst, y);
        float *v = line + 32;
        for (int x = -2; x < 2 * dst->w + 2; x++)
            v[x] = vert5(r0, r1, r2, r3, r4, x);
        for (int x = 0; x < dst->w; x++) {
            const int c = 2 * x;
            d[x] = shift8(v[c - 2] + v[c + 2] + 4.0f * (v[c - 1] + v[c + 1]) + 6.0f * v[c]);
        }
    }
}

void img_sobel_ref(const struct vt_plane *ix, const struct vt_plane *iy, const struct vt_plane *src, int y0, int y1)
{
    for (int y = y0; y < y1; y++) {
        const float *r0 = plane_row(src, y - 1), *r1 = plane_row(src, y), *r2 = plane_row(src, y + 1);
        float *dx = plane_row(ix, y), *dy = plane_row(iy, y);
        for (int x = 0; x < src->w; x++) {
            const float a = r0[x - 1], b = r0[x], c = r0[x + 1], e = r1[x - 1], g = r1[x + 1], m = r2[x - 1], n = r2[x], o = r2[x + 1];
            dx[x] = (c + 2.0f * g + o) - (a + 2.0f * e + m);
            dy[x] = (m + 2.0f * n + o) - (a + 2.0f * b + c);
        }
    }
}

void img_absdiff_ref(const struct vt_plane *dst, const struct vt_plane *a, const struct vt_plane *b, int y0, int y1)
{
    for (int y = y0; y < y1; y++) {
        const float *p = plane_row(a, y), *q = plane_row(b, y);
        float *d = plane_row(dst, y);
        for (int x = 0; x < dst->w; x++)
            d[x] = fabsf(p[x] - q[x]);
    }
}
