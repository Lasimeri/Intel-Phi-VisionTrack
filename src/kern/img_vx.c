/* img_vx.c: the image primitives against vx.h: native MVEX on the card, the lane model on the host.
 *
 * Neighbours along a row come from valignd over whole registers (a rolling window of aligned blocks), never from
 * unaligned loads, which cost two instructions each. Every result's lanes beyond the plane's width land in its right
 * padding, which the caller refills. */
#include "img.h"
#include "vx.h"

#define VLOOP(i, n) for (int i = 0; i < (n); i += VX_LANES)

/* Prefetch distances, in lanes. L2 to L1 only: the rows stream through the L2 on their own. */
#define PF_LANES 64

static const float k_q6 = (float)VT_Q6, k2 = 2.0f, k4 = 4.0f, k6 = 6.0f, k_256th = 1.0f / 256.0f, k_half = 0.5f, k_0 = 0.0f, k_max = 16320.0f;
static const uint32_t k_abs = 0x7fffffffu;

void img_fence(void)
{
#if defined(__KNC__)
    __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory", "cc");
#endif
}

void img_u8_to_q6_vx(const struct vt_plane *dst, const uint8_t *src, int src_stride, int y0, int y1)
{
    const vxf q6 = vx_bcast(&k_q6);
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = src + (ptrdiff_t)y * src_stride;
        float *d = plane_row(dst, y);
        VLOOP(x, dst->w)
            vx_store_nr(d + x, vx_mul_u8(q6, s + x));
    }
    img_fence();
}

void img_q6_to_u8_vx(const struct vt_plane *src, uint8_t *dst, int y0, int y1)
{
    const float inv = 1.0f / (float)VT_Q6;
    const vxf k = vx_bcast(&inv);
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(src, y);
        uint8_t *d = dst + (ptrdiff_t)y * src->w;
        VLOOP(x, src->w)
            vx_store_u8(d + x, vx_mul(vx_load(s + x), k));
    }
    img_fence();
}

void img_norm_vx(const struct vt_plane *dst, const struct vt_plane *src, const float *gain_offset, int y0, int y1)
{
    const vxf g = vx_bcast(gain_offset), o = vx_bcast(gain_offset + 1), lo = vx_bcast(&k_0), hi = vx_bcast(&k_max);
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(src, y);
        float *d = plane_row(dst, y);
        VLOOP(x, dst->w)
            vx_store_nr(d + x, vx_min(vx_max(vx_round(vx_fma(vx_load(s + x), g, o), VX_RND_NEAREST), lo), hi));
    }
    img_fence();
}

void img_minmax_vx(const struct vt_plane *src, float *out, int y0, int y1)
{
    vxf mn = vx_load(plane_row(src, y0)), mx = mn;
    const int whole = src->w & ~15;
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(src, y);
        VLOOP(x, whole) {
            const vxf v = vx_load(s + x);
            mn = vx_min(mn, v);
            mx = vx_max(mx, v);
        }
        /* A width that is no multiple of 16: the last block's lanes up to w, the rest masked by keeping them out. */
        if (whole != src->w) {
            const vxf v = vx_load(s + whole);
            const vxk k = (vxk)((1u << (src->w - whole)) - 1u);
            mn = vx_blend(k, mn, vx_min(mn, v));
            mx = vx_blend(k, mx, vx_max(mx, v));
        }
    }
    out[0] = vx_hmin(mn);
    out[1] = vx_hmax(mx);
}

/* The vertical sum at lanes x .. x + 15 of five rows: at most 261,120, exact. */
static inline vxf vert5(const float *r0, const float *r1, const float *r2, const float *r3, const float *r4, int x, vxf c4, vxf c6)
{
    return vx_fma(vx_load(r2 + x), c6, vx_fma(vx_add(vx_load(r1 + x), vx_load(r3 + x)), c4, vx_add(vx_load(r0 + x), vx_load(r4 + x))));
}

/* (h + 128) >> 8 for an exact integer-valued h below 2^24: h / 256 + 0.5 is exact too, and rounding down finishes it. */
static inline vxf shift8(vxf h, vxf inv256, vxf half) { return vx_round(vx_fma(h, inv256, half), VX_RND_DOWN); }

void img_gauss5_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1)
{
    const vxf c4 = vx_bcast(&k4), c6 = vx_bcast(&k6), inv = vx_bcast(&k_256th), half = vx_bcast(&k_half);
    for (int y = y0; y < y1; y++) {
        const float *r0 = plane_row(src, y - 2), *r1 = plane_row(src, y - 1), *r2 = plane_row(src, y), *r3 = plane_row(src, y + 1), *r4 = plane_row(src, y + 2);
        float *d = plane_row(dst, y);
        vxf prev = vert5(r0, r1, r2, r3, r4, -VX_LANES, c4, c6);
        vxf cur = vert5(r0, r1, r2, r3, r4, 0, c4, c6);
        VLOOP(x, dst->w) {
            const vxf next = vert5(r0, r1, r2, r3, r4, x + VX_LANES, c4, c6);
            const vxf outer = vx_add(vx_alignr(cur, prev, 14), vx_alignr(next, cur, 2));
            const vxf inner = vx_add(vx_alignr(cur, prev, 15), vx_alignr(next, cur, 1));
            const vxf h = vx_fma(cur, c6, vx_fma(inner, c4, outer));
            vx_store_nr(d + x, shift8(h, inv, half));
            prev = cur;
            cur = next;
        }
    }
    img_fence();
}

/* One pyramid step, the plain form: the horizontal filter on all 32 lanes, then every second lane kept. A fused form that
 * deinterleaved first and filtered at half width was written and measured against this: with prefetch on both it LOST at
 * all three sizes (640x480 1818 vs 1709 us, 720p 3100 vs 3020, 1080p 7258 vs 7228; logs/m1/20261102-2100xx), so it is gone. */
static const int32_t idx_even[VX_LANES] VX_ALIGNED = {0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14};
#define HIGH_HALF ((vxk)0xff00)

/* Opaque to the addressing-mode folder: the prefetch gets a plain register as its address. */
static inline const float *pf_at(const float *r, int x)
{
    const float *p = r + x + PF_LANES;
    __asm__ volatile("" : "+r"(p));
    return p;
}


static inline __attribute__((always_inline)) void reduce_core(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1, const int pf)
{
    const vxf c4 = vx_bcast(&k4), c6 = vx_bcast(&k6), inv = vx_bcast(&k_256th), half = vx_bcast(&k_half);
    const vxidx ie = vx_idx(idx_even);
    for (int y = y0; y < y1; y++) {
        const int sy = 2 * y;
        const float *r0 = plane_row(src, sy - 2), *r1 = plane_row(src, sy - 1), *r2 = plane_row(src, sy), *r3 = plane_row(src, sy + 1), *r4 = plane_row(src, sy + 2);
        float *d = plane_row(dst, y);
        vxf prev = vert5(r0, r1, r2, r3, r4, -VX_LANES, c4, c6);
        vxf cur = vert5(r0, r1, r2, r3, r4, 0, c4, c6);
        VLOOP(i, dst->w) {
            const int x = 2 * i;
            if (pf) {
                const float *p0 = pf_at(r0, x + 2 * VX_LANES), *p1 = pf_at(r1, x + 2 * VX_LANES), *p2 = pf_at(r2, x + 2 * VX_LANES), *p3 = pf_at(r3, x + 2 * VX_LANES), *p4 = pf_at(r4, x + 2 * VX_LANES);
                vx_prefetch0(p0);
                vx_prefetch0(p1);
                vx_prefetch0(p2);
                vx_prefetch0(p3);
                vx_prefetch0(p4);
            }
            const vxf next = vert5(r0, r1, r2, r3, r4, x + VX_LANES, c4, c6);
            const vxf after = vert5(r0, r1, r2, r3, r4, x + 2 * VX_LANES, c4, c6);
            const vxf h0 = vx_fma(cur, c6, vx_fma(vx_add(vx_alignr(cur, prev, 15), vx_alignr(next, cur, 1)), c4, vx_add(vx_alignr(cur, prev, 14), vx_alignr(next, cur, 2))));
            const vxf h1 = vx_fma(next, c6, vx_fma(vx_add(vx_alignr(next, cur, 15), vx_alignr(after, next, 1)), c4, vx_add(vx_alignr(next, cur, 14), vx_alignr(after, next, 2))));
            vx_store_nr(d + i, shift8(vx_perm_merge(vx_permi(h0, ie), HIGH_HALF, h1, ie), inv, half));
            prev = next;
            cur = after;
        }
    }
    img_fence();
}

void img_reduce_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1) { reduce_core(dst, src, y0, y1, 1); }
void img_reduce_nopf_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1) { reduce_core(dst, src, y0, y1, 0); }

void img_sobel_vx(const struct vt_plane *ix, const struct vt_plane *iy, const struct vt_plane *src, int y0, int y1)
{
    const vxf c2 = vx_bcast(&k2);
    for (int y = y0; y < y1; y++) {
        const float *r0 = plane_row(src, y - 1), *r1 = plane_row(src, y), *r2 = plane_row(src, y + 1);
        float *dx = plane_row(ix, y), *dy = plane_row(iy, y);
#define S(x) vx_fma(vx_load(r1 + (x)), c2, vx_add(vx_load(r0 + (x)), vx_load(r2 + (x))))
#define D(x) vx_sub(vx_load(r2 + (x)), vx_load(r0 + (x)))
        vxf s_prev = S(-VX_LANES), s_cur = S(0), d_prev = D(-VX_LANES), d_cur = D(0);
        VLOOP(x, src->w) {
            const vxf s_next = S(x + VX_LANES), d_next = D(x + VX_LANES);
            vx_store_nr(dx + x, vx_sub(vx_alignr(s_next, s_cur, 1), vx_alignr(s_cur, s_prev, 15)));
            vx_store_nr(dy + x, vx_fma(d_cur, c2, vx_add(vx_alignr(d_cur, d_prev, 15), vx_alignr(d_next, d_cur, 1))));
            s_prev = s_cur;
            s_cur = s_next;
            d_prev = d_cur;
            d_cur = d_next;
        }
#undef S
#undef D
    }
    img_fence();
}

void img_absdiff_vx(const struct vt_plane *dst, const struct vt_plane *a, const struct vt_plane *b, int y0, int y1)
{
    for (int y = y0; y < y1; y++) {
        const float *p = plane_row(a, y), *q = plane_row(b, y);
        float *d = plane_row(dst, y);
        VLOOP(x, dst->w)
            vx_store_nr(d + x, vx_abs(vx_sub(vx_load(p + x), vx_load(q + x)), &k_abs));
    }
    img_fence();
}
