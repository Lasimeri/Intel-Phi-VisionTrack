/* lk_vx.c: the patch operations against vx.h: native MVEX on the card, the lane model on the host.
 *
 * The patch shares one (u, v), so bilinear sampling needs no gather: the four neighbours are row loads and the weights are constants. Sixteen
 * lanes across a row, with a row's x-interpolated value kept for the next row. Lanes past the patch width hold Tx = Ty = 0, so It x Tx
 * contributes exactly nothing in them and the hot loop needs no mask at all; only the residual, which sums |It|, carries one. */
#include "lk_common.h"
#include "vx.h"

static const float k_two = 2.0f, k_grad = LK_GRAD_SCALE, k_q6i = LK_Q6_INV;
static const uint32_t k_abs = 0x7fffffffu;

/* Every line of a window, issued to the L1 at once: the in-order core stalls on one miss at a time when it demand-loads, and these overlap.
 * Each prefetch takes a plain register: base + index + displacement desynchronizes the ISA audit (docs/compatibility.md). */
static inline void touch_rows(const struct vt_plane *pl, long x0, long y0, int rows, int lanes)
{
    const float *r = plane_row(pl, (int)y0) + x0;
    const ptrdiff_t stride = pl->stride;
    const int lines = lanes / 16 + 2;
    for (int y = 0; y < rows; y++) {
        const float *p = (const float *)((uintptr_t)r & ~(uintptr_t)63);
        for (int k = 0; k < lines; k++) {
            __asm__ volatile("" : "+r"(p));
            vx_prefetch0(p);
            p += 16;
        }
        r += stride;
    }
}

static inline vxk tail_mask(int w)
{
    const int rem = w & 15;
    return rem ? (vxk)((1u << rem) - 1u) : VX_ALL;
}

static void build_vx(const struct vt_plane *pl, long double px, long double py, const struct lk_cfg *c, void *tile, float *g)
{
    struct lk_tile *t = tile;
    const long ix = lk_floor(px), iy = lk_floor(py);
    const float fx = (float)(px - (long double)ix), fy = (float)(py - (long double)iy);
    const vxf vfx = vx_bcast(&fx), vfy = vx_bcast(&fy), two = vx_bcast(&k_two), gs = vx_bcast(&k_grad), zero = vx_zero();
    const int W = c->w, H = c->h, nv = (W + 15) / 16;
    const long xs = ix - W / 2 - 16, y0 = iy - H / 2;
    if (c->prefetch)
        touch_rows(pl, xs, y0 - 1, H + 3, 16 * (nv + 2));
    const int cols = nv + 2;
    /* The interpolated window: S[r] covers rows -1 .. H; vector k covers lanes 16 (k - 1) .. of the patch, one block spare each side. */
    vxf prev[LK_MAX_V + 2];
    {
        const float *a = plane_row(pl, (int)(y0 - 1));
        for (int k = 0; k < cols; k++) {
            const vxf l = vx_loadu(a + xs + 16 * k), n = vx_loadu(a + xs + 16 * k + 1);
            prev[k] = vx_fma(vx_sub(n, l), vfx, l);
        }
    }
    for (int r = 0; r < H + 2; r++) {
        const float *a = plane_row(pl, (int)(y0 + r));
        for (int k = 0; k < cols; k++) {
            const vxf l = vx_loadu(a + xs + 16 * k), n = vx_loadu(a + xs + 16 * k + 1);
            const vxf cur = vx_fma(vx_sub(n, l), vfx, l);
            vx_store(&t->s[r][16 * k], vx_fma(vx_sub(cur, prev[k]), vfy, prev[k]));
            prev[k] = cur;
        }
    }
    const vxk tail = tail_mask(W);
    vxf gxx = zero, gxy = zero, gyy = zero;
    for (int r = 0; r < H; r++) {
        const float *u = t->s[r], *m = t->s[r + 1], *d = t->s[r + 2];
        for (int k = 0; k < nv; k++) {
            const int at = 16 * (k + 1);
            const vxf ul = vx_load(u + at - 16), uc = vx_load(u + at), ur = vx_load(u + at + 16);
            const vxf ml = vx_load(m + at - 16), mc = vx_load(m + at), mr = vx_load(m + at + 16);
            const vxf dl = vx_load(d + at - 16), dc = vx_load(d + at), dr = vx_load(d + at + 16);
            /* x + 1 minus x - 1 per row, then 1 2 1 across the rows. */
            const vxf dxu = vx_sub(vx_alignr(ur, uc, 1), vx_alignr(uc, ul, 15));
            const vxf dxm = vx_sub(vx_alignr(mr, mc, 1), vx_alignr(mc, ml, 15));
            const vxf dxd = vx_sub(vx_alignr(dr, dc, 1), vx_alignr(dc, dl, 15));
            vxf tx = vx_mul(vx_fma(dxm, two, vx_add(dxu, dxd)), gs);
            /* y + 1 minus y - 1 per lane, then 1 2 1 across the lanes. */
            const vxf dyl = vx_sub(dl, ul), dyc = vx_sub(dc, uc), dyr = vx_sub(dr, ur);
            vxf ty = vx_mul(vx_fma(dyc, two, vx_add(vx_alignr(dyc, dyl, 15), vx_alignr(dyr, dyc, 1))), gs);
            if (k == nv - 1) {
                tx = vx_blend(tail, zero, tx);
                ty = vx_blend(tail, zero, ty);
            }
            vx_store(&t->t[r][16 * k], mc);
            vx_store(&t->tx[r][16 * k], tx);
            vx_store(&t->ty[r][16 * k], ty);
            gxx = vx_fma(tx, tx, gxx);
            gxy = vx_fma(tx, ty, gxy);
            gyy = vx_fma(ty, ty, gyy);
        }
    }
    g[0] = vx_hsum(gxx);
    g[1] = vx_hsum(gxy);
    g[2] = vx_hsum(gyy);
}

/* One sampling pass for a compile-time block count: unrolled, every value in a register. */
static inline __attribute__((always_inline)) void iterate_core(const struct vt_plane *pl, long double qx, long double qy, const struct lk_cfg *c, const struct lk_tile *t, float *b, const int nv)
{
    const long jx = lk_floor(qx), jy = lk_floor(qy);
    const float fx = (float)(qx - (long double)jx), fy = (float)(qy - (long double)jy);
    const vxf vfx = vx_bcast(&fx), vfy = vx_bcast(&fy);
    const long x0 = jx - c->w / 2, y0 = jy - c->h / 2;
    vxf prev[LK_MAX_V];
    const float *a = plane_row(pl, (int)y0);
    for (int k = 0; k < nv; k++) {
        const vxf l = vx_loadu(a + x0 + 16 * k), n = vx_loadu(a + x0 + 16 * k + 1);
        prev[k] = vx_fma(vx_sub(n, l), vfx, l);
    }
    vxf bx = vx_zero(), by = vx_zero();
    for (int r = 0; r < c->h; r++) {
        a += pl->stride;
        for (int k = 0; k < nv; k++) {
            const vxf l = vx_loadu(a + x0 + 16 * k), n = vx_loadu(a + x0 + 16 * k + 1);
            const vxf cur = vx_fma(vx_sub(n, l), vfx, l);
            const vxf it = vx_sub(vx_load(&t->t[r][16 * k]), vx_fma(vx_sub(cur, prev[k]), vfy, prev[k]));
            bx = vx_fma(it, vx_load(&t->tx[r][16 * k]), bx);
            by = vx_fma(it, vx_load(&t->ty[r][16 * k]), by);
            prev[k] = cur;
        }
    }
    b[0] = vx_hsum(bx);
    b[1] = vx_hsum(by);
}

static void iterate_vx(const struct vt_plane *pl, long double qx, long double qy, const struct lk_cfg *c, void *tile, float *b)
{
    const struct lk_tile *t = tile;
    const int nv = (c->w + 15) / 16;
    if (c->prefetch)
        touch_rows(pl, lk_floor(qx) - c->w / 2, lk_floor(qy) - c->h / 2, c->h + 1, 16 * nv);
    if (nv == 2)
        iterate_core(pl, qx, qy, c, t, b, 2);
    else if (nv == 1)
        iterate_core(pl, qx, qy, c, t, b, 1);
    else
        iterate_core(pl, qx, qy, c, t, b, 3);
}

static void residual_vx(const struct vt_plane *pl, long double qx, long double qy, const struct lk_cfg *c, void *tile, float *out)
{
    const struct lk_tile *t = tile;
    const long jx = lk_floor(qx), jy = lk_floor(qy);
    const float fx = (float)(qx - (long double)jx), fy = (float)(qy - (long double)jy);
    const vxf vfx = vx_bcast(&fx), vfy = vx_bcast(&fy), q6i = vx_bcast(&k_q6i), zero = vx_zero();
    const int nv = (c->w + 15) / 16;
    const vxk tail = tail_mask(c->w);
    const long x0 = jx - c->w / 2, y0 = jy - c->h / 2;
    vxf prev[LK_MAX_V];
    const float *a = plane_row(pl, (int)y0);
    for (int k = 0; k < nv; k++) {
        const vxf l = vx_loadu(a + x0 + 16 * k), n = vx_loadu(a + x0 + 16 * k + 1);
        prev[k] = vx_fma(vx_sub(n, l), vfx, l);
    }
    vxf sa = zero, ss = zero;
    for (int r = 0; r < c->h; r++) {
        a += pl->stride;
        for (int k = 0; k < nv; k++) {
            const vxf l = vx_loadu(a + x0 + 16 * k), n = vx_loadu(a + x0 + 16 * k + 1);
            const vxf cur = vx_fma(vx_sub(n, l), vfx, l);
            vxf it = vx_mul(vx_sub(vx_load(&t->t[r][16 * k]), vx_fma(vx_sub(cur, prev[k]), vfy, prev[k])), q6i);
            if (k == nv - 1)
                it = vx_blend(tail, zero, it);
            sa = vx_add(sa, vx_abs(it, &k_abs));
            ss = vx_fma(it, it, ss);
            prev[k] = cur;
        }
    }
    /* In long double on both machines: a float expression on x87 divides with excess precision, then rounds again on the store. */
    const long double n = (long double)(c->w * c->h);
    out[0] = (float)((long double)vx_hsum(sa) / n);
    out[1] = (float)((long double)vx_hsum(ss) / n);
}

#define LK_NAME lk_track_vx
#define LK_FLOWBACK_NAME lk_flowback_vx
#define LK_BUILD build_vx
#define LK_ITERATE iterate_vx
#define LK_RESIDUALFN residual_vx
#include "lk_driver.inc"
