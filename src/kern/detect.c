/* detect.c: see detect.h. Against vx.h: native MVEX on the card, the lane model on the host. */
#include "detect.h"

#include "vx.h"

static const float k_two = 2.0f, k_grad = 1.0f / 512.0f, k_harris = 0.04f;

static inline int round_up16(int v) { return (v + 15) & ~15; }

size_t detect_scratch_bytes(int w, int h)
{
    const size_t cols = (size_t)round_up16(w + 16) + 32;
    const size_t rows = (size_t)h + 6;
    /* xx, xy, yy per row, the response, and a candidate list */
    return 4 * cols * rows * sizeof(float) + DETECT_MAX_CANDIDATES * 3 * sizeof(float);
}

int detect_corners(const struct vt_plane *pl, float rx, float ry, float rw, float rh, int margin, int max, float min_dist, float quality, float *x, float *y, void *scratch)
{
    int x0 = (int)rx, y0 = (int)ry, x1 = (int)(rx + rw), y1 = (int)(ry + rh);
    if (x0 < margin) x0 = margin;
    if (y0 < margin) y0 = margin;
    if (x1 > pl->w - margin) x1 = pl->w - margin;
    if (y1 > pl->h - margin) y1 = pl->h - margin;
    if (x1 <= x0 || y1 <= y0)
        return 0;
    /* Lanes start on a 16-lane boundary of the plane: every source load below is aligned. */
    const int ax = x0 & ~15, cols = round_up16(x1 - ax), nv = cols / 16, rows = y1 - y0;
    float *xx = scratch, *xy = xx + (size_t)cols * (size_t)(rows + 4), *yy = xy + (size_t)cols * (size_t)(rows + 4), *resp = yy + (size_t)cols * (size_t)(rows + 4);
    const vxf two = vx_bcast(&k_two), gs = vx_bcast(&k_grad), kh = vx_bcast(&k_harris);
    /* The tensor's three planes, rows y0 - 2 .. y1 + 1. */
    for (int r = 0; r < rows + 4; r++) {
        const int sy = y0 - 2 + r;
        const float *r0 = plane_row(pl, sy - 1), *r1 = plane_row(pl, sy), *r2 = plane_row(pl, sy + 1);
        for (int k = 0; k < nv; k++) {
            const int at = ax + 16 * k;
            const vxf ul = vx_load(r0 + at - 16), uc = vx_load(r0 + at), ur = vx_load(r0 + at + 16);
            const vxf ml = vx_load(r1 + at - 16), mc = vx_load(r1 + at), mr = vx_load(r1 + at + 16);
            const vxf dl = vx_load(r2 + at - 16), dc = vx_load(r2 + at), dr = vx_load(r2 + at + 16);
            const vxf dxu = vx_sub(vx_alignr(ur, uc, 1), vx_alignr(uc, ul, 15));
            const vxf dxm = vx_sub(vx_alignr(mr, mc, 1), vx_alignr(mc, ml, 15));
            const vxf dxd = vx_sub(vx_alignr(dr, dc, 1), vx_alignr(dc, dl, 15));
            const vxf gx = vx_mul(vx_fma(dxm, two, vx_add(dxu, dxd)), gs);
            const vxf dyl = vx_sub(dl, ul), dyc = vx_sub(dc, uc), dyr = vx_sub(dr, ur);
            const vxf gy = vx_mul(vx_fma(dyc, two, vx_add(vx_alignr(dyc, dyl, 15), vx_alignr(dyr, dyc, 1))), gs);
            const size_t i = (size_t)r * (size_t)cols + (size_t)k * 16;
            vx_store(xx + i, vx_mul(gx, gx));
            vx_store(xy + i, vx_mul(gx, gy));
            vx_store(yy + i, vx_mul(gy, gy));
        }
    }
    /* The 5 x 5 box, then det - k trace^2. */
    float best = 0.0f;
    const vxf zero = vx_zero();
#define VSUM(p, r, k)                                                                                                                                                       \
    vx_add(vx_add(vx_add(vx_add(vx_load((p) + (size_t)(r) * (size_t)cols + (size_t)(k) * 16), vx_load((p) + (size_t)((r) + 1) * (size_t)cols + (size_t)(k) * 16)),            \
                         vx_load((p) + (size_t)((r) + 2) * (size_t)cols + (size_t)(k) * 16)),                                                                               \
                  vx_load((p) + (size_t)((r) + 3) * (size_t)cols + (size_t)(k) * 16)),                                                                                      \
           vx_load((p) + (size_t)((r) + 4) * (size_t)cols + (size_t)(k) * 16))
#define HBOX(prev, cur, next) vx_add(vx_add(cur, vx_add(vx_alignr(cur, prev, 15), vx_alignr(next, cur, 1))), vx_add(vx_alignr(cur, prev, 14), vx_alignr(next, cur, 2)))
    for (int r = 0; r < rows; r++) {
        vxf ap = zero, bp = zero, cp = zero;
        vxf ac = VSUM(xx, r, 0), bc = VSUM(xy, r, 0), cc = VSUM(yy, r, 0);
        vxf peak = zero;
        for (int k = 0; k < nv; k++) {
            const vxf an = k + 1 < nv ? VSUM(xx, r, k + 1) : zero, bn = k + 1 < nv ? VSUM(xy, r, k + 1) : zero, cn = k + 1 < nv ? VSUM(yy, r, k + 1) : zero;
            const vxf a = HBOX(ap, ac, an), b = HBOX(bp, bc, bn), c = HBOX(cp, cc, cn);
            const vxf tr = vx_add(a, c);
            const vxf v = vx_fnma(kh, vx_mul(tr, tr), vx_fnma(b, b, vx_mul(a, c)));
            vx_store(resp + (size_t)r * (size_t)cols + (size_t)k * 16, v);
            peak = vx_max(peak, v);
            ap = ac;
            bp = bc;
            cp = cc;
            ac = an;
            bc = bn;
            cc = cn;
        }
        const float m = vx_hmax(peak);
        if (m > best)
            best = m;
    }
    if (best <= 0.0f)
        return 0;
    /* Local maxima above the threshold. Long double for the scalar arithmetic: x87 on both machines, so a replay matches bit for bit. */
    float *cand = resp + (size_t)cols * (size_t)(rows + 4);
    int nc = 0;
    const long double thresh = (long double)best * (long double)quality;
    for (int r = 1; r < rows - 1; r++) {
        const float *up = resp + (size_t)(r - 1) * (size_t)cols, *me = up + cols, *dn = me + cols;
        for (int c = x0 - ax; c < x1 - ax && c < cols - 1; c++) {
            if (c < 1)
                continue;
            const float v = me[c];
            if ((long double)v <= thresh || v < me[c - 1] || v < me[c + 1] || v < up[c] || v < dn[c] || v < up[c - 1] || v < up[c + 1] || v < dn[c - 1] || v < dn[c + 1])
                continue;
            if (nc == DETECT_MAX_CANDIDATES)
                break;
            cand[3 * nc] = v;
            cand[3 * nc + 1] = (float)(ax + c);
            cand[3 * nc + 2] = (float)(y0 + r);
            nc++;
        }
    }
    /* Strongest first (ties by position, so the order is the same everywhere), greedy spacing. */
    int taken = 0;
    const long double min2 = (long double)min_dist * (long double)min_dist;
    while (taken < max) {
        int bi = -1;
        float bv = 0.0f;
        for (int i = 0; i < nc; i++)
            if (cand[3 * i] > bv) {
                bv = cand[3 * i];
                bi = i;
            }
        if (bi < 0)
            break;
        const float cx = cand[3 * bi + 1], cy = cand[3 * bi + 2];
        cand[3 * bi] = 0.0f;
        int far = 1;
        for (int k = 0; k < taken && far; k++) {
            const long double dx = (long double)x[k] - (long double)cx, dy = (long double)y[k] - (long double)cy;
            far = dx * dx + dy * dy >= min2;
        }
        if (far) {
            x[taken] = cx;
            y[taken] = cy;
            taken++;
        }
    }
    return taken;
}
