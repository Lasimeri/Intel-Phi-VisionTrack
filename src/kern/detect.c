/* detect.c: see detect.h. Against vx.h: native MVEX on the card, the lane model on the host. */
#include "detect.h"

#include <string.h>

#include "vt_proto.h"
#include "vx.h"

static const float k_two = 2.0f, k_grad = 1.0f / 512.0f, k_harris = 0.04f;

/* Below this response nothing is a corner, however quiet its neighbourhood: a gradient of 1.5 grey levels a pixel each way over the 5 x 5 window, where sensor noise lives. */
#define DETECT_FLOOR 2500.0L
/* Among corners equally strong for where they stand, the stronger in absolute terms come first. */
#define DETECT_ABS_TIE 0.001L
/* The candidates are ordered by the bits of their score: three passes over 32 bits. */
#define RADIX_BUCKETS 2048u

static inline int round_up16(int v) { return (v + 15) & ~15; }

size_t detect_scratch_bytes(int w, int h)
{
    const size_t cols = (size_t)round_up16(w + 16) + 32;
    const size_t rows = (size_t)h + 6;
    /* xx, xy, yy per row and the response; two grids of 16 x 16 cells, each cell's strongest response and the strongest of it and its eight neighbours; a candidate's score and place, two orders, three histograms */
    return 4 * cols * rows * sizeof(float) + 2 * (cols / 16 + 1) * (rows / 16 + 1) * sizeof(float) + DETECT_MAX_CANDIDATES * (sizeof(float) + 2 * sizeof(int) + 2 * sizeof(uint32_t)) + 3 * RADIX_BUCKETS * sizeof(uint32_t) + 256;
}

int detect_begin(struct detect_job *j, const struct vt_plane *pl, float rx, float ry, float rw, float rh, int margin, int max, float min_dist, float quality, void *scratch)
{
    int x0 = (int)rx, y0 = (int)ry, x1 = (int)(rx + rw), y1 = (int)(ry + rh);
    if (x0 < margin) x0 = margin;
    if (y0 < margin) y0 = margin;
    if (x1 > pl->w - margin) x1 = pl->w - margin;
    if (y1 > pl->h - margin) y1 = pl->h - margin;
    if (x1 <= x0 || y1 <= y0)
        return 0;
    /* Lanes start on a 16-lane boundary of the plane: every source load is aligned. */
    const int ax = x0 & ~15, cols = round_up16(x1 - ax), rows = y1 - y0;
    const size_t tensor = (size_t)cols * (size_t)(rows + 4);
    j->pl = pl;
    j->x0 = x0;
    j->y0 = y0;
    j->x1 = x1;
    j->y1 = y1;
    j->ax = ax;
    j->cols = cols;
    j->nv = cols / 16;
    j->rows = rows;
    j->cell_rows = rows / 16 + 1;
    j->max = max;
    j->min_dist = min_dist;
    j->quality = quality;
    j->xx = scratch;
    j->xy = j->xx + tensor;
    j->yy = j->xy + tensor;
    j->resp = j->yy + tensor;
    j->cell = j->resp + tensor;
    j->near = j->cell + (size_t)j->nv * (size_t)j->cell_rows;
    j->score = j->near + (size_t)j->nv * (size_t)j->cell_rows;
    j->cx = (int *)(j->score + DETECT_MAX_CANDIDATES);
    j->cy = j->cx + DETECT_MAX_CANDIDATES;
    j->order = (uint32_t *)(j->cy + DETECT_MAX_CANDIDATES);
    j->hist = j->order + 2 * DETECT_MAX_CANDIDATES;
    return 1;
}

/* The tensor's three planes, rows y0 - 2 .. y1 + 1. */
void detect_tensor(const struct detect_job *j, int r0, int r1)
{
    const vxf two = vx_bcast(&k_two), gs = vx_bcast(&k_grad);
    const int cols = j->cols;
    for (int r = r0; r < r1; r++) {
        const int sy = j->y0 - 2 + r;
        const float *a = plane_row(j->pl, sy - 1), *m = plane_row(j->pl, sy), *z = plane_row(j->pl, sy + 1);
        for (int k = 0; k < j->nv; k++) {
            const int at = j->ax + 16 * k;
            const vxf ul = vx_load(a + at - 16), uc = vx_load(a + at), ur = vx_load(a + at + 16);
            const vxf ml = vx_load(m + at - 16), mc = vx_load(m + at), mr = vx_load(m + at + 16);
            const vxf dl = vx_load(z + at - 16), dc = vx_load(z + at), dr = vx_load(z + at + 16);
            const vxf dxu = vx_sub(vx_alignr(ur, uc, 1), vx_alignr(uc, ul, 15));
            const vxf dxm = vx_sub(vx_alignr(mr, mc, 1), vx_alignr(mc, ml, 15));
            const vxf dxd = vx_sub(vx_alignr(dr, dc, 1), vx_alignr(dc, dl, 15));
            const vxf gx = vx_mul(vx_fma(dxm, two, vx_add(dxu, dxd)), gs);
            const vxf dyl = vx_sub(dl, ul), dyc = vx_sub(dc, uc), dyr = vx_sub(dr, ur);
            const vxf gy = vx_mul(vx_fma(dyc, two, vx_add(vx_alignr(dyc, dyl, 15), vx_alignr(dyr, dyc, 1))), gs);
            const size_t i = (size_t)r * (size_t)cols + (size_t)k * 16;
            vx_store(j->xx + i, vx_mul(gx, gx));
            vx_store(j->xy + i, vx_mul(gx, gy));
            vx_store(j->yy + i, vx_mul(gy, gy));
        }
    }
}

/* The 5 x 5 box, then det - k trace^2. */
void detect_response(const struct detect_job *j, int r0, int r1)
{
    const vxf kh = vx_bcast(&k_harris), zero = vx_zero();
    const int cols = j->cols, nv = j->nv;
    const float *xx = j->xx, *xy = j->xy, *yy = j->yy;
#define VSUM(p, r, k)                                                                                                                                                                                            \
    vx_add(vx_add(vx_add(vx_add(vx_load((p) + (size_t)(r) * (size_t)cols + (size_t)(k) * 16), vx_load((p) + (size_t)((r) + 1) * (size_t)cols + (size_t)(k) * 16)),                                                \
                         vx_load((p) + (size_t)((r) + 2) * (size_t)cols + (size_t)(k) * 16)),                                                                                                                    \
                  vx_load((p) + (size_t)((r) + 3) * (size_t)cols + (size_t)(k) * 16)),                                                                                                                           \
           vx_load((p) + (size_t)((r) + 4) * (size_t)cols + (size_t)(k) * 16))
#define HBOX(prev, cur, next) vx_add(vx_add(cur, vx_add(vx_alignr(cur, prev, 15), vx_alignr(next, cur, 1))), vx_add(vx_alignr(cur, prev, 14), vx_alignr(next, cur, 2)))
    for (int r = r0; r < r1; r++) {
        vxf ap = zero, bp = zero, cp = zero;
        vxf ac = VSUM(xx, r, 0), bc = VSUM(xy, r, 0), cc = VSUM(yy, r, 0);
        for (int k = 0; k < nv; k++) {
            const vxf an = k + 1 < nv ? VSUM(xx, r, k + 1) : zero, bn = k + 1 < nv ? VSUM(xy, r, k + 1) : zero, cn = k + 1 < nv ? VSUM(yy, r, k + 1) : zero;
            const vxf a = HBOX(ap, ac, an), b = HBOX(bp, bc, bn), c = HBOX(cp, cc, cn);
            const vxf tr = vx_add(a, c);
            vx_store(j->resp + (size_t)r * (size_t)cols + (size_t)k * 16, vx_fnma(kh, vx_mul(tr, tr), vx_fnma(b, b, vx_mul(a, c))));
            ap = ac;
            bp = bc;
            cp = cc;
            ac = an;
            bc = bn;
            cc = cn;
        }
    }
#undef VSUM
#undef HBOX
}

/* Each cell's strongest response. */
void detect_cells(const struct detect_job *j, int c0, int c1)
{
    const vxf zero = vx_zero();
    for (int cr = c0; cr < c1; cr++) {
        const int r1 = (cr + 1) * 16 < j->rows ? (cr + 1) * 16 : j->rows;
        for (int k = 0; k < j->nv; k++) {
            vxf m = zero;
            for (int r = cr * 16; r < r1; r++)
                m = vx_max(m, vx_load(j->resp + (size_t)r * (size_t)j->cols + (size_t)k * 16));
            j->cell[cr * j->nv + k] = vx_hmax(m);
        }
    }
}

/* The candidates in order of score, best first; equal scores in the order they were found, so the answer is the same everywhere. A positive float's bits sort as the float does: three stable passes over the
 * complement of those bits, integers only, in time linear in the candidates. */
static const uint32_t *order_candidates(const struct detect_job *j, int nc)
{
    uint32_t *a = j->order, *b = j->order + DETECT_MAX_CANDIDATES, *h = j->hist;
    memset(h, 0, 3 * RADIX_BUCKETS * sizeof(uint32_t));
    for (int i = 0; i < nc; i++) {
        uint32_t bits;
        memcpy(&bits, &j->score[i], sizeof bits);
        bits = ~bits;
        h[bits & 2047u]++;
        h[RADIX_BUCKETS + ((bits >> 11) & 2047u)]++;
        h[2u * RADIX_BUCKETS + (bits >> 22)]++;
        a[i] = (uint32_t)i;
    }
    for (int pass = 0; pass < 3; pass++) {
        uint32_t *c = h + (unsigned)pass * RADIX_BUCKETS, sum = 0;
        for (unsigned u = 0; u < RADIX_BUCKETS; u++) {
            const uint32_t n = c[u];
            c[u] = sum;
            sum += n;
        }
        for (int i = 0; i < nc; i++) {
            uint32_t bits;
            memcpy(&bits, &j->score[a[i]], sizeof bits);
            b[c[(~bits >> (11 * pass)) & 2047u]++] = a[i];
        }
        uint32_t *s = a;
        a = b;
        b = s;
    }
    return a;
}

int detect_finish(const struct detect_job *j, float *x, float *y)
{
    /* How strong is strong depends on where it stands. The response goes as the gradient to the fourth power: against one threshold for the whole box, one sharp corner on an object's outline, whose patch straddles
     * two motions, erased everything inside the object (measured: 11 corners in a 384 x 307 box over texture). So each corner is judged against its neighbourhood: its cell of 16 x 16 and the eight around it.
     * Scalar arithmetic is long double: x87 on both machines, so a replay matches bit for bit. */
    const int nv = j->nv, ch = j->cell_rows, cols = j->cols;
    float best = 0.0f;
    for (int i = 0; i < nv * ch; i++)
        if (j->cell[i] > best)
            best = j->cell[i];
    if (best <= 0.0f)
        return 0;
    for (int cr = 0; cr < ch; cr++)
        for (int k = 0; k < nv; k++) {
            float m = 0.0f;
            for (int a = cr > 0 ? cr - 1 : 0; a <= cr + 1 && a < ch; a++)
                for (int b = k > 0 ? k - 1 : 0; b <= k + 1 && b < nv; b++)
                    if (j->cell[a * nv + b] > m)
                        m = j->cell[a * nv + b];
            j->near[cr * nv + k] = m;
        }
    /* Local maxima above their neighbourhood's threshold, a vector at a time: most vectors hold none. */
    const vxf zero = vx_zero();
    const int lane_lo = j->x0 - j->ax > 1 ? j->x0 - j->ax : 1, lane_hi = j->x1 - j->ax < cols - 1 ? j->x1 - j->ax : cols - 1;
    int nc = 0;
    for (int r = 1; r < j->rows - 1; r++) {
        const float *up = j->resp + (size_t)(r - 1) * (size_t)cols, *me = up + cols, *dn = me + cols;
        const int cr = r / 16;
        vxf ul = zero, ml = zero, dl = zero, uc = vx_load(up), mc = vx_load(me), dc = vx_load(dn);
        for (int k = 0; k < nv; k++) {
            const vxf un = k + 1 < nv ? vx_load(up + (k + 1) * 16) : zero, mn = k + 1 < nv ? vx_load(me + (k + 1) * 16) : zero, dn_ = k + 1 < nv ? vx_load(dn + (k + 1) * 16) : zero;
            const long double local = (long double)j->near[cr * nv + k], rel = local * (long double)j->quality;
            const float thresh VX_ALIGNED = (float)(rel > DETECT_FLOOR ? rel : DETECT_FLOOR);
            int keep = vx_cmp(vx_bcast(&thresh), mc, VX_CMP_LT);
            /* Only the lanes inside the rectangle. */
            if (16 * k < lane_lo)
                keep &= 0xffff << (lane_lo - 16 * k);
            if (16 * k + 16 > lane_hi)
                keep &= lane_hi > 16 * k ? (1 << (lane_hi - 16 * k)) - 1 : 0;
            if (keep) {
                keep &= vx_cmp(vx_alignr(mc, ml, 15), mc, VX_CMP_LE) & vx_cmp(vx_alignr(mn, mc, 1), mc, VX_CMP_LE) & vx_cmp(uc, mc, VX_CMP_LE) & vx_cmp(dc, mc, VX_CMP_LE);
                keep &= vx_cmp(vx_alignr(uc, ul, 15), mc, VX_CMP_LE) & vx_cmp(vx_alignr(un, uc, 1), mc, VX_CMP_LE) & vx_cmp(vx_alignr(dc, dl, 15), mc, VX_CMP_LE) & vx_cmp(vx_alignr(dn_, dc, 1), mc, VX_CMP_LE);
                for (; keep && nc < DETECT_MAX_CANDIDATES; keep &= keep - 1) {
                    const int c = 16 * k + __builtin_ctz((unsigned)keep);
                    const long double v = (long double)me[c];
                    j->score[nc] = (float)(v / local + DETECT_ABS_TIE * v / (long double)best);
                    j->cx[nc] = j->ax + c;
                    j->cy[nc] = j->y0 + r;
                    nc++;
                }
            }
            ul = uc;
            ml = mc;
            dl = dc;
            uc = un;
            mc = mn;
            dc = dn_;
        }
    }
    /* Best first, greedy spacing. Places are whole pixels, so the spacing is whole-number arithmetic. */
    const uint32_t *ord = order_candidates(j, nc);
    const long double min2 = (long double)j->min_dist * (long double)j->min_dist;
    int need = (int)min2;
    if ((long double)need < min2)
        need++;
    int taken = 0;
    static int tx[VT_MAX_FEATURES], ty[VT_MAX_FEATURES];
    for (int i = 0; i < nc && taken < j->max; i++) {
        const int cx = j->cx[ord[i]], cy = j->cy[ord[i]];
        int far = 1;
        for (int k = 0; k < taken && far; k++) {
            const int dx = tx[k] - cx, dy = ty[k] - cy;
            far = dx * dx + dy * dy >= need;
        }
        if (far) {
            tx[taken] = cx;
            ty[taken] = cy;
            x[taken] = (float)cx;
            y[taken] = (float)cy;
            taken++;
        }
    }
    return taken;
}
