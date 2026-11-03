/* lk_ref.c: the patch operations in plain C: x87 on the card (excess precision inside an expression, rounded on each store to float) and SSE
 * on the host. The scalar reference: its answer differs from the vector kernel's by rounding, and that difference is what the gates report. */
#include <string.h>

#include "lk_common.h"

size_t lk_tile_bytes(void) { return sizeof(struct lk_tile); }

static void build_ref(const struct vt_plane *pl, long double px, long double py, const struct lk_cfg *c, void *tile, float *g)
{
    struct lk_tile *t = tile;
    const long ix = lk_floor(px), iy = lk_floor(py);
    const float fx = (float)(px - (long double)ix), fy = (float)(py - (long double)iy);
    const int W = c->w, H = c->h;
    const long x0 = ix - W / 2, y0 = iy - H / 2;
    /* S covers one row and one column beyond the patch on each side: lanes -1 .. W, rows -1 .. H. */
    for (int r = 0; r < H + 2; r++) {
        const float *a = plane_row(pl, (int)(y0 - 1 + r)), *b = plane_row(pl, (int)(y0 + r));
        for (int k = 0; k < W + 2; k++) {
            const long x = x0 - 1 + k;
            const float top = a[x] + fx * (a[x + 1] - a[x]);
            const float bot = b[x] + fx * (b[x + 1] - b[x]);
            t->s[r][k] = top + fy * (bot - top);
        }
    }
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    for (int r = 0; r < H; r++) {
        const float *u = t->s[r], *m = t->s[r + 1], *d = t->s[r + 2];
        for (int k = 0; k < W; k++) {
            const int i = k + 1;
            const float dx = ((u[i + 1] - u[i - 1]) + 2.0f * (m[i + 1] - m[i - 1]) + (d[i + 1] - d[i - 1])) * LK_GRAD_SCALE;
            const float dy = ((d[i - 1] - u[i - 1]) + 2.0f * (d[i] - u[i]) + (d[i + 1] - u[i + 1])) * LK_GRAD_SCALE;
            t->t[r][k] = m[i];
            t->tx[r][k] = dx;
            t->ty[r][k] = dy;
            gxx += dx * dx;
            gxy += dx * dy;
            gyy += dy * dy;
        }
    }
    g[0] = gxx;
    g[1] = gxy;
    g[2] = gyy;
}

static void iterate_ref(const struct vt_plane *pl, long double qx, long double qy, const struct lk_cfg *c, void *tile, float *b)
{
    const struct lk_tile *t = tile;
    const long jx = lk_floor(qx), jy = lk_floor(qy);
    const float fx = (float)(qx - (long double)jx), fy = (float)(qy - (long double)jy);
    const long x0 = jx - c->w / 2, y0 = jy - c->h / 2;
    float bx = 0.0f, by = 0.0f;
    for (int r = 0; r < c->h; r++) {
        const float *a = plane_row(pl, (int)(y0 + r)), *n = plane_row(pl, (int)(y0 + r + 1));
        for (int k = 0; k < c->w; k++) {
            const long x = x0 + k;
            const float top = a[x] + fx * (a[x + 1] - a[x]);
            const float bot = n[x] + fx * (n[x + 1] - n[x]);
            const float it = t->t[r][k] - (top + fy * (bot - top));
            bx += it * t->tx[r][k];
            by += it * t->ty[r][k];
        }
    }
    b[0] = bx;
    b[1] = by;
}

static void residual_ref(const struct vt_plane *pl, long double qx, long double qy, const struct lk_cfg *c, void *tile, float *out)
{
    const struct lk_tile *t = tile;
    const long jx = lk_floor(qx), jy = lk_floor(qy);
    const float fx = (float)(qx - (long double)jx), fy = (float)(qy - (long double)jy);
    const long x0 = jx - c->w / 2, y0 = jy - c->h / 2;
    float sa = 0.0f, ss = 0.0f;
    for (int r = 0; r < c->h; r++) {
        const float *a = plane_row(pl, (int)(y0 + r)), *n = plane_row(pl, (int)(y0 + r + 1));
        for (int k = 0; k < c->w; k++) {
            const long x = x0 + k;
            const float top = a[x] + fx * (a[x + 1] - a[x]);
            const float bot = n[x] + fx * (n[x + 1] - n[x]);
            const float it = (t->t[r][k] - (top + fy * (bot - top))) * LK_Q6_INV;
            sa += fabsf(it);
            ss += it * it;
        }
    }
    const float n = (float)(c->w * c->h);
    out[0] = sa / n;
    out[1] = ss / n;
}

#define LK_NAME lk_track_ref
#define LK_FLOWBACK_NAME lk_flowback_ref
#define LK_BUILD build_ref
#define LK_ITERATE iterate_ref
#define LK_RESIDUALFN residual_ref
#include "lk_driver.inc"
