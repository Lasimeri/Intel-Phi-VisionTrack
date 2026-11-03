/* attitude.c: see attitude.h. */
#define _GNU_SOURCE
#include "attitude.h"

#include <string.h>

#include "depth.h"
#include "img.h"
#include "vx.h"

typedef long double real;

/* The search's reach at the coarsest level, 4 input pixels each: at level 2 of a 1080p camera that is +-640 and +-480 full-size pixels. */
#define SEARCH_X 40
#define SEARCH_Y 30
#define SEARCH_UNITS ((2 * SEARCH_X + 1) * (2 * SEARCH_Y + 1))
/* A shift with less of the picture in common than this is not evidence of anything. */
#define SEARCH_MIN_SHARE 4

/* What one row of the normal equations adds up. */
enum { T_H00, T_H01, T_H02, T_H11, T_H12, T_H22, T_G0, T_G1, T_G2, T_WEIGHT, T_RR, T_SS, T_RS, T_IN, T_TERMS };

/* Coarse to fine: the kernel's width in normalized residual, and the most steps. Wide first, while most of the picture is still misplaced. */
static const float KERNEL_C[ATT_LEVELS] = {1.6f, 2.0f, 2.6f};
static const int MAX_STEPS[ATT_LEVELS] = {8, 10, 14};
#define STEP_DONE 2e-6L /* radians: a tenth of a thousandth of a pixel's worth at this focal length, near enough */
#define DAMP 1e-3L

struct att_level {
    struct vt_plane cur, nref, ncur, j[3];
    real f, cx, cy;
    int row_step;
};

struct att {
    struct att_cfg c;
    struct att_level lv[ATT_LEVELS];
    char *scratch;
    size_t scratch_each;
    float *search, *rows;
    real q[4];
    /* The round under way. */
    const struct vt_plane *input;
    int level;
    real O[9];
    float kernel_c;
};

static inline int halved(int v, int times)
{
    for (int i = 0; i < times; i++)
        v = (v + 1) / 2;
    return v;
}

size_t att_bytes(const struct att_cfg *c)
{
    size_t n = 0;
    for (int l = 0; l < ATT_LEVELS; l++)
        n += 7 * ((plane_bytes(halved(c->w, l), halved(c->h, l)) + 63) & ~(size_t)63);
    n += (size_t)c->threads * depth_norm_scratch_bytes(c->w) + 64;
    n += sizeof(float) * (size_t)SEARCH_UNITS + 64;
    n += sizeof(float) * (size_t)(c->h * T_TERMS) + 64;
    return n + sizeof(struct att) + 64;
}

struct att *att_create(const struct att_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx)
{
    struct att *a = alloc(ctx, sizeof *a);
    if (!a)
        return NULL;
    memset(a, 0, sizeof *a);
    a->c = *c;
    for (int l = 0; l < ATT_LEVELS; l++) {
        struct att_level *v = &a->lv[l];
        const int w = halved(c->w, l), h = halved(c->h, l);
        const size_t bytes = (plane_bytes(w, h) + 63) & ~(size_t)63;
        struct vt_plane *const planes[7] = {&v->cur, &v->nref, &v->ncur, &v->j[0], &v->j[1], &v->j[2], NULL};
        for (int i = 0; planes[i]; i++) {
            void *m = alloc(ctx, bytes);
            if (!m)
                return NULL;
            memset(m, 0, bytes);
            *planes[i] = plane_at(m, w, h);
        }
        const real s = (real)(1 << l);
        v->f = (real)c->f / s;
        v->cx = (real)c->cx / s;
        v->cy = (real)c->cy / s;
        /* The finest level has far more pixels than the rotation's three numbers need. */
        v->row_step = l == 0 ? 2 : 1;
    }
    a->scratch_each = depth_norm_scratch_bytes(c->w);
    a->scratch = alloc(ctx, (size_t)c->threads * a->scratch_each);
    a->search = alloc(ctx, sizeof(float) * (size_t)SEARCH_UNITS);
    a->rows = alloc(ctx, sizeof(float) * (size_t)(c->h * T_TERMS));
    if (!a->scratch || !a->search || !a->rows)
        return NULL;
    a->q[0] = 1.0L;
    return a;
}

void att_set_lens(struct att *a, double f, double cx, double cy)
{
    a->c.f = f;
    a->c.cx = cx;
    a->c.cy = cy;
    for (int l = 0; l < ATT_LEVELS; l++) {
        const real s = (real)(1 << l);
        a->lv[l].f = (real)f / s;
        a->lv[l].cx = (real)cx / s;
        a->lv[l].cy = (real)cy / s;
    }
}

/* Quaternions, w x y z. */
static void q_normalize(real *q)
{
    const real n = __builtin_sqrtl(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; i++)
        q[i] /= n;
}

static void q_mul(const real *a, const real *b, real *o)
{
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void q_to_matrix(const real *q, real *m)
{
    const real w = q[0], x = q[1], y = q[2], z = q[3];
    m[0] = 1 - 2 * (y * y + z * z);
    m[1] = 2 * (x * y - w * z);
    m[2] = 2 * (x * z + w * y);
    m[3] = 2 * (x * y + w * z);
    m[4] = 1 - 2 * (x * x + z * z);
    m[5] = 2 * (y * z - w * x);
    m[6] = 2 * (x * z - w * y);
    m[7] = 2 * (y * z + w * x);
    m[8] = 1 - 2 * (x * x + y * y);
}

/* Build this level's picture from the one above it. */
static void round_reduce(void *arg, int tid, int n)
{
    struct att *a = arg;
    const struct att_level *v = &a->lv[a->level];
    const struct vt_plane *src = a->level == 1 ? a->input : &a->lv[a->level - 1].cur;
    const int h = v->cur.h, y0 = h * tid / n, y1 = h * (tid + 1) / n;
    img_reduce_vx(&v->cur, src, y0, y1);
    img_pad_rows(&v->cur, y0, y1);
}

static void round_norm_cur(void *arg, int tid, int n)
{
    struct att *a = arg;
    const struct att_level *v = &a->lv[a->level];
    depth_normalize_part(&v->ncur, NULL, a->level ? &v->cur : a->input, a->c.eps, tid, n, a->scratch, a->scratch_each);
}

static void pyramid(struct att *a, const struct vt_plane *in, att_round round, void *ctx)
{
    a->input = in;
    for (int l = 0; l < ATT_LEVELS; l++) {
        a->level = l;
        if (l) {
            round(ctx, round_reduce, a);
            img_pad_tb(&a->lv[l].cur);
        }
        round(ctx, round_norm_cur, a);
    }
}

/* What a small turn of the camera does to the reference, at each pixel: its gradient along the turn's image motion. */
static void round_jacobian(void *arg, int tid, int n)
{
    struct att *a = arg;
    const struct att_level *v = &a->lv[a->level];
    const int w = v->nref.w, h = v->nref.h;
    for (int y = h * tid / n; y < h * (tid + 1) / n; y++) {
        float *j0 = plane_row(&v->j[0], y), *j1 = plane_row(&v->j[1], y), *j2 = plane_row(&v->j[2], y);
        if (y < 1 || y >= h - 1) {
            memset(j0, 0, sizeof(float) * (size_t)w);
            memset(j1, 0, sizeof(float) * (size_t)w);
            memset(j2, 0, sizeof(float) * (size_t)w);
            continue;
        }
        const float *r = plane_row(&v->nref, y), *up = plane_row(&v->nref, y - 1), *dn = plane_row(&v->nref, y + 1);
        const real ny = ((real)y - v->cy) / v->f;
        for (int x = 0; x < w; x++) {
            if (x < 1 || x >= w - 1) {
                j0[x] = j1[x] = j2[x] = 0.0f;
                continue;
            }
            const real gx = 0.5L * ((real)r[x + 1] - (real)r[x - 1]), gy = 0.5L * ((real)dn[x] - (real)up[x]);
            const real nx = ((real)x - v->cx) / v->f;
            /* d(pixel)/d(turn), from the projection of (I + [d]x) (nx, ny, 1). */
            j0[x] = (float)(v->f * (gx * (-nx * ny) + gy * (-(1.0L + ny * ny))));
            j1[x] = (float)(v->f * (gx * (1.0L + nx * nx) + gy * (nx * ny)));
            j2[x] = (float)(v->f * (gx * (-ny) + gy * nx));
        }
    }
}

void att_set_reference(struct att *a, const struct vt_plane *ref, att_round round, void *round_ctx)
{
    pyramid(a, ref, round, round_ctx);
    for (int l = 0; l < ATT_LEVELS; l++) {
        struct att_level *v = &a->lv[l];
        for (int y = 0; y < v->nref.h; y++)
            memcpy(plane_row(&v->nref, y), plane_row(&v->ncur, y), sizeof(float) * (size_t)v->nref.w);
        a->level = l;
        round(round_ctx, round_jacobian, a);
    }
    memset(a->q, 0, sizeof a->q);
    a->q[0] = 1.0L;
}

/* Every shift of the current picture over the reference: one correlation each, in units of chance so that a small overlap cannot win by being small. */
static void round_search(void *arg, int tid, int n)
{
    struct att *a = arg;
    const struct att_level *v = &a->lv[ATT_LEVELS - 1];
    const int w = v->nref.w, h = v->nref.h, across = 2 * SEARCH_X + 1;
    const int least = w * h / SEARCH_MIN_SHARE;
    for (int u = tid; u < SEARCH_UNITS; u += n) {
        const int sx = u % across - SEARCH_X, sy = u / across - SEARCH_Y;
        const int y0 = sy < 0 ? -sy : 0, y1 = sy > 0 ? h - sy : h;
        const int x0 = ((sx < 0 ? -sx : 0) + VX_LANES - 1) & ~(VX_LANES - 1);
        const int x1 = (sx > 0 ? w - sx : w) & ~(VX_LANES - 1);
        const int count = (x1 - x0) * (y1 - y0);
        if (x1 <= x0 || y1 <= y0 || count < least) {
            a->search[u] = -1e30f;
            continue;
        }
        real sum = 0.0L;
        for (int y = y0; y < y1; y++) {
            const float *r = plane_row(&v->nref, y), *c = plane_row(&v->ncur, y + sy) + sx;
            vxf s = vx_zero();
            for (int x = x0; x < x1; x += VX_LANES)
                s = vx_fma(vx_load(r + x), vx_loadu(c + x), s);
            sum += (real)vx_hsum(s);
        }
        a->search[u] = (float)(sum / __builtin_sqrtl((real)count));
    }
}

/* The normal equations of one level: the current picture warped by the rotation held, the residual against the reference, each row's sums kept apart. */
static void round_accumulate(void *arg, int tid, int n)
{
    static const float RAMP[16] VX_ALIGNED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    struct att *a = arg;
    const struct att_level *v = &a->lv[a->level];
    const real *O = a->O;
    const int w = v->nref.w, h = v->nref.h, stride = v->ncur.stride;
    /* In long double: float arithmetic is x87 on the card and SSE on the host, and rounds differently. */
    const float one VX_ALIGNED = 1.0f, ic2 VX_ALIGNED = (float)(1.0L / ((real)a->kernel_c * (real)a->kernel_c)), wm2 VX_ALIGNED = (float)(w - 2), hm2 VX_ALIGNED = (float)(h - 2), wm1 VX_ALIGNED = (float)(w - 1), fstride VX_ALIGNED = (float)stride;
    const vxf ramp = vx_load(RAMP), vone = vx_bcast(&one), vic2 = vx_bcast(&ic2), vw = vx_bcast(&wm2), vh = vx_bcast(&hm2), vlast = vx_bcast(&wm1), vs = vx_bcast(&fstride), zero = vx_zero();
    const real f = v->f, cx = v->cx, cy = v->cy;
    const int rows = (h + v->row_step - 1) / v->row_step;
    for (int i = tid; i < rows; i += n) {
        const int y = i * v->row_step;
        const float *r = plane_row(&v->nref, y), *j0 = plane_row(&v->j[0], y), *j1 = plane_row(&v->j[1], y), *j2 = plane_row(&v->j[2], y);
        const real ny = ((real)y - cy) / f;
        const real a0 = O[0], a1 = O[1] * ny + O[2], b0 = O[3], b1 = O[4] * ny + O[5], c0 = O[6], c1 = O[7] * ny + O[8];
        vxf t[T_TERMS];
        for (int k = 0; k < T_TERMS; k++)
            t[k] = zero;
        for (int x = 0; x < w; x += VX_LANES) {
            /* A rotation's homography, nearly affine over 16 pixels: exact at both ends of the block, one division each, straight between. */
            const real n0 = ((real)x - cx) / f, n1 = ((real)(x + VX_LANES) - cx) / f;
            const real z0 = c0 * n0 + c1, z1 = c0 * n1 + c1;
            const real u0 = f * (a0 * n0 + a1) / z0 + cx, u1 = f * (a0 * n1 + a1) / z1 + cx, v0 = f * (b0 * n0 + b1) / z0 + cy, v1 = f * (b0 * n1 + b1) / z1 + cy;
            const float au VX_ALIGNED = (float)u0, av VX_ALIGNED = (float)v0, du VX_ALIGNED = (float)((u1 - u0) / (real)VX_LANES), dv VX_ALIGNED = (float)((v1 - v0) / (real)VX_LANES), fx0 VX_ALIGNED = (float)x;
            const vxf us = vx_fma(ramp, vx_bcast(&du), vx_bcast(&au)), vy = vx_fma(ramp, vx_bcast(&dv), vx_bcast(&av));
            const vxf fu = vx_round(us, VX_RND_DOWN), fv = vx_round(vy, VX_RND_DOWN);
            /* In view, and a real pixel: a block can run past the width. A point behind the camera is not in view either. */
            const vxk ok = (vxk)(vx_cmp(zero, fu, VX_CMP_LE) & vx_cmp(fu, vw, VX_CMP_LE) & vx_cmp(zero, fv, VX_CMP_LE) & vx_cmp(fv, vh, VX_CMP_LE) & vx_cmp(vx_add(ramp, vx_bcast(&fx0)), vlast, VX_CMP_LE) & (z0 > 0.0L && z1 > 0.0L ? VX_ALL : (vxk)0));
            const vxf cu = vx_min(vx_max(fu, zero), vw), cv = vx_min(vx_max(fv, zero), vh);
            /* Below 2^24 at any size here: an address is exact in F32. */
            const vxf at = vx_fma(cv, vs, cu);
            const vxf s00 = vx_gather(v->ncur.p, vx_to_idx(at));
            const vxf s10 = vx_gather(v->ncur.p, vx_to_idx(vx_add(at, vone)));
            const vxf s01 = vx_gather(v->ncur.p, vx_to_idx(vx_add(at, vs)));
            const vxf s11 = vx_gather(v->ncur.p, vx_to_idx(vx_add(vx_add(at, vs), vone)));
            const vxf tu = vx_sub(us, fu), tv = vx_sub(vy, fv);
            const vxf top = vx_fma(tu, vx_sub(s10, s00), s00), bot = vx_fma(tu, vx_sub(s11, s01), s01);
            const vxf s = vx_blend(ok, zero, vx_fma(tv, vx_sub(bot, top), top));
            const vxf rv = vx_blend(ok, zero, vx_load(r + x));
            const vxf e = vx_sub(s, rv);
            /* Tukey's biweight: no division, and exactly nothing from whatever moved. */
            const vxf u = vx_mul(vx_mul(e, e), vic2);
            const vxf k = vx_max(vx_sub(vone, u), zero);
            const vxf wt = vx_blend(ok, zero, vx_mul(k, k));
            const vxf g0 = vx_load(j0 + x), g1 = vx_load(j1 + x), g2 = vx_load(j2 + x);
            const vxf w0 = vx_mul(wt, g0), w1 = vx_mul(wt, g1), w2 = vx_mul(wt, g2);
            t[T_H00] = vx_fma(w0, g0, t[T_H00]);
            t[T_H01] = vx_fma(w0, g1, t[T_H01]);
            t[T_H02] = vx_fma(w0, g2, t[T_H02]);
            t[T_H11] = vx_fma(w1, g1, t[T_H11]);
            t[T_H12] = vx_fma(w1, g2, t[T_H12]);
            t[T_H22] = vx_fma(w2, g2, t[T_H22]);
            t[T_G0] = vx_fma(w0, e, t[T_G0]);
            t[T_G1] = vx_fma(w1, e, t[T_G1]);
            t[T_G2] = vx_fma(w2, e, t[T_G2]);
            t[T_WEIGHT] = vx_add(t[T_WEIGHT], wt);
            t[T_RR] = vx_fma(rv, rv, t[T_RR]);
            t[T_SS] = vx_fma(s, s, t[T_SS]);
            t[T_RS] = vx_fma(rv, s, t[T_RS]);
            t[T_IN] = vx_add(t[T_IN], vx_blend(ok, zero, vone));
        }
        float *out = a->rows + (size_t)i * T_TERMS;
        for (int k = 0; k < T_TERMS; k++)
            out[k] = vx_hsum(t[k]);
    }
}

/* Row sums added in row order, so the thread count changes nothing. */
static void gather_rows(const struct att *a, real *sum)
{
    const struct att_level *v = &a->lv[a->level];
    const int rows = (v->nref.h + v->row_step - 1) / v->row_step;
    for (int k = 0; k < T_TERMS; k++)
        sum[k] = 0.0L;
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < T_TERMS; k++)
            sum[k] += (real)a->rows[(size_t)i * T_TERMS + (size_t)k];
}

/* The 3 x 3 normal equations, a little damped: a picture that cannot see a turn does not guess one. 0 when it has no information at all. */
static int solve3(const real *t, real *d)
{
    const real h00 = t[T_H00] * (1.0L + DAMP), h11 = t[T_H11] * (1.0L + DAMP), h22 = t[T_H22] * (1.0L + DAMP), h01 = t[T_H01], h02 = t[T_H02], h12 = t[T_H12];
    const real c00 = h11 * h22 - h12 * h12, c01 = h02 * h12 - h01 * h22, c02 = h01 * h12 - h02 * h11;
    const real det = h00 * c00 + h01 * c01 + h02 * c02;
    if (!(det > 1e-12L))
        return 0;
    const real c11 = h00 * h22 - h02 * h02, c12 = h01 * h02 - h00 * h12, c22 = h00 * h11 - h01 * h01;
    d[0] = (c00 * t[T_G0] + c01 * t[T_G1] + c02 * t[T_G2]) / det;
    d[1] = (c01 * t[T_G0] + c11 * t[T_G1] + c12 * t[T_G2]) / det;
    d[2] = (c02 * t[T_G0] + c12 * t[T_G1] + c22 * t[T_G2]) / det;
    return 1;
}

static void measure_at(struct att *a, const real *q, int level, float c, att_round round, void *ctx, real *sum)
{
    a->level = level;
    a->kernel_c = c;
    q_to_matrix(q, a->O);
    round(ctx, round_accumulate, a);
    gather_rows(a, sum);
}

static real correlation(const real *t)
{
    const real d = t[T_RR] * t[T_SS];
    return d > 0.0L ? t[T_RS] / __builtin_sqrtl(d) : 0.0L;
}

void att_q_mul(const double *a, const double *b, double *o)
{
    const real ra[4] = {a[0], a[1], a[2], a[3]}, rb[4] = {b[0], b[1], b[2], b[3]};
    real r[4];
    q_mul(ra, rb, r);
    q_normalize(r);
    for (int i = 0; i < 4; i++)
        o[i] = (double)r[i];
}

uint64_t att_fix_hash(const struct att_fix *f, uint64_t h)
{
    const double v[8] = {f->q[0], f->q[1], f->q[2], f->q[3], f->corr, f->corr_identity, f->used, f->view};
    const unsigned char *b = (const unsigned char *)v;
    const size_t bytes = sizeof v;
    for (size_t i = 0; i < bytes; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void att_measure(struct att *a, const struct vt_plane *cur, int search, att_round round, void *round_ctx, struct att_fix *fix)
{
    memset(fix, 0, sizeof *fix);
    pyramid(a, cur, round, round_ctx);
    if (search) {
        round(round_ctx, round_search, a);
        int best = -1;
        float top = -1e29f;
        for (int u = 0; u < SEARCH_UNITS; u++)
            if (a->search[u] > top) {
                top = a->search[u];
                best = u;
            }
        const int across = 2 * SEARCH_X + 1;
        fix->peak_x = best < 0 ? 0 : best % across - SEARCH_X;
        fix->peak_y = best < 0 ? 0 : best / across - SEARCH_Y;
        fix->peak_z = (double)top;
        /* The shift as a turn: the optical axis is carried to where the picture's centre went, by the shortest way. */
        const struct att_level *v = &a->lv[ATT_LEVELS - 1];
        const real tx = (real)fix->peak_x / v->f, ty = (real)fix->peak_y / v->f, len = __builtin_sqrtl(tx * tx + ty * ty + 1.0L);
        a->q[0] = 1.0L + 1.0L / len;
        a->q[1] = -ty / len;
        a->q[2] = tx / len;
        a->q[3] = 0.0L;
        q_normalize(a->q);
    }
    real sum[T_TERMS];
    const int first = search ? ATT_LEVELS - 1 : 1;
    for (int l = first; l >= 0; l--) {
        for (int it = 0; it < MAX_STEPS[l]; it++) {
            measure_at(a, a->q, l, KERNEL_C[l], round, round_ctx, sum);
            real d[3];
            if (!solve3(sum, d))
                break;
            fix->steps++;
            /* Inverse compositional: the reference moved by d explains the residual, so the warp takes the opposite turn. First order, then renormalized. */
            const real dq[4] = {1.0L, -0.5L * d[0], -0.5L * d[1], -0.5L * d[2]};
            real nq[4];
            q_mul(a->q, dq, nq);
            q_normalize(nq);
            memcpy(a->q, nq, sizeof nq);
            if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < STEP_DONE * STEP_DONE)
                break;
        }
    }
    measure_at(a, a->q, 0, KERNEL_C[0], round, round_ctx, sum);
    fix->corr = (double)correlation(sum);
    fix->used = sum[T_IN] > 0.0L ? (double)(sum[T_WEIGHT] / sum[T_IN]) : 0.0;
    const struct att_level *fine = &a->lv[0];
    const int fine_rows = (fine->nref.h + fine->row_step - 1) / fine->row_step;
    fix->view = (double)(sum[T_IN] / ((real)fine->nref.w * (real)fine_rows));
    static const real identity[4] = {1.0L, 0.0L, 0.0L, 0.0L};
    measure_at(a, identity, 0, KERNEL_C[0], round, round_ctx, sum);
    fix->corr_identity = (double)correlation(sum);
    /* A turn must earn its place. Measured on a camera that had not moved, with the room's light changed and someone in view: the search found a false 4.4 degrees scoring 0.038 where no turn at all scored
     * 0.100, and with someone close, a false 0.3 degrees that gained 0.05. This is also the hysteresis: a change too small to show in the picture changes nothing. */
    fix->held = !((real)fix->corr > (real)fix->corr_identity + (real)ATT_MIN_GAIN);
    /* And it must be a turn at all: a still camera's fit jitters by hundredths of a degree (measured live, D, 2026-11-03), and carried into the chain that noise would walk. */
    if (a->q[1] * a->q[1] + a->q[2] * a->q[2] + a->q[3] * a->q[3] < ATT_MIN_TURN_SQ)
        fix->held = 1;
    if (fix->held)
        memcpy(a->q, identity, sizeof identity);
    q_to_matrix(a->q, a->O);
    for (int i = 0; i < 9; i++)
        fix->O[i] = (double)a->O[i];
    for (int i = 0; i < 4; i++)
        fix->q[i] = (double)a->q[i];
}
