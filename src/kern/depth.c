/* depth.c: see depth.h. */
#include "depth.h"

#include <math.h>
#include <string.h>

#include "vx.h"

#define CHUNK_ROWS 16
#define BOX_PAD 16 /* lanes each side of a row of vertical sums */

typedef long double real;

struct depth {
    struct depth_cfg c;
    int stride;     /* lanes */
    int chunks;
    float *raw;     /* planes x h x stride */
    float *agg;
    float *z, *conf;
    char *scratch;  /* one per thread */
    size_t scratch_each;
    struct depth_view views[DEPTH_MAX_VIEWS];
    int nviews;
    const struct vt_plane *ref, *ref_spread;
};

static const float RAMP[16] VX_ALIGNED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
static const uint32_t ABS_MASK VX_ALIGNED = 0x7fffffffu;

static inline float *vol_row(const struct depth *d, float *vol, int k, int y) { return vol + ((size_t)k * (size_t)d->c.h + (size_t)y) * (size_t)d->stride; }
static inline real inv_z_at(const struct depth_cfg *c, real k)
{
    const real near_i = 1.0L / (real)c->z_near, far_i = 1.0L / (real)c->z_far;
    return near_i + (far_i - near_i) * k / (real)(c->planes - 1);
}

size_t depth_bytes(const struct depth_cfg *c)
{
    const size_t stride = (size_t)((c->w + 15) & ~15);
    const size_t row = stride * sizeof(float);
    const size_t scratch = (size_t)c->threads * (6u * row + (size_t)(2 * BOX_PAD + 32) * sizeof(float) * 6u + 4096u);
    return 2u * (size_t)c->planes * (size_t)c->h * row + 2u * (size_t)c->h * row + scratch + sizeof(struct depth) + 4096u;
}

struct depth *depth_create(const struct depth_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx)
{
    struct depth *d = alloc(ctx, (sizeof *d + 63) & ~(size_t)63);
    if (!d)
        return NULL;
    memset(d, 0, sizeof *d);
    d->c = *c;
    d->stride = (c->w + 15) & ~15;
    d->chunks = (c->h + CHUNK_ROWS - 1) / CHUNK_ROWS;
    const size_t row = (size_t)d->stride * sizeof(float);
    d->raw = alloc(ctx, (size_t)c->planes * (size_t)c->h * row);
    d->agg = alloc(ctx, (size_t)c->planes * (size_t)c->h * row);
    d->z = alloc(ctx, (size_t)c->h * row);
    d->conf = alloc(ctx, (size_t)c->h * row);
    /* Six rows of vertical sums and working values, padded. */
    d->scratch_each = (6u * (row + (size_t)(2 * BOX_PAD) * sizeof(float)) + 4095u) & ~(size_t)63;
    d->scratch = alloc(ctx, (size_t)c->threads * d->scratch_each);
    if (!d->raw || !d->agg || !d->z || !d->conf || !d->scratch)
        return NULL;
    return d;
}

size_t depth_norm_scratch_bytes(int w)
{
    const size_t lanes = (size_t)((w + 15) & ~15) + 2u * BOX_PAD;
    return (4u * lanes * sizeof(float) + 4095u) & ~(size_t)63;
}

/* The box over a row of vertical sums: out[x] = sum of v[x - r .. x + r]. v has BOX_PAD lanes valid each side. */
static void box_row(float *out, const float *v, int w, int r)
{
    for (int x = 0; x < w; x += VX_LANES) {
        vxf s = vx_loadu(v + x - r);
        for (int i = -r + 1; i <= r; i++)
            s = vx_add(s, vx_loadu(v + x + i));
        vx_store(out + x, s);
    }
}

void depth_normalize(const struct vt_plane *dst, const struct vt_plane *spread, const struct vt_plane *src, float eps, int y0, int y1, void *scratch)
{
    const int r = DEPTH_NORM_RADIUS, w = src->w;
    const int lanes = ((w + 15) & ~15) + 2 * BOX_PAD;
    float *base = scratch;
    float *v1 = base + BOX_PAD, *v2 = base + lanes + BOX_PAD, *s1 = base + 2 * lanes + BOX_PAD, *s2 = base + 3 * lanes + BOX_PAD;
    const real count = (real)((2 * r + 1) * (2 * r + 1));
    const int from = -BOX_PAD, to = ((w + 15) & ~15) + BOX_PAD;
    for (int y = y0; y < y1; y++) {
        /* Vertical sums of the pixels and their squares, lanes the padded plane holds. */
        for (int x = from; x < to; x += VX_LANES) {
            vxf a = vx_zero(), b = vx_zero();
            for (int j = -r; j <= r; j++) {
                const vxf p = vx_loadu(plane_row(src, y + j) + x);
                a = vx_add(a, p);
                b = vx_fma(p, p, b);
            }
            vx_store(v1 + x, a);
            vx_store(v2 + x, b);
        }
        box_row(s1, v1, w, r);
        box_row(s2, v2, w, r);
        const float *in = plane_row(src, y);
        float *out = plane_row(dst, y), *sp = spread ? plane_row(spread, y) : NULL;
        for (int x = 0; x < w; x++) {
            const real mean = (real)s1[x] / count;
            const real var = (real)s2[x] / count - mean * mean;
            const real sd = var > 0.0L ? sqrtl(var) : 0.0L;
            out[x] = (float)(((real)in[x] - mean) / (sd + (real)eps));
            if (sp)
                sp[x] = (float)sd;
        }
    }
}

void depth_normalize_part(const struct vt_plane *dst, const struct vt_plane *spread, const struct vt_plane *src, float eps, int tid, int n, char *scratch, size_t scratch_each)
{
    const int h = src->h;
    depth_normalize(dst, spread, src, eps, h * tid / n, h * (tid + 1) / n, scratch + (size_t)tid * scratch_each);
}

void depth_relate(const double *Rb, const double *Tb, const double *Rd, const double *Td, double *R, double *T)
{
    /* X_b = Rb Rd' (X_d - Td) + Tb */
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R[3 * r + c] = (double)((real)Rb[3 * r] * (real)Rd[3 * c] + (real)Rb[3 * r + 1] * (real)Rd[3 * c + 1] + (real)Rb[3 * r + 2] * (real)Rd[3 * c + 2]);
    for (int r = 0; r < 3; r++)
        T[r] = (double)((real)Tb[r] - ((real)R[3 * r] * (real)Td[0] + (real)R[3 * r + 1] * (real)Td[1] + (real)R[3 * r + 2] * (real)Td[2]));
}

void depth_views_from_calib(const struct vt_calib *cal, int scale, const struct vt_plane *norm_r, const struct vt_plane *norm_l, struct depth_view *views)
{
    static const double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, O[3] = {0, 0, 0};
    const real s = (real)scale;
    struct depth_view *r = &views[0], *l = &views[1];
    r->norm = norm_r;
    r->f = (double)((real)cal->fb / s);
    r->cx = (double)((real)cal->cxb / s);
    r->cy = (double)((real)cal->cyb / s);
    r->weight = DEPTH_WEIGHT_R;
    depth_relate(cal->R, cal->T, cal->Rd, cal->Td, r->R, r->T);
    l->norm = norm_l;
    l->f = (double)((real)cal->fa / s);
    l->cx = (double)((real)cal->cxa / s);
    l->cy = (double)((real)cal->cya / s);
    l->weight = DEPTH_WEIGHT_L;
    /* L is the world. */
    depth_relate(I, O, cal->Rd, cal->Td, l->R, l->T);
}

void depth_set_ref(struct depth *d, double f, double cx, double cy)
{
    d->c.f = f;
    d->c.cx = cx;
    d->c.cy = cy;
}

void depth_set_views(struct depth *d, const struct vt_plane *ref_norm, const struct vt_plane *ref_spread, const struct depth_view *views, int nviews)
{
    d->ref = ref_norm;
    d->ref_spread = ref_spread;
    d->nviews = nviews < DEPTH_MAX_VIEWS ? nviews : DEPTH_MAX_VIEWS;
    for (int i = 0; i < d->nviews; i++)
        d->views[i] = views[i];
}

/* Where (x, y) of the reference lands in a view at inverse depth iz. Off the plane behind the camera: far outside any image. */
static inline void project(const struct depth_view *v, real bx, real by, real iz, float *px, float *py)
{
    const real X = (real)v->R[0] * bx + (real)v->R[1] * by + (real)v->R[2] + (real)v->T[0] * iz;
    const real Y = (real)v->R[3] * bx + (real)v->R[4] * by + (real)v->R[5] + (real)v->T[1] * iz;
    const real Z = (real)v->R[6] * bx + (real)v->R[7] * by + (real)v->R[8] + (real)v->T[2] * iz;
    if (Z <= 1e-6L) {
        *px = -1e6f;
        *py = -1e6f;
        return;
    }
    *px = (float)((real)v->f * X / Z + (real)v->cx);
    *py = (float)((real)v->f * Y / Z + (real)v->cy);
}

static void sweep_rows(struct depth *d, int k, int y0, int y1)
{
    const struct depth_cfg *c = &d->c;
    const real iz = inv_z_at(c, (real)k);
    const real rf = (real)c->f, rcx = (real)c->cx, rcy = (real)c->cy;
    const vxf ramp = vx_load(RAMP);
    const vxf zero = vx_zero();
    const float trunc_v VX_ALIGNED = c->trunc;
    const vxf vtrunc = vx_bcast(&trunc_v);
    for (int y = y0; y < y1; y++) {
        const float *ref = plane_row(d->ref, y);
        float *out = vol_row(d, d->raw, k, y);
        const real by = ((real)y - rcy) / rf;
        for (int x = 0; x < c->w; x += VX_LANES) {
            /* A homography, nearly affine over 16 pixels of a coarse level: exact at both ends of the block, one division each, straight between. */
            vxf cost = vx_zero();
            const vxf rv = vx_load(ref + x);
            for (int i = 0; i < d->nviews; i++) {
                const struct depth_view *v = &d->views[i];
                float x0, y0p, x1, y1p;
                project(v, ((real)x - rcx) / rf, by, iz, &x0, &y0p);
                project(v, ((real)(x + VX_LANES) - rcx) / rf, by, iz, &x1, &y1p);
                const float dx VX_ALIGNED = (float)(((real)x1 - (real)x0) / (real)VX_LANES), dy VX_ALIGNED = (float)(((real)y1p - (real)y0p) / (real)VX_LANES);
                const float ax VX_ALIGNED = x0, ay VX_ALIGNED = y0p;
                const float wm1 VX_ALIGNED = (float)(v->norm->w - 1), hm1 VX_ALIGNED = (float)(v->norm->h - 1), stride VX_ALIGNED = (float)v->norm->stride, weight VX_ALIGNED = v->weight;
                const vxf xs = vx_fma(ramp, vx_bcast(&dx), vx_bcast(&ax));
                const vxf ys = vx_fma(ramp, vx_bcast(&dy), vx_bcast(&ay));
                const vxf rx = vx_round(xs, VX_RND_NEAREST), ry = vx_round(ys, VX_RND_NEAREST);
                const vxf vw = vx_bcast(&wm1), vh = vx_bcast(&hm1);
                const vxk ok = (vxk)(vx_cmp(zero, rx, VX_CMP_LE) & vx_cmp(rx, vw, VX_CMP_LE) & vx_cmp(zero, ry, VX_CMP_LE) & vx_cmp(ry, vh, VX_CMP_LE));
                const vxf cx = vx_min(vx_max(rx, zero), vw), cy = vx_min(vx_max(ry, zero), vh);
                /* Below 2^24 at any size here: an address is exact in F32. */
                const vxf at = vx_fma(cy, vx_bcast(&stride), cx);
                const vxf s = vx_gather(v->norm->p, vx_to_idx(at));
                const vxf diff = vx_abs(vx_sub(rv, s), &ABS_MASK);
                const vxf e = vx_blend(ok, vtrunc, vx_min(diff, vtrunc));
                cost = vx_fma(e, vx_bcast(&weight), cost);
            }
            vx_store(out + x, cost);
        }
    }
}

void depth_sweep(struct depth *d, int tid, int n)
{
    const int units = d->c.planes * d->chunks;
    for (int u = tid; u < units; u += n) {
        const int k = u / d->chunks, ch = u % d->chunks;
        const int y0 = ch * CHUNK_ROWS, y1 = y0 + CHUNK_ROWS < d->c.h ? y0 + CHUNK_ROWS : d->c.h;
        sweep_rows(d, k, y0, y1);
    }
}

void depth_aggregate(struct depth *d, int tid, int n)
{
    const int units = d->c.planes * d->chunks, w = d->c.w, h = d->c.h, r = d->c.agg_radius;
    float *base = (float *)(void *)(d->scratch + (size_t)tid * d->scratch_each);
    float *v = base + BOX_PAD;
    for (int u = tid; u < units; u += n) {
        const int k = u / d->chunks, ch = u % d->chunks;
        const int y0 = ch * CHUNK_ROWS, y1 = y0 + CHUNK_ROWS < h ? y0 + CHUNK_ROWS : h;
        for (int y = y0; y < y1; y++) {
            for (int x = 0; x < w; x += VX_LANES) {
                vxf s = vx_zero();
                for (int j = -r; j <= r; j++) {
                    const int yy = y + j < 0 ? 0 : y + j >= h ? h - 1 : y + j;
                    s = vx_add(s, vx_load(vol_row(d, d->raw, k, yy) + x));
                }
                vx_store(v + x, s);
            }
            /* Edges replicated: a window at the border is mostly its border. */
            for (int i = 1; i <= BOX_PAD; i++) {
                v[-i] = v[0];
                v[w - 1 + i] = v[w - 1];
            }
            box_row(vol_row(d, d->agg, k, y), v, w, r);
        }
    }
}

void depth_decide(struct depth *d, int tid, int n)
{
    const struct depth_cfg *c = &d->c;
    const int w = c->w, h = c->h;
    float *base = (float *)(void *)(d->scratch + (size_t)tid * d->scratch_each);
    const int lanes = d->stride + 2 * BOX_PAD;
    float *best_a = base, *idx_a = base + lanes, *sec_a = base + 2 * lanes, *cm_a = base + 3 * lanes, *cp_a = base + 4 * lanes, *inb_a = base + 5 * lanes;
    /* "A different depth" as a distance in inverse depth, not a count of planes: 1.5 of 64. With planes packed more densely a neighbour is the same match seen again, and would read as doubt: 512 planes halved the
     * pixels that were sure, on the same frames. */
    const float big VX_ALIGNED = 1e30f, apart VX_ALIGNED = (float)(1.5L * (real)c->planes / 64.0L);
    const vxf vbig = vx_bcast(&big), v15 = vx_bcast(&apart);
    /* A plane is evidence only where its aggregated cost is under what a window full of out-of-image or saturated samples costs. Measured on the real room: a door L never saw got a confident depth, the few
     * planes that landed inside L winning by default against all the planes that did not. */
    float weight_sum = 0.0f;
    for (int i = 0; i < d->nviews; i++)
        weight_sum += d->views[i].weight;
    const int side = 2 * c->agg_radius + 1;
    const float evidence VX_ALIGNED = 0.98f * c->trunc * weight_sum * (float)(side * side);
    const vxf vevid = vx_bcast(&evidence);
    uint32_t *tally = (uint32_t *)(void *)(inb_a + lanes);
    tally[0] = tally[1] = tally[2] = 0;
    for (int y = tid; y < h; y += n) {
        for (int x = 0; x < w; x += VX_LANES) {
            vxf best = vbig, bidx = vx_zero();
            for (int k = 0; k < c->planes; k++) {
                const vxf e = vx_load(vol_row(d, d->agg, k, y) + x);
                const float kf VX_ALIGNED = (float)k;
                const vxk m = vx_cmp(e, best, VX_CMP_LT);
                best = vx_blend(m, best, e);
                bidx = vx_blend(m, bidx, vx_bcast(&kf));
            }
            vxf sec = vbig, cm = vbig, cp = vbig, inb = vx_zero();
            for (int k = 0; k < c->planes; k++) {
                const vxf e = vx_load(vol_row(d, d->agg, k, y) + x);
                const float kf VX_ALIGNED = (float)k, one VX_ALIGNED = 1.0f;
                const vxf vk = vx_bcast(&kf), v1 = vx_bcast(&one);
                inb = vx_blend(vx_cmp(e, vevid, VX_CMP_LT), inb, vx_add(inb, v1));
                const vxf delta = vx_sub(vk, bidx);
                const vxk below = vx_cmp(vx_add(delta, v1), vx_zero(), VX_CMP_EQ), above = vx_cmp(vx_sub(delta, v1), vx_zero(), VX_CMP_EQ);
                cm = vx_blend(below, cm, e);
                cp = vx_blend(above, cp, e);
                /* The best match at any depth more than one plane from the winner. */
                const vxk far = (vxk)(vx_cmp(v15, delta, VX_CMP_LT) | vx_cmp(delta, vx_sub(vx_zero(), v15), VX_CMP_LT));
                sec = vx_blend(far, sec, vx_min(sec, e));
            }
            vx_store(best_a + x, best);
            vx_store(idx_a + x, bidx);
            vx_store(sec_a + x, sec);
            vx_store(cm_a + x, cm);
            vx_store(cp_a + x, cp);
            vx_store(inb_a + x, inb);
        }
        float *zo = d->z + (size_t)y * (size_t)d->stride, *co = d->conf + (size_t)y * (size_t)d->stride;
        const float *sp = d->ref_spread ? plane_row(d->ref_spread, y) : NULL;
        for (int x = 0; x < w; x++) {
            const long k = lrintf(idx_a[x]);
            const real b = (real)best_a[x], s = (real)sec_a[x], m = (real)cm_a[x], p = (real)cp_a[x];
            const real conf = s > 0.0L && s < 1e29L ? 1.0L - b / s : 0.0L;
            real z = 0.0L;
            /* A winner at the end of the range is no measurement: the true surface may lie beyond it. */
            if (k > 0 && k < c->planes - 1 && conf >= (real)c->conf_min && (!sp || sp[x] >= c->tex_min) && 2.0f * inb_a[x] >= (float)c->planes) {
                const real den = m - 2.0L * b + p;
                real off = den > 1e-9L ? 0.5L * (m - p) / den : 0.0L;
                off = off > 0.5L ? 0.5L : off < -0.5L ? -0.5L : off;
                z = 1.0L / inv_z_at(c, (real)k + off);
            }
            zo[x] = (float)z;
            co[x] = (float)conf;
            const uint32_t textured = !sp || sp[x] >= c->tex_min;
            tally[0] += textured;
            tally[1] += textured & (uint32_t)(conf >= 0.2L);
            tally[2] += (uint32_t)(z > 0.0L);
        }
    }
}

void depth_counts(const struct depth *d, int n, uint32_t *textured, uint32_t *confident, uint32_t *known)
{
    const int lanes = d->stride + 2 * BOX_PAD;
    uint32_t t = 0, s = 0, k = 0;
    for (int i = 0; i < n; i++) {
        const uint32_t *tally = (const uint32_t *)(const void *)(d->scratch + (size_t)i * d->scratch_each + 6u * (size_t)lanes * sizeof(float));
        t += tally[0];
        s += tally[1];
        k += tally[2];
    }
    *textured = t;
    *confident = s;
    *known = k;
}

const float *depth_z(const struct depth *d) { return d->z; }
const float *depth_conf(const struct depth *d) { return d->conf; }
int depth_stride(const struct depth *d) { return d->stride; }
