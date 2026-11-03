/* track.c: see track.h.
 *
 * Every scalar computation is long double, and a float result is one explicit cast: x87 on the card and on the host alike. A float expression would run on x87 with excess
 * precision on the card and on SSE on the host, and whole-run replay would part ways at the first inexact sum. */
#include "track.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "detect.h"
#include "img.h"
#include "kalman.h"
#include "lk.h"
#include "stereo.h"
#include "vt_util.h"
#include "vx.h"

#define APP_W 32 /* the appearance template: lanes, rows at its pyramid level */
#define APP_H 21
#define MAX_SETS 4

typedef long double real;
#define R(x) ((real)(x))

struct pyr {
    struct vt_plane l[VT_MAX_LEVELS];
};

struct job;

struct tracker {
    struct vt_config c;
    struct lk_cfg lk;
    struct stereo_geom g;
    tk_run_fn run_pyr, run_track;
    void *pyr_ctx, *track_ctx;
    struct job *jp, *jt;
    int levels, stereo_level, nsets;
    int w, h;
    struct pyr pyr[MAX_SETS][2]; /* set, camera */
    struct vt_plane input[2], diff; /* before the pyramid, when preprocessing needs a second plane */
    void *tiles[256];
    int threads_track;
    void *det_scratch;
    /* build keys: only tk_build touches these */
    uint32_t b_contrast, b_pre_gauss;
    int debug_level;
    /* the frame in hand */
    int cur, prev; /* sets; prev is -1 before the first frame */
    /* region */
    uint32_t next_id, id, state, missing;
    float cx, cy, rw, rh, rate_u, rate_v;
    uint64_t last_obs_ns, last_acq_ns;
    /* features, structure of arrays */
    int nf;
    float *fx, *fy, *fxr, *fyr, *fdisp, *fsigma, *fu, *fv, *fz;
    uint32_t *fflag;
    struct lk_out *flow;
    long double *p3;
    /* appearance, for taking a lost region back */
    float *app;
    float app_norm, app_fx, app_fy; /* the template's sub-pixel offset from the region's centre */
    int app_level, have_app;
    /* filter */
    struct kf kf;
    uint64_t last_meas_ns;
};

/* ---- parallel jobs ---- */

enum { OP_U8, OP_NORM_IN, OP_GAUSS_IN, OP_REDUCE, OP_DIFF, OP_LK, OP_STEREO, OP_SEARCH };

struct job {
    struct tracker *t;
    int op, level, set, alg;
    const uint8_t *src[2];
    int cameras;
    float gain_offset[2];
    int next VT_CACHELINE;     /* dynamic work: points, candidate rows */
    float partial[256 * 16] VT_CACHELINE; /* per-thread results, a line each */
    int radius, xc, yc;
};

static inline int row_of(int rows, int tid, int n) { return (int)((long)rows * tid / n); }

static void stereo_one(struct tracker *t, int i, int alg, void *tile);

/* The correlation of the stored appearance at (x, y) of the level, in long double on both machines. */
static real search_at(const struct vt_plane *pl, const float *app, real norm_t, int x, int y)
{
    if (x < 1 || y < 1 || x + APP_W >= pl->w - 1 || y + APP_H >= pl->h - 1)
        return -1.0L;
    vxf st = vx_zero(), sj = vx_zero(), sjj = vx_zero();
    for (int r = 0; r < APP_H; r++) {
        const float *row = plane_row(pl, y + r) + x;
        for (int k = 0; k < APP_W; k += VX_LANES) {
            const vxf v = vx_loadu(row + k);
            st = vx_fma(v, vx_load(app + r * APP_W + k), st);
            sj = vx_add(sj, v);
            sjj = vx_fma(v, v, sjj);
        }
    }
    const real n = R(APP_W * APP_H);
    const real a = R(vx_hsum(st)), b = R(vx_hsum(sj)), c = R(vx_hsum(sjj));
    const real var = c - b * b / n;
    return (var <= 0.0L || norm_t <= 0.0L) ? 0.0L : a / sqrtl(norm_t * var);
}

static void job_main(void *arg, int tid, int n)
{
    struct job *j = arg;
    struct tracker *t = j->t;
    const int x87 = j->alg == VT_ALG_X87;
    switch (j->op) {
    case OP_U8:
        for (int c = 0; c < j->cameras; c++) {
            const struct vt_plane *d = (t->b_pre_gauss || t->b_contrast) ? &t->input[c] : &t->pyr[j->set][c].l[0];
            const int y0 = row_of(d->h, tid, n), y1 = row_of(d->h, tid + 1, n);
            if (x87)
                img_u8_to_q6_ref(d, j->src[c], (int)t->c.width, y0, y1);
            else
                img_u8_to_q6_vx(d, j->src[c], (int)t->c.width, y0, y1);
            img_pad_rows(d, y0, y1);
        }
        break;
    case OP_NORM_IN:
        for (int c = 0; c < j->cameras; c++) {
            const struct vt_plane *s = &t->input[c], *d = t->b_pre_gauss ? s : &t->pyr[j->set][c].l[0];
            const int y0 = row_of(d->h, tid, n), y1 = row_of(d->h, tid + 1, n);
            if (x87)
                img_norm_ref(d, s, j->gain_offset, y0, y1);
            else
                img_norm_vx(d, s, j->gain_offset, y0, y1);
            img_pad_rows(d, y0, y1);
        }
        break;
    case OP_GAUSS_IN:
        for (int c = 0; c < j->cameras; c++) {
            const struct vt_plane *d = &t->pyr[j->set][c].l[0];
            const int y0 = row_of(d->h, tid, n), y1 = row_of(d->h, tid + 1, n);
            if (x87)
                img_gauss5_ref(d, &t->input[c], y0, y1);
            else
                img_gauss5_vx(d, &t->input[c], y0, y1);
            img_pad_rows(d, y0, y1);
        }
        break;
    case OP_REDUCE:
        for (int c = 0; c < j->cameras; c++) {
            const struct vt_plane *d = &t->pyr[j->set][c].l[j->level], *s = &t->pyr[j->set][c].l[j->level - 1];
            const int y0 = row_of(d->h, tid, n), y1 = row_of(d->h, tid + 1, n);
            if (x87)
                img_reduce_ref(d, s, y0, y1);
            else
                img_reduce_vx(d, s, y0, y1);
            img_pad_rows(d, y0, y1);
        }
        break;
    case OP_DIFF: {
        const struct vt_plane *a = &t->pyr[t->cur][0].l[0], *b = &t->pyr[t->prev][0].l[0];
        const int y0 = row_of(a->h, tid, n), y1 = row_of(a->h, tid + 1, n);
        img_absdiff_vx(&t->diff, a, b, y0, y1);
        real sum = 0.0L;
        for (int y = y0; y < y1; y++) {
            const float *r = plane_row(&t->diff, y);
            vxf s = vx_zero();
            for (int x = 0; x < (a->w & ~15); x += VX_LANES)
                s = vx_add(s, vx_load(r + x));
            sum += R(vx_hsum(s));
        }
        j->partial[tid * 16] = (float)sum;
        break;
    }
    case OP_LK: {
        const struct vt_plane *I = t->pyr[t->prev][0].l, *J = t->pyr[t->cur][0].l;
        for (;;) {
            const int i = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (i >= t->nf)
                break;
            if (x87) {
                lk_track_ref(I, J, t->levels - 1, &t->lk, t->fx[i], t->fy[i], 0.0f, 0.0f, &t->flow[i], t->tiles[tid]);
                if (t->c.fb_max > 0.0f)
                    lk_flowback_ref(I, J, t->levels - 1, &t->lk, t->fx[i], t->fy[i], t->c.fb_max, &t->flow[i], t->tiles[tid]);
            } else {
                lk_track_vx(I, J, t->levels - 1, &t->lk, t->fx[i], t->fy[i], 0.0f, 0.0f, &t->flow[i], t->tiles[tid]);
                if (t->c.fb_max > 0.0f)
                    lk_flowback_vx(I, J, t->levels - 1, &t->lk, t->fx[i], t->fy[i], t->c.fb_max, &t->flow[i], t->tiles[tid]);
            }
        }
        break;
    }
    case OP_STEREO:
        for (;;) {
            const int i = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (i >= t->nf)
                break;
            stereo_one(t, i, j->alg, t->tiles[tid]);
        }
        break;
    case OP_SEARCH: {
        /* The best correlation of this thread's candidate rows; a line of `partial` per thread. */
        float *out = &j->partial[tid * 16];
        real best = -2.0L;
        int bx = 0, by = 0;
        const struct vt_plane *pl = &t->pyr[t->cur][0].l[t->app_level];
        for (;;) {
            const int r = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (r > 2 * j->radius)
                break;
            const int y = j->yc - j->radius + r;
            for (int x = j->xc - j->radius; x <= j->xc + j->radius; x++) {
                const real v = search_at(pl, t->app, R(t->app_norm), x, y);
                /* Ties broken by position: the same answer at any thread count. */
                if (v > best || (v == best && (y < by || (y == by && x < bx)))) {
                    best = v;
                    bx = x;
                    by = y;
                }
            }
        }
        out[0] = (float)best;
        out[1] = (float)bx;
        out[2] = (float)by;
        break;
    }
    default:
        break;
    }
}

static void run_build_op(struct tracker *t, int op, int level)
{
    struct job *j = t->jp;
    j->op = op;
    j->level = level;
    j->next = 0;
    t->run_pyr(t->pyr_ctx, job_main, j);
}

static void run_track_op(struct tracker *t, int op, int alg)
{
    struct job *j = t->jt;
    j->op = op;
    j->alg = alg;
    j->next = 0;
    t->run_track(t->track_ctx, job_main, j);
}

/* ---- setup ---- */

static void *need(tk_alloc_fn alloc, void *ctx, size_t bytes) { return alloc(ctx, (bytes + 63) & ~(size_t)63); }

static int zeroed_plane(struct vt_plane *pl, int w, int h, tk_alloc_fn alloc, void *ctx)
{
    void *m = need(alloc, ctx, plane_bytes(w, h));
    if (!m) {
        fprintf(stderr, "track: no memory for a plane\n");
        return -1;
    }
    memset(m, 0, plane_bytes(w, h));
    *pl = plane_at(m, w, h);
    return 0;
}

struct tracker *tk_create(const struct vt_config *cfg, int threads_pyr, int threads_track, tk_alloc_fn alloc, void *actx, tk_run_fn run_pyr, void *pyr_ctx, tk_run_fn run_track, void *track_ctx)
{
    (void)threads_pyr;
    struct tracker *t = need(alloc, actx, sizeof *t);
    struct job *jp = need(alloc, actx, sizeof *jp), *jt = need(alloc, actx, sizeof *jt);
    if (!t || !jp || !jt) {
        fprintf(stderr, "track: no memory\n");
        return NULL;
    }
    memset(t, 0, sizeof *t);
    memset(jp, 0, sizeof *jp);
    memset(jt, 0, sizeof *jt);
    t->c = *cfg;
    t->run_pyr = run_pyr;
    t->pyr_ctx = pyr_ctx;
    t->run_track = run_track;
    t->track_ctx = track_ctx;
    t->jp = jp;
    t->jt = jt;
    jp->t = jt->t = t;
    t->w = (int)cfg->width;
    t->h = (int)cfg->height;
    t->levels = (int)cfg->levels;
    t->threads_track = threads_track;
    t->nsets = cfg->pipelines == 2 ? MAX_SETS : 2;
    t->b_contrast = cfg->contrast;
    t->b_pre_gauss = cfg->pre_gauss;
    t->debug_level = -1;
    t->prev = -1;
    if (t->levels < 1 || t->levels > VT_MAX_LEVELS || (t->w & 15) || cfg->max_features > VT_MAX_FEATURES || cfg->patch_w > LK_MAX_W || cfg->patch_h > LK_MAX_H || threads_track > 256) {
        fprintf(stderr, "track: config out of range\n");
        return NULL;
    }
    t->lk = (struct lk_cfg){.w = (int)cfg->patch_w, .h = (int)cfg->patch_h, .max_iter = (int)cfg->max_iter, .eps = cfg->eps, .min_eig = cfg->min_eig, .max_residual = cfg->max_residual, .prefetch = (int)cfg->prefetch};
    t->g = (struct stereo_geom){R(cfg->fx), R(cfg->fy), R(cfg->cx), R(cfg->cy), R(cfg->baseline)};
    /* The scan runs where the largest disparity is at most 64 pixels: its cost is candidates x patch. */
    t->stereo_level = 0;
    while (t->stereo_level < t->levels - 1 && R(cfg->max_disp) / R(1 << t->stereo_level) > 64.0L)
        t->stereo_level++;
    const int cams = cfg->stereo ? 2 : 1;
    for (int s = 0; s < t->nsets; s++)
        for (int c = 0; c < cams; c++) {
            int w = t->w, h = t->h;
            for (int L = 0; L < t->levels; L++) {
                if (zeroed_plane(&t->pyr[s][c].l[L], w, h, alloc, actx))
                    return NULL;
                w = (w + 1) / 2;
                h = (h + 1) / 2;
            }
        }
    /* The preprocessing planes exist whatever the config says: soft configuration can turn them on. */
    for (int c = 0; c < cams; c++)
        if (zeroed_plane(&t->input[c], t->w, t->h, alloc, actx))
            return NULL;
    if (cfg->frame_diff && zeroed_plane(&t->diff, t->w, t->h, alloc, actx))
        return NULL;
    /* One scratch tile a thread, each at a different line offset: identically aligned private blocks alias in the L2 (Cortex, 2026-10-25). */
    for (int i = 0; i < threads_track; i++) {
        char *m = need(alloc, actx, lk_tile_bytes() + 64 * 1024);
        if (!m)
            return NULL;
        t->tiles[i] = m + ((size_t)i * 104729u % 1024u) * 64u;
    }
    t->det_scratch = need(alloc, actx, detect_scratch_bytes(t->w, t->h));
    const size_t nf = VT_MAX_FEATURES;
    t->fx = need(alloc, actx, nf * sizeof(float));
    t->fy = need(alloc, actx, nf * sizeof(float));
    t->fxr = need(alloc, actx, nf * sizeof(float));
    t->fyr = need(alloc, actx, nf * sizeof(float));
    t->fdisp = need(alloc, actx, nf * sizeof(float));
    t->fsigma = need(alloc, actx, nf * sizeof(float));
    t->fu = need(alloc, actx, nf * sizeof(float));
    t->fv = need(alloc, actx, nf * sizeof(float));
    t->fz = need(alloc, actx, nf * sizeof(float));
    t->fflag = need(alloc, actx, nf * sizeof(uint32_t));
    t->flow = need(alloc, actx, nf * sizeof(struct lk_out));
    t->p3 = need(alloc, actx, nf * 3 * sizeof(long double));
    t->app = need(alloc, actx, APP_W * APP_H * sizeof(float));
    if (!t->det_scratch || !t->fx || !t->fy || !t->fxr || !t->fyr || !t->fdisp || !t->fsigma || !t->fu || !t->fv || !t->fz || !t->fflag || !t->flow || !t->p3 || !t->app) {
        fprintf(stderr, "track: no memory for the feature tables\n");
        return NULL;
    }
    t->next_id = 1;
    t->state = VT_ST_NONE;
    return t;
}

int tk_sets(const struct tracker *t) { return t->nsets; }

const struct vt_plane *tk_debug_plane(const struct tracker *t, int set, const struct vt_result *r) { return r->debug_level >= 0 ? &t->pyr[set][0].l[r->debug_level] : NULL; }

const struct vt_plane *tk_level(const struct tracker *t, int set, int level) { return &t->pyr[set][0].l[level]; }

int tk_is_build_key(uint32_t key) { return key == VT_SC_CONTRAST || key == VT_SC_PRE_GAUSS; }

/* ---- the pyramid ---- */

void tk_build(struct tracker *t, int set, const uint8_t *left, const uint8_t *right, const struct vt_msg *m)
{
    if (m->sc_key == VT_SC_CONTRAST)
        t->b_contrast = m->sc_value != 0.0f;
    else if (m->sc_key == VT_SC_PRE_GAUSS)
        t->b_pre_gauss = m->sc_value != 0.0f;
    struct job *j = t->jp;
    const int alg = m->algorithm == VT_ALG_X87 ? VT_ALG_X87 : VT_ALG_MVEX;
    j->alg = alg;
    j->set = set;
    j->src[0] = left;
    j->src[1] = right;
    j->cameras = t->c.stereo ? 2 : 1;
    run_build_op(t, OP_U8, 0);
    const int use_input = t->b_pre_gauss || t->b_contrast;
    for (int c = 0; c < j->cameras; c++)
        img_pad_tb(use_input ? &t->input[c] : &t->pyr[set][c].l[0]);
    if (t->b_contrast) {
        /* One stretch for all cameras, from the left image: both stay comparable. A linear map, so edges keep their ratios. */
        float mm[2];
        const struct vt_plane *s = &t->input[0];
        if (alg == VT_ALG_X87)
            img_minmax_ref(s, mm, 0, s->h);
        else
            img_minmax_vx(s, mm, 0, s->h);
        const real span = R(mm[1]) - R(mm[0]);
        const real gain = span > 0.0L ? 16320.0L / span : 1.0L;
        j->gain_offset[0] = (float)gain;
        j->gain_offset[1] = (float)(-R(mm[0]) * gain);
        run_build_op(t, OP_NORM_IN, 0);
        for (int c = 0; c < j->cameras; c++)
            img_pad_tb(t->b_pre_gauss ? &t->input[c] : &t->pyr[set][c].l[0]);
    }
    if (t->b_pre_gauss) {
        run_build_op(t, OP_GAUSS_IN, 0);
        for (int c = 0; c < j->cameras; c++)
            img_pad_tb(&t->pyr[set][c].l[0]);
    }
    for (int L = 1; L < t->levels; L++) {
        run_build_op(t, OP_REDUCE, L);
        for (int c = 0; c < j->cameras; c++)
            img_pad_tb(&t->pyr[set][c].l[L]);
    }
}

/* ---- a median that reorders its input: Hoare's selection ---- */

static float select_median(float *v, int n)
{
    int lo = 0, hi = n - 1;
    const int k = n / 2;
    while (lo < hi) {
        const float p = v[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] < p)
                i++;
            while (v[j] > p)
                j--;
            if (i <= j) {
                const float s = v[i];
                v[i] = v[j];
                v[j] = s;
                i++;
                j--;
            }
        }
        if (k <= j)
            hi = j;
        else if (k >= i)
            lo = i;
        else
            break;
    }
    return v[k];
}

/* ---- the region ---- */

static void store_appearance(struct tracker *t)
{
    /* The coarsest level where the template still fits INSIDE the region: a template wider than the region would hold the background, which moves differently. */
    int L = 0;
    while (L < t->levels - 1 && R(t->rw) / R(1 << (L + 1)) >= R(APP_W) && R(t->rh) / R(1 << (L + 1)) >= R(APP_H))
        L++;
    const struct vt_plane *pl = &t->pyr[t->cur][0].l[L];
    const real s = 1.0L / R(1 << L);
    const real lx = R(t->cx) * s, ly = R(t->cy) * s;
    const int ix = (int)lx, iy = (int)ly;
    const int x0 = ix - APP_W / 2, y0 = iy - APP_H / 2;
    if (x0 < 1 || y0 < 1 || x0 + APP_W >= pl->w - 1 || y0 + APP_H >= pl->h - 1)
        return;
    vxf sum = vx_zero();
    for (int r = 0; r < APP_H; r++) {
        const float *row = plane_row(pl, y0 + r) + x0;
        for (int k = 0; k < APP_W; k += VX_LANES) {
            const vxf v = vx_loadu(row + k);
            vx_store(t->app + r * APP_W + k, v);
            sum = vx_add(sum, v);
        }
    }
    const float mean = (float)(R(vx_hsum(sum)) / R(APP_W * APP_H));
    const vxf vm = vx_bcast(&mean);
    vxf ss = vx_zero();
    for (int r = 0; r < APP_H; r++)
        for (int k = 0; k < APP_W; k += VX_LANES) {
            const vxf z = vx_sub(vx_load(t->app + r * APP_W + k), vm);
            vx_store(t->app + r * APP_W + k, z);
            ss = vx_fma(z, z, ss);
        }
    t->app_norm = vx_hsum(ss);
    t->app_level = L;
    t->app_fx = (float)(lx - R(ix));
    t->app_fy = (float)(ly - R(iy));
    t->have_app = 1;
}

/* Adds corners inside the region at this frame, away from the features already held. */
static void seed_features(struct tracker *t)
{
    float nx[VT_MAX_FEATURES], ny[VT_MAX_FEATURES];
    const int mw = (int)t->c.patch_w / 2 + 2, mh = (int)t->c.patch_h / 2 + 2;
    const int margin = mw > mh ? mw : mh;
    const int want = (int)t->c.max_features;
    const float rx = (float)(R(t->cx) - 0.5L * R(t->rw)), ry = (float)(R(t->cy) - 0.5L * R(t->rh));
    const int found = detect_corners(&t->pyr[t->cur][0].l[0], rx, ry, t->rw, t->rh, margin, want, t->c.min_dist, t->c.corner_quality, nx, ny, t->det_scratch);
    const real min2 = R(t->c.min_dist) * R(t->c.min_dist);
    for (int i = 0; i < found && t->nf < want; i++) {
        int far = 1;
        for (int k = 0; k < t->nf && far; k++) {
            const real dx = R(t->fx[k]) - R(nx[i]), dy = R(t->fy[k]) - R(ny[i]);
            far = dx * dx + dy * dy >= min2;
        }
        if (!far)
            continue;
        const int k = t->nf++;
        t->fx[k] = nx[i];
        t->fy[k] = ny[i];
        t->fdisp[k] = -1.0f;
        t->fsigma[k] = 0.0f;
        t->fu[k] = 0.0f;
        t->fv[k] = 0.0f;
        t->fflag[k] = VT_FF_NEW;
    }
}

static void end_region(struct tracker *t, uint32_t state)
{
    t->state = state;
    t->nf = 0;
    t->have_app = 0;
    t->kf.valid = 0;
}

static void apply_command(struct tracker *t, const struct vt_msg *d, struct vt_result *o)
{
    if (d->cmd == VT_CMD_DESELECT && t->state != VT_ST_NONE) {
        end_region(t, VT_ST_DESELECTED);
    } else if (d->cmd == VT_CMD_SELECT) {
        end_region(t, VT_ST_NEW);
        t->id = t->next_id++;
        t->cx = (float)(R(d->roi[0]) + 0.5L * R(d->roi[2]));
        t->cy = (float)(R(d->roi[1]) + 0.5L * R(d->roi[3]));
        t->rw = d->roi[2];
        t->rh = d->roi[3];
        t->rate_u = t->rate_v = 0.0f;
        t->missing = 0;
        seed_features(t);
        if (t->nf < (int)t->c.min_features) {
            /* Too little texture to hold: say so, invent nothing. */
            o->flags |= VT_F_NO_FEATURES;
            end_region(t, VT_ST_LOST);
        } else {
            t->last_obs_ns = d->acq_ns;
            store_appearance(t);
        }
    }
}

static inline void predict_position(struct tracker *t, real dt)
{
    if (dt > 0.0L) {
        t->cx = (float)(R(t->cx) + R(t->rate_u) * dt);
        t->cy = (float)(R(t->cy) + R(t->rate_v) * dt);
    }
}

/* What the flow says about the region; moves or does not move it. */
static void region_from_flow(struct tracker *t, const struct vt_msg *d, real dt, struct vt_result *o)
{
    static float du[VT_MAX_FEATURES], dv[VT_MAX_FEATURES], tu[VT_MAX_FEATURES], tv[VT_MAX_FEATURES], px[VT_MAX_FEATURES], py[VT_MAX_FEATURES];
    int n = 0;
    for (int i = 0; i < t->nf; i++)
        if (!t->flow[i].status) {
            du[n] = (float)(R(t->flow[i].x) - R(t->fx[i]));
            dv[n] = (float)(R(t->flow[i].y) - R(t->fy[i]));
            px[n] = t->fx[i];
            py[n] = t->fy[i];
            n++;
        }
    const int before = t->nf;
    if (n >= (int)t->c.min_features) {
        /* Translation plus image scale: x' = x + c + s (x - x_c). A region moving in depth scales about its centre, so the median displacement alone is the centre's motion
         * only when the features happen to sit symmetrically around it. With the medians of displacement and position, c = median(du) - s (median(x) - x_c), and s is the
         * median over the features of the radial stretch relative to those medians. */
        memcpy(tu, du, sizeof(float) * (size_t)n);
        memcpy(tv, dv, sizeof(float) * (size_t)n);
        const real mu0 = R(select_median(tu, n)), mv0 = R(select_median(tv, n));
        memcpy(tu, px, sizeof(float) * (size_t)n);
        memcpy(tv, py, sizeof(float) * (size_t)n);
        const real mx = R(select_median(tu, n)), my = R(select_median(tv, n));
        int ns = 0;
        for (int i = 0; i < n; i++) {
            const real rx = R(px[i]) - mx, ry = R(py[i]) - my, r2 = rx * rx + ry * ry;
            if (r2 >= 16.0L)
                tu[ns++] = (float)(((R(du[i]) - mu0) * rx + (R(dv[i]) - mv0) * ry) / r2);
        }
        const real s = ns >= (int)t->c.min_features ? R(select_median(tu, ns)) : 0.0L;
        const real cu = mu0 - s * (mx - R(t->cx)), cv = mv0 - s * (my - R(t->cy));
        /* The residual of each feature under that motion: how the agreeing ones are told from the rest. */
        for (int i = 0; i < n; i++)
            tu[i] = (float)(fabsl(R(du[i]) - cu - s * (R(px[i]) - R(t->cx))) + fabsl(R(dv[i]) - cv - s * (R(py[i]) - R(t->cy))));
        const real mad = R(select_median(tu, n));
        const real limit = mad * 3.0L > 0.5L ? mad * 3.0L : 0.5L;
        /* Keep the agreeing ones, in place. */
        int w = 0;
        for (int i = 0; i < t->nf; i++) {
            const struct lk_out *f = &t->flow[i];
            if (f->status)
                continue;
            const real u = R(f->x) - R(t->fx[i]), v = R(f->y) - R(t->fy[i]);
            if (fabsl(u - cu - s * (R(t->fx[i]) - R(t->cx))) + fabsl(v - cv - s * (R(t->fy[i]) - R(t->cy))) > limit)
                continue;
            const real var = R(f->cov_xx) + R(f->cov_yy);
            t->fx[w] = f->x;
            t->fy[w] = f->y;
            t->fu[w] = (float)u;
            t->fv[w] = (float)v;
            t->fsigma[w] = (float)sqrtl(var > 0.0L ? var : 0.0L);
            t->fdisp[w] = t->fdisp[i];
            t->fflag[w] = t->fflag[i] & ~(uint32_t)VT_FF_NEW;
            w++;
        }
        const float ratio = before ? (float)(R(w) / R(before)) : 0.0f;
        o->inliers = ratio;
        if (w >= (int)t->c.min_features && ratio >= t->c.min_inliers) {
            t->nf = w;
            t->cx = (float)(R(t->cx) + cu);
            t->cy = (float)(R(t->cy) + cv);
            t->rw = (float)(R(t->rw) * (1.0L + s));
            t->rh = (float)(R(t->rh) * (1.0L + s));
            o->scale = (float)s;
            if (dt > 0.0L) {
                t->rate_u = (float)(cu / dt);
                t->rate_v = (float)(cv / dt);
            }
            o->du = (float)cu;
            o->dv = (float)cv;
            t->state = VT_ST_OBSERVED;
            t->missing = 0;
            t->last_obs_ns = d->acq_ns;
            return;
        }
    }
    /* Not observed: no motion is reported for this frame. The features are not trusted any more; the position shown is a prediction. */
    t->nf = 0;
    t->state = VT_ST_UNOBSERVED;
    t->missing++;
    predict_position(t, dt);
}

/* One axis of a parabola through three correlations: the peak's offset, within half a step. */
static real peak_offset(real l, real c, real r)
{
    const real den = l - 2.0L * c + r;
    if (den >= 0.0L)
        return 0.0L;
    const real o = 0.5L * (l - r) / den;
    return o < -0.5L ? -0.5L : o > 0.5L ? 0.5L : o;
}

/* Looks for the stored appearance around the prediction. */
static int try_reacquire(struct tracker *t, int alg, struct vt_result *o)
{
    if (!t->have_app)
        return 0;
    struct job *j = t->jt;
    const real s = 1.0L / R(1 << t->app_level);
    j->xc = (int)(R(t->cx) * s - R(t->app_fx)) - APP_W / 2;
    j->yc = (int)(R(t->cy) * s - R(t->app_fy)) - APP_H / 2;
    j->radius = (int)t->c.reacq_radius + (int)t->missing;
    if (j->radius > 3 * (int)t->c.reacq_radius)
        j->radius = 3 * (int)t->c.reacq_radius;
    for (int i = 0; i < t->threads_track; i++)
        j->partial[i * 16] = -2.0f;
    run_track_op(t, OP_SEARCH, alg);
    float best = -2.0f, fbx = 0.0f, fby = 0.0f;
    for (int i = 0; i < t->threads_track; i++) {
        const float *p = &j->partial[i * 16];
        if (p[0] > best || (p[0] == best && best > -2.0f && (p[2] < fby || (p[2] == fby && p[1] < fbx)))) {
            best = p[0];
            fbx = p[1];
            fby = p[2];
        }
    }
    const int bx = (int)fbx, by = (int)fby;
    const struct vt_plane *pl = &t->pyr[t->cur][0].l[t->app_level];
    const real norm = R(t->app_norm), c0 = R(best);
    const real ox = peak_offset(search_at(pl, t->app, norm, bx - 1, by), c0, search_at(pl, t->app, norm, bx + 1, by));
    const real oy = peak_offset(search_at(pl, t->app, norm, bx, by - 1), c0, search_at(pl, t->app, norm, bx, by + 1));
    const real scale = R(1 << t->app_level);
    /* The template's corner is its centre less these whole pixels, the same whole division store_appearance used. */
    const int half_w = APP_W / 2, half_h = APP_H / 2;
    o->reacq_ncc = best;
    o->reacq_x = (float)((R(bx) + ox + R(half_w) + R(t->app_fx)) * scale);
    o->reacq_y = (float)((R(by) + oy + R(half_h) + R(t->app_fy)) * scale);
    if (best < t->c.reacq_ncc)
        return 0;
    t->cx = o->reacq_x;
    t->cy = o->reacq_y;
    t->nf = 0;
    seed_features(t);
    if (t->nf < (int)t->c.min_features) {
        t->nf = 0;
        return 0;
    }
    o->flags |= VT_F_REACQUIRED;
    return 1;
}

/* ---- stereo ---- */

static void stereo_one(struct tracker *t, int i, int alg, void *tile)
{
    const struct vt_plane *L = t->pyr[t->cur][0].l, *Rt = t->pyr[t->cur][1].l;
    const float x = t->fx[i], y = t->fy[i];
    uint32_t fl = t->fflag[i] & VT_FF_NEW;
    t->fxr[i] = t->fyr[i] = 0.0f;
    t->fz[i] = 0.0f;
    struct lk_out m;
    float guess = t->fdisp[i];
    int top = 0;
    if (guess <= 0.0f) {
        /* No prior for this feature: the correlation scan at a coarse level says where, and how sure. */
        const int sl = t->stereo_level;
        const real s = 1.0L / R(1 << sl);
        struct stereo_scan sc;
        const int dmin = (int)(R(t->c.min_disp) * s), dmax = (int)(R(t->c.max_disp) * s);
        const float sx = (float)(R(x) * s), sy = (float)(R(y) * s);
        if (alg == VT_ALG_X87)
            stereo_scan_ref(&L[sl], &Rt[sl], sx, sy, dmin, dmax, &sc);
        else
            stereo_scan_vx(&L[sl], &Rt[sl], sx, sy, dmin, dmax, &sc);
        if (sc.disparity < 0.0f || sc.ncc < t->c.min_ncc) {
            t->fflag[i] = fl;
            t->fdisp[i] = -1.0f;
            return;
        }
        if (R(sc.second) > R(sc.ncc) * R(t->c.ambiguity)) {
            t->fflag[i] = fl | VT_FF_AMBIGUOUS;
            t->fdisp[i] = -1.0f;
            return;
        }
        guess = (float)(R(sc.disparity) * R(1 << sl));
        top = sl;
    }
    const float gx = (float)(-R(guess));
    if (alg == VT_ALG_X87) {
        lk_track_ref(L, Rt, top, &t->lk, x, y, gx, 0.0f, &m, tile);
        if (t->c.fb_max > 0.0f)
            lk_flowback_ref(L, Rt, 0, &t->lk, x, y, t->c.fb_max, &m, tile);
    } else {
        lk_track_vx(L, Rt, top, &t->lk, x, y, gx, 0.0f, &m, tile);
        if (t->c.fb_max > 0.0f)
            lk_flowback_vx(L, Rt, 0, &t->lk, x, y, t->c.fb_max, &m, tile);
    }
    if (m.status) {
        t->fflag[i] = fl;
        t->fdisp[i] = -1.0f;
        return;
    }
    t->fxr[i] = m.x;
    t->fyr[i] = m.y;
    const real disparity = R(x) - R(m.x);
    t->fdisp[i] = (float)disparity;
    fl |= VT_FF_STEREO;
    real *p = &t->p3[3 * i];
    if (disparity >= R(t->c.min_disp) && disparity <= R(t->c.max_disp) && stereo_triangulate(&t->g, x, y, m.x, m.y, p)) {
        if (stereo_reprojection(&t->g, p, x, y, m.x, m.y) <= R(t->c.max_reproj)) {
            fl |= VT_FF_3D;
            t->fz[i] = (float)p[2];
        }
    }
    t->fflag[i] = fl;
}

/* The region's 3D measurement from its features' points. Returns how many agreed. */
static int region_measurement(struct tracker *t, real *z, real *r33, real *reproj)
{
    static float ds[VT_MAX_FEATURES], rp[VT_MAX_FEATURES];
    int n = 0;
    for (int i = 0; i < t->nf; i++)
        if (t->fflag[i] & VT_FF_3D) {
            ds[n] = t->fdisp[i];
            rp[n] = (float)stereo_reprojection(&t->g, &t->p3[3 * i], t->fx[i], t->fy[i], t->fxr[i], t->fyr[i]);
            n++;
        }
    if (n < (int)t->c.min_features)
        return 0;
    *reproj = R(select_median(rp, n));
    /* The region's point: its depth from the median disparity, robust against the few bad matches left; X and Y from the region's centre, which does not jump when the feature
     * set changes. A spread of positions across the region is the region's size, not noise, and must not enter the noise. */
    const real d = R(select_median(ds, n));
    if (d <= 0.0L)
        return 0;
    const real zz = t->g.fx * t->g.baseline / d;
    z[0] = (R(t->cx) - t->g.cx) * zz / t->g.fx;
    z[1] = (R(t->cy) - t->g.cy) * zz / t->g.fy;
    z[2] = zz;
    for (int i = 0; i < n; i++)
        rp[i] = (float)fabsl(R(ds[i]) - d);
    const real madd = R(select_median(rp, n));
    /* The disparity's noise: the median's own scatter (pi / 2 x sigma^2 / n, sigma = 1.4826 MAD) and a floor for what scatter cannot show. Features on one surface share errors,
     * so the effective count is n / 4, at least 1. Fitted against truth: docs/measurements.md. */
    const real neff = n >= 4 ? R(n) / 4.0L : 1.0L;
    const real sd = R(t->c.sigma_d), sc = R(t->c.sigma_c);
    const real var_d = 1.5708L * 1.4826L * 1.4826L * madd * madd / neff + sd * sd;
    stereo_covariance(&t->g, z, d, sc * sc, sc * sc, var_d, r33);
    const real floor2 = R(t->c.r_floor) * R(t->c.r_floor);
    r33[0] += floor2;
    r33[4] += floor2;
    r33[8] += floor2;
    return n;
}

/* ---- the rest of one frame ---- */

size_t tk_result_bytes(const struct vt_result *r)
{
    const size_t n = offsetof(struct vt_result, feat) + (size_t)r->nfeat * sizeof(struct vt_feat_out);
    return (n + 63) & ~(size_t)63;
}

/* Soft configuration: one key at this frame, for the track group's stages. */
static void apply_soft_config(struct tracker *t, const struct vt_msg *m, struct vt_result *o)
{
    const float v = m->sc_value;
    switch (m->sc_key) {
    case VT_SC_NONE:
    case VT_SC_CONTRAST:
    case VT_SC_PRE_GAUSS:
        break;
    case VT_SC_FB_MAX: t->c.fb_max = v; break;
    case VT_SC_MAX_ITER: t->lk.max_iter = (int)v; break;
    case VT_SC_EPS: t->lk.eps = v; break;
    case VT_SC_MIN_EIG: t->lk.min_eig = v; break;
    case VT_SC_MAX_RESIDUAL: t->lk.max_residual = v; break;
    case VT_SC_MAX_FEATURES:
        if (v >= 1.0f && v <= (float)VT_MAX_FEATURES) {
            t->c.max_features = (uint32_t)v;
            if (t->nf > (int)v)
                t->nf = (int)v;
        } else
            o->exceptions |= 1u << VT_EXC_BAD_SOFTCONFIG;
        break;
    case VT_SC_Q_ACCEL: t->c.q_accel = v; break;
    case VT_SC_R_FLOOR: t->c.r_floor = v; break;
    case VT_SC_SIGMA_C: t->c.sigma_c = v; break;
    case VT_SC_SIGMA_D: t->c.sigma_d = v; break;
    case VT_SC_PREFETCH: t->lk.prefetch = v != 0.0f; break;
    case VT_SC_MAX_MISSING: t->c.max_missing = (uint32_t)v; break;
    case VT_SC_REACQ_NCC: t->c.reacq_ncc = v; break;
    case VT_SC_DEBUG_LEVEL:
        t->debug_level = v >= 0.0f && (int)v < t->levels ? (int)v : -1;
        break;
    default:
        o->exceptions |= 1u << VT_EXC_BAD_SOFTCONFIG;
        break;
    }
}

void tk_track(struct tracker *t, int set, const struct vt_msg *d, struct vt_result *o)
{
    const uint64_t t0 = vt_tsc();
    memset(o, 0, offsetof(struct vt_result, feat));
    o->hdr = *d;
    o->hdr.type = VT_MSG_RESULT;
    o->hdr.check = 0;
    t->cur = set;
    apply_soft_config(t, d, o);
    int alg = VT_ALG_MVEX;
    if (d->algorithm == VT_ALG_X87)
        alg = VT_ALG_X87;
    else if (d->algorithm != VT_ALG_MVEX)
        o->exceptions |= 1u << VT_EXC_BAD_ALGORITHM;
    /* A destination this session does not have: ignored, and said so. */
    int want_stereo = (d->dest & VT_PIPE_STEREO) != 0;
    if (want_stereo && !t->c.stereo) {
        o->exceptions |= 1u << VT_EXC_ROUTE_STEREO;
        want_stereo = 0;
    }
    const int want_flow = (d->dest & VT_PIPE_FLOW) != 0, want_est = (d->dest & VT_PIPE_EST) != 0;
    const int have_prev = t->prev >= 0;

    if (t->c.frame_diff && have_prev) {
        for (int i = 0; i < t->threads_track; i++)
            t->jt->partial[i * 16] = 0.0f;
        run_track_op(t, OP_DIFF, alg);
        real sum = 0.0L;
        for (int i = 0; i < t->threads_track; i++)
            sum += R(t->jt->partial[i * 16]);
        o->motion = (float)(sum / (R(t->w & ~15) * R(t->h)) / R(VT_Q6));
    }

    const real dt = t->last_acq_ns ? R(d->acq_ns - t->last_acq_ns) * 1e-9L : 0.0L;
    t->last_acq_ns = d->acq_ns;

    /* The flow, for a region that exists and held features last frame. */
    const int active = t->state == VT_ST_NEW || t->state == VT_ST_OBSERVED;
    const int do_flow = want_flow && active && have_prev && t->nf > 0;
    if (do_flow)
        run_track_op(t, OP_LK, alg);
    const uint64_t t1 = vt_tsc();
    o->cycles[VT_T_LK] = t1 - t0;

    if (do_flow) {
        region_from_flow(t, d, dt, o);
    } else if (t->state == VT_ST_UNOBSERVED) {
        t->missing++;
        predict_position(t, dt);
    }
    if (d->cmd != VT_CMD_NONE)
        apply_command(t, d, o);
    /* A region that left its flow: take it back from its appearance, or count it out. */
    if (t->state == VT_ST_UNOBSERVED) {
        if (try_reacquire(t, alg, o)) {
            t->state = VT_ST_OBSERVED;
            t->missing = 0;
            t->last_obs_ns = d->acq_ns;
        } else if (t->missing > t->c.max_missing) {
            end_region(t, VT_ST_LOST);
        }
    }
    if (t->state == VT_ST_OBSERVED && !(o->flags & VT_F_REACQUIRED)) {
        /* Replace features as they die, but not every frame: a region with few corners stays below the line for good, and detection cost 7.2 ms a frame on real video
         * (camcorder, 275 x 320 region, 2026-11-02). Every 16th frame, or at once when the region is close to being unobservable. Refresh the appearance only from a clean frame. */
        if (t->nf < (int)t->c.min_features * 2 || (t->nf < (int)t->c.max_features / 2 && (d->instance % 16u) == 8u))
            seed_features(t);
        if (o->inliers > 0.9f && (d->instance % 16u) == 0u)
            store_appearance(t);
    }
    const uint64_t t2 = vt_tsc();
    o->cycles[VT_T_REGION] = t2 - t1;

    /* Stereo. */
    real z[3] = {0.0L, 0.0L, 0.0L}, r33[9] = {0.0L}, reproj = 0.0L;
    int n3 = 0, ambiguous = 0;
    const int has_feat = (t->state == VT_ST_NEW || t->state == VT_ST_OBSERVED) && t->nf > 0;
    if (want_stereo && has_feat) {
        run_track_op(t, OP_STEREO, alg);
        for (int i = 0; i < t->nf; i++)
            ambiguous += (t->fflag[i] & VT_FF_AMBIGUOUS) != 0;
        n3 = region_measurement(t, z, r33, &reproj);
    }
    const uint64_t t3 = vt_tsc();
    o->cycles[VT_T_STEREO] = t3 - t2;

    /* The filter. Its clock is the acquisition timestamps; a prediction never counts as an observation. */
    if (t->state == VT_ST_LOST || t->state == VT_ST_DESELECTED || t->state == VT_ST_NONE)
        t->kf.valid = 0;
    if (want_est) {
        if (t->kf.valid && dt > 0.0L)
            kf_predict(&t->kf, dt, R(t->c.q_accel));
        if (n3 > 0) {
            for (int i = 0; i < 3; i++)
                o->meas[i] = (double)z[i];
            for (int i = 0; i < 9; i++)
                o->meas_cov[i] = (double)r33[i];
            o->reproj_rms = (double)reproj;
            if (!t->kf.valid) {
                kf_init(&t->kf, z, r33, R(t->c.v0_sigma));
                o->flags |= VT_F_3D_MEASURED;
                t->last_meas_ns = d->acq_ns;
            } else {
                const real nis = kf_innovation(&t->kf, z, r33);
                o->nis = (double)nis;
                if (nis <= KF_GATE_NIS) {
                    kf_update(&t->kf, z, r33);
                    o->flags |= VT_F_3D_MEASURED;
                    t->last_meas_ns = d->acq_ns;
                } else {
                    o->flags |= VT_F_3D_REJECTED;
                }
            }
        }
        if (want_stereo && has_feat && !n3 && ambiguous * 2 >= t->nf)
            o->flags |= VT_F_AMBIGUOUS;
        /* An estimate older than the region's own patience is not reported as metric. */
        if (t->kf.valid) {
            const real age = R(d->acq_ns - t->last_meas_ns) * 1e-9L;
            o->age_s = (double)age;
            const real patience = dt > 0.0L ? dt * R(t->c.max_missing) : 1.0L;
            if (age > patience + 1e-9L) {
                t->kf.valid = 0;
            } else {
                for (int i = 0; i < 3; i++) {
                    o->pos[i] = (double)t->kf.x[i];
                    o->vel[i] = (double)t->kf.x[i + 3];
                    o->vel_var[i] = (double)t->kf.P[i + 3][i + 3];
                    for (int k = 0; k < 3; k++)
                        o->pos_cov[3 * i + k] = (double)t->kf.P[i][k];
                }
                o->flags |= VT_F_METRIC;
            }
        }
    }
    const uint64_t t4 = vt_tsc();
    o->cycles[VT_T_EST] = t4 - t3;

    /* The result. */
    o->state = t->state;
    o->track_id = t->id;
    o->missing = t->missing;
    o->last_obs_ns = t->last_obs_ns;
    if (t->state == VT_ST_UNOBSERVED)
        o->flags |= VT_F_PREDICTED;
    if (t->state == VT_ST_NEW || t->state == VT_ST_OBSERVED || t->state == VT_ST_UNOBSERVED) {
        o->cx = t->cx;
        o->cy = t->cy;
        o->w = t->rw;
        o->h = t->rh;
        o->rate_u = t->rate_u;
        o->rate_v = t->rate_v;
        if (t->state == VT_ST_UNOBSERVED && (t->cx < 0.0f || t->cy < 0.0f || t->cx >= (float)t->w || t->cy >= (float)t->h))
            o->flags |= VT_F_OUT_OF_VIEW;
    }
    o->nfeat = (uint32_t)t->nf;
    int with3d = 0;
    real sig = 0.0L;
    for (int i = 0; i < t->nf; i++) {
        struct vt_feat_out *f = &o->feat[i];
        f->x = t->fx[i];
        f->y = t->fy[i];
        f->u = t->fu[i];
        f->v = t->fv[i];
        f->flags = t->fflag[i];
        f->sigma = t->fsigma[i];
        if (want_stereo && (t->fflag[i] & VT_FF_STEREO)) {
            f->xr = t->fxr[i];
            f->yr = t->fyr[i];
            f->disparity = t->fdisp[i];
            f->z = t->fz[i];
        } else {
            f->xr = f->yr = f->disparity = f->z = 0.0f;
        }
        with3d += (t->fflag[i] & VT_FF_3D) != 0;
        sig += R(t->fsigma[i]);
    }
    o->nfeat_3d = (uint32_t)with3d;
    o->sigma_px = t->nf ? (float)(sig / R(t->nf) / sqrtl(R(t->nf))) : 0.0f;
    o->quality = t->state == VT_ST_OBSERVED ? o->inliers : 0.0f;
    o->hdr.length = (uint32_t)tk_result_bytes(o);
    o->debug_level = t->debug_level;
    if (t->debug_level >= 0) {
        const struct vt_plane *pl = &t->pyr[set][0].l[t->debug_level];
        o->debug_w = (uint32_t)pl->w;
        o->debug_h = (uint32_t)pl->h;
        o->debug_stride = (uint32_t)pl->stride;
    }
    t->prev = set;
    o->cycles[VT_T_TRACK] = vt_tsc() - t0;
}
