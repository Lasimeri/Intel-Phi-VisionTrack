/* track.c: see track.h.
 *
 * Every scalar computation is long double, and a float result is one explicit cast: x87 on the card and on the host alike. A float expression would run on x87 with excess
 * precision on the card and on SSE on the host, and whole-run replay would part ways at the first inexact sum. */
#include "track.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "detect.h"
#include "framegen.h"
#include "img.h"
#include "kalman.h"
#include "lk.h"
#include "stereo.h"
#include "vt_util.h"
#include "vx.h"

#define APP_W 32 /* the appearance template: lanes, rows at its pyramid level */
#define APP_H 21
#define MAX_SETS 4
/* Telling what was selected from what surrounds it: region_from_flow. */
#define RIGID_TOL 1.25L          /* pixels, |x| + |y|: further than this from where the rigid model puts it (or 3 x the members' spread, on a noisy picture), a feature is not moving with the region */
#define ANCHOR_LEAK 0.03L        /* of its deviation a member's anchor gives way each frame: about 33 frames to forgive a slowly changing shape */
#define ANCHOR_LEAK_SPEED 0.05L  /* and more as the region moves, per pixel a frame, */
#define ANCHOR_LEAK_MOVING 0.125L /* up to about 8 frames */
#define TOL_SPEED 3.0L           /* the tolerance is at least this many frames' motion of the region, */
#define TOL_CAP_OF_WIDTH 0.00625L /* up to this much of the image's width: 12 px at 1920 */
#define COAST_DAMP 0.95L         /* a lost region's rate, kept this much per frame: a thing out of sight is not known to keep going */
#define MOVED_MEMORY 0.97L       /* a feature's own motion, remembered over about 33 frames: how a slow mover is told from a still thing */
#define SPREAD_MEMORY 0.9L       /* the members' spread, remembered over about 10 frames */
#define SECOND_SHARE 0.15L       /* of the votes: less than this is not a second thing in the box */
#define WAKE_FRAMES 3            /* frames a second motion near the centre must hold, while the rest is still, before it is what was selected */
#define PROBATION 8              /* frames a new feature rides along, without a vote, before it is a member */
#define SIM_DEAD 0.25L           /* pixels at the members' typical radius: a smaller change of scale or rotation is noise */
#define SIM_MIN_RADIUS2 64.0L    /* below a spread of 8 px the members cannot tell scale or rotation */
#define SIM_MAX_STEP_DOWN 0.64L  /* squared: a model that shrinks by more than 20 % in one frame is not believed */
#define SIM_MAX_STEP_UP 1.5625L  /* squared: nor one that grows by more than 25 % */
#define BOX_ROOM 1.3L            /* a closed box: the members' span times this, plus a margin in pixels */
#define BOX_MARGIN 24.0L
#define BOX_RATE 0.2L            /* of the way to what the members reach, per frame, in a closed box */
#define GROW_REACH 1.6L          /* a closed box looks this far beyond itself for more of the thing */
#define GROW_MOTION 0.1L         /* pixels a frame, |u| + |v|: a closed region moving slower than this proves nothing about a new feature, and takes none in */
#define RECENTRE_RATE 0.05L      /* of the way to the middle of the members, per frame, in a region that found a smaller thing in its box */
#define HOLD_STILL 0.1L          /* pixels a frame, |u| + |v|: a region slower than this is standing still */
#define HOLD_AFTER 15            /* frames standing still before a frame is held */
#define REACQ_MIN_STEPS 2       /* the appearance search, at least this far in pixels of its level */
#define REACQ_FAR_AFTER 8       /* frames out of flow before the whole picture is searched as well */
#define REACQ_FAR_NCC 0.88L     /* far from any prediction, believed at this correlation, */
#define REACQ_FAR_CLEAR 0.04L   /* this far above the best peak elsewhere, */
#define REACQ_AGAIN_PX 24       /* and again on the next frame within these full-resolution pixels */
#define SEARCH_CELL 16          /* template positions to a cell of the search's map */
#define SEARCH_TOP 8            /* cells of the map looked at again in half pixels */
#define APP_MIN_ENERGY 0.0625L  /* of the template's energy: a window with less, contrast under a quarter, is not the thing */
#define REACQ_BAND 32000        /* template positions of the whole-picture search per frame: the rest of the picture on later frames */
#define APP_SLOTS 4             /* appearances remembered */

/* One remembered appearance: a zero-mean template at its level, its sub-pixel offset from the region's centre, and the region's size when it was taken. */
struct app_slot {
    float *pix;
    float norm, fx, fy, rw, rh;
    int level, valid;
};
#define SPACING_MIN 6.0L        /* pixels between features: at most the configured spacing, closer in a small box so that a small thing gets enough of them */
#define SPACING_BUDGET 1.5L      /* the spacing that would fill the feature budget this many times over */

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
    struct app_slot app[APP_SLOTS];
    uint32_t app_ticks;
    int app_turn, scan_y;
    int cand_run, cand_slot; /* seen far away last frame, not yet believed */
    float cand_x, cand_y;
    float *sc_v; /* the search's map, a best per cell */
    int *sc_x, *sc_y;
    /* the rigid model: region_from_flow. Each feature's anchor, own remembered motion, and age. */
    float *fax, *fay, *fmu, *fmv;
    uint8_t *fage;
    float ra, rb, spread, speed;
    float gw, gh; /* the operator's box, scaled with the region */
    int split_run, want_app, closed, recentring;
    struct detect_job det;
    /* A frame held: a member tracked against it cannot accumulate anything. Each feature's place in that frame, negative if it has none. */
    struct vt_plane key[VT_MAX_LEVELS];
    float *fkx, *fky;
    int hold, still_run;
    /* Frame generation. The pyramid goes on above the tracker's own levels, per set; the field starts at the tracker's level 1. */
    struct fg_job fg;
    struct vt_plane fg_more[3][FG_MAX_LEVELS];
    int fg_levels, fg_hist_set, gen_skip;
    uint8_t fg_ready[3];
    void *fg_mem;
    uint64_t fg_cycles[9], fg_runs;
    uint8_t *fg_pic; /* where a generated frame's picture goes, for the window; none: it stays for the tracker alone, and is made only for a region */
    uint64_t last_est_ns;
    /* filter */
    struct kf kf;
    uint64_t last_meas_ns;
};

/* ---- parallel jobs ---- */

enum { OP_U8, OP_NORM_IN, OP_GAUSS_IN, OP_REDUCE, OP_DIFF, OP_LK, OP_STEREO, OP_SEARCH, OP_KEY_COPY, OP_DET_TENSOR, OP_DET_RESPONSE, OP_DET_CELLS, OP_FG_MORE, OP_FG_HIST_A, OP_FG_HIST_B, OP_FG_SEARCH, OP_FG_FILTER, OP_FG_UP, OP_FG_MID, OP_FG_BLEND, OP_FG_PAD, OP_FG_PICTURE };
#define FG_CHUNK 16 /* blocks taken at a time */

struct job {
    struct tracker *t;
    int op, level, set, alg;
    const uint8_t *src[2];
    int cameras;
    float gain_offset[2];
    int next VT_CACHELINE;     /* dynamic work: points, candidate rows */
    float partial[256 * 16] VT_CACHELINE; /* per-thread results, a line each */
    int slot, x0, x1, y0, y1, step, cell_cols, cell_rows; /* the appearance search */
};

static inline int row_of(int rows, int tid, int n) { return (int)((long)rows * tid / n); }

static void stereo_one(struct tracker *t, int i, int alg, void *tile);

/* The correlation of the stored appearance at (x, y) of the level, in long double on both machines. */
static real search_at(const struct vt_plane *pl, const float *app, real norm_t, int x, int y)
{
    if (x < 1 || y < 1 || x + APP_W >= pl->w - 1 || y + APP_H >= pl->h - 1)
        return -1.0L;
    /* Relative to the window's first pixel: whole numbers still, and small ones. In absolute terms the sum of squares of a bright window runs to 2^37 in single precision, whose rounding is larger than the whole variance of a
     * nearly blank one, and such a window read correlations of 1.6 and 10.6 and took the region (live, 2026-11-03). And a window with little energy against the template's cannot be the thing, whatever it correlates. */
    const float first VX_ALIGNED = plane_row(pl, y)[x];
    const vxf base = vx_bcast(&first);
    vxf st = vx_zero(), sj = vx_zero(), sjj = vx_zero();
    for (int r = 0; r < APP_H; r++) {
        const float *row = plane_row(pl, y + r) + x;
        for (int k = 0; k < APP_W; k += VX_LANES) {
            const vxf v = vx_sub(vx_loadu(row + k), base);
            st = vx_fma(v, vx_load(app + r * APP_W + k), st);
            sj = vx_add(sj, v);
            sjj = vx_fma(v, v, sjj);
        }
    }
    const real n = R(APP_W * APP_H);
    const real a = R(vx_hsum(st)), b = R(vx_hsum(sj)), c = R(vx_hsum(sjj));
    const real var = c - b * b / n;
    return (var <= norm_t * APP_MIN_ENERGY || norm_t <= 0.0L) ? 0.0L : a / sqrtl(norm_t * var);
}

long double tk_correlate(const struct vt_plane *pl, const float *app, long double norm, int x, int y) { return search_at(pl, app, norm, x, y); }

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
            /* From where it stood in the held frame, starting from where it stood last frame: no error of an earlier frame stays in the answer. A feature the held frame no longer explains goes on from frame to frame. */
            const float kx = t->fkx[i], ky = t->fky[i];
            if (t->hold && kx >= 0.0f) {
                const float gx = (float)(R(t->fx[i]) - R(kx)), gy = (float)(R(t->fy[i]) - R(ky));
                if (x87) {
                    lk_track_ref(t->key, J, t->levels - 1, &t->lk, kx, ky, gx, gy, &t->flow[i], t->tiles[tid]);
                    if (t->c.fb_max > 0.0f)
                        lk_flowback_ref(t->key, J, t->levels - 1, &t->lk, kx, ky, t->c.fb_max, &t->flow[i], t->tiles[tid]);
                } else {
                    lk_track_vx(t->key, J, t->levels - 1, &t->lk, kx, ky, gx, gy, &t->flow[i], t->tiles[tid]);
                    if (t->c.fb_max > 0.0f)
                        lk_flowback_vx(t->key, J, t->levels - 1, &t->lk, kx, ky, t->c.fb_max, &t->flow[i], t->tiles[tid]);
                }
                if (!t->flow[i].status)
                    continue;
                t->fkx[i] = -1.0f;
            }
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
        /* The best correlation of each cell of 16 x 16 positions. A thread takes a cell at a time, each worked in one order, so the answer is the same at any thread count, and more than one peak can be seen. Whole rows of
         * cells left most of the team idle: seven rows at a quarter of 640 x 480, 2.05 ms a frame while lost. */
        const struct app_slot *a = &t->app[j->slot];
        const struct vt_plane *pl = &t->pyr[t->cur][0].l[a->level];
        const int cells = j->cell_cols * j->cell_rows;
        for (;;) {
            const int cell = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (cell >= cells)
                break;
            const int y0 = j->y0 + (cell / j->cell_cols) * SEARCH_CELL, x0 = j->x0 + (cell % j->cell_cols) * SEARCH_CELL;
            const int y1 = y0 + SEARCH_CELL <= j->y1 + 1 ? y0 + SEARCH_CELL : j->y1 + 1, x1 = x0 + SEARCH_CELL <= j->x1 + 1 ? x0 + SEARCH_CELL : j->x1 + 1;
            float top = -2.0f;
            int bx = x0, by = y0;
            for (int y = y0; y < y1; y += j->step)
                for (int x = x0; x < x1; x += j->step) {
                    const float v = (float)search_at(pl, a->pix, R(a->norm), x, y);
                    if (v > top) {
                        top = v;
                        bx = x;
                        by = y;
                    }
                }
            t->sc_v[cell] = top;
            t->sc_x[cell] = bx;
            t->sc_y[cell] = by;
        }
        break;
    }
    case OP_FG_MORE: {
        /* One more level of the field's pyramid, above the tracker's. */
        const struct vt_plane *d = &t->fg_more[j->set][j->level], *s = j->level == t->levels - 1 ? &t->pyr[j->set][0].l[t->levels - 1] : &t->fg_more[j->set][j->level - 1];
        const int y0 = row_of(d->h, tid, n), y1 = row_of(d->h, tid + 1, n);
        img_reduce_vx(d, s, y0, y1);
        img_pad_rows(d, y0, y1);
        break;
    }
    case OP_FG_HIST_A:
    case OP_FG_HIST_B:
        for (;;) {
            const int s = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
            if (s >= FG_SECTIONS)
                break;
            if (j->op == OP_FG_HIST_A)
                fg_hist_section(&t->fg.a[1], t->fg.hist_a, s);
            else
                fg_hist_section(&t->fg.b[1], t->fg.hist_b, s);
        }
        break;
    case OP_FG_SEARCH:
    case OP_FG_FILTER:
    case OP_FG_UP:
    case OP_FG_MID: {
        const struct fg_field *field = j->op == OP_FG_UP ? &t->fg.up[j->level - 1] : j->op == OP_FG_MID ? &t->fg.mid : &t->fg.raw[j->level];
        const int count = field->w * field->h;
        for (;;) {
            const int i = __atomic_fetch_add(&j->next, FG_CHUNK, __ATOMIC_RELAXED);
            if (i >= count)
                break;
            const int e = i + FG_CHUNK < count ? i + FG_CHUNK : count;
            if (j->op == OP_FG_SEARCH)
                fg_search(&t->fg, j->level, i, e);
            else if (j->op == OP_FG_FILTER)
                fg_filter(&t->fg, j->level, i, e);
            else if (j->op == OP_FG_UP)
                fg_upscale(&t->fg, j->level, i, e);
            else
                fg_mid_vectors(&t->fg, i, e);
        }
        break;
    }
    case OP_FG_BLEND: {
        const int count = fg_tiles(&t->fg);
        for (;;) {
            const int i = __atomic_fetch_add(&j->next, FG_CHUNK, __ATOMIC_RELAXED);
            if (i >= count)
                break;
            fg_blend(&t->fg, i, i + FG_CHUNK < count ? i + FG_CHUNK : count);
        }
        break;
    }
    case OP_FG_PAD:
        img_pad_rows(&t->fg.out, row_of(t->fg.out.h, tid, n), row_of(t->fg.out.h, tid + 1, n));
        break;
    case OP_FG_PICTURE:
        img_q6_to_u8_vx(&t->fg.out, t->fg_pic, row_of(t->fg.out.h, tid, n), row_of(t->fg.out.h, tid + 1, n));
        break;
    case OP_KEY_COPY:
        for (int L = 0; L < t->levels; L++) {
            const struct vt_plane *s = &t->pyr[t->cur][0].l[L], *k = &t->key[L];
            const int all = s->h + 2 * VT_PAD_Y, y0 = row_of(all, tid, n), y1 = row_of(all, tid + 1, n);
            for (int y = y0; y < y1; y++) {
                const float *from = plane_row(s, y - VT_PAD_Y) - VT_PAD_X;
                float *to = plane_row(k, y - VT_PAD_Y) - VT_PAD_X;
                for (int x = 0; x < s->stride; x += VX_LANES)
                    vx_store_nr(to + x, vx_load(from + x));
            }
        }
        break;
    case OP_DET_TENSOR: {
        const int rows = t->det.rows + 4;
        detect_tensor(&t->det, row_of(rows, tid, n), row_of(rows, tid + 1, n));
        break;
    }
    case OP_DET_RESPONSE:
        detect_response(&t->det, row_of(t->det.rows, tid, n), row_of(t->det.rows, tid + 1, n));
        break;
    case OP_DET_CELLS:
        detect_cells(&t->det, row_of(t->det.cell_rows, tid, n), row_of(t->det.cell_rows, tid + 1, n));
        break;
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
    t->nsets = cfg->pipelines == 2 ? MAX_SETS : cfg->framegen ? 3 : 2;
    if (cfg->framegen && (cfg->pipelines == 2 || cfg->levels < 3)) {
        fprintf(stderr, "track: frame generation needs one pipeline and three pyramid levels\n");
        return NULL;
    }
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
    for (int i = 0; i < APP_SLOTS; i++) {
        t->app[i].pix = need(alloc, actx, APP_W * APP_H * sizeof(float));
        if (!t->app[i].pix)
            return NULL;
    }
    const size_t search_cells = (size_t)(t->w / SEARCH_CELL + 2) * (size_t)(t->h / SEARCH_CELL + 2);
    t->sc_v = need(alloc, actx, search_cells * sizeof(float));
    t->sc_x = need(alloc, actx, search_cells * sizeof(int));
    t->sc_y = need(alloc, actx, search_cells * sizeof(int));
    t->fax = need(alloc, actx, nf * sizeof(float));
    t->fay = need(alloc, actx, nf * sizeof(float));
    t->fmu = need(alloc, actx, nf * sizeof(float));
    t->fmv = need(alloc, actx, nf * sizeof(float));
    t->fage = need(alloc, actx, nf);
    t->fkx = need(alloc, actx, nf * sizeof(float));
    t->fky = need(alloc, actx, nf * sizeof(float));
    t->ra = 1.0f;
    t->fg_hist_set = -1;
    if (cfg->framegen) {
        /* The field starts at the tracker's level 1 and goes up as far as AMD's seven levels, while a level is still wide enough to hold a block's search. */
        int w = (t->w + 1) / 2, h = (t->h + 1) / 2, count = 0;
        for (int i = 0; i < FG_MAX_LEVELS && w >= 16 && h >= 16; i++) {
            count++;
            if (i >= t->levels - 1)
                for (int s = 0; s < 3; s++)
                    if (zeroed_plane(&t->fg_more[s][i], w, h, alloc, actx))
                        return NULL;
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        t->fg_levels = count;
        t->fg_mem = need(alloc, actx, fg_bytes((t->w + 1) / 2, (t->h + 1) / 2, count));
        if (!t->fg_mem)
            return NULL;
        fg_init(&t->fg, t->fg_mem, (t->w + 1) / 2, (t->h + 1) / 2, count);
    }
    {
        int w = t->w, h = t->h;
        for (int L = 0; L < t->levels; L++) {
            if (zeroed_plane(&t->key[L], w, h, alloc, actx))
                return NULL;
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
    }
    if (!t->det_scratch || !t->fx || !t->fy || !t->fxr || !t->fyr || !t->fdisp || !t->fsigma || !t->fu || !t->fv || !t->fz || !t->fflag || !t->flow || !t->p3 || !t->sc_v || !t->sc_x || !t->sc_y || !t->fax || !t->fay || !t->fmu || !t->fmv || !t->fage || !t->fkx || !t->fky) {
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
    if (t->c.framegen)
        t->fg_ready[set] = 0;
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

/* ---- generating a frame between two real ones ---- */

void tk_fg_report(const struct tracker *t, const char *name, double cycles_per_us)
{
    if (!t->fg_runs)
        return;
    static const char *const stage[9] = {"more", "scene", "search", "filter", "upscale", "mid", "blend", "pyramid", "picture"};
    fprintf(stderr, "%s: frame generation, %llu frames, us each:", name, (unsigned long long)t->fg_runs);
    for (int i = 0; i < 9; i++)
        fprintf(stderr, " %s %.1f", stage[i], (double)t->fg_cycles[i] / (double)t->fg_runs / cycles_per_us);
    fprintf(stderr, "\n");
}

static void fg_build_more(struct tracker *t, int set)
{
    if (t->fg_ready[set])
        return;
    struct job *j = t->jp;
    j->set = set;
    for (int i = t->levels - 1; i < t->fg_levels; i++) {
        run_build_op(t, OP_FG_MORE, i);
        img_pad_tb(&t->fg_more[set][i]);
    }
    t->fg_ready[set] = 1;
}

void tk_set_picture(struct tracker *t, uint8_t *pixels) { t->fg_pic = pixels; }

int tk_generate(struct tracker *t, int set, const struct vt_msg *m, struct vt_msg *gm)
{
    /* The window doubles every camera's picture whether or not anything is selected; the tracker wants one only for a region it can follow. */
    const int active = (t->state == VT_ST_NEW || t->state == VT_ST_OBSERVED) && t->nf > 0;
    if (!t->c.framegen || t->prev < 0 || t->prev == set || (!active && !t->fg_pic))
        return -1;
    const int p = t->prev, g = (set + 1) % t->nsets;
    uint64_t mark = vt_tsc();
#define FG_LAP(stage)                       \
    do {                                    \
        const uint64_t lap_now = vt_tsc();  \
        t->fg_cycles[stage] += lap_now - mark; \
        mark = lap_now;                     \
    } while (0)
    t->fg_runs++;
    fg_build_more(t, p);
    fg_build_more(t, set);
    FG_LAP(0);
    struct fg_job *f = &t->fg;
    for (int i = 0; i < t->fg_levels; i++) {
        f->a[i] = i + 1 < t->levels ? t->pyr[p][0].l[i + 1] : t->fg_more[p][i];
        f->b[i] = i + 1 < t->levels ? t->pyr[set][0].l[i + 1] : t->fg_more[set][i];
    }
    f->a0 = t->pyr[p][0].l[0];
    f->b0 = t->pyr[set][0].l[0];
    f->out = t->pyr[g][0].l[0];
    /* The older frame's histogram is last time's newer one, if there was a last time. */
    if (t->fg_hist_set == p) {
        uint32_t *s = f->hist_a;
        f->hist_a = f->hist_b;
        f->hist_b = s;
    } else {
        run_track_op(t, OP_FG_HIST_A, VT_ALG_MVEX);
    }
    run_track_op(t, OP_FG_HIST_B, VT_ALG_MVEX);
    t->fg_hist_set = set;
    f->scene_changed = fg_scene_verdict(f);
    FG_LAP(1);
    struct job *j = t->jt;
    for (int L = t->fg_levels - 1; L >= 0; L--) {
        j->level = L;
        run_track_op(t, OP_FG_SEARCH, VT_ALG_MVEX);
        FG_LAP(2);
        run_track_op(t, OP_FG_FILTER, VT_ALG_MVEX);
        FG_LAP(3);
        if (L > 0)
            run_track_op(t, OP_FG_UP, VT_ALG_MVEX);
        FG_LAP(4);
    }
    run_track_op(t, OP_FG_MID, VT_ALG_MVEX);
    FG_LAP(5);
    run_track_op(t, OP_FG_BLEND, VT_ALG_MVEX);
    FG_LAP(6);
    run_track_op(t, OP_FG_PAD, VT_ALG_MVEX);
    img_pad_tb(&f->out);
    /* The rest of its pyramid, as any frame's, if anything is going to read it. */
    if (active) {
        struct job *b = t->jp;
        b->alg = VT_ALG_MVEX;
        b->set = g;
        b->cameras = 1;
        for (int L = 1; L < t->levels; L++) {
            run_build_op(t, OP_REDUCE, L);
            img_pad_tb(&t->pyr[g][0].l[L]);
        }
    }
    FG_LAP(7);
    if (t->fg_pic)
        run_track_op(t, OP_FG_PICTURE, VT_ALG_MVEX);
    FG_LAP(8);
#undef FG_LAP
    *gm = *m;
    gm->cmd = VT_CMD_NONE;
    gm->sc_key = 0;
    gm->dest = VT_PIPE_FLOW;
    gm->generated = 1;
    gm->acq_ns = t->last_acq_ns + (m->acq_ns - t->last_acq_ns) / 2u;
    return g;
}

/* ---- a median that reorders its input: Hoare's selection ---- */

static float select_kth(float *v, int n, int k)
{
    int lo = 0, hi = n - 1;
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

static void store_appearance(struct tracker *t, int slot)
{
    struct app_slot *a = &t->app[slot];
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
            vx_store(a->pix + r * APP_W + k, v);
            sum = vx_add(sum, v);
        }
    }
    const float mean = (float)(R(vx_hsum(sum)) / R(APP_W * APP_H));
    const vxf vm = vx_bcast(&mean);
    vxf ss = vx_zero();
    for (int r = 0; r < APP_H; r++)
        for (int k = 0; k < APP_W; k += VX_LANES) {
            const vxf z = vx_sub(vx_load(a->pix + r * APP_W + k), vm);
            vx_store(a->pix + r * APP_W + k, z);
            ss = vx_fma(z, z, ss);
        }
    a->norm = vx_hsum(ss);
    a->level = L;
    a->fx = (float)(lx - R(ix));
    a->fy = (float)(ly - R(iy));
    a->rw = t->rw;
    a->rh = t->rh;
    a->valid = 1;
}

/* Adds corners inside a part of the region at this frame, away from the features already held, each with its place in the rigid model. Of age PROBATION they are members at once. */
static void seed_part(struct tracker *t, real part, int age)
{
    float nx[VT_MAX_FEATURES], ny[VT_MAX_FEATURES];
    const int mw = (int)t->c.patch_w / 2 + 2, mh = (int)t->c.patch_h / 2 + 2;
    const int margin = mw > mh ? mw : mh;
    const int want = (int)t->c.max_features;
    /* A closed box looks a little beyond itself, as far as the operator's box: more of the thing that was only partly found joins from there. */
    const real ww = t->closed && GROW_REACH * R(t->rw) < R(t->gw) ? GROW_REACH * R(t->rw) : R(t->gw), wh = t->closed && GROW_REACH * R(t->rh) < R(t->gh) ? GROW_REACH * R(t->rh) : R(t->gh);
    const real bw = part * ww, bh = part * wh;
    const float rx = (float)(R(t->cx) - 0.5L * bw), ry = (float)(R(t->cy) - 0.5L * bh);
    /* Spacing from the whole box, whatever part of it this is. */
    const real fill = sqrtl(R(t->rw) * R(t->rh) / (SPACING_BUDGET * R(t->c.max_features)));
    const real room = fill < SPACING_MIN ? SPACING_MIN : fill, spacing = room < R(t->c.min_dist) ? room : R(t->c.min_dist);
    int found = 0;
    if (detect_begin(&t->det, &t->pyr[t->cur][0].l[0], rx, ry, (float)bw, (float)bh, margin, want, (float)spacing, t->c.corner_quality, t->det_scratch)) {
        run_track_op(t, OP_DET_TENSOR, VT_ALG_MVEX);
        run_track_op(t, OP_DET_RESPONSE, VT_ALG_MVEX);
        run_track_op(t, OP_DET_CELLS, VT_ALG_MVEX);
        found = detect_finish(&t->det, nx, ny);
    }
    const real min2 = spacing * spacing;
    const real a = R(t->ra), b = R(t->rb), inv = 1.0L / (a * a + b * b);
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
        const real dx = R(nx[i]) - R(t->cx), dy = R(ny[i]) - R(t->cy);
        t->fax[k] = (float)(inv * (a * dx + b * dy));
        t->fay[k] = (float)(inv * (a * dy - b * dx));
        t->fmu[k] = 0.0f;
        t->fmv[k] = 0.0f;
        t->fage[k] = (uint8_t)age;
        t->fkx[k] = -1.0f;
        t->fky[k] = -1.0f;
    }
}

/* A region's first features. What is in the middle of a box is what was meant: the middle's corners found the region, and the rest of the box rides along until it has shown it moves with them. A middle with too little
 * in it says nothing, and the whole box founds it. */
static void found_region(struct tracker *t)
{
    t->nf = 0;
    seed_part(t, 0.5L, PROBATION);
    if (t->nf < 2 * (int)t->c.min_features) {
        t->nf = 0;
        seed_part(t, 1.0L, PROBATION);
    } else {
        seed_part(t, 1.0L, 0);
    }
}

/* Out of patience: no position is claimed any more, but what it looks like is kept, and it can come back. */
static void lose_region(struct tracker *t)
{
    t->state = VT_ST_LOST;
    t->nf = 0;
    t->kf.valid = 0;
    t->split_run = 0;
    t->hold = 0;
    t->still_run = 0;
    t->rate_u = t->rate_v = 0.0f;
}

static void end_region(struct tracker *t, uint32_t state)
{
    t->state = state;
    t->nf = 0;
    for (int i = 0; i < APP_SLOTS; i++)
        t->app[i].valid = 0;
    t->app_ticks = 0;
    t->app_turn = 0;
    t->scan_y = 1;
    t->cand_run = 0;
    t->kf.valid = 0;
    t->split_run = 0;
    t->want_app = 0;
    t->closed = 0;
    t->recentring = 0;
    t->hold = 0;
    t->still_run = 0;
    t->ra = 1.0f;
    t->rb = 0.0f;
    t->spread = 0.0f;
    t->speed = 0.0f;
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
        t->gw = t->rw;
        t->gh = t->rh;
        t->rate_u = t->rate_v = 0.0f;
        t->missing = 0;
        found_region(t);
        if (t->nf < (int)t->c.min_features) {
            /* Too little texture to hold: say so, invent nothing. */
            o->flags |= VT_F_NO_FEATURES;
            end_region(t, VT_ST_LOST);
        } else {
            t->last_obs_ns = d->acq_ns;
            store_appearance(t, 0);
        }
    }
}

/* A rate decays per real frame, however many are made between them. */
static inline void predict_position(struct tracker *t, real dt, int generated)
{
    if (dt > 0.0L) {
        t->cx = (float)(R(t->cx) + R(t->rate_u) * dt);
        t->cy = (float)(R(t->cy) + R(t->rate_v) * dt);
        if (!generated) {
            t->rate_u = (float)(R(t->rate_u) * COAST_DAMP);
            t->rate_v = (float)(R(t->rate_v) * COAST_DAMP);
        }
    }
}

static float select_median(float *v, int n) { return select_kth(v, n, n / 2); }

/* Votes for one motion: the weight of the features whose remembered motion lies within tol of it. Lanes must be a whole number of vectors, the unused ones of weight zero. */
static float mode_votes(const float *au, const float *av, const float *wt, int lanes, float u0, float v0, float tol)
{
    static const uint32_t ABS_LANE VX_ALIGNED = 0x7fffffffu;
    const float uq VX_ALIGNED = u0, vq VX_ALIGNED = v0, lim VX_ALIGNED = tol;
    const vxf vu = vx_bcast(&uq), vv = vx_bcast(&vq), vl = vx_bcast(&lim), zero = vx_zero();
    vxf s = zero;
    for (int k = 0; k < lanes; k += VX_LANES) {
        const vxf e = vx_add(vx_abs(vx_sub(vx_load(au + k), vu), &ABS_LANE), vx_abs(vx_sub(vx_load(av + k), vv), &ABS_LANE));
        s = vx_add(s, vx_blend(vx_cmp(e, vl, VX_CMP_LE), zero, vx_load(wt + k)));
    }
    return vx_hsum(s);
}

/* The median of those marked. Keeps its input: the scratch is what gets reordered. */
static real median_where(const float *v, const uint8_t *in, int n, float *scratch)
{
    int m = 0;
    for (int k = 0; k < n; k++)
        if (in[k])
            scratch[m++] = v[k];
    return m ? R(select_median(scratch, m)) : 0.0L;
}

/* The rigid model: a feature with anchor A sits at c + [a -b; b a] A. */
struct sim {
    real a, b, cx, cy;
};

/* Least squares over those marked, about their means. The scale and rotation stay as they were unless the fit moves a feature at the members' typical radius by more than SIM_DEAD, or the features are too close
 * together to say: a held model cannot push a still region along, which is what noise in a per-frame stretch does when the features' middle is off the centre. At least one must be marked. */
static void fit_region(const float *nx, const float *ny, const float *ax, const float *ay, const uint8_t *in, int n, real a0, real b0, struct sim *s)
{
    real sx = 0.0L, sy = 0.0L, sa = 0.0L, sb = 0.0L;
    int m = 0;
    for (int k = 0; k < n; k++)
        if (in[k]) {
            sx += R(nx[k]);
            sy += R(ny[k]);
            sa += R(ax[k]);
            sb += R(ay[k]);
            m++;
        }
    const real mx = sx / R(m), my = sy / R(m), ma = sa / R(m), mb = sb / R(m);
    real den = 0.0L, pa = 0.0L, pb = 0.0L;
    for (int k = 0; k < n; k++)
        if (in[k]) {
            const real A = R(ax[k]) - ma, B = R(ay[k]) - mb, X = R(nx[k]) - mx, Y = R(ny[k]) - my;
            den += A * A + B * B;
            pa += X * A + Y * B;
            pb += Y * A - X * B;
        }
    s->a = a0;
    s->b = b0;
    if (den >= R(m) * SIM_MIN_RADIUS2) {
        const real fa = pa / den, fb = pb / den, da = fa - a0, db = fb - b0;
        const real now = fa * fa + fb * fb, was = a0 * a0 + b0 * b0;
        if ((da * da + db * db) * (den / R(m)) > SIM_DEAD * SIM_DEAD && now >= SIM_MAX_STEP_DOWN * was && now <= SIM_MAX_STEP_UP * was) {
            s->a = fa;
            s->b = fb;
        }
    }
    s->cx = mx - (s->a * ma - s->b * mb);
    s->cy = my - (s->b * ma + s->a * mb);
}

/* How far a feature stands from where the model puts it. */
static inline real off_model(const struct sim *s, real x, real y, real A, real B, real *ex, real *ey)
{
    *ex = x - (s->cx + s->a * A - s->b * B);
    *ey = y - (s->cy + s->b * A + s->a * B);
    return fabsl(*ex) + fabsl(*ey);
}

/* Moves the region's centre without moving anything else: the anchors take up the difference. */
static void rename_centre(struct tracker *t, real dx, real dy)
{
    const real a = R(t->ra), b = R(t->rb), inv = 1.0L / (a * a + b * b), dA = inv * (a * dx + b * dy), dB = inv * (a * dy - b * dx);
    for (int i = 0; i < t->nf; i++) {
        t->fax[i] = (float)(R(t->fax[i]) - dA);
        t->fay[i] = (float)(R(t->fay[i]) - dB);
    }
    t->cx = (float)(R(t->cx) + dx);
    t->cy = (float)(R(t->cy) + dy);
}

/* The middle of nine in ten of the members, and half their span. False with too few to say. */
static int members_span(const struct tracker *t, real *mx, real *my, real *half_x, real *half_y)
{
    static float sx[VT_MAX_FEATURES], sy[VT_MAX_FEATURES];
    int m = 0;
    for (int i = 0; i < t->nf; i++)
        if (t->fage[i] >= PROBATION) {
            sx[m] = t->fx[i];
            sy[m] = t->fy[i];
            m++;
        }
    if (m < 2)
        return 0;
    const int lo = m / 10, hi = m - 1 - lo;
    const real x0 = R(select_kth(sx, m, lo)), x1 = R(select_kth(sx, m, hi)), y0 = R(select_kth(sy, m, lo)), y1 = R(select_kth(sy, m, hi));
    *mx = 0.5L * (x0 + x1);
    *my = 0.5L * (y0 + y1);
    *half_x = 0.5L * (x1 - x0);
    *half_y = 0.5L * (y1 - y0);
    return 1;
}

/* A box centred on the region that holds those members, with a margin. */
static inline void box_for_span(const struct tracker *t, real mx, real my, real hx, real hy, real *w, real *h)
{
    *w = 2.0L * (fabsl(mx - R(t->cx)) + hx) * BOX_ROOM + BOX_MARGIN;
    *h = 2.0L * (fabsl(my - R(t->cy)) + hy) * BOX_ROOM + BOX_MARGIN;
}

/* The box closes on the members. The centre stays where the operator put it, on the same point of the thing: that is what was selected. Only when a smaller thing found inside the box does not hold that point, since
 * it lay on the background, does the centre go to the thing; then it keeps refining as more of the thing joins. */
static void close_box(struct tracker *t, int smaller_thing)
{
    real mx, my, hx, hy, nw, nh;
    if (!members_span(t, &mx, &my, &hx, &hy))
        return;
    if (smaller_thing && (fabsl(mx - R(t->cx)) > hx || fabsl(my - R(t->cy)) > hy)) {
        rename_centre(t, mx - R(t->cx), my - R(t->cy));
        t->recentring = 1;
    }
    box_for_span(t, mx, my, hx, hy, &nw, &nh);
    if (nw < R(t->rw))
        t->rw = (float)nw;
    if (nh < R(t->rh))
        t->rh = (float)nh;
    t->closed = 1;
    t->want_app = 1;
}

/* A closed box keeps up with its members, as far as the operator's box. */
static void follow_members(struct tracker *t)
{
    real mx, my, hx, hy, nw, nh;
    if (!members_span(t, &mx, &my, &hx, &hy))
        return;
    if (t->recentring)
        rename_centre(t, RECENTRE_RATE * (mx - R(t->cx)), RECENTRE_RATE * (my - R(t->cy)));
    box_for_span(t, mx, my, hx, hy, &nw, &nh);
    const real tw = nw < R(t->gw) ? nw : R(t->gw), th = nh < R(t->gh) ? nh : R(t->gh);
    t->rw = (float)(R(t->rw) + BOX_RATE * (tw - R(t->rw)));
    t->rh = (float)(R(t->rh) + BOX_RATE * (th - R(t->rh)));
}

/* What the flow says about the region; moves or does not move it.
 *
 * A dragged box always holds background. The median of everything in it is the background's motion when the background is most of the box: zero, so the object's features look wrong and are thrown away, and the
 * region sits on the background reading OBSERVED while the object walks off (measured: frozen at one x, 187.98 px behind, continuity 1.00, scene bigbox). So a region is a rigid model and a set of members. Each
 * feature keeps its place in the model, its anchor, and claims where the centre must be given where it now stands. Members whose motion differs from the region's disagree by more with every frame, however slowly
 * they part, and stand out without a frame-to-frame threshold; the claims of those that agree average instead of accumulating. A new feature rides along without a vote until it has followed for PROBATION frames.
 * Anchors give way slowly, so that a slowly changing shape is forgiven.
 *
 * When most of the box is still and a second motion near its centre holds for WAKE_FRAMES, that is what was selected: the still members go and the box closes on what is left. When a second thing parts from a moving
 * majority, the majority is what was selected, and the box closes likewise. Otherwise the box stays as the operator drew it. */
static void region_from_flow(struct tracker *t, const struct vt_msg *d, real dt, struct vt_result *o)
{
    static float zx[VT_MAX_FEATURES + VX_LANES] VX_ALIGNED, zy[VT_MAX_FEATURES + VX_LANES] VX_ALIGNED, wt[VT_MAX_FEATURES + VX_LANES] VX_ALIGNED;
    static float nx[VT_MAX_FEATURES], ny[VT_MAX_FEATURES], ax[VT_MAX_FEATURES], ay[VT_MAX_FEATURES], mu[VT_MAX_FEATURES], mv[VT_MAX_FEATURES], scratch[VT_MAX_FEATURES];
    static int idx[VT_MAX_FEATURES];
    static uint8_t member[VT_MAX_FEATURES], in_p[VT_MAX_FEATURES], in_q[VT_MAX_FEATURES], fit[VT_MAX_FEATURES];
    const int min_f = (int)t->c.min_features;
    /* Those the flow kept. */
    int n = 0, members = 0, members_in = 0;
    for (int i = 0; i < t->nf; i++) {
        const int m = t->fage[i] >= PROBATION;
        members_in += m;
        if (t->flow[i].status)
            continue;
        idx[n] = i;
        nx[n] = t->flow[i].x;
        ny[n] = t->flow[i].y;
        ax[n] = t->fax[i];
        ay[n] = t->fay[i];
        member[n] = (uint8_t)m;
        members += m;
        n++;
    }
    /* Too few members left, and features that have followed them for a while: they are what there is. */
    if (members < min_f)
        for (int k = 0; k < n; k++)
            if (!member[k] && t->fage[idx[k]] >= PROBATION / 2) {
                member[k] = 1;
                members++;
                members_in++;
            }
    if (members >= min_f) {
        const real a0 = R(t->ra), b0 = R(t->rb), cx0 = R(t->cx), cy0 = R(t->cy), half_w = 0.5L * R(t->rw), half_h = 0.5L * R(t->rh);
        /* Two questions, two tolerances. Who places the centre: the rigid core, within the fixed tolerance or three times its spread. Who stays a member: real things are not rigid, and how far their parts disagree grows with
         * how fast they go. A fixed 1.25 px threw out nearly everything on a thing moving several pixels a frame (live, members down to 8); but one wide tolerance for both let features straddling an outline, moving at 82 %
         * of the object's speed, vote, and moved the centre 2.5 px (scene bigbox). A background the region is moving across still parts from it, by a whole frame's motion each frame. */
        const real by_spread = 3.0L * R(t->spread), tol = by_spread > RIGID_TOL ? by_spread : RIGID_TOL;
        const real by_speed = TOL_SPEED * R(t->speed), cap = R(t->w) * TOL_CAP_OF_WIDTH, capped = by_speed < cap ? by_speed : cap, keep_tol = capped > tol ? capped : tol;
        const real leak_up = ANCHOR_LEAK + ANCHOR_LEAK_SPEED * R(t->speed), leak = leak_up < ANCHOR_LEAK_MOVING ? leak_up : ANCHOR_LEAK_MOVING;
        memset(zx, 0, sizeof zx);
        memset(zy, 0, sizeof zy);
        memset(wt, 0, sizeof wt);
        real total = 0.0L;
        for (int k = 0; k < n; k++) {
            const int i = idx[k];
            const real A = R(ax[k]), B = R(ay[k]);
            zx[k] = (float)(R(nx[k]) - (a0 * A - b0 * B));
            zy[k] = (float)(R(ny[k]) - (b0 * A + a0 * B));
            mu[k] = (float)(MOVED_MEMORY * R(t->fmu[i]) + R(nx[k]) - R(t->fx[i]));
            mv[k] = (float)(MOVED_MEMORY * R(t->fmv[i]) + R(ny[k]) - R(t->fy[i]));
            if (member[k]) {
                const real rx = (R(t->fx[i]) - cx0) / half_w, ry = (R(t->fy[i]) - cy0) / half_h, r2 = rx * rx + ry * ry;
                const real w = r2 <= 0.25L ? 1.0L : r2 >= 1.0L ? 0.25L : 1.25L - r2;
                wt[k] = (float)w;
                total += w;
            }
        }
        const int lanes = (n + VX_LANES - 1) & ~(VX_LANES - 1);
        /* The strongest claim on the centre. */
        int p = 0, nq = 0, others = 0;
        float best = -1.0f;
        for (int k = 0; k < n; k++)
            if (member[k]) {
                const float v = mode_votes(zx, zy, wt, lanes, zx[k], zy[k], (float)tol);
                if (v > best) {
                    best = v;
                    p = k;
                }
            }
        for (int k = 0; k < n; k++) {
            in_p[k] = member[k] && fabsl(R(zx[k]) - R(zx[p])) + fabsl(R(zy[k]) - R(zy[p])) <= tol;
            others += member[k] && !in_p[k];
        }
        /* Is what differs from it one thing? A thing that turns as it moves makes no cluster of claims under the old model: at a degree a frame its parts are 7 px apart after five frames. So it gets a model of its own,
         * fitted to all of them and narrowed twice. */
        real second = 0.0L;
        memset(in_q, 0, (size_t)n);
        if (others >= min_f) {
            struct sim sq;
            real narrow = 4.0L * tol;
            for (int k = 0; k < n; k++)
                fit[k] = member[k] && !in_p[k];
            for (int round = 0; round < 3; round++, narrow *= 0.5L) {
                int m = 0;
                for (int k = 0; k < n; k++)
                    m += fit[k];
                if (!m)
                    break;
                fit_region(nx, ny, ax, ay, fit, n, a0, b0, &sq);
                for (int k = 0; k < n; k++) {
                    real ex, ey;
                    fit[k] = member[k] && !in_p[k] && off_model(&sq, R(nx[k]), R(ny[k]), R(ax[k]), R(ay[k]), &ex, &ey) <= narrow;
                }
            }
            for (int k = 0; k < n; k++) {
                in_q[k] = fit[k];
                nq += fit[k];
                second += fit[k] ? R(wt[k]) : 0.0L;
            }
        }
        /* A second thing near the centre of a box whose majority is still stays on trial instead of being thrown out the frame it parts: its members have not yet moved far enough to say they move. */
        const int p_moves = fabsl(median_where(mu, in_p, n, scratch)) + fabsl(median_where(mv, in_p, n, scratch)) > 2.0L * tol;
        int q_trial = 0, q_moves = 0;
        if (second >= SECOND_SHARE * total && nq >= min_f) {
            q_moves = fabsl(median_where(mu, in_q, n, scratch)) + fabsl(median_where(mv, in_q, n, scratch)) > 2.0L * tol;
            q_trial = !p_moves && fabsl(median_where(nx, in_q, n, scratch) - cx0) <= 0.5L * half_w && fabsl(median_where(ny, in_q, n, scratch) - cy0) <= 0.5L * half_h;
        }
        t->split_run = q_trial && q_moves ? t->split_run + 1 : 0;
        const int woke = t->split_run >= WAKE_FRAMES, on_trial = q_trial && !woke;

        /* The model from those followed; then from every member that model explains, so that a change of scale or rotation which only the middle agreed on at first brings the rest back in. */
        struct sim s;
        memcpy(fit, woke ? in_q : in_p, (size_t)n);
        fit_region(nx, ny, ax, ay, fit, n, a0, b0, &s);
        int nfit = 0;
        for (int k = 0; k < n; k++) {
            real ex, ey;
            fit[k] = member[k] && !(on_trial && !in_p[k]) && off_model(&s, R(nx[k]), R(ny[k]), R(ax[k]), R(ay[k]), &ex, &ey) <= tol;
            nfit += fit[k];
        }
        if (nfit)
            fit_region(nx, ny, ax, ay, fit, n, a0, b0, &s);
        const real inv = 1.0L / (s.a * s.a + s.b * s.b);
        /* A generated frame moves features and the region and decides nothing: its pixels are invented, and an error in them must not cost a member its place. A feature the flow lost on it goes where the model says. If
         * too little agrees the whole frame is passed over, and the next real frame tracks from the last real one. */
        if (d->generated) {
            int agree = 0;
            real gu = 0.0L, gv = 0.0L;
            for (int k = 0; k < n; k++) {
                real ex, ey;
                if (member[k] && !(on_trial && !in_p[k]) && off_model(&s, R(nx[k]), R(ny[k]), R(ax[k]), R(ay[k]), &ex, &ey) <= keep_tol) {
                    agree++;
                    gu += R(nx[k]) - R(t->fx[idx[k]]);
                    gv += R(ny[k]) - R(t->fy[idx[k]]);
                }
            }
            int nkeep = 0;
            for (int k = 0; k < n; k++)
                nkeep += in_p[k];
            const int asked_g = woke ? nq : on_trial ? members_in - (members - nkeep) : members_in;
            const real share_g = asked_g > 0 ? R(agree) / R(asked_g) : 0.0L;
            o->inliers = (float)(share_g < 1.0L ? share_g : 1.0L);
            if (agree < min_f || o->inliers < t->c.min_inliers) {
                t->gen_skip = 1;
                return;
            }
            int k = 0;
            for (int i = 0; i < t->nf; i++) {
                if (t->flow[i].status) {
                    const real A = R(t->fax[i]), B = R(t->fay[i]), nxm = s.cx + s.a * A - s.b * B, nym = s.cy + s.b * A + s.a * B;
                    t->fu[i] = (float)(nxm - R(t->fx[i]));
                    t->fv[i] = (float)(nym - R(t->fy[i]));
                    t->fx[i] = (float)nxm;
                    t->fy[i] = (float)nym;
                } else {
                    t->fu[i] = (float)(R(nx[k]) - R(t->fx[i]));
                    t->fv[i] = (float)(R(ny[k]) - R(t->fy[i]));
                    t->fx[i] = nx[k];
                    t->fy[i] = ny[k];
                    t->fmu[i] = mu[k];
                    t->fmv[i] = mv[k];
                    k++;
                }
                t->fflag[i] &= ~(uint32_t)VT_FF_NEW;
            }
            const real scale_g = sqrtl((s.a * s.a + s.b * s.b) / (a0 * a0 + b0 * b0)), cu_g = gu / R(agree), cv_g = gv / R(agree);
            t->ra = (float)s.a;
            t->rb = (float)s.b;
            t->cx = (float)s.cx;
            t->cy = (float)s.cy;
            t->rw = (float)(R(t->rw) * scale_g);
            t->rh = (float)(R(t->rh) * scale_g);
            t->gw = (float)(R(t->gw) * scale_g);
            t->gh = (float)(R(t->gh) * scale_g);
            t->speed = (float)(0.5L * R(t->speed) + 0.5L * (fabsl(cu_g) + fabsl(cv_g)));
            o->scale = (float)(scale_g - 1.0L);
            if (dt > 0.0L) {
                t->rate_u = (float)(cu_g / dt);
                t->rate_v = (float)(cv_g / dt);
            }
            o->du = (float)cu_g;
            o->dv = (float)cv_g;
            t->state = VT_ST_OBSERVED;
            return;
        }
        /* A rider's last step to membership wants proof: a closed region standing still cannot tell what moves with it from what does not, and would take in the background around it. */
        const real ru = fabsl(s.cx - cx0) + fabsl(s.cy - cy0);
        const int proving = !t->closed || ru >= GROW_MOTION;
        /* Keep those it explains, and a second motion still on trial, which does not move the region. The tables shrink in place: k never gets ahead of the index it reads. */
        int w = 0, kept = 0, nd = 0;
        real su = 0.0L, sv = 0.0L;
        for (int k = 0; k < n; k++) {
            const int i = idx[k];
            const struct lk_out *f = &t->flow[i];
            real ex, ey;
            const real A = R(ax[k]), B = R(ay[k]), off = off_model(&s, R(nx[k]), R(ny[k]), A, B, &ex, &ey);
            /* While a motion is on trial no member is thrown out for differing from the majority: a thing that turns as it moves shows several motions at once until a model of its own explains them all. */
            const int near = off <= keep_tol, trial = on_trial && member[k] && !in_p[k];
            if (!near && !trial)
                continue;
            const int agrees = member[k] && near && !trial;
            const real u = R(f->x) - R(t->fx[i]), v = R(f->y) - R(t->fy[i]), var = R(f->cov_xx) + R(f->cov_yy);
            const uint8_t age = t->fage[i];
            if (agrees) {
                kept++;
                su += u;
                sv += v;
                if (off <= tol)
                    scratch[nd++] = (float)off;
            }
            t->fx[w] = f->x;
            t->fy[w] = f->y;
            t->fu[w] = (float)u;
            t->fv[w] = (float)v;
            t->fsigma[w] = (float)sqrtl(var > 0.0L ? var : 0.0L);
            t->fdisp[w] = t->fdisp[i];
            t->fflag[w] = t->fflag[i] & ~(uint32_t)VT_FF_NEW;
            /* The core that places the centre gives way slowly; what is only being kept, faster, so that a part of a changing shape can come back. */
            const real give = off <= tol ? ANCHOR_LEAK : leak;
            t->fax[w] = (float)(agrees ? A + give * inv * (s.a * ex + s.b * ey) : A);
            t->fay[w] = (float)(agrees ? B + give * inv * (s.a * ey - s.b * ex) : B);
            t->fmu[w] = mu[k];
            t->fmv[w] = mv[k];
            t->fage[w] = member[k] && age < PROBATION ? PROBATION : (age < PROBATION - 1 || proving) && age < 255 ? (uint8_t)(age + 1) : age;
            t->fkx[w] = t->fkx[i];
            t->fky[w] = t->fky[i];
            w++;
        }
        /* How much of what was asked agrees. On waking the rest of the box has just been let go; while a motion is on trial it is not asked. */
        int np = 0;
        for (int k = 0; k < n; k++)
            np += in_p[k];
        const int asked = woke ? nq : on_trial ? members_in - (members - np) : members_in;
        const real share = asked > 0 ? R(kept) / R(asked) : 0.0L;
        const float ratio = (float)(share < 1.0L ? share : 1.0L);
        o->inliers = ratio;
        if (kept >= min_f && ratio >= t->c.min_inliers) {
            const real scale = sqrtl((s.a * s.a + s.b * s.b) / (a0 * a0 + b0 * b0)), cu = su / R(kept), cv = sv / R(kept);
            t->nf = w;
            if (nd)
                t->spread = (float)(SPREAD_MEMORY * R(t->spread) + (1.0L - SPREAD_MEMORY) * R(select_median(scratch, nd)));
            t->ra = (float)s.a;
            t->rb = (float)s.b;
            t->cx = (float)s.cx;
            t->cy = (float)s.cy;
            t->rw = (float)(R(t->rw) * scale);
            t->rh = (float)(R(t->rh) * scale);
            t->gw = (float)(R(t->gw) * scale);
            t->gh = (float)(R(t->gh) * scale);
            if (woke)
                t->split_run = 0;
            /* A region that moves is what still moves with it: its box closes on its members, once. Waiting for a sign of what left failed: 51 of 128 features died in the flow as an object set off, 7 parted, none was
             * riding, and the box and its picture of the background stayed (scene comeback). */
            if (woke || (p_moves && !on_trial && !t->closed))
                close_box(t, woke);
            else if (t->closed)
                follow_members(t);
            t->still_run = fabsl(cu) + fabsl(cv) < HOLD_STILL ? t->still_run + 1 : 0;
            t->speed = (float)(0.5L * R(t->speed) + 0.5L * (fabsl(cu) + fabsl(cv)));
            o->scale = (float)(scale - 1.0L);
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
    if (d->generated) {
        t->gen_skip = 1;
        return;
    }
    /* Not observed: no motion is reported for this frame. The features are not trusted any more; the position shown is a prediction. */
    t->nf = 0;
    t->state = VT_ST_UNOBSERVED;
    t->missing++;
    predict_position(t, dt, 0);
}

/* What a search of one remembered appearance found. */
struct found {
    int slot, hx, hy; /* the template's corner, in half pixels of its level */
    real ncc, second;
};

/* The correlation at a corner in half pixels of the level. A half step averages the two neighbours, a two-tap filter: on fine texture a whole-pixel search at a coarse level can miss a true place by half a pixel and read 0.5
 * where the thing is exactly there. */
static real search_half(const struct vt_plane *pl, const struct app_slot *a, int hx, int hy)
{
    static const float HALF VX_ALIGNED = 0.5f;
    const int x = hx >> 1, y = hy >> 1, ox = hx & 1, oy = hy & 1;
    if (x < 1 || y < 1 || x + APP_W + 1 >= pl->w - 1 || y + APP_H + 1 >= pl->h - 1)
        return -1.0L;
    const vxf half = vx_bcast(&HALF);
    const float first VX_ALIGNED = plane_row(pl, y)[x];
    const vxf base = vx_bcast(&first);
    vxf st = vx_zero(), sj = vx_zero(), sjj = vx_zero();
    for (int r = 0; r < APP_H; r++) {
        const float *row = plane_row(pl, y + r) + x, *below = plane_row(pl, y + r + 1) + x;
        for (int k = 0; k < APP_W; k += VX_LANES) {
            vxf v = vx_loadu(row + k);
            if (ox)
                v = vx_mul(vx_add(v, vx_loadu(row + k + 1)), half);
            if (oy) {
                vxf w = vx_loadu(below + k);
                if (ox)
                    w = vx_mul(vx_add(w, vx_loadu(below + k + 1)), half);
                v = vx_mul(vx_add(v, w), half);
            }
            v = vx_sub(v, base);
            st = vx_fma(v, vx_load(a->pix + r * APP_W + k), st);
            sj = vx_add(sj, v);
            sjj = vx_fma(v, v, sjj);
        }
    }
    const real n = R(APP_W * APP_H), norm = R(a->norm);
    const real sa = R(vx_hsum(st)), sb = R(vx_hsum(sj)), sc = R(vx_hsum(sjj));
    const real var = sc - sb * sb / n;
    return (var <= norm * APP_MIN_ENERGY || norm <= 0.0L) ? 0.0L : sa / sqrtl(norm * var);
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

/* The best place for a remembered appearance in a window of template corners, and the best place clear of it: a thing's peak stands alone, a repeating pattern's or a blank wall's does not. The top cells of the map are
 * looked at again in half pixels before anything is judged. */
static void search_slot(struct tracker *t, int slot, int x0, int x1, int y0, int y1, int step, int alg, struct found *f)
{
    const struct app_slot *a = &t->app[slot];
    const struct vt_plane *pl = &t->pyr[t->cur][0].l[a->level];
    const int lo_x = 1, hi_x = pl->w - APP_W - 2, lo_y = 1, hi_y = pl->h - APP_H - 2;
    x0 = x0 < lo_x ? lo_x : x0;
    x1 = x1 > hi_x ? hi_x : x1;
    y0 = y0 < lo_y ? lo_y : y0;
    y1 = y1 > hi_y ? hi_y : y1;
    f->slot = slot;
    f->ncc = -2.0L;
    f->second = -2.0L;
    f->hx = f->hy = 0;
    if (x1 < x0 || y1 < y0)
        return;
    struct job *j = t->jt;
    j->slot = slot;
    j->x0 = x0;
    j->x1 = x1;
    j->y0 = y0;
    j->y1 = y1;
    j->step = step;
    j->cell_cols = (x1 - x0) / SEARCH_CELL + 1;
    j->cell_rows = (y1 - y0) / SEARCH_CELL + 1;
    run_track_op(t, OP_SEARCH, alg);
    const int cells = j->cell_cols * j->cell_rows;
    int top[SEARCH_TOP], ntop = 0;
    for (int i = 0; i < cells; i++) {
        if (t->sc_v[i] <= -2.0f || (ntop == SEARCH_TOP && t->sc_v[i] <= t->sc_v[top[ntop - 1]]))
            continue;
        int at = ntop < SEARCH_TOP ? ntop++ : SEARCH_TOP - 1;
        for (; at > 0 && t->sc_v[top[at - 1]] < t->sc_v[i]; at--)
            top[at] = top[at - 1];
        top[at] = i;
    }
    /* Around each, to half a pixel: a coarse pass may be a step away from its peak. */
    const int around = step > 1 ? 2 * step - 1 : 1;
    real score[SEARCH_TOP];
    int shx[SEARCH_TOP], shy[SEARCH_TOP];
    for (int c = 0; c < ntop; c++) {
        const int cx = 2 * t->sc_x[top[c]], cy = 2 * t->sc_y[top[c]];
        score[c] = -2.0L;
        shx[c] = cx;
        shy[c] = cy;
        for (int hy = cy - around; hy <= cy + around; hy++)
            for (int hx = cx - around; hx <= cx + around; hx++) {
                const real v = search_half(pl, a, hx, hy);
                if (v > score[c]) {
                    score[c] = v;
                    shx[c] = hx;
                    shy[c] = hy;
                }
            }
        if (score[c] > f->ncc) {
            f->ncc = score[c];
            f->hx = shx[c];
            f->hy = shy[c];
        }
    }
    for (int c = 0; c < ntop; c++) {
        const int dx = shx[c] - f->hx, dy = shy[c] - f->hy;
        /* In half pixels: clear of the peak by half a template. */
        if ((dx * dx > APP_W * APP_W || dy * dy > APP_H * APP_H) && score[c] > f->second)
            f->second = score[c];
    }
}

/* In a remembered appearance's own level, the corner its template would have with the region's centre at (x, y). */
static inline void corner_for(const struct app_slot *a, real x, real y, int *cx, int *cy)
{
    const real s = 1.0L / R(1 << a->level);
    *cx = (int)(x * s - R(a->fx)) - APP_W / 2;
    *cy = (int)(y * s - R(a->fy)) - APP_H / 2;
}

/* Takes the region back where a search found it. False if too little there can be tracked. */
static int take_back(struct tracker *t, const struct found *f, struct vt_result *o)
{
    const struct app_slot *a = &t->app[f->slot];
    const struct vt_plane *pl = &t->pyr[t->cur][0].l[a->level];
    const real scale = R(1 << a->level);
    /* Between the half pixels, by a parabola: the centre stays where it is put from here on, and half a pixel of a coarse level is a pixel or two of the picture, for the rest of the track. */
    const real ox = peak_offset(search_half(pl, a, f->hx - 1, f->hy), f->ncc, search_half(pl, a, f->hx + 1, f->hy));
    const real oy = peak_offset(search_half(pl, a, f->hx, f->hy - 1), f->ncc, search_half(pl, a, f->hx, f->hy + 1));
    /* The template's corner is its centre less these whole pixels, the same whole division store_appearance used. */
    const int half_w = APP_W / 2, half_h = APP_H / 2;
    o->reacq_x = (float)((0.5L * (R(f->hx) + ox) + R(half_w) + R(a->fx)) * scale);
    o->reacq_y = (float)((0.5L * (R(f->hy) + oy) + R(half_h) + R(a->fy)) * scale);
    const float was_cx = t->cx, was_cy = t->cy, was_w = t->rw, was_h = t->rh;
    t->cx = o->reacq_x;
    t->cy = o->reacq_y;
    t->rw = a->rw;
    t->rh = a->rh;
    t->ra = 1.0f;
    t->rb = 0.0f;
    t->spread = 0.0f;
    t->speed = 0.0f;
    t->rate_u = t->rate_v = 0.0f;
    t->hold = 0;
    t->still_run = 0;
    found_region(t);
    if (t->nf < (int)t->c.min_features) {
        t->nf = 0;
        t->cx = was_cx;
        t->cy = was_cy;
        t->rw = was_w;
        t->rh = was_h;
        return 0;
    }
    o->flags |= VT_F_REACQUIRED;
    t->cand_run = 0;
    return 1;
}

/* Looks for a region out of its flow, by what it has looked like. While it can still be predicted, every memory around the prediction, believed at the configured correlation. Otherwise, or if that fails for a while, the
 * whole picture, one memory a frame, believed only on more: a higher correlation, a peak standing clear of any other, and the same place again on the next frame. There is no giving up: only the operator ends a region. */
static int try_reacquire(struct tracker *t, int alg, struct vt_result *o)
{
    int any = 0;
    for (int i = 0; i < APP_SLOTS; i++)
        any |= t->app[i].valid;
    if (!any)
        return 0;
    struct found f;
    real shown = -2.0L;
    const int predicting = t->state == VT_ST_UNOBSERVED;
    /* What was seen far away last frame: is it still there? */
    if (t->cand_run) {
        const struct app_slot *a = &t->app[t->cand_slot];
        int cx, cy;
        corner_for(a, R(t->cand_x), R(t->cand_y), &cx, &cy);
        const int r = REACQ_AGAIN_PX >> a->level;
        const int rr = r < REACQ_MIN_STEPS ? REACQ_MIN_STEPS : r;
        search_slot(t, t->cand_slot, cx - rr, cx + rr, cy - rr, cy + rr, 1, alg, &f);
        o->reacq_ncc = (float)f.ncc;
        if (f.ncc >= REACQ_FAR_NCC && take_back(t, &f, o))
            return 1;
        t->cand_run = 0;
    }
    if (predicting) {
        /* As far in the picture whatever the level: the same number of pixels of a coarse level reached 288 px at 1/8 scale, and a lookalike 160 px away took the region (live, 2026-11-03). */
        const int reach = (int)t->c.reacq_radius + (int)t->missing < 3 * (int)t->c.reacq_radius ? (int)t->c.reacq_radius + (int)t->missing : 3 * (int)t->c.reacq_radius;
        struct found near = {.ncc = -2.0L, .second = -2.0L};
        for (int i = 0; i < APP_SLOTS; i++) {
            const struct app_slot *a = &t->app[i];
            if (!a->valid)
                continue;
            int cx, cy;
            corner_for(a, R(t->cx), R(t->cy), &cx, &cy);
            const int r = (reach >> a->level) < REACQ_MIN_STEPS ? REACQ_MIN_STEPS : reach >> a->level;
            search_slot(t, i, cx - r, cx + r, cy - r, cy + r, a->level == 0 ? 2 : 1, alg, &f);
            if (f.ncc > near.ncc)
                near = f;
        }
        shown = near.ncc;
        o->reacq_ncc = (float)shown;
        if (near.ncc >= R(t->c.reacq_ncc) && take_back(t, &near, o))
            return 1;
        if (t->missing < REACQ_FAR_AFTER)
            return 0;
    }
    /* The whole picture, a band of it a frame, one memory after another. A coarse memory's scan takes a frame or two, a fine one's a second or so: a small thing's search at full resolution was 54 ms a frame by arithmetic, on
     * the path everything else waits for. */
    int slot = t->app_turn;
    for (int i = 0; i < APP_SLOTS; i++, slot = (slot + 1) % APP_SLOTS)
        if (t->app[slot].valid)
            break;
    if (slot != t->app_turn) {
        t->app_turn = slot;
        t->scan_y = 1;
    }
    const struct app_slot *a = &t->app[slot];
    const struct vt_plane *pl = &t->pyr[t->cur][0].l[a->level];
    const int step = a->level <= 1 ? 2 : 1, cols = (pl->w - APP_W) / step + 1;
    const int want_rows = REACQ_BAND * step / cols, rows = want_rows < step ? step : want_rows;
    const int band0 = t->scan_y < 1 ? 1 : t->scan_y, band1 = band0 + rows - 1;
    search_slot(t, slot, 1, pl->w, band0, band1, step, alg, &f);
    t->scan_y = band1 + 1;
    if (t->scan_y > pl->h - APP_H - 2) {
        t->scan_y = 1;
        t->app_turn = (slot + 1) % APP_SLOTS;
    }
    if (f.ncc > shown)
        o->reacq_ncc = (float)f.ncc;
    if (f.ncc >= REACQ_FAR_NCC && f.ncc - f.second >= REACQ_FAR_CLEAR) {
        const real sc = R(1 << a->level);
        const int half_w = APP_W / 2, half_h = APP_H / 2;
        t->cand_slot = slot;
        t->cand_x = (float)((0.5L * R(f.hx) + R(half_w) + R(a->fx)) * sc);
        t->cand_y = (float)((0.5L * R(f.hy) + R(half_h) + R(a->fy)) * sc);
        t->cand_run = 1;
    }
    return 0;
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
        if ((t->fflag[i] & VT_FF_3D) && t->fage[i] >= PROBATION) {
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

    const int gen = d->generated != 0;
    t->gen_skip = 0;
    if (gen)
        o->flags |= VT_F_GENERATED;
    const real dt = t->last_acq_ns ? R(d->acq_ns - t->last_acq_ns) * 1e-9L : 0.0L;
    /* The filter runs on real frames only, and its clock is theirs. */
    const real dt_est = t->last_est_ns ? R(d->acq_ns - t->last_est_ns) * 1e-9L : 0.0L;
    t->last_acq_ns = d->acq_ns;
    if (want_est)
        t->last_est_ns = d->acq_ns;

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
        t->missing += !gen;
        predict_position(t, dt, gen);
    }
    if (d->cmd != VT_CMD_NONE)
        apply_command(t, d, o);
    /* A region that left its flow: take it back from its appearance. Past its patience no position is claimed any more, and the search goes on over the whole picture until the operator says otherwise. */
    if ((t->state == VT_ST_UNOBSERVED || t->state == VT_ST_LOST) && !gen) {
        if (try_reacquire(t, alg, o)) {
            t->state = VT_ST_OBSERVED;
            t->missing = 0;
            t->last_obs_ns = d->acq_ns;
        } else if (t->state == VT_ST_UNOBSERVED && t->missing > t->c.max_missing) {
            lose_region(t);
        }
    }
    if (t->state == VT_ST_OBSERVED && !(o->flags & VT_F_REACQUIRED) && !gen) {
        /* Replace features as they die, but not every frame: a region with few corners stays below the line for good, and detection cost 7.2 ms a frame on real video
         * (camcorder, 275 x 320 region, 2026-11-02). Every 16th frame, or at once when the region is close to being unobservable. Refresh the appearance only from a clean frame. */
        int held = 0, in_key = 0;
        for (int i = 0; i < t->nf; i++)
            if (t->fage[i] >= PROBATION) {
                held++;
                in_key += t->hold && t->fkx[i] >= 0.0f;
            }
        /* A region standing still, with no frame held or too few members in it, holds this one. */
        if (t->still_run >= HOLD_AFTER && 2 * in_key < held) {
            run_track_op(t, OP_KEY_COPY, alg);
            for (int i = 0; i < t->nf; i++) {
                const int m = t->fage[i] >= PROBATION;
                t->fkx[i] = m ? t->fx[i] : -1.0f;
                t->fky[i] = m ? t->fy[i] : -1.0f;
            }
            t->hold = 1;
        }
        /* Counting members, not riders, and not every frame: riders on a background are all thrown out the frame after they arrive when the region moves, and the count alternated 13, 126, 15, 128 with a detection every
         * second frame (live, 433 x 537 box, 2026-11-03). */
        if ((held < (int)t->c.min_features * 2 && (d->instance % 4u) == 0u) || (t->nf < (int)t->c.max_features / 2 && (d->instance % 16u) == 8u))
            seed_part(t, 1.0L, 0);
        /* What it looks like, at several ages: the first since the box last closed, which is never replaced, and about half a second, two and eight seconds ago. A thing that comes back after a while may look like any of them. */
        if (t->want_app) {
            for (int i = 0; i < APP_SLOTS; i++)
                t->app[i].valid = 0;
            store_appearance(t, 0);
            t->want_app = 0;
            t->app_ticks = 0;
        } else if (o->inliers > 0.9f && (d->instance % 16u) == 0u) {
            t->app_ticks++;
            store_appearance(t, 1);
            if ((t->app_ticks & 3u) == 0u)
                store_appearance(t, 2);
            if ((t->app_ticks & 15u) == 0u)
                store_appearance(t, 3);
        }
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
        if (t->kf.valid && dt_est > 0.0L)
            kf_predict(&t->kf, dt_est, R(t->c.q_accel));
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
            const real patience = dt_est > 0.0L ? dt_est * R(t->c.max_missing) : 1.0L;
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
        f->flags = t->fflag[i] | (t->fage[i] >= PROBATION ? VT_FF_MEMBER : 0u);
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
    if (!t->gen_skip)
        t->prev = set;
    o->cycles[VT_T_TRACK] = vt_tsc() - t0;
}
