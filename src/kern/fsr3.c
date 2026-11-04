/* fsr3.c: see fsr3.h. Against vx.h: native MVEX on the card, the lane model on the host. Every scalar is long double, so x87 on both. */
#include "fsr3.h"

#include <math.h>
#include <string.h>

#include "img.h"
#include "vx.h"

#define R(x) ((long double)(x))
#define BLOCK 16
#define Q6_TOP 16320.0f /* 255 grey levels of Q6 */

/* ffx_fsr3upscaler_common.h */
#define FP16_MIN 6.10e-05f
#define FP16_MAX 65504.0f
#define EPSILON FP16_MIN
#define LOCK_THRESHOLD 1.0f
#define LOCK_MAX 2.0f
#define AVG_LANCZOS_WEIGHT (0.74f / 16.0f)
#define ACCUM_ADDED 0.333f  /* the SDK's defaults */
#define ACCUM_DISOCCLUDED -0.333f
#define RECT_BIAS -2.3L
/* What is not in a shader, since a camera has none of it. A jitter sequence of this length is what the SDK gives Quality mode at 1.5 times: only a lock's lifetime reads it. */
#define JITTER_LENGTH 18.0L
#define NEAR_M 1.0L /* a moving block */
#define FAR_M 15.0L /* a still one: farther than fDistanceFactor reads */
#define FRAC_STEPS 64 /* of a pixel: the history sampler's table */
#define RCAS_LIMIT (0.25f - 1.0f / 16.0f)
#define DOWN_TAPS 6 /* Lanczos 2 stretched by up to 3 / 2 */

struct lane_map {
    int32_t i0[16] VX_ALIGNED, i1[16] VX_ALIGNED; /* within a load from base */
    float f[16] VX_ALIGNED;
    int base;
};

struct row_map {
    int r0, r1;
    float f;
};

/* One group of 16 output columns against the source. A shader takes a narrow 3 x 3 slice of a jittered frame and leaves the rest to the next frames' jitter; a fixed camera has none, the same slice would come back for ever, and the result
 * is a staircase (measured: 30.28 dB against 52.19 for a plain Lanczos, scene bigbox). So the whole Lanczos 2 over 4 x 4, one axis at a time, whose weights a table can hold since nothing about them changes from frame to frame. */
struct up_col {
    int32_t i[4][16] VX_ALIGNED; /* ipos - 1 .. ipos + 2, within a load from base */
    float wx[4][16] VX_ALIGNED;  /* Lanczos 2, over their sum */
    float ex[4][16] VX_ALIGNED;  /* the rectification box's weight: exp(-2.3 r^2) = exp(-2.3 x^2) exp(-2.3 y^2) */
    float rc[16] VX_ALIGNED;     /* 1 / the sum of those: a box's total weight is the product of both axes' */
    float x[16] VX_ALIGNED;      /* the pixel's centre */
    int base, block0, block1;
};

struct up_row {
    int iy, by;
    float wy[4], ey[4], rc;
};

/* A second stage's input: a wide Lanczos 2 from a bigger picture, one axis at a time. Output lanes span more than one load: two loads and a merge. */
struct down_col {
    int32_t i[DOWN_TAPS][16] VX_ALIGNED; /* within two loads from base, mod 16 */
    float w[DOWN_TAPS][16] VX_ALIGNED;
    unsigned short hi[DOWN_TAPS];        /* lanes in the second load */
    int base;
};

struct down_row {
    int r0;
    float w[DOWN_TAPS];
};

struct block {
    float mx, my;        /* the vector, render pixels: a thing here was at +(mx, my) in the last frame */
    float v4k;           /* get4KVelocity */
    float reactive, sqrt_reactive, dist, vel_acc, vel_box, vel_inst;
    int ix, iy, fx, fy;  /* render: whole and 64ths */
    int hix, hiy, hfx, hfy; /* the same at upscale */
    uint8_t moving, stamped;
};

struct fsr {
    int rw, rh, uw, uh, dw, dh, bw, bh, frame, scene_changed;
    struct vt_plane hc, hb, hm, hl, hh; /* every source row filtered sideways once: the Lanczos, the box's centre and moment, the middle two's min and max. An output row only mixes four of them. */
    int *mlist, *slist, nmoving, nstamped; /* the blocks that are not plain still ones: the rest need no touching */
    long double mv_scale;
    long double history_steps; /* how many of those steps back the history really is: more than one when frames were let go */
    struct down_col *down_cols;
    struct down_row *down_rows;
    float *temp;
    int temp_stride;
    long double scale_x, scale_y;
    float lock_decay;
    uint32_t con_bits;
    float rcas_con;
    struct vt_plane cur_q6, prev_q6;
    const struct fg_field *field;
    /* Render resolution. */
    struct vt_plane luma[2], acc_next[2], diff, sign, dis, sc_now, acc_now, inst, hist[2][4];
    struct vt_plane md[3], ms[3], sc; /* half, quarter, eighth; half */
    uint8_t *stamp;
    int stamp_stride;
    /* Upscale resolution. */
    struct vt_plane color[2], lock[2], newlock, out2[2];
    int turn; /* which of each pair is this frame's */
    struct block *blocks;
    struct up_col *cols;
    struct up_row *rows;
    struct lane_map *map_up2, *map_up4, *map_h2r;
    struct row_map *row_up2, *row_up4, *row_h2r, *row_r2u;
    struct lane_map *map_r2u;
    int32_t *hi_x, *hi_y;
    float lanczos[FRAC_STEPS][4]; /* weights of the history sampler, normalized, at t = k / FRAC_STEPS */
    int *stamp_x, *stamp_y, nstamps; /* last frame's marks, to clear */
};

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static inline int row_of(int rows, int tid, int n) { return (int)((long)rows * tid / n); }
static const uint32_t ABS_LANE VX_ALIGNED = 0x7fffffffu;
static const float ONE VX_ALIGNED = 1.0f, ZERO VX_ALIGNED = 0.0f, HALF VX_ALIGNED = 0.5f, THREE_HALVES VX_ALIGNED = 1.5f;

static inline vxf sat(vxf x, vxf z, vxf o) { return vx_min(vx_max(x, z), o); }
static inline vxf vs(const float *p) { return vx_bcast(p); }
static inline vxf vf(float v)
{
    const float a VX_ALIGNED = v;
    return vx_bcast(&a);
}

/* ---- scalars of the setup: own series, no libm trigonometry ---- */

static long double sin_pi(long double x) /* sin(pi x), |x| <= 4 */
{
    const long double pi = 3.14159265358979323846264338327950288L, a = pi * x, a2 = a * a;
    long double term = a, sum = a;
    for (int k = 1; k < 28; k++) {
        term = -term * a2 / R((2 * k) * (2 * k + 1));
        sum += term;
    }
    return sum;
}

static long double lanczos2(long double x)
{
    x = fabsl(x);
    if (x >= 2.0L)
        return 0.0L;
    if (x < R(EPSILON))
        return 1.0L;
    const long double pi = 3.14159265358979323846264338327950288L;
    return sin_pi(x) / (pi * x) * (sin_pi(0.5L * x) / (0.5L * pi * x));
}

static long double exp_series(long double x) /* x <= 0, small */
{
    long double term = 1.0L, sum = 1.0L;
    for (int k = 1; k < 60; k++) {
        term = term * x / R(k);
        sum += term;
    }
    return sum;
}

/* ---- memory ---- */

static size_t pl_bytes(int w, int h) { return (plane_bytes(w, h) + 63u) & ~(size_t)63u; }

static size_t table_bytes(int rw, int rh, int uw, int uh)
{
    const size_t bw = (size_t)(rw + BLOCK - 1) / BLOCK, bh = (size_t)(rh + BLOCK - 1) / BLOCK;
    const size_t ug = (size_t)(uw + 15) / 16, hg = (size_t)((rw + 1) / 2 + 15) / 16, rg = (size_t)(rw + 15) / 16;
    return bw * bh * sizeof(struct block) + ug * sizeof(struct up_col) + (size_t)uh * sizeof(struct up_row) + (ug + 2 * hg + rg) * sizeof(struct lane_map) + ((size_t)(rh + uh) + (size_t)rh) * sizeof(struct row_map) * 2 + (size_t)(rw + rh) * 8u + bw * bh * 2u * sizeof(int) + (size_t)((rw + 31) & ~15) * (size_t)(rh + 2) + bw * bh * 2u * sizeof(int) + (bw + bh) * sizeof(int) + 16384u;
}

size_t fsr_bytes(int rw, int rh, int uw, int uh, int dw, int dh)
{
    const int hw = (rw + 1) / 2, hh = (rh + 1) / 2, qw = (hw + 1) / 2, qh = (hh + 1) / 2, ew = (qw + 1) / 2, eh = (qh + 1) / 2;
    return (sizeof(struct fsr) + 4096u) + (size_t)(2 + 2 + 2 + 3 + 1 + 8) * pl_bytes(rw, rh) + 3u * pl_bytes(hw, hh) + 2u * pl_bytes(qw, qh) + 2u * pl_bytes(ew, eh) + 8u * pl_bytes(uw, uh) + 5u * pl_bytes(uw, rh) + table_bytes(rw, rh, uw, uh) + (dw ? ((size_t)(dw + 15) / 16) * sizeof(struct down_col) + (size_t)dh * sizeof(struct down_row) + 256u * (size_t)(uw + 79) * sizeof(float) + 4096u : 0u);
}

static inline long double sat_l(long double x);
/* A block that stands still: all of them the same. */
static void still_block(struct block *b)
{
    memset(b, 0, sizeof *b);
    b->vel_inst = 1.0f;
    b->dist = (float)sat_l(0.75L - FAR_M / 20.0L);
}

static void *take(char **m, size_t n)
{
    void *p = *m;
    *m += (n + 63u) & ~(size_t)63u;
    return p;
}

static struct vt_plane new_plane(char **m, int w, int h)
{
    void *p = take(m, pl_bytes(w, h));
    memset(p, 0, plane_bytes(w, h));
    return plane_at(p, w, h);
}

/* A source column for each of 16 destination lanes, pos = (x + .5) sw / dw - .5. */
static void make_map(struct lane_map *m, int g, int sw, int dw)
{
    int first = 0;
    for (int k = 0; k < 16; k++) {
        const long double pos = (R(g * 16 + k) + 0.5L) * R(sw) / R(dw) - 0.5L, fl = floorl(pos);
        const int ip = (int)fl;
        if (!k)
            first = ip;
        m->i0[k] = ip - first;
        m->i1[k] = ip + 1 - first;
        m->f[k] = (float)(pos - fl);
    }
    m->base = first;
}

static void make_row(struct row_map *m, int y, int sh, int dh)
{
    const long double pos = (R(y) + 0.5L) * R(sh) / R(dh) - 0.5L, fl = floorl(pos);
    m->r0 = (int)fl;
    m->r1 = (int)fl + 1;
    m->f = (float)(pos - fl);
}

struct fsr *fsr_init(void *mem, int rw, int rh, int uw, int uh, int dw, int dh, long double sharpness)
{
    char *m = mem;
    struct fsr *f = take(&m, sizeof *f);
    memset(f, 0, sizeof *f);
    f->rw = rw;
    f->rh = rh;
    f->uw = uw;
    f->uh = uh;
    f->bw = (rw + BLOCK - 1) / BLOCK;
    f->bh = (rh + BLOCK - 1) / BLOCK;
    f->scale_x = R(uw) / R(rw);
    f->scale_y = R(uh) / R(rh);
    f->dw = dw;
    f->dh = dh;
    f->mv_scale = 1.0L;
    f->lock_decay = (float)(0.1L / JITTER_LENGTH);
    /* FsrRcasCon: stops to linear. */
    f->rcas_con = (float)exp_series(-sharpness * 0.693147180559945309417232121458176568L);

    const int hw = (rw + 1) / 2, hh = (rh + 1) / 2, qw = (hw + 1) / 2, qh = (hh + 1) / 2, ew = (qw + 1) / 2, eh = (qh + 1) / 2;
    for (int k = 0; k < 2; k++) {
        f->acc_next[k] = new_plane(&m, rw, rh);
        for (int j = 0; j < 4; j++)
            f->hist[k][j] = new_plane(&m, rw, rh);
    }
    f->diff = new_plane(&m, rw, rh);
    f->sign = new_plane(&m, rw, rh);
    f->dis = new_plane(&m, rw, rh);
    f->sc_now = new_plane(&m, rw, rh);
    f->acc_now = new_plane(&m, rw, rh);
    f->inst = new_plane(&m, rw, rh);
    f->md[0] = new_plane(&m, hw, hh);
    f->ms[0] = new_plane(&m, hw, hh);
    f->md[1] = new_plane(&m, qw, qh);
    f->ms[1] = new_plane(&m, qw, qh);
    f->md[2] = new_plane(&m, ew, eh);
    f->ms[2] = new_plane(&m, ew, eh);
    f->sc = new_plane(&m, hw, hh);
    for (int k = 0; k < 2; k++) {
        f->luma[k] = new_plane(&m, rw, rh);
        f->color[k] = new_plane(&m, uw, uh);
        f->lock[k] = new_plane(&m, uw, uh);
    }
    f->hc = new_plane(&m, uw, rh);
    f->hb = new_plane(&m, uw, rh);
    f->hm = new_plane(&m, uw, rh);
    f->hl = new_plane(&m, uw, rh);
    f->hh = new_plane(&m, uw, rh);
    f->newlock = new_plane(&m, uw, uh);
    f->out2[0] = new_plane(&m, uw, uh);
    f->out2[1] = new_plane(&m, uw, uh);

    const int ug = (uw + 15) / 16, hg = (hw + 15) / 16, rg = (rw + 15) / 16;
    f->blocks = take(&m, (size_t)f->bw * (size_t)f->bh * sizeof(struct block));
    memset(f->blocks, 0, (size_t)f->bw * (size_t)f->bh * sizeof(struct block));
    for (int i = 0; i < f->bw * f->bh; i++)
        still_block(&f->blocks[i]);
    f->mlist = take(&m, (size_t)f->bw * (size_t)f->bh * sizeof(int));
    f->slist = take(&m, (size_t)f->bw * (size_t)f->bh * sizeof(int));
    f->cols = take(&m, (size_t)ug * sizeof(struct up_col));
    f->rows = take(&m, (size_t)uh * sizeof(struct up_row));
    f->map_up2 = take(&m, (size_t)hg * sizeof(struct lane_map));
    f->map_up4 = take(&m, (size_t)hg * sizeof(struct lane_map));
    f->map_h2r = take(&m, (size_t)rg * sizeof(struct lane_map));
    f->map_r2u = take(&m, (size_t)ug * sizeof(struct lane_map));
    f->row_up2 = take(&m, (size_t)hh * sizeof(struct row_map));
    f->row_up4 = take(&m, (size_t)hh * sizeof(struct row_map));
    f->row_h2r = take(&m, (size_t)rh * sizeof(struct row_map));
    f->row_r2u = take(&m, (size_t)uh * sizeof(struct row_map));
    f->hi_x = take(&m, (size_t)rw * sizeof(int32_t));
    f->hi_y = take(&m, (size_t)rh * sizeof(int32_t));
    f->stamp_x = take(&m, (size_t)f->bw * (size_t)f->bh * sizeof(int));
    f->stamp_y = take(&m, (size_t)f->bw * (size_t)f->bh * sizeof(int));
    f->stamp_stride = (rw + 31) & ~15;
    f->stamp = take(&m, (size_t)f->stamp_stride * (size_t)(rh + 2));
    memset(f->stamp, 0, (size_t)f->stamp_stride * (size_t)(rh + 2));

    for (int g = 0; g < hg; g++) {
        make_map(&f->map_up2[g], g, qw, hw);
        make_map(&f->map_up4[g], g, ew, hw);
    }
    for (int g = 0; g < rg; g++)
        make_map(&f->map_h2r[g], g, hw, rw);
    for (int g = 0; g < ug; g++)
        make_map(&f->map_r2u[g], g, rw, uw);
    for (int y = 0; y < hh; y++) {
        make_row(&f->row_up2[y], y, qh, hh);
        make_row(&f->row_up4[y], y, eh, hh);
    }
    for (int y = 0; y < rh; y++)
        make_row(&f->row_h2r[y], y, hh, rh);
    for (int y = 0; y < uh; y++)
        make_row(&f->row_r2u[y], y, rh, uh);
    /* ComputeHrPosFromLrPos */
    for (int x = 0; x < rw; x++)
        f->hi_x[x] = (int)floorl((R(x) + 0.5L) * f->scale_x);
    for (int y = 0; y < rh; y++)
        f->hi_y[y] = (int)floorl((R(y) + 0.5L) * f->scale_y);

    for (int g = 0; g < ug; g++) {
        struct up_col *c = &f->cols[g];
        int first = 0;
        for (int k = 0; k < 16; k++) {
            const int x = g * 16 + k;
            const long double pos = (R(x) + 0.5L) / f->scale_x - 0.5L, is = floorl(pos);
            const int ix = (int)is;
            if (!k)
                first = ix - 1;
            long double total = 0.0L, wtotal = 0.0L;
            for (int t = 0; t < 4; t++) {
                const long double o = pos - (is + R(t - 1));
                const long double w = exp_series(RECT_BIAS * o * o), l = lanczos2(o);
                if (ix + t - 1 - first > 15)
                    return NULL; /* under 1.25 times: 16 lanes no longer fit one load */
                c->i[t][k] = ix + t - 1 - first;
                c->ex[t][k] = (float)w;
                c->wx[t][k] = (float)l;
                total += w;
                wtotal += l;
            }
            for (int t = 0; t < 4; t++)
                c->wx[t][k] = (float)(R(c->wx[t][k]) / wtotal);
            c->rc[k] = (float)(1.0L / total);
            c->x[k] = (float)(R(x) + 0.5L);
        }
        c->base = first;
        c->block0 = clampi((int)floorl((R(g * 16) + 0.5L) / f->scale_x) / BLOCK, 0, f->bw - 1);
        c->block1 = clampi((int)floorl((R(g * 16 + 15) + 0.5L) / f->scale_x) / BLOCK, 0, f->bw - 1);
    }
    for (int y = 0; y < uh; y++) {
        struct up_row *r = &f->rows[y];
        const long double pos = (R(y) + 0.5L) / f->scale_y - 0.5L, is = floorl(pos);
        long double total = 0.0L, wtotal = 0.0L;
        r->iy = (int)is;
        r->by = clampi((int)floorl((R(y) + 0.5L) / f->scale_y) / BLOCK, 0, f->bh - 1);
        for (int t = 0; t < 4; t++) {
            const long double o = pos - (is + R(t - 1));
            const long double w = exp_series(RECT_BIAS * o * o), l = lanczos2(o);
            r->ey[t] = (float)w;
            r->wy[t] = (float)l;
            total += w;
            wtotal += l;
        }
        for (int t = 0; t < 4; t++)
            r->wy[t] = (float)(R(r->wy[t]) / wtotal);
        r->rc = (float)(1.0L / total);
    }
    /* The way down, if there is one: Lanczos 2 stretched by the ratio. */
    if (dw) {
        const long double rx = R(uw) / R(dw), ry = R(uh) / R(dh);
        const int dg = (dw + 15) / 16;
        f->down_cols = take(&m, (size_t)dg * sizeof(struct down_col));
        f->down_rows = take(&m, (size_t)dh * sizeof(struct down_row));
        f->temp_stride = (uw + 79) & ~15;
        f->temp = take(&m, 256u * (size_t)f->temp_stride * sizeof(float));
        for (int g = 0; g < dg; g++) {
            struct down_col *c = &f->down_cols[g];
            for (int t = 0; t < DOWN_TAPS; t++)
                c->hi[t] = 0;
            int first = 0;
            for (int k = 0; k < 16; k++) {
                const long double pos = (R(g * 16 + k) + 0.5L) * rx - 0.5L, is = floorl(pos);
                const int ip = (int)is;
                if (!k)
                    first = ip - 2;
                long double total = 0.0L;
                for (int t = 0; t < DOWN_TAPS; t++) {
                    const long double l = lanczos2((is + R(t - 2) - pos) / rx);
                    const int at = ip + t - 2 - first;
                    if (at > 31)
                        return NULL;
                    c->i[t][k] = at & 15;
                    c->hi[t] = (unsigned short)(c->hi[t] | (unsigned)(at > 15) << k);
                    c->w[t][k] = (float)l;
                    total += l;
                }
                for (int t = 0; t < DOWN_TAPS; t++)
                    c->w[t][k] = (float)(R(c->w[t][k]) / total);
            }
            c->base = first;
        }
        for (int y = 0; y < dh; y++) {
            struct down_row *r = &f->down_rows[y];
            const long double pos = (R(y) + 0.5L) * ry - 0.5L, is = floorl(pos);
            long double total = 0.0L;
            r->r0 = (int)is - 2;
            for (int t = 0; t < DOWN_TAPS; t++) {
                const long double l = lanczos2((is + R(t - 2) - pos) / ry);
                r->w[t] = (float)l;
                total += l;
            }
            for (int t = 0; t < DOWN_TAPS; t++)
                r->w[t] = (float)(R(r->w[t]) / total);
        }
    }
    /* Lanczos2 of the history sampler, at each 64th of a pixel: the four weights over their sum. */
    for (int k = 0; k < FRAC_STEPS; k++) {
        const long double t = R(k) / R(FRAC_STEPS);
        const long double w0 = lanczos2(-1.0L - t), w1 = lanczos2(-t), w2 = lanczos2(1.0L - t), w3 = lanczos2(2.0L - t), s = w0 + w1 + w2 + w3;
        f->lanczos[k][0] = (float)(w0 / s);
        f->lanczos[k][1] = (float)(w1 / s);
        f->lanczos[k][2] = (float)(w2 / s);
        f->lanczos[k][3] = (float)(w3 / s);
    }
    return f;
}

/* ---- the frame's vectors ---- */

static inline long double sat_l(long double x) { return x < 0.0L ? 0.0L : x > 1.0L ? 1.0L : x; }

static void block_offsets(struct fsr *f, struct block *b)
{
    const long double x = R(b->mx), y = R(b->my), fx = floorl(x), fy = floorl(y);
    b->ix = (int)fx;
    b->iy = (int)fy;
    b->fx = (int)floorl((x - fx) * R(FRAC_STEPS) + 0.5L);
    b->fy = (int)floorl((y - fy) * R(FRAC_STEPS) + 0.5L);
    if (b->fx == FRAC_STEPS) {
        b->fx = 0;
        b->ix++;
    }
    if (b->fy == FRAC_STEPS) {
        b->fy = 0;
        b->iy++;
    }
    const long double ux = x * f->scale_x, uy = y * f->scale_y, ufx = floorl(ux), ufy = floorl(uy);
    b->hix = (int)ufx;
    b->hiy = (int)ufy;
    b->hfx = (int)floorl((ux - ufx) * R(FRAC_STEPS) + 0.5L);
    b->hfy = (int)floorl((uy - ufy) * R(FRAC_STEPS) + 0.5L);
    if (b->hfx == FRAC_STEPS) {
        b->hfx = 0;
        b->hix++;
    }
    if (b->hfy == FRAC_STEPS) {
        b->hfy = 0;
        b->hiy++;
    }
    const long double vx4 = x / R(f->rw) * 3840.0L, vy4 = y / R(f->rh) * 2160.0L;
    b->v4k = (float)sqrtl(vx4 * vx4 + vy4 * vy4);
}

static void clear_marks(struct fsr *f)
{
    for (int i = 0; i < f->nstamps; i++)
        for (int r = 0; r < BLOCK; r++)
            memset(f->stamp + (size_t)(f->stamp_y[i] + r) * (size_t)f->stamp_stride + (size_t)f->stamp_x[i], 0, BLOCK);
    f->nstamps = 0;
}

static void clear_blocks(struct fsr *f)
{
    for (int i = 0; i < f->nmoving; i++)
        still_block(&f->blocks[f->mlist[i]]);
    for (int i = 0; i < f->nstamped; i++)
        f->blocks[f->slist[i]].stamped = 0;
    f->nmoving = f->nstamped = 0;
}

void fsr_begin(struct fsr *f, const struct vt_plane *cur, const struct vt_plane *prev, const struct fg_field *field, long double mv_scale, long double history_steps, int scene_changed)
{
    f->cur_q6 = *cur;
    f->prev_q6 = *prev;
    f->field = field;
    f->mv_scale = mv_scale;
    f->history_steps = history_steps;
    f->scene_changed = scene_changed;
    clear_marks(f);
    clear_blocks(f);
    if (scene_changed)
        return;
    for (int by = 0; by < f->bh; by++) {
        const struct fg_vec *row = &field->v[clampi(by, 0, field->h - 1) * field->w];
        for (int bx = 0; bx < f->bw; bx++) {
            const struct fg_vec *v = &row[clampi(bx, 0, field->w - 1)];
            if (!v->x && !v->y)
                continue;
            const int me = by * f->bw + bx;
            struct block *b = &f->blocks[me];
            b->moving = 1;
            /* Half resolution in the field, so twice that here, scaled by how much of a real frame's step this frame covers. FSR_REFINE takes it below a pixel. */
            b->mx = (float)(R(2 * v->x) * mv_scale);
            b->my = (float)(R(2 * v->y) * mv_scale);
            f->mlist[f->nmoving++] = me;
        }
    }
}

/* One Lucas-Kanade step per moving block, against the older frame moved by the field's vector: the field works in whole pixels of half resolution. */
static void refine_block(struct fsr *f, int i)
{
    struct block *b = &f->blocks[i];
    if (!b->moving)
        return;
    const int bx = i % f->bw, by = i / f->bw, px = bx * BLOCK, py = by * BLOCK;
    const int dx = (int)R(b->mx), dy = (int)R(b->my);
    const int x0 = clampi(px + dx, -VT_PAD_X + 2, f->rw + VT_PAD_X - 20), y0 = py + dy;
    vxf gxx = vx_zero(), gxy = vx_zero(), gyy = vx_zero(), bxv = vx_zero(), byv = vx_zero();
    const vxf half = vs(&HALF);
    for (int r = 0; r < BLOCK; r++) {
        const int ya = clampi(y0 + r, -VT_PAD_Y + 1, f->rh + VT_PAD_Y - 2), yc = clampi(py + r, 0, f->rh - 1);
        const float *old = plane_row(&f->prev_q6, ya) + x0;
        const vxf i0 = vx_loadu(old);
        const vxf ix = vx_mul(vx_sub(vx_loadu(old + 1), vx_loadu(old - 1)), half);
        const vxf iy = vx_mul(vx_sub(vx_loadu(plane_row(&f->prev_q6, ya + 1) + x0), vx_loadu(plane_row(&f->prev_q6, ya - 1) + x0)), half);
        const vxf it = vx_sub(vx_load(plane_row(&f->cur_q6, yc) + px), i0);
        gxx = vx_fma(ix, ix, gxx);
        gxy = vx_fma(ix, iy, gxy);
        gyy = vx_fma(iy, iy, gyy);
        bxv = vx_fma(ix, it, bxv);
        byv = vx_fma(iy, it, byv);
    }
    const long double a = R(vx_hsum(gxx)), c = R(vx_hsum(gxy)), d = R(vx_hsum(gyy)), e = R(vx_hsum(bxv)), g = R(vx_hsum(byv));
    const long double det = a * d - c * c, tr = a + d;
    /* A flat or edge-only block says nothing below a pixel. */
    if (det <= 1e-4L * tr * tr || tr <= 0.0L)
        return;
    const long double ux = (d * e - c * g) / det, uy = (a * g - c * e) / det;
    if (fabsl(ux) > 1.5L || fabsl(uy) > 1.5L)
        return;
    b->mx = (float)(R(b->mx) + ux);
    b->my = (float)(R(b->my) + uy);
}

static void finish_blocks(struct fsr *f)
{
    /* The vector so far is a step between the two frames given. The history may lie further back: frames let go in between, moving on as these do. */
    /* Everything else a moving block knows, from its final vector. x87 on one thread, so never over the still ones: 8,160 of them at 1080p. */
    for (int i = 0; i < f->nmoving; i++) {
        struct block *b = &f->blocks[f->mlist[i]];
        b->mx = (float)(R(b->mx) * f->history_steps);
        b->my = (float)(R(b->my) * f->history_steps);
        block_offsets(f, b);
        b->vel_acc = (float)sat_l(R(b->v4k) * 2.0L); /* ComputeBaseAccumulationWeight */
        b->vel_box = (float)sat_l(R(b->v4k) / 20.0L);
        b->vel_inst = (float)(1.0L - sat_l(R(b->v4k) / 20.0L));
        b->dist = (float)sat_l(0.75L - NEAR_M / 20.0L);
    }
    for (int i = 0; i < f->nmoving; i++) {
        const int me = f->mlist[i], bx = me % f->bw, by = me / f->bw;
        struct block *b = &f->blocks[me];
        /* ComputeMotionDivergence: at the block this one came from. */
        const int cx = bx * BLOCK + BLOCK / 2 + b->ix, cy = by * BLOCK + BLOCK / 2 + b->iy;
        if (cx >= 0 && cy >= 0 && cx < f->rw && cy < f->rh) {
            const struct block *o = &f->blocks[(cy / BLOCK) * f->bw + cx / BLOCK];
            const long double dn = o->moving ? NEAR_M : FAR_M, dc = NEAR_M, mx = dn > dc ? dn : dc, mn = dn < dc ? dn : dc;
            const long double fr = (1.0L - sat_l(R(o->v4k) / R(b->v4k))) * (mn / mx) * sat_l(R(b->v4k) / 10.0L);
            b->reactive = (float)fr;
            b->sqrt_reactive = (float)sqrtl(fr);
        }
        /* ReconstructPrevDepth: this near thing stood there last frame. A still block under it has just been uncovered. */
        const int sx = clampi(bx * BLOCK + b->ix, 0, f->stamp_stride - BLOCK), sy = clampi(by * BLOCK + b->iy, 0, f->rh + 2 - BLOCK);
        for (int r = 0; r < BLOCK; r++)
            memset(f->stamp + (size_t)(sy + r) * (size_t)f->stamp_stride + (size_t)sx, 255, BLOCK);
        f->stamp_x[f->nstamps] = sx;
        f->stamp_y[f->nstamps] = sy;
        f->nstamps++;
        for (int oy = sy / BLOCK; oy <= (sy + BLOCK - 1) / BLOCK && oy < f->bh; oy++)
            for (int ox = sx / BLOCK; ox <= (sx + BLOCK - 1) / BLOCK && ox < f->bw; ox++)
                    if (!f->blocks[oy * f->bw + ox].stamped) {
                        f->blocks[oy * f->bw + ox].stamped = 1;
                        f->slist[f->nstamped++] = oy * f->bw + ox;
                    }
    }
}

/* ---- samplers of a plane under a block's vector, and of one plane on another's grid ---- */

static inline const float *row_ptr(const struct vt_plane *p, int x, int y)
{
    const int last = ((p->w + 15) & ~15) + VT_PAD_X - 16;
    return plane_row(p, clampi(y, -VT_PAD_Y, p->h + VT_PAD_Y - 1)) + clampi(x, -VT_PAD_X, last);
}

/* Bilinear at (x + ix + fx / 64, y + iy + fy / 64), the same fractions for all 16 lanes. */
static inline vxf reproject_bilinear(const struct vt_plane *p, int x, int y, const struct block *b)
{
    if (!b->fx && !b->fy)
        return vx_loadu(row_ptr(p, x + b->ix, y + b->iy));
    const float tx VX_ALIGNED = (float)(R(b->fx) / R(FRAC_STEPS)), ty VX_ALIGNED = (float)(R(b->fy) / R(FRAC_STEPS));
    const vxf a0 = vx_loadu(row_ptr(p, x + b->ix, y + b->iy)), a1 = vx_loadu(row_ptr(p, x + b->ix + 1, y + b->iy));
    const vxf c0 = vx_loadu(row_ptr(p, x + b->ix, y + b->iy + 1)), c1 = vx_loadu(row_ptr(p, x + b->ix + 1, y + b->iy + 1));
    const vxf top = vx_fma_s(vx_sub(a1, a0), &tx, a0), bot = vx_fma_s(vx_sub(c1, c0), &tx, c0);
    return vx_fma_s(vx_sub(bot, top), &ty, top);
}

static inline vxf map_bilinear(const struct vt_plane *p, const struct lane_map *m, const struct row_map *r, vxidx i0, vxidx i1, vxf fx)
{
    const vxf a = vx_loadu(row_ptr(p, m->base, r->r0)), c = vx_loadu(row_ptr(p, m->base, r->r1));
    const vxf a0 = vx_permi(a, i0), c0 = vx_permi(c, i0);
    const vxf top = vx_fma(vx_sub(vx_permi(a, i1), a0), fx, a0), bot = vx_fma(vx_sub(vx_permi(c, i1), c0), fx, c0);
    return vx_fma_s(vx_sub(bot, top), &r->f, top);
}

/* ---- FSR_INPUT ---- */

static void stage_input(struct fsr *f, int tid, int n)
{
    static const float SCALE VX_ALIGNED = 1.0f / Q6_TOP;
    const struct vt_plane *o = &f->luma[f->turn];
    const int y0 = row_of(f->rh, tid, n), y1 = row_of(f->rh, tid + 1, n);
    for (int y = y0; y < y1; y++) {
        const float *s = plane_row(&f->cur_q6, y);
        float *d = plane_row(o, y);
        for (int x = 0; x < f->rw; x += 16)
            vx_store(d + x, vx_mul_s(vx_load(s + x), &SCALE));
    }
    img_pad_rows(o, y0, y1);
}

/* ---- FSR_HPASS: each source row across the output's columns, once ---- */

static void stage_hpass(struct fsr *f, int tid, int n)
{
    const struct vt_plane *luma = &f->luma[f->turn];
    const int y0 = row_of(f->rh, tid, n), y1 = row_of(f->rh, tid + 1, n);
    for (int y = y0; y < y1; y++) {
        float *oc = plane_row(&f->hc, y), *ob = plane_row(&f->hb, y), *om = plane_row(&f->hm, y), *ol = plane_row(&f->hl, y), *oh = plane_row(&f->hh, y);
        for (int g = 0; g * 16 < f->uw; g++) {
            const struct up_col *c = &f->cols[g];
            const vxf row = vx_loadu(row_ptr(luma, c->base, y));
            const vxf s[4] = {vx_permi(row, vx_idx(c->i[0])), vx_permi(row, vx_idx(c->i[1])), vx_permi(row, vx_idx(c->i[2])), vx_permi(row, vx_idx(c->i[3]))};
            const vxf e0 = vx_load(c->ex[0]);
            vxf h = vx_mul(s[0], vx_load(c->wx[0])), bc = vx_mul(s[0], e0), bm = vx_mul(s[0], bc);
            for (int k = 1; k < 4; k++) {
                const vxf se = vx_mul(s[k], vx_load(c->ex[k]));
                h = vx_fma(s[k], vx_load(c->wx[k]), h);
                bc = vx_add(bc, se);
                bm = vx_fma(s[k], se, bm);
            }
            const int x = g * 16;
            vx_store(oc + x, h);
            vx_store(ob + x, bc);
            vx_store(om + x, bm);
            vx_store(ol + x, vx_min(s[1], s[2]));
            vx_store(oh + x, vx_max(s[1], s[2]));
        }
    }
}

/* ---- FSR_DIFF: ComputeMinimumDifference, a lane at a time by masks ---- */

#define CSWAP(i, j)                         \
    do {                                    \
        const vxf t_ = vx_min(s[i], s[j]);  \
        s[j] = vx_max(s[i], s[j]);          \
        s[i] = t_;                          \
    } while (0)

static inline void sort5(vxf *s)
{
    CSWAP(0, 3);
    CSWAP(1, 4);
    CSWAP(0, 2);
    CSWAP(1, 3);
    CSWAP(0, 1);
    CSWAP(2, 4);
    CSWAP(1, 2);
    CSWAP(3, 4);
    CSWAP(2, 3);
}

/* s[index[lane]]: the top of the set unless the index says otherwise. */
static inline vxf pick5(const vxf *s, vxf index, const vxf *k)
{
    vxf r = s[4];
    for (int i = 0; i < 4; i++)
        r = vx_blend(vx_cmp(index, k[i], VX_CMP_EQ), r, s[i]);
    return r;
}

static void stage_diff(struct fsr *f, int tid, int n)
{
    const struct vt_plane *cur = &f->luma[f->turn], *old = &f->luma[f->turn ^ 1];
    const int y0 = row_of(f->rh, tid, n), y1 = row_of(f->rh, tid + 1, n);
    const vxf z = vs(&ZERO), one = vs(&ONE), eps = vf(EPSILON), top = vf(FP16_MAX), top1 = vf(FP16_MAX - 1.0f), fp16min = vf(FP16_MIN);
    const vxf k[4] = {z, one, vf(2.0f), vf(3.0f)};
    const int first_frame = f->frame == 0 || f->scene_changed;
    for (int y = y0; y < y1; y++) {
        const struct block *rowb = &f->blocks[clampi(y / BLOCK, 0, f->bh - 1) * f->bw];
        float *od = plane_row(&f->diff, y), *os = plane_row(&f->sign, y);
        for (int x = 0; x < f->rw; x += 16) {
            const struct block *b = &rowb[x / BLOCK];
            /* floor(uv + mv) of the older frame's pixels: the whole part. */
            const int ox = x + b->ix, oy = y + b->iy;
            if (first_frame || oy < 0 || oy >= f->rh || ox + 16 <= 0 || ox >= f->rw) {
                vx_store(od + x, z);
                vx_store(os + x, z);
                continue;
            }
            vxf s0[5], s1[5];
            s0[0] = vx_load(plane_row(cur, y) + x);
            s0[1] = vx_loadu(plane_row(cur, y) + x - 1);
            s0[2] = vx_loadu(plane_row(cur, y) + x + 1);
            s0[3] = vx_load(plane_row(cur, y - 1) + x);
            s0[4] = vx_load(plane_row(cur, y + 1) + x);
            s1[0] = vx_loadu(row_ptr(old, ox, oy));
            s1[1] = vx_loadu(row_ptr(old, ox - 1, oy));
            s1[2] = vx_loadu(row_ptr(old, ox + 1, oy));
            s1[3] = vx_loadu(row_ptr(old, ox, oy - 1));
            s1[4] = vx_loadu(row_ptr(old, ox, oy + 1));
            for (int i = 0; i < 5; i++) {
                s0[i] = vx_max(s0[i], eps);
                s1[i] = vx_max(s1[i], eps);
            }
            sort5(s0);
            sort5(s1);
            vxf md = top1, a = z, c = z;
            unsigned live = 0xffffu;
            for (int i = 0; i < 5 && live; i++) {
                const vxf sa = pick5(s0, a, k), sb = pick5(s1, c, k);
                const vxf d = vx_sub(sa, sb), ad = vx_abs(d, &ABS_LANE);
                const unsigned big = vx_cmp(fp16min, ad, VX_CMP_LT);
                const unsigned go = live & big;
                /* sign(d) (1 - min / max) */
                const vxf ratio = vx_mul(vx_min(sa, sb), vx_rcp(vx_max(sa, sb), &ONE));
                const vxf mag = vx_sub(one, ratio);
                const unsigned neg = vx_cmp(d, z, VX_CMP_LT);
                const vxf fd = vx_blend((vxk)neg, mag, vx_sub(z, mag));
                const unsigned better = vx_cmp(mag, vx_abs(md, &ABS_LANE), VX_CMP_LT);
                md = vx_blend((vxk)(go & better), md, fd);
                /* a += s0[a] < s1[b]; b += s0[a] >= s1[b], the second with a already moved. */
                const unsigned lt = vx_cmp(sa, sb, VX_CMP_LT);
                a = vx_blend((vxk)(go & lt), a, vx_add(a, one));
                const vxf sa2 = pick5(s0, a, k);
                const unsigned ge = vx_cmp(sb, sa2, VX_CMP_LE);
                c = vx_blend((vxk)(go & ge), c, vx_add(c, one));
                /* Identical samples: no change here, and no more to find. */
                md = vx_blend((vxk)(live & ~big), md, top);
                live &= big;
            }
            const unsigned keep = vx_cmp(md, top1, VX_CMP_LT);
            md = vx_blend((vxk)keep, z, md);
            const unsigned neg = vx_cmp(md, z, VX_CMP_LT), pos = vx_cmp(z, md, VX_CMP_LT);
            const vxf sg = vx_blend((vxk)neg, vx_blend((vxk)pos, z, one), vx_sub(z, one));
            vx_store(od + x, md);
            vx_store(os + x, sg);
        }
    }
    img_pad_rows(&f->diff, y0, y1);
    img_pad_rows(&f->sign, y0, y1);
}

/* ---- FSR_MIP*: SPD's reduce: the mean of four ---- */

static void reduce_row(const struct vt_plane *src, const struct vt_plane *dst, int y)
{
    static const int32_t EVEN[16] VX_ALIGNED = {0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14};
    static const float QUARTER VX_ALIGNED = 0.25f;
    const vxidx even = vx_idx(EVEN);
    const float *r0 = plane_row(src, 2 * y), *r1 = plane_row(src, 2 * y + 1);
    float *o = plane_row(dst, y);
    for (int x = 0; x < dst->w; x += 16) {
        /* 32 source lanes make 16: each pair summed where its even lane is, then the even lanes of both halves. */
        const vxf a0 = vx_add(vx_load(r0 + 2 * x), vx_load(r1 + 2 * x)), a1 = vx_add(vx_load(r0 + 2 * x + 16), vx_load(r1 + 2 * x + 16));
        const vxf b0 = vx_add(vx_loadu(r0 + 2 * x + 1), vx_loadu(r1 + 2 * x + 1)), b1 = vx_add(vx_loadu(r0 + 2 * x + 17), vx_loadu(r1 + 2 * x + 17));
        const vxf lo = vx_add(a0, b0), hi = vx_add(a1, b1);
        const vxf m = vx_perm_merge(vx_permi(lo, even), (vxk)0xff00, hi, even);
        vx_store(o + x, vx_mul_s(m, &QUARTER));
    }
}

static void stage_mip(struct fsr *f, int level, int tid, int n)
{
    const struct vt_plane *sd = level ? &f->md[level - 1] : &f->diff, *ss = level ? &f->ms[level - 1] : &f->sign;
    const struct vt_plane *dd = &f->md[level], *ds = &f->ms[level];
    const int y0 = row_of(dd->h, tid, n), y1 = row_of(dd->h, tid + 1, n);
    for (int y = y0; y < y1; y++) {
        reduce_row(sd, dd, y);
        reduce_row(ss, ds, y);
    }
    img_pad_rows(dd, y0, y1);
    img_pad_rows(ds, y0, y1);
}

/* ---- FSR_SC: the largest |difference x sign| over the three mips ---- */

static void stage_sc(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->sc.h, tid, n), y1 = row_of(f->sc.h, tid + 1, n);
    const vxf z = vs(&ZERO), one = vs(&ONE);
    for (int y = y0; y < y1; y++) {
        const float *d0 = plane_row(&f->md[0], y), *s0 = plane_row(&f->ms[0], y);
        float *o = plane_row(&f->sc, y);
        const struct row_map *r2 = &f->row_up2[y], *r4 = &f->row_up4[y];
        for (int x = 0; x < f->sc.w; x += 16) {
            const struct lane_map *m2 = &f->map_up2[x / 16], *m4 = &f->map_up4[x / 16];
            const vxidx a0 = vx_idx(m2->i0), a1 = vx_idx(m2->i1), c0 = vx_idx(m4->i0), c1 = vx_idx(m4->i1);
            const vxf fa = vx_load(m2->f), fc = vx_load(m4->f);
            vxf v = vx_abs(vx_mul(vx_load(d0 + x), vx_load(s0 + x)), &ABS_LANE);
            v = vx_max(v, vx_abs(vx_mul(map_bilinear(&f->md[1], m2, r2, a0, a1, fa), map_bilinear(&f->ms[1], m2, r2, a0, a1, fa)), &ABS_LANE));
            v = vx_max(v, vx_abs(vx_mul(map_bilinear(&f->md[2], m4, r4, c0, c1, fc), map_bilinear(&f->ms[2], m4, r4, c0, c1, fc)), &ABS_LANE));
            vx_store(o + x, sat(v, z, one));
        }
    }
    img_pad_rows(&f->sc, y0, y1);
}

/* ---- FSR_REACT: PrepareReactivity ---- */

static const uint16_t RIDGE_BITS[4] = {(1u << 1) | (1u << 2) | (1u << 4), (1u << 2) | (1u << 3) | (1u << 5), (1u << 4) | (1u << 6) | (1u << 7), (1u << 5) | (1u << 7) | (1u << 8)};

/* ComputeThinFeatureConfidence over 16 lanes: which are similar, by a mask per neighbour; a lane is a thin feature if the middle stands outside the dissimilar ones and no quarter of its neighbourhood is all similar. */
static inline void thin_features(const struct fsr *f, const struct vt_plane *l, int x, int y)
{
    static const float THRESHOLD VX_ALIGNED = 0.9f, LOCK_FLOOR VX_ALIGNED = 0.01f, BIG VX_ALIGNED = 3.4e38f;
    const float *rm = plane_row(l, y - 1), *r0 = plane_row(l, y), *rp = plane_row(l, y + 1);
    const vxf c = vx_load(r0 + x);
    const vxf nb[8] = {vx_loadu(rm + x - 1), vx_load(rm + x), vx_loadu(rm + x + 1), vx_loadu(r0 + x - 1), vx_loadu(r0 + x + 1), vx_loadu(rp + x - 1), vx_load(rp + x), vx_loadu(rp + x + 1)};
    vxf lo = c, hi = c;
    for (int i = 0; i < 8; i++) {
        lo = vx_min(lo, nb[i]);
        hi = vx_max(hi, nb[i]);
    }
    const vxf range = vx_mul_s(vx_sub(hi, lo), &THRESHOLD);
    vxf dlo = vs(&BIG), dhi = vs(&ZERO);
    unsigned similar[9];
    for (int i = 0; i < 8; i++) {
        /* |s - c| / (max - min) < 0.9, multiplied out */
        const unsigned same = vx_cmp(vx_abs(vx_sub(nb[i], c), &ABS_LANE), range, VX_CMP_LT);
        similar[i + 1] = same;
        dlo = vx_blend((vxk)(~same), dlo, vx_min(dlo, nb[i]));
        dhi = vx_blend((vxk)(~same), dhi, vx_max(dhi, nb[i]));
    }
    const unsigned ridge = vx_cmp(dhi, c, VX_CMP_LT) | vx_cmp(c, dlo, VX_CMP_LT);
    unsigned rejected = 0;
    for (int m = 0; m < 4; m++) {
        unsigned all = 0xffffu;
        for (int b = 1; b <= 8; b++)
            if (RIDGE_BITS[m] & (1u << b))
                all &= similar[b];
        rejected |= all;
    }
    unsigned thin = ridge & ~rejected & 0xffffu;
    if (!thin)
        return;
    /* 1 - min / max */
    const vxf lock = vx_sub(vs(&ONE), vx_mul(lo, vx_rcp(vx_max(hi, vf(EPSILON)), &ONE)));
    thin &= vx_cmp(vs(&LOCK_FLOOR), lock, VX_CMP_LT);
    if (!thin)
        return;
    float v[16] VX_ALIGNED;
    vx_store(v, lock);
    float *out = plane_row(&f->newlock, f->hi_y[y]);
    while (thin) {
        const int k = __builtin_ctz(thin);
        thin &= thin - 1;
        out[f->hi_x[x + k]] = v[k];
    }
}

static void stage_react(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->rh, tid, n), y1 = row_of(f->rh, tid + 1, n);
    const struct vt_plane *acc_old = &f->acc_next[f->turn ^ 1], *acc_new = &f->acc_next[f->turn], *luma = &f->luma[f->turn];
    static const float HUNDRED VX_ALIGNED = 100.0f, CENTI VX_ALIGNED = 1.0f, ADD VX_ALIGNED = ACCUM_ADDED, MIN_DIS VX_ALIGNED = ACCUM_DISOCCLUDED;
    const vxf z = vs(&ZERO), one = vs(&ONE);
    const int first_frame = f->frame == 0 || f->scene_changed;
    for (int y = y0; y < y1; y++) {
        const struct block *rowb = &f->blocks[clampi(y / BLOCK, 0, f->bh - 1) * f->bw];
        const struct row_map *rm = &f->row_h2r[y];
        float *od = plane_row(&f->dis, y), *os = plane_row(&f->sc_now, y), *oa = plane_row(&f->acc_now, y), *on = plane_row(acc_new, y);
        const uint8_t *st = f->stamp + (size_t)y * (size_t)f->stamp_stride;
        for (int x = 0; x < f->rw; x += 16) {
            const struct block *b = &rowb[x / BLOCK];
            const struct lane_map *m = &f->map_h2r[x / 16];
            const vxf sc = sat(map_bilinear(&f->sc, m, rm, vx_idx(m->i0), vx_idx(m->i1), vx_load(m->f)), z, one);
            /* Only a still block can be uncovered. */
            const vxf dis = b->stamped && !b->moving ? vx_mul_u8(vf(1.0f / 255.0f), st + x) : z;
            vxf acc = z;
            const int ry = y + b->iy;
            if (!first_frame && ry >= 0 && ry < f->rh)
                acc = sat(reproject_bilinear(acc_old, x, y, b), z, one);
            /* lerp(acc, 0, shading change); lerp(acc, min(-0.333, acc), disocclusion). */
            acc = vx_mul(acc, vx_sub(one, sc));
            acc = vx_fma(vx_sub(vx_min(vs(&MIN_DIS), acc), acc), dis, acc);
            /* acc *= round(acc * 100) > 1 */
            const unsigned on_ = vx_cmp(vs(&CENTI), vx_round(vx_mul_s(acc, &HUNDRED), VX_RND_NEAREST), VX_CMP_LT);
            acc = vx_blend((vxk)on_, z, acc);
            vx_store(on + x, sat(vx_add(acc, vs(&ADD)), z, one));
            vx_store(od + x, dis);
            vx_store(os + x, sc);
            vx_store(oa + x, acc);
            thin_features(f, luma, x, y);
        }
    }
    img_pad_rows(&f->dis, y0, y1);
    img_pad_rows(&f->sc_now, y0, y1);
    img_pad_rows(&f->acc_now, y0, y1);
    img_pad_rows(acc_new, y0, y1);
}

/* ---- FSR_INST: LumaInstability ---- */

static void stage_inst(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->rh, tid, n), y1 = row_of(f->rh, tid + 1, n);
    const struct vt_plane *cur = &f->luma[f->turn];
    const struct vt_plane *oldh = f->hist[f->turn ^ 1], *newh = f->hist[f->turn];
    static const float ACC_NEAR_FULL VX_ALIGNED = 0.9f;
    const vxf z = vs(&ZERO), one = vs(&ONE);
    for (int y = y0; y < y1; y++) {
        const struct block *rowb = &f->blocks[clampi(y / BLOCK, 0, f->bh - 1) * f->bw];
        const float *rc = plane_row(cur, y), *ra = plane_row(&f->acc_now, y), *rd = plane_row(&f->dis, y), *rs = plane_row(&f->sc_now, y);
        float *oi = plane_row(&f->inst, y), *o0 = plane_row(&newh[0], y), *o1 = plane_row(&newh[1], y), *o2 = plane_row(&newh[2], y), *o3 = plane_row(&newh[3], y);
        for (int x = 0; x < f->rw; x += 16) {
            const struct block *b = &rowb[x / BLOCK];
            const unsigned compute = vx_cmp(vs(&ACC_NEAR_FULL), vx_load(ra + x), VX_CMP_LT);
            const int ry = y + b->iy;
            if (!compute || ry < 0 || ry >= f->rh) {
                vx_store(oi + x, z);
                vx_store(o0 + x, z);
                vx_store(o1 + x, z);
                vx_store(o2 + x, z);
                vx_store(o3 + x, z);
                continue;
            }
            const vxf c = vx_load(rc + x);
            const vxf h1 = reproject_bilinear(&oldh[0], x, y, b), h2 = reproject_bilinear(&oldh[1], x, y, b), h3 = reproject_bilinear(&oldh[2], x, y, b), h4 = reproject_bilinear(&oldh[3], x, y, b);
            /* MinDividedByMax compared by cross multiplying: no division. Same sign of difference as the newest one's, and more similar: unstable. */
            const vxf n0 = vx_min(c, h1), d0 = vx_max(c, h1);
            const unsigned differs = vx_cmp(n0, d0, VX_CMP_LT);
            const unsigned pos0 = vx_cmp(h1, c, VX_CMP_LT), neg0 = vx_cmp(c, h1, VX_CMP_LT);
            unsigned unstable = 0;
            const vxf hh[3] = {h2, h3, h4};
            for (int i = 0; i < 3; i++) {
                const vxf ni = vx_min(c, hh[i]), di = vx_max(c, hh[i]);
                const unsigned posi = vx_cmp(hh[i], c, VX_CMP_LT), negi = vx_cmp(c, hh[i], VX_CMP_LT);
                const unsigned same = (pos0 & posi) | (neg0 & negi);
                unstable |= same & vx_cmp(vx_mul(n0, di), vx_mul(ni, d0), VX_CMP_LT);
            }
            unstable &= differs;
            /* * (fLumaHistory[N_MINUS_4] != 0) after the shift: h3 */
            unstable &= ~vx_cmp(h3, z, VX_CMP_EQ) & 0xffffu;
            const float vw VX_ALIGNED = b->vel_inst, rw_ VX_ALIGNED = (float)(1.0L - R(b->reactive));
            const vxf factor = vx_mul(vx_mul_s(vx_mul_s(vx_sub(one, vx_load(rd + x)), &vw), &rw_), vx_sub(one, vx_load(rs + x)));
            vx_store(oi + x, vx_blend((vxk)unstable, z, factor));
            const vxk ck = (vxk)compute;
            vx_store(o0 + x, vx_blend(ck, z, c));
            vx_store(o1 + x, vx_blend(ck, z, h1));
            vx_store(o2 + x, vx_blend(ck, z, h2));
            vx_store(o3 + x, vx_blend(ck, z, h3));
        }
    }
    img_pad_rows(&f->inst, y0, y1);
    for (int i = 0; i < 4; i++)
        img_pad_rows(&newh[i], y0, y1);
}

/* ---- FSR_ACC: Accumulate ---- */

/* The history sampler: Lanczos 2 over 4 x 4, the same weights for all 16 lanes since a block has one vector; a whole-pixel vector is a plain read, and most of a picture from a still camera is one. Deringed by the middle four. */
static inline void history_at(const struct fsr *f, const struct vt_plane *p, int x, int y, const struct block *b, vxf *out)
{
    const int hx = x + b->hix, hy = y + b->hiy;
    if (!b->hfx && !b->hfy) {
        *out = vx_loadu(row_ptr(p, hx, hy));
        return;
    }
    const float *wx = f->lanczos[b->hfx], *wy = f->lanczos[b->hfy];
    vxf acc = vx_zero(), lo = vx_zero(), hi = vx_zero();
    for (int r = 0; r < 4; r++) {
        const float *row = row_ptr(p, hx - 1, hy + r - 1);
        const vxf c0 = vx_loadu(row), c1 = vx_loadu(row + 1), c2 = vx_loadu(row + 2), c3 = vx_loadu(row + 3);
        vxf h = vx_mul_s(c0, wx);
        h = vx_fma_s(c1, wx + 1, h);
        h = vx_fma_s(c2, wx + 2, h);
        h = vx_fma_s(c3, wx + 3, h);
        acc = vx_fma_s(h, wy + r, acc);
        if (r == 1) {
            lo = vx_min(c1, c2);
            hi = vx_max(c1, c2);
        } else if (r == 2) {
            lo = vx_min(lo, vx_min(c1, c2));
            hi = vx_max(hi, vx_max(c1, c2));
        }
    }
    *out = vx_min(vx_max(acc, lo), hi);
}

static inline void accumulate_group(struct fsr *f, const struct up_col *c, const struct lane_map *m, const struct up_row *r, const struct row_map *rm, int x, int y, const struct block *b, vxk mask, vxf dis, vxf shading, vxf acc_in, vxf inst)
{
    const struct vt_plane *ho = &f->color[f->turn ^ 1], *lo = &f->lock[f->turn ^ 1], *hn = &f->color[f->turn], *ln = &f->lock[f->turn];
    (void)m;
    (void)rm;
    const vxf z = vs(&ZERO), one = vs(&ONE);
    const float react_s VX_ALIGNED = b->reactive;
    const vxf react = vs(&react_s);
    const vxf px = vx_load(c->x);
    /* params.bIsExistingSample: uv + mv inside. */
    const float dux VX_ALIGNED = (float)(R(b->mx) * f->scale_x), duy VX_ALIGNED = (float)(R(b->my) * f->scale_y);
    const long double ry = R(y) + 0.5L + R(duy);
    const int row_in = ry >= 0.0L && ry < R(f->uh);
    const float uw_f VX_ALIGNED = (float)f->uw;
    const vxf rx = vx_add(px, vs(&dux));
    unsigned exist = row_in ? (vx_cmp(z, rx, VX_CMP_LE) & vx_cmp(rx, vs(&uw_f), VX_CMP_LT)) : 0u;
    if (f->frame == 0 || f->scene_changed)
        exist = 0u;
    /* ReprojectHistoryColor */
    vxf hist = z, lock = z;
    if (exist) {
        history_at(f, ho, x, y, b, &hist);
        history_at(f, lo, x, y, b, &lock);
        hist = vx_blend((vxk)exist, z, hist);
        lock = vx_blend((vxk)exist, z, lock);
    }
    /* UpdateLockStatus */
    static const float LMAX VX_ALIGNED = LOCK_MAX;
    const vxf life = vx_max(shading, vx_max(react, dis));
    lock = vx_max(z, vx_fnma(life, vs(&LMAX), lock));
    const vxf contrib = sat(vx_sub(lock, vf(LOCK_THRESHOLD)), z, one); /* (fLockMax - fLockThreshold) = 1 */
    const vxf nl = vx_mul(vx_load(plane_row(&f->newlock, y) + x), vx_sub(one, react));
    lock = vx_max(z, vx_min(vx_add(lock, nl), vs(&LMAX)));
    lock = vx_max(z, vx_fnma(vx_sub(one, life), vf(f->lock_decay), lock));
    const vxf nx = vx_sub(px, vs(&dux));
    const long double nyv = R(y) + 0.5L - R(duy);
    const unsigned stays = nyv >= 0.0L && nyv < R(f->uh) ? vx_cmp(z, nx, VX_CMP_LE) & vx_cmp(nx, vs(&uw_f), VX_CMP_LT) : 0u;
    lock = vx_blend((vxk)stays, z, lock);
    /* ComputeBaseAccumulationWeight: min(acc, lerp(acc, 0.15, velocity)) */
    static const float P15 VX_ALIGNED = 0.15f;
    const float vel_a VX_ALIGNED = b->vel_acc;
    vxf hw = vx_min(acc_in, vx_fma_s(vx_sub(vs(&P15), acc_in), &vel_a, acc_in));
    vxf color = z, center = z, moment = z, amin = z, amax = z;
    /* The new frame: Lanczos 2 over 4 x 4, and the box from the same samples, each source row filtered already. Deringed by the middle four, as the history is. */
    for (int t = 0; t < 4; t++) {
        const int sy = r->iy + t - 1;
        color = vx_fma_s(vx_load(plane_row(&f->hc, sy) + x), &r->wy[t], color);
        center = vx_fma_s(vx_load(plane_row(&f->hb, sy) + x), &r->ey[t], center);
        moment = vx_fma_s(vx_load(plane_row(&f->hm, sy) + x), &r->ey[t], moment);
    }
    amin = vx_min(vx_load(plane_row(&f->hl, r->iy) + x), vx_load(plane_row(&f->hl, r->iy + 1) + x));
    amax = vx_max(vx_load(plane_row(&f->hh, r->iy) + x), vx_load(plane_row(&f->hh, r->iy + 1) + x));
    const vxf rcn = vx_mul_s(vx_load(c->rc), &r->rc);
    center = vx_mul(center, rcn);
    moment = vx_mul(moment, rcn);
    color = vx_min(vx_max(color, amin), amax);
    const vxf stddev = vx_sqrt(vx_abs(vx_fnma(center, center, moment), &ABS_LANE), &HALF, &THREE_HALVES);
    /* The weights sum to one; fAverageLanczosWeightPerFrame is what one frame's worth is. */
    vxf uw = vf(AVG_LANCZOS_WEIGHT);
    /* The first sample of a pixel: with no jitter history lives for over a second, so it had better be sharp, not the box's centre. */
    const unsigned initial = vx_cmp(acc_in, z, VX_CMP_EQ);
    uw = vx_blend((vxk)initial, uw, one);
    hw = vx_blend((vxk)initial, hw, z);
    /* RectifyHistory: in one channel the box is an interval, and normalize(x) * box is its near end. */
    static const float LUMA_BOX VX_ALIGNED = 1.7f;
    const float vel_box VX_ALIGNED = b->vel_box, dist VX_ALIGNED = b->dist, sqrt_r VX_ALIGNED = b->sqrt_reactive;
    const vxf tt = vx_max(vs(&vel_box), vx_max(vs(&dist), vx_max(vx_sub(one, acc_in), vx_max(vs(&sqrt_r), shading))));
    /* lerp(3, 1, t) */
    const vxf scale = vx_fnma(tt, vf(2.0f), vf(3.0f));
    const vxf box = vx_mul(vx_mul_s(stddev, &LUMA_BOX), scale);
    const vxf delta = vx_sub(hist, center);
    const unsigned outside = vx_cmp(vx_max(box, vf(1.193e-7f)), vx_abs(delta, &ABS_LANE), VX_CMP_LT);
    if (outside) {
        const vxf clamped = vx_min(vx_max(hist, vx_sub(center, box)), vx_add(center, box));
        const vxf keep = sat(vx_mul(vx_mul(vx_max(inst, contrib), acc_in), vx_sub(one, dis)), z, one);
        hist = vx_blend((vxk)outside, hist, vx_fma(vx_sub(hist, clamped), keep, clamped));
    }
    /* Accumulate */
    static const float FP16MIN VX_ALIGNED = FP16_MIN;
    const unsigned hw_on = vx_cmp(vs(&FP16MIN), hw, VX_CMP_LT);
    hw = vx_blend((vxk)hw_on, z, hw);
    hw = vx_max(vs(&FP16MIN), vx_add(hw, uw));
    const vxf alpha = sat(vx_mul(uw, vx_rcp(hw, &ONE)), z, one);
    hist = vx_max(z, vx_fma(vx_sub(color, hist), alpha, hist));
    /* Written lane by lane under the mask, since two blocks may share 16 columns. */
    float *ch = plane_row(hn, y) + x, *cl = plane_row(ln, y) + x;
    vx_store(ch, vx_blend(mask, vx_load(ch), hist));
    vx_store(cl, vx_blend(mask, vx_load(cl), lock));
}

static void stage_acc(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->uh, tid, n), y1 = row_of(f->uh, tid + 1, n);
    const vxf z = vs(&ZERO), one = vs(&ONE);
    static const float HUNDRED VX_ALIGNED = 100.0f, CENTI VX_ALIGNED = 1.0f;
    for (int y = y0; y < y1; y++) {
        const struct up_row *r = &f->rows[y];
        const struct row_map *rm = &f->row_r2u[y];
        const struct block *rowb = &f->blocks[r->by * f->bw];
        for (int g = 0; g * 16 < f->uw; g++) {
            const int x = g * 16;
            const struct up_col *c = &f->cols[g];
            const struct lane_map *m = &f->map_r2u[g];
            const vxidx i0 = vx_idx(m->i0), i1 = vx_idx(m->i1);
            const vxf fx = vx_load(m->f);
            /* SampleDilatedReactiveMasks: the fifth of a shader's cost a still camera mostly cannot use, but every lane differs. */
            const vxf dis = sat(map_bilinear(&f->dis, m, rm, i0, i1, fx), z, one);
            const vxf shading = sat(map_bilinear(&f->sc_now, m, rm, i0, i1, fx), z, one);
            vxf acc = sat(map_bilinear(&f->acc_now, m, rm, i0, i1, fx), z, one);
            const vxf inst = map_bilinear(&f->inst, m, rm, i0, i1, fx);
            /* acc *= round(acc * 100) > 1 */
            const unsigned on = vx_cmp(vs(&CENTI), vx_round(vx_mul_s(acc, &HUNDRED), VX_RND_NEAREST), VX_CMP_LT);
            acc = vx_blend((vxk)on, z, acc);
            const struct block *b0 = &rowb[c->block0], *b1 = &rowb[c->block1];
            if (c->block0 == c->block1 || (b0->mx == b1->mx && b0->my == b1->my && b0->reactive == b1->reactive && b0->moving == b1->moving)) {
                accumulate_group(f, c, m, r, rm, x, y, b0, (vxk)0xffff, dis, shading, acc, inst);
            } else {
                /* Two blocks over these 16 columns: each writes its own lanes. */
                unsigned lanes = 0;
                for (int k = 0; k < 16; k++)
                    lanes |= (unsigned)((int)floorl((R(x + k) + 0.5L) / f->scale_x) / BLOCK == c->block0) << k;
                accumulate_group(f, c, m, r, rm, x, y, b0, (vxk)lanes, dis, shading, acc, inst);
                accumulate_group(f, c, m, r, rm, x, y, b1, (vxk)(~lanes), dis, shading, acc, inst);
            }
            /* StoreNewLocks(iPxHrPos, 0) */
            vx_store(plane_row(&f->newlock, y) + x, z);
        }
    }
    img_pad_rows(&f->color[f->turn], y0, y1);
    img_pad_rows(&f->lock[f->turn], y0, y1);
}

/* ---- FSR_SHARPEN: FsrRcasF with FSR_RCAS_DENOISE, one channel: the luma of grey is twice grey, and only its ratios are read ---- */

/* A generated frame, through FSR without touching its history: the operator wants the flow first and every frame upscaled, and there are 33 ms for each. What has not moved is the history as it stands, denoised, and costs a read. What has moved is the history brought over by half a step, held to what this frame shows there. Its result is scratch: this turn's planes, which the next real frame writes over. */
static inline void mid_group(struct fsr *f, const struct up_col *c, const struct up_row *r, int x, int y, const struct block *b, vxk mask)
{
    const struct vt_plane *ho = &f->color[f->turn ^ 1];
    float *ch = plane_row(&f->color[f->turn], y) + x;
    const int first = f->frame == 0 || f->scene_changed;
    if (!b->moving && !first) {
        vx_store(ch, vx_blend(mask, vx_load(ch), vx_load(plane_row(ho, y) + x)));
        return;
    }
    const vxf z = vs(&ZERO), one = vs(&ONE);
    vxf color = z, center = z, moment = z;
    for (int t = 0; t < 4; t++) {
        const int sy = r->iy + t - 1;
        color = vx_fma_s(vx_load(plane_row(&f->hc, sy) + x), &r->wy[t], color);
        center = vx_fma_s(vx_load(plane_row(&f->hb, sy) + x), &r->ey[t], center);
        moment = vx_fma_s(vx_load(plane_row(&f->hm, sy) + x), &r->ey[t], moment);
    }
    const vxf amin = vx_min(vx_load(plane_row(&f->hl, r->iy) + x), vx_load(plane_row(&f->hl, r->iy + 1) + x));
    const vxf amax = vx_max(vx_load(plane_row(&f->hh, r->iy) + x), vx_load(plane_row(&f->hh, r->iy + 1) + x));
    const vxf rcn = vx_mul_s(vx_load(c->rc), &r->rc);
    center = vx_mul(center, rcn);
    moment = vx_mul(moment, rcn);
    color = vx_min(vx_max(color, amin), amax);
    /* Inside the picture, the history has it; outside, this frame is all there is. */
    const float dux VX_ALIGNED = (float)(R(b->mx) * f->scale_x), duy VX_ALIGNED = (float)(R(b->my) * f->scale_y), uw_f VX_ALIGNED = (float)f->uw;
    const long double ry = R(y) + 0.5L + R(duy);
    const vxf rx = vx_add(vx_load(c->x), vs(&dux));
    const unsigned exist = !first && ry >= 0.0L && ry < R(f->uh) ? vx_cmp(z, rx, VX_CMP_LE) & vx_cmp(rx, vs(&uw_f), VX_CMP_LT) : 0u;
    vxf out = color;
    if (exist) {
        vxf hist;
        history_at(f, ho, x, y, b, &hist);
        /* RectifyHistory, by how fast this block goes; then a frame's worth of what this frame shows, the less history the faster. */
        static const float LUMA_BOX VX_ALIGNED = 1.7f, P15 VX_ALIGNED = 0.15f;
        const float vel_box VX_ALIGNED = b->vel_box, dist VX_ALIGNED = b->dist, sqrt_r VX_ALIGNED = b->sqrt_reactive, vel_a VX_ALIGNED = b->vel_acc;
        const vxf stddev = vx_sqrt(vx_abs(vx_fnma(center, center, moment), &ABS_LANE), &HALF, &THREE_HALVES);
        const vxf tt = vx_max(vs(&vel_box), vx_max(vs(&dist), vs(&sqrt_r)));
        const vxf box = vx_mul(vx_mul_s(stddev, &LUMA_BOX), vx_fnma(tt, vf(2.0f), vf(3.0f)));
        hist = vx_min(vx_max(hist, vx_sub(center, box)), vx_add(center, box));
        const vxf uw = vf(AVG_LANCZOS_WEIGHT);
        const vxf hw = vx_add(vx_fma_s(vx_sub(vs(&P15), one), &vel_a, one), uw);
        const vxf alpha = sat(vx_mul(uw, vx_rcp(hw, &ONE)), z, one);
        out = vx_blend((vxk)exist, color, vx_max(z, vx_fma(vx_sub(color, hist), alpha, hist)));
    }
    vx_store(ch, vx_blend(mask, vx_load(ch), out));
}

static void stage_mid_acc(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->uh, tid, n), y1 = row_of(f->uh, tid + 1, n);
    for (int y = y0; y < y1; y++) {
        const struct up_row *r = &f->rows[y];
        const struct block *rowb = &f->blocks[r->by * f->bw];
        for (int g = 0; g * 16 < f->uw; g++) {
            const int x = g * 16;
            const struct up_col *c = &f->cols[g];
            const struct block *b0 = &rowb[c->block0], *b1 = &rowb[c->block1];
            if (c->block0 == c->block1 || (b0->mx == b1->mx && b0->my == b1->my && b0->reactive == b1->reactive && b0->moving == b1->moving)) {
                mid_group(f, c, r, x, y, b0, (vxk)0xffff);
            } else {
                unsigned lanes = 0;
                for (int k = 0; k < 16; k++)
                    lanes |= (unsigned)((int)floorl((R(x + k) + 0.5L) / f->scale_x) / BLOCK == c->block0) << k;
                mid_group(f, c, r, x, y, b0, (vxk)lanes);
                mid_group(f, c, r, x, y, b1, (vxk)(~lanes));
            }
        }
    }
    img_pad_rows(&f->color[f->turn], y0, y1);
}

static void stage_sharpen(struct fsr *f, int tid, int n)
{
    const int y0 = row_of(f->uh, tid, n), y1 = row_of(f->uh, tid + 1, n);
    const struct vt_plane *in = &f->color[f->turn];
    static const float QUARTER VX_ALIGNED = 0.25f, FOUR VX_ALIGNED = 4.0f, LIMIT VX_ALIGNED = RCAS_LIMIT, TINY VX_ALIGNED = 1e-6f;
    const float con VX_ALIGNED = f->rcas_con;
    const vxf z = vs(&ZERO), one = vs(&ONE), four = vs(&FOUR), tiny = vs(&TINY);
    for (int y = y0; y < y1; y++) {
        const float *rm = plane_row(in, y - 1), *r0 = plane_row(in, y), *rp = plane_row(in, y + 1);
        float *o = plane_row(&f->out2[f->turn], y);
        for (int x = 0; x < f->uw; x += 16) {
            const vxf b = vx_load(rm + x), d = vx_loadu(r0 + x - 1), e = vx_load(r0 + x), g = vx_loadu(r0 + x + 1), h = vx_load(rp + x);
            const vxf sum = vx_add(vx_add(b, d), vx_add(g, h));
            const vxf mn4 = vx_min(vx_min(b, d), vx_min(g, h)), mx4 = vx_max(vx_max(b, d), vx_max(g, h));
            const vxf mn5 = vx_min(mn4, e), mx5 = vx_max(mx4, e);
            /* nz: noise detection, against the local range */
            const vxf nz0 = vx_abs(vx_sub(vx_mul_s(sum, &QUARTER), e), &ABS_LANE);
            const vxf nz1 = sat(vx_mul(nz0, vx_rcp(vx_max(vx_sub(mx5, mn5), tiny), &ONE)), z, one);
            const vxf nz = vx_fnma(nz1, vs(&HALF), one);
            /* hitMin = min(mn4, e) / (4 mx4), hitMax = (1 - max(mx4, e)) / (4 mn4 - 4) */
            const vxf hit_min = vx_mul(mn5, vx_rcp(vx_max(vx_mul(four, mx4), tiny), &ONE));
            const vxf hit_max = vx_sub(z, vx_mul(vx_sub(one, mx5), vx_rcp(vx_max(vx_sub(four, vx_mul(four, mn4)), tiny), &ONE)));
            const vxf lobe0 = vx_max(vx_sub(z, hit_min), hit_max);
            const vxf lobe = vx_mul(vx_mul_s(vx_max(vx_sub(z, vs(&LIMIT)), vx_min(lobe0, z)), &con), nz);
            /* (lobe sum + e) / (4 lobe + 1) */
            const vxf rcp = vx_rcp(vx_max(vx_fma(four, lobe, one), tiny), &ONE);
            vx_store(o + x, vx_mul(vx_fma(lobe, sum, e), rcp));
        }
    }
    img_pad_rows(&f->out2[f->turn], y0, y1);
}

/* ---- the frame ---- */

void fsr_stage(struct fsr *f, int stage, int tid, int n)
{
    switch (stage) {
    case FSR_REFINE: {
        const int nb = f->nmoving;
        for (int i = row_of(nb, tid, n); i < row_of(nb, tid + 1, n); i++)
            refine_block(f, f->mlist[i]);
        break;
    }
    case FSR_INPUT: stage_input(f, tid, n); break;
    case FSR_HPASS: stage_hpass(f, tid, n); break;
    case FSR_DIFF: stage_diff(f, tid, n); break;
    case FSR_MIP0: stage_mip(f, 0, tid, n); break;
    case FSR_MIP1: stage_mip(f, 1, tid, n); break;
    case FSR_MIP2: stage_mip(f, 2, tid, n); break;
    case FSR_SC: stage_sc(f, tid, n); break;
    case FSR_REACT: stage_react(f, tid, n); break;
    case FSR_INST: stage_inst(f, tid, n); break;
    case FSR_ACC: stage_acc(f, tid, n); break;
    case FSR_MID_ACC: stage_mid_acc(f, tid, n); break;
    case FSR_SHARPEN: stage_sharpen(f, tid, n); break;
    default: break;
    }
}

void fsr_stage_end(struct fsr *f, int stage)
{
    switch (stage) {
    case FSR_REFINE: finish_blocks(f); break;
    case FSR_INPUT: img_pad_tb(&f->luma[f->turn]); break;
    case FSR_HPASS: {
        const struct vt_plane *all[5] = {&f->hc, &f->hb, &f->hm, &f->hl, &f->hh};
        for (int i = 0; i < 5; i++) {
            img_pad_tb(all[i]);
        }
        break;
    }
    case FSR_DIFF:
        img_pad_tb(&f->diff);
        img_pad_tb(&f->sign);
        break;
    case FSR_MIP0:
    case FSR_MIP1:
    case FSR_MIP2: {
        const int l = stage - FSR_MIP0;
        img_pad_tb(&f->md[l]);
        img_pad_tb(&f->ms[l]);
        break;
    }
    case FSR_SC: img_pad_tb(&f->sc); break;
    case FSR_REACT:
        img_pad_tb(&f->dis);
        img_pad_tb(&f->sc_now);
        img_pad_tb(&f->acc_now);
        img_pad_tb(&f->acc_next[f->turn]);
        break;
    case FSR_INST:
        img_pad_tb(&f->inst);
        for (int i = 0; i < 4; i++)
            img_pad_tb(&f->hist[f->turn][i]);
        break;
    case FSR_SHARPEN: img_pad_tb(&f->out2[f->turn]); break;
    case FSR_MID_ACC:
        img_pad_tb(&f->color[f->turn]);
        break;
    case FSR_ACC:
        img_pad_tb(&f->color[f->turn]);
        img_pad_tb(&f->lock[f->turn]);
        break;
    default: break;
    }
}


/* A picture on its way out: at this size, or by Lanczos down to dw x dh, across the columns of a row of source rows' sum. */
/* One row of the second stage: down the columns first, since every lane of a source row is wanted, a row keeping a margin for the sideways taps; then sideways. */
static void down_row(const struct fsr *f, const struct vt_plane *src, int y, uint8_t *d, float *t)
{
    static const float TOP VX_ALIGNED = 255.0f;
    const struct down_row *r = &f->down_rows[y];
    const float *rows[DOWN_TAPS];
    for (int k = 0; k < DOWN_TAPS; k++)
        rows[k] = plane_row(src, clampi(r->r0 + k, -VT_PAD_Y, src->h + VT_PAD_Y - 1));
    for (int x = -16; x < src->w + 32; x += 16) {
        vxf v = vx_mul_s(vx_load(rows[0] + x), &r->w[0]);
        for (int k = 1; k < DOWN_TAPS; k++)
            v = vx_fma_s(vx_load(rows[k] + x), &r->w[k], v);
        vx_store(t + 16 + x, v);
    }
    for (int g = 0; g * 16 < f->dw; g++) {
        const struct down_col *c = &f->down_cols[g];
        const float *from = t + 16 + c->base;
        const vxf lo = vx_loadu(from), hi = vx_loadu(from + 16);
        vxf v = vx_zero();
        for (int k = 0; k < DOWN_TAPS; k++) {
            const vxidx ix = vx_idx(c->i[k]);
            v = vx_fma(vx_perm_merge(vx_permi(lo, ix), (vxk)c->hi[k], hi, ix), vx_load(c->w[k]), v);
        }
        v = vx_mul_s(v, &TOP);
        const int x = g * 16;
        if (x + 16 <= f->dw) {
            vx_store_u8(d + x, v);
        } else {
            uint8_t last[16] VX_ALIGNED;
            vx_store_u8(last, v);
            memcpy(d + x, last, (size_t)(f->dw - x));
        }
    }
}

size_t fsr_down_temp_bytes(const struct fsr *f) { return (size_t)f->temp_stride * sizeof(float); }

static int down_row_wanted(const struct fsr *f, const uint8_t *row_used, int y)
{
    const struct down_row *r = &f->down_rows[y];
    int any = 0;
    for (int k = 0; k < DOWN_TAPS; k++)
        any |= row_used[clampi(r->r0 + k, 0, f->uh - 1)];
    return any;
}

int fsr_down_span(const struct fsr *f, const uint8_t *row_used, int *y0, int *y1)
{
    int first = f->dh, last = 0;
    for (int y = 0; y < f->dh; y++)
        if (down_row_wanted(f, row_used, y)) {
            first = y < first ? y : first;
            last = y + 1;
        }
    *y0 = first < last ? first : 0;
    *y1 = last;
    return first < last;
}

void fsr_down_plane(const struct fsr *f, const struct vt_plane *src, const uint8_t *row_used, int y0, int y1, uint8_t *pixels, float *temp, int tid, int n)
{
    const int a = y0 + row_of(y1 - y0, tid, n), b = y0 + row_of(y1 - y0, tid + 1, n);
    for (int y = a; y < b; y++) {
        uint8_t *d = pixels + (size_t)y * (size_t)f->dw;
        if (down_row_wanted(f, row_used, y))
            down_row(f, src, y, d, temp);
        else
            memset(d, 0, (size_t)f->dw);
    }
}

void fsr_picture(const struct fsr *f, uint8_t *pixels, int tid, int n)
{
    static const float TOP VX_ALIGNED = 255.0f;
    const struct vt_plane *src = &f->out2[f->turn];
    if (!f->dw) {
        const int y0 = row_of(f->uh, tid, n), y1 = row_of(f->uh, tid + 1, n);
        for (int y = y0; y < y1; y++) {
            const float *s = plane_row(src, y);
            uint8_t *d = pixels + (size_t)y * (size_t)f->uw;
            for (int x = 0; x + 16 <= f->uw; x += 16)
                vx_store_u8(d + x, vx_mul_s(vx_load(s + x), &TOP));
            if (f->uw & 15) {
                uint8_t last[16] VX_ALIGNED;
                const int x = f->uw & ~15;
                vx_store_u8(last, vx_mul_s(vx_load(s + x), &TOP));
                memcpy(d + x, last, (size_t)(f->uw & 15));
            }
        }
        return;
    }
    const int y0 = row_of(f->dh, tid, n), y1 = row_of(f->dh, tid + 1, n);
    float *t = f->temp + (size_t)tid * (size_t)f->temp_stride;
    for (int y = y0; y < y1; y++)
        down_row(f, src, y, pixels + (size_t)y * (size_t)f->dw, t);
}

void fsr_end(struct fsr *f)
{
    f->turn ^= 1;
    f->frame++;
}

void fsr_set_sharpness(struct fsr *f, long double stops) { f->rcas_con = (float)exp_series(-stops * 0.693147180559945309417232121458176568L); }
