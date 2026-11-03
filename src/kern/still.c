/* still.c: see still.h. */
#include "still.h"

#include <string.h>

#include "img.h"
#include "lk.h"
#include "vt_util.h"
#include "vx.h"

typedef long double real;

/* The patch: at this level a pixel is four of the camera's. Single level, since the mean and the frame differ by little where anything is still, and where they do not LK fails, which says the same. */
#define PATCH 15
#define LK_ITER 6
#define LK_EPS 0.01f
#define LK_MIN_EIG 0.1f
#define LK_MAX_RESIDUAL 16.0f
#define FB_MAX 0.25f
#define VEC_OVER_SIGMA 12.0L /* LK's covariance is a lower bound: how far past it a vector must reach. Above the still room's largest, 10.4: docs/measurements.md */
#define DIFF_OVER_NOISE 3.0L /* a cell's difference against the camera's median */
#define NOISE_FRAMES 32.0L /* how slowly a camera's usual noise follows its median */
#define EVIDENCE_MIN 2 /* cells with evidence in a 3 x 3, for a cell to count as moving: real motion covers more than one, noise rarely does */
#define MARGIN 9.0f /* half a patch and a pixel: where a patch still fits */
#define CELL_MID_X 8 /* half a cell */
#define CELL_MID_Y 4
_Static_assert(VT_CELL_W == 2 * CELL_MID_X && VT_CELL_H == 2 * CELL_MID_Y && PATCH <= 2 * 9 - 1, "a cell's centre, a patch's margin");
#define TILE_SLIDE (64u * 1024u)
#define HIST_BINS 256
#define HIST_PER_GREY 16 /* a median of cell differences to 1/16 of a grey level, up to 16 levels */

static const uint32_t ABS_MASK VX_ALIGNED = 0x7fffffffu;

struct still {
    struct still_cfg c;
    int cols, rows, ncells;
    struct vt_plane mean;
    struct lk_cfg lk;
    struct vt_cell *cell;
    size_t cell_bytes;
    uint8_t *raw;      /* (cols + 2) x (rows + 2): this frame's own evidence, a border of zeros so the dilation has no edge */
    uint16_t *count;   /* frames in a cell's mean */
    float *weight;     /* for this frame */
    char **tiles;
    uint64_t frames;
    int next;          /* the next cell a thread takes */
    /* the newest frame */
    uint32_t tracked, moving;
    float count_mean, drift_u, drift_v, drift_rms, diff_median, diff_limit;
    long double noise_usual; /* the median cell, over time, while nothing has changed everywhere */
};

static inline int cols_of(int w) { return (w + VT_CELL_W - 1) / VT_CELL_W; }
static inline int rows_of(int h) { return (h + VT_CELL_H - 1) / VT_CELL_H; }
static inline size_t line_up(size_t n) { return (n + 63) & ~(size_t)63; }

size_t still_bytes(const struct still_cfg *c)
{
    const size_t n = (size_t)cols_of(c->w) * (size_t)rows_of(c->h);
    const size_t edge = (size_t)(cols_of(c->w) + 2) * (size_t)(rows_of(c->h) + 2);
    return plane_bytes(c->w, c->h) + (size_t)c->threads * line_up(lk_tile_bytes() + TILE_SLIDE) + line_up((size_t)c->threads * sizeof(char *)) + line_up(n * sizeof(struct vt_cell)) + line_up(edge) + line_up(n * sizeof(uint16_t)) + line_up(n * sizeof(float)) + line_up(sizeof(struct still)) + 16u * 64u;
}

struct still *still_create(const struct still_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx)
{
    struct still *s = alloc(ctx, line_up(sizeof *s));
    if (!s)
        return NULL;
    memset(s, 0, sizeof *s);
    s->c = *c;
    s->cols = cols_of(c->w);
    s->rows = rows_of(c->h);
    s->ncells = s->cols * s->rows;
    s->lk = (struct lk_cfg){.w = PATCH, .h = PATCH, .max_iter = LK_ITER, .eps = LK_EPS, .min_eig = LK_MIN_EIG, .max_residual = LK_MAX_RESIDUAL, .prefetch = 0};
    void *mean = alloc(ctx, plane_bytes(c->w, c->h));
    s->cell_bytes = line_up((size_t)s->ncells * sizeof(struct vt_cell));
    s->cell = alloc(ctx, s->cell_bytes);
    const size_t edge = (size_t)(s->cols + 2) * (size_t)(s->rows + 2);
    s->raw = alloc(ctx, line_up(edge));
    s->count = alloc(ctx, line_up((size_t)s->ncells * sizeof(uint16_t)));
    s->weight = alloc(ctx, line_up((size_t)s->ncells * sizeof(float)));
    s->tiles = alloc(ctx, line_up((size_t)c->threads * sizeof(char *)));
    if (!mean || !s->cell || !s->raw || !s->count || !s->weight || !s->tiles)
        return NULL;
    s->mean = plane_at(mean, c->w, c->h);
    /* Every one of these is read before it is written. Measured: the card's arena comes zeroed and the host's allocator does not, so a count left alone parted the two at the first frame. */
    memset(mean, 0, plane_bytes(c->w, c->h));
    memset(s->cell, 0, s->cell_bytes);
    memset(s->raw, 0, line_up(edge));
    memset(s->count, 0, line_up((size_t)s->ncells * sizeof(uint16_t)));
    memset(s->weight, 0, line_up((size_t)s->ncells * sizeof(float)));
    /* One scratch tile a thread, each at a different line offset: identically aligned private blocks alias in the L2 (as in track.c). */
    for (int i = 0; i < c->threads; i++) {
        char *m = alloc(ctx, line_up(lk_tile_bytes() + TILE_SLIDE));
        if (!m)
            return NULL;
        s->tiles[i] = m + ((size_t)i * 104729u % 1024u) * 64u;
    }
    return s;
}

int still_first(const struct still *s) { return s->frames == 0; }

/* Mean |frame - mean| over one cell, in Q6. */
static float cell_diff(const struct still *s, const struct vt_plane *cur, int col, int row)
{
    const int x = col * VT_CELL_W, y0 = row * VT_CELL_H, y1 = y0 + VT_CELL_H < s->c.h ? y0 + VT_CELL_H : s->c.h;
    real sum = 0.0L;
    for (int y = y0; y < y1; y++) {
        const vxf d = vx_abs(vx_sub(vx_load(plane_row(cur, y) + x), vx_load(plane_row(&s->mean, y) + x)), &ABS_MASK);
        sum += (real)vx_hsum(d);
    }
    return (float)(sum / (real)((y1 - y0) * VT_CELL_W));
}

void still_vectors(struct still *s, const struct vt_plane *cur, int tid, int n)
{
    (void)n;
    void *tile = s->tiles[tid];
    /* A cell at the picture's edge is read as far out as a patch fits. */
    const float margin = MARGIN;
    const float x_max = (float)s->c.w - 1.0f - margin, y_max = (float)s->c.h - 1.0f - margin;
    for (;;) {
        const int i = __atomic_fetch_add(&s->next, 1, __ATOMIC_RELAXED);
        if (i >= s->ncells)
            break;
        const int col = i % s->cols, row = i / s->cols;
        float x = (float)(col * VT_CELL_W + CELL_MID_X), y = (float)(row * VT_CELL_H + CELL_MID_Y);
        x = x < margin ? margin : x > x_max ? x_max : x;
        y = y < margin ? margin : y > y_max ? y_max : y;
        struct lk_out o;
        lk_track_vx(&s->mean, cur, 0, &s->lk, x, y, 0.0f, 0.0f, &o, tile);
        lk_flowback_vx(&s->mean, cur, 0, &s->lk, x, y, FB_MAX, &o, tile);
        struct vt_cell *c = &s->cell[i];
        c->u = (float)((real)o.x - (real)x);
        c->v = (float)((real)o.y - (real)y);
        c->diff = (float)((real)cell_diff(s, cur, col, row) / (real)VT_Q6);
        c->state = o.status;
        c->sigma = o.status ? 0.0f : (float)__builtin_sqrtl((real)o.cov_xx + (real)o.cov_yy);
    }
}

void still_decide(struct still *s)
{
    const int cols = s->cols, rows = s->rows, wide = cols + 2;
    const real px2 = (real)s->c.motion_px * (real)s->c.motion_px;
    /* A vector that lost its patch has moved. One that would not come back, or walked off the picture, is only unreliable: measured in a still room, noise alone fails the round trip in 770 of 1,020 cells of the
     * camcorder. A flat cell (LK_LOWEIG) has no vector at all. */
    const uint32_t bad = LK_RESIDUAL;
    static const int DX[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1}, DY[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    uint32_t hist[HIST_BINS];
    memset(hist, 0, sizeof hist);
    /* This camera's noise: the median cell. Most of a room is still, so what most cells differ by is noise, and each camera has its own. */
    for (int i = 0; i < s->ncells; i++) {
        const long b = (long)((real)s->cell[i].diff * (real)HIST_PER_GREY);
        hist[b < HIST_BINS - 1 ? b : HIST_BINS - 1]++;
    }
    uint32_t seen = 0;
    int med = 0;
    while (med < HIST_BINS - 1 && seen + hist[med] < (uint32_t)s->ncells / 2u)
        seen += hist[med++];
    const real noise = (real)med / (real)HIST_PER_GREY, over = noise * DIFF_OVER_NOISE;
    const real grey = over > (real)s->c.motion_grey ? over : (real)s->c.motion_grey;
    s->diff_median = (float)noise;
    s->diff_limit = (float)grey;
    /* The median leaps when most of the picture changes: a camera knocked, a light turned on. A limit that followed it would then see almost nothing, measured on a frame of another scene: 332 cells of 1,020. So
     * against the median as it usually is: past that, everything has moved. Only ever compared once a frame of noise exists to compare with. */
    const real usual_limit = s->noise_usual * DIFF_OVER_NOISE > (real)s->c.motion_grey ? s->noise_usual * DIFF_OVER_NOISE : (real)s->c.motion_grey;
    const int all_moved = s->frames >= 2 && noise > usual_limit;
    if (!all_moved && s->frames >= 1)
        s->noise_usual = s->frames == 1 ? noise : s->noise_usual + (noise - s->noise_usual) / NOISE_FRAMES;
    /* Each cell's own evidence, no branches: every test is 0 or 1. */
    for (int i = 0; i < s->ncells; i++) {
        const struct vt_cell *c = &s->cell[i];
        const uint32_t lk = c->state & VT_CELL_LK;
        const real u = (real)c->u, v = (real)c->v, sg = (real)c->sigma * VEC_OVER_SIGMA;
        const real need = sg * sg > px2 ? sg * sg : px2;
        const int moved = (int)(lk == 0) & (int)(u * u + v * v > need);
        const int failed = (lk & bad) != 0;
        const int changed = (real)c->diff > grey;
        s->raw[(size_t)(i / cols + 1) * (size_t)wide + (size_t)(i % cols) + 1] = (uint8_t)(moved | failed | changed);
    }
    uint32_t tracked = 0, moving = 0, stillgood = 0;
    real su = 0.0L, sv = 0.0L, suu = 0.0L, count_sum = 0.0L;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            const int i = r * cols + c;
            uint32_t votes = 0;
            for (int k = 0; k < 9; k++)
                votes += s->raw[(size_t)(r + 1 + DY[k]) * (size_t)wide + (size_t)(c + 1 + DX[k])];
            const uint32_t any = votes >= EVIDENCE_MIN || all_moved;
            const uint32_t own = s->raw[(size_t)(r + 1) * (size_t)wide + (size_t)c + 1];
            const uint32_t n = any ? 1u : (uint32_t)s->count[i] + 1u < (uint32_t)s->c.avg ? (uint32_t)s->count[i] + 1u : (uint32_t)s->c.avg;
            s->count[i] = (uint16_t)n;
            s->weight[i] = (float)(1.0L / (real)n);
            struct vt_cell *cell = &s->cell[i];
            const uint32_t lk = cell->state & VT_CELL_LK;
            cell->state = lk | (own ? VT_CELL_CHANGED : 0u) | (any ? VT_CELL_MOVING : 0u) | (n << VT_CELL_COUNT_SHIFT);
            tracked += lk == 0;
            moving += any != 0;
            count_sum += (real)n;
            if (lk == 0 && !any) {
                const real u = (real)cell->u, v = (real)cell->v;
                su += u;
                sv += v;
                suu += u * u + v * v;
                stillgood++;
            }
        }
    s->tracked = tracked;
    s->moving = moving;
    s->count_mean = (float)(count_sum / (real)s->ncells);
    const real mu = stillgood ? su / (real)stillgood : 0.0L, mv = stillgood ? sv / (real)stillgood : 0.0L;
    s->drift_u = (float)mu;
    s->drift_v = (float)mv;
    const real var = stillgood ? suu / (real)stillgood - mu * mu - mv * mv : 0.0L;
    s->drift_rms = var > 0.0L ? (float)__builtin_sqrtl(var) : 0.0f;
}

void still_update(struct still *s, const struct vt_plane *cur, int tid, int n)
{
    const int w = s->c.w;
    for (int r = tid; r < s->rows; r += n) {
        const int y0 = r * VT_CELL_H, y1 = y0 + VT_CELL_H < s->c.h ? y0 + VT_CELL_H : s->c.h;
        const float *wt = s->weight + (size_t)r * (size_t)s->cols;
        for (int y = y0; y < y1; y++) {
            const float *in = plane_row(cur, y);
            float *out = plane_row(&s->mean, y);
            for (int x = 0; x < w; x += VT_CELL_W) {
                const vxf c = vx_load(in + x);
                /* The first frame in a cell is that frame exactly: no rounding through the difference. */
                if (wt[x / VT_CELL_W] == 1.0f) {
                    vx_store(out + x, c);
                    continue;
                }
                const vxf m = vx_load(out + x);
                vx_store(out + x, vx_fma(vx_sub(c, m), vx_bcast(&wt[x / VT_CELL_W]), m));
            }
        }
        img_pad_rows(&s->mean, y0, y1);
    }
}

void still_finish(struct still *s)
{
    img_pad_tb(&s->mean);
    img_fence(); /* the next round's threads read what this one thread just stored */
    s->frames++;
    __atomic_store_n(&s->next, 0, __ATOMIC_RELEASE);
}

uint64_t still_hash(const struct still *s, uint64_t h)
{
    h = vt_fnv(h, s->cell, (size_t)s->ncells * sizeof(struct vt_cell));
    for (int y = 0; y < s->c.h; y++)
        h = vt_fnv(h, plane_row(&s->mean, y), (size_t)s->c.w * sizeof(float));
    return h;
}

const struct vt_plane *still_mean(const struct still *s) { return &s->mean; }
const struct vt_cell *still_cells(const struct still *s) { return s->cell; }
size_t still_cell_bytes(const struct still *s) { return s->cell_bytes; }

void still_motion(const struct still *s, struct vt_motion *m)
{
    m->cols = (uint32_t)s->cols;
    m->rows = (uint32_t)s->rows;
    m->tracked = s->tracked;
    m->moving = s->moving;
    m->count_mean = s->count_mean;
    m->drift_u = s->drift_u;
    m->drift_v = s->drift_v;
    m->drift_rms = s->drift_rms;
    m->diff_median = s->diff_median;
    m->diff_limit = s->diff_limit;
    m->frames = s->frames;
}
