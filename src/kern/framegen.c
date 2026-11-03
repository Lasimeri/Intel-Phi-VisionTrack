/* framegen.c: see framegen.h. Against vx.h: native MVEX on the card, the lane model on the host. */
#include "framegen.h"

#include <string.h>

#include "vx.h"

#define R(x) ((long double)(x))
#define FG_RADIUS 8                /* AMD's search: offsets -8 .. +7 each way, 256 places */
#define FG_TIE_UNIT 1.0f           /* Q6 counts per pixel of offset, added to a place's SAD: a quarter of a grey level at most over a block, so that it only breaks ties, toward less motion */
#define FG_ENOUGH 8192.0f          /* Q6 counts over a block: 2 grey levels a pixel, about sensor noise at half resolution */
#define FG_SCENE_L1 0.5L          /* of a section's samples in different bins, on average over the nine: the picture has changed */
#define FG_MID_FAVOUR_ZERO 64.0f   /* Q6 counts: one grey level over a block, in favour of no motion at the middle instant, for still things under sensor noise */

static const uint32_t ABS_LANE VX_ALIGNED = 0x7fffffffu;

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* A row of a plane from column x, for a load of 16 lanes that stays inside the plane's padding. Rows beyond the padding repeat its last. */
static inline const float *row_at(const struct vt_plane *p, int x, int y)
{
    const int last = ((p->w + 15) & ~15) + VT_PAD_X - 16;
    return plane_row(p, clampi(y, -VT_PAD_Y, p->h + VT_PAD_Y - 1)) + clampi(x, -VT_PAD_X, last);
}

size_t fg_bytes(int w, int h, int levels)
{
    size_t n = 2u * FG_SECTIONS * FG_BINS * sizeof(uint32_t) + 256;
    for (int L = 0; L < levels; L++) {
        const size_t blocks = (size_t)((w + FG_BLOCK - 1) / FG_BLOCK) * (size_t)((h + FG_BLOCK - 1) / FG_BLOCK);
        n += (L == 0 ? 4u : 3u) * blocks * sizeof(struct fg_vec);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    return n;
}

void fg_init(struct fg_job *j, void *mem, int w, int h, int levels)
{
    char *m = mem;
    j->levels = levels;
    j->hist_a = (uint32_t *)m;
    j->hist_b = j->hist_a + FG_SECTIONS * FG_BINS;
    m += 2u * FG_SECTIONS * FG_BINS * sizeof(uint32_t);
    for (int L = 0; L < levels; L++) {
        const int bw = (w + FG_BLOCK - 1) / FG_BLOCK, bh = (h + FG_BLOCK - 1) / FG_BLOCK;
        const size_t bytes = (size_t)bw * (size_t)bh * sizeof(struct fg_vec);
        struct fg_field *f[3] = {&j->raw[L], &j->fil[L], &j->up[L]};
        for (int k = 0; k < 3; k++) {
            f[k]->w = bw;
            f[k]->h = bh;
            f[k]->v = (struct fg_vec *)m;
            m += bytes;
        }
        if (L == 0) {
            j->mid.w = bw;
            j->mid.h = bh;
            j->mid.v = (struct fg_vec *)m;
            m += bytes;
        }
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

/* ---- scene change ---- */

void fg_hist_section(const struct vt_plane *p, uint32_t *hist, int section)
{
    /* Bins of 4 grey levels: Q6 / 256. Every second row is plenty. */
    static const float to_bin VX_ALIGNED = 1.0f / 256.0f;
    const vxf scale = vx_bcast(&to_bin);
    const int sx = section % 3, sy = section / 3;
    const int x0 = (p->w * sx / 3) & ~15, x1 = (p->w * (sx + 1) / 3) & ~15, y0 = p->h * sy / 3, y1 = p->h * (sy + 1) / 3;
    uint32_t *h = hist + section * FG_BINS;
    memset(h, 0, FG_BINS * sizeof(uint32_t));
    for (int y = y0; y < y1; y += 2) {
        const float *row = plane_row(p, y);
        for (int x = x0; x < x1; x += VX_LANES) {
            float v[16] VX_ALIGNED;
            vx_store(v, vx_mul(vx_load(row + x), scale));
            for (int k = 0; k < VX_LANES; k++)
                h[(int)v[k] & (FG_BINS - 1)]++;
        }
    }
}

int fg_scene_verdict(const struct fg_job *j)
{
    uint64_t diff = 0, total = 0;
    for (int i = 0; i < FG_SECTIONS * FG_BINS; i++) {
        const uint32_t a = j->hist_a[i], b = j->hist_b[i];
        diff += a > b ? a - b : b - a;
        total += b;
    }
    /* Half the sum of absolute differences is how many samples are in different bins. */
    return total && R(diff) * 0.5L / R(total) > FG_SCENE_L1;
}

/* ---- AMD's search ---- */

void fg_search(const struct fg_job *j, int level, int i0, int i1)
{
    const struct vt_plane *cur = &j->b[level], *old = &j->a[level];
    const struct fg_field *out = &j->raw[level], *pred = &j->up[level];
    const int top = level == j->levels - 1, row_lanes = (old->w + 15) & ~15;
    float blk[FG_BLOCK][16] VX_ALIGNED;
    float win[2 * FG_RADIUS + FG_BLOCK][32] VX_ALIGNED;
    float cost[2 * FG_RADIUS][16] VX_ALIGNED;
    float tie[16] VX_ALIGNED;
    for (int k = 0; k < 16; k++)
        tie[k] = (float)((k > FG_RADIUS ? k - FG_RADIUS : FG_RADIUS - k)) * FG_TIE_UNIT;
    const vxf tie_x = vx_load(tie);
    for (int i = i0; i < i1; i++) {
        struct fg_vec *o = &out->v[i];
        if (j->scene_changed) {
            o->x = o->y = 0;
            continue;
        }
        const int bx = i % out->w, by = i / out->w, px = bx * FG_BLOCK, py = by * FG_BLOCK;
        /* Around the level above's estimate, kept where the window is still inside the plane's padding. */
        const int want_x = top ? 0 : pred->v[i].x, want_y = top ? 0 : pred->v[i].y;
        const int x0 = clampi(px + want_x - FG_RADIUS, -VT_PAD_X, row_lanes + VT_PAD_X - 32), y0 = py + want_y - FG_RADIUS;
        const int pvx = x0 - px + FG_RADIUS, pvy = want_y;
        for (int r = 0; r < FG_BLOCK; r++)
            vx_store(blk[r], vx_loadu(row_at(cur, px, py + r)));
        /* Where AMD searches all 256 places regardless: on the last level a block the estimate already explains needs no search. Three quarters of the work is on this level, and from a still camera most blocks are this. */
        if (level == 0) {
            static const unsigned short EIGHT_LANES = 0x00ff;
            const vxf nil = vx_zero();
            vxf at_pred = nil;
            for (int r = 0; r < FG_BLOCK; r++)
                at_pred = vx_add(at_pred, vx_blend(EIGHT_LANES, nil, vx_abs(vx_sub(vx_load(blk[r]), vx_loadu(row_at(old, px + pvx, py + r + pvy))), &ABS_LANE)));
            if (vx_hsum(at_pred) <= FG_ENOUGH) {
                o->x = (int16_t)pvx;
                o->y = (int16_t)pvy;
                continue;
            }
        }
        for (int r = 0; r < 2 * FG_RADIUS + FG_BLOCK; r++) {
            const float *s = plane_row(old, clampi(y0 + r, -VT_PAD_Y, old->h + VT_PAD_Y - 1)) + x0;
            vx_store(win[r], vx_loadu(s));
            vx_store(win[r] + 16, vx_loadu(s + 16));
        }
        /* The 256 places at once: a lane per horizontal offset. Each pixel of the block against 16 places of its row, by sliding the row with valignd rather than loading it 8 times; unrolled, since the slide's count is an
         * immediate. msad4 has no equivalent here, and this needs none. */
        vxf lowest = vx_zero();
        for (int dy = 0; dy < 2 * FG_RADIUS; dy++) {
            const float dy_tie VX_ALIGNED = (float)(dy > FG_RADIUS ? dy - FG_RADIUS : FG_RADIUS - dy) * FG_TIE_UNIT;
            vxf acc = vx_add(tie_x, vx_bcast(&dy_tie));
            for (int r = 0; r < FG_BLOCK; r++) {
                const vxf lo = vx_load(win[dy + r]), hi = vx_load(win[dy + r] + 16);
                const float *c = blk[r];
#define FG_COL(n) acc = vx_add(acc, vx_abs(vx_sub(vx_alignr(hi, lo, n), vx_bcast(c + (n))), &ABS_LANE))
                acc = vx_add(acc, vx_abs(vx_sub(lo, vx_bcast(c)), &ABS_LANE));
                FG_COL(1);
                FG_COL(2);
                FG_COL(3);
                FG_COL(4);
                FG_COL(5);
                FG_COL(6);
                FG_COL(7);
#undef FG_COL
            }
            vx_store(cost[dy], acc);
            lowest = dy ? vx_min(lowest, acc) : acc;
        }
        const float best VX_ALIGNED = vx_hmin(lowest);
        const vxf vbest = vx_bcast(&best);
        int fx = FG_RADIUS, fy = FG_RADIUS;
        for (int dy = 0; dy < 2 * FG_RADIUS; dy++) {
            const unsigned k = vx_cmp(vx_load(cost[dy]), vbest, VX_CMP_EQ);
            if (k) {
                fx = __builtin_ctz(k);
                fy = dy;
                break;
            }
        }
        int vx_ = pvx + fx - FG_RADIUS, vy_ = pvy + fy - FG_RADIUS;
        /* On the last level no motion wins if it matches as well as the best found: AMD's fallback. It cannot be among the places searched when the estimate has gone elsewhere. */
        if (level == 0) {
            static const unsigned short EIGHT = 0x00ff;
            vxf sum = vx_zero();
            const vxf zero = vx_zero();
            for (int r = 0; r < FG_BLOCK; r++)
                sum = vx_add(sum, vx_blend(EIGHT, zero, vx_abs(vx_sub(vx_load(blk[r]), vx_loadu(row_at(old, px, py + r))), &ABS_LANE)));
            if (vx_hsum(sum) <= best)
                vx_ = vy_ = 0;
        }
        o->x = (int16_t)vx_;
        o->y = (int16_t)vy_;
    }
}

/* ---- AMD's filter: of the 3 x 3 around it, the vector with the least squared difference from the rest ---- */

void fg_filter(const struct fg_job *j, int level, int i0, int i1)
{
    const struct fg_field *in = &j->raw[level], *out = &j->fil[level];
    for (int i = i0; i < i1; i++) {
        const int bx = i % in->w, by = i / in->w;
        int mx[9], my[9], n = 0;
        for (int xx = -1; xx < 2; xx++)
            for (int yy = -1; yy < 2; yy++) {
                const struct fg_vec *v = &in->v[clampi(by + yy, 0, in->h - 1) * in->w + clampi(bx + xx, 0, in->w - 1)];
                mx[n] = v->x;
                my[n] = v->y;
                n++;
            }
        uint32_t ret = 0xffffffffu;
        for (int a = 0; a < 9; a++) {
            uint32_t d = 0;
            for (int b = 0; b < 9; b++) {
                const int dx = mx[a] - mx[b], dy = my[a] - my[b];
                d += (uint32_t)(dx * dx + dy * dy);
            }
            const uint32_t key = (d << 4) | (uint32_t)a;
            if (key < ret)
                ret = key;
        }
        out->v[i].x = (int16_t)mx[ret & 15u];
        out->v[i].y = (int16_t)my[ret & 15u];
    }
}

/* ---- AMD's upscale: each block of the next level picks among four of this level's by SAD over the 4 x 4 of this level it covers ---- */

static inline float sad4x4(const struct vt_plane *b, const struct vt_plane *a, int x, int y, int vx, int vy)
{
    static const unsigned short FOUR = 0x000f;
    const vxf zero = vx_zero();
    vxf sum = zero;
    for (int r = 0; r < 4; r++)
        sum = vx_add(sum, vx_blend(FOUR, zero, vx_abs(vx_sub(vx_loadu(row_at(b, x, y + r)), vx_loadu(row_at(a, x + vx, y + r + vy))), &ABS_LANE)));
    return vx_hsum(sum);
}

void fg_upscale(const struct fg_job *j, int level, int i0, int i1)
{
    const struct fg_field *in = &j->fil[level], *out = &j->up[level - 1];
    const struct vt_plane *cur = &j->b[level], *old = &j->a[level];
    for (int i = i0; i < i1; i++) {
        if (j->scene_changed) {
            out->v[i].x = out->v[i].y = 0;
            continue;
        }
        const int gx = i % out->w, gy = i / out->w;
        float best = 0.0f;
        int bestx = 0, besty = 0;
        for (int n = 0; n < 4; n++) {
            const int xo = (n % 2) - 1 + gx % 2, yo = (n / 2) - 1 + gy % 2;
            const struct fg_vec *v = &in->v[clampi(gy / 2 + yo, 0, in->h - 1) * in->w + clampi(gx / 2 + xo, 0, in->w - 1)];
            const float s = sad4x4(cur, old, gx * 4, gy * 4, v->x, v->y);
            if (n == 0 || s < best) {
                best = s;
                bestx = v->x;
                besty = v->y;
            }
        }
        out->v[i].x = (int16_t)(bestx * 2);
        out->v[i].y = (int16_t)(besty * 2);
    }
}

/* ---- the vector at the middle instant: not AMD's ---- */

/* How well the two frames agree over a block when each is moved half way along v: a at x + va, b at x - vb, va + vb = v. */
static inline float sad_mid(const struct fg_job *j, int x, int y, int vx, int vy)
{
    static const unsigned short EIGHT = 0x00ff;
    const int ax = vx >> 1, ay = vy >> 1, bx = vx - ax, by = vy - ay;
    const vxf zero = vx_zero();
    vxf sum = zero;
    for (int r = 0; r < FG_BLOCK; r++)
        sum = vx_add(sum, vx_blend(EIGHT, zero, vx_abs(vx_sub(vx_loadu(row_at(&j->a[0], x + ax, y + r + ay)), vx_loadu(row_at(&j->b[0], x - bx, y + r - by))), &ABS_LANE)));
    return vx_hsum(sum);
}

void fg_mid_vectors(const struct fg_job *j, int i0, int i1)
{
    const struct fg_field *f = &j->fil[0], *out = &j->mid;
    for (int i = i0; i < i1; i++) {
        const int bx = i % f->w, by = i / f->w, px = bx * FG_BLOCK, py = by * FG_BLOCK;
        /* No motion first, and it has an edge: a still thing under noise must stay still. */
        float best = sad_mid(j, px, py, 0, 0) - FG_MID_FAVOUR_ZERO;
        int bestx = 0, besty = 0;
        for (int yy = -1; yy < 2; yy++)
            for (int xx = -1; xx < 2; xx++) {
                const struct fg_vec *v = &f->v[clampi(by + yy, 0, f->h - 1) * f->w + clampi(bx + xx, 0, f->w - 1)];
                if ((v->x == bestx && v->y == besty) || (v->x == 0 && v->y == 0))
                    continue;
                const float s = sad_mid(j, px, py, v->x, v->y);
                if (s < best) {
                    best = s;
                    bestx = v->x;
                    besty = v->y;
                }
            }
        out->v[i].x = (int16_t)bestx;
        out->v[i].y = (int16_t)besty;
    }
}

/* ---- the picture ---- */

int fg_tiles(const struct fg_job *j) { return ((j->out.w + 15) / 16) * ((j->out.h + 15) / 16); }

void fg_blend(const struct fg_job *j, int t0, int t1)
{
    /* A block of the field is 16 x 16 at full resolution, and a pixel takes from the blocks whose centres are within 16 of it, by a triangle: per lane, from the left, the block itself and the right. In 32nds, so whole
     * numbers; the two axes multiply to 1024ths, a pixel is the sum of both frames, so 2048ths in all. */
    static const float W_LEFT[16] VX_ALIGNED = {15, 13, 11, 9, 7, 5, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    static const float W_SELF[16] VX_ALIGNED = {17, 19, 21, 23, 25, 27, 29, 31, 31, 29, 27, 25, 23, 21, 19, 17};
    static const float W_RIGHT[16] VX_ALIGNED = {0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15};
    static const float ONE_2048 VX_ALIGNED = 1.0f / 2048.0f, HALF VX_ALIGNED = 0.5f;
    const struct fg_field *f = &j->mid;
    const int tw = (j->out.w + 15) / 16;
    const vxf wh[3] = {vx_load(W_LEFT), vx_load(W_SELF), vx_load(W_RIGHT)};
    const vxf half = vx_bcast(&HALF), scale = vx_bcast(&ONE_2048);
    for (int t = t0; t < t1; t++) {
        const int tx = t % tw, ty = t / tw, x = tx * 16, y0 = ty * 16;
        const int y1 = y0 + 16 < j->out.h ? y0 + 16 : j->out.h;
        /* The nine blocks around. The field may have one block fewer than there are tiles at the edge: it is clamped. */
        struct fg_vec near[3][3];
        int same = 1;
        for (int yy = 0; yy < 3; yy++)
            for (int xx = 0; xx < 3; xx++) {
                near[yy][xx] = f->v[clampi(ty + yy - 1, 0, f->h - 1) * f->w + clampi(tx + xx - 1, 0, f->w - 1)];
                same &= near[yy][xx].x == near[0][0].x && near[yy][xx].y == near[0][0].y;
            }
        if (same) {
            /* Most of a picture: one vector, half of it each way at full resolution, which is a whole number of pixels since the field is at half resolution. */
            const int vx = near[0][0].x, vy = near[0][0].y;
            for (int y = y0; y < y1; y++) {
                const vxf sum = vx_add(vx_loadu(row_at(&j->a0, x + vx, y + vy)), vx_loadu(row_at(&j->b0, x - vx, y - vy)));
                vx_store_nr(plane_row(&j->out, y) + x, vx_round(vx_mul(sum, half), VX_RND_NEAREST));
            }
            continue;
        }
        for (int y = y0; y < y1; y++) {
            const int s = y - y0;
            /* Two rows of blocks bear on a row of pixels: the upper half of a tile takes from above, the lower from below. In 32nds. */
            const int other = s < 8 ? 0 : 2, w_other = s < 8 ? 15 - 2 * s : 2 * s - 15, w_self = 32 - w_other;
            const float wo VX_ALIGNED = (float)w_other, ws VX_ALIGNED = (float)w_self;
            const vxf vw[2] = {vx_bcast(&ws), vx_bcast(&wo)};
            const int rows[2] = {1, other};
            vxf acc = vx_zero();
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 3; b++) {
                    const struct fg_vec *v = &near[rows[a]][b];
                    const vxf sum = vx_add(vx_loadu(row_at(&j->a0, x + v->x, y + v->y)), vx_loadu(row_at(&j->b0, x - v->x, y - v->y)));
                    acc = vx_fma(sum, vx_mul(wh[b], vw[a]), acc);
                }
            vx_store_nr(plane_row(&j->out, y) + x, vx_round(vx_mul(acc, scale), VX_RND_NEAREST));
        }
    }
}
