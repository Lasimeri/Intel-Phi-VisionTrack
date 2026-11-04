/* gimref.c: see gimref.h. */
#include "gimref.h"

#include <string.h>

#include "autoref.h"

typedef long double real;

#define R(x) ((real)(x))
#define DEAD 0.75L     /* camera pixels that must add up before anything is said to have moved */
#define LEAK 0.875L    /* of what has added up, kept for the next frame while nothing has */
#define WINDOW_AGREE 1.0L /* camera pixels from the middle reading: this window agrees */
#define NEXT_AFTER 2   /* frames between choosing windows */

int gimref_init(struct gimref *g, int cols, int rows, int level)
{
    memset(g, 0, sizeof *g);
    g->ar = autoref_create(cols, rows);
    if (!g->ar)
        return -1;
    g->cols = cols;
    g->rows = rows;
    g->level = level;
    g->wc = GIM_SIZE / (VT_CELL_W << level);
    g->wr = GIM_SIZE / (VT_CELL_H << level);
    if (g->wc * g->wr > GIM_CELLS) {
        g->wr = GIM_CELLS / g->wc;
    }
    g->state = GIM_NONE;
    return 0;
}

static int overlaps_box(const struct gimref *g, int col, int row, const float *inside)
{
    const int cw = VT_CELL_W << g->level, ch = VT_CELL_H << g->level;
    return R(col * cw) <= R(inside[2]) && R((col + g->wc) * cw) >= R(inside[0]) && R(row * ch) <= R(inside[3]) && R((row + g->wr) * ch) >= R(inside[1]);
}

void gimref_choose(struct gimref *g, const struct vt_cell *cells, const float *inside, int d_moving)
{
    autoref_feed(g->ar, cells, d_moving);
    g->wait -= g->wait > 0;
    /* A window something moved in is no reference, unless they all are: then it is the camera, and that is what they are for. */
    int moved = 0;
    int was[GIM_WINDOWS];
    for (int i = 0; i < g->nwin; i++) {
        was[i] = 4 * autoref_moved(g->ar, g->col[i], g->row[i], g->wc, g->wr) >= g->wc * g->wr;
        moved += was[i];
    }
    int w = 0;
    for (int i = 0; i < g->nwin; i++) {
        const int drop = (was[i] && moved < g->nwin) || (inside && overlaps_box(g, g->col[i], g->row[i], inside));
        if (drop)
            continue;
        g->col[w] = g->col[i];
        g->row[w] = g->row[i];
        w++;
    }
    g->nwin = w;
    if (g->nwin == GIM_WINDOWS || g->wait)
        return;
    const int cw = VT_CELL_W << g->level, ch = VT_CELL_H << g->level;
    autoref_clear_blocks(g->ar);
    for (int i = 0; i < g->nwin; i++)
        autoref_block(g->ar, g->col[i], g->row[i], g->wc, g->wr);
    if (inside)
        autoref_block(g->ar, (int)(inside[0] / (float)cw), (int)(inside[1] / (float)ch), (int)((inside[2] - inside[0]) / (float)cw) + 2, (int)((inside[3] - inside[1]) / (float)ch) + 2);
    int col, row;
    if (autoref_best(g->ar, g->wc, g->wr, &col, &row)) {
        g->col[g->nwin] = col;
        g->row[g->nwin] = row;
        g->nwin++;
        g->wait = NEXT_AFTER;
    }
}

int gimref_points(const struct gimref *g) { return g->wc * g->wr; }

void gimref_point(const struct gimref *g, int window, int k, float *x, float *y)
{
    const int cw = VT_CELL_W, ch = VT_CELL_H;
    *x = (float)((g->col[window] + k % g->wc) * cw + cw / 2);
    *y = (float)((g->row[window] + k / g->wc) * ch + ch / 2);
}

void gimref_add(struct gimref *g, int window, float du, float dv, int good)
{
    if (!good || g->taken[window] >= GIM_CELLS)
        return;
    const int n = g->taken[window]++;
    g->du[window][n] = du;
    g->dv[window][n] = dv;
}

static real middle(float *v, int n)
{
    for (int i = 1; i < n; i++) {
        const float x = v[i];
        int k = i;
        for (; k > 0 && v[k - 1] > x; k--)
            v[k] = v[k - 1];
        v[k] = x;
    }
    return n & 1 ? R(v[n / 2]) : 0.5L * (R(v[n / 2 - 1]) + R(v[n / 2]));
}

static real size_of(real v) { return v < 0.0L ? -v : v; }

int gimref_finish(struct gimref *g)
{
    const int enough = g->wc * g->wr / 3;
    const real scale = R(1 << g->level);
    float mu[GIM_WINDOWS], mv[GIM_WINDOWS];
    int n = 0;
    for (int i = 0; i < g->nwin; i++) {
        if (g->taken[i] >= enough && g->taken[i] >= 3) {
            mu[n] = (float)(middle(g->du[i], g->taken[i]) * scale);
            mv[n] = (float)(middle(g->dv[i], g->taken[i]) * scale);
            n++;
        }
        g->taken[i] = 0;
    }
    g->move_u = g->move_v = 0.0L;
    g->agreeing = 0;
    if (!g->nwin)
        return g->state = GIM_NONE;
    if (n < 2) {
        g->read_u = g->read_v = 0.0L;
        return g->state = GIM_UNSURE;
    }
    float su[GIM_WINDOWS], sv[GIM_WINDOWS];
    memcpy(su, mu, sizeof su);
    memcpy(sv, mv, sizeof sv);
    const real mid_u = middle(su, n), mid_v = middle(sv, n);
    for (int i = 0; i < n; i++)
        g->agreeing += size_of(R(mu[i]) - mid_u) + size_of(R(mv[i]) - mid_v) <= WINDOW_AGREE;
    g->read_u = mid_u;
    g->read_v = mid_v;
    if (g->agreeing < 2)
        return g->state = GIM_UNSURE;
    g->pend_u += mid_u;
    g->pend_v += mid_v;
    if (size_of(g->pend_u) + size_of(g->pend_v) > DEAD) {
        g->move_u = g->pend_u;
        g->move_v = g->pend_v;
        g->pend_u = g->pend_v = 0.0L;
        return g->state = GIM_MOVED;
    }
    g->pend_u *= LEAK;
    g->pend_v *= LEAK;
    return g->state = GIM_STILL;
}
