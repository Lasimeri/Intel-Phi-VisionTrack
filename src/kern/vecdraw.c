/* vecdraw.c: see vecdraw.h. Scalars are long double, x87 on the card and on the host, so the two agree bit for bit. */
#include "vecdraw.h"

#include <string.h>

typedef long double real;

#define GAIN 4          /* a vector drawn this many times its length: a quarter of a level's pixel is one pixel of the camera, as the window has always drawn it */
#define PEN 2           /* a line's width here: about one and a half pixels once it is brought down by 3/4 */
#define DOT 4           /* a cell's mark, likewise about three */

struct line {
    int x0, y0, x1, y1;
};

struct vecdraw {
    struct vt_plane cov;
    struct line *lines;
    uint8_t *rows;
    int cap, nlines;
};

static size_t round_up(size_t n) { return (n + 63u) & ~(size_t)63u; }

size_t vd_bytes(int w, int h, int cells)
{
    return round_up(sizeof(struct vecdraw)) + round_up(plane_bytes(w, h)) + round_up((size_t)cells * sizeof(struct line)) + round_up((size_t)h);
}

struct vecdraw *vd_init(void *mem, int w, int h, int cells)
{
    char *m = mem;
    struct vecdraw *d = (struct vecdraw *)(void *)m;
    m += round_up(sizeof *d);
    memset(m, 0, plane_bytes(w, h));
    d->cov = plane_at(m, w, h);
    m += round_up(plane_bytes(w, h));
    d->lines = (struct line *)(void *)m;
    m += round_up((size_t)cells * sizeof(struct line));
    d->rows = (uint8_t *)m;
    memset(d->rows, 0, (size_t)h);
    d->cap = cells;
    d->nlines = 0;
    return d;
}

/* A square of `size` pixels from (x, y), what lies inside the plane. */
static inline void pen(const struct vt_plane *p, int x, int y, int size, float value)
{
    for (int r = 0; r < size; r++) {
        if ((unsigned)(y + r) >= (unsigned)p->h)
            continue;
        float *row = plane_row(p, y + r);
        for (int c = 0; c < size; c++)
            if ((unsigned)(x + c) < (unsigned)p->w)
                row[x + c] = value;
    }
}

static void stroke(const struct vt_plane *p, const struct line *l, float value)
{
    const int dx = l->x1 > l->x0 ? l->x1 - l->x0 : l->x0 - l->x1, dy = l->y1 > l->y0 ? l->y0 - l->y1 : l->y1 - l->y0;
    const int sx = l->x0 < l->x1 ? 1 : -1, sy = l->y0 < l->y1 ? 1 : -1;
    int err = dx + dy, x = l->x0, y = l->y0;
    for (;;) {
        pen(p, x, y, PEN, value);
        if (x == l->x1 && y == l->y1)
            break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
    pen(p, l->x0 - DOT / 2, l->y0 - DOT / 2, DOT, value);
}

static inline int to_int(real v) { return (int)v; }

int vd_draw(struct vecdraw *d, const struct vt_cell *cells, int cols, int rows, int level, long double view, const float *inside)
{
    const struct vt_plane *p = &d->cov;
    for (int i = 0; i < d->nlines; i++)
        stroke(p, &d->lines[i], 0.0f);
    memset(d->rows, 0, (size_t)p->h);
    const real scale = (real)(1 << level) * (real)view, gain = scale * (real)GAIN;
    /* What LK may answer is bounded by the picture; a vector past that, or not a number, is drawn as a dot rather than walked for ever. */
    const real limit = (real)(p->w + p->h);
    int n = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols && n < d->cap; c++) {
            const struct vt_cell *e = &cells[r * cols + c];
            if (!(e->state & VT_CELL_CHANGED))
                continue;
            /* Where the cell is in the camera's own pixels: outside the selected box it is drawn no more than it is tracked. */
            const int cam_x = (c * VT_CELL_W + VT_CELL_W / 2) << level, cam_y = (r * VT_CELL_H + VT_CELL_H / 2) << level;
            if (inside && ((real)cam_x < (real)inside[0] || (real)cam_x > (real)inside[2] || (real)cam_y < (real)inside[1] || (real)cam_y > (real)inside[3]))
                continue;
            const int good = (e->state & VT_CELL_LK) == 0;
            real du = good ? (real)e->u * gain : (real)0, dv = good ? (real)e->v * gain : (real)0;
            if (!(du <= limit && du >= -limit && dv <= limit && dv >= -limit))
                du = dv = (real)0;
            const int at_x = c * VT_CELL_W + VT_CELL_W / 2, at_y = r * VT_CELL_H + VT_CELL_H / 2;
            const int x0 = to_int((real)at_x * scale), y0 = to_int((real)at_y * scale);
            struct line *l = &d->lines[n++];
            l->x0 = x0;
            l->y0 = y0;
            l->x1 = x0 + to_int(du);
            l->y1 = y0 + to_int(dv);
            stroke(p, l, 1.0f);
            const int top = (l->y1 < l->y0 ? l->y1 : l->y0) - DOT / 2, bottom = (l->y1 > l->y0 ? l->y1 : l->y0) + DOT;
            for (int y = top < 0 ? 0 : top; y <= bottom && y < p->h; y++)
                d->rows[y] = 1;
        }
    d->nlines = n;
    return n;
}

const struct vt_plane *vd_plane(const struct vecdraw *d) { return &d->cov; }
const uint8_t *vd_rows(const struct vecdraw *d) { return d->rows; }
