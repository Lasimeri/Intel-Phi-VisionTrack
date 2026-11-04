/* autoref.c: see autoref.h. Summed-area tables: every window of a picture is three subtractions, whatever its size. */
#include "autoref.h"

#include <stdlib.h>

#define HOLD 12       /* frames unmoved before a cell may be part of a reference: a second, at the 12 a second these come */
#define CAP 240       /* of what a cell has stood through: twenty seconds is as good as it gets, so a place that once stood long does not outrank what stands now for ever */
#define MARGIN 1      /* cells kept from the picture's edge, where the patches of LK fall off */

struct autoref {
    int cols, rows;
    unsigned short *still, *still_d;
    unsigned char *good;
    /* (cols + 1) x (rows + 1) */
    unsigned *sat_score, *sat_young, *sat_good;
};

struct autoref *autoref_create(int cols, int rows)
{
    struct autoref *a = calloc(1, sizeof *a);
    const size_t cells = (size_t)cols * (size_t)rows, sat = ((size_t)cols + 1u) * ((size_t)rows + 1u);
    if (!a)
        return NULL;
    a->cols = cols;
    a->rows = rows;
    a->still = calloc(cells, sizeof *a->still);
    a->still_d = calloc(cells, sizeof *a->still_d);
    a->good = calloc(cells, 1);
    a->sat_score = calloc(sat, sizeof *a->sat_score);
    a->sat_young = calloc(sat, sizeof *a->sat_young);
    a->sat_good = calloc(sat, sizeof *a->sat_good);
    if (!a->still || !a->still_d || !a->good || !a->sat_score || !a->sat_young || !a->sat_good) {
        autoref_destroy(a);
        return NULL;
    }
    return a;
}

void autoref_destroy(struct autoref *a)
{
    if (!a)
        return;
    free(a->still);
    free(a->still_d);
    free(a->good);
    free(a->sat_score);
    free(a->sat_young);
    free(a->sat_good);
    free(a);
}

void autoref_feed(struct autoref *a, const struct vt_cell *cells, int d_moving)
{
    const size_t n = (size_t)a->cols * (size_t)a->rows;
    for (size_t i = 0; i < n; i++) {
        const unsigned state = cells[i].state;
        a->good[i] = (state & VT_CELL_LK) == 0;
        if (state & VT_CELL_MOVING) {
            a->still[i] = 0;
            a->still_d[i] = 0;
            continue;
        }
        a->still[i] = (unsigned short)(a->still[i] < CAP ? a->still[i] + 1u : CAP);
        a->still_d[i] = (unsigned short)(a->still_d[i] + (d_moving && a->still_d[i] < CAP));
    }
}

static unsigned rect(const unsigned *sat, int stride, int col, int row, int wc, int wr)
{
    const unsigned *top = sat + (size_t)row * (size_t)stride, *low = sat + (size_t)(row + wr) * (size_t)stride;
    return low[col + wc] - low[col] - top[col + wc] + top[col];
}

int autoref_best(struct autoref *a, int wc, int wr, int *col, int *row)
{
    const int cols = a->cols, rows = a->rows, stride = cols + 1;
    for (int r = 0; r < rows; r++) {
        unsigned line_score = 0, line_young = 0, line_good = 0;
        const size_t up = (size_t)r * (size_t)stride, here = up + (size_t)stride;
        for (int c = 0; c < cols; c++) {
            const size_t i = (size_t)r * (size_t)cols + (size_t)c;
            line_score += a->good[i] * (1u + a->still_d[i]);
            line_young += a->still[i] < HOLD;
            line_good += a->good[i];
            a->sat_score[here + (size_t)c + 1u] = a->sat_score[up + (size_t)c + 1u] + line_score;
            a->sat_young[here + (size_t)c + 1u] = a->sat_young[up + (size_t)c + 1u] + line_young;
            a->sat_good[here + (size_t)c + 1u] = a->sat_good[up + (size_t)c + 1u] + line_good;
        }
    }
    /* A third of its cells with corners: a tracker wants a handful of features, and those lie unevenly. */
    const unsigned enough = (unsigned)(wc * wr) / 3u;
    unsigned best = 0;
    int found = 0;
    for (int r = MARGIN; r + wr <= rows - MARGIN; r++)
        for (int c = MARGIN; c + wc <= cols - MARGIN; c++) {
            if (rect(a->sat_young, stride, c, r, wc, wr) || rect(a->sat_good, stride, c, r, wc, wr) < enough)
                continue;
            const unsigned score = rect(a->sat_score, stride, c, r, wc, wr);
            if (score > best) {
                best = score;
                *col = c;
                *row = r;
                found = 1;
            }
        }
    return found;
}

int autoref_moved(const struct autoref *a, int col, int row, int wc, int wr)
{
    int moved = 0;
    for (int r = row; r < row + wr && r < a->rows; r++)
        for (int c = col; c < col + wc && c < a->cols; c++)
            moved += a->still[(size_t)r * (size_t)a->cols + (size_t)c] == 0;
    return moved;
}
