/* gimref_check.c: camera D's reference against readings made here, whose answers are known. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gimref.h"

#define COLS 40
#define ROWS 45
#define LEVEL 1

static int fails;
static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += !ok;
}

/* A frame: textured, nothing moved; each window's cells say what they are told, at the level's pixels. */
static int step(struct gimref *g, const struct vt_cell *cells, const float *box, const float *du, const float *dv)
{
    gimref_choose(g, cells, box, 1);
    const int points = gimref_points(g);
    for (int i = 0; i < g->nwin; i++)
        for (int k = 0; k < points; k++)
            gimref_add(g, i, du[i], dv[i], 1);
    return gimref_finish(g);
}

int main(void)
{
    struct vt_cell *cells = calloc(COLS * ROWS, sizeof *cells);
    struct gimref g;
    if (gimref_init(&g, COLS, ROWS, LEVEL))
        return 2;
    const float zero[3] = {0.0f, 0.0f, 0.0f};
    const float box[4] = {320.0f, 200.0f, 640.0f, 500.0f};
    check(step(&g, cells, box, zero, zero) == GIM_NONE, "nothing held still yet: no windows, no reading");
    for (int i = 0; i < 20; i++)
        step(&g, cells, box, zero, zero);
    check(g.nwin == VT_GIM_WINDOWS, "a moment later, all its windows");
    int clear = 1;
    const int cw = VT_CELL_W << LEVEL, ch = VT_CELL_H << LEVEL;
    for (int i = 0; i < g.nwin; i++)
        clear &= (g.col[i] + g.wc) * cw < 320 || g.col[i] * cw > 640 || (g.row[i] + g.wr) * ch < 200 || g.row[i] * ch > 500;
    check(clear, "none of them in the operator's box");
    int exact = 1;
    for (int i = 0; i < 40; i++)
        exact &= step(&g, cells, box, zero, zero) == GIM_STILL && g.move_u == 0.0L && g.move_v == 0.0L && g.pend_u == 0.0L && g.pend_v == 0.0L;
    check(exact, "nothing moves: exactly nothing, down to the last bit");
    /* Noise of a tenth of a level's pixel, all ways. */
    int quiet = 1;
    for (int i = 0; i < 200; i++) {
        const float s = i & 1 ? 0.1f : -0.1f;
        const float a[3] = {s, -s, s}, b[3] = {-s, s, s};
        quiet &= step(&g, cells, box, a, b) != GIM_MOVED;
    }
    check(quiet, "noise does not add up to a move");
    /* Slow drift: absorbed. */
    int slow = 1;
    const float creep[3] = {0.02f, 0.02f, 0.02f};
    for (int i = 0; i < 400; i++)
        slow &= step(&g, cells, box, creep, creep) != GIM_MOVED;
    check(slow, "slow drift is absorbed, not followed");
    for (int i = 0; i < 40; i++)
        step(&g, cells, box, zero, zero);
    /* A step: 3 px of the level, 1.5 down: 6 and 3 of the camera. */
    const float su[3] = {3.0f, 3.0f, 3.0f}, sv[3] = {1.5f, 1.5f, 1.5f};
    check(step(&g, cells, box, su, sv) == GIM_MOVED && fabsl(g.move_u - 6.0L) < 0.5L && fabsl(g.move_v - 3.0L) < 0.5L, "a move comes through, in the camera's pixels");
    check(step(&g, cells, box, zero, zero) == GIM_STILL && g.move_u == 0.0L, "once");
    /* One window has something crossing it. */
    const float one[3] = {9.0f, 0.0f, 0.0f};
    check(step(&g, cells, box, one, zero) == GIM_STILL && g.move_u == 0.0L, "something crossing one window moves nothing");
    /* All three say something different. */
    const float each_u[3] = {6.0f, 0.0f, -6.0f};
    check(step(&g, cells, box, each_u, zero) == GIM_UNSURE && g.move_u == 0.0L, "windows that disagree: unsure, and nothing moves");
    /* The operator's box comes over a window. */
    const float on[4] = {(float)(g.col[0] * cw), (float)(g.row[0] * ch), (float)((g.col[0] + 1) * cw), (float)((g.row[0] + 1) * ch)};
    const int before = g.nwin, col0 = g.col[0], row0 = g.row[0];
    gimref_choose(&g, cells, on, 1);
    int gone = 1;
    for (int i = 0; i < g.nwin; i++)
        gone &= !(g.col[i] == col0 && g.row[i] == row0);
    check(before == VT_GIM_WINDOWS && gone, "the operator's box coming over a window takes it out");
    /* Something moving in one window: replaced; in all of them: kept. */
    printf("%s\n", fails ? "gimref: FAILED" : "gimref: passed");
    free(cells);
    return fails != 0;
}
