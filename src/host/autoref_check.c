/* autoref_check.c: the scan for static regions against pictures made here, whose answers are known. */
#include <stdio.h>
#include <stdlib.h>

#include "autoref.h"

#define COLS 40
#define ROWS 30
#define WC 8
#define WR 6

static int fails;
static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += !ok;
}

/* Everything flat but the two textured patches, A on the left and B on the right; what moves, moves. */
static void frame(struct vt_cell *cells, int a_moves, int b_moves, int blob_col)
{
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) {
            struct vt_cell *e = &cells[r * COLS + c];
            const int in_a = c >= 4 && c < 14 && r >= 10 && r < 20, in_b = c >= 26 && c < 36 && r >= 10 && r < 20;
            const int in_blob = c >= blob_col && c < blob_col + 6 && r >= 2 && r < 8;
            e->state = (in_a || in_b || in_blob) ? 0u : 4u; /* any LK status: no good vector */
            if ((in_a && a_moves) || (in_b && b_moves) || in_blob)
                e->state |= VT_CELL_CHANGED | VT_CELL_MOVING;
        }
}

int main(void)
{
    struct vt_cell *cells = calloc(COLS * ROWS, sizeof *cells);
    struct autoref *a = autoref_create(COLS, ROWS);
    int c = -1, r = -1;
    frame(cells, 0, 0, 10);
    autoref_feed(a, cells, 0);
    check(!autoref_best(a, WC, WR, &c, &r), "one frame proves nothing");
    for (int i = 0; i < 11; i++)
        autoref_feed(a, cells, 0);
    check(autoref_best(a, WC, WR, &c, &r), "a second still: there is a region");
    check(((c >= 4 && c + WC <= 14) || (c >= 26 && c + WC <= 36)) && r >= 10 && r + WR <= 20, "and it lies on a textured patch, not on flat wall");
    /* D moves, and A moves with it; B does not. */
    for (int i = 0; i < 30; i++) {
        frame(cells, i < 3, 0, 10);
        autoref_feed(a, cells, 1);
    }
    autoref_best(a, WC, WR, &c, &r);
    check(c >= 26 && c + WC <= 36, "of two still places, the one that stood through more of D's motion");
    check(autoref_moved(a, c, r, WC, WR) == 0, "nothing moved in it");
    /* Now what is in it moves. */
    const int was_c = c, was_r = r;
    frame(cells, 0, 1, 10);
    autoref_feed(a, cells, 1);
    check(autoref_moved(a, was_c, was_r, WC, WR) == WC * WR, "motion in the region is seen");
    autoref_best(a, WC, WR, &c, &r);
    check(c >= 4 && c + WC <= 14, "and the next choice is elsewhere");
    /* Both moving, and a moving thing has corners: nothing qualifies. */
    frame(cells, 1, 1, 10);
    autoref_feed(a, cells, 1);
    check(!autoref_best(a, WC, WR, &c, &r), "nothing still with corners: no region, rather than a bad one");
    autoref_destroy(a);
    free(cells);
    printf("%s\n", fails ? "autoref: FAILED" : "autoref: passed");
    return fails != 0;
}
