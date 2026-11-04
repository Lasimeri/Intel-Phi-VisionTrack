/* gimref.h: camera D's virtual gimbal, against places in its own view that hold still (the operator's order, 2026-11-03).
 *
 * L, R and D stand on separate mounts: only D's own picture can say where D points. So D keeps a few windows of its own that have held still, found as L and R's are (autoref.h), never in the
 * operator's box, and each frame asks how far they moved since the last. A median for each window, so a part of one covered does not count, and at least two windows must agree, so one window
 * something crossed does not either, and most of what had been still anywhere in the picture must have moved too: only the camera moves all of it. Small steps add up with a leak: slow drift is absorbed, while a move comes through. Only what comes through moves anything.
 *
 * Frame to frame, not from the camera's still picture: that restarts wherever a cell moved, so its vectors say a move once and then its opposite.
 *
 * Motion and agreement only: nothing here knows what anything is. Scalars are long double, x87 on the card and on the host. */
#ifndef VT_GIMREF_H
#define VT_GIMREF_H

#include "vt_proto.h"

#define GIM_WINDOWS VT_GIM_WINDOWS
#define GIM_SIZE 64    /* a window's side, in the camera's pixels. At 128 only one fitted, on two frames in three: this picture is dark, and all of a window must have held still */
#define GIM_CELLS 64   /* cells a window can hold */

enum gim_state { GIM_NONE, GIM_UNSURE, GIM_STILL, GIM_MOVED };

struct autoref;

struct gimref {
    struct autoref *ar;
    int cols, rows, level, wc, wr, nwin, wait;
    int col[GIM_WINDOWS], row[GIM_WINDOWS], taken[GIM_WINDOWS];
    float du[GIM_WINDOWS][GIM_CELLS], dv[GIM_WINDOWS][GIM_CELLS];
    long double read_u, read_v, pend_u, pend_v, move_u, move_v;
    int state, agreeing, share;
};

/* 0, or -1 with no memory. */
int gimref_init(struct gimref *g, int cols, int rows, int level);
/* A frame's cells: what has held still, windows dropped and chosen. inside, if not NULL: x0, y0, x1, y1 of the operator's box in the camera's pixels. d_moving: D is moving in this frame. */
void gimref_choose(struct gimref *g, const struct vt_cell *cells, const float *inside, int d_moving);
/* Then, for every cell of every window: where its centre is, and what LK says it moved since the last frame, from gimref_point. */
void gimref_point(const struct gimref *g, int window, int k, float *x, float *y);
int gimref_points(const struct gimref *g);
void gimref_add(struct gimref *g, int window, float du, float dv, int good);
/* All added: the frame's reading. Returns its gim_state; move_u, move_v are what came through. */
int gimref_finish(struct gimref *g);

#endif
