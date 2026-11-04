/* still.h: one camera's picture, averaged over the frames it has held still, and the vectors that decide what has.
 *
 * A dim room's frame is mostly noise, and noise biases a truncated absolute difference: averaging before the sweep is what lifted the map from 12.72 % to 17.82 % of textured pixels (docs/measurements.md). A
 * running mean over everything would smear whatever moves, so the card's own vectors gate it: per cell, LK from the camera's mean to this frame. The reference being the mean and not the last frame:
 *   - a slow mover adds up against it, instead of hiding inside each frame's noise;
 *   - what does not move gives the camera's own drift against its past, which does not accumulate as frame-to-frame vectors would;
 *   - where a cell moved its mean restarts from this frame, so the next vector there is frame to frame.
 *
 * The same source is MVEX on the card and the lane model on the host, and the two agree bit for bit (vt_m3 against m3_ref). Scalars are long double, x87 on both.
 *
 * One frame: still_vectors on the team (unless still_first), still_decide on one thread, still_update on the team, still_finish on one thread. A fence between them on the card. */
#ifndef VT_STILL_H
#define VT_STILL_H

#include <stddef.h>
#include <stdint.h>

#include "plane.h"
#include "vt_proto.h"

struct still_cfg {
    int w, h;          /* of the level read */
    int avg;           /* frames at most in a cell's mean: weight 1 / min(count, avg) */
    float motion_px;   /* a good vector longer than this: moved */
    float motion_grey; /* mean |frame - mean| over a cell above this, 8-bit grey: moved */
    int threads;
};

struct still;
struct lk_out;

size_t still_bytes(const struct still_cfg *c);
struct still *still_create(const struct still_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx);

/* Nothing to compare with yet: the first frame only starts the mean. */
int still_first(const struct still *s);
void still_vectors(struct still *s, const struct vt_plane *cur, int tid, int n);
void still_decide(struct still *s);
void still_update(struct still *s, const struct vt_plane *cur, int tid, int n);
void still_finish(struct still *s);
/* How far a point moved from plane a to plane b of this camera, one thread (tid): for what asks about places, not cells. */
void still_point(struct still *s, const struct vt_plane *a, const struct vt_plane *b, float x, float y, struct lk_out *o, int tid);

const struct vt_plane *still_mean(const struct still *s);
const struct vt_cell *still_cells(const struct still *s);
/* Bytes of the cells, a whole number of 64 B lines. */
size_t still_cell_bytes(const struct still *s);
/* Fills everything but cell_at and the cycle counts. */
void still_motion(const struct still *s, struct vt_motion *m);
/* Folds the cells and the mean into a hash: what card and host must agree on. */
uint64_t still_hash(const struct still *s, uint64_t h);

#endif
