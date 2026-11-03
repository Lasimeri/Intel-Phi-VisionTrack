/* lk.h: sparse pyramidal Lucas-Kanade (Bouguet's formulation, the one OpenCV's calcOpticalFlowPyrLK implements) for one point, on
 * Q6 planes (plane.h). Solves I_x u + I_y v + I_t = 0 over a patch, iterated per level, coarse to fine.
 *
 * Two instantiations of one driver (lk_driver.inc): lk_track_ref in plain C, lk_track_vx against vx.h. They differ only in the three
 * patch operations, never in control flow. The driver's own arithmetic is long double in both: x87 on the card and on the host alike, so the
 * host's lane model and the card's MVEX must agree BIT FOR BIT on a track, and scripts/check.sh holds them to it.
 *
 * Status semantics: no measurement is invented. A point whose patch leaves the image at level 0 gets LK_OUT, and its position is only
 * the last estimate. */
#ifndef VT_LK_H
#define VT_LK_H

#include <stddef.h>
#include <stdint.h>

#include "plane.h"

#define LK_MAX_W 48
#define LK_MAX_H 31
#define LK_MAX_V (LK_MAX_W / 16)

enum { LK_OK = 0, LK_OUT = 1, LK_LOWEIG = 2, LK_RESIDUAL = 4, LK_FLOWBACK = 8 };

struct lk_cfg {
    int w, h;          /* patch, pixels: 5 .. LK_MAX_W, 5 .. LK_MAX_H, odd is natural */
    int max_iter;
    float eps;         /* stop when a step is below this, pixels of the level */
    float min_eig;     /* smaller eigenvalue of the structure tensor per patch pixel, gradients in 8-bit grey levels per pixel */
    float max_residual;/* mean |T - J| in 8-bit grey levels */
    int prefetch;      /* 1: every window of a level is prefetched before its first touch. Changes timing, never a result. */
};

struct lk_out {
    float x, y;        /* the estimate at level 0 */
    float min_eig;     /* at level 0 */
    float residual;    /* mean |T - J|, 8-bit grey levels */
    float cov_xx, cov_xy, cov_yy; /* s^2 G^-1, pixels^2: treats the patch's residuals as independent, so a lower bound */
    uint16_t iters;    /* summed over the levels */
    uint16_t status;
};

/* Per-thread scratch: 64 B aligned, lk_tile_bytes() of it. */
size_t lk_tile_bytes(void);

/* Tracks (x, y) from pyramid I to pyramid J, starting at level top with guess (gx, gy) in level-0 pixels. */
void lk_track_vx(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float gx, float gy, struct lk_out *o, void *tile);
void lk_track_ref(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float gx, float gy, struct lk_out *o, void *tile);

/* The forward-backward check: tracks the estimate back from J to I at the same pyramid, starting from minus the found flow. A point whose
 * round trip misses its start by more than max_fb pixels gets LK_FLOWBACK. Catches a confident wrong convergence: the residual alone does
 * not, because the wrong place can look alike (measured, docs/measurements.md). */
void lk_flowback_vx(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float max_fb, struct lk_out *o, void *tile);
void lk_flowback_ref(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float max_fb, struct lk_out *o, void *tile);

#endif
