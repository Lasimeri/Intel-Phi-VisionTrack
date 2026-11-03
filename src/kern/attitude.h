/* attitude.h: how far a camera has turned since the picture its geometry was solved in, from the whole picture, with nothing held: no region, no lights, no still room, no training.
 *
 * Why against its own past. L and R are 56 degrees apart, and matching features between them fails (7 to 15 inliers measured). A camera compared with itself has no such gap, so a knock is measured where
 * appearance holds, and the pair is then corrected in closed form: R' = Ob R Oa', T' = Ob T. The same shape as the New 3DS: stored calibration, and a cheap correction from a live measurement.
 *
 * The method, written for this project, one source for card and host, which must agree bit for bit:
 *   normalize   each pixel's distance from its 9 x 9 mean in units of that window's spread, so a light switched on changes little. depth_normalize.
 *   search      the current picture slid over the reference at a coarse level, every shift: a normalized correlation, which is what finds a knock of many degrees, beyond any gradient's reach.
 *   refine      Gauss-Newton on the three rotation parameters, coarse to fine, inverse compositional: the gradients are the reference's, so they are computed once.
 *   kernel      each residual weighted by Tukey's biweight, (1 - (e / c)^2)^2, zero beyond c: whatever moved has no say at all. Multiplications and a max, so MVEX needs no division.
 *
 * Rotation only. A knock that also moves the camera's centre leaves parallax this cannot remove: corr and used then say how much the picture still disagrees.
 * Scalars are long double, x87 on both machines. No libm: sin and cos differ between the two machines' libraries, so rotations are quaternions, first order and renormalized. Sums are per row and added in row
 * order, so the answer does not depend on the number of threads. */
#ifndef VT_ATTITUDE_H
#define VT_ATTITUDE_H

#include <stddef.h>
#include <stdint.h>

#include "plane.h"

#define ATT_LEVELS 3 /* the input plane, then twice halved */
#define ATT_MIN_GAIN 0.10 /* correlation a turn must add over no turn at all, or it is refused. Measured with someone close in view: false turns gained at most 0.053; a real 0.2 degrees at least 0.19 */
#define ATT_MIN_TURN_SQ 7.6e-7L /* a turn under 0.1 degree is no turn: the square of sin(angle / 2) */

struct att_cfg {
    int w, h;         /* the input plane */
    double f, cx, cy; /* at the input plane's level */
    float eps;        /* normalization's floor on the spread, Q6 */
    int threads;
};

struct att_fix {
    double O[9];             /* a bearing b in the reference's pose is seen at O b now */
    double q[4];             /* the same as a quaternion: w, x, y, z */
    double corr;             /* normalized correlation of reference and warped picture, over all pixels in view */
    double corr_identity;    /* the same with no rotation: what the fix gained */
    double used;             /* the share of pixels in view the kernel kept */
    double view;             /* the share of the reference still in view */
    int peak_x, peak_y;      /* the search's best shift, in pixels of its level; 0, 0 when it did not run */
    double peak_z;           /* how far above chance, in standard deviations: the sum over the root of the count */
    int steps;               /* Gauss-Newton steps taken */
    int held;                /* 1: the fit did not beat no turn at all, so O is the identity */
};

/* One round of a team, or a plain call on one thread. */
typedef void (*att_fn)(void *arg, int tid, int n);
typedef void (*att_round)(void *ctx, att_fn fn, void *arg);

struct att;

size_t att_bytes(const struct att_cfg *c);
/* All memory from alloc, 64 B aligned. NULL if it refuses. */
struct att *att_create(const struct att_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx);
/* The lens, at the input plane's level: before att_set_reference, which computes from it. */
void att_set_lens(struct att *a, double f, double cx, double cy);
/* The picture the geometry belongs to. Its padding must be current. */
void att_set_reference(struct att *a, const struct vt_plane *ref, att_round round, void *round_ctx);
/* The attitude of this picture against the reference. search 1: from nothing, finds a knock. search 0: from the last answer, far cheaper. Its padding must be current. */
void att_measure(struct att *a, const struct vt_plane *cur, int search, att_round round, void *round_ctx, struct att_fix *fix);
/* o = a b: first b, then a. Renormalized. */
void att_q_mul(const double *a, const double *b, double *o);
/* A fix's every number, raw bits, into a running hash: card and host must agree. */
uint64_t att_fix_hash(const struct att_fix *f, uint64_t h);

#endif
