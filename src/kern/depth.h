/* depth.h: a depth map from one reference view and up to two others, by plane sweep: for each of a set of depths, each source view is read where a surface at that depth would put it, and the depth
 * where the views agree best wins. The same source compiles to MVEX on the card and to the lane model on the host, and the two must agree bit for bit.
 *
 * Exposures differ between cameras by a factor of several, so images are compared after local normalization: each pixel's distance from its 9 x 9 mean, in units of that window's spread.
 *
 * A pixel's depth is given only where it can be known: enough texture in the reference, and a best match clearly better than the best match at any other depth. Everything else is 0: no depth.
 * Scalars are long double, x87 on both machines, as everywhere in this project.
 *
 * One frame: depth_normalize per view (by rows), then depth_sweep, depth_aggregate, depth_decide, each a team round with a barrier between. */
#ifndef VT_DEPTH_H
#define VT_DEPTH_H

#include <stddef.h>

#include "plane.h"
#include "vt_proto.h"

#define DEPTH_MAX_VIEWS 2
#define DEPTH_NORM_RADIUS 4 /* inside a plane's vertical padding: VT_PAD_Y */
#define DEPTH_WEIGHT_R 1.0f /* the views' say: L is the noisier camera */
#define DEPTH_WEIGHT_L 0.7f

struct depth_view {
    const struct vt_plane *norm; /* the normalized plane of a source view */
    double f, cx, cy;            /* at that plane's level */
    double R[9], T[3];           /* X_view = R X_ref + T, metres */
    float weight;
};

struct depth_cfg {
    int w, h;            /* of the reference plane: w a multiple of 16 */
    double f, cx, cy;    /* of the reference plane */
    int planes;          /* depths, uniform in 1 / z */
    double z_near, z_far; /* metres */
    int agg_radius;      /* cost aggregation window: 2 r + 1 square */
    float trunc;         /* a per-view cost saturates here: an occluded view cannot outvote a visible one by much */
    float tex_min;       /* reference spread below this, Q6: too little texture, no depth */
    float conf_min;      /* 1 - best / second best below this: no depth */
    int threads;
};

struct depth;

size_t depth_bytes(const struct depth_cfg *c);
/* All memory comes from alloc, 64 B aligned. NULL if it refuses. */
struct depth *depth_create(const struct depth_cfg *c, void *(*alloc)(void *ctx, size_t bytes), void *ctx);

/* dst = (src - mean) / (spread + eps), rows y0 .. y1. Source padding must be current. spread: optional, the window's standard deviation. scratch: depth_norm_scratch_bytes(w), one per thread. */
void depth_normalize(const struct vt_plane *dst, const struct vt_plane *spread, const struct vt_plane *src, float eps, int y0, int y1, void *scratch);
size_t depth_norm_scratch_bytes(int w);
/* The same as one thread's share of a team round: rows split evenly, scratch one depth_norm_scratch_bytes after another. */
void depth_normalize_part(const struct vt_plane *dst, const struct vt_plane *spread, const struct vt_plane *src, float eps, int tid, int n, char *scratch, size_t scratch_each);

/* Camera b as seen from the reference d, each known against the world: X_b = Rb X_w + Tb, X_d = Rd X_w + Td. */
void depth_relate(const double *Rb, const double *Tb, const double *Rd, const double *Td, double *R, double *T);
/* The two source views of a three-camera calibration, D the reference, R first, L second, at a pyramid level `scale` = 2^level. So the live map and the proven offline path cannot part ways. */
void depth_views_from_calib(const struct vt_calib *cal, int scale, const struct vt_plane *norm_r, const struct vt_plane *norm_l, struct depth_view *views);

/* The reference's intrinsics, changed: a camera whose focal length is being searched for. */
void depth_set_ref(struct depth *d, double f, double cx, double cy);
/* What the next sweep reads. */
void depth_set_views(struct depth *d, const struct vt_plane *ref_norm, const struct vt_plane *ref_spread, const struct depth_view *views, int nviews);
void depth_sweep(struct depth *d, int tid, int n);
void depth_aggregate(struct depth *d, int tid, int n);
void depth_decide(struct depth *d, int tid, int n);

/* The results: w x h F32, row stride depth_stride in lanes. depth_z: metres, 0 where unknown. depth_conf: 0 .. 1. */
const float *depth_z(const struct depth *d);
const float *depth_conf(const struct depth *d);
int depth_stride(const struct depth *d);
/* Over the last depth_decide on n threads: pixels with enough texture; of those, with confidence at least 0.2; pixels with a depth. */
void depth_counts(const struct depth *d, int n, uint32_t *textured, uint32_t *confident, uint32_t *known);

#endif
