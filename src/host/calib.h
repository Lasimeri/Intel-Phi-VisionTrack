/* calib.h: how two cameras relate, found from point correspondences alone: no chart, no synchronized shutters, no parallel mounting. Both focal lengths are unknowns; the principal points are taken at
 * the image centres and distortion as none. The result is up to scale, as any two views are; one known length sets it. The method follows what the Nintendo 3DS stores per unit (rotation, translation,
 * angle of view, distance between cameras: 3dbrew, Hardware calibration), reached from the images themselves since nothing here was mounted parallel. docs/architecture.md. */
#ifndef VT_CALIB_H
#define VT_CALIB_H

#include "vt_proto.h"

struct calib_pt {
    double xa, ya, xb, yb;
};

struct calib_problem {
    const struct calib_pt *pts;
    int n;
    int wa, ha, wb, hb;
    double fa_prior, fb_prior;   /* 0: unknown, searched for */
    double prior_weight;         /* pull on log focal, per point: 0 for none */
};

struct calib_solution {
    double fa, fb, R[9], t[3];   /* X_b = R X_a + t, |t| = 1 */
    double rms_px;               /* reprojection over every point in both images */
    double median_px;
    int inliers;
};

/* 0, or -1 with too few points or no geometry with the points in front of both cameras. */
int calib_solve(const struct calib_problem *p, struct calib_solution *s);
/* The distance between points i and j in the solution's unit length. */
double calib_length(const struct calib_problem *p, const struct calib_solution *s, int i, int j);
/* Point i through the solution: its 3D place in camera A's frame, in the solution's unit length, and how far it lands from what was read, pixels, in each image. */
void calib_point(const struct calib_problem *p, const struct calib_solution *s, int i, double *P, double *err_a, double *err_b);
/* A solution with t scaled by a factor, as what the card takes. */
void calib_to_wire(const struct calib_problem *p, const struct calib_solution *s, double scale, struct vt_calib *out);

/* Text: "fa cxa cya k1a k2a fb cxb cyb k1b k2b", then R by rows, then T. 0 or -1. */
int calib_write(const char *path, const struct vt_calib *c, const char *note);
int calib_read(const char *path, struct vt_calib *c);

#endif
