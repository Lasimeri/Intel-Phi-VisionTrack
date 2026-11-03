/* stereo.h: correspondence along a rectified row, and the geometry. The search is zero-mean normalized correlation at one pyramid level, for a coarse disparity
 * and for how ambiguous it is; LK (lk.h) then takes it to sub-pixel in x and y, and the y it finds is the epipolar error. The geometry is long double: x87 on both
 * machines, so a replay matches bit for bit. */
#ifndef VT_STEREO_H
#define VT_STEREO_H

#include "plane.h"

#define STEREO_W 16 /* lanes */
#define STEREO_H 11

struct stereo_scan {
    float disparity;   /* of the best peak, in pixels of the level; negative when no candidate fits */
    float ncc;         /* the best peak's correlation */
    float second;      /* the best correlation at least 3 candidates away from it */
};

/* Scans (x, y) of left against the right row, for integer disparities dmin .. dmax in pixels of the level. */
void stereo_scan_vx(const struct vt_plane *left, const struct vt_plane *right, float x, float y, int dmin, int dmax, struct stereo_scan *out);
void stereo_scan_ref(const struct vt_plane *left, const struct vt_plane *right, float x, float y, int dmin, int dmax, struct stereo_scan *out);

/* Rectified geometry: Z = f B / d. Returns 0 and leaves p alone if d is not positive. */
struct stereo_geom { long double fx, fy, cx, cy, baseline; };
int stereo_triangulate(const struct stereo_geom *g, long double xl, long double yl, long double xr, long double yr, long double *p);

/* The rms distance, pixels, between the observed points in both images and p projected into each. */
long double stereo_reprojection(const struct stereo_geom *g, const long double *p, long double xl, long double yl, long double xr, long double yr);

/* The covariance of p from the image noise: var_x on x in the left image, var_y on the averaged y, var_d on the disparity. */
void stereo_covariance(const struct stereo_geom *g, const long double *p, long double disparity, long double var_x, long double var_y, long double var_d, long double *cov33);

#endif
