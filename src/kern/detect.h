/* detect.h: corners inside an operator's region: the Harris response on a 5 x 5 box of the structure tensor, over Sobel gradients. No
 * sqrt or division in the response, so the lane model and MVEX agree bit for bit. Runs on one thread: a region is small and rare. */
#ifndef VT_DETECT_H
#define VT_DETECT_H

#include "plane.h"

/* Scratch: detect_scratch_bytes(w, h) of it, 64 B aligned, for a region of at most w x h pixels. */
size_t detect_scratch_bytes(int w, int h);

/* Up to max corners inside the rectangle (clipped to the plane's margin), strongest first, at least min_dist apart, response above quality x the
 * strongest. Returns how many. */
int detect_corners(const struct vt_plane *pl, float rx, float ry, float rw, float rh, int margin, int max, float min_dist, float quality, float *x, float *y, void *scratch);

/* Worst-case region this scratch may hold. */
#define DETECT_MAX_CANDIDATES 8192

#endif
