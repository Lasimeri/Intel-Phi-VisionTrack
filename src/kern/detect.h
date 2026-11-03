/* detect.h: corners inside an operator's region: the Harris response on a 5 x 5 box of the structure tensor, over Sobel gradients. No
 * sqrt or division in the response, so the lane model and MVEX agree bit for bit.
 *
 * Three stages run on a whole team, on rows that do not depend on each other, so the answer does not depend on how many threads did them; the last runs on one. On one thread a 384 x 307 box cost 6.1 to 8.0 ms of
 * the tracking path under one threshold for the box, and 26.37 ms under a threshold per neighbourhood with the selection's two passes (scene bigbox, 2026-11-03); on the team, 3.78 ms. A region reseeds while it tracks. */
#ifndef VT_DETECT_H
#define VT_DETECT_H

#include "plane.h"

/* Worst-case number of candidates this scratch may hold. */
#define DETECT_MAX_CANDIDATES 8192

/* Scratch: detect_scratch_bytes(w, h) of it, 64 B aligned, for a region of at most w x h pixels. */
size_t detect_scratch_bytes(int w, int h);

struct detect_job {
    const struct vt_plane *pl;
    int x0, y0, x1, y1, ax, cols, nv, rows, cell_rows, max;
    float min_dist, quality;
    float *xx, *xy, *yy, *resp, *cell, *near, *score;
    int *cx, *cy;
    uint32_t *order, *hist;
};

/* Lays a job out over the scratch. False if nothing of the rectangle lies inside the plane's margin: there is nothing to run. */
int detect_begin(struct detect_job *j, const struct vt_plane *pl, float rx, float ry, float rw, float rh, int margin, int max, float min_dist, float quality, void *scratch);

/* The stages, in this order, each over all of its range: tensor rows 0 .. rows + 4, response rows 0 .. rows, cell rows 0 .. cell_rows. */
void detect_tensor(const struct detect_job *j, int r0, int r1);
void detect_response(const struct detect_job *j, int r0, int r1);
void detect_cells(const struct detect_job *j, int c0, int c1);

/* Up to max corners, best first, at least min_dist apart, each above quality x the strongest of its neighbourhood and above a floor. Returns how many. */
int detect_finish(const struct detect_job *j, float *x, float *y);

#endif
