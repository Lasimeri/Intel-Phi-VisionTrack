/* depthview.h: what the live window does with a depth map from the card: the picture, the room from above, the card's vectors over the cameras, and the map as files. A map is the bytes of one depth buffer
 * (struct vt_depth_out and what follows it, vt_proto.h), copied whole. */
#ifndef VT_DEPTHVIEW_H
#define VT_DEPTHVIEW_H

#include <stdio.h>

#include "vt_proto.h"
#include "x11view.h"

/* The map fills a pane of a whole multiple of its size, nearest depth warm, furthest cold, no depth black. */
void dv_pane(struct x11view *v, const void *map, int x, int y, int pane_w, int pane_h, int mirror);
/* The colour scale, with its depths, along a pane's bottom edge. */
void dv_legend(struct x11view *v, const void *map, int x, int y, int w);
/* Every point of the map from above, in the geometry's world (camera L at the origin), at (cx, cy) in the window, in a box. side: window pixels per metre of X, negative when the display is mirrored. */
void dv_cloud(struct x11view *v, const void *map, const struct vt_calib *cal, int box_x, int box_y, int box_size, int cx, int cy, double side, double px_per_m);
/* The vectors of one camera over its picture. all = 0: cells that moved, at true length times gain. all = 1: every good vector, at a far higher gain, which shows the noise. */
void dv_vectors(struct x11view *v, const void *map, int cam, int pane_x, int pane_y, int pane_w, double view, int mirror, int all);
/* The same from a frame's own vectors, wherever they came from. inside, if not NULL: x0, y0, x1, y1 in the camera's pixels, and only cells whose centre lies there. */
void dv_cells(struct x11view *v, const struct vt_cell *cells, uint32_t cols, uint32_t rows, int flow_level, int pane_x, int pane_y, int pane_w, double view, int mirror, int all, const double *inside);

/* Writes PREFIX.bin (the buffer), PREFIX.ply (points in the world frame, with confidence), PREFIX.pgm (depth). 0, or -1. */
int dv_save(const void *map, const struct vt_calib *cal, const char *prefix);
/* One line per map, everything the card reported: the raw source of any claim about the live map. Writes the header when line is 0. */
void dv_csv(FILE *f, const void *map, double wall_s, double tsc_hz, unsigned line);

#endif
