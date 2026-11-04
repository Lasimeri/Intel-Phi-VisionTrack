/* vecdraw.h: a camera's moved cells, drawn as lines over its picture at the upscaled size, before that picture is brought down.
 *
 * The pictures are one channel, so the lines cannot go into them: they go into a plane of their own, 0 or 1 per pixel, which the second stage's Lanczos brings down
 * with the same tables as the picture, and the window tints the picture by it.
 *
 * One thread draws. The last frame's lines are drawn again in zero rather than the plane being cleared: a frame moves a few hundred lines, and a clear would touch
 * every line of a plane of about 15 MiB. The rows touched are marked, so what brings the plane down can pass over the rest. */
#ifndef VT_VECDRAW_H
#define VT_VECDRAW_H

#include <stddef.h>
#include <stdint.h>

#include "plane.h"
#include "vt_proto.h"

struct vecdraw;

/* A plane w x h, and room for the lines of this many cells. */
size_t vd_bytes(int w, int h, int cells);
struct vecdraw *vd_init(void *mem, int w, int h, int cells);

/* The cells that moved, of a camera's picture at pyramid level `level`; `view` = this plane's width over the camera's. inside, if not NULL: x0, y0, x1, y1 in the camera's pixels, and only cells whose centre lies there. The number of lines drawn. */
int vd_draw(struct vecdraw *d, const struct vt_cell *cells, int cols, int rows, int level, long double view, const float *inside);

const struct vt_plane *vd_plane(const struct vecdraw *d);
/* One byte a row of the plane: non-zero where a line may have touched it. */
const uint8_t *vd_rows(const struct vecdraw *d);

#endif
