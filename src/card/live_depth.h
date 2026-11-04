/* live_depth.h: the depth map of a three-camera session, on the session's own team, between frames: each camera's frame updates its vectors and its still picture (still.h), each frame of camera D
 * then sweeps the three still pictures and sends the map, the vectors after it, the announcement last. */
#ifndef VT_LIVE_DEPTH_H
#define VT_LIVE_DEPTH_H

#include <stddef.h>
#include <stdint.h>

#include "arena.h"
#include "plane.h"
#include "team.h"
#include "vt_dma.h"
#include "vt_proto.h"

struct live_depth;

/* What it will take from the arena. */
size_t ld_bytes(const struct vt_config *cfg, int threads);
/* NULL, with the reason on stderr, when it cannot be done. */
struct live_depth *ld_create(const struct vt_config *cfg, struct team *tm, int threads, struct arena *a, struct vt_dma *dma);
/* A new geometry: until one holds all three cameras no map is made. */
void ld_set_calib(struct live_depth *l, const struct vt_calib *c, uint64_t seq);
/* The pyramid level ld_frame wants. */
int ld_flow_level(const struct live_depth *l);
/* One camera's newest frame, the plane at ld_flow_level. */
void ld_frame(struct live_depth *l, int cam, const struct vt_plane *cur);
/* The same in two parts, for a camera whose vectors run on another team: ld_vectors on that team's first thread, reading nothing ld_rest writes, then ld_rest on the session's. A map between the two copies cells being written: only what the window shows of them with every vector on. */
void ld_vectors(struct live_depth *l, int cam, const struct vt_plane *cur, struct team *tm);
void ld_rest(struct live_depth *l, int cam, const struct vt_plane *cur);
/* ld_rest in its two parts: the mean, on any team; the attitude check, on the session's. A map may read a mean while another team updates it: it sees some of each frame. */
void ld_update(struct live_depth *l, int cam, const struct vt_plane *cur, struct team *tm);
void ld_attitude(struct live_depth *l, int cam);
size_t ld_cell_bytes(const struct live_depth *l, int cam);
const struct vt_cell *ld_cells(const struct live_depth *l, int cam, int *cols, int *rows);
/* After ld_frame of camera D: the map, if it can be made, is sent with status_addr set to its sequence after it. The last DMA line's sequence; 0 when no map was made; UINT64_MAX on failure. */
uint64_t ld_map(struct live_depth *l, const struct vt_msg *m, uint64_t status_addr);

#endif
