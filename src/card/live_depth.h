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
/* After ld_frame of camera D: the map, if it can be made, is sent with status_addr set to its sequence after it. The last DMA line's sequence; 0 when no map was made; UINT64_MAX on failure. */
uint64_t ld_map(struct live_depth *l, const struct vt_msg *m, uint64_t status_addr);

#endif
