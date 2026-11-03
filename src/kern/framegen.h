/* framegen.h: temporal frame generation from whole-frame motion, on the card.
 *
 * The motion field is AMD's FidelityFX Optical Flow algorithm (FSR 3: the luminance pyramid, the exhaustive 16 x 16 SAD search of each 8 x 8 block around the level above's estimate, the 3 x 3 "middle vector" filter, the
 * upscale that picks among four candidates by SAD, the zero-motion fallback on the finest level, motion turned off at a scene change), ported from the published MIT-licensed shaders to MVEX. See NOTICE below.
 * What is not AMD's is the rest. FSR 3's interpolation leans on a game's own motion vectors, which a camera does not have; here a generated frame gets a vector at its own instant from the field around it, the one for
 * which the two real frames agree when each is moved half way, and its picture is blended from both under windows that overlap, so that no block's edge shows. The scene-change test is a simple one of my own over the same
 * nine sections, not AMD's divergence.
 *
 * Every stage takes a range of block or section numbers: a team may run them in parts, in any order, and the answer does not depend on how many threads did. Planes are integer-valued F32, so MVEX and the lane model agree
 * bit for bit.
 *
 * NOTICE: the search, filter and upscale derive from the FidelityFX SDK, Copyright (C) Advanced Micro Devices, Inc., MIT licence: ffx_opticalflow_compute_optical_flow_v5.h, ffx_opticalflow_filter_optical_flow_v5.h,
 * ffx_opticalflow_scale_optical_flow_advanced_v5.h. Full text: LICENSES/FidelityFX-MIT.txt. */
#ifndef VT_FRAMEGEN_H
#define VT_FRAMEGEN_H

#include <stddef.h>
#include <stdint.h>

#include "plane.h"

#define FG_MAX_LEVELS 7 /* AMD's pyramid depth */
#define FG_BLOCK 8      /* pixels of a block at its level */
#define FG_SECTIONS 9   /* of the picture, for the scene-change test */
#define FG_BINS 64

struct fg_vec {
    int16_t x, y;
};

struct fg_field {
    int w, h; /* blocks */
    struct fg_vec *v;
};

/* The vectors point from the newer frame to the older: a thing at x in b is at x + v in a. */
struct fg_job {
    struct vt_plane a[FG_MAX_LEVELS], b[FG_MAX_LEVELS]; /* older and newer frames: [0] the level the field is built at, one level finer than nothing else: half the output's resolution */
    struct vt_plane a0, b0, out;                        /* full resolution */
    int levels;
    struct fg_field raw[FG_MAX_LEVELS], fil[FG_MAX_LEVELS], up[FG_MAX_LEVELS], mid; /* up[L]: what level L starts from */
    uint32_t *hist_a, *hist_b;                          /* FG_SECTIONS x FG_BINS */
    int scene_changed;
};

/* Bytes fg_init needs for a field-level picture of w x h with this many levels. */
size_t fg_bytes(int w, int h, int levels);
void fg_init(struct fg_job *j, void *mem, int w, int h, int levels);

/* Scene change. One section of a plane into its histogram (sections may run on any threads); then the verdict between hist_a and hist_b. */
void fg_hist_section(const struct vt_plane *p, uint32_t *hist, int section);
int fg_scene_verdict(const struct fg_job *j);

/* The blocks of level L, numbered row by row. In this order for each level, coarsest first; each over all its blocks before the next starts. fg_upscale's blocks are those of level L - 1. */
void fg_search(const struct fg_job *j, int level, int i0, int i1);
void fg_filter(const struct fg_job *j, int level, int i0, int i1);
void fg_upscale(const struct fg_job *j, int level, int i0, int i1);

/* After level 0: each block's vector at the middle instant. */
void fg_mid_vectors(const struct fg_job *j, int i0, int i1);

/* The generated picture, 16 x 16 tiles of `out`, numbered row by row. fg_tiles says how many there are. */
int fg_tiles(const struct fg_job *j);
void fg_blend(const struct fg_job *j, int t0, int t1);

#endif
