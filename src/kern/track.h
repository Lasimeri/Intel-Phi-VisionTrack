/* track.h: one frame's whole computation: preprocessing, pyramid, LK, the region state machine, stereo, Kalman. The same file compiles for the card (MVEX, teams) and for the
 * host (the lane model, one thread), so a whole run can be replayed bit for bit.
 *
 * Two halves, so the pyramid of the next frame can run beside the rest of this one: tk_build fills a pyramid set, tk_track consumes sets in order. With one pipeline they run in turn. */
#ifndef VT_TRACK_H
#define VT_TRACK_H

#include <stddef.h>

#include "vt_proto.h"

/* Runs fn on every thread of a team, or once with (arg, 0, 1). */
typedef void (*tk_fn)(void *arg, int tid, int n);
typedef void (*tk_run_fn)(void *ctx, tk_fn fn, void *arg);
/* Memory, 64 B aligned, never freed by track.c. */
typedef void *(*tk_alloc_fn)(void *ctx, size_t bytes);

struct tracker;

/* Everything the config asks for, allocated now. threads_pyr and threads_track size the two groups' per-thread scratch; with one pipeline pass the same run twice. NULL when it
 * cannot be done, the reason on stderr. */
struct tracker *tk_create(const struct vt_config *cfg, int threads_pyr, int threads_track, tk_alloc_fn alloc, void *alloc_ctx, tk_run_fn run_pyr, void *pyr_ctx, tk_run_fn run_track, void *track_ctx);

/* How many pyramid sets exist. Frame k lives in set k mod that count; tk_track of frame k reads sets k and k - 1. */
int tk_sets(const struct tracker *t);

/* The pyramid of a frame, into set `set`. left and (stereo) right: width-stride bytes per row, 16 B aligned. Reads only the message's build keys. Thread: the pyramid group's tid 0. */
void tk_build(struct tracker *t, int set, const uint8_t *left, const uint8_t *right, const struct vt_msg *m);
/* The rest of a frame, whose pyramid is complete in `set`. out is complete on return, except the cycle counts of the stages outside it. Thread: the track group's tid 0. */
void tk_track(struct tracker *t, int set, const struct vt_msg *m, struct vt_result *out);
/* After tk_build of a real frame into `set` and before its tk_track: makes the frame half way between it and the last real one, if frame generation is on and a region exists to gain from it. Returns the set it is in, to
 * pass to tk_track with the message written to *gm, or -1 if there is none. A session of three sets, with real frame k in set k mod 3, the generated frame takes the set frame k + 1 will overwrite. */
int tk_generate(struct tracker *t, int set, const struct vt_msg *m, struct vt_msg *gm);
/* From now on every generated frame's picture is written to these w x h bytes, 64 B aligned, and a frame is generated whether a region exists or not. */
void tk_set_picture(struct tracker *t, uint8_t *pixels);
/* The appearance correlation of a zero-mean template of 32 x 21 against a plane at (x, y): what the search for a lost region is built on, exposed for its gate. */
struct vt_plane;
long double tk_correlate(const struct vt_plane *pl, const float *app, long double norm, int x, int y);
/* What frame generation cost so far, per stage, to stderr. */
void tk_fg_report(const struct tracker *t, const char *name, double cycles_per_us);

/* The plane a result names in debug_level: left camera, of the set it was computed in. NULL if there is none. */
const struct vt_plane *tk_debug_plane(const struct tracker *t, int set, const struct vt_result *r);

/* One level of a set's pyramid, camera 0: valid until tk_build fills that set again. */
const struct vt_plane *tk_level(const struct tracker *t, int set, int level);

/* How many bytes of a result are worth moving: the header and the features in use, in whole lines. */
size_t tk_result_bytes(const struct vt_result *r);

/* A message's soft-configuration key this tracker acts on at the build stage: its session must treat others as tk_track's. */
int tk_is_build_key(uint32_t key);

#endif
