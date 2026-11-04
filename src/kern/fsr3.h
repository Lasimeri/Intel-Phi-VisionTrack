/* fsr3.h: AMD FidelityFX Super Resolution 3.1.6, the temporal upscaler, as MVEX on the card, for one grey camera, and what the user ordered around it: optical flow first at the camera's size, then every frame up by FSR, and a
 * Lanczos down to the size shown. Camera D: 720p12, by the flow 720p24, 1440p24, 1080p24.
 *
 * Passes, as the shaders have them: the luma, the shading change (sorted 5-sample sets, a pyramid of three mips), reactivity and accumulation, luma instability, the accumulate pass (history reprojected through a Lanczos 2 sampler with
 * deringing, a Lanczos of the new frame, a rectification box, thin-feature locks), RCAS with its denoise. Everything a shader computes per pixel and a camera makes constant is computed once:
 *   grey       Co and Cg are zero, so one channel; the box clamp in YCoCg space is a clamp on an interval
 *   exposure   1; no pre-exposure change; no HDR tonemap
 *   jitter     0: fixed cameras cannot have any. So not the shader's narrow 3 x 3 slice of each frame, which only jitter fills in: the whole Lanczos 2 over 4 x 4, with the first sample of a pixel taken from it and not from the box. On a still pixel
 *              this is a temporal denoiser over a spatial upscale, not a gain in detail
 *   motion     one vector for each 16 x 16 block, from AMD's optical flow of the two frames, refined below a pixel by one Lucas-Kanade step: no gather. A still block reads its history directly
 *   depth      none exists at this rate. A moving block stands for near and a still one for far: disocclusion is a still block under where a moving block was, what FSR's reconstructed depth finds with a real depth buffer
 *   reactive   one value a block, from how the motion field diverges
 * Lanes of the output cannot use gather either: 16 output columns span under 16 source columns, so a permute with a table made at the start fetches them. Each source row is filtered across once, and an output row mixes four of them.
 * By the operator's order, 2026-11-03, the flow comes first: its generated frames enter like any other, half a real step from the real frames on each side, so what is invented does enter the history. history_steps says how far back that history lies in the steps of mv_scale, for when frames were let go.
 *
 * Every stage takes (tid, n): a team may run it in parts, and the answer does not depend on how many threads. The same source is native MVEX on the card and a bit-exact model on the host.
 *
 * NOTICE: derives from the FidelityFX SDK shaders ffx_fsr3upscaler_*.h and ffx_fsr1.h, Copyright (C) Advanced Micro Devices, Inc., MIT licence. Full text: LICENSES/FidelityFX-MIT.txt. */
#ifndef VT_FSR3_H
#define VT_FSR3_H

#include <stddef.h>
#include <stdint.h>

#include "framegen.h"
#include "plane.h"

enum fsr_stage { FSR_REFINE, FSR_INPUT, FSR_HPASS, FSR_DIFF, FSR_MIP0, FSR_MIP1, FSR_MIP2, FSR_SC, FSR_REACT, FSR_INST, FSR_ACC, FSR_SHARPEN, FSR_STAGES };

struct fsr;

/* Bytes fsr_init needs: rw x rh in, uw x uh out of FSR, 1.25 to 2 times; dw x dh by Lanczos from there, at most 3/2 times smaller, or 0 x 0 to show it as it is. */
size_t fsr_bytes(int rw, int rh, int uw, int uh, int dw, int dh);
/* sharpness: RCAS stops, 0 the sharpest. NULL if the sizes cannot work. */
struct fsr *fsr_init(void *mem, int rw, int rh, int uw, int uh, int dw, int dh, long double sharpness);
/* The newest frame and the one before it as planes of Q6 grey, and the field between them, blocks of 16 x 16 at this resolution (fg_job.fil[0], level 1). mv_scale: how much of a whole vector lies between these two frames. */
void fsr_begin(struct fsr *f, const struct vt_plane *cur, const struct vt_plane *prev, const struct fg_field *field, long double mv_scale, long double history_steps, int scene_changed);
/* One stage, one thread's part. In order; each over all threads before the next. */
void fsr_stage(struct fsr *f, int stage, int tid, int n);
/* After all threads have done a stage: its padding. Thread 0. */
void fsr_stage_end(struct fsr *f, int stage);
/* The picture as bytes, in thread parts, at the size it goes out. 64 B aligned. */
void fsr_picture(const struct fsr *f, uint8_t *pixels, int tid, int n);
/* The frame is done: what it wrote is now the history. */
/* The second stage for any plane of the upscaled size, not only FSR's own: the same tables, row_used saying which of its rows are worth reading. fsr_down_span finds the rows of the result worth writing; 0 if none. Each thread has its own temp of fsr_down_temp_bytes. */
size_t fsr_down_temp_bytes(const struct fsr *f);
int fsr_down_span(const struct fsr *f, const uint8_t *row_used, int *y0, int *y1);
void fsr_down_plane(const struct fsr *f, const struct vt_plane *src, const uint8_t *row_used, int y0, int y1, uint8_t *pixels, float *temp, int tid, int n);
void fsr_end(struct fsr *f);
/* RCAS, stops, from the next frame on. */
void fsr_set_sharpness(struct fsr *f, long double stops);

#endif
