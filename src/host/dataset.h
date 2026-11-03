/* dataset.h: a rendered sequence, kept on disk so a benchmark and its replay read the same bytes: out/data/NAME-WxH-N-FPS.y8, frames in order, left then right. Timestamps are the dataset's own:
 * acq_ns = 1 s + index / fps, exact integers, so a replay holds the same clock. */
#ifndef VT_DATASET_H
#define VT_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include "synth.h"
#include "vt_proto.h"

struct dataset {
    struct scene s;
    uint8_t *frames;
    size_t frame_bytes; /* one image pair */
    size_t file_bytes;
};

/* Maps the rendered file, rendering it first if it is not there (14 threads). 0, or -1 with a message. */
int dataset_open(struct dataset *d, enum scene_id id, int w, int h, int nframes, double fps);
void dataset_close(struct dataset *d);
static inline const uint8_t *dataset_left(const struct dataset *d, int i) { return d->frames + (size_t)i * d->frame_bytes; }
static inline const uint8_t *dataset_right(const struct dataset *d, int i) { return dataset_left(d, i) + (size_t)d->s.w * (size_t)d->s.h; }
static inline uint64_t dataset_acq_ns(const struct dataset *d, int i) { return 1000000000ull + (uint64_t)((double)i * 1e9 / d->s.fps + 0.5); }
/* Camera 1's clock in a dual scene: its own shutter delay on top. */
static inline uint64_t dataset_acq_ns_cam(const struct dataset *d, int i, int camera) { return dataset_acq_ns(d, i) + (camera == 1 ? (uint64_t)(d->s.b_delay_s * 1e9 + 0.5) : 0ull); }

/* The region an operator would draw on the object at frame i: inside its projected box, a tenth short of each side. */
void dataset_roi(const struct dataset *d, int i, float *roi);
void dataset_roi_cam(const struct dataset *d, int i, int camera, float *roi);

/* The project's default configuration for a size. */
void vt_default_config(struct vt_config *c, const struct scene *s, int stereo);
/* A dual scene's true calibration. */
void dataset_true_calib(const struct dataset *d, struct vt_calib *c);

#endif
