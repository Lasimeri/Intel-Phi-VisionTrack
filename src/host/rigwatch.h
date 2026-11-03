/* rigwatch.h: has this camera moved? Geometry is valid only for the pose it was solved in, and a person moving in front of a camera must not look like the camera moving. So: thumbnails, and only
 * the cells that are textured and have themselves held still over a recent history: a moving person excludes themselves, however much of the frame they fill. The question, of a robust middle: do
 * those cells still line up where they were, or at some other shift? On the desktop host, a few milliseconds a check.
 * A first design without the history failed its test on frames of known answer: a person filling camera D's frame read as a moved camera, at the same mismatch ratio as a real move in camera L. */
#ifndef VT_RIGWATCH_H
#define VT_RIGWATCH_H

#include <stdint.h>

#define RIG_CELL 8      /* image pixels per thumbnail cell */
#define RIG_MAX_SHIFT 6 /* cells searched each way */
#define RIG_HIST 8      /* thumbnails of history */
#define RIG_MIN_CELLS 64 /* fewer usable cells: too dark or too bare to tell, and nothing is claimed */

struct rig_compare {
    float at_zero;      /* the middle mismatch of textured cells at no shift, in grey levels */
    float best;         /* the same at the best shift */
    int dx, dy;         /* the best shift, in image pixels */
    int cells;          /* textured cells used */
    int moved;          /* 1: the camera moved between the two */
};

/* w x h thumbnail cells for an image: (image size) / RIG_CELL. */
void rig_thumb(const uint8_t *gray, int img_w, int img_h, float *thumb);
/* From k recent thumbnails, oldest first: 1 for the cells that held still, by this camera's own noise: their range is compared with the middle cell's range, whatever the light. scratch: w * h floats. */
void rig_stable_mask(const float *const *hist, int k, int w, int h, float *scratch, uint8_t *mask);
/* Compares a thumbnail with its reference, on the mask's cells. scratch: w * h floats. */
void rig_compare(const float *ref, const uint8_t *mask, const float *cur, int w, int h, float *scratch, struct rig_compare *out);

#endif
