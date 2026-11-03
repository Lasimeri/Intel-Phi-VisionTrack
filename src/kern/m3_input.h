/* m3_input.h: what both halves of the depth gate agree on: the file the host writes for the card: geometry as exact doubles, then the images of each frame, D, R, L. With more than one frame the map comes
 * from the still pictures (still.h) those frames leave behind, as it does live. */
#ifndef VT_M3_INPUT_H
#define VT_M3_INPUT_H

#include <stdint.h>

#define M3_MAGIC 0x34444d54u
#define M3_LEVEL_D 2 /* pyramid level of each view the sweep reads */
#define M3_LEVELS 3
#define M3_MAX_FRAMES 64

struct m3_cam {
    int32_t w, h;
    double f, cx, cy; /* at level 0 */
    double R[9], T[3]; /* X_cam = R X_d + T */
    float weight;
    float pad;
};

struct m3_file {
    uint32_t magic;
    int32_t views;      /* source views in use: 1 = R, 2 = R + L */
    int32_t planes, agg_radius;
    double z_near, z_far;
    float trunc, tex_min, conf_min, eps;
    int32_t frames, avg;
    float motion_px, motion_grey;
    struct m3_cam d, r, l;
};

#endif
