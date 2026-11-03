/* synth.h: deterministic synthetic stereo scenes with known geometry, rendered on the desktop host. A pinhole left camera at the world origin looking along
 * +Z, x right, y down; a right camera at (baseline, 0, 0), both rectified. Fronto-parallel textured planes: a background, the object, an optional occluder.
 * Every pixel integrates a few sub-samples, as a sensor does. The truth is analytic: nothing about it comes from the rendered pixels. */
#ifndef VT_SYNTH_H
#define VT_SYNTH_H

#include <stdint.h>

enum scene_id {
    SC_TRANSLATE,   /* a textured rectangle moves at constant image velocity over a still background */
    SC_SHAPES,      /* a disc on a curved path, a second rectangle moving elsewhere */
    SC_CAMERA,      /* the camera moves: the whole scene shifts */
    SC_BRIGHTNESS,  /* the object moves while global gain and offset vary */
    SC_NOISE,       /* additive sensor noise, sigma 4 grey levels */
    SC_OCCLUSION,   /* a bar crosses part of the object */
    SC_LOSS,        /* a bar covers the object completely for a while */
    SC_LEAVE,       /* the object leaves the field of view */
    SC_HELIX,       /* a 3D helix: depth changes, known velocity */
    SC_LINEAR3D,    /* constant 3D velocity: the filter's own model */
    SC_REPEAT,      /* the object carries a 2D periodic pattern with corners, its period inside the disparity range: stereo is geometrically ambiguous */
    SC_FLAT,        /* the object has almost no texture: nothing to track */
    SC_GLITCH,      /* the right camera's image jumps sideways for frames 70 .. 79: wrong stereo measurements the innovation gate must refuse */
    SC_BIGBOX,      /* the object moves over a background as busy as itself, selected with a box 2.4 times its size: what an operator's drag really holds */
    SC_HOLD,        /* a still object under sensor noise, sigma 4: true motion zero, so any travel of the region is error */
    SC_SMALLBOX,    /* a small object in a box four times its size: most of what is in the box, and most of the box's middle, is background */
    SC_STILLSTART,  /* as bigbox, but the object stays still for 2 s after it is selected and only then moves: nothing tells it from the background until it does */
    SC_ROTATE,      /* as bigbox, and the object turns in the image plane at 30 degrees a second */
    SC_SWING,       /* as bigbox, swinging from side to side, up to 19 pixels a frame: real things accelerate and come back */
    SC_DUAL,        /* a second camera off to the side, turned 17.5 degrees, a different focal length, its shutter 11 ms late: the rig of two cameras that were never a stereo pair */
    SC_COUNT,
};

struct scene {
    enum scene_id id;
    const char *name;
    int w, h, nframes;
    double fps;
    double f, cx, cy, baseline;
    int stereo;
    int select_frame;
    int samples;   /* per pixel side */
    /* SC_DUAL, camera 1: X_b = R X_a + T */
    double b_f, b_R[9], b_T[3], b_delay_s;
};

struct truth {
    int on_screen;         /* the object's centre is in the left image */
    double visible;        /* fraction of the object's rectangle that the left camera really sees */
    double u, v, ur;       /* projected centre: left x, y; right x */
    double X, Y, Z;        /* metres */
    double Vx, Vy, Vz;     /* m/s */
    double half_w, half_h; /* projected half size, pixels */
    double ub, vb, half_wb, half_hb; /* SC_DUAL: the centre and half size in camera 1 */
};

void scene_setup(struct scene *s, enum scene_id id, int w, int h, int nframes, double fps);
void scene_truth(const struct scene *s, int frame, struct truth *t);
/* The same at any moment, in seconds from the sequence's start: a camera with a late shutter is compared with the truth of its own instant. */
void scene_truth_at(const struct scene *s, double seconds, struct truth *t);
/* Renders one image: camera 0 left, 1 right. img: h rows of w bytes. Threads: its own pool of up to 14. */
void scene_render(const struct scene *s, int frame, int camera, uint8_t *img);
/* Where a surface point seen at (x0, y0) in frame f0 of the left image is in frame f1, if it lies on the object and is visible in both. Returns 0 otherwise. */
int scene_map_left(const struct scene *s, int f0, double x0, double y0, int f1, double *x1, double *y1);
enum scene_id scene_by_name(const char *name);
const char *scene_name(enum scene_id id);

#endif
