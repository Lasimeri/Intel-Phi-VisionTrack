/* stars.h: the room's LEDs as fixed stars, on the desktop host. After the methods of orbital and satellite navigation: docs/architecture.md.
 *   catalogue    lights that repeat across long exposures: what a star catalogue is to a star tracker.
 *   recursive    each known light looked for close to where it should be (Spratling and Mortari 2009, "recursive" star identification).
 *   lost in space  when lights go missing: every light in the frame votes for the shift that would explain it against the catalogue; the pattern decides, no prior pose.
 *   attitude     the rotation that best turns catalogue bearings into measured ones: Wahba's problem, in its small-angle least-squares form, enough for a camera that was bumped.
 *   RAIM         a light that disagrees with the others (occluded, reflected, moved) is thrown out by its residual, as a receiver throws out a faulty satellite.
 *   GDOP         how much angle one pixel of centroid noise becomes, for the geometry of the lights in use. Reported, because a fix is only as good as its beacons' spread. */
#ifndef VT_STARS_H
#define VT_STARS_H

#include <stdint.h>

#define STAR_MAX 128

struct star {
    double x, y, flux;
    int area;
};

struct star_cat {
    int n;
    struct star s[STAR_MAX];
};

struct star_fix {
    int known, seen, used;      /* in the catalogue; found this time; agreeing with the others */
    int lost_in_space;          /* 1: found through the pattern vote */
    double dx, dy;              /* median shift of the lights in use, pixels */
    double wx, wy, wz;          /* the camera's rotation from the catalogue's pose, radians: x tilt, y pan, z roll */
    double rms_px;              /* residual of the lights in use after that rotation */
    double gdop;                /* degrees of attitude uncertainty per pixel of centroid noise; 0 when not computable */
    int roll_known;             /* 0: too few or too close lights: roll held at zero */
};

/* Every light in a 16-bit frame: bright against a DARK ring. Returns how many, brightest first. k: threshold in noise sigmas. */
int star_detect(const uint16_t *g, int w, int h, double k, struct star *out, int max);
/* One light near (x, y), within radius: its centroid over the local background. 1 if found. */
int star_find(const uint16_t *g, int w, int h, double x, double y, int radius, struct star *out);
/* A camera's fix against its catalogue. f: focal length, pixels; principal point at the image centre. hint_dx, dy: where the last fix left the lights. allow_lost_in_space: may scan the whole frame, tens of milliseconds. */
void star_track(const struct star_cat *c, const uint16_t *g, int w, int h, double f, double hint_dx, double hint_dy, int allow_lost_in_space, struct star_fix *fix);
/* The rotation matrix of a fix's rotation vector: a bearing b in the catalogue's pose is seen at R b now. */
void star_rotation(double wx, double wy, double wz, double *R);
/* Differential correction: X_b = R X_a + T, and each camera has since turned so its bearings are Oa X_a and Ob X_b. Gives the relation that holds now: R' = Ob R Oa', T' = Ob T. A camera's translation is not seen by this. */
void star_correct_pair(const double *R, const double *T, const double *Oa, const double *Ob, double *Rout, double *Tout);

/* Catalogue from two exposures: lights within max_shift of each other, area at least min_area (a hot pixel is one), no other light within min_separation. Returns how many. */
int star_catalogue(const struct star *a, int na, const struct star *b, int nb, double max_shift, int min_area, double min_separation, struct star_cat *out);
int star_cat_write(const char *path, const struct star_cat *c);
int star_cat_read(const char *path, struct star_cat *c);

/* A PGM of 8 or 16 bits as 16-bit grey. Returned with malloc, NULL with a message. */
uint16_t *star_read_pgm(const char *path, int *w, int *h);

#endif
