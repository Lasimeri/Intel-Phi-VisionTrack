/* fuse.h: two independently tracked views of one region, made into one 3D point and one Kalman state. No patch matching between the views and no synchronized shutters: each camera's tracker gives
 * its region's centre at that camera's own time, the older of the two is carried to the newer's time along its measured image rate, both are undistorted to rays, and the rays' closest point
 * is the measurement. Long double, the same on card and host. */
#ifndef VT_FUSE_H
#define VT_FUSE_H

#include "kalman.h"
#include "vt_proto.h"

struct fuse {
    long double fa, cxa, cya, k1a, k2a, fb, cxb, cyb, k1b, k2b;
    long double R[9], T[3];  /* X_b = R X_a + T: camera A is the world frame */
    int have_calib;
    struct fuse_obs {
        int valid;
        long double u, v, rate_u, rate_v;
        uint64_t t_ns;
    } obs[2];
    struct kf kf;
    uint64_t last_meas_ns, last_ns;
    long double sigma_c, r_floor, q_accel, v0_sigma, max_reproj, max_age_s, patience_s;
};

void fuse_init(struct fuse *f, const struct vt_config *c);
void fuse_set_calib(struct fuse *f, const volatile struct vt_calib *c);

/* The closest point of the two rays from image points (ua, va) and (ub, vb). Returns 0 when it lies behind a camera or the rays are near parallel. reproj: rms pixels over both views. deg: the angle
 * between the rays, what depth's accuracy lives on. */
int fuse_triangulate(const struct fuse *f, long double ua, long double va, long double ub, long double vb, long double *p, long double *reproj, long double *deg);

/* A tracker's result for one camera: remembers it; when it comes from camera B, triangulates and updates the filter; always writes the filter's output into the result. */
void fuse_frame(struct fuse *f, int camera, struct vt_result *r);

#endif
