/* kalman.h: the constant-velocity Kalman filter on x = [X Y Z Vx Vy Vz]. Six states on one thread: a small matrix problem, deliberately not spread across the
 * cores. dt always comes from acquisition timestamps. All long double: x87 on the card and on the host alike, so a replay matches bit for bit. */
#ifndef VT_KALMAN_H
#define VT_KALMAN_H

struct kf {
    long double x[6];
    long double P[6][6];
    int valid;
};

/* Starts from one measurement: position = z, velocity 0 with standard deviation v0_sigma. */
void kf_init(struct kf *k, const long double *z, const long double *r33, long double v0_sigma);
/* The transition over dt seconds. q_accel: white-acceleration density, (m/s^2) per sqrt(Hz). */
void kf_predict(struct kf *k, long double dt, long double q_accel);
/* The normalized innovation y' S^-1 y of a measurement, not applied. */
long double kf_innovation(const struct kf *k, const long double *z, const long double *r33);
/* The measurement update, in Joseph form so P stays symmetric positive. */
void kf_update(struct kf *k, const long double *z, const long double *r33);

#define KF_GATE_NIS 16.27L /* chi-square, 3 degrees of freedom, p = 0.001 */

#endif
