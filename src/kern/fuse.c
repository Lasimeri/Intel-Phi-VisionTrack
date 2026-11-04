/* fuse.c: see fuse.h. */
#include "fuse.h"

#include <math.h>
#include <string.h>

typedef long double real;

void fuse_init(struct fuse *f, const struct vt_config *c)
{
    memset(f, 0, sizeof *f);
    f->sigma_c = (real)c->sigma_c;
    f->r_floor = (real)c->r_floor;
    f->q_accel = (real)c->q_accel;
    f->v0_sigma = (real)c->v0_sigma;
    f->max_reproj = (real)c->max_reproj;
    f->max_age_s = 0.1L;
}

void fuse_set_calib(struct fuse *f, const volatile struct vt_calib *c)
{
    f->fa = (real)c->fa;
    f->cxa = (real)c->cxa;
    f->cya = (real)c->cya;
    f->k1a = (real)c->k1a;
    f->k2a = (real)c->k2a;
    f->fb = (real)c->fb;
    f->cxb = (real)c->cxb;
    f->cyb = (real)c->cyb;
    f->k1b = (real)c->k1b;
    f->k2b = (real)c->k2b;
    for (int i = 0; i < 9; i++)
        f->R[i] = (real)c->R[i];
    for (int i = 0; i < 3; i++)
        f->T[i] = (real)c->T[i];
    f->have_calib = c->valid != 0;
}

/* Pixel to a ray with z = 1: the radial distortion inverted by fixed-point iteration. */
static void to_ray(real u, real v, real fl, real cx, real cy, real k1, real k2, real *x, real *y)
{
    const real xd = (u - cx) / fl, yd = (v - cy) / fl;
    real xu = xd, yu = yd;
    for (int i = 0; i < 5; i++) {
        const real r2 = xu * xu + yu * yu;
        const real g = 1.0L + r2 * (k1 + r2 * k2);
        xu = xd / g;
        yu = yd / g;
    }
    *x = xu;
    *y = yu;
}

static void to_pixel(real x, real y, real fl, real cx, real cy, real k1, real k2, real *u, real *v)
{
    const real r2 = x * x + y * y;
    const real g = 1.0L + r2 * (k1 + r2 * k2);
    *u = fl * x * g + cx;
    *v = fl * y * g + cy;
}

int fuse_triangulate(const struct fuse *f, real ua, real va, real ub, real vb, real *p, real *reproj, real *deg)
{
    real ax, ay, bx, by;
    to_ray(ua, va, f->fa, f->cxa, f->cya, f->k1a, f->k2a, &ax, &ay);
    to_ray(ub, vb, f->fb, f->cxb, f->cyb, f->k1b, f->k2b, &bx, &by);
    const real *R = f->R, *T = f->T;
    /* In A's frame: ray a from the origin, ray b from C = -R' T along d = R' (bx, by, 1). */
    const real d[3] = {R[0] * bx + R[3] * by + R[6], R[1] * bx + R[4] * by + R[7], R[2] * bx + R[5] * by + R[8]};
    const real C[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
    const real a[3] = {ax, ay, 1.0L};
    /* s a + q = t d + C, least squares: the closest points of the two lines. */
    const real aa = a[0] * a[0] + a[1] * a[1] + a[2] * a[2], dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ad = a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
    const real ac = a[0] * C[0] + a[1] * C[1] + a[2] * C[2], dc = d[0] * C[0] + d[1] * C[1] + d[2] * C[2];
    const real den = aa * dd - ad * ad;
    if (den <= 1e-12L * aa * dd)
        return 0;
    const real s = (ac * dd - dc * ad) / den, t = (ac * ad - dc * aa) / den;
    if (s <= 0.0L || t <= 0.0L)
        return 0;
    for (int i = 0; i < 3; i++)
        p[i] = 0.5L * (s * a[i] + t * d[i] + C[i]);
    const real cosang = ad / sqrtl(aa * dd);
    *deg = acosl(cosang > 1.0L ? 1.0L : cosang < -1.0L ? -1.0L : cosang) * 57.29577951308232L;
    /* Back into both views. */
    real ra, va2, rb, vb2;
    to_pixel(p[0] / p[2], p[1] / p[2], f->fa, f->cxa, f->cya, f->k1a, f->k2a, &ra, &va2);
    const real q[3] = {R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + T[0], R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + T[1], R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + T[2]};
    if (q[2] <= 0.0L)
        return 0;
    to_pixel(q[0] / q[2], q[1] / q[2], f->fb, f->cxb, f->cyb, f->k1b, f->k2b, &rb, &vb2);
    const real e0 = ra - ua, e1 = va2 - va, e2 = rb - ub, e3 = vb2 - vb;
    *reproj = sqrtl((e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3) * 0.5L);
    return 1;
}

/* The measurement's covariance, carried through the triangulation by differences: what differs from frame to frame, which a filter may average away, and the bias of each view's point, which it may not. */
static void meas_cov(const struct fuse *f, real ua, real va, real ub, real vb, const real *p, real *white, real *bias)
{
    const real h = 0.5L, sig = f->sigma_c * f->sigma_c, var[4] = {f->obs[0].var_u, f->obs[0].var_v, f->obs[1].var_u, f->obs[1].var_v};
    real J[3][4];
    for (int k = 0; k < 4; k++) {
        real q[3], rp, dg;
        const int ok = fuse_triangulate(f, ua + (k == 0 ? h : 0.0L), va + (k == 1 ? h : 0.0L), ub + (k == 2 ? h : 0.0L), vb + (k == 3 ? h : 0.0L), q, &rp, &dg);
        for (int i = 0; i < 3; i++)
            J[i][k] = ok ? (q[i] - p[i]) / h : 0.0L;
    }
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            real s = 0.0L, b = 0.0L;
            for (int k = 0; k < 4; k++) {
                    s += J[r][k] * J[c][k];
                    b += J[r][k] * J[c][k] * var[k];
            }
            white[3 * r + c] = s * sig;
            bias[3 * r + c] = b;
        }
    const real fl = f->r_floor * f->r_floor;
    white[0] += fl;
    white[4] += fl;
    white[8] += fl;
}

void fuse_frame(struct fuse *f, int camera, struct vt_result *r)
{
    const int seen = r->state == VT_ST_OBSERVED || r->state == VT_ST_NEW;
    struct fuse_obs *me = &f->obs[camera & 1];
    me->valid = seen;
    if (seen) {
        me->u = (real)r->pt_x;
        me->v = (real)r->pt_y;
        me->rate_u = (real)r->rate_u;
        me->rate_v = (real)r->rate_v;
        me->t_ns = r->hdr.acq_ns;
        /* A box whose head has not moved: nothing says where in it the thing is. A quarter of its size is one sigma. */
        me->at_rest = (r->flags & VT_F_AT_REST) != 0;
        me->mid = (r->flags & VT_F_POINT_MID) != 0;
        const real qw = me->at_rest ? 0.25L * (real)r->w : (real)r->pt_sx, qh = me->at_rest ? 0.25L * (real)r->h : (real)r->pt_sy;
        me->var_u = qw * qw;
        me->var_v = qh * qh;
        /* A new selection is a new thing: nothing known of the last one applies. */
        if (r->state == VT_ST_NEW) {
            f->kf.valid = 0;
            f->kf_unproven = 0;
            f->lock.run = 0;
        }
    }
    const uint64_t now = r->hdr.acq_ns;
    const real dt = f->last_ns && now > f->last_ns ? (real)(now - f->last_ns) * 1e-9L : 0.0L;
    if (now > f->last_ns) {
        if (f->kf.valid && dt > 0.0L)
            kf_predict(&f->kf, dt, f->q_accel);
        f->last_ns = now;
    }
    const struct fuse_obs *a = &f->obs[0], *b = &f->obs[1];
    if (camera == 1 && seen && a->valid && f->have_calib) {
        /* A's centre carried to B's time along its own measured rate. */
        const real gap = now >= a->t_ns ? (real)(now - a->t_ns) * 1e-9L : -((real)(a->t_ns - now) * 1e-9L);
        if (fabsl(gap) <= f->max_age_s) {
            const real ua = a->u + a->rate_u * gap, va = a->v + a->rate_v * gap;
            real p[3], rp, deg;
            if (fuse_triangulate(f, ua, va, b->u, b->v, p, &rp, &deg)) {
                real white[9], bias[9], cov[9];
                meas_cov(f, ua, va, b->u, b->v, p, white, bias);
                for (int i = 0; i < 9; i++) {
                    cov[i] = white[i] + bias[i];
                    f->bias[i] = bias[i];
                }
                for (int i = 0; i < 3; i++)
                    r->meas[i] = (double)p[i];
                for (int i = 0; i < 9; i++)
                    r->meas_cov[i] = (double)cov[i];
                r->reproj_rms = (double)rp;
                r->parallax_deg = (double)deg;
                r->pair_gap_ms = (double)(gap * 1000.0L);
                r->pair_ua = (float)ua;
                r->pair_va = (float)va;
                const int unproven = a->at_rest || b->at_rest;
                if (unproven)
                    r->flags |= VT_F_UNPROVEN;
                /* The first measure of the thing itself replaces whatever the filter learned before it. */
                if (!unproven && f->kf_unproven) {
                    f->kf.valid = 0;
                    f->kf_unproven = 0;
                }
                /* Rays from two boxes need not meet: their width already says so. */
                const real loose = a->mid || b->mid ? 3.0L * sqrtl(a->var_u + a->var_v + b->var_u + b->var_v) : 0.0L;
                if (!unproven && rp > f->max_reproj + loose) {
                    r->flags |= VT_F_3D_REJECTED;
                } else if (!f->kf.valid || unproven) {
                    /* An unproven measure only says itself again: the same bias a hundred times over is not a hundred measurements. */
                    kf_init(&f->kf, p, unproven ? cov : white, f->v0_sigma);
                    f->kf_unproven = unproven;
                    f->lock.run = 0;
                    r->flags |= VT_F_3D_MEASURED;
                    f->last_meas_ns = now;
                } else {
                    const real nis = kf_innovation(&f->kf, p, white);
                    r->nis = (double)nis;
                    if (nis <= KF_GATE_NIS) {
                        f->lock.run = 0;
                        kf_update(&f->kf, p, white);
                        r->flags |= VT_F_3D_MEASURED;
                        f->last_meas_ns = now;
                    } else {
                        if (kf_refused(&f->lock)) {
                            kf_init(&f->kf, p, white, f->v0_sigma);
                            f->lock.run = 0;
                            r->flags |= VT_F_3D_MEASURED;
                            f->last_meas_ns = now;
                        } else {
                            r->flags |= VT_F_3D_REJECTED;
                        }
                    }
                }
            }
        }
    }
    /* A long way out on prediction alone is not a metric estimate any more. Half a second here: a real rig's rate is not known in advance, and neither is its noise. */
    /* The cameras share no shutter: a frame may be older than the last measurement, whose age is then none, not 2^64 ns. */
    const real age = now > f->last_meas_ns ? (real)(now - f->last_meas_ns) * 1e-9L : 0.0L;
    if (f->kf.valid && age > 0.5L)
        f->kf.valid = 0;
    if (f->kf.valid) {
        r->age_s = (double)age;
        for (int i = 0; i < 3; i++) {
            r->pos[i] = (double)f->kf.x[i];
            r->vel[i] = (double)f->kf.x[i + 3];
            r->vel_var[i] = (double)f->kf.P[i + 3][i + 3];
            for (int k = 0; k < 3; k++)
                r->pos_cov[3 * i + k] = (double)(f->kf.P[i][k] + (f->kf_unproven ? 0.0L : f->bias[3 * i + k]));
        }
        r->flags |= VT_F_METRIC;
        if (f->kf_unproven)
            r->flags |= VT_F_UNPROVEN;
    }
}

void fuse_no_view(struct fuse *f, int camera) { f->obs[camera & 1].valid = 0; }
