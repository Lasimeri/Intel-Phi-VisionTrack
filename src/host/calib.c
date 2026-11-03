/* calib.c: see calib.h. An eight-point start at each candidate pair of focal lengths, then Levenberg-Marquardt over rotation, baseline direction and both focal lengths, with robust weights. */
#include "calib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Eigen-decomposition of a symmetric matrix by Jacobi rotations: values ascending in w, vectors in the columns of V. */
static void jacobi(double *A, int n, double *w, double *V)
{
    for (int i = 0; i < n * n; i++)
        V[i] = i % (n + 1) == 0 ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                off += A[p * n + q] * A[p * n + q];
        if (off < 1e-30)
            break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                const double apq = A[p * n + q];
                if (fabs(apq) < 1e-300)
                    continue;
                const double th = (A[q * n + q] - A[p * n + p]) / (2.0 * apq);
                const double t = (th >= 0 ? 1.0 : -1.0) / (fabs(th) + sqrt(th * th + 1.0));
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; k++) {
                    const double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = c * akp - s * akq;
                    A[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    const double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = c * apk - s * aqk;
                    A[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    const double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < n; i++)
        w[i] = A[i * n + i];
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (w[j] < w[i]) {
                const double t = w[i];
                w[i] = w[j];
                w[j] = t;
                for (int k = 0; k < n; k++) {
                    const double v = V[k * n + i];
                    V[k * n + i] = V[k * n + j];
                    V[k * n + j] = v;
                }
            }
}

static void mat3_mul(const double *a, const double *b, double *o)
{
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
}

static void rodrigues(const double *r, double *R)
{
    const double th = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    const double A = th < 1e-9 ? 1.0 : sin(th) / th, B = th < 1e-9 ? 0.5 : (1.0 - cos(th)) / (th * th);
    const double x = r[0], y = r[1], z = r[2];
    R[0] = 1.0 - B * (y * y + z * z);
    R[1] = -A * z + B * x * y;
    R[2] = A * y + B * x * z;
    R[3] = A * z + B * x * y;
    R[4] = 1.0 - B * (x * x + z * z);
    R[5] = -A * x + B * y * z;
    R[6] = -A * y + B * x * z;
    R[7] = A * x + B * y * z;
    R[8] = 1.0 - B * (x * x + y * y);
}

/* The closest point of the two rays through normalized image points; in A's frame. Returns the parameters along each ray through *sa, *sb. */
static void triangulate(const double *R, const double *t, double ax, double ay, double bx, double by, double *P, double *sa, double *sb)
{
    const double d[3] = {R[0] * bx + R[3] * by + R[6], R[1] * bx + R[4] * by + R[7], R[2] * bx + R[5] * by + R[8]};
    const double C[3] = {-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])};
    const double a[3] = {ax, ay, 1.0};
    const double aa = a[0] * a[0] + a[1] * a[1] + 1.0, dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2], ad = a[0] * d[0] + a[1] * d[1] + d[2];
    const double ac = a[0] * C[0] + a[1] * C[1] + C[2], dc = d[0] * C[0] + d[1] * C[1] + d[2] * C[2];
    const double den = aa * dd - ad * ad;
    const double s = den > 1e-12 ? (ac * dd - dc * ad) / den : 0.0, u = den > 1e-12 ? (ac * ad - dc * aa) / den : 0.0;
    for (int i = 0; i < 3; i++)
        P[i] = 0.5 * (s * a[i] + u * d[i] + C[i]);
    *sa = s;
    *sb = u;
}

struct model {
    double R[9], t[3], fa, fb;
};

/* Reprojection residuals of one point in pixels: 4 values. A point behind a camera is pushed hard. */
static void residuals(const struct calib_problem *p, const struct model *m, int i, double *r)
{
    const struct calib_pt *q = &p->pts[i];
    const double cxa = 0.5 * (p->wa - 1), cya = 0.5 * (p->ha - 1), cxb = 0.5 * (p->wb - 1), cyb = 0.5 * (p->hb - 1);
    const double ax = (q->xa - cxa) / m->fa, ay = (q->ya - cya) / m->fa, bx = (q->xb - cxb) / m->fb, by = (q->yb - cyb) / m->fb;
    double P[3], sa, sb;
    triangulate(m->R, m->t, ax, ay, bx, by, P, &sa, &sb);
    const double *R = m->R, *t = m->t;
    const double Q[3] = {R[0] * P[0] + R[1] * P[1] + R[2] * P[2] + t[0], R[3] * P[0] + R[4] * P[1] + R[5] * P[2] + t[1], R[6] * P[0] + R[7] * P[1] + R[8] * P[2] + t[2]};
    if (sa <= 0.0 || sb <= 0.0 || P[2] <= 1e-6 || Q[2] <= 1e-6) {
        r[0] = r[1] = r[2] = r[3] = 200.0;
        return;
    }
    r[0] = m->fa * P[0] / P[2] + cxa - q->xa;
    r[1] = m->fa * P[1] / P[2] + cya - q->ya;
    r[2] = m->fb * Q[0] / Q[2] + cxb - q->xb;
    r[3] = m->fb * Q[1] / Q[2] + cyb - q->yb;
}

static double total_cost(const struct calib_problem *p, const struct model *m, double huber)
{
    double c = 0.0, r[4];
    for (int i = 0; i < p->n; i++) {
        residuals(p, m, i, r);
        const double e = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
        c += e <= huber ? e * e : huber * (2.0 * e - huber);
    }
    if (p->prior_weight > 0.0) {
        if (p->fa_prior > 0.0) {
            const double d = log(m->fa / p->fa_prior);
            c += p->prior_weight * p->n * d * d;
        }
        if (p->fb_prior > 0.0) {
            const double d = log(m->fb / p->fb_prior);
            c += p->prior_weight * p->n * d * d;
        }
    }
    return c;
}

/* A step of the 7 parameters: rotation, baseline direction by two tangent angles, log focal lengths. */
static void apply_step(const struct model *m0, const double *dx, struct model *m)
{
    double dR[9];
    rodrigues(dx, dR);
    mat3_mul(dR, m0->R, m->R);
    /* Two axes perpendicular to t. */
    const double *t = m0->t;
    double u[3] = {fabs(t[0]) < 0.9 ? 1.0 : 0.0, fabs(t[0]) < 0.9 ? 0.0 : 1.0, 0.0};
    const double dot = u[0] * t[0] + u[1] * t[1] + u[2] * t[2];
    for (int i = 0; i < 3; i++)
        u[i] -= dot * t[i];
    const double un = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (int i = 0; i < 3; i++)
        u[i] /= un;
    const double v[3] = {t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2], t[0] * u[1] - t[1] * u[0]};
    double nt[3];
    for (int i = 0; i < 3; i++)
        nt[i] = t[i] + dx[3] * u[i] + dx[4] * v[i];
    const double nn = sqrt(nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]);
    for (int i = 0; i < 3; i++)
        m->t[i] = nt[i] / nn;
    m->fa = m0->fa * exp(dx[5]);
    m->fb = m0->fb * exp(dx[6]);
}

/* Gaussian elimination with pivoting: A x = b. */
static int solve_linear(double *A, double *b, int n)
{
    for (int i = 0; i < n; i++) {
        int piv = i;
        for (int r = i + 1; r < n; r++)
            if (fabs(A[r * n + i]) > fabs(A[piv * n + i]))
                piv = r;
        if (fabs(A[piv * n + i]) < 1e-15)
            return -1;
        if (piv != i) {
            for (int c = 0; c < n; c++) {
                const double t = A[i * n + c];
                A[i * n + c] = A[piv * n + c];
                A[piv * n + c] = t;
            }
            const double t = b[i];
            b[i] = b[piv];
            b[piv] = t;
        }
        for (int r = i + 1; r < n; r++) {
            const double f = A[r * n + i] / A[i * n + i];
            for (int c = i; c < n; c++)
                A[r * n + c] -= f * A[i * n + c];
            b[r] -= f * b[i];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        for (int c = i + 1; c < n; c++)
            b[i] -= A[i * n + c] * b[c];
        b[i] /= A[i * n + i];
    }
    return 0;
}

static double refine(const struct calib_problem *p, struct model *m, int free_focal, double huber, int iters)
{
    enum { NP = 7 };
    const int np = free_focal ? 7 : 5;
    double lambda = 1e-3, cost = total_cost(p, m, huber);
    const double h[NP] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5};
    for (int it = 0; it < iters; it++) {
        double JTJ[NP * NP], JTr[NP];
        memset(JTJ, 0, sizeof JTJ);
        memset(JTr, 0, sizeof JTr);
        for (int i = 0; i < p->n; i++) {
            double r0[4], J[4][NP];
            residuals(p, m, i, r0);
            const double e = sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2] + r0[3] * r0[3]);
            const double w = e <= huber ? 1.0 : huber / e;
            for (int k = 0; k < np; k++) {
                double dx[NP] = {0, 0, 0, 0, 0, 0, 0}, r1[4];
                dx[k] = h[k];
                struct model mk;
                apply_step(m, dx, &mk);
                residuals(p, &mk, i, r1);
                for (int c = 0; c < 4; c++)
                    J[c][k] = (r1[c] - r0[c]) / h[k];
            }
            for (int a = 0; a < np; a++) {
                for (int c = 0; c < 4; c++)
                    JTr[a] += w * J[c][a] * r0[c];
                for (int b = 0; b < np; b++)
                    for (int c = 0; c < 4; c++)
                        JTJ[a * NP + b] += w * J[c][a] * J[c][b];
            }
        }
        if (free_focal && p->prior_weight > 0.0) {
            if (p->fa_prior > 0.0) {
                JTJ[5 * NP + 5] += p->prior_weight * p->n;
                JTr[5] += p->prior_weight * p->n * log(m->fa / p->fa_prior);
            }
            if (p->fb_prior > 0.0) {
                JTJ[6 * NP + 6] += p->prior_weight * p->n;
                JTr[6] += p->prior_weight * p->n * log(m->fb / p->fb_prior);
            }
        }
        int improved = 0;
        for (int tries = 0; tries < 8 && !improved; tries++) {
            double A[NP * NP], b[NP];
            for (int a = 0; a < np; a++) {
                for (int c = 0; c < np; c++)
                    A[a * np + c] = JTJ[a * NP + c];
                A[a * np + a] += lambda * (JTJ[a * NP + a] + 1e-9);
                b[a] = -JTr[a];
            }
            if (solve_linear(A, b, np))
                break;
            double dx[NP] = {0, 0, 0, 0, 0, 0, 0};
            for (int a = 0; a < np; a++)
                dx[a] = b[a];
            struct model mn;
            apply_step(m, dx, &mn);
            const double c1 = total_cost(p, &mn, huber);
            if (c1 < cost) {
                const double gain = cost - c1;
                *m = mn;
                cost = c1;
                lambda = lambda * 0.3 > 1e-9 ? lambda * 0.3 : 1e-9;
                improved = 1;
                if (gain < 1e-10 * cost)
                    return cost;
            } else {
                lambda *= 4.0;
            }
        }
        if (!improved)
            break;
    }
    return cost;
}

/* The eight-point start: for these focal lengths, the essential matrix and which of its four poses puts the points in front of both cameras. */
static int start_pose(const struct calib_problem *p, struct model *m)
{
    const double cxa = 0.5 * (p->wa - 1), cya = 0.5 * (p->ha - 1), cxb = 0.5 * (p->wb - 1), cyb = 0.5 * (p->hb - 1);
    double AtA[81], w[9], V[81];
    memset(AtA, 0, sizeof AtA);
    for (int i = 0; i < p->n; i++) {
        const struct calib_pt *q = &p->pts[i];
        const double ax = (q->xa - cxa) / m->fa, ay = (q->ya - cya) / m->fa, bx = (q->xb - cxb) / m->fb, by = (q->yb - cyb) / m->fb;
        const double row[9] = {bx * ax, bx * ay, bx, by * ax, by * ay, by, ax, ay, 1.0};
        for (int a = 0; a < 9; a++)
            for (int b = 0; b < 9; b++)
                AtA[a * 9 + b] += row[a] * row[b];
    }
    jacobi(AtA, 9, w, V);
    double E[9];
    for (int i = 0; i < 9; i++)
        E[i] = V[i * 9];
    /* E = U diag(1, 1, 0) V'. From E' E: V. U from E V. */
    double EtE[9], ev[3], Vt[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            EtE[3 * r + c] = E[r] * E[c] + E[3 + r] * E[3 + c] + E[6 + r] * E[6 + c];
    jacobi(EtE, 3, ev, Vt);
    /* Columns of Vt ascending: the smallest is the null vector, then the two largest. */
    double v0[3], v1[3], v2[3];
    for (int i = 0; i < 3; i++) {
        v0[i] = Vt[3 * i + 2];
        v1[i] = Vt[3 * i + 1];
        v2[i] = Vt[3 * i];
    }
    double u0[3], u1[3], u2[3];
    for (int i = 0; i < 3; i++) {
        u0[i] = E[3 * i] * v0[0] + E[3 * i + 1] * v0[1] + E[3 * i + 2] * v0[2];
        u1[i] = E[3 * i] * v1[0] + E[3 * i + 1] * v1[1] + E[3 * i + 2] * v1[2];
    }
    const double n0 = sqrt(u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
    if (n0 < 1e-12)
        return -1;
    for (int i = 0; i < 3; i++)
        u0[i] /= n0;
    const double d01 = u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2];
    for (int i = 0; i < 3; i++)
        u1[i] -= d01 * u0[i];
    const double n1 = sqrt(u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
    if (n1 < 1e-12)
        return -1;
    for (int i = 0; i < 3; i++)
        u1[i] /= n1;
    u2[0] = u0[1] * u1[2] - u0[2] * u1[1];
    u2[1] = u0[2] * u1[0] - u0[0] * u1[2];
    u2[2] = u0[0] * u1[1] - u0[1] * u1[0];
    /* V must be a right-handed frame too. */
    const double det = v2[0] * (v0[1] * v1[2] - v0[2] * v1[1]) - v2[1] * (v0[0] * v1[2] - v0[2] * v1[0]) + v2[2] * (v0[0] * v1[1] - v0[1] * v1[0]);
    if (det < 0.0)
        for (int i = 0; i < 3; i++)
            v2[i] = -v2[i];
    const double U[9] = {u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2]};
    const double Vc[9] = {v0[0], v1[0], v2[0], v0[1], v1[1], v2[1], v0[2], v1[2], v2[2]};
    const double VT[9] = {Vc[0], Vc[3], Vc[6], Vc[1], Vc[4], Vc[7], Vc[2], Vc[5], Vc[8]};
    static const double W[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    static const double WT[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};
    double best = -1.0;
    struct model out = *m;
    for (int k = 0; k < 4; k++) {
        double t0[9], R[9];
        mat3_mul(U, (k & 1) ? WT : W, t0);
        mat3_mul(t0, VT, R);
        const double sgn = (k & 2) ? -1.0 : 1.0;
        const double t[3] = {sgn * u2[0], sgn * u2[1], sgn * u2[2]};
        int front = 0;
        for (int i = 0; i < p->n; i++) {
            const struct calib_pt *q = &p->pts[i];
            double P[3], sa, sb;
            triangulate(R, t, (q->xa - cxa) / m->fa, (q->ya - cya) / m->fa, (q->xb - cxb) / m->fb, (q->yb - cyb) / m->fb, P, &sa, &sb);
            front += sa > 0.0 && sb > 0.0;
        }
        if (front > best) {
            best = front;
            memcpy(out.R, R, sizeof R);
            memcpy(out.t, t, sizeof t);
        }
    }
    *m = out;
    return best > 0 ? 0 : -1;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int calib_solve(const struct calib_problem *p, struct calib_solution *s)
{
    if (p->n < 8)
        return -1;
    const double huber = 6.0;
    /* Focal lengths are searched on a coarse logarithmic grid, or held near a prior; a start at the wrong one still converges when it is not far off. */
    static const double grid[] = {0.5, 0.65, 0.85, 1.1, 1.4, 1.8, 2.4, 3.2, 4.2};
    const int ng = (int)(sizeof grid / sizeof grid[0]);
    struct model best;
    double best_cost = 1e300;
    for (int a = 0; a < (p->fa_prior > 0.0 ? 1 : ng); a++)
        for (int b = 0; b < (p->fb_prior > 0.0 ? 1 : ng); b++) {
            struct model m;
            m.fa = p->fa_prior > 0.0 ? p->fa_prior : grid[a] * p->wa;
            m.fb = p->fb_prior > 0.0 ? p->fb_prior : grid[b] * p->wb;
            if (start_pose(p, &m))
                continue;
            refine(p, &m, 0, huber, 15);
            const double c = refine(p, &m, 1, huber, 40);
            if (c < best_cost) {
                best_cost = c;
                best = m;
            }
        }
    if (best_cost >= 1e300)
        return -1;
    double *errs = malloc(sizeof(double) * (size_t)p->n);
    double sum = 0.0;
    int inl = 0;
    for (int i = 0; i < p->n; i++) {
        double r[4];
        residuals(p, &best, i, r);
        const double e2 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) * 0.5;
        sum += e2;
        errs[i] = sqrt(e2);
        inl += errs[i] <= 3.0;
    }
    qsort(errs, (size_t)p->n, sizeof(double), cmp_d);
    s->median_px = errs[p->n / 2];
    free(errs);
    s->rms_px = sqrt(sum / p->n);
    s->inliers = inl;
    s->fa = best.fa;
    s->fb = best.fb;
    memcpy(s->R, best.R, sizeof best.R);
    memcpy(s->t, best.t, sizeof best.t);
    return 0;
}

double calib_length(const struct calib_problem *p, const struct calib_solution *s, int i, int j)
{
    const double cxa = 0.5 * (p->wa - 1), cya = 0.5 * (p->ha - 1), cxb = 0.5 * (p->wb - 1), cyb = 0.5 * (p->hb - 1);
    double A[3], B[3], sa, sb;
    const struct calib_pt *q = &p->pts[i], *r = &p->pts[j];
    triangulate(s->R, s->t, (q->xa - cxa) / s->fa, (q->ya - cya) / s->fa, (q->xb - cxb) / s->fb, (q->yb - cyb) / s->fb, A, &sa, &sb);
    triangulate(s->R, s->t, (r->xa - cxa) / s->fa, (r->ya - cya) / s->fa, (r->xb - cxb) / s->fb, (r->yb - cyb) / s->fb, B, &sa, &sb);
    return sqrt((A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) + (A[2] - B[2]) * (A[2] - B[2]));
}

void calib_point(const struct calib_problem *p, const struct calib_solution *s, int i, double *P, double *err_a, double *err_b)
{
    const double cxa = 0.5 * (p->wa - 1), cya = 0.5 * (p->ha - 1), cxb = 0.5 * (p->wb - 1), cyb = 0.5 * (p->hb - 1);
    const struct calib_pt *q = &p->pts[i];
    double sa, sb;
    triangulate(s->R, s->t, (q->xa - cxa) / s->fa, (q->ya - cya) / s->fa, (q->xb - cxb) / s->fb, (q->yb - cyb) / s->fb, P, &sa, &sb);
    struct model m;
    memcpy(m.R, s->R, sizeof m.R);
    memcpy(m.t, s->t, sizeof m.t);
    m.fa = s->fa;
    m.fb = s->fb;
    double r[4];
    residuals(p, &m, i, r);
    *err_a = sqrt(r[0] * r[0] + r[1] * r[1]);
    *err_b = sqrt(r[2] * r[2] + r[3] * r[3]);
}

void calib_to_wire(const struct calib_problem *p, const struct calib_solution *s, double scale, struct vt_calib *o)
{
    memset(o, 0, sizeof *o);
    o->valid = 1;
    o->fa = s->fa;
    o->cxa = 0.5 * (p->wa - 1);
    o->cya = 0.5 * (p->ha - 1);
    o->fb = s->fb;
    o->cxb = 0.5 * (p->wb - 1);
    o->cyb = 0.5 * (p->hb - 1);
    memcpy(o->R, s->R, sizeof o->R);
    for (int i = 0; i < 3; i++)
        o->T[i] = s->t[i] * scale;
}

int calib_write(const char *path, const struct vt_calib *c, const char *note)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "# %s\n", note);
    fprintf(f, "%.9g %.9g %.9g %.9g %.9g\n%.9g %.9g %.9g %.9g %.9g\n", c->fa, c->cxa, c->cya, c->k1a, c->k2a, c->fb, c->cxb, c->cyb, c->k1b, c->k2b);
    for (int r = 0; r < 3; r++)
        fprintf(f, "%.12g %.12g %.12g\n", c->R[3 * r], c->R[3 * r + 1], c->R[3 * r + 2]);
    fprintf(f, "%.9g %.9g %.9g\n", c->T[0], c->T[1], c->T[2]);
    if (c->valid_d) {
        fprintf(f, "# camera D\n%.9g %.9g %.9g %.9g %.9g\n", c->fd, c->cxd, c->cyd, c->k1d, c->k2d);
        for (int r = 0; r < 3; r++)
            fprintf(f, "%.12g %.12g %.12g\n", c->Rd[3 * r], c->Rd[3 * r + 1], c->Rd[3 * r + 2]);
        fprintf(f, "%.9g %.9g %.9g\n", c->Td[0], c->Td[1], c->Td[2]);
    }
    fclose(f);
    return 0;
}

int calib_read(const char *path, struct vt_calib *c)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[256];
    double v[39];
    int n = 0;
    while (n < 39 && fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        char *p = line, *e;
        for (;;) {
            const double x = strtod(p, &e);
            if (e == p || n == 39)
                break;
            v[n++] = x;
            p = e;
        }
    }
    fclose(f);
    if (n != 22 && n != 39)
        return -1;
    memset(c, 0, sizeof *c);
    c->valid = 1;
    c->fa = v[0]; c->cxa = v[1]; c->cya = v[2]; c->k1a = v[3]; c->k2a = v[4];
    c->fb = v[5]; c->cxb = v[6]; c->cyb = v[7]; c->k1b = v[8]; c->k2b = v[9];
    memcpy(c->R, v + 10, sizeof c->R);
    memcpy(c->T, v + 19, sizeof c->T);
    if (n == 39) {
        c->valid_d = 1;
        c->fd = v[22]; c->cxd = v[23]; c->cyd = v[24]; c->k1d = v[25]; c->k2d = v[26];
        memcpy(c->Rd, v + 27, sizeof c->Rd);
        memcpy(c->Td, v + 36, sizeof c->Td);
    }
    return 0;
}
