/* kalman.c: see kalman.h. */
#include "kalman.h"

#include <string.h>

void kf_init(struct kf *k, const long double *z, const long double *r33, long double v0_sigma)
{
    memset(k, 0, sizeof *k);
    for (int i = 0; i < 3; i++) {
        k->x[i] = z[i];
        for (int j = 0; j < 3; j++)
            k->P[i][j] = r33[3 * i + j];
        k->P[i + 3][i + 3] = v0_sigma * v0_sigma;
    }
    k->valid = 1;
}

void kf_predict(struct kf *k, long double dt, long double q_accel)
{
    const long double q = q_accel * q_accel;
    const long double d2 = dt * dt, d3 = d2 * dt;
    for (int i = 0; i < 3; i++)
        k->x[i] += dt * k->x[i + 3];
    /* P = F P F' + Q with F = [I dt I; 0 I], written out: the blocks need only additions. */
    long double Pn[6][6];
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++) {
            long double v = k->P[r][c];
            if (r < 3)
                v += dt * k->P[r + 3][c];
            if (c < 3)
                v += dt * k->P[r][c + 3];
            if (r < 3 && c < 3)
                v += d2 * k->P[r + 3][c + 3];
            Pn[r][c] = v;
        }
    for (int i = 0; i < 3; i++) {
        Pn[i][i] += q * d3 / 3.0L;
        Pn[i][i + 3] += q * d2 / 2.0L;
        Pn[i + 3][i] += q * d2 / 2.0L;
        Pn[i + 3][i + 3] += q * dt;
    }
    memcpy(k->P, Pn, sizeof Pn);
}

/* The inverse of a 3 x 3 by cofactors; 0 if singular. */
static int inv33(const long double *a, long double *o)
{
    const long double c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6];
    const long double det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    if (det == 0.0L)
        return 0;
    const long double i = 1.0L / det;
    o[0] = c00 * i;
    o[1] = (a[2] * a[7] - a[1] * a[8]) * i;
    o[2] = (a[1] * a[5] - a[2] * a[4]) * i;
    o[3] = c01 * i;
    o[4] = (a[0] * a[8] - a[2] * a[6]) * i;
    o[5] = (a[2] * a[3] - a[0] * a[5]) * i;
    o[6] = c02 * i;
    o[7] = (a[1] * a[6] - a[0] * a[7]) * i;
    o[8] = (a[0] * a[4] - a[1] * a[3]) * i;
    return 1;
}

static void innov(const struct kf *k, const long double *z, const long double *r, long double *y, long double *S)
{
    for (int i = 0; i < 3; i++) {
        y[i] = z[i] - k->x[i];
        for (int j = 0; j < 3; j++)
            S[3 * i + j] = k->P[i][j] + r[3 * i + j];
    }
}

long double kf_innovation(const struct kf *k, const long double *z, const long double *r33)
{
    long double y[3], S[9], Si[9];
    innov(k, z, r33, y, S);
    if (!inv33(S, Si))
        return 1e30L;
    long double v = 0.0L;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            v += y[i] * Si[3 * i + j] * y[j];
    return v;
}

void kf_update(struct kf *k, const long double *z, const long double *r33)
{
    long double y[3], S[9], Si[9], K[6][3];
    innov(k, z, r33, y, S);
    if (!inv33(S, Si))
        return;
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 3; c++) {
            long double v = 0.0L;
            for (int m = 0; m < 3; m++)
                v += k->P[r][m] * Si[3 * m + c];
            K[r][c] = v;
        }
    for (int r = 0; r < 6; r++)
        k->x[r] += K[r][0] * y[0] + K[r][1] * y[1] + K[r][2] * y[2];
    /* Joseph: (I - K H) P (I - K H)' + K R K'. */
    long double A[6][6], B[6][6];
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            A[r][c] = (r == c ? 1.0L : 0.0L) - (c < 3 ? K[r][c] : 0.0L);
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++) {
            long double v = 0.0L;
            for (int m = 0; m < 6; m++)
                v += A[r][m] * k->P[m][c];
            B[r][c] = v;
        }
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++) {
            long double v = 0.0L;
            for (int m = 0; m < 6; m++)
                v += B[r][m] * A[c][m];
            for (int m = 0; m < 3; m++)
                for (int n = 0; n < 3; n++)
                    v += K[r][m] * r33[3 * m + n] * K[c][n];
            k->P[r][c] = v;
        }
}
