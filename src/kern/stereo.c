/* stereo.c: see stereo.h. The geometry is plain C in double; the row correlation exists in both arithmetics. */
#include "stereo.h"

#include <math.h>

#include "vx.h"

/* Correlation from the three sums, in long double on both machines. */
static inline long double ncc_of(float st, float sj, float sjj, long double norm_t, long double n)
{
    const long double var = (long double)sjj - (long double)sj * (long double)sj / n;
    if (var <= 0 || norm_t <= 0)
        return 0;
    return (long double)st / sqrtl(norm_t * var);
}

static void note_peak(struct stereo_scan *o, int dmin, const long double *curve, int count)
{
    int bi = -1;
    long double bv = -2;
    for (int i = 0; i < count; i++)
        if (curve[i] > bv) {
            bv = curve[i];
            bi = i;
        }
    long double sv = -1;
    for (int i = 0; i < count; i++)
        if ((i < bi - 2 || i > bi + 2) && curve[i] > sv)
            sv = curve[i];
    o->disparity = bi < 0 ? -1.0f : (float)(dmin + bi);
    o->ncc = bi < 0 ? 0.0f : (float)bv;
    o->second = (float)sv;
}

#define MAX_CANDIDATES 512

void stereo_scan_vx(const struct vt_plane *left, const struct vt_plane *right, float x, float y, int dmin, int dmax, struct stereo_scan *out)
{
    const int ix = (int)x, iy = (int)y;
    const int x0 = ix - STEREO_W / 2, y0 = iy - STEREO_H / 2;
    float t[STEREO_H][STEREO_W] VX_ALIGNED;
    vxf sum = vx_zero();
    for (int r = 0; r < STEREO_H; r++) {
        const vxf v = vx_loadu(plane_row(left, y0 + r) + x0);
        vx_store(t[r], v);
        sum = vx_add(sum, v);
    }
    const long double nl = (long double)(STEREO_W * STEREO_H);
    const float mean = (float)((long double)vx_hsum(sum) / nl);
    const vxf vm = vx_bcast(&mean);
    vxf stt = vx_zero();
    for (int r = 0; r < STEREO_H; r++) {
        const vxf z = vx_sub(vx_load(t[r]), vm);
        vx_store(t[r], z);
        stt = vx_fma(z, z, stt);
    }
    const long double norm_t = (long double)vx_hsum(stt);
    long double curve[MAX_CANDIDATES];
    int count = 0;
    for (int d = dmin; d <= dmax && count < MAX_CANDIDATES; d++, count++) {
        const int xr = x0 - d;
        if (xr < 1 || xr + STEREO_W >= right->w - 1) {
            curve[count] = -1;
            continue;
        }
        vxf st = vx_zero(), sj = vx_zero(), sjj = vx_zero();
        for (int r = 0; r < STEREO_H; r++) {
            const vxf j = vx_loadu(plane_row(right, y0 + r) + xr);
            st = vx_fma(j, vx_load(t[r]), st);
            sj = vx_add(sj, j);
            sjj = vx_fma(j, j, sjj);
        }
        curve[count] = ncc_of(vx_hsum(st), vx_hsum(sj), vx_hsum(sjj), norm_t, nl);
    }
    note_peak(out, dmin, curve, count);
}

void stereo_scan_ref(const struct vt_plane *left, const struct vt_plane *right, float x, float y, int dmin, int dmax, struct stereo_scan *out)
{
    const int ix = (int)x, iy = (int)y;
    const int x0 = ix - STEREO_W / 2, y0 = iy - STEREO_H / 2;
    float t[STEREO_H][STEREO_W];
    float sum = 0.0f;
    for (int r = 0; r < STEREO_H; r++) {
        const float *row = plane_row(left, y0 + r) + x0;
        for (int k = 0; k < STEREO_W; k++) {
            t[r][k] = row[k];
            sum += row[k];
        }
    }
    const long double nl = (long double)(STEREO_W * STEREO_H);
    const float mean = (float)((long double)sum / nl);
    float stt = 0.0f;
    for (int r = 0; r < STEREO_H; r++)
        for (int k = 0; k < STEREO_W; k++) {
            t[r][k] -= mean;
            stt += t[r][k] * t[r][k];
        }
    const long double norm_t = stt;
    long double curve[MAX_CANDIDATES];
    int count = 0;
    for (int d = dmin; d <= dmax && count < MAX_CANDIDATES; d++, count++) {
        const int xr = x0 - d;
        if (xr < 1 || xr + STEREO_W >= right->w - 1) {
            curve[count] = -1;
            continue;
        }
        float st = 0.0f, sj = 0.0f, sjj = 0.0f;
        for (int r = 0; r < STEREO_H; r++) {
            const float *row = plane_row(right, y0 + r) + xr;
            for (int k = 0; k < STEREO_W; k++) {
                st += row[k] * t[r][k];
                sj += row[k];
                sjj += row[k] * row[k];
            }
        }
        curve[count] = ncc_of(st, sj, sjj, norm_t, nl);
    }
    note_peak(out, dmin, curve, count);
}

int stereo_triangulate(const struct stereo_geom *g, long double xl, long double yl, long double xr, long double yr, long double *p)
{
    const long double d = xl - xr;
    if (d <= 0.0L)
        return 0;
    const long double z = g->fx * g->baseline / d;
    p[0] = (xl - g->cx) * z / g->fx;
    p[1] = (0.5L * (yl + yr) - g->cy) * z / g->fy;
    p[2] = z;
    return 1;
}

long double stereo_reprojection(const struct stereo_geom *g, const long double *p, long double xl, long double yl, long double xr, long double yr)
{
    const long double ul = g->fx * p[0] / p[2] + g->cx, vl = g->fy * p[1] / p[2] + g->cy;
    const long double ur = g->fx * (p[0] - g->baseline) / p[2] + g->cx;
    const long double el = ul - xl, fl = vl - yl, er = ur - xr, fr = vl - yr;
    return sqrtl((el * el + fl * fl + er * er + fr * fr) * 0.5L);
}

void stereo_covariance(const struct stereo_geom *g, const long double *p, long double disparity, long double var_x, long double var_y, long double var_d, long double *cov33)
{
    const long double zf = p[2] / g->fx, zfy = p[2] / g->fy;
    /* The Jacobian of (X, Y, Z) with respect to (x, y, d): X and Y scale with Z, and Z is proportional to 1 / d. */
    const long double j[3][3] = {
        {zf, 0.0L, -p[0] / disparity},
        {0.0L, zfy, -p[1] / disparity},
        {0.0L, 0.0L, -p[2] / disparity},
    };
    const long double v[3] = {var_x, var_y, var_d};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            long double s = 0.0L;
            for (int k = 0; k < 3; k++)
                s += j[r][k] * v[k] * j[c][k];
            cov33[3 * r + c] = s;
        }
}
