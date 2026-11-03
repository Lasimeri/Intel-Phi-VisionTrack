/* vt_attitude.c: the attitude kernel on saved exposures, on the desktop host, the lane model, one thread.
 *   vt_attitude CALIB REF_PREFIX CUR_PREFIX [--out NEW.txt]       PREFIX-{l,r,d}.pgm: how far each camera has turned, and the geometry carried through it: R' = Ob R Oa', T' = Ob T
 *   vt_attitude --test CALIB PREFIX CAM TILT PAN ROLL             a real exposure of camera l, r or d turned by a known rotation in degrees, and what the kernel recovers
 * Tilt, pan and roll are the rotation vector's x, y and z, in degrees, the same as a star fix's. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "attitude.h"
#include "calib.h"
#include "img.h"
#include "stars.h"

#define LEVEL 2 /* the level the depth map works at: live, these come from the card's own pyramid */

static void *heap(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static void one_thread(void *ctx, att_fn fn, void *arg)
{
    (void)ctx;
    fn(arg, 0, 1);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* An 8-bit image down to the depth map's level. img: w x h bytes, 16 B aligned rows of stride w rounded up to 16. */
static struct vt_plane to_level(const uint8_t *img, int w, int h, int stride)
{
    struct vt_plane p = plane_at(heap(NULL, plane_bytes(w, h)), w, h);
    img_u8_to_q6_vx(&p, img, stride, 0, h);
    img_pad_rows(&p, 0, h);
    img_pad_tb(&p);
    for (int l = 0; l < LEVEL; l++) {
        const int nw = (p.w + 1) / 2, nh = (p.h + 1) / 2;
        struct vt_plane q = plane_at(heap(NULL, plane_bytes(nw, nh)), nw, nh);
        img_reduce_vx(&q, &p, 0, nh);
        img_pad_rows(&q, 0, nh);
        img_pad_tb(&q);
        p = q;
    }
    return p;
}

/* 8 bits, as a camera delivers them: an integrated frame's 16 only carry its sum's precision. */
static uint8_t *read_u8(const char *path, int *w, int *h, int *stride)
{
    uint16_t *g = star_read_pgm(path, w, h);
    if (!g)
        return NULL;
    *stride = (*w + 15) & ~15;
    uint8_t *o = aligned_alloc(64, (((size_t)*stride * (size_t)*h) + 63) & ~(size_t)63);
    if (!o)
        return NULL;
    for (int y = 0; y < *h; y++)
        for (int x = 0; x < *w; x++)
            o[(size_t)y * (size_t)*stride + (size_t)x] = (uint8_t)(g[(size_t)y * (size_t)*w + (size_t)x] >> 8);
    free(g);
    return o;
}

static void rotation_vector(const double *q, double *deg)
{
    const double s = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    const double angle = 2.0 * atan2(s, q[0]), k = s > 1e-12 ? angle / s * 180.0 / M_PI : 360.0 / M_PI;
    for (int i = 0; i < 3; i++)
        deg[i] = q[i + 1] * k;
}

static void focal_of(const struct vt_calib *c, char cam, double *f, double *cx, double *cy)
{
    *f = cam == 'l' ? c->fa : cam == 'r' ? c->fb : c->fd;
    *cx = cam == 'l' ? c->cxa : cam == 'r' ? c->cxb : c->cxd;
    *cy = cam == 'l' ? c->cya : cam == 'r' ? c->cyb : c->cyd;
}

static struct att *make(int w, int h, double f, double cx, double cy)
{
    const struct att_cfg cfg = {w, h, f / (1 << LEVEL), cx / (1 << LEVEL), cy / (1 << LEVEL), 1.0f * VT_Q6, 1};
    return att_create(&cfg, heap, NULL);
}

static int run_test(int argc, char **argv)
{
    if (argc < 8)
        return 2;
    struct vt_calib c;
    if (calib_read(argv[2], &c))
        return 2;
    const char cam = argv[4][0];
    char path[512];
    snprintf(path, sizeof path, "%s-%c.pgm", argv[3], cam);
    int w, h, stride;
    uint8_t *src = read_u8(path, &w, &h, &stride);
    if (!src)
        return 2;
    double f, cx, cy;
    focal_of(&c, cam, &f, &cx, &cy);
    /* A second exposure of the same pose to turn, instead of the reference itself: independent noise, and whatever moved between them. */
    uint8_t *turned_src = src;
    if (argc > 8) {
        int w2, h2, st2;
        snprintf(path, sizeof path, "%s-%c.pgm", argv[8], cam);
        turned_src = read_u8(path, &w2, &h2, &st2);
        if (!turned_src || w2 != w || h2 != h)
            return 2;
    }
    const double put[3] = {atof(argv[5]), atof(argv[6]), atof(argv[7])};
    double O[9];
    star_rotation(put[0] * M_PI / 180.0, put[1] * M_PI / 180.0, put[2] * M_PI / 180.0, O);
    /* The picture a camera turned by O would see: each new pixel's bearing taken back to the old pose, O' b, bilinear; what was never seen is black. */
    uint8_t *dst = aligned_alloc(64, (((size_t)stride * (size_t)h) + 63) & ~(size_t)63);
    if (!dst)
        return 2;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const double b[3] = {((double)x - cx) / f, ((double)y - cy) / f, 1.0};
            const double X = O[0] * b[0] + O[3] * b[1] + O[6] * b[2], Y = O[1] * b[0] + O[4] * b[1] + O[7] * b[2], Z = O[2] * b[0] + O[5] * b[1] + O[8] * b[2];
            const double u = f * X / Z + cx, v = f * Y / Z + cy;
            uint8_t out = 0;
            if (Z > 0.0 && u >= 0.0 && v >= 0.0 && u < (double)(w - 1) && v < (double)(h - 1)) {
                const int iu = (int)u, iv = (int)v;
                const double tu = u - iu, tv = v - iv;
                const uint8_t *s = turned_src + (size_t)iv * (size_t)stride + (size_t)iu;
                const double top = s[0] + tu * ((double)s[1] - s[0]), bot = s[stride] + tu * ((double)s[stride + 1] - s[stride]);
                out = (uint8_t)(top + tv * (bot - top) + 0.5);
            }
            dst[(size_t)y * (size_t)stride + (size_t)x] = out;
        }
    const struct vt_plane ref = to_level(src, w, h, stride), cur = to_level(dst, w, h, stride);
    struct att *a = make(ref.w, ref.h, f, cx, cy);
    if (!a)
        return 2;
    att_set_reference(a, &ref, one_thread, NULL);
    struct att_fix x;
    att_measure(a, &cur, 1, one_thread, NULL, &x);
    double got[3];
    rotation_vector(x.q, got);
    const double miss = sqrt((got[0] - put[0]) * (got[0] - put[0]) + (got[1] - put[1]) * (got[1] - put[1]) + (got[2] - put[2]) * (got[2] - put[2]));
    printf("%c put %+7.2f %+7.2f %+7.2f got %+8.3f %+8.3f %+8.3f miss %.3f deg%s | corr %.3f from %.3f gain %+.3f, kept %.2f, in view %.2f, peak %+d %+d z %.0f, %d steps\n", cam, put[0], put[1], put[2], got[0], got[1], got[2], miss, x.held ? " HELD" : "", x.corr, x.corr_identity, x.corr - x.corr_identity, x.used, x.view, x.peak_x, x.peak_y, x.peak_z, x.steps);
    return 0;
}

/* A bilinear sample of an 8-bit image, or -1 outside it. */
static double sample(const uint8_t *g, int w, int h, int stride, double u, double v)
{
    if (!(u >= 0.0 && v >= 0.0 && u < (double)(w - 1) && v < (double)(h - 1)))
        return -1.0;
    const int iu = (int)u, iv = (int)v;
    const double tu = u - iu, tv = v - iv;
    const uint8_t *s = g + (size_t)iv * (size_t)stride + (size_t)iu;
    const double top = s[0] + tu * ((double)s[1] - s[0]), bot = s[stride] + tu * ((double)s[stride + 1] - s[stride]);
    return top + tv * (bot - top);
}

#define PATCH 20 /* 41 x 41 pixels at full size */
#define REACH 48

/* Landmarks carried from one camera's old picture to its new one: the old patch, turned by the fit, slid over the new picture, normalized correlation with a parabola at the peak. POINTS: "xa ya xb yb label"; cam r
 * reads the second pair and keeps the first, since L has not moved. What cannot be found is left out and said. */
static int run_transfer(int argc, char **argv)
{
    if (argc < 9)
        return 2;
    struct vt_calib c;
    if (calib_read(argv[2], &c))
        return 2;
    char path[512];
    int w, h, st, w2, h2, st2;
    snprintf(path, sizeof path, "%s-r.pgm", argv[3]);
    uint8_t *old = read_u8(path, &w, &h, &st);
    snprintf(path, sizeof path, "%s-r.pgm", argv[4]);
    uint8_t *cur = read_u8(path, &w2, &h2, &st2);
    FILE *in = fopen(argv[5], "r"), *out = fopen(argv[6], "w");
    if (!old || !cur || !in || !out)
        return 2;
    double O[9];
    star_rotation(atof(argv[7]) * M_PI / 180.0, atof(argv[8]) * M_PI / 180.0, (argc > 9 ? atof(argv[9]) : 0.0) * M_PI / 180.0, O);
    const double f = c.fb, cx = c.cxb, cy = c.cyb;
    const double floor_ncc = argc > 10 ? atof(argv[10]) : 0.5;
    fprintf(out, "# L x y, R x y, label. Carried from the old R picture to the new by correlation.\n");
    char line[256];
    int kept = 0, seen = 0;
    while (fgets(line, sizeof line, in)) {
        double xa, ya, xb, yb;
        char label[64];
        if (line[0] == '#' || sscanf(line, "%lf %lf %lf %lf %63s", &xa, &ya, &xb, &yb, label) != 5)
            continue;
        seen++;
        static double t[(2 * PATCH + 1) * (2 * PATCH + 1)];
        double sum = 0.0, sq = 0.0;
        int n = 0;
        for (int dy = -PATCH; dy <= PATCH; dy++)
            for (int dx = -PATCH; dx <= PATCH; dx++) {
                const double s = sample(old, w, h, st, xb + dx, yb + dy);
                t[n++] = s;
                if (s < 0.0) {
                    n = -1;
                    break;
                }
                sum += s;
                sq += s * s;
            }
        const double count = (2.0 * PATCH + 1.0) * (2.0 * PATCH + 1.0);
        const double spread = sqrt(fmax(sq - sum * sum / count, 0.0));
        if (n < 0 || spread < 1.0) {
            printf("%-14s no template\n", label);
            continue;
        }
        const double mean = sum / count;
        static double ncc[2 * REACH + 1][2 * REACH + 1];
        double best = -2.0, second = -2.0;
        int bx = 0, by = 0;
        /* Where the fit says the point went, then the search around it, the patch turned with the camera. */
        for (int sy = -REACH; sy <= REACH; sy++)
            for (int sx = -REACH; sx <= REACH; sx++) {
                double cs = 0.0, css = 0.0, cross = 0.0;
                int i = 0, ok = 1;
                for (int dy = -PATCH; ok && dy <= PATCH; dy++)
                    for (int dx = -PATCH; dx <= PATCH; dx++, i++) {
                        const double nx = (xb + dx - cx) / f, ny = (yb + dy - cy) / f;
                        const double X = O[0] * nx + O[1] * ny + O[2], Y = O[3] * nx + O[4] * ny + O[5], Z = O[6] * nx + O[7] * ny + O[8];
                        const double p = sample(cur, w, h, st2, f * X / Z + cx + sx, f * Y / Z + cy + sy);
                        if (p < 0.0) {
                            ok = 0;
                            break;
                        }
                        cs += p;
                        css += p * p;
                        cross += p * (t[i] - mean);
                    }
                const double sp = ok ? sqrt(fmax(css - cs * cs / count, 0.0)) : 0.0;
                const double v = sp > 1.0 ? cross / (sp * spread) : -2.0;
                ncc[sy + REACH][sx + REACH] = v;
                if (v > best) {
                    best = v;
                    bx = sx;
                    by = sy;
                }
            }
        /* The best answer well away from the peak: how much the peak stands out. */
        for (int sy = -REACH; sy <= REACH; sy++)
            for (int sx = -REACH; sx <= REACH; sx++)
                if ((abs(sx - bx) > 8 || abs(sy - by) > 8) && ncc[sy + REACH][sx + REACH] > second)
                    second = ncc[sy + REACH][sx + REACH];
        double fx = bx, fy = by;
        if (abs(bx) < REACH && abs(by) < REACH) {
            const double *m = &ncc[by + REACH][bx + REACH];
            const double l = m[-1], r = m[1], u = ncc[by + REACH - 1][bx + REACH], d = ncc[by + REACH + 1][bx + REACH];
            const double dxx = l - 2.0 * best + r, dyy = u - 2.0 * best + d;
            if (dxx < 0.0)
                fx += 0.5 * (l - r) / dxx;
            if (dyy < 0.0)
                fy += 0.5 * (u - d) / dyy;
        }
        const double nx = (xb - cx) / f, ny = (yb - cy) / f;
        const double X = O[0] * nx + O[1] * ny + O[2], Y = O[3] * nx + O[4] * ny + O[5], Z = O[6] * nx + O[7] * ny + O[8];
        const double px = f * X / Z + cx + fx, py = f * Y / Z + cy + fy;
        const int good = best >= floor_ncc && best - second >= 0.05 && abs(bx) < REACH && abs(by) < REACH;
        printf("%-14s old %7.1f %7.1f new %7.2f %7.2f, off the fit %+6.1f %+6.1f, ncc %.3f, next %.3f %s\n", label, xb, yb, px, py, fx, fy, best, second, good ? "" : "REFUSED");
        if (good) {
            fprintf(out, "%.2f %.2f %.2f %.2f %s\n", xa, ya, px, py, label);
            kept++;
        }
    }
    fclose(in);
    fclose(out);
    printf("%d of %d carried: %s\n", kept, seen, argv[6]);
    return 0;
}

/* The closest point of two rays, in L's frame: L at the origin, R at -R' T. miss: half their distance apart. */
static void triangulate(const struct vt_calib *c, double xa, double ya, double xb, double yb, double *P, double *miss)
{
    const double da[3] = {(xa - c->cxa) / c->fa, (ya - c->cya) / c->fa, 1.0};
    const double r[3] = {(xb - c->cxb) / c->fb, (yb - c->cyb) / c->fb, 1.0};
    const double *R = c->R, *T = c->T;
    const double db[3] = {R[0] * r[0] + R[3] * r[1] + R[6] * r[2], R[1] * r[0] + R[4] * r[1] + R[7] * r[2], R[2] * r[0] + R[5] * r[1] + R[8] * r[2]};
    const double C[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
    const double aa = da[0] * da[0] + da[1] * da[1] + da[2] * da[2], bb = db[0] * db[0] + db[1] * db[1] + db[2] * db[2], ab = da[0] * db[0] + da[1] * db[1] + da[2] * db[2];
    const double ao = da[0] * C[0] + da[1] * C[1] + da[2] * C[2], bo = db[0] * C[0] + db[1] * C[1] + db[2] * C[2];
    const double det = aa * bb - ab * ab, sa = (ao * bb - bo * ab) / det, sb = (ao * ab - bo * aa) / det;
    double e = 0.0;
    for (int i = 0; i < 3; i++) {
        const double p = sa * da[i], q = C[i] + sb * db[i];
        P[i] = 0.5 * (p + q);
        e += (p - q) * (p - q);
    }
    *miss = 0.5 * sqrt(e);
}

static void bearing(double x, double y, double f, double cx, double cy, double *b)
{
    const double nx = (x - cx) / f, ny = (y - cy) / f, n = sqrt(nx * nx + ny * ny + 1.0);
    b[0] = nx / n;
    b[1] = ny / n;
    b[2] = 1.0 / n;
}

/* Camera R's turn from landmarks it sees in both pictures: the rotation that best carries the old bearings onto the new, Gauss-Newton on the rotation, residuals printed point by point. A residual that grows with
 * nearness says the camera's centre moved. The geometry written holds L and D as they were: R' = O R, T' = O T. */
static int run_solve_r(int argc, char **argv)
{
    if (argc < 6)
        return 2;
    struct vt_calib c;
    if (calib_read(argv[2], &c))
        return 2;
    enum { MAX_POINTS = 64 };
    double r[MAX_POINTS][3], b[MAX_POINTS][3], ox[MAX_POINTS], oy[MAX_POINTS], nx[MAX_POINTS], ny[MAX_POINTS];
    char names[MAX_POINTS][64];
    int n = 0;
    FILE *fn = fopen(argv[4], "r");
    if (!fn)
        return 2;
    char line[256];
    while (fgets(line, sizeof line, fn) && n < MAX_POINTS) {
        double xa, ya, xb, yb;
        if (line[0] == '#' || sscanf(line, "%lf %lf %lf %lf %63s", &xa, &ya, &xb, &yb, names[n]) != 5)
            continue;
        /* Find the same label in the old list. */
        FILE *fo = fopen(argv[3], "r");
        if (!fo)
            return 2;
        char ol[256];
        int found = 0;
        while (fgets(ol, sizeof ol, fo)) {
            double oa, ob, oc, od;
            char label[64];
            if (ol[0] != '#' && sscanf(ol, "%lf %lf %lf %lf %63s", &oa, &ob, &oc, &od, label) == 5 && !strcmp(label, names[n])) {
                ox[n] = oc;
                oy[n] = od;
                found = 1;
            }
        }
        fclose(fo);
        if (!found)
            continue;
        nx[n] = xb;
        ny[n] = yb;
        bearing(ox[n], oy[n], c.fb, c.cxb, c.cyb, r[n]);
        bearing(xb, yb, c.fb, c.cxb, c.cyb, b[n]);
        n++;
    }
    fclose(fn);
    if (n < 3) {
        fprintf(stderr, "vt_attitude: %d landmarks: too few\n", n);
        return 2;
    }
    double O[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int it = 0; it < 16; it++) {
        double A[6] = {0, 0, 0, 0, 0, 0}, g[3] = {0, 0, 0};
        for (int i = 0; i < n; i++) {
            const double p[3] = {O[0] * r[i][0] + O[1] * r[i][1] + O[2] * r[i][2], O[3] * r[i][0] + O[4] * r[i][1] + O[5] * r[i][2], O[6] * r[i][0] + O[7] * r[i][1] + O[8] * r[i][2]};
            const double e[3] = {b[i][0] - p[0], b[i][1] - p[1], b[i][2] - p[2]};
            /* p x e pulls p toward b; (I - p p') is the matrix of a small turn's effect. */
            g[0] += p[1] * e[2] - p[2] * e[1];
            g[1] += p[2] * e[0] - p[0] * e[2];
            g[2] += p[0] * e[1] - p[1] * e[0];
            A[0] += 1.0 - p[0] * p[0];
            A[1] += -p[0] * p[1];
            A[2] += -p[0] * p[2];
            A[3] += 1.0 - p[1] * p[1];
            A[4] += -p[1] * p[2];
            A[5] += 1.0 - p[2] * p[2];
        }
        const double c00 = A[3] * A[5] - A[4] * A[4], c01 = A[2] * A[4] - A[1] * A[5], c02 = A[1] * A[4] - A[2] * A[3];
        const double det = A[0] * c00 + A[1] * c01 + A[2] * c02;
        const double c11 = A[0] * A[5] - A[2] * A[2], c12 = A[1] * A[2] - A[0] * A[4], c22 = A[0] * A[3] - A[1] * A[1];
        const double d[3] = {(c00 * g[0] + c01 * g[1] + c02 * g[2]) / det, (c01 * g[0] + c11 * g[1] + c12 * g[2]) / det, (c02 * g[0] + c12 * g[1] + c22 * g[2]) / det};
        double D[9], N[9];
        star_rotation(d[0], d[1], d[2], D);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                N[3 * i + j] = D[3 * i] * O[j] + D[3 * i + 1] * O[3 + j] + D[3 * i + 2] * O[6 + j];
        memcpy(O, N, sizeof N);
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double p[3] = {O[0] * r[i][0] + O[1] * r[i][1] + O[2] * r[i][2], O[3] * r[i][0] + O[4] * r[i][1] + O[5] * r[i][2], O[6] * r[i][0] + O[7] * r[i][1] + O[8] * r[i][2]};
        const double u = c.fb * p[0] / p[2] + c.cxb, v = c.fb * p[1] / p[2] + c.cyb;
        const double du = nx[i] - u, dv = ny[i] - v;
        sum += du * du + dv * dv;
        printf("%-14s read %7.2f %7.2f, the rotation says %7.2f %7.2f, off %+6.2f %+6.2f px\n", names[i], nx[i], ny[i], u, v, du, dv);
    }
    const double angle = acos(fmin(1.0, fmax(-1.0, 0.5 * (O[0] + O[4] + O[8] - 1.0))));
    const double s = sin(angle) > 1e-12 ? angle / (2.0 * sin(angle)) * 180.0 / M_PI : 90.0 / M_PI;
    printf("R turned tilt %+.3f pan %+.3f roll %+.3f deg, rms %.2f px over %d landmarks\n", (O[7] - O[5]) * s, (O[2] - O[6]) * s, (O[3] - O[1]) * s, sqrt(sum / n), n);
    static const double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    struct vt_calib o = c;
    star_correct_pair(c.R, c.T, I, O, o.R, o.T);
    /* The check that does not care who is in the room: L and the wall have not moved, so each landmark, triangulated through the new geometry from its new pixel, must land where the old geometry had it. */
    FILE *fl = fopen(argv[4], "r");
    if (fl) {
        double total = 0.0;
        int m = 0;
        while (fgets(line, sizeof line, fl)) {
            double xa, ya, xb, yb;
            char label[64];
            if (line[0] == '#' || sscanf(line, "%lf %lf %lf %lf %63s", &xa, &ya, &xb, &yb, label) != 5)
                continue;
            for (int i = 0; i < n; i++) {
                if (strcmp(names[i], label))
                    continue;
                double was[3], now[3], miss0, miss1;
                triangulate(&c, xa, ya, ox[i], oy[i], was, &miss0);
                triangulate(&o, xa, ya, xb, yb, now, &miss1);
                const double dx = now[0] - was[0], dy = now[1] - was[1], dz = now[2] - was[2], moved = sqrt(dx * dx + dy * dy + dz * dz);
                printf("%-14s at z %.3f m: moved %6.1f mm (%+6.1f %+6.1f %+6.1f), rays miss %.1f mm, were %.1f\n", label, was[2], moved * 1000.0, dx * 1000.0, dy * 1000.0, dz * 1000.0, miss1 * 1000.0, miss0 * 1000.0);
                total += moved;
                m++;
            }
        }
        fclose(fl);
        printf("landmarks moved %.1f mm on average, over %d\n", m ? total / m * 1000.0 : 0.0, m);
    }
    if (calib_write(argv[5], &o, "R re-anchored by landmarks carried by correlation; L, D and the lenses as solved"))
        return 2;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "--test"))
        return run_test(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "--transfer"))
        return run_transfer(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "--solve-r"))
        return run_solve_r(argc, argv);
    if (argc < 4) {
        fprintf(stderr, "usage: vt_attitude CALIB REF_PREFIX CUR_PREFIX [--out NEW.txt] | --test CALIB PREFIX CAM TILT PAN ROLL | --transfer CALIB OLD_PREFIX NEW_PREFIX POINTS OUT TILT PAN [ROLL [FLOOR]] | --solve-r CALIB OLD_POINTS NEW_POINTS OUT\n");
        return 2;
    }
    struct vt_calib c;
    if (calib_read(argv[1], &c) || !c.valid || !c.valid_d) {
        fprintf(stderr, "vt_attitude: %s holds no geometry for all three cameras\n", argv[1]);
        return 2;
    }
    const char *out = argc >= 6 && !strcmp(argv[4], "--out") ? argv[5] : NULL;
    static const char CAMS[3] = {'l', 'r', 'd'};
    double O[3][9];
    for (int k = 0; k < 3; k++) {
        char path[512];
        int w, h, sw, cw, ch, cs;
        snprintf(path, sizeof path, "%s-%c.pgm", argv[2], CAMS[k]);
        uint8_t *ri = read_u8(path, &w, &h, &sw);
        snprintf(path, sizeof path, "%s-%c.pgm", argv[3], CAMS[k]);
        uint8_t *ci = read_u8(path, &cw, &ch, &cs);
        if (!ri || !ci || w != cw || h != ch) {
            fprintf(stderr, "vt_attitude: camera %c: missing or different sizes\n", CAMS[k]);
            return 2;
        }
        double f, cx, cy;
        focal_of(&c, CAMS[k], &f, &cx, &cy);
        const struct vt_plane ref = to_level(ri, w, h, sw), cur = to_level(ci, w, h, sw);
        struct att *a = make(ref.w, ref.h, f, cx, cy);
        if (!a)
            return 2;
        att_set_reference(a, &ref, one_thread, NULL);
        struct att_fix x;
        const double t0 = now_s();
        att_measure(a, &cur, 1, one_thread, NULL, &x);
        const double t1 = now_s();
        double deg[3];
        rotation_vector(x.q, deg);
        printf("%c turned tilt %+8.3f pan %+8.3f roll %+8.3f deg%s | corr %.3f from %.3f, kept %.2f, in view %.2f, peak %+d %+d z %.0f, %d steps, host %.2f s\n", CAMS[k], deg[0], deg[1], deg[2], x.held ? " HELD: no better than no turn" : "", x.corr, x.corr_identity, x.used, x.view, x.peak_x, x.peak_y, x.peak_z, x.steps, t1 - t0);
        memcpy(O[k], x.O, sizeof x.O);
    }
    if (out) {
        struct vt_calib n = c;
        star_correct_pair(c.R, c.T, O[0], O[1], n.R, n.T);
        star_correct_pair(c.Rd, c.Td, O[0], O[2], n.Rd, n.Td);
        if (calib_write(out, &n, "carried by vt_attitude from the solved geometry"))
            return 2;
        printf("carried geometry: %s\n", out);
    }
    return 0;
}
