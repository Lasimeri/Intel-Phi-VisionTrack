/* vt_stars.c: the star catalogue and its tests, on the desktop host.
 *   vt_stars list FRAME.pgm [K]                     every light in a frame
 *   vt_stars build OUT.txt A.pgm B.pgm [K]          a catalogue: the lights repeating between two exposures
 *   vt_stars fix CAT.txt FRAME.pgm F                a fix of one frame against a catalogue
 *   vt_stars test CAT.txt FRAME.pgm F TILT PAN ROLL the same frame as a camera sees it after a KNOWN rotation, degrees: what comes back, against what went in */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stars.h"

static void print_fix(const struct star_fix *x)
{
    printf("lights %d known, %d seen, %d used%s | shift %+8.3f %+8.3f px | rotation deg tilt %+7.4f pan %+7.4f roll %+7.4f%s | residual %.3f px | GDOP %.4f deg per px\n", x->known, x->seen, x->used, x->lost_in_space ? ", LOST IN SPACE" : "", x->dx, x->dy, x->wx * 180.0 / M_PI, x->wy * 180.0 / M_PI, x->wz * 180.0 / M_PI, x->roll_known ? "" : " (roll held)", x->rms_px, x->gdop);
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "list")) {
        int w, h;
        uint16_t *g = star_read_pgm(argv[2], &w, &h);
        if (!g)
            return 2;
        static struct star s[STAR_MAX * 4];
        const int n = star_detect(g, w, h, argc >= 4 ? atof(argv[3]) : 12.0, s, STAR_MAX * 4);
        printf("# %s: %d lights\n", argv[2], n);
        for (int i = 0; i < n; i++)
            printf("%8.2f %8.2f  flux %8.0f  area %3d\n", s[i].x, s[i].y, s[i].flux, s[i].area);
        return 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "build")) {
        int w, h, w2, h2;
        uint16_t *a = star_read_pgm(argv[3], &w, &h), *b = star_read_pgm(argv[4], &w2, &h2);
        if (!a || !b || w != w2 || h != h2)
            return 2;
        const double k = argc >= 6 ? atof(argv[5]) : 6.0;
        static struct star sa[STAR_MAX * 4], sb[STAR_MAX * 4];
        const int na = star_detect(a, w, h, k, sa, STAR_MAX * 4), nb = star_detect(b, w, h, k, sb, STAR_MAX * 4);
        static struct star_cat c;
        const int n = star_catalogue(sa, na, sb, nb, 0.5, 3, 40.0, &c);
        printf("%s: %d and %d lights, %d repeat within 0.5 px, area >= 3, no neighbour within 40 px\n", argv[2], na, nb, n);
        for (int i = 0; i < n; i++)
            printf("%8.2f %8.2f  flux %8.0f  area %3d\n", c.s[i].x, c.s[i].y, c.s[i].flux, c.s[i].area);
        return star_cat_write(argv[2], &c) ? 1 : 0;
    }
    if (argc >= 5 && !strcmp(argv[1], "fix")) {
        static struct star_cat c;
        if (star_cat_read(argv[2], &c))
            return 2;
        int w, h;
        uint16_t *g = star_read_pgm(argv[3], &w, &h);
        if (!g)
            return 2;
        struct star_fix x;
        star_track(&c, g, w, h, atof(argv[4]), 0.0, 0.0, 1, &x);
        print_fix(&x);
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "diffcal")) {
        /* The correction's algebra against a known answer: fixed 3D points through a rig, both cameras then turned by known rotations. The rays as the cameras now see them, through the corrected relation, must meet at the same points. */
        const double base_th = 0.6;
        const double R0[9] = {cos(base_th), 0, sin(base_th), 0, 1, 0, -sin(base_th), 0, cos(base_th)};
        const double T0[3] = {-1.1, 0.05, 0.3};
        const double rots[][6] = {{0.01, -0.02, 0.0, 0.0, 0.03, 0.01}, {0.05, 0.03, 0.02, -0.04, 0.02, 0.03}, {0.2, -0.15, 0.1, 0.1, 0.2, -0.05}};
        for (size_t t = 0; t < sizeof rots / sizeof rots[0]; t++) {
            double Oa[9], Ob[9], R1[9], T1[3];
            star_rotation(rots[t][0], rots[t][1], rots[t][2], Oa);
            star_rotation(rots[t][3], rots[t][4], rots[t][5], Ob);
            star_correct_pair(R0, T0, Oa, Ob, R1, T1);
            double worst_old = 0.0, worst_new = 0.0;
            for (int p = 0; p < 200; p++) {
                const double X[3] = {-1.0 + 0.01 * p, -0.5 + 0.005 * p, 1.0 + 0.02 * p};
                /* Where the point is in each camera's frame before the bump, and as each now sees it. */
                const double Xb[3] = {R0[0] * X[0] + R0[1] * X[1] + R0[2] * X[2] + T0[0], R0[3] * X[0] + R0[4] * X[1] + R0[5] * X[2] + T0[1], R0[6] * X[0] + R0[7] * X[1] + R0[8] * X[2] + T0[2]};
                const double Xa2[3] = {Oa[0] * X[0] + Oa[1] * X[1] + Oa[2] * X[2], Oa[3] * X[0] + Oa[4] * X[1] + Oa[5] * X[2], Oa[6] * X[0] + Oa[7] * X[1] + Oa[8] * X[2]};
                const double Xb2[3] = {Ob[0] * Xb[0] + Ob[1] * Xb[1] + Ob[2] * Xb[2], Ob[3] * Xb[0] + Ob[4] * Xb[1] + Ob[5] * Xb[2], Ob[6] * Xb[0] + Ob[7] * Xb[1] + Ob[8] * Xb[2]};
                /* The relation must carry the point seen now by A to the point seen now by B. */
                for (int i = 0; i < 3; i++) {
                    const double viaN = R1[3 * i] * Xa2[0] + R1[3 * i + 1] * Xa2[1] + R1[3 * i + 2] * Xa2[2] + T1[i];
                    const double viaO = R0[3 * i] * Xa2[0] + R0[3 * i + 1] * Xa2[1] + R0[3 * i + 2] * Xa2[2] + T0[i];
                    const double en = fabs(viaN - Xb2[i]), eo = fabs(viaO - Xb2[i]);
                    worst_new = en > worst_new ? en : worst_new;
                    worst_old = eo > worst_old ? eo : worst_old;
                }
            }
            printf("bump %zu: worst miss of a 3D point, uncorrected %.4f m, corrected %.3e m\n", t, worst_old, worst_new);
        }
        return 0;
    }
    if (argc >= 8 && !strcmp(argv[1], "test")) {
        static struct star_cat c;
        if (star_cat_read(argv[2], &c))
            return 2;
        int w, h;
        uint16_t *g = star_read_pgm(argv[3], &w, &h);
        if (!g)
            return 2;
        const double f = atof(argv[4]);
        const double om[3] = {atof(argv[5]) * M_PI / 180.0, atof(argv[6]) * M_PI / 180.0, atof(argv[7]) * M_PI / 180.0};
        uint16_t *o = calloc((size_t)w * (size_t)h, sizeof(uint16_t));
        if (!o)
            return 2;
        /* The known warp: what a camera sees after a TRUE rotation Om: a bearing b becomes R b. A flat image shift is not that off the axis. For each output pixel, its bearing is carried back by R', bilinear. */
        const double th = sqrt(om[0] * om[0] + om[1] * om[1] + om[2] * om[2]);
        const double A = th < 1e-12 ? 1.0 : sin(th) / th, B = th < 1e-12 ? 0.5 : (1.0 - cos(th)) / (th * th);
        const double ox = om[0], oy = om[1], oz = om[2];
        const double R[9] = {1 - B * (oy * oy + oz * oz), -A * oz + B * ox * oy, A * oy + B * ox * oz, A * oz + B * ox * oy, 1 - B * (ox * ox + oz * oz), -A * ox + B * oy * oz, -A * oy + B * ox * oz, A * ox + B * oy * oz, 1 - B * (ox * ox + oy * oy)};
        const double cx = 0.5 * (w - 1), cy = 0.5 * (h - 1);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const double u = (x - cx) / f, v = (y - cy) / f;
                const double bx = R[0] * u + R[3] * v + R[6], by = R[1] * u + R[4] * v + R[7], bz = R[2] * u + R[5] * v + R[8];
                if (bz <= 0.0)
                    continue;
                const double sx = f * bx / bz + cx, sy = f * by / bz + cy;
                const int ix = (int)floor(sx), iy = (int)floor(sy);
                if (ix < 0 || iy < 0 || ix + 1 >= w || iy + 1 >= h)
                    continue;
                const double fx = sx - ix, fy = sy - iy;
                const uint16_t *p = g + (size_t)iy * (size_t)w + (size_t)ix;
                o[(size_t)y * (size_t)w + (size_t)x] = (uint16_t)lrint((1 - fy) * ((1 - fx) * p[0] + fx * p[1]) + fy * ((1 - fx) * p[w] + fx * p[w + 1]));
            }
        struct star_fix x;
        star_track(&c, o, w, h, atof(argv[4]), 0.0, 0.0, 1, &x);
        printf("put in deg: tilt %+6.3f pan %+6.3f roll %+6.3f | ", om[0] * 180.0 / M_PI, om[1] * 180.0 / M_PI, om[2] * 180.0 / M_PI);
        print_fix(&x);
        free(o);
        return 0;
    }
    fprintf(stderr, "usage: vt_stars list FRAME [K] | build OUT A B [K] | fix CAT FRAME F | test CAT FRAME F TILT PAN ROLL | diffcal\n");
    return 2;
}
