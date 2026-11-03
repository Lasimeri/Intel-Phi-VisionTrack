/* vt_mapcheck.c: holds a live depth map against what is known without it. Each landmark read by eye in L and R is triangulated through the geometry, carried into D, and its depth there compared with what the
 * sweep found at that place. Two maps of a still room: how much the depth moves between them. All on the desktop host, from files.
 *   vt_mapcheck MAP.bin CALIB.txt TRIPLES.txt [MAP2.bin]
 * TRIPLES: "Lx Ly Rx Ry Dx Dy label", full-size pixels. Not independent of the geometry: the armrest tips and the LEDs helped place D. What it can show is whether the sweep and the triangulation agree. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calib.h"
#include "vt_proto.h"

static uint8_t a[VT_DEPTH_BYTES], b[VT_DEPTH_BYTES];

static int cmp_f(const void *p, const void *q)
{
    const float x = *(const float *)p, y = *(const float *)q;
    return x < y ? -1 : x > y;
}

/* The closest point of two rays, in L's frame: L at the origin, R at -R' T. err: half their distance apart. */
static void triangulate(const struct vt_calib *c, double xa, double ya, double xb, double yb, double *P, double *err)
{
    const double da[3] = {(xa - c->cxa) / c->fa, (ya - c->cya) / c->fa, 1.0};
    const double r[3] = {(xb - c->cxb) / c->fb, (yb - c->cyb) / c->fb, 1.0};
    const double *R = c->R, *T = c->T;
    const double db[3] = {R[0] * r[0] + R[3] * r[1] + R[6] * r[2], R[1] * r[0] + R[4] * r[1] + R[7] * r[2], R[2] * r[0] + R[5] * r[1] + R[8] * r[2]};
    const double O[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
    const double aa = da[0] * da[0] + da[1] * da[1] + da[2] * da[2], bb = db[0] * db[0] + db[1] * db[1] + db[2] * db[2], ab = da[0] * db[0] + da[1] * db[1] + da[2] * db[2];
    const double ao = da[0] * O[0] + da[1] * O[1] + da[2] * O[2], bo = db[0] * O[0] + db[1] * O[1] + db[2] * O[2];
    const double det = aa * bb - ab * ab;
    const double s = (ao * bb - bo * ab) / det, t = (ao * ab - bo * aa) / det;
    double e = 0.0;
    for (int i = 0; i < 3; i++) {
        const double p = s * da[i], q = O[i] + t * db[i];
        P[i] = 0.5 * (p + q);
        e += (p - q) * (p - q);
    }
    *err = 0.5 * sqrt(e);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: vt_mapcheck MAP.bin CALIB.txt TRIPLES.txt [MAP2.bin]\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(a, sizeof a, 1, f) != 1)
        return 2;
    fclose(f);
    struct vt_calib c;
    if (calib_read(argv[2], &c) || !c.valid || !c.valid_d) {
        fprintf(stderr, "vt_mapcheck: %s holds no geometry for all three cameras\n", argv[2]);
        return 2;
    }
    const struct vt_depth_out *h = (const struct vt_depth_out *)a;
    const float *z = vt_depth_z(a), *conf = vt_depth_conf(a);
    const int scale = 1 << VT_DEPTH_LEVEL, reach = 3;
    f = fopen(argv[3], "r");
    if (!f)
        return 2;
    char line[256];
    double arm[2][3];
    int have_arm[2] = {0, 0};
    printf("%-12s %8s %8s %8s %8s %6s %7s %14s\n", "landmark", "rays mm", "tri z_D", "map z", "map-tri", "known", "conf", "tri-eye px, D");
    while (fgets(line, sizeof line, f)) {
        double la, lb, ra, rb, da, db;
        char label[64];
        if (line[0] == '#' || sscanf(line, "%lf %lf %lf %lf %lf %lf %63s", &la, &lb, &ra, &rb, &da, &db, label) != 7)
            continue;
        double P[3], err;
        triangulate(&c, la, lb, ra, rb, P, &err);
        const double *R = c.Rd, *T = c.Td;
        const double Q[3] = {R[0] * P[0] + R[1] * P[1] + R[2] * P[2] + T[0], R[3] * P[0] + R[4] * P[1] + R[5] * P[2] + T[1], R[6] * P[0] + R[7] * P[1] + R[8] * P[2] + T[2]};
        /* Where the triangulated point falls in D, against where the eye read it. */
        const double px = c.fd * Q[0] / Q[2] + c.cxd, py = c.fd * Q[1] / Q[2] + c.cyd;
        const int mx = (int)lrint(da / scale), my = (int)lrint(db / scale);
        static float got[64];
        int n = 0;
        double cs = 0.0;
        for (int y = my - reach; y <= my + reach; y++)
            for (int x = mx - reach; x <= mx + reach; x++) {
                if (x < 0 || y < 0 || x >= (int)h->w || y >= (int)h->h)
                    continue;
                const float v = z[(size_t)y * h->stride + (size_t)x];
                if (v > 0.0f) {
                    got[n++] = v;
                    cs += (double)conf[(size_t)y * h->stride + (size_t)x];
                }
            }
        qsort(got, (size_t)n, sizeof(float), cmp_f);
        const double med = n ? (double)got[n / 2] : 0.0;
        printf("%-12s %8.1f %8.3f %8.3f %+8.3f %3d/49 %7.3f %+7.1f %+6.1f\n", label, err * 1000.0, Q[2], med, n ? med - Q[2] : 0.0, n, n ? cs / n : 0.0, px - da, py - db);
        if (!strcmp(label, "arm-A") || !strcmp(label, "arm-B")) {
            /* The tip in 3D as the map has it. */
            const int t = label[4] == 'B';
            if (n) {
                arm[t][0] = (da - c.cxd) / c.fd * med;
                arm[t][1] = (db - c.cyd) / c.fd * med;
                arm[t][2] = med;
                have_arm[t] = 1;
            }
        }
    }
    fclose(f);
    if (have_arm[0] && have_arm[1])
        printf("armrest, tip to tip, from the map's own depths: %.0f mm (from the L-R landmarks: 301 mm)\n", 1000.0 * sqrt((arm[0][0] - arm[1][0]) * (arm[0][0] - arm[1][0]) + (arm[0][1] - arm[1][1]) * (arm[0][1] - arm[1][1]) + (arm[0][2] - arm[1][2]) * (arm[0][2] - arm[1][2])));
    if (argc > 4) {
        f = fopen(argv[4], "rb");
        if (!f || fread(b, sizeof b, 1, f) != 1)
            return 2;
        fclose(f);
        const struct vt_depth_out *h2 = (const struct vt_depth_out *)b;
        const float *z2 = vt_depth_z(b), *conf2 = vt_depth_conf(b);
        static float diff[1 << 20];
        long both = 0, one = 0, near = 0;
        const double near_i = 1.0 / (double)h->z_near, far_i = 1.0 / (double)h->z_far, step = (near_i - far_i) / (double)(h->planes - 1);
        for (uint32_t y = 0; y < h->h; y++)
            for (uint32_t x = 0; x < h->w; x++) {
                const float p = z[y * h->stride + x], q = z2[y * h2->stride + x];
                const int cp = p > 0.0f && conf[y * h->stride + x] >= 0.2f, cq = q > 0.0f && conf2[y * h2->stride + x] >= 0.2f;
                one += cp != cq;
                if (cp && cq) {
                    diff[both++] = fabsf(p - q);
                    near += fabs(1.0 / (double)p - 1.0 / (double)q) <= step;
                }
            }
        qsort(diff, (size_t)both, sizeof(float), cmp_f);
        printf("map %llu against map %llu, confident in both: %ld pixels, in one only %ld; |dz| m p50 %.4f p90 %.4f p99 %.4f; within one plane %.1f %%\n", (unsigned long long)h->seq, (unsigned long long)h2->seq, both, one, both ? (double)diff[both / 2] : 0.0, both ? (double)diff[both * 9 / 10] : 0.0, both ? (double)diff[both * 99 / 100] : 0.0, both ? 100.0 * (double)near / (double)both : 0.0);
    }
    return 0;
}
