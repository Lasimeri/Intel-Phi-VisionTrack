/* vt_stitch.c: camera D joined to the L-R geometry, on the desktop host.
 *   vt_stitch LR.txt POINTS.txt TRIPLES.txt --fd F --hub l|r --fh F --prior W --out OUT.txt [--report]
 * LR.txt: the geometry so far, camera L the world, X_r = R X_l + T, metres. POINTS.txt: "xd yd xh yh label": camera D against the hub camera, whichever shares more with it. --fh: the hub's focal length as a
 * prior. TRIPLES.txt: "xl yl xr yr xd yd label": points seen by all three: their distance from the hub by way of L + R, in metres, against their distance by way of D, in the solve's unit, is what scales it.
 * Hub R: X_r = S X_d + s t, so X_d = S' Rlr X_l + S' (Tlr - s t). Hub L: X_l = S X_d + s t, so X_d = S' X_l - s S' t. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calib.h"

#define MAX_PTS 512

static struct calib_pt pts[MAX_PTS];
static char labels[MAX_PTS][32];

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* The point nearest both rays, returned in camera B's frame (X_b = R X_a + T). Normalized image points. */
static void triangulate_b(const double *R, const double *T, double ax, double ay, double bx, double by, double *Pb)
{
    const double da[3] = {R[0] * ax + R[1] * ay + R[2], R[3] * ax + R[4] * ay + R[5], R[6] * ax + R[7] * ay + R[8]};
    const double db[3] = {bx, by, 1.0};
    /* Ray a from T along da, ray b from the origin along db. */
    const double aa = da[0] * da[0] + da[1] * da[1] + da[2] * da[2], bb = db[0] * db[0] + db[1] * db[1] + 1.0, ab = da[0] * db[0] + da[1] * db[1] + da[2];
    const double at = da[0] * T[0] + da[1] * T[1] + da[2] * T[2], bt = db[0] * T[0] + db[1] * T[1] + T[2];
    const double den = aa * bb - ab * ab;
    const double s = den > 1e-12 ? (ab * bt - bb * at) / den : 0.0, u = den > 1e-12 ? (aa * bt - ab * at) / den : 0.0;
    for (int i = 0; i < 3; i++)
        Pb[i] = 0.5 * (T[i] + s * da[i] + u * db[i]);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: vt_stitch LR.txt DR_POINTS.txt TRIPLES.txt --fd F --fb F --prior W --out OUT.txt [--report]\n");
        return 2;
    }
    struct vt_calib lr;
    if (calib_read(argv[1], &lr)) {
        fprintf(stderr, "vt_stitch: %s unreadable\n", argv[1]);
        return 2;
    }
    double fd = 0.0, fh = 0.0, prior = 8.0;
    const char *out = NULL;
    int report = 0, hub_l = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--fd") && i + 1 < argc) fd = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fh") && i + 1 < argc) fh = atof(argv[++i]);
        else if (!strcmp(argv[i], "--hub") && i + 1 < argc) hub_l = argv[++i][0] == 'l';
        else if (!strcmp(argv[i], "--prior") && i + 1 < argc) prior = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--report")) report = 1;
        else {
            fprintf(stderr, "vt_stitch: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    FILE *f = fopen(argv[2], "r");
    if (!f) {
        perror(argv[2]);
        return 2;
    }
    struct calib_problem p;
    memset(&p, 0, sizeof p);
    p.pts = pts;
    p.wa = 1280;
    p.ha = 720;
    p.wb = 1920;
    p.hb = 1080;
    p.fa_prior = fd;
    p.fb_prior = fh > 0.0 ? fh : (hub_l ? lr.fa : lr.fb);
    p.prior_weight = prior;
    char line[256];
    while (p.n < MAX_PTS && fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        struct calib_pt q;
        char lab[32] = "";
        if (sscanf(line, "%lf %lf %lf %lf %31s", &q.xa, &q.ya, &q.xb, &q.yb, lab) >= 4) {
            pts[p.n] = q;
            memcpy(labels[p.n], lab, sizeof lab);
            p.n++;
        }
    }
    fclose(f);
    struct calib_solution s;
    if (calib_solve(&p, &s)) {
        fprintf(stderr, "vt_stitch: no D-R solution from %d points\n", p.n);
        return 1;
    }
    const double th = acos(((s.R[0] + s.R[4] + s.R[8]) - 1.0) * 0.5) * 180.0 / M_PI;
    printf("D-%c: %d points, within 3 px %d, rms %.2f px, median %.2f; fd %.1f fhub %.1f; rotation %.1f deg; baseline direction in hub %.3f %.3f %.3f\n", hub_l ? 'L' : 'R', p.n, s.inliers, s.rms_px, s.median_px, s.fa, s.fb, th, s.t[0], s.t[1], s.t[2]);
    /* The scale: each triple's place in R's frame by L, in metres, against by D, in units. */
    f = fopen(argv[3], "r");
    if (!f) {
        perror(argv[3]);
        return 2;
    }
    double ratios[MAX_PTS];
    int nr = 0;
    while (nr < MAX_PTS && fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        double xl, yl, xr, yr, xd, yd;
        char lab[32] = "";
        if (sscanf(line, "%lf %lf %lf %lf %lf %lf %31s", &xl, &yl, &xr, &yr, &xd, &yd, lab) < 6)
            continue;
        double Pr[3], Pd[3];
        triangulate_b(lr.R, lr.T, (xl - lr.cxa) / lr.fa, (yl - lr.cya) / lr.fa, (xr - lr.cxb) / lr.fb, (yr - lr.cyb) / lr.fb, Pr);
        double Ph[3];
        if (hub_l) {
            /* The same point in L's frame: X_l = R' (X_r - T). */
            const double q[3] = {Pr[0] - lr.T[0], Pr[1] - lr.T[1], Pr[2] - lr.T[2]};
            for (int i = 0; i < 3; i++)
                Ph[i] = lr.R[i] * q[0] + lr.R[3 + i] * q[1] + lr.R[6 + i] * q[2];
            triangulate_b(s.R, s.t, (xd - 639.5) / s.fa, (yd - 359.5) / s.fa, (xl - 959.5) / s.fb, (yl - 539.5) / s.fb, Pd);
        } else {
            memcpy(Ph, Pr, sizeof Ph);
            triangulate_b(s.R, s.t, (xd - 639.5) / s.fa, (yd - 359.5) / s.fa, (xr - 959.5) / s.fb, (yr - 539.5) / s.fb, Pd);
        }
        const double nl = sqrt(Ph[0] * Ph[0] + Ph[1] * Ph[1] + Ph[2] * Ph[2]), nd = sqrt(Pd[0] * Pd[0] + Pd[1] * Pd[1] + Pd[2] * Pd[2]);
        ratios[nr++] = nl / nd;
        printf("  %-14s from the hub by L + R %.3f m, by D %.3f units: ratio %.3f\n", lab, nl, nd, nl / nd);
    }
    fclose(f);
    if (!nr) {
        fprintf(stderr, "vt_stitch: no triples to scale by\n");
        return 1;
    }
    qsort(ratios, (size_t)nr, sizeof ratios[0], cmp_d);
    const double scale = ratios[nr / 2];
    printf("scale: D to hub %.3f m, spread of ratios %.3f .. %.3f\n", scale, ratios[0], ratios[nr - 1]);
    if (report) {
        for (int i = 0; i < p.n; i++) {
            double P[3], ea, eb;
            calib_point(&p, &s, i, P, &ea, &eb);
            printf("%-16s errD %6.2f errR %6.2f   X %7.3f Y %7.3f Z %7.3f m in D's frame\n", labels[i], ea, eb, P[0] * scale, P[1] * scale, P[2] * scale);
        }
    }
    /* Into L's world. */
    struct vt_calib o = lr;
    o.valid_d = 1;
    o.fd = s.fa;
    o.cxd = 639.5;
    o.cyd = 359.5;
    double tn[3];
    for (int r = 0; r < 3; r++)
        tn[r] = (hub_l ? 0.0 : lr.T[r]) - scale * s.t[r];
    static const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double *to_hub = hub_l ? identity : lr.R;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            o.Rd[3 * r + c] = s.R[r] * to_hub[c] + s.R[3 + r] * to_hub[3 + c] + s.R[6 + r] * to_hub[6 + c];
        o.Td[r] = s.R[r] * tn[0] + s.R[3 + r] * tn[1] + s.R[6 + r] * tn[2];
    }
    if (out) {
        char note[200];
        snprintf(note, sizeof note, "L-R as solved; D-%c %d pts rms %.2f px, %.3f m by %d shared points", hub_l ? 'L' : 'R', p.n, s.rms_px, scale, nr);
        if (calib_write(out, &o, note)) {
            perror(out);
            return 1;
        }
        printf("wrote %s\n", out);
    }
    return 0;
}
