/* vt_calib.c: solves two cameras' relative geometry from correspondences, on the desktop host.
 *   vt_calib POINTS.txt --wa W --ha H --wb W --hb H [--fa F] [--fb F] [--prior W] [--scale I J METRES] [--out FILE] [--report]
 *     POINTS.txt: "xa ya xb yb [label]" per line, # comments. --scale: points I and J (from 0) are METRES apart; without it the baseline is reported in unit length, and no metric file is written.
 *   vt_calib --synthetic N NOISE_PX [--seed S]: a known rig, N points, noisy: what the method recovers, against the truth. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calib.h"
#include "vt_util.h"

#define MAX_PTS 20000

static struct calib_pt pts[MAX_PTS];
static char labels[MAX_PTS][32];

static double rot_angle_deg(const double *A, const double *B)
{
    double tr = 0.0;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            tr += A[3 * r + c] * B[3 * r + c];
    const double v = (tr - 1.0) * 0.5;
    return acos(v > 1.0 ? 1.0 : v < -1.0 ? -1.0 : v) * 180.0 / M_PI;
}

static double gauss(uint64_t *s)
{
    const double a = ((double)(vt_rand(s) >> 11) + 0.5) / 9007199254740992.0, b = ((double)(vt_rand(s) >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(a)) * cos(2.0 * M_PI * b);
}

static int synthetic(int n, double noise, uint64_t seed)
{
    /* Roughly this desk: camera B a metre to the side, turned 45 degrees toward what both see, 1 to 3 m away, B a wide webcam, A a longer lens. */
    const double th = 45.0 * M_PI / 180.0, ct = cos(th), st = sin(th);
    const double R[9] = {ct, 0.0, st, 0.0, 1.0, 0.0, -st, 0.0, ct};
    const double C[3] = {1.1, -0.1, 0.45};
    double T[3];
    for (int i = 0; i < 3; i++)
        T[i] = -(R[3 * i] * C[0] + R[3 * i + 1] * C[1] + R[3 * i + 2] * C[2]);
    const double fa = 1600.0, fb = 693.0;
    const struct calib_problem p0 = {.pts = pts, .n = n, .wa = 1920, .ha = 1080, .wb = 640, .hb = 480, .fa_prior = 0.0, .fb_prior = 693.0, .prior_weight = 0.5};
    uint64_t s = seed;
    int got = 0;
    double P0[3] = {0, 0, 0}, P1[3] = {0, 0, 0};
    while (got < n) {
        const double u = (double)(vt_rand(&s) >> 11) / 9007199254740992.0, v = (double)(vt_rand(&s) >> 11) / 9007199254740992.0, w = (double)(vt_rand(&s) >> 11) / 9007199254740992.0;
        const double Z = 1.0 + 2.0 * w, X = (u - 0.5) * 1.1 * Z, Y = (v - 0.5) * 0.6 * Z;
        const double Q[3] = {R[0] * X + R[1] * Y + R[2] * Z + T[0], R[3] * X + R[4] * Y + R[5] * Z + T[1], R[6] * X + R[7] * Y + R[8] * Z + T[2]};
        if (Q[2] < 0.3)
            continue;
        const double xa = fa * X / Z + 959.5, ya = fa * Y / Z + 539.5, xb = fb * Q[0] / Q[2] + 319.5, yb = fb * Q[1] / Q[2] + 239.5;
        if (xa < 0 || xa >= 1920 || ya < 0 || ya >= 1080 || xb < 0 || xb >= 640 || yb < 0 || yb >= 480)
            continue;
        if (got == 0) {
            P0[0] = X; P0[1] = Y; P0[2] = Z;
        } else if (got == 1) {
            P1[0] = X; P1[1] = Y; P1[2] = Z;
        }
        pts[got].xa = xa + noise * gauss(&s);
        pts[got].ya = ya + noise * gauss(&s);
        pts[got].xb = xb + noise * gauss(&s);
        pts[got].yb = yb + noise * gauss(&s);
        got++;
    }
    struct calib_solution sol;
    if (calib_solve(&p0, &sol)) {
        printf("synthetic n %d noise %.1f: no solution\n", n, noise);
        return 1;
    }
    const double tn = sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);
    const double dot = (sol.t[0] * T[0] + sol.t[1] * T[1] + sol.t[2] * T[2]) / tn;
    const double L = sqrt((P0[0] - P1[0]) * (P0[0] - P1[0]) + (P0[1] - P1[1]) * (P0[1] - P1[1]) + (P0[2] - P1[2]) * (P0[2] - P1[2]));
    const double unit = calib_length(&p0, &sol, 0, 1);
    printf("synthetic n %3d noise %.1f px seed %llu: rotation err %.2f deg, baseline dir err %.2f deg, fa %.0f (true 1600, %+.1f %%), fb %.0f (true 693), rms %.2f px, baseline from a known %.3f m length %.3f m (true %.3f, %+.1f %%)\n", n, noise, (unsigned long long)seed, rot_angle_deg(sol.R, R), acos(dot > 1 ? 1 : dot) * 180.0 / M_PI, sol.fa, (sol.fa / 1600.0 - 1.0) * 100.0, sol.fb, sol.rms_px, L, L / unit, tn, (L / unit / tn - 1.0) * 100.0);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "--synthetic")) {
        uint64_t seed = 1234567;
        if (argc >= 6 && !strcmp(argv[4], "--seed"))
            seed = strtoull(argv[5], NULL, 0);
        return synthetic(atoi(argv[2]), atof(argv[3]), seed);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: vt_calib POINTS.txt --wa W --ha H --wb W --hb H [--fa F] [--fb F] [--prior W] [--scale I J METRES] [--out FILE]\n");
        return 2;
    }
    struct calib_problem p;
    memset(&p, 0, sizeof p);
    p.pts = pts;
    p.prior_weight = 0.5;
    int si = -1, sj = -1, report = 0;
    double metres = 0.0;
    const char *out = NULL;
    for (int i = 2; i < argc; i++) {
#define OPT(s) (!strcmp(argv[i], s) && i + 1 < argc)
        if (OPT("--wa")) p.wa = atoi(argv[++i]);
        else if (OPT("--ha")) p.ha = atoi(argv[++i]);
        else if (OPT("--wb")) p.wb = atoi(argv[++i]);
        else if (OPT("--hb")) p.hb = atoi(argv[++i]);
        else if (OPT("--fa")) p.fa_prior = atof(argv[++i]);
        else if (OPT("--fb")) p.fb_prior = atof(argv[++i]);
        else if (OPT("--prior")) p.prior_weight = atof(argv[++i]);
        else if (OPT("--out")) out = argv[++i];
        else if (!strcmp(argv[i], "--report")) report = 1;
        else if (!strcmp(argv[i], "--scale") && i + 3 < argc) {
            si = atoi(argv[++i]);
            sj = atoi(argv[++i]);
            metres = atof(argv[++i]);
        } else {
            fprintf(stderr, "vt_calib: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
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
        fprintf(stderr, "vt_calib: no solution from %d points\n", p.n);
        return 1;
    }
    const double th = acos(((s.R[0] + s.R[4] + s.R[8]) - 1.0) * 0.5 > 1.0 ? 1.0 : ((s.R[0] + s.R[4] + s.R[8]) - 1.0) * 0.5) * 180.0 / M_PI;
    printf("points %d, inliers within 3 px %d, rms %.2f px, median %.2f px\n", p.n, s.inliers, s.rms_px, s.median_px);
    printf("fa %.1f px (horizontal view %.1f deg)  fb %.1f px (%.1f deg)\n", s.fa, 2.0 * atan(0.5 * p.wa / s.fa) * 180.0 / M_PI, s.fb, 2.0 * atan(0.5 * p.wb / s.fb) * 180.0 / M_PI);
    printf("rotation between the cameras %.1f deg; baseline direction in B %.3f %.3f %.3f\n", th, s.t[0], s.t[1], s.t[2]);
    double scale = 1.0;
    if (si >= 0) {
        const double u = calib_length(&p, &s, si, sj);
        scale = metres / u;
        printf("scale: %s .. %s = %.4f m assumed: baseline %.3f m\n", labels[si], labels[sj], metres, scale);
    }
    if (report) {
        printf("%-18s %7s %7s   %8s %8s %8s  %s\n", "point", "errA px", "errB px", "X", "Y", "Z", si >= 0 ? "metres, camera A's frame" : "unit baseline, camera A's frame");
        for (int i = 0; i < p.n; i++) {
            double P[3], ea, eb;
            calib_point(&p, &s, i, P, &ea, &eb);
            printf("%-18s %7.2f %7.2f   %8.3f %8.3f %8.3f\n", labels[i], ea, eb, P[0] * scale, P[1] * scale, P[2] * scale);
        }
    }
    if (out && si >= 0) {
        struct vt_calib c;
        calib_to_wire(&p, &s, scale, &c);
        char note[200];
        snprintf(note, sizeof note, "%d pts rms %.2f px, base %.3f m if %s-%s = %.3f m", p.n, s.rms_px, scale, labels[si], labels[sj], metres);
        if (calib_write(out, &c, note)) {
            perror(out);
            return 1;
        }
        printf("wrote %s\n", out);
    }
    return 0;
}
