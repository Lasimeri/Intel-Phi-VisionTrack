/* vt_align.c: camera D placed in the L-R geometry from the real room, on the desktop host, no chart and no hand-read chair.
 *   vt_align R.pgm L.pgm D.pgm LR.txt TRIPLES.txt [--fd F] [--out OUT.txt] [--trials N]
 * Three steps, each after orbital and satellite navigation:
 *   the map     a depth map from R and L: thousands of pixels with a known place in the room.
 *   resection   D sees known beacons (TRIPLES.txt: the LEDs, the armrest's tips). At each candidate position in a grid, Wahba's problem gives the attitude that best turns the bearings to the beacons into the
 *               bearings D measures (Davenport's q method); the leftover angle ranks the positions.
 *   the vote    each pose is scored by the room: the map's pixels projected into D, D's normalized image against R's. The right pose is where they agree. Pose and focal length are then refined on that score,
 *               and it is reported against what wrong poses score: a pose no better than chance is no pose. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calib.h"
#include "depth.h"
#include "img.h"
#include "stars.h"

#define LEVEL 2
#define MAX_BEACONS 32
#define MAX_POINTS 60000
#define TRUNC 1.5f

struct pt3 {
    float x, y, z; /* in R's frame, metres */
    float v;       /* R's normalized value */
};

static struct pt3 pts[MAX_POINTS];
static int npts;

static void *host_alloc(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

/* A level of a frame, normalized. */
static struct vt_plane level_of(const char *path, struct vt_plane *spread, float eps)
{
    int w, h;
    uint16_t *g = star_read_pgm(path, &w, &h);
    if (!g)
        exit(2);
    uint8_t *img = aligned_alloc(64, ((size_t)w * (size_t)h + 63) & ~(size_t)63);
    if (!img)
        exit(2);
    for (size_t i = 0; i < (size_t)w * (size_t)h; i++)
        img[i] = (uint8_t)(g[i] >> 8);
    free(g);
    struct vt_plane l = plane_at(aligned_alloc(64, plane_bytes(w, h)), w, h);
    img_u8_to_q6_vx(&l, img, w, 0, h);
    img_pad_rows(&l, 0, h);
    img_pad_tb(&l);
    free(img);
    for (int i = 1; i <= LEVEL; i++) {
        const int nw = (l.w + 1) / 2, nh = (l.h + 1) / 2;
        struct vt_plane n = plane_at(aligned_alloc(64, plane_bytes(nw, nh)), nw, nh);
        img_reduce_vx(&n, &l, 0, nh);
        img_pad_rows(&n, 0, nh);
        img_pad_tb(&n);
        l = n;
    }
    struct vt_plane out = plane_at(aligned_alloc(64, plane_bytes(l.w, l.h)), l.w, l.h);
    void *sc = aligned_alloc(64, depth_norm_scratch_bytes(l.w));
    depth_normalize(&out, spread, &l, eps, 0, l.h, sc);
    free(sc);
    return out;
}

/* Davenport's q method: the rotation A maximizing sum w_i b_i' A r_i, for bearings r in the reference frame and b in the body: the eigenvector of the largest eigenvalue, by power iteration on a shifted matrix. */
static void wahba(const double r[][3], const double b[][3], const double *w, int n, double *A)
{
    double B[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < n; i++)
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++)
                B[p][q] += w[i] * b[i][p] * r[i][q];
    const double S[3][3] = {{B[0][0] + B[0][0], B[0][1] + B[1][0], B[0][2] + B[2][0]}, {B[1][0] + B[0][1], B[1][1] + B[1][1], B[1][2] + B[2][1]}, {B[2][0] + B[0][2], B[2][1] + B[1][2], B[2][2] + B[2][2]}};
    const double tr = B[0][0] + B[1][1] + B[2][2];
    const double z[3] = {B[1][2] - B[2][1], B[2][0] - B[0][2], B[0][1] - B[1][0]};
    const double K[4][4] = {{S[0][0] - tr, S[0][1], S[0][2], z[0]}, {S[1][0], S[1][1] - tr, S[1][2], z[1]}, {S[2][0], S[2][1], S[2][2] - tr, z[2]}, {z[0], z[1], z[2], tr}};
    double wsum = 0.0;
    for (int i = 0; i < n; i++)
        wsum += w[i];
    const double shift = 2.0 * wsum + 1.0;
    double q[4] = {0.3, 0.3, 0.3, 0.8};
    for (int it = 0; it < 200; it++) {
        double o[4];
        for (int i = 0; i < 4; i++) {
            o[i] = shift * q[i];
            for (int j = 0; j < 4; j++)
                o[i] += K[i][j] * q[j];
        }
        const double nn = sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
        for (int i = 0; i < 4; i++)
            q[i] = o[i] / nn;
    }
    const double x = q[0], y = q[1], s = q[2], t = q[3];
    A[0] = x * x - y * y - s * s + t * t;
    A[1] = 2 * (x * y + s * t);
    A[2] = 2 * (x * s - y * t);
    A[3] = 2 * (x * y - s * t);
    A[4] = -x * x + y * y - s * s + t * t;
    A[5] = 2 * (y * s + x * t);
    A[6] = 2 * (x * s + y * t);
    A[7] = 2 * (y * s - x * t);
    A[8] = -x * x - y * y + s * s + t * t;
}

struct pose {
    double A[9], b[3]; /* X_d = A X_r + b */
    double f;
};

/* What the room says about a pose: mean truncated difference between R's value and D's at each map pixel D can see, and the share of them it can. */
static double score_pose(const struct pose *p, const struct vt_plane *nd, double *seen)
{
    const double f = p->f / (1 << LEVEL), cx = 639.5 / (1 << LEVEL), cy = 359.5 / (1 << LEVEL);
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < npts; i++) {
        const struct pt3 *q = &pts[i];
        const double X = p->A[0] * q->x + p->A[1] * q->y + p->A[2] * q->z + p->b[0];
        const double Y = p->A[3] * q->x + p->A[4] * q->y + p->A[5] * q->z + p->b[1];
        const double Z = p->A[6] * q->x + p->A[7] * q->y + p->A[8] * q->z + p->b[2];
        if (Z <= 0.05)
            continue;
        const double u = f * X / Z + cx, v = f * Y / Z + cy;
        const int iu = (int)floor(u), iv = (int)floor(v);
        if (iu < 0 || iv < 0 || iu + 1 >= nd->w || iv + 1 >= nd->h)
            continue;
        const double fu = u - iu, fv = v - iv;
        const float *r0 = plane_row(nd, iv), *r1 = plane_row(nd, iv + 1);
        const double s = (1 - fv) * ((1 - fu) * r0[iu] + fu * r0[iu + 1]) + fv * ((1 - fu) * r1[iu] + fu * r1[iu + 1]);
        const double e = fabs(s - (double)q->v);
        sum += e < TRUNC ? e : TRUNC;
        n++;
    }
    *seen = (double)n / npts;
    /* A pixel D cannot see says nothing for the pose: it costs as much as a mismatch, so a pose cannot win by looking away. */
    return (sum + (double)(npts - n) * TRUNC) / npts;
}

static void rotate_left(double *A, double wx, double wy, double wz)
{
    double E[9], o[9];
    star_rotation(wx, wy, wz, E);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            o[3 * r + c] = E[3 * r] * A[c] + E[3 * r + 1] * A[3 + c] + E[3 * r + 2] * A[6 + c];
    memcpy(A, o, sizeof o);
}

/* The beacons, for the refinement to stay bound to: unweighted the room alone walked D 2.95 m from R, past L, on a score hardly better than chance. */
static double g_world[MAX_BEACONS][3], g_body[MAX_BEACONS][3], g_weight[MAX_BEACONS];
static int g_nb;
#define BEACON_COST 0.02 /* per squared degree of weighted miss: 2 degrees costs about what the room's margin over chance is */

static double beacon_cost(const struct pose *p)
{
    double sum = 0.0;
    for (int i = 0; i < g_nb; i++) {
        const double *w = g_world[i];
        const double X = p->A[0] * w[0] + p->A[1] * w[1] + p->A[2] * w[2] + p->b[0], Y = p->A[3] * w[0] + p->A[4] * w[1] + p->A[5] * w[2] + p->b[1], Z = p->A[6] * w[0] + p->A[7] * w[1] + p->A[8] * w[2] + p->b[2];
        const double n = sqrt(X * X + Y * Y + Z * Z);
        const double c = (X * g_body[i][0] + Y * g_body[i][1] + Z * g_body[i][2]) / n;
        const double ang = acos(c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c) * 180.0 / M_PI;
        sum += g_weight[i] * ang * ang;
    }
    return BEACON_COST * sum;
}

/* Coordinate descent on the room's score and the beacons: each of attitude, position, focal length, in shrinking steps. */
static double refine(struct pose *p, const struct vt_plane *nd, double rot_step, double pos_step, int free_f)
{
    double seen;
    double best = score_pose(p, nd, &seen) + beacon_cost(p);
    for (int round = 0; round < 7; round++) {
        for (int pass = 0; pass < 3; pass++)
            for (int k = 0; k < (free_f ? 7 : 6); k++)
                for (int sign = -1; sign <= 1; sign += 2) {
                    struct pose t = *p;
                    if (k < 3) {
                        const double d = sign * rot_step;
                        const double w[3] = {k == 0 ? d : 0.0, k == 1 ? d : 0.0, k == 2 ? d : 0.0};
                        /* Keep the camera's centre: C = -A' b. */
                        double C[3];
                        for (int i = 0; i < 3; i++)
                            C[i] = -(p->A[i] * p->b[0] + p->A[3 + i] * p->b[1] + p->A[6 + i] * p->b[2]);
                        rotate_left(t.A, w[0], w[1], w[2]);
                        for (int i = 0; i < 3; i++)
                            t.b[i] = -(t.A[3 * i] * C[0] + t.A[3 * i + 1] * C[1] + t.A[3 * i + 2] * C[2]);
                    } else if (k < 6) {
                        t.b[k - 3] += sign * pos_step;
                    } else {
                        t.f *= 1.0 + sign * 0.5 * rot_step;
                    }
                    const double s = score_pose(&t, nd, &seen) + beacon_cost(&t);
                    if (s < best) {
                        best = s;
                        *p = t;
                    }
                }
        rot_step *= 0.6;
        pos_step *= 0.6;
    }
    return best;
}

/* The point nearest both rays, in the frame of camera B of X_b = R X_a + T. Normalized image points. */
static void triangulate_b(const double *R, const double *T, double ax, double ay, double bx, double by, double *Pb)
{
    const double da[3] = {R[0] * ax + R[1] * ay + R[2], R[3] * ax + R[4] * ay + R[5], R[6] * ax + R[7] * ay + R[8]};
    const double db[3] = {bx, by, 1.0};
    const double aa = da[0] * da[0] + da[1] * da[1] + da[2] * da[2], bb = db[0] * db[0] + db[1] * db[1] + 1.0, ab = da[0] * db[0] + da[1] * db[1] + da[2];
    const double at = da[0] * T[0] + da[1] * T[1] + da[2] * T[2], bt = db[0] * T[0] + db[1] * T[1] + T[2];
    const double den = aa * bb - ab * ab;
    const double s = den > 1e-12 ? (ab * bt - bb * at) / den : 0.0, u = den > 1e-12 ? (aa * bt - ab * at) / den : 0.0;
    for (int i = 0; i < 3; i++)
        Pb[i] = 0.5 * (T[i] + s * da[i] + u * db[i]);
}

struct seed {
    struct pose p;
    double angle, score;
};

static int cmp_seed(const void *a, const void *b)
{
    const double x = ((const struct seed *)a)->score, y = ((const struct seed *)b)->score;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: vt_align R.pgm L.pgm D.pgm LR.txt TRIPLES.txt [--fd F] [--out OUT.txt]\n");
        return 2;
    }
    double fd = 1272.0;
    const char *out = NULL;
    int fix_only = 0;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--fix")) fix_only = 1;
        else if (!strcmp(argv[i], "--fd") && i + 1 < argc) fd = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    }
    struct vt_calib lr;
    if (calib_read(argv[4], &lr))
        return 2;
    const float eps = 1.0f * VT_Q6;
    int w, h;
    {
        uint16_t *g = star_read_pgm(argv[1], &w, &h);
        if (!g)
            return 2;
        free(g);
    }
    const int lw = w >> LEVEL, lh = h >> LEVEL, s = 1 << LEVEL;
    struct vt_plane sr = plane_at(aligned_alloc(64, plane_bytes(lw, lh)), lw, lh);
    struct vt_plane nr = level_of(argv[1], &sr, eps), nl = level_of(argv[2], NULL, eps), nd = level_of(argv[3], NULL, eps);
    /* The map, R the reference. L relative to R: X_l = R' (X_r - T). */
    struct depth_cfg cfg = {.w = lw, .h = lh, .f = lr.fb / s, .cx = lr.cxb / s, .cy = lr.cyb / s, .planes = 64, .z_near = 0.8, .z_far = 12.0, .agg_radius = 3, .trunc = TRUNC, .tex_min = 1.5f * VT_Q6, .conf_min = 0.06f, .threads = 1};
    struct depth *d = depth_create(&cfg, host_alloc, NULL);
    if (!d)
        return 2;
    struct depth_view v = {.norm = &nl, .f = lr.fa / s, .cx = lr.cxa / s, .cy = lr.cya / s, .weight = 1.0f};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++)
            v.R[3 * r + c] = lr.R[3 * c + r];
        v.T[r] = -(lr.R[r] * lr.T[0] + lr.R[3 + r] * lr.T[1] + lr.R[6 + r] * lr.T[2]);
    }
    depth_set_views(d, &nr, &sr, &v, 1);
    depth_sweep(d, 0, 1);
    depth_aggregate(d, 0, 1);
    depth_decide(d, 0, 1);
    const float *z = depth_z(d);
    const int st = depth_stride(d);
    for (int y = 4; y < lh - 4; y++)
        for (int x = 4; x < lw - 4 && npts < MAX_POINTS; x++) {
            const float zv = z[(size_t)y * (size_t)st + (size_t)x];
            if (zv <= 0.0f)
                continue;
            const double bx = ((double)x - cfg.cx) / cfg.f, by = ((double)y - cfg.cy) / cfg.f;
            pts[npts] = (struct pt3){(float)(bx * zv), (float)(by * zv), zv, plane_row(&nr, y)[x]};
            npts++;
        }
    printf("the map: %d pixels with a place in the room, R the reference, fb %.1f fa %.1f\n", npts, lr.fb, lr.fa);
    if (npts < 500) {
        fprintf(stderr, "vt_align: too small a map\n");
        return 1;
    }
    /* The beacons, in R's frame, and what D measures of them. */
    double world[MAX_BEACONS][3], body[MAX_BEACONS][3], weights[MAX_BEACONS];
    char names[MAX_BEACONS][32];
    int nb = 0;
    FILE *f = fopen(argv[5], "r");
    if (!f) {
        perror(argv[5]);
        return 2;
    }
    char line[256];
    while (nb < MAX_BEACONS && fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        double xl, yl, xr, yr, xd, yd;
        if (sscanf(line, "%lf %lf %lf %lf %lf %lf %31s", &xl, &yl, &xr, &yr, &xd, &yd, names[nb]) < 6)
            continue;
        triangulate_b(lr.R, lr.T, (xl - lr.cxa) / lr.fa, (yl - lr.cya) / lr.fa, (xr - lr.cxb) / lr.fb, (yr - lr.cyb) / lr.fb, world[nb]);
        const double bx = (xd - 639.5) / fd, by = (yd - 359.5) / fd, n = sqrt(bx * bx + by * by + 1.0);
        body[nb][0] = bx / n;
        body[nb][1] = by / n;
        body[nb][2] = 1.0 / n;
        /* The LEDs are read to a tenth of a pixel, the armrest's soft tips to several. */
        weights[nb] = !strncmp(names[nb], "led", 3) ? 1.0 : 0.3;
        memcpy(g_world[nb], world[nb], sizeof g_world[nb]);
        memcpy(g_body[nb], body[nb], sizeof g_body[nb]);
        g_weight[nb] = weights[nb];
        g_nb = nb + 1;
        printf("  beacon %-10s X %7.3f Y %7.3f Z %7.3f m in R's frame\n", names[nb], world[nb][0], world[nb][1], world[nb][2]);
        nb++;
    }
    fclose(f);
    /* Resection: every candidate position in the room, the attitude from Wahba, ranked by the room's own score. */
    enum { MAX_SEEDS = 200000 };
    static struct seed seeds[MAX_SEEDS];
    int ns = 0;
    for (double X = -3.0; X <= 3.0; X += 0.2)
        for (double Y = -1.2; Y <= 1.2; Y += 0.3)
            for (double Z = -1.5; Z <= 3.0; Z += 0.2) {
                double rb[MAX_BEACONS][3];
                int ok = 1;
                for (int i = 0; i < nb; i++) {
                    const double dx = world[i][0] - X, dy = world[i][1] - Y, dz = world[i][2] - Z, n = sqrt(dx * dx + dy * dy + dz * dz);
                    if (n < 0.2) {
                        ok = 0;
                        break;
                    }
                    rb[i][0] = dx / n;
                    rb[i][1] = dy / n;
                    rb[i][2] = dz / n;
                }
                if (!ok || ns >= MAX_SEEDS)
                    continue;
                struct pose p;
                p.f = fd;
                wahba(rb, body, weights, nb, p.A);
                double worst = 0.0;
                for (int i = 0; i < nb; i++) {
                    const double m0 = p.A[0] * rb[i][0] + p.A[1] * rb[i][1] + p.A[2] * rb[i][2], m1 = p.A[3] * rb[i][0] + p.A[4] * rb[i][1] + p.A[5] * rb[i][2], m2 = p.A[6] * rb[i][0] + p.A[7] * rb[i][1] + p.A[8] * rb[i][2];
                    const double c = m0 * body[i][0] + m1 * body[i][1] + m2 * body[i][2];
                    const double ang = acos(c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c) * 180.0 / M_PI;
                    worst = ang > worst ? ang : worst;
                }
                /* A pose missing a beacon by more than this cannot be near the truth. */
                if (worst > 6.0)
                    continue;
                for (int i = 0; i < 3; i++)
                    p.b[i] = -(p.A[3 * i] * X + p.A[3 * i + 1] * Y + p.A[3 * i + 2] * Z);
                double seen;
                seeds[ns].p = p;
                seeds[ns].angle = worst;
                seeds[ns].score = score_pose(&p, &nd, &seen);
                ns++;
            }
    if (!ns) {
        fprintf(stderr, "vt_align: no position agrees with the beacons within 6 degrees\n");
        return 1;
    }
    /* What the beacons alone say, before the room votes. */
    {
        double best_angle = 1e30;
        int at = 0;
        for (int i = 0; i < ns; i++)
            if (seeds[i].angle < best_angle) {
                best_angle = seeds[i].angle;
                at = i;
            }
        const struct pose *q = &seeds[at].p;
        /* The homographies that turn R and L to face as D does, the rotation alone: for vt_match, so what is left between the pictures is only the cameras' displacement. X_d ~ A X_r, X_r = R X_l + T. */
        {
            const double kd[3] = {fd, 639.5, 359.5};
            double AL[9];
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    AL[3 * r + c] = q->A[3 * r] * lr.R[c] + q->A[3 * r + 1] * lr.R[3 + c] + q->A[3 * r + 2] * lr.R[6 + c];
            for (int cam = 0; cam < 2; cam++) {
                const double *M = cam ? AL : q->A;
                const double fs = cam ? lr.fa : lr.fb, cxs = cam ? lr.cxa : lr.cxb, cys = cam ? lr.cya : lr.cyb;
                /* H = Kd M Ks^-1. Ks^-1 maps (u, v, 1) to ((u - cx) / f, (v - cy) / f, 1). */
                double H[9];
                for (int r = 0; r < 3; r++) {
                    const double row0 = r == 0 ? kd[0] * M[0] + kd[1] * M[6] : r == 1 ? kd[0] * M[3] + kd[2] * M[6] : M[6];
                    const double row1 = r == 0 ? kd[0] * M[1] + kd[1] * M[7] : r == 1 ? kd[0] * M[4] + kd[2] * M[7] : M[7];
                    const double row2 = r == 0 ? kd[0] * M[2] + kd[1] * M[8] : r == 1 ? kd[0] * M[5] + kd[2] * M[8] : M[8];
                    H[3 * r] = row0 / fs;
                    H[3 * r + 1] = row1 / fs;
                    H[3 * r + 2] = row2 - row0 * cxs / fs - row1 * cys / fs;
                }
                printf("H_%c", cam ? 'L' : 'R');
                for (int i = 0; i < 9; i++)
                    printf(" %.10g", H[i]);
                printf("\n");
            }
        }
        printf("beacons alone: worst miss %.2f deg at centre %.2f %.2f %.2f m in R's frame, room score %.4f\n", best_angle, -(q->A[0] * q->b[0] + q->A[3] * q->b[1] + q->A[6] * q->b[2]), -(q->A[1] * q->b[0] + q->A[4] * q->b[1] + q->A[7] * q->b[2]), -(q->A[2] * q->b[0] + q->A[5] * q->b[1] + q->A[8] * q->b[2]), seeds[at].score);
        for (int i = 0; i < ns; i++) {
            const struct pose *r = &seeds[i].p;
            if (seeds[i].angle < 1.5)
                printf("  within 1.5 deg: miss %.2f centre %6.2f %6.2f %6.2f room %.4f\n", seeds[i].angle, -(r->A[0] * r->b[0] + r->A[3] * r->b[1] + r->A[6] * r->b[2]), -(r->A[1] * r->b[0] + r->A[4] * r->b[1] + r->A[7] * r->b[2]), -(r->A[2] * r->b[0] + r->A[5] * r->b[1] + r->A[8] * r->b[2]), seeds[i].score);
        }
    }
    qsort(seeds, (size_t)ns, sizeof seeds[0], cmp_seed);
    const double median_score = seeds[ns / 2].score;
    printf("resection: %d candidate positions agree with the beacons within 6 deg; the room's score: best %.4f, median %.4f, worst %.4f\n", ns, seeds[0].score, median_score, seeds[ns - 1].score);
    /* The room refines the seeds the beacons like most, under the beacons' own cost. */
    for (int i = 0; i < ns; i++)
        seeds[i].score = seeds[i].angle;
    qsort(seeds, (size_t)ns, sizeof seeds[0], cmp_seed);
    if (fix_only) {
        /* A navigation fix: the beacons alone, solved continuously from every seed, the distinct answers listed. The room measured no better than chance, so it has no vote here. Focal length stays as given. */
        struct pose found[64];
        double miss[64];
        int nf = 0;
        for (int i = 0; i < ns && nf < 64; i += ns > 400 ? ns / 400 : 1) {
            struct pose p = seeds[i].p;
            double rot = 0.02, pos = 0.06;
            double cost = beacon_cost(&p);
            for (int round = 0; round < 40; round++) {
                int improved = 0;
                for (int k = 0; k < 6; k++)
                    for (int sign = -1; sign <= 1; sign += 2) {
                        struct pose t = p;
                        if (k < 3) {
                            const double dv = sign * rot;
                            double C0[3];
                            for (int a = 0; a < 3; a++)
                                C0[a] = -(p.A[a] * p.b[0] + p.A[3 + a] * p.b[1] + p.A[6 + a] * p.b[2]);
                            rotate_left(t.A, k == 0 ? dv : 0.0, k == 1 ? dv : 0.0, k == 2 ? dv : 0.0);
                            for (int a = 0; a < 3; a++)
                                t.b[a] = -(t.A[3 * a] * C0[0] + t.A[3 * a + 1] * C0[1] + t.A[3 * a + 2] * C0[2]);
                        } else {
                            t.b[k - 3] += sign * pos;
                        }
                        const double c2 = beacon_cost(&t);
                        if (c2 < cost) {
                            cost = c2;
                            p = t;
                            improved = 1;
                        }
                    }
                if (!improved) {
                    rot *= 0.5;
                    pos *= 0.5;
                }
            }
            /* Is it a new answer? */
            double Cn[3];
            for (int a = 0; a < 3; a++)
                Cn[a] = -(p.A[a] * p.b[0] + p.A[3 + a] * p.b[1] + p.A[6 + a] * p.b[2]);
            int same = 0;
            for (int j = 0; j < nf; j++) {
                double Cj[3];
                for (int a = 0; a < 3; a++)
                    Cj[a] = -(found[j].A[a] * found[j].b[0] + found[j].A[3 + a] * found[j].b[1] + found[j].A[6 + a] * found[j].b[2]);
                const double dc = sqrt((Cn[0] - Cj[0]) * (Cn[0] - Cj[0]) + (Cn[1] - Cj[1]) * (Cn[1] - Cj[1]) + (Cn[2] - Cj[2]) * (Cn[2] - Cj[2]));
                if (dc < 0.15) {
                    same = 1;
                    if (cost < miss[j]) {
                        miss[j] = cost;
                        found[j] = p;
                    }
                }
            }
            if (!same) {
                found[nf] = p;
                miss[nf] = cost;
                nf++;
            }
        }
        for (int j = 0; j < nf; j++) {
            if (miss[j] > 0.5)
                continue;
            double Cj[3];
            for (int a = 0; a < 3; a++)
                Cj[a] = -(found[j].A[a] * found[j].b[0] + found[j].A[3 + a] * found[j].b[1] + found[j].A[6 + a] * found[j].b[2]);
            printf("fix %d: beacon cost %.4f, centre %6.3f %6.3f %6.3f m in R's frame, %.3f m from R\n", j, miss[j], Cj[0], Cj[1], Cj[2], sqrt(Cj[0] * Cj[0] + Cj[1] * Cj[1] + Cj[2] * Cj[2]));
            if (out) {
                struct vt_calib o = lr;
                o.valid_d = 1;
                o.fd = fd;
                o.cxd = 639.5;
                o.cyd = 359.5;
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++)
                        o.Rd[3 * r + c] = found[j].A[3 * r] * lr.R[c] + found[j].A[3 * r + 1] * lr.R[3 + c] + found[j].A[3 * r + 2] * lr.R[6 + c];
                    o.Td[r] = found[j].A[3 * r] * lr.T[0] + found[j].A[3 * r + 1] * lr.T[1] + found[j].A[3 * r + 2] * lr.T[2] + found[j].b[r];
                }
                char path[300], note[120];
                snprintf(path, sizeof path, "%s-%d.txt", out, j);
                snprintf(note, sizeof note, "D by resection on LEDs + armrest, beacon cost %.4f", miss[j]);
                calib_write(path, &o, note);
            }
        }
        return 0;
    }
    struct seed best;
    best.score = 1e30;
    const int top = ns < 40 ? ns : 40;
    for (int i = 0; i < top; i++) {
        struct pose p = seeds[i].p;
        const double sc = refine(&p, &nd, 0.015, 0.05, 1);
        if (sc < best.score) {
            best.score = sc;
            best.p = p;
        }
    }
    double seen;
    const double final_score = score_pose(&best.p, &nd, &seen);
    /* What a wrong pose scores: the same pose with each map pixel's value taken from another pixel: the score of agreement by chance, seen as much. */
    float *saved = malloc(sizeof(float) * (size_t)npts);
    for (int i = 0; i < npts; i++)
        saved[i] = pts[i].v;
    for (int i = 0; i < npts; i++)
        pts[i].v = saved[(i * 7919L + 104729L) % npts];
    double seen2;
    const double chance = score_pose(&best.p, &nd, &seen2);
    for (int i = 0; i < npts; i++)
        pts[i].v = saved[i];
    free(saved);
    const double C[3] = {-(best.p.A[0] * best.p.b[0] + best.p.A[3] * best.p.b[1] + best.p.A[6] * best.p.b[2]), -(best.p.A[1] * best.p.b[0] + best.p.A[4] * best.p.b[1] + best.p.A[7] * best.p.b[2]), -(best.p.A[2] * best.p.b[0] + best.p.A[5] * best.p.b[1] + best.p.A[8] * best.p.b[2])};
    printf("best pose: score %.4f, seen %.1f %%; chance on the same pose %.4f; fd %.1f; D's centre in R's frame %.3f %.3f %.3f m (%.3f m from R)\n", final_score, 100.0 * seen, chance, best.p.f, C[0], C[1], C[2], sqrt(C[0] * C[0] + C[1] * C[1] + C[2] * C[2]));
    for (int i = 0; i < nb; i++) {
        const double X = best.p.A[0] * world[i][0] + best.p.A[1] * world[i][1] + best.p.A[2] * world[i][2] + best.p.b[0], Y = best.p.A[3] * world[i][0] + best.p.A[4] * world[i][1] + best.p.A[5] * world[i][2] + best.p.b[1], Z = best.p.A[6] * world[i][0] + best.p.A[7] * world[i][1] + best.p.A[8] * world[i][2] + best.p.b[2];
        const double u = best.p.f * X / Z + 639.5, v2 = best.p.f * Y / Z + 359.5;
        const double bu = body[i][0] / body[i][2] * fd + 639.5, bv = body[i][1] / body[i][2] * fd + 359.5;
        printf("  %-10s projects to %7.1f %6.1f in D; measured %7.1f %6.1f; off by %.1f px; Z %.2f m\n", names[i], u, v2, bu, bv, sqrt((u - bu) * (u - bu) + (v2 - bv) * (v2 - bv)), Z);
    }
    if (out) {
        struct vt_calib o = lr;
        o.valid_d = 1;
        o.fd = best.p.f;
        o.cxd = 639.5;
        o.cyd = 359.5;
        /* X_d = A X_r + b, X_r = R X_l + T. */
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                o.Rd[3 * r + c] = best.p.A[3 * r] * lr.R[c] + best.p.A[3 * r + 1] * lr.R[3 + c] + best.p.A[3 * r + 2] * lr.R[6 + c];
            o.Td[r] = best.p.A[3 * r] * lr.T[0] + best.p.A[3 * r + 1] * lr.T[1] + best.p.A[3 * r + 2] * lr.T[2] + best.p.b[r];
        }
        char note[200];
        snprintf(note, sizeof note, "L-R by landmarks; D by resection + room score %.3f (chance %.3f)", final_score, chance);
        if (calib_write(out, &o, note)) {
            perror(out);
            return 1;
        }
        printf("wrote %s\n", out);
    }
    return 0;
}
