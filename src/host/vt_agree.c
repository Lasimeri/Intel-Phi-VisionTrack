/* vt_agree.c: do L and R, each referred to the centre camera, say the same thing? On the desktop host, the lane model, one thread.
 *   vt_agree CALIB PREFIX [--dump OUT]     PREFIX-{d,r,l}.pgm
 *
 * The map's own confidence cannot judge geometry with anyone in view: a pose 3 degrees wrong scored higher than the right one. This can. For each of D's pixels the depth is found from D and R alone, and again
 * from D and L alone. They share only D's picture, so they agree only if both cameras really are where the geometry says, relative to D. Whoever is in the room, it is the same question.
 *
 * Reported: how much of D each camera can see at all; how many pixels each pair is sure of; for pixels both are sure of, how far apart the two depths are, in planes of the sweep; and the same for unrelated pixels,
 * which is what chance looks like. --dump: OUT-r.pgm, OUT-l.pgm, OUT-both.pgm, nearest bright, nothing black. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "calib.h"
#include "depth.h"
#include "img.h"
#include "stars.h"
#include "vt_proto.h"

#define SURE 0.2f

static void *heap(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

static struct vt_plane level_of(const char *prefix, char cam, int *full_w)
{
    char path[512];
    snprintf(path, sizeof path, "%s-%c.pgm", prefix, cam);
    int w, h;
    uint16_t *g = star_read_pgm(path, &w, &h);
    if (!g) {
        fprintf(stderr, "vt_agree: %s\n", path);
        exit(2);
    }
    const int stride = (w + 15) & ~15;
    uint8_t *img = heap(NULL, (size_t)stride * (size_t)h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            img[(size_t)y * (size_t)stride + (size_t)x] = (uint8_t)(g[(size_t)y * (size_t)w + (size_t)x] >> 8);
    free(g);
    struct vt_plane p = plane_at(heap(NULL, plane_bytes(w, h)), w, h);
    img_u8_to_q6_vx(&p, img, stride, 0, h);
    img_pad_rows(&p, 0, h);
    img_pad_tb(&p);
    for (int l = 0; l < VT_DEPTH_LEVEL; l++) {
        const int nw = (p.w + 1) / 2, nh = (p.h + 1) / 2;
        struct vt_plane q = plane_at(heap(NULL, plane_bytes(nw, nh)), nw, nh);
        img_reduce_vx(&q, &p, 0, nh);
        img_pad_rows(&q, 0, nh);
        img_pad_tb(&q);
        p = q;
    }
    *full_w = w;
    return p;
}

static int cmp_f(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static void solve(struct depth *d, const struct vt_plane *nd, const struct vt_plane *sd, const struct depth_view *v, int views, float *z, float *conf, size_t n)
{
    depth_set_views(d, nd, sd, v, views);
    depth_sweep(d, 0, 1);
    depth_aggregate(d, 0, 1);
    depth_decide(d, 0, 1);
    memcpy(z, depth_z(d), n * sizeof(float));
    memcpy(conf, depth_conf(d), n * sizeof(float));
}

static void dump_pgm(const char *out, const char *tag, const float *z, const float *conf, int w, int h, int st, float z_near, float z_far)
{
    char path[512];
    snprintf(path, sizeof path, "%s-%s.pgm", out, tag);
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    const float a = 1.0f / z_far, b = 1.0f / z_near;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const size_t i = (size_t)y * (size_t)st + (size_t)x;
            int v = 0;
            if (z[i] > 0.0f && conf[i] >= SURE)
                v = 40 + (int)(215.0f * (1.0f / z[i] - a) / (b - a));
            fputc(v < 0 ? 0 : v > 255 ? 255 : v, f);
        }
    fclose(f);
}

/* Pixels where D+R, as this view stands, arrives at the depth D+L arrived at. The objective: chance cannot make thousands of pixels agree, and whoever is in the room does not change what it asks. */
static long agreed_with(struct depth *d, const struct vt_plane *nd, const struct vt_plane *sd, const struct depth_view *v, const float *zl, const float *cl, float *z, float *conf, size_t n, double step)
{
    solve(d, nd, sd, v, 1, z, conf, n);
    long pass = 0;
    for (size_t i = 0; i < n; i++)
        pass += z[i] > 0.0f && zl[i] > 0.0f && conf[i] >= SURE && cl[i] >= SURE && fabs(1.0 / (double)z[i] - 1.0 / (double)zl[i]) / step <= 2.0;
    return pass;
}

/* R's pose relative to D, by coordinate descent on that objective: its three turns and three shifts, in shrinking steps. D and L are held: neither moved. */
static void refine_view(struct depth *d, const struct vt_plane *nd, const struct vt_plane *sd, struct depth_view *v, const float *zl, const float *cl, float *z, float *conf, size_t n, double step)
{
    long best = agreed_with(d, nd, sd, v, zl, cl, z, conf, n, step);
    printf("refine: start %ld agreed\n", best);
    double rot = 1.0 * M_PI / 180.0, pos = 0.08;
    for (int round = 0; round < 6; round++) {
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < 6; k++)
                for (int sign = -1; sign <= 1; sign += 2) {
                    struct depth_view t = *v;
                    if (k < 3) {
                        const double a = sign * rot;
                        double E[9];
                        star_rotation(k == 0 ? a : 0.0, k == 1 ? a : 0.0, k == 2 ? a : 0.0, E);
                        for (int r = 0; r < 3; r++) {
                            for (int c = 0; c < 3; c++)
                                t.R[3 * r + c] = E[3 * r] * v->R[c] + E[3 * r + 1] * v->R[3 + c] + E[3 * r + 2] * v->R[6 + c];
                            t.T[r] = E[3 * r] * v->T[0] + E[3 * r + 1] * v->T[1] + E[3 * r + 2] * v->T[2];
                        }
                    } else {
                        t.T[k - 3] += sign * pos;
                    }
                    const long s = agreed_with(d, nd, sd, &t, zl, cl, z, conf, n, step);
                    if (s > best) {
                        best = s;
                        *v = t;
                    }
                }
        printf("refine: steps %.3f deg, %.1f cm: %ld agreed\n", rot * 180.0 / M_PI, pos * 100.0, best);
        rot *= 0.5;
        pos *= 0.5;
    }
}

/* D's own pose and focal length: the centre camera moved until R and L, each referred to it, agree. L and R are held: their relation is the one the far landmarks support. */
static long agreed_through(struct depth *d, const struct vt_plane *nd, const struct vt_plane *sd, const struct vt_plane *nr, const struct vt_plane *nl, const struct vt_calib *c, int scale, float *zr, float *cr, float *zl, float *cl, size_t n, double step)
{
    struct depth_view v[2];
    depth_views_from_calib(c, scale, nr, nl, v);
    depth_set_ref(d, c->fd / scale, c->cxd / scale, c->cyd / scale);
    solve(d, nd, sd, &v[1], 1, zl, cl, n);
    return agreed_with(d, nd, sd, &v[0], zl, cl, zr, cr, n, step);
}

static void refine_d(struct depth *d, const struct vt_plane *nd, const struct vt_plane *sd, const struct vt_plane *nr, const struct vt_plane *nl, struct vt_calib *c, int scale, float *zr, float *cr, float *zl, float *cl, size_t n, double step)
{
    long best = agreed_through(d, nd, sd, nr, nl, c, scale, zr, cr, zl, cl, n, step);
    printf("refine D: start %ld agreed\n", best);
    double rot = 1.0 * M_PI / 180.0, pos = 0.08, fs = 0.02;
    for (int round = 0; round < 6; round++) {
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < 7; k++)
                for (int sign = -1; sign <= 1; sign += 2) {
                    struct vt_calib t = *c;
                    if (k < 3) {
                        const double a = sign * rot;
                        double E[9];
                        star_rotation(k == 0 ? a : 0.0, k == 1 ? a : 0.0, k == 2 ? a : 0.0, E);
                        for (int r = 0; r < 3; r++) {
                            for (int q = 0; q < 3; q++)
                                t.Rd[3 * r + q] = E[3 * r] * c->Rd[q] + E[3 * r + 1] * c->Rd[3 + q] + E[3 * r + 2] * c->Rd[6 + q];
                            t.Td[r] = E[3 * r] * c->Td[0] + E[3 * r + 1] * c->Td[1] + E[3 * r + 2] * c->Td[2];
                        }
                    } else if (k < 6) {
                        t.Td[k - 3] += sign * pos;
                    } else {
                        t.fd = c->fd * (1.0 + sign * fs);
                    }
                    const long s = agreed_through(d, nd, sd, nr, nl, &t, scale, zr, cr, zl, cl, n, step);
                    if (s > best) {
                        best = s;
                        *c = t;
                    }
                }
        printf("refine D: steps %.3f deg, %.1f cm, %.2f %% of fd: %ld agreed, fd %.1f\n", rot * 180.0 / M_PI, pos * 100.0, fs * 100.0, best, c->fd);
        rot *= 0.5;
        pos *= 0.5;
        fs *= 0.5;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: vt_agree CALIB PREFIX [--planes N] [--agg R] [--dump OUT] [--save FILE] [--against FILE] [--refine-r NEW.txt] [--refine-d NEW.txt] [--grid 1]\n");
        return 2;
    }
    struct vt_calib c;
    if (calib_read(argv[1], &c) || !c.valid || !c.valid_d) {
        fprintf(stderr, "vt_agree: %s holds no geometry for all three cameras\n", argv[1]);
        return 2;
    }
    const char *out = NULL, *save = NULL, *against = NULL, *refine_r = NULL, *refine_d_out = NULL;
    int planes = 64, agg = 3, grid = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--dump"))
            out = argv[i + 1];
        else if (!strcmp(argv[i], "--refine-d"))
            refine_d_out = argv[i + 1];
        else if (!strcmp(argv[i], "--grid"))
            grid = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--refine-r"))
            refine_r = argv[i + 1];
        else if (!strcmp(argv[i], "--save"))
            save = argv[i + 1];
        else if (!strcmp(argv[i], "--against"))
            against = argv[i + 1];
        else if (!strcmp(argv[i], "--planes"))
            planes = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--agg"))
            agg = atoi(argv[i + 1]);
    }
    int fw;
    struct vt_plane ld = level_of(argv[2], 'd', &fw), lr = level_of(argv[2], 'r', &fw), ll = level_of(argv[2], 'l', &fw);
    const int scale = 1 << VT_DEPTH_LEVEL;
    const float z_near = 0.8f, z_far = 8.0f, eps = 1.0f * VT_Q6;
    struct vt_plane nd = plane_at(heap(NULL, plane_bytes(ld.w, ld.h)), ld.w, ld.h), sd = plane_at(heap(NULL, plane_bytes(ld.w, ld.h)), ld.w, ld.h);
    struct vt_plane nr = plane_at(heap(NULL, plane_bytes(lr.w, lr.h)), lr.w, lr.h), nl = plane_at(heap(NULL, plane_bytes(ll.w, ll.h)), ll.w, ll.h);
    void *sc = heap(NULL, depth_norm_scratch_bytes(lr.w > ld.w ? lr.w : ld.w));
    depth_normalize(&nd, &sd, &ld, eps, 0, ld.h, sc);
    depth_normalize(&nr, NULL, &lr, eps, 0, lr.h, sc);
    depth_normalize(&nl, NULL, &ll, eps, 0, ll.h, sc);
    /* As the live window asks for it. */
    const struct depth_cfg cfg = {.w = ld.w, .h = ld.h, .f = c.fd / scale, .cx = c.cxd / scale, .cy = c.cyd / scale, .planes = planes, .z_near = z_near, .z_far = z_far, .agg_radius = agg, .trunc = 1.5f, .tex_min = 2.0f * VT_Q6, .conf_min = 0.05f, .threads = 1};
    struct depth *d = depth_create(&cfg, heap, NULL);
    if (!d)
        return 2;
    struct depth_view views[2];
    depth_views_from_calib(&c, scale, &nr, &nl, views);

    /* How much of D each camera can see at all, for a surface at a few depths. */
    printf("of D's picture in view of, at depth:");
    static const double DEPTHS[4] = {1.0, 1.5, 2.5, 4.5};
    for (int i = 0; i < 4; i++) {
        long total = 0, in[2] = {0, 0}, both = 0;
        for (int y = 0; y < ld.h; y += 4)
            for (int x = 0; x < ld.w; x += 4) {
                const double X = ((double)x - cfg.cx) / cfg.f * DEPTHS[i], Y = ((double)y - cfg.cy) / cfg.f * DEPTHS[i], Z = DEPTHS[i];
                int seen = 0;
                for (int v = 0; v < 2; v++) {
                    const struct depth_view *V = &views[v];
                    const double a = V->R[0] * X + V->R[1] * Y + V->R[2] * Z + V->T[0], b = V->R[3] * X + V->R[4] * Y + V->R[5] * Z + V->T[1], e = V->R[6] * X + V->R[7] * Y + V->R[8] * Z + V->T[2];
                    if (e > 0.0) {
                        const double u = V->f * a / e + V->cx, w = V->f * b / e + V->cy;
                        if (u >= 0.0 && w >= 0.0 && u < (double)V->norm->w && w < (double)V->norm->h) {
                            in[v]++;
                            seen++;
                        }
                    }
                }
                both += seen == 2;
                total++;
            }
        printf("  %.1f m: R %.0f %% L %.0f %% both %.0f %%", DEPTHS[i], 100.0 * (double)in[0] / (double)total, 100.0 * (double)in[1] / (double)total, 100.0 * (double)both / (double)total);
    }
    printf("\n");

    const int st = depth_stride(d);
    const size_t n = (size_t)st * (size_t)ld.h;
    float *zr = heap(NULL, n * sizeof(float)), *cr = heap(NULL, n * sizeof(float)), *zl = heap(NULL, n * sizeof(float)), *cl = heap(NULL, n * sizeof(float)), *zb = heap(NULL, n * sizeof(float)), *cb = heap(NULL, n * sizeof(float));
    if (refine_d_out) {
        const double unit = (1.0 / (double)z_near - 1.0 / (double)z_far) / 63.0;
        refine_d(d, &nd, &sd, &nr, &nl, &c, scale, zr, cr, zl, cl, n, unit);
        if (calib_write(refine_d_out, &c, "D refined until R and L agree through it; L, R and their lenses as they were"))
            return 2;
        printf("D refined: written %s\n", refine_d_out);
        depth_views_from_calib(&c, scale, &nr, &nl, views);
        depth_set_ref(d, c.fd / scale, c.cxd / scale, c.cyd / scale);
    }
    solve(d, &nd, &sd, &views[1], 1, zl, cl, n);
    if (refine_r) {
        const double unit = (1.0 / (double)z_near - 1.0 / (double)z_far) / 63.0;
        refine_view(d, &nd, &sd, &views[0], zl, cl, zr, cr, n, unit);
        /* Back to the file's form. X_r = Rv X_d + Tv and X_d = Rd X_a + Td, so X_r = (Rv Rd) X_a + (Rv Td + Tv). */
        struct vt_calib o = c;
        for (int r = 0; r < 3; r++) {
            for (int k = 0; k < 3; k++)
                o.R[3 * r + k] = views[0].R[3 * r] * c.Rd[k] + views[0].R[3 * r + 1] * c.Rd[3 + k] + views[0].R[3 * r + 2] * c.Rd[6 + k];
            o.T[r] = views[0].R[3 * r] * c.Td[0] + views[0].R[3 * r + 1] * c.Td[1] + views[0].R[3 * r + 2] * c.Td[2] + views[0].T[r];
        }
        if (calib_write(refine_r, &o, "R refined until R and L agree through D; L, D and the lenses as solved"))
            return 2;
        printf("R refined: baseline %.3f m, written %s\n", sqrt(o.T[0] * o.T[0] + o.T[1] * o.T[1] + o.T[2] * o.T[2]), refine_r);
    }
    solve(d, &nd, &sd, &views[0], 1, zr, cr, n);
    solve(d, &nd, &sd, views, 2, zb, cb, n);

    /* A fixed unit, whatever the sweep's own count: one step of the 64 planes it used to have, so runs compare. */
    const double near_i = 1.0 / (double)z_near, far_i = 1.0 / (double)z_far, step = (near_i - far_i) / 63.0;
    /* What one of the sweep's own planes means in each camera's pixels at this level, at the centre of D. */
    for (int v = 0; v < 2; v++) {
        const struct depth_view *V = &views[v];
        const double baseline = sqrt(V->T[0] * V->T[0] + V->T[1] * V->T[1] + V->T[2] * V->T[2]);
        const double span_inv = near_i - far_i;
        printf("%c: %.2f m from D; the depth range spans about %.0f of its pixels at most; %d planes = %.2f pixels a plane\n", "RL"[v], baseline, V->f * baseline * span_inv, planes, V->f * baseline * span_inv / (double)(planes - 1));
    }
    long tex = 0, sure_r = 0, sure_l = 0, sure_b = 0, both = 0, w1 = 0, w2 = 0, w4 = 0;
    float *diff = heap(NULL, n * sizeof(float)), *ir = heap(NULL, n * sizeof(float)), *il = heap(NULL, n * sizeof(float));
    for (int y = 0; y < ld.h; y++) {
        const float *sp = plane_row(&sd, y);
        for (int x = 0; x < ld.w; x++) {
            const size_t i = (size_t)y * (size_t)st + (size_t)x;
            tex += sp[x] >= cfg.tex_min;
            const int a = zr[i] > 0.0f && cr[i] >= SURE, b = zl[i] > 0.0f && cl[i] >= SURE;
            sure_r += a;
            sure_l += b;
            sure_b += zb[i] > 0.0f && cb[i] >= SURE;
            if (a && b) {
                const double dp = fabs(1.0 / (double)zr[i] - 1.0 / (double)zl[i]) / step;
                ir[both] = zr[i];
                il[both] = zl[i];
                diff[both++] = (float)dp;
                w1 += dp <= 1.0;
                w2 += dp <= 2.0;
                w4 += dp <= 4.0;
            }
        }
    }
    printf("textured %ld; sure from D+R alone %ld, from D+L alone %ld, from all three %ld; sure from each alone %ld\n", tex, sure_r, sure_l, sure_b, both);
    if (both) {
        /* Chance: the same two lists of depths, paired with unrelated pixels. */
        long c1 = 0, c2 = 0, c4 = 0;
        unsigned rng = 1;
        for (long i = 0; i < both; i++) {
            rng = rng * 1664525u + 1013904223u;
            const long j = (long)((unsigned long)rng % (unsigned long)both);
            const double dp = fabs(1.0 / (double)ir[i] - 1.0 / (double)il[j]) / step;
            c1 += dp <= 1.0;
            c2 += dp <= 2.0;
            c4 += dp <= 4.0;
        }
        qsort(diff, (size_t)both, sizeof(float), cmp_f);
        qsort(ir, (size_t)both, sizeof(float), cmp_f);
        qsort(il, (size_t)both, sizeof(float), cmp_f);
        const double b = (double)both;
        printf("R against L, in steps of 0.0179 / m: median %.1f, p90 %.1f; within 1: %.1f %%, 2: %.1f %%, 4: %.1f %%\n", (double)diff[both / 2], (double)diff[both * 9 / 10], 100.0 * (double)w1 / b, 100.0 * (double)w2 / b, 100.0 * (double)w4 / b);
        printf("if unrelated, chance:                       within 1: %.1f %%, 2: %.1f %%, 4: %.1f %%\n", 100.0 * (double)c1 / b, 100.0 * (double)c2 / b, 100.0 * (double)c4 / b);
        printf("depths, m: R-only p10 %.2f p50 %.2f p90 %.2f; L-only p10 %.2f p50 %.2f p90 %.2f\n", (double)ir[both / 10], (double)ir[both / 2], (double)ir[both * 9 / 10], (double)il[both / 10], (double)il[both / 2], (double)il[both * 9 / 10]);
    }
    /* The map as the two cameras would have it together: a depth only where each, on its own, arrived at it from D. Chance: how many would pass if R's depths were paired with L's from elsewhere. */
    {
        long pass = 0;
        float *zc = heap(NULL, n * sizeof(float)), *cc = heap(NULL, n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            const int ok = zr[i] > 0.0f && zl[i] > 0.0f && cr[i] >= SURE && cl[i] >= SURE && fabs(1.0 / (double)zr[i] - 1.0 / (double)zl[i]) / step <= 2.0;
            zc[i] = ok ? (float)(2.0 / (1.0 / (double)zr[i] + 1.0 / (double)zl[i])) : 0.0f;
            cc[i] = ok ? 1.0f : 0.0f;
            pass += ok;
        }
        printf("agreed by both, within 2 steps: %ld pixels of %ld textured\n", pass, tex);
        if (out)
            dump_pgm(out, "agreed", zc, cc, ld.w, ld.h, st, z_near, z_far);
    }
    /* Where in D's picture: 4 x 3 regions. Textured pixels, sure from each, agreed, and each camera's median depth where it is sure. */
    if (grid) {
        printf("region  textured  sure R  sure L  agreed  z R   z L\n");
        static float a[1 << 18], b[1 << 18];
        for (int gy = 0; gy < 3; gy++)
            for (int gx = 0; gx < 4; gx++) {
                long t = 0, sr = 0, sl = 0, ag = 0;
                for (int y = ld.h * gy / 3; y < ld.h * (gy + 1) / 3; y++) {
                    const float *sp = plane_row(&sd, y);
                    for (int x = ld.w * gx / 4; x < ld.w * (gx + 1) / 4; x++) {
                        const size_t i = (size_t)y * (size_t)st + (size_t)x;
                        t += sp[x] >= cfg.tex_min;
                        const int p = zr[i] > 0.0f && cr[i] >= SURE, q = zl[i] > 0.0f && cl[i] >= SURE;
                        if (p)
                            a[sr++] = zr[i];
                        if (q)
                            b[sl++] = zl[i];
                        ag += p && q && fabs(1.0 / (double)zr[i] - 1.0 / (double)zl[i]) / step <= 2.0;
                    }
                }
                qsort(a, (size_t)sr, sizeof(float), cmp_f);
                qsort(b, (size_t)sl, sizeof(float), cmp_f);
                printf("  %d,%d  %8ld %7ld %7ld %7ld  %4.2f  %4.2f\n", gx, gy, t, sr, sl, ag, sr ? (double)a[sr / 2] : 0.0, sl ? (double)b[sl / 2] : 0.0);
            }
    }
    /* Depths where the two cameras agreed, kept: surfaces of a still room whose depth is known, for D's pixels, whatever is in the room later. */
    if (save) {
        float *keep = heap(NULL, n * sizeof(float));
        long kept = 0;
        for (size_t i = 0; i < n; i++) {
            const int ok = zr[i] > 0.0f && zl[i] > 0.0f && cr[i] >= SURE && cl[i] >= SURE && fabs(1.0 / (double)zr[i] - 1.0 / (double)zl[i]) / step <= 1.0;
            keep[i] = ok ? (float)(2.0 / (1.0 / (double)zr[i] + 1.0 / (double)zl[i])) : 0.0f;
            kept += ok;
        }
        FILE *f = fopen(save, "wb");
        if (!f || fwrite(keep, sizeof(float), n, f) != n)
            return 2;
        fclose(f);
        printf("kept %ld pixels where R and L agree within one step: %s\n", kept, save);
    }
    /* Each camera's depth today against those known depths. */
    if (against) {
        float *known = heap(NULL, n * sizeof(float));
        FILE *f = fopen(against, "rb");
        if (!f || fread(known, sizeof(float), n, f) != n)
            return 2;
        fclose(f);
        const float *const zs[2] = {zr, zl}, *const cs[2] = {cr, cl};
        for (int v = 0; v < 2; v++) {
            long all = 0, sure = 0, a1 = 0, a2 = 0, a4 = 0;
            float *e = heap(NULL, n * sizeof(float));
            for (size_t i = 0; i < n; i++) {
                if (known[i] <= 0.0f)
                    continue;
                all++;
                if (zs[v][i] > 0.0f && cs[v][i] >= SURE) {
                    const double dp = fabs(1.0 / (double)zs[v][i] - 1.0 / (double)known[i]) / step;
                    e[sure++] = (float)dp;
                    a1 += dp <= 1.0;
                    a2 += dp <= 2.0;
                    a4 += dp <= 4.0;
                }
            }
            qsort(e, (size_t)sure, sizeof(float), cmp_f);
            printf("D+%c against the known depths: %ld known pixels, %ld sure now; median %.1f steps; within 1: %.1f %%, 2: %.1f %%, 4: %.1f %%\n", "RL"[v], all, sure, sure ? (double)e[sure / 2] : 0.0, sure ? 100.0 * (double)a1 / (double)sure : 0.0, sure ? 100.0 * (double)a2 / (double)sure : 0.0, sure ? 100.0 * (double)a4 / (double)sure : 0.0);
        }
    }
    if (out) {
        dump_pgm(out, "r", zr, cr, ld.w, ld.h, st, z_near, z_far);
        dump_pgm(out, "l", zl, cl, ld.w, ld.h, st, z_near, z_far);
        dump_pgm(out, "both", zb, cb, ld.w, ld.h, st, z_near, z_far);
    }
    return 0;
}
