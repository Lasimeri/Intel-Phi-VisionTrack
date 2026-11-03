/* m3_ref.c: the depth map of the real room on the desktop host, the lane model, one thread: real frames of the three cameras and a solved geometry. Prints the hash the card's vt_m3 must reproduce, writes
 * the card's input, and the map as an image.
 *   m3_ref D.pgm R.pgm L.pgm CALIB.txt [--views 1|2] [--planes N] [--near M] [--far M] [--agg R] [--trunc T] [--conf C] [--tex T] [--write FILE] [--pgm PREFIX] [--seq PREFIX N] [--avg A] [--refine OUT] [--rounds N] [--source-l]
 * Reference is camera D. CALIB.txt as vt_calib and vt_stitch write it: camera L is the world frame, X_r = R X_l + T, X_d = Rd X_l + Td. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "attitude.h"
#include "calib.h"
#include "depth.h"
#include "img.h"
#include "m3_input.h"
#include "stars.h"
#include "still.h"
#include "vt_util.h"

static void *host_alloc(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

struct pyr {
    struct vt_plane l[M3_LEVELS];
};

static void att_one_thread(void *ctx, att_fn fn, void *arg)
{
    (void)ctx;
    fn(arg, 0, 1);
}

static void make_pyr(struct pyr *p, int w, int h)
{
    for (int i = 0; i < M3_LEVELS; i++) {
        void *m = aligned_alloc(64, plane_bytes(w, h));
        if (!m)
            exit(2);
        p->l[i] = plane_at(m, w, h);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

static void build_pyr(const struct pyr *p, const uint8_t *img, int stride)
{
    img_u8_to_q6_vx(&p->l[0], img, stride, 0, p->l[0].h);
    img_pad_rows(&p->l[0], 0, p->l[0].h);
    img_pad_tb(&p->l[0]);
    for (int i = 1; i < M3_LEVELS; i++) {
        img_reduce_vx(&p->l[i], &p->l[i - 1], 0, p->l[i].h);
        img_pad_rows(&p->l[i], 0, p->l[i].h);
        img_pad_tb(&p->l[i]);
    }
}

/* An 8-bit image, 64 B aligned, from a PGM of 8 or 16 bits: an integrated frame keeps its sum's precision only to here, which is what a camera's 8 bits gave anyway. */
static uint8_t *read_image(const char *path, int *w, int *h)
{
    uint16_t *g = star_read_pgm(path, w, h);
    if (!g)
        return NULL;
    const size_t n = (size_t)*w * (size_t)*h;
    uint8_t *o = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (!o)
        return NULL;
    for (size_t i = 0; i < n; i++)
        o[i] = (uint8_t)(g[i] >> 8);
    free(g);
    return o;
}

/* How much of the reference's textured picture the views agree with at some depth: percent with confidence at least 0.2, and each view must agree on its own: one view cannot carry a wrong pose. */
static double one_view_score(struct depth *d, const struct vt_plane *sd, const struct depth_view *v, const struct vt_plane *nd, int w, int h, float tex_min)
{
    depth_set_views(d, nd, sd, v, 1);
    depth_sweep(d, 0, 1);
    depth_aggregate(d, 0, 1);
    depth_decide(d, 0, 1);
    const float *conf = depth_conf(d);
    const int st = depth_stride(d);
    long tex = 0, sure = 0;
    for (int y = 0; y < h; y++) {
        const float *sp = plane_row(sd, y);
        for (int x = 0; x < w; x++)
            if (sp[x] >= tex_min) {
                tex++;
                sure += conf[(size_t)y * (size_t)st + (size_t)x] >= 0.2f;
            }
    }
    return tex ? 100.0 * (double)sure / (double)tex : 0.0;
}

static const struct vt_plane *g_nd;

static double view_score(struct depth *d, const struct vt_plane *sd, const struct m3_file *F, const struct depth_view *views, const struct depth_cfg *c, int scale)
{
    (void)scale;
    const double a = one_view_score(d, sd, &views[0], g_nd, c->w, c->h, F->tex_min), b = one_view_score(d, sd, &views[1], g_nd, c->w, c->h, F->tex_min);
    return a < b ? a + 0.25 * (b - a) : b + 0.25 * (a - b);
}

static void write_pgm(const char *path, const float *z, int w, int h, int stride, double near_z, double far_z)
{
    FILE *f = fopen(path, "wb");
    if (!f)
        return;
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const float v = z[(size_t)y * (size_t)stride + (size_t)x];
            int g = 0;
            if (v > 0.0f) {
                g = (int)(255.0 * (1.0 / v - 1.0 / far_z) / (1.0 / near_z - 1.0 / far_z));
                g = g < 16 ? 16 : g > 255 ? 255 : g;
            }
            fputc(g, f);
        }
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: m3_ref D.pgm R.pgm L.pgm CALIB.txt [--views 1|2] [--planes N] [--near M] [--far M] [--agg R] [--trunc T] [--conf C] [--tex T] [--write FILE] [--pgm PREFIX]\n");
        return 2;
    }
    static struct m3_file F;
    F.magic = M3_MAGIC;
    F.views = 2;
    F.planes = 64;
    F.agg_radius = 3;
    F.z_near = 0.5;
    F.z_far = 8.0;
    F.trunc = 1.5f;
    F.tex_min = 2.0f * VT_Q6;
    F.conf_min = 0.05f;
    F.eps = 1.0f * VT_Q6;
    F.frames = 1;
    F.avg = 128;
    F.motion_px = 0.25f;
    F.motion_grey = 2.0f;
    const char *write_file = NULL, *pgm = NULL, *refine_out = NULL, *seq_prefix = NULL, *dump_file = NULL;
    int source_l = 0, refine_rounds = 8, seq_n = 0;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--views") && i + 1 < argc) F.views = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq") && i + 2 < argc) {
            seq_prefix = argv[++i];
            seq_n = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--avg") && i + 1 < argc) F.avg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_file = argv[++i];
        else if (!strcmp(argv[i], "--refine") && i + 1 < argc) refine_out = argv[++i];
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) refine_rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--source-l")) source_l = 1;
        else if (!strcmp(argv[i], "--planes") && i + 1 < argc) F.planes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--near") && i + 1 < argc) F.z_near = atof(argv[++i]);
        else if (!strcmp(argv[i], "--far") && i + 1 < argc) F.z_far = atof(argv[++i]);
        else if (!strcmp(argv[i], "--agg") && i + 1 < argc) F.agg_radius = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trunc") && i + 1 < argc) F.trunc = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--conf") && i + 1 < argc) F.conf_min = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tex") && i + 1 < argc) F.tex_min = (float)atof(argv[++i]) * VT_Q6;
        else if (!strcmp(argv[i], "--write") && i + 1 < argc) write_file = argv[++i];
        else if (!strcmp(argv[i], "--pgm") && i + 1 < argc) pgm = argv[++i];
        else {
            fprintf(stderr, "m3_ref: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    int dw, dh, rw, rh, lw, lh;
    uint8_t *img_d = read_image(argv[1], &dw, &dh), *img_r = read_image(argv[2], &rw, &rh), *img_l = read_image(argv[3], &lw, &lh);
    if (!img_d || !img_r || !img_l)
        return 2;
    struct vt_calib cal;
    if (calib_read(argv[4], &cal) || !cal.valid || !cal.valid_d) {
        fprintf(stderr, "m3_ref: %s has no geometry for all three cameras\n", argv[4]);
        return 2;
    }
    F.d = (struct m3_cam){.w = dw, .h = dh, .f = cal.fd, .cx = cal.cxd, .cy = cal.cyd};
    F.r = (struct m3_cam){.w = rw, .h = rh, .f = cal.fb, .cx = cal.cxb, .cy = cal.cyb, .weight = DEPTH_WEIGHT_R};
    F.l = (struct m3_cam){.w = lw, .h = lh, .f = cal.fa, .cx = cal.cxa, .cy = cal.cya, .weight = DEPTH_WEIGHT_L};
    /* At level 0, as the file carries them. */
    struct depth_view whole[2];
    depth_views_from_calib(&cal, 1, NULL, NULL, whole);
    memcpy(F.r.R, whole[0].R, sizeof F.r.R);
    memcpy(F.r.T, whole[0].T, sizeof F.r.T);
    memcpy(F.l.R, whole[1].R, sizeof F.l.R);
    memcpy(F.l.T, whole[1].T, sizeof F.l.T);
    /* One source view, and it is L: its image and geometry take the first place. */
    if (source_l) {
        F.r = F.l;
        F.r.weight = 1.0f;
        F.views = 1;
        uint8_t *t = img_r;
        img_r = img_l;
        img_l = t;
        int tw = rw, th = rh;
        rw = lw;
        rh = lh;
        lw = tw;
        lh = th;
    }
    /* A run of real frames first, PREFIX-i-{d,r,l}.pgm, and then the frames named on the line: the map comes from what they leave in the still pictures, as it does live. */
    static uint8_t *imgs[M3_MAX_FRAMES][3];
    if (seq_n < 0 || seq_n + 1 > M3_MAX_FRAMES) {
        fprintf(stderr, "m3_ref: at most %d frames\n", M3_MAX_FRAMES);
        return 2;
    }
    for (int i = 0; i < seq_n; i++) {
        char path[256];
        int w, h;
        for (int k = 0; k < 3; k++) {
            snprintf(path, sizeof path, "%s-%d-%c.pgm", seq_prefix, i, "drl"[k]);
            imgs[i][k] = read_image(path, &w, &h);
            if (!imgs[i][k] || w != (k == 0 ? dw : k == 1 ? rw : lw) || h != (k == 0 ? dh : k == 1 ? rh : lh)) {
                fprintf(stderr, "m3_ref: %s is missing or another size\n", path);
                return 2;
            }
        }
    }
    imgs[seq_n][0] = img_d;
    imgs[seq_n][1] = img_r;
    imgs[seq_n][2] = img_l;
    F.frames = seq_n + 1;
    if (write_file) {
        FILE *f = fopen(write_file, "wb");
        if (!f || fwrite(&F, sizeof F, 1, f) != 1)
            return 2;
        for (int i = 0; i < F.frames; i++)
            if (fwrite(imgs[i][0], (size_t)dw * (size_t)dh, 1, f) != 1 || fwrite(imgs[i][1], (size_t)rw * (size_t)rh, 1, f) != 1 || fwrite(imgs[i][2], (size_t)lw * (size_t)lh, 1, f) != 1)
                return 2;
        fclose(f);
    }
    struct pyr pd, pr, pl;
    make_pyr(&pd, dw, dh);
    make_pyr(&pr, rw, rh);
    make_pyr(&pl, lw, lh);
    build_pyr(&pd, img_d, dw);
    build_pyr(&pr, img_r, rw);
    build_pyr(&pl, img_l, lw);
    const int lv = M3_LEVEL_D, scale = 1 << lv;
    const struct vt_plane *ld = &pd.l[lv], *lr = &pr.l[lv], *ll = &pl.l[lv];
    if (F.frames > 1) {
        const struct still_cfg sc_d = {.w = dw >> lv, .h = dh >> lv, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = 1};
        const struct still_cfg sc_r = {.w = rw >> lv, .h = rh >> lv, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = 1};
        const struct still_cfg sc_l = {.w = lw >> lv, .h = lh >> lv, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = 1};
        struct still *st[3] = {still_create(&sc_l, host_alloc, NULL), still_create(&sc_r, host_alloc, NULL), still_create(&sc_d, host_alloc, NULL)};
        if (!st[0] || !st[1] || !st[2])
            return 2;
        const struct m3_cam *const lens[3] = {&F.l, &F.r, &F.d};
        struct att *att[3];
        struct att_fix fix[3];
        memset(fix, 0, sizeof fix);
        uint64_t att_h = 0xcbf29ce484222325ull;
        for (int k = 0; k < 3; k++) {
            const struct att_cfg ac = {lens[k]->w >> lv, lens[k]->h >> lv, lens[k]->f / (1 << lv), lens[k]->cx / (1 << lv), lens[k]->cy / (1 << lv), F.eps, 1};
            att[k] = att_create(&ac, host_alloc, NULL);
            if (!att[k])
                return 2;
        }
        uint64_t still_h = VT_FNV_INIT;
        /* Where card and host part ways, when they do: every cell of every frame, as vt_m3's third argument writes them. */
        FILE *dump = dump_file ? fopen(dump_file, "wb") : NULL;
        for (int i = 0; i < F.frames; i++) {
            build_pyr(&pd, imgs[i][0], dw);
            build_pyr(&pr, imgs[i][1], rw);
            build_pyr(&pl, imgs[i][2], lw);
            const struct vt_plane *cur[3] = {ll, lr, ld};
            for (int k = 0; k < 3; k++) {
                if (!still_first(st[k]))
                    still_vectors(st[k], cur[k], 0, 1);
                still_decide(st[k]);
                still_update(st[k], cur[k], 0, 1);
                still_finish(st[k]);
            }
            struct vt_motion m[3];
            for (int k = 0; k < 3; k++) {
                still_motion(st[k], &m[k]);
                still_h = still_hash(st[k], still_h);
                if (dump)
                    fwrite(still_cells(st[k]), still_cell_bytes(st[k]), 1, dump);
            }
            printf("still_frame %d moving %u %u %u tracked %u %u %u noise %.3f %.3f %.3f\n", i, m[0].moving, m[1].moving, m[2].moving, m[0].tracked, m[1].tracked, m[2].tracked, (double)m[0].diff_median, (double)m[1].diff_median, (double)m[2].diff_median);
            /* The attitude kernel, as live: the still picture after a few frames is the reference, each later one measured against it. */
            for (int k = 0; k < 3; k++) {
                if (i < M3_ATT_REF_FRAME)
                    continue;
                if (i == M3_ATT_REF_FRAME) {
                    att_set_reference(att[k], still_mean(st[k]), att_one_thread, NULL);
                } else {
                    att_measure(att[k], still_mean(st[k]), 1, att_one_thread, NULL, &fix[k]);
                    att_h = att_fix_hash(&fix[k], att_h);
                }
            }
        }
        printf("still_hash %016llx\n", (unsigned long long)still_h);
        if (F.frames > M3_ATT_REF_FRAME + 1) {
            for (int k = 0; k < 3; k++)
                printf("att_last %d q %.9f %.9f %.9f %.9f corr %.6f none %.6f used %.6f view %.6f held %d\n", k, fix[k].q[0], fix[k].q[1], fix[k].q[2], fix[k].q[3], fix[k].corr, fix[k].corr_identity, fix[k].used, fix[k].view, fix[k].held);
            printf("att_hash %016llx\n", (unsigned long long)att_h);
        }
        if (dump)
            fclose(dump);
        ll = still_mean(st[0]);
        lr = still_mean(st[1]);
        ld = still_mean(st[2]);
    }
    struct vt_plane nd = plane_at(aligned_alloc(64, plane_bytes(ld->w, ld->h)), ld->w, ld->h), sd = plane_at(aligned_alloc(64, plane_bytes(ld->w, ld->h)), ld->w, ld->h);
    struct vt_plane nr = plane_at(aligned_alloc(64, plane_bytes(lr->w, lr->h)), lr->w, lr->h), nl = plane_at(aligned_alloc(64, plane_bytes(ll->w, ll->h)), ll->w, ll->h);
    void *sc = aligned_alloc(64, depth_norm_scratch_bytes(rw >> lv));
    depth_normalize(&nd, &sd, ld, F.eps, 0, ld->h, sc);
    depth_normalize(&nr, NULL, lr, F.eps, 0, lr->h, sc);
    depth_normalize(&nl, NULL, ll, F.eps, 0, ll->h, sc);
    struct depth_cfg cfg = {.w = ld->w, .h = ld->h, .f = F.d.f / scale, .cx = F.d.cx / scale, .cy = F.d.cy / scale, .planes = F.planes, .z_near = F.z_near, .z_far = F.z_far, .agg_radius = F.agg_radius, .trunc = F.trunc, .tex_min = F.tex_min, .conf_min = F.conf_min, .threads = 1};
    struct depth *d = depth_create(&cfg, host_alloc, NULL);
    if (!d)
        return 2;
    struct depth_view views[2] = {{.norm = &nr, .f = F.r.f / scale, .cx = F.r.cx / scale, .cy = F.r.cy / scale, .weight = F.r.weight}, {.norm = &nl, .f = F.l.f / scale, .cx = F.l.cx / scale, .cy = F.l.cy / scale, .weight = F.l.weight}};
    memcpy(views[0].R, F.r.R, sizeof views[0].R);
    memcpy(views[0].T, F.r.T, sizeof views[0].T);
    memcpy(views[1].R, F.l.R, sizeof views[1].R);
    memcpy(views[1].T, F.l.T, sizeof views[1].T);
    depth_set_views(d, &nd, &sd, views, F.views);
    g_nd = &nd;
    if (refine_out) {
        /* The reference camera's place and focal length, by coordinate descent on how much of its picture the other views agree with at some depth. A rotated control lost it steadily: 16.65 % at the fix, 14.16 at 1.5 degrees, 11.14 at 3, 7.06 at 6. */
        double best = view_score(d, &sd, &F, views, &cfg, scale);
        const double start = best;
        double rot = 0.008, pos = 0.04, fstep = 0.02;
        for (int round = 0; round < refine_rounds; round++) {
            for (int k = 0; k < 7; k++)
                for (int sign = -1; sign <= 1; sign += 2) {
                    struct depth_view trial[2];
                    memcpy(trial, views, sizeof trial);
                    struct depth_cfg c2 = cfg;
                    if (k < 3) {
                        double E[9], Rn[9];
                        const double dv = sign * rot;
                        star_rotation(k == 0 ? dv : 0.0, k == 1 ? dv : 0.0, k == 2 ? dv : 0.0, E);
                        /* The reference turns by E: X_ref' = E X_ref. So X_v = R E' X_ref' + T. */
                        for (int v = 0; v < 2; v++) {
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++)
                                    Rn[3 * r + c] = trial[v].R[3 * r] * E[3 * c] + trial[v].R[3 * r + 1] * E[3 * c + 1] + trial[v].R[3 * r + 2] * E[3 * c + 2];
                            memcpy(trial[v].R, Rn, sizeof Rn);
                        }
                    } else if (k < 6) {
                        /* The reference moves by d: X_ref' = X_ref + d. So X_v = R X_ref' + T - R d. */
                        double dp[3] = {0, 0, 0};
                        dp[k - 3] = sign * pos;
                        for (int v = 0; v < 2; v++)
                            for (int r = 0; r < 3; r++)
                                trial[v].T[r] -= trial[v].R[3 * r] * dp[0] + trial[v].R[3 * r + 1] * dp[1] + trial[v].R[3 * r + 2] * dp[2];
                    } else {
                        c2.f = cfg.f * (1.0 + sign * fstep);
                    }
                    depth_set_ref(d, c2.f, c2.cx, c2.cy);
                    const double s = view_score(d, &sd, &F, trial, &c2, scale);
                    if (s > best) {
                        best = s;
                        memcpy(views, trial, sizeof trial);
                        cfg = c2;
                    }
                }
            depth_set_ref(d, cfg.f, cfg.cx, cfg.cy);
            printf("refine round %d: score %.3f, steps %.3f deg %.3f m %.2f %%; fd %.1f\n", round, best, rot * 180.0 / M_PI, pos, 100.0 * fstep, cfg.f * scale);
            rot *= 0.7;
            pos *= 0.7;
            fstep *= 0.7;
        }
        printf("refined: score %.3f from %.3f\n", best, start);
        /* Back into the world: X_l = R1 X_d + T1, so X_d = R1' X_l - R1' T1; and L -> R unchanged. */
        struct vt_calib o = cal;
        const struct depth_view *vl = &views[1];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                o.Rd[3 * r + c] = vl->R[3 * c + r];
            o.Td[r] = -(vl->R[r] * vl->T[0] + vl->R[3 + r] * vl->T[1] + vl->R[6 + r] * vl->T[2]);
        }
        o.fd = cfg.f * scale;
        char note[160];
        snprintf(note, sizeof note, "L-R by landmarks; D by LED + armrest resection, refined on the room: score %.2f from %.2f", best, start);
        if (calib_write(refine_out, &o, note)) {
            perror(refine_out);
            return 1;
        }
        depth_set_views(d, &nd, &sd, views, F.views);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    depth_sweep(d, 0, 1);
    depth_aggregate(d, 0, 1);
    depth_decide(d, 0, 1);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    const float *z = depth_z(d), *conf = depth_conf(d);
    const int st = depth_stride(d);
    uint64_t hash = VT_FNV_INIT;
    long known = 0;
    double sum_z = 0.0, sum_c = 0.0;
    for (int y = 0; y < cfg.h; y++) {
        hash = vt_fnv(hash, z + (size_t)y * (size_t)st, (size_t)cfg.w * sizeof(float));
        hash = vt_fnv(hash, conf + (size_t)y * (size_t)st, (size_t)cfg.w * sizeof(float));
        for (int x = 0; x < cfg.w; x++) {
            const float v = z[(size_t)y * (size_t)st + (size_t)x];
            if (v > 0.0f) {
                known++;
                sum_z += (double)v;
                sum_c += (double)conf[(size_t)y * (size_t)st + (size_t)x];
            }
        }
    }
    printf("depth_hash %016llx\n", (unsigned long long)hash);
    /* How well the views agree at their best depth: what a right geometry should raise. Over textured reference pixels only. */
    double agree = 0.0, conf_sum = 0.0;
    long tex = 0, sure20 = 0;
    for (int y = 0; y < cfg.h; y++) {
        const float *sp = plane_row(&sd, y);
        for (int x = 0; x < cfg.w; x++)
            if (sp[x] >= F.tex_min) {
                const float cv = conf[(size_t)y * (size_t)st + (size_t)x];
                conf_sum += (double)cv;
                sure20 += cv >= 0.2f;
                tex++;
            }
    }
    agree = tex ? conf_sum / (double)tex : 0.0;
    printf("textured %ld pixels: mean confidence %.4f, confidence >= 0.2 on %.2f %%\n", tex, agree, tex ? 100.0 * (double)sure20 / (double)tex : 0.0);
    printf("%d x %d, %ld pixels with depth = %.1f %%, mean depth %.2f m, mean confidence %.3f; views %d planes %d near %.2f far %.2f agg %d trunc %.2f conf %.2f; host lane model one thread %.2f s\n", cfg.w, cfg.h, known, 100.0 * (double)known / ((double)cfg.w * cfg.h), known ? sum_z / (double)known : 0.0, known ? sum_c / (double)known : 0.0, F.views, F.planes, F.z_near, F.z_far, F.agg_radius, (double)F.trunc, (double)F.conf_min, (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec));
    if (pgm) {
        char path[256];
        snprintf(path, sizeof path, "%s-depth.pgm", pgm);
        write_pgm(path, z, cfg.w, cfg.h, st, F.z_near, F.z_far);
    }
    return 0;
}
