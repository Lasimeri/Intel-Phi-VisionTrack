/* m3_gate.c: the depth map on the card: the same input and source as m3_ref on the host, a team of hardware threads, MVEX. Prints the hash m3_ref must match, and times each stage.
 *   vt_m3 INPUT_FILE [ROUNDS] */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arena.h"
#include "attitude.h"
#include "depth.h"
#include "img.h"
#include "m3_input.h"
#include "still.h"
#include "team.h"
#include "topo.h"
#include "vt_util.h"

static struct arena ar;

static double tsc_hz(void)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    const uint64_t t0 = vt_tsc();
    struct timespec nap = {0, 200000000};
    nanosleep(&nap, NULL);
    const uint64_t t1 = vt_tsc();
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (double)(t1 - t0) / ((double)(b.tv_sec - a.tv_sec) + 1e-9 * (double)(b.tv_nsec - a.tv_nsec));
}

struct pyr {
    struct vt_plane l[M3_LEVELS];
};

static void make_pyr(struct pyr *p, int w, int h)
{
    for (int i = 0; i < M3_LEVELS; i++) {
        void *m = arena_take(&ar, plane_bytes(w, h), 64);
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

static void *arena_alloc(void *ctx, size_t bytes) { return arena_take(ctx, bytes, 64); }

struct norm_job {
    const struct vt_plane *dst, *spread, *src;
    float eps;
    char *scratch;
    size_t scratch_each;
};

static void team_norm(void *arg, int tid, int n)
{
    struct norm_job *j = arg;
    const int h = j->src->h;
    depth_normalize(j->dst, j->spread, j->src, j->eps, h * tid / n, h * (tid + 1) / n, j->scratch + (size_t)tid * j->scratch_each);
}

struct frame_job {
    struct still *s;
    const struct vt_plane *cur;
};

static void att_team_round(void *ctx, att_fn fn, void *arg)
{
    team_run(ctx, fn, arg);
    img_fence();
}

static void team_vectors(void *arg, int tid, int n)
{
    struct frame_job *j = arg;
    still_vectors(j->s, j->cur, tid, n);
}

static void team_update(void *arg, int tid, int n)
{
    struct frame_job *j = arg;
    still_update(j->s, j->cur, tid, n);
}

static void team_sweep(void *arg, int tid, int n) { depth_sweep(arg, tid, n); }
static void team_agg(void *arg, int tid, int n) { depth_aggregate(arg, tid, n); }
static void team_decide(void *arg, int tid, int n) { depth_decide(arg, tid, n); }

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return 2;
    const int rounds = argc > 2 ? atoi(argv[2]) : 9;
    setvbuf(stdout, NULL, _IOLBF, 0);
    static struct m3_file F;
    FILE *in = fopen(argv[1], "rb");
    if (!in || fread(&F, sizeof F, 1, in) != 1 || F.magic != M3_MAGIC) {
        fprintf(stderr, "vt_m3: bad input %s\n", argv[1]);
        return 2;
    }
    struct topo t;
    if (topo_discover(&t))
        return 2;
    int cpus[256], core_of[256];
    const int n = topo_select(&t, 0, t.nusable, 2, cpus, core_of);
    if (n < 1)
        return 2;
    const size_t img_bytes = ((size_t)F.d.w * (size_t)F.d.h + 2u * (size_t)F.r.w * (size_t)F.r.h + 256u) * (size_t)F.frames;
    const struct still_cfg sc_d = {.w = F.d.w >> M3_LEVEL_D, .h = F.d.h >> M3_LEVEL_D, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = n};
    const struct still_cfg sc_r = {.w = F.r.w >> M3_LEVEL_D, .h = F.r.h >> M3_LEVEL_D, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = n};
    const struct still_cfg sc_l = {.w = F.l.w >> M3_LEVEL_D, .h = F.l.h >> M3_LEVEL_D, .avg = F.avg, .motion_px = F.motion_px, .motion_grey = F.motion_grey, .threads = n};
    const size_t still_all = F.frames > 1 ? still_bytes(&sc_d) + still_bytes(&sc_r) + still_bytes(&sc_l) : 0;
    struct depth_cfg cfg ={.w = F.d.w >> M3_LEVEL_D, .h = F.d.h >> M3_LEVEL_D, .f = F.d.f / (1 << M3_LEVEL_D), .cx = F.d.cx / (1 << M3_LEVEL_D), .cy = F.d.cy / (1 << M3_LEVEL_D), .planes = F.planes, .z_near = F.z_near, .z_far = F.z_far, .agg_radius = F.agg_radius, .trunc = F.trunc, .tex_min = F.tex_min, .conf_min = F.conf_min, .threads = n};
    if (arena_open(&ar, img_bytes + still_all + depth_bytes(&cfg) + 8u * plane_bytes(F.r.w, F.r.h) + (64u << 20)))
        return 2;
    struct team *tm = team_create(cpus, core_of, n);
    if (!tm)
        return 2;
    const double hz = tsc_hz();
    printf("topology  cpus %d cores %d usable %d, team %d threads (card)\n", t.ncpus, t.ncores, t.nusable, n);
    static uint8_t *imgs[M3_MAX_FRAMES][3];
    if (F.frames < 1 || F.frames > M3_MAX_FRAMES)
        return 2;
    for (int i = 0; i < F.frames; i++) {
        uint8_t *img_d = arena_take(&ar, (size_t)F.d.w * (size_t)F.d.h, 64), *img_r = arena_take(&ar, (size_t)F.r.w * (size_t)F.r.h, 64), *img_l = arena_take(&ar, (size_t)F.l.w * (size_t)F.l.h, 64);
        if (!img_d || !img_r || !img_l || fread(img_d, (size_t)F.d.w * (size_t)F.d.h, 1, in) != 1 || fread(img_r, (size_t)F.r.w * (size_t)F.r.h, 1, in) != 1 || fread(img_l, (size_t)F.l.w * (size_t)F.l.h, 1, in) != 1)
            return 2;
        imgs[i][0] = img_d;
        imgs[i][1] = img_r;
        imgs[i][2] = img_l;
    }
    fclose(in);
    struct pyr pd, pr, pl;
    make_pyr(&pd, F.d.w, F.d.h);
    make_pyr(&pr, F.r.w, F.r.h);
    make_pyr(&pl, F.l.w, F.l.h);
    build_pyr(&pd, imgs[F.frames - 1][0], F.d.w);
    build_pyr(&pr, imgs[F.frames - 1][1], F.r.w);
    build_pyr(&pl, imgs[F.frames - 1][2], F.l.w);
    const int lv = M3_LEVEL_D, scale = 1 << lv;
    const struct vt_plane *ld = &pd.l[lv], *lr = &pr.l[lv], *ll = &pl.l[lv];
    if (F.frames > 1) {
        /* As live: each camera's frames in turn, the vectors deciding what its still picture takes in. */
        struct still *st[3] = {still_create(&sc_l, arena_alloc, &ar), still_create(&sc_r, arena_alloc, &ar), still_create(&sc_d, arena_alloc, &ar)};
        if (!st[0] || !st[1] || !st[2])
            return 2;
        const struct m3_cam *const lens[3] = {&F.l, &F.r, &F.d};
        struct att *att[3];
        struct att_fix fix[3];
        memset(fix, 0, sizeof fix);
        uint64_t att_h = 0xcbf29ce484222325ull, a_cycles = 0;
        for (int k = 0; k < 3; k++) {
            const struct att_cfg ac = {lens[k]->w >> lv, lens[k]->h >> lv, lens[k]->f / scale, lens[k]->cx / scale, lens[k]->cy / scale, F.eps, team_size(tm)};
            att[k] = att_create(&ac, arena_alloc, &ar);
            if (!att[k])
                return 2;
        }
        static uint64_t tv[M3_MAX_FRAMES];
        uint64_t still_h = VT_FNV_INIT;
        /* Where card and host part ways, when they do: every cell of every frame, as m3_ref --dump writes them. */
        FILE *dump = argc > 3 ? fopen(argv[3], "wb") : NULL;
        for (int i = 0; i < F.frames; i++) {
            build_pyr(&pd, imgs[i][0], F.d.w);
            build_pyr(&pr, imgs[i][1], F.r.w);
            build_pyr(&pl, imgs[i][2], F.l.w);
            img_fence(); /* tid 0's stores, before the team reads them */
            const struct vt_plane *cur[3] = {ll, lr, ld};
            const uint64_t t0 = vt_tsc();
            for (int k = 0; k < 3; k++) {
                struct frame_job j = {st[k], cur[k]};
                if (!still_first(st[k])) {
                    team_run(tm, team_vectors, &j);
                    img_fence();
                }
                still_decide(st[k]);
                team_run(tm, team_update, &j);
                img_fence();
                still_finish(st[k]);
            }
            tv[i] = vt_tsc() - t0;
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
                const uint64_t a0 = vt_tsc();
                if (i == M3_ATT_REF_FRAME) {
                    att_set_reference(att[k], still_mean(st[k]), att_team_round, tm);
                } else {
                    att_measure(att[k], still_mean(st[k]), 1, att_team_round, tm, &fix[k]);
                    att_h = att_fix_hash(&fix[k], att_h);
                    if (a_cycles < vt_tsc() - a0)
                        a_cycles = vt_tsc() - a0;
                }
            }
        }
        printf("still_hash %016llx\n", (unsigned long long)still_h);
        if (F.frames > M3_ATT_REF_FRAME + 1) {
            for (int k = 0; k < 3; k++)
                printf("att_last %d q %.9f %.9f %.9f %.9f corr %.6f none %.6f used %.6f view %.6f held %d\n", k, fix[k].q[0], fix[k].q[1], fix[k].q[2], fix[k].q[3], fix[k].corr, fix[k].corr_identity, fix[k].used, fix[k].view, fix[k].held);
            printf("att_hash %016llx\n", (unsigned long long)att_h);
            printf("attitude us, one camera, searching, worst (card): %.0f\n", (double)a_cycles * 1e6 / hz);
        }
        if (dump)
            fclose(dump);
        qsort(tv, (size_t)F.frames, 8, cmp_u64);
        const uint64_t tv_mid = tv[F.frames >> 1];
        printf("still us a frame of all three cameras, median of %d (card): %.0f\n", F.frames, (double)tv_mid * 1e6 / hz);
        ll = still_mean(st[0]);
        lr = still_mean(st[1]);
        ld = still_mean(st[2]);
    }
    struct vt_plane nd =plane_at(arena_take(&ar, plane_bytes(ld->w, ld->h), 64), ld->w, ld->h), sd = plane_at(arena_take(&ar, plane_bytes(ld->w, ld->h), 64), ld->w, ld->h);
    struct vt_plane nr = plane_at(arena_take(&ar, plane_bytes(lr->w, lr->h), 64), lr->w, lr->h), nl = plane_at(arena_take(&ar, plane_bytes(ll->w, ll->h), 64), ll->w, ll->h);
    const size_t sc_each = depth_norm_scratch_bytes(F.r.w >> lv);
    char *sc = arena_take(&ar, (size_t)n * sc_each, 64);
    struct depth *d = depth_create(&cfg, arena_alloc, &ar);
    if (!d || !sc)
        return 2;
    struct depth_view views[2] = {{.norm = &nr, .f = F.r.f / scale, .cx = F.r.cx / scale, .cy = F.r.cy / scale, .weight = F.r.weight}, {.norm = &nl, .f = F.l.f / scale, .cx = F.l.cx / scale, .cy = F.l.cy / scale, .weight = F.l.weight}};
    memcpy(views[0].R, F.r.R, sizeof views[0].R);
    memcpy(views[0].T, F.r.T, sizeof views[0].T);
    memcpy(views[1].R, F.l.R, sizeof views[1].R);
    memcpy(views[1].T, F.l.T, sizeof views[1].T);
    depth_set_views(d, &nd, &sd, views, F.views);
    static uint64_t tn[64], ts[64], ta[64], tc[64];
    uint64_t hash = 0;
    const int rr = rounds < 64 ? rounds : 64;
    for (int r = 0; r < rr; r++) {
        struct norm_job a = {&nd, &sd, ld, F.eps, sc, sc_each}, b = {&nr, NULL, lr, F.eps, sc, sc_each}, c = {&nl, NULL, ll, F.eps, sc, sc_each};
        const uint64_t t0 = vt_tsc();
        team_run(tm, team_norm, &a);
        team_run(tm, team_norm, &b);
        if (F.views > 1)
            team_run(tm, team_norm, &c);
        img_fence();
        const uint64_t t1 = vt_tsc();
        team_run(tm, team_sweep, d);
        img_fence();
        const uint64_t t2 = vt_tsc();
        team_run(tm, team_agg, d);
        img_fence();
        const uint64_t t3 = vt_tsc();
        team_run(tm, team_decide, d);
        const uint64_t t4 = vt_tsc();
        tn[r] = t1 - t0;
        ts[r] = t2 - t1;
        ta[r] = t3 - t2;
        tc[r] = t4 - t3;
        uint64_t h = VT_FNV_INIT;
        const float *z = depth_z(d), *conf = depth_conf(d);
        const int st = depth_stride(d);
        for (int y = 0; y < cfg.h; y++) {
            h = vt_fnv(h, z + (size_t)y * (size_t)st, (size_t)cfg.w * sizeof(float));
            h = vt_fnv(h, conf + (size_t)y * (size_t)st, (size_t)cfg.w * sizeof(float));
        }
        if (r && h != hash) {
            printf("depth_hash DIFFERS between rounds\n");
            return 1;
        }
        hash = h;
    }
    printf("depth_hash %016llx\n", (unsigned long long)hash);
    qsort(tn, (size_t)rr, 8, cmp_u64);
    qsort(ts, (size_t)rr, 8, cmp_u64);
    qsort(ta, (size_t)rr, 8, cmp_u64);
    qsort(tc, (size_t)rr, 8, cmp_u64);
    const double us = 1e6 / hz;
    const uint64_t mid_n = tn[rr / 2], mid_s = ts[rr / 2], mid_a = ta[rr / 2], mid_c = tc[rr / 2];
    printf("stage us median of %d (card): normalize %.0f sweep %.0f aggregate %.0f decide %.0f total %.0f; %dx%d, %d planes, %d views\n", rr, (double)mid_n * us, (double)mid_s * us, (double)mid_a * us, (double)mid_c * us, (double)(mid_n + mid_s + mid_a + mid_c) * us, cfg.w, cfg.h, cfg.planes, F.views);
    team_destroy(tm);
    arena_close(&ar);
    return 0;
}
