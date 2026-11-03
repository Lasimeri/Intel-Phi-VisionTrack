/* m1_gate.c: M1 on the card: one frame through every image primitive. Prints the output hashes (compared with the
 * host's m1_ref by scripts/check.sh and against this binary's own scalar run), then times x87 scalar and native MVEX,
 * one pinned thread, interleaved, and a team of each thread count asked for. Results as CSV on stdout (lines "csv,").
 *
 *   vt_m1 W H [tpc-list as digits, default 1234]    kernel,w,h,threads,tpc,variant,median_us,min_us,max_us,mpix_s */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arena.h"
#include "img.h"
#include "m1_input.h"
#include "team.h"
#include "topo.h"

#define ROUNDS 7

static double g_hz;
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

static struct vt_plane new_plane(int w, int h)
{
    void *m = arena_take(&ar, plane_bytes(w, h), 64);
    if (!m) {
        fprintf(stderr, "m1: arena full\n");
        exit(2);
    }
    /* A pseudo-random line offset between planes would go here if planes aliased in the L2; none has so far. */
    return plane_at(m, w, h);
}

static uint64_t hash_plane(const struct vt_plane *pl)
{
    uint64_t h = VT_FNV_INIT;
    for (int y = 0; y < pl->h; y++)
        h = vt_fnv(h, plane_row(pl, y), (size_t)pl->w * sizeof(float));
    return h;
}

/* One kernel as a function of a row range, whatever its planes. */
struct job;
typedef void (*kern_fn)(const struct job *j, int y0, int y1);
struct job {
    const char *name;
    kern_fn ref, vx;
    int rows;         /* output rows to split */
    double pixels;    /* output pixels a run makes */
    const uint8_t *u8;
    int u8_stride;
    struct vt_plane a, b, c, d;
    float go[2];
    kern_fn run; /* team_main's choice */
    uint64_t expect; /* the single-thread output hash: a team must reproduce it */
};

#define WRAP(fname, call) \
    static void fname(const struct job *j, int y0, int y1) { call; }
WRAP(u8_ref, img_u8_to_q6_ref(&j->a, j->u8, j->u8_stride, y0, y1))
WRAP(u8_vx, img_u8_to_q6_vx(&j->a, j->u8, j->u8_stride, y0, y1))
WRAP(norm_ref, img_norm_ref(&j->a, &j->b, j->go, y0, y1))
WRAP(norm_vx, img_norm_vx(&j->a, &j->b, j->go, y0, y1))
WRAP(gauss_ref, img_gauss5_ref(&j->a, &j->b, y0, y1))
WRAP(gauss_vx, img_gauss5_vx(&j->a, &j->b, y0, y1))
WRAP(reduce_ref, img_reduce_ref(&j->a, &j->b, y0, y1))
WRAP(reduce_vx, img_reduce_vx(&j->a, &j->b, y0, y1))
WRAP(reduce_nopf, img_reduce_nopf_vx(&j->a, &j->b, y0, y1))
WRAP(sobel_ref, img_sobel_ref(&j->a, &j->c, &j->b, y0, y1))
WRAP(sobel_vx, img_sobel_vx(&j->a, &j->c, &j->b, y0, y1))
WRAP(diff_ref, img_absdiff_ref(&j->a, &j->b, &j->c, y0, y1))
WRAP(diff_vx, img_absdiff_vx(&j->a, &j->b, &j->c, y0, y1))

static void team_main(void *arg, int tid, int n)
{
    const struct job *j = arg;
    const int y0 = (int)((long)j->rows * tid / n), y1 = (int)((long)j->rows * (tid + 1) / n);
    j->run(j, y0, y1);
}

static int g_team_bad;
static uint64_t job_hash(const struct job *j)
{
    uint64_t h = hash_plane(&j->a);
    if (j->c.p && j->ref == sobel_ref)
        h ^= hash_plane(&j->c);
    return h;
}

static void report(const char *kern, int w, int h, int threads, int tpc, const char *variant, uint64_t *t, int n, double pixels)
{
    uint64_t lo = ~0ull, hi = 0;
    for (int i = 0; i < n; i++) {
        if (t[i] < lo) lo = t[i];
        if (t[i] > hi) hi = t[i];
    }
    const double med = (double)vt_median_u64(t, n) / g_hz;
    printf("csv,%s,%d,%d,%d,%d,%s,%.1f,%.1f,%.1f,%.1f\n", kern, w, h, threads, tpc, variant, med * 1e6, (double)lo / g_hz * 1e6, (double)hi / g_hz * 1e6, pixels / med / 1e6);
}

static void one_thread(struct job *j, int w, int h)
{
    uint64_t tr[ROUNDS], tv[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        uint64_t t0 = vt_tsc();
        j->vx(j, 0, j->rows);
        tv[r] = vt_tsc() - t0;
        t0 = vt_tsc();
        j->ref(j, 0, j->rows);
        tr[r] = vt_tsc() - t0;
    }
    report(j->name, w, h, 1, 1, "mvex", tv, ROUNDS, j->pixels);
    report(j->name, w, h, 1, 1, "scalar", tr, ROUNDS, j->pixels);
    j->expect = job_hash(j);
}

static void team_time(struct team *tm, struct job *j, int w, int h, int tpc, kern_fn f, const char *variant)
{
    uint64_t t[ROUNDS];
    j->run = f;
    for (int r = 0; r < ROUNDS; r++) {
        const uint64_t t0 = vt_tsc();
        team_run(tm, team_main, j);
        t[r] = vt_tsc() - t0;
    }
    report(j->name, w, h, team_size(tm), tpc, variant, t, ROUNDS, j->pixels);
    if (job_hash(j) != j->expect) {
        printf("MISMATCH team %s %s threads %d\n", j->name, variant, team_size(tm));
        g_team_bad = 1;
    }
}

int main(int argc, char **argv)
{
    const int w = argc > 1 ? atoi(argv[1]) : 640, h = argc > 2 ? atoi(argv[2]) : 480;
    const char *tpcs = argc > 3 ? argv[3] : "1234";
    setvbuf(stdout, NULL, _IOLBF, 0);
    struct topo t;
    if (topo_discover(&t))
        return 2;
    if (arena_open(&ar, 8u * plane_bytes(w, h) + (64u << 20)))
        return 2;
    int cpus[256], core_of[256];
    if (topo_select(&t, 0, 1, 1, cpus, core_of) != 1)
        return 2;
    topo_pin(cpus[0]);
    g_hz = tsc_hz();
    printf("topology  cpus %d cores %d usable %d; other_busy_cpus %d (card)\n", t.ncpus, t.ncores, t.nusable, topo_busy_cpus(200));

    const int stride8 = (w + 15) & ~15;
    uint8_t *img = arena_take(&ar, (size_t)stride8 * (size_t)h, 64);
    m1_make_image(img, w, h, stride8);
    const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
    struct vt_plane q = new_plane(w, h), g = new_plane(w, h), x1 = new_plane(w, h), x2 = new_plane(w, h), x3 = new_plane(w, h), x4 = new_plane(w, h);
    struct vt_plane r1 = new_plane(w2, h2), r2 = new_plane(w2, h2);

    /* Correctness: MVEX against this card's x87 reference; the hash lines go to the host's. */
    int bad = 0;
#define SAME(name, pl1, pl2)                                                                                     \
    do {                                                                                                         \
        const uint64_t h1 = hash_plane(&pl1), h2_ = hash_plane(&pl2);                                            \
        printf("kern %-12s %dx%d %016llx\n", name, w, h, (unsigned long long)h1);                                \
        if (h1 != h2_) {                                                                                         \
            printf("MISMATCH %s: x87 %016llx mvex %016llx\n", name, (unsigned long long)h1, (unsigned long long)h2_); \
            bad = 1;                                                                                             \
        }                                                                                                        \
    } while (0)
    img_u8_to_q6_ref(&q, img, stride8, 0, h);
    img_u8_to_q6_vx(&g, img, stride8, 0, h);
    img_pad_rows(&q, 0, h);
    img_pad_tb(&q);
    SAME("u8_to_q6", q, g);
    const float go[2] = {0.75f, 512.0f};
    img_norm_ref(&x1, &q, go, 0, h);
    img_norm_vx(&x2, &q, go, 0, h);
    SAME("norm", x1, x2);
    float m0[2], m1[2];
    img_minmax_ref(&q, m0, 0, h);
    img_minmax_vx(&q, m1, 0, h);
    printf("kern minmax       %dx%d %.1f %.1f\n", w, h, (double)m0[0], (double)m0[1]);
    if (m0[0] != m1[0] || m0[1] != m1[1]) {
        printf("MISMATCH minmax\n");
        bad = 1;
    }
    img_gauss5_ref(&x1, &q, 0, h);
    img_gauss5_vx(&x2, &q, 0, h);
    SAME("gauss5", x1, x2);
    img_reduce_ref(&r1, &q, 0, h2);
    img_reduce_vx(&r2, &q, 0, h2);
    SAME("reduce", r1, r2);
    img_sobel_ref(&x1, &x2, &q, 0, h);
    img_sobel_vx(&x3, &x4, &q, 0, h);
    SAME("sobel_x", x1, x3);
    SAME("sobel_y", x2, x4);
    img_gauss5_ref(&g, &q, 0, h);
    img_absdiff_ref(&x1, &q, &g, 0, h);
    img_absdiff_vx(&x2, &q, &g, 0, h);
    SAME("absdiff", x1, x2);
    printf("correctness %s (card, %dx%d, x87 vs MVEX, hash per kernel)\n", bad ? "FAILED" : "identical", w, h);

    const double px = (double)w * h, px2 = (double)w2 * h2;
    struct job jobs[] = {
        {.name = "u8_to_q6", .ref = u8_ref, .vx = u8_vx, .rows = h, .pixels = px, .u8 = img, .u8_stride = stride8, .a = x1},
        {.name = "norm", .ref = norm_ref, .vx = norm_vx, .rows = h, .pixels = px, .a = x1, .b = q, .go = {0.75f, 512.0f}},
        {.name = "gauss5", .ref = gauss_ref, .vx = gauss_vx, .rows = h, .pixels = px, .a = x1, .b = q},
        {.name = "reduce", .ref = reduce_ref, .vx = reduce_vx, .rows = h2, .pixels = px2, .a = r1, .b = q},
        {.name = "sobel", .ref = sobel_ref, .vx = sobel_vx, .rows = h, .pixels = px, .a = x1, .c = x2, .b = q},
        {.name = "absdiff", .ref = diff_ref, .vx = diff_vx, .rows = h, .pixels = px, .a = x1, .b = q, .c = g},
    };
    const int njobs = (int)(sizeof jobs / sizeof jobs[0]);
    printf("csv,kernel,w,h,threads,tpc,variant,median_us,min_us,max_us,mpix_s\n");
    for (int i = 0; i < njobs; i++)
        one_thread(&jobs[i], w, h);
    /* The reduce variants on one thread, interleaved with each other. */
    struct job *rj = &jobs[3];
    uint64_t ta[ROUNDS], tb[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        uint64_t t0 = vt_tsc();
        reduce_vx(rj, 0, rj->rows);
        ta[r] = vt_tsc() - t0;
        t0 = vt_tsc();
        reduce_nopf(rj, 0, rj->rows);
        tb[r] = vt_tsc() - t0;
    }
    report("reduce", w, h, 1, 1, "mvex_pf", ta, ROUNDS, px2);
    report("reduce", w, h, 1, 1, "mvex_nopf", tb, ROUNDS, px2);

    for (const char *p = tpcs; *p; p++) {
        const int tpc = *p - '0';
        const int n = topo_select(&t, 0, t.nusable, tpc, cpus, core_of);
        if (n < 0)
            continue;
        struct team *tm = team_create(cpus, core_of, n);
        if (!tm)
            return 2;
        for (int i = 0; i < njobs; i++)
            team_time(tm, &jobs[i], w, h, tpc, jobs[i].vx, "mvex");
        team_time(tm, rj, w, h, tpc, reduce_nopf, "mvex_nopf");
        team_destroy(tm);
    }
    /* Core scaling at 2 threads a core. */
    for (int cores = 1; cores <= t.nusable; cores = cores < 8 ? cores * 2 : cores + (cores < 32 ? 8 : 16)) {
        const int n = topo_select(&t, 0, cores, 2, cpus, core_of);
        if (n < 0)
            break;
        struct team *tm = team_create(cpus, core_of, n);
        if (!tm)
            return 2;
        team_time(tm, rj, w, h, 2, reduce_vx, "mvex_scale");
        team_time(tm, &jobs[2], w, h, 2, gauss_vx, "mvex_scale");
        team_destroy(tm);
    }
    bad |= g_team_bad;
    printf("team_outputs %s (card, every team run re-hashed against the single-thread output)\n", g_team_bad ? "FAILED" : "identical");
    printf("m1_card_done %s\n", bad ? "FAILED" : "ok");
    return bad;
}
