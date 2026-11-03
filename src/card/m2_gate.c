/* m2_gate.c: M2's LK kernel on the card: known sub-pixel displacements through the pyramid, vector kernel against scalar on the same machine.
 * Prints a hash of every estimate (compared with the host's lane model by scripts/check.sh), endpoint errors against the exact truth, the
 * scalar-versus-vector difference, then times both.
 *
 *   vt_m2 W H [PATCH_W PATCH_H LEVELS] */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arena.h"
#include "img.h"
#include "lk.h"
#include "m2_input.h"
#include "team.h"
#include "topo.h"
#include "vt_util.h"

#define MAX_LEVELS 6
#define MAX_POINTS 4096
#define ROUNDS 7

static struct arena ar;
static double g_hz;

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
    struct vt_plane l[MAX_LEVELS];
};

static void make_pyr(struct pyr *p, int w, int h, int levels)
{
    for (int i = 0; i < levels; i++) {
        void *m = arena_take(&ar, plane_bytes(w, h), 64);
        if (!m)
            exit(2);
        p->l[i] = plane_at(m, w, h);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

static void build_pyr(const struct pyr *p, const uint8_t *img, int stride, int levels)
{
    img_u8_to_q6_vx(&p->l[0], img, stride, 0, p->l[0].h);
    img_pad_rows(&p->l[0], 0, p->l[0].h);
    img_pad_tb(&p->l[0]);
    for (int i = 1; i < levels; i++) {
        img_reduce_vx(&p->l[i], &p->l[i - 1], 0, p->l[i].h);
        img_pad_rows(&p->l[i], 0, p->l[i].h);
        img_pad_tb(&p->l[i]);
    }
}

static float px[MAX_POINTS], py[MAX_POINTS];
static struct lk_out out_v[MAX_POINTS], out_r[MAX_POINTS];

static int cmp_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

struct job {
    const struct pyr *a, *b;
    int top, n;
    const struct lk_cfg *cfg;
    char **tiles;
    int next __attribute__((aligned(64)));
};

static void team_lk(void *arg, int tid, int nthreads)
{
    (void)nthreads;
    struct job *j = arg;
    for (;;) {
        const int at = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (at >= j->n)
            return;
        lk_track_vx(j->a->l, j->b->l, j->top, j->cfg, px[at], py[at], 0.0f, 0.0f, &out_v[at], j->tiles[tid]);
    }
}

int main(int argc, char **argv)
{
    const int w = argc > 1 ? atoi(argv[1]) : 640, h = argc > 2 ? atoi(argv[2]) : 480;
    struct lk_cfg cfg = {.w = argc > 3 ? atoi(argv[3]) : 21, .h = argc > 4 ? atoi(argv[4]) : 21, .max_iter = 10, .eps = 0.005f, .min_eig = 0.1f, .max_residual = 40.0f};
    const int levels = argc > 5 ? atoi(argv[5]) : 4;
    setvbuf(stdout, NULL, _IOLBF, 0);
    struct topo t;
    if (topo_discover(&t) || arena_open(&ar, 12u * plane_bytes(w, h) + (96u << 20)))
        return 2;
    int cpus[256], core_of[256];
    if (topo_select(&t, 0, 1, 1, cpus, core_of) != 1)
        return 2;
    topo_pin(cpus[0]);
    g_hz = tsc_hz();
    printf("topology  cpus %d cores %d usable %d; other_busy_cpus %d (card)\n", t.ncpus, t.ncores, t.nusable, topo_busy_cpus(200));
    const int stride = (w + 15) & ~15;
    uint8_t *img0 = arena_take(&ar, (size_t)stride * (size_t)h, 64), *img1 = arena_take(&ar, (size_t)stride * (size_t)h, 64);
    struct pyr pa, pb;
    make_pyr(&pa, w, h, levels);
    make_pyr(&pb, w, h, levels);
    /* Feature grid, away from the edges by more than the largest test displacement. */
    int n = 0;
    const int margin = 40 + cfg.w;
    for (int y = margin; y < h - margin && n < MAX_POINTS; y += 24)
        for (int x = margin; x < w - margin && n < MAX_POINTS; x += 24) {
            px[n] = (float)x;
            py[n] = (float)y;
            n++;
        }
    printf("points %d patch %dx%d levels %d frame %dx%d\n", n, cfg.w, cfg.h, levels, w, h);
    void *tile0 = arena_take(&ar, lk_tile_bytes(), 64);
    m2_render(img0, w, h, stride, M2_OFFSET, M2_OFFSET);
    build_pyr(&pa, img0, stride, levels);
    int failed = 0;
    for (int c = 0; c < M2_NCASES; c++) {
        const int dx = m2_cases[c].dx, dy = m2_cases[c].dy;
        /* The pattern moves by +d: the second image samples the fine grid at offset - d. */
        m2_render(img1, w, h, stride, M2_OFFSET - dx, M2_OFFSET - dy);
        build_pyr(&pb, img1, stride, levels);
        for (int i = 0; i < n; i++) {
            lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_v[i], tile0);
            lk_track_ref(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_r[i], tile0);
        }
        uint64_t hash = VT_FNV_INIT;
        printf("lk_first %d,%d x %08x y %08x eig %08x res %08x cxx %08x cxy %08x cyy %08x it %u st %u\n", dx, dy, vt_float_bits(out_v[0].x), vt_float_bits(out_v[0].y), vt_float_bits(out_v[0].min_eig), vt_float_bits(out_v[0].residual), vt_float_bits(out_v[0].cov_xx), vt_float_bits(out_v[0].cov_xy), vt_float_bits(out_v[0].cov_yy), out_v[0].iters, out_v[0].status);
        static float epe[MAX_POINTS];
        int ok = 0;
        double maxdiff = 0, sumdiff = 0, maxe = 0;
        for (int i = 0; i < n; i++) {
            const struct lk_out *v = &out_v[i], *r = &out_r[i];
            hash = vt_fnv(hash, v, sizeof *v);
            const double ex = (double)v->x - ((double)px[i] + 0.25 * dx), ey = (double)v->y - ((double)py[i] + 0.25 * dy);
            if (!v->status) {
                const double e = sqrt(ex * ex + ey * ey);
                epe[ok++] = (float)e;
                if (e > maxe)
                    maxe = e;
            }
            const double ddx = (double)v->x - (double)r->x, ddy = (double)v->y - (double)r->y;
            const double dd = sqrt(ddx * ddx + ddy * ddy);
            sumdiff += dd;
            if (dd > maxdiff)
                maxdiff = dd;
        }
        qsort(epe, (size_t)ok, sizeof(float), cmp_float);
        printf("lk_hash %d,%d %016llx\n", dx, dy, (unsigned long long)hash);
        printf("case d=(%.2f, %.2f)  ok %d of %d  EPE px median %.4f p95 %.4f max %.4f  mvex-vs-x87 px mean %.6f max %.6f (card)\n", 0.25 * dx, 0.25 * dy, ok, n, ok ? (double)epe[ok / 2] : 0.0,
               ok ? (double)epe[ok * 95 / 100] : 0.0, maxe, sumdiff / n, maxdiff);
        if (ok * 10 < n * 9 || (ok && epe[ok / 2] > 0.1f))
            failed = 1;
    }
    /* What the forward-backward check does to the hardest case: points that said OK and were wrong, before and after. */
    {
        const int dx = m2_cases[M2_NCASES - 1].dx, dy = m2_cases[M2_NCASES - 1].dy;
        int ok_before = 0, wrong_before = 0, ok_after = 0, wrong_after = 0;
        for (int i = 0; i < n; i++) {
            struct lk_out o;
            lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &o, tile0);
            const double ex = (double)o.x - ((double)px[i] + 0.25 * dx), ey = (double)o.y - ((double)py[i] + 0.25 * dy);
            const int wrong = sqrt(ex * ex + ey * ey) > 1.0;
            if (!o.status) {
                ok_before++;
                wrong_before += wrong;
            }
            lk_flowback_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.5f, &o, tile0);
            if (!o.status) {
                ok_after++;
                wrong_after += wrong;
            }
        }
        printf("flowback d=(%.2f, %.2f): said OK %d, of them wrong by over 1 px %d; with the check said OK %d, wrong %d (card)\n", 0.25 * dx, 0.25 * dy, ok_before, wrong_before, ok_after, wrong_after);
    }
    /* A case the pyramid holds, so the timing is of working tracks. Prefetch A/B, one thread: each pass walks all points, so the L2 cannot hold them. */
    m2_render(img1, w, h, stride, M2_OFFSET - 49, M2_OFFSET + 30);
    build_pyr(&pb, img1, stride, levels);
    {
        uint64_t a[ROUNDS], b[ROUNDS];
        for (int r = 0; r < ROUNDS; r++) {
            cfg.prefetch = 1;
            uint64_t t0 = vt_tsc();
            for (int i = 0; i < n; i++)
                lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_v[i], tile0);
            a[r] = vt_tsc() - t0;
            cfg.prefetch = 0;
            t0 = vt_tsc();
            for (int i = 0; i < n; i++)
                lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_v[i], tile0);
            b[r] = vt_tsc() - t0;
        }
        const double ma = (double)vt_median_u64(a, ROUNDS), mb = (double)vt_median_u64(b, ROUNDS);
        printf("patch_prefetch_ab us/point with %.2f [%.2f %.2f] without %.2f [%.2f %.2f] (card, one thread, median of %d, interleaved)\n", ma / g_hz / n * 1e6, (double)a[0] / g_hz / n * 1e6, (double)a[ROUNDS - 1] / g_hz / n * 1e6, mb / g_hz / n * 1e6, (double)b[0] / g_hz / n * 1e6, (double)b[ROUNDS - 1] / g_hz / n * 1e6, ROUNDS);
    }
    /* Timing on the last, hardest case: per point, one thread, interleaved. */
    uint64_t tv[ROUNDS], tr[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        uint64_t t0 = vt_tsc();
        for (int i = 0; i < n; i++)
            lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_v[i], tile0);
        tv[r] = vt_tsc() - t0;
        t0 = vt_tsc();
        for (int i = 0; i < n; i++)
            lk_track_ref(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_r[i], tile0);
        tr[r] = vt_tsc() - t0;
    }
    long iters = 0;
    for (int i = 0; i < n; i++)
        iters += out_v[i].iters;
    const double mv = (double)vt_median_u64(tv, ROUNDS), mr = (double)vt_median_u64(tr, ROUNDS);
    printf("csv,variant,threads,tpc,points,iters_per_point,us_per_point,points_per_s\n");
    printf("csv,mvex,1,1,%d,%.1f,%.2f,%.0f\n", n, (double)iters / n, mv / g_hz / n * 1e6, n * g_hz / mv);
    printf("csv,x87,1,1,%d,%.1f,%.2f,%.0f\n", n, (double)iters / n, mr / g_hz / n * 1e6, n * g_hz / mr);
    printf("one_thread_speedup %.2f x (card, LK whole, MVEX over x87; the 2x2 solves stay x87 in both)\n", mr / mv);
    /* Teams. */
    /* One scratch tile per thread at a pseudo-random line offset (identically aligned private blocks alias in the L2), once for every team size. */
    static char *tiles[256];
    for (int i = 0; i < 256; i++) {
        char *m = arena_take(&ar, lk_tile_bytes() + 64 * 1024, 64);
        if (!m)
            return 2;
        tiles[i] = m + ((size_t)i * 104729u % 1024u) * 64u;
    }
    for (int pf = 0; pf < 2; pf++)
    for (int tpc = 1; tpc <= 4; tpc++) {
        const int nth = topo_select(&t, 0, t.nusable, tpc, cpus, core_of);
        if (nth < 0)
            continue;
        struct team *tm = team_create(cpus, core_of, nth);
        static struct job j;
        j.a = &pa;
        j.b = &pb;
        j.top = levels - 1;
        j.n = n;
        cfg.prefetch = pf;
        j.cfg = &cfg;
        j.tiles = tiles;
        uint64_t tt[ROUNDS];
        uint64_t hv = VT_FNV_INIT;
        for (int r = 0; r < ROUNDS; r++) {
            j.next = 0;
            const uint64_t t0 = vt_tsc();
            team_run(tm, team_lk, &j);
            tt[r] = vt_tsc() - t0;
        }
        for (int i = 0; i < n; i++)
            hv = vt_fnv(hv, &out_v[i], sizeof out_v[i]);
        const double mt = (double)vt_median_u64(tt, ROUNDS);
        printf("csv,mvex_pf%d,%d,%d,%d,%.1f,%.3f,%.0f\n", pf, nth, tpc, n, (double)iters / n, mt / g_hz / n * 1e6, n * g_hz / mt);
        printf("team_hash threads %d %016llx\n", nth, (unsigned long long)hv);
        team_destroy(tm);
    }
    printf("m2_card_done %s\n", failed ? "FAILED" : "ok");
    return failed;
}
