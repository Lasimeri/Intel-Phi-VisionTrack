/* m0_gate.c: Milestone 0 on the card. Discovers the topology, proves the vector operations match the host
 * model bit for bit, times one kernel as x87 scalar and as MVEX on one pinned thread, then runs it on every
 * usable core with a pin check, and times an empty team round.
 *
 *   vt_m0 [tpc]    threads per core for the team part, default 2 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#include "m0_kern.h"
#include "team.h"
#include "topo.h"
#include "vt_util.h"

#define N 4096L
#define ROUNDS 9

static float buf_a[N] __attribute__((aligned(64))), buf_b[N] __attribute__((aligned(64))), buf_c[N] __attribute__((aligned(64)));
static float base_c[N] __attribute__((aligned(64))), out_v[N] __attribute__((aligned(64))), out_s[N] __attribute__((aligned(64)));

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

struct team_job {
    long passes;
    const int *cpus;
    float *block[256];
    int cpu_ok[256];
    uint64_t cycles[256];
};

static void team_fma(void *arg, int tid, int n)
{
    (void)n;
    struct team_job *j = arg;
    j->cpu_ok[tid] = sched_getcpu() == j->cpus[tid];
    float *c = j->block[tid];
    const uint64_t t0 = vt_tsc();
    for (long p = 0; p < j->passes; p++)
        m0_fma_vx(c, buf_a, buf_b, N);
    j->cycles[tid] = vt_tsc() - t0;
}

static void team_empty(void *arg, int tid, int n)
{
    (void)arg;
    (void)tid;
    (void)n;
}

static void fill_buffers(void)
{
    uint64_t s = 7;
    for (long i = 0; i < N; i++) {
        buf_a[i] = vt_rand_float(&s, 120u);
        buf_b[i] = vt_rand_float(&s, 120u);
        base_c[i] = vt_rand_float(&s, 127u);
    }
}

int main(int argc, char **argv)
{
    const int tpc = argc > 1 ? atoi(argv[1]) : 2;
    setvbuf(stdout, NULL, _IOLBF, 0);

    struct topo t;
    if (topo_discover(&t))
        return 2;
    printf("topology  cpus %d cores %d usable %d (card)\n", t.ncpus, t.ncores, t.nusable);
    for (int c = 0; c < t.ncores; c++)
        if (t.core[c].reserved)
            printf("reserved  core_id %d cpus %d %d %d %d why %s%s\n", t.core[c].id, t.core[c].cpu[0], t.core[c].cpu[1], t.core[c].cpu[2],
                   t.core[c].cpu[3], (t.core[c].reserved & 1) ? "cpu0 " : "", (t.core[c].reserved & 2) ? "block-poller" : "");
    int histo[5] = {0};
    for (int c = 0; c < t.ncores; c++)
        histo[t.core[c].ncpus]++;
    printf("threads_per_core  1:%d 2:%d 3:%d 4:%d\n", histo[1], histo[2], histo[3], histo[4]);
    printf("pollers  %d on cpus", t.npollers);
    for (int i = 0; i < t.npollers; i++)
        printf(" %d", t.poller_cpu[i]);
    printf("\nother_busy_cpus  %d (over 200 ms, before the run)\n", topo_busy_cpus(200));

    /* The operations against the model. */
    uint64_t hash[M0_NOPS];
    m0_vx_selftest(hash);
    for (int i = 0; i < M0_NOPS; i++)
        printf("vxop %-12s %016llx\n", m0_op_name[i], (unsigned long long)hash[i]);
    uint32_t sub_first;
    const uint64_t sub = m0_vx_subnormal(&sub_first);
    printf("subnormal %016llx first_bits %08x\n", (unsigned long long)sub, sub_first);

    /* Output: MVEX against the model (bit exact, by hash), and against x87 scalar (counted). */
    fill_buffers();
    memcpy(out_v, base_c, sizeof out_v);
    memcpy(out_s, base_c, sizeof out_s);
    m0_fma_vx(out_v, buf_a, buf_b, N);
    m0_fma_scalar(out_s, buf_a, buf_b, N);
    long diff = 0;
    long maxulp = 0;
    for (long i = 0; i < N; i++) {
        const long d = labs((long)vt_float_bits(out_v[i]) - (long)vt_float_bits(out_s[i]));
        diff += d != 0;
        if (d > maxulp)
            maxulp = d;
    }
    printf("fma_hash %016llx (MVEX)\n", (unsigned long long)vt_fnv(VT_FNV_INIT, out_v, sizeof out_v));
    printf("scalar_vs_mvex  %ld of %ld lanes differ, max %ld ulp (card, x87 double rounding)\n", diff, N, maxulp);

    /* Speed on one thread of one core, nothing else of ours running, interleaved. */
    int cpus[256], core_of[256];
    const int one = topo_select(&t, 0, 1, 1, cpus, core_of);
    if (one != 1)
        return 2;
    topo_pin(cpus[0]);
    const double hz = tsc_hz();
    printf("tsc_hz  %.0f (against the card's CLOCK_MONOTONIC)\n", hz);
    for (int sz = 0; sz < 2; sz++) {
        /* 3 x 4 KiB stays in the 32 KiB L1; 3 x 16 KiB does not, and runs from the L2. */
        const long len = sz == 0 ? 1024 : N;
        const long pv = sz == 0 ? 8000 : 2000, ps = sz == 0 ? 400 : 100;
        const double elems_v = (double)(pv * len), elems_s = (double)(ps * len);
        uint64_t tv[ROUNDS], ts[ROUNDS];
        for (int r = 0; r < ROUNDS; r++) {
            memcpy(buf_c, base_c, sizeof buf_c);
            uint64_t t0 = vt_tsc();
            for (long p = 0; p < pv; p++)
                m0_fma_vx(buf_c, buf_a, buf_b, len);
            tv[r] = vt_tsc() - t0;
            memcpy(buf_c, base_c, sizeof buf_c);
            t0 = vt_tsc();
            for (long p = 0; p < ps; p++)
                m0_fma_scalar(buf_c, buf_a, buf_b, len);
            ts[r] = vt_tsc() - t0;
        }
        uint64_t lo_v = ~0ull, hi_v = 0, lo_s = ~0ull, hi_s = 0;
        for (int r = 0; r < ROUNDS; r++) {
            if (tv[r] < lo_v) lo_v = tv[r];
            if (tv[r] > hi_v) hi_v = tv[r];
            if (ts[r] < lo_s) lo_s = ts[r];
            if (ts[r] > hi_s) hi_s = ts[r];
        }
        const double mv = (double)vt_median_u64(tv, ROUNDS), ms = (double)vt_median_u64(ts, ROUNDS);
        const char *where = sz == 0 ? "L1" : "L2";
        printf("one_thread_mvex_%s    %.3f cycles/elem median of %d, min %.3f max %.3f (cpu %d)\n", where, mv / elems_v, ROUNDS, (double)lo_v / elems_v, (double)hi_v / elems_v, cpus[0]);
        printf("one_thread_scalar_%s  %.3f cycles/elem median of %d, min %.3f max %.3f (cpu %d)\n", where, ms / elems_s, ROUNDS, (double)lo_s / elems_s, (double)hi_s / elems_s, cpus[0]);
        printf("one_thread_speedup_%s %.2f x (MVEX over x87 scalar, same thread)\n", where, (ms / elems_s) / (mv / elems_v));
    }

    /* The team: every usable core. */
    const int n = topo_select(&t, 0, t.nusable, tpc, cpus, core_of);
    if (n < 0)
        return 2;
    struct team *tm = team_create(cpus, core_of, n);
    if (!tm)
        return 2;
    static struct team_job job;
    job.passes = 2000;
    job.cpus = cpus;
    /* Each thread's own buffer at a pseudo-random line offset: identically aligned private blocks alias
     * in the L2 at 120 threads (measured on this card, Cortex 2026-10-25, 75 vs 218 GFLOP/s). */
    void *mem;
    const size_t stride = N * sizeof(float) + (1u << 12);
    if (posix_memalign(&mem, 64, stride * 256 + (1u << 20)))
        return 2;
    for (int i = 0; i < n; i++) {
        job.block[i] = (float *)((char *)mem + stride * (size_t)i + ((size_t)i * 104729u % 8192u) * 64u);
        memcpy(job.block[i], base_c, N * sizeof(float));
    }
    team_run(tm, team_fma, &job); /* first touch, pin check */
    int pinned = 0;
    for (int i = 0; i < n; i++)
        pinned += job.cpu_ok[i];
    printf("team_pinned  %d of %d threads on their assigned cpu\n", pinned, n);
    uint64_t wall[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        const uint64_t t0 = vt_tsc();
        team_run(tm, team_fma, &job);
        wall[r] = vt_tsc() - t0;
    }
    const double wall_cy = (double)vt_median_u64(wall, ROUNDS);
    const double total_elems = (double)job.passes * (double)N * n;
    printf("team_fma  %.1f Gelem/s = %.1f GFLOP/s measured, median of %d, threads %d (peak bound 16 lanes x 2 flops x %d cores x %.3f GHz = %.0f GFLOP/s, not a prediction)\n",
           total_elems / (wall_cy / hz) / 1e9, 2.0 * total_elems / (wall_cy / hz) / 1e9, ROUNDS, n, t.nusable, hz / 1e9, 32.0 * t.nusable * hz / 1e9);
    uint64_t rounds_empty[9];
    for (int r = 0; r < 9; r++) {
        uint64_t t0 = vt_tsc();
        for (int k = 0; k < 2000; k++)
            team_run(tm, team_empty, NULL);
        rounds_empty[r] = (vt_tsc() - t0) / 2000;
    }
    const uint64_t round_cy = vt_median_u64(rounds_empty, 9);
    printf("team_round_empty  %llu cycles = %.2f us median of 9x2000, threads %d cores %d tpc %d\n", (unsigned long long)round_cy, (double)round_cy / hz * 1e6, n, t.nusable, tpc);
    team_destroy(tm);
    free(mem);
    printf("m0_card_done\n");
    return 0;
}
