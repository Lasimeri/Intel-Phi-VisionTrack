/* team.c: see team.h. */
#define _GNU_SOURCE
#include "team.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

#include "topo.h"

#define LINE __attribute__((aligned(64)))
#define SPIN_CYCLES 8u
#define FANOUT 4

/* The round's whole order travels in the core's own line: go, the function, its argument, the team's
 * size. No thread reads any line that every core shares, so a round does not make all the cores queue
 * for one line (a shared fn/arg line, rewritten each round: 13.06 us for 118 threads, 2026-11-02). */
struct tcore {
    uint32_t go LINE;     /* written last by the parent core; read by every thread of this core */
    int n;
    team_fn fn;
    void *arg;
    uint32_t done LINE;   /* one byte per thread of this core, the core's leader excluded; same L1 */
    uint32_t up LINE;     /* written by this core's leader: its subtree has finished the round */
    uint32_t done_mask;   /* which bytes of done count */
    int nthreads;
    int child[FANOUT];
    int nchildren;
};

struct tthread {
    struct team *t;
    int tid, cpu, core, slot; /* slot: index among the core's threads; 0 = the core's leader */
    pthread_t thread;
};

struct team {
    int n, ncores;
    struct tcore *core;
    struct tthread *thr;
    uint32_t round;
    int count LINE; /* team_barrier: arrivals */
    int sense LINE; /* team_barrier: release */
};

void team_delay(unsigned cycles)
{
#if defined(__KNC__)
    __asm__ volatile("delay %0" : : "r"(cycles));
#else
    (void)cycles;
    __asm__ volatile("pause");
#endif
}

static inline uint32_t bytes_of(uint32_t round, uint32_t mask) { return (round & 0xffu) * 0x01010101u & mask; }

/* The round's way down from this core. The child's order is written before its go: x86 keeps store order. */
static inline void wake_children(struct team *t, const struct tcore *c, uint32_t round, team_fn fn, void *arg)
{
    for (int i = 0; i < c->nchildren; i++) {
        struct tcore *k = &t->core[c->child[i]];
        k->fn = fn;
        k->arg = arg;
        __atomic_store_n(&k->go, round, __ATOMIC_RELEASE);
    }
}

/* The leader's way back: its own core's threads, its child cores, then its own word up. */
static inline void leader_finish(struct team *t, struct tcore *c, uint32_t round)
{
    const uint32_t want = bytes_of(round, c->done_mask);
    while ((__atomic_load_n(&c->done, __ATOMIC_ACQUIRE) & c->done_mask) != want)
        team_delay(SPIN_CYCLES);
    for (int i = 0; i < c->nchildren; i++)
        while (__atomic_load_n(&t->core[c->child[i]].up, __ATOMIC_ACQUIRE) != round)
            team_delay(SPIN_CYCLES);
    __atomic_store_n(&c->up, round, __ATOMIC_RELEASE);
}

static void *worker_main(void *p)
{
    struct tthread *me = p;
    struct team *t = me->t;
    struct tcore *c = &t->core[me->core];
    const int tid = me->tid, slot = me->slot;
    topo_pin(me->cpu);
    uint32_t seen = 0;
    for (;;) {
        uint32_t g;
        while ((g = __atomic_load_n(&c->go, __ATOMIC_ACQUIRE)) == seen)
            team_delay(SPIN_CYCLES);
        seen = g;
        const team_fn fn = c->fn;
        void *const arg = c->arg;
        if (slot == 0)
            wake_children(t, c, g, fn, arg);
        if (!fn)
            return NULL;
        fn(arg, tid, c->n);
        if (slot == 0)
            leader_finish(t, c, g);
        else
            __atomic_store_n((uint8_t *)&c->done + slot, (uint8_t)g, __ATOMIC_RELEASE);
    }
}

struct team *team_create(const int *cpus, const int *core_of, int n)
{
    struct team *t;
    if (n < 1 || posix_memalign((void **)&t, 64, sizeof *t))
        return NULL;
    *t = (struct team){.n = n, .ncores = core_of[n - 1] + 1};
    if (posix_memalign((void **)&t->core, 64, sizeof(struct tcore) * (size_t)t->ncores) ||
        posix_memalign((void **)&t->thr, 64, sizeof(struct tthread) * (size_t)n))
        return NULL;
    for (int c = 0; c < t->ncores; c++) {
        t->core[c] = (struct tcore){.n = n};
        for (int k = 0; k < FANOUT; k++) {
            const int child = FANOUT * c + 1 + k;
            if (child < t->ncores)
                t->core[c].child[t->core[c].nchildren++] = child;
        }
    }
    for (int i = 0; i < n; i++) {
        struct tcore *c = &t->core[core_of[i]];
        const int slot = c->nthreads++;
        if (slot >= 4)
            return NULL;
        if (slot > 0)
            c->done_mask |= 0xffu << (8 * slot);
        t->thr[i] = (struct tthread){.t = t, .tid = i, .cpu = cpus[i], .core = core_of[i], .slot = slot};
    }
    topo_pin(cpus[0]);
    for (int i = 1; i < n; i++)
        if (pthread_create(&t->thr[i].thread, NULL, worker_main, &t->thr[i]))
            return NULL;
    return t;
}

int team_size(const struct team *t) { return t->n; }

void team_run(struct team *t, team_fn fn, void *arg)
{
    const uint32_t g = ++t->round;
    struct tcore *c = &t->core[0];
    c->fn = fn;
    c->arg = arg;
    __atomic_store_n(&c->go, g, __ATOMIC_RELEASE);
    wake_children(t, c, g, fn, arg);
    fn(arg, 0, t->n);
    leader_finish(t, c, g);
}

void team_barrier(struct team *t)
{
    const int s = __atomic_load_n(&t->sense, __ATOMIC_RELAXED);
    if (__atomic_add_fetch(&t->count, 1, __ATOMIC_ACQ_REL) == t->n) {
        __atomic_store_n(&t->count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&t->sense, !s, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&t->sense, __ATOMIC_ACQUIRE) == s)
            team_delay(SPIN_CYCLES);
    }
}

void team_destroy(struct team *t)
{
    const uint32_t g = ++t->round;
    struct tcore *c = &t->core[0];
    c->fn = NULL;
    c->arg = NULL;
    __atomic_store_n(&c->go, g, __ATOMIC_RELEASE);
    wake_children(t, c, g, NULL, NULL);
    for (int i = 1; i < t->n; i++)
        pthread_join(t->thr[i].thread, NULL);
    free(t->core);
    free(t->thr);
    free(t);
}
