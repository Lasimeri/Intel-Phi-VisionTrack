/* team.h: pinned threads that run one function together, for the card's kernels. Workers spin between
 * rounds: a frame's short parallel steps cannot afford a futex each.
 *
 * Topology-aware fan-out: all hardware threads of a core spin on ONE `go` line, which the core's L1
 * already holds, so a round crosses the ring once per CORE, not once per thread. The cores form a
 * 4-ary tree. The way back: per-core bytes in one word, then one word per core up the tree. */
#ifndef VT_TEAM_H
#define VT_TEAM_H

struct team;

/* tid in 0 .. n-1. */
typedef void (*team_fn)(void *arg, int tid, int n);

/* n threads, pinned to cpus[tid], with core_of[tid] their core index (0-based, ascending, tid 0 on
 * core 0). The caller becomes tid 0 and is pinned too. NULL on failure. */
struct team *team_create(const int *cpus, const int *core_of, int n);
void team_destroy(struct team *t);
void team_run(struct team *t, team_fn fn, void *arg);
int team_size(const struct team *t);

/* Inside fn: waits until all n threads have arrived. Costs one contended line per arrival. */
void team_barrier(struct team *t);

/* A wait that gives the thread's issue slots to its core's other threads. */
void team_delay(unsigned cycles);

#endif
