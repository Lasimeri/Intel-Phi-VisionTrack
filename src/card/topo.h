/* topo.h: the card's CPU topology, discovered at run time from sysfs and /proc. Nothing in the
 * project names a core number: CPU 0's core and the block pollers' core are found, not assumed. */
#ifndef VT_TOPO_H
#define VT_TOPO_H

#define TOPO_MAX_CPUS 256
#define TOPO_MAX_CORES 64
#define TOPO_MAX_TPC 4

struct topo_core {
    int id;                  /* sysfs core_id */
    int ncpus;               /* online logical CPUs on the core */
    int cpu[TOPO_MAX_TPC];   /* ascending */
    int reserved;            /* bit 0: holds CPU 0; bit 1: runs a block poller */
};

struct topo {
    int ncpus;
    int ncores;
    int nusable;             /* cores with reserved == 0 */
    struct topo_core core[TOPO_MAX_CORES];
    int poller_cpu[8];
    int npollers;
};

/* 0, or -1 with a message on stderr. */
int topo_discover(struct topo *t);

/* Fills cpus[] with tpc threads on each of ncores usable cores, tid order core by core, and core_of[]
 * with each thread's core index 0 .. ncores-1. Returns the thread count, or -1 if the card has fewer
 * usable cores or fewer threads per core than asked. first_core skips usable cores: disjoint groups. */
int topo_select(const struct topo *t, int first_core, int ncores, int tpc, int *cpus, int *core_of);

/* Pins the calling thread to one CPU; 0 or -1. */
int topo_pin(int cpu);

/* CPUs above 50 % busy over a window of ms milliseconds: other load on the card, recorded beside a measurement. */
int topo_busy_cpus(int ms);

#endif
