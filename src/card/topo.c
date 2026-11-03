/* topo.c: see topo.h. */
#define _GNU_SOURCE
#include "topo.h"

#include <ctype.h>
#include <dirent.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int read_int(const char *path, int *out)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    const int ok = fscanf(f, "%d", out) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

static int core_index(struct topo *t, int id)
{
    for (int i = 0; i < t->ncores; i++)
        if (t->core[i].id == id)
            return i;
    if (t->ncores == TOPO_MAX_CORES)
        return -1;
    t->core[t->ncores] = (struct topo_core){.id = id};
    return t->ncores++;
}

/* The running CPU of every kernel thread named phiblk*-poll: field 39 of /proc/PID/stat. */
static void find_pollers(struct topo *t)
{
    DIR *d = opendir("/proc");
    if (!d)
        return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!isdigit((unsigned char)e->d_name[0]) || t->npollers == 8)
            continue;
        char path[300], line[512];
        snprintf(path, sizeof path, "/proc/%s/stat", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        const size_t n = fread(line, 1, sizeof line - 1, f);
        fclose(f);
        line[n] = 0;
        const char *open = strchr(line, '(');
        const char *close = strrchr(line, ')');
        if (!open || !close || strncmp(open + 1, "phiblk", 6) != 0 || !strstr(open, "-poll"))
            continue;
        /* field 3 starts after ") "; the processor is field 39, 36 fields on */
        const char *p = close + 2;
        for (int field = 3; field < 39 && p; field++) {
            p = strchr(p, ' ');
            if (p)
                p++;
        }
        if (p)
            t->poller_cpu[t->npollers++] = atoi(p);
    }
    closedir(d);
}

int topo_discover(struct topo *t)
{
    memset(t, 0, sizeof *t);
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed)) {
        perror("topo: sched_getaffinity");
        return -1;
    }
    for (int cpu = 0; cpu < TOPO_MAX_CPUS; cpu++) {
        char path[96];
        int id;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        if (read_int(path, &id))
            continue;
        if (!CPU_ISSET((size_t)cpu, &allowed))
            continue;
        const int c = core_index(t, id);
        if (c < 0 || t->core[c].ncpus == TOPO_MAX_TPC) {
            fprintf(stderr, "topo: core %d of CPU %d does not fit\n", id, cpu);
            return -1;
        }
        t->core[c].cpu[t->core[c].ncpus++] = cpu;
        t->ncpus++;
    }
    if (!t->ncpus) {
        fprintf(stderr, "topo: no CPU found in sysfs\n");
        return -1;
    }
    find_pollers(t);
    for (int c = 0; c < t->ncores; c++) {
        for (int k = 0; k < t->core[c].ncpus; k++) {
            if (t->core[c].cpu[k] == 0)
                t->core[c].reserved |= 1;
            for (int p = 0; p < t->npollers; p++)
                if (t->core[c].cpu[k] == t->poller_cpu[p])
                    t->core[c].reserved |= 2;
        }
        if (!t->core[c].reserved)
            t->nusable++;
    }
    return 0;
}

int topo_select(const struct topo *t, int first_core, int ncores, int tpc, int *cpus, int *core_of)
{
    int skipped = 0, taken = 0, n = 0;
    for (int c = 0; c < t->ncores && taken < ncores; c++) {
        const struct topo_core *k = &t->core[c];
        if (k->reserved)
            continue;
        if (skipped++ < first_core)
            continue;
        if (k->ncpus < tpc)
            return -1;
        for (int i = 0; i < tpc; i++) {
            cpus[n] = k->cpu[i];
            if (core_of)
                core_of[n] = taken;
            n++;
        }
        taken++;
    }
    return taken == ncores ? n : -1;
}

int topo_pin(int cpu)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((size_t)cpu, &s);
    return sched_setaffinity(0, sizeof s, &s) ? -1 : 0;
}

static int read_stat(long long *busy, long long *total)
{
    FILE *f = fopen("/proc/stat", "r");
    if (!f)
        return -1;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int cpu;
        long long u, ni, sy, id, io, ir, si;
        if (sscanf(line, "cpu%d %lld %lld %lld %lld %lld %lld %lld", &cpu, &u, &ni, &sy, &id, &io, &ir, &si) == 8 && cpu < TOPO_MAX_CPUS) {
            busy[cpu] = u + ni + sy + ir + si;
            total[cpu] = busy[cpu] + id + io;
        }
    }
    fclose(f);
    return 0;
}

int topo_busy_cpus(int ms)
{
    static long long b0[TOPO_MAX_CPUS], t0[TOPO_MAX_CPUS], b1[TOPO_MAX_CPUS], t1[TOPO_MAX_CPUS];
    if (read_stat(b0, t0))
        return -1;
    struct timespec nap = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&nap, NULL);
    if (read_stat(b1, t1))
        return -1;
    int busy = 0;
    for (int c = 0; c < TOPO_MAX_CPUS; c++) {
        const long long dt = t1[c] - t0[c];
        if (dt > 0 && (b1[c] - b0[c]) * 2 > dt)
            busy++;
    }
    return busy;
}
