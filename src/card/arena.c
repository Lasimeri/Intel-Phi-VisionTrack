/* arena.c: see arena.h. */
#define _GNU_SOURCE
#include "arena.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int arena_open(struct arena *a, size_t bytes)
{
    const size_t len = (bytes + ARENA_PAGE - 1) / ARENA_PAGE * ARENA_PAGE;
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "arena: no %zu MiB of huge pages: reserve them first (scripts/card_run.sh does)\n", len >> 20);
        return -1;
    }
    mlock(p, len);
    a->base = p;
    a->bytes = len;
    a->used = 0;
    a->pages = (int)(len / ARENA_PAGE);
    a->phys = malloc(sizeof(uint64_t) * (size_t)a->pages);
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0 || !a->phys) {
        perror("arena: pagemap");
        return -1;
    }
    for (int i = 0; i < a->pages; i++) {
        uint64_t e = 0, e2 = 0;
        const uintptr_t v = (uintptr_t)a->base + (uintptr_t)i * ARENA_PAGE;
        if (pread(fd, &e, 8, (off_t)(v / 4096 * 8)) != 8 || pread(fd, &e2, 8, (off_t)((v + ARENA_PAGE - 4096) / 4096 * 8)) != 8 || !(e >> 63)) {
            fprintf(stderr, "arena: page %d has no frame\n", i);
            close(fd);
            return -1;
        }
        a->phys[i] = (e & ((1ULL << 55) - 1)) * 4096;
        /* one physical 2 MiB frame: its last 4 KiB must follow its first */
        if ((e2 & ((1ULL << 55) - 1)) * 4096 != a->phys[i] + ARENA_PAGE - 4096) {
            fprintf(stderr, "arena: page %d is not contiguous\n", i);
            close(fd);
            return -1;
        }
    }
    close(fd);
    return 0;
}

void *arena_take(struct arena *a, size_t n, size_t align)
{
    const size_t at = (a->used + align - 1) & ~(align - 1);
    if (at + n > a->bytes)
        return NULL;
    a->used = at + n;
    return a->base + at;
}

uint64_t arena_phys(const struct arena *a, const void *p)
{
    const size_t off = (size_t)((const uint8_t *)p - a->base);
    return a->phys[off / ARENA_PAGE] + off % ARENA_PAGE;
}

void arena_close(struct arena *a)
{
    munmap(a->base, a->bytes);
    free(a->phys);
    a->base = NULL;
}
