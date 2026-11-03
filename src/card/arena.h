/* arena.h: 2 MiB huge pages for image buffers and DMA targets, every page's physical address known. */
#ifndef VT_ARENA_H
#define VT_ARENA_H

#include <stddef.h>
#include <stdint.h>

#define ARENA_PAGE (2u << 20)

struct arena {
    uint8_t *base;
    size_t bytes, used;
    uint64_t *phys; /* one per page */
    int pages;
};

/* Reserves bytes (rounded up to pages), populated and locked; 0, or -1 with a message on stderr. */
int arena_open(struct arena *a, size_t bytes);
/* The next n bytes at the given alignment (a power of two); NULL when the arena is full. */
void *arena_take(struct arena *a, size_t n, size_t align);
/* The physical address of an arena pointer. */
uint64_t arena_phys(const struct arena *a, const void *p);
void arena_close(struct arena *a);

#endif
