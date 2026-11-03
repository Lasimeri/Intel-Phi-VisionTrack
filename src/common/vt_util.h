/* vt_util.h: small tools shared by card and host: the cycle counter, a 64-bit FNV-1a over raw bits, a
 * deterministic generator whose floats are built from bits (never from float arithmetic, which is x87 with
 * excess precision on the card and SSE on the host), medians. */
#ifndef VT_UTIL_H
#define VT_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t vt_tsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t)hi << 32 | lo;
}

/* A wait that gives the thread's issue slots to its core's other threads. */
static inline void vt_delay(unsigned cycles)
{
#if defined(__KNC__)
    __asm__ volatile("delay %0" : : "r"(cycles));
#else
    (void)cycles;
    __asm__ volatile("pause");
#endif
}

#define VT_FNV_INIT 0xcbf29ce484222325ull

static inline uint64_t vt_fnv(uint64_t h, const void *p, size_t n)
{
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* xorshift64*: the same stream everywhere. */
static inline uint64_t vt_rand(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* A float in +-[2^(exp-127), 2^(exp-126)), from bits. */
static inline float vt_rand_float(uint64_t *s, unsigned exp)
{
    const uint64_t r = vt_rand(s);
    const uint32_t bits = (uint32_t)(r >> 63) << 31 | exp << 23 | (uint32_t)(r & 0x7fffff);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline uint32_t vt_float_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/* The median of n uint64 values, by insertion sort: run records, tens of values. Reorders v. */
static inline uint64_t vt_median_u64(uint64_t *v, int n)
{
    for (int i = 1; i < n; i++) {
        const uint64_t x = v[i];
        int j = i;
        for (; j > 0 && v[j - 1] > x; j--)
            v[j] = v[j - 1];
        v[j] = x;
    }
    return v[n / 2];
}

#endif
