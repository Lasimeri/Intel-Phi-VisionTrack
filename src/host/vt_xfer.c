/* vt_xfer.c: the transport measured on the desktop host against a running vtrackd on the card.
 *   vt_xfer OUT.csv
 * Small-message round trip (host word to card, card word back), then host-to-card and card-to-host copies by the card's DMA
 * channel at each size, timed by the host around the request and by the card around the copy. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vt_link.h"
#include "vt_util.h"

static uint64_t seq_now;

static uint64_t request(struct vt_ctl *c, uint64_t op, uint64_t bytes, uint64_t *card_cycles)
{
    const uint64_t s = ++seq_now;
    c->xfer_cmd = op;
    c->xfer_bytes = bytes;
    _mm_sfence();
    const uint64_t t0 = vt_now_ns();
    c->xfer_seq = s;
    while (c->xfer_done != s)
        _mm_pause();
    const uint64_t t1 = vt_now_ns();
    if (card_cycles)
        *card_cycles = c->xfer_card_cycles;
    return t1 - t0;
}

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv)
{
    FILE *csv = fopen(argc > 1 ? argv[1] : "/dev/stdout", "w");
    struct vt_link l;
    if (!csv || vt_link_open(&l))
        return 2;
    char pcie[96];
    vt_link_pcie(pcie, sizeof pcie);
    printf("host: PCIe link %s\n", pcie);
    struct vt_config cfg = {.kind = VT_SESSION_XFER};
    if (vt_session_start(&l, &cfg, 20.0))
        return 3;
    struct vt_ctl *c = l.ctl;
    fprintf(csv, "test,bytes,reps,host_median_us,host_p95_us,host_p99_us,host_min_us,host_max_us,card_median_us,mb_s_host,mb_s_card\n");
    enum { REPS = 2000 };
    static uint64_t t[REPS], cc[REPS];
    /* A fresh pattern each copy, so nothing is served by a stale line and the region really carries data. */
    uint8_t *buf = l.part + VT_XFER_OFFSET;
    memset(buf, 0x5a, VT_XFER_BYTES);
    const double hz = 1238089000.0; /* the card's TSC as measured by vt_m0: card-side times only */
#define EMIT(name, bytes, reps)                                                                                                     \
    do {                                                                                                                            \
        qsort(t, (size_t)(reps), 8, cmp_u64);                                                                                       \
        qsort(cc, (size_t)(reps), 8, cmp_u64);                                                                                      \
        const double med = (double)t[(reps) / 2] * 1e-3, card = (double)cc[(reps) / 2] / hz * 1e6;                                  \
        fprintf(csv, "%s,%llu,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f\n", name, (unsigned long long)(bytes), (reps), med,         \
                (double)t[(reps) * 95 / 100] * 1e-3, (double)t[(reps) * 99 / 100] * 1e-3, (double)t[0] * 1e-3, (double)t[(reps) - 1] * 1e-3, \
                card, (double)(bytes) / med, card > 0 ? (double)(bytes) / card : 0.0);                                              \
    } while (0)
    for (int i = 0; i < REPS; i++)
        t[i] = request(c, VT_XFER_PING, 0, &cc[i]);
    EMIT("ping", 0ull, REPS);
    static const uint64_t sizes[] = {64, 1024, 4096, 16384, 65536, 262144, 524288, 1048576, 2097152, 307200, 921600, 2073600, 4147200, 8294400};
    const int nsizes = (int)(sizeof sizes / sizeof sizes[0]);
    for (int round = 0; round < 2; round++) {
        for (int s = 0; s < nsizes; s++) {
            const uint64_t bytes = (sizes[s] + 63) & ~63ull;
            const int reps = bytes >= (1u << 20) ? 100 : 500;
            for (int i = 0; i < reps; i++) {
                buf[(size_t)i * 64 % bytes] = (uint8_t)i;
                t[i] = request(c, VT_XFER_H2C, bytes, &cc[i]);
            }
            EMIT("h2c", bytes, reps);
            for (int i = 0; i < reps; i++)
                t[i] = request(c, VT_XFER_C2H, bytes, &cc[i]);
            EMIT("c2h", bytes, reps);
        }
    }
    fclose(csv);
    vt_session_end(&l);
    vt_session_wait_end(&l, 10.0);
    printf("host: done\n");
    vt_link_close(&l);
    return 0;
}
