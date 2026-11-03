/* vtrackd: the card's long-lived program. Idle, one thread naps on the session word in host memory and every other thread of the
 * project is gone, so an idle vtrackd costs the card nothing. The host starts a session by writing a config and session_seq: the program
 * builds what the config asks (teams, buffers), serves it, tears it down, and naps again.
 *
 * Started detached by scripts/vtrackd.sh. Stopped by the quit word (host: vt_quit). */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "arena.h"
#include "session.h"
#include "topo.h"
#include "vt_dma.h"
#include "vt_format.h"
#include "vt_proto.h"

/* The region's swap guard: below 4 GiB of region the card's whole /dev/phiblk1 is swap, and writing a part would corrupt it. */
static int region_is_swap(void)
{
    FILE *f = fopen("/proc/swaps", "r");
    if (!f)
        return 0;
    char line[256];
    int bad = 0;
    while (fgets(line, sizeof line, f))
        if (strstr(line, "/dev/phiblk1"))
            bad = 1;
    fclose(f);
    return bad;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    printf("vtrackd: start (card)\n");
    if (region_is_swap()) {
        fprintf(stderr, "vtrackd: /dev/phiblk1 is swap: the shared region is not free; refused\n");
        return 3;
    }
    const int fd = open("/dev/phihost", O_RDWR);
    if (fd < 0) {
        perror("vtrackd: /dev/phihost");
        return 3;
    }
    struct vt_ctl *ctl = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)VT_PART_OFFSET);
    close(fd);
    if (ctl == MAP_FAILED) {
        perror("vtrackd: mmap /dev/phihost");
        return 3;
    }
    struct arena dma_page;
    if (arena_open(&dma_page, ARENA_PAGE))
        return 3;
    void *ring = arena_take(&dma_page, 8192, 4096);
    struct vt_dma dma;
    if (vt_dma_open(&dma, ring, arena_phys(&dma_page, ring)))
        return 3;
    printf("vtrackd: DMA channel %d, region at card address 0x%llx, %llu MiB (card)\n", dma.chan, (unsigned long long)dma.region_addr,
           (unsigned long long)(dma.region_bytes >> 20));
    struct topo topo;
    if (topo_discover(&topo))
        return 3;
    printf("vtrackd: %d cpus, %d cores, %d usable (card)\n", topo.ncpus, topo.ncores, topo.nusable);
    uint64_t seen = ctl->session_seq;
    ctl->card_session = 0;
    /* The last run's order to leave is still in host memory: it was for that run. */
    ctl->quit = 0;
    const uint64_t format = vt_format_hash();
    ctl->card_format = format;
    ctl->card_build = argc > 1 ? (uint64_t)atoll(argv[1]) : 0;
    printf("vtrackd: wire format %016llx (card)\n", (unsigned long long)format);
    while (!ctl->quit) {
        const uint64_t want = ctl->session_seq;
        if (want == seen) {
            struct timespec nap = {0, 2000000};
            nanosleep(&nap, NULL);
            continue;
        }
        seen = want;
        struct vt_config cfg;
        memcpy(&cfg, (const void *)&ctl->config, sizeof cfg);
        printf("vtrackd: session %llu kind %u (card)\n", (unsigned long long)want, cfg.kind);
        int rc;
        if (cfg.format_hash != format) {
            fprintf(stderr, "vtrackd: refused: the host's wire format is %016llx, mine %016llx\n", (unsigned long long)cfg.format_hash, (unsigned long long)format);
            ctl->card_session = (uint64_t)-(int64_t)want;
            continue;
        }
        switch (cfg.kind) {
        case VT_SESSION_XFER:
            rc = session_xfer(ctl, &dma, want);
            break;
        case VT_SESSION_TRACK:
            rc = session_track(ctl, &dma, &topo, &cfg, want);
            break;
        default:
            rc = -1;
        }
        printf("vtrackd: session %llu ended rc %d (card)\n", (unsigned long long)want, rc);
        ctl->card_session = rc == 0 ? 0 : (uint64_t)-(int64_t)want;
    }
    vt_dma_close(&dma);
    printf("vtrackd: quit (card)\n");
    return 0;
}
