/* vt_dma.c: see vt_dma.h. */
#define _GNU_SOURCE
#include "vt_dma.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "vt_util.h"

#define SBOX_DMA_PAGE 0x08007DA000ULL /* SBOX + 0xA000 */
#define REGION_HEADER 0x10000000ULL   /* phictl's header: "PHIR", the region's card address at +32, its size at +40 */
#define DCR_OFFSET 0x280
#define CHAN_STRIDE 0x40
#define R_DCAR 0x00
#define R_DHPR 0x04
#define R_DTPR 0x08
#define R_DRAR_LO 0x14
#define R_DRAR_HI 0x18
#define R_DCHERRMSK 0x30
#define DCAR_MASK_BOTH 0x03000000u /* both interrupt masks: the card kernel has no handler for this */
#define RING_DESC 256u             /* one 4 KiB page of 16-byte descriptors */
#define LINE_DESC 4u               /* copy, status for the host, status for this process, NOP */
#define IN_FLIGHT (RING_DESC / LINE_DESC - 1u)
#define ADDR_MASK ((1ULL << 40) - 1)
#define TYPE_COPY (1ULL << 60)
#define TYPE_STATUS (2ULL << 60)

/* 0 and 1 are phictl's. The others were each claimed by some project, none of which is required to be running: flock decides. */
static const int channels[] = {5, 3, 2, 6, 7, 4};

static inline uint32_t rd(const struct vt_dma *d, int off) { return d->regs[(d->chan * CHAN_STRIDE + off) / 4]; }
static inline void wr(struct vt_dma *d, int off, uint32_t v) { d->regs[(d->chan * CHAN_STRIDE + off) / 4] = v; }
static inline void full_fence(void) { __asm__ volatile("lock addq $0, (%%rsp)" ::: "memory", "cc"); }

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int take_lock(int chan)
{
    char path[64];
    snprintf(path, sizeof path, "/tmp/phi-cdma%d.lock", chan);
    const int fd = open(path, O_CREAT | O_RDWR, 0600);
    if (fd < 0)
        return -1;
    if (flock(fd, LOCK_EX | LOCK_NB)) {
        close(fd);
        return -1;
    }
    return fd;
}

int vt_dma_open(struct vt_dma *d, void *page, uint64_t page_phys)
{
    memset(d, 0, sizeof *d);
    d->lock_fd = -1;
    for (unsigned i = 0; i < sizeof channels / sizeof channels[0] && d->lock_fd < 0; i++) {
        d->lock_fd = take_lock(channels[i]);
        d->chan = channels[i];
    }
    if (d->lock_fd < 0) {
        fprintf(stderr, "vt_dma: every channel of the list is locked\n");
        return -1;
    }
    const int mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem < 0) {
        perror("vt_dma: /dev/mem");
        return -1;
    }
    void *regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)SBOX_DMA_PAGE);
    void *hdr = mmap(NULL, 4096, PROT_READ, MAP_SHARED, mem, (off_t)REGION_HEADER);
    close(mem);
    if (regs == MAP_FAILED || hdr == MAP_FAILED) {
        perror("vt_dma: mmap");
        return -1;
    }
    d->regs = regs;
    if (memcmp(hdr, "PHIR", 4)) {
        fprintf(stderr, "vt_dma: no PHIR header at 0x%llx\n", (unsigned long long)REGION_HEADER);
        return -1;
    }
    memcpy(&d->region_addr, (const char *)hdr + 32, 8);
    memcpy(&d->region_bytes, (const char *)hdr + 40, 8);
    munmap(hdr, 4096);
    if (!d->region_addr || !d->region_bytes || (d->region_addr & 4095)) {
        fprintf(stderr, "vt_dma: implausible region 0x%llx, %llu bytes\n", (unsigned long long)d->region_addr, (unsigned long long)d->region_bytes);
        return -1;
    }
    memset(page, 0, 8192);
    d->ring = page;
    d->done = (volatile uint64_t *)((char *)page + 4096);
    d->ring_phys = page_phys;
    d->done_phys = page_phys + 4096;

    volatile uint32_t *dcr = &d->regs[DCR_OFFSET / 4];
    const uint32_t mine = 3u << (2 * d->chan), before = *dcr;
    const uint32_t en = 2u << (2 * d->chan), own = 1u << (2 * d->chan);
    if ((before & en) && !(before & own)) {
        /* Enabled and card owned, with its lock free: the last owner died. Its queued copies get 2 s to drain. */
        const double t0 = now_s();
        while (rd(d, R_DTPR) != rd(d, R_DHPR) && now_s() - t0 < 2.0) {
        }
        if (rd(d, R_DTPR) != rd(d, R_DHPR)) {
            fprintf(stderr, "vt_dma: channel %d still busy\n", d->chan);
            return -1;
        }
    }
    *dcr = before & ~mine; /* card owned, disabled; nobody else's bits */
    const uint32_t after = *dcr;
    if ((after & ~mine) != (before & ~mine) || (after & mine)) {
        fprintf(stderr, "vt_dma: DCR 0x%08x became 0x%08x\n", before, after);
        return -1;
    }
    wr(d, R_DCAR, DCAR_MASK_BOTH);
    wr(d, R_DCHERRMSK, 0);
    wr(d, R_DRAR_LO, (uint32_t)d->ring_phys);
    wr(d, R_DRAR_HI, (RING_DESC << 4) | (uint32_t)((d->ring_phys >> 32) & 0xF));
    /* The head on a line boundary at or after the hardware's tail; everything between is a NOP. */
    d->head = ((rd(d, R_DTPR) % RING_DESC + LINE_DESC - 1) / LINE_DESC * LINE_DESC) % RING_DESC;
    full_fence();
    wr(d, R_DHPR, d->head);
    *dcr = (*dcr & ~mine) | en;
    if (!(*dcr & en)) {
        fprintf(stderr, "vt_dma: the enable did not stick\n");
        return -1;
    }
    const double t0 = now_s();
    while (rd(d, R_DTPR) % RING_DESC != d->head && now_s() - t0 < 2.0) {
    }
    return 0;
}

uint64_t vt_dma_copy(struct vt_dma *d, uint64_t src, uint64_t dst, uint32_t bytes, uint64_t status_addr, uint64_t status_value)
{
    if (!bytes || (bytes & 63) || bytes > VT_DMA_MAX || ((src | dst) & 63))
        return 0;
    while (__atomic_exchange_n(&d->submit_lock, 1, __ATOMIC_ACQUIRE))
        vt_delay(16u);
    while (d->seq - *d->done >= IN_FLIGHT)
        vt_delay(16u);
    const uint64_t seq = ++d->seq;
    volatile uint64_t *q = d->ring + (size_t)d->head * 2;
    q[0] = (src & ADDR_MASK) | ((uint64_t)(bytes / 64) << 46);
    q[1] = (dst & ADDR_MASK) | TYPE_COPY;
    q[2] = status_addr ? status_value : 0;
    q[3] = status_addr ? (status_addr & ADDR_MASK) | TYPE_STATUS : 0;
    q[4] = seq;
    q[5] = (d->done_phys & ADDR_MASK) | TYPE_STATUS;
    q[6] = 0;
    q[7] = 0;
    d->head = (d->head + LINE_DESC) % RING_DESC;
    full_fence();
    wr(d, R_DHPR, d->head);
    __atomic_store_n(&d->submit_lock, 0, __ATOMIC_RELEASE);
    return seq;
}

int vt_dma_wait(struct vt_dma *d, uint64_t seq, double timeout_s)
{
    for (int i = 0; i < 400000; i++) {
        if (*d->done >= seq)
            return 0;
        vt_delay(16u);
    }
    const double t0 = now_s();
    while (*d->done < seq) {
        if (now_s() - t0 > timeout_s)
            return -1;
        vt_delay(256u);
    }
    return 0;
}

void vt_dma_close(struct vt_dma *d)
{
    if (d->regs) {
        const uint32_t en = 2u << (2 * d->chan);
        d->regs[DCR_OFFSET / 4] &= ~en;
    }
    if (d->lock_fd >= 0)
        close(d->lock_fd);
    d->regs = NULL;
    d->lock_fd = -1;
}
