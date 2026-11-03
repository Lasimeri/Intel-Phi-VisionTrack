/* vt_dma.h: one channel of the card's SBOX DMA engine, taken and driven from user space on the card.
 *
 * Written for this project from the hardware contract: Intel SSDG 328207-002 section 2.1.8.2.1 and the stack's
 * host/asm/phictl/defs.inc, dma.md. Channel n's registers sit at SBOX + 0xA000 + 0x40 n, DCR at SBOX + 0xA280: bit 2n the owner
 * (0 card), bit 2n + 1 the enable. A descriptor is 16 bytes: memcpy (40-bit source, lines of 64 B in bits 59:46; 40-bit
 * destination, type 1 in bits 63:60), status (a value, then a 40-bit address, type 2), or zero = NOP. Addresses are card
 * physical: GDDR, or the shared region's card address + an offset. */
#ifndef VT_DMA_H
#define VT_DMA_H

#include <stdint.h>

struct vt_dma {
    int chan, lock_fd;
    volatile uint32_t *regs;
    volatile uint64_t *ring;
    volatile uint64_t *done;   /* the last line finished */
    uint64_t ring_phys, done_phys;
    uint64_t region_addr, region_bytes;
    uint32_t head;
    uint64_t seq;
    int submit_lock;           /* two threads submit in a tracking session: the frames' ingest and the results */
};

/* The first channel of the list that can be locked and taken, its region located. 0, or -1 with a message. */
int vt_dma_open(struct vt_dma *d, void *page, uint64_t page_phys);

/* A copy of bytes (a multiple of 64, at most VT_DMA_MAX) between 64 B aligned card addresses, then, when status_addr is nonzero, the
 * 8-byte status_value written there, after the copy's data. Returns the line's sequence number. Safe from several threads. */
#define VT_DMA_MAX (16383u * 64u)
uint64_t vt_dma_copy(struct vt_dma *d, uint64_t src, uint64_t dst, uint32_t bytes, uint64_t status_addr, uint64_t status_value);

/* Spins until line seq has finished; 0, or -1 after timeout_s seconds. */
int vt_dma_wait(struct vt_dma *d, uint64_t seq, double timeout_s);

void vt_dma_close(struct vt_dma *d);

#endif
