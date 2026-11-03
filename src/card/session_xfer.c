/* session_xfer.c: the card's half of the transport measurement (src/host/vt_xfer.c). */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>

#include "arena.h"
#include "session.h"
#include "vt_util.h"

int session_xfer(struct vt_ctl *ctl, struct vt_dma *dma, uint64_t seq)
{
    struct arena a;
    if (arena_open(&a, VT_XFER_BYTES))
        return -1;
    /* The buffer's pages need not be contiguous: copies go page by page. */
    const uint64_t host = dma->region_addr + VT_PART_OFFSET + VT_XFER_OFFSET;
    ctl->card_session = seq;
    uint64_t done = 0;
    while (!ctl->session_end) {
        const uint64_t s = ctl->xfer_seq;
        if (s == done) {
            vt_delay(16u);
            continue;
        }
        done = s;
        const uint64_t op = ctl->xfer_cmd, bytes = ctl->xfer_bytes;
        const uint64_t t0 = vt_tsc();
        if (op == VT_XFER_H2C || op == VT_XFER_C2H) {
            uint64_t last = 0, off = 0;
            while (off < bytes) {
                const uint64_t in_page = ARENA_PAGE - off % ARENA_PAGE;
                const uint32_t n = (uint32_t)(bytes - off < in_page ? bytes - off : in_page);
                const uint32_t piece = n > VT_DMA_MAX ? VT_DMA_MAX : n;
                const uint64_t card_addr = arena_phys(&a, a.base + off);
                last = op == VT_XFER_H2C ? vt_dma_copy(dma, host + off, card_addr, piece, 0, 0) : vt_dma_copy(dma, card_addr, host + off, piece, 0, 0);
                if (!last) {
                    fprintf(stderr, "vtrackd: bad copy of %u bytes\n", piece);
                    return -1;
                }
                off += piece;
            }
            if (vt_dma_wait(dma, last, 5.0)) {
                fprintf(stderr, "vtrackd: DMA timeout\n");
                return -1;
            }
        }
        ctl->xfer_card_cycles = vt_tsc() - t0;
        ctl->xfer_done = s;
    }
    arena_close(&a);
    return 0;
}
