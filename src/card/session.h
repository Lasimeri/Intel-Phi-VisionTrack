/* session.h: what vtrackd can be asked to do. Each returns 0, or -1 on failure with its reason on stderr. */
#ifndef VT_SESSION_H
#define VT_SESSION_H

#include <stdint.h>

#include "arena.h"
#include "topo.h"
#include "vt_dma.h"
#include "vt_proto.h"

/* A virtual range of the arena to contiguous addresses of the region, cut at huge-page edges. The last DMA line's sequence, 0 if one was refused. */
uint64_t session_copy_out(struct vt_dma *d, const struct arena *a, void *p, uint64_t host_addr, uint64_t bytes);

int session_xfer(struct vt_ctl *ctl, struct vt_dma *dma, uint64_t seq);
int session_track(struct vt_ctl *ctl, struct vt_dma *dma, const struct topo *topo, const struct vt_config *cfg, uint64_t seq);

#endif
