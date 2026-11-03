/* vt_link.h: the host's view of the shared part: the control page, the frame slots, the result ring. */
#ifndef VT_LINK_H
#define VT_LINK_H

#include <stdint.h>
#include <time.h>

#include "vt_proto.h"

struct vt_link {
    int fd;
    uint8_t *part;
    struct vt_ctl *ctl;
};

/* Maps the part of /dev/shm/phi-hostmem. 0, or -1 with a message. */
int vt_link_open(struct vt_link *l);
void vt_link_close(struct vt_link *l);

static inline uint64_t vt_now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Publishes a config and waits for the card's answer. 0, or -1 with a message: card absent or session refused. */
int vt_session_start(struct vt_link *l, const struct vt_config *cfg, double timeout_s);
void vt_session_end(struct vt_link *l);
/* Waits until vtrackd has acknowledged and left its current session (card_session back to 0). */
int vt_session_wait_end(struct vt_link *l, double timeout_s);

/* The negotiated PCIe link from sysfs, "5.0 GT/s PCIe x8 of x16". */
void vt_link_pcie(char *out, int n);

#endif
