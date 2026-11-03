/* vt_link.c: see vt_link.h. */
#define _GNU_SOURCE
#include "vt_link.h"

#include "vt_format.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define REGION_PATH "/dev/shm/phi-hostmem"
#define CARD_SYSFS "/sys/bus/pci/devices/0000:2f:00.0"

int vt_link_open(struct vt_link *l)
{
    l->fd = open(REGION_PATH, O_RDWR);
    if (l->fd < 0) {
        perror("vt_link: " REGION_PATH);
        return -1;
    }
    void *p = mmap(NULL, VT_PART_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, l->fd, (off_t)VT_PART_OFFSET);
    if (p == MAP_FAILED) {
        perror("vt_link: mmap");
        return -1;
    }
    l->part = p;
    l->ctl = p;
    return 0;
}

void vt_link_close(struct vt_link *l)
{
    munmap(l->part, VT_PART_BYTES);
    close(l->fd);
}

int vt_session_start(struct vt_link *l, const struct vt_config *cfg, double timeout_s)
{
    struct vt_ctl *c = l->ctl;
    /* A session still open belongs to a program that died without ending it: end it first, or the card serves a ghost until its watchdog. */
    if ((int64_t)c->card_session > 0) {
        c->session_end = 1;
        vt_session_wait_end(l, 3.0);
    }
    c->magic = VT_MAGIC;
    c->config = *cfg;
    c->config.format_hash = vt_format_hash();
    /* check.sh's test of the refusal: a program of another build, without building one. */
    if (getenv("VT_TEST_BAD_FORMAT"))
        c->config.format_hash ^= 1;
    c->frames_pub = 0;
    c->frames_taken = 0;
    c->exceptions = 0;
    c->last_exception = 0;
    c->results_pub = 0;
    c->depth_pub = 0;
    c->xfer_seq = 0;
    c->xfer_done = 0;
    c->session_end = 0;
    c->card_session = 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    const uint64_t seq = c->session_seq + 1;
    c->session_seq = seq;
    const uint64_t t0 = vt_now_ns();
    while (c->card_session != seq) {
        if (c->card_session == (uint64_t)-(int64_t)seq) {
            if (c->card_format != c->config.format_hash)
                fprintf(stderr, "vt_link: the card runs another build: its wire format %016llx, this program's %016llx. Rebuild, then scripts/vtrackd.sh stop and start.\n", (unsigned long long)c->card_format, (unsigned long long)c->config.format_hash);
            else
                fprintf(stderr, "vt_link: the card refused the session (see /tmp/vt/vtrackd.log on the card)\n");
            return -1;
        }
        if ((double)(vt_now_ns() - t0) * 1e-9 > timeout_s) {
            fprintf(stderr, "vt_link: no answer from vtrackd after %.0f s (host side waited; is it running on the card?)\n", timeout_s);
            return -1;
        }
        struct timespec nap = {0, 1000000};
        nanosleep(&nap, NULL);
    }
    return 0;
}

void vt_session_end(struct vt_link *l) { l->ctl->session_end = 1; }

int vt_session_wait_end(struct vt_link *l, double timeout_s)
{
    const uint64_t t0 = vt_now_ns();
    while (l->ctl->card_session != 0) {
        if ((double)(vt_now_ns() - t0) * 1e-9 > timeout_s)
            return -1;
        struct timespec nap = {0, 1000000};
        nanosleep(&nap, NULL);
    }
    return 0;
}

static void read_line(const char *name, char *out, int n)
{
    char path[160];
    snprintf(path, sizeof path, "%s/%s", CARD_SYSFS, name);
    out[0] = 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    if (fgets(out, n, f))
        out[strcspn(out, "\n")] = 0;
    fclose(f);
}

void vt_link_pcie(char *out, int n)
{
    char sp[40], w[16], mw[16];
    read_line("current_link_speed", sp, sizeof sp);
    read_line("current_link_width", w, sizeof w);
    read_line("max_link_width", mw, sizeof mw);
    snprintf(out, (size_t)n, "%s x%s of x%s", sp, w, mw);
}
