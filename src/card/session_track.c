/* session_track.c: the tracking session on the card. Long-lived: one set of teams and one tracker for the whole session, no process per frame.
 *
 *   ingest:   on a free hardware thread of a reserved core, found at run time. Polls the host's publication word, checks each message object, pulls its frame into device memory by
 *             DMA, frees the host's slot. So the next frame's transfer overlaps this frame's computation. Device memory full is the backpressure: the host's slots fill, and its
 *             policy decides.
 *   pipelines = 1: one team, tid 0 builds the pyramid and tracks, in turn.
 *   pipelines = 2: two groups of cores with a pyramid ring between them: the group building frame n + 1 runs beside the group tracking frame n. Each group's tid 0 is the only
 *             thread calling its half of the tracker.
 *
 * Results go back by DMA, a status word after each. */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"
#include "fuse.h"
#include "img.h"
#include "gimref.h"
#include "lk.h"
#include "vecdraw.h"
#include "live_depth.h"
#include "plane.h"
#include "session.h"
#include "team.h"
#include "track.h"
#include "fsr3.h"
#include "vt_util.h"

#define DEV_SLOTS 8
#define IDLE_POLL_CYCLES 64u
#define NO_FRAME_S 30.0
#define MAX_SETS 8

struct ingest {
    struct vt_ctl *ctl;
    struct vt_dma *dma;
    struct arena *a;
    const struct vt_config *cfg;
    uint64_t host_frames;   /* card address of the frame area */
    uint64_t slot_bytes;
    uint8_t *dev[DEV_SLOTS];
    struct vt_msg msg[DEV_SLOTS];
    uint64_t dma_cycles[DEV_SLOTS];
    uint64_t landed_tsc[DEV_SLOTS];
    int cpu;
    uint64_t landed VT_CACHELINE;  /* frames in device memory, written by ingest */
    uint64_t taken VT_CACHELINE;   /* frames whose pixels nobody needs any more */
    int stop VT_CACHELINE;
    int failed;
};

/* A virtual range of the arena against contiguous card-physical addresses of the region, cut at huge-page edges. */
static uint64_t copy_range(struct vt_dma *d, const struct arena *a, void *p, uint64_t host_addr, uint64_t bytes, int to_card)
{
    uint64_t last = 0, off = 0;
    while (off < bytes) {
        uint8_t *v = (uint8_t *)p + off;
        const uint64_t in_page = ARENA_PAGE - ((uint64_t)(v - a->base) % ARENA_PAGE);
        uint64_t n = bytes - off < in_page ? bytes - off : in_page;
        if (n > VT_DMA_MAX)
            n = VT_DMA_MAX;
        const uint64_t card = arena_phys(a, v);
        last = to_card ? vt_dma_copy(d, host_addr + off, card, (uint32_t)n, 0, 0) : vt_dma_copy(d, card, host_addr + off, (uint32_t)n, 0, 0);
        if (!last)
            return 0;
        off += n;
    }
    return last;
}

uint64_t session_copy_out(struct vt_dma *d, const struct arena *a, void *p, uint64_t host_addr, uint64_t bytes) { return copy_range(d, a, p, host_addr, bytes, 0); }

static void *ingest_main(void *arg)
{
    struct ingest *g = arg;
    topo_pin(g->cpu);
    uint64_t seen = 0, next = 0, last_instance[VT_MAX_CAMERAS] = {0, 0, 0};
    int have_instance[VT_MAX_CAMERAS] = {0, 0, 0};
    const uint32_t ncams = vt_cameras(g->cfg);
    const uint64_t expect[VT_MAX_CAMERAS] = {vt_slot_bytes(g->cfg->width, g->cfg->height, g->cfg->stereo), vt_slot_bytes(g->cfg->width2, g->cfg->height2, 0), vt_slot_bytes(g->cfg->width3, g->cfg->height3, 0)};
    while (!__atomic_load_n(&g->stop, __ATOMIC_ACQUIRE)) {
        if (g->ctl->frames_pub <= seen) {
            vt_delay(IDLE_POLL_CYCLES);
            continue;
        }
        /* Device memory full: wait. The host's slots then fill behind this, and its policy acts. */
        while (next - __atomic_load_n(&g->taken, __ATOMIC_ACQUIRE) >= DEV_SLOTS) {
            if (__atomic_load_n(&g->stop, __ATOMIC_ACQUIRE))
                return NULL;
            vt_delay(IDLE_POLL_CYCLES);
        }
        const uint32_t di = (uint32_t)(next % DEV_SLOTS);
        struct vt_msg *m = &g->msg[di];
        /* Word by word from uncached host memory. */
        const volatile uint64_t *src = (const volatile uint64_t *)&g->ctl->desc[seen % VT_DESC_RING];
        uint64_t *out = (uint64_t *)m;
        for (size_t i = 0; i < sizeof *m / 8; i++)
            out[i] = src[i];
        /* The header is checked before anything acts on it. A bad message is dropped and reported, the session goes on. */
        enum vt_exception ex = VT_EXC_NONE;
        if (m->magic != VT_MSG_MAGIC || m->check != vt_msg_check(m))
            ex = VT_EXC_BAD_CHECK;
        else if (m->type != VT_MSG_FRAME || !(m->dest & VT_PIPE_PYR) || m->camera >= ncams)
            ex = VT_EXC_BAD_TYPE;
        else if (m->slot >= g->cfg->slots)
            ex = VT_EXC_BAD_SLOT;
        else if (m->length != expect[m->camera])
            ex = VT_EXC_BAD_LENGTH;
        else if (have_instance[m->camera] && m->instance <= last_instance[m->camera])
            ex = VT_EXC_INSTANCE_ORDER;
        if (ex != VT_EXC_NONE) {
            g->ctl->last_exception = ex;
            g->ctl->exceptions = g->ctl->exceptions + 1;
            seen++;
            g->ctl->frames_taken = seen;
            continue;
        }
        last_instance[m->camera] = m->instance;
        have_instance[m->camera] = 1;
        const uint64_t t0 = vt_tsc();
        const uint64_t last = copy_range(g->dma, g->a, g->dev[di], g->host_frames + (uint64_t)m->slot * g->slot_bytes, m->length, 1);
        if (!last || vt_dma_wait(g->dma, last, 5.0)) {
            fprintf(stderr, "vtrackd: frame DMA failed\n");
            g->failed = 1;
            return NULL;
        }
        const uint64_t t1 = vt_tsc();
        g->dma_cycles[di] = t1 - t0;
        g->landed_tsc[di] = t1;
        next++;
        __atomic_store_n(&g->landed, next, __ATOMIC_RELEASE);
        seen++;
        g->ctl->frames_taken = seen;
    }
    return NULL;
}

/* The highest CPU of a reserved core that is neither CPU 0 nor a block poller: ingest's home. */
static int service_cpu(const struct topo *t)
{
    int best = -1;
    for (int c = 0; c < t->ncores; c++) {
        if (!t->core[c].reserved)
            continue;
        for (int k = 0; k < t->core[c].ncpus; k++) {
            const int cpu = t->core[c].cpu[k];
            int poller = 0;
            for (int p = 0; p < t->npollers; p++)
                poller |= t->poller_cpu[p] == cpu;
            if (cpu != 0 && !poller && cpu > best)
                best = cpu;
        }
    }
    return best;
}

static void *arena_alloc(void *ctx, size_t bytes) { return arena_take(ctx, bytes, 64); }
static void team_runner(void *ctx, tk_fn fn, void *arg) { team_run(ctx, fn, arg); }

/* What the pyramid half hands to the track half, per set. */
struct built {
    struct vt_msg msg;
    uint64_t dma_cycles, queue_cycles, pyr_cycles;
    uint32_t skipped;
};

struct flow {
    struct ingest *g;
    struct tracker *tr;
    const struct vt_config *cfg;
    int nsets;
    uint64_t next;       /* the next landed frame the pyramid half looks at */
    struct built info[MAX_SETS];
    uint64_t built VT_CACHELINE;   /* sets complete */
    uint64_t tracked VT_CACHELINE; /* frames finished by the track half */
    int stop VT_CACHELINE;
    /* the pyramid group, when it has its own thread */
    const int *cpus, *core_of;
    int n;
    struct team *team;
    int team_ready;
};

/* One frame's pyramid, if a frame is ready and a set is free. Returns 1 when it built. */
static int build_one(struct flow *f)
{
    struct ingest *g = f->g;
    const uint64_t landed = __atomic_load_n(&g->landed, __ATOMIC_ACQUIRE);
    if (landed <= f->next)
        return 0;
    /* A set is free once its frame has been tracked and is no longer the next frame's previous one. */
    const uint64_t b = f->built;
    if (b + 2 > __atomic_load_n(&f->tracked, __ATOMIC_ACQUIRE) + (uint64_t)f->nsets)
        return 0;
    uint32_t skipped = 0;
    uint32_t di = (uint32_t)(f->next % DEV_SLOTS);
    /* A live frame goes ahead of whatever waits before it: the newest landed one, if it asks for that. */
    const uint32_t newest = (uint32_t)((landed - 1) % DEV_SLOTS);
    if (landed - f->next > 1 && g->msg[newest].priority == VT_PRIO_LIVE) {
        skipped = (uint32_t)(landed - 1 - f->next);
        f->next = landed - 1;
        di = newest;
    }
    const uint64_t t0 = vt_tsc();
    const uint8_t *left = g->dev[di];
    const uint8_t *right = f->cfg->stereo ? left + (size_t)f->cfg->width * f->cfg->height : NULL;
    const int set = (int)(b % (uint64_t)f->nsets);
    tk_build(f->tr, set, left, right, &g->msg[di]);
    struct built *i = &f->info[set];
    i->msg = g->msg[di];
    i->dma_cycles = g->dma_cycles[di];
    i->queue_cycles = t0 - g->landed_tsc[di];
    i->pyr_cycles = vt_tsc() - t0;
    i->skipped = skipped;
    f->next++;
    __atomic_store_n(&g->taken, f->next, __ATOMIC_RELEASE);
    __atomic_store_n(&f->built, b + 1, __ATOMIC_RELEASE);
    return 1;
}

/* A picture from FSR, on the same channel ahead of its result: it has landed when the result says so. */
static int send_fsr_picture(struct vt_dma *dma, const struct arena *a, uint8_t *pic, const struct vt_config *cfg, struct vt_result *res, uint32_t *turn)
{
    const uint32_t slot = (*turn)++ % VT_FSR_SLOTS;
    if (!copy_range(dma, a, pic, dma->region_addr + VT_PART_OFFSET + vt_fsr_offset(slot), (uint64_t)cfg->fsr_w * cfg->fsr_h, 0)) {
        fprintf(stderr, "vtrackd: FSR picture copy refused\n");
        return -1;
    }
    res->fsr_slot = slot + 1;
    res->fsr_w = cfg->fsr_w;
    res->fsr_h = cfg->fsr_h;
    return 0;
}

/* The row and picture of the frame between the last real one and this one, sent before this one's row. Made only now: its box lies between the two real ones, and the newer has just been tracked. `real` stays as it is. Waits for its own copy: the next result may follow at once. Nonzero if the channel refused or timed out. */
static int send_generated(struct tracker *t, const struct vt_msg *gm, uint64_t gen_cycles, const struct vt_result *real, struct vt_result *gres, uint64_t gres_phys, struct vt_dma *dma, const struct arena *a, uint8_t *pic, uint64_t pic_bytes, uint32_t *pic_turn, uint8_t *fsr_pic, const struct vt_config *cfg, uint32_t *fsr_turn, uint64_t *npub, uint64_t *last_dma, uint64_t host_res, uint64_t pub_addr)
{
    tk_generated_result(t, gm, real, gres);
    gres->cycles[VT_T_GEN] = gen_cycles;
    if (pic) {
        /* The picture first, on the same channel: it has landed when the result says so. */
        const uint32_t slot = (*pic_turn)++ % VT_GEN_SLOTS;
        if (!copy_range(dma, a, pic, dma->region_addr + VT_PART_OFFSET + vt_gen_offset(gm->camera, slot), pic_bytes, 0)) {
            fprintf(stderr, "vtrackd: generated picture copy refused\n");
            return -1;
        }
        gres->picture = slot + 1;
    }
    if (fsr_pic && send_fsr_picture(dma, a, fsr_pic, cfg, gres, fsr_turn))
        return -1;
    (*npub)++;
    gres->seq = *npub;
    *last_dma = vt_dma_copy(dma, gres_phys, host_res + (*npub - 1) % VT_RES_SLOTS * VT_RES_BYTES, gres->hdr.length, pub_addr, *npub);
    if (!*last_dma) {
        fprintf(stderr, "vtrackd: generated result copy refused\n");
        return -1;
    }
    if (vt_dma_wait(dma, *last_dma, 5.0)) {
        fprintf(stderr, "vtrackd: generated result DMA timed out\n");
        return -1;
    }
    return 0;
}

static void *pyramid_main(void *arg)
{
    struct flow *f = arg;
    f->team = team_create(f->cpus, f->core_of, f->n);
    __atomic_store_n(&f->team_ready, f->team ? 1 : -1, __ATOMIC_RELEASE);
    if (!f->team)
        return NULL;
    while (!__atomic_load_n(&f->stop, __ATOMIC_ACQUIRE))
        if (!build_one(f))
            vt_delay(IDLE_POLL_CYCLES);
    team_destroy(f->team);
    return NULL;
}

/* Optical flow and FSR apart from the tracker, the operator's order, 2026-11-04: a team on each core's third and fourth hardware threads, which the tracker does not use. It shares the cores, so it slows the tracker while both run, but the tracker never waits for it. Busy when a frame arrives, the newest waits and the one it replaces is let go: a picture late is no use, and a tracker late is worse. */
struct flow_slot {
    struct tracker *t;
    uint32_t seen;
    uint32_t want VT_CACHELINE, done VT_CACHELINE;
};

struct fsr_worker {
    int cpus[256], core_of[256], n;
    struct team *team;
    struct flow_slot slots[VT_MAX_CAMERAS];
    int fsr_cam, next, inside_left;
    int ncams VT_CACHELINE;
    int stop VT_CACHELINE, team_ok, frames, dropped, with_mid;
};

static void fsr_worker_run(void *ctx, tk_fn fn, void *arg) { team_run(((struct fsr_worker *)ctx)->team, fn, arg); }

/* One other camera whose frame is waiting, each in turn. 1 if there was one. */
static int flow_other(struct fsr_worker *w)
{
    const int ncams = __atomic_load_n(&w->ncams, __ATOMIC_ACQUIRE);
    for (int i = 0; i < ncams; i++) {
        const int k = (w->next + i) % ncams;
        struct flow_slot *s = &w->slots[k];
        if (k == w->fsr_cam || !s->t)
            continue;
        const uint32_t job = __atomic_load_n(&s->want, __ATOMIC_ACQUIRE);
        if (job == s->seen)
            continue;
        s->seen = job;
        tk_generate_make(s->t);
        __atomic_store_n(&s->done, job, __ATOMIC_RELEASE);
        w->next = (k + 1) % ncams;
        return 1;
    }
    return 0;
}

/* Between FSR's stages, a limited number: a camera at 33 ms a frame need not wait for a whole frame of FSR, and FSR's camera must not wait for all of theirs. With no limit, 2026-11-04, the team asked for 105 % of its time and FSR's camera lost: 15.83 pictures a second, L and R 59.3. */
#define FLOW_INSIDE_FSR 2
static void flow_between(void *arg)
{
    struct fsr_worker *w = arg;
    if (w->inside_left > 0)
        w->inside_left -= flow_other(w);
}

static void *fsr_worker_main(void *arg)
{
    struct fsr_worker *w = arg;
    w->team = team_create(w->cpus, w->core_of, w->n);
    __atomic_store_n(&w->team_ok, w->team ? 1 : -1, __ATOMIC_RELEASE);
    if (!w->team)
        return NULL;
    while (!__atomic_load_n(&w->stop, __ATOMIC_ACQUIRE)) {
        struct flow_slot *d = &w->slots[w->fsr_cam];
        const uint32_t job = __atomic_load_n(&d->want, __ATOMIC_ACQUIRE);
        if (job != d->seen) {
            d->seen = job;
            /* FSR's camera first: the field its motion comes from, FSR on the real frame, and the frame between it and the last from their pictures. They go together. */
            w->inside_left = FLOW_INSIDE_FSR;
            tk_generate_make(d->t);
            w->with_mid = tk_fsr_one(d->t);
            __atomic_store_n(&d->done, job, __ATOMIC_RELEASE);
            continue;
        }
        if (!flow_other(w))
            team_delay(2000);
    }
    team_destroy(w->team);
    return NULL;
}

/* The orange vectors of FSR's camera, on the core nothing else uses: the operator's order, 2026-11-03. Its vectors, which cells moved, the lines at the upscaled size, and those brought down with the picture's own tables. The rest of that frame's work, and all the copying, stay with the session. */
struct vec_worker {
    struct live_depth *ld;
    struct vecdraw *draw;
    const struct fsr *fsr;
    uint8_t *pic;
    char *temp;
    size_t temp_each;
    int cam, level;
    long double view;
    int cpus[8], core_of[8], n;
    struct team *team;
    const struct vt_plane *cur;
    float inside[4];
    int has_box;
    uint64_t instance;
    int y0, y1, lines, any;
    uint64_t cycles_vectors, cycles_draw, cycles_down, cycles_gim;
    struct gimref gim;
    long double total_u, total_v;
    uint32_t applied;
    const struct vt_plane *prev;
    uint32_t want VT_CACHELINE, done VT_CACHELINE;
    int stop VT_CACHELINE, team_ok;
};

static void vec_down_team(void *arg, int tid, int n)
{
    const struct vec_worker *w = arg;
    fsr_down_plane(w->fsr, vd_plane(w->draw), vd_rows(w->draw), w->y0, w->y1, w->pic, (float *)(void *)(w->temp + (size_t)tid * w->temp_each), tid, n);
}

static void *vec_worker_main(void *arg)
{
    struct vec_worker *w = arg;
    w->team = team_create(w->cpus, w->core_of, w->n);
    __atomic_store_n(&w->team_ok, w->team ? 1 : -1, __ATOMIC_RELEASE);
    if (!w->team)
        return NULL;
    uint32_t seen = 0;
    while (!__atomic_load_n(&w->stop, __ATOMIC_ACQUIRE)) {
        const uint32_t job = __atomic_load_n(&w->want, __ATOMIC_ACQUIRE);
        if (job == seen) {
            team_delay(2000);
            continue;
        }
        seen = job;
        const uint64_t t0 = vt_tsc();
        ld_vectors(w->ld, w->cam, w->cur, w->team);
        const uint64_t t1 = vt_tsc();
        int cols, rows;
        const struct vt_cell *cells = ld_cells(w->ld, w->cam, &cols, &rows);
        /* Where D holds still, and how far that moved since the last frame. */
        gimref_choose(&w->gim, cells, w->has_box ? w->inside : NULL, w->lines >= 3);
        if (w->prev) {
            const int points = gimref_points(&w->gim);
            for (int i = 0; i < w->gim.nwin; i++)
                for (int k = 0; k < points; k++) {
                    float x, y;
                    struct lk_out o;
                    gimref_point(&w->gim, i, k, &x, &y);
                    ld_point(w->ld, w->cam, w->prev, w->cur, x, y, &o, 0);
                    gimref_add(&w->gim, i, (float)((long double)o.x - (long double)x), (float)((long double)o.y - (long double)y), o.status == 0);
                }
            gimref_finish(&w->gim);
        }
        w->prev = w->cur;
        const uint64_t t_gim = vt_tsc();
        w->cycles_gim = t_gim - t1;
        w->lines = vd_draw(w->draw, cells, cols, rows, w->level, w->view, w->has_box ? w->inside : NULL);
        w->any = fsr_down_span(w->fsr, vd_rows(w->draw), &w->y0, &w->y1);
        const uint64_t t2 = vt_tsc();
        if (w->any) {
            team_run(w->team, vec_down_team, w);
            img_fence();
        }
        const uint64_t t3 = vt_tsc();
        w->cycles_vectors = t1 - t0;
        w->cycles_draw = t2 - t_gim;
        w->cycles_down = t3 - t2;
        __atomic_store_n(&w->done, job, __ATOMIC_RELEASE);
    }
    team_destroy(w->team);
    return NULL;
}

/* One frame's vectors of L or R, on the channel: the word that names them last. */
static uint64_t cells_publish(struct vt_dma *dma, const struct arena *a, const struct live_depth *ld, int cam, uint64_t instance, struct vt_cells_meta *meta, uint64_t meta_phys, uint64_t seq)
{
    const uint64_t part = dma->region_addr + VT_PART_OFFSET;
    const uint32_t slot = (uint32_t)(seq % VT_CELL_SLOTS);
    int cols, rows;
    const struct vt_cell *cells = ld_cells(ld, cam, &cols, &rows);
    if (!session_copy_out(dma, a, (void *)(uintptr_t)cells, part + vt_cells_offset((uint32_t)cam, slot), ld_cell_bytes(ld, cam)))
        return 0;
    memset(meta, 0, sizeof *meta);
    meta->instance = instance;
    meta->cam = (uint32_t)cam;
    meta->slot = slot;
    meta->cols = (uint32_t)cols;
    meta->rows = (uint32_t)rows;
    meta->level = (uint32_t)ld_flow_level(ld);
    return vt_dma_copy(dma, meta_phys, part + offsetof(struct vt_ctl, cells_meta) + (uint64_t)cam * sizeof(struct vt_cells_meta), sizeof *meta, part + offsetof(struct vt_ctl, cells_pub) + (uint64_t)cam * sizeof(uint64_t), seq + 1);
}

/* The rows that have lines, then what they are, on the channel: the word that names them last. */
static uint64_t vec_publish(struct vt_dma *dma, const struct arena *a, const struct vec_worker *w, const struct vt_config *cfg, struct vt_vec_meta *meta, uint64_t meta_phys, uint64_t host_meta, uint64_t host_pub, uint64_t seq)
{
    const uint32_t slot = (uint32_t)(seq % VT_VEC_SLOTS);
    const uint64_t pw = cfg->fsr_w;
    if (w->any && !copy_range(dma, a, w->pic + (uint64_t)w->y0 * pw, dma->region_addr + VT_PART_OFFSET + vt_vec_offset(slot) + (uint64_t)w->y0 * pw, (uint64_t)(w->y1 - w->y0) * pw, 0))
        return 0;
    memset(meta, 0, sizeof *meta);
    meta->instance = w->instance;
    meta->slot = slot;
    meta->w = cfg->fsr_w;
    meta->h = cfg->fsr_h;
    meta->y0 = w->any ? (uint32_t)w->y0 : 0u;
    meta->y1 = w->any ? (uint32_t)w->y1 : 0u;
    meta->lines = (uint32_t)w->lines;
    meta->threads = (uint32_t)w->n;
    meta->cycles_vectors = w->cycles_vectors;
    meta->cycles_draw = w->cycles_draw;
    meta->cycles_down = w->cycles_down;
    meta->cycles_gim = w->cycles_gim;
    meta->gim_total_u = (float)w->total_u;
    meta->gim_total_v = (float)w->total_v;
    meta->gim_applied = w->applied;
    meta->gim_share = (uint32_t)w->gim.share;
    meta->gim_state = (uint32_t)w->gim.state;
    meta->gim_windows = (uint32_t)w->gim.nwin;
    meta->gim_read_u = (float)w->gim.read_u;
    meta->gim_read_v = (float)w->gim.read_v;
    meta->gim_move_u = (float)w->gim.move_u;
    meta->gim_move_v = (float)w->gim.move_v;
    meta->gim_w = (uint16_t)(w->gim.wc * (VT_CELL_W << w->gim.level));
    meta->gim_h = (uint16_t)(w->gim.wr * (VT_CELL_H << w->gim.level));
    for (int i = 0; i < w->gim.nwin; i++) {
        meta->gim_win[i][0] = (uint16_t)(w->gim.col[i] * (VT_CELL_W << w->gim.level));
        meta->gim_win[i][1] = (uint16_t)(w->gim.row[i] * (VT_CELL_H << w->gim.level));
    }
    return vt_dma_copy(dma, meta_phys, host_meta, sizeof *meta, host_pub, seq + 1);
}

/* A real frame's finished picture, and the one half way to it if there is one, on the channel: the word that names them last. One meta for both: they finish together, and a second word so soon would pass the first unread. */
static uint64_t fsr_publish(struct vt_dma *dma, const struct arena *a, uint8_t *const *pics, int with_mid, const struct vt_config *cfg, struct vt_fsr_meta *meta, uint64_t meta_phys, uint64_t host_meta, uint64_t host_pub, uint64_t seq, uint64_t instance)
{
    const uint32_t mid = (uint32_t)(2u * seq % VT_FSR_SLOTS), slot = mid + 1u;
    const uint64_t bytes = (uint64_t)cfg->fsr_w * cfg->fsr_h;
    if ((with_mid && !copy_range(dma, a, pics[0], dma->region_addr + VT_PART_OFFSET + vt_fsr_offset(mid), bytes, 0)) || !copy_range(dma, a, pics[1], dma->region_addr + VT_PART_OFFSET + vt_fsr_offset(slot), bytes, 0))
        return 0;
    meta->instance = instance;
    meta->slot = slot;
    meta->slot_mid = with_mid ? mid : VT_FSR_SLOTS;
    meta->w = cfg->fsr_w;
    meta->h = cfg->fsr_h;
    return vt_dma_copy(dma, meta_phys, host_meta, sizeof *meta, host_pub, seq + 1);
}

/* A generated picture of a camera that does not go through FSR, on the channel: the word that names it last. Its row went ahead, as soon as the tracker had the newer frame. */
static uint64_t gen_publish(struct vt_dma *dma, const struct arena *a, uint8_t *pic, uint64_t bytes, uint32_t cam, uint32_t *turn, uint64_t cycles, struct vt_gen_meta *meta, uint64_t meta_phys, uint64_t seq, uint64_t instance)
{
    const uint64_t part = dma->region_addr + VT_PART_OFFSET;
    const uint32_t slot = (*turn)++ % VT_GEN_SLOTS;
    if (!copy_range(dma, a, pic, part + vt_gen_offset(cam, slot), bytes, 0))
        return 0;
    memset(meta, 0, sizeof *meta);
    meta->instance = instance;
    meta->slot = slot;
    meta->cycles_make = cycles;
    return vt_dma_copy(dma, meta_phys, part + offsetof(struct vt_ctl, gen_meta) + (uint64_t)cam * sizeof(struct vt_gen_meta), sizeof *meta, part + offsetof(struct vt_ctl, gen_pub) + (uint64_t)cam * sizeof(uint64_t), seq + 1);
}

static int fsr_camera_w(const struct vt_config *c) { return (int)(c->fsr_camera == 2 ? c->width2 : c->fsr_camera == 3 ? c->width3 : c->width); }
static int fsr_camera_h(const struct vt_config *c) { return (int)(c->fsr_camera == 2 ? c->height2 : c->fsr_camera == 3 ? c->height3 : c->height); }

int session_track(struct vt_ctl *ctl, struct vt_dma *dma, const struct topo *topo, const struct vt_config *cfg, uint64_t seq)
{
    const int cores = cfg->cores ? (int)cfg->cores : topo->nusable;
    const int tpc = cfg->tpc ? (int)cfg->tpc : 2;
    const int two = cfg->pipelines == 2;
    const int pyr_cores = two ? (int)cfg->pyr_cores : 0;
    int cpus[256], core_of[256], pcpus[256], pcore_of[256];
    if (two && (pyr_cores < 1 || pyr_cores >= cores)) {
        fprintf(stderr, "vtrackd: %d pyramid cores of %d do not leave two groups\n", pyr_cores, cores);
        return -1;
    }
    const int n = topo_select(topo, pyr_cores, cores - pyr_cores, tpc, cpus, core_of);
    const int pn = two ? topo_select(topo, 0, pyr_cores, tpc, pcpus, pcore_of) : 0;
    if (n < 1 || pn < 0) {
        fprintf(stderr, "vtrackd: %d cores at %d threads a core do not fit\n", cores, tpc);
        return -1;
    }
    const uint64_t slot_bytes = vt_slot_stride(cfg);
    if (!cfg->slots || cfg->slots > VT_MAX_SLOTS || slot_bytes * cfg->slots > VT_FRAME_AREA) {
        fprintf(stderr, "vtrackd: %u slots of %llu bytes do not fit the region\n", cfg->slots, (unsigned long long)slot_bytes);
        return -1;
    }
    const int svc = service_cpu(topo);
    if (svc < 0) {
        fprintf(stderr, "vtrackd: no free thread on a reserved core for ingest\n");
        return -1;
    }
    /* Pyramids, scratch, tiles, frames: sized from the config. */
    const size_t plane = plane_bytes((int)cfg->width, (int)cfg->height);
    const size_t plane2 = cfg->dual ? plane_bytes((int)cfg->width2, (int)cfg->height2) : 0;
    const size_t plane3 = cfg->dual && cfg->width3 ? plane_bytes((int)cfg->width3, (int)cfg->height3) : 0;
    const int with_depth = cfg->dual && cfg->width3 && cfg->depth_planes;
    /* The orange vectors are drawn at the size FSR's picture has before any second stage. */
    const int vd_w = (int)(cfg->fsr_up_w ? cfg->fsr_up_w : cfg->fsr_w), vd_h = (int)(cfg->fsr_up_w ? cfg->fsr_up_h : cfg->fsr_h);
    const size_t need = (cfg->vec_overlay && vd_w ? vd_bytes(vd_w, vd_h, VT_DEPTH_BYTES / (int)sizeof(struct vt_cell)) + (size_t)cfg->fsr_w * cfg->fsr_h + (1u << 20) : 0u) + (cfg->fsr_camera ? (cfg->fsr_up_w ? fsr_bytes(fsr_camera_w(cfg), fsr_camera_h(cfg), (int)cfg->fsr_up_w, (int)cfg->fsr_up_h, (int)cfg->fsr_w, (int)cfg->fsr_h) : fsr_bytes(fsr_camera_w(cfg), fsr_camera_h(cfg), (int)cfg->fsr_w, (int)cfg->fsr_h, 0, 0)) + 2u * (size_t)cfg->fsr_w * cfg->fsr_h + (2u << 20) : 0u) + (two ? 26 : 20) * plane + 20 * plane2 + 20 * plane3 + DEV_SLOTS * slot_bytes + (cfg->dual ? (128u << 20) : (64u << 20)) + (with_depth ? ld_bytes(cfg, n) : 0) + (cfg->gen_pictures ? 3 * VT_GEN_PIC_BYTES : 0);
    struct arena a;
    if (arena_open(&a, need))
        return -1;
    struct team *tm = team_create(cpus, core_of, n);
    if (!tm) {
        fprintf(stderr, "vtrackd: team of %d failed\n", n);
        return -1;
    }
    static struct ingest g;
    memset(&g, 0, sizeof g);
    g.ctl = ctl;
    g.dma = dma;
    g.a = &a;
    g.cfg = cfg;
    g.host_frames = dma->region_addr + VT_PART_OFFSET + VT_FRAME_OFFSET;
    g.slot_bytes = slot_bytes;
    g.cpu = svc;
    static struct flow f;
    memset(&f, 0, sizeof f);
    f.g = &g;
    f.cfg = cfg;
    pthread_t pyr_thread;
    if (two) {
        f.cpus = pcpus;
        f.core_of = pcore_of;
        f.n = pn;
        if (pthread_create(&pyr_thread, NULL, pyramid_main, &f)) {
            fprintf(stderr, "vtrackd: no pyramid thread\n");
            return -1;
        }
        /* The tracker must not build before it exists, so the group waits on nothing yet: build_one needs f.tr, set below. */
        while (!__atomic_load_n(&f.team_ready, __ATOMIC_ACQUIRE))
            vt_delay(IDLE_POLL_CYCLES);
        if (f.team_ready < 0) {
            fprintf(stderr, "vtrackd: pyramid team of %d failed\n", pn);
            return -1;
        }
    }
    struct tracker *tr = tk_create(cfg, two ? pn : n, n, arena_alloc, &a, team_runner, two ? (void *)f.team : (void *)tm, team_runner, tm);
    if (!tr)
        return -1;
    /* More views: a tracker of its own size for each, sharing the team. Cameras 0 and 1 fuse into one 3D point; camera 2 has no vote in it. */
    struct tracker *trk[VT_MAX_CAMERAS] = {tr, NULL, NULL};
    static struct vt_config cam_cfg[VT_MAX_CAMERAS];
    static struct fuse fu;
    if (cfg->dual) {
        for (uint32_t k = 1; k < vt_cameras(cfg); k++) {
            cam_cfg[k] = *cfg;
            cam_cfg[k].width = k == 1 ? cfg->width2 : cfg->width3;
            cam_cfg[k].height = k == 1 ? cfg->height2 : cfg->height3;
            trk[k] = tk_create(&cam_cfg[k], n, n, arena_alloc, &a, team_runner, tm, team_runner, tm);
            if (!trk[k])
                return -1;
        }
        fuse_init(&fu, cfg);
    }
    /* A byte picture for each camera's generated frames, if the window wants them. */
    uint8_t *gen_pic[VT_MAX_CAMERAS] = {NULL, NULL, NULL};
    uint64_t gen_bytes[VT_MAX_CAMERAS] = {(uint64_t)cfg->width * cfg->height, (uint64_t)cfg->width2 * cfg->height2, (uint64_t)cfg->width3 * cfg->height3};
    uint32_t gen_turn[VT_MAX_CAMERAS] = {0, 0, 0};
    if (cfg->gen_pictures && cfg->framegen)
        for (uint32_t k = 0; k < vt_cameras(cfg); k++) {
            if (cfg->gen_cameras && !(cfg->gen_cameras >> k & 1u))
                continue;
            if (gen_bytes[k] > VT_GEN_PIC_BYTES) {
                fprintf(stderr, "vtrackd: camera %u's picture does not fit a slot\n", k);
                return -1;
            }
            gen_pic[k] = arena_take(&a, gen_bytes[k], 64);
            if (!gen_pic[k]) {
                fprintf(stderr, "vtrackd: no memory for the generated pictures\n");
                return -1;
            }
            tk_set_picture(trk[k], gen_pic[k]);
        }
    struct live_depth *ld = NULL;
    if (with_depth) {
        ld = ld_create(cfg, tm, n, &a, dma);
        if (!ld) {
            fprintf(stderr, "vtrackd: no depth map\n");
            return -1;
        }
    }
    const uint64_t depth_pub_addr = dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, depth_pub);
    uint64_t cam_frames[VT_MAX_CAMERAS] = {0, 0, 0}, calib_seen = 0;
    f.nsets = tk_sets(tr);
    __atomic_store_n(&f.tr, tr, __ATOMIC_RELEASE);
    for (int i = 0; i < DEV_SLOTS; i++) {
        g.dev[i] = arena_take(&a, slot_bytes, 64);
        if (!g.dev[i]) {
            fprintf(stderr, "vtrackd: no memory for the frames\n");
            return -1;
        }
    }
    /* A camera through FSR: its memory and its picture. */
    const int fsr_cam = (int)cfg->fsr_camera - 1;
    uint8_t *fsr_pics[2] = {NULL, NULL};
    uint32_t fsr_turn = 0;
    static struct fsr_worker fsr_w;
    memset(&fsr_w, 0, sizeof fsr_w);
    pthread_t fsr_th;

    uint64_t fsr_seq = 0;
    if (cfg->fsr_camera) {
        struct tracker *ft = fsr_cam <= 0 ? tr : trk[fsr_cam];
        fsr_pics[0] = arena_take(&a, (uint64_t)cfg->fsr_w * cfg->fsr_h, 64);
        fsr_pics[1] = arena_take(&a, (uint64_t)cfg->fsr_w * cfg->fsr_h, 64);
        if (cfg->fsr_async) {
            /* The tracker has the first threads of every core it uses; these are the rest. */
            int nw = 0, taken = 0;
            for (int c = 0; c < topo->ncores; c++) {
                const struct topo_core *k = &topo->core[c];
                if (k->reserved || k->ncpus <= tpc)
                    continue;
                for (int i = tpc; i < k->ncpus && nw < 256; i++) {
                    fsr_w.cpus[nw] = k->cpu[i];
                    fsr_w.core_of[nw] = taken;
                    nw++;
                }
                taken++;
            }
            fsr_w.n = nw;
            fsr_w.fsr_cam = fsr_cam;
            if (!nw || pthread_create(&fsr_th, NULL, fsr_worker_main, &fsr_w)) {
                fprintf(stderr, "vtrackd: no threads left for FSR\n");
                return -1;
            }
            int ok;
            while (!(ok = __atomic_load_n(&fsr_w.team_ok, __ATOMIC_ACQUIRE)))
                vt_delay(IDLE_POLL_CYCLES);
            if (ok < 0)
                return -1;
            printf("vtrackd: FSR team %d threads, on the threads the tracker leaves\n", nw);
        }
        if ((uint64_t)cfg->fsr_w * cfg->fsr_h > VT_FSR_PIC_BYTES || !fsr_pics[0] || !fsr_pics[1] || tk_fsr_setup(ft, (int)cfg->fsr_up_w, (int)cfg->fsr_up_h, (int)cfg->fsr_w, (int)cfg->fsr_h, (long double)cfg->fsr_sharp, arena_alloc, &a, fsr_pics[0], fsr_pics[1], cfg->fsr_async ? fsr_worker_run : NULL, cfg->fsr_async ? (void *)&fsr_w : NULL)) {
            fprintf(stderr, "vtrackd: no room for FSR at %ux%u\n", cfg->fsr_w, cfg->fsr_h);
            return -1;
        }
    }
    /* Every camera's flow on that team, the operator's order, 2026-11-04. The session's team keeps the rest. */
    const int flow_on = cfg->fsr_async && cfg->fsr_camera;
    struct vt_gen_meta *gen_meta[VT_MAX_CAMERAS] = {NULL, NULL, NULL};
    uint64_t gen_meta_phys[VT_MAX_CAMERAS] = {0, 0, 0}, gen_dma[VT_MAX_CAMERAS] = {0, 0, 0}, gen_seq[VT_MAX_CAMERAS] = {0, 0, 0};
    uint32_t flow_seen[VT_MAX_CAMERAS] = {0, 0, 0};
    int flow_pending[VT_MAX_CAMERAS] = {0, 0, 0}, flow_skipped[VT_MAX_CAMERAS] = {0, 0, 0};
    uint64_t flow_instance[VT_MAX_CAMERAS] = {0, 0, 0}, flow_pending_instance[VT_MAX_CAMERAS] = {0, 0, 0};
    uint64_t flow_waits = 0, flow_replaced = 0;
    if (flow_on) {
        for (uint32_t k = 0; k < vt_cameras(cfg); k++) {
            if (!gen_pic[k])
                continue;
            if (tk_flow_apart(trk[k], arena_alloc, &a, fsr_worker_run, &fsr_w)) {
                fprintf(stderr, "vtrackd: no room for camera %u's flow on the second team\n", k);
                return -1;
            }
            gen_meta[k] = arena_take(&a, sizeof(struct vt_gen_meta), 64);
            if (!gen_meta[k])
                return -1;
            gen_meta_phys[k] = arena_phys(&a, gen_meta[k]);
            fsr_w.slots[k].t = trk[k];
        }
        tk_fsr_between(trk[fsr_cam], flow_between, &fsr_w);
        __atomic_store_n(&fsr_w.ncams, (int)vt_cameras(cfg), __ATOMIC_RELEASE);
    }
    /* L and R's vectors on this session's team, at half their rate, each the moment nothing is waiting: the frames never wait for them. Their own way to the window, so a map can no longer hold them up. */
    int cams_on = 0, att_wanted[VT_CELL_CAMERAS] = {0, 0};
    uint32_t cam_tick[VT_CELL_CAMERAS] = {0, 0};
    uint64_t cam_seq[VT_CELL_CAMERAS] = {0, 0}, cam_dma[VT_CELL_CAMERAS] = {0, 0}, cells_meta_phys[VT_CELL_CAMERAS] = {0, 0};
    struct vt_cells_meta *cells_meta[VT_CELL_CAMERAS] = {NULL, NULL};
    if (cfg->vec_overlay && cfg->fsr_async && ld && fsr_cam == 2 && ld_cell_bytes(ld, 0) <= VT_CELL_SLOT_BYTES && ld_cell_bytes(ld, 1) <= VT_CELL_SLOT_BYTES) {
        cells_meta[0] = arena_take(&a, 64, 64);
        cells_meta[1] = arena_take(&a, 64, 64);
        if (cells_meta[0] && cells_meta[1]) {
            cells_meta_phys[0] = arena_phys(&a, cells_meta[0]);
            cells_meta_phys[1] = arena_phys(&a, cells_meta[1]);
            cams_on = 1;
        }
    }
    /* The orange vectors on the core phitop shows empty: a block poller's, which sleeps while nothing swaps. Its other threads; not the poller's own, nor ingest's. */
    static struct vec_worker vec_w;
    memset(&vec_w, 0, sizeof vec_w);
    pthread_t vec_th;
    int vec_on = 0;
    uint32_t vec_seen = 0;
    uint64_t vec_seq = 0, vec_dma = 0;
    int vec_rest = 0; /* a frame's lines are out, and the session's part of that frame is still to do */
    long double rig_u = 0.0L, rig_v = 0.0L;
    int rig_waiting = 0;
    struct vt_msg vec_msg;
    memset(&vec_msg, 0, sizeof vec_msg);
    struct vt_vec_meta *vec_meta = NULL;
    uint64_t vec_meta_phys = 0;
    if (cfg->vec_overlay && cfg->fsr_async && ld && fsr_cam >= 0 && (uint64_t)cfg->fsr_w * cfg->fsr_h <= VT_VEC_PIC_BYTES) {
        int nv = 0;
        for (int c = 0; c < topo->ncores && !nv; c++) {
            const struct topo_core *k = &topo->core[c];
            if (k->reserved != 2)
                continue;
            for (int i = 0; i < k->ncpus && nv < 8; i++) {
                int taken_cpu = k->cpu[i] == svc;
                for (int p = 0; p < topo->npollers; p++)
                    taken_cpu |= topo->poller_cpu[p] == k->cpu[i];
                if (taken_cpu)
                    continue;
                vec_w.cpus[nv] = k->cpu[i];
                vec_w.core_of[nv] = 0;
                nv++;
            }
        }
        const size_t draw_bytes = vd_bytes(vd_w, vd_h, VT_DEPTH_BYTES / (int)sizeof(struct vt_cell));
        void *draw_mem = nv ? arena_take(&a, draw_bytes, 64) : NULL;
        vec_w.fsr = tk_fsr_of(fsr_cam <= 0 ? tr : trk[fsr_cam]);
        vec_w.temp_each = vec_w.fsr ? (fsr_down_temp_bytes(vec_w.fsr) + 63u) & ~(size_t)63u : 0u;
        vec_w.temp = nv ? arena_take(&a, (uint64_t)nv * vec_w.temp_each + 64u, 64) : NULL;
        vec_w.pic = nv ? arena_take(&a, (uint64_t)cfg->fsr_w * cfg->fsr_h, 64) : NULL;
        vec_meta = nv ? arena_take(&a, sizeof(struct vt_vec_meta), 64) : NULL;
        if (!draw_mem || !vec_w.fsr || !vec_w.temp || !vec_w.pic || !vec_meta) {
            fprintf(stderr, "vtrackd: no spare core or no room for the orange vectors: the window draws them as before\n");
        } else {
            vec_meta_phys = arena_phys(&a, vec_meta);
            vec_w.draw = vd_init(draw_mem, vd_w, vd_h, VT_DEPTH_BYTES / (int)sizeof(struct vt_cell));
            vec_w.ld = ld;
            vec_w.cam = fsr_cam;
            vec_w.level = ld_flow_level(ld);
            vec_w.view = (long double)vd_w / (long double)fsr_camera_w(cfg);
            vec_w.n = nv;
            {
                int gc, gr;
                ld_cells(ld, fsr_cam, &gc, &gr);
                if (gimref_init(&vec_w.gim, gc, gr, ld_flow_level(ld))) {
                    fprintf(stderr, "vtrackd: no memory for camera D's reference\n");
                    return -1;
                }
            }
            if (pthread_create(&vec_th, NULL, vec_worker_main, &vec_w)) {
                fprintf(stderr, "vtrackd: no thread for the orange vectors\n");
                return -1;
            }
            int ok;
            while (!(ok = __atomic_load_n(&vec_w.team_ok, __ATOMIC_ACQUIRE)))
                vt_delay(IDLE_POLL_CYCLES);
            if (ok < 0)
                return -1;
            vec_on = 1;
            printf("vtrackd: orange vectors on the spare core, %d threads, cpus %d to %d\n", nv, vec_w.cpus[0], vec_w.cpus[nv - 1]);
        }
    }
    struct vt_result *res = arena_take(&a, VT_RES_BYTES, ARENA_PAGE);
    if (!res)
        return -1;
    const uint64_t res_phys = arena_phys(&a, res);
    struct vt_fsr_meta *fsr_meta = cfg->fsr_async ? arena_take(&a, 64, 64) : NULL;
    const uint64_t fsr_meta_phys = fsr_meta ? arena_phys(&a, fsr_meta) : 0;
    struct vt_result *gres = cfg->framegen ? arena_take(&a, VT_RES_BYTES, ARENA_PAGE) : NULL;
    if (cfg->framegen && !gres)
        return -1;
    const uint64_t gres_phys = gres ? arena_phys(&a, gres) : 0;
    const uint64_t host_res = dma->region_addr + VT_PART_OFFSET + VT_RES_OFFSET;
    const uint64_t pub_addr = dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, results_pub);
    pthread_t ing;
    if (pthread_create(&ing, NULL, ingest_main, &g)) {
        fprintf(stderr, "vtrackd: no ingest thread\n");
        return -1;
    }
    printf("vtrackd: track session: %ux%u %s, %s, track team %d threads, pyramid team %d, %d x %d, ingest on cpu %d, arena %zu MiB (card)\n", cfg->width, cfg->height, cfg->stereo ? "stereo" : "mono", two ? "2 pipelines" : "1 pipeline", n, pn, cores, tpc, svc, a.bytes >> 20);
    ctl->card_session = seq;

    uint64_t nres = 0, last_dma = 0, map_dma = 0;
    uint64_t idle_since = vt_tsc();
    const double hz = 1.238e9; /* the watchdog needs no precision */
    int rc = 0;
    unsigned spin = 0;
    /* Tracking comes first. Each result leaves as soon as it exists, and the still pictures' flow and the map are done only while no frame is waiting, on each camera's newest frame: a job not reached is replaced
     * by the next, and that frame's pyramid is not written again until the job after it has replaced it. If the card falls behind it is the flow that slows, not the tracker or what the window shows. */
    struct {
        struct tracker *t;
        int set, wanted;
        struct vt_msg msg;
        float inside[4];  /* the selected box at this frame, in the camera's pixels */
        int has_box;
    } flow_job[VT_MAX_CAMERAS];
    memset(flow_job, 0, sizeof flow_job);
    struct vt_msg map_msg;
    memset(&map_msg, 0, sizeof map_msg);
    int map_wanted = 0;
    uint64_t depth_cycles = 0;
    uint64_t npub = 0; /* results sent: one per frame taken, and one per frame generated */
    for (;;) {
        /* What the second team has finished, whenever it finishes, and what has waited for it. FSR's camera: the picture half way and the real one, together. The others: the picture made between two real frames, whose row has gone ahead. */
        if (flow_on) {
            int flow_failed = 0;
            for (uint32_t k = 0; k < vt_cameras(cfg) && !flow_failed; k++) {
                if (!fsr_w.slots[k].t)
                    continue;
                struct flow_slot *s = &fsr_w.slots[k];
                const uint32_t r = __atomic_load_n(&s->done, __ATOMIC_ACQUIRE);
                if (r != flow_seen[k]) {
                    if ((int)k == fsr_cam) {
                        /* The meta and its pictures are read by the channel: the last copy goes first. */
                        flow_failed = last_dma && vt_dma_wait(dma, last_dma, 5.0);
                        const uint64_t sent = flow_failed ? 0 : fsr_publish(dma, &a, fsr_pics, fsr_w.with_mid, cfg, fsr_meta, fsr_meta_phys, dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, fsr_meta), dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, fsr_pub), fsr_seq, flow_instance[k]);
                        flow_failed = !sent;
                        last_dma = sent ? sent : last_dma;
                        fsr_seq++;
                    } else {
                        flow_failed = gen_dma[k] && vt_dma_wait(dma, gen_dma[k], 5.0);
                        gen_dma[k] = flow_failed ? 0 : gen_publish(dma, &a, gen_pic[k], gen_bytes[k], k, &gen_turn[k], tk_flow_cycles(trk[k]), gen_meta[k], gen_meta_phys[k], gen_seq[k], flow_instance[k]);
                        flow_failed = !gen_dma[k];
                        last_dma = gen_dma[k] ? gen_dma[k] : last_dma;
                        gen_seq[k]++;
                    }
                    flow_seen[k] = r;
                }
                /* The newest real frame that came while it was busy goes the moment it is free: a little late costs a little, not a whole frame. What it writes must have gone first. */
                if (!flow_failed && flow_pending[k] && flow_seen[k] == s->want) {
                    flow_failed = gen_dma[k] && vt_dma_wait(dma, gen_dma[k], 5.0);
                    flow_pending[k] = 0;
                    tk_flow_take(trk[k], flow_skipped[k]);
                    flow_skipped[k] = 0;
                    flow_instance[k] = flow_pending_instance[k];
                    __atomic_store_n(&s->want, s->want + 1u, __ATOMIC_RELEASE);
                    fsr_w.frames += (int)k == fsr_cam;
                }
            }
            if (flow_failed) {
                fprintf(stderr, "vtrackd: a picture copy refused or timed out\n");
                rc = -1;
                break;
            }
        }
        /* The spare core: a camera's frame goes to it at once and its finished lines come back at once, whether or not a frame is waiting here, or its work would take the time between frames and more. */
        if (vec_on) {
            const uint32_t vr = __atomic_load_n(&vec_w.done, __ATOMIC_ACQUIRE);
            if (vr != vec_seen) {
                vec_seen = vr;
                if (last_dma && vt_dma_wait(dma, last_dma, 5.0)) {
                    fprintf(stderr, "vtrackd: result DMA timed out\n");
                    rc = -1;
                    break;
                }
                vec_dma = vec_publish(dma, &a, &vec_w, cfg, vec_meta, vec_meta_phys, dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, vec_meta), dma->region_addr + VT_PART_OFFSET + offsetof(struct vt_ctl, vec_pub), vec_seq);
                if (!vec_dma) {
                    fprintf(stderr, "vtrackd: copy of the orange vectors refused\n");
                    rc = -1;
                    break;
                }
                last_dma = vec_dma;
                vec_seq++;
                vec_rest = 1;
                /* What D's own reference says moved: for its head, at its next frame. */
                if (cfg->gimbal_ref && vec_w.gim.state == GIM_MOVED) {
                    rig_u += vec_w.gim.move_u;
                    rig_v += vec_w.gim.move_v;
                    rig_waiting = 1;
                }
            }
            if (flow_job[vec_w.cam].wanted && !vec_rest && __atomic_load_n(&vec_w.want, __ATOMIC_ACQUIRE) == vec_seen) {
                /* Its picture must have gone before the next is made over it. */
                if (vec_dma && vt_dma_wait(dma, vec_dma, 5.0)) {
                    fprintf(stderr, "vtrackd: copy of the orange vectors timed out\n");
                    rc = -1;
                    break;
                }
                flow_job[vec_w.cam].wanted = 0;
                vec_msg = flow_job[vec_w.cam].msg;
                vec_w.instance = vec_msg.instance;
                vec_w.cur = tk_level(flow_job[vec_w.cam].t, flow_job[vec_w.cam].set, ld_flow_level(ld));
                vec_w.has_box = flow_job[vec_w.cam].has_box;
                memcpy(vec_w.inside, flow_job[vec_w.cam].inside, sizeof vec_w.inside);
                __atomic_store_n(&vec_w.want, vec_w.want + 1u, __ATOMIC_RELEASE);
            }
        }
        if (g.failed) {
            rc = -1;
            break;
        }
        if (!two && !cfg->dual)
            build_one(&f);
        if ((cfg->dual ? __atomic_load_n(&g.landed, __ATOMIC_ACQUIRE) : __atomic_load_n(&f.built, __ATOMIC_ACQUIRE)) <= nres) {
            /* No frame waiting: one piece of what is below tracking, then look again. */
            if (ld) {
                /* L and R's vectors first of all that waits for a gap. Their own copy to the window. */
                if (cams_on && (flow_job[0].wanted || flow_job[1].wanted)) {
                    const int ck = flow_job[0].wanted ? 0 : 1;
                    const uint64_t t3 = vt_tsc();
                    flow_job[ck].wanted = 0;
                    /* What it fills must have gone before it is filled again. */
                    if (cam_dma[ck] && vt_dma_wait(dma, cam_dma[ck], 5.0)) {
                        fprintf(stderr, "vtrackd: copy of L or R's vectors timed out\n");
                        rc = -1;
                        break;
                    }
                    const struct vt_plane *cc = tk_level(flow_job[ck].t, flow_job[ck].set, ld_flow_level(ld));
                    ld_vectors(ld, ck, cc, tm);
                    ld_update(ld, ck, cc, tm);
                    cam_dma[ck] = cells_publish(dma, &a, ld, ck, flow_job[ck].msg.instance, cells_meta[ck], cells_meta_phys[ck], cam_seq[ck]);
                    if (!cam_dma[ck]) {
                        fprintf(stderr, "vtrackd: copy of L or R's vectors refused\n");
                        rc = -1;
                        break;
                    }
                    last_dma = cam_dma[ck];
                    cam_seq[ck]++;
                    att_wanted[ck] = 1;
                    depth_cycles += vt_tsc() - t3;
                    continue;
                }
                /* What the spare core has finished: the session's part of that frame, then a map can read it. */
                if (vec_rest) {
                    const uint64_t t3 = vt_tsc();
                    vec_rest = 0;
                    ld_rest(ld, vec_w.cam, vec_w.cur);
                    map_msg = vec_msg;
                    map_wanted = 1;
                    depth_cycles += vt_tsc() - t3;
                    continue;
                }
                /* The checks of where L and R point, in what gaps there are: they can wait. */
                if (cams_on && (att_wanted[0] || att_wanted[1])) {
                    const int ak = att_wanted[0] ? 0 : 1;
                    const uint64_t t3 = vt_tsc();
                    att_wanted[ak] = 0;
                    ld_attitude(ld, ak);
                    depth_cycles += vt_tsc() - t3;
                    continue;
                }
                /* Those on other teams were handed over above. */
                int k = 0;
                while (k < VT_MAX_CAMERAS && (!flow_job[k].wanted || (vec_on && k == vec_w.cam) || (cams_on && k < VT_CELL_CAMERAS)))
                    k++;
                if (k < VT_MAX_CAMERAS || map_wanted) {
                    const uint64_t t3 = vt_tsc();
                    if (k < VT_MAX_CAMERAS) {
                        flow_job[k].wanted = 0;
                        ld_frame(ld, k, tk_level(flow_job[k].t, flow_job[k].set, ld_flow_level(ld)));
                        if (k == 2) {
                            map_msg = flow_job[k].msg;
                            map_wanted = 1;
                        }
                    } else {
                        map_wanted = 0;
                        /* The map's own buffers: the last one's copy must have finished. */
                        if (map_dma && vt_dma_wait(dma, map_dma, 5.0)) {
                            fprintf(stderr, "vtrackd: depth map DMA timed out\n");
                            rc = -1;
                            break;
                        }
                        const uint64_t sent = ld_map(ld, &map_msg, depth_pub_addr);
                        if (sent == UINT64_MAX) {
                            fprintf(stderr, "vtrackd: depth map copy refused\n");
                            rc = -1;
                            break;
                        }
                        if (sent)
                            map_dma = sent;
                    }
                    depth_cycles += vt_tsc() - t3;
                    continue;
                }
            }
            /* One uncached read every few thousand polls: the session word is rare, the local counter is not. */
            if ((++spin & 0xfff) == 0) {
                if (ctl->session_end || ctl->quit)
                    break;
                if ((double)(vt_tsc() - idle_since) / hz > NO_FRAME_S) {
                    fprintf(stderr, "vtrackd: no frame for %.0f s: ending the session\n", NO_FRAME_S);
                    break;
                }
            }
            vt_delay(IDLE_POLL_CYCLES);
            continue;
        }
        /* The last result's copy reads res: it must have finished. */
        if (last_dma && vt_dma_wait(dma, last_dma, 5.0)) {
            fprintf(stderr, "vtrackd: result DMA timed out\n");
            rc = -1;
            break;
        }
        int set;
        struct tracker *cur = tr;
        if (cfg->dual) {
            /* Each camera's frames go through its own tracker, in the order they landed; then the two views are one point. */
            const uint32_t di = (uint32_t)(nres % DEV_SLOTS);
            const struct vt_msg *m = &g.msg[di];
            const int cam = (int)m->camera;
            cur = trk[cam];
            set = (int)(cam_frames[cam]++ % (uint64_t)tk_sets(cur));
            const int apart = flow_on && fsr_w.slots[cam].t != NULL;
            /* A set the second team is still reading must not be built over. It has two frames' time: that it should ever wait here is worth counting. */
            if (apart) {
                int waited = 0;
                while (__atomic_load_n(&fsr_w.slots[cam].done, __ATOMIC_ACQUIRE) != fsr_w.slots[cam].want && tk_flow_reads(cur, set)) {
                    waited = 1;
                    vt_delay(IDLE_POLL_CYCLES);
                }
                flow_waits += (uint64_t)waited;
            }
            const uint64_t t0 = vt_tsc();
            tk_build(cur, set, g.dev[di], NULL, m);
            const uint64_t t1 = vt_tsc();
            struct vt_msg gm;
            const int gen_set = apart ? tk_generate_plan(cur, set, m, &gm) : tk_generate(cur, set, m, &gm);
            const uint64_t t_gen = vt_tsc();
            if (gen_set >= 0 && apart) {
                /* Free only once the last job's pictures have gone. If not, the newest waits; the one it replaces is the one let go. */
                struct flow_slot *fs = &fsr_w.slots[cam];
                if (flow_seen[cam] == fs->want) {
                    if (gen_dma[cam] && vt_dma_wait(dma, gen_dma[cam], 5.0)) {
                        fprintf(stderr, "vtrackd: generated picture DMA timed out\n");
                        rc = -1;
                        break;
                    }
                    tk_flow_take(cur, flow_skipped[cam]);
                    flow_skipped[cam] = 0;
                    flow_instance[cam] = m->instance;
                    __atomic_store_n(&fs->want, fs->want + 1u, __ATOMIC_RELEASE);
                    fsr_w.frames += cam == fsr_cam;
                } else {
                    fsr_w.dropped += cam == fsr_cam ? flow_pending[cam] : 0;
                    flow_replaced += (uint64_t)flow_pending[cam];
                    flow_skipped[cam] += flow_pending[cam];
                    flow_pending[cam] = 1;
                    flow_pending_instance[cam] = m->instance;
                }
            } else if (gen_set >= 0 && fsr_cam == cam)
                tk_fsr_frame(cur);
            if (rig_waiting && vec_on && cam == vec_w.cam) {
                if (tk_rig_moved(cur, rig_u, rig_v)) {
                    vec_w.total_u += rig_u;
                    vec_w.total_v += rig_v;
                    vec_w.applied++;
                }
                rig_u = rig_v = 0.0L;
                rig_waiting = 0;
            }
            tk_track(cur, set, m, res);
            if (gen_set >= 0 && send_generated(cur, &gm, t_gen - t1, res, gres, gres_phys, dma, &a, apart ? NULL : gen_pic[cam], gen_bytes[cam], &gen_turn[cam], fsr_cam == cam && !cfg->fsr_async ? fsr_pics[0] : NULL, cfg, &fsr_turn, &npub, &last_dma, host_res, pub_addr)) {
                rc = -1;
                break;
            }
            if (gen_set >= 0 && fsr_cam == cam && !cfg->fsr_async && send_fsr_picture(dma, &a, fsr_pics[1], cfg, res, &fsr_turn)) {
                rc = -1;
                break;
            }
            const uint64_t seq_now = ctl->calib_seq;
            if (seq_now != calib_seen) {
                calib_seen = seq_now;
                struct vt_calib local;
                const volatile uint64_t *from = (const volatile uint64_t *)&ctl->calib;
                uint64_t *to = (uint64_t *)&local;
                for (size_t i = 0; i < sizeof local / 8; i++)
                    to[i] = from[i];
                fuse_set_calib(&fu, &local);
                if (ld)
                    ld_set_calib(ld, &local, seq_now);
            }
            const uint64_t t2 = vt_tsc();
            if (cam < 2) {
                /* A static reference of L or R is not the thing the other camera sees: no 3D from it, the window having left the estimate out of the frame's orders. */
                if (m->dest & VT_PIPE_EST)
                    fuse_frame(&fu, cam, res);
                else
                    fuse_no_view(&fu, cam);
            }
            res->cycles[VT_T_EST] = vt_tsc() - t2;
            if (ld) {
                /* L and R are camera D's static reference: their vectors at half their rate, the operator's order, 2026-11-03. */
                if (!(cams_on && cam < VT_CELL_CAMERAS) || (cam_tick[cam]++ & 1u) == 0u) {
                    flow_job[cam].t = cur;
                    flow_job[cam].set = set;
                    flow_job[cam].msg = *m;
                    flow_job[cam].has_box = res->state == VT_ST_NEW || res->state == VT_ST_OBSERVED || res->state == VT_ST_UNOBSERVED;
                    flow_job[cam].inside[0] = (float)((long double)res->cx - 0.5L * (long double)res->w);
                    flow_job[cam].inside[1] = (float)((long double)res->cy - 0.5L * (long double)res->h);
                    flow_job[cam].inside[2] = (float)((long double)res->cx + 0.5L * (long double)res->w);
                    flow_job[cam].inside[3] = (float)((long double)res->cy + 0.5L * (long double)res->h);
                    flow_job[cam].wanted = 1;
                }
                res->cycles[VT_T_DEPTH] = depth_cycles; /* since the last result */
                depth_cycles = 0;
            }
            nres++;
            res->seq = nres;
            res->cycles[VT_T_DMA_IN] = g.dma_cycles[di];
            res->cycles[VT_T_QUEUE] = t0 - g.landed_tsc[di];
            res->cycles[VT_T_PYR] = t1 - t0;
            __atomic_store_n(&g.taken, nres, __ATOMIC_RELEASE);
        } else {
            set = (int)(nres % (uint64_t)f.nsets);
            const struct built *b = &f.info[set];
            struct vt_msg gm;
            const uint64_t t_gen0 = vt_tsc();
            const int gen_set = tk_generate(tr, set, &b->msg, &gm);
            const uint64_t t_gen = vt_tsc();
            if (gen_set >= 0 && fsr_cam == 0)
                tk_fsr_frame(tr);
            tk_track(tr, set, &b->msg, res);
            if (gen_set >= 0 && send_generated(tr, &gm, t_gen - t_gen0, res, gres, gres_phys, dma, &a, gen_pic[0], gen_bytes[0], &gen_turn[0], fsr_cam == 0 ? fsr_pics[0] : NULL, cfg, &fsr_turn, &npub, &last_dma, host_res, pub_addr)) {
                rc = -1;
                break;
            }
            if (gen_set >= 0 && fsr_cam == 0 && send_fsr_picture(dma, &a, fsr_pics[1], cfg, res, &fsr_turn)) {
                rc = -1;
                break;
            }
            nres++;
            res->seq = nres;
            res->frames_skipped = b->skipped;
            res->cycles[VT_T_DMA_IN] = b->dma_cycles;
            res->cycles[VT_T_QUEUE] = b->queue_cycles;
            res->cycles[VT_T_PYR] = b->pyr_cycles;
        }
        /* Looking at the card's own memory, when asked: the plane goes first on the same channel, so it has landed when the result says so. */
        const struct vt_plane *dbg = tk_debug_plane(cur, set, res);
        if (dbg) {
            const uint64_t plane_bytes_now = (uint64_t)dbg->stride * (uint64_t)dbg->h * sizeof(float);
            if (plane_bytes_now <= VT_XFER_BYTES && !copy_range(dma, &a, plane_row(dbg, 0), dma->region_addr + VT_PART_OFFSET + VT_XFER_OFFSET, plane_bytes_now, 0)) {
                fprintf(stderr, "vtrackd: plane copy refused\n");
                rc = -1;
                break;
            }
        }
        npub++;
        last_dma = vt_dma_copy(dma, res_phys, host_res + (npub - 1) % VT_RES_SLOTS * VT_RES_BYTES, res->hdr.length, pub_addr, npub);
        if (!last_dma) {
            fprintf(stderr, "vtrackd: result copy refused\n");
            rc = -1;
            break;
        }
        __atomic_store_n(&f.tracked, nres, __ATOMIC_RELEASE);
        idle_since = vt_tsc();
    }
    __atomic_store_n(&g.stop, 1, __ATOMIC_RELEASE);
    pthread_join(ing, NULL);
    if (two) {
        __atomic_store_n(&f.stop, 1, __ATOMIC_RELEASE);
        pthread_join(pyr_thread, NULL);
    }
    if (last_dma)
        vt_dma_wait(dma, last_dma, 5.0);
    if (cfg->fsr_async && cfg->fsr_camera) {
        __atomic_store_n(&fsr_w.stop, 1, __ATOMIC_RELEASE);
        pthread_join(fsr_th, NULL);
        printf("vtrackd: FSR took %d real frames of camera %u, let %d go; flow let %llu go in all, and the tracker waited for it %llu times (card)\n", fsr_w.frames, cfg->fsr_camera - 1u, fsr_w.dropped, (unsigned long long)flow_replaced, (unsigned long long)flow_waits);
    }
    if (vec_on) {
        __atomic_store_n(&vec_w.stop, 1, __ATOMIC_RELEASE);
        pthread_join(vec_th, NULL);
    }
    if (map_dma)
        vt_dma_wait(dma, map_dma, 5.0);
    if (cfg->dual) {
        for (uint32_t k = 0; k < vt_cameras(cfg); k++)
        {
            tk_fg_report(trk[k], "vtrackd", hz * 1e-6);
            tk_fsr_report(trk[k], "vtrackd", hz * 1e-6);
        }
    } else {
        tk_fg_report(tr, "vtrackd", hz * 1e-6);
        tk_fsr_report(tr, "vtrackd", hz * 1e-6);
    }
    team_destroy(tm);
    arena_close(&a);
    printf("vtrackd: track session: %llu results (card)\n", (unsigned long long)nres);
    return rc;
}
