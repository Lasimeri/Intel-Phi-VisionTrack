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
#include "live_depth.h"
#include "plane.h"
#include "session.h"
#include "team.h"
#include "track.h"
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

/* The frame between the last real one and this one, tracked, and its result sent, before this one's. Waits for its own result's copy: `res` is written again at once. Nonzero if the channel refused or timed out. */
static int send_generated(struct tracker *t, int set, const struct vt_msg *m, struct vt_result *res, struct vt_dma *dma, const struct arena *a, uint8_t *pic, uint64_t pic_bytes, uint32_t *pic_turn, uint64_t *npub, uint64_t *last_dma, uint64_t res_phys, uint64_t host_res, uint64_t pub_addr)
{
    struct vt_msg gm;
    const uint64_t t0 = vt_tsc();
    const int g = tk_generate(t, set, m, &gm);
    if (g < 0)
        return 0;
    const uint64_t t1 = vt_tsc();
    tk_track(t, g, &gm, res);
    res->cycles[VT_T_GEN] = t1 - t0;
    if (pic) {
        /* The picture first, on the same channel: it has landed when the result says so. */
        const uint32_t slot = (*pic_turn)++ % VT_GEN_SLOTS;
        if (!copy_range(dma, a, pic, dma->region_addr + VT_PART_OFFSET + vt_gen_offset(m->camera, slot), pic_bytes, 0)) {
            fprintf(stderr, "vtrackd: generated picture copy refused\n");
            return -1;
        }
        res->picture = slot + 1;
    }
    (*npub)++;
    res->seq = *npub;
    *last_dma = vt_dma_copy(dma, res_phys, host_res + (*npub - 1) % VT_RES_SLOTS * VT_RES_BYTES, res->hdr.length, pub_addr, *npub);
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
    const size_t need = (two ? 26 : 20) * plane + 20 * plane2 + 20 * plane3 + DEV_SLOTS * slot_bytes + (cfg->dual ? (128u << 20) : (64u << 20)) + (with_depth ? ld_bytes(cfg, n) : 0) + (cfg->gen_pictures ? 3 * VT_GEN_PIC_BYTES : 0);
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
    struct vt_result *res = arena_take(&a, VT_RES_BYTES, ARENA_PAGE);
    if (!res)
        return -1;
    const uint64_t res_phys = arena_phys(&a, res);
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
    } flow_job[VT_MAX_CAMERAS];
    memset(flow_job, 0, sizeof flow_job);
    struct vt_msg map_msg;
    memset(&map_msg, 0, sizeof map_msg);
    int map_wanted = 0;
    uint64_t depth_cycles = 0;
    uint64_t npub = 0; /* results sent: one per frame taken, and one per frame generated */
    for (;;) {
        if (g.failed) {
            rc = -1;
            break;
        }
        if (!two && !cfg->dual)
            build_one(&f);
        if ((cfg->dual ? __atomic_load_n(&g.landed, __ATOMIC_ACQUIRE) : __atomic_load_n(&f.built, __ATOMIC_ACQUIRE)) <= nres) {
            /* No frame waiting: one piece of what is below tracking, then look again. */
            if (ld) {
                int k = 0;
                while (k < VT_MAX_CAMERAS && !flow_job[k].wanted)
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
            const uint64_t t0 = vt_tsc();
            tk_build(cur, set, g.dev[di], NULL, m);
            const uint64_t t1 = vt_tsc();
            if (send_generated(cur, set, m, res, dma, &a, gen_pic[cam], gen_bytes[cam], &gen_turn[cam], &npub, &last_dma, res_phys, host_res, pub_addr)) {
                rc = -1;
                break;
            }
            tk_track(cur, set, m, res);
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
            if (cam < 2)
                fuse_frame(&fu, cam, res);
            res->cycles[VT_T_EST] = vt_tsc() - t2;
            if (ld) {
                flow_job[cam].t = cur;
                flow_job[cam].set = set;
                flow_job[cam].msg = *m;
                flow_job[cam].wanted = 1;
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
            if (send_generated(tr, set, &b->msg, res, dma, &a, gen_pic[0], gen_bytes[0], &gen_turn[0], &npub, &last_dma, res_phys, host_res, pub_addr)) {
                rc = -1;
                break;
            }
            tk_track(tr, set, &b->msg, res);
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
    if (map_dma)
        vt_dma_wait(dma, map_dma, 5.0);
    if (cfg->dual) {
        for (uint32_t k = 0; k < vt_cameras(cfg); k++)
            tk_fg_report(trk[k], "vtrackd", hz * 1e-6);
    } else {
        tk_fg_report(tr, "vtrackd", hz * 1e-6);
    }
    team_destroy(tm);
    arena_close(&a);
    printf("vtrackd: track session: %llu results (card)\n", (unsigned long long)nres);
    return rc;
}
