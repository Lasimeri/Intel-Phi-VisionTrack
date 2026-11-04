/* vt_view.c: the live operator window, on the desktop host. Each camera is always shown. Frames go to the card as message objects; each result is drawn on the very frame it was computed from, so what is
 * on screen is what the card's code saw and concluded. Everything is monochrome, measured and predicted look different, and metric output is shown only when it exists: with one camera, or without a
 * calibration, never.
 *
 * Three cameras are three independent trackers on the card. L (camcorder, the user's left) and R (NexiGo, the user's right) are the primary pair: a region selected in each becomes one 3D point through
 * input/calib/current.txt (reloaded when it changes, frames keep flowing). D (the 720p C310) is the depth map's view and has no vote in that point. Every moment a region is observed in L and in R or D
 * yields a correspondence, appended to out/run/pairs-lr.txt or pairs-ld.txt, tagged with its selection epoch.
 *
 * It outlives the card program: a rebuilt vtrackd (scripts/live.sh restarts it) only costs a banner, then the session comes back and the regions are selected again where they were. While
 * out/run/bench.lock exists a benchmark owns the card: this program leaves the region alone and shows the cameras with a banner.
 *
 *   vt_view [--dev D] [--size WxH] [--fps F] [--dev2 D] [--size2 WxH] [--fps2 F] [--dev3 D] [--size3 WxH] [--fps3 F]  [--rate F] [--rate2 F] [--rate3 F] [--cams 1|2|3] [--at X,Y] [--cores C] [--tpc T] [--levels L] [--feat N]
 *   Keys: drag select a region in any view; right click or d deselect; v next view (camera, then the card's own pyramid planes); c contrast; g blur; [ ] feature budget; m mirror the display; q quit
 *   Orders from out/run/view.cmd, one line, consumed when read: "select CAM X Y W H", "deselect", "view N", "snap N" (N distinct frames summed per camera into out/run/snap-{a,b}.pgm, 16 bits), "map" (the newest depth map into out/run/map.{bin,pgm,ply}).
 *
 * With three cameras the card makes a depth map at every frame of D (live_depth.c): shown beside D, from above in the panel, one line per map in logs/live/depth-*.csv. --avg N: frames a still picture is averaged over. f shows all of the card's vectors, not only the cells that moved. */
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "attitude.h"
#include "calib.h"
#include "dataset.h"
#include "depthview.h"
#include "plane.h"
#include "stars.h"
#include "v4l2cap.h"
#include "vt_link.h"
#include "x11view.h"

#define RING 32
#define SLOTS 9
#define PANEL_ONE_VIEW 640
#define DRAW_MIN_S (1.0 / 60.0) /* with no results to pace them: the cameras as they come */
#define PANEL_S 0.05            /* the panel and the depth pane, at most this often */
#define SHOW_Q 8                /* pictures of one camera waiting for their moment */
#define ARRIVALS 64             /* how late they arrive, remembered over about a second */
#define SHOW_MARGIN_NS 1500000ull
#define CAMERAS_PATH "input/calib/cameras.txt"
#define LOCK_PATH "out/run/bench.lock"
#define CMD_PATH "out/run/view.cmd"
#define POS_PATH "out/run/view.pos"
#define PAIRS_LR_PATH "out/run/pairs-lr.txt"
#define PAIRS_LD_PATH "out/run/pairs-ld.txt"
#define CALIB_PATH "input/calib/current.txt"
#define TSC_HZ 1238089000.0
#define TRAIL 512
#define WHITE 0xffffffu
#define BLACK 0x000000u
#define GREY 0xa0a0a0u
#define DIM 0x707070u

enum { LINK_STARTING, LINK_TRACKING, LINK_NO_CARD, LINK_PAUSED };

/* A picture waiting for its moment, and what the tracker knew of that very picture. A frame generated between two real ones exists only once the newer has arrived, so the newer is held back until the frame before it has had its turn: every picture is shown a fixed delay after its own instant. */
struct shown {
    uint64_t at_ns, recv_ns;
    const uint8_t *pic;
    int generated;
    struct vt_result r;
};

struct cam {
    double vs;        /* how much bigger than the camera its pane is: 1, or FSR's */
    int vw, vh;       /* its pane */
    struct vt_result *row_at; /* FSR's camera: each recent instant's result, real then generated, waiting for its picture */
    const char *dev;
    int w, h, fps;          /* what the device is asked for */
    double rate;            /* frames a second kept of those: the operator's order, 2026-11-03. Each is half of what is shown */
    uint64_t next_ns, skipped; /* when the next one to keep is due; how many have been let go */
    uint8_t *ring;
    uint64_t acq[RING];
    uint64_t captured;      /* distinct frames so far */
    uint64_t raw, repeats;  /* everything the device delivered; how much of it repeated the frame before */
    uint64_t bad;           /* frames that failed to decode */
    const char *name;       /* what is on the screen beside it */
    int yuyv;               /* 1: YUYV, not MJPEG */
    int state;              /* 0 no frame yet, 1 frames coming, 2 frames stopped */
    /* The operator's orders, for the link thread. */
    int cmd;
    float roi[4];
    /* The link thread's results. */
    struct vt_result last;
    uint64_t results;
    uint64_t recv_ns;
    struct rate { double t0; uint64_t n0; double value; } r_new, r_raw, r_res;
    uint64_t generated; /* of its results, how many came from generated frames */
    /* The pictures on their way to the screen. */
    struct shown *q;
    uint64_t q_in, q_out, delay_ns, shown, shown_made, passed;
    uint32_t arrive_us[ARRIVALS], arrivals;
    struct rate r_shown, r_made;
    pthread_t thread;
    /* An integrated snapshot under way: the sum of the frames so far. */
    uint32_t *sum;
    int integ_n, integ_left;
    /* Its fixed lights, a check only: stars.h. They do not carry the geometry and cannot turn the map off: a search around the last fix is a lock, and a light hidden by someone walked it off to something
     * else, which read as the camera moving (2026-11-03: 109 px on a camera that had not moved, 0.07 px on the same pixels with no memory). */
    struct star_cat cat;
    struct star_fix fix;
    uint16_t *star16;
    uint32_t *star_sum;
    /* How far it has turned, from the card's attitude kernel. */
    double q_now[4];           /* since this session's first picture */
    double q_file[4];          /* at the moment the geometry on file was true */
    double att_applied[3];     /* the rotation vector the geometry has been carried through, from the file's */
    int att_pending;
    int lost_run;              /* seconds the camera has not been able to find itself in its reference */
    /* All of the above belongs to the navigation thread. This is what the window may see of it, under A.m. */
    struct nav_view { int used, known, state, lost; double gdop, turned[3]; float corr, corr_none; uint32_t keys; } nav;
};

static struct {
    struct cam c[VT_MAX_CAMERAS];
    int ncam, levels, feat, cores, tpc, quit;
    uint32_t sc_key;
    float sc_value;
    int fsr, fsr_cam, fsr_sharp_new, lr_gen; /* camera D through FSR, in its pane; L and R generated pictures beside it */
    /* The orange vectors, drawn at FSR's upscaled size on the card's spare core and brought down with its pictures: each slot's instant and which rows it has. */
    int vec;
    struct vec_plane { uint64_t instance; uint32_t y0, y1, w, valid; } vec_at[VT_VEC_SLOTS];
    struct vt_vec_meta vec_last;
    uint64_t vec_count;
    /* L and R's vectors, a frame at a time, each slot's instant. */
    struct cells_plane { uint64_t instance; uint32_t cols, rows, level, valid; } cells_at[VT_CELL_CAMERAS][VT_CELL_SLOTS];
    uint64_t cells_count[VT_CELL_CAMERAS];
    float fsr_sharp;
    int contrast, pre_gauss, view_level, mirror;
    uint32_t pair_epoch; /* changes at every selection: pairs from different selections are not one object's path */
    pthread_mutex_t m;
    int link_state;
    uint64_t dropped, exceptions, card_build, pairs;
    struct vt_link l;
    /* The calibration as the card holds it. */
    struct vt_calib calib;      /* as the card holds it: the solved geometry, carried by the star fixes */
    struct vt_calib calib_base; /* as solved */
    char calib_note[120];
    time_t calib_mtime;
    int calib_new;
    double trail[TRAIL][3];
    int trail_n, trail_i;
    /* The depth map: the link thread fills the buffer that is not current, and only then makes it current. */
    int map_i, depth_avg, depth_planes, vectors_all, save_map, flow_level, session_restart, framegen;
    uint64_t map_seq;
    struct rate map_rate;
    struct rate cells_rate[VT_CELL_CAMERAS];
} A;

static uint8_t map_store[2][VT_DEPTH_BYTES] __attribute__((aligned(64)));

static double now_s(void) { return (double)vt_now_ns() * 1e-9; }

/* Told to stop (a rebuild restarts this window): leave through the front door, so the session on the card ends. */
static void on_term(int sig)
{
    (void)sig;
    A.quit = 1;
}

/* In a dark room the picture lies inside one grey level of a sensor's noise, and one frame cannot show it. N distinct frames summed can, as 16 bits a pixel, big-endian as PGM has it. */
static void integrate_frame(struct cam *k, const uint8_t *g)
{
    const size_t bytes = (size_t)k->w * (size_t)k->h;
    for (size_t i = 0; i < bytes; i++)
        k->sum[i] += g[i];
    if (__atomic_sub_fetch(&k->integ_left, 1, __ATOMIC_ACQ_REL))
        return;
    char path[64];
    const int cam = (int)(k - A.c);
    snprintf(path, sizeof path, "out/run/snap-%c.pgm", "lrd"[cam]);
    uint8_t *o = malloc(bytes * 2);
    FILE *f = o ? fopen(path, "wb") : NULL;
    if (f) {
        const uint32_t n = (uint32_t)k->integ_n;
        for (size_t i = 0; i < bytes; i++) {
            const uint32_t v = (uint32_t)(((uint64_t)k->sum[i] * 257u) / n);
            o[2 * i] = (uint8_t)(v >> 8);
            o[2 * i + 1] = (uint8_t)v;
        }
        fprintf(f, "P5\n%d %d\n65535\n", k->w, k->h);
        fwrite(o, 2, bytes, f);
        fclose(f);
        fprintf(stderr, "vt_view: snap camera %d: %u frames integrated, %s\n", cam, n, path);
    }
    free(o);
}

static void *capture_main(void *arg)
{
    struct cam *k = arg;
    struct v4l2cap *c = NULL;
    uint64_t last_hash = 0;
    const size_t bytes = (size_t)k->w * (size_t)k->h;
    while (!A.quit) {
        if (!c) {
            c = v4l2cap_open(k->dev, k->w, k->h, k->fps, k->yuyv);
            if (!c) {
                sleep(1);
                continue;
            }
        }
        const uint64_t i = __atomic_load_n(&k->captured, __ATOMIC_RELAXED);
        uint8_t *dst = k->ring + (i % RING) * bytes;
        uint64_t acq;
        const int r = v4l2cap_next(c, dst, &acq, 500);
        k->bad = v4l2cap_bad(c);
        if (r > 0) {
            k->state = 1;
            k->raw++;
            /* A capture card fed a slower signal delivers the same picture twice. A repeat is no measurement: sent on, it reads as zero motion, and the next frame as double. So it stops
             * here, and the frame keeps the time it really arrived. Sampled every 61st byte: two different sensor frames do not agree on 33,000 samples. */
            uint64_t h = 0xcbf29ce484222325ull;
            for (size_t b = 0; b < bytes; b += 61) {
                h ^= dst[b];
                h *= 0x100000001b3ull;
            }
            if (h == last_hash) {
                k->repeats++;
                continue;
            }
            last_hash = h;
            if (__atomic_load_n(&k->integ_left, __ATOMIC_ACQUIRE) > 0)
                integrate_frame(k, dst);
            /* Kept by when it was taken, not by counting: the rate holds if the device slips. Within half a device frame of its time is its time. */
            if (k->rate < (double)k->fps) {
                const uint64_t period = (uint64_t)(1e9 / k->rate), half = 500000000ull / (uint64_t)k->fps;
                if (acq + half < k->next_ns) {
                    k->skipped++;
                    continue;
                }
                k->next_ns = k->next_ns && acq < k->next_ns + period ? k->next_ns + period : acq + period;
            }
            k->acq[i % RING] = acq;
            __atomic_store_n(&k->captured, i + 1, __ATOMIC_RELEASE);
        } else if (r == 0) {
            if (k->state == 1)
                k->state = 2;
        } else {
            fprintf(stderr, "vt_view: capture error on %s: reopening\n", k->dev);
            v4l2cap_close(c);
            c = NULL;
        }
    }
    if (c)
        v4l2cap_close(c);
    return NULL;
}

/* The session this window asks of the card. */
static void session_config(struct vt_config *c)
{
    struct scene s;
    memset(&s, 0, sizeof s);
    s.id = SC_COUNT;
    s.w = A.c[0].w;
    s.h = A.c[0].h;
    s.f = s.w;
    s.cx = 0.5 * (s.w - 1);
    s.cy = 0.5 * (s.h - 1);
    vt_default_config(c, &s, 0);
    c->cores = (uint32_t)A.cores;
    c->tpc = (uint32_t)A.tpc;
    c->levels = (uint32_t)A.levels;
    c->max_features = (uint32_t)A.feat;
    c->max_missing = (uint32_t)(A.c[0].rate * 2.0);
    c->contrast = (uint32_t)A.contrast;
    c->pre_gauss = (uint32_t)A.pre_gauss;
    c->slots = SLOTS;
    if (A.ncam >= 2) {
        c->dual = 1;
        c->width2 = (uint32_t)A.c[1].w;
        c->height2 = (uint32_t)A.c[1].h;
        c->width3 = A.ncam == 3 ? (uint32_t)A.c[2].w : 0;
        c->height3 = A.ncam == 3 ? (uint32_t)A.c[2].h : 0;
        /* The rays are as good as the calibration, not as the pixels: hand-picked geometry is off by far more than a tracker's noise. */
        c->sigma_c = 1.0f;
        c->r_floor = 0.005f;
        c->q_accel = 1.0f;
        c->max_reproj = 25.0f;
    }
    if (A.ncam == 3) {
        /* As the geometry was refined and judged offline (docs/measurements.md). */
        c->depth_planes = (uint32_t)A.depth_planes;
        c->depth_agg = 3;
        c->depth_avg = (uint32_t)A.depth_avg;
        c->depth_near = 0.8f;
        c->depth_far = 8.0f;
        c->depth_trunc = 1.5f;
        c->depth_tex = 2.0f * VT_Q6;
        c->depth_conf = 0.05f;
        c->depth_eps = 1.0f * VT_Q6;
        /* A quarter of a pixel at level 2, the same distance in the world at any level: at native flow an unscaled limit called 22 % of R's still cells moving. */
        c->motion_px = 0.25f * (float)(1 << (VT_DEPTH_LEVEL - A.flow_level));
        c->motion_grey = 2.0f;
        c->flow_level = (uint32_t)A.flow_level;
        c->framegen = (uint32_t)A.framegen;
        c->gen_pictures = (uint32_t)A.framegen;
        /* 720p12.5 up to 1440p by FSR, doubled to 25 a second by the flow from the pictures, and a Lanczos to 1080p: the user's order, 2026-11-03. */
        if (A.fsr && A.framegen) {
            c->fsr_camera = 3;
            c->fsr_up_w = 2u * (uint32_t)A.c[2].w;
            c->fsr_up_h = 2u * (uint32_t)A.c[2].h;
            c->fsr_w = 3u * (uint32_t)A.c[2].w / 2u;
            c->fsr_h = 3u * (uint32_t)A.c[2].h / 2u;
            c->fsr_sharp = A.fsr_sharp;
            c->vec_overlay = (uint32_t)A.vec;
            c->fsr_async = 1;
            /* L and R's generated pictures are only for the eye: the tracker never reads them. --lr-gen 0 gives their time to FSR. */
            c->gen_cameras = A.lr_gen ? 0u : 1u << 2;
        }
    }
}

/* A correspondence: both regions observed, camera L's centre carried to the other camera's time along its rate. */
static void note_pair(FILE *f, const struct vt_result *a, const struct vt_result *b, double *last)
{
    if (a->state != VT_ST_OBSERVED || b->state != VT_ST_OBSERVED || ((a->flags | b->flags) & VT_F_AT_REST))
        return;
    const double gap = ((double)b->hdr.acq_ns - (double)a->hdr.acq_ns) * 1e-9;
    if (gap > 0.04 || gap < -0.04)
        return;
    const double xa = (double)a->pt_x + (double)a->rate_u * gap, ya = (double)a->pt_y + (double)a->rate_v * gap;
    const double xb = (double)b->pt_x, yb = (double)b->pt_y;
    /* Only a real change of place adds information. */
    if ((xa - last[0]) * (xa - last[0]) + (ya - last[1]) * (ya - last[1]) + (xb - last[2]) * (xb - last[2]) + (yb - last[3]) * (yb - last[3]) < 36.0)
        return;
    last[0] = xa;
    last[1] = ya;
    last[2] = xb;
    last[3] = yb;
    fprintf(f, "%.3f %.3f %.3f %.3f e%u\n", xa, ya, xb, yb, A.pair_epoch);
    fflush(f);
    A.pairs++;
}

/* The picture of the result just merged into K->last, due at its own instant plus this camera's delay: the 95th percentile of how late pictures arrive, measured. A generated picture is the latest to arrive against its own instant, half a frame plus the card's time, so with generation on the delay is about that, and with it off it falls by itself to the card's time alone. At most 2.5 frames: a slot of the part is written again after three. */
static void show_push(struct cam *K, const struct vt_result *r, const uint8_t *pic, int generated)
{
    const uint64_t now = vt_now_ns(), acq = r->hdr.acq_ns;
    K->arrive_us[K->arrivals % ARRIVALS] = now > acq ? (uint32_t)((now - acq) / 1000u) : 0u;
    K->arrivals++;
    if (K->arrivals >= ARRIVALS && (K->arrivals & 15u) == 0u) {
        uint32_t s[ARRIVALS];
        for (int i = 0; i < ARRIVALS; i++) {
            const uint32_t v = K->arrive_us[i];
            int at = i;
            for (; at > 0 && s[at - 1] > v; at--)
                s[at] = s[at - 1];
            s[at] = v;
        }
        const uint64_t most = K->rate > 0.0 ? (uint64_t)(2.5e9 / K->rate) : 100000000ull, want = (uint64_t)s[ARRIVALS * 19 / 20] * 1000u + SHOW_MARGIN_NS;
        K->delay_ns = want < most ? want : most;
    }
    struct shown *s = &K->q[K->q_in % SHOW_Q];
    s->at_ns = acq + K->delay_ns;
    s->recv_ns = now;
    s->pic = pic;
    s->generated = generated;
    memcpy(&s->r, r, offsetof(struct vt_result, feat) + (size_t)r->nfeat * sizeof(struct vt_feat_out));
    K->q_in++;
    if (K->q_in - K->q_out > SHOW_Q) {
        K->q_out = K->q_in - SHOW_Q;
        K->passed++;
    }
}

static void *link_main(void *arg)
{
    (void)arg;
    struct vt_ctl *c = A.l.ctl;
    int active = 0;
    uint64_t seq = 0, pub = 0, seen = 0, sent[VT_MAX_CAMERAS] = {0, 0, 0}, instance[VT_MAX_CAMERAS] = {0, 0, 0}, last_taken = 0;
    double last_progress = 0.0;
    float resume_roi[VT_MAX_CAMERAS][4];
    int resume[VT_MAX_CAMERAS] = {0, 0, 0}, debug_sent = -1;
    FILE *pf[VT_MAX_CAMERAS] = {NULL, NULL, NULL}, *track_csv = NULL, *depth_csv = NULL;
    uint64_t fsr_seen = 0, depth_seen = 0, vec_seen = 0;
    uint64_t cells_seen[VT_CELL_CAMERAS] = {0, 0};
    unsigned depth_lines = 0;
    double pair_last[VT_MAX_CAMERAS][4];
    for (int k = 0; k < VT_MAX_CAMERAS; k++)
        for (int i = 0; i < 4; i++)
            pair_last[k][i] = 1e9;
    size_t stride = 0;
    while (!A.quit) {
        if (access(LOCK_PATH, F_OK) == 0) {
            if (active) {
                vt_session_end(&A.l);
                vt_session_wait_end(&A.l, 2.0);
                active = 0;
            }
            A.link_state = LINK_PAUSED;
            usleep(100000);
            continue;
        }
        /* The order "flow N": a new session on the card, at that level. */
        if (active && __atomic_exchange_n(&A.session_restart, 0, __ATOMIC_ACQ_REL)) {
            vt_session_end(&A.l);
            vt_session_wait_end(&A.l, 2.0);
            active = 0;
        }
        if (!active) {
            struct vt_config cfg;
            session_config(&cfg);
            if (vt_session_start(&A.l, &cfg, 2.0)) {
                A.link_state = LINK_NO_CARD;
                usleep(500000);
                continue;
            }
            seq = c->session_seq;
            active = 1;
            pub = seen = depth_seen = 0;
            vec_seen = 0;
            cells_seen[0] = cells_seen[1] = 0;
            stride = vt_slot_stride(&cfg);
            for (int k = 0; k < A.ncam; k++)
                sent[k] = __atomic_load_n(&A.c[k].captured, __ATOMIC_ACQUIRE);
            last_progress = now_s();
            last_taken = 0;
            debug_sent = -1;
            A.card_build = c->card_build;
            A.link_state = LINK_TRACKING;
            pthread_mutex_lock(&A.m);
            A.calib_new = 1;
            /* A new card program: the regions go on where they were. */
            for (int k = 0; k < A.ncam; k++) {
                const struct vt_result *r = &A.c[k].last;
                resume[k] = r->state == VT_ST_OBSERVED || r->state == VT_ST_NEW || r->state == VT_ST_UNOBSERVED;
                if (resume[k]) {
                    resume_roi[k][0] = r->cx - 0.5f * r->w;
                    resume_roi[k][1] = r->cy - 0.5f * r->h;
                    resume_roi[k][2] = r->w;
                    resume_roi[k][3] = r->h;
                }
            }
            pthread_mutex_unlock(&A.m);
        }
        if (c->card_session != seq) {
            active = 0;
            A.link_state = LINK_NO_CARD;
            continue;
        }
        pthread_mutex_lock(&A.m);
        if (A.calib_new && A.ncam >= 2) {
            c->calib = A.calib;
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            c->calib_seq = c->calib_seq + 1;
            A.calib_new = 0;
        }
        pthread_mutex_unlock(&A.m);
        int did = 0;
        for (int k = 0; k < A.ncam; k++) {
            struct cam *K = &A.c[k];
            const uint64_t cap = __atomic_load_n(&K->captured, __ATOMIC_ACQUIRE);
            if (cap <= sent[k])
                continue;
            const uint64_t newest = cap - 1;
            const size_t bytes = (size_t)K->w * (size_t)K->h;
            if (pub - c->frames_taken < SLOTS) {
                const uint32_t slot = (uint32_t)(pub % SLOTS);
                memcpy(A.l.part + VT_FRAME_OFFSET + (size_t)slot * stride, K->ring + (newest % RING) * bytes, bytes);
                struct vt_msg m;
                memset(&m, 0, sizeof m);
                m.magic = VT_MSG_MAGIC;
                m.type = VT_MSG_FRAME;
                m.dest = VT_PIPE_PYR | VT_PIPE_FLOW | VT_PIPE_EST;
                m.priority = VT_PRIO_ORDERED;
                m.algorithm = VT_ALG_MVEX;
                m.camera = (uint32_t)k;
                m.length = (uint32_t)vt_slot_bytes((uint32_t)K->w, (uint32_t)K->h, 0);
                m.instance = ++instance[k];
                m.acq_ns = K->acq[newest % RING];
                m.pub_ns = newest; /* the frame's place in this window's ring: it comes back with the result */
                m.slot = slot;
                pthread_mutex_lock(&A.m);
                if (resume[k]) {
                    m.cmd = VT_CMD_SELECT;
                    memcpy(m.roi, resume_roi[k], sizeof m.roi);
                    resume[k] = 0;
                } else if (K->cmd != VT_CMD_NONE) {
                    m.cmd = (uint32_t)K->cmd;
                    memcpy(m.roi, K->roi, sizeof m.roi);
                    K->cmd = VT_CMD_NONE;
                    A.pair_epoch++;
                }
                if (k == 0) {
                    if (A.view_level != debug_sent) {
                        m.sc_key = VT_SC_DEBUG_LEVEL;
                        m.sc_value = (float)A.view_level;
                        debug_sent = A.view_level;
                    } else if (A.sc_key != VT_SC_NONE) {
                        m.sc_key = A.sc_key;
                        m.sc_value = A.sc_value;
                        A.sc_key = VT_SC_NONE;
                    }
                }
                if (k == A.fsr_cam && A.fsr_sharp_new) {
                    m.sc_key = VT_SC_FSR_SHARP;
                    m.sc_value = A.fsr_sharp;
                    A.fsr_sharp_new = 0;
                }
                pthread_mutex_unlock(&A.m);
                m.check = vt_msg_check(&m);
                c->desc[pub % VT_DESC_RING] = m;
                __atomic_thread_fence(__ATOMIC_SEQ_CST);
                c->frames_pub = ++pub;
            } else {
                A.dropped += cap - sent[k];
            }
            sent[k] = cap;
            did = 1;
        }
        const uint64_t rp = c->results_pub;
        const size_t head = offsetof(struct vt_result, feat);
        for (; seen < rp; seen++) {
            const uint8_t *src = A.l.part + VT_RES_OFFSET + seen % VT_RES_SLOTS * VT_RES_BYTES;
            const uint32_t from = ((const struct vt_result *)src)->hdr.camera;
            const uint32_t k = from < (uint32_t)A.ncam ? from : 0;
            struct cam *K = &A.c[k];
            pthread_mutex_lock(&A.m);
            const struct vt_result *fresh = (const struct vt_result *)src;
            const int generated = (fresh->flags & VT_F_GENERATED) != 0;
            if (generated) {
                /* A generated frame measures nothing: its box and features lie between the two real frames around it, and what is known in 3D stays as the last real frame left it. */
                K->last.state = fresh->state;
                K->last.flags |= VT_F_GENERATED;
                K->last.hdr.acq_ns = fresh->hdr.acq_ns;
                K->last.hdr.instance = fresh->hdr.instance;
                K->last.cx = fresh->cx;
                K->last.cy = fresh->cy;
                K->last.w = fresh->w;
                K->last.h = fresh->h;
                K->last.pt_x = fresh->pt_x;
                K->last.pt_y = fresh->pt_y;
                K->last.rate_u = fresh->rate_u;
                K->last.rate_v = fresh->rate_v;
                K->last.inliers = fresh->inliers;
                K->last.nfeat = fresh->nfeat;
                K->last.reacq_ncc = fresh->reacq_ncc;
            } else {
                memcpy(&K->last, src, head);
            }
            memcpy(K->last.feat, src + head, (size_t)K->last.nfeat * sizeof(struct vt_feat_out));
            K->results++;
            K->generated += generated;
            K->recv_ns = vt_now_ns();
            /* FSR's camera: the picture comes later and on its own; what was known of that instant waits for it. */
            if (K->row_at) {
                memcpy(&K->row_at[(K->last.hdr.instance & 15u) * 2u + (generated != 0)], &K->last, offsetof(struct vt_result, feat) + (size_t)K->last.nfeat * sizeof(struct vt_feat_out));
            } else
            /* What this result is a picture of: a generated picture from the part, a real one from this window's ring, where its place travelled with the frame. */
            if (generated) {
                if (fresh->picture)
                    show_push(K, &K->last, A.l.part + vt_gen_offset(k, fresh->picture - 1), 1);
            } else {
                show_push(K, &K->last, K->ring + (K->last.hdr.pub_ns % RING) * (size_t)K->w * (size_t)K->h, 0);
            }
            A.exceptions = c->exceptions;
            /* Every 3D result, as the card computed it: what any claim about the live numbers is measured from, not what a screen showed. */
            if (K->last.state != VT_ST_NONE || (K->last.flags & (VT_F_3D_MEASURED | VT_F_3D_REJECTED | VT_F_METRIC))) {
                if (!track_csv) {
                    char path[80];
                    const time_t wall = time(NULL);
                    struct tm tm;
                    localtime_r(&wall, &tm);
                    strftime(path, sizeof path, "logs/live/track-%Y%m%d-%H%M%S.csv", &tm);
                    track_csv = fopen(path, "w");
                    if (track_csv)
                        fprintf(track_csv, "acq_s,camera,instance,state,flags,epoch,cx,cy,meas_x,meas_y,meas_z,reproj_px,parallax_deg,gap_ms,nis,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,sig_x,sig_y,sig_z,age_s,w,h,nfeat,inliers,reacq_ncc,lat_ms\n");
                }
                if (track_csv) {
                    const struct vt_result *q = &K->last;
                    fprintf(track_csv, "%.6f,%u,%llu,%u,%u,%u,%.3f,%.3f,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.1f,%.1f,%u,%.3f,%.3f,%.3f\n", (double)q->hdr.acq_ns * 1e-9, k, (unsigned long long)q->hdr.instance, q->state, q->flags, A.pair_epoch, (double)q->cx, (double)q->cy, q->meas[0], q->meas[1], q->meas[2], q->reproj_rms, q->parallax_deg, q->pair_gap_ms, q->nis, q->pos[0], q->pos[1], q->pos[2], q->vel[0], q->vel[1], q->vel[2], sqrt(q->pos_cov[0]), sqrt(q->pos_cov[4]), sqrt(q->pos_cov[8]), q->age_s, (double)q->w, (double)q->h, q->nfeat, (double)q->inliers, (double)q->reacq_ncc, (double)(K->recv_ns - q->hdr.acq_ns) * 1e-6);
                    fflush(track_csv);
                }
            }
            if (k >= 1 && !generated) {
                if (!pf[k])
                    pf[k] = fopen(k == 1 ? PAIRS_LR_PATH : PAIRS_LD_PATH, "a");
                if (pf[k])
                    note_pair(pf[k], &A.c[0].last, &K->last, pair_last[k]);
            }
            if ((K->last.flags & VT_F_METRIC) && !generated) {
                memcpy(A.trail[A.trail_i], K->last.pos, sizeof A.trail[0]);
                A.trail_i = (A.trail_i + 1) % TRAIL;
                if (A.trail_n < TRAIL)
                    A.trail_n++;
            }
            pthread_mutex_unlock(&A.m);
            did = 1;
        }
        /* Two pictures from FSR when they are ready, each due at its own instant: 30 a second from 15. */
        const uint64_t fp = c->fsr_pub;
        if (A.fsr && fp != fsr_seen) {
            fsr_seen = fp;
            const uint32_t fslot = c->fsr_meta.slot, fmid = c->fsr_meta.slot_mid;
            const uint64_t finst = c->fsr_meta.instance;
            struct cam *F = &A.c[A.fsr_cam];
            const struct vt_result *row_real = &F->row_at[(finst & 15u) * 2u], *row_mid = &F->row_at[(finst & 15u) * 2u + 1u];
            pthread_mutex_lock(&A.m);
            if (fslot < VT_FSR_SLOTS && fmid < VT_FSR_SLOTS && row_real->hdr.instance == finst) {
                if (row_mid->hdr.instance == finst)
                    show_push(F, row_mid, A.l.part + vt_fsr_offset(fmid), 1);
                show_push(F, row_real, A.l.part + vt_fsr_offset(fslot), 0);
            }
            pthread_mutex_unlock(&A.m);
        }
        /* A frame's vectors of L or R, likewise. */
        for (int cv = 0; A.vec && cv < VT_CELL_CAMERAS; cv++) {
            const uint64_t cp = c->cells_pub[cv];
            if (cp == cells_seen[cv])
                continue;
            cells_seen[cv] = cp;
            struct vt_cells_meta cm;
            memcpy(&cm, &c->cells_meta[cv], sizeof cm);
            if (cm.slot < VT_CELL_SLOTS) {
                pthread_mutex_lock(&A.m);
                A.cells_at[cv][cm.slot] = (struct cells_plane){cm.instance, cm.cols, cm.rows, cm.level, 1u};
                A.cells_count[cv]++;
                pthread_mutex_unlock(&A.m);
            }
        }
        /* A real frame's orange vectors, in a slot of their own: found by instant when its picture comes due. */
        const uint64_t vp = c->vec_pub;
        if (A.vec && vp != vec_seen) {
            vec_seen = vp;
            struct vt_vec_meta vm;
            memcpy(&vm, &c->vec_meta, sizeof vm);
            if (vm.slot < VT_VEC_SLOTS) {
                pthread_mutex_lock(&A.m);
                A.vec_at[vm.slot] = (struct vec_plane){vm.instance, vm.y0, vm.y1, vm.w, 1u};
                A.vec_last = vm;
                A.vec_count++;
                pthread_mutex_unlock(&A.m);
            }
        }
        /* The newest map, whole: the card wrote every byte of this buffer before it wrote the word that names it. */
        const uint64_t dp = c->depth_pub;
        if (dp && dp != depth_seen) {
            depth_seen = dp;
            const int next = !A.map_i;
            memcpy(map_store[next], A.l.part + VT_DEPTH_OFFSET + ((dp - 1) & 1) * VT_DEPTH_BYTES, VT_DEPTH_BYTES);
            pthread_mutex_lock(&A.m);
            A.map_i = next;
            A.map_seq = dp;
            pthread_mutex_unlock(&A.m);
            if (!depth_csv) {
                char path[80];
                const time_t wall = time(NULL);
                struct tm tm;
                localtime_r(&wall, &tm);
                strftime(path, sizeof path, "logs/live/depth-%Y%m%d-%H%M%S.csv", &tm);
                depth_csv = fopen(path, "w");
            }
            if (depth_csv)
                dv_csv(depth_csv, map_store[next], (double)time(NULL), TSC_HZ, depth_lines++);
            did = 1;
        }
        const uint64_t taken = c->frames_taken;
        if (taken != last_taken) {
            last_taken = taken;
            last_progress = now_s();
        } else if (pub > taken && now_s() - last_progress > 3.0) {
            /* Published frames not being taken: that program is gone. */
            active = 0;
            A.link_state = LINK_NO_CARD;
            continue;
        }
        if (!did)
            usleep(100);
    }
    if (active) {
        vt_session_end(&A.l);
        vt_session_wait_end(&A.l, 2.0);
    }
    for (int k = 0; k < VT_MAX_CAMERAS; k++)
        if (pf[k])
            fclose(pf[k]);
    if (track_csv)
        fclose(track_csv);
    if (depth_csv)
        fclose(depth_csv);
    return NULL;
}

/* ---- drawing ---- */

static void outline_line(struct x11view *v, int x0, int y0, int x1, int y1, uint32_t c, int dashed)
{
    if (dashed) {
        x11view_dash(v, x0 + 1, y0 + 1, x1 + 1, y1 + 1, BLACK, 10, 8);
        x11view_dash(v, x0, y0, x1, y1, c, 10, 8);
    } else {
        x11view_line(v, x0 + 1, y0 + 1, x1 + 1, y1 + 1, BLACK);
        x11view_line(v, x0, y0, x1, y1, c);
    }
}

static void text_outlined(struct x11view *v, int x, int y, int s, uint32_t c, const char *t)
{
    x11view_text(v, x + 1, y + 1, s, BLACK, t);
    x11view_text(v, x, y, s, c, t);
}

static void banner(struct x11view *v, int w, const char *t)
{
    x11view_fill(v, 0, 0, w, 26, BLACK);
    x11view_text(v, 8, 5, 2, WHITE, t);
}

/* The cameras face the user: as they deliver it, the camera at the user's left reads as a view from the right, and the other way round. So the DISPLAY mirrors, like a mirror on the desk: the card's
 * frames never do, a mirrored image would reverse a pinhole camera's handedness and the geometry would solve for a reflected world. Everything in image coordinates goes through this on its way to the screen. */
static int show_x(int ox, int w, double x)
{
    return A.mirror ? ox + w - 1 - (int)x : ox + (int)x;
}

static void show_plane(struct x11view *v, int w, int h, int level, const struct vt_result *r)
{
    /* The card's own pyramid plane, as it holds it: Q6 grey, shown at its pixels' true extent. */
    const float *pl = (const float *)(A.l.part + VT_XFER_OFFSET);
    const int s = (int)r->debug_stride;
    for (int y = 0; y < h; y++) {
        const int py = y >> level;
        if (py >= (int)r->debug_h)
            break;
        const float *row = pl + (size_t)py * (size_t)s;
        uint32_t *d = v->px + (size_t)y * (size_t)v->w;
        for (int x = 0; x < w; x++) {
            const int px = x >> level;
            const int g = px < (int)r->debug_w ? (int)row[px] >> 6 : 0;
            d[A.mirror ? w - 1 - x : x] = (uint32_t)(g < 0 ? 0 : g > 255 ? 255 : g) * 0x010101u;
        }
    }
}

static const char *state_name(uint32_t s)
{
    switch (s) {
    case VT_ST_NEW: return "NEWLY SELECTED";
    case VT_ST_OBSERVED: return "OBSERVED";
    case VT_ST_UNOBSERVED: return "UNOBSERVED";
    case VT_ST_LOST: return "LOST";
    case VT_ST_DESELECTED: return "DESELECTED";
    default: return "NO REGION";
    }
}

static void rate_update(struct rate *r, uint64_t n, double t)
{
    if (t - r->t0 >= 0.5) {
        r->value = (double)(n - r->n0) / (t - r->t0);
        r->n0 = n;
        r->t0 = t;
    }
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int panel_x;
static void panel_line(struct x11view *v, int *y, uint32_t c, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static void panel_line(struct x11view *v, int *y, uint32_t c, const char *fmt, ...)
{
    char buf[96];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    x11view_text(v, panel_x + 10, *y, 2, c, buf);
    *y += 20;
}

/* One view: its frame and what the card concluded about it. */
static void draw_view(struct x11view *v, int ox, int oy, const struct cam *K, const struct vt_result *r, int tracking)
{
    const int has_region = tracking && (r->state == VT_ST_NEW || r->state == VT_ST_OBSERVED || r->state == VT_ST_UNOBSERVED);
    if (!has_region)
        return;
    const int pred = (r->flags & VT_F_PREDICTED) != 0;
    const double s = K->vs;
    const int ex0 = show_x(ox, K->vw, (double)(r->cx - 0.5f * r->w) * s), ex1 = show_x(ox, K->vw, (double)(r->cx + 0.5f * r->w) * s);
    const int x0 = ex0 < ex1 ? ex0 : ex1, x1 = ex0 < ex1 ? ex1 : ex0, y0 = oy + (int)((double)(r->cy - 0.5f * r->h) * s), y1 = oy + (int)((double)(r->cy + 0.5f * r->h) * s);
    outline_line(v, x0, y0, x1, y0, WHITE, pred);
    outline_line(v, x1, y0, x1, y1, WHITE, pred);
    outline_line(v, x1, y1, x0, y1, WHITE, pred);
    outline_line(v, x0, y1, x0, y0, WHITE, pred);
    char tag[64];
    snprintf(tag, sizeof tag, "REGION %u  %s", r->track_id, pred ? "PREDICTED, NOT MEASURED" : state_name(r->state));
    text_outlined(v, x0, y0 - 20 < oy ? y0 + 4 : y0 - 20, 2, WHITE, tag);
    for (unsigned k = 0; k < r->nfeat; k++) {
        const struct vt_feat_out *f = &r->feat[k];
        const int fx = show_x(ox, K->vw, (double)f->x * s), fy = oy + (int)((double)f->y * s);
        /* A vector eight times its length, from where the feature was: a pixel of motion is visible. */
        outline_line(v, fx, fy, show_x(ox, K->vw, (double)(f->x - 8.0f * f->u) * s), fy - (int)((double)(8.0f * f->v) * s), GREY, 0);
        x11view_fill(v, fx - 2, fy - 2, 5, 5, BLACK);
        x11view_fill(v, fx - 1, fy - 1, 3, 3, (f->flags & VT_FF_MEMBER) ? WHITE : GREY);
    }
}

/* The epipolar line: where a point seen in one camera must lie in the other, if the calibration is right. A check anyone can read off the screen. */
static void draw_epipolar(struct x11view *v, const struct vt_calib *c, const struct vt_result *a, const double *R, const double *T, double fb, double cxb, double cyb, const struct cam *B, int ox, int oy)
{
    const double ax = ((double)a->pt_x - c->cxa) / c->fa, ay = ((double)a->pt_y - c->cya) / c->fa;
    const double p[3] = {R[0] * ax + R[1] * ay + R[2], R[3] * ax + R[4] * ay + R[5], R[6] * ax + R[7] * ay + R[8]};
    /* T x p, in normalized coordinates of B: l0 x + l1 y + l2 = 0. */
    const double l0 = T[1] * p[2] - T[2] * p[1], l1 = T[2] * p[0] - T[0] * p[2], l2 = T[0] * p[1] - T[1] * p[0];
    if (l0 == 0.0 && l1 == 0.0)
        return;
    int have = 0, x0 = 0, y0 = 0;
    for (int xp = 0; xp < B->w; xp += 8) {
        const double nx = ((double)xp - cxb) / fb;
        if (l1 == 0.0)
            break;
        const double ny = -(l0 * nx + l2) / l1;
        const int yp = (int)(ny * fb + cyb);
        if (yp < 0 || yp >= B->h) {
            have = 0;
            continue;
        }
        if (have)
            x11view_dash(v, show_x(ox, B->vw, x0 * B->vs), oy + (int)(y0 * B->vs), show_x(ox, B->vw, xp * B->vs), oy + (int)(yp * B->vs), GREY, 3, 5);
        have = 1;
        x0 = xp;
        y0 = yp;
    }
}

/* Top-down: the reconstructed path, camera A at the bottom, the bar one metre. */
static void draw_trail(struct x11view *v, int x, int y, int size, const struct vt_calib *calib)
{
    x11view_fill(v, x, y, size, size, 0x0c0c0cu);
    x11view_dash(v, x + size / 2, y, x + size / 2, y + size, 0x303030u, 2, 6);
    /* Mirrored with the pictures above it, so a move to one side there is a move to the same side here. Six metres: the rear wall is about 4.8 from camera A. */
    const double px_per_m = (double)size / 6.0, side = A.mirror ? -px_per_m : px_per_m;
    const int cx = x + size / 2, cy = y + size - 4;
    x11view_fill(v, cx - 2, cy - 2, 5, 5, GREY);
    x11view_text(v, cx + 6, cy - 6, 1, DIM, "camera A");
    x11view_fill(v, x + 8, y + 8, (int)px_per_m, 2, DIM);
    x11view_text(v, x + 8, y + 12, 1, DIM, "1 m");
    if (calib->valid) {
        const double *R = calib->R, *T = calib->T;
        const double C[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
        const int bx = cx + (int)(C[0] * side), by = cy - (int)(C[2] * px_per_m);
        if (bx > x && bx < x + size && by > y && by < y + size) {
            x11view_fill(v, bx - 2, by - 2, 5, 5, GREY);
            x11view_text(v, bx + 6, by - 3, 1, DIM, "camera B");
        }
    }
    if (calib->valid_d) {
        const double *R = calib->Rd, *T = calib->Td;
        const double C[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
        const int bx = cx + (int)(C[0] * side), by = cy - (int)(C[2] * px_per_m);
        if (bx > x && bx < x + size && by > y && by < y + size) {
            x11view_fill(v, bx - 2, by - 2, 5, 5, GREY);
            x11view_text(v, bx + 6, by - 3, 1, DIM, "camera D");
        }
        if (A.map_seq)
            dv_cloud(v, map_store[A.map_i], calib, x, y, size, cx, cy, side, px_per_m);
    }
    int lx = 0, ly = 0, have = 0;
    for (int i = 0; i < A.trail_n; i++) {
        const double *p = A.trail[(A.trail_i - A.trail_n + i + TRAIL) % TRAIL];
        const int px = cx + (int)(p[0] * side), py = cy - (int)(p[2] * px_per_m);
        if (px < x || px >= x + size || py < y || py >= y + size) {
            have = 0;
            continue;
        }
        const uint32_t g = (uint32_t)(60 + 195 * i / (A.trail_n > 1 ? A.trail_n - 1 : 1)) * 0x010101u;
        if (have)
            x11view_line(v, lx, ly, px, py, g);
        have = 1;
        lx = px;
        ly = py;
    }
    if (have)
        x11view_fill(v, lx - 2, ly - 2, 5, 5, WHITE);
}

/* The window's panes, in window pixels: the cameras at 1:1, the depth map beside camera D, the panel. */
struct pane {
    int x, y, w, h;
};
static struct pane pane[VT_MAX_CAMERAS], depth_pane, panel;

static void layout(int *win_w, int *win_h)
{
    for (int k = 0; k < A.ncam; k++) {
        A.c[k].vs = 1.0;
        A.c[k].vw = A.c[k].w;
        A.c[k].vh = A.c[k].h;
    }
    /* Camera D as FSR makes it: half as big again, and its pane with it. */
    const int with_fsr = A.fsr && A.framegen && A.ncam == 3;
    if (with_fsr) {
        A.c[2].vs = 1.5;
        A.c[2].vw = 3 * A.c[2].w / 2;
        A.c[2].vh = 3 * A.c[2].h / 2;
        A.c[2].row_at = calloc(32, sizeof(struct vt_result));
        if (!A.c[2].row_at)
            exit(2);
    }
    const struct cam *a = &A.c[0], *b = &A.c[1], *c = &A.c[2];
    pane[0] = (struct pane){0, 0, a->w, a->h};
    if (A.ncam == 1) {
        panel = (struct pane){a->w, 0, PANEL_ONE_VIEW, a->h};
        *win_w = a->w + PANEL_ONE_VIEW;
        *win_h = a->h;
        return;
    }
    pane[1] = (struct pane){a->w, 0, b->w, b->h};
    const int top = a->h > b->h ? a->h : b->h, total = a->w + b->w;
    if (A.ncam == 2) {
        panel = (struct pane){0, top, total, 720};
    } else {
        pane[2] = (struct pane){0, top, c->vw, c->vh};
        if (with_fsr) {
            depth_pane = (struct pane){c->vw, top, c->w / 2, c->h / 2};
            panel = (struct pane){c->vw + c->w / 2, top, total - c->vw - c->w / 2, c->vh};
        } else {
            depth_pane = (struct pane){c->w, top, c->w, c->h};
            panel = (struct pane){2 * c->w, top, total - 2 * c->w, c->h};
        }
    }
    *win_w = total;
    *win_h = top + panel.h;
}

/* The camera whose picture this window point is in, or -1. */
static int pane_at(int x, int y)
{
    for (int k = 0; k < A.ncam; k++)
        if (x >= pane[k].x && x < pane[k].x + pane[k].w && y >= pane[k].y && y < pane[k].y + pane[k].h)
            return k;
    return -1;
}

/* Starts an integrated snapshot of N distinct frames in every camera: the capture threads finish it. */
static void start_snapshot(int n)
{
    for (int k = 0; k < A.ncam; k++) {
        struct cam *K = &A.c[k];
        if (__atomic_load_n(&K->integ_left, __ATOMIC_ACQUIRE) > 0)
            continue;
        const size_t bytes = (size_t)K->w * (size_t)K->h;
        if (!K->sum)
            K->sum = malloc(bytes * sizeof(uint32_t));
        if (!K->sum)
            continue;
        memset(K->sum, 0, bytes * sizeof(uint32_t));
        K->integ_n = n;
        __atomic_store_n(&K->integ_left, n, __ATOMIC_RELEASE);
    }
}

static void apply_command_file(void)
{
    FILE *f = fopen(CMD_PATH, "r");
    if (!f)
        return;
    char line[96] = "";
    const int got = fgets(line, sizeof line, f) != NULL;
    fclose(f);
    remove(CMD_PATH);
    if (!got)
        return;
    float r[4];
    int n, cam;
    pthread_mutex_lock(&A.m);
    if (sscanf(line, "select %d %f %f %f %f", &cam, &r[0], &r[1], &r[2], &r[3]) == 5 && cam >= 0 && cam < A.ncam) {
        A.c[cam].cmd = VT_CMD_SELECT;
        memcpy(A.c[cam].roi, r, sizeof r);
    } else if (!strncmp(line, "deselect", 8)) {
        for (int k = 0; k < A.ncam; k++)
            A.c[k].cmd = VT_CMD_DESELECT;
    } else if (sscanf(line, "view %d", &n) == 1) {
        A.view_level = n;
    } else if (!strncmp(line, "snap", 4)) {
        const int want = atoi(line + 4);
        start_snapshot(want > 0 && want <= 1 << 16 ? want : 1);
    } else if (!strncmp(line, "map", 3)) {
        A.save_map = 1;
    } else if (sscanf(line, "flow %d", &n) == 1 && n >= 0 && n <= VT_DEPTH_LEVEL) {
        A.flow_level = n;
        __atomic_store_n(&A.session_restart, 1, __ATOMIC_RELEASE);
    } else if (sscanf(line, "framegen %d", &n) == 1 && (n == 0 || n == 1)) {
        A.framegen = n;
        __atomic_store_n(&A.session_restart, 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&A.m);
    fprintf(stderr, "vt_view: command file: %s", line);
}

#define STAR_AVG_FRAMES 16
#define ATT_APPLY_RAD (0.03 * M_PI / 180.0) /* a change the kernel must show twice running before the geometry follows it */
#define ATT_LOST_TICKS 5     /* seconds without a good answer before the geometry is called lost */
#define ATT_LOST_VIEW 0.25f  /* of the reference still in view */
#define ATT_LOST_CORR 0.2f   /* the picture against its reference */
#define ATT_KEEP_TICKS 10    /* seconds a carried geometry must hold still before it goes on file */

/* The geometry as it holds now: what was solved, carried through each camera's attitude: star_correct_pair. */
static void geometry_from_attitudes(void)
{
    double Oa[9], Ob[9], Od[9];
    const double *a = A.c[0].att_applied, *b = A.c[1].att_applied, *d = A.c[2].att_applied;
    star_rotation(a[0], a[1], a[2], Oa);
    star_rotation(b[0], b[1], b[2], Ob);
    star_rotation(d[0], d[1], d[2], Od);
    pthread_mutex_lock(&A.m);
    A.calib = A.calib_base;
    if (A.calib_base.valid)
        star_correct_pair(A.calib_base.R, A.calib_base.T, Oa, Ob, A.calib.R, A.calib.T);
    if (A.calib_base.valid_d)
        star_correct_pair(A.calib_base.Rd, A.calib_base.Td, Oa, Od, A.calib.Rd, A.calib.Td);
    A.calib_new = 1;
    pthread_mutex_unlock(&A.m);
}

/* Once a second, each camera: a fix against its fixed lights, for the log and the panel. Each from the catalogue itself, with no memory of the last one. */
static void star_tick(void)
{
    static unsigned tick;
    static FILE *csv;
    tick++;
    for (int k = 0; k < A.ncam; k++) {
        struct cam *K = &A.c[k];
        const uint64_t cap = __atomic_load_n(&K->captured, __ATOMIC_ACQUIRE);
        if (!cap)
            continue;
        if (!K->cat.n) {
            char path[64];
            snprintf(path, sizeof path, "input/calib/stars-%c.txt", "lrd"[k]);
            if (star_cat_read(path, &K->cat) || !K->cat.n)
                continue;
            fprintf(stderr, "vt_view: stars: camera %c catalogue of %d\n", "LRD"[k], K->cat.n);
        }
        const size_t n = (size_t)K->w * (size_t)K->h;
        if (!K->star16)
            K->star16 = malloc(n * sizeof(uint16_t));
        if (!K->star16)
            continue;
        /* A short long-exposure: the mean of the last frames. One frame of the camcorder in this light cannot show a blurred LED. The newest 16, none of them the one being written. */
        const int nf = cap >= STAR_AVG_FRAMES ? STAR_AVG_FRAMES : (int)cap;
        if (!K->star_sum)
            K->star_sum = malloc(n * sizeof(uint32_t));
        if (!K->star_sum)
            continue;
        memset(K->star_sum, 0, n * sizeof(uint32_t));
        for (int fr = 0; fr < nf; fr++) {
            const uint8_t *frame = K->ring + ((cap - 1 - (uint64_t)fr) % RING) * (uint64_t)K->w * (uint64_t)K->h;
            for (size_t i = 0; i < n; i++)
                K->star_sum[i] += frame[i];
        }
        const uint32_t scale = 257u, divide = (uint32_t)nf;
        for (size_t i = 0; i < n; i++)
            K->star16[i] = (uint16_t)(K->star_sum[i] * scale / divide);
        const double f = k == 0 ? A.calib_base.fa : k == 1 ? A.calib_base.fb : A.calib_base.fd;
        /* A whole-frame search only now and then: it costs tens of milliseconds, and a light behind the user is no reason to spend them every second. */
        star_track(&K->cat, K->star16, K->w, K->h, f > 1.0 ? f : (double)K->w, 0.0, 0.0, (tick & 7u) == 0, &K->fix);
        const struct star_fix *x = &K->fix;
        if (!csv) {
            char path[80];
            const time_t wall = time(NULL);
            struct tm tm;
            localtime_r(&wall, &tm);
            strftime(path, sizeof path, "logs/live/stars-%Y%m%d-%H%M%S.csv", &tm);
            csv = fopen(path, "w");
            if (csv)
                fprintf(csv, "wall_s,camera,known,seen,used,lost_in_space,dx_px,dy_px,tilt_deg,pan_deg,roll_deg,residual_px,gdop_deg_per_px\n");
        }
        if (csv) {
            fprintf(csv, "%lld,%c,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.4f\n", (long long)time(NULL), "LRD"[k], x->known, x->seen, x->used, x->lost_in_space, x->dx, x->dy, x->wx * 180.0 / M_PI, x->wy * 180.0 / M_PI, x->wz * 180.0 / M_PI, x->rms_px, x->gdop);
            fflush(csv);
        }
    }
}

/* The rotation vector of a quaternion, w x y z. */
static void q_to_vector(const double *q, double *v)
{
    const double s = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    const double k = s > 1e-12 ? 2.0 * atan2(s, q[0]) / s : 2.0;
    v[0] = q[1] * k;
    v[1] = q[2] * k;
    v[2] = q[3] * k;
}

/* The carried geometry onto the file, so a restart does not fall back to a pose the rig has left. The file then is the truth, and the turns start again from here. */
static void keep_on_file(void)
{
    pthread_mutex_lock(&A.m);
    const struct vt_calib now = A.calib;
    pthread_mutex_unlock(&A.m);
    static const char note[] = "carried by the attitude kernel from the solved geometry";
    const char *tmp = CALIB_PATH ".new";
    struct stat st;
    if (calib_write(tmp, &now, note) || rename(tmp, CALIB_PATH) || stat(CALIB_PATH, &st)) {
        fprintf(stderr, "vt_view: the carried geometry could not be kept on file\n");
        return;
    }
    pthread_mutex_lock(&A.m);
    A.calib_base = now;
    A.calib_mtime = st.st_mtime;
    snprintf(A.calib_note, sizeof A.calib_note, "%s", note);
    for (int k = 0; k < A.ncam; k++) {
        memcpy(A.c[k].q_file, A.c[k].q_now, sizeof A.c[k].q_file);
        memset(A.c[k].att_applied, 0, sizeof A.c[k].att_applied);
    }
    pthread_mutex_unlock(&A.m);
    fprintf(stderr, "vt_view: the carried geometry kept on file\n");
}

/* Once a second: each camera's turn as the card's attitude kernel last measured it, carried into the geometry. It holds nothing, so it cannot be walked off: each answer comes from the whole picture again. */
static void attitude_tick(void)
{
    static uint64_t last_seq;
    static FILE *csv;
    static int unkept, steady;
    struct vt_motion mo[VT_MAX_CAMERAS];
    uint64_t seq = 0;
    pthread_mutex_lock(&A.m);
    if (A.map_seq) {
        const struct vt_depth_out *h = (const void *)map_store[A.map_i];
        memcpy(mo, h->cam, sizeof mo);
        seq = h->seq;
    }
    pthread_mutex_unlock(&A.m);
    if (!seq || !A.calib_base.valid)
        return;
    int changed = 0;
    /* A new session on the card: its first picture is a new origin. The geometry on file holds. */
    if (seq < last_seq) {
        for (int k = 0; k < A.ncam; k++) {
            static const double one[4] = {1.0, 0.0, 0.0, 0.0};
            memcpy(A.c[k].q_file, one, sizeof one);
            memcpy(A.c[k].q_now, one, sizeof one);
            memset(A.c[k].att_applied, 0, sizeof A.c[k].att_applied);
        }
        changed = 1;
        unkept = 0;
    }
    last_seq = seq;
    if (!csv) {
        char path[80];
        const time_t wall = time(NULL);
        struct tm tm;
        localtime_r(&wall, &tm);
        strftime(path, sizeof path, "logs/live/attitude-%Y%m%d-%H%M%S.csv", &tm);
        csv = fopen(path, "w");
        if (csv)
            fprintf(csv, "wall_s,camera,state,tilt_deg,pan_deg,roll_deg,corr,corr_none,used,view,peak_z,keys,card_ms,stars_used,stars_dx,stars_dy\n");
    }
    for (int k = 0; k < A.ncam; k++) {
        struct cam *K = &A.c[k];
        const struct vt_motion *m = &mo[k];
        if (m->att_state == VT_ATT_NONE) {
            K->lost_run = 0;
            continue;
        }
        memcpy(K->q_now, m->att_q, sizeof K->q_now);
        /* Since the file's moment: now, after undoing the file's. */
        const double back[4] = {K->q_file[0], -K->q_file[1], -K->q_file[2], -K->q_file[3]};
        double rel[4], v[3];
        att_q_mul(K->q_now, back, rel);
        q_to_vector(rel, v);
        const double d0 = v[0] - K->att_applied[0], d1 = v[1] - K->att_applied[1], d2 = v[2] - K->att_applied[2];
        if (sqrt(d0 * d0 + d1 * d1 + d2 * d2) > ATT_APPLY_RAD) {
            if (K->att_pending++) {
                memcpy(K->att_applied, v, sizeof v);
                K->att_pending = 0;
                changed = 1;
                unkept = 1;
                steady = 0;
                fprintf(stderr, "vt_view: attitude: camera %c turned tilt %+.3f pan %+.3f roll %+.3f deg, correlation %.3f: geometry carried\n", "LRD"[k], v[0] * 180.0 / M_PI, v[1] * 180.0 / M_PI, v[2] * 180.0 / M_PI, (double)m->att_corr);
            }
        } else {
            K->att_pending = 0;
        }
        const float corr = m->att_state == VT_ATT_TURNED ? m->att_corr : m->att_corr_none;
        const int good = m->att_view >= ATT_LOST_VIEW && corr >= ATT_LOST_CORR;
        const int was_lost = K->lost_run >= ATT_LOST_TICKS;
        K->lost_run = good ? 0 : K->lost_run + 1;
        if (!was_lost && K->lost_run >= ATT_LOST_TICKS)
            fprintf(stderr, "vt_view: attitude: camera %c cannot find itself: correlation %.3f, %.2f in view\n", "LRD"[k], (double)corr, (double)m->att_view);
        if (csv)
            fprintf(csv, "%lld,%c,%u,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.1f,%u,%.3f,%d,%.2f,%.2f\n", (long long)time(NULL), "LRD"[k], m->att_state, v[0] * 180.0 / M_PI, v[1] * 180.0 / M_PI, v[2] * 180.0 / M_PI, (double)m->att_corr, (double)m->att_corr_none, (double)m->att_used, (double)m->att_view, (double)m->att_peak_z, m->att_keys, (double)m->cycles_att * 1e3 / TSC_HZ, K->fix.used, K->fix.dx, K->fix.dy);
    }
    if (csv)
        fflush(csv);
    if (changed)
        geometry_from_attitudes();
    if (unkept && ++steady >= ATT_KEEP_TICKS) {
        keep_on_file();
        unkept = 0;
        steady = 0;
    }
}

/* A camera that has not been able to find itself: its geometry is not known to hold. With A.m held. L and R make the 3D point; the map needs D as well. */
static int calib_lost_pair(void)
{
    return A.calib.valid && (A.c[0].nav.lost || A.c[1].nav.lost);
}

static void load_calibration(void)
{
    struct stat st;
    if (stat(CALIB_PATH, &st)) {
        /* Withdrawn: the card must stop using it as well. */
        if (A.calib.valid) {
            pthread_mutex_lock(&A.m);
            memset(&A.calib, 0, sizeof A.calib);
            memset(&A.calib_base, 0, sizeof A.calib_base);
            A.calib_new = 1;
            A.calib_mtime = 0;
            A.calib_note[0] = 0;
            A.trail_n = 0;
            pthread_mutex_unlock(&A.m);
            fprintf(stderr, "vt_view: calibration withdrawn: %s is gone\n", CALIB_PATH);
        }
        return;
    }
    if (st.st_mtime == A.calib_mtime)
        return;
    struct vt_calib c;
    if (calib_read(CALIB_PATH, &c)) {
        fprintf(stderr, "vt_view: %s is unreadable\n", CALIB_PATH);
        pthread_mutex_lock(&A.m);
        A.calib_mtime = st.st_mtime;
        pthread_mutex_unlock(&A.m);
        return;
    }
    /* The file's first line says what its scale rests on: the panel shows exactly that. */
    char note[sizeof A.calib_note] = "";
    FILE *nf = fopen(CALIB_PATH, "r");
    if (nf) {
        if (fgets(note, sizeof note, nf) && note[0] == '#') {
            memmove(note, note + 2, strlen(note + 2) + 1);
            note[strcspn(note, "\n")] = 0;
        } else {
            note[0] = 0;
        }
        fclose(nf);
    }
    pthread_mutex_lock(&A.m);
    A.calib = c;
    A.calib_base = c;
    /* A newly solved geometry belongs to the pose of its own moment: the attitudes start again from it. So a catalogue must be rebuilt whenever geometry is solved anew. */
    for (int k = 0; k < VT_MAX_CAMERAS; k++)
        memset(A.c[k].att_applied, 0, sizeof A.c[k].att_applied);
    A.calib_new = 1;
    A.calib_mtime = st.st_mtime;
    A.trail_n = 0;
    memcpy(A.calib_note, note, sizeof note);
    pthread_mutex_unlock(&A.m);
    /* A geometry and its catalogues are one pose: the catalogues are read again. The turns count from here. */
    for (int k = 0; k < A.ncam; k++) {
        struct cam *K = &A.c[k];
        K->cat.n = 0;
        memset(&K->fix, 0, sizeof K->fix);
        memcpy(K->q_file, K->q_now, sizeof K->q_file);
        K->att_pending = K->lost_run = 0;
    }
    fprintf(stderr, "vt_view: calibration loaded: fa %.1f fb %.1f baseline %.3f m\n", c.fa, c.fb, sqrt(c.T[0] * c.T[0] + c.T[1] * c.T[1] + c.T[2] * c.T[2]));
}

/* Everything that reads whole frames once a second, on a thread of its own. On the window's thread it stopped the drawing for 33 to 40 ms at every tick, 2 or 3 displayed frames lost each second
 * (measured, docs/measurements.md), with the cameras and the card running steadily behind it. The window sees only what is published here. */
static void *nav_main(void *arg)
{
    (void)arg;
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    for (int k = 0; k < A.ncam; k++)
        A.c[k].q_now[0] = A.c[k].q_file[0] = 1.0;
    while (!A.quit) {
        load_calibration();
        star_tick();
        attitude_tick();
        pthread_mutex_lock(&A.m);
        struct vt_motion mo[VT_MAX_CAMERAS];
        const int have = A.map_seq != 0;
        if (have)
            memcpy(mo, ((const struct vt_depth_out *)(const void *)map_store[A.map_i])->cam, sizeof mo);
        for (int k = 0; k < A.ncam; k++) {
            struct cam *K = &A.c[k];
            K->nav = (struct nav_view){K->fix.used, K->fix.known, have ? (int)mo[k].att_state : VT_ATT_NONE, K->lost_run >= ATT_LOST_TICKS, K->fix.gdop, {K->att_applied[0], K->att_applied[1], K->att_applied[2]}, have ? mo[k].att_corr : 0.0f, have ? mo[k].att_corr_none : 0.0f, have ? mo[k].att_keys : 0u};
        }
        pthread_mutex_unlock(&A.m);
        next.tv_sec++;
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    struct cam *a = &A.c[0], *b = &A.c[1], *d = &A.c[2];
    /* By by-id path from a local file: a replug renumbers /dev/videoN, and left and right must not swap with it. Those paths carry serial numbers, so they are not in this source. */
    static char devs[3][V4L2CAP_PATH] = {"/dev/video0", "/dev/video2", "/dev/video4"};
    if (v4l2cap_names(CAMERAS_PATH, devs) != 3)
        fprintf(stderr, "vt_view: %s does not name all three cameras; the others go by number, which a replug can swap\n", CAMERAS_PATH);
    a->dev = devs[0];
    a->name = "L  camcorder";
    a->w = 1920;
    a->h = 1080;
    a->fps = 30;
    a->rate = 24.0;
    b->dev = devs[1];
    b->name = "R  NexiGo";
    b->w = 1920;
    b->h = 1080;
    b->fps = 30;
    b->rate = 24.0;
    d->dev = devs[2];
    d->name = "D  C310";
    d->w = 1280;
    d->h = 720;
    /* The rates are the user's, 11-03. In MJPEG this camera claims its largest alternate setting at any frame rate, so it must not share a USB 2 bus with the NexiGo, which does the same: STREAMON
     * answers ENOSPC. Measured at 720p15 and 1080p30 on one bus; solved by the two being on different controllers. */
    d->fps = 25;
    d->rate = 12.5; /* every second frame of its own 25: even, 80 ms, where 12.5 of 15 would leave a gap of 133 */
    A.ncam = 3;
    A.mirror = 1;
    A.levels = 5;
    A.feat = 128;
    A.tpc = 2;
    A.view_level = -1;
    A.depth_avg = 128;
    A.depth_planes = 64;
    /* Half of native: the user's choice, 2026-11-03, native being laggy to use though it holds its frame rate (capture to result 47.5 ms, here 35.4). docs/measurements.md. */
    A.flow_level = 1;
    A.framegen = 1;
    A.fsr = 1;
    A.lr_gen = 1; /* 60, 60 and 29 a second with FSR running, measured 2026-11-03: --lr-gen 0 gives the card back if it is ever short */
    A.fsr_sharp = 0.4f; /* AMD: (1 - 0.8) x 2 stops, its sample's default */
    A.vec = 1;
    A.fsr_cam = 2;
    int at_x = -1, at_y = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dev") && i + 1 < argc) a->dev = argv[++i];
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &a->w, &a->h);
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) a->fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev2") && i + 1 < argc) b->dev = argv[++i];
        else if (!strcmp(argv[i], "--size2") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &b->w, &b->h);
        else if (!strcmp(argv[i], "--fps2") && i + 1 < argc) b->fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev3") && i + 1 < argc) d->dev = argv[++i];
        else if (!strcmp(argv[i], "--size3") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &d->w, &d->h);
        else if (!strcmp(argv[i], "--fps3") && i + 1 < argc) d->fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rate") && i + 1 < argc) a->rate = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rate2") && i + 1 < argc) b->rate = atof(argv[++i]);
        else if (!strcmp(argv[i], "--rate3") && i + 1 < argc) d->rate = atof(argv[++i]);
        else if (!strcmp(argv[i], "--raw3") && i + 1 < argc) d->yuyv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cams") && i + 1 < argc) A.ncam = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--at") && i + 1 < argc) sscanf(argv[++i], "%d,%d", &at_x, &at_y);
        else if (!strcmp(argv[i], "--cores") && i + 1 < argc) A.cores = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tpc") && i + 1 < argc) A.tpc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--levels") && i + 1 < argc) A.levels = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--feat") && i + 1 < argc) A.feat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--avg") && i + 1 < argc) A.depth_avg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--flow") && i + 1 < argc) A.flow_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--framegen") && i + 1 < argc) A.framegen = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "--fsr") && i + 1 < argc) A.fsr = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "--lr-gen") && i + 1 < argc) A.lr_gen = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "--vec") && i + 1 < argc) A.vec = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "--noframe")) x11view_noframe(1);
        else if (!strcmp(argv[i], "--planes") && i + 1 < argc) A.depth_planes = atoi(argv[++i]);
        else {
            fprintf(stderr, "vt_view: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (A.ncam < 1 || A.ncam > VT_MAX_CAMERAS) {
        fprintf(stderr, "vt_view: 1 to %d cameras\n", VT_MAX_CAMERAS);
        return 2;
    }
    for (int k = 0; k < A.ncam; k++)
        if (A.c[k].w & 15) {
            fprintf(stderr, "vt_view: camera %d: width %d is not a multiple of 16\n", k, A.c[k].w);
            return 2;
        }
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);
    pthread_mutex_init(&A.m, NULL);
    for (int k = 0; k < A.ncam; k++) {
        A.c[k].ring = malloc((size_t)RING * (size_t)A.c[k].w * (size_t)A.c[k].h);
        A.c[k].q = malloc(SHOW_Q * sizeof(struct shown));
        if (!A.c[k].ring || !A.c[k].q)
            return 2;
    }
    if (vt_link_open(&A.l))
        return 2;
    struct x11view v;
    int win_w, win_h;
    layout(&win_w, &win_h);
    if (x11view_open(&v, win_w, win_h, at_x, at_y, "VisionTrack: live on the Xeon Phi 7120A"))
        return 2;
    panel_x = panel.x;
    char view_build[16] = "?";
    struct stat st;
    if (!stat("/proc/self/exe", &st)) {
        struct tm tm;
        localtime_r(&st.st_mtime, &tm);
        strftime(view_build, sizeof view_build, "%H:%M:%S", &tm);
    }
    for (int k = 0; k < A.ncam; k++)
        pthread_create(&A.c[k].thread, NULL, capture_main, &A.c[k]);
    pthread_t lt, nt;
    pthread_create(&lt, NULL, link_main, NULL);
    pthread_create(&nt, NULL, nav_main, NULL);

    int dragging = 0, drag_cam = 0, dx0 = 0, dy0 = 0, dx1 = 0, dy1 = 0;
    uint64_t last_res[VT_MAX_CAMERAS] = {0, 0, 0}, last_cap[VT_MAX_CAMERAS] = {0, 0, 0};
    static double lat[128];
    int nlat = 0, ilat = 0;
    double lat_med = 0.0, lat_p95 = 0.0, place_t = 0.0, last_draw = 0.0, last_panel = 0.0, draw_ms = 0.0, present_ms = 0.0;
    double last_shown[VT_MAX_CAMERAS] = {0.0, 0.0, 0.0}, delay_ms[VT_MAX_CAMERAS] = {0.0, 0.0, 0.0};
    const uint8_t *show_pic[VT_MAX_CAMERAS] = {NULL, NULL, NULL};
    const uint8_t *show_cover = NULL;
    const struct vt_cell *show_cells[VT_CELL_CAMERAS] = {NULL, NULL};
    struct cells_plane show_cell_at[VT_CELL_CAMERAS] = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    FILE *rates_csv = NULL;
    double rates_t = 0.0;
    struct vec_plane show_lines = {0, 0, 0, 0, 0};
    uint64_t vec_count_was = 0;
    double vec_t_was = 0.0, vec_rate = 0.0;
    int show_made[VT_MAX_CAMERAS] = {0, 0, 0};
    FILE *shown_csv = NULL;
    unsigned shown_lines = 0;
    static struct vt_result r[VT_MAX_CAMERAS];
    while (!A.quit) {
        struct x11event e;
        while (x11view_poll(&v, &e) != X11_NONE) {
            if (e.type == X11_QUIT || (e.type == X11_KEY && (e.key == 'q' || e.key == 27))) {
                A.quit = 1;
            } else if (e.type == X11_PRESS && e.button == 1) {
                const int at = pane_at(e.x, e.y);
                if (at >= 0) {
                    dragging = 1;
                    drag_cam = at;
                    dx0 = dx1 = e.x;
                    dy0 = dy1 = e.y;
                }
            } else if (e.type == X11_MOTION && dragging) {
                dx1 = e.x;
                dy1 = e.y;
            } else if (e.type == X11_RELEASE && e.button == 1 && dragging) {
                dragging = 0;
                const int ox = pane[drag_cam].x, oy = pane[drag_cam].y, pw = pane[drag_cam].w;
                /* Back from the mirror the screen shows to the image's own pixels, the only coordinates the card knows. */
                const int ix0 = A.mirror ? pw - 1 - (dx0 - ox) : dx0 - ox, ix1 = A.mirror ? pw - 1 - (e.x - ox) : e.x - ox;
                const int xa = ix0 < ix1 ? ix0 : ix1, ya = (dy0 < e.y ? dy0 : e.y) - oy, xb = ix0 < ix1 ? ix1 : ix0, yb = (dy0 < e.y ? e.y : dy0) - oy;
                fprintf(stderr, "vt_view: drag camera %d %d,%d .. %d,%d\n", drag_cam, xa, ya, xb, yb);
                if (xb - xa >= 16 && yb - ya >= 16) {
                    pthread_mutex_lock(&A.m);
                    A.c[drag_cam].cmd = VT_CMD_SELECT;
                    /* In the camera's own pixels, whatever size its pane is. */
                    const double dsc = A.c[drag_cam].vs;
                    A.c[drag_cam].roi[0] = (float)(xa / dsc);
                    A.c[drag_cam].roi[1] = (float)(ya / dsc);
                    A.c[drag_cam].roi[2] = (float)((xb - xa) / dsc);
                    A.c[drag_cam].roi[3] = (float)((yb - ya) / dsc);
                    pthread_mutex_unlock(&A.m);
                }
            } else if ((e.type == X11_PRESS && e.button == 3) || (e.type == X11_KEY && e.key == 'd')) {
                pthread_mutex_lock(&A.m);
                if (e.type == X11_PRESS) {
                    const int at = pane_at(e.x, e.y);
                    if (at >= 0)
                        A.c[at].cmd = VT_CMD_DESELECT;
                } else {
                    for (int k = 0; k < A.ncam; k++)
                        A.c[k].cmd = VT_CMD_DESELECT;
                }
                pthread_mutex_unlock(&A.m);
            } else if (e.type == X11_KEY) {
                pthread_mutex_lock(&A.m);
                if (e.key == '[' || e.key == ']') {
                    const float sharp = A.fsr_sharp + (e.key == '[' ? 0.2f : -0.2f);
                    A.fsr_sharp = sharp < 0.0f ? 0.0f : sharp > 3.0f ? 3.0f : sharp;
                    A.fsr_sharp_new = 1;
                } else if (e.key == 'm')
                    A.mirror = !A.mirror;
                else if (e.key == 'f')
                    A.vectors_all = !A.vectors_all;
                else if (e.key == 'v')
                    A.view_level = A.view_level + 1 >= A.levels ? -1 : A.view_level + 1;
                else if (e.key == 'c') {
                    A.contrast = !A.contrast;
                    A.sc_key = VT_SC_CONTRAST;
                    A.sc_value = (float)A.contrast;
                } else if (e.key == 'g') {
                    A.pre_gauss = !A.pre_gauss;
                    A.sc_key = VT_SC_PRE_GAUSS;
                    A.sc_value = (float)A.pre_gauss;
                } else if (e.key == '[' || e.key == ']') {
                    A.feat += e.key == ']' ? 16 : -16;
                    A.feat = A.feat < 16 ? 16 : A.feat > VT_MAX_FEATURES ? VT_MAX_FEATURES : A.feat;
                    A.sc_key = VT_SC_MAX_FEATURES;
                    A.sc_value = (float)A.feat;
                }
                pthread_mutex_unlock(&A.m);
            }
        }
        apply_command_file();
        const double t = now_s();
        /* Once a second: where the user has put the window, for the next restart with new code. */
        if (t - place_t > 1.0) {
            place_t = t;
            int px, py;
            FILE *pf = !x11view_where(&v, &px, &py) ? fopen(POS_PATH, "w") : NULL;
            if (pf) {
                fprintf(pf, "%d,%d\n", px, py);
                fclose(pf);
            }
        }
        /* What is due. Each camera has its own clock, so each pane is drawn and shown when its own next picture is due, and nothing else moves. */
        const int panel_due = t - last_panel >= PANEL_S;
        uint64_t caps[VT_MAX_CAMERAS] = {0, 0, 0};
        unsigned dirty = 0, paced = 0;
        const uint64_t due_ns = vt_now_ns();
        pthread_mutex_lock(&A.m);
        const int link = A.link_state, tracking = link == LINK_TRACKING, cameras_due = t - last_draw >= DRAW_MIN_S;
        for (int k = 0; k < A.ncam; k++) {
            struct cam *K = &A.c[k];
            caps[k] = __atomic_load_n(&K->captured, __ATOMIC_ACQUIRE);
            last_res[k] = K->results;
            delay_ms[k] = (double)K->delay_ns * 1e-6;
            const struct shown *due = NULL;
            uint64_t taken = 0;
            while (K->q_out < K->q_in && K->q[K->q_out % SHOW_Q].at_ns <= due_ns) {
                due = &K->q[K->q_out % SHOW_Q];
                K->q_out++;
                taken++;
            }
            if (due) {
                memcpy(&r[k], &due->r, offsetof(struct vt_result, feat) + (size_t)due->r.nfeat * sizeof(struct vt_feat_out));
                show_pic[k] = due->pic;
                if (A.vec && k < VT_CELL_CAMERAS) {
                    /* This picture's own vectors, or the newest before them. */
                    int best = -1;
                    for (int i = 0; i < VT_CELL_SLOTS; i++)
                        if (A.cells_at[k][i].valid && A.cells_at[k][i].instance <= due->r.hdr.instance && (best < 0 || A.cells_at[k][i].instance > A.cells_at[k][best].instance))
                            best = i;
                    if (best >= 0) {
                        show_cell_at[k] = A.cells_at[k][best];
                        show_cells[k] = (const struct vt_cell *)(const void *)(A.l.part + vt_cells_offset((uint32_t)k, (uint32_t)best));
                    }
                }
                if (A.vec && k == A.fsr_cam) {
                    /* This picture's own lines, or the newest before them if they are late. */
                    int best = -1;
                    for (int i = 0; i < VT_VEC_SLOTS; i++)
                        if (A.vec_at[i].valid && A.vec_at[i].instance <= due->r.hdr.instance && (best < 0 || A.vec_at[i].instance > A.vec_at[best].instance))
                            best = i;
                    if (best >= 0) {
                        show_lines = A.vec_at[best];
                        show_cover = A.l.part + vt_vec_offset((uint32_t)best);
                    }
                }
                show_made[k] = due->generated;
                K->passed += taken - 1;
                K->shown++;
                K->shown_made += (uint64_t)due->generated;
                last_shown[k] = t;
                dirty |= 1u << k;
                paced |= 1u << k;
                if (k == 0 && !due->generated) {
                    lat[ilat] = (double)(due->recv_ns - due->r.hdr.acq_ns) * 1e-3;
                    ilat = (ilat + 1) % 128;
                    if (nlat < 128)
                        nlat++;
                }
            } else if (caps[k] && caps[k] != last_cap[k] && cameras_due && (!tracking || t - last_shown[k] > 0.5)) {
                /* Nothing from the card to go with them: the cameras as they are. */
                show_pic[k] = K->ring + ((caps[k] - 1) % RING) * (size_t)K->w * (size_t)K->h;
                last_cap[k] = caps[k];
                dirty |= 1u << k;
            } else if (!caps[k] && panel_due) {
                dirty |= 1u << k;
            }
        }
        if (!dirty && !panel_due) {
            pthread_mutex_unlock(&A.m);
            usleep(200);
            continue;
        }
        if (dirty & ~paced)
            last_draw = t;
        const uint64_t dropped = A.dropped, exc = A.exceptions, card_build = A.card_build, pairs = A.pairs;
        const int view_level = A.view_level, contrast = A.contrast, pre_gauss = A.pre_gauss, feat = A.feat;
        const struct vt_calib calib = A.calib;
        const int stale = calib_lost_pair(), map_lost = stale || A.c[2].nav.lost;
        struct nav_view nav[VT_MAX_CAMERAS];
        char calib_note[sizeof A.calib_note];
        for (int k = 0; k < A.ncam; k++)
            nav[k] = A.c[k].nav;
        memcpy(calib_note, A.calib_note, sizeof calib_note);
        const uint64_t map_seq = A.map_seq;
        const void *map = map_seq ? map_store[A.map_i] : NULL;
        const int save_map = A.save_map, vectors_all = A.vectors_all;
        A.save_map = 0;
        pthread_mutex_unlock(&A.m);
        if (save_map && map && !dv_save(map, &calib, "out/run/map"))
            fprintf(stderr, "vt_view: depth map %llu saved: out/run/map.{bin,pgm,ply}\n", (unsigned long long)map_seq);
        const uint64_t draw_t0 = vt_now_ns();
        if (panel_due) {
            last_panel = t;
            for (int k = 0; k < A.ncam; k++) {
                rate_update(&A.c[k].r_new, caps[k], t);
                rate_update(&A.c[k].r_raw, A.c[k].raw, t);
                rate_update(&A.c[k].r_res, last_res[k], t);
                rate_update(&A.c[k].r_shown, A.c[k].shown, t);
                rate_update(&A.c[k].r_made, A.c[k].shown_made, t);
            }
            if (nlat >= 16) {
                static double tmp[128];
                memcpy(tmp, lat, sizeof(double) * (size_t)nlat);
                qsort(tmp, (size_t)nlat, sizeof(double), cmp_d);
                lat_med = tmp[nlat / 2];
                lat_p95 = tmp[nlat * 95 / 100];
            }
        }
        for (int k = 0; k < A.ncam; k++) {
            if (!(dirty >> k & 1u))
                continue;
            const struct cam *K = &A.c[k];
            const int ox = pane[k].x, oy = pane[k].y;
            if (k == 0 && tracking && view_level >= 0 && r[0].debug_level == view_level && r[0].debug_w)
                show_plane(&v, K->w, K->h, view_level, &r[0]);
            else if (caps[k] && A.mirror)
                x11view_gray_mirror(&v, show_pic[k], K->vw, K->vh, K->vw, ox, oy);
            else if (caps[k])
                x11view_gray(&v, show_pic[k], K->vw, K->vh, K->vw, ox, oy, 1);
            else
                x11view_fill(&v, ox, oy, K->vw, K->vh, 0x101010u);
            /* Under the tracker's own marks, and out of the way altogether while a target is being picked. */
            const int baked = A.vec && k == A.fsr_cam && A.vec_count && !vectors_all;
            if (tracking && !dragging && baked && show_cover && show_lines.y1)
                x11view_tint(&v, show_cover, (int)show_lines.w, (int)show_lines.y0, (int)show_lines.y1, ox, oy, A.mirror, 0xff8040u);
            /* L and R's own, of this very picture, need no map. */
            const int own = A.vec && k < VT_CELL_CAMERAS && A.cells_count[k];
            if (tracking && !dragging && own && show_cells[k])
                dv_cells(&v, show_cells[k], show_cell_at[k].cols, show_cell_at[k].rows, (int)show_cell_at[k].level, ox, oy, K->vw, K->vs, A.mirror, vectors_all);
            if (tracking && map && !dragging && !baked && !own)
                dv_vectors(&v, map, k, ox, oy, K->vw, K->vs, A.mirror, vectors_all);
            draw_view(&v, ox, oy, K, &r[k], tracking);
            text_outlined(&v, ox + 8, oy + K->h - 22, 2, caps[k] ? WHITE : GREY, K->name);
            if (!caps[k])
                text_outlined(&v, ox + 8, oy + 8, 2, WHITE, K->state ? "no frames" : "waiting for the device");
        }
        if (tracking && (r[0].state == VT_ST_OBSERVED || r[0].state == VT_ST_NEW)) {
            if (A.ncam >= 2 && calib.valid && (dirty & 2u))
                draw_epipolar(&v, &calib, &r[0], calib.R, calib.T, calib.fb, calib.cxb, calib.cyb, b, pane[1].x, pane[1].y);
            if (A.ncam == 3 && calib.valid && calib.valid_d && (dirty & 4u))
                draw_epipolar(&v, &calib, &r[0], calib.Rd, calib.Td, calib.fd, calib.cxd, calib.cyd, d, pane[2].x, pane[2].y);
        }
        if (A.ncam == 3 && panel_due) {
            if (A.ncam == 3) {
                x11view_fill(&v, depth_pane.x, depth_pane.y, depth_pane.w, depth_pane.h, 0x0c0c0cu);
                const int live = tracking && map && calib.valid && calib.valid_d && !map_lost;
                if (live) {
                    const struct vt_depth_out *h = map;
                    dv_pane(&v, map, depth_pane.x, depth_pane.y, depth_pane.w, depth_pane.h, A.mirror);
                    dv_legend(&v, map, depth_pane.x + 10, depth_pane.y + depth_pane.h - 18, depth_pane.w - 20);
                    char t[96];
                    snprintf(t, sizeof t, "DEPTH MAP, FROM VIEW D: %u x %u, map %llu", h->w, h->h, (unsigned long long)h->seq);
                    text_outlined(&v, depth_pane.x + 10, depth_pane.y + 10, 2, WHITE, t);
                } else {
                    x11view_text(&v, depth_pane.x + 10, depth_pane.y + 10, 2, WHITE, "DEPTH MAP, FROM VIEW D");
                    x11view_text(&v, depth_pane.x + 10, depth_pane.y + 34, 2, GREY, !calib.valid || !calib.valid_d ? "UNAVAILABLE: no geometry holds camera D" : !tracking ? "UNAVAILABLE: card not tracking" : !map ? "waiting for the card's first map" : "UNAVAILABLE: A CAMERA CANNOT FIND ITSELF");
                }
            }
        }
        if (dragging && (dirty >> drag_cam & 1u)) {
            outline_line(&v, dx0, dy0, dx1, dy0, GREY, 1);
            outline_line(&v, dx1, dy0, dx1, dy1, GREY, 1);
            outline_line(&v, dx1, dy1, dx0, dy1, GREY, 1);
            outline_line(&v, dx0, dy1, dx0, dy0, GREY, 1);
        }
        if (dirty & 1u) {
            if (!caps[0])
                banner(&v, a->w, a->state ? "CAMERA A: no frames" : "CAMERA A: waiting for the device");
            else if (a->state == 2)
                banner(&v, a->w, "CAMERA A: frames stopped");
            else if (link == LINK_PAUSED)
                banner(&v, a->w, "CARD: a benchmark owns it: cameras only until it finishes");
            else if (link == LINK_NO_CARD)
                banner(&v, a->w, "CARD: waiting for vtrackd (restarting with new code?): cameras only");
            else if (view_level >= 0) {
                char bn[80];
                snprintf(bn, sizeof bn, "VIEW: card memory, pyramid level %d, what LK reads", view_level);
                banner(&v, a->w, bn);
            }
        }

        if (panel_due) {
            /* The panel: everything known, and only what is known. */
            x11view_fill(&v, panel.x, panel.y, panel.w, panel.h, 0x181818u);
            const int trail_size = 400;
            const int side_x = panel.x + panel.w - trail_size - 12;
            int y = panel.y + 8;
            panel_line(&v, &y, WHITE, "VISIONTRACK LIVE  7120A  %d VIEW%s", A.ncam, A.ncam > 1 ? "S" : "");
            panel_line(&v, &y, DIM, "code: card %02llu:%02llu:%02llu  view %s", (unsigned long long)card_build / 10000, (unsigned long long)card_build / 100 % 100, (unsigned long long)card_build % 100, view_build);
            panel_line(&v, &y, tracking ? WHITE : GREY, "CARD %s  bad messages %llu", tracking ? "TRACKING" : link == LINK_PAUSED ? "BENCHMARK" : link == LINK_NO_CARD ? "NOT ANSWERING" : "STARTING", (unsigned long long)exc);
            for (int k = 0; k < A.ncam; k++) {
                const struct cam *K = &A.c[k];
                panel_line(&v, &y, GREY, "%c %4.1f new, shown %4.1f, made %4.1f, repeats %2.0f%%, bad %llu", "LRD"[k], K->r_new.value, K->r_shown.value, K->r_made.value, K->raw ? 100.0 * (double)K->repeats / (double)K->raw : 0.0, (unsigned long long)K->bad);
            }
            panel_line(&v, &y, GREY, "L capture to result ms: median %.2f p95 %.2f", lat_med * 1e-3, lat_p95 * 1e-3);
            panel_line(&v, &y, GREY, "capture to screen ms: L %.0f R %.0f D %.0f  draw %.1f present %.1f", delay_ms[0], delay_ms[1], delay_ms[2], draw_ms, present_ms);
            panel_line(&v, &y, GREY, "frames dropped at the host %llu", (unsigned long long)dropped);
            {
                char rig[96];
                int at = snprintf(rig, sizeof rig, "attitude:");
                for (int k = 0; k < A.ncam && at < (int)sizeof rig; k++) {
                    const float shown = nav[k].state == VT_ATT_TURNED ? nav[k].corr : nav[k].corr_none;
                    at += snprintf(rig + at, sizeof rig - (size_t)at, " %c %s %.2f", "LRD"[k], nav[k].lost ? "LOST" : nav[k].state == VT_ATT_TURNED ? "turned" : nav[k].state == VT_ATT_HELD ? "still" : "-", (double)shown);
                }
                panel_line(&v, &y, GREY, "%s", rig);
            }
            {
                char line[96], line2[96];
                int at = snprintf(line, sizeof line, "lights, a check:"), at2 = snprintf(line2, sizeof line2, "carried deg:");
                for (int k = 0; k < A.ncam; k++) {
                    at += snprintf(line + at, sizeof line - (size_t)at, " %c %d/%d", "LRD"[k], nav[k].used, nav[k].known);
                    at2 += snprintf(line2 + at2, sizeof line2 - (size_t)at2, " %c%+.2f%+.2f%+.2f", "LRD"[k], nav[k].turned[0] * 180.0 / M_PI, nav[k].turned[1] * 180.0 / M_PI, nav[k].turned[2] * 180.0 / M_PI);
                }
                panel_line(&v, &y, GREY, "%s", line);
                panel_line(&v, &y, GREY, "%s", line2);
            }
    #define US(k) ((double)r[0].cycles[k] * 1e6 / TSC_HZ)
            panel_line(&v, &y, GREY, "card us L: dma %.0f pyr %.0f flow %.0f region %.0f", US(VT_T_DMA_IN), US(VT_T_PYR), US(VT_T_LK), US(VT_T_REGION));
            y += 6;
            for (int k = 0; k < A.ncam; k++) {
                const struct vt_result *q = &r[k];
                const int on = tracking && (q->state == VT_ST_NEW || q->state == VT_ST_OBSERVED || q->state == VT_ST_UNOBSERVED);
                panel_line(&v, &y, WHITE, "%c: %s", "LRD"[k], tracking ? state_name(q->state) : "-");
                if (on) {
                    panel_line(&v, &y, GREY, " at %7.1f %6.1f  size %5.0f %4.0f  %3.0f %%", (double)q->cx, (double)q->cy, (double)q->w, (double)q->h, (double)q->inliers * 100.0);
                    if (q->flags & VT_F_PREDICTED)
                        panel_line(&v, &y, WHITE, " PREDICTED, missing %u, best match %.2f", q->missing, (double)q->reacq_ncc);
                    else
                        panel_line(&v, &y, GREY, " %+6.1f %+6.1f px/s  features %u/%d", (double)q->rate_u, (double)q->rate_v, q->nfeat, feat);
                } else if (tracking && (q->flags & VT_F_NO_FEATURES)) {
                    panel_line(&v, &y, WHITE, " too little texture there");
                } else if (tracking && q->state == VT_ST_LOST) {
                    panel_line(&v, &y, WHITE, " searching the whole picture, best match %.2f", (double)q->reacq_ncc);
                }
            }
            y += 6;
            panel_line(&v, &y, WHITE, "3D POSITION, VELOCITY");
            const struct vt_result *m3 = (r[1].flags & VT_F_METRIC) && r[1].hdr.acq_ns >= r[0].hdr.acq_ns ? &r[1] : &r[0];
            if (A.ncam < 2) {
                panel_line(&v, &y, GREY, "UNAVAILABLE: one camera");
            } else if (!calib.valid) {
                panel_line(&v, &y, GREY, "UNAVAILABLE: no calibration yet");
                panel_line(&v, &y, DIM, "correspondences so far: %llu", (unsigned long long)pairs);
            } else if (stale) {
                /* No number at all: a position computed through a pose the rig has left only looks like a measurement. */
                panel_line(&v, &y, WHITE, "UNAVAILABLE: L OR R CANNOT FIND ITSELF");
                panel_line(&v, &y, GREY, "its picture no longer matches its reference");
            } else if (tracking && (m3->flags & VT_F_METRIC)) {
                const int measured = m3->age_s < 0.05;
                const int unproven = (m3->flags & VT_F_UNPROVEN) != 0;
                panel_line(&v, &y, unproven ? GREY : measured ? WHITE : GREY, "%s, age %.0f ms", unproven ? "NOT PROVEN: BOXES AT REST, NOT THE THING" : measured ? "MEASURED" : "PREDICTED, NOT MEASURED", m3->age_s * 1000.0);
                panel_line(&v, &y, WHITE, "X %+6.3f  Y %+6.3f  Z %+6.3f m", m3->pos[0], m3->pos[1], m3->pos[2]);
                panel_line(&v, &y, GREY, "+- %5.3f    %5.3f    %5.3f", sqrt(m3->pos_cov[0]), sqrt(m3->pos_cov[4]), sqrt(m3->pos_cov[8]));
                panel_line(&v, &y, WHITE, "V %+6.3f    %+6.3f    %+6.3f m/s", m3->vel[0], m3->vel[1], m3->vel[2]);
                panel_line(&v, &y, GREY, "+- %5.3f    %5.3f    %5.3f", sqrt(m3->vel_var[0]), sqrt(m3->vel_var[1]), sqrt(m3->vel_var[2]));
                panel_line(&v, &y, GREY, "rays %.1f deg apart, miss %.1f px, gap %.1f ms", r[1].parallax_deg, r[1].reproj_rms, r[1].pair_gap_ms);
                panel_line(&v, &y, DIM, "%.72s", calib_note);
            } else {
                panel_line(&v, &y, GREY, "UNAVAILABLE: select the region in L and in R");
            }
            panel_line(&v, &y, DIM, "correspondences %llu  contrast %s  blur %s", (unsigned long long)pairs, contrast ? "on" : "off", pre_gauss ? "on" : "off");
            if (A.fsr)
                panel_line(&v, &y, DIM, "D: FSR 720p%.4g > flow 1440p%.4g > Lanczos 1080p  RCAS %.1f stops  [ ] change it", A.c[2].rate, 2.0 * A.c[2].rate, (double)A.fsr_sharp);
            if (A.ncam == 3 && map) {
                const struct vt_depth_out *h = map;
                const double us = 1e-3 / TSC_HZ * 1e6;
                rate_update(&A.map_rate, map_seq, t);
                y += 6;
                panel_line(&v, &y, WHITE, "DEPTH %.1f maps/s  confident %.2f %% of %u textured", A.map_rate.value, h->textured ? 100.0 * (double)h->confident / (double)h->textured : 0.0, h->textured);
                panel_line(&v, &y, GREY, " %.1f %% of pixels have a depth", 100.0 * (double)h->known / ((double)h->w * (double)h->h));
                panel_line(&v, &y, GREY, " card ms: normalize %.1f sweep %.1f agg %.1f decide %.1f", (double)h->cycles_normalize * us, (double)h->cycles_sweep * us, (double)h->cycles_aggregate * us, (double)h->cycles_decide * us);
                panel_line(&v, &y, GREY, " flow at level %u: L %u x %u cells", h->flow_level, h->cam[0].cols, h->cam[0].rows);
                panel_line(&v, &y, GREY, " vectors ms L %.1f R %.1f D %.1f  moving %u %u %u", (double)h->cam[0].cycles_vectors * us, (double)h->cam[1].cycles_vectors * us, (double)h->cam[2].cycles_vectors * us, h->cam[0].moving, h->cam[1].moving, h->cam[2].moving);
                if (A.vec && A.cells_count[0]) {
                    rate_update(&A.cells_rate[0], A.cells_count[0], t);
                    rate_update(&A.cells_rate[1], A.cells_count[1], t);
                    panel_line(&v, &y, GREY, " orange: L %.1f R %.1f/s, on FSR's threads between its frames", A.cells_rate[0].value, A.cells_rate[1].value);
                }
                /* A line a second, for what happens when things move: maps, planes of D, frames of L and R, cells moving. */
                if (t - rates_t >= 1.0) {
                    rates_t = t;
                    if (!rates_csv) {
                        char path[80];
                        const time_t wall = time(NULL);
                        struct tm tm;
                        localtime_r(&wall, &tm);
                        strftime(path, sizeof path, "logs/live/rates-%Y%m%d-%H%M%S.csv", &tm);
                        rates_csv = fopen(path, "w");
                        if (rates_csv)
                            fprintf(rates_csv, "t,maps,orange_d,orange_l,orange_r,moving_l,moving_r,moving_d,capture_to_result_us\n");
                    }
                    if (rates_csv) {
                        fprintf(rates_csv, "%.3f,%llu,%llu,%llu,%llu,%u,%u,%u,%.0f\n", t, (unsigned long long)map_seq, (unsigned long long)A.vec_count, (unsigned long long)A.cells_count[0], (unsigned long long)A.cells_count[1], h->cam[0].moving, h->cam[1].moving, h->cam[2].moving, lat_med);
                        fflush(rates_csv);
                    }
                }
                if (A.vec && A.vec_count) {
                    if (t - vec_t_was >= 1.0) {
                        vec_rate = vec_t_was > 0.0 ? (double)(A.vec_count - vec_count_was) / (t - vec_t_was) : 0.0;
                        vec_count_was = A.vec_count;
                        vec_t_was = t;
                    }
                    panel_line(&v, &y, GREY, " orange: D on the spare core, %u threads, %.1f/s: vectors %.1f lines %.2f down %.1f ms, %u lines", A.vec_last.threads, vec_rate, (double)A.vec_last.cycles_vectors * us, (double)A.vec_last.cycles_draw * us, (double)A.vec_last.cycles_down * us, A.vec_last.lines);
                }
                panel_line(&v, &y, GREY, " frames in the means L %.0f R %.0f D %.0f", (double)h->cam[0].count_mean, (double)h->cam[1].count_mean, (double)h->cam[2].count_mean);
                panel_line(&v, &y, GREY, " drift, camera px: L %+.2f%+.2f R %+.2f%+.2f D %+.2f%+.2f", 4.0 * (double)h->cam[0].drift_u, 4.0 * (double)h->cam[0].drift_v, 4.0 * (double)h->cam[1].drift_u, 4.0 * (double)h->cam[1].drift_v, 4.0 * (double)h->cam[2].drift_u, 4.0 * (double)h->cam[2].drift_v);
            }
            if (A.ncam >= 2)
                draw_trail(&v, side_x, panel.y + 10, trail_size, &calib);
            static const char *const help[] = {"drag: select, in each view", "right click, d: deselect", "v: view   c: contrast", "g: blur   f: all vectors", "m: mirror   [ ]: features   q: quit"};
            for (int i = 0; i < 5; i++)
                x11view_text(&v, side_x, panel.y + trail_size + 24 + 20 * i, 2, DIM, help[i]);
        }
        const uint64_t draw_t1 = vt_now_ns();
        for (int k = 0; k < A.ncam; k++)
            if (dirty >> k & 1u)
                x11view_present_rect(&v, pane[k].x, pane[k].y, pane[k].w, pane[k].h);
        if (panel_due) {
            x11view_present_rect(&v, panel.x, panel.y, panel.w, panel.h);
            if (A.ncam == 3)
                x11view_present_rect(&v, depth_pane.x, depth_pane.y, depth_pane.w, depth_pane.h);
        }
        x11view_sync(&v);
        const uint64_t draw_t2 = vt_now_ns();
        if (dirty) {
            draw_ms += 0.05 * ((double)(draw_t1 - draw_t0) * 1e-6 - draw_ms);
            present_ms += 0.05 * ((double)(draw_t2 - draw_t1) * 1e-6 - present_ms);
        }
        /* Every picture, as it went to the screen: what any claim about the rate on the screen is measured from. */
        if (paced && !shown_csv) {
            char path[80];
            const time_t wall = time(NULL);
            struct tm tm;
            localtime_r(&wall, &tm);
            strftime(path, sizeof path, "logs/live/shown-%Y%m%d-%H%M%S.csv", &tm);
            shown_csv = fopen(path, "w");
            if (shown_csv)
                fprintf(shown_csv, "shown_s,camera,generated,acq_s,instance,draw_ms,present_ms\n");
        }
        if (shown_csv)
            for (int k = 0; k < A.ncam; k++)
                if (paced >> k & 1u) {
                    fprintf(shown_csv, "%.6f,%d,%d,%.6f,%llu,%.3f,%.3f\n", (double)draw_t2 * 1e-9, k, show_made[k], (double)r[k].hdr.acq_ns * 1e-9, (unsigned long long)r[k].hdr.instance, (double)(draw_t1 - draw_t0) * 1e-6, (double)(draw_t2 - draw_t1) * 1e-6);
                    if ((++shown_lines & 255u) == 0u)
                        fflush(shown_csv);
                }
    }
    pthread_join(lt, NULL);
    pthread_join(nt, NULL);
    for (int k = 0; k < A.ncam; k++)
        pthread_join(A.c[k].thread, NULL);
    x11view_close(&v);
    vt_link_close(&A.l);
    return 0;
}
