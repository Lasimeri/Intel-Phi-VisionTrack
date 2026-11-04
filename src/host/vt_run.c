/* vt_run.c: one scene through the card, on the desktop host's side: publishes message objects, collects results, scores them against the scene's analytic truth.
 * Options: run_opts.c. --json FILE writes the summary there too. --gate exits 1 when the scene's documented tolerances are missed (tests/scenes.md). */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run_opts.h"
#include "score.h"
#include "vt_link.h"
#include "vt_util.h"

#define TSC_HZ 1238089000.0 /* the card's, measured by vt_m0: for card-side stage times only */

struct recv {
    struct vt_link *l;
    struct vt_result *res;
    uint64_t *t_recv;
    uint64_t *pic_hash; /* of a generated result's picture, as it lands: the slot is written again within a few frames */
    const struct vt_config *cfg;
    int max;
    int count;
    int stop;
};

static void *recv_main(void *arg)
{
    struct recv *r = arg;
    struct vt_ctl *c = r->l->ctl;
    uint64_t have = 0;
    const uint8_t *ring = r->l->part + VT_RES_OFFSET;
    while (have < (uint64_t)r->max) {
        const uint64_t pub = c->results_pub;
        if (pub <= have) {
            if (__atomic_load_n(&r->stop, __ATOMIC_ACQUIRE))
                break;
            _mm_pause();
            continue;
        }
        const uint64_t t = vt_now_ns();
        if (pub - have > VT_RES_SLOTS - 1)
            fprintf(stderr, "host: result ring overrun: %llu results behind\n", (unsigned long long)(pub - have));
        const size_t head = offsetof(struct vt_result, feat);
        for (uint64_t i = have; i < pub; i++) {
            const uint8_t *src = ring + i % VT_RES_SLOTS * VT_RES_BYTES;
            struct vt_result *o = &r->res[i];
            memcpy(o, src, head);
            memcpy(o->feat, src + head, (size_t)o->nfeat * sizeof(struct vt_feat_out));
            r->t_recv[i] = t;
            if (o->fsr_slot)
                r->pic_hash[i] ^= vt_fnv(VT_FNV_INIT, r->l->part + vt_fsr_offset(o->fsr_slot - 1), (size_t)o->fsr_w * o->fsr_h);
            if (o->picture) {
                const size_t bytes = o->hdr.camera == 0 ? (size_t)r->cfg->width * r->cfg->height : (size_t)r->cfg->width2 * r->cfg->height2;
                r->pic_hash[i] ^= vt_fnv(VT_FNV_INIT, r->l->part + vt_gen_offset(o->hdr.camera, o->picture - 1), bytes);
            }
        }
        have = pub;
        __atomic_store_n(&r->count, (int)have, __ATOMIC_RELEASE);
    }
    return NULL;
}

/* A deliberately malformed message, for the exception path: exactly one thing wrong with it. */
static void publish_bad(struct vt_ctl *c, const struct run_opts *o, const struct dataset *ds, uint64_t *pub)
{
    struct vt_msg m;
    run_opts_msg(o, ds, 0, 0, &m);
    switch (o->bad_kind) {
    case VT_EXC_BAD_TYPE: m.type = 99; break;
    case VT_EXC_BAD_SLOT: m.slot = 9999; break;
    case VT_EXC_BAD_LENGTH: m.length = 64; break;
    case VT_EXC_INSTANCE_ORDER: m.instance = 0; m.slot = 0; break;
    default: break;
    }
    m.check = vt_msg_check(&m);
    if (o->bad_kind == VT_EXC_BAD_CHECK)
        m.check ^= 1;
    c->desc[*pub % VT_DESC_RING] = m;
    _mm_sfence();
    c->frames_pub = ++*pub;
}

int main(int argc, char **argv)
{
    struct run_opts o;
    struct dataset ds;
    if (run_opts_parse(&o, argc, argv, &ds))
        return 2;
    const int w = o.w, h = o.h, nframes = o.nframes;
    struct vt_link l;
    if (vt_link_open(&l))
        return 2;
    if (vt_session_start(&l, &o.cfg, 30.0))
        return 3;
    const int cams = o.dual ? 2 : 1, nmsg = nframes * cams;
    const int nresults = o.cfg.framegen ? 2 * nmsg : nmsg; /* a generated frame's result as well as each real one's */
    struct vt_result *res = malloc(sizeof(struct vt_result) * (size_t)nresults);
    uint64_t *t_recv = calloc((size_t)nresults, 8), *t_pub = calloc((size_t)nmsg, 8), *pic_hash = calloc((size_t)nresults, 8);
    if (!res || !t_recv || !t_pub || !pic_hash)
        return 2;
    struct recv rcv = {.l = &l, .res = res, .t_recv = t_recv, .pic_hash = pic_hash, .cfg = &o.cfg, .max = nresults};
    pthread_t rt;
    pthread_create(&rt, NULL, recv_main, &rcv);

    struct vt_ctl *c = l.ctl;
    if (o.dual) {
        dataset_true_calib(&ds, &c->calib);
        _mm_sfence();
        c->calib_seq = c->calib_seq + 1;
    }
    const size_t slot_bytes = vt_slot_stride(&o.cfg);
    uint8_t *frames = l.part + VT_FRAME_OFFSET;
    const uint64_t t_start = vt_now_ns();
    uint64_t pub = 0, bad_published = 0;
    int ndrop = 0, nsent = 0;
    for (int i = 0; i < nframes; i++) {
        if (o.paced) {
            const uint64_t due = t_start + (uint64_t)((double)i * 1e9 / o.fps);
            while (vt_now_ns() < due)
                _mm_pause();
        }
        if (i == o.bad_at) {
            publish_bad(c, &o, &ds, &pub);
            bad_published++;
        }
        for (int cam = 0; cam < cams; cam++) {
            if (pub - c->frames_taken >= o.cfg.slots) {
                if (o.drop) {
                    ndrop++;
                    continue;
                }
                while (pub - c->frames_taken >= o.cfg.slots)
                    _mm_pause();
            }
            const uint32_t slot = (uint32_t)(pub % o.cfg.slots);
            uint8_t *dst = frames + (size_t)slot * slot_bytes;
            t_pub[i * cams + cam] = vt_now_ns();
            memcpy(dst, cam == 0 ? dataset_left(&ds, i) : dataset_right(&ds, i), (size_t)w * (size_t)h);
            if (!o.mono && !o.dual)
                memcpy(dst + (size_t)w * (size_t)h, dataset_right(&ds, i), (size_t)w * (size_t)h);
            struct vt_msg m;
            run_opts_msg(&o, &ds, i, cam, &m);
            m.pub_ns = t_pub[i * cams + cam];
            m.slot = slot;
            m.check = vt_msg_check(&m);
            c->desc[pub % VT_DESC_RING] = m;
            _mm_sfence();
            c->frames_pub = ++pub;
            nsent++;
        }
    }
    /* Every good frame comes back, unless the card skipped it on purpose. */
    int waited = 0;
    for (;;) {
        const int n = __atomic_load_n(&rcv.count, __ATOMIC_ACQUIRE);
        uint64_t skipped = 0, real = 0;
        for (int i = 0; i < n; i++) {
            skipped += res[i].frames_skipped;
            real += !(res[i].flags & VT_F_GENERATED);
        }
        if (real + skipped >= (uint64_t)nsent)
            break;
        if (waited++ > 10000) {
            fprintf(stderr, "host: only %d results of %d frames after 10 s (exceptions %llu, last %llu)\n", n, nsent, (unsigned long long)c->exceptions, (unsigned long long)c->last_exception);
            break;
        }
        struct timespec nap = {0, 1000000};
        nanosleep(&nap, NULL);
    }
    const uint64_t t_end = vt_now_ns();
    __atomic_store_n(&rcv.stop, 1, __ATOMIC_RELEASE);
    pthread_join(rt, NULL);
    const int nres = rcv.count;
    const uint64_t exceptions = c->exceptions, last_exception = c->last_exception;
    vt_session_end(&l);
    vt_session_wait_end(&l, 10.0);

    struct score *s = score_new(&ds, o.csv);
    FILE *hf = o.hash ? fopen(o.hash, "w") : NULL;
    int alg_n[2] = {0, 0};
    double alg_us[2] = {0.0, 0.0};
    for (int r = 0; r < nres; r++) {
        struct vt_result *x = &res[r];
        const int fi = (int)x->hdr.instance;
        if (hf) {
            /* The hash covers everything computed from the frames, nothing timed. */
            static struct vt_result copy;
            const size_t bytes = offsetof(struct vt_result, feat) + (size_t)x->nfeat * sizeof(struct vt_feat_out);
            memcpy(&copy, x, bytes);
            copy.seq = 0;
            copy.hdr.pub_ns = 0;
            copy.hdr.slot = 0;
            copy.frames_skipped = 0;
            copy.picture = 0;
            copy.fsr_slot = 0;
            memset(copy.cycles, 0, sizeof copy.cycles);
            fprintf(hf, (x->flags & VT_F_GENERATED) ? "%dg %016llx\n" : "%d %016llx\n", fi, (unsigned long long)(vt_fnv(VT_FNV_INIT, &copy, bytes) ^ pic_hash[r]));
        }
        const int key = fi * cams + (int)x->hdr.camera;
        const double lat_us = key < nmsg && t_pub[key] ? (double)(t_recv[r] - t_pub[key]) * 1e-3 : 0.0;
        score_frame(s, x, lat_us, 1e6 / TSC_HZ);
        const int a = x->hdr.algorithm == VT_ALG_X87;
        alg_n[a]++;
        alg_us[a] += (double)(x->cycles[VT_T_PYR] + x->cycles[VT_T_TRACK]) * 1e6 / TSC_HZ;
    }
    if (hf)
        fclose(hf);
    char extra[420];
    snprintf(extra, sizeof extra, "\"cores\":%u,\"tpc\":%u,\"pipelines\":%u,\"pyr_cores\":%u,\"paced\":%d,\"exceptions\":%llu,\"last_exception\":%llu,\"bad_published\":%llu,\"mean_compute_us\":{\"mvex\":%.1f,\"n_mvex\":%d,\"x87\":%.1f,\"n_x87\":%d},", o.cfg.cores, o.cfg.tpc, o.cfg.pipelines, o.cfg.pyr_cores, o.paced, (unsigned long long)exceptions, (unsigned long long)last_exception, (unsigned long long)bad_published, alg_n[0] ? alg_us[0] / alg_n[0] : 0.0, alg_n[0], alg_n[1] ? alg_us[1] / alg_n[1] : 0.0, alg_n[1]);
    const double wall = (double)(t_end - t_start) * 1e-9;
    const char *mode = o.dual ? "dual" : o.mono ? "A" : "B";
    score_print(s, stdout, mode, nres, ndrop, wall, extra);
    if (o.json) {
        FILE *j = fopen(o.json, "w");
        if (j) {
            score_print(s, j, mode, nres, ndrop, wall, extra);
            fclose(j);
        }
    }
    int bad = nres ? 0 : 1;
    if (o.gate)
        bad += score_gate(s, stdout);
    vt_link_close(&l);
    dataset_close(&ds);
    return bad ? 1 : 0;
}
