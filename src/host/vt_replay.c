/* vt_replay.c: the same scene and the same message stream through the same tracker source on the desktop host: the lane model, one thread. Prints the hash vt_run --hash wrote from the card's
 * results; with --json or --csv, scores it too (no card needed, so parameter fits run here). VT_TRACE=1 prints a line per frame on stderr. Options: run_opts.c. */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fuse.h"
#include "run_opts.h"
#include "score.h"
#include "track.h"
#include "vt_util.h"

static void *host_alloc(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, bytes);
}

static void host_run(void *ctx, tk_fn fn, void *arg)
{
    (void)ctx;
    fn(arg, 0, 1);
}

/* How close a generated picture is to the true picture of its instant, against what needs no motion at all: the older frame shown again, and the two frames averaged. Over the thing with a margin, and over the whole picture, where a still background flatters everything. */
struct fg_sse {
    double gen, repeat, mean;
    long n;
};

static void fg_add(struct fg_sse *q, const uint8_t *gen, const uint8_t *truth, const uint8_t *older, const uint8_t *newer, int w, int x0, int y0, int x1, int y1)
{
    for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
            const size_t i = (size_t)y * (size_t)w + (size_t)x;
            const double t = truth[i], g = (double)gen[i] - t, r = (double)older[i] - t, m = 0.5 * ((double)older[i] + (double)newer[i]) - t;
            q->gen += g * g;
            q->repeat += r * r;
            q->mean += m * m;
            q->n++;
        }
}

static double psnr(double sse, long n) { return sse > 0.0 && n ? 10.0 * log10(255.0 * 255.0 * (double)n / sse) : 99.0; }

int main(int argc, char **argv)
{
    struct run_opts o;
    struct dataset ds;
    if (run_opts_parse(&o, argc, argv, &ds))
        return 2;
    const int cams = o.dual ? 2 : 1;
    struct tracker *t[2] = {NULL, NULL};
    for (int c = 0; c < cams; c++) {
        t[c] = tk_create(&o.cfg, 1, 1, host_alloc, NULL, host_run, NULL, host_run, NULL);
        if (!t[c])
            return 2;
    }
    uint8_t *pics[2] = {NULL, NULL};
    size_t pic_bytes[2] = {(size_t)o.cfg.width * o.cfg.height, (size_t)o.cfg.width2 * o.cfg.height2};
    if (o.cfg.gen_pictures)
        for (int c = 0; c < cams; c++) {
            pics[c] = host_alloc(NULL, pic_bytes[c]);
            if (!pics[c])
                return 2;
            tk_set_picture(t[c], pics[c]);
        }
    const int fgcheck = getenv("VT_FGCHECK") != NULL && pics[0];
    uint8_t *fg_truth = fgcheck ? malloc(pic_bytes[0]) : NULL;
    struct fg_sse fg_thing = {0.0, 0.0, 0.0, 0}, fg_whole = {0.0, 0.0, 0.0, 0};
    static struct fuse fu;
    fuse_init(&fu, &o.cfg);
    if (o.dual) {
        struct vt_calib cal;
        dataset_true_calib(&ds, &cal);
        fuse_set_calib(&fu, &cal);
    }
    struct score *s = score_new(&ds, o.csv);
    static struct vt_result r;
    const int trace = getenv("VT_TRACE") != NULL, nsets = tk_sets(t[0]);
    for (int i = 0; i < o.nframes; i++) {
        for (int c = 0; c < cams; c++) {
            struct vt_msg m;
            run_opts_msg(&o, &ds, i, c, &m);
            const int set = i % nsets;
            const uint8_t *left = c == 0 ? dataset_left(&ds, i) : dataset_right(&ds, i);
            tk_build(t[c], set, left, (o.mono || o.dual) ? NULL : dataset_right(&ds, i), &m);
            struct vt_msg gm;
            const int g = tk_generate(t[c], set, &m, &gm);
            if (g >= 0) {
                tk_track(t[c], g, &gm, &r);
                r.seq = 0;
                r.hdr.pub_ns = 0;
                r.hdr.slot = 0;
                r.frames_skipped = 0;
                memset(r.cycles, 0, sizeof r.cycles);
                const size_t gbytes = offsetof(struct vt_result, feat) + (size_t)r.nfeat * sizeof(struct vt_feat_out);
                printf("%d %016llx\n", i, (unsigned long long)(vt_fnv(VT_FNV_INIT, &r, gbytes) ^ (pics[c] ? vt_fnv(VT_FNV_INIT, pics[c], pic_bytes[c]) : 0ull)));
                if (fgcheck && c == 0 && i > 0) {
                    const double mid = ((double)gm.acq_ns - 1e9) * 1e-9;
                    struct truth tr;
                    scene_truth_at(&ds.s, mid, &tr);
                    scene_render_at(&ds.s, mid, i, 0, fg_truth);
                    const int w = ds.s.w, h = ds.s.h, x0 = (int)(tr.u - tr.half_w) - 8, x1 = (int)(tr.u + tr.half_w) + 8, y0 = (int)(tr.v - tr.half_h) - 8, y1 = (int)(tr.v + tr.half_h) + 8;
                    fg_add(&fg_whole, pics[0], fg_truth, dataset_left(&ds, i - 1), dataset_left(&ds, i), w, 0, 0, w, h);
                    fg_add(&fg_thing, pics[0], fg_truth, dataset_left(&ds, i - 1), dataset_left(&ds, i), w, x0 < 0 ? 0 : x0, y0 < 0 ? 0 : y0, x1 > w ? w : x1, y1 > h ? h : y1);
                }
                score_frame(s, &r, 0.0, 0.0);
            }
            tk_track(t[c], set, &m, &r);
            if (o.dual)
                fuse_frame(&fu, c, &r);
            if (trace) {
                struct truth tr;
                scene_truth_at(&ds.s, ((double)r.hdr.acq_ns - 1e9) * 1e-9, &tr);
                fprintf(stderr, "f %3d c%d st %u fl %3u miss %2u nf %3u box %5.0f %5.0f c %8.2f %7.2f truth %8.2f %7.2f vis %.2f ncc %6.3f nis %7.3f z %7.4f pos %7.4f %7.4f %7.4f tr %7.4f %7.4f %7.4f rp %.3f par %.1f\n", i, c, r.state, r.flags, r.missing, r.nfeat, (double)r.w, (double)r.h, (double)r.cx, (double)r.cy, c ? tr.ub : tr.u, c ? tr.vb : tr.v, tr.visible, (double)r.reacq_ncc, r.nis, r.meas[2], r.pos[0], r.pos[1], r.pos[2], tr.X, tr.Y, tr.Z, r.reproj_rms, r.parallax_deg);
            }
            r.seq = 0;
            r.hdr.pub_ns = 0;
            r.hdr.slot = 0;
            r.frames_skipped = 0;
            memset(r.cycles, 0, sizeof r.cycles);
            const size_t bytes = offsetof(struct vt_result, feat) + (size_t)r.nfeat * sizeof(struct vt_feat_out);
            printf("%d %016llx\n", i, (unsigned long long)vt_fnv(VT_FNV_INIT, &r, bytes));
            score_frame(s, &r, 0.0, 0.0);
        }
    }
    if (o.json) {
        FILE *j = fopen(o.json, "w");
        if (j) {
            score_print(s, j, o.dual ? "dual" : o.mono ? "A" : "B", o.nframes * cams, 0, 0.0, "\"replay\":1,");
            fclose(j);
        }
    }
    if (fgcheck)
        fprintf(stderr, "fgcheck %s: over the thing, dB: generated %.2f, older frame again %.2f, the two averaged %.2f; whole picture: generated %.2f, older frame again %.2f, the two averaged %.2f\n", ds.s.name, psnr(fg_thing.gen, fg_thing.n), psnr(fg_thing.repeat, fg_thing.n), psnr(fg_thing.mean, fg_thing.n), psnr(fg_whole.gen, fg_whole.n), psnr(fg_whole.repeat, fg_whole.n), psnr(fg_whole.mean, fg_whole.n));
    /* --gate here as well: scenes can be judged with no card at all. */
    int bad = 0;
    if (o.gate) {
        FILE *nul = fopen("/dev/null", "w");
        if (nul) {
            score_print(s, nul, "replay", o.nframes * cams, 0, 0.0, "\"replay\":1,");
            fclose(nul);
        }
        bad = score_gate(s, stderr);
    }
    dataset_close(&ds);
    return bad ? 1 : 0;
}
