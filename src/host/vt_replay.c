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

/* FSR against the truth: the scene rendered clean at the upscaled size, against what needs no history at all, a Lanczos 2 upscale of the same noisy frame. From the 30th frame on, once history has built up. */
struct fsr_check {
    struct scene hi;
    uint8_t *truth;
    float *lz, *tmp;
    double sse_fsr, sse_lz, sse_fsr_thing, sse_lz_thing, sse_fsr_rest, sse_lz_rest;
    long n, n_thing, n_rest;
};

static double l2(double x)
{
    x = fabs(x);
    if (x >= 2.0)
        return 0.0;
    if (x < 1e-6)
        return 1.0;
    return sin(M_PI * x) / (M_PI * x) * (sin(0.5 * M_PI * x) / (0.5 * M_PI * x));
}

/* Lanczos 2, one axis at a time. */
static void lanczos_up(const uint8_t *in, int w, int h, float *out, float *tmp, int uw, int uh)
{
    for (int y = 0; y < h; y++)
        for (int x = 0; x < uw; x++) {
            const double p = (x + 0.5) * w / uw - 0.5, fl = floor(p);
            double s = 0.0, t = 0.0;
            for (int k = -1; k <= 2; k++) {
                const int sx = (int)fl + k, cx = sx < 0 ? 0 : sx >= w ? w - 1 : sx;
                const double wt = l2(p - sx);
                s += wt * in[y * w + cx];
                t += wt;
            }
            tmp[y * uw + x] = (float)(s / t);
        }
    for (int y = 0; y < uh; y++) {
        const double p = (y + 0.5) * h / uh - 0.5, fl = floor(p);
        for (int x = 0; x < uw; x++) {
            double s = 0.0, t = 0.0;
            for (int k = -1; k <= 2; k++) {
                const int sy = (int)fl + k, cy = sy < 0 ? 0 : sy >= h ? h - 1 : sy;
                const double wt = l2(p - sy);
                s += wt * tmp[cy * uw + x];
                t += wt;
            }
            const double v = s / t;
            out[y * uw + x] = (float)(v < 0.0 ? 0.0 : v > 255.0 ? 255.0 : v);
        }
    }
}

static void fsr_check_frame(struct fsr_check *q, const struct dataset *ds, int i, double at_frame, const uint8_t *pic, int uw, int uh)
{
    const double t = at_frame / ds->s.fps;
    scene_render_at(&q->hi, t, i, 0, q->truth);
    lanczos_up(dataset_left(ds, i), ds->s.w, ds->s.h, q->lz, q->tmp, uw, uh);
    struct truth tr;
    scene_truth_at(&q->hi, t, &tr);
    const int x0 = (int)(tr.u - tr.half_w), x1 = (int)(tr.u + tr.half_w), y0 = (int)(tr.v - tr.half_h), y1 = (int)(tr.v + tr.half_h);
    for (int y = 0; y < uh; y++)
        for (int x = 0; x < uw; x++) {
            const double tv = q->truth[y * uw + x], a = (double)pic[y * uw + x] - tv, b = (double)q->lz[y * uw + x] - tv;
            q->sse_fsr += a * a;
            q->sse_lz += b * b;
            q->n++;
            if (x >= x0 && x < x1 && y >= y0 && y < y1) {
                q->sse_fsr_thing += a * a;
                q->sse_lz_thing += b * b;
                q->n_thing++;
            } else {
                q->sse_fsr_rest += a * a;
                q->sse_lz_rest += b * b;
                q->n_rest++;
            }
        }
}

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
    uint8_t *fsr_pic = NULL, *fsr_gen = NULL;
    static struct fsr_check fq, fm;
    if (o.cfg.fsr_camera) {
        const size_t fb = (size_t)o.cfg.fsr_w * o.cfg.fsr_h;
        fsr_pic = host_alloc(NULL, fb);
        fsr_gen = host_alloc(NULL, fb);
        if (!fsr_pic || !fsr_gen || tk_fsr_setup(t[0], (int)o.cfg.fsr_up_w, (int)o.cfg.fsr_up_h, (int)o.cfg.fsr_w, (int)o.cfg.fsr_h, (long double)o.cfg.fsr_sharp, host_alloc, NULL, fsr_gen, fsr_pic, NULL, NULL))
            return 2;
        scene_setup(&fq.hi, ds.s.id, (int)o.cfg.fsr_w, (int)o.cfg.fsr_h, ds.s.nframes, ds.s.fps);
        fq.hi.clean = 1;
        fq.truth = malloc(fb);
        fq.lz = malloc(fb * sizeof(float));
        fq.tmp = malloc(fb * sizeof(float));
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
    static struct vt_result r, rg;
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
            const int fsr_ran = g >= 0 && fsr_pic && c == 0;
            if (fsr_ran)
                tk_fsr_frame(t[c]);
            tk_track(t[c], set, &m, &r);
            if (fsr_ran) {
                r.fsr_w = o.cfg.fsr_w;
                r.fsr_h = o.cfg.fsr_h;
                if (i >= 30)
                {
                    if (!fm.truth)
                        fm = fq;
                    fsr_check_frame(&fq, &ds, i, (double)i, fsr_pic, (int)o.cfg.fsr_w, (int)o.cfg.fsr_h);
                    /* The frame half way, against the truth half way: Lanczos here is of the real frame it came after, what the eye had before it. */
                    fsr_check_frame(&fm, &ds, i, (double)i - 0.5, fsr_gen, (int)o.cfg.fsr_w, (int)o.cfg.fsr_h);
                }
            }
            if (g >= 0) {
                tk_generated_result(t[c], &gm, &r, &rg);
                if (fsr_ran) {
                    rg.fsr_w = o.cfg.fsr_w;
                    rg.fsr_h = o.cfg.fsr_h;
                }
                rg.hdr.pub_ns = 0;
                rg.hdr.slot = 0;
                memset(rg.cycles, 0, sizeof rg.cycles);
                const size_t gbytes = offsetof(struct vt_result, feat) + (size_t)rg.nfeat * sizeof(struct vt_feat_out);
                printf("%dg %016llx\n", i, (unsigned long long)(vt_fnv(VT_FNV_INIT, &rg, gbytes) ^ (pics[c] ? vt_fnv(VT_FNV_INIT, pics[c], pic_bytes[c]) : 0ull) ^ (fsr_ran ? vt_fnv(VT_FNV_INIT, fsr_gen, (size_t)o.cfg.fsr_w * o.cfg.fsr_h) : 0ull)));
                if (fgcheck && c == 0 && i > 0) {
                    const double mid = ((double)gm.acq_ns - 1e9) * 1e-9;
                    struct truth tr;
                    scene_truth_at(&ds.s, mid, &tr);
                    scene_render_at(&ds.s, mid, i, 0, fg_truth);
                    const int w = ds.s.w, h = ds.s.h, x0 = (int)(tr.u - tr.half_w) - 8, x1 = (int)(tr.u + tr.half_w) + 8, y0 = (int)(tr.v - tr.half_h) - 8, y1 = (int)(tr.v + tr.half_h) + 8;
                    fg_add(&fg_whole, pics[0], fg_truth, dataset_left(&ds, i - 1), dataset_left(&ds, i), w, 0, 0, w, h);
                    fg_add(&fg_thing, pics[0], fg_truth, dataset_left(&ds, i - 1), dataset_left(&ds, i), w, x0 < 0 ? 0 : x0, y0 < 0 ? 0 : y0, x1 > w ? w : x1, y1 > h ? h : y1);
                }
                score_frame(s, &rg, 0.0, 0.0);
            }
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
            printf("%d %016llx\n", i, (unsigned long long)(vt_fnv(VT_FNV_INIT, &r, bytes) ^ (fsr_ran ? vt_fnv(VT_FNV_INIT, fsr_pic, (size_t)o.cfg.fsr_w * o.cfg.fsr_h) : 0ull)));
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
    if (fsr_pic) {
        tk_fsr_report(t[0], ds.s.name, 1.0);
        fprintf(stderr, "fsrcheck %s %dx%d to %ux%u, dB against the clean truth: whole FSR %.2f Lanczos %.2f; on the thing FSR %.2f Lanczos %.2f; elsewhere FSR %.2f Lanczos %.2f\n", ds.s.name, ds.s.w, ds.s.h, o.cfg.fsr_w, o.cfg.fsr_h, psnr(fq.sse_fsr, fq.n), psnr(fq.sse_lz, fq.n), psnr(fq.sse_fsr_thing, fq.n_thing), psnr(fq.sse_lz_thing, fq.n_thing), psnr(fq.sse_fsr_rest, fq.n_rest), psnr(fq.sse_lz_rest, fq.n_rest));
        fprintf(stderr, "fsrgen %s %dx%d to %ux%u, dB against the clean truth: whole FSR %.2f Lanczos %.2f; on the thing FSR %.2f Lanczos %.2f; elsewhere FSR %.2f Lanczos %.2f\n", ds.s.name, ds.s.w, ds.s.h, o.cfg.fsr_w, o.cfg.fsr_h, psnr(fm.sse_fsr, fm.n), psnr(fm.sse_lz, fm.n), psnr(fm.sse_fsr_thing, fm.n_thing), psnr(fm.sse_lz_thing, fm.n_thing), psnr(fm.sse_fsr_rest, fm.n_rest), psnr(fm.sse_lz_rest, fm.n_rest));
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
