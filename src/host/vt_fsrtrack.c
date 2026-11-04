/* vt_fsrtrack SCENE WxH --mono --fsr WxH --fsr-up WxH --fsr-sharp S [--frames N]: is a tracker steadier on FSR's pictures of real frames than on the raw frames? The user's question, 2026-11-03.
 * Three trackers over one scene: one makes the pictures and tracks nothing; one follows the raw frames; one follows the pictures, its box scaled up and its answers scaled back, so every number is in raw pixels. Generated pictures go to neither:
 * measured, they cost a tracker its target. Compares the centre against the truth and, what stability means, each frame's step against the truth's step. */
#define _GNU_SOURCE
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run_opts.h"
#include "track.h"

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

struct tally {
    double err2, step2, abs_step;
    double *errs;
    int n, steps, observed, visible;
    double px, py, tx, ty;
    int have;
};

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void note(struct tally *t, const struct vt_result *r, double x, double y, const struct truth *tr, int counted)
{
    if (!counted)
        return;
    t->visible++;
    const int seen = r->state == VT_ST_OBSERVED;
    t->observed += seen;
    if (!seen) {
        t->have = 0;
        return;
    }
    const double e = hypot(x - tr->u, y - tr->v);
    t->errs[t->n++] = e;
    t->err2 += e * e;
    if (t->have) {
        /* How far this frame's step differs from the thing's: jitter, whatever the box's offset. */
        const double d = hypot((x - t->px) - (tr->u - t->tx), (y - t->py) - (tr->v - t->ty));
        t->step2 += d * d;
        t->abs_step += d;
        t->steps++;
    }
    t->px = x;
    t->py = y;
    t->tx = tr->u;
    t->ty = tr->v;
    t->have = 1;
}

static void report(const char *name, struct tally *t)
{
    qsort(t->errs, (size_t)t->n, sizeof(double), cmp_d);
    printf("  %-5s continuity %6.4f  centre median %7.3f p95 %7.3f px  step error mean %7.4f rms %7.4f px\n", name, t->visible ? (double)t->observed / t->visible : 0.0, t->n ? t->errs[t->n / 2] : 0.0, t->n ? t->errs[(int)(t->n * 0.95)] : 0.0, t->steps ? t->abs_step / t->steps : 0.0, t->steps ? sqrt(t->step2 / t->steps) : 0.0);
}

int main(int argc, char **argv)
{
    struct run_opts o;
    struct dataset ds;
    if (run_opts_parse(&o, argc, argv, &ds))
        return 2;
    if (!o.cfg.fsr_w) {
        fprintf(stderr, "vt_fsrtrack: needs --fsr\n");
        return 2;
    }
    const int fw = (int)o.cfg.fsr_w, fh = (int)o.cfg.fsr_h;
    const double sx = (double)fw / ds.s.w, sy = (double)fh / ds.s.h;
    /* The maker. */
    struct tracker *maker = tk_create(&o.cfg, 1, 1, host_alloc, NULL, host_run, NULL, host_run, NULL);
    uint8_t *gen_small = host_alloc(NULL, (size_t)ds.s.w * (size_t)ds.s.h), *pic_mid = host_alloc(NULL, (size_t)fw * (size_t)fh), *pic = host_alloc(NULL, (size_t)fw * (size_t)fh);
    tk_set_picture(maker, gen_small);
    if (!maker || tk_fsr_setup(maker, (int)o.cfg.fsr_up_w, (int)o.cfg.fsr_up_h, fw, fh, (long double)o.cfg.fsr_sharp, host_alloc, NULL, pic_mid, pic, NULL, NULL))
        return 2;
    /* The two followers: alike but for size. */
    struct vt_config raw_cfg, fsr_cfg;
    struct scene hi;
    vt_default_config(&raw_cfg, &ds.s, 0);
    scene_setup(&hi, ds.s.id, fw, fh, ds.s.nframes, ds.s.fps);
    vt_default_config(&fsr_cfg, &hi, 0);
    struct tracker *raw = tk_create(&raw_cfg, 1, 1, host_alloc, NULL, host_run, NULL, host_run, NULL), *on_fsr = tk_create(&fsr_cfg, 1, 1, host_alloc, NULL, host_run, NULL, host_run, NULL);
    if (!raw || !on_fsr)
        return 2;
    static struct vt_result r0, r1, r2;
    struct tally a = {0}, b = {0};
    a.errs = malloc(sizeof(double) * (size_t)o.nframes);
    b.errs = malloc(sizeof(double) * (size_t)o.nframes);
    const int nsets = tk_sets(maker), nsets2 = tk_sets(raw);
    int fsr_ready = 0;
    for (int i = 0; i < o.nframes; i++) {
        struct vt_msg m;
        run_opts_msg(&o, &ds, i, 0, &m);
        const uint8_t *left = dataset_left(&ds, i);
        struct vt_msg made = m, gm;
        made.cmd = VT_CMD_NONE;
        tk_build(maker, i % nsets, left, NULL, &made);
        const int g = tk_generate(maker, i % nsets, &made, &gm);
        if (g >= 0) {
            tk_fsr_frame(maker);
            fsr_ready = 1;
        }
        tk_track(maker, i % nsets, &made, &r0);
        tk_build(raw, i % nsets2, left, NULL, &m);
        tk_track(raw, i % nsets2, &m, &r1);
        struct truth tr;
        scene_truth_at(&ds.s, ((double)m.acq_ns - 1e9) * 1e-9, &tr);
        const int counted = i > ds.s.select_frame && tr.on_screen && tr.visible >= 0.6;
        note(&a, &r1, (double)r1.cx, (double)r1.cy, &tr, counted);
        /* The same message and the same box, in the picture's pixels. There is no picture yet on the first frame: it waits. */
        struct vt_msg mf = m;
        mf.roi[0] = (float)(((double)m.roi[0] + 0.5) * sx - 0.5);
        mf.roi[1] = (float)(((double)m.roi[1] + 0.5) * sy - 0.5);
        mf.roi[2] = (float)((double)m.roi[2] * sx);
        mf.roi[3] = (float)((double)m.roi[3] * sy);

        if (fsr_ready) {
            tk_build(on_fsr, i % nsets2, pic, NULL, &mf);
            tk_track(on_fsr, i % nsets2, &mf, &r2);
            note(&b, &r2, ((double)r2.cx + 0.5) / sx - 0.5, ((double)r2.cy + 0.5) / sy - 0.5, &tr, counted);
        }
    }
    printf("%s, %d frames, %dx%d, pictures %dx%d (up %ux%u, RCAS %.1f), all in raw pixels\n", ds.s.name, o.nframes, ds.s.w, ds.s.h, fw, fh, o.cfg.fsr_up_w, o.cfg.fsr_up_h, (double)o.cfg.fsr_sharp);
    report("raw", &a);
    report("FSR", &b);
    dataset_close(&ds);
    return 0;
}
