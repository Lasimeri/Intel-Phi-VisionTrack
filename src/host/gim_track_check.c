/* gim_track_check SCENE WxH [run options]: a stopped head when the whole picture jumps, as if the camera were knocked, with the jump reported and without. And that a run in which nothing is reported is the same to the bit: the reference does nothing until it has something to say. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "run_opts.h"
#include "track.h"

#define JUMP_AT 70
#define JUMP_X 6
#define JUMP_Y 3
#define REPORT_LAG 1 /* frames late, as the spare core is */

static void *host_alloc(void *ctx, size_t bytes)
{
    (void)ctx;
    return aligned_alloc(64, (bytes + 63u) & ~(size_t)63u);
}

static void host_run(void *ctx, tk_fn fn, void *arg)
{
    (void)ctx;
    fn(arg, 0, 1);
}

static int fails;
static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += !ok;
}

static void shifted(const uint8_t *src, uint8_t *dst, int w, int h, int sx, int sy)
{
    for (int y = 0; y < h; y++) {
        const int yy = y - sy < 0 ? 0 : y - sy >= h ? h - 1 : y - sy;
        for (int x = 0; x < w; x++) {
            const int xx = x - sx < 0 ? 0 : x - sx >= w ? w - 1 : x - sx;
            dst[(size_t)y * (size_t)w + (size_t)x] = src[(size_t)yy * (size_t)w + (size_t)xx];
        }
    }
}

struct outcome {
    float before_x, before_y, after_x, after_y;
    int woke, stopped;
    uint64_t hash;
};

/* jump: the picture jumps. report: and the tracker is told, a frame late. */
static void run(const struct run_opts *o, const struct dataset *ds, int jump, int report, struct outcome *out)
{
    struct vt_config cfg;
    vt_default_config(&cfg, &ds->s, 0);
    struct tracker *t = tk_create(&cfg, 1, 1, host_alloc, NULL, host_run, NULL, host_run, NULL);
    const int w = ds->s.w, h = ds->s.h, sets = tk_sets(t);
    uint8_t *frame = host_alloc(NULL, (size_t)w * (size_t)h);
    static struct vt_result r;
    memset(out, 0, sizeof *out);
    uint64_t hash = 0xcbf29ce484222325ull;
    for (int i = 0; i < o->nframes; i++) {
        struct vt_msg m;
        run_opts_msg(o, ds, i, 0, &m);
        const int gone = jump && i >= JUMP_AT;
        shifted(dataset_left(ds, i), frame, w, h, gone ? JUMP_X : 0, gone ? JUMP_Y : 0);
        tk_build(t, i % sets, frame, NULL, &m);
        if (report && i == JUMP_AT + REPORT_LAG)
            tk_rig_moved(t, (long double)JUMP_X, (long double)JUMP_Y);
        tk_track(t, i % sets, &m, &r);
        if (i == JUMP_AT - 1) {
            out->before_x = r.cx;
            out->before_y = r.cy;
            out->stopped = (r.flags & VT_F_AT_REST) != 0;
        }
        if (i >= JUMP_AT && !(r.flags & VT_F_AT_REST))
            out->woke = 1;
        out->after_x = r.cx;
        out->after_y = r.cy;
        r.hdr.pub_ns = 0;
        r.hdr.slot = 0;
        memset(r.cycles, 0, sizeof r.cycles);
        const uint8_t *b = (const uint8_t *)&r;
        const size_t bytes = offsetof(struct vt_result, feat) + (size_t)r.nfeat * sizeof(struct vt_feat_out);
        for (size_t k = 0; k < bytes; k++)
            hash = (hash ^ b[k]) * 0x100000001b3ull;
    }
    out->hash = hash;
}

int main(int argc, char **argv)
{
    struct run_opts o;
    struct dataset ds;
    if (run_opts_parse(&o, argc, argv, &ds))
        return 2;
    struct outcome still_a, still_b, plain, told;
    run(&o, &ds, 0, 0, &still_a);
    run(&o, &ds, 0, 0, &still_b);
    check(still_a.hash == still_b.hash, "the same run twice is the same to the bit");
    run(&o, &ds, 1, 0, &plain);
    run(&o, &ds, 1, 1, &told);
    const double plain_dx = (double)(plain.after_x - plain.before_x), plain_dy = (double)(plain.after_y - plain.before_y);
    const double told_dx = (double)(told.after_x - told.before_x), told_dy = (double)(told.after_y - told.before_y);
    printf("the picture jumped %d, %d px at frame %d. Not told: the window went %+.2f, %+.2f, head started %d. Told: %+.2f, %+.2f, head started %d\n", JUMP_X, JUMP_Y, JUMP_AT, plain_dx, plain_dy, plain.woke, told_dx, told_dy, told.woke);
    if (told.stopped) {
        check(fabs(told_dx - JUMP_X) <= 0.5 && fabs(told_dy - JUMP_Y) <= 0.5, "stopped head, told: the window stays on the same place in the world");
        check(!told.woke, "stopped head, told: the jump starts nothing");
    } else {
        check(told.hash == plain.hash, "moving head: it has taken the jump through its median, and being told changes nothing, to the bit");
    }
    printf("%s\n", fails ? "gim_track: FAILED" : "gim_track: passed");
    dataset_close(&ds);
    return fails != 0;
}
