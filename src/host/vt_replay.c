/* vt_replay.c: the same scene and the same message stream through the same tracker source on the desktop host: the lane model, one thread. Prints the hash vt_run --hash wrote from the card's
 * results; with --json or --csv, scores it too (no card needed, so parameter fits run here). VT_TRACE=1 prints a line per frame on stderr. Options: run_opts.c. */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
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
            tk_track(t[c], set, &m, &r);
            if (o.dual)
                fuse_frame(&fu, c, &r);
            if (trace) {
                struct truth tr;
                scene_truth_at(&ds.s, ((double)r.hdr.acq_ns - 1e9) * 1e-9, &tr);
                fprintf(stderr, "f %3d c%d st %u fl %3u miss %2u nf %3u c %8.2f %7.2f truth %8.2f %7.2f vis %.2f ncc %6.3f nis %7.3f z %7.4f pos %7.4f %7.4f %7.4f tr %7.4f %7.4f %7.4f rp %.3f par %.1f\n", i, c, r.state, r.flags, r.missing, r.nfeat, (double)r.cx, (double)r.cy, c ? tr.ub : tr.u, c ? tr.vb : tr.v, tr.visible, (double)r.reacq_ncc, r.nis, r.meas[2], r.pos[0], r.pos[1], r.pos[2], tr.X, tr.Y, tr.Z, r.reproj_rms, r.parallax_deg);
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
    dataset_close(&ds);
    return 0;
}
