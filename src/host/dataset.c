/* dataset.c: see dataset.h. */
#define _GNU_SOURCE
#include "dataset.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vt_link.h"

int dataset_open(struct dataset *d, enum scene_id id, int w, int h, int nframes, double fps)
{
    memset(d, 0, sizeof *d);
    scene_setup(&d->s, id, w, h, nframes, fps);
    d->frame_bytes = 2 * (size_t)w * (size_t)h;
    d->file_bytes = d->frame_bytes * (size_t)nframes;
    char dir[] = "out/data", path[256];
    mkdir("out", 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/%s-%dx%d-%d-%g.y8", dir, d->s.name, w, h, nframes, fps);
    struct stat st;
    if (stat(path, &st) || (size_t)st.st_size != d->file_bytes) {
        const uint64_t t0 = vt_now_ns();
        char tmp[300];
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        FILE *f = fopen(tmp, "w");
        if (!f) {
            perror(tmp);
            return -1;
        }
        uint8_t *img = malloc((size_t)w * (size_t)h);
        for (int i = 0; i < nframes; i++)
            for (int cam = 0; cam < 2; cam++) {
                scene_render(&d->s, i, cam, img);
                if (fwrite(img, (size_t)w * (size_t)h, 1, f) != 1) {
                    perror(tmp);
                    fclose(f);
                    return -1;
                }
            }
        free(img);
        fclose(f);
        if (rename(tmp, path)) {
            perror(path);
            return -1;
        }
        fprintf(stderr, "host: rendered %s in %.1f s\n", path, (double)(vt_now_ns() - t0) * 1e-9);
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return -1;
    }
    void *p = mmap(NULL, d->file_bytes, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        perror("mmap");
        return -1;
    }
    d->frames = p;
    return 0;
}

void dataset_close(struct dataset *d) { munmap(d->frames, d->file_bytes); }

void dataset_roi(const struct dataset *d, int i, float *roi)
{
    struct truth t;
    scene_truth(&d->s, i, &t);
    /* Inside the object, except where the point of the scene is a box with background in it. */
    const enum scene_id id = d->s.id;
    const double k = id == SC_SMALLBOX ? 4.0 : id == SC_BIGBOX || id == SC_STILLSTART || id == SC_ROTATE || id == SC_SWING || id == SC_COMEBACK ? 2.4 : 0.8;
    roi[0] = (float)(t.u - k * t.half_w);
    roi[1] = (float)(t.v - k * t.half_h);
    roi[2] = (float)(2.0 * k * t.half_w);
    roi[3] = (float)(2.0 * k * t.half_h);
}

void dataset_roi_cam(const struct dataset *d, int i, int camera, float *roi)
{
    if (camera == 0) {
        dataset_roi(d, i, roi);
        return;
    }
    struct truth t;
    scene_truth_at(&d->s, (double)i / d->s.fps + d->s.b_delay_s, &t);
    roi[0] = (float)(t.ub - 0.8 * t.half_wb);
    roi[1] = (float)(t.vb - 0.8 * t.half_hb);
    roi[2] = (float)(1.6 * t.half_wb);
    roi[3] = (float)(1.6 * t.half_hb);
}

void dataset_true_calib(const struct dataset *d, struct vt_calib *c)
{
    memset(c, 0, sizeof *c);
    c->valid = 1;
    c->fa = d->s.f;
    c->cxa = c->cxb = d->s.cx;
    c->cya = c->cyb = d->s.cy;
    c->fb = d->s.b_f;
    memcpy(c->R, d->s.b_R, sizeof c->R);
    memcpy(c->T, d->s.b_T, sizeof c->T);
}

void vt_default_config(struct vt_config *c, const struct scene *s, int stereo)
{
    memset(c, 0, sizeof *c);
    c->kind = VT_SESSION_TRACK;
    c->width = (uint32_t)s->w;
    c->height = (uint32_t)s->h;
    c->stereo = (uint32_t)stereo;
    c->slots = 4;
    c->cores = 0;
    c->tpc = 2;
    c->pipelines = 1;
    /* Fitted against truth on the synthetic scenes (docs/measurements.md). A real rig must fit its own. */
    c->sigma_c = 0.1f;
    c->sigma_d = 0.01f;
    c->levels = 4;
    c->patch_w = 21;
    c->patch_h = 21;
    c->max_iter = 10;
    c->eps = 0.01f;
    c->min_eig = 0.1f;
    c->max_residual = 25.0f;
    c->fb_max = 0.5f;
    c->max_features = 128;
    c->min_dist = (float)s->w / 80.0f;
    c->corner_quality = 0.02f;
    c->min_features = 6;
    c->min_inliers = 0.35f;
    c->prefetch = 1;
    c->max_missing = 45;
    c->reacq_ncc = 0.82f; /* near the prediction. True takes back read 0.978 and 0.988; 0.785 and 0.774 landed about 120 px from the last observed position (live, 2026-11-03) */
    c->reacq_radius = 12;
    c->fx = (float)s->f;
    c->fy = (float)s->f;
    c->cx = (float)s->cx;
    c->cy = (float)s->cy;
    c->baseline = (float)s->baseline;
    c->min_disp = 2.0f;
    c->max_disp = (float)s->w / 8.0f;
    c->min_ncc = 0.6f;
    c->ambiguity = 0.85f;
    c->max_reproj = 1.0f;
    c->q_accel = 0.5f;
    c->r_floor = 0.002f;
    c->v0_sigma = 2.0f;
}
