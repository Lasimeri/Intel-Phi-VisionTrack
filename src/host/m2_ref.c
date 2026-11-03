/* m2_ref.c: the M2 LK cases through the host lane model: the hash lines the card's vt_m2 must reproduce.
 *   m2_ref W H [PATCH_W PATCH_H LEVELS] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "img.h"
#include "lk.h"
#include "m2_input.h"
#include "vt_util.h"

#define MAX_LEVELS 6
#define MAX_POINTS 4096

struct pyr { struct vt_plane l[MAX_LEVELS]; };

static void make_pyr(struct pyr *p, int w, int h, int levels)
{
    for (int i = 0; i < levels; i++) {
        void *m = aligned_alloc(64, plane_bytes(w, h));
        if (!m)
            exit(2);
        p->l[i] = plane_at(m, w, h);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

static void build_pyr(const struct pyr *p, const uint8_t *img, int stride, int levels)
{
    img_u8_to_q6_vx(&p->l[0], img, stride, 0, p->l[0].h);
    img_pad_rows(&p->l[0], 0, p->l[0].h);
    img_pad_tb(&p->l[0]);
    for (int i = 1; i < levels; i++) {
        img_reduce_vx(&p->l[i], &p->l[i - 1], 0, p->l[i].h);
        img_pad_rows(&p->l[i], 0, p->l[i].h);
        img_pad_tb(&p->l[i]);
    }
}

static float px[MAX_POINTS], py[MAX_POINTS];
static struct lk_out out_v[MAX_POINTS];

int main(int argc, char **argv)
{
    const int w = argc > 1 ? atoi(argv[1]) : 640, h = argc > 2 ? atoi(argv[2]) : 480;
    const struct lk_cfg cfg = {.w = argc > 3 ? atoi(argv[3]) : 21, .h = argc > 4 ? atoi(argv[4]) : 21, .max_iter = 10, .eps = 0.005f, .min_eig = 0.1f, .max_residual = 40.0f};
    const int levels = argc > 5 ? atoi(argv[5]) : 4;
    const int stride = (w + 15) & ~15;
    uint8_t *img0 = aligned_alloc(64, (size_t)stride * (size_t)h), *img1 = aligned_alloc(64, (size_t)stride * (size_t)h);
    struct pyr pa, pb;
    make_pyr(&pa, w, h, levels);
    make_pyr(&pb, w, h, levels);
    int n = 0;
    const int margin = 40 + cfg.w;
    for (int y = margin; y < h - margin && n < MAX_POINTS; y += 24)
        for (int x = margin; x < w - margin && n < MAX_POINTS; x += 24) {
            px[n] = (float)x;
            py[n] = (float)y;
            n++;
        }
    void *tile = aligned_alloc(64, (lk_tile_bytes() + 63) & ~(size_t)63);
    m2_render(img0, w, h, stride, M2_OFFSET, M2_OFFSET);
    build_pyr(&pa, img0, stride, levels);
    for (int c = 0; c < M2_NCASES; c++) {
        const int dx = m2_cases[c].dx, dy = m2_cases[c].dy;
        m2_render(img1, w, h, stride, M2_OFFSET - dx, M2_OFFSET - dy);
        build_pyr(&pb, img1, stride, levels);
        uint64_t hash = VT_FNV_INIT;
        for (int i = 0; i < n; i++) {
            lk_track_vx(pa.l, pb.l, levels - 1, &cfg, px[i], py[i], 0.0f, 0.0f, &out_v[i], tile);
            hash = vt_fnv(hash, &out_v[i], sizeof out_v[i]);
        }
        printf("lk_first %d,%d x %08x y %08x eig %08x res %08x cxx %08x cxy %08x cyy %08x it %u st %u\n", dx, dy, vt_float_bits(out_v[0].x), vt_float_bits(out_v[0].y), vt_float_bits(out_v[0].min_eig), vt_float_bits(out_v[0].residual), vt_float_bits(out_v[0].cov_xx), vt_float_bits(out_v[0].cov_xy), vt_float_bits(out_v[0].cov_yy), out_v[0].iters, out_v[0].status);
        printf("lk_hash %d,%d %016llx\n", dx, dy, (unsigned long long)hash);
    }
    return 0;
}
