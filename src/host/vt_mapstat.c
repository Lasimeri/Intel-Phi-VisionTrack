/* vt_mapstat.c: reads a depth buffer the live window saved (out/run/map.bin) and says what is in it: per camera, how its cells stand and how their differences and vectors spread; of the map, how its depths spread.
 *   vt_mapstat out/run/map.bin */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vt_proto.h"

static uint8_t buf[VT_DEPTH_BYTES];

static int cmp_f(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static float pct(const float *s, int n, double p) { return n ? s[(int)(p * (n - 1))] : 0.0f; }

int main(int argc, char **argv)
{
    FILE *f = argc > 1 ? fopen(argv[1], "rb") : NULL;
    if (!f || fread(buf, sizeof buf, 1, f) != 1) {
        fprintf(stderr, "usage: vt_mapstat MAP.bin\n");
        return 2;
    }
    fclose(f);
    const struct vt_depth_out *h = (const struct vt_depth_out *)buf;
    printf("map %llu, frame %llu of D, %u x %u, %u planes %.2f .. %.2f m, %u textured, %u confident = %.2f %%, %u with depth, geometry %llu\n", (unsigned long long)h->seq, (unsigned long long)h->instance, h->w, h->h, h->planes, (double)h->z_near, (double)h->z_far, h->textured, h->confident, h->textured ? 100.0 * h->confident / h->textured : 0.0, h->known, (unsigned long long)h->geometry);
    static float d[1 << 20], len[1 << 20], res[1 << 20], over[1 << 20];
    for (int k = 0; k < VT_MAX_CAMERAS; k++) {
        const struct vt_motion *m = &h->cam[k];
        const struct vt_cell *c = vt_depth_cells(buf, k);
        const int n = (int)(m->cols * m->rows);
        int nl = 0, st[16] = {0}, changed = 0, moving = 0;
        long count = 0;
        for (int i = 0; i < n; i++) {
            d[i] = c[i].diff;
            const unsigned lk = c[i].state & VT_CELL_LK;
            for (int b = 0; b < 4; b++)
                st[b] += (lk >> b) & 1;
            st[15] += lk == 0;
            if (lk == 0) {
                const float l = sqrtf(c[i].u * c[i].u + c[i].v * c[i].v);
                over[nl] = c[i].sigma > 0.0f ? l / c[i].sigma : 0.0f;
                len[nl++] = l;
            }
            changed += (c[i].state & VT_CELL_CHANGED) != 0;
            moving += (c[i].state & VT_CELL_MOVING) != 0;
            count += c[i].state >> VT_CELL_COUNT_SHIFT;
        }
        qsort(d, (size_t)n, sizeof(float), cmp_f);
        qsort(len, (size_t)nl, sizeof(float), cmp_f);
        qsort(over, (size_t)nl, sizeof(float), cmp_f);
        printf("%c: %u x %u cells, %d frames in a mean on average, frames %llu; changed %d, moving %d\n", "LRD"[k], m->cols, m->rows, (int)(count / (n ? n : 1)), (unsigned long long)m->frames, changed, moving);
        printf("   LK: good %d, out %d, low eigenvalue %d, residual %d, flowback %d\n", st[15], st[0], st[1], st[2], st[3]);
        printf("   |frame - mean| grey: min %.3f p10 %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f\n", (double)pct(d, n, 0), (double)pct(d, n, 0.1), (double)pct(d, n, 0.5), (double)pct(d, n, 0.9), (double)pct(d, n, 0.99), (double)pct(d, n, 1));
        printf("   limit %.3f grey\n", (double)m->diff_limit);
        printf("   |vector| px of the level: p50 %.3f p90 %.3f p99 %.3f max %.3f; drift %+.4f %+.4f rms %.4f\n", (double)pct(len, nl, 0.5), (double)pct(len, nl, 0.9), (double)pct(len, nl, 0.99), (double)pct(len, nl, 1), (double)m->drift_u, (double)m->drift_v, (double)m->drift_rms);
        printf("   |vector| / its sigma: p50 %.1f p90 %.1f p99 %.1f max %.1f\n", (double)pct(over, nl, 0.5), (double)pct(over, nl, 0.9), (double)pct(over, nl, 0.99), (double)pct(over, nl, 1));
    }
    const float *z = vt_depth_z(buf), *conf = vt_depth_conf(buf);
    int nz = 0;
    for (uint32_t r = 0; r < h->h; r++)
        for (uint32_t x = 0; x < h->w; x++)
            if (z[r * h->stride + x] > 0.0f && conf[r * h->stride + x] >= 0.2f)
                res[nz++] = z[r * h->stride + x];
    qsort(res, (size_t)nz, sizeof(float), cmp_f);
    printf("depth at confidence >= 0.2, %d pixels, m: p5 %.2f p25 %.2f p50 %.2f p75 %.2f p95 %.2f\n", nz, (double)pct(res, nz, 0.05), (double)pct(res, nz, 0.25), (double)pct(res, nz, 0.5), (double)pct(res, nz, 0.75), (double)pct(res, nz, 0.95));
    return 0;
}
