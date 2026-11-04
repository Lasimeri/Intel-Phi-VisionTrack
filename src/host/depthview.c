/* depthview.c: see depthview.h. */
#define _GNU_SOURCE
#include "depthview.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* 256 entries of a perceptually even rainbow, from Anton Mikhailov's polynomial fit to Turbo, built once: 0 is far and cold, 255 near and warm. */
static uint32_t lut[256];

static void lut_build(void)
{
    if (lut[255])
        return;
    for (int i = 0; i < 256; i++) {
        const double t = (double)i / 255.0;
        const double r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
        const double g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
        const double b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814.0 - t * (22569.18 - t * 6838.66))));
        const int R = r < 0.0 ? 0 : r > 255.0 ? 255 : (int)r, G = g < 0.0 ? 0 : g > 255.0 ? 255 : (int)g, B = b < 0.0 ? 0 : b > 255.0 ? 255 : (int)b;
        lut[i] = (uint32_t)R << 16 | (uint32_t)G << 8 | (uint32_t)B;
    }
}

void dv_pane(struct x11view *v, const void *map, int x, int y, int pane_w, int pane_h, int mirror)
{
    const struct vt_depth_out *h = map;
    if (!h->w || !h->h)
        return;
    lut_build();
    const float *z = vt_depth_z(map), *conf = vt_depth_conf(map);
    const int sx = pane_w / (int)h->w, sy = pane_h / (int)h->h;
    if (sx < 1 || sy < 1)
        return;
    const float near_i = 1.0f / h->z_near, far_i = 1.0f / h->z_far, gain = 255.0f / (near_i - far_i);
    /* One line of the pane per row of the map, then copied down: a whole pane at the camera's rate costs a screen's copy, not a screen's arithmetic. */
    static uint32_t line[8192];
    const int pw = (int)h->w * sx;
    if (pw > 8192 || x < 0 || x + pw > v->w)
        return;
    for (int r = 0; r < (int)h->h; r++) {
        const float *row = z + (size_t)r * h->stride, *crow = conf + (size_t)r * h->stride;
        for (int c = 0; c < (int)h->w; c++) {
            const int at = mirror ? (int)h->w - 1 - c : c;
            const float d = row[at];
            uint32_t px = 0;
            if (d > 0.0f) {
                const float t = (1.0f / d - far_i) * gain;
                px = lut[t < 0.0f ? 0 : t > 255.0f ? 255 : (int)t];
                /* Unsure depths are shown as unsure, a quarter as bright: measured, the LEDs' come out at confidence 0.07 and are metres wrong. */
                static const uint32_t KEEP[2] = {0xffffffu, 0x3f3f3fu};
                const uint32_t unsure = crow[at] < 0.2f;
                px = (px >> (unsure * 2u)) & KEEP[unsure];
            }
            for (int k = 0; k < sx; k++)
                line[c * sx + k] = px;
        }
        for (int k = 0; k < sy; k++) {
            const int yy = y + r * sy + k;
            if (yy >= 0 && yy < v->h)
                memcpy(v->px + (size_t)yy * (size_t)v->w + (size_t)x, line, (size_t)pw * sizeof(uint32_t));
        }
    }
}

void dv_legend(struct x11view *v, const void *map, int x, int y, int w)
{
    const struct vt_depth_out *h = map;
    lut_build();
    const int bar = w - 200;
    for (int i = 0; i < bar; i++)
        x11view_fill(v, x + 100 + i, y, 1, 10, lut[255 - i * 255 / bar]);
    char t[32];
    snprintf(t, sizeof t, "%.1f m", (double)h->z_near);
    x11view_text(v, x, y - 2, 2, 0xffffffu, t);
    snprintf(t, sizeof t, "%.1f m", (double)h->z_far);
    x11view_text(v, x + 110 + bar, y - 2, 2, 0xffffffu, t);
}

/* A pixel and its depth, in the world: X_w = Rd' (X_d - Td). */
static inline void to_world(const struct vt_depth_out *h, const struct vt_calib *c, int r, int k, double d, double *o)
{
    const double X = ((double)k - (double)h->cx) / (double)h->f * d, Y = ((double)r - (double)h->cy) / (double)h->f * d;
    const double a = X - c->Td[0], b = Y - c->Td[1], e = d - c->Td[2];
    const double *R = c->Rd;
    o[0] = R[0] * a + R[3] * b + R[6] * e;
    o[1] = R[1] * a + R[4] * b + R[7] * e;
    o[2] = R[2] * a + R[5] * b + R[8] * e;
}

void dv_cloud(struct x11view *v, const void *map, const struct vt_calib *cal, int box_x, int box_y, int box_size, int cx, int cy, double side, double px_per_m)
{
    const struct vt_depth_out *h = map;
    if (!h->w || !cal->valid_d || !(v->w > 0))
        return;
    lut_build();
    const float *z = vt_depth_z(map), *conf = vt_depth_conf(map);
    const float near_i = 1.0f / h->z_near, far_i = 1.0f / h->z_far, gain = 255.0f / (near_i - far_i);
    for (int r = 0; r < (int)h->h; r++) {
        const float *zr = z + (size_t)r * h->stride, *cr = conf + (size_t)r * h->stride;
        for (int k = 0; k < (int)h->w; k++) {
            const float d = zr[k];
            if (d <= 0.0f || cr[k] < 0.2f)
                continue;
            double o[3];
            to_world(h, cal, r, k, (double)d, o);
            const int px = cx + (int)(o[0] * side), py = cy - (int)(o[2] * px_per_m);
            if (px < box_x || px >= box_x + box_size || py < box_y || py >= box_y + box_size)
                continue;
            const float t = (1.0f / d - far_i) * gain;
            v->px[(size_t)py * (size_t)v->w + (size_t)px] = lut[t < 0.0f ? 0 : t > 255.0f ? 255 : (int)t];
        }
    }
}

void dv_cells(struct x11view *v, const struct vt_cell *cells, uint32_t cols, uint32_t rows, int flow_level, int pane_x, int pane_y, int pane_w, double view, int mirror, int all)
{
    const int scale = 1 << flow_level;
    /* Pixels of the level drawn in pixels of the camera, and then made visible: a vector of a quarter of a level's pixel is one pixel of the camera. */
    const double gain = (double)scale * view * (all ? 64.0 : 4.0);
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            const struct vt_cell *e = &cells[r * cols + c];
            const int good = (e->state & VT_CELL_LK) == 0, changed = (e->state & VT_CELL_CHANGED) != 0;
            if (!(all ? good : changed))
                continue;
            int x0 = (int)((double)((int)(c * VT_CELL_W + VT_CELL_W / 2) * scale) * view);
            const int y0 = (int)((double)((int)(r * VT_CELL_H + VT_CELL_H / 2) * scale) * view);
            double du = good ? (double)e->u * gain : 0.0;
            const double dv = good ? (double)e->v * gain : 0.0;
            if (mirror) {
                x0 = pane_w - 1 - x0;
                du = -du;
            }
            const uint32_t colour = changed ? 0xff8040u : 0x40c0ffu;
            x11view_line(v, pane_x + x0, pane_y + y0, pane_x + x0 + (int)du, pane_y + y0 + (int)dv, colour);
            x11view_fill(v, pane_x + x0 - 1, pane_y + y0 - 1, 3, 3, colour);
        }
}

void dv_vectors(struct x11view *v, const void *map, int cam, int pane_x, int pane_y, int pane_w, double view, int mirror, int all)
{
    const struct vt_depth_out *h = map;
    const struct vt_motion *m = &h->cam[cam];
    if (m->cols && m->cell_at)
        dv_cells(v, vt_depth_cells(map, cam), m->cols, m->rows, (int)h->flow_level, pane_x, pane_y, pane_w, view, mirror, all);
}

int dv_save(const void *map, const struct vt_calib *cal, const char *prefix)
{
    const struct vt_depth_out *h = map;
    char path[256];
    snprintf(path, sizeof path, "%s.bin", prefix);
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fwrite(map, VT_DEPTH_BYTES, 1, f);
    fclose(f);
    const float *z = vt_depth_z(map), *conf = vt_depth_conf(map);
    snprintf(path, sizeof path, "%s.pgm", prefix);
    f = fopen(path, "wb");
    if (!f)
        return -1;
    fprintf(f, "P5\n%u %u\n255\n", h->w, h->h);
    const float near_i = 1.0f / h->z_near, far_i = 1.0f / h->z_far, gain = 239.0f / (near_i - far_i);
    for (uint32_t r = 0; r < h->h; r++)
        for (uint32_t k = 0; k < h->w; k++) {
            const float d = z[r * h->stride + k];
            const float t = d > 0.0f ? 16.0f + (1.0f / d - far_i) * gain : 0.0f;
            fputc(t < 0.0f ? 0 : t > 255.0f ? 255 : (int)t, f);
        }
    fclose(f);
    long points = 0;
    for (uint32_t i = 0; i < h->h * h->stride; i++)
        points += z[i] > 0.0f;
    snprintf(path, sizeof path, "%s.ply", prefix);
    f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "ply\nformat ascii 1.0\ncomment VisionTrack depth map, seq %llu, frame %llu of camera D; world = camera L, metres\nelement vertex %ld\nproperty float x\nproperty float y\nproperty float z\nproperty float confidence\nend_header\n", (unsigned long long)h->seq, (unsigned long long)h->instance, points);
    for (uint32_t r = 0; r < h->h; r++)
        for (uint32_t k = 0; k < h->w; k++) {
            const float d = z[r * h->stride + k];
            if (d <= 0.0f)
                continue;
            double o[3];
            to_world(h, cal, (int)r, (int)k, (double)d, o);
            fprintf(f, "%.4f %.4f %.4f %.3f\n", o[0], o[1], o[2], (double)conf[r * h->stride + k]);
        }
    fclose(f);
    return 0;
}

void dv_csv(FILE *f, const void *map, double wall_s, double tsc_hz, unsigned line)
{
    const struct vt_depth_out *h = map;
    if (!line) {
        fprintf(f, "wall_s,seq,instance,geometry,textured,confident,confident_pct,known,normalize_us,sweep_us,aggregate_us,decide_us");
        for (int k = 0; k < VT_MAX_CAMERAS; k++) {
            const char c = "lrd"[k];
            fprintf(f, ",%c_frames,%c_tracked,%c_moving,%c_count,%c_drift_u,%c_drift_v,%c_drift_rms,%c_diff,%c_vectors_us,%c_mean_us,%c_changed,%c_chg_x,%c_chg_y,%c_chg_spread", c, c, c, c, c, c, c, c, c, c, c, c, c, c);
        }
        fputc('\n', f);
    }
    const double us = 1e6 / tsc_hz;
    fprintf(f, "%.3f,%llu,%llu,%llu,%u,%u,%.3f,%u,%.0f,%.0f,%.0f,%.0f", wall_s, (unsigned long long)h->seq, (unsigned long long)h->instance, (unsigned long long)h->geometry, h->textured, h->confident, h->textured ? 100.0 * (double)h->confident / (double)h->textured : 0.0, h->known, (double)h->cycles_normalize * us, (double)h->cycles_sweep * us, (double)h->cycles_aggregate * us, (double)h->cycles_decide * us);
    for (int k = 0; k < VT_MAX_CAMERAS; k++) {
        const struct vt_motion *m = &h->cam[k];
        fprintf(f, ",%llu,%u,%u,%.1f,%.4f,%.4f,%.4f,%.3f,%.0f,%.0f", (unsigned long long)m->frames, m->tracked, m->moving, (double)m->count_mean, (double)m->drift_u, (double)m->drift_v, (double)m->drift_rms, (double)m->diff_median, (double)m->cycles_vectors * us, (double)m->cycles_mean * us);
        /* Where this camera saw something change, in its own pixels: a small thing moving is one place in each view at one moment, with no appearance to match. */
        const struct vt_cell *cells = vt_depth_cells(map, k);
        const double cell_x = (double)(VT_CELL_W << h->flow_level), cell_y = (double)(VT_CELL_H << h->flow_level);
        double sx = 0.0, sy = 0.0, sxx = 0.0;
        unsigned n = 0;
        for (uint32_t r = 0; r < m->rows; r++)
            for (uint32_t c = 0; c < m->cols; c++)
                if (cells[r * m->cols + c].state & VT_CELL_CHANGED) {
                    const double x = ((double)c + 0.5) * cell_x, y = ((double)r + 0.5) * cell_y;
                    sx += x;
                    sy += y;
                    sxx += x * x + y * y;
                    n++;
                }
        const double mx = n ? sx / n : 0.0, my = n ? sy / n : 0.0, spread = n ? sxx / n - mx * mx - my * my : 0.0;
        fprintf(f, ",%u,%.1f,%.1f,%.1f", n, mx, my, spread > 0.0 ? sqrt(spread) : 0.0);
    }
    fputc('\n', f);
    fflush(f);
}
