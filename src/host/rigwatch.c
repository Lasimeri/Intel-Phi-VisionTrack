/* rigwatch.c: see rigwatch.h. */
#include "rigwatch.h"

#include <math.h>
#include <stdlib.h>

/* A cell is textured when it differs from its neighbours by this many grey levels: a uniform wall shows nothing about a move. */
#define TEXTURE_MIN 1.5f
/* Moved: the best shift is not zero, and the zero-shift mismatch is this much worse than its. Between what was measured on frames of known answer, 11-03: 15 still frames at most 1.013, 3 moved
 * frames at least 1.117 (camera D's excluded: its known move reads 1.00 in a dark room, a miss this cannot catch). */
#define MOVED_RATIO 1.06f
/* Held still: a range within this many times the middle cell's range. */
#define STILL_RANGE 1.5f

static float median_inplace(float *a, int n)
{
    /* Hoare's selection: the middle element, the array left in pieces. */
    int lo = 0, hi = n - 1;
    const int k = n / 2;
    while (lo < hi) {
        const float p = a[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (a[i] < p)
                i++;
            while (a[j] > p)
                j--;
            if (i <= j) {
                const float t = a[i];
                a[i] = a[j];
                a[j] = t;
                i++;
                j--;
            }
        }
        if (k <= j)
            hi = j;
        else if (k >= i)
            lo = i;
        else
            break;
    }
    return a[k];
}

void rig_thumb(const uint8_t *gray, int img_w, int img_h, float *thumb)
{
    const int w = img_w / RIG_CELL, h = img_h / RIG_CELL;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint32_t s = 0;
            for (int j = 0; j < RIG_CELL; j++) {
                const uint8_t *row = gray + (size_t)(y * RIG_CELL + j) * (size_t)img_w + (size_t)x * RIG_CELL;
                for (int i = 0; i < RIG_CELL; i++)
                    s += row[i];
            }
            thumb[y * w + x] = (float)s / (float)(RIG_CELL * RIG_CELL);
        }
}

void rig_stable_mask(const float *const *hist, int k, int w, int h, float *scratch, uint8_t *mask)
{
    const int n = w * h;
    float gains[32];
    for (int f = 0; f < k && f < 32; f++) {
        for (int i = 0; i < n; i++)
            scratch[i] = hist[f][i];
        const float m = median_inplace(scratch, n);
        gains[f] = m > 0.5f ? 1.0f / m : 1.0f;
    }
    for (int i = 0; i < n; i++) {
        float lo = 1e30f, hi = -1e30f;
        for (int f = 0; f < k && f < 32; f++) {
            const float v = hist[f][i] * gains[f];
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        scratch[i] = hi - lo;
    }
    /* mask is written from a second pass: the selection below scrambles scratch, so the ranges are recomputed against the middle it found. */
    const float middle = median_inplace(scratch, n);
    const float limit = STILL_RANGE * middle;
    for (int i = 0; i < n; i++) {
        float lo = 1e30f, hi = -1e30f;
        for (int f = 0; f < k && f < 32; f++) {
            const float v = hist[f][i] * gains[f];
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        mask[i] = hi - lo <= limit;
    }
}

void rig_compare(const float *ref, const uint8_t *mask, const float *cur, int w, int h, float *scratch, struct rig_compare *out)
{
    /* The room's light follows whatever a screen shows: one gain between the two, from their middles. */
    int n = 0;
    for (int i = 0; i < w * h; i++)
        scratch[n++] = ref[i];
    const float mr = median_inplace(scratch, n);
    n = 0;
    for (int i = 0; i < w * h; i++)
        scratch[n++] = cur[i];
    const float mc = median_inplace(scratch, n);
    const float gain = mc > 0.5f ? mr / mc : 1.0f;

    /* Which cells of the reference can show a move. Stepped by two: this is an estimate of a middle, not a sum. */
    const int margin = RIG_MAX_SHIFT + 1;
    int *cells = malloc(sizeof(int) * (size_t)(w * h));
    int nc = 0;
    for (int y = margin; y < h - margin; y += 2)
        for (int x = margin; x < w - margin; x += 2) {
            const float c = ref[y * w + x];
            const float g = fabsf(c - ref[y * w + x + 1]) + fabsf(c - ref[y * w + x - 1]) + fabsf(c - ref[(y + 1) * w + x]) + fabsf(c - ref[(y - 1) * w + x]);
            if (g >= 4.0f * TEXTURE_MIN && mask[y * w + x])
                cells[nc++] = y * w + x;
        }
    out->cells = nc;
    out->moved = 0;
    out->dx = out->dy = 0;
    out->at_zero = out->best = 0.0f;
    if (nc < RIG_MIN_CELLS) {
        free(cells);
        return;
    }
    float best = 1e30f;
    int bx = 0, by = 0;
    for (int dy = -RIG_MAX_SHIFT; dy <= RIG_MAX_SHIFT; dy++)
        for (int dx = -RIG_MAX_SHIFT; dx <= RIG_MAX_SHIFT; dx++) {
            const int off = dy * w + dx;
            for (int i = 0; i < nc; i++)
                scratch[i] = fabsf(ref[cells[i]] - gain * cur[cells[i] + off]);
            const float m = median_inplace(scratch, nc);
            if (dx == 0 && dy == 0)
                out->at_zero = m;
            if (m < best) {
                best = m;
                bx = dx;
                by = dy;
            }
        }
    free(cells);
    out->best = best;
    out->dx = bx * RIG_CELL;
    out->dy = by * RIG_CELL;
    out->moved = (bx || by) && out->at_zero > MOVED_RATIO * best;
}
