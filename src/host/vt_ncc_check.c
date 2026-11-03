/* vt_ncc_check: the appearance correlation may never read over 1, and a nearly blank window may never read as a match. It did: on windows with under a quarter of a grey level of noise the old arithmetic read up to 6.13, and
 * live a region was taken by a blank patch at 10.6, since the search takes whatever scores highest. Uses the tracker's own function on the lane model, which the gates hold the card to. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "plane.h"
#include "track.h"
#include "vx.h"

#define W 32
#define H 21
#define PW 64
#define PH 42

int main(void)
{
    static float app[W * H] VX_ALIGNED;
    static float mem[(PW + 2 * VT_PAD_X) * (PH + 2 * VT_PAD_Y)] VX_ALIGNED;
    const struct vt_plane pl = plane_at(mem, PW, PH);
    srand(7);
    float sum = 0.0f;
    for (int i = 0; i < W * H; i++) {
        app[i] = (float)(6000 + rand() % 4000);
        sum += app[i];
    }
    const float mean = (float)((long double)sum / (long double)(W * H));
    long double norm = 0.0L;
    for (int i = 0; i < W * H; i++) {
        app[i] -= mean;
        norm += (long double)app[i] * (long double)app[i];
    }
    static const int levels[] = {16320, 16000, 12800, 6400, 640};
    static const int steps[] = {0, 1, 2, 4, 8, 16, 32, 64};
    long double worst = -2.0L;
    int over = 0, matched = 0, trials = 0;
    for (unsigned l = 0; l < sizeof levels / sizeof levels[0]; l++)
        for (unsigned s = 0; s < sizeof steps / sizeof steps[0]; s++)
            for (int trial = 0; trial < 5000; trial++) {
                for (int y = 0; y < PH; y++)
                    for (int x = 0; x < PW; x++)
                        plane_row(&pl, y)[x] = (float)(levels[l] - (steps[s] ? rand() % (steps[s] + 1) : 0));
                const long double v = tk_correlate(&pl, app, norm, 4, 4);
                trials++;
                worst = v > worst ? v : worst;
                over += v > 1.0L;
                matched += v >= 0.5L;
            }
    /* The thing itself must still score: the template on any level, at full contrast and at a third of it. */
    for (int y = 0; y < PH; y++)
        for (int x = 0; x < PW; x++)
            plane_row(&pl, y)[x] = 8000.0f;
    for (int r = 0; r < H; r++)
        for (int k = 0; k < W; k++)
            plane_row(&pl, 4 + r)[4 + k] = 8000.0f + app[r * W + k];
    const long double same = tk_correlate(&pl, app, norm, 4, 4);
    for (int r = 0; r < H; r++)
        for (int k = 0; k < W; k++)
            plane_row(&pl, 4 + r)[4 + k] = 8000.0f + rintf(app[r * W + k] / 3.0f);
    const long double dim = tk_correlate(&pl, app, norm, 4, 4);
    printf("%d nearly blank windows: worst correlation %.4Lf, over 1: %d, reading 0.5 or more: %d. The template itself %.6Lf, at a third of its contrast %.6Lf\n", trials, worst, over, matched, same, dim);
    return over || matched || same < 0.999L || dim < 0.999L;
}
