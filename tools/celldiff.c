/* celldiff.c: where two dumps of still cells part ways (m3_ref --dump, vt_m3's third argument). tcc -run tools/celldiff.c HOST CARD FRAMES [SHOW] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cell {
    float u, v, diff;
    uint32_t state;
    float sigma, pad;
};

static const int CELLS[3] = {1020, 1020, 460};

int main(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    FILE *a = fopen(argv[1], "rb"), *b = fopen(argv[2], "rb");
    const int frames = atoi(argv[3]), show = argc > 4 ? atoi(argv[4]) : 8;
    if (!a || !b)
        return 2;
    int shown = 0;
    for (int f = 0; f < frames; f++)
        for (int k = 0; k < 3; k++) {
            const size_t bytes = ((size_t)CELLS[k] * sizeof(struct cell) + 63) & ~(size_t)63;
            static struct cell x[2048], y[2048];
            if (fread(x, bytes, 1, a) != 1 || fread(y, bytes, 1, b) != 1) {
                printf("short file at frame %d camera %d\n", f, k);
                return 1;
            }
            int uv = 0, diff = 0, state = 0, sigma = 0;
            for (int i = 0; i < CELLS[k]; i++) {
                const int duv = memcmp(&x[i].u, &y[i].u, 8) != 0, dd = memcmp(&x[i].diff, &y[i].diff, 4) != 0, ds = x[i].state != y[i].state, dg = memcmp(&x[i].sigma, &y[i].sigma, 4) != 0;
                uv += duv;
                diff += dd;
                state += ds;
                sigma += dg;
                if ((duv || dd || ds || dg) && shown < show) {
                    shown++;
                    printf("  frame %d cam %d cell %d: host u %.9g v %.9g diff %.9g state %x sigma %.9g | card u %.9g v %.9g diff %.9g state %x sigma %.9g\n", f, k, i, (double)x[i].u, (double)x[i].v, (double)x[i].diff, x[i].state, (double)x[i].sigma, (double)y[i].u, (double)y[i].v, (double)y[i].diff, y[i].state, (double)y[i].sigma);
                }
            }
            if (uv || diff || state || sigma)
                printf("frame %d cam %d: cells differing: vector %d, difference %d, state %d, sigma %d of %d\n", f, k, uv, diff, state, sigma, CELLS[k]);
        }
    return 0;
}
