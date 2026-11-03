/* vt_rigcheck.c: the rig watch on saved frames of one camera, on the desktop host: vt_rigcheck K FRAME.pgm ...: the first K frames make the history, the oldest of them the reference, and every later frame
 * is compared. 8 or 16 bits. For testing the watch against frames whose answer is known. */
#include <stdio.h>
#include <stdlib.h>

#include "rigwatch.h"

static uint8_t *read_pgm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    int maxv;
    if (fscanf(f, "P5 %d %d %d", w, h, &maxv) != 3 || (maxv != 255 && maxv != 65535)) {
        fprintf(stderr, "%s: not a PGM\n", path);
        fclose(f);
        return NULL;
    }
    fgetc(f);
    const size_t n = (size_t)*w * (size_t)*h, bytes = maxv == 255 ? 1 : 2;
    uint8_t *raw = malloc(n * bytes), *g = malloc(n);
    if (!raw || !g || fread(raw, bytes, n, f) != n) {
        fprintf(stderr, "%s: short\n", path);
        fclose(f);
        return NULL;
    }
    fclose(f);
    for (size_t i = 0; i < n; i++)
        g[i] = raw[i * bytes]; /* 16-bit PGM is big-endian: the high byte */
    free(raw);
    return g;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: vt_rigcheck K FRAME.pgm ... (the first K are the history)\n");
        return 2;
    }
    const int k = atoi(argv[1]);
    if (k < 2 || k > 32 || argc < 2 + k + 1) {
        fprintf(stderr, "vt_rigcheck: K %d does not fit %d frames\n", k, argc - 2);
        return 2;
    }
    int w = 0, h = 0;
    float *hist[32], *thumb = NULL, *scratch = NULL;
    uint8_t *mask = NULL;
    int tw = 0, th = 0;
    for (int i = 0; i < argc - 2; i++) {
        int bw, bh;
        uint8_t *g = read_pgm(argv[i + 2], &bw, &bh);
        if (!g)
            return 2;
        if (!i) {
            w = bw;
            h = bh;
            tw = w / RIG_CELL;
            th = h / RIG_CELL;
            thumb = malloc(sizeof(float) * (size_t)(tw * th));
            scratch = malloc(sizeof(float) * (size_t)(tw * th));
            mask = malloc((size_t)(tw * th));
            if (!thumb || !scratch || !mask)
                return 2;
        } else if (bw != w || bh != h) {
            fprintf(stderr, "%s: another size\n", argv[i + 2]);
            return 2;
        }
        if (i < k) {
            hist[i] = malloc(sizeof(float) * (size_t)(tw * th));
            if (!hist[i])
                return 2;
            rig_thumb(g, w, h, hist[i]);
            if (i == k - 1)
                rig_stable_mask((const float *const *)hist, k, tw, th, scratch, mask);
        } else {
            rig_thumb(g, w, h, thumb);
            struct rig_compare c;
            rig_compare(hist[0], mask, thumb, tw, th, scratch, &c);
            printf("%-44s %s  zero %5.2f best %5.2f at %+4d %+4d px  cells %4d\n", argv[i + 2], c.cells < RIG_MIN_CELLS ? "UNSURE" : c.moved ? "MOVED " : "still ", (double)c.at_zero, (double)c.best, c.dx, c.dy, c.cells);
        }
        free(g);
    }
    return 0;
}
