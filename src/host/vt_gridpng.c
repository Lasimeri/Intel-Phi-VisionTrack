/* vt_gridpng.c: a snapshot made readable, on the desktop host. Crops a grey PGM (8 or 16 bits), stretches its contrast from the crop's own 1st to 99th percentile, enlarges it by an integer factor, and draws a grid
 * labelled in the ORIGINAL image's pixel coordinates, so a coordinate read by eye from the picture is a coordinate in the frame.
 *   vt_gridpng IN.pgm OUT.png [X Y W H [SCALE [STEP]]]    defaults: the whole image, scale 1, a grid line every 100 pixels
 * The PNG is 8-bit grey in stored deflate blocks: nothing here needs to be small. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t crc_table[256];

static void crc_init(void)
{
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
        crc_table[n] = c;
    }
}

static uint32_t crc_add(uint32_t c, const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        c = crc_table[(c ^ p[i]) & 255u] ^ (c >> 8);
    return c;
}

static void put32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static void chunk(FILE *f, const char *type, const uint8_t *d, size_t n)
{
    uint8_t b[4];
    put32(b, (uint32_t)n);
    fwrite(b, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (n)
        fwrite(d, 1, n, f);
    uint32_t c = crc_add(0xffffffffu, (const uint8_t *)type, 4);
    c = crc_add(c, d, n);
    put32(b, c ^ 0xffffffffu);
    fwrite(b, 1, 4, f);
}

/* 3 x 5 digits, a row per byte, the low 3 bits. */
static const uint8_t digit[10][5] = {{7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1}, {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}};

static void draw_number(uint8_t *px, int w, int h, int x, int y, int v)
{
    char s[12];
    const int n = snprintf(s, sizeof s, "%d", v);
    const int size = 2;
    for (int by = y - 1; by < y + 5 * size + 1; by++)
        for (int bx = x - 1; bx < x + n * 4 * size; bx++)
            if (bx >= 0 && bx < w && by >= 0 && by < h)
                px[by * w + bx] = 0;
    for (int i = 0; i < n; i++) {
        const uint8_t *g = digit[s[i] - '0'];
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 3; c++)
                if (g[r] >> (2 - c) & 1)
                    for (int a = 0; a < size; a++)
                        for (int b = 0; b < size; b++) {
                            const int X = x + i * 4 * size + c * size + b, Y = y + r * size + a;
                            if (X >= 0 && X < w && Y >= 0 && Y < h)
                                px[Y * w + X] = 255;
                        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: vt_gridpng IN.pgm OUT.png [X Y W H [SCALE [STEP]]]\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    int iw, ih, maxv;
    if (fscanf(f, "P5 %d %d %d", &iw, &ih, &maxv) != 3 || (maxv != 255 && maxv != 65535)) {
        fprintf(stderr, "%s: not an 8-bit or 16-bit PGM\n", argv[1]);
        return 2;
    }
    fgetc(f);
    const size_t npix = (size_t)iw * (size_t)ih, width = maxv == 255 ? 1 : 2;
    uint8_t *file = malloc(npix * width);
    uint16_t *in = malloc(npix * sizeof(uint16_t));
    if (!file || !in || fread(file, width, npix, f) != npix) {
        fprintf(stderr, "%s: short\n", argv[1]);
        return 2;
    }
    fclose(f);
    for (size_t i = 0; i < npix; i++)
        in[i] = maxv == 255 ? (uint16_t)(file[i] * 257u) : (uint16_t)((file[2 * i] << 8) | file[2 * i + 1]);
    free(file);
    int X = 0, Y = 0, W = iw, H = ih, S = 1, step = 100;
    if (argc >= 7) {
        X = atoi(argv[3]);
        Y = atoi(argv[4]);
        W = atoi(argv[5]);
        H = atoi(argv[6]);
    }
    if (argc >= 8)
        S = atoi(argv[7]);
    if (argc >= 9)
        step = atoi(argv[8]);
    if (X < 0)
        X = 0;
    if (Y < 0)
        Y = 0;
    if (X + W > iw)
        W = iw - X;
    if (Y + H > ih)
        H = ih - Y;
    if (W <= 0 || H <= 0 || S < 1 || step < 1) {
        fprintf(stderr, "vt_gridpng: nothing to show\n");
        return 2;
    }
    /* The crop's own histogram sets the stretch. */
    static uint64_t hist[65536];
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            hist[in[(y + Y) * iw + x + X]]++;
    const uint64_t total = (uint64_t)W * (uint64_t)H;
    int lo = 0, hi = 65535;
    uint64_t run = 0;
    for (int i = 0; i < 65536; i++) {
        run += hist[i];
        if (run * 100 < total)
            lo = i + 1;
        if (run * 100 < total * 99)
            hi = i + 1;
    }
    if (hi <= lo)
        hi = lo + 1;
    const int ow = W * S, oh = H * S;
    uint8_t *px = malloc((size_t)ow * (size_t)oh);
    if (!px)
        return 2;
    /* A square-root tone curve after the stretch: the shadows are where the room is. */
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            const size_t at = (size_t)(y / S + Y) * (size_t)iw + (size_t)(x / S + X);
            const double t = ((double)in[at] - lo) / (double)(hi - lo);
            px[y * ow + x] = t <= 0.0 ? 0 : t >= 1.0 ? 255 : (uint8_t)(sqrt(t) * 255.0);
        }
    /* The grid, inverted so it reads on any picture: dotted, so what is under it stays visible. The labels name the frame's own pixels. */
    const int fx = (X + step - 1) / step * step, fy = (Y + step - 1) / step * step;
    for (int gx = fx; gx < X + W; gx += step) {
        const int ox = (gx - X) * S + S / 2;
        for (int y = 0; y < oh; y += 4)
            px[y * ow + ox] ^= 0xff;
    }
    for (int gy = fy; gy < Y + H; gy += step) {
        const int oy = (gy - Y) * S + S / 2;
        for (int x = 0; x < ow; x += 4)
            px[oy * ow + x] ^= 0xff;
    }
    for (int gx = fx; gx < X + W; gx += step)
        for (int gy = fy; gy < Y + H; gy += step)
            draw_number(px, ow, oh, (gx - X) * S + S / 2 + 3, (gy - Y) * S + S / 2 + 3, gx);
    for (int gy = fy; gy < Y + H; gy += step)
        for (int gx = fx; gx < X + W; gx += step)
            draw_number(px, ow, oh, (gx - X) * S + S / 2 + 3, (gy - Y) * S + S / 2 + 16, gy);
    /* PNG: a filter byte per row, stored blocks of at most 65535 bytes, Adler-32 at the end. */
    const size_t raw = ((size_t)ow + 1) * (size_t)oh;
    const size_t blocks = raw / 65535 + 1;
    uint8_t *z = malloc(2 + raw + blocks * 5 + 4);
    if (!z)
        return 2;
    size_t zn = 0;
    z[zn++] = 0x78;
    z[zn++] = 0x01;
    uint32_t a = 1, b = 0;
    size_t left = raw, in_block = 0;
    for (int y = 0; y < oh; y++)
        for (int x = -1; x < ow; x++) {
            if (!in_block) {
                const size_t n = left > 65535 ? 65535 : left;
                z[zn++] = left <= 65535;
                z[zn++] = (uint8_t)n;
                z[zn++] = (uint8_t)(n >> 8);
                z[zn++] = (uint8_t)~n;
                z[zn++] = (uint8_t)(~n >> 8);
                in_block = n;
            }
            const uint8_t v = x < 0 ? 0 : px[y * ow + x];
            z[zn++] = v;
            a += v;
            if (a >= 65521)
                a -= 65521;
            b += a;
            if (b >= 65521)
                b -= 65521;
            in_block--;
            left--;
        }
    put32(z + zn, (b << 16) | a);
    zn += 4;
    FILE *o = fopen(argv[2], "wb");
    if (!o) {
        perror(argv[2]);
        return 2;
    }
    crc_init();
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
    fwrite(sig, 1, 8, o);
    uint8_t ihdr[13];
    put32(ihdr, (uint32_t)ow);
    put32(ihdr + 4, (uint32_t)oh);
    ihdr[8] = 8;
    ihdr[9] = 0;
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    chunk(o, "IHDR", ihdr, 13);
    chunk(o, "IDAT", z, zn);
    chunk(o, "IEND", z, 0);
    fclose(o);
    printf("%s: %d x %d from %d,%d size %d x %d, scale %d, grid %d, stretch %d .. %d\n", argv[2], ow, oh, X, Y, W, H, S, step, lo, hi);
    return 0;
}
