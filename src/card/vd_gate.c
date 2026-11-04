/* vd_gate.c: the orange vectors' plane and its way down, against what they must do, with the same answer on the card and the host (vd_ref). No camera: the cells are made here.
 *
 * Holds: a line lands where its cell and vector say, in the plane and after the Lanczos; nothing lands outside what was marked; the next frame's erase leaves the plane all zero, bit for bit;
 * and drawing again in zero is cheaper than clearing the plane, which is why it is done that way. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fsr3.h"
#include "vecdraw.h"
#include "vt_proto.h"

#define UW 1280
#define UH 720
#define RW 640
#define RH 360
#define DW 960
#define DH 540
#define LEVEL 1
#define COLS 20  /* of a 320 x 180 level: 16 x 8 cells */
#define ROWS 23
#define THREADS 3

static inline uint64_t cycles_now(void)
{
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static int fails;
static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += !ok;
}

static void *block(size_t n)
{
    void *p = aligned_alloc(64, (n + 63u) & ~(size_t)63u);
    if (!p) {
        fprintf(stderr, "vd: no memory for %zu bytes\n", n);
        exit(2);
    }
    return p;
}

static int all_zero(const struct vt_plane *p)
{
    /* Every lane the plane has, its padding too: a read of the padding is part of the way down. */
    const float *f = plane_row(p, -VT_PAD_Y) - VT_PAD_X;
    const size_t lanes = (size_t)p->stride * (size_t)(p->h + 2 * VT_PAD_Y);
    uint32_t any = 0;
    for (size_t i = 0; i < lanes; i++) {
        uint32_t bits;
        memcpy(&bits, &f[i], 4);
        any |= bits;
    }
    return any == 0;
}

int main(void)
{
    struct vecdraw *d = vd_init(block(vd_bytes(UW, UH, COLS * ROWS)), UW, UH, COLS * ROWS);
    const struct vt_plane *p = vd_plane(d);
    struct fsr *f = fsr_init(block(fsr_bytes(RW, RH, UW, UH, DW, DH)), RW, RH, UW, UH, DW, DH, 0.4L);
    uint8_t *pic = block((size_t)DW * DH);
    float *temp = block(fsr_down_temp_bytes(f) * THREADS);
    struct vt_cell *cells = block(sizeof(struct vt_cell) * COLS * ROWS);
    memset(cells, 0, sizeof(struct vt_cell) * COLS * ROWS);
    /* Cell 7 of row 11 moved 3.5 px right and 1.25 up, at its level. Its centre is (120, 92) there, (480, 368) here, since this plane is 4 x that level: the level x 2 x view 2. */
    struct vt_cell *c = &cells[11 * COLS + 7];
    c->u = 3.5f;
    c->v = -1.25f;
    c->state = VT_CELL_CHANGED;
    /* One unchanged with a vector: not to be drawn. One changed with no good vector: a dot. One not a number: a dot and no walk. */
    cells[2 * COLS + 2].u = 9.0f;
    cells[4 * COLS + 15].state = VT_CELL_CHANGED | 2u;
    cells[4 * COLS + 15].u = 1000.0f;
    uint32_t nan_bits = 0x7fc00000u;
    memcpy(&cells[18 * COLS + 3].u, &nan_bits, 4);
    cells[18 * COLS + 3].state = VT_CELL_CHANGED;
    const long double view = (long double)UW / (long double)RW;
    /* Only inside a box: the long line's cell, centred at (240, 184) in the camera's pixels, is in this one, the others are not. */
    const float box[4] = {200.0f, 150.0f, 300.0f, 220.0f}, away[4] = {0.0f, 0.0f, 40.0f, 40.0f};
    check(vd_draw(d, cells, COLS, ROWS, LEVEL, view, box) == 1 && plane_row(p, 368)[480] == 1.0f && plane_row(p, 144)[992] == 0.0f, "inside a box, only the cell inside it");
    check(vd_draw(d, cells, COLS, ROWS, LEVEL, view, away) == 0 && all_zero(p), "a box with no cell in it: nothing");
    const int n = vd_draw(d, cells, COLS, ROWS, LEVEL, view, NULL);
    check(n == 3, "three cells changed, three lines");
    /* The gain is 4: 3.5 x 2 x 2 x 4 = 56 right, 1.25 x 16 = 20 up. */
    check(plane_row(p, 368)[480] == 1.0f, "the line starts at the cell's centre");
    check(plane_row(p, 348)[536] == 1.0f, "and ends where the vector says");
    check(plane_row(p, 368)[560] == 0.0f, "and goes no further");
    check(plane_row(p, 40)[40] == 0.0f, "an unchanged cell draws nothing");
    /* Column 15, row 4: (248, 36) at its level, (992, 144) here. */
    check(plane_row(p, 144)[992] == 1.0f && plane_row(p, 144)[1004] == 0.0f, "a changed cell with no good vector is a dot");
    /* Column 3, row 18: (56, 148), (224, 592) here. */
    check(plane_row(p, 592)[224] == 1.0f && plane_row(p, 592)[236] == 0.0f, "a vector not a number is a dot");
    check(vd_rows(d)[348] && vd_rows(d)[368] && !vd_rows(d)[200], "the rows touched are marked, the others not");
    uint64_t lit = 0;
    for (int y = 0; y < UH; y++)
        for (int x = 0; x < UW; x++)
            lit += plane_row(p, y)[x] != 0.0f;
    printf("lit pixels %llu\n", (unsigned long long)lit);
    int y0 = 0, y1 = 0;
    check(fsr_down_span(f, vd_rows(d), &y0, &y1) && y0 < 110 && y1 > 440 && y1 <= DH, "the rows to bring down span all three");
    memset(pic, 0x55, (size_t)DW * DH);
    for (int tid = 0; tid < THREADS; tid++)
        fsr_down_plane(f, p, vd_rows(d), y0, y1, pic, (float *)(void *)((char *)temp + (size_t)tid * fsr_down_temp_bytes(f)), tid, THREADS);
    /* At 3/4, the middle of the long line: (508, 358) here is (381, 268.5) there. */
    check(pic[268 * DW + 381] > 64 || pic[269 * DW + 381] > 64, "the line is there after the Lanczos, at 3/4 of its place");
    check(pic[200 * DW + 700] == 0, "a row with no line inside the span reads zero");
    const uint64_t hash = fnv(0xcbf29ce484222325ull, pic + (size_t)y0 * DW, (size_t)(y1 - y0) * DW);
    printf("down_hash %016llx rows %d %d\n", (unsigned long long)hash, y0, y1);
    /* What bringing it down costs, what the next frame's erase costs, and what clearing would have. */
    uint64_t t0 = cycles_now();
    for (int tid = 0; tid < THREADS; tid++)
        fsr_down_plane(f, p, vd_rows(d), y0, y1, pic, (float *)(void *)((char *)temp + (size_t)tid * fsr_down_temp_bytes(f)), tid, THREADS);
    const uint64_t down = cycles_now() - t0;
    t0 = cycles_now();
    const int gone = vd_draw(d, cells, 0, 0, LEVEL, view, NULL);
    const uint64_t erase = cycles_now() - t0;
    check(gone == 0 && all_zero(p), "the next frame's erase leaves the plane all zero");
    t0 = cycles_now();
    memset(plane_row(p, -VT_PAD_Y) - VT_PAD_X, 0, plane_bytes(UW, UH));
    const uint64_t clear = cycles_now() - t0;
    printf("cycles one thread: down %llu erase %llu clear %llu\n", (unsigned long long)down, (unsigned long long)erase, (unsigned long long)clear);
    check(erase < clear, "drawing the lines again in zero is cheaper than clearing");
    /* With no second stage, the way the live chain goes: the plane's own rows to bytes, nothing moved and nothing softened. */
    {
        struct fsr *straight = fsr_init(block(fsr_bytes(RW, RH, UW, UH, 0, 0)), RW, RH, UW, UH, 0, 0, 0.4L);
        uint8_t *whole = block((size_t)UW * UH);
        vd_draw(d, cells, COLS, ROWS, LEVEL, view, NULL);
        int s0 = 0, s1 = 0;
        check(fsr_down_span(straight, vd_rows(d), &s0, &s1) && s0 <= 144 && s1 > 592 && s1 <= UH, "straight: the rows to take out span all three");
        memset(whole, 0x55, (size_t)UW * UH);
        for (int tid = 0; tid < THREADS; tid++)
            fsr_down_plane(straight, p, vd_rows(d), s0, s1, whole, NULL, tid, THREADS);
        check(whole[368 * UW + 480] == 255 && whole[348 * UW + 536] == 255, "straight: the line where its cell and vector say, at full strength");
        check(whole[368 * UW + 560] == 0 && whole[500 * UW + 40] == 0, "straight: nothing beyond it");
        const uint64_t straight_hash = fnv(0xcbf29ce484222325ull, whole + (size_t)s0 * UW, (size_t)(s1 - s0) * UW);
        printf("straight_hash %016llx rows %d %d\n", (unsigned long long)straight_hash, s0, s1);
    }
    printf("%s\n", fails ? "vd: FAILED" : "vd: passed");
    return fails != 0;
}
