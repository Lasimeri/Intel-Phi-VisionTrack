/* m1_ref.c: M1 on the desktop host: every image primitive, plain C against the lane model of the same vector
 * source. Prints one hash line per kernel for one frame size: the lines the card's vt_m1 must reproduce.
 *
 *   m1_ref W H */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "img.h"
#include "m1_input.h"

static struct vt_plane new_plane(int w, int h)
{
    void *m = aligned_alloc(64, plane_bytes(w, h));
    if (!m)
        exit(2);
    memset(m, 0, plane_bytes(w, h));
    return plane_at(m, w, h);
}

static uint64_t hash_plane(const struct vt_plane *pl)
{
    uint64_t h = VT_FNV_INIT;
    for (int y = 0; y < pl->h; y++)
        h = vt_fnv(h, plane_row(pl, y), (size_t)pl->w * sizeof(float));
    return h;
}

int main(int argc, char **argv)
{
    const int w = argc > 1 ? atoi(argv[1]) : 640, h = argc > 2 ? atoi(argv[2]) : 480;
    const int stride8 = (w + 15) & ~15;
    uint8_t *img = aligned_alloc(64, (size_t)stride8 * (size_t)h);
    m1_make_image(img, w, h, stride8);
    struct vt_plane a = new_plane(w, h), b = new_plane(w, h), c = new_plane(w, h), d = new_plane(w, h);
    const int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
    struct vt_plane r = new_plane(w2, h2), r2 = new_plane(w2, h2);
    int bad = 0;
#define SAME(name, pl1, pl2)                                                                                     \
    do {                                                                                                         \
        const uint64_t h1 = hash_plane(&pl1), h2_ = hash_plane(&pl2);                                            \
        printf("kern %-12s %dx%d %016llx\n", name, w, h, (unsigned long long)h1);                                \
        if (h1 != h2_) {                                                                                         \
            printf("MISMATCH %s: reference %016llx lane model %016llx\n", name, (unsigned long long)h1, (unsigned long long)h2_); \
            bad = 1;                                                                                             \
        }                                                                                                        \
    } while (0)
    img_u8_to_q6_ref(&a, img, stride8, 0, h);
    img_u8_to_q6_vx(&b, img, stride8, 0, h);
    img_pad_rows(&a, 0, h);
    img_pad_tb(&a);
    SAME("u8_to_q6", a, b);

    const float go[2] = {0.75f, 512.0f};
    img_norm_ref(&c, &a, go, 0, h);
    img_norm_vx(&d, &a, go, 0, h);
    SAME("norm", c, d);

    float m0[2], m1[2];
    img_minmax_ref(&a, m0, 0, h);
    img_minmax_vx(&a, m1, 0, h);
    printf("kern minmax       %dx%d %.1f %.1f\n", w, h, (double)m0[0], (double)m0[1]);
    if (m0[0] != m1[0] || m0[1] != m1[1]) {
        printf("MISMATCH minmax\n");
        bad = 1;
    }

    img_gauss5_ref(&c, &a, 0, h);
    img_gauss5_vx(&d, &a, 0, h);
    SAME("gauss5", c, d);

    img_reduce_ref(&r, &a, 0, h2);
    img_reduce_vx(&r2, &a, 0, h2);
    SAME("reduce", r, r2);

    struct vt_plane e = new_plane(w, h), f = new_plane(w, h);
    img_sobel_ref(&c, &d, &a, 0, h);
    img_sobel_vx(&e, &f, &a, 0, h);
    SAME("sobel_x", c, e);
    SAME("sobel_y", d, f);

    img_gauss5_ref(&b, &a, 0, h);
    img_absdiff_ref(&c, &a, &b, 0, h);
    img_absdiff_vx(&d, &a, &b, 0, h);
    SAME("absdiff", c, d);
    return bad;
}
