/* img.h: image primitives. Each has a plain C reference (_ref) and a kernel against vx.h (_vx: native MVEX on the
 * card, the lane model on the host). A kernel works on output rows y0 .. y1 of its own, so a team splits a plane by
 * rows. Unless noted, the source plane's padding must be current, and the output's padding is filled by the
 * caller (img_pad_rows per thread, img_pad_rows_tb once). */
#ifndef VT_IMG_H
#define VT_IMG_H

#include "plane.h"

/* The padding of rows y0 .. y1 (left and right), and the rows above and below the plane. */
void img_pad_rows(const struct vt_plane *pl, int y0, int y1);
void img_pad_tb(const struct vt_plane *pl);

/* src: h rows of w bytes, src_stride a multiple of 16, src 16 B aligned. dst = src x 64. */
void img_u8_to_q6_ref(const struct vt_plane *dst, const uint8_t *src, int src_stride, int y0, int y1);
void img_u8_to_q6_vx(const struct vt_plane *dst, const uint8_t *src, int src_stride, int y0, int y1);
/* Back to bytes, rows y0 .. y1 of a plane whose width is a multiple of 16: dst holds h rows of w bytes, 64 B aligned. */
void img_q6_to_u8_vx(const struct vt_plane *src, uint8_t *dst, int y0, int y1);

/* dst = clamp(round(src x gain + offset), 0, 16320). The configurable normalization / linear contrast stretch. */
void img_norm_ref(const struct vt_plane *dst, const struct vt_plane *src, const float *gain_offset, int y0, int y1);
void img_norm_vx(const struct vt_plane *dst, const struct vt_plane *src, const float *gain_offset, int y0, int y1);

/* out[0] = min, out[1] = max over the rows. For the contrast stretch. */
void img_minmax_ref(const struct vt_plane *src, float *out, int y0, int y1);
void img_minmax_vx(const struct vt_plane *src, float *out, int y0, int y1);

/* The 5 x 5 binomial (1 4 6 4 1)^2 / 256, (h + 128) >> 8 in exact arithmetic. */
void img_gauss5_ref(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1);
void img_gauss5_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1);

/* One pyramid step: the same filter, every second pixel of every second row. dst is (w + 1) / 2 x (h + 1) / 2.
 *   _vx       the horizontal filter on every lane, every second one kept, software prefetch on
 *   _nopf_vx  the same without software prefetch */
void img_reduce_ref(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1);
void img_reduce_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1);
void img_reduce_nopf_vx(const struct vt_plane *dst, const struct vt_plane *src, int y0, int y1);

/* Sobel: ix = the [-1 0 1] x [1 2 1] filter, iy its transpose, exact. */
void img_sobel_ref(const struct vt_plane *ix, const struct vt_plane *iy, const struct vt_plane *src, int y0, int y1);
void img_sobel_vx(const struct vt_plane *ix, const struct vt_plane *iy, const struct vt_plane *src, int y0, int y1);

/* dst = |a - b|. */
void img_absdiff_ref(const struct vt_plane *dst, const struct vt_plane *a, const struct vt_plane *b, int y0, int y1);
void img_absdiff_vx(const struct vt_plane *dst, const struct vt_plane *a, const struct vt_plane *b, int y0, int y1);

/* A fence for the kernels' non-globally-ordered stores: call before another thread reads their output. */
void img_fence(void);

#endif
