/* m0_kern.h: the Milestone 0 kernels: a scalar and a vector fused multiply-add over one buffer, and a
 * self-test of every operation in vx.h. Only pointers and integers cross this boundary. */
#ifndef VT_M0_KERN_H
#define VT_M0_KERN_H

#include <stdint.h>

/* c[i] = a[i] * b[i] + c[i]. All 64 B aligned, n a multiple of 16. */
void m0_fma_vx(float *restrict c, const float *restrict a, const float *restrict b, long n);
/* The same in plain C: x87 on the card (extended-precision product and sum, rounded on the store), SSE on the host. */
void m0_fma_scalar(float *restrict c, const float *restrict a, const float *restrict b, long n);

#define M0_NOPS 18
extern const char *const m0_op_name[M0_NOPS];
/* Runs every vx.h operation over deterministic inputs; hash[i] covers operation i's raw output bits.
 * The host model and the card must print the same 18 numbers. */
void m0_vx_selftest(uint64_t *hash);
/* The same model-versus-card question for F32 subnormal inputs and results, reported apart: the card's
 * behaviour there was unmeasured (APL gate 2.1 did not reach that range). */
uint64_t m0_vx_subnormal(uint32_t *first_result_bits);

#endif
