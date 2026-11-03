/* m0_kern.c: see m0_kern.h. One source: native MVEX under the KNC clang, the lane model on the host. */
#include "m0_kern.h"

#include "vt_util.h"
#include "vx.h"

void m0_fma_vx(float *restrict c, const float *restrict a, const float *restrict b, long n)
{
    for (long i = 0; i < n; i += VX_LANES)
        vx_store(c + i, vx_fma(vx_load(a + i), vx_load(b + i), vx_load(c + i)));
}

void m0_fma_scalar(float *restrict c, const float *restrict a, const float *restrict b, long n)
{
    for (long i = 0; i < n; i++)
        c[i] = a[i] * b[i] + c[i];
}

const char *const m0_op_name[M0_NOPS] = {
    "load+store", "bcast",  "loadu",  "load_u8", "add",   "sub",  "mul",   "fma",   "fnma",
    "fma_s+mul_s", "minmax", "abs",    "alignr",  "perm",  "blend+cmp", "round_near", "round_down", "hsum",
};

#define BLOCKS 32

static float in_a[BLOCKS * VX_LANES] VX_ALIGNED, in_b[BLOCKS * VX_LANES] VX_ALIGNED, in_c[BLOCKS * VX_LANES] VX_ALIGNED;
static uint8_t in_u8[BLOCKS * VX_LANES] __attribute__((aligned(16)));

static void fill_inputs(void)
{
    uint64_t s = 0x1234567890abcdefull;
    for (int i = 0; i < BLOCKS * VX_LANES; i++) {
        in_a[i] = vt_rand_float(&s, 127u + (unsigned)(vt_rand(&s) % 16)); /* 1 .. 65536, either sign */
        in_b[i] = vt_rand_float(&s, 120u + (unsigned)(vt_rand(&s) % 16));
        in_c[i] = vt_rand_float(&s, 130u);
        in_u8[i] = (uint8_t)vt_rand(&s);
    }
}

static inline uint64_t mix(uint64_t h, vxf v)
{
    float out[VX_LANES] VX_ALIGNED;
    vx_store(out, v);
    return vt_fnv(h, out, sizeof out);
}

void m0_vx_selftest(uint64_t *hash)
{
    static const uint32_t abs_mask = 0x7fffffffu;
    static const int32_t rev[VX_LANES] VX_ALIGNED = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    fill_inputs();
    for (int i = 0; i < M0_NOPS; i++)
        hash[i] = VT_FNV_INIT;
    for (int b = 0; b < BLOCKS - 1; b++) {
        const float *pa = in_a + b * VX_LANES, *pb = in_b + b * VX_LANES, *pc = in_c + b * VX_LANES;
        const vxf a = vx_load(pa), v = vx_load(pb), c = vx_load(pc), a2 = vx_load(pa + VX_LANES);
        int op = 0;
        hash[op] = mix(hash[op], a);
        op++;
        hash[op] = mix(hash[op], vx_bcast(pa + 3));
        op++;
        hash[op] = mix(hash[op], vx_loadu(pa + (b % 15) + 1));
        op++;
        hash[op] = mix(hash[op], vx_load_u8(in_u8 + b * VX_LANES));
        op++;
        hash[op] = mix(hash[op], vx_add(a, v));
        op++;
        hash[op] = mix(hash[op], vx_sub(a, v));
        op++;
        hash[op] = mix(hash[op], vx_mul(a, v));
        op++;
        hash[op] = mix(hash[op], vx_fma(a, v, c));
        op++;
        hash[op] = mix(hash[op], vx_fnma(a, v, c));
        op++;
        hash[op] = mix(mix(hash[op], vx_fma_s(a, pb + 5, c)), vx_mul_s(a, pb + 7));
        op++;
        hash[op] = mix(mix(hash[op], vx_min(a, v)), vx_max(a, v));
        op++;
        hash[op] = mix(hash[op], vx_abs(a, &abs_mask));
        op++;
        hash[op] = mix(mix(mix(hash[op], vx_alignr(a2, a, 1)), vx_alignr(a2, a, 7)), vx_alignr(a2, a, 15));
        op++;
        hash[op] = mix(hash[op], vx_perm(a, rev));
        op++;
        hash[op] = mix(hash[op], vx_blend(vx_cmp(a, v, VX_CMP_LT), a, v));
        op++;
        hash[op] = mix(hash[op], vx_round(a, VX_RND_NEAREST));
        op++;
        hash[op] = mix(hash[op], vx_round(a, VX_RND_DOWN));
        op++;
        const float h = vx_hsum(a);
        hash[op] = vt_fnv(hash[op], &h, 4);
    }
}

uint64_t m0_vx_subnormal(uint32_t *first_result_bits)
{
    float a[VX_LANES] VX_ALIGNED, b[VX_LANES] VX_ALIGNED, c[VX_LANES] VX_ALIGNED;
    uint64_t s = 42;
    for (int i = 0; i < VX_LANES; i++) {
        a[i] = vt_rand_float(&s, 1u + (unsigned)i);   /* just above the subnormal range */
        b[i] = vt_rand_float(&s, 120u);               /* about 2^-7: the product falls below 2^-126 */
        c[i] = 0.0f;
    }
    float out[VX_LANES] VX_ALIGNED;
    vx_store(out, vx_mul(vx_load(a), vx_load(b)));
    *first_result_bits = vt_float_bits(out[0]);
    uint64_t h = vt_fnv(VT_FNV_INIT, out, sizeof out);
    vx_store(out, vx_fma(vx_load(a), vx_load(b), vx_load(c)));
    return vt_fnv(h, out, sizeof out);
}
