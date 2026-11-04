/* vx.h: every Knights Corner vector operation the kernels use, one name each, two bodies.
 *
 *   __KNC__ (Intel-Phi-LLVM clang, -march=knc): the builtin itself, one MVEX instruction per
 *     operation (vx_loadu and vx_hsum are the named sequences). vxf is a real 512-bit register.
 *   anything else (host cc, clangd): a lane-by-lane model that rounds as the card does: fmaf for
 *     every fused multiply-add, one IEEE operation per lane. vxf is a plain struct.
 *
 * The same kernel source therefore compiles to native MVEX on the card and to a bit-exact model of
 * itself on the host, so the model's output is a reference for the card's output, every lane. The
 * struct also makes vector operators a compile error: `a + b` on vxf does not build on the host.
 * That is deliberate. On the card `a + b` on a generic 512-bit vector compiles but SCALARIZES to x87
 * (measured 2026-11-02, docs/compatibility.md), the worst of both worlds.
 *
 * Alignment contract (Intel-Phi-APL gate 2.1, measured on this card): an up-converting memory
 * operand must be aligned to its own 16-element size: 16 B for {uint8}; one byte off is SIGSEGV. A
 * plain vector load or store is 64 B aligned unless the name says u.
 *
 * Assumes round-to-nearest-even. The model's agreement with the card is gated per kernel, not assumed. */
#ifndef VT_VX_H
#define VT_VX_H

#include <stdint.h>

#define VX_LANES 16
#define VX_BYTES 64
#define VX_ALIGNED __attribute__((aligned(VX_BYTES)))

/* MVEX SSS codes (327364-001, tables 2-3, 2-4). */
#define VX_SSS_NONE 0
#define VX_SSS_1TO16 1
#define VX_SSS_UINT8 4

/* vcmpps predicates. */
#define VX_CMP_EQ 0
#define VX_CMP_LT 1
#define VX_CMP_LE 2

/* vrndfxpntps immediate: bits 1:0 the rounding (0 nearest, 1 down, 2 up, 3 truncate). */
#define VX_RND_NEAREST 0
#define VX_RND_DOWN 1

#if defined(__KNC__)

#include <immintrin.h>

typedef float vxf __attribute__((vector_size(64), aligned(64)));
typedef int vxi __attribute__((vector_size(64), aligned(64)));
typedef unsigned short vxk;

#define VX_ALL ((vxk)0xffff)
#define VX_U ((__v16sf)_mm512_undefined_ps())
#define VX_UI ((__v16si)_mm512_undefined_epi32())

static inline vxf vx_zero(void) { return (vxf)_mm512_setzero_ps(); }
static inline vxf vx_load(const float *p) { return (vxf)_mm512_load_ps(p); }
static inline void vx_store(float *p, vxf v) { _mm512_store_ps(p, (__m512)v); }
/* vmovnrngoaps: a whole line written with no read-for-ownership. Only for lines not read back at once. */
static inline void vx_store_nr(float *p, vxf v) { __builtin_ia32_knc_vmovnrngoaps_mz(VX_ALL, p, (__v16sf)v, 0); }
/* vmovaps {uint8}: 16 floats to 16 bytes, rounded, saturated to 0 .. 255. p 16 B aligned. */
static inline void vx_store_u8(uint8_t *p, vxf v) { __builtin_ia32_knc_vmovaps_mz(VX_ALL, p, (__v16sf)v, VX_SSS_UINT8); }
/* vbroadcastss: p[0] in every lane. */
static inline vxf vx_bcast(const float *p) { return (vxf)__builtin_ia32_knc_vbroadcastss_zm(VX_U, VX_ALL, p, 0); }
/* 16 elements from any address: the unpack pair, two instructions. */
static inline vxf vx_loadu(const float *p)
{
    __v16sf v = __builtin_ia32_knc_vloadunpacklps_zm(VX_U, VX_ALL, p, 0);
    return (vxf)__builtin_ia32_knc_vloadunpackhps_zm(v, VX_ALL, (const char *)p + 64, 0);
}
/* vmovaps {uint8}: 16 bytes to 16 floats, 0.0 .. 255.0, exact. p 16 B aligned. */
static inline vxf vx_load_u8(const uint8_t *p) { return (vxf)__builtin_ia32_knc_vmovaps_zm(VX_U, VX_ALL, p, VX_SSS_UINT8); }
static inline vxf vx_add(vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vaddps_zzz(VX_U, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
static inline vxf vx_sub(vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vsubps_zzz(VX_U, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
static inline vxf vx_mul(vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vmulps_zzz(VX_U, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* a * b + c, one rounding. */
static inline vxf vx_fma(vxf a, vxf b, vxf c) { return (vxf)__builtin_ia32_knc_vfmadd231ps_zzz((__v16sf)c, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* c - a * b, one rounding. */
static inline vxf vx_fnma(vxf a, vxf b, vxf c) { return (vxf)__builtin_ia32_knc_vfnmadd231ps_zzz((__v16sf)c, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* a * *s + c with *s a {1to16} memory operand: no broadcast register. */
static inline vxf vx_fma_s(vxf a, const float *s, vxf c) { return (vxf)__builtin_ia32_knc_vfmadd231ps_zzm((__v16sf)c, VX_ALL, (__v16sf)a, s, VX_SSS_1TO16); }
static inline vxf vx_mul_s(vxf a, const float *s) { return (vxf)__builtin_ia32_knc_vmulps_zzm(VX_U, VX_ALL, (__v16sf)a, s, VX_SSS_1TO16); }
static inline vxf vx_min(vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vgminps_zzz(VX_U, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
static inline vxf vx_max(vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vgmaxps_zzz(VX_U, VX_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* The sign bit cleared: one vpandd with a {1to16} memory mask. */
static inline vxf vx_abs(vxf a, const uint32_t *mask7f)
{
    return (vxf)__builtin_ia32_knc_vpandd_zzm(VX_UI, VX_ALL, (__v16si)a, mask7f, VX_SSS_1TO16);
}
/* The 32 lanes hi:lo, shifted down by n lanes, the low 16 kept. n = 0 .. 15, a compile-time constant. */
#define vx_alignr(hi, lo, n) ((vxf)__builtin_ia32_knc_valignd_zzzi(VX_UI, VX_ALL, (__v16si)(hi), (__v16si)(lo), (n)))
/* Lane i takes v[idx[i]]. vpermd's indices are its register operand, its data the last operand, so the index
 * table cannot fold as memory: it would become the data (the M0 gate caught exactly that). */
static inline vxf vx_perm(vxf v, const int32_t *idx)
{
    const __v16si ix = (__v16si)_mm512_load_epi32(idx);
    return (vxf)__builtin_ia32_knc_vpermd_zzz(VX_UI, VX_ALL, ix, (__v16si)v);
}
/* Lanes of k from b, the rest from a. */
static inline vxf vx_blend(vxk k, vxf a, vxf b) { return (vxf)__builtin_ia32_knc_vblendmps_zzz(k, (__v16sf)a, (__v16sf)b, 0); }
#define vx_cmp(a, b, pred) ((vxk)__builtin_ia32_knc_vcmpps_kzzi(VX_ALL, (__v16sf)(a), (__v16sf)(b), (pred), 0))
/* Round to an integer value, in F32: vrndfxpntps. */
#define vx_round(v, mode) ((vxf)__builtin_ia32_knc_vrndfxpntps_zzi(VX_U, VX_ALL, (__v16sf)(v), (mode), 0))
/* The pairs i^1, i^2, i^4, i^8, in that order: this exact tree is what the model reproduces. */
static inline float vx_hsum(vxf v)
{
    __v16sf x = (__v16sf)v;
    x = __builtin_ia32_knc_vaddps_zzz(VX_U, VX_ALL, x, x, 1); /* {cdab} */
    x = __builtin_ia32_knc_vaddps_zzz(VX_U, VX_ALL, x, x, 2); /* {badc} */
    x = __builtin_ia32_knc_vaddps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0xB1), 0);
    x = __builtin_ia32_knc_vaddps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0x4E), 0);
    return x[0];
}
/* An index table held in a register, for a run of permutes. */
typedef __v16si vxidx;
static inline vxidx vx_idx(const int32_t *idx) { return (__v16si)_mm512_load_epi32(idx); }
static inline vxf vx_permi(vxf v, vxidx ix) { return (vxf)__builtin_ia32_knc_vpermd_zzz(VX_UI, VX_ALL, ix, (__v16si)v); }
/* Lanes of k take v[ix[i]]; the rest keep dst: one vpermd with a write mask, no separate blend. */
static inline vxf vx_perm_merge(vxf dst, vxk k, vxf v, vxidx ix) { return (vxf)__builtin_ia32_knc_vpermd_zzz((__v16si)dst, k, ix, (__v16si)v); }
/* Whole-valued lanes to int32. */
static inline vxidx vx_to_idx(vxf v) { return (__v16si)__builtin_ia32_knc_vcvtfxpntps2dq_zzi(VX_UI, VX_ALL, (__v16sf)v, 0, 0); }
/* base[ix[lane]], in src/kern/vx_gather.S: this compiler has no gather builtin and no safe way to name zmm registers in inline assembly. A program using it links that object. */
void vx_gather_knc(const float *base, const int32_t *index, float *out);
static inline vxf vx_gather(const float *base, vxidx ix)
{
    int32_t in[16] VX_ALIGNED;
    float out[16] VX_ALIGNED;
    _mm512_store_epi32(in, ix);
    vx_gather_knc(base, in, out);
    return vx_load(out);
}
/* a * *p with 16 bytes up-converted as the memory operand: one instruction, no load. p 16 B aligned. */
static inline vxf vx_mul_u8(vxf a, const uint8_t *p) { return (vxf)__builtin_ia32_knc_vmulps_zzm(VX_U, VX_ALL, (__v16sf)a, p, VX_SSS_UINT8); }
/* The extreme over 16 lanes, the same tree as vx_hsum. */
static inline float vx_hmin(vxf v)
{
    __v16sf x = (__v16sf)v;
    x = __builtin_ia32_knc_vgminps_zzz(VX_U, VX_ALL, x, x, 1);
    x = __builtin_ia32_knc_vgminps_zzz(VX_U, VX_ALL, x, x, 2);
    x = __builtin_ia32_knc_vgminps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0xB1), 0);
    x = __builtin_ia32_knc_vgminps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0x4E), 0);
    return x[0];
}
static inline float vx_hmax(vxf v)
{
    __v16sf x = (__v16sf)v;
    x = __builtin_ia32_knc_vgmaxps_zzz(VX_U, VX_ALL, x, x, 1);
    x = __builtin_ia32_knc_vgmaxps_zzz(VX_U, VX_ALL, x, x, 2);
    x = __builtin_ia32_knc_vgmaxps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0xB1), 0);
    x = __builtin_ia32_knc_vgmaxps_zzz(VX_U, VX_ALL, x, __builtin_ia32_knc_vpermf32x4_zzi(VX_U, VX_ALL, x, 0x4E), 0);
    return x[0];
}
static inline void vx_prefetch0(const void *p) { __builtin_ia32_knc_vprefetch0_m(p); }
static inline void vx_prefetch1(const void *p) { __builtin_ia32_knc_vprefetch1_m(p); }
/* Division and square root by Newton from the bits: this card has no vector divide, and vrcp23ps is an approximation the host could only copy by table. Fixed iterations of fused operations round the same on both sides, so a kernel
 * stays bit-exact. Relative error under 1e-6 from 3 iterations (measured: vt_m4). x > 0. */
static inline vxf vx_bits_rsub(vxf x, const uint32_t *k) { return (vxf)__builtin_ia32_knc_vpsubrd_zzm(VX_UI, VX_ALL, (__v16si)x, k, VX_SSS_1TO16); }
static inline vxf vx_bits_shr1(vxf x) { return (vxf)__builtin_ia32_knc_vpsrld_zzi(VX_UI, VX_ALL, (__v16si)x, 1, 0); }

#else /* the host model */

#include <math.h>
#include <string.h>

typedef struct { float l[16]; } vxf;
typedef unsigned short vxk;

#define VX_ALL ((vxk)0xffff)
#define VX_LANE_LOOP for (int i = 0; i < 16; i++)

static inline vxf vx_zero(void) { vxf r; VX_LANE_LOOP r.l[i] = 0.0f; return r; }
static inline vxf vx_load(const float *p) { vxf r; memcpy(r.l, p, sizeof r.l); return r; }
static inline void vx_store(float *p, vxf v) { memcpy(p, v.l, sizeof v.l); }
static inline void vx_store_nr(float *p, vxf v) { memcpy(p, v.l, sizeof v.l); }
static inline void vx_store_u8(uint8_t *p, vxf v) { VX_LANE_LOOP { const float s = rintf(v.l[i]); p[i] = s <= 0.0f ? 0u : s >= 255.0f ? 255u : (uint8_t)s; } }
static inline vxf vx_bcast(const float *p) { vxf r; VX_LANE_LOOP r.l[i] = *p; return r; }
static inline vxf vx_loadu(const float *p) { return vx_load(p); }
static inline vxf vx_load_u8(const uint8_t *p) { vxf r; VX_LANE_LOOP r.l[i] = (float)p[i]; return r; }
static inline vxf vx_add(vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = a.l[i] + b.l[i]; return r; }
static inline vxf vx_sub(vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = a.l[i] - b.l[i]; return r; }
static inline vxf vx_mul(vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = a.l[i] * b.l[i]; return r; }
static inline vxf vx_fma(vxf a, vxf b, vxf c) { vxf r; VX_LANE_LOOP r.l[i] = fmaf(a.l[i], b.l[i], c.l[i]); return r; }
static inline vxf vx_fnma(vxf a, vxf b, vxf c) { vxf r; VX_LANE_LOOP r.l[i] = fmaf(-a.l[i], b.l[i], c.l[i]); return r; }
static inline vxf vx_fma_s(vxf a, const float *s, vxf c) { vxf r; VX_LANE_LOOP r.l[i] = fmaf(a.l[i], *s, c.l[i]); return r; }
static inline vxf vx_mul_s(vxf a, const float *s) { vxf r; VX_LANE_LOOP r.l[i] = a.l[i] * *s; return r; }
static inline vxf vx_min(vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = fminf(a.l[i], b.l[i]); return r; }
static inline vxf vx_max(vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = fmaxf(a.l[i], b.l[i]); return r; }
static inline vxf vx_abs(vxf a, const uint32_t *mask7f) { (void)mask7f; vxf r; VX_LANE_LOOP r.l[i] = fabsf(a.l[i]); return r; }
static inline vxf vx_alignr_model(vxf hi, vxf lo, int n)
{
    vxf r;
    VX_LANE_LOOP r.l[i] = i + n < 16 ? lo.l[i + n] : hi.l[i + n - 16];
    return r;
}
#define vx_alignr(hi, lo, n) vx_alignr_model((hi), (lo), (n))
static inline vxf vx_perm(vxf v, const int32_t *idx) { vxf r; VX_LANE_LOOP r.l[i] = v.l[idx[i] & 15]; return r; }
static inline vxf vx_blend(vxk k, vxf a, vxf b) { vxf r; VX_LANE_LOOP r.l[i] = ((k >> i) & 1) ? b.l[i] : a.l[i]; return r; }
static inline vxk vx_cmp_model(vxf a, vxf b, int pred)
{
    unsigned k = 0;
    VX_LANE_LOOP {
        const int t = pred == VX_CMP_EQ ? a.l[i] == b.l[i] : pred == VX_CMP_LT ? a.l[i] < b.l[i] : a.l[i] <= b.l[i];
        k |= (unsigned)t << i;
    }
    return (vxk)k;
}
#define vx_cmp(a, b, pred) vx_cmp_model((a), (b), (pred))
static inline vxf vx_round_model(vxf v, int mode)
{
    vxf r;
    VX_LANE_LOOP r.l[i] = mode == VX_RND_DOWN ? floorf(v.l[i]) : rintf(v.l[i]);
    return r;
}
#define vx_round(v, mode) vx_round_model((v), (mode))
static inline float vx_hsum(vxf v)
{
    for (int step = 1; step <= 8; step <<= 1) {
        vxf t;
        VX_LANE_LOOP t.l[i] = v.l[i] + v.l[i ^ step];
        v = t;
    }
    return v.l[0];
}
typedef struct { int32_t l[16]; } vxidx;
static inline vxidx vx_idx(const int32_t *idx) { vxidx r; memcpy(r.l, idx, sizeof r.l); return r; }
static inline vxf vx_permi(vxf v, vxidx ix) { vxf r; VX_LANE_LOOP r.l[i] = v.l[ix.l[i] & 15]; return r; }
static inline vxf vx_perm_merge(vxf dst, vxk k, vxf v, vxidx ix) { vxf r; VX_LANE_LOOP r.l[i] = ((k >> i) & 1) ? v.l[ix.l[i] & 15] : dst.l[i]; return r; }
static inline vxidx vx_to_idx(vxf v) { vxidx r; VX_LANE_LOOP r.l[i] = (int32_t)lrintf(v.l[i]); return r; }
static inline vxf vx_gather(const float *base, vxidx ix) { vxf r; VX_LANE_LOOP r.l[i] = base[ix.l[i]]; return r; }
static inline vxf vx_mul_u8(vxf a, const uint8_t *p) { vxf r; VX_LANE_LOOP r.l[i] = a.l[i] * (float)p[i]; return r; }
static inline float vx_hmin(vxf v)
{
    for (int step = 1; step <= 8; step <<= 1) {
        vxf t;
        VX_LANE_LOOP t.l[i] = fminf(v.l[i], v.l[i ^ step]);
        v = t;
    }
    return v.l[0];
}
static inline float vx_hmax(vxf v)
{
    for (int step = 1; step <= 8; step <<= 1) {
        vxf t;
        VX_LANE_LOOP t.l[i] = fmaxf(v.l[i], v.l[i ^ step]);
        v = t;
    }
    return v.l[0];
}
static inline void vx_prefetch0(const void *p) { (void)p; }
static inline void vx_prefetch1(const void *p) { (void)p; }
static inline vxf vx_bits_rsub(vxf x, const uint32_t *k) { vxf r; VX_LANE_LOOP { uint32_t b; memcpy(&b, &x.l[i], 4); b = *k - b; memcpy(&r.l[i], &b, 4); } return r; }
static inline vxf vx_bits_shr1(vxf x) { vxf r; VX_LANE_LOOP { uint32_t b; memcpy(&b, &x.l[i], 4); b >>= 1; memcpy(&r.l[i], &b, 4); } return r; }

#endif

/* 1 / x and 1 / sqrt(x), x > 0: on both sides the same fused operations from the same bits. */
static inline vxf vx_rcp(vxf x, const float *one)
{
    static const uint32_t K VX_ALIGNED = 0x7EF311C7u;
    vxf y = vx_bits_rsub(x, &K);
    const vxf o = vx_bcast(one);
    for (int i = 0; i < 3; i++)
        y = vx_fma(y, vx_fnma(x, y, o), y);
    return y;
}
static inline vxf vx_sqrt(vxf x, const float *half, const float *three_halves)
{
    static const uint32_t K VX_ALIGNED = 0x5F375A86u;
    vxf y = vx_bits_rsub(vx_bits_shr1(x), &K);
    const vxf h = vx_bcast(half), t = vx_bcast(three_halves);
    const vxf hx = vx_mul(x, h);
    for (int i = 0; i < 3; i++)
        y = vx_mul(y, vx_fnma(hx, vx_mul(y, y), t));
    return vx_mul(x, y);
}

#endif
