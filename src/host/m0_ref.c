/* m0_ref.c: the same M0 kernels through the host lane model of vx.h: the lines the card's vt_m0 must print. */
#include <stdio.h>
#include <string.h>

#include "m0_kern.h"
#include "vt_util.h"

#define N 4096L
static float a[N], b[N], c[N];

int main(void)
{
    uint64_t hash[M0_NOPS];
    m0_vx_selftest(hash);
    for (int i = 0; i < M0_NOPS; i++)
        printf("vxop %-12s %016llx\n", m0_op_name[i], (unsigned long long)hash[i]);
    uint32_t first;
    const uint64_t sub = m0_vx_subnormal(&first);
    printf("subnormal %016llx first_bits %08x\n", (unsigned long long)sub, first);
    uint64_t s = 7;
    for (long i = 0; i < N; i++) {
        a[i] = vt_rand_float(&s, 120u);
        b[i] = vt_rand_float(&s, 120u);
        c[i] = vt_rand_float(&s, 127u);
    }
    m0_fma_vx(c, a, b, N);
    printf("fma_hash %016llx (MVEX)\n", (unsigned long long)vt_fnv(VT_FNV_INIT, c, sizeof c));
    return 0;
}
