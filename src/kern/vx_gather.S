/* vx_gather.S: void vx_gather_knc(const float *base, const int32_t *index, float *out), 16 lanes of each, index and out 64 B aligned: out[i] = base[index[i]].
 *
 * vgatherdps takes the lanes of one cache line per pass and clears their mask bits, so it loops until the mask is empty.
 *
 * Out of line because it must be: this compiler has no gather builtin and no zmm operand for inline assembly, and inline assembly that names zmm or k registers does not stop the allocator using them across it
 * (docs/compatibility.md). Across a call every one of them is the caller's to save, so nothing live can be hit. */
    .text
    .globl vx_gather_knc
    .type vx_gather_knc, @function
    .p2align 4
vx_gather_knc:
    vmovdqa32 (%rsi), %zmm30
    kxnor %k1, %k1
1:
    vgatherdps (%rdi,%zmm30,4), %zmm31{%k1}
    kortest %k1, %k1
    jnz 1b
    vmovaps %zmm31, (%rdx)
    ret
    .size vx_gather_knc, .-vx_gather_knc
    .section .note.GNU-stack,"",@progbits
