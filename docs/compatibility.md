# Compatibility and device-topology report (Milestone 0)

Audit date 2026-11-02, host clock CST. Every line names its command or its raw log. Raw M0 run:
`logs/m0/20261102-205211-card.txt`; host model lines: `logs/m0/host-ref.txt`.

## Revisions inspected

| Tree | Revision | State |
| --- | --- | --- |
| Intel Phi 7120A (`/media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A`) | `aeb1d1c` | clean (`git status --porcelain`) |
| Intel Phi LLVM | `e673241` (S0 to S3 of its plan; S4, S5 partly, S6 not built) | clean; clang 23.1.1 at `~/.cache/intel-phi-llvm-build/llvm/bin` |
| Intel Phi AVX-512 C | HEAD `db092f6` plus 24 uncommitted files | see "57-core assumptions" |

None of the three is modified by this project. They are used as tools only.

## The card, as measured

| Property | Value | Source |
| --- | --- | --- |
| Device | `2f:00.0` 8086:225c, subsystem 7d99, rev 0x20 = 7120A | `scripts/verify-card.sh`, `lspci -nn` |
| Link | 5.0 GT/s x8; the card's maximum is x16; the parent bridge `00:03.1` is x8 | sysfs `current_link_*`, `max_link_*` |
| Logical CPUs | 244 online, 61 cores, 4 threads on all 61 | `vt_m0`, from sysfs `topology/core_id` |
| CPU 0 | core_id 60, with CPUs 241, 242, 243 | `vt_m0` |
| Block pollers | `phiblk0-poll` and `phiblk1-poll` on CPUs 239, 238 = core_id 59 | `vt_m0`, `/proc/PID/stat` field 39 |
| Usable cores | 59 (the two above excluded at run time) | `vt_m0` |
| TSC | 1,238,089,313 to 1,238,089,556 Hz against the card's `CLOCK_MONOTONIC` | `vt_m0` |
| Memory | 15,789,924 kB total; project cap 12 GB = 5,722 pages of 2 MiB | `/proc/meminfo`; user's rule |
| Sensors | hwmon `knc` on the card: `core_mhz`, `in0_input`, `temp*_input`, `power` | `ls /sys/class/hwmon/hwmon0` |

The brief's numbers against this: 1.238 GHz agrees; ARK says 1.24 GHz base and 1.33 GHz turbo, and the card ran
at 1238 MHz. The brief's 352 GB/s is ARK's maximum memory bandwidth. An earlier GEMV on this card reached
177.9 GB/s (Intel-Phi-APL, 2026-11-01). The 2.42 TFLOP figure is a bound nothing here will reach.

## Runtime discovery replaces every core-number assumption

`src/card/topo.c` reads `topology/core_id` for each online CPU in the affinity mask, marks the core holding CPU 0,
finds the pollers by name in `/proc/*/stat`, and hands out disjoint CPU groups. No core number appears in this
project's source. `other_busy_cpus` is recorded before a measurement, so other load on the card is visible.

## 57-core assumptions inherited from the 3120A

**The brief's list does not match the checked-out tree.** `card/vpu/` and `host/asm/phi512/` do not exist in the 7120A
repository: `ls card/vpu` fails. They live in Intel Phi AVX-512 C.

| File (AVX-512 C) | HEAD `db092f6` | Working tree (uncommitted, 24 files) |
| --- | --- | --- |
| `card/vpu/defs.inc` | `NCORES` 57, `MAX_POOL` 227, CPU 0 = core 56 | `NCORES_MAX` 64, `MAX_POOL` 255 (4 x 64 - 1), core count read at start |
| `card/vpu/worker.S` | 57 cores | counted from the online CPUs; falls back to 57 at line 982 if the count fails |
| `card/vpu/rows.S`, `matmul.inc`, `resident.S`, `sched.S` | 57 | adjusted in the same uncommitted change |
| `host/asm/phi512/card.S`, `scripts/phi-vpu.sh`, `phi512.sh`, `gates.sh`, `bench/common.sh` | 114 threads, core 55 | adjusted in the same uncommitted change |

`docs/7120a.md` in the 7120A tree describes HEAD, so it is stale against that working tree. It says the worker "has
not been run" on this card; I have not run it either, and this project does not.

**Consequence: isolated, not fixed.** The tracker does not use that worker (the brief: no transparent AVX-512
translation as the primary compute path). Its own team (`src/card/team.c`) takes its size from `topo.c`.

## Compiler: what exists

**Build recipe, verified.** KNC clang `-c` for every card source, `knc-cc -static` for the link, then the release
`phi-isa-audit`, which must read `0 illegal, 0 suspect`.

| Path tried | Result |
| --- | --- |
| `clang --config=config/knc.cfg` whole-program link | **Fails**: `cannot open crtbeginS.o`, `libclang_rt.builtins.a` (runtimes not built in this LLVM build root) |
| KNC clang `-c` + `knc-cc -static` link | **Works**: `vt_m0` audit `0 illegal, 0 suspect, 69 KNC vector`, runs on the card |
| `knc-cc` from `env.sh` under zsh | **Fails**: `command not found`. The tool shell is zsh. Solved by calling knc-cc by its full path |

**Instruction coverage.** 297 builtins (`BuiltinsX86KNC.td`), 119 of Intel's 595 names in `kncintrin.h` (LLVM repo count,
`scripts/knc-intrinsics.sh`). No gather or scatter builtins, no mask-writing forms. `llvm-mc -mcpu=knc` assembles
`vgatherdps`, `{uint8}`, `{1to16}`, `{cdab}`, `valignd`, so hand-written `.S` needs no `.byte` encoder.

**Known compiler bug.** Inline asm naming `zmm` or `k` registers does not stop the allocator using them across the asm
(Worldgen evidence, 2026-10-22). Rule here: vector asm only in out-of-line `.S`.

**Gather.** `vx_gather` is the one vector instruction with no builtin, so it lives in `src/kern/vx_gather.S`. It first went in as inline assembly naming `zmm30`, `zmm31` and `k1`, against the rule above; every hash matched, which only showed that compile had not been hit. Written there: braces in inline assembly must be `%{%%k1%}`; `jknzd` is refused by the assembler, so the loop is `kortest` then `jnz`; a zmm output constraint fails with "invalid output size" for both `+x` and `+v`.

## Dependency failure: the brief's "compiler-generated vectorized" tier does not exist

- **Expected (brief section 8):** a compiler-generated KNC vectorized version beside scalar and explicit MVEX.
- **Observed (`clang -march=knc -O3 -fvectorize -fslp-vectorize -ffast-math`, and again with `-mllvm -force-vector-width=16`):**
  the axpy loop stays x87 (`flds`, `fmuls`, `fadds`), 0 `vfmadd`, `vmulps`, `vaddps`. A generic `vector_size(64)`
  `y + x * x` loads zmm, stores it to the stack, and computes per lane in x87: slower than plain scalar, with no
  warning.
- **Cause:** stage S6 ("generic vectors, cost model, vectorizers") of the Intel-Phi-LLVM plan is optional and unbuilt.
- **Correction:** the brief says not to modify the compiler unless integration requires it, and it does not. The tiers are
  now **(1) scalar C, (2) intrinsic C** (builtins, instruction selection and scheduling by the compiler), **(3) hand
  `.S`** where measured faster. `vx.h` makes the trap uncompilable on the host: the lane type is a struct, so `a + b`
  does not build.

## Numerical contract, measured

`vx.h` has two bodies: MVEX on the card, a lane model on the host (`fmaf` for FMA). `vt_m0` prints a hash of every
operation's raw bits and `m0_ref` prints the same list. **18 of 18 operations plus the FMA kernel are identical,
card against host model** (`diff`, run 20:52).

- **Caught by this gate:** `vpermd` takes its data from the memory operand, so folding the index table made the *vector*
  the index. First run: 17 of 18 agreed; the fix uses the register form.
- **F32 subnormals: no flush to zero.** The hash over products in the subnormal range matches the model (first result
  bits `8001c732`). Intel-Phi-APL's gate 2.1 had left this open.
- **x87 scalar against MVEX FMA:** 0 of 4096 lanes differ on this input. x87's 64-bit mantissa holds the 48-bit product
  exactly; a difference needs double rounding in the sum. This is one input, not a guarantee.

## M0 speed, card, one pinned thread (CPU 1), nothing else running (`other_busy_cpus 0`), interleaved, median of 9

| Buffers | MVEX cycles/elem (min..max) | x87 cycles/elem (min..max) | Speedup |
| --- | --- | --- | --- |
| 3 x 4 KiB (L1) | 0.993 (0.990..1.001) | 14.272 (14.223..14.432) | **14.37 x** |
| 3 x 16 KiB (L2) | 3.617 (3.613..3.620) | 16.841 (16.791..17.061) | 4.66 x |

Team, 59 cores x 2 threads: 118 of 118 threads on their assigned CPU; 42.0 GFLOP/s on the 16 KiB buffers, 1.8 % of
the 2,338 GFLOP/s arithmetic bound. Per core that equals the one-thread L2 rate, so what limits it is the core's
L1-miss stall: L1 fills from L2 are not prefetched by hardware. That points the kernels at tile sizes that fit L1 and at
`vprefetch0` (measured in M1).

Empty team round, 118 threads: **9.29 us** (11,502 cycles). The first layout, with one shared `fn`/`arg` line rewritten
each round, took 13.06 us. Intel-Phi-APL's tree reads 14.0 to 14.3 us at 118 threads (its own measurement, same
card). A pipeline with many short rounds pays this per round, so stages merge rounds where they can.

## Transport inspected

| Interface | Owner | Use here |
| --- | --- | --- |
| Region `/dev/shm/phi-hostmem` (host), `/dev/phihost` (card, UC) | phictl `--host-mem 4G` | frame slots, control words |
| Region header at card physical `0x10000000` (`PHIR`, address +32, size +40) | phictl | card finds the region's bus address |
| SBOX DMA channels, SBOX + 0xA000 + 0x40 n, DCR at +0x280 | hardware (SSDG 328207-002 2.1.8.2.1; `host/asm/phictl/defs.inc`) | one card-owned channel, opened by this project |
| `phi run`, `phi put`, `phi get` | agent over control socket | program start and log fetch only; a long `phi run` blocks the socket, so the worker starts detached |

Not used: the SSH forward (32 connections for the life of a boot), MPSS, the AVX-512 translation worker.

**Region range chosen:** `[1536 MiB, 1792 MiB)`. Taken already: wrapper below 768 MiB, APL 768 to 1024, Worldgen at 1024,
Cortex at 1344, QwenPU 1472 to 1536, vgd at 1792, card swap from 2 GiB (`grep` of the part offsets across all trees).
The worker refuses to start if `/dev/phiblk1` itself is listed in `/proc/swaps`, in which case the region is swap.

**DMA channel:** 0 and 1 belong to phictl. The rest are claimed historically by projects that are not all running, so the
worker tries an ordered list, takes the first it can lock (`/tmp/phi-cdma<N>.lock`, `flock`), and reports which it got.

**Hugepages:** `nr_hugepages` was 0 at the audit. A run raises it, and restores the old count on exit.

**Fresh code, not copies.** The card-side DMA driver and team are written here from the hardware contract above. Worldgen,
Cortex and APL each carry a `cdma.c`, read for the protocol. A standing rule forbids copying between Phi projects without
asking. Writing fresh breaks nothing and lets the team be topology-aware.

## Other card users

Another project (QwenGPU) is in development on this host, host-side at the time of this audit. Its predecessor's files stay
in the card's `/tmp` (the user kept that checkpoint); this project writes only under `/tmp/vt`. Card reboots are not
performed: they clear `/tmp`.

## Exit criterion

M0's exit criterion, "a validated native scalar and SIMD test can execute on the actual 7120A": **met** on the card,
with the gate above. Native vector execution evidence: audit `69 KNC vector`, output bit-identical to the model, 14.37 x
faster than x87 on the same thread.

Not established by M0: real-time behaviour, DMA throughput, any computer-vision correctness. Those are later milestones.
