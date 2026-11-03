# Measurements

Measured results only. Arithmetic bounds are labelled as such. Every table names its raw data. All card runs: 7120A,
`other_busy_cpus 0` at the start, TSC 1.238 GHz. Card-timed durations may read about 0.48 % short if the host clock is the
true one (the card's clock lag, measured earlier and untraced); this does not affect the ratios.

**What a clone cannot recheck.** `input/calib/`, `logs/live/` and `out/` stay out of the repository: they describe a private room, so frames, depth maps, point clouds, landmarks and live logs are withheld. Numbers citing them are as measured on the rig; everything in `logs/` and `results/` that is in the repository can be read here, and the synthetic scenes rerun from source.

## M1: image primitives (2026-11-02 21:01, card)

Raw: `logs/m1/20261102-210147-*-card.txt`, `results/m1-{640x480,1280x720,1920x1080}.csv`. Command: `scripts/card_run.sh -p 96 vt_m1 W H 1234`.

**Correctness.** For u8-to-Q6, normalize, min/max, Gaussian, pyramid reduce, Sobel x and y, frame difference, at all three
sizes: card MVEX = card x87 = host C = host lane model, bit for bit (hash per kernel, `diff`). Every team run at 1, 2, 3
and 4 threads a core was re-hashed against the single-thread output: identical.

**One pinned thread, median of 7, interleaved (us per frame):**

| Kernel | 640x480 MVEX / x87 | 1280x720 MVEX / x87 | 1920x1080 MVEX / x87 | Speedup at 1080p |
| --- | --- | --- | --- | --- |
| u8 to Q6 | 399 / 4,996 | 1,196 / 14,849 | 2,598 / 33,389 | 12.85 x |
| normalize | 1,132 / 37,708 | 3,284 / 112,919 | 7,387 / 253,812 | 34.36 x |
| Gaussian 5x5 | 1,666 / 23,969 | 4,770 / 73,911 | 16,280 / 169,473 | 10.41 x |
| pyramid reduce | 1,694 / 9,512 | 3,023 / 25,854 | 7,239 / 59,534 | 8.22 x |
| Sobel x + y | 1,570 / 18,340 | 4,479 / 54,880 | 10,084 / 124,969 | 12.39 x |
| frame difference | 1,518 / 4,501 | 4,283 / 13,482 | 9,493 / 30,277 | 3.19 x |

Reading these:

- **normalize's 34 x exceeds the 16 lanes**, and that is not pure SIMD: the scalar form spends its time in x87 `frndint` and two
  compare-and-branch clamps per pixel, where MVEX has one `vrndfxpntps` and two `vgmaxps`/`vgminps` for 16 lanes.
- **frame difference's 3.19 x is memory-bound**: it streams three frame-sized planes through one core, whose rate a scalar
  loop nearly reaches too.
- **An earlier baseline was rigged and was corrected.** A `(int)` cast per pixel on x87 rewrites the control word twice, and
  showed Sobel at 52.7 x and Gaussian at 24 x. The reference now uses exact x87 F32 arithmetic and one `fistp`. Those numbers are
  withdrawn. Raw: `logs/m1/20261102-2059*`.

**A clever kernel, condemned.** A fused pyramid step deinterleaved the vertical sums and filtered at half width: 27 instructions
per 16 output lanes against the plain form's 35. It lost anyway, with prefetch on both (us, one thread):

| Size | Fused + prefetch (min..max) | Plain + prefetch (min..max) |
| --- | --- | --- |
| 640x480 | 1,817.5 (1,749.7..1,828.2) | 1,708.9 (1,660.7..1,726.1) |
| 1280x720 | 3,100.4 (3,066.0..3,164.8) | 3,020.2 (3,013.8..3,071.4) |
| 1920x1080 | 7,258.0 (7,205.8..8,014.2) | 7,228.4 (7,211.9..7,362.8) |

The plain form is the kernel. The fused one is deleted; raw: `logs/m1/` runs before 21:01.

**Prefetch earns its place.** `vprefetch0` per row, reduce, one thread: 640x480 2,170 to 1,698 us (-22 %), 720p 3,384 to 3,030 (-10 %),
1080p 8,968 to 7,252 (-19 %).

**Team, 1920x1080, 59 cores (us, median of 7 [min max]):**

| Threads per core | u8 to Q6 | Gaussian | Reduce |
| --- | --- | --- | --- |
| 1 | 45.5 [43.0 978.3] | 267.5 [263.6 311.5] | 107.7 [102.4 152.1] |
| 2 | 36.4 [34.9 502.1] | 193.4 [192.1 227.1] | 68.9 [68.8 95.1] |
| 3 | 33.8 [33.1 65.4] | 150.6 [147.2 191.4] | 61.9 [60.7 79.4] |
| 4 | 31.3 [30.3 1841.2] | 123.3 [121.8 215.7] | 53.1 [52.8 87.8] |

**These team numbers are an upper bound, not the pipeline's rate.** The gate repeats one frame, so at scale the source sits in each
core's L2 (superlinear steps show it: reduce at 1080p, 2 threads 4,712 us, 96 threads 87.9 us = 53.6 x for 48 x the threads). A real
frame arrives by DMA, which invalidates those lines. The real rate is measured in the transport-backed pipeline (M2 onward).

At 640x480 the round itself dominates: reduce takes 14.7 us at 118 threads, of which an empty round costs 9.29 us (M0).

## Transport (2026-11-02 21:07, card DMA channel 5, driven from `vtrackd`, timed from the desktop host)

Raw: `results/transport-20261102-210659.csv`, `logs/xfer/20261102-210659.txt`. Command: `scripts/vtrackd.sh start 64; out/host/vt_xfer FILE`.
Negotiated link from sysfs: 5.0 GT/s x8 of x16.

| Test | Bytes | Host median us (p99) | MB/s (host-timed) |
| --- | --- | --- | --- |
| Small-message round trip: host word, card poll, card word back | 8 | 3.16 (3.56) | |
| H2C / C2H | 64 | 4.23 / 3.82 | |
| H2C / C2H | 65,536 | 22.17 / 21.70 | 2,956 / 3,020 |
| H2C / C2H, one 640x480 image | 307,200 | 88.35 / 88.11 | 3,477 / 3,487 |
| H2C / C2H, one 1280x720 image | 921,600 | 256.66 / 256.97 | 3,591 / 3,586 |
| H2C / C2H, one 1920x1080 image | 2,073,600 | 573.16 / 574.00 | 3,618 / 3,613 |
| H2C / C2H, 1920x1080 stereo pair | 4,147,200 | 1,144.09 / 1,145.05 | 3,625 / 3,622 |
| H2C / C2H | 8,294,400 | 2,284.86 / 2,287.03 | 3,630 / 3,627 |

Host and card agree on large copies within 1 % (card-timed 3,652 MB/s at 8.3 MB). This is PCIe's rate, not GDDR's or host memory's: arithmetic for
Gen2 x8 is 4,000 MB/s after 8b/10b, so the channel delivers about 91 % of the wire. Both directions are equal.

## M2: the LK kernel (2026-11-02 21:15, card)

Raw: `logs/m2/20261102-211509-640x480-card.txt`. Command: `scripts/card_run.sh -p 128 vt_m2 640 480`. 330 points on a grid, patch 21 x 21, 4 levels, eps 0.005.

Input with exact truth: two images rendered from integers on a grid 4 x finer than the pixels, each pixel the box average of its 4 x 4 cells, so the displacement is exactly a
multiple of 1/4 pixel (`src/kern/m2_input.h`).

| Displacement, px | Status OK | EPE median | EPE p95 | EPE max | MVEX vs x87, max px |
| --- | --- | --- | --- | --- | --- |
| (0, 0) | 330 / 330 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| (0.25, 0) | 330 / 330 | 0.0073 | 0.0138 | 0.0264 | 0.000008 |
| (0, 0.25) | 330 / 330 | 0.0067 | 0.0137 | 0.0254 | 0.000000 |
| (0.5, 0.5) | 330 / 330 | 0.0069 | 0.0162 | 0.0289 | 0.000008 |
| (-0.75, 0.25) | 330 / 330 | 0.0111 | 0.0203 | 0.0417 | 0.000031 |
| (1.25, -1.5) | 330 / 330 | 0.0090 | 0.0181 | 0.0316 | 0.000000 |
| (3.25, 1.75) | 330 / 330 | 0.0115 | 0.0209 | 0.0285 | 0.000015 |
| (-5.5, 4.25) | 330 / 330 | 0.0088 | 0.0184 | 0.0339 | 0.000031 |
| (12.25, -7.5) | 330 / 330 | 0.0089 | 0.0181 | 0.0314 | 0.000031 |
| (-25.25, 16.5) | **303 / 330** | 0.0171 | **45.2494** | **61.6750** | 0.000785 |

**Card MVEX = host lane model, bit for bit,** every field of every estimate (20 of 20 hash and raw-bit lines, `diff`). x87 and MVEX differ by rounding only, at most 0.00003 px up to 12 px of motion.

**The last row is a real failure, and the median hides it.** This pattern's energy sits at 4 and 8 pixel wavelengths, so at level 3 there is nothing to lock onto: **of the 303 points that said OK, 137 were wrong by over 1 px.** The residual test cannot see it, because the wrong place can look alike. The forward-backward check does:

| At (-25.25, 16.5) | Said OK | Of them wrong by over 1 px |
| --- | --- | --- |
| LK alone | 303 | 137 |
| LK + forward-backward, 0.5 px | 171 | 5 |

So it cuts false confidence by 96 % and costs 44 % of the coverage at this displacement, and the 5 left are real. The tracker runs with it on, and it doubles the cost of a feature.

**Speed, us per point** (the (12.25, -7.5) case, 15.8 LK iterations per point):

| Variant | 1 thread | 59 threads (1 per core) | 118 (2) | 177 (3) | 236 (4) |
| --- | --- | --- | --- | --- | --- |
| x87 | 743.57 | | | | |
| MVEX, no patch prefetch | 81.94 | 2.479 | 1.477 | 1.139 | 1.313 |
| MVEX, patch prefetch | 71.31 | 1.653 | 1.211 | 1.184 | 1.282 |

One thread, MVEX over x87 on the same thread: **9.07 x without patch prefetch** (the 2 x 2 solves stay x87 in both). Patch prefetch: 71.31 vs 82.05 us per point, interleaved, median of 7, so it stays. A team hash at every size equals the single-thread one. Best: 3 threads a core by about 2 % over 2; the fourth loses. An early run read 3.70 us at 118 threads because of my work-chunk size of 8 (330 points, 41 chunks, one chunk 776 us); that number is withdrawn.

Every number here repeats memory that the L2 already holds in part (one image pair, repeated). The pipeline numbers below are the real ones.

## The transport-backed pipeline (2026-11-02 21:32 to 21:34, card; 640x480, 150 frames at dataset 30 fps, 2 threads a core, unpaced)

Raw: `results/runs/20261102-2132*`, `2133*`, `2134*` (`.csv` per frame, `.json` summary). Command: `scripts/vtrackd.sh start 512; scripts/scene.sh NAME 640x480`.

**Every run below replayed bit-identically:** `scripts/scene.sh` runs the scene through the card (team, DMA, MVEX), then through `vt_replay` on the host (one thread, the lane model, the same source), and compares a hash of each frame's whole result: 150 of 150 frames equal, 12 of 12 scenes.

| Scene | Observed | EPE median px | EPE p95 px | Centre error median px | 3D measured RMSE m | Filter RMSE m | Velocity RMSE m/s | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| translate | 144 / 144 | 0.0093 | 0.0185 | 0.005 | 0.00075 | 0.00027 | 0.00354 | |
| shapes (disc) | 144 / 144 | 0.0088 | 0.1502 | 0.028 | 0.00063 | 0.00067 | 0.02967 | |
| camera moves | 144 / 144 | 0.0067 | 0.0156 | 0.005 | 0.00072 | 0.00056 | 0.01526 | |
| brightness varies | 144 / 144 | 0.0515 | 0.1442 | 0.260 | 0.00182 | 0.00190 | 0.00261 | 5.5 x the plain EPE |
| noise, sigma 4 grey | 144 / 144 | 0.0572 | 0.1263 | 0.061 | 0.00191 | 0.00140 | 0.01415 | |
| partial occlusion | 144 / 144 | 0.0071 | 0.0156 | 0.019 | 0.00065 | 0.00029 | 0.00308 | |
| full occlusion, 1 s (`loss`) | 54 / 114 visible | 0.0067 | 0.0144 | 0.005 | 0.00076 | 0.00315 | 0.00387 | **FAILS: never reacquired** |
| leaves the view | 144 / 144 | 0.0044 | 0.0101 | 0.016 | 0.00073 | 0.00083 | 0.00758 | |
| helix (depth changes) | 144 / 144 | 0.0088 | 0.0183 | 0.250 | 0.00225 | 0.00252 | 0.02735 | |
| linear3d | 144 / 144 | 0.0087 | 0.0184 | 0.186 | 0.00119 | 0.00126 | 0.00121 | |
| repeat (stripes) | 0: `NO_FEATURES` | | | | | | | **VOID: ambiguity never exercised** |
| flat (no texture) | 0: `NO_FEATURES` | | | | | | | correct: nothing invented |

**Open, and stated plainly:**

- **`loss` does not take the region back.** 0 false observations while hidden (30 hidden frames, 0 said observed), but after the occluder leaves it stays unobserved and goes LOST.
- **`repeat` is void.** 1D stripes have no corners, so the selection fails before stereo runs. The brief's "no high confidence for ambiguous geometry" is **untested**.
- **The covariance is conservative and uncalibrated.** Mean NIS is 0.000 to 0.135, where 3 is expected; `z_pos_rms` 0.02 to 0.88, `z_vel_rms` 0.006 to 0.2, where 1 is expected. **So the innovation gate has never fired and "rejected 0" proves nothing yet.**

Speed from these runs: mono 487 us per frame (pyramid 165, LK 245, region 51), about 1,969 fps; stereo 1,086 us (pyramid 304, LK 402, stereo 248), about 900 fps. **Those rates are one 150-frame burst at 640x480.** Their latency figures (5.8 ms mono, 13 ms stereo) are **queueing artifacts of an unpaced feed, not the pipeline's latency**, and are withdrawn; paced runs are the real ones.

**Superseded since this section was written:** `loss` now takes the region back (template fits inside the region, parabola: NCC 0.988, continuity 1.0, 0.22 px). `repeat` now exercises ambiguity with 2D bumps (145 of 145 reported ambiguous). The gate fires (`glitch`: 10 of 10 rejected, filter unmoved). `scripts/check.sh`: 34 passed, 0 failed, `logs/check/20261102-221101.txt`.

## Live, one camera (2026-11-02, desktop host + card, 1080p camcorder)

Capture to result 17.5 to 18 ms median at 60 requested, about 16 ms of it before the card; DMA 570 us, pyramid about 400 us, flow 262 to 279 us. A 730 x 740 selection cost 41 ms of detection; every 16th frame instead of every frame took it from 7,247 to 103 us on ordinary frames.

## The dual path, with TRUE calibration (2026-11-02 22:21, card + host replay)

`scripts/scene.sh dual 640x480 --dual`, `results/runs/*-dual.{csv,json}`. Rig: 17 to 18 degrees of parallax, focal lengths differ, shutter 11 ms apart.

300 results, 0 dropped, 1,641.9 messages/s unpaced, **replay IDENTICAL 300**. 3D RMSE 11.20 mm at about 4 m, filter 11.81 mm, velocity 7.31 mm/s, reprojection median 0.0275 px. NEES 7.28 against 3, NIS 0.013: not consistent. **This proves the fusion, not a calibration.**

## Three cameras live (2026-11-03 01:08, panel)

| Camera | New fps | Repeats | Bad decodes | Card results/s |
| --- | --- | --- | --- | --- |
| L camcorder 1080p30 | 29.4 | 0 % | 0 | 29.4 |
| R NexiGo 1080p30 | 31.4 | 0 % | 0 | 31.4 |
| D C310 720p15 | 15.7 | 0 % | 0 | 15.7 |

0 bad messages. L capture to result median 29.8 to 32.0 ms at 30 requested, against 17.5 to 18 at 60: unexplained. Requesting 30 removed the capture card's duplicates (50 % at 60).

USB, `lsusb -v`, `journalctl -k`, sysfs `bAlternateSetting`: C310 alt 11 = 3 x 1020 B per microframe, NexiGo alt 6 = 3 x 1024; together 6,132 against USB 2's periodic ceiling of 6,000 (arithmetic). Both claimed their maximum at every rate tried: C310 at 15 fps, NexiGo at 30 fps. `STREAMON` answered `ENOSPC` on one bus. **My prediction that C310 YUYV 720p10 would fit (alt 10, 5,760) was wrong: it got `ENOSPC` too.** The user moved the NexiGo to another controller, which solved it.

## L-R geometry (2026-11-03 01:15 to 01:30, desktop host)

`input/calib/lr-seed.txt` (first pose, void: the cameras were then re-aimed), `input/calib/lr-pose2.txt` (26 points by eye). `out/host/vt_calib ... --report`.

| fb (held near) | fa | rms px | Rotation deg | Baseline m | Wall Z m | TV Z m |
| --- | --- | --- | --- | --- | --- | --- |
| 871 | 1363 | 1.64 | 48.5 | 0.82 | 25 to 215 | 1.66 |
| 983 | 1528 | 1.65 | 52.7 | 1.12 | 5.2 to 6.7 | 1.96 |
| 1082 | 1672 | 1.67 | 56.7 | 1.40 | 3.9 to 4.7 | 2.22 |
| 1281 | 1951 | 1.70 | 64.2 | 2.01 | 3.3 to 3.8 | 2.68 |
| 1582 | 2342 | 1.73 | 74.2 | 2.97 | 3.3 to 3.8 | 3.30 |

Metres from one TV edge of 460 mm, from its EDID, its corners read by eye. Poster corner angles in 3D, rms off 90 degrees: fb 934: 17.5, 983: 6.7, 1032: 3.4, 1082: 2.2, 1131: 2.0, 1181 to 1231: 1.9, 1281: 2.0, 1331 to 1582: 2.3 to 2.4. So a floor near 1080 and no ceiling.

Sign check, independent of the solver: L lies on R's image-right, the user's left, as the user said. The unconstrained solves of the two poses differ: fb 800 and 878 (9.7 %), fa 1264 and 1371 (8.5 %), ratio 1.579 and 1.561 (1.1 %). **Geometry in use: fb 1262, fa 1929, 63.6 degrees, baseline 1.960 m, mid-band, no ground truth.**

## The LED star tracker (2026-11-03 01:29 to 01:42)

Between two 5 s exposures (`snap 150`): L 0.098 px, R 0.010 px, D 0.041 px.

Live, 50 s at 1 Hz from a 16-frame mean, `logs/live/stars-20261103-014127.csv`:

| Camera | Fixes | Lights used | Shift std x, y px | Residual px | GDOP deg/px |
| --- | --- | --- | --- | --- | --- |
| L | 51 of 51 | 1 | 0.067, 0.057 | | 0.040 |
| R | 50 of 50 | 5.0 of 7 | 0.060, 0.047 | 0.248 | 0.086 |
| D | 50 of 50 | 1 | 0.025, 0.032 | | 0.062 |

L's mean offset from its catalogue: +4.484, +2.591 px, steady, nearly radial from the image centre, about 0.9 % outward (one light).

Known true rotations of a real exposure, `vt_stars test`, tilt, pan, roll in degrees:

| Camera | Put in | Recovered | |
| --- | --- | --- | --- |
| R | 0.05, -0.03, 0 | 0.0455, -0.0295, -0.0135 | |
| R | 0.3, -0.2, 0 | 0.2983, -0.1948, +0.0037 | |
| R | 0, 0, 0.5 | -0.0019, +0.0037, +0.4988 | |
| R | 2, 1.5, 0 | 1.9845, 1.5049, -0.0252 | lost in space |
| R | 4, 3, 0 | 3.9351, 3.0273, -0.0785 | lost in space |
| R | 1, 1, 1 | none | **no fix**, earlier build; shift vote cannot follow roll |
| L | 2, 1.5, 0 | 1.9984, 1.5190, 0 (held) | lost in space, one light |
| L | 0, 0, 0.5 | -0.1465, -0.0564, 0 (held) | **false**: one light cannot see roll; earlier build |

Correction algebra, `vt_stars diffcal`: worst miss of a 3D point 0.18 to 1.31 m uncorrected, 2e-15 m corrected. Algebra only: no real camera was turned.

## First live 3D (2026-11-03 01:30, 20 s, `logs/live/track-20261103-013010.csv`)

One large selection in each of L and R. 601 R frames: 44 accepted, 556 rejected. Rays 60.1 degrees apart, missing by 49.0 px mean, gate 25 px. L's region slid to (1113, 804), 255 x 306, off what R's still held. **Accuracy unverified.** The poster (a still target, true velocity zero) could not be run: L went LOST on it in this light, R held 18 features.

## Rig watch (`vt_rigcheck`, frames of known answer)

First design: D false-alarmed at the same 1.19 ratio as a real move on L: rejected. With cells that held still over 8 thumbnails: 15 known-still all still, at most 1.013; 3 known moves: R 1.166 caught, L 1.117 caught only after setting the threshold to 1.06, D 1.00 missed. Live: at least 5 false "D MOVED". Small sample.

## Calibrating from the empty room (2026-11-03 01:57 to 02:14, desktop host)

Exposures: `snap 600` per camera, room empty, `input/calib/sets/pose3-015721-a-{l,r,d}.pgm`. Lights, `vt_stars list`, centroids: lower LED L 1521.57, 745.57; R 579.80, 603.12; D 752.01, 189.52. Upper LED R 575.95, 587.34; D 750.21, 167.77; L read by eye 1521.5, 720. The lower is the brighter in all (flux ratio D 4.4, R 6.3), which identifies the pair across views.

L and R, `input/calib/lr-pose3.txt`, 15 points, `vt_calib --fb F --prior 40 --scale 13 14 0.460 --report`:

| fb | fa | rms px | Rotation deg | Baseline m | Wall Z m | LED Z m | LED pair mm | Armrest mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 910 | 1413 | 1.65 | 47.9 | 0.86 | 17.7 to 37.9 | 16.5 | 914 | 298 |
| 996 | 1548 | 1.65 | 51.8 | 1.12 | 5.6 to 7.0 | 5.9 | 118 | 298 |
| 1096 | 1696 | 1.65 | 56.0 | 1.42 | 4.0 to 4.8 | 4.3 | 69 | 301 |
| 1196 | 1837 | 1.65 | 60.1 | 1.72 | 3.5 to 4.1 | 3.8 | 54 | 304 |
| 1296 | 1972 | 1.66 | 63.9 | 2.03 | 3.3 to 3.9 | 3.6 | 48 | 307 |
| 1396 | 2100 | 1.66 | 67.4 | 2.35 | 3.3 to 3.8 | 3.6 | 44 | 311 |
| 1596 | 2338 | 1.66 | 73.9 | 2.98 | 3.3 to 3.8 | 3.6 | 39 | 321 |

D as arbiter, `tools/arbiter.sh`, fd 1386 given, the room's score (per cent of D's textured pixels confident, the lesser of R and L plus a quarter of the difference) after refinement:

| fb | Score from fix | Refined | Two-view confident | fd after |
| --- | --- | --- | --- | --- |
| 1000 | 24.50 | 28.82 | 19.73 % | 1386.0 |
| 1100 | 23.56 | **28.83** | 17.82 % | **1377.3** |
| 1200 | 21.29 | 24.48 | 14.62 % | 1390.8 |
| 1295 | 19.02 | 21.76 | 12.28 % | 1339.3 |
| 1400 | 16.68 | 19.82 | 11.38 % | 1386.0 |
| 1500 | no fix within the limit | | | |

From fd 1272 instead, fb 1100: 23.12 to 27.64, fd staying 1272.0; fb 1295: 22.17 to 25.19, fd 1246.6. Logitech lists 60 degrees and 4.4 mm, fixed focus, for the C310 (support.logi.com article 360023464573); over the 4:3 sensor that is 1386 px, over the 720p crop 1272 px (arithmetic). Free to move, fd reached 1377.3.

Control: the fix for fb 1100, D rotated about y, `tools/perturb.awk`, two views, before refinement:

| D pose | Textured confident >= 0.2 | Mean confidence |
| --- | --- | --- |
| as fixed | 16.65 % | 0.1073 |
| 1.5 degrees off | 14.16 % | 0.0998 |
| 3 degrees off | 11.14 % | 0.0903 |
| 6 degrees off | 7.06 % | 0.0786 |

**Conclusion, with no ground truth:** poster corners give a floor near fb 1080 (rms off 90 degrees: 3.4 at 1032, 2.2 at 1082, 1.9 to 2.4 above), D gives a ceiling near 1100, and fb 1000 puts the wall 5.6 to 7 m away. `input/calib/current.txt`: fb 1095.8, fa 1695.5, fd 1377.3, 56 degrees, 1.415 m, armrest 301 mm, on a TV edge of 460 mm (from its EDID, `tools/edid.c`). fb 1096 is a 90.2 degree diagonal (arithmetic); NexiGo lists 90 degrees for some models, and this camera's model is not identified.

**What failed before that, all on real frames:**

| Attempt | Result | Why |
| --- | --- | --- |
| D against R by eye, 8 to 11 points | rms 20 px, rotation 100 degrees | a "frame" in D 54 x 57 px could not be R's 95 x 125 px at one depth |
| D against L by eye, 10 points on chair piping | rms 4 to 127 px, points behind cameras, fd 1188 to 1515 | a soft chair from two sides is not reliably one point |
| SIFT + MAGSAC, `tools/vt_match.cpp`, 600-frame exposures | 7 to 15 inliers from 1,194 to 1,657 keypoints | 50 to 60 degrees between views, dim |
| The same with the rotation removed by homography | 369 keypoints left in the canvas, 9 to 13 inliers | most of the frame falls outside D's canvas |
| Map pixels projected into D, D's value against R's, free | D 2.95 m from R, past L; 0.3521 against chance 0.4261 | a small map, nothing holding the pose |
| The same bound to the beacons | 0.4847 against chance 0.4905, 0.5540 against 0.5556, 0.5706 against 0.6024 | no information |
| L and R alone, confidence against fb | rises steadily, 8.38 % at 900 to 20.49 % at 1800 | no peak: packed planes make confidence cheap |

## The depth kernel on real frames (2026-11-03 02:03 to 02:13)

`scripts/card_run.sh -p 128 vt_m3 /tmp/vt/m3-rl.bin 9` against `m3_ref`: R the reference, L the source, exposures of pose 3, fb 1295 at that moment.

**Card `2bbe3861e245d7bb`, host `2bbe3861e245d7bb`.** Median of 9 on the card, 118 threads, 480 x 270, 64 planes, 1 view: normalize 1,090 us, sweep 6,976, aggregate 2,768, decide 2,702, **total 13,536 us**. 33,816 textured pixels, mean confidence 0.0937, 16.70 % at 0.2 or over, 12.4 % of all pixels with depth. Host lane model, one thread, 0.46 s for the process.

**That hash predates the evidence rule, added after a door L never saw got a confident depth.** The current kernel has not been compared with the card, and D as reference with two views has not been timed on the card.

Host, the current geometry, D the reference, two views, 320 x 180: 17.82 % of textured pixels confident. By eye in the map: the armrest one consistent grey, chair back and seat in coherent regions; the rotated control visibly noisier. No per-pixel truth exists for it.

## Card against host on the current kernel (2026-11-03 02:24)

D the reference, R and L the sources, 600-frame exposures of pose 3, `input/calib/current.txt`, 64 planes, 0.8 to 8 m. `out/host/m3_ref ... --views 2 --write`, then `scripts/card_run.sh -p 128 vt_m3 /tmp/vt/m3-d2.bin 9`.

**Card `07b6a82717b7d47d`, host `07b6a82717b7d47d`.** Median of 9 on the card, 118 threads, 320 x 180: normalize 1,313 us, sweep 5,664, aggregate 1,166, decide 1,178, **total 9,321 us**: 14 % of D's 66.7 ms between frames (arithmetic). Host: 8,907 textured pixels, 17.82 % of them confident at 0.2 or over; 5,618 pixels with depth, 9.8 % of all, mean 1.50 m.

Star catalogues from pose 3, `input/calib/sets/pose3-cat-{a,b}-*.pgm`: L 1 light (1521.67, 745.43), R 9, D 2 (752.00, 189.37 and 353.59, 223.72, the second unidentified).

## The depth map, live on the card (2026-11-03 02:39 to 02:52)

`src/card/live_depth.c`, `src/kern/still.c`, 3 cameras, 64 planes, 0.8 to 8 m, 320 x 180. Raw: `logs/live/depth-*.csv`, one line a map, as the card reported it.

**The first run's motion gate was wrong** (`depth-20261103-023925.csv`, empty room): L 1,010 to 1,019 of 1,020 cells "moving", so no averaging at all there; R about 350 to 410, D about 220 of 460. Confident 12.5 to 14.4 %. `vt_mapstat` on a saved map gave the cause: L had 770 of its cells failing the round trip and none failing for lack of texture, so noise itself was supplying the gradients; and its median cell difference, 2.07 grey, was above my fixed 2.0. A failed round trip means a vector is unreliable, not that anything moved. Rules changed, as in [architecture.md](architecture.md).

**After the fix** (`depth-20261103-024243.csv`, 7,775 maps, empty room, this log spanning two restarts of the card program):

| | Value |
| --- | --- |
| Maps | 15.0 a second, every frame of D (D itself dropped to 13.6 fps at one reading on the panel: host capture, not the card) |
| Confident of textured pixels | mean **18.34 %**, max 19.68, min 12.90 at a restart. Offline with 600-frame sums: 17.82 |
| How fast | 14.46 % at the first map, about 18 % by map 101, 7 s (arithmetic) |
| Moving cells, all cameras | 5.2 a map on average |
| Frames in a mean | L about 112 to 126, R 124 to 127, D 113 to 126, of 128 at most |
| Card map | mean 10,362 us live; `vt_m3` alone 9,262 us |
| Card vectors + mean | 2,263 us a frame of all three cameras (`vt_m3`) |
| L capture to result | median 32.33 ms, p95 35.61 (one panel reading; before: 29.8 to 32.0) |

Still-room noise from one saved map (`vt_mapstat`): |frame - mean| median L 1.50 grey, R 0.288, D 0.002 (MJPEG flattens its dark areas); vector length median at the level L 0.348 px, R 0.069, D 0.160; vector over its own sigma, largest L 10.4, R 8.2, D 13.0, of which 12 is the limit and one cell alone never counts.

**Card = host over real frames.** 8 frames of `input/calib/sets/010917`, then the single frame of pose 3, `m3_ref ... --seq ... 8 --write`, then `scripts/card_run.sh -p 256 vt_m3 FILE 9`: `still_hash 18320686bdb23bd4`, `depth_hash 0dd065d27f16e16e`, all nine per-frame lines identical on both. Now a gate in `check.sh`. It failed at first: `tools/celldiff.c` on a dump of every cell found the host counting 128 frames where the card counted 1, at frame 0. `still_create` had left its counts unzeroed; the card's arena comes zeroed and the host's allocator does not. My first guess, a missing fence, changed nothing and was wrong: the card's hash did not move.

**Gather out of line** (`src/kern/vx_gather.S`, [compatibility.md](compatibility.md)): both hashes unchanged; sweep 5,575 to 5,946 us, +6.7 %, one run each, median of 9; total 9,655 us. Live on the final binaries, `depth-20261103-025323.csv`, 1,279 maps: 15.9 maps/s on the panel, confident 18.41 % after the first 8 s, map mean 9,796 us, sweep 6,045, 0.8 moving cells a map; L capture to result median 32.32 ms, p95 32.97 (one panel reading); star fixes L 1 of 1, R 8 of 9, D 2 of 2.

**All gates on the final binaries:** `scripts/check.sh`, `logs/check/20261103-025752.txt`: 36 passed, 0 failed, 0 skipped, including the real-frame gate above and the `dual` scene against its limit of 15 mm, set from the measured 11.20. `make compdb`: 88 files, 0 diagnostics. Audit of every card program: 0 illegal, 0 suspect.

**The gate on a real change of scene:** that sequence's last frame is another pose. Before the whole-picture rule only 332 of 1,020 cells flagged, the noise-relative limit having risen with the median. After it R flags 1,020, but L still 332 and D 242: nine frames with movement in them leave those two's usual noise high enough to stay under it. So the rule works in one camera of three on this test. **Live movement:** see "The rig knocked" below; whether the flagged cells are where things moved has not been checked.

**Against landmarks** (`vt_mapcheck MAP CALIB input/calib/triples-pose3.txt`, the 22 s apart maps 444 and 774). Not independent: the armrest tips and LEDs helped place D. It shows whether the sweep and the triangulation agree.

| Landmark | Rays miss mm | Triangulated z in D, m | Map z, m | Map - tri, m | Pixels of 49 | Mean confidence | tri - eye in D, px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LED lower | 1.5 | 4.283 | 3.113 | **-1.171** | 14 | 0.072 | +6.7, +4.7 |
| LED upper | 0.7 | 4.295 | 1.435 | **-2.861** | 17 | 0.076 | +7.2, +5.3 |
| Armrest A | 7.3 | 1.971 | 1.952 | -0.019 | 8 | 0.141 | +5.2, -5.2 |
| Armrest B | 3.7 | 1.962 | 1.930 | -0.032 | 38 | 0.171 | +1.6, -6.1 |

The armrest from the map's own depths, tip to tip: **303 mm**, against 301 mm from the L-R landmarks. The LEDs fail: a point of light against black has nothing else in its window, and their low confidence is right to be low. So the pane now dims every depth under 0.2.

The offsets in the last column are open: x always positive, y changing sign between the top of the picture and the bottom. Refinement moved D away from these landmarks by this much for the sake of the room's score, about 0.3 degrees (arithmetic). Four points cannot say which of the two is right.

**Repeatability**, maps 444 and 774, an empty room: 1,605 pixels confident in both, 279 in one only; |dz| p50 0.3 mm, p90 1.5, p99 8.2; 100 % within one plane. That is stability, not accuracy.

**The user's verdict (2026-11-03 05:10):** having seen the live result in the room itself, the user judged the geometry correct ("perfect"). That is the one independent check so far and it is by eye: there is still no tape measure of the baseline, and the offsets above stay unexplained. It can only describe the geometry from before the knocks below, the first of which came 11 minutes earlier.

## The rig knocked (2026-11-03 04:59 to 05:30)

`logs/live/depth-20261103-025323.csv`, `stars-20261103-025322.csv`. Cameras R and D were knocked out of position; L was not.

| | Value |
| --- | --- |
| Before, 108,628 maps | confident mean 18.60 %, min 9.13, max 20.06; moving cells about 0.2 per camera per map |
| R | first star fix over 15 px 04:59:36, every fix for 60 s by 05:01:02; first whole-camera restart 04:59:48 |
| D | first whole-camera restart 05:10:30, first star fix over 15 px 05:10:41, every fix for 60 s by 05:17:42 |
| L | one star fix over 15 px at 05:10:47, never for 60 s |
| Newest fixes | R pan -11.0, roll 1.0, tilt 0.07 degrees, 3 lights, residual 1.2 px. D tilt -17.7, pan 6.7, 2 lights, residual 12.8 px, so D's size is doubtful |
| Whole-camera restarts | L 2 maps, R 19, D 276: the first live data for that rule, against one camera of three on the saved frames above |
| Moving cells | up to 130 a map, minute mean; back to 1.4, 3.9, 4.5 in a quiet spell |
| After | confident 5.5 to 8 %, about what a deliberately rotated D read offline (4.41 to 7.16 %) |

The window said so as designed: "GEOMETRY STALE", no 3D numbers. Consistent with stale geometry, not proof of it, and the textured count rose at the same time (8,950 to 13,620 and later 28,947), so D's view or the room's light changed too. No epipolar or motion analysis may use this window: its geometry was stale.

## The window's freezes (2026-11-03 05:39 to 05:47)

Reported by the user as periodic freezes. Measured without touching the process: each thread's run time from `/proc/PID/task/TID/schedstat`, sampled about every millisecond, one run each way.

| | Draw thread, before | Draw thread, after |
| --- | --- | --- |
| Runs over 16 ms without a break | 44 in 44 s | **0 in 30 s** |
| Their length | 33 to 40 ms, median 36 to 37 | |
| Their period | 1.00 s | |
| Blocked over 30 ms | 0 | 0 |
| CPU | 23.8 to 24.0 % | 22.7 % |

**Cause:** `star_tick` (16 whole frames summed per camera) and `rig_tick` ran on the draw thread once a second, so 2 or 3 displayed frames were lost at every tick. They now run on `nav_main`, which carries the same work (median 33 ms, max 38, period 1.00 s), and the window reads a snapshot published under the mutex. `draw_trail` read the geometry unlocked, safe only while one thread did everything; it now reads the snapshot. Card programs unchanged by hash.

**The cameras and the card were not stalling:** over 148,921 maps (2.8 h), no stall of 133 ms or more on L or R; 3 maps got no L frame and 0 no R frame. No USB events in the kernel log in 60 minutes. Tracking on the card never went through the draw thread, so whether tracking itself improves is not measured.

## Constant attitude from the picture (2026-11-03 05:52 to 06:30)

`src/kern/attitude.{h,c}`, original C, one source for card and host. `out/host/vt_attitude`; live: `logs/live/attitude-*.csv`. Each camera against its own reference picture: normalized correlation over shifts, then Gauss-Newton on the rotation with a Tukey kernel. No lock on any region or light, no still rig, no empty room.

**Known rotations of a second real exposure** (independent noise, same light, someone in view), `vt_attitude --test CALIB A CAM T P R B`:

| Camera | Put in, tilt pan roll | Recovered | Miss deg | Correlation gain |
| --- | --- | --- | --- | --- |
| R | 0 0 0 | held | 0.000 | 0.000 |
| R | 0.1 0 0 | 0.106 -0.001 -0.002 | 0.006 | 0.062 |
| R | 0 0.2 0 | 0.006 0.201 0.001 | 0.006 | 0.197 |
| R | 0 0.5 0 | 0.006 0.499 -0.002 | 0.007 | 0.596 |
| R | 1 -2 0.5 | 1.007 -1.999 0.498 | 0.007 | 0.883 |
| R | 0 -9 0 | 0.008 -8.997 -0.004 | 0.009 | 0.871 |
| L | 0 0.5 0 | 0.007 0.504 0.003 | 0.008 | 0.660 |
| D | 0 0.2 0 | -0.008 0.194 0.024 | 0.026 | 0.192 |
| D | 0.5 0 0 | 0.496 -0.009 0.016 | 0.019 | 0.404 |
| D | 3 -2 1 | 2.992 -2.013 1.025 | 0.029 | 0.563 |

The gain the 0.05 and 0.1 degree trials showed before the final gate (0.016 to 0.072) now falls under it, so those are held: **turns of about 0.1 degree and under go uncorrected.** The turn itself is warped in software: this tests the arithmetic against real pictures, not a camera really being knocked.

**False turns, live on D, someone close in view:** without a gate, up to 0.33 degrees, largest gain 0.053 of 68 measures, and a still camera's fit jittered up to 0.08 degrees and walked. Hence the gate at a gain of 0.10 and at 0.1 degree. With it, about a minute live: 0 turns on all three, gains at most 0.021, card 2.6 to 3.6 ms a measurement, once a second per camera, correlation L 0.71, R 0.80, D 0.55.

**Failures on the way:**

| What | Result | What it taught |
| --- | --- | --- |
| Night reference against morning frames, someone in view | L 0.03 deg, correct. D a false 4.4 deg at 0.038 where no turn scored 0.100. R -10.06, correlation 0.134 | a reference must share the light; a turn must beat no turn |
| Map's confidence as the judge, D-L, neither moved | as solved 16.38 %, 1.5 deg off 16.66, 3 deg off 18.35, 6 deg off 17.82, -3 deg 16.53 | **with someone in view the map's score says nothing about geometry**; 18.60 % held only for the empty room |
| The star fix as the yardstick | R -11.31 where the LED's own pixels give -9.51 exactly (arithmetic): a small-angle solver, tested to 4 degrees | not a reference beyond a few degrees |
| The live star tracker on a still rig | L 33, 77 px, R 109, 23 px, D lost; the same pixels with no memory L 0.46, -0.78, R -0.07, 0.18, D 2.2, 0.08, and the kernel held all three | **each fix searched around the last: a lock. A hidden light walked it off and it called that a move.** Log only now, no memory |

## Do L and R agree through D? (2026-11-03 06:27 to 06:37)

`out/host/vt_agree`, on saved exposures, the lane model. For each of D's pixels the depth is found from D and R alone, and again from D and L alone. They share only D's picture, so they agree only if both really are where the geometry says relative to D, and it is the same question whoever is in the room. Differences are in steps of 0.0179 per metre, one of the original 64 planes. Chance: the same depths paired with unrelated pixels.

| Frames, planes | Sure from each alone | Median steps apart | Within 1 step | Chance | Within 2, of all textured |
| --- | --- | --- | --- | --- | --- |
| Empty room, 64 | 909 of 8,907 | 1.3 | 39.9 % | 16.8 % | 519 |
| Empty room, 256 | 1,254 | 1.0 | 50.1 % | 21.2 % | 845 |
| Empty room, 256, 11 x 11 window | 1,168 | 0.8 | 59.2 % | 30.5 % | 947 |
| Morning, someone in view, 64 | 1,016 of 31,876 | 14.4 | 7.2 % | 3.2 % | 131 |
| Morning, 256 | 1,400 | 16.4 | 5.6 % | 3.1 % | 202 |

**The map has only ever worked on one near object.** By region of D's picture, in the empty room the pixels that agree are the near chair, at 1.92 against 1.95 m; the far wall at the top reads 0.83 against 4.04 m and 1.29 against 4.42 m. Every good number before this section came from that one object. With someone in view, nothing agrees much anywhere: the map today is mostly speckle.

**Three causes found, none yet the whole answer:**

| Cause | Evidence | State |
| --- | --- | --- |
| Too few planes. D is 0.83 and 0.77 m from R and L, so 0.8 to 8 m spans up to 255 and 367 pixels at level 2: 64 planes is 4.05 and 5.82 pixels a plane, past the true match | 519 to 845 agreed pixels at 256 planes, empty room | not live: 256 planes would add about 28 ms to each map (arithmetic), which the budget below does not have |
| "A different depth" was 1.5 planes, whatever the count, so more planes read as more doubt | 512 planes: 2,517 sure, 5,870 after, same frames | fixed; unchanged at 64 planes, hash the same |
| D's pose. Moved until R and L agree, by coordinate descent on that count: 177 to 389 on its own exposure, **80 to 162 on one held back**, within 1 step 8.9 % against 3.5 % chance. fd 1363.5, about 1.2 degrees | real, and small | not installed: too little agreement to trust. The same search on R gained nothing held back (80 to 86) |

**R's re-anchored pose cannot be judged on these frames:** the empty-room pixels that make a known-depth yardstick are on the chair, which has moved, so even D+L, which did not move, misses them.

## Optical flow resolution, against the real card (2026-11-03 06:44 to 06:57)

The user's direction: double it until the frame rate drops. Flow is the still pictures' vectors, at a pyramid level set at run time (`flow N` to `out/run/view.cmd`, `--flow N`). The map and the attitude kernel stay at level 2, reduced from it. `logs/live/depth-*.csv`; rates from each camera's frame counter over the run; latency and drops from the panel, one reading each.

| Flow | Cells, L and R / D | Over | Frames/s L, R, D | Worst 5 s window | Maps/s | Vectors L, us | "Mean" L, us | Capture to result, median, p95 ms | Dropped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Level 2, 480 x 270 | 1,020 / 460 | 25 s | 31.0, 31.0, 15.6 | | 15.6 | 399 | 508 | 33.2, 36.7 | 0 |
| Level 1, 960 x 540 | 4,080 / 1,800 | 25 s | 30.3, 30.4, 15.2 | | 15.2 | 1,448 | 1,648 | 35.4, 36.0 | 0 |
| **Level 0, native** | **16,200 / 7,200** | **75 s** | **30.2, 30.2, 15.1** | **29.6, 29.8, 15.0** | **15.1** | **4,515** | **5,825** | **47.5, 51.4** | **0** |
| Native, cells half as tall | 32,400 / 14,400 | 38 s | **16.8, 25.7, 7.3** | 12.8, 20.2, 4.2 | **7.3** | 8,285 | 11,101 | **371.9, 382.3** | **1,651** |

The rates over 25 s read a little high: wall time in that log is whole seconds. **Native holds its rate and the next doubling collapses.** Level 0 is about 873 ms of the session's thread a second, level 1 about 367 (arithmetic: 30 x 10.3 x 2, and 15 x (5.3 + 10.5) for D and its map; 30 x 3.1 x 2, and 15 x (1.6 + 10.5)).

**The user then halved it, native being laggy to use: level 1 is the default.** Measured, 55 s: worst 5 s windows 29.8, 29.8, 15.0, maps 15.0, 0 dropped.

**Then tracking was put first,** the user's direction. Each result used to leave after that frame's flow and map, and the window shows a frame only when its result returns, so that work delayed the picture. Now the result leaves at once, and flow and the map run only while no frame is waiting, on each camera's newest frame. One panel reading each:

| Flow | Frames/s L, R, D, worst 5 s | Capture to result, median, p95 ms, before | After |
| --- | --- | --- | --- |
| Level 1 | 29.8, 29.8, 15.0 | 36.0, 37.8 | **33.9, 34.6** |
| Native | 29.8, 29.8, 15.0 | 47.5, 51.4 | **36.7, 44.3** |

A frame that lands during a job still waits for that job, up to about 10.5 ms behind a map. **That flow gives way under overload rather than the tracker is by design and has not been seen:** neither row is overloaded. On screen, the flow vectors were drawn over the tracker's marks; they now go under them, and are not drawn at all while a target is being dragged.

**That ceiling may be this code rather than the card.** "Mean" nearly doubled when the cells doubled with the same pixels, so about 5.3 of its 5.8 ms scale with cells and 0.5 with pixels (arithmetic). That points at `still_decide`, which runs on one thread. It has not been timed apart, or moved onto the team.

**The motion gate is too quick at finer levels.** R's cells called moving: 3.5 % at level 2, 10.4 % at level 1, 22.8 % at native, with someone in view. Scaling `motion_px` with the level did not help (still 21 to 23 %), so a vector is not what triggers it; R's averaging falls to about 76 frames of 128. Unsolved. A panel reading of D's rate at native showed 13.8 against 15.1 from the counters: unexplained.

**Found on the way:** `vtrackd.sh stop` waited a fixed second for the card program to end. Under the overloaded test it took longer, the new binary was refused as "Text file busy", and `live.sh` exited, leaving the old window and program running, whose log then looked like a measurement of the new build. It now waits for the program to end, and a failed start no longer ends `live.sh`.

**All gates:** `scripts/check.sh`, `logs/check/20261103-070644.txt` on the build with tracking first: 36 passed, 0 failed, 0 skipped, with the attitude kernel over real frames card = host. The gate runs the flow at level 2, as `vt_m3` always has; native flow on the card is measured live, as above, not by a hash.

**R re-anchored after the knock**, once, by landmarks carried from the old R picture to the new by correlation (`--transfer`, `--solve-r`): wall points correlate 0.66 to 0.985 from night to day. Tilt -0.274, pan -9.222, roll -0.369 degrees, 2.19 px rms over 9 points at 4.0 to 4.8 m. The landmarks then land 14.9 mm from where they were, which restates the residual and is not independent; near objects are untested, and any move of R's centre is unmeasured. D was not moved at all: its landmarks are where they were.

## Holding a selected region (2026-11-03 07:30 to 08:10, card; scenes 640x480, 150 frames; raw `results/runs/`)

**The old tracker on a box like an operator's.** Scene `bigbox` (box 2.4 times the object, background as busy as the object): centre error median 74.376 px, p95 176.775, max 187.975, continuity 1.0000, 144 of 144 frames reading OBSERVED, frozen at x 273.9 while the truth walked from 231.5 at selection to about 462 (arithmetic). Nothing had shown this: every earlier scene selected from inside the object.

**Why, in order found:**

| Cause | Measured |
| --- | --- |
| One corner threshold for the box: the Harris response goes as the gradient to the fourth power, so the outline's corners erase the inside's | 11 corners in 384 x 307. Those straddle two motions: the region moved 1.31 px a frame where truth moved 1.60, EPE 0.30 px |
| A median over all features: the background's motion when background is the majority | the frozen run above |
| Detection on the tracking thread | 6.1 to 8.0 ms each on 384 x 307, on every frame while features were low (frames 5, 6, 8, 24, 38, 40, 56 of that run, among others). About 24 to 30 ms on the largest box an operator has drawn, 924 x 484 (arithmetic, by area) |
| Search for a lost region counted in pixels of its own level | 288 px of reach at 1/8 (arithmetic); a 160 px jump, live |
| Reseeding counted riders, every frame | features 13, 126, 15, 128 and a detection every second frame, live, 433 x 537 box |

**The new region model** ([architecture.md](architecture.md)). Card, replay against the host identical on every line:

| Scene | What it provokes | Centre median | p95 | max | Continuity | EPE median |
| --- | --- | --- | --- | --- | --- | --- |
| bigbox | box 2.4 x the object | 0.053 px | 0.076 | 0.368 | 1.0000 | 0.0067 px |
| smallbox | box 4 x a smaller object | 0.270 | 0.326 | 4.736 | 1.0000 | 0.0071 |
| stillstart | still 2 s after selection, then moves | 0.053 | 0.072 | 0.218 | 1.0000 | 0.0027 |
| rotate | turning 30 degrees a second | 0.388 | 0.608 | 2.595 | 1.0000 | 0.0417 |
| swing | to and fro, up to 19 px a frame | 0.002 | 0.006 | 0.015 | 1.0000 | 0.0052 |
| hold | still under noise of sigma 4 (the old tracker, over 450 frames: 0.076, 0.099, 0.117) | 0.049 | 0.065 | 0.073 | 1.0000 | 0.0589 |

**These centres flatter the method by one thing:** each scene's box is centred on the object, and the region keeps the operator's centre when it lies on the thing, so the centre starts exactly right. What they measure is that it stays there. An off-centre drag keeps its offset by design.

**They also spare it:** no motion blur, no change of shape, no change of light on the object.

**Wrong turns, with their numbers.** New features voting at a quarter weight and on motion remembered over 8 frames: still frozen, 64.9 px. Holding them out of the vote by age, with a fall back to all when too few: died at frame 47, continuity 0.29. A second thing found as a clump of claims under the old model: a turning object has no clump, and the region stayed on the background, 144.6 px. Moving the centre to the middle of the survivors on closing: 24 to 43 px on `rotate`, for throwing away the point that was selected.

**Detection on the team:** judging corners by neighbourhood, with the selection's two passes, cost 26.37 ms on one thread, and 3.78 ms on the team, 7.0 times, same box, one run each. Against the old 6.25 ms on that frame, which found 11 corners where this finds 128 (`results/runs/20261103-071854-bigbox.csv`, column `us_region`).

**All gates:** `scripts/check.sh`, `logs/check/20261103-080347.txt`: 42 passed, 0 failed, 0 skipped, with the six new scenes among them (`results/runs/20261103-0806*`). The old tracker's frozen run is `results/runs/20261103-071854-bigbox.json`.

**A still target, live, the same two posters as before, 42 s** (`logs/live/track-20261103-071056.csv` before, `logs/live/track-20261103-075213.csv` after, one run each):

| Camera | Results observed, before and after | Centre std before | after | First to last fifth before | after |
| --- | --- | --- | --- | --- | --- |
| R | 1,290, 1,261 | 0.13, 0.21 px | 0.059, 0.058 px | not recorded | +0.145, +0.049 px |
| L | 1,287, 1,260 | 6.31, 4.02 px | 0.518, 0.172 px | +21.5, +13.1 px over 39 s | +1.340, +0.456 px |

**L still creeps,** +3.4, +3.0 px over 126 s of the same run, steadily, where R wanders within 0.27 px with no direction. An earlier statement in the project's notes that L's error was a random walk was wrong: the first run's numbers show a steady crawl. An LED in L's view moved -0.6, +0.7 px over the same stretch (`logs/live/attitude-20261103-074057.csv`), so at most a small part of it is the whole picture moving. Flow bias accumulating frame to frame and real change on the target are not yet told apart.

## Generated frames (2026-11-03 08:15 to 08:50, card; 640x480 unless said; replay identical on every line)

**Centre error on the real frames, px, median and p95, generated frames off and on.** `results/runs/`, the last run of each.

| Scene | Off | On |
| --- | --- | --- |
| bigbox | 0.223, 0.301 | 0.084, 0.102 |
| smallbox | 0.301, 0.693 | 0.296, 0.390 |
| stillstart | 0.079, 0.140 | 0.059, 0.080 |
| rotate | 0.363, 0.680 | 0.203, 0.471 |
| hold | 0.018, 0.030 | 0.017, 0.028 |
| swing | 0.003, 0.006 | 0.016, 0.034: **worse** |

One run each, on a deterministic scene: there is no spread to give, and none to hide behind.

**Cost per generated frame.** 1.68 to 1.93 ms at 640x480. At 1080p, `scripts/scene.sh bigbox 1920x1080 --frames 40 --framegen`: 5.95 ms median, 6.57 max. Stage means from the card's log (`scripts/vtrackd.sh log`), us:

| Run | More pyramid | Scene | Search | Filter | Upscale | Middle | Blend | Pyramid | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scene, 1080p, 34 frames | 117 | 423 | 2,787 | 648 | 394 | 246 | 990 | 372 | 5,977 |
| Live L, 1080p, 144 frames | 72 | 409 | 2,785 | 640 | 408 | 595 | 1,718 | 404 | 6,631 |
| Live R, 1080p, 144 frames | 71 | 414 | 2,774 | 638 | 415 | 437 | 1,440 | 410 | 6,109 |
| Live D, 720p, 71 frames | 38 | 206 | 1,512 | 381 | 248 | 233 | 687 | 300 | 3,105 |

My estimate from area before measuring 1080p was 13 ms. It was more than twice the measurement. Those live lines predate the early out on still blocks, whose effect at 640x480 on a busy scene was nil and whose effect live is not measured.

**The rates, live.** `logs/live/track-20261103-083232.csv`, windows of 5 s with a region held on all three cameras: L 150 real and 150 generated results, R 150 and 150, D 75 and up to 75. That is 60, 60 and up to 30 a second, with the real frames at their full rate. These are results the tracker produced, not frames in the window. Observed: L and R on all but 0 to 2 real frames of those windows, on averages of 16 to 75 features, riders included. There is no like-for-like earlier figure: the log before this one held only results with a 3D flag. D read unobserved on 22 to 49 of 75 in its worst windows, and generated nothing while it did.

## Losing and taking back (2026-11-03 08:35 to 08:50, card)

The old rule ended a region after 2 s out of sight, its appearance erased. Measured on scene `comeback` (`--frames 240`: out of the picture for 3 s, back from the other side):

| Build | Taken back | Error | Best match while away |
| --- | --- | --- | --- |
| The old rule | never: it had no memory left | | |
| Kept, one memory holding background | frame 210, when the object reached the exact spot it was selected at, correlation 1.000 | 0.109 px | up to 0.52 |
| The box closes on a moving region, half-pixel scoring | **frame 182**, the frame the object is 95 % visible, correlation 0.995 | 0.605 px | up to 0.44 |

The region ran out of patience, read LOST for 101 results, and kept its track id. The first row of the table was not run: it is the code that was there. 

**Retracted, on the way.** A gap of 1.57 s, a jump of 164 px and R's rate falling to 68 to 112 per 5 s, read from `logs/live/track-20261103-080810.csv` as a stall, were not. That log wrote only results with a 3D flag, so the gaps meant no 3D result. It now writes every result of a region.

**All gates on this build:** `scripts/check.sh`, `logs/check/20261103-084917.txt`: 54 passed, 0 failed, 0 skipped, with `comeback` and eleven runs with generated frames among them, card = host on every result.
