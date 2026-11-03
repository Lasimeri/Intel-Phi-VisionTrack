# Architecture

A single 7120A does the numeric work; the desktop host acquires, transports and displays. "Lock-on" here means visual continuity of a region an operator chose: an anonymous image region with no class. No weapon interface, fire control, ballistics, aiming output or recognition exists.

## The two programs

| Program | Runs on | Does |
| --- | --- | --- |
| `vtrackd` | card, one long-lived process, started detached | sessions: teams, device-resident buffers, DMA, the whole tracker |
| `vt_run`, `vt_xfer`, `vt_replay`, viewer | desktop host | frames, transport control, scoring against truth, display |

An idle `vtrackd` is one thread napping on one word: its team is destroyed between sessions, so it does not hold the card at full power.

## One frame

| Stage | Where | Measured at 640x480 mono (us, median) |
| --- | --- | --- |
| Acquire, stamp, copy to a slot of the shared region | host | |
| Pull the slot into device memory by DMA | card, ingest thread | 86 |
| u8 to Q6, optional stretch and blur, pyramid | card, team | 165 |
| LK with forward-backward check, per feature | card, team | 245 |
| Region state machine, detection when needed | card, tid 0 | 51 |
| Stereo correspondence and triangulation | card, team | 0 in mono; 248 in stereo |
| Kalman | card, tid 0 | under 1 in mono; 22 in stereo |
| Result to host by DMA, status word after it | card | |

Ingest runs on a free hardware thread of a reserved core, found at run time, so frame transfer overlaps computation.

## Decisions, and what each rests on

| Decision | Why | Evidence |
| --- | --- | --- |
| Planes hold integer-valued F32 in Q6 | blurs and gradients are exact below 2^24, so MVEX = x87 = SSE bit for bit, and LK needs no conversion | M1 hashes |
| All scalar pipeline arithmetic is `long double` | x87 on both machines; any plain float expression broke replay at the first inexact sum | replay failed from frame 6, then passed |
| One tracker source for card and host | whole-run tracks must match bit for bit | 12 of 12 scenes |
| Planes read with `valignd`, patches with `vloadunpack` | no gather: a patch shares one (u, v), so bilinear weights are constants | M2 |
| Fused pyramid kernel removed | it lost to the plain form at every size | M1 |
| LK solves in x87, with a cached inverse | 2 x 2 per level; vectorizing it would gain nothing | M2 |
| Kalman on one thread | six states; it takes 22 us | pipeline runs |
| Fresh DMA driver and team | a standing rule forbids copying between Phi projects unasked; the team needed per-core fan-out anyway | M0: 9.29 us against 14.0 |

## Three cameras (user's direction, 2026-11-03)

| Camera | Where | Mode | Role |
| --- | --- | --- | --- |
| 0, L: camcorder via capture card | the user's left | 1080p30 MJPEG, USB 3 | primary 3D view |
| 1, R: NexiGo | the user's right | 1080p30 MJPEG, USB 2, own controller | primary 3D view |
| 2, D: Logitech C310 | centre | 720p15 MJPEG, USB 2, own controller | depth map view; no vote in the 3D point |

Each camera is one tracker on the card, sharing the team, frames interleaved in the order they land. `fuse.c` intersects L's and R's rays and holds one Kalman state. Devices are bound by their `/dev/v4l/by-id` paths, named in the local `input/calib/cameras.txt` because those paths carry serial numbers: a replug renumbers `/dev/videoN`, and left and right must not swap. The NexiGo and C310 cannot share a USB 2 controller in MJPEG (they claim alt 6 and alt 11 at any frame rate: `ENOSPC`).

## Navigation by fixed lights (user's direction, 2026-11-03)

The room's LEDs are treated as a star tracker treats stars. `src/host/stars.{h,c}`, on the desktop host, a fix per camera per second from a 16-frame mean.

| Orbital or GPS method | Here |
| --- | --- |
| Star catalogue, ephemeris | `input/calib/stars-{l,r,d}.txt`: lights repeating between two long exposures within 0.5 px; no similar neighbour within 40 px, as catalogues drop close doubles; flux above 5 % of the brightest |
| Recursive star identification | each light looked for within 6 px of where the last fix puts it |
| Lost-in-space identification | every light in the frame votes for the shift explaining it against the catalogue. Shifts only, so not as general as the method it is named after |
| Wahba's problem | small-angle least squares on the rotation from catalogue bearings to measured ones. Not QUEST: enough for a bump, and tested to 4 degrees |
| RAIM | the worst light out per round while any residual exceeds 1.5 px, never below 2 lights |
| GDOP | degrees of attitude per pixel of centroid noise, on the panel |
| Differential correction | with at least 3 agreeing lights and residual under 0.7 px, R' = Ob R Oa', T' = Ob T, pushed to the card. Fewer lights only raise an alarm: a blurred LED's offset looked exactly like a 0.12 degree pan |
| Coasting | a hidden light keeps the last attitude: in view "most of the time", the user's words |

A camera's translation is invisible to this, so beyond 5 degrees or 15 px the geometry is declared stale and no 3D is shown. References: Spratling and Mortari 2009, "A Survey on Star Identification Algorithms", mdpi.com/1999-4893/2/1/93; Shuster on QUEST, malcolmdshuster.com/Pubp_010_031z_J_JAS1290_cquest_MDS.pdf.

**What the design needs next:** lights spread in view of L and R. Today they share one bright LED and its faint companion 45 mm apart, which is one bearing.

## How the geometry is found, and how it is not

After the method Nintendo describes for the 3DS, per unit stored rotation, translation, angle of view and camera distance, reached here from the images because nothing was mounted parallel. All of it offline on the desktop host, from a still room and long exposures, one pose.

| Step | Tool | What it does |
| --- | --- | --- |
| Exposures | `snap 600` to `out/run/view.cmd` | N distinct frames summed per camera, 16 bits |
| L and R | `vt_gridpng`, `vt_calib` | landmarks read on gridded crops, in the true overlap only, LEDs by centroid; essential matrix from a grid of focal lengths, then Levenberg-Marquardt with Huber weights. Scale from one known length |
| D's fix | `vt_align --fix` | GPS-style resection: at each candidate position Davenport's q method gives the attitude from bearings to the LEDs and armrest tips; solved on continuously. The result is a valley of poses, not a point |
| D's refinement | `m3_ref --refine` | coordinate descent on D's six degrees of freedom and focal length, scored by the depth kernel itself: how much of D's textured picture each of R and L agrees with at some depth, the lesser of the two ruling |
| The focal band | `tools/arbiter.sh` | repeats that for each fb: D, an independent third view, judges what reprojection cannot |
| Controls | `tools/rect.awk`, `tools/perturb.awk` | poster corners at 90 degrees in 3D; D deliberately rotated, which must lose |

**Reprojection cannot judge this.** It is flat across a factor of two in focal length while the rear wall moves from 215 m to 3.3 m. So physical tests judge it: wall depth and flatness (`--report`), poster corners at 90 degrees, and D's agreement. The first version of this reasoning concluded fb about 771 from two rulers agreeing, on geometry that put the rear wall 25 to 215 m away: retracted.

**Geometry in use, `input/calib/current.txt`:** fb 1095.8, fa 1695.5, fd 1377.3 px at full size; L to R rotation 56 degrees, baseline 1.415 m on a 460 mm TV edge. The user judged it correct by eye; there is no measured ground truth. What failed on the way, with numbers: [measurements.md](measurements.md).

## The depth map

`src/kern/depth.{h,c}`, one source for card and host. Plane sweep with D as reference, at pyramid level 2 of each view: D 320 x 180, R and L 480 x 270.

| Stage | How |
| --- | --- |
| Normalize | each pixel's distance from its 9 x 9 mean, in units of that window's spread: exposures differ by a factor of several. Vector sums, x87 for the root and division |
| Sweep | 64 planes uniform in inverse depth. For each 16 lanes the homography is computed exactly at both ends, one x87 division each, straight between (FMA). Round, clamp, address in F32 (exact below 2^24), then `vgatherdps`. Truncated absolute difference, weighted per view |
| Aggregate | a box, by vertical sums and `vloadunpack` horizontally |
| Decide | winner per lane by masks, then a second pass for the best plane more than one from it, the winner's neighbours, and how many planes were real evidence. x87 per pixel for parabola, confidence, depth |

No depth without texture, a clearly best match, and at least half the planes inside the other view: without that rule a door L never saw received a confident depth.

**Live, on the card (`src/card/live_depth.c`).** In the three-camera session, on the session's own team, between frames. Every frame of every camera updates that camera's still picture; every frame of D then normalizes the three still pictures, sweeps, aggregates and decides. z, confidence, each camera's cells and last a head go by DMA to one of two 1 MiB buffers in the free [2 MiB, 4 MiB) of the region part, and the head's DMA status writes `depth_pub`, so the word that names a map comes after every byte of it on the same channel. The depth copy goes before that frame's result, so the wait for the result also protects the map's buffers. The relation of the views comes from one function, `depth_views_from_calib`, used by this and by `m3_ref`, recomputed at every `calib_seq`, so the star fixes carry the map too.

`vt_view` draws it beside D (`src/host/depthview.c`): nearest warm, furthest cold, no depth black, confidence below 0.2 at a quarter brightness, because measured, those depths can be metres wrong. Also from above in the panel, one line a map in `logs/live/depth-*.csv`, and on the order `map`, `out/run/map.{bin,pgm,ply}`.

## The card's vector data (the user's direction, 2026-11-03)

LK flow is what the card measures best, and motion, unlike appearance, does not change with a 50 degree turn of view. Every correspondence failure in calibration was appearance changing.

**Built: the still pictures (`src/kern/still.{h,c}`).** One frame of this dim room is mostly noise, and noise biases a truncated absolute difference: the 600-frame sums gave 17.82 % against 12.72 % from one frame. So each camera keeps a running mean of its level 2 plane, and the card's vectors decide what enters it.

| Part | How |
| --- | --- |
| Cell | 16 x 8 pixels of the level: one vector register across, so one weight per load. 1,020 cells in L and R, 460 in D |
| Vector | LK from that camera's **mean** to this frame, one level, 15 x 15 patch, with a round trip |
| Why the mean | a slow mover adds up against it instead of hiding inside each frame's noise; a cell that moved restarts there, so its next vector is frame to frame; the still cells' mean vector is the camera's drift against its own past, which does not accumulate |
| Moved | a vector longer than 0.25 pixels **and** 12 of LK's own sigmas; LK's residual failing; or the cell's mean difference above 2 grey levels and 3 times this camera's median. At least 2 such cells in a 3 x 3 for any to count. A round-trip failure is not evidence: noise alone fails it |
| Everything moved | a camera's median difference above 3 times its usual median: all its cells restart. Otherwise a camera knocked or a light switched on would raise its own limit and go unseen |
| Mean | weight 1 / min(count, 128) per cell, one FMA per 16 lanes; the first frame in a cell is that frame exactly |

The choice over the design first written here, averaging the cost volume: three small planes against 15 MB at 15 Hz, it uses all of L's and R's 30 frames a second, it removes noise before the difference where it biases, and it reproduces the very exposures the 17.82 % came from.

**Written, waiting for motion:** each map's CSV line has where each camera saw change, in its own pixels. A small thing moving is one place in each view at one moment: a test of the geometry, and later more correspondences for it, with no appearance to match. Nothing has been measured with it yet: the first run with movement began about 05:10 on 2026-11-03 and is not analysed. **Not built:** adjusting the geometry from those correspondences.

## Design influences

Public descriptions only, used as general principles. Nothing here claims to reproduce, or to know the algorithms of, any operational military system.

**US7418574B2 (Lockheed Martin, the peer-vector machine).** Read in full 2026-11-02 at the user's direction; link and checksum in `input/refs/README.md`. Google's page lists it as expired with an adjusted expiry of 2024-03-16, and says that status is its assumption, not a legal conclusion; I make no legal claim.

| Patent's method | Here |
| --- | --- |
| Host and accelerator as peers, exchanging data vectors | host and card exchange frames and results by DMA, each with its own memory |
| Host for decisions, pipelines for math | on the card: tid 0 holds decisions and small matrices, teams hold the vector kernels. The brief puts estimation on the card, so this split lives inside the card |
| An interface memory as a FIFO | frame slots in the shared region, bounded, with a stated policy when full |
| Raw data straight to the accelerator | the card pulls frames from the region itself, and the host does no pixel work |
| Message objects: destination, priority, length, unique instance, type, algorithm, camera | done: `vt_msg`, checked word by word, malformed ones dropped and counted |
| One object may carry data and commands | done: a frame descriptor carries the operator's select or deselect, applied at that exact frame |
| Soft configuration, changing function without changing layout | done: `vt_sc_key`, gates in `check.sh` |
| Pipelines check headers and report routing exceptions | done: `vt_exception` |
| Several independent pipelines | two pipeline groups with a pyramid ring exist and replay identically; brief section 9's dynamic groups are not measured |

**What cannot carry over.** Claim 1 generates pipeline data "without executing any program instruction". A KNC core executes instructions; there is no FPGA here. So the influence is architectural, and the speed of hardwired logic is not on offer.

**F-22 CIP, early F-35 RACE++, LITENING (the brief's references).** Taken as principles: separate supervisory logic from numeric pipelines, local memory per processing group, sensor processing separate from overlays. Sources reached: Avionics Handbook chapter 32 (opening sections), the LITENING SPIE abstract only. The Mercury datasheet refused the fetch (HTTP 403), so RACE++ rests on the brief's own summary.
