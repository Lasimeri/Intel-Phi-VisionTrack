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
| Making a frame between two real ones, while a camera holds a region | card, team | 1,680 to 1,930; 5,950 at 1080p |
| Region state machine; detection when needed, on the team | card, tid 0 | 51 (measured before the region model of 2026-11-03; 106 to 330 mean on the box scenes since) |
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

## Holding a selected region (2026-11-03)

An operator's drag holds the thing and whatever is around it. The first tracker took the median motion of all it held, so when most of the box was background the region sat on the background reading OBSERVED while the thing walked away: 187.98 px behind with continuity 1.00 in the scene that was added to show it. Nothing had caught this, since every scene until then selected from inside the object. Numbers: [measurements.md](measurements.md).

**What runs now: the first tracker's median, on a virtual gimbal (the operator's direction, 2026-11-03).** Two designs came between. A rigid region model held the known scenes well and stood still on real targets, because its assumptions suited a camera on a gimbal. An inversion, tracking what differs from a fixed-camera background, passed all 79 gates of its day and failed live at once: its box covered over half the picture on 32.9, 28.5 and 71.8 % of L, R and D's results. The operator's order was to go back to the older method that had worked and give it a gimbal in software. The head points, and the window is where it points. Nothing on the rig moves.

| Part | What it does | What it replaced, and why |
| --- | --- | --- |
| The median | the first tracker's: median displacement of the voters, scale and turn from the median stretch about them, and whatever lies over 3 MAD or 0.5 px from that dropped | a rigid model fitted to members, with anchors, which a real non-rigid mover broke apart |
| The head | its speed is every feature's starting guess in LK. The target then stands about still in the head's view while the background streaks past and fails the MAD test by itself | LK's zero guess, which lost what moved fast |
| At rest | a head that has not moved, or has parked, takes no median, so nothing integrates noise: `hold` reads 0.000 px | summing, then a held frame |
| Starting | at least `min_features` features whose remembered motion is over 2.5 px, coherent with each other, for 3 real frames. Still features cannot outvote them, and are dropped when it starts | a 15 % share of the votes and one rigid fit |
| Catching up | at the start the window jumps by the median distance moved by the features that were there from the beginning. Later ones travelled less and would hold it short: 3.2 to 3.3 px on three scenes before this | |
| Looking there | a big box holds few features on a small thing, 5 or 6 of 128 on one scene. When at least 3 features have moved or died in the flow, and too few are moving to start the head, every other still feature gives way to new ones around that place. Deaths count, since blur kills a feature rather than moving it | founders taken only from the box's inner half |
| Outsiders | a feature that moved before it had settled was born moving, and motion first seen at the box's edge after the first 15 frames came from outside. Neither can start the head, which is what `passby` holds | |
| Riders | a new feature has no vote until the head has gone 2 px since its birth; it is then dropped unless it went with the head to within half that distance. At 8 px it votes that same frame, since at speed it may not live to the next | riders graduating by age, which is how the background became the majority |
| Parking | under 0.1 px a frame for 15 real frames. A parked head starts again only if at least half its voters moved, so a part of the thing waving does not move the window | |
| The box | closes once when the head first moves, then goes half way each frame to the members' span times 1.25 plus 8 px, never past the operator's box. The centre stays on the point chosen, unless that lies outside the members' span, or their scatter (MAD over 1 px) says the thing is not rigid: it then settles toward their middle at 5 % a frame | the box as drawn, which the operator found too padded |
| Generated frames | nothing: for the eye only; a generated row's box is the middle of the two real boxes | |

**Corners are judged where they stand.** The Harris response goes as the gradient to the fourth power, so against one threshold for the box, a sharp corner on an object's outline, whose patch straddles two motions, erased everything inside the object: 11 corners in 384 x 307, and a region lagging 18 %. Now each corner answers to the strongest of its 16 x 16 cell and the eight around it, above a floor. Small boxes space their features closer, down to 6 px, so a small thing gets enough.

**Detection is on the team.** It ran on the tracking thread: 6.1 to 8.0 ms for 384 x 307, on every frame while features were low, and judging corners by neighbourhood took it to 26.37 ms on a selection frame. The tensor, response and cell stages now run on all threads on rows that do not depend on each other, and the last stage scans a vector at a time and orders candidates by a radix sort over the complement of their score's bits, integers only. 3.78 ms. While the head moves it reseeds on any real frame short of members or features; at rest, on every 4th or 16th.

**A take back keeps the head moving.** A region taken back at speed was then told it had stopped, and new features that all voted stood it still on the background. Now the head keeps its speed, and at speed the new features ride.

**Known fault: 3D before the head moves.** Until then 3D comes from all the box's features, mostly background, and the filter starts from that ([limitations](limitations.md)).

### Losing and taking back

There is no time-out. Past its patience (2 s live) a region claims no position and reads LOST, but only the operator ends it. It used to end there, its appearance erased, and could never come back.

| Part | What it does |
| --- | --- |
| Memory | Four appearances: the first since the box closed, never replaced, and about 0.5, 2 and 8 s old, from frames over 90 % agreed. Each has its level, offset and the box it had. The box closing clears them, since until then they may hold background |
| Near the prediction | While predicting, every memory within 12 to 36 px, as far in the picture at any level. Believed at 0.82 |
| The whole picture | After 8 frames, or once lost: a band of 32,000 positions a frame, one memory after another, a map of the best per 16 x 16 places, on the team. About 0.5 ms a frame. A coarse memory's scan takes a frame or two, a small thing's about half a second. Believed at 0.88, 0.04 clear of any other peak in the band, and again on the next frame within 24 px |
| Half pixels | The map's top 8 cells are scored again at half-pixel steps before anything is judged: on fine texture a whole-pixel search at 1/4 scale can stand half a pixel off and read 0.5 at the right place. A parabola places the winner between them |
| Blank windows | Sums run relative to the window's first pixel, and a window with under 1/16 of the template's energy scores 0. In absolute terms rounding read correlations up to 10.6 on nearly blank patches, which took the region |
| Coasting | A lost region's rate decays 5 % a frame: a thing out of sight is not known to keep going |

### Generated frames (user's direction, 2026-11-03)

Between two real frames the card makes one more for each camera, selected or not, which takes L and R to 60 pictures a second on the screen and D to 30.

**To the screen.** The picture is reduced to bytes on the team (`vmovaps {uint8}`, 78 microseconds at 1080p) and copied to a slot at the top of the shared part, on the same channel as its result and before it, so it has landed when the result says so. The window holds a queue per camera: every picture, real or made, with what the tracker knew of that very picture, is due at its own instant plus a delay, the 95th percentile of how late pictures arrive. A generated picture is the latest against its own instant, half a frame plus the card's time, so the real frame after it waits its turn; with generation off the delay falls by itself to the card's time alone. Each pane is drawn and presented when its own next picture is due, and nothing else moves. Options set aside: vectors to the host and a warp there, heavy work on the host against the project's rule; copying Q6 floats, 4 times the DMA; redrawing the 3840 x 1800 window for each picture.

| Stage | What | Whose |
| --- | --- | --- |
| Pyramid | The tracker's own levels from 1 up, then more, to at most 7 levels | |
| Search | Each 8 x 8 block, from the level above's estimate, tries 256 places by SAD. A lane per horizontal offset, the row slid with `valignd` instead of loaded 8 times, unrolled since the slide's count is an immediate. No motion wins on the last level if it matches as well | AMD's |
| Filter | Of the 3 x 3 around, the vector with the least squared difference from the rest | AMD's |
| Upscale | Each block of the next level picks among four by SAD over 4 x 4 | AMD's |
| Early out | On the last level a block the estimate already explains, within 2 grey levels a pixel, is not searched | this project's |
| Middle vector | The field says where a block of the newer frame came from; a generated frame needs what is at its own instant. Each block tries the 3 x 3 of vectors around and none, moving both frames half way, and takes the one on which they agree | this project's |
| Blend | Both frames moved half way, under triangular windows over the nine blocks around each tile, so that no block's edge shows; one vector over all nine is a plain average | this project's |

FSR 3's interpolation leans on a game's own motion vectors, which a camera does not have, so only the field is AMD's. The source is MIT licensed: `LICENSES/FidelityFX-MIT.txt`.

**Why on the card and not AMD's shaders on the GPU.** The only GPU here is an RTX 3090 Ti, so AMF is out, and the SDK needs DX12 or Vulkan. Running its shaders there would put image processing on a second chip, with a copy each way per frame. On the card both frames and the result are already in its memory.

**The tracker does not read a generated frame.** Its pixels are invented, and following them cost it the target (`dart`) or a steady drift (`smallbox`, about .067 px a frame): each real frame started from a place measured on a blended picture. A real row is the same to the bit with generation on or off. A generated row is made afterwards: the middle of the two real boxes, and each feature half a step back along its own real step.

**It costs a real frame milliseconds, not a frame.** A generated frame needs the newer real frame, so it is made once that frame has landed, and its result reports an instant half a frame old. The real frame's own result waits for it: about 6 ms to make and 1 to track at 1080p (arithmetic from the stages; the latency itself is not yet measured live).

**What it can and cannot do.** It adds no information: everything in a generated frame comes from the two real ones. What it gives the tracker is two half steps for one whole one, which is why it helps where features are lost or torn between motions, and adds a little noise where nothing was wrong (`swing`).

### Camera D through FSR (the operator's order, 2026-11-03)

720p at 12.5 frames a second goes up to 1440p by FSR 3.1.6 (`src/kern/fsr3.c`), on real frames only, so nothing invented enters its history. Flow makes the frame half way from the two finished pictures, 1440p25. A Lanczos brings both to 1080p. FSR has its own team on each core's third and fourth hardware threads: it slows the tracker while both run, the tracker never waits for it, and it lets a frame go if it is still busy. The tracker follows the raw frames.

One channel, one vector a block, no gather: each source row is filtered across once, by permute tables, and an output row mixes four of them; a still block reads its history directly. There is no vector divide on this card and its reciprocal instruction is an approximation, so division and square root are Newton steps from the bits, the same on card and host.

**The orange vectors on the spare core (the operator's order, 2026-11-03).** D's motion vectors, which cells moved, lines at 1440p, and those brought down with the picture's own tables, all on the core nothing else uses (`src/kern/vecdraw.c`, `struct vec_worker`). The session hands a frame over and takes the plane back at the top of its loop; only its own part of that frame waits for a gap. Pictures have one channel, so the lines travel as a plane of their own, and the window finds the one of that instant when the picture comes due. L and R's vectors and means run on FSR's team between its frames, each camera at least once a frame of FSR's, and go to the window alone; a map can no longer hold them up, and a map reading a mean being updated sees some of each frame.

## Three cameras (user's direction, 2026-11-03)

| Camera | Where | Mode | Role |
| --- | --- | --- | --- |
| 0, L: camcorder via capture card | the user's left | 1080p30 MJPEG, USB 3 | primary 3D view |
| 1, R: NexiGo | the user's right | 1080p30 MJPEG, USB 2, own controller | primary 3D view |
| 2, D: Logitech C310 | centre | 720p15 MJPEG, USB 2, own controller | depth map view; no vote in the 3D point |

Each camera is one tracker on the card, sharing the team, frames interleaved in the order they land. `fuse.c` intersects L's and R's rays and holds one Kalman state. Devices are bound by their `/dev/v4l/by-id` paths, named in the local `input/calib/cameras.txt` because those paths carry serial numbers: a replug renumbers `/dev/videoN`, and left and right must not swap. The NexiGo and C310 cannot share a USB 2 controller in MJPEG (they claim alt 6 and alt 11 at any frame rate: `ENOSPC`).

## Navigation by fixed lights (user's direction, 2026-11-03)

The room's LEDs are treated as a star tracker treats stars. `src/host/stars.{h,c}`, on the desktop host, a fix per camera per second from a 16-frame mean, on its own thread (`nav_main`): on the draw thread it froze the window for 33 to 40 ms every second. Since 2026-11-03 this is a check only, as "Calibration that holds itself" explains; the table below is the method, and the carrying it describes is no longer done by the lights.

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

## Calibration that holds itself (the user's direction, 2026-11-03)

"Constant and convolutional", with no lock. The first solve below is still how a rig is first calibrated; this keeps it true afterwards. Original C throughout, the published work read for its mathematics only.

The method rests on one observation: L and R are 56 degrees apart and appearance fails across that gap, but **a camera compared with itself has no gap**. So each camera measures only its own turn, against its own reference, and the pair follows in closed form, R' = Ob R Oa', T' = Ob T. A New 3DS has the same shape, stored constants plus a closed-form correction from a live measurement, though its eye tracking has no place here.

| Stage | How |
| --- | --- |
| Picture | the camera's still picture at the map's level, averaged and gated by the card's vectors, already on the card |
| Normalize | each pixel's distance from its 9 x 9 mean, in units of that window's spread, so a light switched on changes little |
| Search | every shift over the reference at a quarter of that level, a correlation each, in units of chance so a small overlap cannot win. Reaches +-640 and +-480 pixels of a 1080p camera |
| Refine | Gauss-Newton on three rotation parameters, coarse to fine, inverse compositional: the reference's gradients are computed once. Homography exact at each end of 16 lanes, straight between, four `vgatherdps` for the bilinear sample |
| Kernel | Tukey's biweight on each residual, no division on a machine without one: whatever moved has no say |
| Earn it | a turn must beat no turn by 0.10 of correlation and exceed 0.1 degree, or nothing changes |
| Chain | a turn that lasts 5 measurements while the picture is quiet becomes the new reference, and so does a reference the light has left behind, at most every 30 s |
| Bit-exact | quaternions, first order and renormalized, because `sin` differs between the two libm; sums per row added in row order, so 1 thread and 118 agree |

On the card once a second per camera, 2.6 to 3.6 ms each. `vt_view` carries the geometry when a turn shows twice, and keeps it on file after 10 steady seconds, so a restart does not fall back to a pose the rig has left. A camera that cannot find itself for 5 seconds (under 0.2 correlation or a quarter of the view) hides the map and the 3D point.

**Why not the others.** A CNN (the one published method found covers +-2.5 degrees, and needs training pairs this rig does not have) was ruled out by the user's rule and the numbers alike. Tracking the essential matrix between L and R needs matches that this angle does not give (7 to 15 inliers). Tuning on the map's confidence fails with anyone in view (measurements.md). Motion barcodes, correlating motion along epipolar lines, would need no appearance at all and remain the way to check L against R directly: not built.

**The star tracker is now a check.** Each fix searched around the last, which is a lock: a hidden light walked it onto something else, and it called that a move. It has no memory now, and neither carries the geometry nor turns the map off.

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

**What this does not yet achieve.** Each view's cost is summed against D, so nothing requires R and L to arrive at the same depth by themselves, and measured they mostly do not: `vt_agree` asks exactly that, and it has become the honest judge, since the map's own confidence cannot judge geometry with anyone in view. 64 planes is also too few for this rig: 4 to 6 pixels a plane at this level. See measurements.md, "Do L and R agree through D?".

**Live, on the card (`src/card/live_depth.c`).** In the three-camera session, on the session's own team, between frames. Every frame of every camera updates that camera's still picture; every frame of D then normalizes the three still pictures, sweeps, aggregates and decides. z, confidence, each camera's cells and last a head go by DMA to one of two 1 MiB buffers in the free [2 MiB, 4 MiB) of the region part, and the head's DMA status writes `depth_pub`, so the word that names a map comes after every byte of it on the same channel. The next map waits for that copy to finish before writing its buffers. The relation of the views comes from one function, `depth_views_from_calib`, used by this and by `m3_ref`, recomputed at every `calib_seq`, so the star fixes carry the map too.

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

**Resolution.** The still pictures run at a pyramid level set at run time, level 1 by default: 4,080 cells for L and R, 1,800 for D. Native (16,200 and 7,200) is the finest that holds on one card, and the user found it laggy (measurements.md). The table's cell counts are for level 2, where this began. The map and the attitude kernel read level 2, reduced from it on the team, so the three costs stay separate.

**Tracking comes first.** Each result leaves as soon as it exists. Flow and the map run in the session's idle moments, one piece at a time, each camera's job always its newest frame, so work not reached is replaced rather than queued. The window shows a frame only when its result returns, so this is also what keeps the picture live. On screen, flow vectors lie under the tracker's marks and vanish while a target is dragged.

The choice over the design first written here, averaging the cost volume: three small planes against 15 MB at 15 Hz, it uses all of L's and R's 30 frames a second, it removes noise before the difference where it biases, and it reproduces the very exposures the 17.82 % came from.

**Written, waiting for motion:** each map's CSV line has where each camera saw change, in its own pixels. A small thing moving is one place in each view at one moment: a test of the geometry, and later more correspondences for it, with no appearance to match. Nothing has been measured with it yet: the first movement came with cameras being knocked on 2026-11-03 (from 04:59:36), when the geometry was stale, so that run cannot test it. **Not built:** adjusting the geometry from those correspondences.

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
