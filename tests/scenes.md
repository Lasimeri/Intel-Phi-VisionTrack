# Scene tolerances

What `vt_run --gate` holds each scene to (`score_gate` in `src/host/score.c`). They were set from measured runs at 640x480, as a margin above what the tracker does today, and exist to catch regressions and false claims. They are not targets. Truth is analytic (`src/host/synth.c`).

Every scene, always: a hidden region never reads as observed, and the track id does not change.

`check.sh` also runs eleven of them with `--framegen`, to the same tolerances on the real frames, card against host on every result, and `vt_ncc_check`: nearly blank windows must never match in the appearance search, which no scene could show (an exactly flat patch has zero variance and was already refused).

**What these do not cover.** Every moving thing here is rigid, without motion blur or change of light, and most of what its box's middle holds. All of them pass while a real selected target can still be left behind by a window that stands still ([limitations](../docs/limitations.md)). A scene with a small non-rigid mover off the middle of a large window does not exist yet, and is the first step of fixing that.

| Scene | What it provokes | Continuity | EPE median | Centre median | 3D RMSE | Other |
| --- | --- | --- | --- | --- | --- | --- |
| translate | constant image motion, still background | >= 0.99 | <= 0.02 px | <= 0.05 px | <= 5 mm | |
| camera | the camera moves, not the object | >= 0.99 | <= 0.02 px | <= 0.05 px | <= 5 mm | |
| leave | the object leaves the view | >= 0.99 of frames it is visible | <= 0.02 px | <= 0.05 px | <= 5 mm | |
| shapes | a disc on a curved path, a second mover | >= 0.99 | <= 0.02 px | <= 0.1 px | | |
| brightness | gain and offset vary | >= 0.99 | <= 0.1 px | <= 0.5 px | <= 5 mm | |
| noise | additive noise, sigma 4 grey levels | >= 0.99 | <= 0.1 px | <= 0.5 px | <= 5 mm | |
| occlusion | a bar crosses part of the object | >= 0.95 | <= 0.02 px | | | |
| loss | a block hides the object entirely for 1 s | >= 0.95 of visible frames | | | | taken back at least once, within 2 px |
| helix | 3D curve, depth changes | >= 0.99 | <= 0.02 px | <= 0.5 px | <= 5 mm | velocity RMSE <= 50 mm/s |
| linear3d | constant 3D velocity | >= 0.99 | <= 0.02 px | <= 0.5 px | <= 5 mm | velocity RMSE <= 50 mm/s |
| glitch | the right image jumps 14 px for 10 frames | | | | | the gate rejects; filter stays within 20 mm |
| repeat | 2D periodic texture, period inside the disparity range | | | | | reported ambiguous; ambiguity, not rejection, stops it |
| flat | no texture | | | | | selection refused, nothing observed |
| bigbox | a box 2.4 times the object, over a background as busy as the object: what an operator's drag really holds | >= 0.95 | | <= 4 px | | centre p95 <= 12 px |
| smallbox | a box 4 times a smaller object: most of the box and of its middle is background | >= 0.95 | | <= 4 px | | centre p95 <= 12 px |
| stillstart | as bigbox, still for 2 s after selection and only then moving: nothing tells it from the background until it does | >= 0.95 | | <= 4 px | | centre p95 <= 12 px |
| rotate | as bigbox, turning 30 degrees a second: no cluster of motion to find | >= 0.95 | | <= 4 px | | centre p95 <= 12 px |
| swing | as bigbox, swinging to and fro, up to 19 px a frame | >= 0.95 | | <= 4 px | | centre p95 <= 12 px |
| hold | a still object under noise of sigma 4: any travel of the region is error | >= 0.99 | | <= 0.5 px | | centre p95 <= 1 px |
| comeback | as bigbox, out of the picture to one side for 3 s, past any patience, and back from the other: `--frames 240` | | | <= 4 px | | runs out of patience, is taken back, within 3 px; never reads observed while away |
| offbox | **fails today, on purpose not yet a gate:** rigid, wholly outside the middle half of a box 5 times its size. Window went 0.000 px, thing 211.2 px, continuity 1.0000 | | | | | window travel median <= 4 px, p95 <= 12 px |
| jointed | **fails today:** six parts with their own motions, turns and shutter blur, in the middle of its box: window travel median 27.262 px | | | | | as offbox |
| offjointed | **fails today:** the same off the middle: window went 0.001 px, thing 211.2 px | | | | | as offbox |
| passby | a still thing selected, another crosses the box without touching it: the window must not leave. Holds today, 0.013 px | | | | | window travel max <= 3 px |
| dual | two independently tracked views, rotated, focal lengths differ, shutter 11 ms apart; TRUE calibration given (`scene.sh dual 640x480 --dual --gate`) | see `score_gate` | | | <= 15 mm | proves the fusion mechanism, not any calibration. Set from the measured 11.20 mm at 4 m with little parallax: the first limit, 10 mm, was written before the scene had ever run and failed it |

The depth map has no scene here, by the user's direction: it runs on the real room, which has no per-pixel truth. Its checks are in [measurements.md](../docs/measurements.md): card against host hash over real frames (a gate in `check.sh`, where those frames exist), a deliberately rotated pose must lose, the sweep's depth against depths triangulated from landmarks (`vt_mapcheck`), and physical sizes.
