# Scene tolerances

What `vt_run --gate` holds each scene to (`score_gate` in `src/host/score.c`). They were set from measured runs at 640x480, as a margin above what the tracker does today, and exist to catch regressions and false claims. They are not targets. Truth is analytic (`src/host/synth.c`).

Every scene, always: a hidden region never reads as observed, and the track id does not change.

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
| dual | two independently tracked views, rotated, focal lengths differ, shutter 11 ms apart; TRUE calibration given (`scene.sh dual 640x480 --dual --gate`) | see `score_gate` | | | <= 15 mm | proves the fusion mechanism, not any calibration. Set from the measured 11.20 mm at 4 m with little parallax: the first limit, 10 mm, was written before the scene had ever run and failed it |

The depth map has no scene here, by the user's direction: it runs on the real room, which has no per-pixel truth. Its checks are in [measurements.md](../docs/measurements.md): card against host hash over real frames (a gate in `check.sh`, where those frames exist), a deliberately rotated pose must lose, the sweep's depth against depths triangulated from landmarks (`vt_mapcheck`), and physical sizes.
