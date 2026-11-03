# Known limitations

Current as of 2026-11-03 05:30. Numbers and their sources: [measurements.md](measurements.md). Items leave this list only with a measurement.

## Failing, or not built

- **The card's vectors feed the depth map but not yet the geometry.** They gate the still pictures. Adjusting the geometry from motion, the rest of the user's direction of 2026-11-03, is not built, and needs something moving in view: the room has been empty.
- **The motion gate is not yet evaluated on movement live.** A first such run began about 05:10 on 2026-11-03 and is not analysed. Otherwise measured only on an empty room (almost nothing flagged, which is half of what it must do) and on saved frames of a change of scene, where the whole-picture rule fired in one camera of three (R 1,020 cells, L 332, D 242).
- **The map's depths under confidence 0.2 can be metres wrong.** The two LEDs read 3.11 and 1.44 m against 4.28 and 4.30 m triangulated, at confidence 0.07. They are shown at a quarter brightness, and sit in the CSV's `known` count, not its `confident` one.
- **The sweep's reference is not synchronized.** A map pairs D's frame with L's and R's still pictures as they stand, up to 33 ms apart; a moving thing is therefore misplaced in depth by however far it moved.
- **Every map delays the tracker's frames behind it,** the two sharing one thread of control: about 10 ms at 15 Hz, seen as L capture to result 32.33 ms median against 29.8 to 32.0 before (one panel reading).
- **No live 3D or depth number has a measured ground truth.** The user, who can see the room, judged the geometry correct by eye on 2026-11-03; beyond that it is bounded by physical tests, not measured. Scale rests on one TV edge whose corners were read by eye to about 5 px; one tape measure between L and R would test it. An armrest of 301 mm is plausible, and no more than that.
- **D's pose is chosen from a valley.** The LEDs and armrest allow many poses; the kernel's own agreement chose among them. A rotated control loses steadily, so the signal is real, but the remaining error is unmeasured.
- **Lens distortion is not modelled in any solve.** The wire and `fuse.c` carry k1, k2; every solver writes zero. An 82 degree horizontal lens surely has some.
- **The camcorder is a poor measuring instrument as it stands.** Autofocus turned a sharp LED into a 40 px disc and moved it 5.2 px, nearly radially, about 0.9 %: what focus breathing would look like (one light: consistent, not proven). If so, fa is not constant, and geometry solved at one focus is off at another. This camcorder's focus cannot be locked (the user, 2026-11-03), so fa would have to be estimated as it runs; nothing does that yet. Dim targets cannot be tracked in it in this light.
- **The map covers little:** about 10 % of a dim room's pixels have a depth at this level, about 3 % a confident one (1,742 of 57,600), most of the rest being refused for lack of texture or evidence. The rear wall, the posters and the TV have none.
- **The landmarks do not sit where D's pose puts them.** Triangulated from L and R they project 6 to 9 px from where the eye read them in D (x always positive, y changing sign from top to bottom). Refinement traded that for the room's score. Which is right is unknown.
- **Live 3D from L and R is mostly refused, correctly, on large selections.** One run: 44 accepted, 556 rejected in 20 s, rays missing by 49 px mean, because a large selection in the noisy camcorder slid off what it was on. The two trackers follow regions; nothing guarantees their centres are one physical point. That run predates the current geometry.
- **The LEDs cannot establish geometry alone.** L and R share one bright LED and its faint neighbour, 45 mm apart: effectively one bearing. L has 1 catalogued light, D has 1, R has 7. They anchor and raise alarms; carrying geometry needs at least 3 agreeing lights, which only R has.
- **One light cannot see roll.** A true 0.5 degree roll reads on L as tilt -0.15, pan -0.06 degrees, so cameras with one light never correct themselves, only raise an alarm above 15 px.
- **Lost-in-space recovery searches shifts only.** A combined 1, 1, 1 degree move on R finds nothing, and says so. D loses lock at 2 degrees and beyond.
- **The rig watch false-alarms on a dim camera,** and missed one real move on D. It is used only for a camera with no light in view.
- **Covariance is not calibrated.** On the `dual` scene: NEES mean 7.28 against 3, NIS mean 0.013. In `session_config` the two-view noise values (sigma_c 1.0, r_floor 0.005, q 1.0, max_reproj 25) are guesses. Noise set A from the sweep is not fully applied in `vt_default_config`.
- **Real-time benchmarks are unfinished:** the paced latency matrix, stage groups, scalar against SIMD over a whole pipeline, temperature and clock.
- **Detection on a large selection takes 4.7 to 41 ms** on tid 0; it should be on the team.
- **The window creeps 64 px down per restart** (`view.pos`).
- **`tools/vt_match.cpp` and `tools/celldiff.c` are outside the clean-compile gate:** `make compdb` covers `src/` (88 files, 0 diagnostics); those two are built by hand.

## By design

- Mode A has no metric output: an uncalibrated single camera cannot give depth, and depth is reported as unavailable.
- Geometry that a camera's move has outdated shows no 3D numbers at all.
- A pixel gets depth only with texture, a clearly best match, and at least half its depths actually inside the other view.
- The display mirrors; the card's frames, `view.cmd` and `snap` never do.
- Two views are never rectified or synchronized: each camera is tracked alone, the rays are intersected at camera B's time, camera A carried along its image rate.
- A region tracks translation plus uniform image scale. Rotation and perspective change are not modelled, so features die and get replaced under them.
- The constant-velocity model only: outside the brief's scope to do more.
- Calibration is offline on the desktop host. OpenCV appears only in `tools/vt_match.cpp`, which found too little to be used.
- What the cameras saw stays out of the repository: frames, depth maps, point clouds, landmarks, poses and live logs (`input/calib/`, `logs/live/`, `out/`). So does each camera's serial number, in the local `input/calib/cameras.txt`. A clone therefore needs its own rig and calibration, and skips the gate over real frames.
- No synthetic rig for the depth map: the user's direction. Real frames have no per-pixel truth, so its accuracy comes from physical checks.

## Inherited

- **The compiler has no vectorizer for KNC:** generic vector code compiles but runs on x87, so the brief's compiler-generated tier does not exist ([compatibility.md](compatibility.md)).
- **Card-timed durations may read about 0.48 % short** (the card's clock lag; untraced).
- **The per-feature `sigma` is a lower bound:** it treats a patch's residuals as independent.
- **In `--variant 1` (x87) detection, appearance search and frame difference stay MVEX,** so it is not a pure scalar pipeline.
