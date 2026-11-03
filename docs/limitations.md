# Known limitations

Current as of 2026-11-03 07:00. Numbers and their sources: [measurements.md](measurements.md). Items leave this list only with a measurement.

## Failing, or not built

- **The depth map does not yet tie L and R to D properly, and today it is mostly speckle.** D+R and D+L agree within a step on 5.6 to 7.2 % of the pixels both are sure of (chance 3.2 %), with someone in view; 50 % in the empty room, and there only on the near chair. The far wall has never agreed. Too few planes and a wrong D pose are both shown to contribute, neither is the whole answer, and neither fix is live ([measurements.md](measurements.md)).
- **The map's planes are still 64.** It needs about 256 (about 28 ms more per map, arithmetic), which level 1 flow has room for, at about 367 ms of the session's thread a second. Not done yet.
- **A frame landing during background work still waits for it,** up to about 10.5 ms behind a map: tracking comes first between jobs, not within one. That the flow, rather than the tracker, gives way under overload has not been seen, since nothing has been overloaded since the change.
- **The motion gate is too quick at finer flow:** 10.4 % of R's cells called moving at level 1 and 22.8 % at native, against 3.5 % at level 2, so its averaging suffers. Scaling the vector limit did not cure it.
- **The constant calibration has not yet seen a real knock.** Live it has only held still, correctly, with someone in view. Its recoveries are of turns warped in software into real pictures (0.006 to 0.029 degrees missed), and the real knock of 04:59 came before it ran. Nudging a camera is the test still owed.
- **It measures rotation only.** A camera's centre moving leaves parallax it cannot remove and biases it: on R's knock the whole picture said -10.06 degrees where the far wall says -9.22. How far R's centre moved is unmeasured, and near objects in the re-anchored geometry are untested, since the chair the near landmarks were on has moved.
- **The map's confidence cannot judge geometry with anyone in view.** For two cameras that had not moved, a pose 3 degrees wrong scored higher (18.35 %) than the right one (16.38 %). 18.60 % held only in the empty room, so today's 6 to 8 % says nothing either way, and continuous tuning on that score was abandoned.
- **Turns of about 0.1 degree and under go uncorrected,** about half a pixel at the map's level: below that a false turn cannot be told from a true one. For the same reason a camera creeping slower than the reference refreshes, at most every 30 s, is not seen.
- **A knock while no session is running is not seen.** The reference is the session's own first settled picture, so it starts from wherever the camera now is. The star catalogues, read fresh with no memory, are the only witness.
- **After a carried turn the star catalogues describe the old pose,** so the lights check reads the offset until they are rebuilt. They no longer affect anything.
- **L against R is not checked directly.** Each camera is checked only against itself. Motion barcodes could check the pair with no appearance at all: not built.
- **The card's LK vectors gate the depth map's averaging but do not adjust the geometry.** The attitude kernel correlates pictures instead. Adjusting geometry from motion, the rest of the user's direction of 2026-11-03, is not built.
- **The motion gate is only partly evaluated on movement live.** During the knocks of 2026-11-03 its whole-camera rule fired in all three cameras, and its counts fell back when things went quiet; whether the flagged cells are where things moved is unchecked. Otherwise measured on an empty room (almost nothing flagged, which is half of what it must do) and on saved frames of a change of scene, where the whole-picture rule fired in one camera of three (R 1,020 cells, L 332, D 242).
- **The map's depths under confidence 0.2 can be metres wrong.** The two LEDs read 3.11 and 1.44 m against 4.28 and 4.30 m triangulated, at confidence 0.07. They are shown at a quarter brightness, and sit in the CSV's `known` count, not its `confident` one.
- **The sweep's reference is not synchronized.** A map pairs D's frame with L's and R's still pictures as they stand, up to 33 ms apart; a moving thing is therefore misplaced in depth by however far it moved.
- **No live 3D or depth number has a measured ground truth.** The user, who can see the room, judged the geometry correct by eye on 2026-11-03; beyond that it is bounded by physical tests, not measured. Scale rests on one TV edge whose corners were read by eye to about 5 px; one tape measure between L and R would test it. An armrest of 301 mm is plausible, and no more than that.
- **D's pose is chosen from a valley.** The LEDs and armrest allow many poses; the kernel's own agreement chose among them. A rotated control loses steadily, so the signal is real, but the remaining error is unmeasured.
- **Lens distortion is not modelled in any solve.** The wire and `fuse.c` carry k1, k2; every solver writes zero. An 82 degree horizontal lens surely has some.
- **The camcorder is a poor measuring instrument as it stands.** Autofocus turned a sharp LED into a 40 px disc and moved it 5.2 px, nearly radially, about 0.9 %: what focus breathing would look like (one light: consistent, not proven). If so, fa is not constant, and geometry solved at one focus is off at another. This camcorder's focus cannot be locked (the user, 2026-11-03), so fa would have to be estimated as it runs; nothing does that yet. Dim targets cannot be tracked in it in this light.
- **The map covers little:** about 10 % of a dim room's pixels have a depth at this level, about 3 % a confident one (1,742 of 57,600), most of the rest being refused for lack of texture or evidence. The rear wall, the posters and the TV have none.
- **The landmarks do not sit where D's pose puts them.** Triangulated from L and R they project 6 to 9 px from where the eye read them in D (x always positive, y changing sign from top to bottom). Refinement traded that for the room's score. Which is right is unknown.
- **Live 3D from L and R is mostly refused, correctly, on large selections.** One run: 44 accepted, 556 rejected in 20 s, rays missing by 49 px mean, because a large selection in the noisy camcorder slid off what it was on. The two trackers follow regions; nothing guarantees their centres are one physical point. That run predates the current geometry.
- **The LEDs are a check, and a weak one.** L and R share one bright LED and its faint neighbour, 45 mm apart: effectively one bearing. L has 1 catalogued light, D 2, R 7, some of R's possibly daylight glints. One light cannot see roll. The solver is small-angle: it read R's turn as -11.31 degrees where the true value is -9.51. Lost-in-space recovery searches shifts only, and can pick a wrong light.
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
