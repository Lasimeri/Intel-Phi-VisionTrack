# Local calibration data

Nothing here but this file is in the repository. Everything else describes one private room and rig: the landmarks read from it, the poses solved in it, the frames it was solved from, and the cameras' serial numbers. Results that cite these files, or `logs/live/`, cannot be rechecked from a clone; they are reported in [measurements.md](../../docs/measurements.md) with their numbers.

To run on another rig, make your own, as the README's "Geometry and the star catalogues" says.

| File | Format | Made by |
| --- | --- | --- |
| `cameras.txt` | one line per camera, `L PATH`, `R PATH`, `D PATH`; `#` starts a comment. Use `/dev/v4l/by-id/...` paths: a replug renumbers `/dev/videoN`, and left and right must not swap with it | by hand |
| `current.txt` | the geometry `vt_view` loads, and reloads when it changes: `fa cx cy k1 k2`, the same for camera B, 3 rows of R, T in metres; then camera D the same way | `vt_calib`, `vt_align`, `m3_ref --refine` |
| `stars-{l,r,d}.txt` | `x y flux area`, one per fixed light, in that camera's pixels | `vt_stars build` |
| `TRIPLES`, for `vt_align` and `vt_mapcheck` | `Lx Ly Rx Ry Dx Dy label`, full-size pixels; labels `led*` are weighted as lights, `arm-A` and `arm-B` as one known pair | by hand, on `vt_gridpng` crops |
| `sets/` | saved exposures, PGM. `scripts/check.sh` runs its real-frame gate where these exist and skips it where they do not | `snap N` to `out/run/view.cmd` |
