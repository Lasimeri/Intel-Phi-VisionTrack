# Intel-Phi-VisionTrack

A passive, non-weaponized computer-vision tracking proof of concept on one Intel Xeon Phi 7120A (Knights Corner): sparse pyramidal Lucas-Kanade on native 512-bit MVEX, an operator-selected
anonymous image region, calibrated stereo reconstruction, and a constant-velocity Kalman filter on the card. The host handles acquisition, transport and display.

"Lock-on" means visual continuity of a region the operator chose. No weapon interfaces, fire control, ballistics, external aiming, face recognition, biometrics, or semantic classification exist here or are planned.

Built on [Intel-Phi-7120A](https://github.com/Lasimeri/Intel-Phi-7120A) and [Intel-Phi-LLVM](https://github.com/Lasimeri/Intel-Phi-LLVM), used as tools, not modified.

## Status

| Milestone | State | Evidence |
| --- | --- | --- |
| M0 audit, topology, native execution | **done, on the card** | [compatibility](docs/compatibility.md) |
| M1 SIMD numerical primitives | **done, on the card** | [measurements](docs/measurements.md) |
| M2 optical-flow tracking | **kernel and tracking run on the card**; frames generated between real ones for the tracker (effective 60, 60 and up to 30 a second); a lost region searched for over the whole picture until the operator ends it. On known truth in scenes; **a selected region still prefers to stand still on real targets with fixed cameras, not fixed** ([limitations](docs/limitations.md)) | [measurements](docs/measurements.md) |
| M3 stereo reconstruction | **runs on the card, 1 to 2 mm RMSE on synthetic scenes**; ambiguity reported, gate fires | [measurements](docs/measurements.md) |
| M4 temporal state estimation | **runs on the card**; covariance not calibrated | [measurements](docs/measurements.md) |
| M5 live visualization, integration | **three cameras and a depth map live on the card** (L + R 1080p30, D 720p15; 15.9 maps/s at 9.8 ms, 18.41 % of textured pixels confident); LED star tracker live; the card's vectors gate the depth map's averaging; tracking first, with flow and the map in the card's idle moments; optical flow at half native (4,080 cells for L and R; native, 16,200, also holds); each camera's turn measured constantly on the card from its own picture, with no lock, carrying the geometry; **the map is not yet right: L and R mostly do not agree through D except on one near object; no measured ground truth; a real knock not yet seen live** | [measurements](docs/measurements.md) |
| M6 optimization, validation | partial | [limitations](docs/limitations.md) |

Open items: [limitations](docs/limitations.md). Design: [architecture](docs/architecture.md).

## Build and run

```
make                        # host tools + card programs (14 jobs); every card binary is ISA-audited
make compdb                 # compile_commands.json + clangd, 0 diagnostics or exit 1
scripts/vtrackd.sh start 512
scripts/scene.sh linear3d 640x480   # the card, then its replay on the host, compared frame by frame
scripts/scene.sh bigbox 640x480 --framegen   # with generated frames between the real ones
scripts/scene.sh comeback 640x480 --frames 240 --gate   # past any patience and back; scene names and tolerances: tests/scenes.md
out/host/vt_ncc_check       # nearly blank windows must never match; no card needed
scripts/check.sh            # every gate; it takes the card, and a live window restarts after
scripts/vtrackd.sh stop     # also puts the card's huge pages back
```

The live window, following every rebuild, on the desktop with the card up:

```
scripts/live.sh --at X,Y     # L camcorder, R NexiGo, D C310; which device is which: input/calib/cameras.txt, which you write
```

Geometry and the star catalogues, on a still rig, from one pose:

```
echo 'snap 150' > out/run/view.cmd                            # integrated frames: out/run/snap-{l,r,d}.pgm; twice, saved apart
out/host/vt_gridpng out/run/snap-r.pgm r.png X Y W H SCALE STEP   # read landmarks by eye, in the shared overlap only
out/host/vt_calib POINTS --wa 1920 --ha 1080 --wb 1920 --hb 1080 --fb F --prior 40 --scale I J METRES --report --out LR.txt
out/host/vt_align R.pgm L.pgm D.pgm LR.txt TRIPLES --fd F --fix --out FIX      # D: resection on the LEDs and armrest: FIX-N.txt
out/host/m3_ref D.pgm R.pgm L.pgm FIX-N.txt --views 2 --planes 64 --near 0.8 --far 8 --rounds 6 --refine input/calib/current.txt   # D refined on the room
tools/arbiter.sh FB SCRATCH_DIR FRAME_PREFIX                                   # the same for one focal length, the score to compare
out/host/vt_stars build input/calib/stars-l.txt A.pgm B.pgm 6     # per camera; never mix a catalogue and a geometry from different poses
```

`--report` is not optional: a low rms says nothing about whether the 3D is physically sane ([limitations](docs/limitations.md)).

After that the card holds it by itself ([architecture](docs/architecture.md)): one line a second per camera in `logs/live/attitude-*.csv`. To look at it, or to re-anchor a camera that moved while nothing was running:

```
out/host/vt_attitude CALIB REF_PREFIX CUR_PREFIX [--out NEW.txt]             # PREFIX-{l,r,d}.pgm: how far each has turned, and the geometry carried
out/host/vt_attitude --test CALIB PREFIX CAM TILT PAN ROLL [SECOND_PREFIX]   # a known turn of a real exposure, and what it recovers
out/host/vt_attitude --transfer CALIB OLD NEW POINTS OUT TILT PAN [ROLL [FLOOR]]   # landmarks carried from R's old picture to its new, by correlation
out/host/vt_attitude --solve-r CALIB OLD_POINTS NEW_POINTS OUT               # R's turn from them; L, D and the lenses held
out/host/vt_agree CALIB PREFIX [--planes N] [--agg R] [--grid 1] [--dump OUT]  # do R and L, each referred to D, arrive at the same depth? The honest judge of the map
echo 'flow 0' > out/run/view.cmd                                             # optical flow at pyramid level 0 (native), 1 or 2: a new session on the card
echo 'framegen 0' > out/run/view.cmd                                         # generated frames between the real ones for the tracker, on by default (--framegen 0 at start): a new session on the card
```

The depth map runs by itself in the live window once `current.txt` holds all three cameras. To look at it closely, from the desktop host:

```
echo map > out/run/view.cmd                    # the newest map: out/run/map.{bin,pgm,ply}
out/host/vt_mapstat out/run/map.bin            # what is in it: cells, vectors, noise, depths
out/host/vt_mapcheck out/run/map.bin input/calib/current.txt input/calib/triples-pose3.txt [MAP2.bin]   # against landmarks; a second map: how still it is
```

Live numbers come from `logs/live/depth-*.csv`, one line a map. `--avg N` on `live.sh` sets how many frames a still picture averages, 1 for single frames; key `f` shows all of the card's vectors, not only the cells that changed.

The kernels from saved frames, host and card, which must print the same hashes (a gate in `check.sh`):

```
out/host/m3_ref D.pgm R.pgm L.pgm input/calib/current.txt --views 2 --planes 64 --near 0.8 --far 8 [--seq PREFIX N] --write m3.bin --pgm MAP   # PREFIX-i-{d,r,l}.pgm first; MAP-depth.pgm
bash -c '. scripts/lib.sh; phic put m3.bin /tmp/vt/m3.bin'; scripts/card_run.sh -p 256 vt_m3 /tmp/vt/m3.bin 9
```

Requires the 7120A stack and the KNC clang in the default locations of `config.mk` (each can be overridden), and a running card (`phi up`).

## Not in this repository

Frames, depth maps, point clouds, landmarks, poses, live logs and camera serial numbers: they describe a private room. So the live and real-frame results in [measurements](docs/measurements.md) cannot be rechecked from a clone, `scripts/check.sh` skips its real-frame gate, and another rig needs its own calibration ([input/calib](input/calib/README.md)). The synthetic scenes, the card-against-host gates on them, and the raw logs of those runs are all here.
