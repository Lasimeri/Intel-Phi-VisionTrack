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
| M2 optical-flow tracking | **kernel and tracking run on the card**, including reacquisition after full occlusion | [measurements](docs/measurements.md) |
| M3 stereo reconstruction | **runs on the card, 1 to 2 mm RMSE on synthetic scenes**; ambiguity reported, gate fires | [measurements](docs/measurements.md) |
| M4 temporal state estimation | **runs on the card**; covariance not calibrated | [measurements](docs/measurements.md) |
| M5 live visualization, integration | **three cameras and a depth map live on the card** (L + R 1080p30, D 720p15; 15.9 maps/s at 9.8 ms, 18.41 % of textured pixels confident); LED star tracker live; the card's vectors gate the depth map's averaging; **geometry solved from the real room and judged correct by the user's eye, no measured ground truth; two cameras since knocked, so it is being solved again; geometry not yet adjusted from motion** | [measurements](docs/measurements.md) |
| M6 optimization, validation | partial | [limitations](docs/limitations.md) |

Open items: [limitations](docs/limitations.md). Design: [architecture](docs/architecture.md).

## Build and run

```
make                        # host tools + card programs (14 jobs); every card binary is ISA-audited
make compdb                 # compile_commands.json + clangd, 0 diagnostics or exit 1
scripts/vtrackd.sh start 512
scripts/scene.sh linear3d 640x480   # the card, then its replay on the host, compared frame by frame
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
