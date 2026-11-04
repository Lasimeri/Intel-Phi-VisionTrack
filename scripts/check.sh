#!/usr/bin/env bash
# check.sh: every gate. Host reference programs, then on the card: the vector operations, the image kernels and LK against the host's lane model, then every scene against its
# tolerances and its replay, the message-exception paths, the format handshake, soft configuration, and two pipelines against one. Exit 0 all passed, 1 a failure, 77 card down.
# Writes logs/check/STAMP.txt. Leaves vtrackd stopped and the card's huge pages as they were.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$vt_root"
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p logs/check
log="logs/check/$stamp.txt"
exec > >(tee "$log") 2>&1
vt_stamp
s=$(mktemp -d)
bench_begin
cleanup() {
    scripts/vtrackd.sh stop >/dev/null 2>&1 || true
    # A live window is waiting for the card: give it a program to talk to.
    if pgrep -x vt_view >/dev/null; then scripts/vtrackd.sh start 1024 >/dev/null 2>&1 || true; fi
    bench_end
    rm -rf "$s"
}
trap cleanup EXIT

same() { # same NAME EXPECTED_LINES FILE_A FILE_B
    if [[ $(wc -l <"$3") == "$2" && $(wc -l <"$4") == "$2" ]] && cmp -s "$3" "$4"; then pass "$1 ($2 lines identical)"; else fail "$1"; diff "$3" "$4" | head -4; fi
}

out/host/m0_ref >"$s/m0h"
for wh in 640,480 1280,720 1920,1080; do
    out/host/m1_ref "${wh%,*}" "${wh#*,}" >"$s/m1h-$wh" || fail "host m1_ref $wh: reference and lane model differ"
done
out/host/m2_ref 640 480 >"$s/m2h"
pass "host reference programs"
if out/host/vt_ncc_check >"$s/ncc" 2>&1; then pass "appearance correlation: nearly blank windows never match, never read over 1: $(cut -c1-90 "$s/ncc")"; else fail "appearance correlation"; head -3 "$s/ncc"; fi

if ! card_up; then
    skip "the card is not reachable"
    verdict
fi

scripts/card_run.sh -p 16 vt_m0 2 >"$s/m0c"
/usr/bin/grep -aE '^(vxop|subnormal|fma_hash)' "$s/m0c" >"$s/m0cf" || true
same "M0 vector operations, card = lane model" 20 "$s/m0cf" "$s/m0h"
for wh in 640,480 1280,720 1920,1080; do
    scripts/card_run.sh -p 96 vt_m1 "${wh%,*}" "${wh#*,}" 2 >"$s/m1c" || fail "vt_m1 $wh exit"
    /usr/bin/grep -aE '^kern' "$s/m1c" >"$s/m1cf" || true
    /usr/bin/grep -aE '^kern' "$s/m1h-$wh" >"$s/m1hf" || true
    same "M1 kernels $wh, card = host" 8 "$s/m1cf" "$s/m1hf"
done
scripts/card_run.sh -p 128 vt_m2 640 480 >"$s/m2c" || fail "vt_m2 exit"
/usr/bin/grep -aE '^lk_(hash|first)' "$s/m2c" >"$s/m2cf" || true
same "M2 LK, card = lane model" 20 "$s/m2cf" "$s/m2h"

# The depth map and the still pictures, over real frames of the room. They stay out of the repository (they may show the user), so this gate runs where they exist.
m3_seq=input/calib/sets/010917
m3_one=input/calib/sets/pose3-015721-one
if [[ -f $m3_seq-0-d.pgm && -f $m3_one-d.pgm && -f input/calib/current.txt ]]; then
    out/host/m3_ref "$m3_one-d.pgm" "$m3_one-r.pgm" "$m3_one-l.pgm" input/calib/current.txt --views 2 --planes 64 --near 0.8 --far 8 --seq "$m3_seq" 8 --write "$s/m3.bin" >"$s/m3h" 2>&1 || fail "m3_ref exit"
    phic put "$s/m3.bin" "$vt_card_dir/m3.bin" >/dev/null
    scripts/card_run.sh -p 256 vt_m3 "$vt_card_dir/m3.bin" 3 >"$s/m3c" || fail "vt_m3 exit"
    phic run rm -f "$vt_card_dir/m3.bin" >/dev/null
    /usr/bin/grep -aE '^(depth_hash|still_hash|still_frame|att_last|att_hash)' "$s/m3c" >"$s/m3cf" || true
    /usr/bin/grep -aE '^(depth_hash|still_hash|still_frame|att_last|att_hash)' "$s/m3h" >"$s/m3hf" || true
    same "M3 still pictures, attitude and depth map over 9 real frames, card = host" 15 "$s/m3cf" "$s/m3hf"
else
    skip "M3 over real frames: they are not on this machine"
fi

scripts/vtrackd.sh stop >/dev/null 2>&1 || true
scripts/vtrackd.sh start 1024 >/dev/null

for n in translate shapes camera brightness noise occlusion loss leave helix linear3d glitch repeat flat bigbox smallbox stillstart rotate swing hold offbox jointed offjointed passby dart; do
    if scripts/scene.sh "$n" 640x480 --gate >"$s/scene" 2>&1; then pass "scene $n: tolerances and replay"; else fail "scene $n"; /usr/bin/grep -aE 'GATE MISS|DIFFERS' "$s/scene" | head -6; fi
done
if scripts/scene.sh comeback 640x480 --frames 240 --gate >"$s/scene" 2>&1; then pass "scene comeback, past any patience and back: tolerances and replay"; else fail "scene comeback"; /usr/bin/grep -aE 'GATE MISS|DIFFERS' "$s/scene" | head -6; fi
# Generated frames between the real ones: the same tolerances on the real frames, card = host on every result.
for n in translate occlusion loss bigbox smallbox stillstart rotate swing hold offbox jointed offjointed passby dart; do
    if scripts/scene.sh "$n" 640x480 --gate --framegen >"$s/scene" 2>&1; then pass "scene $n with generated frames: tolerances and replay"; else fail "scene $n with generated frames"; /usr/bin/grep -aE 'GATE MISS|DIFFERS' "$s/scene" | head -6; fi
done
if scripts/scene.sh comeback 640x480 --frames 240 --gate --framegen >"$s/scene" 2>&1; then pass "scene comeback with generated frames: tolerances and replay"; else fail "scene comeback with generated frames"; /usr/bin/grep -aE 'GATE MISS|DIFFERS' "$s/scene" | head -6; fi
if scripts/scene.sh dual 640x480 --dual --gate --framegen >"$s/scene" 2>&1; then pass "scene dual with generated frames: tolerances and replay"; else fail "scene dual with generated frames"; /usr/bin/grep -aE 'GATE|replay' "$s/scene" | head -6; fi
if scripts/scene.sh dualbox 640x480 --dual --gate --framegen >"$s/scene" 2>&1; then pass "scene dualbox with generated frames: tolerances and replay"; else fail "scene dualbox with generated frames"; /usr/bin/grep -aE 'GATE|replay' "$s/scene" | head -6; fi
if scripts/scene.sh dual 640x480 --dual --gate >"$s/scene" 2>&1; then pass "scene dual, two independently tracked views: tolerances and replay"; else fail "scene dual"; /usr/bin/grep -aE 'GATE|replay' "$s/scene" | head -6; fi
if scripts/scene.sh dualbox 640x480 --dual --gate >"$s/scene" 2>&1; then pass "scene dualbox, two boxes drawn by hand over a still thing: tolerances and replay"; else fail "scene dualbox"; /usr/bin/grep -aE 'GATE|replay' "$s/scene" | head -6; fi
# L and R's static regions: chosen from motion and texture, against pictures whose answers are known.
out/host/autoref_check >"$s/ar" 2>&1 && pass "static references: every check" || { fail "static references"; /usr/bin/grep -a FAIL "$s/ar" | head -4; }

# The orange vectors: drawn at the upscaled size, down with FSR's tables, card = host; where they land, what is left after the next erase, and that erasing beats clearing, on the card's own clock.
out/host/vd_ref >"$s/vdh" 2>&1 && pass "orange vectors: every check on the host" || { fail "orange vectors on the host"; /usr/bin/grep -a FAIL "$s/vdh" | head -4; }
scripts/card_run.sh -p 256 vt_vd >"$s/vdc" 2>&1 && pass "orange vectors: every check on the card" || { fail "orange vectors on the card"; /usr/bin/grep -a FAIL "$s/vdc" | head -4; }
if [[ "$(/usr/bin/grep -a down_hash "$s/vdh")" == "$(/usr/bin/grep -a down_hash "$s/vdc")" && -n "$(/usr/bin/grep -a down_hash "$s/vdc")" ]]; then pass "orange vectors: card = host, bit for bit"; else fail "orange vectors: card and host differ"; fi
# FSR on camera D's path: 720p class up, flow between the pictures, Lanczos down. Card = host on every picture. What it looks like is the operator's to judge; this only catches a broken chain: on a noisy still thing it must take noise out.
# FSR on camera D's path: 720p class up, flow between the pictures, Lanczos down. Card = host on every picture. What it looks like is the operator's to judge; this only catches a broken chain: on a noisy still thing it must take noise out.
for n in hold bigbox; do
    if scripts/scene.sh "$n" 640x480 --frames 60 --fsr 960x720 --fsr-up 1280x960 --fsr-sharp 0.4 >"$s/scene" 2>&1; then pass "$n through FSR, up, flow, Lanczos down: replay identical"; else fail "$n through FSR"; /usr/bin/grep -aE 'replay|refused|no room' "$s/scene" | head -4; fi
done
out/host/vt_replay hold 640x480 --frames 80 --fsr 960x720 --fsr-up 1280x960 --fsr-sharp 0.4 >/dev/null 2>"$s/fq" || true
gain=$(/usr/bin/grep -a fsrcheck "$s/fq" | sed -E 's/.*whole FSR ([0-9.]+) Lanczos ([0-9.]+).*/\1 \2/' | awk '{print $1 - $2}')
if awk -v g="${gain:-0}" 'BEGIN{exit !(g >= 6)}'; then pass "FSR takes noise out of a still scene: ${gain} dB over a plain Lanczos against the clean truth"; else fail "FSR noise reduction ${gain:-none} dB, under 6"; fi
# A generated picture is for the eye: a real row is the same to the bit with generation on or off, on the host's lane model, which the scenes above prove equal to the card.
for spec in "bigbox" "dart" "rotate" "smallbox" "jointed" "dual --dual" "dualbox --dual" "comeback --frames 240"; do
    n=${spec%% *}; a=(); [[ $spec == *" "* ]] && read -r -a a <<<"${spec#* }"
    out/host/vt_replay "$n" 640x480 "${a[@]}" >"$s/plain" 2>/dev/null || true
    out/host/vt_replay "$n" 640x480 "${a[@]}" --framegen 2>/dev/null | /usr/bin/grep -av g >"$s/gen" || true
    if [[ -s $s/plain ]] && cmp -s "$s/plain" "$s/gen"; then pass "$n: real rows identical with generation on and off ($(wc -l <"$s/plain") rows)"; else fail "$n: generation changed a real row"; fi
done
if scripts/scene.sh translate 640x480 --mono --gate >"$s/scene" 2>&1; then pass "scene translate, mode A: tolerances and replay"; else fail "scene translate, mode A"; /usr/bin/grep -aE 'GATE MISS|DIFFERS' "$s/scene" | head -6; fi
if /usr/bin/grep -aq '"metric_frames":0,' "$s/scene"; then pass "mode A reports no metric output"; else fail "mode A reported metric output"; fi

# Whatever the team shape and pipeline count, the same tracks.
if scripts/scene.sh linear3d 640x480 --tpc 3 >"$s/scene" 2>&1; then pass "3 threads a core: replay identical"; else fail "3 threads a core"; fi
if scripts/scene.sh linear3d 640x480 --pipes 2 --pyr-cores 20 >"$s/scene" 2>&1; then pass "2 pipelines: replay identical"; else fail "2 pipelines"; /usr/bin/grep -a DIFFERS "$s/scene"; fi

# A malformed message is dropped, counted and named, and every good frame still comes back.
for k in 1 2 3 4 5; do
    out/host/vt_run translate 640x480 --frames 60 --mono --bad-at 30 --bad-kind "$k" >"$s/bad" 2>&1 || true
    if /usr/bin/grep -aq "\"results\":60," "$s/bad" && /usr/bin/grep -aq "\"exceptions\":1,\"last_exception\":$k," "$s/bad"; then pass "exception kind $k: dropped, reported, 60 results"; else fail "exception kind $k"; head -1 "$s/bad" | cut -c1-240; fi
done
out/host/vt_run translate 640x480 --frames 60 --mono --bad-route 20 >"$s/bad" 2>&1 || true
if /usr/bin/grep -aq '"results_with_exceptions":1' "$s/bad"; then pass "a stereo destination in a mono session: ignored and reported"; else fail "route exception"; fi
out/host/vt_run translate 640x480 --frames 60 --mono --sc 20:no_such_key:1 >"$s/bad" 2>&1 || true
if /usr/bin/grep -aq '"results_with_exceptions":1' "$s/bad"; then pass "an unknown soft-configuration key: ignored and reported"; else fail "soft-config exception"; fi

# The handshake refuses a program with another wire format, with its reason; the next session still works.
if VT_TEST_BAD_FORMAT=1 out/host/vt_run translate 640x480 --frames 10 --mono >"$s/fmt" 2>&1; then fail "a wrong wire format was accepted"; else
    if /usr/bin/grep -aq 'wire format' "$s/fmt"; then pass "wrong wire format refused, with its reason"; else fail "refusal without its reason"; cat "$s/fmt" | head -3; fi
fi
if out/host/vt_run translate 640x480 --frames 10 --mono >/dev/null 2>&1; then pass "next session after a refusal runs"; else fail "next session after a refusal"; fi

# Soft configuration changes what runs at the frame it names; the replay gets the same stream, so it must agree.
if scripts/scene.sh linear3d 640x480 --sc 40:max_features:40 --sc 80:fb_max:0 >"$s/scene" 2>&1; then pass "soft configuration mid-stream: replay identical"; else fail "soft configuration mid-stream"; fi
latest=$(ls -t results/runs/*-linear3d.csv | head -1)
over=$(awk -F, 'NR>1 && $2>=41 && $6>40 {n++} END {print n+0}' "$latest")
if [[ $over == 0 ]]; then pass "soft configuration took effect: no frame after 40 holds over 40 features"; else fail "max_features ignored in $over frames"; fi


verdict
