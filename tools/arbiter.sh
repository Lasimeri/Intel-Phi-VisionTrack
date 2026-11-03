#!/usr/bin/env bash
# One focal length: L-R solved at it, D fixed by the beacons, the tightest fixes refined, the best score reported.
fb=$1; S=$2; P=$3
out/host/vt_calib input/calib/lr-pose3.txt --wa 1920 --ha 1080 --wb 1920 --hb 1080 --fb "$fb" --prior 40 --scale 13 14 0.460 --out "$S/lr-$fb.txt" > /dev/null 2>&1
rm -f "$S/arb-$fb"-*.txt
timeout 60 out/host/vt_align "$P-r.pgm" "$P-l.pgm" "$P-d.pgm" "$S/lr-$fb.txt" input/calib/triples-pose3.txt --fd 1386 --fix --out "$S/arb-$fb" > /dev/null 2>&1
best=0; bf=""
for f in "$S/arb-$fb"-*.txt; do
    [[ -f $f ]] || continue
    c=$(head -1 "$f" | awk '{print $NF}')
    awk -v c="$c" 'BEGIN{exit !(c<0.004)}' || continue
    r=$(timeout 20 out/host/m3_ref "$P-d.pgm" "$P-r.pgm" "$P-l.pgm" "$f" --views 1 --planes 64 --near 0.8 --far 8 | awk '/^textured/{print $(NF-1)}')
    if awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}'; then best=$r; bf=$f; fi
done
[[ -n $bf ]] || { echo "fb $fb: no tight fix"; exit 0; }
s=$(timeout 200 out/host/m3_ref "$P-d.pgm" "$P-r.pgm" "$P-l.pgm" "$bf" --views 2 --planes 64 --near 0.8 --far 8 --rounds 6 --refine "$S/arb-best-$fb.cal" --pgm "$S/arb-best-$fb" | awk '/^refined/{r=$3" from "$5} /round 5/{f=$NF} /^textured/{t=$(NF-1)} END{print r", two-view "t" %, fd "f}')
b=$(sed -n '1p' "$S/arb-best-$fb.cal" >/dev/null; awk 'NR==8{printf "%.3f", sqrt($1*$1+$2*$2+$3*$3)}' "$S/lr-$fb.txt")
echo "fb $fb (baseline $b m): score $s"
