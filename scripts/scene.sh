#!/usr/bin/env bash
# scene.sh NAME WxH [vt_run options]: one scene through the card, then its replay on the desktop host, compared frame by frame. Writes results/runs/STAMP-NAME.{csv,json} and prints the summary.
# Exit 1 if the replay differs: tracks must be bit-identical, at any team size and pipeline count. A run that uses the x87 kernels or may drop frames has nothing to compare with. Requires
# vtrackd running on the card. A live window gives up the card for the run.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$vt_root"
name=$1 size=$2
shift 2
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p results/runs
hc=$(mktemp) hh=$(mktemp)
held=${VT_BENCH_HELD:-}
bench_begin
cleanup() {
    [[ -n $held ]] || bench_end
    rm -f "$hc" "$hh"
}
trap cleanup EXIT
frames=150
perframe=1
args=("$@")
compare=1
generated=0
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --frames) frames=${args[$((i + 1))]} ;;
        --variant | --alternate | --drop | --live | --bad-at) compare=0 ;;
        --dual) perframe=2 ;;
        --framegen) generated=1 ;;
    esac
done
frames=$((frames * perframe))
out="results/runs/$stamp-$name"
rcard=0
out/host/vt_run "$name" "$size" --hash "$hc" --csv "$out.csv" --json "$out.json" "$@" || rcard=$?
if ((compare == 0)); then
    echo "replay: nothing to compare with for these options"
elif { rreplay=0; out/host/vt_replay "$name" "$size" "$@" >"$hh" || rreplay=$?; ((rreplay <= 1)); } && { ((generated)) && (($(wc -l <"$hc") >= frames)) || [[ $(wc -l <"$hc") == "$frames" ]]; } && [[ $(wc -l <"$hh") == $(wc -l <"$hc") ]] && cmp -s "$hc" "$hh"; then
    echo "replay: IDENTICAL, $frames frames, card MVEX = host one-thread lane model"
else
    echo "replay: DIFFERS"
    diff "$hc" "$hh" | head -4
    exit 1
fi
exit "$rcard"
