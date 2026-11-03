# lib.sh: sourced by every script (bash). One place for paths, the card verbs, verdict lines, and the
# provenance stamp each run record starts with.
set -euo pipefail
vt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PHI_TREE:=/media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A}"
: "${KNC_CLANG:=$HOME/.cache/intel-phi-llvm-build/llvm/bin/clang}"
: "${KNC_SYSROOT:=$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
: "${KNC_LINK:=$HOME/.cache/intel-phi-7120a/toolchain/clang/knc-cc}"
: "${ISA_AUDIT:=$HOME/.cache/intel-phi-7120a/host/target/release/phi-isa-audit}"
: "${PHI_CARD:=0}"
vt_card_dir=/tmp/vt   # on the card (tmpfs): every file this project puts there, nothing outside it

phic() { phi -c "$PHI_CARD" "$@" </dev/null; }
card_up() { phic run true >/dev/null 2>&1; }

# Copies card programs into $vt_card_dir on the card, one control-socket call per file.
card_put() {
    phic run mkdir -p "$vt_card_dir" >/dev/null
    local f
    for f in "$@"; do
        phic put "$f" "$vt_card_dir/${f##*/}" >/dev/null
    done
    phic run sh -c "chmod +x $vt_card_dir/*" >/dev/null
}

vt_stamp() {
    printf 'date   %s\n' "$(date '+%F %T %Z')"
    printf 'tree   %s (git %s)\n' "$vt_root" "$(git -C "$vt_root" rev-parse --short HEAD 2>/dev/null || echo 'no commits')"
    printf 'stack  %s (git %s)\n' "$PHI_TREE" "$(git -C "$PHI_TREE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    printf 'cc     %s\n' "$("$KNC_CLANG" --version 2>/dev/null | head -1)"
}

n_pass=0 n_fail=0 n_skip=0
pass() { n_pass=$((n_pass + 1)); printf 'PASS  %s\n' "$*"; }
fail() { n_fail=$((n_fail + 1)); printf 'FAIL  %s\n' "$*"; }
skip() { n_skip=$((n_skip + 1)); printf 'SKIP  %s\n' "$*"; }
verdict() {
    printf -- '-- %d passed, %d failed, %d skipped\n' "$n_pass" "$n_fail" "$n_skip"
    ((n_fail == 0)) || exit 1
    ((n_skip == 0)) || exit 77
    exit 0
}

# A benchmark's claim on the card. While the file exists the live window (vt_view) leaves the card alone and shows the camera with a banner, so a measurement never shares the card with
# it. VT_BENCH_HELD stops a nested script from taking or dropping the claim its caller holds.
LOCK_FILE="$vt_root/out/run/bench.lock"
bench_begin() {
    [[ -n ${VT_BENCH_HELD:-} ]] && return 0
    mkdir -p "$vt_root/out/run"
    : >"$LOCK_FILE"
    export VT_BENCH_HELD=1
    if pgrep -x vt_view >/dev/null; then sleep 1; fi
}
bench_end() {
    rm -f "$LOCK_FILE"
}
