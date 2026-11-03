#!/usr/bin/env bash
# live.sh [vt_view options]: the live window, kept current with the code. Starts vtrackd on the card and vt_view on the desktop; whenever make replaces out/card/vtrackd it restarts vtrackd
# (the window stays up, shows a banner, and takes its region back); whenever make replaces out/host/vt_view it restarts the window, where the user last put it (vt_view keeps out/run/view.pos
# current). Ends when the window is closed, and stops vtrackd.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$vt_root"
export DISPLAY="${DISPLAY:-:0}"
mkdir -p logs/live out/run
log="logs/live/$(date +%Y%m%d-%H%M%S).log"
echo "live: log $log"
scripts/vtrackd.sh stop >/dev/null 2>&1 || true
scripts/vtrackd.sh start 512 >>"$log" 2>&1
at=""
args=()
while (($#)); do
    if [[ $1 == --at ]]; then
        at=$2
        shift 2
    else
        args+=("$1")
        shift
    fi
done
start_view() {
    out/host/vt_view ${at:+--at "$at"} "${args[@]}" >>"$log" 2>&1 &
    view_pid=$!
    echo "live: vt_view pid $view_pid at ${at:-a place the window manager picks}"
}
start_view
card_t=$(stat -c %Y out/card/vtrackd)
view_t=$(stat -c %Y out/host/vt_view)
# A new binary counts once it has stopped changing: the audit runs after the link.
settled() { (($(date +%s) - $1 >= 2)); }
while kill -0 "$view_pid" 2>/dev/null; do
    sleep 1
    [[ -f $LOCK_FILE ]] && continue
    t=$(stat -c %Y out/card/vtrackd 2>/dev/null || echo 0)
    if [[ $t != 0 && $t != "$card_t" ]] && settled "$t"; then
        card_t=$t
        echo "live: new card code, restarting vtrackd on the card ($(date +%T))" | tee -a "$log"
        scripts/vtrackd.sh stop >>"$log" 2>&1 || true
        scripts/vtrackd.sh start 512 >>"$log" 2>&1
    fi
    t=$(stat -c %Y out/host/vt_view 2>/dev/null || echo 0)
    if [[ $t != 0 && $t != "$view_t" ]] && settled "$t"; then
        view_t=$t
        echo "live: new host code, restarting the window ($(date +%T))" | tee -a "$log"
        [[ -s out/run/view.pos ]] && at=$(cat out/run/view.pos)
        kill "$view_pid" 2>/dev/null || true
        wait "$view_pid" 2>/dev/null || true
        start_view
    fi
done
scripts/vtrackd.sh stop >>"$log" 2>&1 || true
echo "live: window closed, vtrackd stopped"
