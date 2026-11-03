#!/usr/bin/env bash
# vtrackd.sh start [PAGES] | stop | log: the card program, detached so it does not hold the control socket. Start reserves 2 MiB pages (default 512 = 1 GiB, refused over 5,722 = the user's
# 12 GB) and remembers the old count; stop asks it to quit through the region and puts the count back. Only /tmp/vt on the card is written.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
state="$vt_root/out/run"
case "${1:-}" in
    start)
        pages=${2:-512}
        if ((pages > 5722)); then
            echo "vtrackd.sh: refused: $pages pages are over the 12 GB maximum" >&2
            exit 3
        fi
        mkdir -p "$state"
        was=$(phic run cat /proc/sys/vm/nr_hugepages)
        [[ -f $state/hugepages.was ]] || echo "$was" >"$state/hugepages.was"
        base=$(cat "$state/hugepages.was")
        phic run sh -c "echo $((base + pages)) > /proc/sys/vm/nr_hugepages" >/dev/null
        have=$(phic run cat /proc/sys/vm/nr_hugepages)
        card_put "$vt_root/out/card/vtrackd" >/dev/null
        # The binary's own time goes with it, so a window can say which code is running.
        built=$(date -r "$vt_root/out/card/vtrackd" +%H%M%S)
        phic run sh -c "cd $vt_card_dir && setsid ./vtrackd $built > vtrackd.log 2>&1 < /dev/null &" >/dev/null
        sleep 1
        echo "vtrackd started on the card: nr_hugepages $have (was $base), code of $built"
        phic run cat "$vt_card_dir/vtrackd.log"
        ;;
    stop)
        "$vt_root/out/host/vt_quit" || true
        # It finishes its session first, which under load takes seconds. A new binary over a running one is refused: "Text file busy".
        for _ in {1..40}; do
            [[ -z $(phic run sh -c 'pgrep vtrackd' 2>/dev/null || true) ]] && break
            sleep 0.5
        done
        if [[ -n $(phic run sh -c 'pgrep vtrackd' 2>/dev/null || true) ]]; then
            echo "vtrackd.sh: still running 20 s after being asked: stopping it" >&2
            phic run sh -c 'pkill vtrackd' >/dev/null 2>&1 || true
            sleep 1
        fi
        if [[ -f $state/hugepages.was ]]; then
            base=$(cat "$state/hugepages.was")
            phic run sh -c "echo $base > /proc/sys/vm/nr_hugepages" >/dev/null
            rm -f "$state/hugepages.was"
            echo "card: nr_hugepages back to $(phic run cat /proc/sys/vm/nr_hugepages)"
        fi
        ;;
    log)
        phic run cat "$vt_card_dir/vtrackd.log"
        ;;
    *)
        echo "usage: vtrackd.sh start [PAGES] | stop | log" >&2
        exit 2
        ;;
esac
