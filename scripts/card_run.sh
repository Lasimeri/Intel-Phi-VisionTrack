#!/usr/bin/env bash
# card_run.sh [-p PAGES] PROG [ARGS...]: runs out/card/PROG on the card: reserves 2 MiB pages, copies the program to /tmp/vt, runs it with the arguments, and puts the old reservation back.
# Refuses over 5,722 pages (the user's 12 GB for the card) before touching anything. Prints the program's output and its exit code as "rc N". A live window gives up the card for the run.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
pages=64
while getopts "p:" o; do
    case $o in
        p) pages=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
prog=$1
shift
if ((pages > 5722)); then
    echo "card_run: refused: $pages pages of 2 MiB are over the 12 GB maximum" >&2
    exit 3
fi
held=${VT_BENCH_HELD:-}
bench_begin
was=$(phic run cat /proc/sys/vm/nr_hugepages)
phic run sh -c "echo $((was + pages)) > /proc/sys/vm/nr_hugepages" >/dev/null
cleanup() {
    phic run sh -c "echo $was > /proc/sys/vm/nr_hugepages" >/dev/null
    [[ -n $held ]] || bench_end
}
trap cleanup EXIT
card_put "$vt_root/out/card/$prog" 2>&1 | /usr/bin/grep -v '^phictl put' || true
phic run sh -c "cd $vt_card_dir && ./$prog $*; echo rc \$?"
