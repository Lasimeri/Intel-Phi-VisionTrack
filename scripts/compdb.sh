#!/usr/bin/env bash
# compdb.sh: writes compile_commands.json and proves it clean: clangd over every .c and .h, 14 at once.
# Every file gets the host flags, so clangd shows the host lane model of vx.h (the same source the KNC clang
# compiles under __KNC__, checked there by -Werror in the Makefile). A diagnostic is a line "E[time] [code]
# Line N: ..." or W[...]. --check-locations=0 skips clangd's hover/tweak pass, whose "tweak ==> FAIL" lines
# are not diagnostics of the code. Exit 0 only when every file reports none.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$vt_root"
db=compile_commands.json
mapfile -t sources < <(find src tests -name '*.c' 2>/dev/null | LC_ALL=C sort)
mapfile -t checked < <(find src tests -name '*.[ch]' 2>/dev/null | LC_ALL=C sort)
{
    echo "["
    for i in "${!sources[@]}"; do
        sep=","
        ((i == ${#sources[@]} - 1)) && sep=""
        printf '  {"directory": "%s", "file": "%s", "arguments": ["clang", "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-Isrc/common", "-Isrc/kern", "-Isrc/card", "-Isrc/host", "-c", "%s"]}%s\n' \
            "$vt_root" "${sources[$i]}" "${sources[$i]}" "$sep"
    done
    echo "]"
} >"$db"
one() {
    local log
    log=$(clangd --check="$1" --check-locations=0 2>&1) || true
    if ! /usr/bin/grep -a -q 'All checks completed' <<<"$log"; then
        printf 'clangd did not finish  %s\n' "$1"
        return 1
    fi
    local d
    d=$(/usr/bin/grep -a -E '^[EW]\[[0-9:.]+\] \[' <<<"$log" || true)
    if [[ -n $d ]]; then
        printf 'clangd %d diagnostics  %s\n' "$(wc -l <<<"$d")" "$1"
        head -5 <<<"$d"
        return 1
    fi
    return 0
}
export -f one
if printf '%s\0' "${checked[@]}" | xargs -0 -P 14 -n 1 bash -c 'one "$0"'; then
    echo "clangd over the tree: ${#checked[@]} files, 0 diagnostics"
else
    echo "clangd over the tree: FAILED (${#checked[@]} files)"
    exit 1
fi
