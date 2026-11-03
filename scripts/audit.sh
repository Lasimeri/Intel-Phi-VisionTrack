#!/usr/bin/env bash
# audit.sh BIN: the stack's phi-isa-audit over a card binary. Removes the binary and fails unless the result
# line reads 0 illegal, 0 suspect: a card program that audits dirty never reaches the card.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
out=$("$ISA_AUDIT" "$1" 2>&1 || true)
line=$(/usr/bin/grep -a '^result:' <<<"$out" || true)
printf 'audit %s: %s\n' "${1##*/}" "${line:-no result line}"
if [[ $line != "result: 0 illegal, 0 suspect,"* ]]; then
    /usr/bin/grep -a -m8 -E 'illegal|suspect' <<<"$out" >&2 || true
    rm -f "$1"
    exit 1
fi
