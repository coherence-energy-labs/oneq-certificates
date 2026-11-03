#!/usr/bin/env bash
# R198: the 24 time-shift certificates of the 3-colour cycle (weight <= 10, anchors in rounds 0-1, T = 1..12,
# memory-Z and memory-X) -- the step that turns "d_circ = 11 for every T" from MEASURED into MACHINE-CHECKED.
#
# Resumable: a (basis, T) cell already VERIFIED in out/cert_shift.jsonl is skipped. Cooperative: before each cell,
# while out/PAUSE exists (another session's wall), it waits; the cell in progress always finishes first. Each cell
# compiles the engine from THIS lane's sources, so a cell runs the code of the checkout it was started in.
# Launch it DETACHED (launch_detached.ps1) so it outlives the Claude session that starts it; finish with
# `python finish_3c_certs.py --import` once 24/24 are verified. Outputs stay in out/ (git-ignored; proofs are
# hundreds of MB and are deleted after their check).
set -u
CAMP="$(cd "$(dirname "$0")" && pwd)"
LANE="$(dirname "$CAMP")"
OUT="$CAMP/out"
THREADS=${THREADS:-20}
mkdir -p "$OUT/work"
cd "$LANE" || exit 2
log() { echo "$(date '+%F %T') $*" >> "$OUT/campaign.log"; }
done_cell() {
  [ -f "$OUT/cert_shift.jsonl" ] && python - "$1" "$2" "$(cygpath -w "$OUT/cert_shift.jsonl")" <<'PY'
import json, sys
b, t, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in open(path):
    r = json.loads(line)
    if r["basis"] == b and r["rounds"] == t and r.get("checker_exit") == 0:
        sys.exit(0)
sys.exit(1)
PY
}
log "time-shift certificates: start (threads $THREADS, pid $$)"
for b in Z X; do
  for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if done_cell $b $t; then continue; fi
    while [ -f "$OUT/PAUSE" ]; do sleep 60; done
    log "start certify shift $b T=$t"
    python run_lane.py certify --circuit 3c --basis $b --rounds $t --w 10 --maxround 1 --threads "$THREADS" \
      --work "$OUT/work" >> "$OUT/cert_shift.jsonl" 2>> "$OUT/errors.log"
    rm -f "$OUT/work"/proof_3c_${b}${t}_w10.bin
    log "done  certify shift $b T=$t"
  done
done
log "time-shift certificates: complete"
