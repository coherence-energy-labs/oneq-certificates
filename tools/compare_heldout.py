#!/usr/bin/env python3
r"""Did the held-out re-run reproduce? Compare the SCIENCE, not the clock.

WHY THIS EXISTS. REPRODUCE.md told a stranger that a correct re-run
"reproduces evidence/qldpc_heldout.json exactly". It does not, and it
cannot: the artifact embeds wall-clock timings (median_ms, p95_ms,
lp_ms_per_shot), so a byte comparison fails on any machine whose load
differs from ours -- which is every machine. A reproducer following that
instruction would see a mismatch, and the honest ones would report a
failure that is not one.

Run in a clean clone of HEAD, every difference was timing and every
scientific field was identical: counts, tier splits, exact/open, all
2,019 records across 8 rungs.

So this compares what the claim is actually about and IGNORES what it is
not. Timing keys are listed explicitly rather than pattern-matched --
"anything with ms in the name" would silently excuse a future field whose
name happens to contain it.

    python tools/compare_heldout.py                       # vs git HEAD
    python tools/compare_heldout.py --stored a.json --fresh b.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = "evidence/qldpc_heldout.json"

# Wall-clock only. Every one of these is a measurement OF THE MACHINE,
# not of the code, and none of them appears in any published claim.
TIMING_KEYS = {
    "timing", "seconds", "elapsed_s", "wall_s",
    "lp_ms_per_shot", "ms_per_shot", "cert_ms_per_shot",
    "median_ms", "p95_ms", "max_ms",
}


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items()
                if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip_timing(x) for x in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", default=None,
                    help="default: the copy committed at git HEAD")
    ap.add_argument("--fresh", default=str(ROOT / ART))
    a = ap.parse_args()

    if a.stored:
        stored = json.loads(pathlib.Path(a.stored).read_text(
            encoding="utf-8"))
    else:
        r = subprocess.run(["git", "show", f"HEAD:{ART}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"cannot read {ART} from git HEAD: {r.stderr[:200]}")
            return 1
        stored = json.loads(r.stdout)
    fresh = json.loads(pathlib.Path(a.fresh).read_text(encoding="utf-8"))

    s, f = strip_timing(stored), strip_timing(fresh)
    same = s == f

    sr, fr = s.get("rungs", {}), f.get("rungs", {})
    print(f"held-out comparison: {len(sr)} rung(s) stored, "
          f"{len(fr)} fresh")
    for k in sorted(set(sr) | set(fr)):
        x, y = sr.get(k), fr.get(k)
        if x == y:
            n = (x or {}).get("shots", (x or {}).get("n", "?"))
            print(f"  {k:14s} IDENTICAL   ({n} records)")
        else:
            print(f"  {k:14s} *** DIFFERS ***")
            for fld in sorted(set(x or {}) | set(y or {})):
                if (x or {}).get(fld) != (y or {}).get(fld):
                    print(f"      {fld}: {(x or {}).get(fld)!r} != "
                          f"{(y or {}).get(fld)!r}")

    # A comparison that skipped everything would agree trivially.
    if not sr or not fr:
        print("\nVACUOUS: no rungs to compare")
        return 1

    if same:
        print("\nREPRODUCED: every scientific field identical. "
              "Wall-clock timings differ, as they must on different "
              "hardware, and are excluded by name.")
        return 0
    print("\nNOT REPRODUCED: a field that is not wall-clock differs. "
          "That is a finding and we want it reported.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
