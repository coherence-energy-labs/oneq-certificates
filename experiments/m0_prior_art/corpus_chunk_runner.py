r"""Run ONE chunk of ONE corpus config and write ONE artifact -- the atom of
the resumable pattern.

The measured environment on this workstation kills workers on a schedule
nothing here controls: paging-file exhaustion at spawn, MemoryError on ~1 MiB
allocations when a CI runner or a release build spikes the commit charge. The
million-shot campaign survived it with config-per-process; the escalated
corpus runs lost 113 of 200 and then 78 of 100 chunks to the same class,
because inside one process a dead chunk is dead forever. This runner makes
the CHUNK the unit of survival: it either writes a good artifact or exits
nonzero, and the driver retries whatever is missing until coverage is
literally complete. A crash costs one chunk, one retry -- never a run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-attack", action="store_true")
    a = ap.parse_args()

    import google_corpus_gate as gate

    name, r = gate._work((a.config, a.lo, a.hi, not a.no_attack))
    if r.get("failed_chunk"):
        print(f"chunk {a.lo}:{a.hi} failed: {r['reason']}", file=sys.stderr)
        return 1
    r["rungs"] = dict(r["rungs"])
    r["per_attack"] = dict(r["per_attack"])
    r["per_attack_accepted"] = dict(r["per_attack_accepted"])
    doc = {"config": name, "lo": a.lo, "hi": a.hi, **r}
    out = pathlib.Path(a.out)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(out)                    # atomic: no torn artifacts on a kill
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
