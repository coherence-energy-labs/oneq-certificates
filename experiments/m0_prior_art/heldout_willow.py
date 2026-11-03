r"""Paper 1's held-out Willow sample: the frozen base ladder on 50-round configurations no generation ever saw.

    python experiments/m0_prior_art/heldout_willow.py --workers 3      # certify, every receipt kept
    python experiments/m0_prior_art/heldout_willow.py --merge          # summaries + the pooled result

The protocol is fixed in docs/PREREG_heldout_willow_2026-11-02.md, committed before this ran. This file adds four
configurations to the epsilon rerun's own machinery (experiments/m0_prior_art/epsilon_rerun.py: the same chunking,
producer call, checker, receipt format and forgery battery) under names prefixed `heldout_`, so nothing that reads
the rerun's `willow_*` results sees them, and runs them through the rerun's own `run` and `merge`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
import epsilon_rerun as er  # noqa: E402

HELDOUT = ("d3_at_q4_5_X_r50", "d3_at_q4_5_Z_r50", "d5_at_q8_7_X_r50", "d5_at_q8_7_Z_r50")
SHOTS, CHUNK = 5000, 250
OUT = ROOT / "evidence" / "epsilon_rerun" / "heldout_willow_r50.json"
PREREG = "docs/PREREG_heldout_willow_2026-11-02.md"

for _c in HELDOUT:
    er.CONFIGS[f"heldout_{_c}"] = {"kind": "willow", "corpus_config": _c, "chunk": CHUNK,
                                   "seeds": (1, SHOTS // CHUNK), "method": "certify",
                                   "reproduces": f"held out, preregistered in {PREREG}"}


def cp95(k: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    names = [f"heldout_{c}" for c in HELDOUT]
    if not a.merge:
        for nm in names:
            er.run(nm, a.workers)
        return 0
    rows = {}
    for nm in names:
        er.merge(nm)
        s = json.loads((er.INDEX_DIR / nm / "summary.json").read_text(encoding="utf-8"))
        rows[nm] = {k: s[k] for k in ("shots", "nontrivial", "accepted", "accepted_rate", "largest_accepted_epsilon",
                                      "largest_accepted_epsilon_float", "forgeries_attempted", "forgeries_accepted",
                                      "chunk_failures_retried")}
        rows[nm]["exact_optimal"] = s["status"].get("EXACT_OPTIMAL", 0) if isinstance(s.get("status"), dict) else None
        rows[nm]["not_proven"] = s["status"].get("NOT_PROVEN", 0) if isinstance(s.get("status"), dict) else None
    n = sum(r["nontrivial"] for r in rows.values())
    k = sum(r["accepted"] for r in rows.values())
    lo, hi = cp95(k, n)
    out = {"schema": "oneq-heldout-willow/1", "preregistration": PREREG,
           "configurations": rows, "pooled": {"nontrivial": n, "accepted": k, "rate": k / n,
                                             "cp95_two_sided": [lo, hi]},
           "forgeries_accepted_total": sum(r["forgeries_accepted"] for r in rows.values())}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["pooled"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
