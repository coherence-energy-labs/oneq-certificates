r"""Merge the million-shot main run with its disjoint-seed supplements.

The main run lost chunks to an ENVIRONMENTAL class (paging-file exhaustion at
scipy import -- recorded verbatim in its artifact), so completion runs with
disjoint seed bases finish the count. This merge is legitimate ONLY because
chunk death is independent of shot content: workers died at spawn, before any
syndrome was seen, so no shot was selected out by its difficulty. That
argument is stated in the merged artifact, where a reader can weigh it.

Counts are summed; the gap distribution is re-summarized from the per-run
summaries conservatively (n-weighted mean; max of maxes; medians reported
per-run because medians do not merge). Nothing is recomputed from trust:
every number here is copied or summed from gate-produced artifacts.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"


def main() -> int:
    parts = []
    for name in sys.argv[1:] or ["million_shot_gate_d5.json",
                                 "million_shot_gate_d5_supplement.json",
                                 "million_shot_gate_d5_supplement2.json"]:
        p = EVID / name
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2
        parts.append((name, json.loads(p.read_text(encoding="utf-8"))))

    keys = ("shots", "nontrivial", "certified", "forgeries_attempted",
            "forgeries_accepted", "failed_chunks",
            "shots_lost_to_failed_chunks")
    tot = {k: sum(d.get(k) or 0 for _n, d in parts) for k in keys}
    gaps = [(d["uncertified_gap"], _n) for _n, d in parts
            if d.get("uncertified_gap")]
    gap_n = sum(g["n"] for g, _ in gaps)
    merged = {
        "schema": "oneq-million-shot-merged/1",
        "constituent_artifacts": [n for n, _ in parts],
        "why_merging_is_legitimate": (
            "lost chunks died at worker SPAWN (paging-file exhaustion at "
            "scipy import), before any syndrome was decoded, so chunk death "
            "is independent of shot content and completion runs with "
            "disjoint seed bases introduce no selection on difficulty"),
        **tot,
        "certified_rate": tot["certified"] / tot["nontrivial"],
        "uncertified_gap_merged": {
            "n": gap_n,
            "nweighted_mean": (sum(g["mean"] * g["n"] for g, _ in gaps)
                               / gap_n if gap_n else 0.0),
            "max_of_maxes": max((g["max"] for g, _ in gaps), default=0.0),
            "per_run_medians": {n: g["median"] for g, n in gaps},
            "note": "medians do not merge; reported per constituent run",
        },
        "per_run": {n: {k: d.get(k) for k in
                        ("shots", "certified", "certified_rate",
                         "forgeries_attempted", "forgeries_accepted")}
                    for n, d in parts},
    }
    out = EVID / "million_shot_gate_d5_merged.json"
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"{tot['shots']:,} shots  {tot['certified']:,} certified "
          f"({100 * merged['certified_rate']:.2f}%)  forgeries "
          f"{tot['forgeries_attempted']:,} attempted, "
          f"{tot['forgeries_accepted']} accepted")
    print(f"evidence -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
