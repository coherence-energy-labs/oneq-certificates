#!/usr/bin/env python3
"""Measure retrospective certificate availability from verified gap traces.

This is an analysis of recorded local times, not a hard-timeout executor or a
claim about hardware control deadlines. Model setup, training and sampling
costs remain separate from the per-shot reference pipeline.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from replay_corrected_gaps import percentile, verify


def timing(values):
    return None if not values else {"n": len(values), "mean_ms": sum(values) / len(values) / 1e6,
                                    "p95_ms": percentile(values, 95) / 1e6,
                                    "max_ms": max(values) / 1e6}


def analyze(path, budgets_ms):
    replay = verify(path)
    document = json.loads(path.read_text())
    records = [json.loads(line) for line in gzip.decompress(
        (path.parent / document["trace"]["path"]).read_bytes()).splitlines()]
    rows = []
    for summary in document["rows"]:
        shots = [r for r in records if r["phase"] == "HELDOUT"
                 and r["distance"] == summary["distance"] and r["syndrome"]]
        accepted_times, refused_times = [], []
        for shot in shots:
            reference = shot["reference"]
            mwpm = shot["candidates"]["mwpm"]
            elapsed = (mwpm["decode_ns"] + mwpm["projection_ns"]
                       + reference["certify_including_internal_checks_ns"]
                       + reference["retained_witness_recheck_ns"])
            (accepted_times if reference["accepted_exact"] else refused_times).append(elapsed)
        availability = [{"budget_ms": budget,
                         "exact_reference_within_observed_budget": sum(t <= budget * 1_000_000 for t in accepted_times),
                         "nontrivial_attempts": len(shots),
                         "fraction": sum(t <= budget * 1_000_000 for t in accepted_times) / len(shots)
                                      if shots else None}
                        for budget in sorted(budgets_ms)]
        rows.append({"distance": summary["distance"], "nontrivial": len(shots),
                     "accepted": timing(accepted_times), "refused": timing(refused_times),
                     "training_seconds": summary["training"]["elapsed_ns"] / 1e9,
                     "retrospective_availability": availability})
    return {"schema": "oneq-corrected-gap-operating-cost/1", "verified_input": str(path.name),
            "input_trace_sha256": document["trace"]["sha256"], "input_replay": replay,
            "primary_bp_baseline": "bp_osd_train_selected",
            "strong_graph_baseline": "mwpm",
            "pipeline_cost": "MWPM decode + exact projection + certification including internal checks + retained-witness recheck",
            "exclusions": ["model construction", "TRAIN tuning (reported separately)", "syndrome sampling", "artifact serialization"],
            "claim_limit": "Retrospective local wall times; no hard interruption, live QPU, or deadline guarantee.",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budgets-ms", nargs="+", type=int, default=[1, 5, 10, 25, 50, 100, 250])
    args = parser.parse_args()
    if any(budget < 0 for budget in args.budgets_ms) or len(set(args.budgets_ms)) != len(args.budgets_ms):
        parser.error("budgets must be distinct nonnegative integer milliseconds")
    if args.out.exists():
        parser.error("output exists; choose a new path")
    result = analyze(args.summary, args.budgets_ms)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
