r"""Logical failures of each decoder on paper 1's preregistered gap run (final referee, MINOR 6, 2026-11-03).

    python -B experiments/m0_prior_art/gap_distribution_logical.py
      -> evidence/gap_distribution_v4/prereg_2026-11-03/logical_failures.json

Sec. VIII measures each decoder's excess WEIGHT over the minimum. Belief-matching does not aim at the minimum-weight
correction under the published weights (it re-weights the graph from belief propagation), so its excess measures how
far its objective is from the one the reference minimises, not how often it fails. This reports the other quantity
on the same shots: a correction fails when the parity of the observable over its edge set (each matching-graph edge
carrying PyMatching's own observable label, read from the graph the run decoded on) differs from the shot's recorded
`observable_flip`. Every decoder is scored on its projected edge set, the object Sec. VIII weighs. Only nontrivial
shots are recorded by the run; on a trivial shot every decoder predicts no flip.

Nothing is decoded again: the per-shot record keeps each decoder's edge set and the true flip.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "evidence" / "gap_distribution_v4" / "prereg_2026-11-03"
OUT = RUN / "logical_failures.json"
ARMS = ("pymatching", "bposd_merged", "bposd_component", "belief_matching")
B = -1


def key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def predicted_flip(edge_set, flips_of: dict) -> bool:
    """Observable parity of an edge set: XOR of its edges' observable labels."""
    out = False
    for u, v in edge_set:
        out ^= flips_of[key(int(u), int(v))]
    return out


def score(recs: list[dict], flips_of: dict) -> dict:
    fails = {a: 0 for a in ARMS}
    for r in recs:
        for a in ARMS:
            fails[a] += predicted_flip(r[a]["edges"], flips_of) != r["observable_flip"]
    return {"nontrivial": len(recs), "true_flips": sum(r["observable_flip"] for r in recs), "logical_failures": fails}


def component_check(dem, flips_of: dict) -> dict:
    """ASSERT what scoring by edge labels assumes (final referee, 2026-11-03), on every component of every error of the
    decomposed model: a graphlike component (one or two detectors) flips the observable exactly when PyMatching's
    label on its edge does, and a detector-free component flips no observable. A decoder that selected a component
    whose own observable differed from its edge's label would be scored wrongly; this refuses rather than scores."""
    n = mism = free_flip = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        group: list = []
        for t in list(inst.targets_copy()) + [None]:
            if t is not None and not t.is_separator():
                group.append(t)
                continue
            dets = [int(x.val) for x in group if x.is_relative_detector_id()]
            obs = sorted(int(x.val) for x in group if x.is_logical_observable_id())
            group = []
            if not dets:
                free_flip += bool(obs)
                continue
            if len(dets) > 2:
                raise SystemExit(f"a decomposed component touches {len(dets)} detectors: not graphlike")
            n += 1
            k = key(dets[0], dets[1] if len(dets) == 2 else B)
            if k not in flips_of:
                raise SystemExit(f"component {k} has no edge in the matching graph")
            mism += (0 in obs) != flips_of[k]
    if mism or free_flip:
        raise SystemExit(f"{mism} component(s) whose observable differs from their edge's label, {free_flip} "
                         f"detector-free component(s) that flip an observable: scoring by edge labels would be wrong")
    return {"components_checked": n, "label_mismatches": 0, "detector_free_components_flipping_an_observable": 0}


def edge_observables(d: int, p: float) -> tuple[dict, dict]:
    import pymatching
    import stim
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = circ.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    flips = {key(int(u), B if v is None else int(v)): (0 in set(attr.get("fault_ids", ()))) for u, v, attr in m.edges()}
    return flips, component_check(dem, flips)


def main() -> int:
    summ = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for row in summ["rows"]:
        body = gzip.decompress((ROOT / row["per_shot"]).read_bytes())
        if hashlib.sha256(body).hexdigest() != row["per_shot_sha256"]:
            raise SystemExit(f"d={row['distance']}: the per-shot record does not hash to its summary")
        recs = [json.loads(x) for x in body.decode().splitlines()]
        flips, checked = edge_observables(row["distance"], row["p"])
        res = score(recs, flips)
        rows.append({"distance": row["distance"], **res, "assumption_checked": checked})
        print(row["distance"], json.dumps(res), flush=True)
    doc = {"schema": "oneq-gap-logical-failures/1", "run": summ["head"], "rows": rows,
           "rule": "failure = observable parity of the decoder's projected edge set != the shot's observable_flip; "
                   "edge observables from PyMatching's graph of the run's own circuit; nontrivial shots only",
           "belief_matching_settings": "BeliefMatching(dem, max_bp_iters=20) (gap_distribution.py)"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
