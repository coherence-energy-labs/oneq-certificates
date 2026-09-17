r"""The measured model through the CERTIFIED accept path, held out.

`measured_model_lane` (R23) showed that a decoding graph estimated from the detector statistics
alone beats the calibration DEM 80-200x on held-out logical errors. It scores those errors with
`hardware_noise_floor.score`, a plain matching. The certified pipeline that Phase 7 ran on hardware
(`rep_code_cert.certify_shots`) accepts a correction only with a proof that it is minimum weight
under the DECLARED model, and it has been handed the uniform and the calibrated models, never the
measured one. This lane hands it the measured one.

For each record the model is estimated on the FIRST half of the shots and the SECOND half goes
through `certify_shots(hard=True)` three times -- the record's calibration DEM, the uniform graph,
the measured graph -- with the observed logical (`truth`, the final data register) supplied, so
each pass reports certified rate, logical errors on the declared cut and on the opposite cut. The
measured model's count is also taken from `score()` on the same half, because two decoders that
should agree and are never compared is how a convention drifts (see the cut history in
rep_code_cert.logical_cut_edges).

Preregistered in docs/PREREG_measured_model_program_2026-09-16.md, section A.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ibm_live"))
sys.path.insert(0, str(ROOT / "experiments" / "dual_observable"))

from oneq.dem_measured import MAX_SPAN, measured_graph              # noqa: E402
from measured_model_lane import load                                # noqa: E402
from hardware_noise_floor import score                              # noqa: E402
from rep_code_cert import certify_shots, decode_graph, logical_cut_edges  # noqa: E402

RECORDS = (
    "evidence/rep_d3_r13_kingston.json",
    "evidence/rep_d3_r40_kingston.json",
    "evidence/hardware_noise_floor_fez_d3_20k.json",
    "evidence/rep_d3_r40_fez.json",
    "evidence/rep_d5_r13_fez.json",
    "evidence/rep_d3_r13_fez_nested.json",
)


def one_record(path: str, max_span: int) -> dict:
    doc, dem_edges, det = load(path)
    d, r = doc["circuit"]["distance"], doc["circuit"]["rounds"]
    n_nodes = det.shape[1]
    positions, n_rounds = d - 1, r + 1
    truth = np.array(doc["truth"], dtype=np.uint8)
    half = det.shape[0] // 2
    g = measured_graph(dem_edges, det[:half], positions, n_rounds, n_nodes, max_span=max_span)
    det3 = det[half:].reshape(-1, n_rounds, positions)
    truth2 = truth[half:]

    models = {"dem": list(dem_edges), "uniform": decode_graph(d, r), "measured": list(g.edges)}
    row = {"record": path, "backend": doc.get("backend"), "distance": d, "rounds": r,
           "held_out_shots": int(det3.shape[0]), "fitted_on_shots": int(half),
           "measured_graph_stats": g.stats, "models": {}}
    for name, edges in models.items():
        t0 = time.perf_counter()
        out = certify_shots(det3, d, r, hard=True, edges=edges, observed=truth2)
        out.pop("nonreproducible", None)
        out["edges"] = len(edges)
        out["seconds"] = round(time.perf_counter() - t0, 1)
        row["models"][name] = out
    # the lane's own scorer on the same half and the same measured edges
    cut = logical_cut_edges(d, r)
    _rec, _pr, _g, correct = score(list(g.edges), cut, det[half:], truth2, n_nodes, budget_s=1e-9)
    row["score_measured_errors"] = int(sum(1 for x in correct if not x))
    return row


def verdicts(row: dict) -> dict:
    m = row["models"]
    meas, dem = m["measured"], m["dem"]
    ob = meas["observable"]
    v = {}
    v["A1_certified"] = {"certified_rate": meas["certified_rate"], "nontrivial": meas["nontrivial"],
                         "met": (meas["certified_rate"] >= 0.99) if meas["nontrivial"] >= 100 else None}
    dem_err = dem["observable"]["logical_errors"]
    v["A2_beats_dem"] = {"measured": ob["logical_errors"], "dem": dem_err,
                         "uniform": m["uniform"]["observable"]["logical_errors"],
                         "met": (ob["logical_errors"] <= dem_err / 1.5) if dem_err >= 30 else None}
    diff = abs(ob["logical_errors"] - row["score_measured_errors"])
    v["A3_agrees_with_score"] = {"certified_lane": ob["logical_errors"],
                                 "score": row["score_measured_errors"], "met": diff <= 3}
    v["A4_cut_discriminates"] = {"declared": ob["logical_errors"],
                                 "opposite": ob["logical_errors_opposite_cut"],
                                 "met": (ob["logical_errors_opposite_cut"] >= 10 * ob["logical_errors"])
                                 if ob["logical_errors"] >= 1 else None}
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default=None, help="comma-separated record paths")
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)
    runs = a.runs.split(",") if a.runs else list(RECORDS)
    out = {"schema": "oneq/certified-measured-lane/v1", "max_span": a.max_span, "runs": []}
    for path in runs:
        row = one_record(path, a.max_span)
        row["verdicts"] = verdicts(row)
        out["runs"].append(row)
        m = row["models"]
        print(f"{path}: held out {row['held_out_shots']} shots")
        for name in ("dem", "uniform", "measured"):
            o = m[name]
            print(f"  {name:9s} certified {o['certified']}/{o['nontrivial']} ({o['certified_rate']:.4f})  "
                  f"logical errors {o['observable']['logical_errors']:5d}  opposite cut "
                  f"{o['observable']['logical_errors_opposite_cut']:5d}  edges {o['edges']}  {o['seconds']} s")
        print(f"  score() on the measured edges: {row['score_measured_errors']} errors; verdicts "
              + ", ".join(f"{k.split('_')[0]}={v['met']}" for k, v in row["verdicts"].items()))
    # THE CORPUS TOTALS, WRITTEN BY THE PRODUCER. A number a document types must exist in an
    # artifact: `tools/claims_provenance_gate.py` reads every number in the docs and refuses any
    # that appears nowhere in the evidence tree. Summing the rows inside the paper's generator
    # made the paper the producer of its own headline, which is exactly the orphan the gate
    # exists to catch, and it caught it (2026-09-16).
    rows = out["runs"]
    mdl = lambda r, k: r["models"][k]                                   # noqa: E731
    tot = {
        "records": len(rows),
        "devices": sorted({r["backend"] for r in rows}),
        "distances": sorted({r["distance"] for r in rows}),
        "rounds": sorted({r["rounds"] for r in rows}),
        "held_out_shots": sum(r["held_out_shots"] for r in rows),
        "nontrivial": sum(mdl(r, "measured")["nontrivial"] for r in rows),
        "certified_under_measured": sum(mdl(r, "measured")["certified"] for r in rows),
        "measured_graph_edges_min": min(mdl(r, "measured")["edges"] for r in rows),
        "measured_graph_edges_max": max(mdl(r, "measured")["edges"] for r in rows),
        "logical_errors": {k: sum(mdl(r, k)["observable"]["logical_errors"] for r in rows)
                           for k in ("dem", "uniform", "measured")},
        "rounds_with_refusals": sorted({r["rounds"] for r in rows
                                        if mdl(r, "measured")["nontrivial"] > mdl(r, "measured")["certified"]}),
    }
    tot["refused_under_measured"] = tot["nontrivial"] - tot["certified_under_measured"]
    tot["dem_over_measured_errors"] = (tot["logical_errors"]["dem"]
                                       / max(tot["logical_errors"]["measured"], 1))
    tot["note"] = ("sums over the records above, written here so every number a document types "
                   "exists in an artifact; the ratio is against OUR declared baselines on short "
                   "chains and is not comparable to the estimated-DEM or neural-decoder literature")
    out["totals"] = tot
    print(f"TOTALS: certified {tot['certified_under_measured']}/{tot['nontrivial']} nontrivial over "
          f"{tot['records']} records on {len(tot['devices'])} devices; logical errors "
          f"DEM {tot['logical_errors']['dem']}, uniform {tot['logical_errors']['uniform']}, "
          f"measured {tot['logical_errors']['measured']} "
          f"({tot['dem_over_measured_errors']:.1f}x)")
    if a.json:
        p = ROOT / a.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
