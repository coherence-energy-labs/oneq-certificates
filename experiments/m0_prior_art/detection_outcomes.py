r"""Detection power WITH its base rates, under the sound epsilon checker.

Answers audit items M6 / A-03 and M7 / A-04 on the two detection experiments
of paper 1 (decoder_mutation_gate.py, stale_calibration_fault.py). Those runs
reported how often a refusal coincided with a defective correction, but not

  * how often the detector refuses a CLEAN output (the false-alarm base rate,
    which is producer slack, not decoder error),
  * whether a refusal is PROVEN suboptimality or merely NOT_PROVEN,
  * ground truth independent of PyMatching (the old runs called a correction
    suboptimal when it was heavier than PyMatching's own answer on the true
    graph, and PyMatching discretises weights internally),
  * the discordant logical-outcome cells behind "the logical error rate is
    unchanged".

Same circuits, same seeds, same mutants, same drift construction as the runs of
record; the old headline counts are recomputed here and compared, so a change
of harness cannot pass silently.

Per evaluated correction J on the TRUE graph:
  INVALID            J does not explain the syndrome (syndrome check catches it)
  CERTIFIED          the producer's certificate is accepted by check_epsilon
                     (EXACT_OPTIMAL or EPSILON_OPTIMAL at eps_max = 1e-6)
  PROVEN_SUBOPTIMAL  refused, and an independently re-verified lighter
                     correction exists (the exact oracle's, re-checked here)
  NOT_PROVEN         refused, and no lighter correction by more than eps_max

Ground truth: excess(J) = w(J) - w(J*) in exact rationals over the double
weights, w(J*) from exact_min_tjoin, whose answer is re-verified here (odd set,
exact weight) and never taken on its word. J is DEFECTIVE iff it is INVALID or
excess > eps_max. A false alarm is a refusal of a non-defective J. A miss is an
acceptance of a defective J; soundness says there are none and the run fails
if one appears.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from decoder_mutation_gate import mutants, odd_set             # noqa: E402
from stale_calibration_fault import drifted_dem                # noqa: E402

from oneq.certifying_decoder import CertifyingDecoder          # noqa: E402
from oneq.exact_decode import exact_min_tjoin                  # noqa: E402
from oneq.matching_cert import BOUNDARY                        # noqa: E402

EPS_MAX = Fraction(1, 10**6)
DRIFTS = (1.0, 1.5, 2.0, 3.0)
OUT = ROOT / "evidence" / "detection_outcomes"


def _circuit(d, rounds, p):
    import stim
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def _edges(m):
    return [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
             float(e[2].get("weight", 1.0))) for e in m.edges()]


class Judge:
    """Ground truth and the detector's verdict for one graph."""

    def __init__(self, edges):
        self.edges = edges
        self.cd = CertifyingDecoder(edges, epsilon_max=EPS_MAX)
        self.WX = {k: Fraction(w) for k, w in self.cd.W.items()}

    def optimum(self, fired):
        corr, w, why = exact_min_tjoin(self.edges, fired)
        if corr is None:
            return None, f"oracle gave no answer: {why}"
        keys = tuple(self.cd.key(e) for e in corr)
        if any(k not in self.WX for k in keys):
            return None, "oracle correction uses an edge outside the graph"
        if odd_set(keys) != set(fired):
            return None, "oracle correction does not explain the syndrome"
        wx = sum((self.WX[k] for k in keys), Fraction(0))
        if wx != w:
            return None, "oracle weight does not recompute"
        return wx, None

    def outcome(self, fired, corr, opt, *, hard=False):
        keys = tuple(self.cd.key(e) for e in corr)
        if any(k not in self.WX for k in keys) or odd_set(keys) != set(fired):
            return {"state": "INVALID", "defective": True, "excess": None}
        excess = sum((self.WX[k] for k in keys), Fraction(0)) - opt
        defective = excess > EPS_MAX
        if hard:
            r = self.cd.certify_hard(fired, keys)
        else:
            r = self.cd.certify(fired, keys)
        if r.accepted:
            state = "CERTIFIED"
        elif excess > EPS_MAX:
            state = "PROVEN_SUBOPTIMAL"
        else:
            state = "NOT_PROVEN"
        if r.accepted and defective:
            raise SystemExit(f"SOUNDNESS VIOLATION: accepted a correction {float(excess)} "
                             f"above the optimum (eps_max {EPS_MAX})")
        return {"state": state, "defective": defective, "excess": excess}


def _tally(rows):
    t = {"n": len(rows), "defective": 0, "invalid": 0, "certified": 0,
         "proven_suboptimal": 0, "not_proven": 0,
         "false_alarms": 0, "missed": 0, "not_proven_on_defective": 0}
    for o in rows:
        t["defective"] += o["defective"]
        t["invalid"] += o["state"] == "INVALID"
        t["certified"] += o["state"] == "CERTIFIED"
        t["proven_suboptimal"] += o["state"] == "PROVEN_SUBOPTIMAL"
        t["not_proven"] += o["state"] == "NOT_PROVEN"
        t["false_alarms"] += (not o["defective"]) and o["state"] != "CERTIFIED"
        t["missed"] += o["defective"] and o["state"] == "CERTIFIED"
        t["not_proven_on_defective"] += o["defective"] and o["state"] == "NOT_PROVEN"
    ex = [o["excess"] for o in rows if o["defective"] and o["excess"] is not None]
    t["mean_excess_of_suboptimal"] = float(sum(ex) / len(ex)) if ex else 0.0
    return t


def seeded_faults(d, shots, p):
    import pymatching
    circ = _circuit(d, d, p)
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    edges = _edges(m)
    J = Judge(edges)
    det, _ = circ.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    clean, clean_hard, per = [], [], {}
    legacy = {}
    oracle_failures = []
    prev = None
    for s in range(shots):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        matched = tuple(J.cd.key((u, v if v >= 0 else BOUNDARY))
                        for u, v in m.decode_to_edges_array(det[s]))
        opt, why = J.optimum(fired)
        if why:
            oracle_failures.append({"shot": s, "why": why})
            prev = matched
            continue
        clean.append(J.outcome(fired, matched, opt))
        clean_hard.append(J.outcome(fired, matched, opt, hard=True)
                          if clean[-1]["state"] != "CERTIFIED" else clean[-1])
        base_w = J.cd.weight(matched)
        for name, mut in mutants(J.cd.W, matched, fired, prev):
            per.setdefault(name, []).append(J.outcome(fired, mut, opt))
            # the run of record's definition, recomputed for reconciliation
            lg = legacy.setdefault(name, {"n": 0, "truly_defective": 0})
            lg["n"] += 1
            lg["truly_defective"] += int(odd_set(mut) != set(fired)
                                         or sum(J.cd.W[k] for k in mut) > base_w + 1e-9)
        prev = matched
    return {"distance": d, "rounds": d, "shots": shots, "p": p, "sampler_seed": 7,
            "clean_output_base_ladder": _tally(clean),
            "clean_output_full_ladder": _tally(clean_hard),
            "clean_output_pymatching_above_optimum": sum(o["defective"] for o in clean),
            "clean_output_pymatching_excess_positive": sum(
                1 for o in clean if o["excess"] is not None and o["excess"] > 0),
            "per_mutant": {k: _tally(v) for k, v in sorted(per.items())},
            "legacy_definition": legacy,
            "oracle_failures": oracle_failures}


def stale_calibration(d, shots, p):
    import pymatching
    circ = _circuit(d, d, p)
    true_dem = circ.detector_error_model(decompose_errors=True)
    true_m = pymatching.Matching.from_detector_error_model(true_dem)
    edges = _edges(true_m)
    J = Judge(edges)
    det, obs = circ.compile_detector_sampler(seed=20260731).sample(shots, separate_observables=True)
    rows = []
    for factor in DRIFTS:
        rng = np.random.default_rng(7)
        stale_m = pymatching.Matching.from_detector_error_model(drifted_dem(true_dem, factor, rng))
        outs, cells = [], {"both_correct": 0, "stale_only_wrong": 0,
                           "true_only_wrong": 0, "both_wrong": 0}
        legacy_sub = 0
        skipped = {"not_expressible": 0, "decode_error": 0, "oracle": 0}
        for s in range(shots):
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            try:
                corr = tuple(J.cd.key((u, v if v >= 0 else BOUNDARY))
                             for u, v in stale_m.decode_to_edges_array(det[s]))
            except Exception:                                   # noqa: BLE001
                skipped["decode_error"] += 1
                continue
            if any(k not in J.cd.W for k in corr):
                skipped["not_expressible"] += 1
                continue
            opt, why = J.optimum(fired)
            if why:
                skipped["oracle"] += 1
                continue
            outs.append(J.outcome(fired, corr, opt))
            true_corr = tuple(J.cd.key((u, v if v >= 0 else BOUNDARY))
                              for u, v in true_m.decode_to_edges_array(det[s]))
            legacy_sub += int(J.cd.weight(corr) > J.cd.weight(true_corr) + 1e-9)
            tw = bool(true_m.decode(det[s])[0]) != bool(obs[s][0])
            sw = bool(stale_m.decode(det[s])[0]) != bool(obs[s][0])
            key = ("both_wrong" if tw and sw else "true_only_wrong" if tw
                   else "stale_only_wrong" if sw else "both_correct")
            cells[key] += 1
        rows.append({"drift_factor": factor, "outcomes": _tally(outs),
                     "logical_cells": cells, "legacy_suboptimal_vs_true_decoder": legacy_sub,
                     "skipped": skipped})
    return {"distance": d, "rounds": d, "shots": shots, "p": p, "sampler_seed": 20260731,
            "drift_rng_seed": 7,
            "drift_definition": ("each error mechanism's probability is multiplied by an "
                                 "independent factor exp(U), U uniform on [-ln f, ln f], "
                                 "clipped to [1e-12, 0.49]; the stale graph is rebuilt from "
                                 "the drifted model; the true model is the reference supplied "
                                 "to the checker"),
            "rows": rows}


def _reconcile(out, d):
    """The new harness must reproduce the run of record's counts."""
    ev = ROOT / "evidence"
    old = json.loads((ev / f"decoder_mutation_gate_d{d}.json").read_text(encoding="utf-8"))
    for name, lg in out["seeded_faults"]["legacy_definition"].items():
        o = old["per_mutant"][name]
        if (o["n"], o["truly_defective"]) != (lg["n"], lg["truly_defective"]):
            raise SystemExit(f"d={d} {name}: harness does not reproduce the run of record "
                             f"({o['n']},{o['truly_defective']}) vs ({lg['n']},{lg['truly_defective']})")
    old = json.loads((ev / f"stale_calibration_d{d}.json").read_text(encoding="utf-8"))
    for r in out["stale_calibration"]["rows"]:
        o = next(x for x in old["rows"] if x["drift_factor"] == r["drift_factor"])
        c = r["logical_cells"]
        if (o["shots"], o["genuinely_suboptimal"],
                o["logical_errors_true_model"], o["logical_errors_stale_model"]) != (
                r["outcomes"]["n"], r["legacy_suboptimal_vs_true_decoder"],
                c["true_only_wrong"] + c["both_wrong"], c["stale_only_wrong"] + c["both_wrong"]):
            raise SystemExit(f"d={d} drift {r['drift_factor']}: stale harness does not reproduce "
                             f"the run of record")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("distance", type=int, choices=(3, 5, 7))
    ap.add_argument("--p", type=float, default=5e-3)
    a = ap.parse_args()
    t0 = time.time()
    out = {"schema": "oneq-detection-outcomes/1", "eps_max": str(EPS_MAX),
           "command": f"python experiments/m0_prior_art/detection_outcomes.py {a.distance}",
           "seeded_faults": seeded_faults(a.distance, 300, a.p),
           "stale_calibration": stale_calibration(a.distance, 800, a.p)}
    _reconcile(out, a.distance)
    out["wall_seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"d{a.distance}.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {dest.relative_to(ROOT)} in {out['wall_seconds']} s")
    sf = out["seeded_faults"]
    print("clean base ladder:", sf["clean_output_base_ladder"])
    print("clean full ladder:", sf["clean_output_full_ladder"])
    for r in out["stale_calibration"]["rows"]:
        print("drift", r["drift_factor"], r["outcomes"], r["logical_cells"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
