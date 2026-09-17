r"""The device's error model, read off its own detectors, and judged HELD OUT.

R13-R22 established that IBM's calibration-derived DEM misprices this code
and that a hand-built correction fixes it. Every one of those corrections
was a GUESS with a constant in it -- a per-family scale fitted on half a
run, then one flat burst price. This lane removes the guess: it estimates
the model from the detector statistics alone (`oneq.dem_measured`), using
no calibration data and never reading a logical outcome, and then decodes
with it.

THREE THINGS ARE MEASURED HERE, and each is reported with its control.

  1. WHICH WAY THE DEM IS WRONG, by family. A uniform error cannot explain
     the campaign (see the module docstring); this reports the DEM's own
     per-family probabilities against the measured ones, so the direction
     is a number rather than an inference.

  2. WHETHER THE BURST IS IN THE DATA. Same-position correlations at
     round separations the DEM does not model at all, reported as a
     fraction of the adjacent-round correlation -- against a SYNTHETIC
     CONTROL sampled from an independent model at the same rate, where
     that fraction must be ~0. Without the control the tail is an
     estimator artifact for all anyone can tell. And against the SHARPER
     null: the same tail within strata of each shot's total detector
     count, which removes a shot-to-shot wobble in the error rate (R8
     measured a lag-1 shot memory here). Both arms are stratified the
     same way because stratifying biases the estimate down by itself.

  3. WHETHER IT DECODES BETTER, HELD OUT. The model is estimated on the
     first half of the shots and judged on the second. In-sample is
     reported too, and the honest reading is the held-out column: the
     estimator never sees a logical outcome, but it does see detectors,
     and estimating on the shots you decode is still estimating on them.

  4. WHETHER IT IS THE DEVICE'S MODEL AT ALL -- and it is NOT. R21a's own
     test, turned on this model: sample from its weights and compare the
     detector statistics to the run's. It over-predicts firing 1.10-2.13x,
     and on fez it describes the device WORSE than the DEM it beats 61x at
     decoding. `graph_likeness` says the same algebraically. Reported on
     every run, because a decoding win is the easiest thing in this field
     to mis-read as a measurement of the hardware.

The lane REFUSES (rc 6) if the measured model stops beating the device's
own model on held-out shots. That is the regression this exists to catch.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ibm_live"))
sys.path.insert(0, str(ROOT / "experiments" / "dual_observable"))

from oneq.dem_families import (apply_burst_model,                  # noqa: E402
                               edge_family)
from oneq.dem_measured import (MAX_SPAN, graph_likeness,           # noqa: E402
                               measure_pairs, measured_graph, sample_from)
from hardware_noise_floor import score                             # noqa: E402
from rep_code_cert import logical_cut_edges                        # noqa: E402

#: The runs this lane judges by default: every committed hardware artifact
#: that carries pinned weights and a truth column, across BOTH devices.
DEFAULT_RUNS = (
    ("marrakesh B'' 20k", "evidence/hardware_noise_floor_B2_20k.json"),
    ("marrakesh B 20k", "evidence/hardware_noise_floor_B_20k.json"),
    ("marrakesh A 40k", "evidence/hardware_noise_floor_A_40k.json"),
    ("fez d=3 20k", "evidence/hardware_noise_floor_fez_d3_20k.json"),
    # R25 (2026-09-08): the preregistered nested pair. d=5 is the campaign's
    # only d>3 decode result (52 -> 1); the d=3 arm is the middle five of the
    # same nine qubits (41 -> 1). Carried here so the wall watches both.
    ("fez d=5 40k", "evidence/rep_d5_r13_fez.json"),
    ("fez d=3 nested 20k", "evidence/rep_d3_r13_fez_nested.json"),
)

#: The R21 burst price, carried here UNCHANGED as the arm to beat. It was
#: chosen by looking at error counts on marrakesh; the measured model was
#: not, which is why the held-out comparison between them is the fair one.
BURST_COST = 0.5


def load(path):
    doc = json.loads((ROOT / path).read_text(encoding="utf-8"))
    prov = (doc.get("edge_provenance") or {}).get("edges")
    if not prov:
        raise SystemExit(f"REFUSING: {path} carries no pinned weights.")
    edges = [(int(k.partition("-")[0]), int(k.partition("-")[2]),
              float(Fraction(v["weight_exact"])))
             for k, v in prov.items()]
    sh = doc["detector_shape"]
    det = np.unpackbits(np.array(doc["detectors_packed"], dtype=np.uint8)
                        )[: sh[0] * sh[1]].reshape(sh).astype(np.uint8)
    return doc, edges, det


def _p(w):
    return 1.0 / (1.0 + math.exp(w))


def family_comparison(dem_edges, det, positions):
    """The DEM's per-family probabilities against the measured ones."""
    pairs = sorted({(min(int(u), int(v)), max(int(u), int(v)))
                    for (u, v, _w) in dem_edges if v >= 0})
    est, _means = measure_pairs(det, pairs)
    out = {}
    for fam in ("time-like", "space-like"):
        dem_p, got_p = [], []
        for (u, v, w) in dem_edges:
            if v < 0 or edge_family(int(u), int(v), positions) != fam:
                continue
            m = est.get((min(int(u), int(v)), max(int(u), int(v))))
            if m is None or m <= 0:
                continue
            dem_p.append(_p(w))
            got_p.append(m)
        if not dem_p:
            continue
        a = sum(dem_p) / len(dem_p)
        b = sum(got_p) / len(got_p)
        out[fam] = {"dem_p": a, "measured_p": b, "ratio": a / b, "n": len(dem_p)}
    return out


def _tail(d, positions, n_rounds, spans):
    means = d.mean(axis=0)

    def corr(i, j):
        return float((d[:, i] * d[:, j]).mean()) - float(means[i] * means[j])

    def avg(span):
        vals = []
        for q in range(positions):
            for t in range(n_rounds - span):
                vals.append(corr(t * positions + q, (t + span) * positions + q))
        return sum(vals) / len(vals) if vals else 0.0

    base = avg(1)
    out = {"adjacent": base}
    for s in spans:
        if s < n_rounds:
            v = avg(s)
            out[f"span{s}"] = v
            out[f"span{s}_fraction"] = (v / base) if base else 0.0
    return out


def correlation_tail(det, positions, n_rounds, spans=(2, 4, 6, 8), strata=0):
    r"""Same-position correlation at each round separation. The DEM models
    NONE of these.

    `strata` > 0 computes the same quantity WITHIN bins of the shot's total
    detector count and pools it. That is the sharper null for R23's burst
    claim: a shot-to-shot wobble in the error rate -- and R8 measured a
    lag-1 shot memory on exactly this data -- makes every detector in a
    noisy shot fire together, which shows up as correlation at EVERY span
    without any multi-round mechanism existing. Stratifying removes it.

    Stratification also induces a negative bias of its own (conditioning on
    a sum anti-correlates its terms), so the number that means something is
    the EXCESS over the synthetic independent control computed the same
    way. Measured: the excess is essentially unchanged by stratification on
    three of four runs, and fez loses about a third. The tail is a
    mechanism, not the shot wobble.
    """
    det = np.asarray(det)
    if strata <= 1:
        return _tail(det.astype(np.float64), positions, n_rounds, spans)
    order = np.argsort(det.sum(axis=1), kind="stable")
    acc, n = {}, 0
    for chunk in np.array_split(order, strata):
        if chunk.size < 50:
            continue
        t = _tail(det[chunk].astype(np.float64), positions, n_rounds, spans)
        for k, v in t.items():
            if not k.endswith("_fraction"):
                acc[k] = acc.get(k, 0.0) + v * chunk.size
        n += chunk.size
    if not n:
        return _tail(det.astype(np.float64), positions, n_rounds, spans)
    out = {k: v / n for k, v in acc.items()}
    base = out.get("adjacent", 0.0)
    for s in spans:
        if f"span{s}" in out:
            out[f"span{s}_fraction"] = (out[f"span{s}"] / base) if base else 0.0
    return out


def _independent_control(dem_edges, n_nodes, positions, n_rounds, shots, seed,
                         strata=0):
    """The tail measured on data with NO multi-round mechanism in it."""
    det, _t = sample_from(dem_edges, n_nodes, positions, shots, seed)
    return correlation_tail(det, positions, n_rounds, strata=strata)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated artifact paths; default is every "
                         "committed hardware run across both devices")
    ap.add_argument("--max-span", type=int, default=MAX_SPAN)
    ap.add_argument("--control-shots", type=int, default=40000)
    ap.add_argument("--strata", type=int, default=10,
                    help="bins of total detector count for the stratified "
                         "tail; the sharper null for the burst claim")
    ap.add_argument("--no-in-sample", action="store_true",
                    help="held-out only; the wall's fast path")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)

    runs = DEFAULT_RUNS
    if a.runs:
        runs = tuple((p.split("/")[-1], p) for p in a.runs.split(","))

    t0 = time.time()
    out = {"runs": [], "burst_cost": BURST_COST, "max_span": a.max_span}
    tot = {"dem": 0, "burst": 0, "pricing_only": 0, "structure_only": 0,
           "measured": 0}
    tot_in = {"dem": 0, "burst": 0, "measured": 0}
    for label, path in runs:
        doc, edges, det = load(path)
        d, r = doc["circuit"]["distance"], doc["circuit"]["rounds"]
        n_nodes = doc["detector_shape"][1]
        positions, n_rounds = d - 1, r + 1
        truth = np.array(doc["truth"], dtype=np.uint8)
        cut = logical_cut_edges(d, r)
        half = det.shape[0] // 2

        def errs(edge_list, lo, hi):
            _rec, _pr, _g, correct = score(edge_list, cut, det[lo:hi],
                                           truth[lo:hi], n_nodes,
                                           budget_s=1e-9)
            return sum(1 for x in correct if not x)

        # HELD OUT: estimated on the first half, judged on the second.
        g = measured_graph(edges, det[:half], positions, n_rounds, n_nodes,
                           max_span=a.max_span)
        # *** THE DECOMPOSITION. *** `measured_graph` changes TWO things at
        # once -- it re-prices the DEM's edges from data AND adds
        # multi-round edges the DEM does not have. Reporting only the total
        # would credit the estimator for a gain that might be entirely
        # structural. Found the honest way, by a test: a graph with the
        # added edges and every weight set to a constant still beat the DEM
        # 7.2x on fez, because a flat weight on a span-6 edge IS the burst
        # model at its limit. So both halves are separated here.
        #   pricing only  the DEM's own edge set, re-priced from data
        #   structure only  the added edges, but every weight uniform
        g_price = measured_graph(edges, det[:half], positions, n_rounds,
                                 n_nodes, max_span=1)
        g_struct = [(u, v, 1.0) for (u, v, _w) in g.edges]
        held = {"dem": errs(list(edges), half, None),
                "burst": errs(apply_burst_model(edges, BURST_COST, positions,
                                                n_rounds), half, None),
                "pricing_only": errs(g_price.edges, half, None),
                "structure_only": errs(g_struct, half, None),
                "measured": errs(g.edges, half, None),
                "shots": int(det.shape[0] - half)}
        for k in tot:
            tot[k] += held[k]

        # THE SELF-REFUTATION, on every run, in the shipped artifact.
        real_rate = float(det.mean())
        gen = {}
        for name, el in (("dem", list(edges)), ("measured", g.edges)):
            sdet, _st = sample_from(el, n_nodes, positions,
                                    min(20000, int(det.shape[0])),
                                    seed=20260906)
            gen[name] = float(sdet.mean()) / real_rate if real_rate else 0.0
        row = {"label": label, "artifact": path, "distance": d, "rounds": r,
               "held_out": held, "graph_stats": g.stats,
               "graph_likeness": graph_likeness(edges, det, positions,
                                                n_rounds, n_nodes,
                                                max_span=a.max_span),
               "generative_rate_vs_real": gen,
               "families": family_comparison(edges, det, positions),
               "correlation_tail": correlation_tail(det, positions, n_rounds),
               "independent_control": _independent_control(
                   edges, n_nodes, positions, n_rounds,
                   min(a.control_shots, int(det.shape[0])), seed=20260906),
               # THE SHARPER NULL: the same tail computed within strata of
               # the shot's total detector count, which removes a
               # shot-to-shot wobble in the error rate. Each with its own
               # control, because stratifying biases the estimate down.
               "correlation_tail_stratified": correlation_tail(
                   det, positions, n_rounds, strata=a.strata),
               "independent_control_stratified": _independent_control(
                   edges, n_nodes, positions, n_rounds,
                   min(a.control_shots, int(det.shape[0])), seed=20260906,
                   strata=a.strata)}
        row["tail_excess_over_control"] = {
            f"span{s}{suf}": (row["correlation_tail" + key].get(f"span{s}", 0.0)
                              - row["independent_control" + key].get(f"span{s}", 0.0))
            for s in (2, 4, 6, 8)
            for suf, key in (("", ""), ("_stratified", "_stratified"))}
        if not a.no_in_sample:
            gi = measured_graph(edges, det, positions, n_rounds, n_nodes,
                                max_span=a.max_span)
            ins = {"dem": errs(list(edges), 0, None),
                   "burst": errs(apply_burst_model(edges, BURST_COST,
                                                   positions, n_rounds),
                                 0, None),
                   "measured": errs(gi.edges, 0, None),
                   "shots": int(det.shape[0])}
            row["in_sample"] = ins
            for k in tot_in:
                tot_in[k] += ins[k]
        out["runs"].append(row)

        print(f"{label:20s} held-out {held['shots']:6d} shots: "
              f"DEM {held['dem']:5d}  burst {held['burst']:5d}  "
              f"pricing {held['pricing_only']:5d}  "
              f"structure {held['structure_only']:5d}  "
              f"MEASURED {held['measured']:5d}")
        for fam, f in row["families"].items():
            print(f"    {fam:11s} DEM p {f['dem_p']:.5f}  measured "
                  f"{f['measured_p']:.5f}  ratio {f['ratio']:6.2f}x  "
                  f"({'OVERstated' if f['ratio'] > 1 else 'UNDERstated'})")
        gl = row["graph_likeness"]
        print(f"    NOT the device's model: predicted detector rate "
              f"{gen['measured']:.2f}x real (DEM {gen['dem']:.2f}x); the "
              f"marginal constraint has no solution on "
              f"{gl['unsatisfiable']}/{gl['n_nodes']} detectors")
        ct, ic = row["correlation_tail"], row["independent_control"]
        ex = row["tail_excess_over_control"]
        for s in (2, 8):
            k = f"span{s}_fraction"
            if k in ct:
                print(f"    span {s} correlation: hardware "
                      f"{ct[k]:6.2f}x adjacent   independent control "
                      f"{ic.get(k, 0.0):6.2f}x   excess over control "
                      f"{ex[f'span{s}']:+.5f}, and {ex[f'span{s}_stratified']:+.5f} "
                      f"STRATIFIED by shot rate")

    out["total_held_out"] = tot
    out["total_in_sample"] = tot_in if not a.no_in_sample else None
    out["held_out_ratio_vs_dem"] = tot["dem"] / max(1, tot["measured"])
    out["held_out_ratio_vs_burst"] = tot["burst"] / max(1, tot["measured"])
    out["seconds"] = round(time.time() - t0, 1)
    # THE VERDICT THE ROADMAP READS. The two refusals below are the gate's
    # criterion; an artifact that carried the numbers without the verdict
    # would let the roadmap derive CLOSED from the file merely existing.
    compounds = all(tot["measured"] < tot[k]
                    for k in ("pricing_only", "structure_only"))
    out["passed"] = bool(tot["measured"] < tot["dem"] and compounds)

    print(f"\nHELD OUT across {len(out['runs'])} run(s): DEM {tot['dem']}, "
          f"burst {tot['burst']}, MEASURED {tot['measured']}")
    print(f"  decomposition -- pricing alone (DEM's edges, measured p): "
          f"{tot['pricing_only']}")
    print(f"                   structure alone (added edges, flat weights): "
          f"{tot['structure_only']}")
    print(f"  measured vs the device's own model: "
          f"{out['held_out_ratio_vs_dem']:.2f}x better")
    print(f"  measured vs the R21 burst model:    "
          f"{out['held_out_ratio_vs_burst']:.2f}x")
    print("  -- and it used NO calibration data and never read a logical "
          "outcome")
    ug = sum(r["graph_likeness"]["unsatisfiable"] for r in out["runs"])
    ng = sum(r["graph_likeness"]["n_nodes"] for r in out["runs"])
    # The damping figures are the FOUR-RUN result from R23a and are labelled
    # as such: this line prints on any run set, and an unlabelled "45 -> 124"
    # under a one-run report reads as that run's own number.
    print(f"  BUT IT IS NOT THE DEVICE'S MODEL: the marginal constraint has "
          f"no solution on {ug} of {ng} detectors. Damping the added edges "
          f"toward a model that IS consistent costs the gain (45 -> 124 over "
          f"the four default runs, R23a -- not this run set). A "
          f"better-shaped prior, not a truer description.")

    if a.json:
        dest = ROOT / a.json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.json}")

    # THE REGRESSION THIS LANE EXISTS TO CATCH. A change that made the
    # estimator worse than the model it replaces would otherwise land
    # looking like any other refactor.
    if tot["measured"] >= tot["dem"]:
        print(f"REFUSING: the measured model ({tot['measured']}) does not "
              f"beat the device's own DEM ({tot['dem']}) on held-out shots.",
              file=sys.stderr)
        return 6
    # THE COMPOUNDING CLAIM, as a regression. Judged on the TOTAL and said
    # so: on `marrakesh B''` alone structure-only (13) beats the full model
    # (17), and a per-run form of this check would refuse a true result.
    # Across the four runs the margin is 45 against 104 and 201.
    worse = [k for k in ("pricing_only", "structure_only")
             if tot["measured"] >= tot[k]]
    if worse and len(out["runs"]) == len(DEFAULT_RUNS):
        print(f"REFUSING: the measured model ({tot['measured']}) no longer "
              f"beats its own half/halves {worse} "
              f"({', '.join(f'{k}={tot[k]}' for k in worse)}) across the full "
              f"run set -- the two halves stopped compounding.", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
