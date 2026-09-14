r"""Is the checker really O(nnz)? Measure it; do not assert it.

The whole value proposition rests on the checker being cheap and simple enough
that a stranger can re-run it. "O(nnz), solver-free" is easy to write and easy
to be wrong about -- the constant hidden in |support(z)| could grow with the
graph, in which case verification scales worse than the claim and the argument
quietly weakens as codes get bigger, which is exactly when it matters.

WHAT IS ACTUALLY BEING CLAIMED. Feasibility is one pass over the edges,
accumulating for each edge the duals whose boundary it crosses. That is
O(nnz x |support(z)|) in the naive form, not O(nnz). The honest claim is
therefore:

    checking cost grows LINEARLY in nnz x |support(z)|

and the thing worth measuring is whether |support(z)| itself stays small --
because if it grew with the graph, the product would be superlinear in nnz and
the headline would be misleading even though every individual step is cheap.

METHOD. Sweep graph size by distance and round count, measure the checker ALONE
(the producer is excluded: it is the expensive half and it is not what a
verifier runs), and fit a power law. Timing is done on an otherwise idle
machine because a measurement taken under load measures the load.

THE FALSIFIER. If the fitted exponent on nnz x |support(z)| is materially above
1.0, the claim is wrong and the paper says so instead.
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

from oneq.certifying_decoder import CertifyingDecoder                # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate, check  # noqa: E402


def main() -> int:
    import pymatching
    import stim
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=60,
                    help="certified shots to time per configuration")
    ap.add_argument("--repeats", type=int, default=5,
                    help="timed repetitions per shot; the MINIMUM is kept, "
                         "because the minimum is the least contaminated by "
                         "whatever else the machine was doing")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    configs = [(3, 3), (3, 9), (5, 5), (5, 15), (5, 25), (7, 7), (7, 15), (7, 25)]
    rows = []
    for d, rounds in configs:
        p = 5e-3
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds,
            after_clifford_depolarization=p, before_measure_flip_probability=p,
            after_reset_flip_probability=p, before_round_data_depolarization=p)
        m = pymatching.Matching.from_detector_error_model(
            circ.detector_error_model(decompose_errors=True))
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
                  float(e[2].get("weight", 1.0))) for e in m.edges()]
        cd = CertifyingDecoder(edges)
        nnz = len(cd.W)
        det, _ = circ.compile_detector_sampler(seed=808).sample(
            a.shots * 3, separate_observables=True)

        cases = []
        for s in range(det.shape[0]):
            if len(cases) >= a.shots:
                break
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                         for u, v in m.decode_to_edges_array(det[s]))
            r = cd.certify(fired, corr)
            if r.accepted:
                cases.append((fired, corr, r.cert))
        if not cases:
            continue

        times, supports = [], []
        for fired, corr, cert in cases:
            best = float("inf")
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                check(edges=edges, syndrome=fired, cert=cert)
                best = min(best, time.perf_counter() - t0)
            times.append(best)
            supports.append(len(cert.blossom_duals))
        t = np.array(times)
        sup = np.array(supports)
        row = {"distance": d, "rounds": rounds, "nnz": nnz,
               "nodes": len({u for u, _v, _w in edges} | {v for _u, v, _w in edges}),
               "shots_timed": len(cases),
               "support_mean": float(sup.mean()), "support_max": int(sup.max()),
               "check_ms_mean": float(1e3 * t.mean()),
               "check_ms_median": float(1e3 * np.median(t)),
               "work_units": float(nnz * sup.mean())}
        rows.append(row)
        print(f"d={d:2d} r={rounds:3d}  nnz={nnz:6d}  |z| mean {sup.mean():6.1f} "
              f"max {sup.max():4d}   check {1e3*np.median(t):8.3f} ms  "
              f"({nnz*sup.mean():10.0f} work units)")

    # Fit log(time) = c + k*log(work). k ~ 1 means linear in nnz x |support|.
    fit = None
    if len(rows) >= 3:
        x = np.log(np.array([r["work_units"] for r in rows]))
        y = np.log(np.array([r["check_ms_median"] for r in rows]))
        k, c = np.polyfit(x, y, 1)
        pred = c + k * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        fit = {"exponent": float(k), "intercept": float(c),
               "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
               "x": "log(nnz * mean|support(z)|)", "y": "log(median check ms)"}
        print(f"\nfit: time ~ (nnz * |z|)^{k:.3f}   R^2 = {fit['r_squared']:.4f}")
        verdict = ("LINEAR as claimed" if k <= 1.15 else
                   "SUPERLINEAR -- the O(nnz) claim is WRONG and the paper must "
                   "say so")
        print(f"verdict: {verdict}")

    # THE FIT THAT ACTUALLY ANSWERS THE QUESTION: time against nnz directly.
    #
    # An earlier version reported "check ~ nnz^(1+k)" by fitting |support(z)|
    # against nnz and ADDING ONE -- valid only while the cost really was
    # nnz * |support|, which it was when the checker scanned every edge once
    # per dual. After the walk-each-moat's-boundary rewrite the cost is the sum
    # of the moats' volumes, so that derived exponent describes an
    # implementation that no longer exists. Fitting the thing itself removes
    # the assumption.
    supp_fit = direct = None
    if len(rows) >= 3:
        lz = np.log(np.array([r["nnz"] for r in rows]))
        k2, c2 = np.polyfit(lz, np.log(np.array([r["support_mean"] for r in rows])), 1)
        supp_fit = {"exponent": float(k2), "intercept": float(c2),
                    "note": "|support(z)| vs nnz -- reported because a support "
                            "that grows with the graph is what would make a "
                            "per-dual scan superlinear"}
        yt = np.log(np.array([r["check_ms_median"] for r in rows]))
        k3, c3 = np.polyfit(lz, yt, 1)
        pred = c3 + k3 * lz
        ssr = float(((yt - pred) ** 2).sum()); sst = float(((yt - yt.mean()) ** 2).sum())
        direct = {"exponent": float(k3), "intercept": float(c3),
                  "r_squared": 1.0 - ssr / sst if sst > 0 else None,
                  "note": "median check ms vs nnz -- THE headline number"}
        print(f"support growth: |z| ~ nnz^{k2:.3f}")
        print(f"DIRECT: check time ~ nnz^{k3:.3f}   R^2 = {direct['r_squared']:.4f}"
              f"   -> {'LINEAR in nnz' if k3 <= 1.15 else 'SUPERLINEAR in nnz'}")

    out = {"schema": "oneq-checker-scaling/2", "rows": rows,
           "fit_time_vs_work": fit, "fit_support_vs_nnz": supp_fit,
           "fit_time_vs_nnz": direct,
           "what_is_claimed": (
               "feasibility is one pass over the edges accumulating, per edge, "
               "the duals whose boundary it crosses -- O(nnz * |support(z)|), "
               "NOT O(nnz). The headline is only honest if |support(z)| stays "
               "small as the graph grows, so both fits are reported."),
           "why_the_producer_is_excluded": (
               "the producer is the expensive half and it is not what a "
               "verifier runs. Timing them together would hide the property "
               "the argument depends on: that CHECKING is cheap even though "
               "PRODUCING is not."),
           "method": ("minimum of repeated timings per shot -- the minimum is "
                      "the least contaminated by other load. Run on an "
                      "otherwise idle machine; a measurement taken under load "
                      "measures the load."),
           "falsifier": ("an exponent materially above 1.0 on nnz*|z| means "
                         "the claim is wrong")}
    dest = pathlib.Path(a.out) if a.out else ROOT / "evidence" / "checker_scaling.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nevidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
