"""R190: the shield size m for the ion run, decided on CHANNEL-FAITHFUL circuits.

WHY. The m = 6 design sits at its measured optima (the optimal two-use code, the best decoder, a program form with no
synchronization idle), and its faithful margin is only +0.017 to +0.024 (R189 part 2). The one lever never measured
on ions is a larger shield: the ideal two-use F_e rises with d = 2^m (0.7948 at m = 6, 0.8158 at m = 7, 0.8318 at
m = 8, the closed form), while each step adds 4 qubits and roughly 4 two-qubit gates per unit. Which wins is a
number, not an argument.

WHAT IT MEASURES. For each m and device model, N units drawn from the FULL design law (strata in proportion to their
weight) are evaluated with R186's exact evaluator in three compilations of the same unit:
  * faithful  -- the builder circuit translated with NO optimization (level 0): every gate the builder writes, which
                 is what the one-program Helios stream executes (R189: its executed entangling count equals the
                 builder's). On m = 6 it is compared with R189's measured executed arm;
  * barriers  -- the builder with barriers at both channel boundaries, level 3 (faithful; the model idles every qubit
                 at a barrier);
  * frozen    -- the builder at level 3 without barriers (NOT faithful: it merges across the channel), for reference.
The estimator is R186's control variate: f = f_ideal(m) - sum_s W_s mean_s(ideal_u - E[S|u]), exact ideal per unit,
stratified, with its standard error; F_e = 1.5 f - 0.5. The certificate's own view is reported too: only the HIGH
groups' errors (the ones that lower f), as R189's projection and the error-indicator certificate read them. Every
unit's ideal outcome must be deterministic (top or bottom of its support) for that certificate to apply; the
fraction is reported per m, and a non-deterministic unit is counted, never hidden.

  python r190_m_scan.py --m 6 7 8 --device spec:helios_1 spec:h2_2_mem56 --units 400   -> JSON on stdout
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

DECODER = "mix"
SUPPORT = {"kept": (0.0, 2.0), "ee_x": (0.0, 1.0)}


def _kind(u) -> str:
    if u.branch != "erased_erased":
        return "kept"
    return "ee_x" if u.inp[0] == "X" else "ee_const"


def scan_one(m: int, device: str, n_units: int, seed: int) -> dict:
    import hw_faithful as HF
    import hw_two_use as H
    import r186_faithful_requirements as R
    import r189_executed_projection as X
    from qiskit import transpile
    dev = R.make_device(device, m, DECODER)
    lay = dev.layout if dev.layout is not None else None
    strata = R.faithful_strata(m)
    W = np.array([w for _, w in strata])
    alloc = np.maximum(2, np.round(n_units * W / W.sum()).astype(int))
    rng = np.random.default_rng(seed)
    f_ideal = (2 * H.f_e_closed(m) + 1) / 3
    arms = ("faithful", "barriers", "frozen")
    loss_by = {a: [] for a in arms}
    high_by: dict = {a: {} for a in arms}
    n2q = {a: [] for a in arms}
    deterministic = nondet = 0
    t0 = time.time()
    for s, ((key, w), n_s) in enumerate(zip(strata, alloc)):
        per = {a: [] for a in arms}
        for j in range(int(n_s)):
            u = R.sample_faithful(m, key, rng)
            kind = _kind(u)
            score = HF.score(m, u, DECODER)
            qc = HF.circuit(m, u, DECODER, erase="none")
            circuits = {
                "faithful": transpile(qc, dev.backend, initial_layout=lay or list(range(qc.num_qubits)),
                                      optimization_level=0, coupling_map=dev.coupling_map),
                "barriers": dev.transpile(X.staged_circuit(m, u, DECODER), seed=7000 + j),
                "frozen": dev.transpile(qc, seed=7000 + j),
            }
            ideal0 = None
            for a, tq in circuits.items():
                ideal, full = (float(v) for v in R.evaluate(
                    R.compile_circuit(tq, dev.snap, dev.params, [R.IDEAL, R.FULL]), score, tq.num_clbits))
                if ideal0 is None:
                    ideal0 = ideal
                elif abs(ideal - ideal0) > 1e-9:
                    raise RuntimeError(f"m={m}: the {a} compile changed the unit's IDEAL ({ideal} vs {ideal0})")
                per[a].append(ideal - full)
                n2q[a].append(sum(1 for x in tq.data if x.operation.num_qubits == 2))
            if kind != "ee_const":
                lo, hi = SUPPORT[kind]
                p_top = (ideal0 - lo) / (hi - lo)
                if abs(p_top - 1) < 1e-9 or abs(p_top) < 1e-9:
                    deterministic += 1
                    sign = "high" if abs(p_top - 1) < 1e-9 else "low"
                    if sign == "high":
                        for a in arms:
                            g = high_by[a].setdefault(f"{kind}/high", dict(w=0.0, errs=[], width=hi - lo))
                            g["errs"].append(per[a][-1] / (hi - lo))
                else:
                    nondet += 1
            else:
                deterministic += 1
        for a in arms:
            loss_by[a].append((w, np.array(per[a])))
        print(f"   m={m} {device}: stratum {s + 1}/{len(strata)} ({int(n_s)} units), {time.time() - t0:.0f} s",
              file=sys.stderr, flush=True)
    out = dict(m=m, d=2 ** m, device=device, units=int(alloc.sum()), f_ideal=f_ideal,
               F_e_ideal=1.5 * f_ideal - 0.5, deterministic_ideal_units=deterministic, nondeterministic_units=nondet,
               seconds=round(time.time() - t0, 1), arms={})
    for a in arms:
        mean_loss = sum(w * x.mean() for w, x in loss_by[a])
        var = sum((w ** 2) * x.var(ddof=1) / len(x) for w, x in loss_by[a] if len(x) > 1)
        f = f_ideal - mean_loss
        out["arms"][a] = dict(F_e=1.5 * f - 0.5, F_e_se=1.5 * math.sqrt(var), mean_2q=float(np.mean(n2q[a])),
                              high_group_error={g: float(np.mean(v["errs"])) for g, v in high_by[a].items()})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, nargs="+", required=True)
    ap.add_argument("--device", nargs="+", required=True)
    ap.add_argument("--units", type=int, default=400)
    ap.add_argument("--seed", type=int, default=190)
    a = ap.parse_args(argv)
    res = [scan_one(m, dv, a.units, a.seed + 10 * m) for m in a.m for dv in a.device]
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
