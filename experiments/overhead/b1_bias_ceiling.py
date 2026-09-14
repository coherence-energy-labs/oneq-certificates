r"""B-1: the bias ceiling across six IBM devices, offline, and why the
ABSTAIN does not depend on which ceiling you believe.

THE SPEC (FTQC_IMPLEMENTATION, week 2) records this as pre-satisfied:

    reproduce the bias ceiling across six IBM devices, 30 s, offline
    => ABSTAIN 6/6, ceilings 1.52-3.56. Max ceiling 3.56 against a bar
       of 10. B-1 pre-satisfied before you write it.

No code in this repository produces it, so this is the independent
implementation -- same discipline as H-4, applied to the other overhead
number. `fake_provider` snapshots only: no network, no QPU, no token.

WHAT IS UNAMBIGUOUS, AND WHAT IS NOT. Most of the recorded table is
defined by the calibration data and nothing else:

  eta_idle    T1/T2 - 1/2, from the Pauli-twirled amplitude-damping +
              dephasing channel: p_x = p_y = t/(4 T1), p_z = t/(2 T2) -
              t/(4 T1), so the Z-to-(X+Y) ratio is T1/T2 - 1/2. It is 0
              exactly when T2 = 2 T1 (no pure dephasing), which is the
              right sanity check.
  T2 > 2T1    a PHYSICALITY violation, not a bias: T2 <= 2 T1 always, so
              any qubit above it is a calibration artefact and is counted
              rather than quietly averaged in.
  RO asym     max(p01, p10) / min(p01, p10) per qubit.
  error mass  the share of total error rate carried by 2Q gates, readout,
              idle and 1Q gates.

ETA_MAX IS NOT. The spec quotes ceilings 1.52-3.56 but does not state the
formula, and inventing one that reproduces 3.56 would be fitting to the
answer -- the exact failure this program spends its gates preventing. So
this file does the opposite: it computes the ceiling under THREE
independently defensible definitions, prints all three, and reports the
verdict only if they AGREE.

  strict     idle mass is perfectly biased, everything else depolarizing
  ro-biased  readout asymmetry counted as biased mass too (generous)
  optimistic every non-2Q error treated as perfectly biased (the most
             generous ceiling any accounting could justify)

The decision B-1 exists to make is ABSTAIN vs BUILD against a bar of 10.
If even the optimistic ceiling sits far below 10 on all six devices, the
ABSTAIN is robust to the choice of formula, and that is a stronger claim
than matching one unstated number.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BAR = 10.0          # the spec's bar: below this, a bias-tailored code is
                    # not worth its complexity

DEVICES = ["Marrakesh", "Torino", "Sherbrooke", "Fez", "Brisbane", "Kyiv"]

RECORDED = {   # the spec's table, for comparison -- not for fitting
    "Marrakesh": dict(n=156, eta_med=1.01, frac_gt3=0.19, t2_gt_2t1=1,
                      ro_med=2.21, eta_max=3.56),
    "Torino":    dict(n=133, eta_med=0.69, frac_gt3=0.09, t2_gt_2t1=5,
                      ro_med=1.72, eta_max=3.51),
    "Sherbrooke": dict(n=127, eta_med=1.05, frac_gt3=0.28, t2_gt_2t1=0,
                       ro_med=1.89, eta_max=2.17),
    "Fez":       dict(n=156, eta_med=1.10, frac_gt3=0.20, t2_gt_2t1=1,
                      ro_med=2.67, eta_max=2.71),
    "Brisbane":  dict(n=127, eta_med=0.99, frac_gt3=0.15, t2_gt_2t1=2,
                      ro_med=1.90, eta_max=2.37),
    "Kyiv":      dict(n=127, eta_med=1.80, frac_gt3=0.32, t2_gt_2t1=0,
                      ro_med=1.51, eta_max=1.52),
}


def load(name: str):
    import qiskit_ibm_runtime.fake_provider as fp
    return getattr(fp, f"Fake{name}")()


def device_report(name: str) -> dict:
    b = load(name)
    props = b.properties()
    n = b.num_qubits

    etas, ro_asym, t2_viol = [], [], 0
    ro_mass = 0.0
    idle_mass = 0.0
    for q in range(n):
        d = props.qubit_property(q)
        t1 = d.get("T1", (None,))[0]
        t2 = d.get("T2", (None,))[0]
        p01 = d.get("prob_meas0_prep1", (None,))[0]
        p10 = d.get("prob_meas1_prep0", (None,))[0]
        if t1 and t2:
            if t2 > 2 * t1 + 1e-15:
                t2_viol += 1          # unphysical: counted, never averaged
            else:
                etas.append(t1 / t2 - 0.5)
            # idle error mass proxy: the qubit's own decoherence rate
            idle_mass += 1.0 / t1 + 1.0 / t2
        if p01 and p10 and min(p01, p10) > 0:
            ro_asym.append(max(p01, p10) / min(p01, p10))
        ro = d.get("readout_error", (None,))[0]
        if ro:
            ro_mass += ro

    tgt = b.target
    two_q_mass = one_q_mass = 0.0
    for op in tgt.operation_names:
        try:
            insts = tgt[op]
        except Exception:              # noqa: BLE001
            continue
        for key, ip in insts.items():
            e = getattr(ip, "error", None)
            if e is None:
                continue
            if key is not None and len(key) == 2:
                two_q_mass += e
            elif op in ("x", "sx", "rz", "id"):
                one_q_mass += e

    # normalise the masses onto a common scale. The idle proxy is a RATE
    # (1/s) while the others are per-operation probabilities, so it is
    # converted with a representative 2Q duration -- stated, because a
    # hidden time constant here would silently move every ceiling.
    dur = []
    for op in ("ecr", "cz", "cx"):
        if op in tgt.operation_names:
            dur = [ip.duration for ip in tgt[op].values()
                   if getattr(ip, "duration", None)]
            break
    t_gate = float(np.median(dur)) if dur else 5e-7
    idle_mass_p = idle_mass * t_gate

    total = two_q_mass + ro_mass + idle_mass_p + one_q_mass
    m2q, mro, midle, m1q = (two_q_mass / total, ro_mass / total,
                            idle_mass_p / total, one_q_mass / total)

    eta_med = float(np.median(etas)) if etas else 0.0
    frac_gt3 = float(np.mean([e > 3 for e in etas])) if etas else 0.0

    # --- THREE ceilings, all stated, none fitted -------------------
    def ceil_from(biased: float, unbiased: float) -> float:
        """Perfectly biased mass over depolarizing mass. Depolarizing
        error contributes a third of itself to Z and two thirds to X+Y."""
        if unbiased <= 0:
            return float("inf")
        return (biased + unbiased / 3.0) / (2.0 * unbiased / 3.0)

    strict = ceil_from(midle, m2q + mro + m1q)
    ro_biased = ceil_from(midle + mro, m2q + m1q)
    optimistic = ceil_from(midle + mro + m1q, m2q)

    return {"device": name, "n": n,
            "eta_idle_med": round(eta_med, 3),
            "frac_eta_gt_3": round(frac_gt3, 3),
            "t2_exceeds_2t1": t2_viol,
            "ro_asym_med": round(float(np.median(ro_asym)), 3),
            "ro_asym_p90": round(float(np.percentile(ro_asym, 90)), 2),
            "ro_asym_max": round(float(np.max(ro_asym)), 1),
            "mass_2q": round(m2q, 3), "mass_ro": round(mro, 3),
            "mass_idle": round(midle, 3), "mass_1q": round(m1q, 3),
            "ceiling_strict": round(strict, 2),
            "ceiling_ro_biased": round(ro_biased, 2),
            "ceiling_optimistic": round(optimistic, 2),
            "abstain": max(strict, ro_biased, optimistic) < BAR}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    rows = []
    print(f"B-1 bias ceiling, offline snapshots, bar = {BAR}")
    print(f"  {'device':11s} {'n':>4s} {'eta_med':>8s} {'>3':>5s} "
          f"{'T2>2T1':>7s} {'RO_med':>7s} "
          f"{'2Q/RO/idle':>18s} {'strict':>7s} {'ro-b':>6s} {'optim':>6s}")
    for name in DEVICES:
        r = device_report(name)
        rows.append(r)
        print(f"  {r['device']:11s} {r['n']:4d} {r['eta_idle_med']:8.2f} "
              f"{r['frac_eta_gt_3']:5.2f} {r['t2_exceeds_2t1']:7d} "
              f"{r['ro_asym_med']:7.2f} "
              f"{r['mass_2q']:.3f}/{r['mass_ro']:.3f}/{r['mass_idle']:.3f}  "
              f"{r['ceiling_strict']:7.2f} {r['ceiling_ro_biased']:6.2f} "
              f"{r['ceiling_optimistic']:6.2f}")

    all_abstain = all(r["abstain"] for r in rows)
    worst = max(max(r["ceiling_strict"], r["ceiling_ro_biased"],
                    r["ceiling_optimistic"]) for r in rows)
    print(f"\n  worst ceiling under ANY of the three definitions: "
          f"{worst:.2f}  (bar {BAR})")
    print(f"  ABSTAIN {sum(r['abstain'] for r in rows)}/6"
          + ("  -- and the verdict does not depend on which ceiling you "
             "believe" if all_abstain else ""))

    # compare against the recorded table WITHOUT fitting to it
    print("\n  vs the spec's recorded table (comparison, not calibration):")
    for r in rows:
        rec = RECORDED[r["device"]]
        print(f"    {r['device']:11s} n {r['n']:4d}/{rec['n']:<4d} "
              f"eta_med {r['eta_idle_med']:5.2f}/{rec['eta_med']:<5.2f} "
              f"T2>2T1 {r['t2_exceeds_2t1']:2d}/{rec['t2_gt_2t1']:<2d} "
              f"RO_med {r['ro_asym_med']:5.2f}/{rec['ro_med']:<5.2f}")

    out = {"schema": "oneq-b1-bias-ceiling/1",
           "source": ("qiskit_ibm_runtime fake_provider calibration "
                      "snapshots -- offline, no network, no QPU, no token"),
           "bar": BAR,
           "devices": rows,
           "abstain_count": sum(r["abstain"] for r in rows),
           "worst_ceiling_any_definition": worst,
           "verdict": ("ABSTAIN 6/6" if all_abstain else "NOT UNANIMOUS"),
           "why_three_ceilings": (
               "the spec quotes ceilings 1.52-3.56 but does not state the "
               "formula. Inventing one that reproduces 3.56 would be "
               "fitting to the answer, so the ceiling is computed under "
               "three independently defensible definitions and the "
               "verdict is reported only because all three agree. That is "
               "a stronger claim than matching one unstated number: the "
               "ABSTAIN is robust to the accounting."),
           "definitions": {
               "strict": "idle mass perfectly biased, all else depolarizing",
               "ro_biased": "readout asymmetry counted as biased mass too",
               "optimistic": ("every non-2Q error perfectly biased -- the "
                              "most generous ceiling any accounting could "
                              "justify")},
           "unambiguous_quantities": (
               "eta_idle = T1/T2 - 1/2 (Pauli-twirled damping+dephasing; "
               "0 exactly when T2 = 2T1), T2 > 2T1 counted as a "
               "PHYSICALITY violation rather than averaged in, readout "
               "asymmetry max/min per qubit, and error-mass shares")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "b1_bias_ceiling.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nevidence -> {dest}")
    return 0 if all_abstain else 1


if __name__ == "__main__":
    raise SystemExit(main())
