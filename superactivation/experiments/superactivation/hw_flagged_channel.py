r"""Driver for docs/PREREG_superactivation_hw_floor_2026-10-12.md -- the flagged superactivation
channel on half of a Bell pair, the transfer noise floor, and the per-CZ error, on ibm_kingston.

Modes:
  --validate   build all 25 circuits, check the pinned layout's live calibration, transpile, print
               the plan and the quota; submits NOTHING (default)
  --submit     OWNER ACT: requires --i-understand-this-spends-credits AND ONEQ_IBM_SUBMIT=1, the
               prereg hash to match, the layout to be healthy, and >= 120 s of quota remaining
  --fetch      pull the counts for the archived job and write the analysis artifact
Analysis is pure arithmetic on counts (`analyse`), so it is testable without the device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
PREREG = ROOT / "docs" / "PREREG_superactivation_hw_floor_2026-10-12.md"
PREREG_SHA256 = "b37938fd9c5f66c8c63a6e39e34c956f0dd5dfbadec487d4eeb3d7dac25a68b8"
RECORDS = ROOT / "evidence" / "superactivation"
BACKEND = "ibm_kingston"
LAYOUT = [148, 149]                 # R, A
SHOTS = 4000
PAULIS = ("I", "X", "Y", "Z")
BASES = ("ZZ", "XX", "YY")
DEPTHS = (2, 4, 8)
CALS = ("00", "01", "10", "11")
P_FLIP = math.sqrt(2.0) - 1.0
MIN_QUOTA_S = 120
F_IDEAL = 0.5 + (1 - P_FLIP) / 4    # 0.646447
DELTA_BUDGET = 1.38020e-4 / (0.7501380 - 0.25)   # certified n=17 margin / (F - 1/4)


def prereg_ok() -> bool:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


# ------------------------------------------------------------------ circuits
def circuits():
    """(name, QuantumCircuit) for every circuit in the preregistration, in a fixed order."""
    from qiskit import QuantumCircuit
    out = []

    def bell_then(insert, basis, name):
        qc = QuantumCircuit(2, 2, name=name)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        insert(qc)
        qc.barrier()
        for q in (0, 1):
            if basis[q] == "X":
                qc.h(q)
            elif basis[q] == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure([0, 1], [0, 1])
        return qc

    for s in PAULIS:
        for b in BASES:
            def ins(qc, s=s):
                if s == "X":
                    qc.x(1)
                elif s == "Y":
                    qc.y(1)
                elif s == "Z":
                    qc.z(1)
                else:
                    qc.id(1)
            out.append((f"pauli_{s}_{b}", bell_then(ins, b, f"pauli_{s}_{b}")))
    for m in DEPTHS:
        for b in BASES:
            def ins(qc, m=m):
                for i in range(m):
                    qc.cz(0, 1)
                    if i < m - 1:
                        qc.barrier()
            out.append((f"depth_{m}_{b}", bell_then(ins, b, f"depth_{m}_{b}")))
    for c in CALS:
        qc = QuantumCircuit(2, 2, name=f"cal_{c}")
        for q, bit in enumerate(c):          # c[q] is qubit q's prepared bit
            if bit == "1":
                qc.x(q)
        qc.measure([0, 1], [0, 1])
        out.append((f"cal_{c}", qc))
    return out


# ------------------------------------------------------------------ analysis (counts only)
def _bits(key: str):
    """Qiskit count keys are 'c1c0'; return (bit of qubit 0, bit of qubit 1)."""
    k = key.replace(" ", "")
    return int(k[-1]), int(k[-2])


def correlator(counts: dict) -> tuple[float, float, int]:
    n = sum(counts.values())
    s = sum(v * (1 if (_bits(k)[0] ^ _bits(k)[1]) == 0 else -1) for k, v in counts.items())
    mean = s / n
    return mean, math.sqrt(max(1 - mean * mean, 0.0) / n), n


def bell_fidelity(cxx, cyy, czz):
    (xx, sx, _), (yy, sy, _), (zz, sz, _) = correlator(cxx), correlator(cyy), correlator(czz)
    F = (1 + xx - yy + zz) / 4
    return F, math.sqrt(sx ** 2 + sy ** 2 + sz ** 2) / 4


def assignment_matrix(cal: dict) -> np.ndarray:
    """A[measured, prepared] over the 4 two-bit states, index = 2*bit(q1) + bit(q0)."""
    A = np.zeros((4, 4))
    for c, counts in cal.items():
        prep = 2 * int(c[1]) + int(c[0])
        n = sum(counts.values())
        for k, v in counts.items():
            b0, b1 = _bits(k)
            A[2 * b1 + b0, prep] += v / n
    return A


def mitigate(counts: dict, Ainv: np.ndarray) -> dict:
    n = sum(counts.values())
    vec = np.zeros(4)
    for k, v in counts.items():
        b0, b1 = _bits(k)
        vec[2 * b1 + b0] += v
    q = Ainv @ vec
    return {f"{i >> 1}{i & 1}": float(q[i]) for i in range(4)} | {"_n": n}


def _mit_counts(counts, Ainv):
    m = mitigate(counts, Ainv)
    n = m.pop("_n")
    # back to pseudo-counts with key 'c1c0'
    return {k: v for k, v in m.items()}, n


def analyse(counts: dict) -> dict:
    """counts: name -> {bitstring: count}. Pure arithmetic; every number in the artifact."""
    res = {"raw": {}, "mitigated": {}}
    cal = {c: counts[f"cal_{c}"] for c in CALS}
    A = assignment_matrix(cal)
    Ainv = np.linalg.inv(A)
    for mode in ("raw", "mitigated"):
        def get(name):
            if mode == "raw":
                return counts[name]
            mc, _ = _mit_counts(counts[name], Ainv)
            return {k: max(v, 0.0) for k, v in mc.items()}
        F = {}
        for s in PAULIS:
            F[s] = bell_fidelity(get(f"pauli_{s}_XX"), get(f"pauli_{s}_YY"), get(f"pauli_{s}_ZZ"))
        depth = {0: F["I"]}
        for m in DEPTHS:
            depth[m] = bell_fidelity(get(f"depth_{m}_XX"), get(f"depth_{m}_YY"), get(f"depth_{m}_ZZ"))
        FI, sI = F["I"]
        delta0 = 4 * (1 - FI) / 3
        wts = {"I": (1 - P_FLIP) / 2, "X": P_FLIP / 2, "Y": P_FLIP / 2, "Z": (1 - P_FLIP) / 2}
        Fn = 0.5 * FI + 0.5 * sum(wts[s] * F[s][0] for s in PAULIS)
        sn = math.sqrt((0.5 + 0.5 * wts["I"]) ** 2 * sI ** 2
                       + sum((0.5 * wts[s]) ** 2 * F[s][1] ** 2 for s in "XYZ"))
        model = (1 - delta0) * F_IDEAL + delta0 / 4
        # Fn - model is LINEAR in (F_I, F_X, F_Y, F_Z): both use F_I, so the standard error of the
        # difference comes from its own coefficients, not from adding two independent errors.
        cI = 0.5 + 0.5 * wts["I"] - (F_IDEAL - 0.25) * 4 / 3
        se_diff = math.sqrt((cI * sI) ** 2 + sum((0.5 * wts[s] * F[s][1]) ** 2 for s in "XYZ"))
        ms = np.array(sorted(depth))
        # Fit the DEPOLARIZING PARAMETER lam = (4F - 1)/3, not F: under the stated model
        # lam(m) = lam0 (1 - eps)^m exactly, while F = (1 + 3 lam)/4 decays slower and a fit of
        # log F under-reads eps by ~25% (caught by test_superactivation_hw_analysis before any run).
        lams = np.array([(4 * depth[m][0] - 1) / 3 for m in ms])
        slope = float(np.polyfit(ms, np.log(np.clip(lams, 1e-12, None)), 1)[0])
        eps = 1 - math.exp(slope)
        res[mode] = {
            "F_per_pauli": {s: {"F": F[s][0], "se": F[s][1]} for s in PAULIS},
            "F_depth": {str(m): {"F": depth[m][0], "se": depth[m][1]} for m in ms},
            "delta0": delta0, "delta0_se": 4 * sI / 3,
            "F_flagged_channel": Fn, "F_flagged_channel_se": sn,
            "F_flagged_ideal": F_IDEAL, "F_flagged_model_from_delta0": model,
            "flagged_minus_model": Fn - model, "flagged_minus_model_se": se_diff,
            "P2_within_2se": bool(abs(Fn - model) <= 2 * se_diff),
            "eps_CZ": eps,
            "infeasibility_factor_R": 16 * eps / DELTA_BUDGET,
        }
    res["assignment_matrix"] = A.tolist()
    res["delta_budget"] = DELTA_BUDGET
    r = res["raw"]
    res["predictions"] = {
        "P1_delta0_in_[0.003,0.03]": bool(0.003 <= r["delta0"] <= 0.03),
        "P2_flagged_channel_matches_model": r["P2_within_2se"],
        "P3_epsCZ_in_[0.001,0.015]": bool(0.001 <= r["eps_CZ"] <= 0.015),
        "P4_R_at_least_50": bool(r["infeasibility_factor_R"] >= 50),
    }
    return res


# ------------------------------------------------------------------ device modes
def _service_backend():
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    return svc, svc.backend(BACKEND)


def _layout_health(backend) -> list[str]:
    t = backend.target
    bad = []
    for q in LAYOUT:
        e = t["measure"][(q,)].error
        if e is None or e > 0.05:
            bad.append(f"qubit {q} readout error {e}")
    cz = t["cz"].get(tuple(LAYOUT)) or t["cz"].get(tuple(reversed(LAYOUT)))
    if cz is None or cz.error is None or cz.error > 0.01:
        bad.append(f"pair {LAYOUT} CZ error {None if cz is None else cz.error}")
    return bad


def _transpile(backend):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=LAYOUT)
    named = circuits()
    return [n for n, _ in named], [pm.run(qc) for _, qc in named]


def mode_validate() -> int:
    print(f"prereg hash matches: {prereg_ok()}")
    svc, backend = _service_backend()
    bad = _layout_health(backend)
    print("layout health:", "OK" if not bad else bad)
    names, isa = _transpile(backend)
    n2q = [sum(1 for ins in c.data if ins.operation.num_qubits == 2) for c in isa]
    used = sorted({isa[0].find_bit(q).index for ins in isa[0].data for q in ins.qubits})
    print(f"{len(isa)} circuits x {SHOTS} shots = {len(isa) * SHOTS} shots; two-qubit gates per circuit: {n2q}")
    print(f"physical qubits used by the first circuit: {used} (pinned {LAYOUT})")
    u = svc.usage()
    print(f"quota: consumed {u.get('usage_consumed_seconds')} of {u.get('usage_limit_seconds')} s")
    return 0 if not bad else 3


def mode_submit(a) -> int:
    if not (a.i_understand_this_spends_credits and os.environ.get("ONEQ_IBM_SUBMIT") == "1"):
        print("REFUSED: needs --i-understand-this-spends-credits AND ONEQ_IBM_SUBMIT=1")
        return 3
    if not prereg_ok():
        print("REFUSED: preregistration does not hash to the pinned value")
        return 3
    svc, backend = _service_backend()
    bad = _layout_health(backend)
    if bad:
        print(f"REFUSED (postpone, do not re-site): {bad}")
        return 3
    u = svc.usage()
    remaining = (u.get("usage_limit_seconds") or 0) - (u.get("usage_consumed_seconds") or 0)
    if remaining < MIN_QUOTA_S:
        print(f"REFUSED: {remaining} s of quota remaining < {MIN_QUOTA_S}")
        return 3
    from qiskit_ibm_runtime import SamplerV2
    names, isa = _transpile(backend)
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run(isa)
    rec = {"prereg": PREREG.name, "prereg_sha256": PREREG_SHA256, "backend": backend.name,
           "layout": LAYOUT, "shots": SHOTS, "circuit_names": names, "job_id": job.job_id(),
           "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quota_remaining_before_s": remaining}
    RECORDS.mkdir(parents=True, exist_ok=True)
    p = RECORDS / f"hw_floor_{job.job_id()}.job.json"
    p.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(f"submitted {len(isa)} circuits to {backend.name}: job {job.job_id()} -> {p}")
    return 0


def mode_fetch(a) -> int:
    jobs = sorted(RECORDS.glob("hw_floor_*.job.json"))
    if not jobs:
        print("no archived job")
        return 2
    rec = json.loads(jobs[-1].read_text(encoding="utf-8"))
    svc, _ = _service_backend()
    job = svc.job(rec["job_id"])
    st = str(job.status())
    if "DONE" not in st:
        print(f"job {rec['job_id']}: {st}")
        return 4
    result = job.result()
    counts = {name: result[i].data.c.get_counts() for i, name in enumerate(rec["circuit_names"])}
    spans = result.metadata.get("execution", {}).get("execution_spans")
    out = {"job": rec, "counts": counts, "analysis": analyse(counts),
           "execution_seconds": (spans.stop - spans.start).total_seconds() if spans else None}
    usage = None
    try:
        usage = job.usage()
    except Exception:  # noqa: BLE001
        pass
    out["job_usage_seconds"] = usage
    dest = RECORDS / f"hw_floor_{rec['job_id']}.result.json"
    dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(out["analysis"]["raw"], indent=2))
    print(json.dumps(out["analysis"]["predictions"], indent=2))
    print(f"-> {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    if a.submit:
        return mode_submit(a)
    if a.fetch:
        return mode_fetch(a)
    return mode_validate()


if __name__ == "__main__":
    sys.exit(main())
