"""Is the engine's stand-in for mid-circuit read + feed-forward (hw_dynamic.engine_circuit: virtual zero-error CZ edges
plus an exact mixture over the read-out's wrong correction) the same model as the REAL dynamic circuit? (R147)

Both sides get the SAME noise model, the engine's: on every CZ a uniform non-identity two-qubit Pauli with total
probability 1.25 x CZ factor x vendor error; on every sx/x a single-qubit Pauli with total 1.5 x sx error; on EVERY
measurement, mid-circuit ones included, the per-qubit asymmetric confusion (p01, p10) of the calibration. Idle is OFF
on both (Aer schedules none), which isolates exactly the stand-in. Aer runs the real if_test circuits shot by shot.
The R145 static circuits of the same units are run the same way, so dynamic - static is compared in ONE model.

  python validate_standin.py UNITS SHOTS
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, ".")
import hw_dynamic as D  # noqa: E402
import hw_faithful as FA  # noqa: E402
import hw_levers as V  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import noise as N  # noqa: E402

M = 5


def aer_noise(snap, phys_of, params):
    """an Aer NoiseModel on COMPACT indices (logical i -> physical phys_of[i]) mirroring oneq.hw.noise.compile_ops"""
    from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error
    nm = NoiseModel(basis_gates=["cz", "rz", "sx", "x", "measure", "if_else"])
    n = len(phys_of)
    for i in range(n):
        e = min(params.sx_factor * 1.5 * snap.sx_error[phys_of[i]], 0.75)
        if e > 0:
            nm.add_quantum_error(pauli_error([("X", e / 3), ("Y", e / 3), ("Z", e / 3), ("I", 1 - e)]), ["sx", "x"], [i])
        p01, p10 = snap.p01[phys_of[i]], snap.p10[phys_of[i]]
        nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [i])
    paulis = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = sorted((phys_of[i], phys_of[j]))
            if (a, b) not in snap.cz:
                continue
            p = min(params.cz_factor * 1.25 * snap.cz[(a, b)][0], 15 / 16)
            nm.add_quantum_error(pauli_error([(s, p / 15) for s in paulis] + [("II", 1 - p)]), ["cz"], [i, j])
    return nm


def compact_isa(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["cz", "rz", "sx", "x", "measure", "if_else"], optimization_level=1,
                     seed_transpiler=1)


def aer_mean(qc_isa, nm, score, shots, seed):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(noise_model=nm)
    res = sim.run(qc_isa, shots=shots, seed_simulator=seed).result().get_counts()
    nb = qc_isa.num_clbits
    vals = []
    for k, c in res.items():
        bits = tuple(int(x) for x in reversed(k.replace(" ", "")))[:nb]
        vals += [score(bits)] * c
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def main(units=10, shots=20000):
    be, snap = V.backend(), V.snapshot()
    params = N.NoiseParams(cz_factor=V.CZ_FACTOR_V4, idle=False)
    lay0, _ = D.hand_layout(snap, be, M)
    tau = float(np.median([x for x in snap.meas_duration if math.isfinite(x)])) + D.TAU_FF
    rng = np.random.default_rng(3)
    erng = np.random.default_rng(4)
    nm = aer_noise(snap, lay0, params)                  # dynamic circuits live on the hand layout, compact = logical
    d_stand, d_aer, rows = [], [], []
    for k in range(units):
        key = ("kept_kept" if k % 2 == 0 else "kept_erased", "s0", 0, 0)
        u = H.sample_in_stratum(M, key, rng)
        while not D.measure_type(u.choice):
            u = H.sample_in_stratum(M, key, rng)
        (c0, virt), (c1, _) = D.engine_circuit(M, u, "cz2", lay0, be, tau, early=True)
        sn = D.augmented(snap, virt)
        mt = D.measure_type(u.choice)
        r = 0.5 * (1 - math.prod(1 - 2 * snap.readout_error(lay0[4 + t]) for t in mt))
        r0 = E.expectation(c0, FA.score(M, u, "cz2"), sn, params, 60, erng)
        r1 = E.expectation(c1, FA.score(M, u, "cz2"), sn, params, 60, erng)
        stand = (1 - r) * r0.mean + r * r1.mean
        stand_se = math.hypot((1 - r) * r0.se, r * r1.se)          # the engine's own Monte Carlo error
        dyn = compact_isa(D.lean_dynamic(M, u, "cz2", early=True, one_register=True))
        a_mean, a_se = aer_mean(dyn, nm, FA.score(M, u, "cz2"), shots, 10 + k)
        # the SAME unit as the R145 static circuit, all-to-all on the same physical qubits' calibration (no routing):
        # isolates "measure-and-correct vs CZ" from routing
        stat = compact_isa(H.circuit(M, u))
        nm_s = aer_noise(snap, lay0, params)
        # static CZs between key0 and the shields have no calibrated edge on the hand layout: give them the
        # median calibrated CZ error, the cost the static design pays per key-shield CZ before any routing
        from qiskit_aer.noise import pauli_error
        med = float(np.median([v[0] for v in snap.cz.values()]))
        p = min(params.cz_factor * 1.25 * med, 15 / 16)
        paulis = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]
        for t in range(M):
            for pair in ((0, 4 + t), (4 + t, 0)):
                nm_s.add_quantum_error(pauli_error([(s, p / 15) for s in paulis] + [("II", 1 - p)]), ["cz"], list(pair))
        s_mean, s_se = aer_mean(stat, nm_s, H.score(M, u), shots, 100 + k)
        ideal = FA.lean_ideal(M, u, "cz2")
        rows.append((u.name, ideal, stand, a_mean, math.hypot(a_se, stand_se), s_mean))
        d_stand.append(stand - a_mean)
        print(f"{u.name[:44]:44s} ideal {ideal:.3f}  engine stand-in {stand:.4f}  Aer real dynamic {a_mean:.4f}"
              f"+-{a_se:.4f}  diff {stand - a_mean:+.4f} | Aer static (no routing) {s_mean:.4f}", flush=True)
        d_aer.append(a_mean - s_mean)
    se = math.sqrt(sum(r[4] ** 2 for r in rows)) / len(rows)
    print(f"\nmean (stand-in - Aer dynamic) = {np.mean(d_stand):+.4f} +- {se:.4f} (Aer shots AND engine trajectories)"
          f"   [same model: should be ~0; idle OFF on both sides, so the read-out hold tau is NOT tested here]")
    print(f"mean (Aer dynamic - Aer static, no routing) = {np.mean(d_aer):+.4f}   [the measure-and-correct trade "
          f"before routing: read-out error vs CZ error]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
