"""Which levers move the two-use protocol toward 3/4 on heavy-hex, by the engine, on the R145 run-day calibration (R147).

For each (m, decoder) it predicts the RAW F_e of
  LEAN      the R145 branch compilation (hw_two_use.py) with Bob's pair decoder cz2 (as run), mix or cz1
            (hw_faithful.pair_decode), on a layout chosen by oneq.hw.layout for THAT decoder's circuits;
  FAITHFUL  the channel-faithful protocol (hw_faithful.py), obtained EXACTLY from the lean strata in the engine's model:
              - both kept: use 1's Bell pairs and P act after its copy is taken and never reach a measured qubit, so
                its F equals the lean one;
              - an erased use: every operation on its key preserves the Z basis and the key is read in Z, so P's real
                gates, the copy CNOT and any routing add only an independent flip of probability q_sigma (sigma = the
                sector of P's unitary on that use). The lean strata are split by that flip, so
                    F_faithful = sum_sigma W_sigma [(1 - q_sigma) V(sigma) + q_sigma V(1 - sigma)].
            q_sigma is the engine's flip probability of the use-1 subcircuit (key, copy, Bell pairs, P) routed on its own,
            averaged over P's law in sector sigma.
Scenarios (labelled; none is a measurement): the R145 model as frozen (CZ factor 2.01, measured); idle errors x 0.3
(a dynamical-decoupling what-if). Each kept stratum's value is its EXACT ideal mean minus the mean per-unit loss (the
control variate of hw_predict.py): the ideal varies with the input inside some strata, the loss barely does.

  python hw_levers.py M DECODER UNITS TRAJ [--tag T]
"""
from __future__ import annotations

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
import hardware_design as hd  # noqa: E402
import hw_faithful as F  # noqa: E402
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "shield_family" / "levers"
E.MAX_ACTIVE = 25           # routed m >= 7 circuits touch one ancilla beyond the default 24 (2^25 amplitudes, ~2 GiB peak)
FROZEN = ROOT / "evidence" / "superactivation" / "hw_run_m5" / "frozen_20261015T195926Z.json"
CZ_FACTOR_V4 = 2.01                       # R145's ladder, edge 47-48 (MEASURED)


def snapshot():
    rec = json.loads(FROZEN.read_text(encoding="utf-8"))
    return S.DeviceSnapshot.from_json(rec["snapshot"])


def backend():
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def representatives(m, decoder, seed=1):
    rng = np.random.default_rng(seed)
    units = [H.Unit("kept_kept", ("s1", (0,) * m), 0, ("Z", 1), (0, 0)),
             H.Unit("kept_erased", ("s1", (1,) * m), 1, ("Y", -1), (0, 1)),
             H.Unit("kept_kept", ("s0", ("I",) * m), 1, ("X", 1), (0, 0)),
             H.Unit("kept_kept", ("s0", ("Z",) * m), 0, ("X", 1), (0, 0)),
             H.sample_unit(m, "kept_kept", rng), H.sample_unit(m, "kept_erased", rng)]
    return [F.lean_circuit(m, u, decoder) for u in units]


def isa(be, qc, layout, seed):
    from qiskit import transpile
    return transpile(qc, be, initial_layout=layout, optimization_level=3, seed_transpiler=seed)


_EXACT: dict = {}


def stratum_exact(m, key):
    """the exact ideal mean score of a kept stratum: every input enumerated (the ideal depends on the input, not on
    which unitary of the sector was drawn -- P's decomposition is exact for every draw)"""
    k = (m, tuple(key))
    if k not in _EXACT:
        branch, sector, beta, f1 = key
        rng = np.random.default_rng(0)
        lab = (tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)) if sector == "s0"
               else tuple(int(x) for x in rng.integers(2, size=m)))
        _EXACT[k] = float(np.mean([F.lean_ideal(m, H.Unit(branch, (sector, lab), beta, inp, (0, f1)), "cz2")
                                   for inp in H.INPUTS]))
    return _EXACT[k]


def finish_row(m, key, W, ideal, noisy, czs):
    """control variate: value = exact stratum ideal - mean per-unit loss; se from the per-unit losses"""
    loss = np.array(ideal) - np.array(noisy)
    ex = stratum_exact(m, key)
    return dict(key=list(key), weight=W, exact_ideal=ex, sampled_ideal=float(np.mean(ideal)),
                raw_noisy=float(np.mean(noisy)), loss=float(loss.mean()),
                loss_se=float(loss.std(ddof=1) / math.sqrt(len(loss))), noisy=ex - float(loss.mean()),
                mean_cz=float(np.mean(czs)))


def lean_strata(m, decoder, units, traj, params, be, snap, layout, seed):
    unit_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    rows, i = {}, 0
    for key, W in H.strata(m):
        ideal, noisy, czs = [], [], []
        for _ in range(units):
            u = H.sample_in_stratum(m, key, unit_rng)
            tqc = isa(be, F.lean_circuit(m, u, decoder), layout, 1000 + i)
            i += 1
            r = E.expectation(tqc, F.score(m, u, decoder), snap, params, traj, rng)
            ncz = tqc.count_ops().get("cz", 0)
            ideal.append(F.lean_ideal(m, u, decoder))
            noisy.append(r.mean)
            czs.append(ncz)
        rows["|".join(map(str, key))] = finish_row(m, key, W, ideal, noisy, czs)
    ee = {}
    for w, u in H.erased_erased_units(m):
        tqc = isa(be, H.circuit(m, u), layout, 1)
        r = E.expectation(tqc, H.score(m, u), snap, params, traj, rng)
        ee[(u.inp, u.flips)] = r.mean
    return rows, ee


def q_flip(m, sector, params, be, snap, n=6, seed=5):
    """the engine's probability that use 1's own gates (copy CNOT, Bell pairs, P's real gates, routing) flip its key's
    Z read-out, averaged over P's law within the sector; the key starts in a Z eigenstate (Pauli noise: the flip law
    does not depend on the key's state)"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        lab = (tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)) if sector == 0
               else tuple(int(x) for x in rng.integers(2, size=m)))
        q = QuantumRegister(2 + 2 * m, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        s = [2 + t for t in range(m)]
        a = [2 + m + t for t in range(m)]
        for t in range(m):
            qc.h(a[t])
            qc.cx(a[t], s[t])
        qc.cx(0, 1)
        H._apply_p(qc, ("s0" if sector == 0 else "s1", lab), key=0, shields=s)
        qc.measure(0, 0)
        best = None
        for sd in range(1, 5):
            t = transpile(qc, be, optimization_level=3, seed_transpiler=sd)
            try:
                pf = L.proxy_log_fidelity(t, snap, params)
            except Exception:  # noqa: BLE001 -- an uncalibrated qubit: skip that routing
                continue
            if best is None or pf > best[0]:
                best = (pf, t)
        want = sector                                    # sector 1 flips the key (X); sector 0 does not
        r = E.expectation(best[1], lambda bits: 1.0 if bits[0] != want else 0.0, snap,
                          N.NoiseParams(**{**params.describe(), "readout": False}), 20, rng)
        vals.append(r.mean)
    return float(np.mean(vals))


def compose(m, rows, ee, q, field):
    """F_e of LEAN and of FAITHFUL from the strata (F_avg units), exactly as the docstring states"""
    p = H.p_flip(m)
    lean = sum(r["weight"] * r[field] for r in rows.values())
    faith = 0.0
    for r in rows.values():
        branch, sector, beta, f1 = r["key"]
        if branch == "kept_kept":
            faith += r["weight"] * r[field]
            continue
        other = rows["|".join(map(str, [branch, sector, beta, 1 - f1]))]
        faith += r["weight"] * ((1 - q[f1]) * r[field] + q[f1] * other[field])
    p_eff = (1 - p) * q[0] + p * (1 - q[1])
    lean_ee = faith_ee = 0.0
    for (inp, flips), v in ee.items():
        w = H.BRANCH_WEIGHT["erased_erased"] / 6
        lean_ee += w * math.prod(p if f else 1 - p for f in flips) * v
        faith_ee += w * math.prod(p_eff if f else 1 - p_eff for f in flips) * v
    to_fe = lambda fa: (3 * fa - 1) / 2  # noqa: E731
    return to_fe(lean + lean_ee), to_fe(faith + faith_ee)


def run(m, decoder, units, traj, tag=""):
    t0 = time.time()
    be, snap = backend(), snapshot()
    full = N.NoiseParams(cz_factor=CZ_FACTOR_V4)
    choice = L.choose_layout(representatives(m, decoder), be, snap, full)
    layout = choice.initial_layout
    out = dict(m=m, d=2 ** m, decoder=decoder, snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
               cz_factor=CZ_FACTOR_V4, layout=layout, layout_mean_cz=choice.best.mean_cz, units_per_stratum=units,
               trajectories=traj, F_e_ideal=H.f_e_closed(m), scenarios={})
    for name, params in (("model", full), ("dd_idle_x0.3", N.NoiseParams(cz_factor=CZ_FACTOR_V4, idle_factor=0.3))):
        rows, ee = lean_strata(m, decoder, units, traj, params, be, snap, layout, seed=7)
        q = {0: q_flip(m, 0, params, be, snap), 1: q_flip(m, 1, params, be, snap)}
        se = 1.5 * math.sqrt(sum((r["weight"] * r["loss_se"]) ** 2 for r in rows.values()))
        lean, faith = compose(m, rows, ee, q, "noisy")
        out["scenarios"][name] = dict(lean=lean, faithful=faith, se=se, q=q,
                                      mean_cz_kept=float(np.mean([r["mean_cz"] for r in rows.values()])),
                                      strata=rows)
        print(f"m={m} {decoder:4s} {name:13s} lean {lean:.4f} faithful {faith:.4f} "
              f"(+-{se:.4f}; q0 {q[0]:.4f} q1 {q[1]:.4f}; "
              f"CZ {out['scenarios'][name]['mean_cz_kept']:.1f})", flush=True)
    out["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"levers_m{m}_{decoder}{('_' + tag) if tag else ''}.json").write_text(
        json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    run(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
        tag=sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "")


# ---------------------------------------------------------------- an all-to-all device (trapped-ion class), and the
# composition theorem checked directly on it (no routing, so the lean, faithful and use-1 circuits share their gates)

HELIOS = dict(e2=8e-4, e1=3e-5, ro=3e-4, mem=5e-4)   # Quantinuum Helios data sheet v1.00, Table 1, typical values


def all_to_all(n, e2=HELIOS["e2"], e1=HELIOS["e1"], ro=HELIOS["ro"], mem=HELIOS["mem"]):
    """(backend, snapshot): full connectivity; vendor 2q error e2 (the engine's Pauli rate is 1.25 e2 x cz_factor),
    1q e1, symmetric readout ro, and dephasing T2 set so one 2q-gate layer costs `mem` per idle qubit"""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap
    be = GenericBackendV2(num_qubits=n, basis_gates=["cz", "sx", "x", "rz", "id"],
                          coupling_map=CouplingMap.from_full(n), seed=1)
    tau = 200e-6                                        # one 2q layer
    t2 = tau / (2 * mem)                                # idle_pauli: pz ~ t / (2 T2)
    snap = S.DeviceSnapshot(name=f"all_to_all_{n}", num_qubits=n, calibrated_utc=None, dt=1e-9,
                            cz={(a, b): (e2, tau) for a in range(n) for b in range(a + 1, n)},
                            t1=(1e3,) * n, t2=(t2,) * n, p01=(ro,) * n, p10=(ro,) * n, sx_error=(e1,) * n,
                            sx_duration=(10e-6,) * n, meas_duration=(tau,) * n)
    return be, snap


def theorem_check(m=3, units=4, traj=40, seed=3, idle=True):
    """per matched unit: engine(faithful circuit) vs (1 - q) V(sigma) + q V(1 - sigma), with V from the lean circuit
    of the same use-0 unit and q from the same use-1 gates; kept_kept: engine(faithful) vs V(lean)"""
    be, snap = all_to_all(4 + 4 * m)
    params = N.NoiseParams(cz_factor=2.0, idle=idle)
    rng = np.random.default_rng(seed)
    erng = np.random.default_rng(seed + 1)
    ident = list(range(4 + 4 * m))
    worst = []
    for branch in ("kept_kept", "kept_erased"):
        for _ in range(units):
            c0 = hd.sample_choice(m, rng)
            c1 = hd.sample_choice(m, rng)
            beta = int(c0[0] == "s1")                    # the matched basis: the informative half of the design
            inp = H.INPUTS[int(rng.integers(6))]
            fu = F.FUnit(branch, (c0, c1), beta, inp)
            direct = E.expectation(isa(be, F.circuit(m, fu, "cz1", erase="none"), ident, 1), F.score(m, fu, "cz1"),
                                   snap, params, traj, erng)
            sig = int(c1[0] == "s1")

            def lean_v(f1):
                lu = H.Unit(branch, c0, beta, inp, (0, f1))
                lq = F.lean_circuit(m, lu, "cz1")
                return E.expectation(isa(be, lq, list(range(lq.num_qubits)), 1), F.score(m, lu, "cz1"), snap, params,
                                     traj, erng)
            if branch == "kept_kept":
                v = lean_v(0)
                comp, se = v.mean, v.se
            else:
                q = q_single(m, c1, params, be, snap, erng)
                v0, v1 = lean_v(sig), lean_v(1 - sig)
                comp, se = (1 - q) * v0.mean + q * v1.mean, math.hypot(v0.se, v1.se)
            worst.append((branch, direct.mean, comp, math.hypot(direct.se, se)))
            print(f"  {branch:12s} direct {direct.mean:.5f}  composed {comp:.5f}  diff {direct.mean - comp:+.5f} "
                  f"(+-{math.hypot(direct.se, se):.5f})", flush=True)
    return worst


def q_single(m, choice, params, be, snap, rng):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    q = QuantumRegister(2 + 2 * m, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c)
    s = [2 + t for t in range(m)]
    a = [2 + m + t for t in range(m)]
    for t in range(m):
        qc.h(a[t])
        qc.cx(a[t], s[t])
    qc.cx(0, 1)
    H._apply_p(qc, choice, key=0, shields=s)
    qc.measure(0, 0)
    want = int(choice[0] == "s1")
    r = E.expectation(isa(be, qc, list(range(qc.num_qubits)), 1), lambda b: 1.0 if b[0] != want else 0.0, snap,
                      N.NoiseParams(**{**params.describe(), "readout": False}), 20, rng)
    return r.mean


def ion_predict(m, decoder="cz2", units=6, traj=30, cz_factor=1.0, idle_factor=1.0, seed=7):
    """the Helios-class all-to-all model (HELIOS above): no routing, so lean and faithful use the R145 circuits as
    they are. cz_factor scales the vendor 2q error to an in-circuit rate (1 = the data sheet; IBM's in-circuit rate
    was ~2x its vendor figure)."""
    n = 4 + 2 * m
    be, snap = all_to_all(n)
    params = N.NoiseParams(cz_factor=cz_factor, idle_factor=idle_factor)
    layout = list(range(n))
    rows, ee = lean_strata(m, decoder, units, traj, params, be, snap, layout, seed)
    be1, snap1 = all_to_all(2 + 2 * m)
    rng = np.random.default_rng(seed + 9)
    q = {}
    for sec in (0, 1):
        vals = []
        for _ in range(4):
            lab = (("s0", tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m))) if sec == 0
                   else ("s1", tuple(int(x) for x in rng.integers(2, size=m))))
            vals.append(q_single(m, lab, params, be1, snap1, rng))
        q[sec] = float(np.mean(vals))
    lean, faith = compose(m, rows, ee, q, "noisy")
    se = 1.5 * math.sqrt(sum((r["weight"] * r["loss_se"]) ** 2 for r in rows.values()))
    rec = dict(m=m, decoder=decoder, device="all-to-all, Helios data-sheet typical values", helios=HELIOS,
               cz_factor=cz_factor, idle_factor=idle_factor, F_e_ideal=H.f_e_closed(m), F_e_lean=lean,
               F_e_faithful=faith, se=se, q=q, strata=rows,
               mean_cz_kept=float(np.mean([r["mean_cz"] for r in rows.values()])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ion_m{m}_{decoder}_cz{cz_factor:g}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n",
                                                                  encoding="utf-8")
    print(f"ION m={m} {decoder} cz x{cz_factor:g}: lean {lean:.4f} faithful {faith:.4f} +- {se:.4f} (ideal "
          f"{H.f_e_closed(m):.4f}; CZ {rec['mean_cz_kept']:.1f}; q {q[0]:.4f}/{q[1]:.4f})", flush=True)
    return rec
