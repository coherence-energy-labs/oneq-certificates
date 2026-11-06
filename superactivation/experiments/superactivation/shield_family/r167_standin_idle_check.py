"""R167: is R147's engine stand-in for a mid-circuit read the device, once IDLES are on? (OFFLINE; Aer: the tw-venv)

R147's dynamic units keep a held-out miss of -0.033 +- 0.002 that neither the twirl dressing (R165) nor the measured
feed-forward stall (R163) moves. The engine has no mid-circuit read. Its stand-in (hw_dynamic.engine_circuit):
  - holds each read shield idle for tau (read + stall) right after Alice's encoding;
  - replaces the read and the conditional Z by a zero-error CZ(key, shield) on a VIRTUAL edge;
  - leaves the shield ENTANGLED (with its partner and the key) until pair decoding, which REUSES it;
  - charges every idle as a Pauli-twirled channel (oneq.hw.noise.idle_pauli), exact only under a random frame (R165).
validate_standin.py checked it against Aer running the real if_test circuits with idle OFF on both sides, so the
hold, the reuse and the relaxation of a projected shield were never tested.

Four predictions per unit, all from ONE op list -- the engine's own compile_ops of the frozen stand-in on V3c:
  engine     the stand-in prediction exactly as R147 / R152 compute it (hw_dynamic_run._predict_from)
  A replay   Aer runs the stand-in's ops as they are (virtual CZ, Pauli idles), c0/c1 mixed as the engine mixes:
             must equal the engine within Monte Carlo error, or the translation is wrong
  B read     the virtual CZ replaced by what the device does -- a real mid-circuit read of the shield at the START of
             its hold (after its last encoding gate) and a conditional Z on the key where the CZ was; Pauli idles.
             B - A is the deferred-read representation, with idles on
  C exact    B with every idle as EXACT thermal relaxation (amplitude damping toward |0>, state-dependent).
             C - B is the Pauli-twirled idle in this context; C - engine is the stand-in's whole error
The ops are translated by a PURE planner (plan_arm: no Aer), tested in the suite; only build_circuit needs Aer.

  <tw-venv>/Scripts/python.exe r167_standin_idle_check.py REPO_ROOT [--units N] [--shots S] [--traj T]
      -> evidence/superactivation/r167_standin/standin_idle_check.json
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

ARMS = ("A_replay", "B_read", "C_exact")
DIAGONAL = ("rz", "quasistatic", "detuning")      # commute with a Z read: may sit between the hold and the CZ


class PlanRefused(ValueError):
    """the op list cannot be translated faithfully"""


# ------------------------------------------------------------------ the pure planner
def aer_label(pair: tuple) -> str:
    """the engine applies PAULIS2 pair (a, b) as a on op.qubits[0] and b on op.qubits[1]; qiskit's labels are
    little-endian (the RIGHTMOST letter acts on the first qarg)"""
    a, b = pair
    return b + a


def idle_seconds(pp: tuple, t1: float, idle_factor: float = 1.0) -> float:
    """the idle's duration, recovered exactly from its Pauli-twirled probabilities: px = (1 - e^{-t/T1}) / 4"""
    px = pp[0] / idle_factor
    if not (0.0 < px < 0.25):
        raise PlanRefused(f"an idle with px = {pp[0]} does not encode a duration")
    return -t1 * math.log(1.0 - 4.0 * px)


def read_sites(ops: list, virtual: dict) -> dict:
    """per virtual CZ (op index): (key, shield, index AFTER which the shield is read). virtual: {sorted active pair:
    (key, shield)} -- ORIENTED as the stand-in made it (hw_dynamic.engine_circuit's virt edges are (key, shield));
    never inferred from the indices, which are sorted by PHYSICAL qubit. The read starts the hold: right after the
    shield's last non-diagonal gate before the CZ (idles and diagonal ops in between are the hold, or commute with a
    Z read)"""
    out = {}
    for k, op in enumerate(ops):
        if op.name != "cz" or tuple(sorted(op.qubits)) not in virtual:
            continue
        key, shield = virtual[tuple(sorted(op.qubits))]
        if set(op.qubits) != {key, shield}:
            raise PlanRefused(f"virtual pair {(key, shield)} does not match the CZ on {op.qubits}")
        site = None
        for j in range(k - 1, -1, -1):
            o = ops[j]
            if shield not in o.qubits or o.name == "idle" or o.name in DIAGONAL:
                continue
            site = j
            break
        if site is None:
            raise PlanRefused(f"virtual CZ {op.qubits} at op {k}: the shield has no gate before it")
        out[k] = (key, shield, site)
    return out


def oriented(virt: list, idx: dict) -> dict:
    """the stand-in's virtual edges [[key_phys, shield_phys], ...] as {sorted active pair: (key, shield)}"""
    out = {}
    for e in virt:
        key, shield = idx[int(e[0])], idx[int(e[1])]
        out[tuple(sorted((key, shield)))] = (key, shield)
    keys = {v[0] for v in out.values()}
    if len(keys) != 1:
        raise PlanRefused(f"the virtual edges name {len(keys)} keys; the stand-in has exactly one")
    return out


def plan_arm(ops: list, meas: list, arm: str, virtual: dict, t1: dict, t2: dict, idle_factor: float = 1.0) -> list:
    """the op list as a list of abstract instructions for one arm (pure; see the module docstring):
      ("unitary", matrix, qubits) | ("pauli1", (px, py, pz), q) | ("pauli2", {label: p}, (q0, q1)) |
      ("relax", t, T1, T2, q) | ("measure", q, clbit) | ("cond_z", key, clbit)
    measurement bits: the protocol's clbits as the engine maps them; the mid-circuit reads take clbits after them.
    t1, t2: per ACTIVE index (T2 clamped to 2 T1 as the engine clamps)"""
    if arm not in ARMS:
        raise PlanRefused(f"unknown arm {arm!r}; expected one of {ARMS}")
    from oneq.hw.noise import PAULIS2
    sites = read_sites(ops, virtual) if arm != "A_replay" else {}
    after_site: dict = {}
    n_prot = len(meas)
    mcm_bit = {}
    for i, (k, (key, shield, site)) in enumerate(sorted(sites.items())):
        after_site.setdefault(site, []).append((shield, n_prot + i))
        mcm_bit[k] = (key, n_prot + i)
    out: list = []
    for k, op in enumerate(ops):
        if k in mcm_bit:                               # the read's conditional Z replaces the virtual CZ
            key, b = mcm_bit[k]
            if op.p_error:
                raise PlanRefused(f"virtual CZ at op {k} carries an error {op.p_error}")
            out.append(("cond_z", key, b))
        elif op.name == "idle":
            if arm == "C_exact":
                q = op.qubits[0]
                out.append(("relax", idle_seconds(op.pauli_probs, t1[q], idle_factor), t1[q], min(t2[q], 2 * t1[q]),
                            q))
            else:
                out.append(("pauli1", tuple(op.pauli_probs), op.qubits[0]))
        else:
            if op.matrix is not None:
                out.append(("unitary", op.matrix, tuple(op.qubits)))
            if op.p_error > 0:
                if len(op.qubits) == 1:
                    out.append(("pauli1", tuple(op.pauli_probs), op.qubits[0]))
                elif len(op.qubits) == 2:
                    probs = op.pauli_probs or tuple(op.p_error / 15 for _ in PAULIS2)
                    out.append(("pauli2", {aer_label(p): pr for p, pr in zip(PAULIS2, probs) if pr > 0},
                                tuple(op.qubits)))
                else:
                    raise PlanRefused(f"op {op.name} on {len(op.qubits)} qubits")
        for shield, b in after_site.get(k, ()):
            out.append(("measure", shield, b))
    for q, c in meas:
        out.append(("measure", q, c))
    return out


SEED_SPAN = 10_000


def seed_for(unit: int, arm: int, part: int, shots: int) -> int:
    """a seed whose per-shot stream is DISJOINT from every other run's. Aer seeds shot i with seed + i, so two seeds
    closer than the shot count replay almost the same stream: seeds 1 and 2 gave counts one apart (measured
    2026-10-17), and seeds 10 apart made arms share their noise draws while their errors were combined as
    independent"""
    if not (0 < shots <= SEED_SPAN):
        raise PlanRefused(f"shots must be in (0, {SEED_SPAN}] for disjoint seed streams: {shots}")
    if not (0 <= arm < len(ARMS) and part in (0, 1) and unit >= 0):
        raise PlanRefused(f"no seed for unit {unit}, arm {arm}, part {part}")
    return 1 + SEED_SPAN * ((unit * len(ARMS) + arm) * 2 + part)


# ------------------------------------------------------------------ Aer (tw-venv only)
def build_circuit(instrs: list, n_qubits: int, n_clbits: int):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.noise import pauli_error, thermal_relaxation_error
    qc = QuantumCircuit(QuantumRegister(n_qubits, "q"), ClassicalRegister(n_clbits, "c"))
    for ins in instrs:
        kind = ins[0]
        if kind == "unitary":
            qc.unitary(ins[1], list(ins[2]))
        elif kind == "pauli1":
            px, py, pz = ins[1]
            if px + py + pz > 0:
                qc.append(pauli_error([("X", px), ("Y", py), ("Z", pz), ("I", 1 - px - py - pz)]).to_instruction(),
                          [ins[2]])
        elif kind == "pauli2":
            tot = sum(ins[1].values())
            qc.append(pauli_error(list(ins[1].items()) + [("II", 1 - tot)]).to_instruction(), list(ins[2]))
        elif kind == "relax":
            _, t, T1, T2, q = ins
            qc.append(thermal_relaxation_error(T1, T2, t).to_instruction(), [q])
        elif kind == "measure":
            qc.measure(ins[1], ins[2])
        elif kind == "cond_z":
            with qc.if_test((qc.clbits[ins[2]], 1)):
                qc.z(ins[1])
        else:
            raise PlanRefused(f"unknown instruction {kind}")
    return qc


def aer_score(qc, readout: dict, score, n_prot: int, shots: int, seed: int) -> tuple:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
    nm = NoiseModel()
    for q, (p01, p10) in readout.items():
        if p01 or p10:
            nm.add_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]), [q])
    sim = AerSimulator(noise_model=nm, method="statevector", seed_simulator=seed)
    counts = sim.run(qc, shots=shots).result().get_counts()
    vals = []
    for k, c in counts.items():
        bits = tuple(int(x) for x in reversed(k.replace(" ", "")))[:n_prot]
        vals.append((score(bits), c))
    n = sum(c for _, c in vals)
    mean = sum(v * c for v, c in vals) / n
    var = sum(c * (v - mean) ** 2 for v, c in vals) / n
    return float(mean), float(math.sqrt(var / n))


# ------------------------------------------------------------------ the run
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--units", type=int, default=24)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--traj", type=int, default=200)
    ap.add_argument("--replay-units", type=int, default=4, help="units that also run arm A (the translation check)")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    import hw_dynamic as D
    import hw_dynamic_run as R147
    import hw_faithful as FA
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    from oneq.hw import noise as N
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, 1)
    rec = P["record147"]
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    params = N.NoiseParams(cz_factor=1.0, schedule=P["params146"].schedule, quasistatic="coherent")
    stand = P["stand_ins"]
    dyn = [(k, u) for k, u in enumerate(rec["units"])
           if u["prediction_inputs"]["kind"] == "dynamic" and u["prediction_inputs"]["virt"]]
    pick = [dyn[int(i)] for i in np.linspace(0, len(dyn) - 1, min(a.units, len(dyn)))]
    rows = []
    t_start = time.time()
    for n_done, (k, u) in enumerate(pick):
        inp = u["prediction_inputs"]
        uu = R147.unit_from(u["unit"])
        score = FA.score(R147.M, uu, R147.DECODER)
        c0, c1 = stand[inp["c0"]], (stand[inp["c1"]] if inp["c1"] else None)
        eng = R147._predict_from(c0, c1, inp["virt"], inp["mix_qubits"], score, m7, np.random.default_rng([31, k]),
                                 params, a.traj)
        sn = D.augmented(m7, [tuple(e) for e in inp["virt"]])
        row = dict(unit=u["name"], kind=u["kind"], weight=u["weight"], engine=eng)
        for arm in ARMS if n_done < a.replay_units else ARMS[1:]:
            mixes = [(c0, 1.0)] if arm != "A_replay" or c1 is None else None
            if mixes is None:
                r = 0.5 * (1 - math.prod(1 - 2 * m7.readout_error(q) for q in inp["mix_qubits"]))
                mixes = [(c0, 1 - r), (c1, r)]
            mean, var = 0.0, 0.0
            for j, (circ, w) in enumerate(mixes):
                ops, active, meas = N.compile_ops(circ, sn, params)
                idx = {p: i for i, p in enumerate(active)}
                virtual = oriented(inp["virt"], idx)
                t1 = {i: m7.t1[p] for i, p in enumerate(active)}
                t2 = {i: m7.t2[p] for i, p in enumerate(active)}
                instrs = plan_arm(ops, meas, arm, virtual, t1, t2, params.idle_factor)
                n_mcm = sum(1 for x in instrs if x[0] == "measure") - len(meas)
                qc = build_circuit(instrs, len(active), len(meas) + n_mcm)
                ro = {i: (m7.p01[p], m7.p10[p]) for i, p in enumerate(active)}
                m_, s_ = aer_score(qc, ro, score, len(meas), a.shots, seed_for(k, ARMS.index(arm), j, a.shots))
                mean += w * m_
                var += (w * s_) ** 2
            row[arm] = (mean, math.sqrt(var))
        rows.append(row)
        el = time.time() - t_start
        print(f"[{n_done + 1}/{len(pick)} {el:5.0f} s] {u['name'][:40]:40s} engine {eng[0]:.4f}"
              + "".join(f"  {arm} {row[arm][0]:.4f}+-{row[arm][1]:.4f}" for arm in ARMS if arm in row), flush=True)

    def diff(x, y):
        d = [(r[x][0] if x != "engine" else r["engine"][0]) - (r[y][0] if y != "engine" else r["engine"][0])
             for r in rows if x in r and y in r]
        s = [math.hypot(r[x][1], r[y][1]) for r in rows if x in r and y in r]
        return dict(n=len(d), mean=float(np.mean(d)), se=float(math.sqrt(sum(v * v for v in s)) / len(d)))
    summary = {"A_replay - engine (translation check)": diff("A_replay", "engine"),
               "B_read - engine (deferred read, Pauli idles)": diff("B_read", "engine"),
               "C_exact - B_read (Pauli-twirled vs exact relaxation)": diff("C_exact", "B_read"),
               "C_exact - engine (the stand-in's whole error)": diff("C_exact", "engine")}
    out = dict(schema="oneq.r167.standin_idle_check/1", label="EXPLORATORY -- outside every preregistered verdict",
               r152_job=art["job"], model="V3c (R152 repair, vendor sx, R155 white T2 + sigma, CZ ladder x 1)",
               shots=a.shots, traj=a.traj, units_dynamic_total=len(dyn), rows=rows, summary=summary,
               r147_main_residual_to_explain=-0.033)
    d = root / "evidence" / "superactivation" / "r167_standin"
    d.mkdir(parents=True, exist_ok=True)
    (d / "standin_idle_check.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                               newline="\n")
    for k_, v in summary.items():
        print(f"{k_:56s} {v['mean']:+.4f} +- {v['se']:.4f}  ({v['n']} units)")
    print("wrote standin_idle_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
