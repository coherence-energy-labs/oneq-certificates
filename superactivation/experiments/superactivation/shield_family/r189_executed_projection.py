"""R189 part 2: the device model evaluated on the stream the one-program run EXECUTES, not on the frozen circuits' own
compilation.

WHY. R186 part 2's projections (H2-2 0.7803, Helios 0.7819 at m = 6) were computed on the frozen builder circuits
transpiled at optimization level 3 for each device model. The parameterized-shot program executes a different
compilation of the same unitaries: Guppy's per-gate lowering, with a classical branch between gates, so no gate is
merged across a branch. A projection derived from one compilation and quoted for another is the trap this repository
has recorded before (a fresh freeze compiles today's builder, not the circuit the projection came from). This lane
measures the difference instead of assuming it away.

WHAT IT DOES. For every frozen unit it evaluates, with R186's exact evaluator and one device model:
  * FROZEN: the builder circuit transpiled as the R188 dry run does (`dev.transpile`, level 3);
  * EXECUTED: the shot's native stream from `tools/verify_ion_param_shots.py --dump-native` (rz/rx/rzz, plus the
    measured qubits in clbit order), translated to the model's basis at optimization level 0 -- translation only, so
    every executed gate is played;
and requires the executed stream's IDEAL score to be the unit's frozen ideal exactly. Each arm's per-unit error
rate e = (ideal - E[score]) / width then gives the group means and the projected F_e = 1.5 (f_ideal - sum_g width_g
W_g ebar_g) - 0.5; the two arms are evaluated on the SAME units, so their difference is paired.

  python r189_executed_projection.py DESIGN_DIR NATIVE_JSONL DEVICE [--limit N]   -> JSON on stdout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


class ProjectionRefused(ValueError):
    pass


def executed_circuit(rec: dict):
    """the dumped native stream + its terminal measurements, clbit j <- measured[j]"""
    import math
    from qiskit import ClassicalRegister, QuantumCircuit, qasm2
    # rzz is not in strict qelib1.inc; qiskit's dumper writes it as its legacy-include gate
    body = qasm2.loads(rec["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    meas = list(rec["measured"])
    c = ClassicalRegister(len(meas), "c")
    qc = QuantumCircuit(*body.qregs, c)
    # ONE NATIVE ENTANGLING GATE IS ONE CZ. Every executed two-qubit gate is ZZ(+-pi/2) (66,167 of 66,167 on
    # 2026-10-19), and ZZ(t) = CZ . Rz(t) x Rz(t) up to a global phase for t = +-pi/2 (exact to 3e-16). Left as rzz,
    # qiskit's level-0 translation into the model's CZ basis plays it as TWO CZs, and the first projection read
    # F_e 0.7544 instead of 0.7801 -- the translation's cost, not the device's. Any other angle is refused, never
    # priced by a guess
    for inst in body.data:
        qs = [body.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "rzz":
            t = math.remainder(float(inst.operation.params[0]), 2 * math.pi)
            if abs(abs(t) - math.pi / 2) > 1e-9:
                raise ProjectionRefused(f"an executed ZZ({t}) is not +-pi/2: it is not one CZ, and no price is "
                                        "assumed for it")
            qc.rz(t, qs[0])
            qc.rz(t, qs[1])
            qc.cz(qs[0], qs[1])
        else:
            qc.append(inst.operation, qs)
    qc.measure(meas, list(range(len(meas))))
    return qc


def staged_circuit(m: int, u, decoder: str, barriers: bool = True):
    """hw_faithful.circuit(m, u, decoder, erase='none') with a barrier on every qubit after Alice's encoding and after
    the channels -- the CHANNEL-FAITHFUL compilation boundary. Without barriers it is gate for gate the builder's
    circuit (tests/test_r189_executed_projection.py holds that), so the two arms differ only in what an optimizer may
    merge. Measured 2026-10-19 on 250 kept units: level 3 removes 7 CZs per unit without the barriers and none with
    them, i.e. every saving merged Alice's gates with Bob's THROUGH a channel that is the identity in that branch --
    a shortcut a channel-faithful demonstration may not take"""
    import hw_faithful as HF
    import hw_two_use as H
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    L = HF.Q(m)
    q = QuantumRegister(L.n, "q")
    pre = QuantumCircuit(q)
    H._prep(pre, 0, u.inp)
    pre.cx(0, 1)
    pre.h(0)
    pre.h(1)
    for uu in range(2):
        for t in range(m):
            pre.h(L.a(uu, t))
            pre.cx(L.a(uu, t), L.s(uu, t))
        pre.cx(uu, 2 + uu)
    if barriers:
        pre.barrier()
    for uu in range(2):
        H._apply_p(pre, u.choices[uu], key=uu, shields=[L.s(uu, t) for t in range(m)])
    if barriers:
        pre.barrier()
    if u.branch == "erased_erased":
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c, name=u.name)
        qc.compose(pre, inplace=True)
        qc.measure([0, 1], [0, 1])
        return qc
    body = pre.copy()
    body.cx(0, 2)
    bits = []
    for t in range(m):
        bits += HF.pair_decode(body, L.a(0, t), L.s(0, t), u.beta, decoder)
    H._measure_in(body, 0, HF.BOB_BASIS[u.inp[0]][0])
    order = [2] + bits + [3 if u.branch == "kept_kept" else 1, 0]
    c = ClassicalRegister(len(order), "c")
    qc = QuantumCircuit(q, c, name=u.name)
    qc.compose(body, inplace=True)
    qc.measure(order, list(range(len(order))))
    return qc


ALL_ARMS = ("frozen", "barriers", "exec")


def project(design: dict, natives: list, device: str, limit: int | None = None, arms: tuple = ALL_ARMS) -> dict:
    import hw_faithful as HF
    import ion_run_package as P
    import r186_faithful_requirements as R
    from qiskit import transpile
    units = design["units"]
    by_shot = {int(r["shot"]): r for r in natives}
    n = len(units) if limit is None else min(limit, len(units))
    missing = [i for i in range(n) if i not in by_shot]
    if missing:
        raise ProjectionRefused(f"{len(missing)} units have no executed stream, e.g. shots {missing[:5]}")
    dev = R.make_device(device, design["m"], design["decoder"])
    lay = dev.layout if dev.layout is not None else None
    rows, t0 = [], time.time()
    for i in range(n):
        u = units[i]
        rec = by_shot[i]
        if int(rec["uid"]) != i:
            raise ProjectionRefused(f"shot {i} carries unit {rec['uid']}")
        unit = P._unit(u)
        qc = HF.circuit(design["m"], unit, design["decoder"], erase="none")
        if hashlib.sha256(P.export_qasm(qc).encode("utf-8")).hexdigest() != u["qasm_sha256"]:
            raise ProjectionRefused(f"{u['name']}: the rebuilt circuit is not the frozen one")
        score = HF.score(design["m"], unit, design["decoder"])
        lo, hi = P.SUPPORT[u["group"].split("/")[0]]
        e_f = e_b = float("nan")
        tq_f = tq_b = None
        if "frozen" in arms:
            tq_f = dev.transpile(qc, seed=5000 + i)
            e_f = float(R.evaluate(R.compile_circuit(tq_f, dev.snap, dev.params, [R.FULL]), score,
                                   tq_f.num_clbits)[0])
        if "barriers" in arms:
            tq_b = dev.transpile(staged_circuit(design["m"], unit, design["decoder"]), seed=5000 + i)
            ideal_b, e_b = (float(v) for v in R.evaluate(
                R.compile_circuit(tq_b, dev.snap, dev.params, [R.IDEAL, R.FULL]), score, tq_b.num_clbits))
            if abs(ideal_b - u["ideal_score"]) > 1e-9:
                raise ProjectionRefused(f"{u['name']}: the barriered circuit's ideal {ideal_b} is not the frozen "
                                        "ideal")
        qx = executed_circuit(rec)
        tq_x = transpile(qx, dev.backend, initial_layout=lay or list(range(qx.num_qubits)), optimization_level=0,
                         coupling_map=dev.coupling_map)
        ideal_x, e_x = (float(v) for v in R.evaluate(
            R.compile_circuit(tq_x, dev.snap, dev.params, [R.IDEAL, R.FULL]), score, tq_x.num_clbits))
        if abs(ideal_x - u["ideal_score"]) > 1e-9:
            raise ProjectionRefused(f"{u['name']}: the executed stream's ideal score {ideal_x} is not the frozen "
                                    f"ideal {u['ideal_score']}")
        width = hi - lo
        def n2q(tq):
            return float("nan") if tq is None else sum(1 for x in tq.data if x.operation.num_qubits == 2)
        rows.append(dict(group=u["group"], e_frozen=(u["ideal_score"] - e_f) / width,
                         e_barriers=(u["ideal_score"] - e_b) / width, e_exec=(u["ideal_score"] - e_x) / width,
                         n2q_frozen=n2q(tq_f), n2q_barriers=n2q(tq_b), n2q_exec=n2q(tq_x)))
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{n} units, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    if "exec" not in arms:
        raise ProjectionRefused("the executed arm is the projection's subject; it cannot be skipped")
    out = dict(schema="oneq.r189.executed_projection/2", device=device, units=n, arms_evaluated=list(arms),
               arms=dict(frozen="the frozen builder circuit, level 3 (the R186/R188 projection; merges across the "
                                "channel boundary)",
                         barriers="the same with barriers at the channel boundaries, level 3 (channel-faithful; the "
                                  "model idles every qubit at each barrier)",
                         exec="the executed one-program stream, translation only (channel-faithful by its branches)"),
               groups={})
    f_lo = {arm: design["f_ideal"] for arm in arms}
    for g, meta in design["groups"].items():
        sel = [r for r in rows if r["group"] == g]
        if not sel:
            continue
        e = {arm: np.array([r[f"e_{arm}"] for r in sel]) for arm in arms}
        d = e["exec"] - e["frozen"] if "frozen" in arms else None
        out["groups"][g] = dict(n=len(sel), **{f"ebar_{arm}": float(e[arm].mean()) for arm in arms},
                                paired_diff_exec_minus_frozen=None if d is None else float(d.mean()),
                                paired_diff_se=float(d.std(ddof=1) / np.sqrt(len(d))) if d is not None and len(d) > 1
                                else None,
                                **{f"n2q_{arm}_mean": float(np.mean([r[f"n2q_{arm}"] for r in sel])) for arm in arms})
        for arm in arms:
            f_lo[arm] -= meta["width"] * meta["W"] * float(e[arm].mean())
    out["F_e_projected"] = {arm: 1.5 * f - 0.5 for arm, f in f_lo.items()}
    out["F_e_shift_exec_minus_frozen"] = (out["F_e_projected"]["exec"] - out["F_e_projected"]["frozen"]
                                          if "frozen" in arms else None)
    out["per_unit"] = {f"e_{arm}": [round(r[f"e_{arm}"], 9) for r in rows] for arm in arms}
    out["per_unit"]["group"] = [r["group"] for r in rows]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design_dir")
    ap.add_argument("native_jsonl")
    ap.add_argument("device")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=list(ALL_ARMS))
    a = ap.parse_args(argv)
    design = json.loads((pathlib.Path(a.design_dir) / "design.json").read_text(encoding="utf-8"))
    natives = [json.loads(ln) for ln in pathlib.Path(a.native_jsonl).read_text(encoding="utf-8").splitlines() if ln]
    print(json.dumps(project(design, natives, a.device, a.limit, tuple(a.arms)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
