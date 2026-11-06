"""R172 (b): dynamical decoupling in the long waits of R147's dynamic circuits, priced on the measured model
(EXPLORATORY; item 20 of the owner's 25-item list).

R166/R173 MEASURED the feed-forward: the conditional lands 2.70 us after its read starts. Qubits that wait superposed
through such a stretch dephase. R155 measured the dephasing as a mix: white, plus a quasi-static spread sigma/2pi of
median 11 kHz, about 0.19 rad over 2.7 us. An echo refocuses the quasi-static part; the white part it cannot. The
runtime's own DD refuses dynamic circuits (runtime 0.50), so DD is client-side: x pulses written into the circuit,
which with server twirling OFF (R173; client twirling, oneq.hw.client_twirl) play as one pulse each.

with_dd schedules a stand-in on the engine's own schedule (oneq.hw.frame.schedule, ALAP). Every padded wait of at
least MIN_WAIT on a qubit that has left |0> gets an XY-free X-X echo: T/4, x, T/2, x, T/4. The engine prices the two
pulses (their error, and T1 while an X frame inverts a Z eigenstate) against the echo it credits to sigma (its phase
windows invert at an x).

Model: V3c with the compiled pulses (the client-twirl arm of R172: no server template, no dressing). R147 on its frozen
stand-ins with and without DD. NOTHING is fitted.

  python r172_dd_heldout.py REPO_ROOT [--traj N] -> evidence/superactivation/r172_design/dd_heldout_r152.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

MIN_WAIT_S = (0.5e-6, 1.0e-6)


class DDRefused(ValueError):
    """the stand-in cannot be decoupled faithfully"""


def with_dd(qc, snap, min_wait_s: float):
    """qc on the engine's ALAP schedule with an X-X echo in every padded wait >= min_wait_s on a qubit that has left
    |0> (its first sx or x ran). Returns (circuit, number of waits decoupled)"""
    from qiskit.circuit import Delay
    from qiskit.circuit.library import XGate

    from oneq.hw import frame as FR
    if not min_wait_s > 0:
        raise DDRefused(f"the minimum wait must be positive: {min_wait_s}")
    sched = FR.schedule(qc, snap, "alap")
    dt = snap.dt
    started: set = set()
    out = sched.copy_empty_like()
    n = 0
    for inst in sched.data:
        name = inst.operation.name
        qs = [sched.find_bit(b).index for b in inst.qubits]
        if name == "delay" and qs[0] in started:
            op = inst.operation
            if op.unit != "dt":
                raise DDRefused(f"a padded wait in {op.unit}: the engine's schedule is in dt")
            T = int(op.duration)
            if T * dt >= min_wait_s:
                q = inst.qubits[0]
                quarter = T // 4
                half = T - 2 * quarter - 2 * int(round(snap.sx_duration[qs[0]] / dt))
                if half < 0:
                    out.append(op, inst.qubits, inst.clbits)
                    continue
                out.append(Delay(quarter, unit="dt"), [q])
                out.append(XGate(), [q])
                out.append(Delay(half, unit="dt"), [q])
                out.append(XGate(), [q])
                out.append(Delay(quarter, unit="dt"), [q])
                n += 1
                continue
        if name in ("sx", "x"):
            started.update(qs)
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    import hw_dynamic as D
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    rec147 = P["record147"]
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    kw = dict(quasistatic="coherent")
    out: dict = dict(schema="oneq.r172.dd_heldout/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, measured_r147=P["meas147"],
                     model="V3c, compiled pulses (client-twirl arm)", variants={})
    f7, k7 = P["fe147"](m7, **kw)
    out["variants"]["no DD"] = dict(r147=f7, residual=k7, waits_decoupled=0)
    for mw in MIN_WAIT_S:
        stand, total = {}, 0
        for u in rec147["units"]:
            inp = u["prediction_inputs"]
            if inp["kind"] != "dynamic":
                continue
            sn = D.augmented(m7, [tuple(e) for e in inp["virt"]])
            for key in ("c0", "c1"):
                name = inp[key]
                if name and name not in stand:
                    stand[name], n = with_dd(P["stand_ins"][name], sn, mw)
                    total += n
        merged = dict(P["stand_ins"], **stand)
        f7, k7 = P["fe147"](m7, stand_ins=merged, **kw)
        out["variants"][f"X-X echo in waits >= {mw * 1e6:.1f} us"] = dict(r147=f7, residual=k7, waits_decoupled=total)
    base = out["variants"]["no DD"]["r147"]
    for name, v in out["variants"].items():
        v["gain_vs_no_dd"] = v["r147"] - base
        print(f"{name:34s} R147 {v['r147']:.4f} (gain {v['gain_vs_no_dd']:+.4f}; {v['waits_decoupled']} waits) "
              f"residual {json.dumps({k: round(x, 4) for k, x in v['residual'].items()})}", flush=True)
    d = root / "evidence" / "superactivation" / "r172_design"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dd_heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                            newline="\n")
    print("wrote dd_heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
