"""R167 (b): the feed-forward stall on EVERY qubit of a dynamic circuit, held out on R152 (EXPLORATORY: outside every
preregistered verdict).

R167 cleared R147's stand-in: with idles on, a real mid-circuit read and exact relaxation change the prediction by
+0.0000 +- 0.0022 per unit, against the -0.033 the dynamic units miss. R163 tried the measured feed-forward stall
only inside the HOLD (the key and the read shields): <= 0.005. But R155 B2 measured that the if_else -- not the read --
stalls DISTANT qubits: contrast ratios 0.991, 0.960, 0.897, from each far qubit's idle map L = 0.39, 0.85, 2.19 us.
Every other qubit of a dynamic circuit waits through that stall too, mostly superposed.

Each arm:
- keeps the hold at read + L on the key and the read shields (r155_heldout_r152.with_hold);
- at the conditional, the first virtual CZ (where the device runs the if_else), syncs every qubit the circuit uses
  with a barrier;
- delays every OTHER used qubit by L.
Base: V3c with the twirl dressing at R165's measured 1.54. NOTHING is fitted to R147's outcomes.

  python r167_global_stall_heldout.py REPO_ROOT [--traj N]
      -> evidence/superactivation/r167_standin/global_stall_heldout_r152.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

STALLS_US = (0.0, 0.39, 0.85, 2.19)       # 0: the sync alone; then R155 B2's three measured sites
MODEL_KW = dict(quasistatic="coherent", twirl="active-accum", dressing_factor=1.54)


class StallRefused(ValueError):
    """the stand-in cannot be given a global stall faithfully"""


def with_global_stall(qc, virt: list, held: set, stall_dt: int):
    """the stand-in with, immediately before its FIRST virtual CZ (the device's if_else), a barrier over every qubit
    the circuit uses, a delay of stall_dt on every used qubit not in `held` (the key and the read shields, whose hold
    already carries the stall), and a second barrier. virt: the stand-in's virtual edges [[key, shield], ...] in the
    circuit's qubit indices. Returns (circuit, the qubits delayed)"""
    from qiskit.circuit import Barrier, Delay
    if type(stall_dt) is not int or stall_dt < 0:
        raise StallRefused(f"a stall is a whole number of dt >= 0: {stall_dt!r}")
    pairs = {tuple(sorted(map(int, e))) for e in virt}
    if not pairs:
        raise StallRefused("no virtual edge: the circuit has no feed-forward to stall at")
    used = sorted({qc.find_bit(b).index for inst in qc.data for b in inst.qubits
                   if inst.operation.name not in ("barrier", "delay")})
    first = next((k for k, inst in enumerate(qc.data) if inst.operation.name == "cz"
                  and tuple(sorted(qc.find_bit(b).index for b in inst.qubits)) in pairs), None)
    if first is None:
        raise StallRefused(f"no CZ on the virtual edges {sorted(pairs)}")
    delayed = [q for q in used if q not in held]
    out = qc.copy_empty_like()
    for k, inst in enumerate(qc.data):
        if k == first:
            out.append(Barrier(len(used)), [qc.qubits[q] for q in used])
            if stall_dt:
                for q in delayed:
                    out.append(Delay(stall_dt, unit="dt"), [qc.qubits[q]])
            out.append(Barrier(len(used)), [qc.qubits[q] for q in used])
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out, delayed


def stalled_stand_ins(stand_ins: dict, rec147: dict, stall_dt: int, with_hold=None) -> tuple:
    """every dynamic unit's stand-ins (c0, c1) with the global stall; the hold first set to read + stall by
    with_hold(stand_ins, rec147, hold_s) when given. Returns (stand_ins, {name: qubits delayed})"""
    dt = rec147["snapshot"]["dt"]
    base = stand_ins
    if with_hold is not None:
        read_s = rec147["tau_s"] - rec147["tau_ff_assumed_s"]
        base, _ = with_hold(stand_ins, rec147, read_s + stall_dt * dt)
    out, delayed = dict(base), {}
    for u in rec147["units"]:
        inp = u["prediction_inputs"]
        if inp["kind"] != "dynamic" or not inp["virt"]:
            continue
        for key in ("c0", "c1"):
            name = inp[key]
            if not name or name in delayed:
                continue
            qc = base[name]
            phys = {int(p) for e in inp["virt"] for p in e}            # the key and the read shields (physical)
            held = {q for q in range(qc.num_qubits) if q in phys}
            out[name], delayed[name] = with_global_stall(qc, inp["virt"], held, stall_dt)
    return out, delayed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    rec147 = P["record147"]
    dt = rec147["snapshot"]["dt"]
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    out: dict = dict(schema="oneq.r167.global_stall_heldout/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, measured_r147=P["meas147"], model=dict(MODEL_KW, base="V3c"),
                     variants={})
    held_only = P["fe147"](m7, **MODEL_KW)
    out["variants"]["R163 hold only, L = 0.65 us (assumed)"] = dict(r147=held_only[0],
                                                                    r147_miss=P["meas147"] - held_only[0],
                                                                    residual=held_only[1])
    for L in STALLS_US:
        stall_dt = int(round(L * 1e-6 / dt / 16)) * 16
        stand, delayed = stalled_stand_ins(P["stand_ins"], rec147, stall_dt, HO.with_hold if L else None)
        f7, k7 = P["fe147"](m7, stand_ins=stand, **MODEL_KW)
        out["variants"][f"global stall L = {L} us"] = dict(
            stall_dt=stall_dt, circuits=len(delayed), qubits_delayed_mean=(
                sum(len(v) for v in delayed.values()) / max(len(delayed), 1)),
            r147=f7, r147_miss=P["meas147"] - f7, residual=k7)
    for name, v in out["variants"].items():
        print(f"{name:42s} R147 {v['r147']:.4f} (miss {v['r147_miss']:+.4f})  residual "
              f"{json.dumps({k: round(x, 4) for k, x in v['residual'].items()})}", flush=True)
    d = root / "evidence" / "superactivation" / "r167_standin"
    d.mkdir(parents=True, exist_ok=True)
    (d / "global_stall_heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n",
                                                      encoding="utf-8", newline="\n")
    print("wrote global_stall_heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
