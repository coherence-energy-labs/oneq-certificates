"""R155's measured idle physics, HELD OUT on R152's two protocols (EXPLORATORY: outside every preregistered verdict).

R152 (job db9c50oc77vs73817rog) measured R146's p0 F_e and R147's F_e. The engine's best transferring model, V1s, needed
every CZ rate multiplied by s = 2.95, FITTED on R146 -- a factor no ladder measured. R155 (job db9dsqpgv2cc73e5c8fg, two
hours later, same device) measured why a single exponential T2 is the wrong idle model: per qubit a white rate and a
quasi-static detuning spread sigma (r155_idle_map.py). This scores, with NOTHING fitted to R146 or R147:

  V2w  R152's own in-circuit repair (readout, sx, laddered CZ rates and shapes at factor 1) with t2 := R155's WHITE T2
  V2c  V2w + R155's sigma, quasi-static windows 'coherent' (a twirled CZ's frame is undone at the CZ)
  V2l  V2w + R155's sigma, 'layer' (every CZ on a qubit closes its window: the frame merged into neighbouring gates)

and, with --sx-floor, the same three with the VENDOR sx error in place of the repair's (V3w, V3c, V3l) plus the vendor's
echo T2 for idles (V3e). The repair's sx error comes from R146's sk family -- |0>-survival after sx^(4k) -- and sits
BELOW the T1/T2 floor of a 36 ns gate (median 1e-5 against 1.5e-4, 37 of 39 qubits): relaxation pulls toward the very
state that observable scores, and the drive refocuses dephasing, so sk cannot see the gate's decoherence. The vendor's
randomized-benchmarking figure (2.4-3.0e-4) is above the floor.

Qubits the idle map lacks keep the repair's in-circuit T2 and no sigma; the record lists them.

  python r155_heldout_r152.py REPO_ROOT [--traj N]   -> evidence/superactivation/r155_diag/heldout_r152_<job>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import r152_posthoc as P152  # noqa: E402

from oneq.hw import snapshot as S  # noqa: E402

R152_DIR = pathlib.Path("evidence/superactivation/campaign_r152/20261017T003152Z")
R155_DIR = pathlib.Path("evidence/superactivation/r155_diag")
IDLE_MAP = "idle_map_db9dsqpgv2cc73e5c8fg.json"


def with_idle_map(model: S.DeviceSnapshot, idle: dict, sigma: bool) -> tuple:
    """the model with R155's white T2 (clamped to 2 T1) on every mapped qubit, and its sigma when asked; plus the
    qubits it could not map"""
    d = model.to_json()
    sig = {}
    for q, row in idle.items():
        qi = int(q)
        t2w = row["white_T2_s"]
        if t2w and t2w != float("inf") and qi < model.num_qubits and d["t1"][qi]:
            d["t2"][qi] = min(float(t2w), 2 * d["t1"][qi])
        if sigma and row["sigma_rad_per_s"] > 0:
            sig[qi] = float(row["sigma_rad_per_s"])
    if sig:
        d["sigma"] = {str(q): v for q, v in sorted(sig.items())}
    return S.DeviceSnapshot.from_json(d)


def with_vendor_sx(model: S.DeviceSnapshot, vendor: S.DeviceSnapshot) -> S.DeviceSnapshot:
    """the model with the vendor's sx error on every qubit (see the module docstring for why the repair's is wrong)"""
    return S.DeviceSnapshot(**{**model.__dict__, "sx_error": tuple(vendor.sx_error)})


def with_vendor_t2(model: S.DeviceSnapshot, vendor: S.DeviceSnapshot) -> S.DeviceSnapshot:
    return S.DeviceSnapshot(**{**model.__dict__, "t2": tuple(vendor.t2)})


def floor_violations(model: S.DeviceSnapshot, qubits) -> list:
    """qubits whose sx error is below the decoherence floor of the gate itself, t_g / (3 T1) + t_g / (3 T2)"""
    return [q for q in qubits if model.sx_error[q] < model.sx_duration[q] * (1 / model.t1[q] + 1 / model.t2[q]) / 3]


def with_hold(stand_ins: dict, rec147: dict, hold_s: float) -> tuple:
    """R147's frozen stand-ins with every HOLD -- a delay of exactly the record's tau (the read plus the assumed
    feed-forward stall), on the key and the read shields -- set to hold_s; and, per stand-in, how many it changed. No
    other instruction is touched, and a delay of any other length is refused (it would not be a hold)."""
    from qiskit.circuit import Delay
    dt = rec147["snapshot"]["dt"]
    old, new = int(round(rec147["tau_s"] / dt)), int(round(hold_s / dt))
    out, changed = {}, {}
    for name, qc in stand_ins.items():
        c, n = qc.copy_empty_like(), 0
        for inst in qc.data:
            op = inst.operation
            if op.name == "delay":
                if op.unit != "dt" or int(op.duration) != old:
                    raise ValueError(f"{name}: a delay of {op.duration} {op.unit} is not R147's hold of {old} dt")
                op, n = Delay(new, unit="dt"), n + 1
            c.append(op, inst.qubits, inst.clbits)
        out[name], changed[name] = c, n
    return out, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    ap.add_argument("--sx-floor", action="store_true", help="also the V3 variants: the vendor sx error")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    art = P152.load(root / R152_DIR)
    idle = json.loads((root / R155_DIR / IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    out: dict = dict(schema="oneq.r155.heldout_r152/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], idle_map=IDLE_MAP, traj=a.traj,
                     measured=dict(r146=P["meas146"], r147=P["meas147"]),
                     unmapped=dict(r146=[q for q in P["qubits146"] if str(q) not in idle],
                                   r147=[q for q in P["qubits147"] if str(q) not in idle]),
                     variants={})
    prior = json.loads((root / R152_DIR / f"posthoc_{art['job']}.json").read_text(encoding="utf-8"))["models"]
    out["reference"] = {k: dict(r146=v["r146"], r147=v["r147"]) for k, v in prior["variants"].items()}
    out["sx_below_gate_floor"] = dict(r146=floor_violations(P["m146"], P["qubits146"]),
                                      r147=floor_violations(P["m147"], P["qubits147"]),
                                      vendor_r146=floor_violations(P["vendor"], P["qubits146"]),
                                      vendor_r147=floor_violations(P["vendor"], P["qubits147"]))
    arms = [("V2w", False, "coherent", False), ("V2c", True, "coherent", False), ("V2l", True, "layer", False)]
    if a.sx_floor:
        arms += [("V3w", False, "coherent", True), ("V3c", True, "coherent", True), ("V3l", True, "layer", True),
                 ("V3e", None, "coherent", True)]
    for name, sigma, mode, vsx in arms:
        if sigma is None:                              # V3e: the vendor's echo T2, no idle map
            m6, m7 = with_vendor_t2(P["m146"], P["vendor"]), with_vendor_t2(P["m147"], P["vendor"])
        else:
            m6, m7 = with_idle_map(P["m146"], idle, sigma), with_idle_map(P["m147"], idle, sigma)
        if vsx:
            m6, m7 = with_vendor_sx(m6, P["vendor"]), with_vendor_sx(m7, P["vendor"])
        f6 = P["fe146"](m6, quasistatic=mode)
        f7, k7 = P["fe147"](m7, quasistatic=mode)
        out["variants"][name] = dict(quasistatic=mode if sigma else None, vendor_sx=vsx,
                                     idle="vendor echo T2" if sigma is None else "R155 white T2",
                                     r146=f6, r146_miss=P["meas146"] - f6,
                                     r147=f7, r147_miss=P["meas147"] - f7, residual=k7)
        print(f"{name}: R146 {f6:.4f} (miss {P['meas146'] - f6:+.4f})  R147 {f7:.4f} (miss {P['meas147'] - f7:+.4f})"
              f"  residual {json.dumps({k: round(v, 4) for k, v in k7.items()})}", flush=True)
    dest = root / R155_DIR / f"heldout_r152_{art['job']}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
