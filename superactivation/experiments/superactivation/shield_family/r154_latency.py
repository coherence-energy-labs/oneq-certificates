"""R154: does a mid-circuit read SYNCHRONIZE the whole circuit? Mechanism (B), cross-validated on R147's own units.

R147's engine stand-ins model a mid-circuit read as a hold on the read shields only (their read-out time plus an
ASSUMED 0.65 us feed-forward latency), with every other qubit free to run on. If the device instead holds EVERY active
qubit until the read and its feed-forward complete, each mid-circuit read costs every superposed qubit an idle of the
read time plus a latency L -- a cost that does not grow with the CZ count, as R153's residuals require.
The hypothesis, in the stand-in: after the shields' hold, a barrier on every qubit and a delay L on every qubit.
Static (sector-1) circuits have no mid-circuit read, so they CANNOT move: a built-in control.
CROSS-VALIDATION, fixed before any number: R147's sector-0 kept units split by unit index parity. Fit L on the EVEN
half (its mean per-unit residual to zero, by interpolation on a grid), test on the ODD half. The model underneath is
the repaired + crosstalk model, with the static term R154_forms SELECTED (if any) at its R146-fitted value.

  python r154_latency.py COMPOSITE_JSON [--traj T]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_dynamic_run as R147  # noqa: E402
import hw_faithful as FA  # noqa: E402
import hw_two_use as H  # noqa: E402
import r154_forms as RF  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import noise as N  # noqa: E402

LATENCY_GRID_US = (None, 0.0, 1.0, 3.0, 6.0)          # None: R147's stand-in as frozen (no synchronization)


def synchronized(qc, hold_dt: int, latency_dt: int):
    """the stand-in with a whole-circuit synchronization after the read shields' hold: a barrier on every qubit, then
    a delay of latency_dt on every qubit. The hold is the delay of exactly hold_dt the stand-in places on each read
    shield; the sync goes after the LAST of them."""
    from qiskit import QuantumCircuit
    holds = [k for k, ins in enumerate(qc.data) if ins.operation.name == "delay"
             and int(round(float(ins.operation.duration))) == hold_dt]
    if not holds:
        return qc
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    for k, ins in enumerate(qc.data):
        out.append(ins.operation, ins.qubits, ins.clbits)
        if k == holds[-1]:
            out.barrier()
            if latency_dt > 0:
                for q in out.qubits:
                    out.delay(latency_dt, q, unit="dt")
            out.barrier()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--traj", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    rp = pathlib.Path(a.record).resolve()
    loaded, per, rec146, circ146, snap146, params146, snapb = RF.setup(rp)
    sel_path = rp.parent / "r154_forms.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    form = sel.get("selected")
    knob = sel["forms"][form]["fitted"] if form else None
    rec147 = loaded["parts"]["r147"].record
    circ147 = F.load(rec147, loaded["parts"]["r147"].blob, check_sources=False)
    pred147 = R147.load_prediction_circuits(rec147, loaded["pred147"])
    dt = rec147["snapshot"]["dt"]
    hold_dt = int(round(rec147["tau_s"] / dt))
    base = N.NoiseParams(cz_factor=R147.REPAIRED_CZ_FACTOR, schedule="alap", idle=True)
    params = RF.params_for(base, form, knob) if form else base
    # the kept units of each sector; sector 0 split by index parity
    from oneq.hw import stats as T
    units = rec147["units"]
    stat = [T.Unit(u["name"], u["weight"], rec147["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    tallies = T.tally_all(stat, {u["name"]: per["r147"][u["name"]] for u in units},
                          {u["name"]: R147._score(u) for u in units})
    groups = {"s0_even": [], "s0_odd": [], "s1": []}
    for k, u in enumerate(units):
        if u["kind"] != "main" or u["stratum"] == "erased_erased":
            continue
        sec = u["stratum"].split("|")[1]
        if sec == "s1":
            groups["s1"].append(u)
        else:
            groups["s0_even" if k % 2 == 0 else "s0_odd"].append(u)
    rows = []
    for lat in LATENCY_GRID_US:
        if lat is None:
            pc = pred147
        else:
            ldt = int(round(lat * 1e-6 / dt))
            pc = {n: synchronized(c, hold_dt, ldt) for n, c in pred147.items()}
        names = {u["name"] for g in groups.values() for u in g}
        preds = R147.predict_units(rec147, circ147, pc, snapb, params, traj=a.traj, names=names)
        res = {g: float(np.mean([tallies[u["name"]].mean - preds[u["name"]][0] for u in us]))
               for g, us in groups.items()}
        rows.append((lat, res))
        print(f"  latency {'none (as frozen)' if lat is None else f'{lat:.1f} us + sync'}: mean per-unit residual "
              f"{ {k: round(v, 4) for k, v in res.items()} }", flush=True)
    # fit on s0_even over the synchronized rows, test on s0_odd and s1
    syn = [(lat, r) for lat, r in rows if lat is not None]
    xs = np.array([lat for lat, _ in syn])
    ye = np.array([r["s0_even"] for _, r in syn])
    fitted = None
    if ye.min() <= 0 <= ye.max():
        order = np.argsort(ye)
        fitted = float(np.interp(0.0, ye[order], xs[order]))
    out = dict(schema="oneq.r154_latency/1", rule=__doc__.split("CROSS-VALIDATION")[1].split("  python")[0].strip(),
               static_form=form, static_knob=knob, hold_dt=hold_dt, traj=a.traj,
               scan=[dict(latency_us=lat, residual=r) for lat, r in rows], fitted_latency_us_on_even=fitted)
    if fitted is not None:
        ldt = int(round(fitted * 1e-6 / dt))
        pc = {n: synchronized(c, hold_dt, ldt) for n, c in pred147.items()}
        preds = R147.predict_units(rec147, circ147, pc, snapb, params, traj=a.traj)
        test = {g: float(np.mean([tallies[u["name"]].mean - preds[u["name"]][0] for u in us]))
                for g, us in groups.items()}
        full = R147.analyse(R147.with_predictions(rec147, preds, snapb), per["r147"])
        out.update(test_residuals_at_fit=test, r147_F_e_predicted=full["predicted"]["F_e"],
                   r147_F_e_measured=full["measured"]["F_e"],
                   r147_delta=full["measured"]["F_e"] - full["predicted"]["F_e"])
        print(f"FITTED latency (even half) {fitted:.2f} us -> TEST: odd half residual {test['s0_odd']:+.4f}, "
              f"s1 {test['s1']:+.4f}; R147 F_e predicted {out['r147_F_e_predicted']:.4f} vs "
              f"{out['r147_F_e_measured']:.4f} (delta {out['r147_delta']:+.4f})", flush=True)
    else:
        print("no latency on the grid zeroes the even half's residual", flush=True)
    out["seconds"] = time.time() - t0
    (rp.parent / "r154_latency.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8",
                                                 newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
