"""R177 priced: R146's design with every two-qubit block resynthesized against its contract (oneq.hw.contract_synth),
scored on device model v2 against the circuits that ran, on R152's calibration, nothing fitted (OFFLINE, 2026-10-18).

The rewrite is EXACT on every unit circuit: each one's outcome distribution over every classical bit is checked
against the original's (oneq.hw.dynamic_exact) and the run refuses above 1e-12. Its saving is then priced by the SAME
scoring that closes R146 held out (r152_posthoc.predictors with the rewritten circuits substituted, base V3c with the
idle map and vendor sx, quasi-static coherent), in each twirl mode's preset (oneq.hw.twirl_mode), with common random
numbers so the two arms differ only by their circuits. "projected" adds the gain to R146's measured F_e: the model's
miss carried over unchanged (v2 misses R146 by +0.003 +- 0.003).

  python r177_price.py [--traj N] -> evidence/superactivation/r177_kak_floor/price_r146.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "evidence" / "superactivation" / "r177_kak_floor"
EXACT = 1e-12


class PriceRefused(ValueError):
    """a rewritten circuit that is not exactly the original's distribution is never priced"""


def rewrite_all(circuits: dict, names: list) -> tuple:
    """(rewritten circuits, per-circuit report): every named circuit rewritten with its INPUT wire protected
    (contract_synth.input_wires: the unit's input lives on logical qubit 0), then checked twice -- exact for this
    unit's input, and FAITHFUL for every input on that wire (six Pauli preparations). R177's first version skipped
    the second check and specialized 17 R146 units to their inputs."""
    from oneq.hw import contract_synth as CS
    out, rows = dict(circuits), {}
    for n in names:
        (wire,) = CS.input_wires(circuits[n])
        new, rep = CS.rewrite(circuits[n], unknown=(wire,))
        diff = CS.equivalent(circuits[n], new)
        if diff > EXACT:
            raise PriceRefused(f"{n}: the rewrite moves an outcome probability by {diff:.3e}")
        unfaithful = CS.faithful(circuits[n], new, wire)
        if unfaithful > EXACT:
            raise PriceRefused(f"{n}: the rewrite is exact for this input only (another moves by {unfaithful:.3e})")
        out[n] = new
        rows[n] = dict(rep.to_json(), max_abs_dp=diff, max_abs_dp_any_input=unfaithful, input_wire=wire)
    return out, rows


def _cz_pairs(qc) -> set:
    return {tuple(sorted(qc.find_bit(q).index for q in i.qubits)) for i in qc.data if i.operation.name == "cz"}


def rewrite_standins(stand_ins: dict, real: dict, edges: set) -> tuple:
    """R147's prediction stand-ins rewritten like the circuits that ran. A stand-in is a MODEL of its circuit: it
    carries CZs the device never runs (the zero-duration virtual feed-forward edge, and the copy structure that stands
    for a mid-circuit read), and a rewrite that removed those would delete the model's charge for a read, not a gate.
    So: only pairs its real circuit also has may be rewritten (every other pair is skipped), its input wire must be
    its circuit's, the CZs it saves must equal what the protected rewrite saves on the circuit itself, and it is
    checked exact and faithful. Any miss is refused -- found 2026-10-18, when an unguarded version saved 2,049 CZs on
    the stand-ins against 453 on the circuits."""
    from oneq.hw import contract_synth as CS
    out, rows = dict(stand_ins), {}
    real_saving: dict = {}
    for k, qc in stand_ins.items():
        (wire,) = CS.input_wires(qc)
        name = k.split("__", 1)[1]
        if name not in real:
            raise PriceRefused(f"{k}: no circuit {name!r} for this stand-in")
        if CS.input_wires(real[name]) != (wire,):
            raise PriceRefused(f"{k}: its input wire {wire} is not its circuit's {CS.input_wires(real[name])}")
        if name not in real_saving:
            _, rr = CS.rewrite(real[name], unknown=(wire,))
            real_saving[name] = rr.cz_before - rr.cz_after
        model_only = (_cz_pairs(qc) - edges) | (_cz_pairs(qc) - _cz_pairs(real[name]))
        new, rep = CS.rewrite(qc, unknown=(wire,), skip_pairs=model_only)
        if rep.cz_before - rep.cz_after != real_saving[name]:
            raise PriceRefused(f"{k}: saves {rep.cz_before - rep.cz_after} CZ where its circuit saves "
                               f"{real_saving[name]}: the stand-in rewrite is not the circuit's")
        virtual = model_only
        diff = CS.equivalent(qc, new)
        if diff > EXACT:
            raise PriceRefused(f"{k}: the rewrite moves an outcome probability by {diff:.3e}")
        unfaithful = CS.faithful(qc, new, wire)
        if unfaithful > EXACT:
            raise PriceRefused(f"{k}: the rewrite is exact for this input only (another moves by {unfaithful:.3e})")
        out[k] = new
        rows[k] = dict(rep.to_json(), max_abs_dp=diff, max_abs_dp_any_input=unfaithful, input_wire=wire,
                       model_only_pairs=sorted(map(list, virtual)), circuit_saving=real_saving[name])
    return out, rows


AS_RUN = "server-active-accum"            # the mode R152 was submitted in: the one whose miss is measured


def gains(original: dict, rewritten: dict, measured: float, as_run: str = AS_RUN) -> dict:
    """per mode: both F_e, the rewrite's gain within the mode, and the projection of the rewritten design in that
    mode onto hardware: R152's measured F_e plus (model, this mode, rewritten) - (model, as run, original) -- the
    as-run miss carried over unchanged, which is an assumption for every mode but the one that ran (pure)"""
    if as_run not in original:
        raise KeyError(f"no as-run mode {as_run!r}")
    out = {}
    for mode in original:
        out[mode] = dict(original=original[mode], rewritten=rewritten[mode], gain=rewritten[mode] - original[mode],
                         projected=measured + rewritten[mode] - original[as_run],
                         projected_original=measured + original[mode] - original[as_run])
    return out


def price_r147(art: dict, traj: int) -> int:
    """R147 (the channel-faithful dynamic design) priced on its prediction stand-ins: rewritten with the input wire
    protected and the virtual feed-forward edges skipped, each checked exact and faithful, then scored by the same
    fe147 that closes R147 held out, in every twirl mode"""
    import hw_dynamic_run as R147
    import r152_posthoc as P152
    import r155_heldout_r152 as HO

    from oneq.hw import freeze as F
    from oneq.hw import twirl_mode as TM
    rec = art["parts"]["r147"]
    real = F.load(rec, (art["dir"] / "r147.qpy").read_bytes(), check_sources=False)
    stand = R147.load_prediction_circuits(rec, (art["dir"] / "r147.pred.qpy").read_bytes())
    edges = {tuple(sorted(map(int, e.split("_")))) for e in rec["snapshot"]["cz"]}
    t0 = time.time()
    new, rows = rewrite_standins(stand, real, edges)
    print(f"rewrote {len(stand)} stand-ins in {time.time() - t0:.0f} s; CZ "
          f"{sum(r['cz_before'] for r in rows.values())} -> {sum(r['cz_after'] for r in rows.values())}", flush=True)
    idle = json.loads((ROOT / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, traj)
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    preds = {"original": {}, "rewritten": {}}
    for mode in TM.MODES:
        for arm, si in (("original", None), ("rewritten", new)):
            t1 = time.time()
            preds[arm][mode] = P["fe147"](m7, stand_ins=si, quasistatic="coherent", **TM.noise_kwargs(mode))[0]
            print(f"  {arm:9s} {mode:20s} F_e {preds[arm][mode]:.4f}  ({time.time() - t1:.0f} s)", flush=True)
    measured = P["meas147"]
    out = dict(schema="oneq.r177.price/1", label="EXPLORATORY -- a recompiled design priced on the model; no run",
               protocol="r147", r152_job=art["job"], traj=traj, measured_r147=measured,
               cz=dict(before=sum(r["cz_before"] for r in rows.values()),
                       after=sum(r["cz_after"] for r in rows.values())),
               worst_max_abs_dp=max(r["max_abs_dp"] for r in rows.values()),
               worst_max_abs_dp_any_input=max(r["max_abs_dp_any_input"] for r in rows.values()),
               by_mode=gains(preds["original"], preds["rewritten"], measured), per_standin=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "price_r147.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for mode, v in out["by_mode"].items():
        print(f"{mode:20s} {v['original']:.4f} -> {v['rewritten']:.4f}  gain {v['gain']:+.4f}  projected "
              f"{v['projected']:.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=int, default=24)
    ap.add_argument("--protocol", choices=["r146", "r147"], default="r146")
    a = ap.parse_args()
    import r152_posthoc as P152
    import r155_heldout_r152 as HO

    from oneq.hw import freeze as F
    from oneq.hw import twirl_mode as TM
    art = P152.load(ROOT / HO.R152_DIR)
    if a.protocol == "r147":
        return price_r147(art, a.traj)
    r146 = art["parts"]["r146"]
    c146 = F.load(r146, (art["dir"] / "r146.qpy").read_bytes(), check_sources=False)
    names = [u["name"] for u in r146["units"]]
    t0 = time.time()
    new146, rows = rewrite_all(c146, names)
    print(f"rewrote {len(names)} circuits in {time.time() - t0:.0f} s; CZ "
          f"{sum(r['cz_before'] for r in rows.values())} -> {sum(r['cz_after'] for r in rows.values())}", flush=True)
    idle = json.loads((ROOT / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    preds = {}
    for arm, circs in (("original", None), ("rewritten", new146)):
        P = P152.predictors(art, a.traj, circuits146=circs)
        m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
        preds[arm] = {}
        for mode in TM.MODES:
            t1 = time.time()
            preds[arm][mode] = P["fe146"](m6, quasistatic="coherent", **TM.noise_kwargs(mode))
            print(f"  {arm:9s} {mode:20s} F_e {preds[arm][mode]:.4f}  ({time.time() - t1:.0f} s)", flush=True)
        measured = P["meas146"]
    out = dict(schema="oneq.r177.price/1", label="EXPLORATORY -- a recompiled design priced on the model; no run",
               r152_job=art["job"], traj=a.traj, measured_r146=measured,
               cz=dict(before=sum(r["cz_before"] for r in rows.values()),
                       after=sum(r["cz_after"] for r in rows.values())),
               worst_max_abs_dp=max(r["max_abs_dp"] for r in rows.values()),
               by_mode=gains(preds["original"], preds["rewritten"], measured), per_circuit=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "price_r146.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for mode, v in out["by_mode"].items():
        print(f"{mode:20s} {v['original']:.4f} -> {v['rewritten']:.4f}  gain {v['gain']:+.4f}  projected "
              f"{v['projected']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
