"""R153: the repaired model plus the two terms R152's calibration could not see, tested HELD OUT across experiments.

Terms (oneq.hw.noise):
  crosstalk  the always-on conditional phase per coupled edge (DeviceSnapshot.zz), fitted from each block's own xs
             family by zz_fit.py; the qubit's T2 is then its INTRINSIC value (coupling divided out), so the xs decay is
             not charged twice;
  spectator  ONE global probability of a Z error on every superposed neighbour of a running CZ (NoiseParams.spectator).
             No family of this job measures it (the ladders park neighbours in |0>), so it is FITTED -- on one
             experiment only.
The rule, fixed here before any R147 number is computed:
  1. FIT on R146's held-out p0 (R145's 208 static circuits on R146's qubits): scan the spectator probability, take the
     value that matches p0's measured F_e (linear interpolation on the scan).
  2. PREDICT R147 (other qubits, other design, mid-circuit reads) with that value frozen, on R147's own block fit.
     R147's counts enter only the comparison.
A model that closes R147 held out has earned a hardware attempt; one that does not, has not.

  python r153_heldout.py RECORD [--traj T]       (RECORD: the campaign's composite.json)
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_campaign_run as CR  # noqa: E402
import hw_diag_run as D146  # noqa: E402
import hw_dynamic_run as R147  # noqa: E402
import hw_superact_run as HS  # noqa: E402
import hw_two_use as H  # noqa: E402
import zz_fit as ZZ  # noqa: E402
from oneq.hw import campaign as C  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402
from oneq.hw import stats as T  # noqa: E402

SPECTATOR_GRID = (0.0, 0.002, 0.005, 0.01, 0.02)


def with_terms(snap: S.DeviceSnapshot, zzfit: dict | None) -> S.DeviceSnapshot:
    """the repaired snapshot plus the fitted crosstalk, with T2 set to the intrinsic value on every block qubit"""
    if zzfit is None:
        return snap
    zz = {}
    for k, e in zzfit["edges"].items():
        a, b = (int(x) for x in k.split("_"))
        zz[(min(a, b), max(a, b))] = float(e["zeta_rad_s"])
    t2 = list(snap.t2)
    for q, v in zzfit["t2"].items():
        if math.isfinite(v["t2_intrinsic_s"]):
            t2[int(q)] = min(float(v["t2_intrinsic_s"]), 2 * snap.t1[int(q)])
    return S.DeviceSnapshot(**{**snap.__dict__, "zz": zz, "t2": tuple(t2)})


def r146_measured(rec, counts):
    units = rec["units"]
    stat = [T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    scores = {u["name"]: H.score(rec["m"], HS.unit_from(u["unit"])) for u in units}
    t = T.tally_all(stat, {u["name"]: counts[u["name"]] for u in units}, scores)
    means = {k: v.mean for k, v in t.items()}
    var = {k: max(v.s2 / v.n - v.mean ** 2, 0.0) for k, v in t.items()}
    est = HS.estimate(means, units, var, {k: v.n for k, v in t.items()})
    return est, means


def r146_predict(rec, circ, snap, params, traj):
    units = rec["units"]
    rng = np.random.default_rng(11)
    pred = {u["name"]: E.expectation(circ[u["name"]], H.score(rec["m"], HS.unit_from(u["unit"])), snap, params, traj,
                                     rng).mean for u in units}
    return HS.estimate(pred, units)["F_e"], pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--traj", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    rp = pathlib.Path(a.record).resolve()
    loaded = CR.load(rp)
    counts = json.loads(next(rp.parent.glob("counts_*.json")).read_text(encoding="utf-8"))
    per = C.split(loaded["rec"], counts)
    p146, pblk, p147 = (loaded["parts"][t] for t in ("r146", "blk147", "r147"))
    # ---- the two calibration blocks, fitted on their calibration families ONLY
    rec146 = p146.record
    circ146 = F.load(rec146, p146.blob, check_sources=False)
    fit_counts = {k: v for k, v in per["r146"].items() if not k.startswith(D146.HELD_OUT)}
    fit146 = D146.fit_in_circuit(rec146, fit_counts, circ146)
    snap146 = D146.repaired_snapshot(rec146, fit146)
    params146 = D146.repaired_params(fit146)
    zz146 = ZZ.fit_zz(rec146, fit_counts, fit146["readout"])
    fitb = D146.fit_device(pblk.record, per["blk147"])
    snapb = D146.repaired_snapshot(pblk.record, fitb)
    zzb = ZZ.fit_zz(pblk.record, per["blk147"], fitb["readout"])
    meas146, _ = r146_measured(rec146, per["r146"])
    print(f"R146 p0 measured F_e {meas146['F_e']:.4f} +- {meas146['sd_F_e']:.4f}", flush=True)
    # ---- 1. FIT the spectator probability on R146's p0, with and without the crosstalk term
    scans = {}
    for label, zzfit in (("repaired", None), ("repaired+crosstalk", zz146)):
        s = with_terms(snap146, zzfit)
        rows = []
        for p in SPECTATOR_GRID:
            prm = N.NoiseParams(**{**params146.describe(), "spectator": p})
            fe, _ = r146_predict(rec146, circ146, s, prm, a.traj)
            rows.append((p, fe))
            print(f"  R146 p0 [{label}] spectator {p:.3f}: predicted F_e {fe:.4f}  (delta {meas146['F_e'] - fe:+.4f})",
                  flush=True)
        scans[label] = rows
    fitted = {}
    for label, rows in scans.items():
        ps, fes = zip(*rows)
        target = meas146["F_e"]
        if not (min(fes) <= target <= max(fes)):
            fitted[label] = None                       # the scan does not bracket the measurement: no value fits
            continue
        fitted[label] = float(np.interp(target, fes[::-1], ps[::-1]))
    print("FITTED on R146 p0:", fitted, flush=True)
    # ---- 2. PREDICT R147 held out, each variant with its frozen spectator value
    rec147 = p147.record
    circ147 = F.load(rec147, p147.blob, check_sources=False)
    pred147 = R147.load_prediction_circuits(rec147, loaded["pred147"])
    out147 = {}
    variants = [("repaired (R152's rule)", None, 0.0)]
    for label, zzfit in (("repaired", None), ("repaired+crosstalk", zzb)):
        if fitted[label] is not None:
            variants.append((f"{label} + spectator {fitted[label]:.4f}", zzfit, fitted[label]))
    for name, zzfit, p in variants:
        model = with_terms(snapb, zzfit)
        params = N.NoiseParams(cz_factor=R147.REPAIRED_CZ_FACTOR, schedule="alap", idle=True, spectator=p)
        preds = R147.predict_units(rec147, circ147, pred147, model, params, traj=a.traj)
        res = R147.analyse(R147.with_predictions(rec147, preds, model), per["r147"])
        by_sector = {}
        for u in rec147["units"]:
            if u["kind"] != "main" or u["stratum"] == "erased_erased":
                continue
            sec = u["stratum"].split("|")[1]
            m = res["per_unit"][u["name"]]
            by_sector.setdefault(sec, []).append(m["measured"] - m["predicted"])
        out147[name] = dict(F_e_measured=res["measured"]["F_e"], sd=res["measured"]["sd_F_e"],
                            F_e_predicted=res["predicted"]["F_e"],
                            delta=res["measured"]["F_e"] - res["predicted"]["F_e"],
                            V2_within_0p03=abs(res["measured"]["F_e"] - res["predicted"]["F_e"]) <= 0.03,
                            residual_by_sector={k: float(np.mean(v)) for k, v in by_sector.items()},
                            dynamic_minus_static=res["dynamic_minus_static"])
        print(f"R147 HELD OUT [{name}]: predicted {res['predicted']['F_e']:.4f} vs measured "
              f"{res['measured']['F_e']:.4f} +- {res['measured']['sd_F_e']:.4f} (delta "
              f"{out147[name]['delta']:+.4f}); per-unit residual by sector {out147[name]['residual_by_sector']}",
              flush=True)
    rec = dict(schema="oneq.r153_heldout/1", rule=__doc__.split("The rule")[1].split("  python")[0].strip(),
               traj=a.traj, spectator_grid=SPECTATOR_GRID, r146_measured=meas146["F_e"],
               r146_measured_sd=meas146["sd_F_e"], r146_scans={k: [list(r) for r in v] for k, v in scans.items()},
               spectator_fitted_on_r146=fitted, r147_held_out=out147,
               crosstalk_r146=zz146["summary"], crosstalk_blk147=zzb["summary"], seconds=time.time() - t0)
    dest = rp.parent / "r153_heldout.json"
    dest.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    dest = dest.resolve()
    print("->", dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
