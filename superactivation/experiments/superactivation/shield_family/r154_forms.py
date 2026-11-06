"""R154: WHICH missing error? One-parameter forms of the static-circuit deficit, chosen on R146 and tested on R147.

R153 showed the deficit is real but not per-CZ. Candidate forms, each ONE knob on the repaired model (crosstalk on):
  spectator   Z on every superposed neighbour of a running CZ                    (NoiseParams.spectator)
  idle        every idle error multiplied                                        (NoiseParams.idle_factor)
  cz          every CZ error multiplied                                          (NoiseParams.cz_factor)
  sx          every single-qubit gate error multiplied                           (NoiseParams.sx_factor)
  readout     every readout error multiplied                                     (NoiseParams.readout_factor)
THE RULE (fixed before any R147 number): for each form, fit its knob to R146 p0's measured F_e; score the form by
the chi^2 of R146's 13 strata at that knob (the INTERNAL structure the total does not constrain); the form with the
lowest chi^2 is SELECTED. Every form is then predicted on R147, but only the selected one is the test.

  python r154_forms.py COMPOSITE_JSON FORM [--traj T]      (one form per process: they run in parallel)
  python r154_forms.py COMPOSITE_JSON --collect             (selects, writes r154_forms.json)
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
import r153_heldout as R153  # noqa: E402
import zz_fit as ZZ  # noqa: E402
from oneq.hw import campaign as C  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import stats as T  # noqa: E402

GRIDS = {"spectator": (0.0, 0.005, 0.01, 0.02), "idle": (1.0, 2.0, 4.0, 8.0), "cz": (1.0, 1.5, 2.0, 3.0),
         "sx": (1.0, 3.0, 6.0, 12.0), "readout": (1.0, 2.0, 3.0, 5.0)}


def params_for(base: N.NoiseParams, form: str, v: float) -> N.NoiseParams:
    d = base.describe()
    if form == "spectator":
        d["spectator"] = v
    elif form == "idle":
        d["idle_factor"] = v
    elif form == "cz":
        d["cz_factor"] = base.cz_factor * v
    elif form == "sx":
        d["sx_factor"] = base.sx_factor * v
    elif form == "readout":
        d["readout_factor"] = v
    return N.NoiseParams(**d)


def strata_chi2(rec, counts, pred):
    """chi^2 of R146's strata: per stratum, the weighted measured mean against the predicted, with the measured
    stratum mean's sd from the spread of its units (design + shot); the enumerated both-erased stratum by shot sd"""
    units = rec["units"]
    stat = [T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in units]
    scores = {u["name"]: H.score(rec["m"], HS.unit_from(u["unit"])) for u in units}
    t = T.tally_all(stat, {u["name"]: counts[u["name"]] for u in units}, scores)
    by: dict = {}
    for u in units:
        by.setdefault(u["stratum"], []).append(u)
    chi2, rows = 0.0, {}
    for s, us in by.items():
        W = sum(u["weight"] for u in us)
        m = sum(u["weight"] * t[u["name"]].mean for u in us) / W
        p = sum(u["weight"] * pred[u["name"]] for u in us) / W
        if len(us) > 1 and s != "erased_erased":
            d = [t[u["name"]].mean - pred[u["name"]] for u in us]
            sd = float(np.std(d, ddof=1) / math.sqrt(len(us)))
        else:
            sd = math.sqrt(sum((u["weight"] / W) ** 2 * max(t[u["name"]].s2 / t[u["name"]].n - t[u["name"]].mean ** 2,
                                                             1e-6) / t[u["name"]].n for u in us))
        sd = max(sd, 1e-4)
        chi2 += ((m - p) / sd) ** 2
        rows[s] = dict(measured=m, predicted=p, sd=sd)
    return chi2, rows


def setup(rp):
    loaded = CR.load(rp)
    counts = json.loads(next(rp.parent.glob("counts_*.json")).read_text(encoding="utf-8"))
    per = C.split(loaded["rec"], counts)
    p146, pblk = loaded["parts"]["r146"], loaded["parts"]["blk147"]
    rec146 = p146.record
    circ146 = F.load(rec146, p146.blob, check_sources=False)
    fit_counts = {k: v for k, v in per["r146"].items() if not k.startswith(D146.HELD_OUT)}
    fit146 = D146.fit_in_circuit(rec146, fit_counts, circ146)
    snap146 = R153.with_terms(D146.repaired_snapshot(rec146, fit146), ZZ.fit_zz(rec146, fit_counts, fit146["readout"]))
    fitb = D146.fit_device(pblk.record, per["blk147"])
    snapb = R153.with_terms(D146.repaired_snapshot(pblk.record, fitb), ZZ.fit_zz(pblk.record, per["blk147"],
                                                                                 fitb["readout"]))
    return loaded, per, rec146, circ146, snap146, D146.repaired_params(fit146), snapb


def run_form(rp, form, traj):
    t0 = time.time()
    loaded, per, rec146, circ146, snap146, params146, snapb = setup(rp)
    meas, _ = R153.r146_measured(rec146, per["r146"])
    scan = []
    for v in GRIDS[form]:
        fe, _ = R153.r146_predict(rec146, circ146, snap146, params_for(params146, form, v), traj)
        scan.append((v, fe))
        print(f"  [{form}] {v:g}: R146 p0 predicted {fe:.4f} (measured {meas['F_e']:.4f})", flush=True)
    vs, fes = zip(*scan)
    if not (min(fes) <= meas["F_e"] <= max(fes)):
        out = dict(form=form, scan=scan, fitted=None, note="the grid does not bracket the measurement")
    else:
        order = np.argsort(fes)
        v_star = float(np.interp(meas["F_e"], np.array(fes)[order], np.array(vs)[order]))
        fe_star, pred = R153.r146_predict(rec146, circ146, snap146, params_for(params146, form, v_star), traj)
        chi2, rows = strata_chi2(rec146, per["r146"], pred)
        print(f"[{form}] fitted {v_star:.4g}: R146 F_e {fe_star:.4f}; R146 strata chi2 {chi2:.1f} (13 strata)",
              flush=True)
        # R147, held out, at the fitted knob
        rec147 = loaded["parts"]["r147"].record
        circ147 = F.load(rec147, loaded["parts"]["r147"].blob, check_sources=False)
        pred147 = R147.load_prediction_circuits(rec147, loaded["pred147"])
        base147 = N.NoiseParams(cz_factor=R147.REPAIRED_CZ_FACTOR, schedule="alap", idle=True)
        preds = R147.predict_units(rec147, circ147, pred147, snapb, params_for(base147, form, v_star), traj=traj)
        res = R147.analyse(R147.with_predictions(rec147, preds, snapb), per["r147"])
        sec = {}
        for u in rec147["units"]:
            if u["kind"] == "main" and u["stratum"] != "erased_erased":
                m = res["per_unit"][u["name"]]
                sec.setdefault(u["stratum"].split("|")[1], []).append(m["measured"] - m["predicted"])
        out = dict(form=form, scan=scan, fitted=v_star, r146_F_e_at_fit=fe_star, r146_strata_chi2=chi2,
                   r146_strata=rows, r147_predicted=res["predicted"]["F_e"], r147_measured=res["measured"]["F_e"],
                   r147_sd=res["measured"]["sd_F_e"], r147_delta=res["measured"]["F_e"] - res["predicted"]["F_e"],
                   r147_residual_by_sector={k: float(np.mean(v)) for k, v in sec.items()})
        print(f"[{form}] R147 held out: predicted {out['r147_predicted']:.4f} vs {out['r147_measured']:.4f} "
              f"(delta {out['r147_delta']:+.4f}); per-unit residual by sector {out['r147_residual_by_sector']}",
              flush=True)
    out["seconds"] = time.time() - t0
    (rp.parent / f"r154_form_{form}.json").write_text(json.dumps(out, indent=1, default=str) + "\n",
                                                       encoding="utf-8", newline="\n")
    return out


def collect(rp):
    forms = {}
    for f in GRIDS:
        p = rp.parent / f"r154_form_{f}.json"
        if p.exists():
            forms[f] = json.loads(p.read_text(encoding="utf-8"))
    fit = {f: v for f, v in forms.items() if v.get("fitted") is not None}
    selected = min(fit, key=lambda f: fit[f]["r146_strata_chi2"]) if fit else None
    rec = dict(schema="oneq.r154_forms/1", rule=__doc__.split("THE RULE")[1].split("  python")[0].strip(),
               forms=forms, selected=selected,
               test=None if selected is None else dict(form=selected, r147_delta=fit[selected]["r147_delta"],
                                                       r147_residual_by_sector=fit[selected]["r147_residual_by_sector"],
                                                       passes_V2_0p03=abs(fit[selected]["r147_delta"]) <= 0.03))
    (rp.parent / "r154_forms.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8",
                                               newline="\n")
    for f, v in forms.items():
        tag = " <- SELECTED" if f == selected else ""
        if v.get("fitted") is None:
            print(f"{f:10s} no fit ({v.get('note')})")
        else:
            print(f"{f:10s} knob {v['fitted']:.4g}  R146 strata chi2 {v['r146_strata_chi2']:7.1f}  R147 delta "
                  f"{v['r147_delta']:+.4f}  sectors {v['r147_residual_by_sector']}{tag}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("form", nargs="?")
    ap.add_argument("--traj", type=int, default=40)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    rp = pathlib.Path(a.record).resolve()
    if a.collect:
        collect(rp)
    else:
        run_form(rp, a.form, a.traj)
