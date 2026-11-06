"""R152 post hoc: everything the local session read from job db9c50oc77vs73817rog's committed shots, reproducible.

Every analysis reads ONLY the committed evidence of the record directory (the parts' records and QPYs, counts, the
flight record, the span slices, the result) -- no network, no quota. EXPLORATORY: none of it is a preregistered
verdict (the ledger's "R152 -- corrections and the post-hoc analysis" states each with its evidence class).

  integrity      counts == the per-shot record; every circuit its registered shots; every readout circuit reads its
                 own prepared pattern; the spans cover every shot
  twirl_blocks   every PUB's data is (randomizations x shots/randomization), read off the span SLICES (not assumed);
                 the chi2 of each circuit's per-randomization means against binomial, per family
  double_read    R147's MCMCAL: per shield and preparation, the four (first read, second read) outcomes -- a wrong
                 START is wrong on both reads, a MISREAD on one, a change BETWEEN the reads on the second only
  vendor_rank    per metric (readout, dephasing over the xs wait, sx, CZ, T1): the in-circuit value against the
                 vendor's, Spearman rank correlation and median ratio, per qubit set
  loss_location  R147's F_e deficit by stratum (weighted share), input basis, sector, branch
  models         (--models; minutes) V0 as preregistered, V1 = vendor echo T2 for idles, V1s = V1 with every CZ rate
                 x s, s FITTED on R146 only, all TESTED on R147; and R147's error budget on V1s

  python r152_posthoc.py RECORD_DIR [--models] [--traj N]      -> RECORD_DIR/posthoc_<job>.json
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from oneq.hw import campaign as C  # noqa: E402

M = 5                                   # R147's shields (the MCMCAL circuits read each one twice)


def load(d: pathlib.Path) -> dict:
    comp = json.loads((d / "composite.json").read_text(encoding="utf-8"))
    job = next(p.name.split("_", 1)[1].rsplit(".", 1)[0] for p in d.glob("counts_*.json"))
    art = dict(dir=d, job=job, composite=comp,
               counts=json.loads((d / f"counts_{job}.json").read_text(encoding="utf-8")),
               index=json.loads((d / f"flight_{job}.json").read_text(encoding="utf-8")),
               arrays=dict(np.load(d / f"flight_{job}.npz")),
               result=json.loads((d / f"result_{job}.json").read_text(encoding="utf-8")),
               parts={p["tag"]: json.loads((d / f"{p['tag']}.json").read_text(encoding="utf-8")) for p in comp["parts"]})
    sp = d / f"spans_{job}.json"
    art["spans"] = json.loads(sp.read_text(encoding="utf-8"))["spans"] if sp.exists() else None
    return art


def _bits(art, name) -> np.ndarray:
    return np.array([[int(c) for c in s[::-1]] for s in C.bitstrings(art["index"], art["arrays"], name)])


# ------------------------------------------------------------------ integrity
def integrity(art) -> dict:
    order, rec = art["composite"]["circuit_order"], art["composite"]
    bad_counts = [n for n in order if collections.Counter(C.bitstrings(art["index"], art["arrays"], n))
                  != collections.Counter({k: int(v) for k, v in art["counts"][n].items()})]
    bad_shots = [n for n in order if sum(int(v) for v in art["counts"][n].values()) != rec["shots"][n]]
    agree = []
    for tag in ("r146", "blk147"):
        for x in art["parts"][tag]["diag"]:
            if x["meta"]["family"] != "ro":
                continue
            for prefix in (f"{tag}__", f"info__end_{tag}_"):
                b = _bits(art, prefix + x["name"])
                agree.append(float((b == np.array(x["meta"]["prepared"])).mean()))
    return dict(circuits=len(order), counts_equal_flight=len(order) - len(bad_counts), shots_exact=len(order) -
                len(bad_shots), readout_reads_its_pattern_min=min(agree), readout_circuits=len(agree),
                spans_cover_every_shot=art["index"]["spans_cover_every_shot"])


# ------------------------------------------------------------------ twirl blocks
def twirl_blocks(art) -> dict:
    """the per-randomization means of each circuit against binomial: chi2 with R-1 dof if the randomizations agree"""
    order = art["composite"]["circuit_order"]
    R = art["composite"]["randomizations"]
    shape_ok, checked = 0, 0
    if art["spans"]:
        for s in art["spans"]:
            for i, sl in s.get("slices", {}).items():
                checked += 1
                shape = sl[0]
                shape_ok += int(len(shape) == 2 and shape[0] == R
                                and shape[0] * shape[1] == art["composite"]["shots"][order[int(i)]])
    fams: dict = collections.defaultdict(list)
    for name in order:
        tag, _, inner = name.partition("__")
        b = _bits(art, name)
        n = len(b)
        if n % R:
            continue
        x = (b.sum(axis=1) / b.shape[1]).astype(float)
        blocks = x.reshape(R, n // R).mean(axis=1)
        var = x.var(ddof=1) / (n // R)
        if var <= 0:
            continue
        chi2 = float(((blocks - x.mean()) ** 2).sum() / var)
        key = f"{tag}:{inner[:2] if tag in ('r146', 'blk147', 'info') else inner[0]}"
        fams[key].append(chi2)
    return dict(randomizations=R, slice_shapes_checked=checked, slice_shapes_randomization_major=shape_ok,
                expected_mean_chi2=R - 1,
                families={k: dict(circuits=len(v), mean_chi2=float(np.mean(v)), frac_p_below_0_01=float(
                    np.mean(np.array(v) > 18.475))) for k, v in sorted(fams.items())})


# ------------------------------------------------------------------ double read
def double_read(art) -> dict:
    out = {}
    for cal in art["parts"]["r147"]["mcm_cal"]:
        cnt = art["counts"]["r147__" + cal["name"]]
        tot = sum(int(c) for c in cnt.values())
        pat = cal["pattern"]
        rows = {}
        for t, q in enumerate(cal["qubits"]):
            n = np.zeros((2, 2), int)            # [first wrong][second wrong], INTEGER indices
            for k, c in cnt.items():
                b = k[::-1]
                n[int(int(b[t]) != pat), int(int(b[M + t]) != pat)] += int(c)
            rows[str(q)] = dict(right_right=int(n[0, 0]), second_only_wrong=int(n[0, 1]),
                                first_only_wrong=int(n[1, 0]), both_wrong=int(n[1, 1]),
                                state_wrong_at_first_read=n[1, 1] / tot, misassignment_first_read=n[1, 0] / tot,
                                change_between_reads=n[0, 1] / tot)
        out[f"prepared_{pat}"] = dict(shots=tot, per_qubit=rows)
    return out


# ------------------------------------------------------------------ vendor vs in-circuit
def vendor_rank(art) -> dict:
    from scipy.stats import spearmanr
    out = {}
    for tag, key in (("r146", "r146"), ("blk147", "block")):
        fit = art["result"]["r146"]["fit"] if key == "r146" else art["result"]["block"]["fit"]
        rec = art["parts"][tag]
        sn, t = rec["snapshot"], rec["wait_dt"] * rec["snapshot"]["dt"]
        qs = [int(q) for q in fit["readout"]]
        t1 = art["result"]["exploratory"]["t1"][tag]["per_qubit"]
        metrics = {
            "readout": ([0.5 * (fit["readout"][str(q)]["p01"] + fit["readout"][str(q)]["p10"]) for q in qs],
                        [0.5 * (sn["p01"][q] + sn["p10"][q]) for q in qs]),
            "dephasing_over_wait": ([-math.log(max(fit["xs"][str(q)]["contrast_w"], 1e-6)
                                               / max(fit["xs"][str(q)]["contrast_w0"], 1e-6)) for q in qs],
                                    [t / sn["t2"][q] for q in qs]),
            "sx": ([fit["sx"][str(q)]["p_sx"] for q in qs], [1.5 * sn["sx_error"][q] for q in qs]),
            "cz": ([v["p"] for v in fit["ladders"].values()], [1.25 * v["vendor_r"] for v in fit["ladders"].values()]),
            "t1": ([t1[str(q)]["t1_in_circuit_s"] for q in qs], [sn["t1"][q] for q in qs]),
        }
        rows = {}
        for name, (a, b) in metrics.items():
            ok = [(x, y) for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
            rho, p = spearmanr([x for x, _ in ok], [y for _, y in ok])
            rows[name] = dict(n=len(ok), spearman=float(rho), p=float(p),
                              median_in_circuit_over_vendor=float(np.median([x / y for x, y in ok if y])))
        deph = metrics["dephasing_over_wait"]
        rows["worst_dephasers"] = [dict(qubit=q, in_circuit=x, vendor=v) for x, v, q in
                                   sorted(zip(deph[0], deph[1], qs), reverse=True)[:5]]
        out[tag] = rows
    return out


# ------------------------------------------------------------------ loss location
def loss_location(art) -> dict:
    pu = art["result"]["r147"]["per_unit"]
    by: dict = collections.defaultdict(list)
    for u in art["parts"]["r147"]["units"]:
        if u["kind"] != "main":
            continue
        loss = u["ideal"] - pu[u["name"]]["measured"]
        by[("stratum", u["stratum"])].append((u["weight"], loss))
        if u["stratum"] != "erased_erased":
            for k, v in (("input", u["unit"]["inp"][0]), ("sector", u["unit"]["choice"][0]),
                         ("branch", u["unit"]["branch"])):
                by[(k, v)].append((u["weight"], loss))
    tot = sum(w * x for k, v in by.items() if k[0] == "stratum" for w, x in v)
    out: dict = dict(F_avg_deficit=tot, F_e_deficit=1.5 * tot)
    for (kind, val), v in sorted(by.items()):
        row = dict(n=len(v), mean_loss=float(np.mean([x for _, x in v])))
        if kind == "stratum":
            row["share_of_deficit"] = sum(w * x for w, x in v) / tot
        out.setdefault(kind, {})[str(val)] = row
    return out


# ------------------------------------------------------------------ models (slow)
def predictors(art, traj: int = 24) -> dict:
    """the two held-out predictions every model variant is scored on, as functions of a device model: R146's p0 F_e
    (static, routed) and R147's F_e (dynamic stand-ins) with its per-kind residuals -- plus the snapshots they start
    from (the vendor's; each block's in-circuit repair) and the measured values. Shared by models() and
    r155_heldout_r152.py, so a variant is never scored by a second copy of the scoring."""
    import hw_diag_run as D146
    import hw_dynamic_run as R147
    import hw_superact_run as HS
    import hw_two_use as H

    from oneq.hw import engine as E
    from oneq.hw import freeze as F
    from oneq.hw import noise as N
    from oneq.hw import snapshot as S
    d, res, counts = art["dir"], art["result"], art["counts"]
    r146, blk, r147 = art["parts"]["r146"], art["parts"]["blk147"], art["parts"]["r147"]
    c146 = F.load(r146, (d / "r146.qpy").read_bytes(), check_sources=False)
    c147 = F.load(r147, (d / "r147.qpy").read_bytes(), check_sources=False)
    p147 = R147.load_prediction_circuits(r147, (d / "r147.pred.qpy").read_bytes())
    q146 = D146.repaired_params(res["r146"]["fit"])

    def fe146(model, **kw):
        params = N.NoiseParams(**{**q146.describe(), **kw})
        pr = {u["name"]: E.expectation(c146[u["name"]], H.score(r146["m"], HS.unit_from(u["unit"])), model, params,
                                       traj, np.random.default_rng([11, k])).mean for k, u in enumerate(r146["units"])}
        return HS.estimate(pr, r146["units"])["F_e"]

    def fe147(model, stand_ins=None, **kw):
        params = N.NoiseParams(cz_factor=1.0, schedule=q146.schedule, **kw)
        preds = R147.predict_units(r147, c147, p147 if stand_ins is None else stand_ins, model, params, traj=traj)
        out = R147.analyse(R147.with_predictions(r147, preds, model),
                           {n: counts["r147__" + n] for n in r147["circuit_order"]})
        kinds: dict = collections.defaultdict(list)
        for v in out["per_unit"].values():
            kinds[v["kind"]].append(v["measured"] - v["predicted"])
        return out["predicted"]["F_e"], {k: float(np.mean(r)) for k, r in kinds.items()}

    return dict(fe146=fe146, fe147=fe147, stand_ins=p147, record147=r147,
                vendor=S.DeviceSnapshot.from_json(r146["snapshot"]),
                m146=D146.repaired_snapshot(r146, res["r146"]["fit"]),
                m147=D146.repaired_snapshot(blk, res["block"]["fit"]), params146=q146,
                meas146=res["r146"]["held_out"]["p0"]["F_e_measured"], meas147=res["r147"]["measured"]["F_e"],
                qubits146=sorted({c.find_bit(x).index for n in (u["name"] for u in r146["units"])
                                  for c in (c146[n],) for i in c.data for x in i.qubits}),
                qubits147=sorted({c.find_bit(x).index for c in c147.values() for i in c.data for x in i.qubits}))


def models(art, traj: int = 24) -> dict:
    from oneq.hw import snapshot as S
    P = predictors(art, traj)
    fe146, fe147, vendor, m146, m147 = P["fe146"], P["fe147"], P["vendor"], P["m146"], P["m147"]

    def variant(model, t2_vendor, s):
        dd = model.to_json()
        if t2_vendor:
            dd["t2"] = list(vendor.t2)
        dd["cz"] = {k: [r * s, dur] for k, (r, dur) in dd["cz"].items()}
        return S.DeviceSnapshot.from_json(dd)

    meas146, meas147 = P["meas146"], P["meas147"]
    out: dict = dict(traj=traj, measured=dict(r146=meas146, r147=meas147), variants={})
    for name, t2v in (("V0", False), ("V1", True)):
        f6 = fe146(variant(m146, t2v, 1.0))
        f7, k7 = fe147(variant(m147, t2v, 1.0))
        out["variants"][name] = dict(r146=f6, r146_miss=meas146 - f6, r147=f7, r147_miss=meas147 - f7, residual=k7)
    lo, hi = 0.8, 3.0                                   # F_e falls monotonically with s: bisection on R146 ONLY
    for _ in range(9):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fe146(variant(m146, True, mid)) > meas146 else (lo, mid)
    s = (lo + hi) / 2
    f6 = fe146(variant(m146, True, s))
    f7, k7 = fe147(variant(m147, True, s))
    out["variants"]["V1s"] = dict(s_fitted_on_r146=s, r146=f6, r146_miss=meas146 - f6, r147=f7,
                                  r147_miss_held_out=meas147 - f7, residual=k7)
    full = variant(m147, True, s)
    budget = {"full": f7}
    for label, mdl, kw in (("no_cz", variant(m147, True, 0.0), {}), ("no_readout", full, dict(readout=False)),
                           ("no_sx", full, dict(sx_factor=0.0)), ("no_idle", full, dict(idle=False)),
                           ("cz_x0.5", variant(m147, True, s * 0.5), {}), ("cz_x0.25", variant(m147, True, s * 0.25), {})):
        budget[label] = fe147(mdl, **kw)[0]
    out["r147_budget_on_V1s"] = budget
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record_dir")
    ap.add_argument("--models", action="store_true", help="also the model variants and the budget (minutes)")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    art = load(pathlib.Path(a.record_dir))
    out = dict(schema="oneq.r152.posthoc/1", job=art["job"], label="EXPLORATORY -- outside every preregistered verdict",
               integrity=integrity(art), twirl_blocks=twirl_blocks(art), double_read=double_read(art),
               vendor_rank=vendor_rank(art), loss_location=loss_location(art))
    if a.models:
        out["models"] = models(art, a.traj)
    dest = art["dir"] / f"posthoc_{art['job']}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(integrity=out["integrity"], models=out.get("models", {}).get("variants")), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
