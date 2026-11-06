"""WHERE R152's missing error lives: every unit's residual regressed on its first-order loss by error source
(EXPLORATORY: outside every preregistered verdict).

R152's in-circuit model (each block's own repair, CZ at the ladder rate) over-predicted both protocols: R146 p0 by
0.054 and R147 by 0.037. One global knob (V1s: every CZ x 2.95, fitted on R146) closes both, but a knob is not a
location. The engine's linear response (oneq.hw.sensitivity: exact for every reachable error, first order for the
blocked) gives each unit u its loss L_k(u) = sum_e w_e dS/dw_e from each error source k. If source k's true rate is
(1 + beta_k) times the model's, the unit's residual is, to first order,

    measured_u - predicted_u = sum_k beta_k L_k(u)

so the ~500 units -- which use the sources in different proportions -- locate the excess by weighted least squares
(weights: shot variance). Sources: CZ, sx, idle (white T1/T2), terminal readout, and for R147's dynamic units the
mid-circuit read's assignment mix. Fitted on each protocol, and each fit PREDICTS the other's F_e residual.

  python r152_attribution.py REPO_ROOT [--traj N] [--edges]   -> <R152 record dir>/attribution_<job>.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import hw_dynamic as D  # noqa: E402
import hw_dynamic_run as R147  # noqa: E402
import hw_superact_run as HS  # noqa: E402
import hw_two_use as H  # noqa: E402
import r152_posthoc as P152  # noqa: E402

from oneq.hw import engine as E  # noqa: E402
from oneq.hw import freeze as F  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import sensitivity as SE  # noqa: E402
from oneq.hw import stats as T  # noqa: E402

R152_DIR = pathlib.Path("evidence/superactivation/campaign_r152/20261017T003152Z")
CLASSES = ("cz", "sx", "idle", "readout")


def losses(qc, score, snap, params, edges: bool, qubits: bool | None = None) -> dict:
    """the unit's first-order loss per source class (and per CZ edge, and per QUBIT, when asked); '_value' is the
    linear response's own expectation (its reachable part). Per qubit: every error's w dS/dw, split equally over the
    qubits its Pauli acts on (a ZI after a CZ is qubit a's alone), plus that qubit's readout -- the place a defect of
    ONE qubit (a fixed detuning, a bad frequency) shows whichever gate it hides behind."""
    s = SE.sensitivity(qc, score, snap, params)
    qubits = edges if qubits is None else qubits
    out = dict.fromkeys(CLASSES, 0.0)
    out["_value"] = s.value
    for key, v in s.loss_budget.items():
        if key[0] in out:
            out[key[0]] += v
        if edges and key[0] == "cz":
            out[f"cz_{key[1]}_{key[2]}"] = out.get(f"cz_{key[1]}_{key[2]}", 0.0) + v
    if qubits:
        for _, _, _, paulis, w, ds in s.per_error:
            hit = [q for q, g in paulis if g != "I"]
            for q in hit:
                out[f"q{q}"] = out.get(f"q{q}", 0.0) + w * ds / len(hit)
    for key, v in s.components.items():                  # readout: p01 dS/dp01 + p10 dS/dp10, each linear
        if key[0] in ("p01", "p10"):
            q = key[1]
            p = (snap.p01 if key[0] == "p01" else snap.p10)[q] * params.readout_factor
            out["readout"] += p * v
            if qubits:
                out[f"q{q}"] = out.get(f"q{q}", 0.0) + p * v
    return out


def _mix(a: dict, b: dict, r: float) -> dict:
    return {k: (1 - r) * a.get(k, 0.0) + r * b.get(k, 0.0) for k in set(a) | set(b)}


def r146_rows(art, P, traj, edges) -> list:
    d, r146 = art["dir"], art["parts"]["r146"]
    c146 = F.load(r146, (d / "r146.qpy").read_bytes(), check_sources=False)
    model, params = P["m146"], P["params146"]
    stat = [T.Unit(u["name"], u["weight"], u["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in r146["units"]]
    scores = {u["name"]: H.score(r146["m"], HS.unit_from(u["unit"])) for u in r146["units"]}
    tallies = T.tally_all(stat, {u["name"]: art["counts"]["r146__" + u["name"]] for u in r146["units"]}, scores)
    rows = []
    for k, u in enumerate(r146["units"]):
        n = u["name"]
        pred = E.expectation(c146[n], scores[n], model, params, traj, np.random.default_rng([11, k]))
        t = tallies[n]
        L = losses(c146[n], scores[n], model, params, edges)
        L.pop("_value")
        rows.append(dict(name=n, kind="p0", weight=u["weight"], measured=t.mean, predicted=pred.mean,
                         var=max(t.s2 / t.n - t.mean ** 2, 1e-6) / t.n + pred.se ** 2, L=L))
    return rows


def r147_rows(art, P, traj, edges) -> list:
    d, r147 = art["dir"], P["record147"]
    c147 = F.load(r147, (d / "r147.qpy").read_bytes(), check_sources=False)
    model = P["m147"]
    params = N.NoiseParams(cz_factor=1.0, schedule=P["params146"].schedule)
    preds = R147.predict_units(r147, c147, P["stand_ins"], model, params, traj=traj)
    out = R147.analyse(R147.with_predictions(r147, preds, model),
                       {n: art["counts"]["r147__" + n] for n in r147["circuit_order"]})
    stat = [T.Unit(u["name"], u["weight"], r147["shots"], u["n_bits"], 0.0, u["hi"], u["stratum"]) for u in r147["units"]]
    tallies = T.tally_all(stat, {u["name"]: art["counts"]["r147__" + u["name"]] for u in r147["units"]},
                          {u["name"]: R147._score(u) for u in r147["units"]})
    rows = []
    for u in r147["units"]:
        n, inp = u["name"], u["prediction_inputs"]
        uu = R147.unit_from(u["unit"])
        score = R147.FA.score(R147.M, uu, R147.DECODER) if inp["score"] == "FA" else H.score(R147.M, uu)
        if inp["kind"] == "dynamic":
            sn = D.augmented(model, [tuple(e) for e in inp["virt"]])
            r = 0.5 * (1 - math.prod(1 - 2 * model.readout_error(q) for q in inp["mix_qubits"]))
            L0 = losses(P["stand_ins"][inp["c0"]], score, sn, params, edges)
            L1 = losses(P["stand_ins"][inp["c1"]], score, sn, params, edges) if inp["c1"] else L0
            L = _mix(L0, L1, r)
            # the mid-circuit read's assignment error mixes in the wrong-correction circuit: its loss is r (e1 - e0)
            L["mcm_readout"] = r * (L1["_value"] - L0["_value"])
        else:
            L = losses(c147[n], score, model, params, edges)
            L["mcm_readout"] = 0.0
        L.pop("_value")
        t, pu = tallies[n], out["per_unit"][n]
        rows.append(dict(name=n, kind=u["kind"], weight=u["weight"], measured=t.mean, predicted=pu["predicted"],
                         var=max(t.s2 / t.n - t.mean ** 2, 1e-6) / t.n + preds[n][1] ** 2, L=L))
    return rows


def fit(rows: list, keys: list) -> dict:
    """weighted least squares of the residuals on the losses; beta, its sd, the fit's chi2 against the no-excess
    model, and the design's condition number (collinear sources cannot be told apart)"""
    X = np.array([[r["L"].get(k, 0.0) for k in keys] for r in rows])
    y = np.array([r["measured"] - r["predicted"] for r in rows])
    w = np.array([1 / r["var"] for r in rows])
    Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.pinv(Xw.T @ Xw)
    chi0, chi1 = float(yw @ yw), float(((yw - Xw @ beta) ** 2).sum())
    sv = np.linalg.svd(Xw, compute_uv=False)
    return dict(keys=keys, beta={k: float(b) for k, b in zip(keys, beta)},
                sd={k: float(math.sqrt(max(cov[i, i], 0.0))) for i, k in enumerate(keys)},
                chi2_no_excess=chi0, chi2_fit=chi1, n=len(rows), condition=float(sv[0] / max(sv[-1], 1e-300)),
                correlation=np.corrcoef(X.T).round(3).tolist() if len(keys) > 1 else [[1.0]])


def fe_shift(rows: list, beta: dict) -> float:
    """the F_e change the fitted excess implies for these rows' design (F_e = 3/2 sum_u w_u mean_u - 1/2)"""
    return 1.5 * sum(r["weight"] * sum(b * r["L"].get(k, 0.0) for k, b in beta.items()) for r in rows
                     if r["kind"] in ("p0", "main"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    ap.add_argument("--edges", action="store_true", help="also fit one CZ excess per edge, and one excess per QUBIT")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    art = P152.load(root / R152_DIR)
    P = P152.predictors(art, a.traj)
    rows6 = r146_rows(art, P, a.traj, a.edges)
    print(f"R146 p0: {len(rows6)} units", flush=True)
    rows7 = r147_rows(art, P, a.traj, a.edges)
    print(f"R147: {len(rows7)} units", flush=True)
    out: dict = dict(schema="oneq.r152.attribution/1", label="EXPLORATORY -- outside every preregistered verdict",
                     job=art["job"], traj=a.traj, base_model="each block's in-circuit repair, CZ at factor 1 (V0)",
                     fits={})
    k6, k7 = list(CLASSES), list(CLASSES) + ["mcm_readout"]
    out["fits"]["r146"] = f6 = fit(rows6, k6)
    out["fits"]["r147"] = f7 = fit(rows7, k7)
    out["fits"]["joint"] = fit(rows6 + rows7, k7)
    for one in ("cz", "sx", "idle", "readout"):          # the single-source readings, each on its own
        out["fits"][f"r146_{one}_only"] = fit(rows6, [one])
        out["fits"][f"r147_{one}_only"] = fit(rows7, [one])
    meas6 = 1.5 * sum(r["weight"] * (r["measured"] - r["predicted"]) for r in rows6)
    main7 = [r for r in rows7 if r["kind"] == "main"]
    meas7 = 1.5 * sum(r["weight"] * (r["measured"] - r["predicted"]) for r in main7)
    out["cross"] = dict(
        r146_residual_F_e=meas6, r147_residual_F_e_stratified=meas7,
        r146_fit_predicts_r147=fe_shift(rows7, f6["beta"]), r147_fit_predicts_r146=fe_shift(rows6, f7["beta"]),
        cz_only_r146_predicts_r147=fe_shift(rows7, out["fits"]["r146_cz_only"]["beta"]))
    if a.edges:
        e6 = sorted({k for r in rows6 for k in r["L"] if k.startswith("cz_")})
        e7 = sorted({k for r in rows7 for k in r["L"] if k.startswith("cz_")})
        out["fits"]["r146_edges"] = fit(rows6, e6 + ["sx", "idle", "readout"])
        out["fits"]["r147_edges"] = fit(rows7, e7 + ["sx", "idle", "readout", "mcm_readout"])
        q6 = sorted({k for r in rows6 for k in r["L"] if k.startswith("q")}, key=lambda k: int(k[1:]))
        q7 = sorted({k for r in rows7 for k in r["L"] if k.startswith("q")}, key=lambda k: int(k[1:]))
        out["fits"]["r146_qubits"] = fit(rows6, q6)
        out["fits"]["r147_qubits"] = fit(rows7, q7 + ["mcm_readout"])
        for name in ("r146_qubits", "r147_qubits"):
            f = out["fits"][name]
            print(name, {k: f"{f['beta'][k]:+.2f}+-{f['sd'][k]:.2f}" for k in f["keys"]},
                  f"chi2 {f['chi2_no_excess']:.0f} -> {f['chi2_fit']:.0f} (cond {f['condition']:.0f})", flush=True)
    out["rows"] = dict(r146=rows6, r147=rows7)
    dest = root / R152_DIR / f"attribution_{art['job']}.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    for name in ("r146", "r147", "joint"):
        f = out["fits"][name]
        print(name, {k: f"{f['beta'][k]:+.2f} +- {f['sd'][k]:.2f}" for k in f["keys"]},
              f"chi2 {f['chi2_no_excess']:.0f} -> {f['chi2_fit']:.0f} (n {f['n']}, cond {f['condition']:.0f})")
    print(json.dumps(out["cross"], indent=1))
    print("wrote", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
