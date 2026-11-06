"""R225 -- rule 'betting_ad' (the F_e + antidegradability-witness mixture) on the frozen R224 package: its operating
characteristics with and without the futility stop, and its validity simulated on null devices.

  oc            P(holds) and E[HQC] per noise direction x lambda (r225_witness_choice's device: R195's executed
                Helios projection per (group, basis), kept's basis profile by direction), at 5 sigma (and 3 sigma in
                the model direction), with the run's OWN constants (ion_betting.ad_components on the package). The
                wealth is a function of the per-look counts by category, so a run is simulated at that level
                (multinomial per batch; SIMULATED, each holds with its standard error).
  oc_futility   the same with the run's futility stop applied by ion_betting.futility_bound_ad itself, on the same
                count draws (common random numbers).
  futility      THE CARRY-OVER RULE (declared before this file's first run, 2026-10-26): R209's parameters
                (ion_betting.FUTILITY, fitted to F_e) carry over to the mixture iff, in every cell where the mixture
                without futility holds >= 0.5, the stop costs at most 0.002 of P(holds). Otherwise the run is not
                ready: the grid is to be searched by R209's criterion.
  validity      the false-claim rate at delta = 0.05 on non-anticipating devices IN the mixture's null (BOTH
                components' running means on or above their thresholds at every look, or judged on their own
                running averages): R203's six families, rebuilt for four categories at the CORNER of the null (both
                thresholds met with equality); validity_control runs them on scores scaled by 0.5, which each must
                fail. Ville says each rate is <= delta (ion_betting's R225 comment; lean: ville_family).

  python r225_mixture_oc.py [REPS]     -> evidence/superactivation/r225_witness/oc_mixture.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ad_witness as A  # noqa: E402
import ion_betting as B  # noqa: E402
import ion_sequential as Q  # noqa: E402
import r203_betting as R3  # noqa: E402
import r225_witness_choice as C  # noqa: E402

PKG = C.PKG
OUT = C.EV / "r225_witness" / "oc_mixture.json"
CLAIMS = {"5sigma": Q.CLAIM_DELTA, "3sigma": 0.00135}
FUTILITY_COST_CAP, FUTILITY_FLOOR = 0.002, 0.5


def package() -> tuple:
    """R224's design and its sequential record READ by rule 'betting_ad' (the record itself once adopted)"""
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((PKG / "sequential.json").read_text(encoding="utf-8"))
    if seq.get("rule") != Q.AD_RULE:
        seq = dict({k: v for k, v in seq.items() if k != "witness"}, rule=Q.AD_RULE, witness=Q.witness_fields())
    Q.check_betting_package(design, seq)
    return design, seq


def oc_mc(comps: list, probs: dict, looks: list, costs: list, delta: float, reps: int, seed: int,
          futility: dict | None = None, chunk: int = 25_000) -> dict:
    """the mixture's stopping law: P(clears at each look), P(stopped for futility at each look), E[HQC]. The claim
    test is the run's own (log mean exp of the components >= log(1/delta) + FLOAT_GUARD); with `futility`, a look that
    does not clear, from min_look on and before the last, stops when ion_betting.futility_bound_ad < tau"""
    cats = list(A.CATS)
    pv = [probs[c] for c in cats] + [1 - sum(probs[c] for c in cats)]
    if min(pv) < 0:
        raise ValueError(f"shot probabilities {pv} are not a distribution")
    bar = math.log(1 / delta) + B.FLOAT_GUARD
    cum = np.cumsum(costs)
    clears, futile = np.zeros(len(looks)), np.zeros(len(looks))
    rng = np.random.default_rng(seed)
    done = 0
    while done < reps:
        n = min(chunk, reps - done)
        k = np.zeros((n, len(cats)), np.int64)
        alive = np.ones(n, bool)
        prev = 0
        for li, L in enumerate(looks):
            k += rng.multinomial(L - prev, pv, size=n)[:, :len(cats)]
            prev = L
            n0 = L - k.sum(axis=1)
            lw = np.stack([n0 * math.log1p(c["bet"] * c["m"]) + sum(
                k[:, j] * math.log1p(c["bet"] * (c["m"] - c["kappa"][cat])) for j, cat in enumerate(cats))
                for c in comps])
            top = lw.max(axis=0)
            hit = (top + np.log(np.mean(np.exp(lw - top), axis=0)) >= bar) & alive
            clears[li] += hit.sum()
            alive &= ~hit
            if futility is not None and li < len(looks) - 1 and L >= futility["min_look"] and alive.any():
                idx = np.nonzero(alive)[0]
                u = B.futility_bound_ad(L, {cat: k[idx, j] for j, cat in enumerate(cats)}, looks[li + 1:], comps,
                                        delta, futility["alpha_opt"])
                quit_ = np.zeros(n, bool)
                quit_[idx] = np.asarray(u) < futility["tau"]
                futile[li] += quit_.sum()
                alive &= ~quit_
        done += n
    pc, pf = clears / reps, futile / reps
    never = 1 - pc.sum() - pf.sum()
    out = dict(clears_by_look=[round(float(x), 6) for x in pc], holds=round(float(pc.sum()), 6),
               se=round(float(math.sqrt(max(pc.sum() * (1 - pc.sum()), 1e-12) / reps)), 6),
               expected_hqc=round(float((pc * cum).sum() + (pf * cum).sum() + never * cum[-1]), 1),
               max_hqc=round(float(cum[-1]), 1))
    if futility is not None:
        out.update(futile_by_look=[round(float(x), 6) for x in pf], futile=round(float(pf.sum()), 6))
    return out


# ------------------------------------------------------------------ validity on null devices
FAMILIES = ("stationary", "rare_only", "drift_down", "banker", "memory", "coupled")
NULL_TOL = 1e-9


def corner(comps: list, share: dict) -> dict:
    """per-category error rates (given the category) on BOTH thresholds at once: rare at rate a, kept's three
    bases at b times the model profile -- two linear equations in (a, b)"""
    fe, wit = comps[0], comps[3]
    rates = C.device_rates()
    prof = {c: rates[c] / rates["kept/high"] for c in A.CATS if c.startswith("kept")}
    mat = np.array([[share["ee_x/high|X"] * t["kappa"]["ee_x/high|X"],
                     sum(share[c] * t["kappa"][c] * prof[c] for c in prof)] for t in (fe, wit)])
    a, b = np.linalg.solve(mat, np.array([fe["m"], wit["m"]]))
    if a <= 0 or b <= 0:
        raise ValueError(f"no corner device with positive rates: a={a}, b={b}")
    return {"ee_x/high|X": float(a), **{c: float(b * prof[c]) for c in prof}}


def device_runs(kind: str, comps: list, share: dict, rng, reps: int, n: int) -> tuple:
    """reps runs of one family: (S_F, S_W, mu_F, mu_W), each reps x n -- the two score processes the components bet
    on, and their conditional means given the past"""
    cats = list(A.CATS)
    fe, wit = comps[0], comps[3]
    base = corner(comps, share)
    cum = np.cumsum([share[c] for c in cats])
    cat = np.searchsorted(cum, rng.random((reps, n)) * cum[-1])          # each shot's category, iid
    kF = np.array([fe["kappa"][c] for c in cats])
    kW = np.array([wit["kappa"][c] for c in cats])
    rare_only = {c: 0.0 for c in cats}
    rare_only["ee_x/high|X"] = max(fe["m"] / (share["ee_x/high|X"] * fe["kappa"]["ee_x/high|X"]),
                                   wit["m"] / (share["ee_x/high|X"] * wit["kappa"]["ee_x/high|X"]))
    SF, SW, muF, muW = (np.zeros((reps, n)) for _ in range(4))
    for i in range(n):
        if kind == "stationary":
            e = np.tile([base[c] for c in cats], (reps, 1))
        elif kind == "rare_only":
            e = np.tile([rare_only[c] for c in cats], (reps, 1))
        elif kind == "drift_down":
            e = np.tile([base[c] * (1.5 if i < n // 2 else 0.5) for c in cats], (reps, 1))
        elif kind == "banker":
            e = np.tile([base[c] * (3.0 if i < n // 3 else 0.0) for c in cats], (reps, 1))
        elif kind == "memory":                          # a kept shot right after an ee_x one never errs; on the
            prev_rare = (cat[:, i - 1] == 0) if i else np.zeros(reps, bool)     # corner on average
            boost = 1 / (1 - share["ee_x/high|X"])
            e = np.stack([np.full(reps, base[cats[0]])] + [np.where(prev_rare, 0.0, base[c] * boost)
                                                           for c in cats[1:]], axis=1)
        elif kind == "coupled":                         # 1.075 x the corner after a clean shot, 0 after an error
            f = np.where(SF[:, i - 1] == 0, 1.075, 0.0) if i else np.full(reps, 1.075)
            e = np.stack([base[c] * f for c in cats], axis=1)
        else:
            raise ValueError(kind)
        p = np.array([share[c] for c in cats])
        muF[:, i] = (e * p * kF).sum(axis=1)
        muW[:, i] = (e * p * kW).sum(axis=1)
        err = rng.random(reps) < e[np.arange(reps), cat[:, i]]
        SF[:, i] = kF[cat[:, i]] * err
        SW[:, i] = kW[cat[:, i]] * err
    return SF, SW, muF, muW


def validity(comps: list, share: dict, delta: float = 0.05, reps: int = 4000, n: int = 1500, step: int = 50,
             seed: int = 225, mis_weight: float = 1.0) -> dict:
    """false claims -- the mixture at log(1/delta) + FLOAT_GUARD at a look while BOTH running means are on or above
    their thresholds -- per family. mis_weight < 1 scales every score down: a WRONG certificate each must catch"""
    fe, wit = comps[0], comps[3]
    bar = math.log(1 / delta) + B.FLOAT_GUARD
    rng = np.random.default_rng(seed)
    out = {}
    for kind in FAMILIES:
        SF, SW, muF, muW = device_runs(kind, comps, share, rng, reps, n)
        lk = np.stack([np.cumsum(np.log1p(c["bet"] * (c["m"] - mis_weight * (SF if c["test"] == "F_e" else SW))),
                                 axis=1) for c in comps])                      # comps x reps x n
        mF = np.cumsum(muF, axis=1) / np.arange(1, n + 1)
        mW = np.cumsum(muW, axis=1) / np.arange(1, n + 1)
        done = np.zeros(reps, bool)
        false = claims = null_looks = looks = 0
        for t in range(step, n + 1, step):
            col = lk[:, :, t - 1]
            top = col.max(axis=0)
            hit = (top + np.log(np.mean(np.exp(col - top), axis=0)) >= bar) & ~done
            in_null = (mF[:, t - 1] >= fe["m"] * (1 - NULL_TOL)) & (mW[:, t - 1] >= wit["m"] * (1 - NULL_TOL))
            claims += int(hit.sum())
            false += int((hit & in_null).sum())
            null_looks += int(in_null.sum())
            looks += reps
            done |= hit
        out[kind] = dict(reps=reps, false_claims=false, rate=false / reps, claims=claims,
                         share_of_looks_in_null=round(null_looks / looks, 4),
                         wilson_upper_95=round(R3._wilson_upper(false, reps), 5))
    return dict(delta=delta, shots=n, look_every=step, null_tolerance=NULL_TOL, mis_weight=mis_weight, families=out)


def shares(design: dict) -> dict:
    """P(a shot's category) under the design law: pi for ee_x (all X), 1 - pi split by kept/high's exact basis
    weights"""
    W = C.basis_weights(int(design["m"]))
    kept = sum(W[("kept/high", b)] for b in A.BASES)
    pi = float(Q.PI_RARE)
    return {"ee_x/high|X": pi, **{f"kept/high|{b}": (1 - pi) * float(W[("kept/high", b)] / kept) for b in A.BASES}}


def run(reps: int = 200_000, validity_reps: int = 4000, seed: int = 2251) -> dict:
    design, seq = package()
    comps = B.ad_components(design, seq)
    looks = list(seq["looks"])
    costs, cost_src = R3.batch_costs(looks)
    rates = C.device_rates()
    out: dict = dict(schema="oneq.r225_mixture_oc/1", package=PKG.relative_to(ROOT).as_posix() if PKG.is_relative_to(ROOT) else str(PKG),
                     rule=Q.AD_RULE, witness=Q.witness_fields(), pi_rare=seq["pi_rare"], f_e_bets=seq["bets"],
                     looks=looks, batch_hqc=costs, batch_hqc_source=cost_src,
                     components=[dict(test=c["test"], m=c["m"], bet=c["bet"], kappa=c["kappa"]) for c in comps],
                     rates=rates, rates_source="R195 executed projection, Helios, exec arm, per (group, input basis)",
                     futility_rule=dict(B.FUTILITY), reps=reps, seed=seed, oc={}, oc_futility={})
    cells = [(d, lam, "5sigma") for d in C.DIRECTIONS for lam in C.LAMS] + [("model", lam, "3sigma") for lam in C.LAMS]
    for i, (d, lam, claim) in enumerate(cells):
        probs = C.shot_probs(rates, d, lam, float(seq["pi_rare"]))
        key = f"{claim}|{d}|{lam}"
        out["oc"][key] = oc_mc(comps, probs, looks, costs, CLAIMS[claim], reps, seed + i)
        out["oc_futility"][key] = oc_mc(comps, probs, looks, costs, CLAIMS[claim], reps, seed + i, futility=B.FUTILITY)
    worst = min(((out["oc_futility"][k]["holds"] - v["holds"], k) for k, v in out["oc"].items()
                 if v["holds"] >= FUTILITY_FLOOR), default=(0.0, None))
    out["futility"] = dict(rule=f"R209's parameters carry over iff the stop costs <= {FUTILITY_COST_CAP} of P(holds) in "
                                f"every cell holding >= {FUTILITY_FLOOR} without it",
                           worst_cost=dict(delta_holds=round(worst[0], 6), cell=worst[1]),
                           carries_over=bool(worst[0] >= -FUTILITY_COST_CAP))
    sh = shares(design)
    out["validity"] = validity(comps, sh, reps=validity_reps)
    out["validity_control"] = validity(comps, sh, reps=validity_reps, mis_weight=0.5)
    out["corner_rates"] = corner(comps, sh)
    out["caveats"] = ["SIMULATED device rates (R195's projection) scaled by lambda; directions are scenarios",
                      "Monte Carlo over per-look counts: each holds carries its standard error se"]
    return out


def main(argv) -> int:
    reps = int(argv[0]) if argv else 200_000
    out = run(reps)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out["oc"].items():
        f = out["oc_futility"][k]
        print(f"{k:22s} holds {v['holds']:.4f}  E[HQC] {v['expected_hqc']:7.1f}   with futility: holds "
              f"{f['holds']:.4f}  E[HQC] {f['expected_hqc']:7.1f}")
    print("futility:", json.dumps(out["futility"]))
    for name in ("validity", "validity_control"):
        for k, v in out[name]["families"].items():
            print(f"{name:16s} {k:10s} false {v['false_claims']}/{v['reps']} = {v['rate']:.4f} "
                  f"(95% upper {v['wilson_upper_95']}), in null {v['share_of_looks_in_null']}")
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
