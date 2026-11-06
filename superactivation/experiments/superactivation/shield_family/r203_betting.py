"""R203 -- the betting certificate's operating characteristics (EXACT) and its validity (SIMULATED), for the m = 8
sequential ion package.

  oc        P(the run stops at each look) and the expected HQC, per lambda and claim level, for the chosen design
            (PI_RARE, BETS, LOOKS_R203 of ion_sequential). The mixture wealth at a look depends only on the shot count
            and the error counts by group, and those are trinomial per shot, so a DP over the cumulative counts
            (k_rare, k_common) gives the stopping law exactly. No Monte Carlo.
  tuning    the same for the grid the design was chosen from: pi_rare x look schedule, at 5 sigma.
  validity  the false-claim rate at a LARGE delta (0.05, so it can be seen) for six non-anticipating device families
            IN the null (FAMILIES: on m, rare-only, drifting down, banking errors early, R202's one-shot memory,
            outcome-coupled). Ville says each is <= delta; validity_control runs the same families on a certificate
            mis-weighted by 0.5, which each must catch.
  versus    R201's and R202's stratified Clopper-Pearson rows from r202_composition.json, for comparison.

Rates: R195's executed Helios projection (group means) x lambda (SIMULATED inputs). Per-batch HQC: the R203
package's receipts when present, else 5 + units x 0.1702 (R201's verified per-shot price), flagged.

  python r203_betting.py      -> evidence/superactivation/r203_betting/oc.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from scipy.stats import binom

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ion_betting as B  # noqa: E402
import ion_sequential as Q  # noqa: E402
import r200_ion_robustness as R  # noqa: E402

EV = ROOT / "evidence" / "superactivation"
PKG_R203 = EV / "ion_run_m8_r224"          # R224: the both-erasure-orders re-freeze (R203's was folded, R213)
PKG_DESIGN = EV / "ion_run_m8_sequential"          # any m = 8 package: groups, W, f_ideal are the design law's
OUT = EV / "r203_betting" / "oc.json"
CLAIMS = {"5sigma": Q.CLAIM_DELTA, "3sigma": 0.00135}
LAMS = (1.0, 1.5, 2.0, 2.5, 3.0)
PER_SHOT_FALLBACK = 0.1702
TAIL = 1e-13


def design_constants() -> dict:
    src = PKG_R203 if (PKG_R203 / "design.json").is_file() else PKG_DESIGN
    return json.loads((src / "design.json").read_text(encoding="utf-8"))


def batch_costs(looks: list) -> tuple:
    rc = [PKG_R203 / f"batch_{k}" / "param_shots_receipt.json" for k in range(1, len(looks) + 1)]
    if all(p.is_file() for p in rc):
        return [json.loads(p.read_text(encoding="utf-8"))["hqc"]["total"] for p in rc], "R224 package receipts"
    sizes = [b - a for a, b in zip([0] + list(looks[:-1]), looks)]
    return [5 + s * PER_SHOT_FALLBACK for s in sizes], f"estimate: 5 + units x {PER_SHOT_FALLBACK} per batch"


def oc_exact(design: dict, pi: float, bets, looks: list, costs: list, lam: float, rates: dict, delta: float,
             futility: dict | None = None) -> dict:
    """the exact stopping law of the mixture test over the looks (trinomial per shot). With `futility` (R209,
    ion_betting.FUTILITY's keys), a state that does not clear at a look from min_look on, before the last look, stops
    when ion_betting.futility_bound -- the very function the run applies -- falls below tau: that mass claims
    nothing and spends only the batches up to that look"""
    q_r = pi * min(1.0, lam * rates[B.RARE])
    q_c = (1 - pi) * min(1.0, lam * rates[B.COMMON])
    N = looks[-1]
    k1max = int(min(N, binom.ppf(1 - TAIL, N, q_r))) + 2
    k2max = int(min(N, binom.ppf(1 - TAIL, N, q_c))) + 2
    k1 = np.arange(k1max + 1)[:, None]
    k2 = np.arange(k2max + 1)[None, :]
    m, a = B.threshold(design)["m"], B.scores(design, pi)
    B.check_bets(design, pi, bets)
    bar = math.log(1 / delta)
    mass = np.zeros((k1max + 1, k2max + 1))
    mass[0, 0] = 1.0
    prev, stop, futile = 0, [], []
    for li, L in enumerate(looks):
        s = L - prev
        new = np.zeros_like(mass)
        p1 = binom.pmf(np.arange(k1max + 1), s, q_r)
        for d1 in np.nonzero(p1 > 1e-300)[0]:
            p2 = binom.pmf(np.arange(k2max + 1), s - d1, q_c / (1 - q_r))
            shifted = np.zeros_like(mass)
            shifted[d1:, :] = mass[:k1max + 1 - d1, :]
            conv = np.zeros_like(mass)
            for d2 in np.nonzero(p2 > 1e-300)[0]:
                conv[:, d2:] += shifted[:, :k2max + 1 - d2] * p2[d2]
            new += p1[d1] * conv
        mass = new
        n0 = L - k1 - k2
        lw = np.stack([n0 * math.log1p(b * m) + k1 * math.log1p(b * (m - a[B.RARE]))
                       + k2 * math.log1p(b * (m - a[B.COMMON])) for b in bets])
        top = lw.max(axis=0)
        ok = (top + np.log(np.mean(np.exp(lw - top), axis=0)) >= bar) & (n0 >= 0)
        stop.append(float(mass[ok].sum()))
        mass = np.where(ok, 0.0, mass)
        f = 0.0
        if futility is not None and li < len(looks) - 1 and L >= futility["min_look"]:
            u = B.futility_bound(L, k1, k2, looks[li + 1:], design, pi, bets, delta, futility["alpha_opt"])
            quit_ = (u < futility["tau"]) & (n0 >= 0)
            f = float(mass[quit_].sum())
            mass = np.where(quit_, 0.0, mass)
        futile.append(f)
        prev = L
    cum = np.cumsum(costs)
    never = max(0.0, 1.0 - sum(stop) - sum(futile))
    out = dict(clears_by_look=[round(x, 6) for x in stop], holds=round(sum(stop), 6),
               expected_hqc=round(float(sum(x * c for x, c in zip(stop, cum)) + sum(x * c for x, c in zip(futile, cum))
                                        + never * cum[-1]), 1),
               max_hqc=round(float(cum[-1]), 1))
    if futility is not None:
        out.update(futile_by_look=[round(x, 6) for x in futile], futile=round(sum(futile), 6))
    return out


# The families, all non-anticipating (shot i's label and unit are fresh; its error rate depends only on i and on what
# ran, and how it came out, before i). Corrected 2026-10-21 by the R203 adversary pass: the first harness put every
# family EXACTLY on m and let floating-point rounding decide "mu_bar >= m", its drift family sat below m at every
# look but the last (it could never fail), and its adaptive family sat at 1.49 m (it never claimed). Now each family
# is IN the null at every look or is judged on its own running average with a tolerance, and each must FAIL a
# mis-weighted certificate (its own negative control).
FAMILIES = ("stationary", "rare_only", "drift_down", "banker", "memory", "coupled")
NULL_TOL = 1e-9


def device_runs(kind: str, design: dict, pi: float, rng, reps: int, n: int) -> tuple:
    """reps runs of one device family, side by side: (W, mu), each reps x n"""
    c, m = B.threshold(design)["c"], B.threshold(design)["m"]
    a = B.scores(design, pi)
    rare = rng.random((reps, n)) < pi
    W, mu = np.zeros((reps, n)), np.zeros((reps, n))

    def split(deficit):                                     # the rare group carries 10% of a deficit
        return 0.1 * deficit / c[B.RARE], 0.9 * deficit / c[B.COMMON]
    for i in range(n):
        if kind == "stationary":                            # on m at every shot
            er, ec = split(np.full(reps, m))
        elif kind == "rare_only":                           # on m, the whole deficit on the rare group
            er, ec = np.full(reps, m / c[B.RARE]), np.zeros(reps)
        elif kind == "drift_down":                          # 1.5 m, then 0.5 m: mu_bar >= m at EVERY look
            er, ec = split(np.full(reps, 1.5 * m if i < n // 2 else 0.5 * m))
        elif kind == "banker":                              # 3 m for the first third, then CLEAN: mu_bar >= m
            er, ec = split(np.full(reps, 3.0 * m if i < n // 3 else 0.0))   # throughout, the wealth recovers late
        elif kind == "memory":                              # R202's refuting witness: a kept circuit right after an
            er = np.zeros(reps)                             # ee_x one never errs; on m on average
            ec = np.where(rare[:, i - 1], 0.0, m / (c[B.COMMON] * (1 - pi))) if i else np.full(reps, m / c[B.COMMON])
        elif kind == "coupled":                             # 1.075 m after a clean shot, 0 after an error
            d = np.where(W[:, i - 1] == 0, 1.075 * m, 0.0) if i else np.full(reps, 1.075 * m)
            er, ec = split(d)
        else:
            raise ValueError(kind)
        mu[:, i] = c[B.RARE] * er + c[B.COMMON] * ec
        err = rng.random(reps) < np.where(rare[:, i], er, ec)
        W[:, i] = np.where(rare[:, i], a[B.RARE], a[B.COMMON]) * err
    return W, mu


def validity(design: dict, pi: float, bets, delta: float = 0.05, reps: int = 4000, n: int = 1500, step: int = 50,
             seed: int = 203, mis_weight: float = 1.0, families=FAMILIES) -> dict:
    """false claims -- K >= 1/delta at a look while the run-average deficit mu_bar >= m (1 - NULL_TOL) -- per
    family. mis_weight < 1 scales every score down: a WRONG certificate, which each family must catch"""
    m = B.threshold(design)["m"]
    bets = np.array(B.check_bets(design, pi, bets))
    bar = math.log(1 / delta)
    rng = np.random.default_rng(seed)
    out = {}
    for kind in families:
        W, mu = device_runs(kind, design, pi, rng, reps, n)
        W = W * mis_weight
        lk = np.cumsum(np.log1p(bets[:, None, None] * (m - W)[None, :, :]), axis=2)      # bets x reps x n
        mubar = np.cumsum(mu, axis=1) / np.arange(1, n + 1)
        done = np.zeros(reps, bool)
        false = claims = null_looks = looks = 0
        for t in range(step, n + 1, step):
            col = lk[:, :, t - 1]
            top = col.max(axis=0)
            hit = (top + np.log(np.mean(np.exp(col - top), axis=0)) >= bar) & ~done
            in_null = mubar[:, t - 1] >= m * (1 - NULL_TOL)
            claims += int(hit.sum())
            false += int((hit & in_null).sum())
            null_looks += int(in_null.sum())
            looks += reps
            done |= hit
        out[kind] = dict(reps=reps, false_claims=false, rate=false / reps, claims=claims,
                         share_of_looks_in_null=round(null_looks / looks, 4),
                         wilson_upper_95=round(_wilson_upper(false, reps), 5))
    return dict(delta=delta, shots=n, look_every=step, null_tolerance=NULL_TOL, mis_weight=mis_weight, families=out)


def _wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def run(validity_reps: int = 4000) -> dict:
    design = design_constants()
    rates = R.projected_group_rates("helios_1", "exec")
    looks = list(Q.LOOKS_R203)
    costs, cost_src = batch_costs(looks)
    out: dict = dict(schema="oneq.r203_betting/1", pi_rare=Q.PI_RARE, bets=list(Q.BETS), looks=looks,
                     batch_hqc=costs, batch_hqc_source=cost_src, threshold=B.threshold(design), rates=rates,
                     rates_source="R195 executed projection, Helios, group means (r200_ion_robustness)",
                     oc={}, tuning={}, versus={}, futility_rule=dict(B.FUTILITY), oc_futility={}, futility_grid={})
    for claim, delta in CLAIMS.items():
        for lam in LAMS:
            out["oc"][f"{claim}|{lam}"] = oc_exact(design, Q.PI_RARE, Q.BETS, looks, costs, lam, rates, delta)
            out["oc_futility"][f"{claim}|{lam}"] = oc_exact(design, Q.PI_RARE, Q.BETS, looks, costs, lam, rates,
                                                            delta, futility=B.FUTILITY)
    # R209: the grid the futility rule was chosen from, at 5 sigma (the chosen point: P(holds) loss <= 0.0011 at
    # lambda <= 2, the largest savings at 2.5x and 3x among the rules that keep that)
    for alpha_opt in (0.05, 0.1, 0.2, 0.3):
        for min_look in (1000, 2000):
            for tau in (0.05, 0.1):
                rule = dict(alpha_opt=alpha_opt, min_look=min_look, tau=tau)
                for lam in (1.0, 2.0, 2.25, 2.5, 3.0):
                    out["futility_grid"][f"{alpha_opt}|{min_look}|{tau}|{lam}"] = oc_exact(
                        design, Q.PI_RARE, Q.BETS, looks, costs, lam, rates, Q.CLAIM_DELTA, futility=rule)
    schedules = {"mixed10": looks, "8x1000": [1000 * k for k in range(1, 9)]}
    for pi in (0.04, 0.06, 0.08, 0.10, 0.12):
        for name, lk in schedules.items():
            c = [5 + (b - a) * PER_SHOT_FALLBACK for a, b in zip([0] + lk[:-1], lk)]
            for lam in (1.0, 1.5, 2.0, 2.5):
                try:
                    out["tuning"][f"{pi}|{name}|{lam}"] = oc_exact(design, pi, Q.BETS, lk, c, lam, rates,
                                                                   Q.CLAIM_DELTA)
                except B.BettingRefused as e:      # a small pi makes c/pi large: the chosen bets leave the range
                    out["tuning"][f"{pi}|{name}|{lam}"] = dict(refused=str(e))
    comp = EV / "r202_composition" / "composition.json"
    if comp.is_file():
        s = json.loads(comp.read_text(encoding="utf-8"))["sequential"]
        out["versus"] = {k: v for k, v in s.items() if k.startswith("5sigma|")
                         and k.split("|")[2] in ("R201_committed", "R202_committed", "uniform_20")}
    out["validity"] = validity(design, Q.PI_RARE, Q.BETS, reps=validity_reps)
    out["validity_control"] = validity(design, Q.PI_RARE, Q.BETS, reps=validity_reps, mis_weight=0.5)
    return out


def main() -> int:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out["oc"].items():
        f = out["oc_futility"][k]
        print(f"{k:12s} holds {v['holds']:.4f}  E[HQC] {v['expected_hqc']:7.1f}   with futility: holds "
              f"{f['holds']:.4f}  E[HQC] {f['expected_hqc']:7.1f}  max {v['max_hqc']}")
    for k, v in out["validity"]["families"].items():
        print(f"validity {k:10s} false {v['false_claims']}/{v['reps']} = {v['rate']:.4f} (95% upper "
              f"{v['wilson_upper_95']}) at delta {out['validity']['delta']}")
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
