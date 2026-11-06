"""R202 -- the m = 8 sequential ion package's group COMPOSITION and ORDER, decided on the exact certificate.

The package certifies f_lower = f_ideal - sum_g width_g W_g CP_upper(k_g, n_g, a) at a = delta / looks / groups, after
each look, and stops at the first look whose 1.5 f_lower - 0.5 clears 3/4. Which units sit in each look's prefix --
n_g(look) -- is a free design choice before any data. R201 inherited the Neyman split (width W sqrt(ebar), 5.55%
ee_x) in acceptance order, where the rejection sampler fills the ee_x quota early.

This lane computes, EXACTLY (no Monte Carlo):
  per_look    P(one look clears) as a function of its ee_x count, per lambda and claim level. Exact binomial sums.
  sequential  P(the run stops at each look) and the expected HQC, for each candidate ORDER, by a DP over the joint
              cumulative error counts (K_ee, K_kept). The two groups' counts are independent Markov chains across the
              looks, so the unstopped mass at look L is T_ee' M_{L-1} T_kept, cut by the clearing region.
  last_look   lambda* (the largest lambda with P(clear) >= 0.9) at the full package against its ee_x share.

The rates are R195's executed projection for Helios (group means, `r200_ion_robustness.projected_group_rates`) times
lambda: SIMULATED inputs, exact arithmetic. Per-batch HQC from each package's own receipts.

  python r202_composition.py      -> evidence/superactivation/r202_composition/composition.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import beta, binom

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ion_run_package as P  # noqa: E402
import ion_sequential as Q  # noqa: E402
import r200_ion_robustness as R  # noqa: E402

PKG_R201 = ROOT / "evidence" / "superactivation" / "ion_run_m8_sequential"
PKG_R202 = ROOT / "evidence" / "superactivation" / "ion_run_m8_r202"
OUT = ROOT / "evidence" / "superactivation" / "r202_composition" / "composition.json"
EE, KE = Q.RARE, Q.COMMON
CLAIMS = {"3sigma": 0.00135, "5sigma": Q.CLAIM_DELTA}
LAMS = (1.0, 1.5, 2.0, 2.5)
TAIL = 1e-13


def _design(pkg: pathlib.Path) -> dict:
    return json.loads((pkg / "design.json").read_text(encoding="utf-8"))


def certificate(design: dict) -> dict:
    """the clearing condition in the error-rate plane: c_ee U_ee + c_ke U_ke < budget"""
    g = design["groups"]
    return dict(c_ee=g[EE]["width"] * g[EE]["W"], c_ke=g[KE]["width"] * g[KE]["W"],
                budget=design["f_ideal"] - (design["bar_F_e"] + 0.5) / 1.5)


def ucb(n: int, a: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    out = np.ones(kmax + 1)
    m = k < n
    out[m] = beta.ppf(1 - a, k[m] + 1, n - k[m])
    return out


def _kmax(n: int, p: float) -> int:
    return int(min(n, binom.ppf(1 - TAIL, n, p) + 2)) if n else 0


def p_clear(n_ee: int, n_ke: int, p_ee: float, p_ke: float, a: float, cert: dict) -> float:
    """exact P(one look at these counts clears)"""
    if n_ke <= 0 or n_ee <= 0:
        return 0.0
    ke, ee = _kmax(n_ke, p_ke), _kmax(n_ee, p_ee)
    ok = cert["c_ee"] * ucb(n_ee, a, ee)[:, None] + cert["c_ke"] * ucb(n_ke, a, ke)[None, :] < cert["budget"]
    pe, pk = binom.pmf(np.arange(ee + 1), n_ee, p_ee), binom.pmf(np.arange(ke + 1), n_ke, p_ke)
    return float((pe[:, None] * pk[None, :] * ok).sum())


def _transition(n_prev: int, n_next: int, p: float, kmax: int) -> np.ndarray:
    """T[k, k'] = P(cumulative count k' after the segment | k before), the segment Binomial(n_next - n_prev, p)"""
    seg = binom.pmf(np.arange(kmax + 1), n_next - n_prev, p)
    T = np.zeros((kmax + 1, kmax + 1))
    for k in range(kmax + 1):
        T[k, k:] = seg[:kmax + 1 - k]
    return T


def sequential_exact(counts: list, costs: list, p_ee: float, p_ke: float, a: float, cert: dict) -> dict:
    """counts[L] = (n_ee, n_ke) in the prefix of look L; costs[L] = HQC of batch L. Exact stopping law."""
    ee, ke = _kmax(counts[-1][0], p_ee), _kmax(counts[-1][1], p_ke)
    mass = np.zeros((ee + 1, ke + 1))
    mass[0, 0] = 1.0
    prev = (0, 0)
    stop = []
    for (n_ee, n_ke) in counts:
        mass = _transition(prev[0], n_ee, p_ee, ee).T @ mass @ _transition(prev[1], n_ke, p_ke, ke)
        ok = cert["c_ee"] * ucb(n_ee, a, ee)[:, None] + cert["c_ke"] * ucb(n_ke, a, ke)[None, :] < cert["budget"]
        stop.append(float(mass[ok].sum()))
        mass = np.where(ok, 0.0, mass)
        prev = (n_ee, n_ke)
    cum = np.cumsum(costs)
    never = max(0.0, 1.0 - sum(stop))
    return dict(clears_by_look=[round(s, 6) for s in stop], holds=round(sum(stop), 6),
                expected_hqc=round(float(sum(s * c for s, c in zip(stop, cum)) + never * cum[-1]), 1))


def prefix_counts(units: list, looks: list) -> list:
    out = []
    for n in looks:
        g = [u["group"] for u in units[:n]]
        out.append((g.count(EE), g.count(KE)))
    return out


def batch_costs(pkg: pathlib.Path, n_batches: int) -> list:
    return [json.loads((pkg / f"batch_{k}" / "param_shots_receipt.json").read_text(encoding="utf-8"))["hqc"]["total"]
            for k in range(1, n_batches + 1)]


def orders(looks: list) -> dict:
    """the candidate orders, each as its looks' (n_ee, n_kept) prefix counts"""
    d201 = _design(PKG_R201)
    out = {"R201_committed": prefix_counts(d201["units"], looks)}
    e = Q.ebar_rule()
    out["neyman_stratified"] = [(P.allocation(n, m=8, ebar=e)["alloc"][EE], P.allocation(n, m=8, ebar=e)["alloc"][KE])
                                for n in looks]
    for share in (0.10, 0.15, 0.20, 0.25):
        out[f"uniform_{int(round(share * 100))}"] = [(round(share * n), n - round(share * n)) for n in looks]
    if (PKG_R202 / "design.json").is_file():
        out["R202_committed"] = prefix_counts(_design(PKG_R202)["units"], looks)
    return out


def lam_star(n: int, n_ee: int, a: float, rates: dict, cert: dict, target: float = 0.9):
    def ok(lam):
        return p_clear(n_ee, n - n_ee, min(1, lam * rates[EE]), min(1, lam * rates[KE]), a, cert) >= target
    if not ok(1.0):
        return None
    lo, hi = 1.0, 4.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return round(lo, 3)


def run() -> dict:
    d201 = _design(PKG_R201)
    seq = json.loads((PKG_R201 / "sequential.json").read_text(encoding="utf-8"))
    looks = seq["looks"]
    cert = certificate(d201)
    rates = R.projected_group_rates("helios_1", "exec")
    costs = {"R201": batch_costs(PKG_R201, len(looks))}
    if (PKG_R202 / "batch_4" / "param_shots_receipt.json").is_file():
        costs["R202"] = batch_costs(PKG_R202, len(looks))
    cand = orders(looks)
    out: dict = dict(schema="oneq.r202_composition/1", looks=looks, certificate=cert, rates=rates,
                     rates_source="R195 executed projection, Helios, group means (r200_ion_robustness)",
                     batch_hqc=costs, orders={k: [list(c) for c in v] for k, v in cand.items()},
                     per_look=[], sequential={}, last_look=[])
    for claim, delta in CLAIMS.items():
        a = delta / len(looks) / 2
        for lam in LAMS:
            p_ee, p_ke = min(1, lam * rates[EE]), min(1, lam * rates[KE])
            for L in looks[:-1]:
                grid = list(range(8, min(1600, L) + 1, 8))
                curve = [(n, p_clear(n, L - n, p_ee, p_ke, a, cert)) for n in grid]
                best = max(curve, key=lambda t: t[1])
                out["per_look"].append(dict(claim=claim, lam=lam, look=L, best_n_ee=best[0],
                                            best_p=round(best[1], 6),
                                            curve=[[n, round(p, 6)] for n, p in curve[::4]]))
            for name, cnt in cand.items():
                c = costs["R202"] if name == "R202_committed" and "R202" in costs else costs["R201"]
                out["sequential"][f"{claim}|{lam}|{name}"] = sequential_exact(cnt, c, p_ee, p_ke, a, cert)
        n = looks[-1]
        for n_ee in (222, 444, 800, 1200, 1600, 2000):
            out["last_look"].append(dict(claim=claim, n=n, n_ee=n_ee, lam_star_p90=lam_star(n, n_ee, a, rates, cert)))
    return out


def main() -> int:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out["sequential"].items():
        if k.startswith("5sigma"):
            print(f"{k:34s} holds {v['holds']:.4f}  E[HQC] {v['expected_hqc']:7.1f}  by look {v['clears_by_look']}")
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
