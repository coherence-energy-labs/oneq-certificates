"""R225 -- the mixture's double-precision claim decision implies the exact-real claim the Lean theorem is about.

The run claims at a look when log_mean_exp(lw) >= log(1/delta) + FLOAT_GUARD (ion_betting.verdict_ad), lw the six
components' log wealths computed in doubles from the look's counts by category. The theorem (lean: ville_family)
bounds the event 1/delta <= K, K = (1/6) sum_j prod_i (1 + b_j (m_j - S_ij)) in exact reals, with b, m, kappa and
delta the EXACT values of the doubles the producer uses. R211 checked R203's 2-category decision state by state
(103,307,010 states); with four categories the reachable states are ~1e13, so this lane does it by a bound:

  THE BOUND (DERIVED). Each component's log wealth is n0 x0 + sum_c k_c x_c with x = log1p(b (m - kappa)):
    - b (m - kappa) in doubles: two roundings, relative error <= 2u (u = 2^-53), so |x err| <= 2u |y| / (1 + y) + the
      log1p's own <= 1 ulp of |x| -- at most ERR_X = 4u max(|x|, |y| / (1 + y));
    - k_c x_c: k_c <= 8,000 is exact; one rounding;  the 5-term sum: 4 roundings of partial sums <= 8,000 max|x|;
    - so |lw float - lw exact| <= 8,000 (ERR_X + 6u max|x|) for each component, and log_mean_exp adds <= 8u (|top| + 3).
  With max|x| = 2.64 (the witness's largest bet against its rare score) and n <= 8,000 this is 7.9e-11 (`derived_bound`,
  recorded); FLOAT_GUARD = 1e-9 is 12x it, so a float claim puts the exact log K above log(1/delta), and Lean's event
  holds.

  THE DRILL (MEASURED). At every look, random states and states BISECTED onto the float decision's boundary (in each
  category's count, the others drawn at random) are recomputed in 60-digit Decimal: |float - exact| must stay under
  the bound, and every float claim must be an exact claim. Two MUTANTS must be caught, so the drill can fail:
    - 'float32': the log wealths in single precision -- the error check must see it exceed the bound;
    - 'bias': the decision shifted by 0.1 the wrong way -- the implication check must see claims the exact test does
      not make. (A shift as small as the guard is not observable: the counts are integers and one error moves log K by
      0.27 or more, so no state sits within 1e-9 of the bar by chance; `nearest_to_bar` records how close any comes.)

  python r225_float_guard.py     -> evidence/superactivation/r225_witness/float_guard.json
"""
from __future__ import annotations

import decimal
import json
import math
import pathlib
import sys
from decimal import Decimal

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ad_witness as A  # noqa: E402
import ion_betting as B  # noqa: E402
import r225_mixture_oc as M  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "r225_witness" / "float_guard.json"
U = 2.0 ** -53
CTX = decimal.Context(prec=60)


def derived_bound(comps: list, n_max: int) -> float:
    """the module docstring's bound on |log K float - log K exact|, from the constants"""
    xs, ys = [], []
    for c in comps:
        for k in [None] + list(A.CATS):
            y = c["bet"] * (c["m"] - (c["kappa"][k] if k else 0.0))
            ys.append(abs(y) / (1 + y))
            xs.append(abs(math.log1p(y)))
    xmax, rmax = max(xs), max(ys)
    per_component = n_max * (4 * U * max(xmax, rmax) + 6 * U * xmax)
    top = n_max * xmax
    return per_component + 8 * U * (top + 3)


def exact_log_k(n: int, counts: dict, comps: list) -> Decimal:
    """log K in 60-digit Decimal from the EXACT values of the doubles (Decimal(float) is exact)"""
    k = sum(counts.values())
    total = Decimal(0)
    for c in comps:
        b, m = Decimal(c["bet"]), Decimal(c["m"])
        lw = (n - k) * CTX.ln(1 + b * m)
        for cat, kc in counts.items():
            lw += kc * CTX.ln(1 + b * (m - Decimal(c["kappa"][cat])))
        total += CTX.exp(lw)
    return CTX.ln(total / len(comps))


def float_log_k(n: int, counts: dict, comps: list, single: bool = False) -> float:
    """the run's float log K; single=True is the 'float32' MUTANT (every operation in single precision)"""
    if not single:
        return B.log_mean_exp(B.ad_log_wealth(n, counts, comps))
    f = np.float32
    k = sum(counts.values())
    lw = [f(n - k) * np.log1p(f(c["bet"]) * f(c["m"])) + sum(
        f(kc) * np.log1p(f(c["bet"]) * (f(c["m"]) - f(c["kappa"][cat]))) for cat, kc in counts.items()) for c in comps]
    top = max(lw)
    return float(top + np.log(sum(np.exp(x - top) for x in lw) / f(len(lw))))


def boundary_states(comps: list, delta: float, looks: list, per_look: int, rng, shift: float = 0.0) -> list:
    """(n, counts) pairs where the float decision flips: one category's count bisected, the others random"""
    bar = math.log(1 / delta) + B.FLOAT_GUARD
    cats = list(A.CATS)
    out = []
    for n in looks:
        for _ in range(per_look):
            free = cats[int(rng.integers(len(cats)))]
            counts = {c: int(rng.integers(0, max(1, n // 40))) for c in cats if c != free}
            claims = lambda kf: float_log_k(n, dict(counts, **{free: kf}), comps) + shift >= bar  # noqa: E731
            lo, hi = 0, n - sum(counts.values())
            if not claims(lo) or claims(hi):
                continue
            while hi - lo > 1:                      # claims(lo) and not claims(hi): monotone in every count
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if claims(mid) else (lo, mid)
            out += [(n, dict(counts, **{free: lo})), (n, dict(counts, **{free: hi}))]
    return out


def drill(comps: list, delta: float, looks: list, n_random: int = 4000, per_look: int = 60, seed: int = 225,
          shift: float = 0.0, single: bool = False) -> dict:
    """random and boundary states in Decimal: the largest |float - exact|, every float claim that is not an exact
    claim, and the nearest any state comes to the bar. shift / single are the MUTANTS"""
    rng = np.random.default_rng(seed)
    bar_f = math.log(1 / delta) + B.FLOAT_GUARD
    bar_x = CTX.ln(1 / Decimal(delta))
    cats = list(A.CATS)
    states = boundary_states(comps, delta, looks, per_look, rng, shift)
    n_boundary = len(states)
    for _ in range(n_random):
        n = int(rng.choice(looks))
        counts = {c: int(rng.integers(0, max(1, n // 25))) for c in cats}
        if sum(counts.values()) <= n:
            states.append((n, counts))
    worst, near, bad = 0.0, math.inf, []
    for n, counts in states:
        f = float_log_k(n, counts, comps, single)
        x = exact_log_k(n, counts, comps)
        worst = max(worst, abs(float(Decimal(f) - x)))
        near = min(near, abs(float(x - bar_x)))
        if f + shift >= bar_f and x < bar_x:
            bad.append(dict(n=n, counts=counts, float_log_k=f, exact_minus_bar=float(x - bar_x)))
    return dict(states=len(states), boundary_states=n_boundary, worst_abs_error=worst, nearest_to_bar=near,
                float_claims_not_exact=len(bad), examples=bad[:3], shift=shift, single=single)


def run() -> dict:
    design, seq = M.package()
    comps = B.ad_components(design, seq)
    looks = list(seq["looks"])
    delta = float(seq["claim_delta"])
    bound = derived_bound(comps, looks[-1])
    out = dict(schema="oneq.r225_float_guard/1", rule=seq["rule"], float_guard=B.FLOAT_GUARD,
               derived_bound=bound, guard_over_bound=B.FLOAT_GUARD / bound, claim_delta=delta, looks=looks)
    out["drill"] = drill(comps, delta, looks)
    out["mutant_float32"] = drill(comps, delta, looks, n_random=400, per_look=6, seed=226, single=True)
    out["mutant_bias"] = drill(comps, delta, looks, n_random=0, per_look=20, seed=227, shift=0.1)
    out["verdict"] = ("PASS" if (bound * 10 < B.FLOAT_GUARD and out["drill"]["worst_abs_error"] <= bound
                                 and out["drill"]["float_claims_not_exact"] == 0
                                 and out["mutant_float32"]["worst_abs_error"] > bound
                                 and out["mutant_bias"]["float_claims_not_exact"] > 0) else "FAIL")
    return out


def main() -> int:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    d, m32, mb = out["drill"], out["mutant_float32"], out["mutant_bias"]
    print(f"bound {out['derived_bound']:.2e} (guard {out['float_guard']:.0e}, x{out['guard_over_bound']:.1f}); "
          f"{d['states']} states ({d['boundary_states']} on the boundary): worst |float - exact| "
          f"{d['worst_abs_error']:.2e}, nearest to the bar {d['nearest_to_bar']:.2e}, float claims not exact "
          f"{d['float_claims_not_exact']}; mutants: float32 error {m32['worst_abs_error']:.2e}, bias caught on "
          f"{mb['float_claims_not_exact']} of {mb['states']} -> {out['verdict']}")
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
