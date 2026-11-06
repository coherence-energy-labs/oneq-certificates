r"""R211: the producer's double-precision claim decision is the exact-real decision the Lean theorem is about, on EVERY
reachable state of the frozen package (the adversary's F7).

    python r203_float_boundary.py        # -> evidence/superactivation/r203_betting/float_boundary.json (~1 min)

THE TWO DECISIONS. At a look of n shots with k_rare rare-group and k_common common-group errors:
  - the producer (`ion_betting.verdict`) claims iff log_mean_exp(log_wealth(n, k_rare, k_common)) >= log(1/delta),
    in doubles;
  - the theorem (`Statement.frozen_run_valid`) is about the exact real
    K = (1/3) sum_b (1 + b m)^(n-k_r-k_c) (1 + b (m - s_r))^k_r (1 + b (m - s_c))^k_c >= 1/delta,
    with the same doubles read as exact rationals (`wealth_counts` is the product-to-counts step).

THE DRILL covers every (n, k_rare, k_common) with k_rare + k_common <= n at every look: a superset of the states any
label draw can reach. For each look n and each k_rare:
  1. bisect k_common on the PRODUCER's own decision for its boundary kc_d (its last claiming k_common, -1 if none),
     keeping decide(lo) = claim and decide(hi) = no claim;
  2. certify the REAL decision at kc_d (claims) and at kc_d + 1 (does not), wherever they are in range. This uses
     60-digit Decimal arithmetic, whose ln and exp are correctly rounded; a sign counts as certain when |g| > 1e-40,
     and exact Fractions are the fallback;
  3. cover every other k_common with the SLOPE LEMMA. One more common error multiplies every bet's wealth by
     r_b = (1 + b (m - s_c)) / (1 + b m) <= r_max < 1 (checked exactly), so g = log K - log(1/delta) falls by at
     least gap = -log r_max per step. The real decision therefore claims for every k_common <= kc_d and fails for
     every k_common > kc_d.
     The producer agrees wherever its absolute error in g is below gap:
     - DERIVED bound EPS: |error| <= 1e-9. log_wealth is three integer multiples (<= 8,000) of log1p values, each
       within a few ulps, summed; log_mean_exp and log(1/delta) add a few ulps of quantities below 1e4. So the
       error is about 1e4 * 1e-16 * small constants, about 1e-11.
     - MEASURED: the largest |g_double - g_exact| over every evaluated point and a seeded sample of 20,000 states.
       It must sit below EPS / 100.
Agreement everywhere then follows from two facts: EPS < gap, and the boundary certified on both sides.
"""
from __future__ import annotations

import decimal
import json
import math
import pathlib
import random
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "evidence" / "superactivation" / "ion_run_m8_r224"   # R224 re-freeze (both erasure orders)
OUT = ROOT / "evidence" / "superactivation" / "r203_betting" / "float_boundary.json"
EPS = 1e-9            # DERIVED bound on the producer's absolute error in g (module docstring)
CERTAIN = decimal.Decimal("1e-40")
SAMPLE = 20_000
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import ion_betting as B  # noqa: E402
from roster_digest import roster_digest  # noqa: E402

CTX = decimal.Context(prec=60)


class Exact:
    """the theorem's statistic for the package, as g = log K - log(1/delta) in 60-digit Decimal, Fractions behind"""

    def __init__(self, design: dict, seq: dict):
        sc = B.scores(design, seq["pi_rare"])
        self.m, self.sr, self.sc = (Fraction(x) for x in (B.threshold(design)["m"], sc[B.RARE], sc[B.COMMON]))
        self.bets = [Fraction(b) for b in B.check_bets(design, seq["pi_rare"], seq["bets"])]
        self.delta = Fraction(seq["claim_delta"])
        self.f0 = [1 + b * self.m for b in self.bets]
        self.fr = [1 + b * (self.m - self.sr) for b in self.bets]
        self.fc = [1 + b * (self.m - self.sc) for b in self.bets]
        d = self._dec
        self.L = [(CTX.ln(d(a)), CTX.ln(d(r)), CTX.ln(d(c))) for a, r, c in zip(self.f0, self.fr, self.fc)]
        self.bar = CTX.ln(d(1 / self.delta))
        self.r_max = max(c / a for a, c in zip(self.f0, self.fc))

    @staticmethod
    def _dec(f: Fraction) -> decimal.Decimal:
        return CTX.divide(decimal.Decimal(f.numerator), decimal.Decimal(f.denominator))

    def g(self, n: int, kr: int, kc: int) -> decimal.Decimal:
        lw = [CTX.add(CTX.add(CTX.multiply(n - kr - kc, a), CTX.multiply(kr, r)), CTX.multiply(kc, c))
              for a, r, c in self.L]
        top = max(lw)
        s = sum((CTX.exp(CTX.subtract(x, top)) for x in lw), decimal.Decimal(0))
        return CTX.subtract(CTX.add(top, CTX.ln(CTX.divide(s, len(lw)))), self.bar)

    def claims(self, n: int, kr: int, kc: int) -> tuple[bool, decimal.Decimal, bool]:
        """(the real decision, g, whether the exact Fraction fallback decided it)"""
        g = self.g(n, kr, kc)
        if abs(g) > CERTAIN:
            return g >= 0, g, False
        k = sum(a ** (n - kr - kc) * r ** kr * c ** kc for a, r, c in zip(self.f0, self.fr, self.fc)) / len(self.bets)
        return k >= 1 / self.delta, g, True


def producer(design: dict, seq: dict):
    """the producer's own decision and g, as ion_betting.verdict computes them"""
    pi, bets, delta = float(seq["pi_rare"]), seq["bets"], float(seq["claim_delta"])
    bar = math.log(1 / delta)

    def decide(n: int, kr: int, kc: int) -> tuple[bool, float]:
        lk = B.log_mean_exp(B.log_wealth(n, kr, kc, design, pi, bets))
        return lk >= bar, lk - bar
    return decide


def drill(design: dict, seq: dict, looks=None, decide=None, sample: int = SAMPLE) -> dict:
    ex = Exact(design, seq)
    decide = decide or producer(design, seq)
    gap = -math.log(float(ex.r_max))
    if not (ex.r_max < 1 and EPS < gap):
        raise ValueError(f"the slope lemma does not cover the error bound: r_max {float(ex.r_max)}, gap {gap}")
    max_err, fallbacks, disagree, rows = 0.0, 0, [], []

    def err(n, kr, kc, g_dbl):
        nonlocal max_err
        max_err = max(max_err, abs(g_dbl - float(ex.g(n, kr, kc))))

    for n in (looks or seq["looks"]):
        certified, margin = 0, None
        for kr in range(n + 1):
            top = n - kr
            d0, g0 = decide(n, kr, 0)
            err(n, kr, 0, g0)
            if not d0:
                kc_d = -1
            else:
                dt, gt = decide(n, kr, top)
                err(n, kr, top, gt)
                if dt:
                    kc_d = top
                else:
                    lo, hi = 0, top
                    while hi - lo > 1:
                        mid = (lo + hi) // 2
                        dm, gm = decide(n, kr, mid)
                        err(n, kr, mid, gm)
                        lo, hi = (mid, hi) if dm else (lo, mid)
                    kc_d = lo
            for kc, want in ((kc_d, True), (kc_d + 1, False)):
                if 0 <= kc <= top:
                    real, g, fb = ex.claims(n, kr, kc)
                    fallbacks += fb
                    certified += 1
                    margin = abs(float(g)) if margin is None else min(margin, abs(float(g)))
                    if real != want:
                        disagree.append(dict(look=n, k_rare=kr, k_common=kc, producer=want, real=real, g=str(g)))
        rows.append(dict(look=n, k_rare_values=n + 1, states=(n + 1) * (n + 2) // 2, boundaries_certified=certified,
                         min_boundary_margin=margin))
    rng = random.Random(211)
    for _ in range(sample):
        n = rng.choice(list(looks or seq["looks"]))
        kr = rng.randint(0, n)
        kc = rng.randint(0, n - kr)
        err(n, kr, kc, decide(n, kr, kc)[1])
    return dict(schema="oneq.r203_float_boundary/1", looks=rows, states=sum(r["states"] for r in rows),
                r_max=[ex.r_max.numerator, ex.r_max.denominator], gap=gap, eps_derived=EPS,
                max_measured_error=max_err, fraction_fallbacks=fallbacks, disagreements=disagree,
                agrees_everywhere=(not disagree and max_err < EPS / 100))


def run() -> dict:
    design = json.loads((PACKAGE / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((PACKAGE / "sequential.json").read_text(encoding="utf-8"))
    out = drill(design, seq)
    out["package"] = {p: roster_digest(PACKAGE / p) for p in ("design.json", "sequential.json")}
    out["producer"] = {p: roster_digest(HERE / p) for p in ("ion_betting.py",)}
    return out


def main() -> int:
    out = run()
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{out['states']:,} states over {len(out['looks'])} looks; disagreements {len(out['disagreements'])}; "
          f"max measured error {out['max_measured_error']:.2e} (bound {EPS:g}, gap {out['gap']:.3f}); "
          f"Fraction fallbacks {out['fraction_fallbacks']} -> {OUT}")
    return 0 if out["agrees_everywhere"] else 1


if __name__ == "__main__":
    sys.exit(main())
