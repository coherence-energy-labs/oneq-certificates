"""Certified decisions from hardware counts: the score function that graded the prediction grades the run.

A protocol registers UNITS -- one circuit (or one parameter binding) each, with its weight in the estimand, its
score range and its shot count -- and supplies score(bits) per unit. The estimand is
    F = sum_u w_u mu_u,        mu_u = E[score_u] on the device,     sum_u w_u = 1.

Bounds (all one-sided lower confidence bounds at level 1 - alpha, distribution-free):
  WEIGHTED HOEFFDING -- the design is ENUMERATED (every unit of the estimand is run, weights exact):
      F_hat = sum_u w_u (S_u / n_u),   P(F_hat - F >= t) <= exp(-2 t^2 / sum_u w_u^2 (hi_u - lo_u)^2 / n_u)
      (Hoeffding for independent bounded shots; equal weights per shot recover the textbook 1/N form.)
  TWO-STAGE HOEFFDING -- units are SAMPLED iid from the design law within strata (weights W_b): each sampled
      unit's mean shot score is one bounded draw with mean F_b, so the bound runs over UNIT means:
      halfwidth = sqrt(ln(1/alpha) sum_b W_b^2 (hi_b - lo_b)^2 / C_b / 2)
      Shot noise is inside each unit mean, so this is valid for any shots per unit -- and it is the one to use
      whenever the run's circuits are a SAMPLE of the design, because then the circuit-to-circuit spread of the
      true means is part of the error. Treating sampled units as if enumerated is the error this module refuses
      to make silently: `lower_bound` takes the design kind explicitly.
  EMPIRICAL BERNSTEIN (Maurer & Pontil 2009, Thm 4) over iid unit means, reported beside two-stage Hoeffding.
Controls: `consistent_with` (two-sided Hoeffding about a value) and `not_certified_above` (a control that must NOT
clear the bar). Completeness: every registered unit present with exactly its registered shots, every key a valid
bitstring of the unit's width; anything else raises IncompleteRunError before any number is computed.

Bit order: qiskit count keys print classical bit 0 RIGHTMOST; bits_of returns them in clbit order (bit 0 first),
the order every score function reads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


class IncompleteRunError(ValueError):
    """a registered unit is missing, has the wrong shot count, or carries malformed outcome keys"""


@dataclass(frozen=True)
class Unit:
    name: str
    weight: float                   # weight in the estimand (enumerated) or 1/C_b within stratum b (sampled)
    shots: int
    n_bits: int
    lo: float = 0.0                 # score range: every score(bits) must lie in [lo, hi]
    hi: float = 1.0
    stratum: str = "all"


@dataclass(frozen=True)
class Tally:
    n: int
    s1: float                       # sum of scores
    s2: float                       # sum of squared scores

    @property
    def mean(self) -> float:
        return self.s1 / self.n


@dataclass(frozen=True)
class Bound:
    estimate: float
    halfwidth: float
    alpha: float
    method: str

    @property
    def lower(self) -> float:
        return self.estimate - self.halfwidth

    @property
    def upper(self) -> float:
        return self.estimate + self.halfwidth


@dataclass
class Decision:
    claim: str
    bound: Bound
    bar: float
    passed: bool
    detail: dict = field(default_factory=dict)


def bits_of(key: str, n_bits: int) -> tuple:
    """a qiskit count key -> bits in clbit order (bit 0 first)"""
    k = key.replace(" ", "")
    if k.startswith("0x"):
        v = int(k, 16)
        if v >> n_bits:
            raise IncompleteRunError(f"outcome {key!r} does not fit in {n_bits} bits")
        return tuple((v >> j) & 1 for j in range(n_bits))
    if len(k) != n_bits or set(k) - {"0", "1"}:
        raise IncompleteRunError(f"outcome key {key!r} is not a {n_bits}-bit string")
    return tuple(int(c) for c in reversed(k))


def tally(counts: dict, score, unit: Unit) -> Tally:
    n = 0
    s1 = s2 = 0.0
    for key, c in counts.items():
        c = int(c)
        if c < 0:
            raise IncompleteRunError(f"{unit.name}: negative count for {key!r}")
        v = float(score(bits_of(key, unit.n_bits)))
        if not (unit.lo - 1e-12 <= v <= unit.hi + 1e-12):
            raise IncompleteRunError(f"{unit.name}: score {v} outside its declared range [{unit.lo}, {unit.hi}]")
        n += c
        s1 += c * v
        s2 += c * v * v
    return Tally(n, s1, s2)


def tally_all(units: list, counts_by_name: dict, scores: dict) -> dict:
    """completeness first, then tallies: every registered unit, exactly its shots"""
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        raise IncompleteRunError("duplicate unit names: results could be looked up by the wrong circuit")
    missing = [n for n in names if n not in counts_by_name]
    if missing:
        raise IncompleteRunError(f"{len(missing)} registered units have no counts, e.g. {missing[:3]}")
    extra = [n for n in counts_by_name if n not in set(names)]
    if extra:
        raise IncompleteRunError(f"counts for unregistered units, e.g. {extra[:3]}")
    out = {}
    for u in units:
        t = tally(counts_by_name[u.name], scores[u.name], u)
        if t.n != u.shots:
            raise IncompleteRunError(f"{u.name}: {t.n} shots recorded, {u.shots} registered")
        out[u.name] = t
    return out


# ------------------------------------------------------------------ bounds
def hoeffding_enumerated(units: list, tallies: dict, alpha: float) -> Bound:
    w = sum(u.weight for u in units)
    if abs(w - 1) > 1e-9:
        raise ValueError(f"enumerated weights must sum to 1, got {w}")
    est = sum(u.weight * tallies[u.name].mean for u in units)
    var_proxy = sum((u.weight * (u.hi - u.lo)) ** 2 / tallies[u.name].n for u in units)
    return Bound(est, math.sqrt(math.log(1 / alpha) * var_proxy / 2), alpha, "hoeffding-enumerated")


def _strata(units: list, tallies: dict) -> dict:
    by: dict = {}
    for u in units:
        by.setdefault(u.stratum, []).append((u, tallies[u.name].mean))
    return by


def hoeffding_two_stage(units: list, tallies: dict, stratum_weight: dict, alpha: float) -> Bound:
    by = _strata(units, tallies)
    if set(by) != set(stratum_weight) or abs(sum(stratum_weight.values()) - 1) > 1e-9:
        raise ValueError(f"strata {sorted(by)} must match weights {sorted(stratum_weight)} that sum to 1")
    est = 0.0
    var_proxy = 0.0
    for b, rows in by.items():
        C = len(rows)
        lo, hi = min(u.lo for u, _ in rows), max(u.hi for u, _ in rows)
        est += stratum_weight[b] * sum(mu for _, mu in rows) / C
        var_proxy += (stratum_weight[b] * (hi - lo)) ** 2 / C
    return Bound(est, math.sqrt(math.log(1 / alpha) * var_proxy / 2), alpha, "hoeffding-two-stage")


def empirical_bernstein(values: list, lo: float, hi: float, alpha: float) -> Bound:
    """Maurer-Pontil: for iid X_i in [lo, hi], with prob >= 1 - alpha,
    E X >= mean - sqrt(2 V ln(2/alpha) / n) - 7 (hi - lo) ln(2/alpha) / (3 (n - 1)), V the unbiased sample variance"""
    n = len(values)
    if n < 2:
        raise ValueError("empirical Bernstein needs at least two draws")
    mean = sum(values) / n
    V = sum((x - mean) ** 2 for x in values) / (n - 1)
    L = math.log(2 / alpha)
    return Bound(mean, math.sqrt(2 * V * L / n) + 7 * (hi - lo) * L / (3 * (n - 1)), alpha, "empirical-bernstein")


def lower_bound(design: str, units: list, tallies: dict, alpha: float, stratum_weight: dict | None = None) -> Bound:
    """design = 'enumerated' (every unit of the estimand ran, exact weights) or 'sampled' (units drawn from the
    design law within strata). There is deliberately no default."""
    if design == "enumerated":
        return hoeffding_enumerated(units, tallies, alpha)
    if design == "sampled":
        if stratum_weight is None:
            raise ValueError("a sampled design needs its stratum weights")
        return hoeffding_two_stage(units, tallies, stratum_weight, alpha)
    raise ValueError(f"design must be 'enumerated' or 'sampled', not {design!r}")


def decide(claim: str, bound: Bound, bar: float) -> Decision:
    """PASS iff the one-sided lower bound clears the bar strictly"""
    return Decision(claim, bound, bar, bound.lower > bar)


def consistent_with(bound: Bound, value: float) -> bool:
    """two-sided: the value lies inside estimate +- halfwidth (pass a Bound computed at alpha/2 per side)"""
    return abs(bound.estimate - value) <= bound.halfwidth


def not_certified_above(bound: Bound, bar: float) -> bool:
    return not bound.lower > bar


def required_halfwidth_shots(range_: float, alpha: float, halfwidth: float) -> int:
    """shots N for which a single-stratum, equal-weight Hoeffding halfwidth reaches `halfwidth`"""
    return math.ceil(math.log(1 / alpha) * range_ ** 2 / (2 * halfwidth ** 2))
