"""R177: independent rational non-2-extendibility witness for a logical qubit channel.

A Bell-diagonal two-qubit Choi state with lambda=(I,X,Y,Z) is symmetrically
extendible exactly when sum(lambda^2)-4 sqrt(prod lambda) <= 1/2.
The criterion is an imported theorem, not an ONE_Q discovery.

This module gives:
  * exact rational verification from declared Bell populations;
  * an interval-certified sufficient NOT-2-extendible result from six-state
    survival probability intervals;
  * a finite-shot IID Bernoulli Hoeffding *simultaneous* interval factory,
    separate from any weighted 0..2 hardware estimator;
  * an explicit REFUSAL/INCONCLUSIVE state for weak or unsupported evidence.

A positive certificate shows that the PAULI-TWIRLED Choi is not 2-extendible;
this is sufficient to exclude 2-extendibility of the pre-twirled Choi.
It does NOT prove nonzero quantum capacity, remote entanglement, or the
superactivation of two physical channels without separate channel witnesses.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any

AXES = ("X", "Y", "Z")
SIGNS = ("+", "-")
PAULIS = ("I", "X", "Y", "Z")
CERT_TYPE = "oneq.quantum_witness.not_two_extendible/1"


class WitnessRefused(ValueError):
    """The instrument is being asked to certify data it cannot interpret."""


def rat(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise WitnessRefused("booleans are not probabilities")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as e:
            raise WitnessRefused(f"not a rational: {value!r}") from e
    raise WitnessRefused("rational witnesses accept integer, fraction or decimal text; floats not accepted")


def _interval(pair) -> tuple[Fraction, Fraction]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise WitnessRefused("expected [lower, upper] interval")
    a, b = rat(pair[0]), rat(pair[1])
    if not 0 <= a <= b <= 1:
        raise WitnessRefused("confidence interval must obey 0 <= lower <= upper <= 1")
    return a, b


def probabilities_to_t(intervals) -> dict[str, tuple[Fraction, Fraction]]:
    if not isinstance(intervals, dict) or set(intervals) != set(AXES):
        raise WitnessRefused("six-state record needs exactly X, Y, Z axes")
    out = {}
    for b in AXES:
        v = intervals[b]
        if not isinstance(v, dict) or set(v) != set(SIGNS):
            raise WitnessRefused(f"axis {b} needs exactly + and - settings")
        a, c = _interval(v["+"]), _interval(v["-"])
        # Pauli transfer diagonal t_b = P(correct|+b) + P(correct|-b) - 1.
        out[b] = a[0] + c[0] - 1, a[1] + c[1] - 1
    return out


# signs multiplying (tx, ty, tz) for lambda_I,X,Y,Z
_PARITY = {
    "I": (+1, +1, +1),
    "X": (+1, -1, -1),
    "Y": (-1, +1, -1),
    "Z": (-1, -1, +1),
}


def _bell_bounds(t: dict, signs) -> tuple[Fraction, Fraction]:
    low, high = Fraction(1), Fraction(1)
    for b, s in zip(AXES, signs):
        lo, hi = t[b]
        low += s * (lo if s > 0 else hi)
        high += s * (hi if s > 0 else lo)
    return low / 4, high / 4


def bell_bounds(t: dict) -> dict:
    return {name: _bell_bounds(t, signs) for name, signs in _PARITY.items()}


def exact_margin_sign(lambdas) -> dict:
    """Exact sign proof without numerical square roots: m>0 iff A>0 and A²>16 Πλ.

    Here A=sum lambda² - 1/2, m=A-4 sqrt(prod lambda). The Choi
    populations must be a valid probability simplex. This is not a
    finite-data statistical certificate.
    """
    if not isinstance(lambdas, dict) or set(lambdas) != set(PAULIS):
        raise WitnessRefused("exact witness requires four Pauli populations")
    p = {k: rat(lambdas[k]) for k in PAULIS}
    if any(x < 0 for x in p.values()) or sum(p.values()) != 1:
        raise WitnessRefused("Pauli populations must be nonnegative and sum exactly to 1")
    a = sum((x * x for x in p.values()), Fraction()) - Fraction(1, 2)
    rhs = 16 * math.prod(p.values())
    return dict(
        not_two_extendible=(a > 0 and a * a > rhs),
        A=str(a), A_squared_minus_16_prod=str(a * a - rhs),
        lambdas={k: str(v) for k, v in p.items()},
        mathematical_scope="exact Bell-diagonal Choi populations; no finite-shot uncertainty",
    )


def _box_certificate(t: dict) -> dict:
    """A conservative interval bound for every physically allowed point of t."""
    bounds = bell_bounds(t)
    if any(high < 0 or low > 1 for low, high in bounds.values()):
        return dict(vacuous=True, proved=False, reason="Pauli triangle excludes this entire box")
    clip = {k: (max(Fraction(0), v[0]), min(Fraction(1), v[1]))
            for k, v in bounds.items()}
    a_lower = sum((v[0] * v[0] for v in clip.values()), Fraction()) - Fraction(1, 2)
    product_upper = 16 * math.prod(v[1] for v in clip.values())
    proved = a_lower > 0 and a_lower * a_lower > product_upper
    return dict(vacuous=False, proved=proved, a_lower=a_lower,
                surplus=a_lower * a_lower - product_upper, lambda_bounds=clip)


def interval_certificate(intervals: dict, *, max_boxes: int = 2048) -> dict:
    """Sufficient proof valid for ALL physical channels within 6 declared intervals.

    Splitting improves tightness of interval enclosure; failing to prove is
    INCONCLUSIVE, never a positive or negative physical theorem.
    The intervals must have a separate sampling model and auditable origin.
    """
    if type(max_boxes) is not int or not 1 <= max_boxes <= 65536:
        raise WitnessRefused("max_boxes must be positive and bounded")
    t = probabilities_to_t(intervals)
    todo = [t]
    visited = 0
    accepted = 0
    tightest = None
    while todo:
        box = todo.pop()
        visited += 1
        val = _box_certificate(box)
        if val["vacuous"]:
            continue
        if val["proved"]:
            accepted += 1
            tightest = val["surplus"] if tightest is None else min(tightest, val["surplus"])
            continue
        if visited + len(todo) + 2 > max_boxes:
            return dict(schema=CERT_TYPE, verdict="INCONCLUSIVE",
                        boxes_checked=visited, boxes_proved=accepted,
                        rationale="not enough interval resolution; cannot promote to measured witness",
                        confidence_intervals=intervals)
        b = max(AXES, key=lambda x: box[x][1] - box[x][0])
        lo, hi = box[b]
        if lo == hi:
            return dict(schema=CERT_TYPE, verdict="INCONCLUSIVE",
                        boxes_checked=visited, boxes_proved=accepted,
                        rationale="exact point fails to establish non-2-extendibility",
                        confidence_intervals=intervals)
        mid = (lo + hi) / 2
        left, right = dict(box), dict(box)
        left[b], right[b] = (lo, mid), (mid, hi)
        todo.extend((left, right))
    if accepted == 0:
        return dict(schema=CERT_TYPE, verdict="INCONCLUSIVE",
                    boxes_checked=visited, boxes_proved=0,
                    rationale="no physically allowed point inside declared intervals")
    return dict(schema=CERT_TYPE, verdict="PROVED_WITHIN_DECLARED_INTERVALS",
                boxes_checked=visited, boxes_proved=accepted,
                min_strict_surplus=str(tightest),
                lambda_scope="two-qubit Choi after correlated Pauli twirl",
                confidence_intervals=intervals,
                statistical_scope="conditional on externally verified interval coverage",
                capacity_claim=False, remote_link_claim=False)


def _upward_eps(n: int, delta: float) -> Fraction:
    val = math.sqrt(math.log(12 / delta) / (2 * n))
    # Rounded outward in units 1e-12, plus one tick for strict coverage.
    return Fraction(math.ceil(val * 10 ** 12) + 1, 10 ** 12)


def simultaneous_hoeffding(counts: dict, *, alpha: float = 0.01,
                             score_contract: str | None = None) -> dict:
    """Six IID Bernoulli survival experiments, each separately counted.

    Union bound over six two-sided intervals: joint coverage >= 1-alpha.
    DOES NOT apply to weighted R145/R147 per-shot scores with range [0,2],
    correlated samples, selected subsets, or adaptive post-processing.
    """
    if score_contract != "raw_unweighted_iid_bernoulli":
        raise WitnessRefused("six-state Hoeffding requires an explicit raw_unweighted_iid_bernoulli score contract, "
                             "not R145/R147 weighted [0,2] estimators")
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise WitnessRefused("alpha must be between 0 and 1")
    if not isinstance(counts, dict) or set(counts) != set(AXES):
        raise WitnessRefused("counts require exactly X, Y, Z")
    intervals = {}
    for b in AXES:
        if not isinstance(counts[b], dict) or set(counts[b]) != set(SIGNS):
            raise WitnessRefused(f"counts for {b} need + and -")
        intervals[b] = {}
        for sign in SIGNS:
            obj = counts[b][sign]
            if not isinstance(obj, dict) or set(obj) != {"correct", "shots"}:
                raise WitnessRefused("each setting needs correct, shots")
            k, n = obj["correct"], obj["shots"]
            if type(k) is not int or type(n) is not int or n < 1 or not 0 <= k <= n:
                raise WitnessRefused("successes/shots must be bounded integers")
            eps = _upward_eps(n, float(alpha))
            p = Fraction(k, n)
            intervals[b][sign] = [str(max(Fraction(), p - eps)), str(min(Fraction(1), p + eps))]
    return dict(model="IID six-state Bernoulli (six fixed settings), no cherry-picked shots",
                score_contract="raw_unweighted_iid_bernoulli",
                simultaneous_coverage_at_least=1 - alpha, alpha=alpha,
                intervals=intervals)
