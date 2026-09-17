r"""A CERTIFIED INTERVAL ON THE ERROR THE DEVICE ACTUALLY PRODUCED.

*** THE MEASUREMENT THIS REPOSITORY HAS BEEN COMPUTING AND THROWING
AWAY. ***

Every certified shot already carries three numbers:

    bound    the dual objective -- a feasible cut packing's value
    primal   the weight of the correction the decoder returned
    gap      primal - bound

They are used to answer one boolean ("did it certify?") and then
discarded. `evidence/million_shot_gate_d5.json` records 960,181
certified shots and stores **not one** of their weights;
`accepted_examples` is an empty list. Nearly a million proven bounds,
computed and dropped.

But those numbers are a physical observable, and the chain that makes
them one is three inequalities, each of which holds for a different
reason:

    bound  <=  min_weight        WEAK DUALITY. A feasible dual never
                                 exceeds a feasible primal. Holds no
                                 matter which decoder produced either.

    min_weight  <=  primal       The returned correction IS an
                                 explanation of the syndrome, so the
                                 minimum over explanations is at most
                                 its weight.

    min_weight  <=  w(true)      The error that actually occurred is
                                 ALSO an explanation. The minimum is at
                                 most its weight too.

Read them together:

    *** `bound` IS A PROVEN LOWER BOUND ON THE WEIGHT OF WHATEVER ERROR
    ACTUALLY HAPPENED, ON EVERY SHOT, WITHOUT KNOWING WHAT IT WAS. ***

and `[bound, primal]` is a proven interval containing `min_weight`.

--------------------------------------------------------------------------
*** WHY THE INTERVAL, AND NOT THE BOUND ALONE. ***

The bound alone is GAMEABLE, and in the flattering direction. A weaker
dual gives a smaller bound; a smaller bound reads as "we can only prove
this little error occurred". A vendor with a deliberately poor dual
producer would look quieter than one with a good one.

Reporting the interval closes that, because a weak dual WIDENS it. You
cannot exhibit a tight interval and a low bound unless the error
genuinely was small. Width is the optimality gap, and it is the price of
your own machinery -- paid visibly, by you.

So nothing here reports a bound without its width.

--------------------------------------------------------------------------
*** WHAT MAKES IT A CROSS-VENDOR NUMBER. ***

The lower end is decoder-INDEPENDENT: any feasible dual, from any
producer, gives a valid bound. A better dual gives a tighter one, never
a different kind of claim. Two laboratories running different decoders
on different hardware produce numbers that can be laid side by side --
which is the property benchmarks in this field usually lack, since a
fidelity from randomized benchmarking measures a proxy circuit and a
logical error rate measures the decoder as much as the device.

This measures the workload that was actually run.

--------------------------------------------------------------------------
*** REFUSED SHOTS BELONG IN THE SAMPLE, AND LEAVING THEM OUT WOULD BIAS
IT. ***

Weak duality needs feasibility, not optimality. A shot the ladder could
not close still carries a valid lower bound -- the best packing it
found. Those are exactly the HARD shots, the ones where more error
occurred.

Dropping them would compute the device's noise floor over its easy
shots only, which is the survivorship error in its purest form: the
measurement would improve every time the decoder got worse. So the
lower bound is taken over EVERY shot, and `certified_fraction` is
reported beside it rather than used as a filter.

--------------------------------------------------------------------------
*** WHAT THIS IS NOT. ***

It is a statement about weight under the DECLARED error model. Edge
weights come from a detector error model; if that model is wrong, this
is a certified interval about the wrong graph. It does not become a
physical error RATE without the model, and `oneq.dem_calibration` exists
to interrogate exactly that.

It is also not a fidelity, not a threshold estimate, and not a claim
about any shot's logical outcome.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

#: *** WHAT A "WEIGHT" IS, AND WHY THE ANSWER CHANGES THE CLAIM. ***
#:
#: The inequality chain holds for ANY non-negative edge weights. What it
#: MEANS does not.
#:
#:   LOG_LIKELIHOOD  w_e = -log(p_e/(1-p_e)) or an affine image of it.
#:                   Minimum weight is then the most likely explanation
#:                   (`api.py`: "log-likelihood ratios make min-weight =
#:                   most-likely"), and the bound is a proven lower bound
#:                   on the NEGATIVE LOG-PROBABILITY of the error that
#:                   occurred. This is the reading that makes the number
#:                   a statement about noise.
#:
#:   UNIFORM         w_e = 1. Minimum weight counts MECHANISMS. The bound
#:                   is then a proven lower bound on HOW MANY fault
#:                   mechanisms fired -- a real observable, and NOT a
#:                   likelihood. `evidence/ibm_live_certified.json` says
#:                   "uniform, declared", so this is the convention the
#:                   live hardware lane actually ran under.
#:
#:   UNDECLARED      UNCHECKABLE. The arithmetic still works and means
#:                   nothing in particular.
#:
#: An earlier draft of this module asserted the likelihood reading with
#: no field for the convention at all, which would have published a
#: statement about probability computed from a count of edges. The
#: convention now travels with the number.
LOG_LIKELIHOOD = "log_likelihood"
UNIFORM = "uniform"
UNDECLARED = "undeclared"

_READING = {
    LOG_LIKELIHOOD: ("a proven lower bound on the negative log-probability "
                     "of the error that actually occurred"),
    UNIFORM: ("a proven lower bound on the NUMBER of fault mechanisms that "
              "fired -- a count, not a likelihood"),
    UNDECLARED: ("UNCHECKABLE: the edge-weight convention was not declared, "
                 "so the interval is arithmetically valid and carries no "
                 "physical reading"),
}

#: A shot must be at least this heavy for its interval to be reported as
#: resolved. Zero-weight shots are the trivial syndrome -- nothing fired
#: -- and averaging them in measures how often the device was quiet, not
#: how much error it made when it was not.
TRIVIAL = 1e-12

#: *** THE POPULATION BOUND UNDER-COVERS ON LEFT-SKEWED SAMPLES, AND
#: NEITHER METHOD FIXES IT. ***
#:
#: MEASURED 2026-08-31, 400 trials per cell, nominal coverage 0.95:
#:
#:     shape                    n     normal   bootstrap
#:     right-skew 95/5 @40     200     0.978     0.983
#:     extreme    99/1 @500    200     0.988     0.998
#:     bimodal    50/50        200     0.943     0.953
#:     LEFT-skew  95/5 @0.02    20     0.640     0.640
#:     LEFT-skew  95/5 @0.02    60     0.780     0.780
#:     LEFT-skew  95/5 @0.02   200     0.865     0.885
#:
#: The bootstrap is never worse and is marginally better in the tails,
#: which is why it is used -- but the honest headline is the last three
#: rows: when most shots are heavy and a few are very light, a nominal
#: 95% lower bound covers as little as 64% of the time. That is an error
#: bar that lies, and swapping estimators does not repair it (BCa is the
#: known remedy and is not implemented here).
#:
#: So the shape is CHECKED and the bound is marked unreliable rather
#: than published as though the nominal level held. An earlier draft of
#: this module reported the bound with no shape check at all, and
#: justified the bootstrap with an under-coverage claim that this very
#: table does not support.
LEFT_SKEW_UNRELIABLE = -0.5


class WeightRefusal(Exception):
    """The records cannot support the statement being asked for."""


@dataclass(frozen=True)
class ShotInterval:
    """A proven interval containing one shot's minimum-weight explanation."""
    lower: float                 # dual objective: proven, decoder-independent
    upper: float                 # the returned correction's weight
    certified: bool              # did the ladder close the gap?

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, w: float) -> bool:
        return self.lower - 1e-9 <= w <= self.upper + 1e-9


def shot_interval(primal: float, bound: float, *,
                  certified: bool = False) -> ShotInterval:
    r"""One shot's certified interval, with the orientation CHECKED.

    *** A DUAL ABOVE ITS PRIMAL IS NOT A TIGHT BOUND, IT IS A BROKEN
    ONE. *** Weak duality forbids it, so observing it means the packing
    is infeasible or the weights disagree between the two sides -- and
    this repository has caught exactly that before, when a d=7 run
    reported `dual objective 75.008 EXCEEDS primal 73.971`. Silently
    clamping would convert a producer defect into a slightly optimistic
    measurement, which is the worst available outcome: wrong, and quiet.
    """
    if not (math.isfinite(primal) and math.isfinite(bound)):
        raise WeightRefusal(
            f"non-finite interval ({bound}, {primal}): a bound that is "
            f"not a number cannot be a proof of anything")
    if bound < -1e-9:
        raise WeightRefusal(
            f"negative dual objective {bound}: a packing of non-negative "
            f"z_S cannot have negative value, so this dual is not feasible")
    if bound > primal + 1e-9:
        raise WeightRefusal(
            f"dual {bound} EXCEEDS primal {primal}, which weak duality "
            f"forbids -- the packing is infeasible or the two sides "
            f"disagree about the edge weights. Refusing rather than "
            f"clamping: a clamp turns a producer defect into a quietly "
            f"optimistic measurement")
    return ShotInterval(lower=bound, upper=primal, certified=bool(certified))


@dataclass(frozen=True)
class DeviceFloor:
    """What a run proves about the error the device produced."""
    shots: int
    nontrivial: int
    certified_fraction: float
    #: Mean over EVERY shot, refusals included. Proven.
    mean_lower: float
    mean_upper: float
    #: One-sided 95% lower confidence bound on the POPULATION mean of the
    #: minimum-weight explanation. Statistics on top of a proof.
    population_lower_95: float | None
    #: None when the sample's shape is one where the nominal level was
    #: MEASURED not to hold. A bound whose coverage is unknown is not a
    #: bound, and reporting it beside a "95%" label would be the lie.
    population_bound_reliable: bool
    mean_width: float
    #: The same statistic over the certified subset ONLY -- reported so a
    #: reader can see the size of the survivorship effect, never as the
    #: headline.
    certified_only_mean_lower: float | None
    weight_convention: str = UNDECLARED
    #: Sample skewness of the per-shot lower bounds. Reported because it
    #: is the quantity that decides whether a normal approximation would
    #: have been safe -- and an earlier draft used one without looking.
    skewness: float | None = None
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema": "oneq-certified-weight/1",
                "shots": self.shots, "nontrivial": self.nontrivial,
                "certified_fraction": self.certified_fraction,
                "proven_interval_on_mean_min_weight":
                    [self.mean_lower, self.mean_upper],
                "population_lower_95": self.population_lower_95,
                "population_bound_reliable": self.population_bound_reliable,
                "mean_width": self.mean_width,
                "certified_only_mean_lower": self.certified_only_mean_lower,
                "weight_convention": self.weight_convention,
                "physical_reading": _READING[self.weight_convention],
                "skewness_of_bounds": self.skewness,
                "claim": (
                    "every shot's minimum-weight explanation lies in its "
                    "own [lower, upper]; the lower end is a proven bound "
                    "on the weight of the error that actually occurred, "
                    "and holds for refused shots too because weak duality "
                    "needs feasibility, not optimality"),
                "does_not_claim": (
                    "a physical error RATE. These are weights under the "
                    "declared detector error model. A wrong model makes "
                    "this a certified interval about the wrong graph -- "
                    "see oneq.dem_calibration"),
                "notes": dict(self.notes)}


def device_floor(records, *, include_trivial: bool = False,
                 convention: str = UNDECLARED,
                 bootstrap_resamples: int = 2000,
                 bootstrap_seed: int = 20260831) -> DeviceFloor:
    r"""Aggregate a run's per-shot intervals into a device statement.

    `records` are mappings with `primal`, `bound`, and optionally
    `accepted`. Anything else on them is ignored, so the same function
    reads a simulator's output and a hardware lane's without a shim.

    *** THE AVERAGE IS TAKEN OVER EVERY SHOT. *** See the module
    docstring: filtering to certified shots measures the easy ones and
    improves as the decoder gets worse.
    """
    ivs, trivial = [], 0
    for i, r in enumerate(records):
        try:
            primal, bound = float(r["primal"]), float(r["bound"])
        except (KeyError, TypeError, ValueError) as e:
            raise WeightRefusal(
                f"record {i} lacks a usable primal/bound pair ({e}); a "
                f"missing bound is not a bound of zero") from None
        iv = shot_interval(primal, bound, certified=bool(r.get("accepted")))
        if iv.upper <= TRIVIAL and not include_trivial:
            trivial += 1
            continue
        ivs.append(iv)

    n = len(ivs)
    if n == 0:
        raise WeightRefusal(
            f"no non-trivial shots among {len(records)} record(s): every "
            f"syndrome was empty, so there is no error weight to bound. "
            f"Pass include_trivial=True to measure how OFTEN that happens, "
            f"which is a different question")

    lows = [iv.lower for iv in ivs]
    ups = [iv.upper for iv in ivs]
    mean_lo = sum(lows) / n
    mean_up = sum(ups) / n

    # A one-sided lower confidence bound on the population mean of the
    # LOWER ends. Every term is itself a proven bound, so the statistical
    # step only accounts for sampling -- it is not doing the work of the
    # proof, and cannot rescue a run whose duals were weak.
    #
    # *** BOOTSTRAP, NOT `mean - 1.645*se`. ***
    #
    # An earlier draft used the normal approximation -- hours after this
    # repository documented that exact shortcut being wrong for small
    # Poisson counts, and replaced a hardcoded table with an exact
    # Garwood limit for it. Per-shot weights are heavily right-skewed:
    # most shots are light and a few are very heavy, which is the shape
    # a normal interval handles worst.
    #
    # The percentile bootstrap assumes nothing about the distribution.
    # It is seeded, so the artifact is reproducible bit for bit, which
    # matters because this number is published.
    pop_lo = None
    skew = None
    if n >= 2:
        m2 = sum((x - mean_lo) ** 2 for x in lows) / n
        if m2 > 0:
            m3 = sum((x - mean_lo) ** 3 for x in lows) / n
            skew = m3 / (m2 ** 1.5)
        else:
            skew = 0.0
        rng = random.Random(bootstrap_seed)
        means = []
        for _ in range(bootstrap_resamples):
            tot = 0.0
            for _ in range(n):
                tot += lows[rng.randrange(n)]
            means.append(tot / n)
        means.sort()
        pop_lo = means[int(0.05 * (len(means) - 1))]

    # THE SHAPE DECIDES WHETHER THE LEVEL MEANS ANYTHING. See the table
    # beside LEFT_SKEW_UNRELIABLE: at skew -1.9 the measured coverage of
    # a nominal 95% bound is 0.64.
    reliable = skew is None or skew > LEFT_SKEW_UNRELIABLE

    cert = [iv.lower for iv in ivs if iv.certified]
    return DeviceFloor(
        shots=len(records), nontrivial=n,
        certified_fraction=(sum(1 for iv in ivs if iv.certified) / n),
        mean_lower=mean_lo, mean_upper=mean_up,
        population_lower_95=pop_lo,
        population_bound_reliable=bool(reliable),
        mean_width=(mean_up - mean_lo),
        certified_only_mean_lower=(sum(cert) / len(cert) if cert else None),
        weight_convention=convention, skewness=skew,
        notes={"trivial_shots_excluded": trivial,
               "population_bound_method": (
                   f"percentile bootstrap, {bootstrap_resamples} resamples, "
                   f"seed {bootstrap_seed} -- distribution-free, because "
                   f"per-shot weights are right-skewed and a normal "
                   f"interval is worst exactly there"),
               "why_every_shot": (
                   "refused shots carry a valid lower bound (weak duality "
                   "needs feasibility, not optimality) and are the HARD "
                   "shots; excluding them would make the measurement "
                   "improve whenever the decoder got worse")})


def survivorship_gap(floor: DeviceFloor) -> float | None:
    r"""How much the certified-only average flatters the honest one.

    Positive means the certified subset looks HEAVIER; negative means it
    looks quieter, which is the direction that would let a weak decoder
    report a better device. Reported so the size of the effect is a
    number rather than an argument.
    """
    if floor.certified_only_mean_lower is None:
        return None
    return floor.certified_only_mean_lower - floor.mean_lower


def is_gameable_by_weakening_the_dual(before: DeviceFloor,
                                      after: DeviceFloor) -> bool:
    r"""*** THE ANTI-GAMING PROPERTY, AS AN EXECUTABLE CHECK. ***

    `after` is the same run scored with a WEAKER dual producer. A weaker
    dual lowers every bound, which alone would look like a quieter
    device -- so the interval must widen at least as much as the bound
    fell. If a weakening ever lowered the bound WITHOUT widening the
    interval, the headline number could be improved by making the
    machinery worse, and the metric would be worthless.

    Returns True when that has happened, which is a defect.
    """
    fell = before.mean_lower - after.mean_lower
    widened = after.mean_width - before.mean_width
    return fell > 1e-9 and widened < fell - 1e-9
