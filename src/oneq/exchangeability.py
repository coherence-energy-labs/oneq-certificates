r"""IS THE i.i.d. ASSUMPTION TRUE? ASK WITHOUT A MODEL.

*** EVERY DETECTOR ERROR MODEL IN THIS FIELD ASSUMES SHOTS ARE
INDEPENDENT AND IDENTICALLY DISTRIBUTED, AND ALMOST NOBODY TESTS IT. ***

A DEM assigns each mechanism a fixed probability. That is a statement
about every shot equally: no drift, no bursts, no dependence on what the
previous shot did. It is load-bearing for every logical error rate ever
published from one, and for `oneq.dem_calibration`, and for
`oneq.certified_weight`'s aggregate -- a mean over correlated shots has
a different variance than the one reported.

It is also the rare assumption that can be tested with NO MODEL AT ALL.

--------------------------------------------------------------------------
*** EXCHANGEABILITY MAKES THE NULL EXACT. ***

If shots are i.i.d., their ORDER carries no information: every
permutation of the sequence is equally likely. So for any statistic that
depends on order, the null distribution is obtained by permuting the
observed values and recomputing. That null is exact -- not asymptotic,
not approximate, not conditional on a distributional family. It inherits
whatever shape the data has, including the heavily skewed shapes that
break normal approximations elsewhere in this repository.

The only requirement is that shot order was preserved. It usually is not
recorded, which is one reason this goes untested; `hardware_noise_floor`
stores the detector array in submission order precisely so this is
possible offline.

--------------------------------------------------------------------------
*** A NEGATIVE RESULT WITHOUT ITS SENSITIVITY IS WORTHLESS. ***

"No drift detected" from a sample that could not have detected drift is
the dark-gate shape: the output reads the same whether the device is
stable or the test is blind. MEASURED on ibm_marrakesh, 4,000 shots:
the lag-1 test resolves |r1| >~ 0.047, and observed +0.022. So the
honest report is "no correlation detected, and anything below 0.05 would
have been invisible" -- which is a different sentence from "the device
is stable", and the only one the data supports.

Every report here carries the minimum effect it could have seen.

--------------------------------------------------------------------------
*** SEVERAL BLOCK SIZES ARE NOT SEVERAL EXPERIMENTS. ***

Block over-dispersion at 50, 100, 200 and 400 all pointed the same way
on that run, and none was significant. It is tempting to read four
agreeing signs as corroboration. They are four views of ONE dataset,
strongly correlated with each other, so agreement between them is close
to guaranteed and means far less than four independent tests would.

`block_scan` therefore returns the family and refuses to combine it into
a single p-value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


class ExchangeabilityRefusal(Exception):
    """The sample cannot support a statement about order."""


#: Below this many observations a permutation null is too coarse for the
#: p-value to mean much: with n points there are only so many distinct
#: orderings the statistic can take, and the test degenerates.
MIN_OBSERVATIONS = 30

#: Permutations. The p-value's own resolution is 1/n, so this bounds how
#: small a p-value can be reported at all -- stated rather than implied.
DEFAULT_PERMUTATIONS = 20_000


def lag1_autocorrelation(values) -> float:
    r"""Correlation between consecutive observations, in the order given.

    The natural statistic for "did this shot resemble the one before
    it?", which is what a burst or a slow drift produces and what an
    i.i.d. model forbids.
    """
    v = [float(x) for x in values]
    n = len(v)
    if n < 2:
        raise ExchangeabilityRefusal("lag-1 needs at least two observations")
    m = sum(v) / n
    dev = [x - m for x in v]
    denom = sum(d * d for d in dev)
    if denom <= 0.0:
        raise ExchangeabilityRefusal(
            "every observation is identical, so order cannot carry "
            "information and no correlation is defined. This is a real "
            "answer about the sample, not a correlation of zero")
    return sum(dev[i] * dev[i + 1] for i in range(n - 1)) / denom


def block_dispersion(values, block: int) -> float:
    r"""Variance of block means -- sensitive to SLOW drift.

    Lag-1 sees a shot resembling its neighbour. A device warming over
    thousands of shots produces almost no lag-1 signal and a large
    spread between blocks, so the two statistics look at different
    failures and neither subsumes the other.
    """
    v = [float(x) for x in values]
    k = len(v) // block
    if k < 2:
        raise ExchangeabilityRefusal(
            f"block size {block} gives {k} block(s) from {len(v)} "
            f"observations; a variance needs at least two")
    means = [sum(v[i * block:(i + 1) * block]) / block for i in range(k)]
    mu = sum(means) / k
    return sum((x - mu) ** 2 for x in means) / k


@dataclass(frozen=True)
class PermutationResult:
    statistic: str
    observed: float
    p_value: float
    null_mean: float
    null_sd: float
    permutations: int
    #: The smallest |effect| this sample could have separated from noise
    #: at three null standard deviations. A negative result without it
    #: cannot be told from a blind test.
    sensitivity: float
    resolution: float          # the finest p-value reportable, 1/permutations
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"statistic": self.statistic, "observed": self.observed,
                "p_value": self.p_value, "null_mean": self.null_mean,
                "null_sd": self.null_sd, "permutations": self.permutations,
                "sensitivity_3sigma": self.sensitivity,
                "p_value_resolution": self.resolution,
                "claim": ("an EXACT test: under i.i.d. shot order is "
                          "exchangeable, so permuting the observed values "
                          "gives the null distribution with no "
                          "distributional assumption at all"),
                "does_not_claim": (
                    "that the device is stable when p is large. It claims "
                    "only that no effect ABOVE the stated sensitivity was "
                    "found; a smaller one would have been invisible"),
                "notes": dict(self.notes)}


def permutation_test(values, statistic, *, seed: int = 20260903,
                     permutations: int = DEFAULT_PERMUTATIONS,
                     two_sided: bool = True,
                     name: str = "statistic") -> PermutationResult:
    r"""The exact test. `statistic` is any callable over an ordered list.

    Seeded, because this number is published and a permutation p-value
    that moved between runs of the same data would break every hash that
    covers it.
    """
    v = [float(x) for x in values]
    if len(v) < MIN_OBSERVATIONS:
        raise ExchangeabilityRefusal(
            f"{len(v)} observations is below {MIN_OBSERVATIONS}: a "
            f"permutation null over so few points is too coarse for its "
            f"p-value to mean anything")
    observed = statistic(v)
    rng = random.Random(seed)
    work = list(v)
    null = []
    for _ in range(permutations):
        rng.shuffle(work)
        null.append(statistic(work))
    mean = sum(null) / len(null)
    var = sum((x - mean) ** 2 for x in null) / len(null)
    sd = math.sqrt(var)
    if two_sided:
        hits = sum(1 for x in null if abs(x - mean) >= abs(observed - mean))
    else:
        hits = sum(1 for x in null if x >= observed)
    return PermutationResult(
        statistic=name, observed=observed,
        p_value=hits / len(null), null_mean=mean, null_sd=sd,
        permutations=len(null), sensitivity=3.0 * sd,
        resolution=1.0 / len(null),
        notes={"seed": seed, "two_sided": two_sided,
               "why_exact": ("shot order is exchangeable under i.i.d., so "
                             "the permutation null needs no model")})


def block_scan(values, blocks=(50, 100, 200, 400), **kw) -> dict:
    r"""Over-dispersion across several block sizes -- as a FAMILY.

    *** THESE ARE NOT INDEPENDENT EXPERIMENTS. *** On ibm_marrakesh all
    four pointed the same way and none was significant, and reading four
    agreeing signs as corroboration is the temptation this function
    exists to refuse: they are four views of one dataset, strongly
    correlated with each other, so agreement is close to guaranteed.

    The family is returned whole. No combined p-value is offered,
    because combining correlated tests as though independent is how a
    non-result becomes a finding.
    """
    out = []
    for b in blocks:
        try:
            r = permutation_test(values, lambda v, b=b: block_dispersion(v, b),
                                 two_sided=False, name=f"block_dispersion_{b}",
                                 **kw)
        except ExchangeabilityRefusal as e:
            out.append({"block": b, "refused": str(e)})
            continue
        d = r.to_json()
        d["block"] = b
        out.append(d)
    return {"family": out,
            "no_combined_p_value": (
                "these share one dataset and are strongly correlated, so "
                "agreement between them is close to guaranteed and a "
                "combined p-value computed as though they were "
                "independent would overstate the evidence badly"),
            "how_to_read": (
                "one block size clearing its own bar is a finding; four "
                "of them leaning the same way without clearing it is a "
                "hint that a larger sample could settle")}
