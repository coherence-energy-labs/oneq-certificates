r"""SHOTS ARE NOT INTERCHANGEABLE, AND THIS REPOSITORY CAN PROVE IT.

*** THE CURRENCY OF EVERY EXPECTATION-VALUE ALGORITHM IS VARIANCE, AND
EVERY SHOT IS CURRENTLY WEIGHTED THE SAME. ***

A quantum experiment that repeats ONE preparation and averages -- a
logical memory's error rate, a repeated-preparation expectation value --
runs the circuit N times and counts each shot once, because nothing
tells you which shots were trustworthy.

*** THE SCOPE, NARROWED AFTER AN OVERCLAIM. *** An earlier draft of this
docstring listed VQE, QAOA and shadow tomography. Those do not fit the
derivation below, which assumes a single fixed `s` across every shot: in
a variational algorithm each shot samples a distribution whose mean IS
the estimand, so the logical value is not constant and `Y_i = X_i / a_i`
is estimating a different thing each time. Extending to that case needs
the per-shot reliability folded into the sampling model, not into a
debias of a constant, and is NOT what this module does.

`experiments/bakeoff/e2_bakeoff_gate.py` produces exactly that missing
thing. Relabelling the logical observable as an extra detector and
decoding BOTH parities gives a weight difference: the complementary
gap, which is a log-likelihood ratio. It is calibrated, MEASURED at
d=5, p=3e-3:

    gap in [1,2)    misprediction 0.184 observed   0.182 predicted
    gap in [2,4)                  0.044            0.047

So each shot arrives with a probability that its logical outcome is
wrong. Weighting by it is then not a heuristic -- it is the
minimum-variance estimator, and the gain is a straight multiplier on
effective shot count with no change to the hardware.

--------------------------------------------------------------------------
*** THE ESTIMATOR, DERIVED RATHER THAN ASSERTED. ***

Let `s = (-1)^L` be the logical value being estimated, `pred_i` the
decoder's prediction, `g_i >= 0` the complementary gap, and

    p_i = 1/(1 + e^{g_i})            misprediction probability
    a_i = 1 - 2 p_i = tanh(g_i / 2)  RELIABILITY, in [0, 1)

With `X_i = (-1)^{pred_i}`:

    E[X_i | g_i] = (1 - p_i) s + p_i (-s) = a_i s

so `Y_i = X_i / a_i` is unbiased for `s`, and since `X_i^2 = 1`,

    Var(Y_i) = 1/a_i^2 - 1 = (1 - a_i^2) / a_i^2

Inverse-variance weighting is optimal among linear combinations, giving

    w_i = a_i^2 / (1 - a_i^2)        and        Var(s_hat) = 1 / sum(w_i)

*** THE ESTIMATOR IS NOT THE CONTRIBUTION. *** This is the Chair-Varshney
fusion rule, textbook in distributed detection since 1986. What is new
is having a CALIBRATED, per-shot, certificate-backed reliability out of
a QEC decoder that does not natively emit one. The statistics are old;
the input is not.

--------------------------------------------------------------------------
*** THE ASSUMPTION THAT CAN KILL IT, STATED IN FRONT. ***

`E[X_i | g_i] = a_i s` requires the calibration to be THE SAME for both
logical values. If mispredictions at a given gap are more common when
`L = 1` than when `L = 0` -- entirely possible, since the two parities
decode different graphs -- then one `a_i` cannot debias both, and the
weighted estimate acquires a bias that MORE SHOTS DO NOT REMOVE.

That is the whole risk, and it is testable: split by predicted value and
compare the two calibration curves. `parity_calibration_split` does it,
and `weighted_estimate` refuses to report a gain when the split is
significant rather than quietly returning a smaller number.

Note that DISCARDING low-gap shots rests on exactly the same assumption
and throws away information as well, so weighting dominates
post-selection whenever both are available. Neither is safe if the
split is real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: A reliability of exactly 1 is a shot asserted to be certainly right,
#: which would carry infinite weight and let one shot decide the answer.
#: Gaps that large come from a saturated float, not from evidence.
MAX_RELIABILITY = 1.0 - 1e-12

#: Below this, a shot is a coin flip and carries no information. It gets
#: weight ZERO rather than being dropped, so `shots` still counts it --
#: a shot that told you nothing is not a shot that did not happen.
MIN_RELIABILITY = 1e-12


class EstimatorRefusal(Exception):
    """The sample cannot support the estimate being asked for."""


#: How far outside [-1, 1] an estimate must land before it counts as a
#: violation rather than sampling noise. An ABSOLUTE epsilon cried wolf:
#: MEASURED on ibm_marrakesh, the measured-confidence estimate came in
#: at +1.0002 with a standard error of 0.00089 -- an overshoot of 0.28
#: sigma, which is a value pinned at the boundary of its range, not a
#: broken model. The genuinely broken one, +1.3033, was hundreds of
#: sigma out. A check that cannot tell those apart flags every honest
#: run and gets switched off.
OUT_OF_RANGE_SIGMA = 3.0


def _out_of_range(value: float, variance: float) -> bool:
    """Is `|value| > 1` by more than sampling can explain?"""
    se = math.sqrt(variance) if variance > 0 else 0.0
    return abs(value) > 1.0 + OUT_OF_RANGE_SIGMA * se + 1e-12


def reliability(gap: float) -> float:
    r"""`a = tanh(g/2)`, the calibrated 1 - 2p for a complementary gap.

    A NEGATIVE gap is refused rather than folded. The gap is defined as
    a magnitude, `|w1 - w0|`; a negative one means the caller has handed
    over a signed log-likelihood ratio with the sign still attached, and
    silently taking `abs` would DISCARD which parity it favoured -- the
    one piece of information the sign carries.
    """
    if not math.isfinite(gap):
        raise EstimatorRefusal(
            f"non-finite gap {gap}: a reliability derived from it would "
            f"be a number nothing measured")
    if gap < 0:
        raise EstimatorRefusal(
            f"negative gap {gap}: the complementary gap is a MAGNITUDE. "
            f"A signed value means the caller still holds the sign, and "
            f"folding it here would silently discard which parity the "
            f"evidence favoured")
    return min(math.tanh(gap / 2.0), MAX_RELIABILITY)


def inverse_variance_weight(gap: float) -> float:
    """`w = a^2 / (1 - a^2)`, the optimal linear-combination weight."""
    a = reliability(gap)
    if a <= MIN_RELIABILITY:
        return 0.0
    return (a * a) / (1.0 - a * a)


@dataclass(frozen=True)
class Estimate:
    """A weighted estimate and the honest account of what it cost."""
    value: float
    variance: float
    shots: int
    informative_shots: int          # those with non-zero weight
    #: Variance of the SAME debiased terms under equal weighting. The
    #: ratio to `variance` is how many times more shots equal weighting
    #: would need for the same precision.
    equal_weight_variance: float
    effective_shot_multiplier: float
    bias_warning: str | None = None
    #: *** |s| > 1 IS THE ESTIMATOR FALSIFYING ITSELF. ***
    #:
    #: `s = (-1)^L` lies in [-1, 1] by definition, so an estimate
    #: outside it cannot be a measurement of s -- it is proof that the
    #: reliabilities fed in were wrong. MEASURED on ibm_marrakesh: the
    #: gap clamp that removed a fantasy 49,881x multiplier pushed every
    #: reliability DOWN, the debias `X/a` then over-corrected, and the
    #: estimate came back +1.3033. The clamp fixed the variance and
    #: broke the point estimate, and only this check notices.
    out_of_range: bool = False
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema": "oneq-soft-estimate/1",
                "value": self.value, "variance": self.variance,
                "stderr": math.sqrt(self.variance) if self.variance > 0
                else 0.0,
                "shots": self.shots,
                "informative_shots": self.informative_shots,
                "equal_weight_variance": self.equal_weight_variance,
                "effective_shot_multiplier": self.effective_shot_multiplier,
                "bias_warning": self.bias_warning,
                "out_of_range": self.out_of_range,
                "out_of_range_means": (
                    "s = (-1)^L lies in [-1, 1] by definition. A value "
                    "outside it is not a measurement of s: it proves the "
                    "reliabilities are wrong -- most often a gap clamp "
                    "that understates confidence, so the debias "
                    "over-corrects"),
                "claim": (
                    "the minimum-variance linear estimator of the logical "
                    "value, weighting each shot by its calibrated "
                    "complementary gap"),
                "does_not_claim": (
                    "(a) unbiasedness when the calibration curve DIFFERS "
                    "between the two logical values -- see "
                    "parity_calibration_split; that bias does not shrink "
                    "with more shots. (b) a gain against CURRENT "
                    "PRACTICE. The multiplier compares against equal "
                    "weighting of the SAME debiased terms, which nobody "
                    "actually does; a majority vote is biased, so "
                    "comparing to it needs mean-squared error rather "
                    "than variance and is a different measurement"),
                "notes": dict(self.notes)}


def weighted_estimate(preds, gaps, *, bias_warning: str | None = None,
                      max_calibrated_gap: float | None = None) -> Estimate:
    r"""Minimum-variance estimate of `s = (-1)^L` from soft decisions.

    `preds` are the decoder's per-shot logical predictions (0/1 or
    False/True); `gaps` the matching complementary gaps.

    *** `max_calibrated_gap` IS NOT OPTIONAL POLISH. ***

    MEASURED on the first real run of `certified_noise_floor`: at d=3,
    p=3e-3, the reported effective-shot multiplier was **49,881**. That
    is not a variance reduction anyone achieved. 316 of 400 shots had
    gaps above 8, where `p = 1/(1+e^g)` claims an error rate of 1e-5 --
    a rate those 316 shots expect to see 0.003 times, and therefore
    cannot check. The weights were enormous because the model was
    EXTRAPOLATED far past any evidence, and the reported variance was
    fantasy.

    Clamping the gap to where calibration has power turns the multiplier
    back into a claim about measured confidence. The count of clamped
    shots is reported, because a run that is mostly extrapolation should
    say so rather than quietly hand back a large number.
    """
    preds = list(preds)
    gaps = list(gaps)
    if len(preds) != len(gaps):
        raise EstimatorRefusal(
            f"{len(preds)} predictions against {len(gaps)} gaps: pairing "
            f"them by position would attach confidences to the wrong shots")
    if not preds:
        raise EstimatorRefusal("no shots to estimate from")

    num = wsum = 0.0
    eq_var_terms = []
    informative = 0
    clamped = 0
    for pred, gap in zip(preds, gaps):
        gap = float(gap)
        if max_calibrated_gap is not None and gap > max_calibrated_gap:
            gap = max_calibrated_gap
            clamped += 1
        w = inverse_variance_weight(gap)
        if w <= 0.0:
            continue
        informative += 1
        a = reliability(gap)
        x = -1.0 if int(pred) else 1.0
        num += w * (x / a)
        wsum += w
        eq_var_terms.append((1.0 - a * a) / (a * a))

    if wsum <= 0.0:
        raise EstimatorRefusal(
            f"every one of {len(preds)} shot(s) had zero reliability: the "
            f"decoder expressed no preference on any of them, so there is "
            f"nothing to average. This is a real answer about the run, not "
            f"an estimate of zero")

    n_inf = len(eq_var_terms)
    var_opt = 1.0 / wsum
    value = num / wsum
    out_of_range = _out_of_range(value, var_opt)
    var_eq = sum(eq_var_terms) / (n_inf * n_inf)
    return Estimate(
        value=value, variance=var_opt, out_of_range=out_of_range,
        shots=len(preds), informative_shots=informative,
        equal_weight_variance=var_eq,
        effective_shot_multiplier=(var_eq / var_opt if var_opt > 0 else 1.0),
        bias_warning=bias_warning,
        notes={"weighting": "inverse-variance on tanh(gap/2)",
               "zero_weight_shots": len(preds) - informative,
               "max_calibrated_gap": max_calibrated_gap,
               "shots_clamped_to_calibrated_range": clamped,
               "why_clamp": (
                   "beyond the gap where calibration has power, "
                   "p = 1/(1+e^g) is an extrapolation; weighting by it "
                   "produced a measured 49,881x multiplier that no "
                   "evidence supported")})


def parity_calibration_split(preds, gaps, correct, *, bins=None) -> dict:
    r"""*** THE ASSUMPTION THE ESTIMATOR LIVES OR DIES ON. ***

    Splits the shots by PREDICTED logical value and compares the
    observed misprediction rate per gap bin against `1/(1+e^gap)` in
    each half.

    If the two halves calibrate differently, one reliability cannot
    debias both, and the weighted estimate carries a bias that more
    shots will not remove. Reported per bin with counts, because a
    divergence over four shots is not a divergence.
    """
    bins = list(bins) if bins is not None else [0.0, 1.0, 2.0, 4.0, 8.0]
    preds, gaps, correct = list(preds), list(gaps), list(correct)
    if not (len(preds) == len(gaps) == len(correct)):
        raise EstimatorRefusal(
            "predictions, gaps and outcomes must line up shot for shot")

    def _bin(g):
        for i in range(len(bins) - 1, -1, -1):
            if g >= bins[i]:
                return i
        return 0

    rows: dict[tuple[int, int], dict] = {}
    for pred, gap, ok in zip(preds, gaps, correct):
        key = (int(bool(int(pred))), _bin(float(gap)))
        r = rows.setdefault(key, {"n": 0, "wrong": 0, "gap_sum": 0.0,
                                  "pred_sum": 0.0})
        r["n"] += 1
        r["wrong"] += 0 if ok else 1
        r["gap_sum"] += float(gap)
        # PER SHOT, NOT AT THE MEAN GAP. The logistic is convex, so
        # p(mean g) under-predicts E[p(g)] -- see the note in
        # oneq.dem_calibration, where the same slip biased every
        # honest run toward looking overconfident.
        r["pred_sum"] += 1.0 / (1.0 + math.exp(float(gap)))

    out = []
    for (parity, b), r in sorted(rows.items()):
        out.append({"predicted": parity, "bin": b,
                    "bin_low": bins[b],
                    "n": r["n"], "wrong": r["wrong"],
                    "observed": r["wrong"] / r["n"],
                    "predicted_rate": r["pred_sum"] / r["n"]})

    # A divergence only counts where BOTH parities have enough shots to
    # say anything. `MIN_PER_CELL` is the count below which a difference
    # is noise; naming it stops the report reading as evidence.
    MIN_PER_CELL = 25
    worst, worst_bin = 0.0, None
    for b in range(len(bins)):
        two = {r["predicted"]: r for r in out if r["bin"] == b}
        if len(two) != 2:
            continue
        if min(two[0]["n"], two[1]["n"]) < MIN_PER_CELL:
            continue
        d = abs(two[0]["observed"] - two[1]["observed"])
        if d > worst:
            worst, worst_bin = d, b

    return {"schema": "oneq-parity-calibration/1",
            "rows": out, "worst_split": worst, "worst_split_bin": worst_bin,
            "min_shots_per_cell": MIN_PER_CELL,
            "comparable_bins": sum(
                1 for b in range(len(bins))
                if len({r["predicted"] for r in out if r["bin"] == b}) == 2),
            "why": ("the weighted estimator assumes one calibration curve "
                    "serves both logical values; where these two columns "
                    "diverge, one reliability cannot debias both and the "
                    "resulting bias does not shrink with more shots")}


# ================================================================
# MEASURED CONFIDENCE, INSTEAD OF THE MODEL'S
# ================================================================
#
# *** THE CLAMP WAS THE WRONG MECHANISM, AND THE ESTIMATOR SAID SO. ***
#
# Weighting by `p = 1/(1+e^g)` trusts the model at every gap, including
# gaps far past any evidence -- MEASURED, that produced a 49,881x
# effective-shot multiplier on stim. Clamping the gap to the last
# verified bin removed the fantasy but pushed every reliability DOWN,
# so the debias `X/a` over-corrected and the hardware estimate came back
# at +1.3033, outside the [-1, 1] that `(-1)^L` can occupy at all.
#
# Both failures have one cause: the reliability came from the MODEL. So
# it is measured instead, and the two uses are separated, because they
# want opposite kinds of error:
#
#   THE DEBIAS wants an unbiased rate. Too low and the correction
#   overshoots -- that is the +1.3033. Jeffreys `(k+0.5)/(n+1)` is used:
#   it is well behaved at k = 0, where the MLE would claim certainty
#   from having seen nothing go wrong.
#
#   THE WEIGHTS want a rate that cannot be optimistic, because an
#   overstated confidence understates the variance and every error bar
#   downstream inherits it. The exact one-sided 95% Poisson upper limit
#   is used -- the same Garwood machinery `hardware_lambda` uses, for
#   the same reason.
#
# MEASURED on ibm_marrakesh: the gap>=8 bin held 3,896 shots with ZERO
# errors. The device model claimed p = 1.84e-05; 3,896 clean shots
# support only p <= 7.69e-04 at 95%. The model was claiming SEVEN TIMES
# more confidence than the data can carry, and nothing was checking.

def _poisson_upper_95(k: int) -> float:
    """Exact one-sided 95% upper limit on a Poisson mean from k events."""
    from scipy.stats import chi2
    return float(0.5 * chi2.ppf(0.95, 2 * (k + 1)))


def empirical_reliability_map(gaps, correct, *, bins=None) -> dict:
    r"""Confidence as MEASURED, per gap bin, with monotonicity enforced.

    Returns `{bin_index: {"lo", "hi", "n", "wrong", "p_hat", "p_upper",
    "a_point", "a_conservative"}}`.

    *** MONOTONE, BECAUSE CONFIDENCE THAT RISES WITH DOUBT IS NOISE. ***
    A larger gap means the decoder was surer, so the error rate must not
    increase with it. Sampling noise routinely violates that in thin
    bins, and an unsmoothed map would hand a high-gap bin a WORSE rate
    than a low-gap one -- inverting the weighting the estimator exists
    to apply. A running maximum from the top down enforces it in the
    conservative direction.
    """
    bins = list(bins) if bins is not None else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    gaps, correct = list(gaps), list(correct)
    if len(gaps) != len(correct):
        raise EstimatorRefusal(
            "gaps and outcomes must line up shot for shot")
    if not gaps:
        raise EstimatorRefusal("no shots to calibrate against")

    cells: dict[int, dict] = {}
    for g, ok in zip(gaps, correct):
        g = float(g)
        b = 0
        for i in range(len(bins) - 1, -1, -1):
            if g >= bins[i]:
                b = i
                break
        c = cells.setdefault(b, {"n": 0, "wrong": 0})
        c["n"] += 1
        c["wrong"] += 0 if ok else 1

    out = {}
    for b, c in cells.items():
        n, k = c["n"], c["wrong"]
        p_hat = (k + 0.5) / (n + 1.0)                 # Jeffreys
        p_upper = min(_poisson_upper_95(k) / n, 0.5)  # exact, conservative
        out[b] = {"lo": bins[b],
                  "hi": bins[b + 1] if b + 1 < len(bins) else None,
                  "n": n, "wrong": k,
                  "p_hat": p_hat, "p_upper": p_upper}

    # MONOTONE FROM THE TOP DOWN: a bin may not be reported as more
    # error-prone than a less confident one below it.
    running_hat = running_up = 0.0
    for b in sorted(out, reverse=True):
        running_hat = max(running_hat, out[b]["p_hat"])
        running_up = max(running_up, out[b]["p_upper"])
        out[b]["p_hat"] = running_hat
        out[b]["p_upper"] = running_up
        out[b]["a_point"] = max(0.0, 1.0 - 2.0 * running_hat)
        out[b]["a_conservative"] = max(0.0, 1.0 - 2.0 * running_up)
    return out


def measured_estimate(preds, gaps, correct, *, bins=None) -> Estimate:
    r"""The estimate built on MEASURED confidence rather than the model's.

    The point estimate debiases with the Jeffreys rate and the weights
    use the conservative upper limit, so an over-confident model can
    neither inflate the multiplier nor push the answer out of range.

    A gap above every bin edge inherits the TOP bin's measured rate --
    never a rate extrapolated past the data, which is the whole defect
    this replaces.
    """
    preds, gaps, correct = list(preds), list(gaps), list(correct)
    if not (len(preds) == len(gaps) == len(correct)):
        raise EstimatorRefusal(
            "predictions, gaps and outcomes must line up shot for shot")
    cmap = empirical_reliability_map(gaps, correct, bins=bins)

    def cell(g):
        b = None
        for i, lo in enumerate(sorted(cmap)):
            if float(g) >= cmap[lo]["lo"]:
                b = lo
        return cmap[b] if b is not None else cmap[min(cmap)]

    num = wsum = 0.0
    eq_terms = []
    informative = 0
    for pred, g in zip(preds, gaps):
        c = cell(g)
        a_pt, a_cons = c["a_point"], c["a_conservative"]
        if a_pt <= MIN_RELIABILITY or a_cons <= MIN_RELIABILITY:
            continue
        informative += 1
        w = (a_cons * a_cons) / (1.0 - a_cons * a_cons)
        x = -1.0 if int(pred) else 1.0
        num += w * (x / a_pt)
        wsum += w
        eq_terms.append((1.0 - a_cons * a_cons) / (a_cons * a_cons))

    if wsum <= 0.0:
        raise EstimatorRefusal(
            "no shot carried measurable confidence: every gap bin's "
            "observed error rate was consistent with a coin flip")

    value = num / wsum
    var_opt = 1.0 / wsum
    var_eq = sum(eq_terms) / (len(eq_terms) ** 2)
    return Estimate(
        value=value, variance=var_opt,
        out_of_range=_out_of_range(value, var_opt),
        shots=len(preds), informative_shots=informative,
        equal_weight_variance=var_eq,
        effective_shot_multiplier=(var_eq / var_opt if var_opt > 0 else 1.0),
        notes={"weighting": "MEASURED confidence, not the model's",
               "debias_rate": "Jeffreys (k+0.5)/(n+1)",
               "weight_rate": "exact one-sided 95% Poisson upper limit",
               "why_two_rates": (
                   "the debias wants an unbiased rate -- too low and it "
                   "overshoots, which is how a clamped model produced "
                   "+1.3033 on hardware. The weights want a rate that "
                   "cannot be optimistic, because overstated confidence "
                   "understates every error bar downstream"),
               "calibration_map": {str(k): v for k, v in cmap.items()}})
