r"""IS THE DETECTOR ERROR MODEL THE DEVICE'S MODEL? ASK THE DECODER.

*** THE BOUNDARY EVERY OTHER MODULE HERE LEANS ON. ***

`oneq.matching_cert` proves a correction is minimum weight. `oneq.certified_weight`
turns the dual into a proven interval. Both are statements about weight
UNDER A DECLARED DETECTOR ERROR MODEL. Hand them the wrong weights and
they will prove, correctly and uselessly, an optimal answer to a
question the device was not asked.

So the DEM is the load-bearing assumption, and this module attacks it.

--------------------------------------------------------------------------
*** THE SIGNAL, AND WHY IT IS INDEPENDENT OF THE USUAL ONE. ***

The complementary gap is computed FROM the model: relabel the logical
observable as a detector, decode both parities, take the weight
difference. If the model is the truth, that gap is a log-likelihood
ratio and the misprediction rate at gap `g` is exactly

    p(g) = 1 / (1 + e^g)

The outcomes, however, come from the DEVICE. So the residual between
the two is a comparison between what the model predicts about itself
and what the hardware did.

The standard way to validate a DEM is to compare predicted against
observed DETECTOR CORRELATION frequencies -- a first- and second-moment
check on the graph's edges. This is a different instrument:

  * it reads the decoder's POSTERIOR GEOMETRY, not the edge marginals,
    so a mechanism that preserves pairwise correlations while moving
    probability mass between explanations is visible here and invisible
    there;
  * it is sensitive to what the model gets wrong ON THE CIRCUIT THAT
    RAN, weighted by how much each error mattered to a decision.

The two disagree in interesting ways, and neither subsumes the other.

--------------------------------------------------------------------------
*** DIRECTION IS THE DIAGNOSIS. ***

    observed > predicted    the model is OVERCONFIDENT. The device makes
                            errors the DEM does not know about --
                            leakage, crosstalk, correlated bursts,
                            drift since calibration. This is the common
                            failure and the dangerous one: every
                            downstream logical error rate is optimistic.

    observed < predicted    the model is PESSIMISTIC. Weights overstate
                            the noise; decoding is leaving performance
                            unclaimed.

A bare "the model is miscalibrated" is a shrug. The sign says where to
look.

--------------------------------------------------------------------------
*** SIGNIFICANCE IS NOT SIZE, AND THIS REPORTS BOTH. ***

With a million shots a chi-square test rejects a calibration that is
wrong in the fourth decimal, which is true and useless. With four
hundred it rejects nothing, which is also useless. So every report
carries:

    p_value              is the deviation more than sampling noise?
    calibration_error    how big is it, in probability?

and a run is only called miscalibrated when BOTH clear their bars. A
gate keyed on the p-value alone would fire on every large run; one keyed
on the effect size alone would fire on every small one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Below this a cell cannot distinguish a rate from noise, so it is
#: excluded from the statistic and REPORTED as excluded -- a bin quietly
#: dropped is a bin nobody knows was not checked.
MIN_PER_BIN = 30

#: *** SHOTS ARE NOT POWER. ***
#:
#: MEASURED on the first real run of `certified_noise_floor`: 316 of 400
#: shots landed in the gap>=8 bin, where the model predicts an error
#: rate of 1.03e-05. Those 316 shots expect 0.003 errors. Observing zero
#: is consistent with the model -- and equally consistent with a model
#: TEN TIMES worse, because the bin has no power to tell them apart.
#:
#: The report called that run CONSISTENT. That is the dark-gate shape:
#: a verdict that reads the same whether the model is right or merely
#: untested. A bin verifies a rate only when it expects enough events to
#: notice one going missing, so power is now computed per bin, and a run
#: with no powered bin is UNCHECKABLE rather than consistent.
MIN_EXPECTED_EVENTS = 5.0

#: The deviation, in absolute probability, below which a rejection is
#: statistically real and practically irrelevant.
#:
#: *** DERIVED, NOT CHOSEN. *** An earlier draft simply asserted 0.02,
#: which is the shape this estimate refuses elsewhere: a constant nobody
#: compares to an artifact drifts. Two independent anchors put it here.
#:
#: FLOOR -- it must not fire on an honestly calibrated system. E2
#: measures this repository's own soft output at d=5, p=3e-3:
#:
#:     gap [1,2)   observed 0.184   predicted 0.182   residual 0.002
#:     gap [2,4)   observed 0.044   predicted 0.047   residual 0.003
#:
#: so an honest residual here is 0.002-0.003. A bar at 0.02 sits ~7x
#: above that, far enough not to chase sampling noise.
#:
#: CEILING -- it must fire before the deviation matters downstream. The
#: soft estimator debiases by `a = 1 - 2p`; a calibration error `d` in
#: `p` shifts that by `2d`, so the relative bias in a weighted estimate
#: is about `2d/a`. At a typical `a ~ 0.6` (gap ~ 1.4), `d = 0.02` is
#: already ~6.7% relative bias in the answer -- material by any reading.
#:
#: `test_the_materiality_bar_SITS_BETWEEN_its_two_anchors` pins both
#: ends, so moving this number has to break a test that says why.
MATERIAL_ERROR = 0.02

#: The honest residual E2 measured, kept here so the floor above is a
#: comparison against a number and not a memory.
MEASURED_HONEST_RESIDUAL = 0.003

#: Default gap edges. Wide at the top because the interesting failures
#: live where the model claims near-certainty and the device disagrees.
DEFAULT_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


class CalibrationRefusal(Exception):
    """The sample cannot support a statement about calibration."""


def predicted_rate(gap: float) -> float:
    """`1/(1+e^gap)` -- what the model says its own error rate is."""
    if not math.isfinite(gap) or gap < 0:
        raise CalibrationRefusal(
            f"gap {gap} is not a usable magnitude")
    return 1.0 / (1.0 + math.exp(gap))


@dataclass(frozen=True)
class CalibrationReport:
    bins: list
    n_used: int
    n_excluded_thin: int
    calibration_error: float          # shot-weighted mean |obs - pred|
    signed_error: float               # >0 means the device is NOISIER
    chi2: float | None
    dof: int
    p_value: float | None
    verdict: str
    #: Bins expecting at least MIN_EXPECTED_EVENTS errors under the
    #: model. Zero means the run cannot distinguish this model from a
    #: much worse one, whatever the p-value says.
    powered_bins: int = 0
    powered_shots: int = 0
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema": "oneq-dem-calibration/1",
                "bins": list(self.bins),
                "shots_used": self.n_used,
                "shots_excluded_thin_bins": self.n_excluded_thin,
                "calibration_error": self.calibration_error,
                "signed_error": self.signed_error,
                "chi2": self.chi2, "dof": self.dof,
                "p_value": self.p_value,
                "verdict": self.verdict,
                "powered_bins": self.powered_bins,
                "powered_shots": self.powered_shots,
                "direction": (
                    "the device is NOISIER than the model claims"
                    if self.signed_error > 0 else
                    "the model overstates the noise"),
                "claim": (
                    "whether the declared detector error model predicts "
                    "its own misprediction rate, measured against device "
                    "outcomes through the decoder's posterior geometry"),
                "does_not_claim": (
                    "which mechanism is wrong. A miscalibration says the "
                    "model and the device disagree and in which "
                    "direction, not which edge weight to change"),
                "notes": dict(self.notes)}


def calibration_report(gaps, correct, *, bins=DEFAULT_BINS
                       ) -> CalibrationReport:
    r"""Compare the model's self-predicted error rate against outcomes.

    `gaps` are complementary gaps; `correct[i]` is whether the decoder's
    logical prediction for shot `i` was right.
    """
    gaps, correct = list(gaps), list(correct)
    if len(gaps) != len(correct):
        raise CalibrationRefusal(
            f"{len(gaps)} gaps against {len(correct)} outcomes: pairing "
            f"them by position would score shots against other shots' "
            f"answers")
    if not gaps:
        raise CalibrationRefusal("no shots to calibrate against")

    edges = list(bins)
    cells: dict[int, dict] = {}
    for g, ok in zip(gaps, correct):
        g = float(g)
        if not math.isfinite(g) or g < 0:
            raise CalibrationRefusal(f"gap {g} is not a usable magnitude")
        b = 0
        for i in range(len(edges) - 1, -1, -1):
            if g >= edges[i]:
                b = i
                break
        c = cells.setdefault(b, {"n": 0, "wrong": 0, "gap_sum": 0.0,
                                 "pred_sum": 0.0})
        c["n"] += 1
        c["wrong"] += 0 if ok else 1
        c["gap_sum"] += g
        # *** PER SHOT, NOT AT THE BIN'S MEAN GAP -- JENSEN. ***
        # The logistic is convex over the gaps that matter, so
        # E[p(g)] > p(E[g]). Evaluating at the mean UNDER-predicts the
        # error rate: MEASURED at +0.0070 on the [2,4) bin, which is a
        # third of the materiality bar. Every honestly calibrated run
        # would have read as OVERCONFIDENT -- a systematic false
        # positive in exactly the direction this module exists to flag.
        c["pred_sum"] += predicted_rate(g)

    rows, used, thin = [], 0, 0
    powered, powered_shots = 0, 0
    chi2 = 0.0
    dof = 0
    err_num = 0.0
    signed_num = 0.0
    for b in sorted(cells):
        c = cells[b]
        mean_gap = c["gap_sum"] / c["n"]
        pred = c["pred_sum"] / c["n"]        # mean of p(g), not p(mean g)
        obs = c["wrong"] / c["n"]
        row = {"bin": b, "bin_low": edges[b],
               "bin_high": (edges[b + 1] if b + 1 < len(edges) else None),
               "n": c["n"], "wrong": c["wrong"],
               "observed": obs, "predicted": pred,
               "mean_gap": mean_gap}
        if c["n"] < MIN_PER_BIN:
            row["excluded"] = f"only {c['n']} shots (< {MIN_PER_BIN})"
            thin += c["n"]
            rows.append(row)
            continue
        # POWER, NOT JUST COUNT. A bin expecting 0.003 events cannot
        # distinguish this model from one ten times worse.
        row["expected_events"] = c["n"] * pred
        # *** POWER MUST COUNT OBSERVED EVENTS TOO, OR THE GUARD IS
        # BLIND EXACTLY WHEN THE MODEL IS MOST WRONG. ***
        #
        # MEASURED on ibm_marrakesh (job d9su5b9d..., 4000 shots, device
        # calibration): the gap>=2 bin held 55 shots, predicted a 0.0228
        # error rate, and OBSERVED 0.0909 -- four times worse. Five real
        # errors against 1.25 expected, which Poisson puts at p ~ 0.009.
        #
        # The first version of this guard keyed only on EXPECTED events,
        # so it scored that bin powerless and the whole run UNCHECKABLE.
        # But expected events are computed FROM the model, and an
        # overconfident model understates exactly that quantity. The
        # guard therefore discarded its own strongest evidence, and did
        # so in the one direction that matters: a model claiming the
        # device is quiet silences the check that would catch it.
        #
        # Observed events carry information regardless of what the model
        # predicted. Either arm is enough.
        row["observed_events"] = c["wrong"]
        row["has_power"] = bool(
            row["expected_events"] >= MIN_EXPECTED_EVENTS
            or c["wrong"] >= MIN_EXPECTED_EVENTS)
        rows.append(row)
        used += c["n"]
        if row["has_power"]:
            powered += 1
            powered_shots += c["n"]
        err_num += c["n"] * abs(obs - pred)
        signed_num += c["n"] * (obs - pred)
        # A binomial cell's chi-square term. Guarded: a predicted rate of
        # exactly 0 or 1 has no variance and would divide by zero.
        var = pred * (1.0 - pred)
        if var > 1e-15:
            chi2 += c["n"] * (obs - pred) ** 2 / var
            dof += 1

    if used == 0:
        raise CalibrationRefusal(
            f"every gap bin held fewer than {MIN_PER_BIN} shots, so no "
            f"cell can tell a rate from noise. This is UNCHECKABLE, not "
            f"a passing calibration")

    p = None
    if dof > 0:
        try:
            from scipy.stats import chi2 as _chi2
            p = float(_chi2.sf(chi2, dof))
        except Exception:                        # noqa: BLE001
            p = None

    cal_err = err_num / used
    signed = signed_num / used
    significant = (p is not None and p < 0.01)
    material = cal_err >= MATERIAL_ERROR
    if powered == 0:
        # *** THE DARK VERDICT THIS GUARD EXISTS FOR. ***
        # Every bin is consistent with the model AND with a much worse
        # one. Calling that CONSISTENT is a green light over an
        # untested claim.
        verdict = (
            f"UNCHECKABLE: no gap bin either expects or observes at "
            f"least {MIN_EXPECTED_EVENTS} errors, so the run is "
            f"consistent with this model and with far worse ones. "
            f"More shots at the LOW-gap end, or a noisier operating "
            f"point, would give the comparison power")
    elif p is None:
        verdict = "UNCHECKABLE: no p-value available (scipy absent)"
    elif significant and material:
        verdict = "MISCALIBRATED"
    elif significant:
        verdict = ("significant but immaterial: real, and smaller than "
                   f"{MATERIAL_ERROR}")
    elif material:
        verdict = ("large but not significant: this sample cannot "
                   "distinguish it from noise")
    else:
        verdict = "CONSISTENT"

    return CalibrationReport(
        bins=rows, n_used=used, n_excluded_thin=thin,
        calibration_error=cal_err, signed_error=signed,
        chi2=chi2, dof=dof, p_value=p, verdict=verdict,
        powered_bins=powered, powered_shots=powered_shots,
        notes={"min_per_bin": MIN_PER_BIN,
               "min_expected_events": MIN_EXPECTED_EVENTS,
               "why_power": (
                   "a bin expecting 0.003 errors is consistent with the "
                   "model and with one ten times worse; shots are not "
                   "power, and a verdict from a powerless bin reads the "
                   "same whether the model is right or untested"),
               "material_error": MATERIAL_ERROR,
               "why_two_bars": (
                   "a chi-square on a million shots rejects a deviation in "
                   "the fourth decimal, and on four hundred rejects "
                   "nothing; significance and size answer different "
                   "questions and both are required")})
