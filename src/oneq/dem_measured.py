r"""The error model measured FROM THE DETECTORS, trusting no calibration.

WHY THIS MODULE EXISTS -- the contradiction that forced it. Two measured
facts of the R13-R22 campaign pull in opposite directions and cannot both
come from one fault in the device's calibration-derived DEM:

  * it OVERSTATES the experiment. Sampling from its own weights produces
    ~10x the logical error rate the device delivers -- 9.89x on ibm_fez,
    ~10x on ibm_marrakesh (R21a, R22: two devices).
  * it UNDER-uses the time-like route. Making time-like chains CHEAPER is
    what fixes the decoder, 710 -> 147 on marrakesh and 245 -> 3 on fez
    (R19, R21, R22).

A uniform inflation of every p cannot do both. Weight is w = log((1-p)/p),
so for small p, w ~ -log p; inflating every p by a factor k SUBTRACTS
~log k from every weight, which makes LONG paths relatively cheaper. A
uniformly over-stated model would therefore over-use the time-like route,
not under-use it. The observed direction is the opposite, so the fault is
DIFFERENTIAL: some families are overstated and others understated, and no
single scale can express that.

Which family is wrong is not a thing to guess. For a graph-like error model
-- one in which every mechanism flips at most two detectors -- the pairwise
error probability is DETERMINED by the detector statistics alone, with no
calibration data of any kind. That is the standard two-point estimator used
for surface-code experiments:

    C_ij = <x_i x_j> - <x_i><x_j>
    p_ij = 1/2 - 1/2 * sqrt(1 - 4 C_ij / ((1 - 2<x_i>)(1 - 2<x_j>) + 4 C_ij))

and the single-detector marginal then fixes each boundary edge, because a
detector fires iff an odd number of its incident mechanisms fired:

    prod over incident e of (1 - 2 p_e) = 1 - 2 <x_i>

WHAT IT MEASURED (R23, four runs across two devices). The DEM's space-like
(data-qubit) probabilities are overstated 17.4x on fez and 37.6x on
marrakesh, while its time-like probabilities are within 1.73x and on fez
UNDERSTATED (0.69x). That is the differential fault, measured rather than
inferred -- and it explains both facts above at once.

It also settles what R21a said a DEM-built simulator never could: the
long-range same-position correlations the DEM omits are VISIBLE in the
detector data. At span 8 rounds the hardware carries 0.38x (marrakesh) and
0.12x (fez) of the adjacent-round correlation, where an independent model
sampled at the same rate gives 0.01x. The burst is not a hypothesis fitted
to make the decoder work; it is in the detectors.

WHAT THIS BUYS. A decoding graph built only from detector statistics --
no calibration, and never seeing a logical outcome -- beats the device's
own model 6.25x in-sample (737 -> 118 errors) and 8.47x on HELD-OUT shots
(381 -> 45), where it also beats the hand-built burst model of R21 (66).
The held-out number being the BETTER one is the signature of a model that
is estimated rather than fitted: the burst constant 0.5 was chosen by
looking at error counts elsewhere, and this was not.

AND THE TWO HALVES ARE NOT SEPARABLE. This changes two things at once --
it re-prices the DEM's own edges from data, and it adds the multi-round
edges the DEM lacks -- so `measured_model_lane` reports each alone. Over
the same four held-out runs: pricing alone 381 -> 201 (1.90x), structure
alone 381 -> 104 (3.66x), both 381 -> 45 (8.47x). 1.90 * 3.66 = 6.95, so
the halves COMPOUND rather than add: the added edges are only worth their
cost when they are priced from the same data that justifies them.

Run `marrakesh A 40k` is the witness. Its multi-round correlation tail is
0.06x of adjacent where the other three runs read 0.33-0.51x, and there
structure alone is worth almost nothing (54 -> 51) while the full measured
model reaches 5. A flat burst price is a good approximation exactly where
a burst is strong and a poor one where it is not; an estimator needs no
such assumption because it reads the strength off each run.

*** AND IT IS NOT THE DEVICE'S MODEL. *** The result above is a DECODING
result and stops there. Turning R21a's own test on this model -- sample
from its weights, compare the detector statistics to the run's -- it
over-predicts the detector firing rate by 1.10x to 2.13x, and on ibm_fez
it is a WORSE description of the device than the DEM it beats 61x at
decoding (1.60x against 1.07x). `graph_likeness` finds the same thing
algebraically: on 108 of 112 detectors across the four runs the pairwise
probabilities already explain more than the detector fires, so the
marginal constraint has no solution and there is no valid model there.

The cause is derivable rather than mysterious. Five independent
same-position edges leaving one node permit five concurrent bursts, where
the physical event chooses ONE endpoint; the mass is over-counted by
construction. And damping it back (see `measured_graph`) monotonically
destroys the decoding gain, 45 errors to 124. So this is a better-shaped
PRIOR and not a truer model, which is exactly what R21's burst price also
was -- a flat cost is not a probability at all.

WHAT IT DOES NOT CLAIM. The estimator is exact only for a graph-like model.
Real devices have mechanisms that flip three or more detectors (leakage,
crosstalk); those bias every pair they touch, and this module cannot see
them. It also cannot see a mechanism that flips exactly one detector at a
node whose marginal it then explains as a boundary edge -- the marginal
constraint absorbs those by construction. Both limits are why the code-edge
STRUCTURE of the DEM is preserved here (see `measured_graph`) rather than
the graph being rebuilt from correlations alone.
"""
from __future__ import annotations

import math

import numpy as np

from .matching_cert import BOUNDARY

__all__ = ["MAX_SPAN", "PFLOOR", "MeasuredGraph", "candidate_pairs",
           "graph_likeness", "measure_pairs", "measured_graph",
           "pair_probability", "sample_from"]

#: Probability floor. An edge the data prices below this is not deleted --
#: deleting a CODE edge removes a cut and the observable node with it (the
#: decoder then raises `shots array should have at least N columns`, the
#: 2026-09-05 regression) -- it is priced at the floor instead, which is
#: dear but finite. Non-code pairs below the floor are simply not added.
PFLOOR = 1e-5

#: Longest same-position round separation given its own edge. Beyond this
#: the estimator's variance exceeds the correlation it is measuring: the
#: measured span-8 correlation is already 0.12-0.38x of adjacent, and each
#: extra span multiplies the candidate count by the round depth.
MAX_SPAN = 6


class MeasuredGraph(tuple):
    """`(edges, stats)` with the provenance the artifact has to record."""

    __slots__ = ()

    def __new__(cls, edges, stats):
        return super().__new__(cls, (list(edges), dict(stats)))

    @property
    def edges(self):
        return self[0]

    @property
    def stats(self):
        return self[1]


def pair_probability(x_i, x_j, x_ij):
    r"""The two-point estimator, or None where it leaves its branch.

    Returns None rather than a number when the denominator vanishes or the
    radicand goes negative. BOTH happen on real data -- a pair with no
    shared mechanism has C ~ 0 plus sampling noise, which can put the
    radicand either side of 1 -- and a caller that silently received 0.0
    would add a certain edge where the data said nothing. The None is the
    honest answer and every caller here treats it as "no evidence".
    """
    c = x_ij - x_i * x_j
    den = (1 - 2 * x_i) * (1 - 2 * x_j) + 4 * c
    if den == 0:
        return None
    v = 1 - 4 * c / den
    if v < 0:
        return None
    return 0.5 - 0.5 * math.sqrt(v)


def measure_pairs(det, pairs):
    """p for each (i, j) from the detector array alone.

    `det` is the (shots, nodes) uint8 detector array. Nothing else is
    consulted -- in particular not the logical outcomes, which is what
    makes a model built from this unable to overfit the metric it is
    judged on.
    """
    d = np.asarray(det, dtype=np.float64)
    means = d.mean(axis=0)
    out = {}
    for i, j in pairs:
        x_ij = float((d[:, i] * d[:, j]).mean())
        out[(i, j)] = pair_probability(float(means[i]), float(means[j]),
                                       x_ij)
    return out, means


def candidate_pairs(dem_edges, positions, n_rounds, max_span=MAX_SPAN):
    r"""Every pair worth measuring: the DEM's own code edges, plus the
    multi-round same-position pairs the DEM does not model at all.

    The DEM's edges are kept as candidates even where the data prices them
    at the floor, because they carry the code's STRUCTURE -- which pairs of
    detectors a single physical fault can touch. The estimator supplies
    probabilities; it is not asked to discover the code.
    """
    pairs = set()
    for (u, v, _w) in dem_edges:
        if v >= 0:
            pairs.add((min(int(u), int(v)), max(int(u), int(v))))
    for q in range(positions):
        for t in range(n_rounds):
            for s in range(2, max_span + 1):
                if t + s < n_rounds:
                    pairs.add((t * positions + q, (t + s) * positions + q))
    return sorted(pairs)


def measured_graph(dem_edges, det, positions, n_rounds, n_nodes,
                   max_span=MAX_SPAN, pfloor=PFLOOR, damping=1.0):
    r"""A decoding graph whose every probability came from the detectors.

    The DEM supplies which pairs a fault can touch; the DATA supplies how
    likely each is, and adds the multi-round edges the DEM omits. Boundary
    edges are not measured pairwise -- they are what is LEFT of a
    detector's marginal once the pair edges are accounted for, via
    prod(1 - 2 p_e) = 1 - 2 <x_i>.

    `damping` scales the ADDED multi-round probabilities only, and exists
    because this graph is provably not a generative model of the device
    (see `graph_likeness`): five independent same-position edges out of one
    node permit five concurrent bursts, where the physical event chooses
    ONE endpoint, so their mass is over-counted. damping = 1.0 keeps the
    raw estimate and damping -> 0 approaches a model whose node marginals
    are satisfiable.

    *** AND THE TWO OBJECTIVES ARE IN TENSION. *** Damping toward
    generative consistency monotonically DESTROYS the decoding gain, over
    the same four held-out runs:

        damping   generative rate vs real   held-out errors
          1.00          1.10 - 2.13               45
          0.50          1.03 - 1.50               87
          0.25          1.00 - 1.20               75
          0.10          0.96 - 1.05              107
          0.05          0.99 - 1.02              124

    So the over-count is not a defect to be corrected -- it is doing the
    work. What this module produces is a better-shaped PRIOR, not a truer
    description, and the default stays at 1.0 for that reason. The same
    caveat applies to R21's burst price, which is not a probability at all.
    """
    pairs = candidate_pairs(dem_edges, positions, n_rounds, max_span)
    est, means = measure_pairs(det, pairs)
    code_pairs = {(min(int(u), int(v)), max(int(u), int(v)))
                  for (u, v, _w) in dem_edges if v >= 0}
    edges = []
    prod = {i: 1.0 for i in range(n_nodes)}
    floored = 0
    added = 0
    for (i, j) in pairs:
        p = est.get((i, j))
        if p is None or p <= pfloor:
            # A code edge is PRICED at the floor, never dropped: dropping
            # one removes a logical cut and the observable node with it.
            if (i, j) not in code_pairs:
                continue
            p = pfloor
            floored += 1
        elif (i, j) not in code_pairs:
            p = p * damping
            if p <= pfloor:
                continue
            added += 1
        edges.append((i, j, math.log((1 - p) / p)))
        prod[i] *= (1 - 2 * p)
        prod[j] *= (1 - 2 * p)
    unsatisfiable = 0
    for i in range(n_nodes):
        pb = pfloor
        solved = False
        if prod[i] > 0:
            v = (1 - (1 - 2 * float(means[i])) / prod[i]) / 2
            if v > pfloor:
                pb = min(v, 0.49)
                solved = True
        if not solved:
            unsatisfiable += 1
        edges.append((i, BOUNDARY, math.log((1 - pb) / pb)))
    stats = {"pairs_measured": len(pairs), "code_edges": len(code_pairs),
             "floored": floored, "multi_round_added": added,
             "max_span": max_span, "pfloor": pfloor, "damping": damping,
             "unsatisfiable_marginals": unsatisfiable, "n_nodes": n_nodes}
    return MeasuredGraph(edges, stats)


def graph_likeness(dem_edges, det, positions, n_rounds, n_nodes,
                   max_span=MAX_SPAN, pfloor=PFLOOR):
    r"""How far this run is from the assumption the estimator rests on.

    THE ESTIMATOR'S OWN SELF-REFUTATION, and the honest reading of R23.
    The two-point formula is exact only where every mechanism flips at most
    two detectors. Nothing here can see a three-detector mechanism
    directly -- but its consequence is visible for free, because the
    pairwise probabilities imply a firing rate for each detector and the
    detector's actual rate is measured:

        predicted <x_i> = (1 - prod over incident e of (1 - 2 p_e)) / 2

    When the pair edges already explain MORE than the node fires, the
    boundary solve goes negative and there is no valid model. Measured on
    the four committed hardware runs, that happens on 108 of 112 nodes,
    with a median excess of 1.08x to 1.75x. The measured model is
    therefore NOT a generative description of these devices, and any claim
    that it "reads the device's true error model" is refuted by this
    function on the same data that motivates it.

    WHAT CAUSES IT HERE, corrected the same day (ledger R23a). R23 read the
    violation as evidence of a mechanism touching three or more detectors.
    The connected third cumulant tests that directly, and REFUTES it: the
    apparent three-way signal is entirely a shot-to-shot wobble in the
    error rate, and vanishes when the same statistic is computed within
    strata of a shot's total detector count. The violation is the wobble
    inflating the two-point estimate about fivefold, not a hyper-edge.

    So this function measures a REAL inconsistency and its cause is a
    nuisance parameter, not a hidden mechanism. Removing that nuisance
    properly -- estimating the same pairs within strata -- very nearly
    restores consistency (108 of 112 unsatisfiable nodes fall to 14) and
    costs 4.6x the decoding gain. Do not "fix" the inconsistency expecting
    a better decoder; three independent removals were tried and all three
    were worse.

    Returns the per-run numbers rather than a verdict: `unsatisfiable` is
    the node count, `median_excess` the ratio of predicted to observed
    firing at the median node.
    """
    pairs = candidate_pairs(dem_edges, positions, n_rounds, max_span)
    est, means = measure_pairs(det, pairs)
    prod = {i: 1.0 for i in range(n_nodes)}
    for (i, j) in pairs:
        p = est.get((i, j))
        if p is None or p <= pfloor:
            continue
        prod[i] *= (1 - 2 * p)
        prod[j] *= (1 - 2 * p)
    unsatisfiable = 0
    excess = []
    for i in range(n_nodes):
        if prod[i] <= 0:
            unsatisfiable += 1
        else:
            v = (1 - (1 - 2 * float(means[i])) / prod[i]) / 2
            if v <= pfloor:
                unsatisfiable += 1
        m = float(means[i])
        if m > 0:
            excess.append(((1 - prod[i]) / 2) / m)
    excess.sort()
    return {"n_nodes": n_nodes, "unsatisfiable": unsatisfiable,
            "unsatisfiable_fraction": unsatisfiable / n_nodes if n_nodes else 0.0,
            "median_excess": excess[len(excess) // 2] if excess else float("nan")}


def sample_from(edges, n_nodes, positions, shots, seed):
    r"""Sample detectors and EXACT truth from a planted edge model.

    This is the estimator's answer key and nothing else. It exists so that
    `measured_graph` is validated against a model whose probabilities are
    KNOWN before any hardware number is read from it -- an estimator that
    has not recovered a planted answer says nothing about an unknown one.
    """
    rng = np.random.default_rng(seed)
    det = np.zeros((shots, n_nodes), dtype=np.uint8)
    truth = np.zeros(shots, dtype=np.uint8)
    for (u, v, w) in edges:
        p = 1.0 / (1.0 + math.exp(w))
        fire = rng.random(shots) < p
        det[fire, int(u)] ^= 1
        if v != BOUNDARY and v >= 0:
            det[fire, int(v)] ^= 1
        elif int(u) % positions == positions - 1:
            truth[fire] ^= 1
    return det, truth
