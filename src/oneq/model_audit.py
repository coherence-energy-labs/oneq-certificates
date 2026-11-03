r"""Is the model a certificate binds to STRUCTURALLY MISSPECIFIED? A syndrome-only reading.

*** THE HOLE THIS CLOSES. *** Every certificate in this repository proves the same thing: that a
correction is minimum-weight UNDER A DECLARED MODEL. That is sound, and it has always carried an
unstated gap -- is the model any good? The ledger has the expensive proof that the gap is real:

    R78: an exactly-certified maximum-likelihood decoder made 268 logical errors where ALWAYS
         PREDICTING CLASS 0 made 97. "The arithmetic was never wrong; the model's support was."

A perfect certificate about a wrong model is SOUND AND USELESS, and nothing in the passport could
tell the two apart. README's hardware row says the quiet part already: "the error model is ours ...
against a declared uniform model -- not the device's physics."

*** THE STATISTIC (ledger R94/R96a/R100/R105). *** For each mechanism family -- space-like,
time-like, space-time -- take the model's pairwise probability against the device's Spitz two-point
estimate on the same shots, giving a per-family ratio. Then

    dominance = |log r| of the worst family  /  |log r| of the next worst

Four properties, each measured rather than assumed:

  * **It needs NO logical outcome.** It is computable on a machine that does not know its own
    answer, which is the only situation that matters in a real computation.
  * **It is a RATIO OF RATIOS**, so a uniform mis-scaling of the whole model cancels out of it. That
    is deliberate: R94 measured a uniform error costing 3% of the logical error rate against 233%
    for a family-structured one of identical statistical size. It sees only the damaging kind.
  * **It is NULL-CALIBRATED**, so a reading is an absolute distance from a correct model rather
    than a ranking. R96a's quoted 1.00-1.09 is the null of the SPREAD (max/min); `dominance`
    (max/second) is a noisier quantity and its own null, measured here over nine draws on nine IBM
    datasets, runs **1.10 - 4.44**. Quoting the spread's null for dominance would overstate the
    instrument, so the threshold is applied against dominance's own null and the draws travel with
    the reading.
  * **It responds to a REPAIR.** When the burst defect is removed it falls on 9 of 9 datasets,
    median 5.62x, landing on its own null (R105). Every earlier use was observational; that is the
    intervention that separates "measures mispricing" from "fingerprints the device".

*** WHAT IT DOES NOT DO, and the passport must not imply otherwise. *** It is a BINARY detector, not
a magnitude predictor: rho against available decoding gain is +0.000 on held-out devices (R100) and
+0.250 against the gain from a known repair (R105). It says whether a family-separable defect is
present. It says nothing about how much that defect costs, and nothing about soundness -- a
certificate over a misspecified model is still a valid proof of minimum weight under that model.

*** THE THRESHOLD IS RELATIVE TO THE NULL, not absolute, and a verdict must clear the WORST draw.
*** A device's own null is not 1 (finite shots, and a ratio of ratios is heavy-tailed), so
reading/null is what carries meaning -- and a reading that sits inside the null's own RANGE is a
reading the null explains, however large it looks beside the median. Measured on the nine IBM
datasets with `NULL_DRAWS = 9`:

    verdict              n   reading        median null   reading/null
    MISSPECIFIED         6   4.64 - 38.41   1.10 - 4.44   3.58 - 31.24
    INCONCLUSIVE         3   3.17 -  7.26   2.02 - 2.81   1.57 -  2.58
    NO_DEFECT_DETECTED   0

The three INCONCLUSIVE rows are the instrument declining rather than guessing: marrakesh A_40k
reads 7.26 at 2.58x its median null and is still refused, because its null draws reach past 7.26.
An instrument that never says "I cannot tell" is one whose quiet answers mean nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

BOUNDARY = -1
FAMILIES = ("space-like", "time-like", "space-time")
SCHEMA = "oneq-model-audit/1"

#: reading/null at or above this is a structural defect. Between the two populations, not at a
#: cliff: sound models measured 1.15-1.16, misspecified ones 2.5-20.3.
MISSPECIFIED_AT = 2.0
#: below this the two are indistinguishable and the audit declines to call it either way.
SOUND_BELOW = 1.5


def edge_family(u: int, v: int, positions: int) -> str:
    """space-like / time-like / space-time / boundary for a repetition-code detector layout.

    Detector `i` sits at position `i % positions` and round `i // positions`. This is index
    arithmetic and therefore exact -- it is why the IBM half of this lane was untouched by the
    variable-length-coordinate defect that emptied a whole family on Willow (R98).
    """
    if u == BOUNDARY or v == BOUNDARY or u < 0 or v < 0:
        return "boundary"
    pi, ri = u % positions, u // positions
    pj, rj = v % positions, v // positions
    if ri == rj:
        return "space-like"
    if pi == pj and abs(ri - rj) == 1:
        return "time-like"
    return "space-time"


def coords_from_records(records, *, space_indices, time_index) -> dict:
    """Detector index -> (space tuple, time), with BOTH slot sets named by the caller.

    *** THE EXPLICITNESS IS THE WHOLE POINT, AND R98 IS WHY. *** A Willow detector coordinate
    is `[x, y, t, <the detector it is compared against>]` -- VARIABLE LENGTH -- and reading
    `c[-1]` as the time slot emptied `time-like` on all twelve configurations while the audit
    went on returning a verdict over the families that were left. A silently empty family is
    worse than a crash: `dominance` is a ratio of the worst family to the next worst, so
    deleting a family changes the number without changing its type. There is deliberately no
    default and no "last element is time" convention here; a caller that does not know which
    slot is which cannot express the question.

    stim's `circuit.get_detector_coordinates()` gives `[x, y, t]` for a rotated surface code,
    so `space_indices=(0, 1), time_index=2`.
    """
    ti = int(time_index)
    si = tuple(int(i) for i in space_indices)
    if ti in si:
        raise ValueError(f"time_index {ti} is also listed as a space index {si}")
    out = {}
    for i, c in (records.items() if hasattr(records, "items") else enumerate(records)):
        need = max((ti,) + si) + 1
        if len(c) < need:
            raise ValueError(f"detector {i} has {len(c)} coordinates, needs at least {need}")
        out[int(i)] = (tuple(c[j] for j in si), c[ti])
    return out


def coordinate_layout(coords):
    """A family classifier for ANY code whose detectors carry (space, time), surface included.

    The classification itself was never repetition-specific -- only the index arithmetic that
    recovered the coordinates was. Same time is space-like, same place one round apart is
    time-like, anything else is space-time.
    """
    def fam(u: int, v: int) -> str:
        if u == BOUNDARY or v == BOUNDARY or u < 0 or v < 0:
            return "boundary"
        # A MISSING DETECTOR RAISES. Classifying it as `boundary` would drop the pair out of
        # every family silently, which is R98's failure wearing a different hat.
        su, tu = coords[int(u)]
        sv, tv = coords[int(v)]
        if tu == tv:
            return "space-like"
        if su == sv and abs(tu - tv) == 1:
            return "time-like"
        return "space-time"
    return fam


def _classifier(positions):
    """`positions` may be the repetition stride (int), a coordinate map, or a classifier."""
    if callable(positions):
        return positions
    if hasattr(positions, "items"):
        return coordinate_layout(positions)
    p = int(positions)
    return lambda u, v: edge_family(u, v, p)


def families_present(edges, positions) -> set:
    """Which families the MODEL actually contains. An empty one is a fact, not a silence.

    R97 found the d=3 repetition DEM has no space-time edges at all, so an ablation of that
    family moved nothing by exactly 1.00x; R98 found a coordinate bug emptying `time-like`
    and looking identical from the outside. Both are answered by asking the question directly
    before reading a dominance out of whatever survived.
    """
    fam = _classifier(positions)
    seen = set()
    for (u, v, _p) in edges:
        if u == BOUNDARY or v == BOUNDARY or u < 0 or v < 0:
            continue
        seen.add(fam(int(u), int(v)))
    return seen & set(FAMILIES)


def observed_pij(det: np.ndarray, pairs) -> dict:
    """The Spitz two-point estimator on the pairs the model actually couples."""
    n = det.shape[0]
    mean = det.mean(axis=0).astype(float)
    out = {}
    for (i, j) in pairs:
        xi, xj = mean[i], mean[j]
        num = float((det[:, i].astype(np.float64) * det[:, j]).sum() / n) - xi * xj
        den = (1.0 - 2.0 * xi) * (1.0 - 2.0 * xj)
        out[(i, j)] = num / den if abs(den) > 1e-12 else 0.0
    return out


def family_ratios(edges, det: np.ndarray, positions) -> dict:
    """model / observed, per family. Only two-detector edges: a family is a claim about a PAIR.

    `positions` is the repetition stride (int) for the layout this was built on, or a
    coordinate map / classifier for any other code -- see `coordinate_layout`.
    """
    pairs, model = [], {}
    for (u, v, p) in edges:
        if u == BOUNDARY or v == BOUNDARY or u < 0 or v < 0:
            continue
        key = (min(int(u), int(v)), max(int(u), int(v)))
        pairs.append(key)
        model[key] = float(p)
    if not pairs:
        return {}
    obs = observed_pij(det, pairs)
    fam = _classifier(positions)
    out = {}
    for f in FAMILIES:
        prs = [k for k in pairs if fam(k[0], k[1]) == f]
        if not prs:
            continue
        mo = float(np.mean([max(model[k], 1e-12) for k in prs]))
        ob = float(np.mean([max(obs[k], 1e-12) for k in prs]))
        out[f] = mo / ob
    return out


def dominance(ratios: dict) -> float | None:
    """How far the worst-mispriced family stands above the next. None if fewer than two families.

    `max/second`, NOT `max/sum`: the latter has a floor of 1/k and so is not comparable across
    models with different family counts, which silently flattered a cross-vendor comparison in
    R100 until it was checked.
    """
    d = sorted((abs(math.log(max(v, 1e-12))) for k, v in ratios.items() if k in FAMILIES),
               reverse=True)
    if len(d) < 2:
        return None
    return d[0] / max(d[1], 1e-12)


def sample_from_edges(edges, n_det: int, shots: int, seed: int = 0) -> np.ndarray:
    """Shots drawn from the model itself -- the null that makes a reading absolute, not relative."""
    rng = np.random.default_rng(seed)
    det = np.zeros((shots, n_det), dtype=np.uint8)
    for (u, v, p) in edges:
        fires = rng.random(shots) < float(p)
        for node in (u, v):
            if node != BOUNDARY and 0 <= node < n_det:
                det[fires, node] ^= 1
    return det


#: how many independent null draws. ONE IS NOT ENOUGH, and this module shipped with one until it
#: was validated against hardware: `dominance` is a ratio of ratios, so its null is heavy-tailed,
#: and single draws on the nine IBM datasets ranged 1.04 to **4.87**. On marrakesh A_40k that one
#: unlucky draw made a model reading 7.26 -- misspecified by every other measure -- come out
#: NO_DEFECT_DETECTED. A verdict resting on one draw of a noisy statistic is the blank-draw trap.
NULL_DRAWS = 9


@dataclass(frozen=True)
class Audit:
    """A reading, its null DISTRIBUTION, and the verdict they support -- never a verdict alone."""

    reading: float | None
    null: float | None                       # the median draw
    verdict: str
    null_max: float | None = None            # the worst draw; the reading must clear it
    null_draws: tuple = ()
    ratios: dict = field(default_factory=dict)
    shots: int = 0
    n_families: int = 0

    @property
    def relative(self) -> float | None:
        # `self.null in (None, 0)` was correct at runtime but is NOT a narrowing form:
        # mypy cannot discharge Optional through `in`, so the division below read as
        # `float / None`. Spelled out, it narrows and it says what it means.
        if self.reading is None or self.null is None or self.null == 0:
            return None
        return self.reading / self.null


def verdict_for(reading: float | None, null: float | None,
                null_max: float | None = None) -> str:
    """The ONLY place a verdict is derived. Both the builder and the re-check call this, so the two
    cannot drift apart -- a verdict computed twice by two rules is how a claim gets inflated between
    them (the failure `passport.build_core` guards against for the soundness block).

    MISSPECIFIED requires BOTH that the reading is `MISSPECIFIED_AT` times the median null AND that
    it clears the WORST null draw. The second condition is the one that can fail: a reading inside
    the null's own range is a reading the null explains, however large it looks beside the median.
    """
    if reading is None or null is None or null <= 0:
        return "INCONCLUSIVE"
    rel = reading / null
    if rel >= MISSPECIFIED_AT and (null_max is None or reading > null_max):
        return "MISSPECIFIED"
    if rel < SOUND_BELOW:
        return "NO_DEFECT_DETECTED"
    return "INCONCLUSIVE"


def audit(edges, det: np.ndarray, positions, *, n_det: int | None = None,
          seed: int = 0, null_draws: int = NULL_DRAWS) -> Audit:
    """Read the model against the device, and against itself `null_draws` times.

    `positions`: the repetition stride (int), or a `{detector: (space, time)}` map, or a
    classifier. The surface-code path goes through `coords_from_records`, which makes the
    caller name which coordinate slots are space and which is time.
    """
    if n_det is None:
        n_det = int(det.shape[1])
    ratios = family_ratios(edges, det, positions)
    reading = dominance(ratios)
    draws = []
    for k in range(max(1, int(null_draws))):
        nd = sample_from_edges(edges, n_det, int(det.shape[0]), seed=seed + 1000 * k)
        v = dominance(family_ratios(edges, nd, positions))
        if v is not None:
            draws.append(float(v))
    null = float(np.median(draws)) if draws else None
    nmax = float(max(draws)) if draws else None
    return Audit(reading=reading, null=null, verdict=verdict_for(reading, null, nmax),
                 null_max=nmax, null_draws=tuple(draws),
                 ratios=ratios, shots=int(det.shape[0]), n_families=len(ratios))


def _ratio(x: float | None) -> dict | None:
    """Floats are refused by a passport core (cross-language formatting is a forgery surface), so a
    measured quantity travels as an exact ratio."""
    if x is None:
        return None
    fr = Fraction(float(x)).limit_denominator(10 ** 6)
    return {"num": int(fr.numerator), "den": int(fr.denominator)}


def passport_block(a: Audit) -> dict:
    """The block a passport core carries. Records the READING and its NULL, never a bare verdict:
    a verdict without the numbers behind it cannot be re-derived by a verifier, and this repository
    does not ask anyone to take a producer's word for a number."""
    return {
        "schema": SCHEMA,
        "statistic": "family_dominance",
        "reading": _ratio(a.reading),
        "null": _ratio(a.null),
        "null_max": _ratio(a.null_max),
        "null_draws": int(len(a.null_draws)),
        "verdict": a.verdict,
        "shots": int(a.shots),
        "n_families": int(a.n_families),
        "misspecified_at": _ratio(MISSPECIFIED_AT),
        "sound_below": _ratio(SOUND_BELOW),
        "means": ("family_dominance measures whether the model is STRUCTURALLY misspriced between "
                  "mechanism families, from syndromes alone. It is a binary detector, not a "
                  "magnitude: it does not say what the defect costs (rho +0.00 to +0.25 against "
                  "decoding gain, R100/R105), and it says NOTHING about soundness -- a certificate "
                  "over a misspecified model is still a valid proof of minimum weight under it."),
    }


def check_block(block) -> None:
    """Re-derive the verdict from the block's own numbers and refuse a disagreement.

    A caller can hand-build a block, and the signed core is the last place a claim can be inflated.
    `passport.build_core` calls this for exactly the reason it re-checks the soundness block: "the
    producer already checked" is the trust this repository does not extend to producers.
    """
    if not isinstance(block, dict) or block.get("schema") != SCHEMA:
        raise ValueError(f"model-audit block has schema {block.get('schema')!r}, expected {SCHEMA}")

    def val(key):
        r = block.get(key)
        if r is None:
            return None
        if not isinstance(r, dict) or "num" not in r or "den" not in r or not r["den"]:
            raise ValueError(f"model-audit {key} must be an exact ratio, got {r!r}")
        return float(r["num"]) / float(r["den"])

    reading, null, nmax = val("reading"), val("null"), val("null_max")
    want = verdict_for(reading, null, nmax)
    got = block.get("verdict")
    if got != want:
        raise ValueError(
            f"model-audit block claims {got!r} but its own reading {reading} against null {null} "
            f"supports {want!r}. Understating is free; overstating is the forgery this repository "
            f"was audited for")
