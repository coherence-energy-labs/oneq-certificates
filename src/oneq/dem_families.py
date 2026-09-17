r"""Edge families of the repetition-code decoding graph, and per-family weight scales.

Every edge of the rep-code memory's decoding graph belongs to one of three
families, by index arithmetic with node = round * positions + position and
positions = d - 1:

    time-like    same position, adjacent rounds   |u - v| == positions
                 (measurement / idle error between rounds)
    space-like   adjacent positions, same round    |u - v| == 1, same round
                 (data-qubit error)
    boundary     one endpoint is the boundary      u < 0 or v < 0
                 (data error on an end qubit)

Anything else is "other" and is never scaled.

WHY THIS MODULE EXISTS. R13-R17 (docs/RESEARCH_LEDGER.md) found that the
device's published calibration yields a DEM whose FAMILY BALANCE is wrong on
a chain by a stable factor, and that one weight scale per family, fitted
once on half a run, cuts the certifying decoder's logical errors 2-7x on
runs it never saw. The family arithmetic and the scaling were written three
times inside experiment probes; the noise-floor lane needs them as a
DECLARED calibration layer whose scaled weights are pinned in the artifact.
One operator, one home.

Scaling is applied to the edge WEIGHT (log((1 - p) / p)); matching sees only
relative weights, so a uniform scale changes nothing about the decoder and
only its confidence -- measured 2026-09-05 (R16: 206 -> 206 errors under a
uniform 0.5x, calibration 0.0092 -> 0.0370).
"""
from __future__ import annotations

from fractions import Fraction

FAMILIES = ("time-like", "space-like", "boundary")

__all__ = ["FAMILIES", "apply_burst_model", "apply_family_scales",
           "burst_edges", "control_verdict", "degeneracy_verdict",
           "edge_family", "longest_timelike_run", "parse_scales"]

#: The synthetic control refuses an optimum below this: on shots sampled
#: from the model that decodes them the right answer is 1.0.
CONTROL_TOL = 0.5


def edge_family(u: int, v: int, positions: int) -> str:
    """The family of edge (u, v); a negative endpoint is the boundary."""
    if u < 0 or v < 0:
        return "boundary"
    if abs(u - v) == positions:
        return "time-like"
    if abs(u - v) == 1 and min(u, v) // positions == max(u, v) // positions:
        return "space-like"
    return "other"


def apply_family_scales(edges, scales: dict, positions: int):
    """Scale each edge's weight by its family's factor.

    `edges` is an iterable of (u, v, w). Families absent from `scales`, and
    "other", keep 1.0. Weights are multiplied EXACTLY when they are
    Fractions (the pinned form) and as floats otherwise; the caller
    decides which form it is carrying. Returns a new list, same order.
    """
    out: list[tuple] = []
    for u, v, w in edges:
        s = scales.get(edge_family(u, v, positions), 1)
        if isinstance(w, Fraction):
            out.append((u, v, w * Fraction(str(s))))
        else:
            out.append((u, v, float(w) * float(s)))
    return out


def parse_scales(text: str) -> dict:
    """'time-like=0.5,space-like=2.1,boundary=1.1' -> {family: scale}.

    Every named family must be one of FAMILIES and every family must be
    named: a missing family silently kept at 1.0 would be a scale set that
    reads as complete and is not.
    """
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in FAMILIES:
            raise ValueError(f"unknown family {name!r}; expected one of {FAMILIES}")
        out[name] = float(val)
    missing = [f for f in FAMILIES if f not in out]
    if missing:
        raise ValueError(f"scales missing for {missing}")
    return out


def longest_timelike_run(edges_uv, positions: int) -> int:
    """The longest chain of consecutive-round time-like EDGES at one
    detector position -- edges, not detector rounds, so a d=3 r=13 circuit
    (14 detector rounds) tops out at 13. A burst shows here; independent
    measurement noise does not."""
    per_pos: dict[int, set] = {}
    for u, v in edges_uv:
        if edge_family(u, v, positions) != "time-like":
            continue
        per_pos.setdefault(min(u, v) % positions, set()).add(min(u, v) // positions)
    best = 0
    for rounds in per_pos.values():
        rs = sorted(rounds)
        cur = run = 1
        for a, b in zip(rs, rs[1:]):
            cur = cur + 1 if b == a + 1 else 1
            run = max(run, cur)
        best = max(best, run)
    return best


def control_verdict(curve: dict) -> dict:
    """On synthetic shots the model is right, so the optimum must be near
    1.0 and small scales must hurt. Three-valued, and able to fail."""
    if not curve:
        return {"verdict": "NOT_TESTED", "reading": "no synthetic curve"}
    best = min(curve, key=lambda s: curve[s])
    at_one = curve.get(1.0)
    if at_one is None:
        return {"verdict": "NOT_TESTED", "reading": "the grid omits 1.0"}
    small = min((s for s in curve if s <= 0.15), default=None)
    if small is None:
        return {"verdict": "NOT_TESTED", "reading": "the grid has no small scale"}
    if best < CONTROL_TOL:
        return {"verdict": "REFUTED", "best_scale": best,
                "reading": "the optimum on a CORRECT model sits far below 1.0, so the "
                           "gain on hardware is the decoder's and not the device's"}
    if curve[small] <= at_one:
        return {"verdict": "REFUTED", "best_scale": best,
                "reading": "a hardware-winning scale does not hurt where the model is "
                           "right by construction"}
    return {"verdict": "UPHELD", "best_scale": best,
            "ratio_small_over_one": curve[small] / at_one if at_one else float("inf"),
            "reading": "the correction hurts where the model is correct, so on hardware "
                       "it is correcting the DEVICE's model"}


def degeneracy_verdict(flip_predictions: int, errors: int, true_flips: int) -> dict:
    """Scaling by cut membership must DESTROY the decoder (it stops ever
    predicting a flip), not improve it."""
    if flip_predictions == 0 and errors == true_flips:
        return {"verdict": "DEGENERATE",
                "reading": "the cut-scaled decoder never predicts a logical flip; its "
                           "errors are exactly the shots whose truth is a flip"}
    return {"verdict": "NOT_DEGENERATE",
            "reading": "the cut-scaled decoder still predicts flips; the kill recorded "
                       "in R19 does not reproduce and must be re-read"}


def burst_edges(positions: int, n_rounds: int, cost: float, min_span: int = 2):
    """Time-like edges spanning MORE than one round, all at the same `cost`.

    The mechanism R20 measured: a burst -- a leaked or stuck ancilla -- makes
    a detector misreport for many rounds at once. An independent-error model
    prices a k-round chain at k*w, which is exponentially unlikely; a burst
    costs about ONE event whatever k is. These are the edges that say so.

    `min_span` 2 skips the ordinary one-round edges, which the DEM already
    carries; `apply_burst_model` re-prices those separately.
    """
    return [(t * positions + pos, t2 * positions + pos, float(cost))
            for pos in range(positions)
            for t in range(n_rounds)
            for t2 in range(t + min_span, n_rounds)]


def apply_burst_model(edges, cost: float, positions: int, n_rounds: int):
    """The burst model, taken seriously: a time-like chain of ANY length
    costs about one event.

    Every time-like weight becomes `cost` -- not scaled, REPLACED, because
    under a burst the per-round probability is not the quantity that sets
    the price -- and the multi-round edges are added at the same `cost`.
    Space-like and boundary weights are untouched: they carry the data
    errors, which are what the observable actually depends on.

    MEASURED 2026-09-05 over 120,000 ibm_marrakesh shots at cost 0.5, with
    no fitting of any kind: 710 logical errors under the device's own model,
    147 under this one. On SYNTHETIC shots, where the model that decodes is
    the model that sampled and no burst exists, it is far worse (1,767 ->
    4,885) -- which is how we know it is modelling the device and not the
    decoder.
    """
    out = [(u, v, float(cost) if edge_family(u, v, positions) == "time-like" else w)
           for u, v, w in edges]
    return out + burst_edges(positions, n_rounds, cost)
