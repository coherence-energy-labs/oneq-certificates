r"""Minimal exact checker for T-join optimality certificates.

This is a deliberately smaller trust boundary than ``matching_cert.py``.
It accepts ONE dual form only: a non-negative packing of T-odd cuts, which is
the literal LP dual of the T-join cut formulation on the ORIGINAL graph.
There are no node potentials, no metric-closure declaration to trust, no
repair search, and no tolerance in the acceptance decision.

Primal (T-join cut formulation)
-------------------------------

    min  sum_e w_e x_e
    s.t. x(delta(S)) >= 1       for every S with |S intersect T| odd
         x_e >= 0

Dual
----

    max  sum_S z_S
    s.t. sum_{S: e in delta(S)} z_S <= w_e    for every edge e
         z_S >= 0
         S is T-odd

Therefore a valid correction F and a feasible cut packing z with

    w(F) == sum_S z_S

prove optimality immediately by weak duality.  Complementary slackness need not
be checked: equality of a feasible primal and feasible dual is already the
certificate.

Why exact rationals are part of the schema
-------------------------------------------
The graph weights are the ORIGINAL IEEE doubles, converted with
``Fraction(float)`` so their binary values are preserved exactly.  A producer
must export each z_S as an exact numerator/denominator pair.  Repairing a
rounded dual is a producer job; the verifier does not mutate evidence into a
certificate and then certify its own mutation.

This module is intentionally boring.  That is a security property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

BOUNDARY = -1


def _ekey(u, v):
    u, v = int(u), int(v)
    # BOUNDARY IS ALWAYS SECOND. Written boundary-first,
    # u = -1 makes `u <= v` already true, so the swap never
    # fired: (-1, 3) and (3, -1) both survived as distinct
    # keys, the same edge was counted twice, and a valid
    # document was refused with GRAPH MISMATCH -- the exact
    # outcome graph_fingerprint exists to avoid.
    if u == BOUNDARY:
        return (v, u)
    if v == BOUNDARY:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def _frac(value):
    """Parse an exact rational dual value; floats are forbidden for duals."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        num, den = int(value[0]), int(value[1])
        if den == 0:
            raise ValueError("zero denominator")
        return Fraction(num, den)
    if isinstance(value, Mapping) and "num" in value and "den" in value:
        den = int(value["den"])
        if den == 0:
            raise ValueError("zero denominator")
        return Fraction(int(value["num"]), den)
    raise TypeError("dual values must be exact {num,den} pairs, not floats")


@dataclass(frozen=True)
class ExactCutCertificate:
    matched_edges: tuple[tuple[int, int], ...]
    cut_duals: Mapping[frozenset, object]

    def to_json(self):
        rows = []
        for S, value in sorted(self.cut_duals.items(), key=lambda kv: sorted(kv[0])):
            z = _frac(value)
            rows.append({"set": sorted(int(x) for x in S),
                         "z": {"num": z.numerator, "den": z.denominator}})
        return {
            "schema": "oneq/tjoin-exact-cut-certificate/v1",
            "matched_edges": [list(map(int, e)) for e in self.matched_edges],
            "cut_duals": rows,
        }


@dataclass
class StrictCheckResult:
    accepted: bool
    reason: str
    primal: Fraction | None = None
    dual: Fraction | None = None
    checks: dict = field(default_factory=dict)


def check_exact_cut(*, edges: Iterable[tuple[int, int, float]],
                    syndrome: Sequence[int], cert: ExactCutCertificate):
    """Verify a T-join and an exact T-odd-cut dual.  No solver, no repair."""
    # ANNOTATED, and the feasibility loop below no longer reuses `w`.
    # mypy read `w` as the float it is bound to in the edges loop, so
    # rebinding it to a Fraction was an error it reported and nobody
    # acted on -- in the checker whose whole claim is exact arithmetic.
    W: dict[tuple[int, int], Fraction] = {}
    vertices: set[int] = set()
    for u, v, w in edges:
        k = _ekey(u, v)
        fw = Fraction(float(w))             # exact IEEE-double value
        if fw < 0:
            return StrictCheckResult(False, f"negative edge weight on {k}")
        if k not in W or fw < W[k]:
            W[k] = fw
        vertices.update(k)
    vertices.discard(BOUNDARY)

    # Primal: the shipped correction itself must be a T-join.
    deg: dict[int, int] = {}
    primal = Fraction(0)
    for u, v in cert.matched_edges:
        k = _ekey(u, v)
        if k not in W:
            return StrictCheckResult(False, f"correction edge {k} absent", primal=primal)
        primal += W[k]
        for q in k:
            if q != BOUNDARY:
                deg[q] = deg.get(q, 0) + 1
    T = {int(q) for q in syndrome}
    odd = {q for q, d in deg.items() if d & 1}
    if odd != T:
        return StrictCheckResult(False, "correction is not a T-join", primal=primal)

    # Dual: exact values and, critically, only constraints that actually exist
    # in the primal are allowed to contribute to the objective.
    Z: dict[frozenset, Fraction] = {}
    try:
        for S0, value in cert.cut_duals.items():
            S = frozenset(int(q) for q in S0)
            z = _frac(value)
            if z < 0:
                return StrictCheckResult(False, "negative cut dual", primal=primal)
            if z == 0:
                continue
            if len(S & T) % 2 != 1:
                return StrictCheckResult(
                    False,
                    f"dual set {sorted(S)} is T-even; it is not a primal cut constraint",
                    primal=primal)
            Z[S] = Z.get(S, Fraction(0)) + z
    except (TypeError, ValueError) as exc:
        return StrictCheckResult(False, f"malformed exact dual: {exc}", primal=primal)

    if not Z:
        return StrictCheckResult(False, "empty cut dual cannot prove positive optimum",
                                 primal=primal, dual=Fraction(0))

    # Feasibility on EVERY graph edge, including boundary edges.
    for e, cap in W.items():
        a, b = e
        load = sum((z for S, z in Z.items() if ((a in S) != (b in S))),
                   Fraction(0))
        if load > cap:
            return StrictCheckResult(False,
                                     f"dual infeasible on edge {e}: {load} > {cap}",
                                     primal=primal)

    dual = sum(Z.values(), Fraction(0))
    if dual != primal:
        return StrictCheckResult(False,
                                 f"exact primal/dual gap: {primal - dual}",
                                 primal=primal, dual=dual,
                                 checks={"t_join": True, "t_odd": True,
                                         "dual_feasible": True})

    return StrictCheckResult(
        True,
        "exact feasible T-odd cut dual attains exact correction weight",
        primal=primal, dual=dual,
        checks={"t_join": True, "t_odd": True, "dual_feasible": True,
                "exact_attainment": True})
