r"""The epsilon-optimality checker for matching receipts: exact arithmetic, declared epsilon.

WHY THIS EXISTS (external audit 2026-09-14, A-01 / B1). Paper 1's scale runs
accepted a certificate when a FLOATING-POINT gap was at most 1e-6, and the
released checker (matching_cert.check) demands EXACT optimality over double
weights -- a stronger claim than the paper argues is meaningful, which it
therefore could not reproduce (842 of 1,588 audited certificates). Neither
enforced the claim the paper makes. This module enforces exactly that claim.

THE THEOREM (paper 1, Lemma 1). Let z >= 0 be supported on T-odd vertex sets,
load_e = sum of z_S over sets S whose boundary contains e, and
theta = min(1, min over loaded edges of w_e / load_e). Then theta*z is a
feasible cut packing, so LB = theta * sum(z) is a lower bound on every
T-join, and for the candidate J

        w(J) - OPT  <=  epsilon := w(J) - LB.

THE CONTRACT. Everything is converted to fractions.Fraction (every FINITE
double converts exactly); no floating-point comparison decides anything.
The verdict is one of

  EXACT_OPTIMAL      epsilon == 0
  EPSILON_OPTIMAL    0 < epsilon <= eps_max, with epsilon reported
  NOT_PROVEN         epsilon > eps_max; says nothing about the decoder
  INVALID_INPUT      non-finite or negative weights or duals, an even set,
                     a correction that does not explain the syndrome, an
                     edge absent from the graph, nonzero node potentials
  VERIFIER_FAILURE   LB > w(J): impossible for valid inputs, so an internal
                     fault, never an acceptance

`eps_max` is declared by the caller and must be a finite, non-negative
rational; it travels with the verdict. A refusal is never evidence of
suboptimality -- only a lighter feasible correction is.

This is a CHECKER: it imports no producer (tools/standards_gate.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

BOUNDARY = -1

EXACT_OPTIMAL = "EXACT_OPTIMAL"
EPSILON_OPTIMAL = "EPSILON_OPTIMAL"
NOT_PROVEN = "NOT_PROVEN"
INVALID_INPUT = "INVALID_INPUT"
VERIFIER_FAILURE = "VERIFIER_FAILURE"


@dataclass(frozen=True)
class EpsilonVerdict:
    status: str
    reason: str
    eps_max: Fraction
    epsilon: Fraction | None = None
    lower_bound: Fraction | None = None
    primal: Fraction | None = None

    @property
    def accepted(self) -> bool:
        return self.status in (EXACT_OPTIMAL, EPSILON_OPTIMAL)


def _rational(x, what: str) -> Fraction:
    """A finite value as an exact Fraction; refuses NaN, inf and non-numbers."""
    if isinstance(x, bool):
        raise ValueError(f"{what}: boolean is not a number")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"{what}: non-finite value {x!r}")
        return Fraction(x)
    raise ValueError(f"{what}: unsupported type {type(x).__name__}")


def _key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if (u, v) <= (v, u) else (v, u)


def _graph(edges: Iterable) -> tuple[dict, dict]:
    """Canonical weights (parallel edges collapse to the minimum) and adjacency."""
    W: dict[tuple[int, int], Fraction] = {}
    for u, v, w in edges:
        if isinstance(u, bool) or isinstance(v, bool) or not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(f"edge ({u!r}, {v!r}): vertices must be integers")
        if u == v:
            raise ValueError(f"edge ({u}, {v}): self-loop")
        wf = _rational(w, f"weight of edge ({u}, {v})")
        if wf < 0:
            raise ValueError(f"edge ({u}, {v}): negative weight")
        k = _key(u, v)
        if k not in W or wf < W[k]:
            W[k] = wf
    adj: dict[int, list] = {}
    for k in W:
        for x in k:
            adj.setdefault(x, []).append(k)
    return W, adj


def _correction(W, syndrome: Sequence[int], matched_edges) -> tuple[Fraction, set]:
    """Exact weight of the correction, after checking it explains the syndrome."""
    T = set()
    for s in syndrome:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"syndrome entry {s!r} is not an integer")
        if s in T:
            raise ValueError(f"syndrome lists detector {s} twice")
        T.add(s)
    deg: dict[int, int] = {}
    primal = Fraction(0)
    seen = set()
    for e in matched_edges:
        u, v = int(e[0]), int(e[1])
        k = _key(u, v)
        if k not in W:
            raise ValueError(f"correction uses edge {k} absent from the graph")
        if k in seen:
            raise ValueError(f"correction lists edge {k} twice")
        seen.add(k)
        primal += W[k]
        for x in k:
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    if {x for x, n in deg.items() if n % 2} != T:
        raise ValueError("correction's odd-degree set is not the syndrome")
    return primal, T


def _scaled_bound(W, adj, T: set, blossom_duals) -> Fraction:
    """theta * sum(z) for a validated packing (Lemma 1)."""
    Tp = set(T) | ({BOUNDARY} if len(T) % 2 else set())
    load: dict[tuple[int, int], Fraction] = {}
    total = Fraction(0)
    for S, z in blossom_duals.items():
        members = set(S)
        if not members:
            raise ValueError("empty dual set")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in members):
            raise ValueError("dual set members must be integers")
        zf = _rational(z, f"dual of set of size {len(members)}")
        if zf < 0:
            raise ValueError("negative dual")
        if len(members & Tp) % 2 == 0:
            raise ValueError("dual set is not odd against the syndrome")
        if zf == 0:
            continue
        total += zf
        for k in {k for x in members for k in adj.get(x, ())}:
            if (k[0] in members) != (k[1] in members):
                load[k] = load.get(k, Fraction(0)) + zf
    theta = Fraction(1)
    for k, L in load.items():
        if L > W[k]:
            theta = min(theta, W[k] / L)
    return theta * total


def check_epsilon(*, edges, syndrome: Sequence[int], cert, eps_max) -> EpsilonVerdict:
    """Verdict on `cert` proving its correction within `eps_max` of optimal."""
    try:
        emax = _rational(eps_max, "eps_max")
    except ValueError as exc:
        raise ValueError(f"eps_max must be a finite rational: {exc}") from exc
    if emax < 0:
        raise ValueError("eps_max must be non-negative")
    try:
        W, adj = _graph(edges)
        for k, v in dict(cert.node_potentials or {}).items():
            if _rational(v, f"node potential {k}") != 0:
                raise ValueError("nonzero node potentials are not a T-join dual")
        primal, T = _correction(W, syndrome, cert.matched_edges)
        lb = _scaled_bound(W, adj, T, dict(cert.blossom_duals))
    except ValueError as exc:
        return EpsilonVerdict(INVALID_INPUT, str(exc), emax)
    eps = primal - lb
    if eps < 0:
        return EpsilonVerdict(VERIFIER_FAILURE, "lower bound exceeds the correction's weight",
                              emax, eps, lb, primal)
    if eps == 0:
        return EpsilonVerdict(EXACT_OPTIMAL, "the scaled packing attains the correction's weight",
                              emax, eps, lb, primal)
    if eps <= emax:
        return EpsilonVerdict(EPSILON_OPTIMAL, "within the declared epsilon", emax, eps, lb, primal)
    return EpsilonVerdict(NOT_PROVEN, "proven slack exceeds the declared epsilon; says nothing "
                          "about the decoder", emax, eps, lb, primal)
