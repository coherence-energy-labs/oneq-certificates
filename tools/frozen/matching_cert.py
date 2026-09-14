r"""Certifying MWPM decoding: an independent, solver-free checker.

THE CONTRIBUTION, scoped by milestone 0.0's kill sweep. The LP-dual optimality
certificate for a QEC decoder is NOT new -- Wu et al. (arXiv:2508.04969,
HyperBlossom) publish a theorem titled "Certifying Optimum" with complementary
slackness and per-solution bounds. What is absent from that work and from every
QEC decoder paper found is the CERTIFYING-ALGORITHMS half (Mehlhorn & McConnell,
CACM 2011): their certificate is an internal invariant -- the solver computes
the duals, the solver compares objectives, the solver announces optimality. A
buggy solver is buggy in both halves, so nothing is actually checked.

This module is the other half:

    solver  ->  (matching, dual)  ->  CHECKER  ->  accept / refuse

The checker shares no code with any solver, runs in O(nnz) with no LP and no
matching algorithm inside it, and has no false positives BY CONSTRUCTION: it
accepts only when weak duality is tight, and weak duality is a theorem.

    primal   min  sum_e w_e x_e     s.t. every syndrome node is matched
    dual     max  sum_v y_v + sum_S z_S
             s.t. y_u + y_v + sum_{S : e in delta(S)} z_S  <=  w_e   for all e
                  z_S >= 0

    WEAK DUALITY: any feasible dual value <= any feasible primal value.
    So if a feasible dual ACHIEVES the primal's weight, both are optimal --
    and that implication does not depend on where either came from.

SOUNDNESS BOUNDARY -- READ BEFORE QUOTING ANY CERTIFICATION RATE.

QEC decoding is a minimum-weight T-JOIN (odd-degree set = syndrome), which is
equivalent to minimum-weight perfect matching on the METRIC CLOSURE over the
syndrome nodes -- pairwise shortest-path distances, not single edges. The dual
of that problem therefore constrains PAIRS in T:

        y_u + y_v <= dist(u, v)        for all u, v in T

The edge-local constraints this checker currently enforces, y_u + y_v <= w_e for
graph edges e, are the dual of a DIFFERENT primal. They are necessary but NOT
sufficient: a dual can satisfy every edge constraint and still exceed the true
T-join optimum, because no constraint binds a pair joined by a path rather than
an edge.

This is not hypothetical. Running against PyMatching at d=7, the checker's own
weak-duality guard fired -- "dual objective 75.008 EXCEEDS primal 73.971,
impossible under weak duality" -- catching its own dual PRODUCER supplying an
invalid bound. The guard did its job; the formulation underneath it was wrong.

CONSEQUENCE, stated plainly: certification counts produced with an edge-local
dual DO NOT ESTABLISH optimality and must not be reported as if they did. The
accept path is only sound once the supplied dual is feasible for the metric
closure (or carries the blossom/odd-set terms that make the edge form
equivalent). Closing that is the next task, not a caveat to file away.

WHAT THE CHECKER IS ALLOWED TO ASSUME: nothing about the solver. It re-reads
the graph, re-sums the weight, tests every edge's dual constraint, and tests
complementary slackness on matched edges. If the decoder lied about its
matching, the weight check fails; if it lied about the dual, feasibility fails;
if it produced a suboptimal matching, the objectives do not meet.

TARGET FAILURE MODE. Not algorithmic approximation -- sparse blossom is exact.
The target is IMPLEMENTATION DEFECTS in production decoders, where the incumbent
assurance method is randomized differential testing against another
implementation (which cannot run per-shot in production and inherits the
reference's bugs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

BOUNDARY = -1


@dataclass(frozen=True)
class MatchingCertificate:
    """What a decoder must EXPORT for its answer to be checkable.

    Deliberately plain data: ints and floats, serializable to JSON, carrying no
    object the checker must trust. `blossom_duals` maps a frozenset of node ids
    (an odd set S) to z_S >= 0; an empty mapping is the common case on the
    graphs QEC decoding produces and is not an error.
    """
    matched_edges: tuple[tuple[int, int], ...]
    node_potentials: Mapping[int, float]
    blossom_duals: Mapping[frozenset, float] = field(default_factory=dict)
    claimed_weight: float | None = None
    # Set only when the dual is feasible for the METRIC CLOSURE (pairwise
    # shortest-path constraints), which is what a T-join bound actually needs.
    # Default False, so an edge-local dual cannot accidentally claim optimality.
    metric_closure_certified: bool = False

    def to_json(self) -> dict:
        return {
            "schema": "oneq-matching-certificate/1",
            "matched_edges": [list(e) for e in self.matched_edges],
            "node_potentials": {str(k): float(v)
                                for k, v in self.node_potentials.items()},
            "blossom_duals": [{"set": sorted(int(x) for x in S), "z": float(z)}
                              for S, z in self.blossom_duals.items()],
            "claimed_weight": self.claimed_weight,
        }


@dataclass
class CheckResult:
    accepted: bool
    reason: str
    primal_weight: float | None = None
    dual_objective: float | None = None
    gap: float | None = None
    checks: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema": "oneq-matching-check/1", "accepted": self.accepted,
                "reason": self.reason, "primal_weight": self.primal_weight,
                "dual_objective": self.dual_objective, "gap": self.gap,
                "checks": self.checks}


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if (v == BOUNDARY or u <= v) else (v, u)


def check(
    *,
    edges: Iterable[tuple[int, int, float]],
    syndrome: Sequence[int],
    cert: MatchingCertificate,
    tol: float = 1e-9,
) -> CheckResult:
    """Accept only if the certificate PROVES the matching is minimum-weight.

    `edges` is the decoding graph as (u, v, weight); v may be BOUNDARY (-1).
    `syndrome` is the list of node ids that fired. No solver is invoked, no
    matching is computed, and nothing the decoder said is taken on trust.

    Refusal is the default: every early return below is a refusal, and the
    accept path is reached only after all four checks pass.
    """
    W: dict[tuple[int, int], float] = {}
    # dict of SETS -- the annotation must match the set the code deliberately
    # uses (see the double-count note at the setdefault below); the previous
    # `list` annotation lied, and a lying type in the checker is how the next
    # reader re-introduces the duplicate-incidence bug the set exists to kill
    incident: dict[int, set[tuple[int, int]]] = {}
    for u, v, w in edges:
        k = _edge_key(int(u), int(v))
        if w < -tol:
            return CheckResult(False, f"negative edge weight {w} on {k}")
        # keep the CHEAPEST parallel edge: a decoder may not pick an expensive
        # duplicate and call it optimal
        if k not in W or w < W[k]:
            W[k] = float(w)
        for endpoint in k:
            # A SET, NOT A LIST. Parallel mechanisms in a DEM produce several
            # input edges with the SAME canonical key; W collapses them to the
            # cheapest, but an incidence LIST would keep one entry per input.
            # The feasibility walk adds z_S once per incidence, so a duplicate
            # entry double-counts the dual load on that edge and refuses a
            # perfectly valid packing for arithmetic that never happened.
            incident.setdefault(endpoint, set()).add(k)

    checks: dict = {}

    # --- 1. the correction is a valid T-JOIN: odd-degree nodes == syndrome.
    #
    # A WRONG MODEL OF THE DECODER'S OUTPUT, corrected by running at d=7. The
    # first version required every node to appear in at most one edge -- i.e.
    # it assumed the decoder returns a MATCHING on syndrome nodes. It does not.
    # It returns the EDGES OF A CORRECTION, which form PATHS: an interior node
    # of a path has degree 2 and is not in the syndrome at all. At d=3 the
    # paths are short enough that this almost never showed; at d=7 it refused
    # three of PyMatching's perfectly correct answers, and the refusal said
    # "node 84 matched twice" about a node that was not even fired.
    #
    # The real invariant is the T-join condition: the correction's ODD-DEGREE
    # vertices are exactly the fired detectors. Everything else may have any
    # even degree. A checker built on the wrong model does not fail safe -- it
    # accuses a correct decoder, which is the most expensive kind of wrong.
    deg: dict[int, int] = {}
    primal = 0.0
    for (u, v) in cert.matched_edges:
        k = _edge_key(int(u), int(v))
        if k not in W:
            return CheckResult(False, f"correction edge {k} is not in the graph")
        primal += W[k]
        for endpoint in k:
            if endpoint != BOUNDARY:
                deg[endpoint] = deg.get(endpoint, 0) + 1
    fired = {int(s) for s in syndrome}
    odd = {n for n, c in deg.items() if c % 2 == 1}
    if odd != fired:
        missing = sorted(fired - odd)
        extra = sorted(odd - fired)
        return CheckResult(
            False,
            f"NOT A VALID CORRECTION: odd-degree set != syndrome "
            f"(fired but even degree: {missing[:6]}; odd degree but quiet: "
            f"{extra[:6]})",
            primal_weight=primal)
    checks["t_join_valid"] = True
    checks["primal_weight"] = primal
    if cert.claimed_weight is not None and abs(cert.claimed_weight - primal) > 1e-6:
        return CheckResult(False,
                           f"decoder claimed weight {cert.claimed_weight} but "
                           f"its own edges sum to {primal}",
                           primal_weight=primal)

    # --- 2. dual feasibility on EVERY edge (this is the O(nnz) core)
    y = {int(k): float(v) for k, v in cert.node_potentials.items()}
    z = {frozenset(int(x) for x in S): float(zz)
         for S, zz in cert.blossom_duals.items()}
    for S, zz in z.items():
        if zz < -tol:
            return CheckResult(False, f"negative blossom dual z={zz}",
                               primal_weight=primal)
    # WALK EACH MOAT'S OWN BOUNDARY, DO NOT SCAN EVERY EDGE PER DUAL.
    #
    # The first version tested every (edge, dual) pair: O(nnz * |support(z)|).
    # Measuring it -- as the milestone required, rather than asserting O(nnz) --
    # showed why that matters. Cost was linear in nnz*|z| (exponent 0.83,
    # R^2 0.997), but |z| ITSELF grows as nnz^0.86, so the product put the
    # checker at ~nnz^1.86: very nearly quadratic, and the "linear-time
    # checking" headline was wrong.
    #
    # The quadratic was in the implementation, not the mathematics. An edge is
    # in delta(S) only if it touches S, so the work required is the sum of the
    # moats' volumes -- the certificate's own size -- not the graph's size
    # times the certificate's. Moats are local, so that is dramatically less.
    #
    # THE BOUNDARY IS A REAL VERTEX FOR CUT PURPOSES. Excluding it from every
    # set gave the checker a DIFFERENT delta(S) from the producer's, and valid
    # cut packings were reported DUAL INFEASIBLE on boundary edges 67 times in
    # 150 shots. It carries no potential, but it is inside or outside a set
    # like any other vertex, and both sides must agree.
    load: dict[tuple[int, int], float] = {}
    for S, zz in z.items():
        if zz == 0.0:
            continue
        for node in S:
            for k in incident.get(node, ()):
                other = k[1] if k[0] == node else k[0]
                if other not in S:          # exactly one endpoint inside
                    load[k] = load.get(k, 0.0) + zz

    # An edge carrying no dual load can only violate its constraint through a
    # node potential, so the full scan is needed only when potentials exist.
    # They never do for a cut packing; the branch keeps the legacy dual form
    # honest rather than quietly unchecked.
    to_check = W.items() if y else ((k, W[k]) for k in load)
    for (u, v), w in to_check:
        lhs = load.get((u, v), 0.0)
        if y:
            lhs += (y.get(u, 0.0) if u != BOUNDARY else 0.0) + \
                   (y.get(v, 0.0) if v != BOUNDARY else 0.0)
        if lhs > w + tol:
            return CheckResult(
                False,
                f"DUAL INFEASIBLE on edge {(u, v)}: {lhs:.9f} > weight {w:.9f}",
                primal_weight=primal)
    checks["dual_feasible"] = True

    # --- 3. THE OBJECTIVES, BEFORE complementary slackness.
    #
    # ORDER MATTERS, and getting it wrong produced a misleading refusal on real
    # PyMatching shots: the checker blamed COMPLEMENTARY SLACKNESS when the
    # actual situation was a dual that simply did not reach the primal (gap
    # 0.547). CS failing is EXPECTED whenever the supplied dual is suboptimal,
    # so reporting it as the fault points the reader at the wrong thing --
    # "your decoder violated slackness" reads as a decoder bug, when the truth
    # was "this certificate does not prove optimality". A refusal reason is
    # acted upon, so it has to name the real cause.
    #
    # By LP theory, when a feasible dual ATTAINS the primal value both are
    # optimal and CS holds automatically. So CS is checked only in that regime,
    # where its failure genuinely means a malformed certificate.
    dual_obj = sum(y.get(n, 0.0) for n in fired) + sum(z.values())
    gap = primal - dual_obj
    checks["dual_objective"] = dual_obj
    checks["gap"] = gap
    if gap < -1e-6:
        return CheckResult(
            False,
            f"dual objective {dual_obj:.9f} EXCEEDS primal {primal:.9f} -- "
            f"impossible under weak duality, so the certificate is malformed",
            primal_weight=primal, dual_objective=dual_obj, gap=gap, checks=checks)
    if gap > 1e-6:
        return CheckResult(
            False,
            f"NOT PROVEN OPTIMAL: gap {gap:.9f} (primal {primal:.9f}, dual "
            f"{dual_obj:.9f}). The matching may well BE optimal -- this "
            f"certificate does not show it. A dual without odd-set (blossom) "
            f"terms cannot always close the gap.",
            primal_weight=primal, dual_objective=dual_obj, gap=gap, checks=checks)

    # --- 4. complementary slackness: tight on every matched edge
    for (u, v) in cert.matched_edges:
        k = _edge_key(int(u), int(v))
        w = W[k]
        a, b = k
        lhs = (y.get(a, 0.0) if a != BOUNDARY else 0.0) + \
              (y.get(b, 0.0) if b != BOUNDARY else 0.0)
        for S, zz in z.items():
            inside = (a in S) + (b in S)
            if inside == 1:
                lhs += zz
        if abs(lhs - w) > 1e-6:
            return CheckResult(
                False,
                f"COMPLEMENTARY SLACKNESS violated on matched edge {k}: "
                f"dual {lhs:.9f} != weight {w:.9f}",
                primal_weight=primal)
    checks["complementary_slackness"] = True

    # A CUT-PACKING dual (z over T-odd sets) is edge-locally SOUND: its
    # constraints are exactly the T-join dual's, so feasibility here is
    # feasibility there. Node potentials alone are not, which is the defect
    # the d=7 run exposed. So the accept path requires EITHER odd-set terms
    # or an explicit metric-closure declaration -- never bare y.
    if not z and not cert.metric_closure_certified:
        return CheckResult(
            False,
            f"NOT PROVEN (formulation): dual attains the primal weight "
            f"({primal:.9f}), but it carries no odd-set terms and is not "
            f"declared feasible for the METRIC CLOSURE. Edge-local "
            f"feasibility does not bound a T-join -- see the soundness "
            f"boundary in this module's docstring. Refusing rather than "
            f"reporting an optimality this cannot support.",
            primal_weight=primal, dual_objective=dual_obj, gap=gap, checks=checks)

    return CheckResult(
        True,
        "OPTIMAL: feasible dual attains the primal weight, so weak duality "
        "makes both optimal -- verified without invoking any solver",
        primal_weight=primal, dual_objective=dual_obj, gap=gap, checks=checks)


def degraded_receipt(
    *,
    edges: Iterable[tuple[int, int, float]],
    syndrome: Sequence[int],
    matched_edges: Sequence[tuple[int, int]],
    dual_lower_bound: float,
) -> dict:
    """Milestone 0.3: the receipt for decoders with no optimality dual.

    Union-find, BP-OSD and neural decoders cannot produce a dual, so they
    cannot be certified optimal. They can still be given an honest receipt:
    syndrome-consistent, weight <= W, and an optimality GAP measured against
    any valid dual lower bound (typically one obtained from MWPM's certificate
    on the same shot). Stating the gap is not a weaker version of the
    certificate -- it is a different, quantified claim, and the gap
    distribution per decoder per distance is a number that does not currently
    exist in the literature.
    """
    W: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        k = _edge_key(int(u), int(v))
        if k not in W or w < W[k]:
            W[k] = float(w)
    weight = sum(W[_edge_key(int(u), int(v))] for u, v in matched_edges)
    covered: set[int] = set()
    ok = True
    for u, v in matched_edges:
        for e in (int(u), int(v)):
            if e == BOUNDARY:
                continue
            if e in covered:
                ok = False
            covered.add(e)
    consistent = ok and covered == {int(s) for s in syndrome}
    return {
        "schema": "oneq-degraded-receipt/1",
        "syndrome_consistent": consistent,
        "weight": weight,
        "dual_lower_bound": dual_lower_bound,
        "optimality_gap": weight - dual_lower_bound,
        "claim": ("syndrome-consistent with weight W and optimality gap at most "
                  "(W - dual bound). NOT a proof of optimality: no dual for "
                  "this decoder's own answer is supplied."),
    }
