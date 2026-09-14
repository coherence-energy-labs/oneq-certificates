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

SOUNDNESS BOUNDARY -- CLOSED 2026-08-18. READ THIS FIRST.

The break described below was REAL and is now FIXED: `check` refuses a
non-empty `node_potentials` outright (see step 2 of `check`), which
collapses this checker onto exactly the contract
`matching_cert_strict.check_exact_cut` enforces -- a non-negative packing
of T-odd cuts and nothing else. The paragraphs that follow are the record
of the finding and the argument for the fix, not a live caveat.

Two things made the fix safe, and both were MEASURED rather than assumed:
every producer in this tree already passed an empty potential map, and no
shipped document in evidence/, conformance/ or release/ carried one -- so
the field was dead except to an attacker. The real cost sat in the
equivalence gate, which built its random corpus OUT OF potentials; that
corpus was ported to T-odd cuts FIRST, and only then was the refusal
added. In the other order the gate would have compared two checkers that
both refuse everything -- a green light over a dark checker, which is a
worse failure than the one being fixed.

--- the original finding, kept as the record ---

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

CONSEQUENCE, as it stood: certification counts produced with an edge-local
dual DO NOT ESTABLISH optimality and must not be reported as if they did.

HOW IT WAS CLOSED, and why the obvious repair was the wrong one. The
tempting fix is to make the dual feasible for the metric closure. That
widens the checker to verify a DIFFERENT primal, and the whole difficulty
is that the metric-closure dual is a relaxation in which a feasible dual
can exceed the true T-join optimum. So the fix goes the other way: refuse
the mixed object entirely and verify only the dual that belongs to the
primal being claimed. A narrower contract is what makes the theorem in
`_exact_optimality` applicable again.

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
from fractions import Fraction
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
    #: *** DID THE ODD-DEGREE SET EQUAL THE SYNDROME? ***
    #:
    #: `True` past the T-join test, `False` at the return that fails it,
    #: `None` where the question was never reached (a malformed input
    #: rejected earlier) -- three states, because "not asked" is not
    #: "answered no".
    #:
    #: It is a field because the consumer used to recover it by
    #: searching this class's `reason` for the string "odd-degree"
    #: (`certifying_decoder.Certified.degraded`), and publish the
    #: result as `syndrome_consistent` in a receipt whose own text
    #: claims the correction explains the syndrome. Two producers
    #: already phrase that reason differently and both happen to
    #: contain the hyphenated token; drop the hyphen anywhere and an
    #: invalid correction starts publishing `syndrome_consistent:
    #: true`. The absence of a phrase was being read as good news.
    #:
    #: DECLARED only where the T-join test itself decides. Every later
    #: refusal DERIVES it from `checks["t_join_valid"]`, which this
    #: checker was already recording -- the structured answer existed
    #: the whole time and the consumer grepped English anyway. Deriving
    #: rather than repeating means a return site added later inherits
    #: the right value instead of quietly defaulting to "not asked".
    syndrome_consistent_declared: bool | None = None
    #: True only when every completed check says the correction and dual are
    #: valid and a tighter dual packing is the sole missing proof obligation.
    #:
    #: This is a machine-readable control-flow result. Consumers must not
    #: infer it by searching the human-readable refusal reason, because a
    #: harmless rewording would otherwise change which ladder rungs execute.
    only_obstruction_is_gap: bool = False

    @property
    def syndrome_consistent(self) -> bool | None:
        """Did the correction's odd-degree set equal the syndrome?

        THREE STATES. `None` means the question was never reached -- a
        malformed input rejected before the T-join test ran -- and that
        is not the same answer as `False`.
        """
        if self.syndrome_consistent_declared is not None:
            return self.syndrome_consistent_declared
        got = self.checks.get("t_join_valid")
        return bool(got) if got is not None else None

    def to_json(self) -> dict:
        return {"schema": "oneq-matching-check/1", "accepted": self.accepted,
                "reason": self.reason, "primal_weight": self.primal_weight,
                "dual_objective": self.dual_objective, "gap": self.gap,
                "syndrome_consistent": self.syndrome_consistent,
                "only_obstruction_is_gap": self.only_obstruction_is_gap,
                "checks": self.checks}


def _edge_key(u: int, v: int) -> tuple[int, int]:
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


def _finite(v, what: str):
    """Every numeric field, checked for finiteness at the boundary.

    EXTERNAL AUDIT 2026-08-04, finding 5 (CONFIRMED and fixed here). A
    NaN certificate was ACCEPTED as OPTIMAL, by both this checker and the
    JS one, because every comparison against NaN is False: `w < -tol` is
    False, `lhs > w + tol` is False, `gap > 1e-6` is False, so a
    non-finite value walked through every refusal in the file and out the
    accept path. That is fail-OPEN in a checker whose entire purpose is
    to fail closed, and it is the worst class of defect this program can
    have.

    Comparisons cannot catch NaN. Only an explicit test can, so this runs
    before any comparison sees the value.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, f"{what} is not a number: {v!r}"
    if f != f or f in (float("inf"), float("-inf")):
        return None, f"{what} is not finite: {v!r} -- refused"
    return f, None


def _collapse_graph(edges, tol):
    """The decoding graph as (cheapest weight per key, incidence sets).

    Returns ((W, incident), error).

    CACHED BY CONTENT (2026-09-08, OPEN_WORK F1 item 3 / O-4). The graph is
    the same for every shot of a run and this was rebuilt per shot: 6.6
    CPU-h per 1e6 shots at d=7. The key is the edge tuple itself, so a
    changed weight is a different key and a mutated list cannot serve a
    stale graph; the incidence sets are returned as fresh copies so a
    caller's mutation cannot leak into the next shot either. `check()`'s
    standalone signature is untouched.
    """
    key = (tuple((int(u), int(v), float(w)) for u, v, w in edges), float(tol))
    hit = _COLLAPSE_CACHE.get(key)
    if hit is not None:
        built, err = hit
        if built is None:
            return None, err
        W, incident = built
        return (dict(W), {n: set(s) for n, s in incident.items()}), None
    built, err = _collapse_graph_uncached(edges, tol)
    if len(_COLLAPSE_CACHE) >= _COLLAPSE_CACHE_MAX:
        _COLLAPSE_CACHE.pop(next(iter(_COLLAPSE_CACHE)))
    _COLLAPSE_CACHE[key] = (built, err)
    if built is None:
        return None, err
    W, incident = built
    return (dict(W), {n: set(s) for n, s in incident.items()}), None


_COLLAPSE_CACHE: dict = {}
_COLLAPSE_CACHE_MAX = 8


def _collapse_graph_uncached(edges, tol):
    W: dict[tuple[int, int], float] = {}
    # dict of SETS -- the annotation must match the set the code deliberately
    # uses (see the double-count note at the setdefault below); the previous
    # `list` annotation lied, and a lying type in the checker is how the next
    # reader re-introduces the duplicate-incidence bug the set exists to kill
    incident: dict[int, set[tuple[int, int]]] = {}
    for u, v, w in edges:
        k = _edge_key(int(u), int(v))
        w, err = _finite(w, f"edge weight on {k}")
        if err:
            return None, CheckResult(False, err)
        if w < -tol:
            return None, CheckResult(False, f"negative edge weight {w} on {k}")
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
    return (W, incident), None


def _tjoin_primal(cert, W, syndrome):
    """Is the correction a valid T-JOIN, and what does it weigh?

    Returns ((primal, fired), error).

    A WRONG MODEL OF THE DECODER'S OUTPUT, corrected by running at d=7. The
    first version required every node to appear in at most one edge -- i.e.
    it assumed the decoder returns a MATCHING on syndrome nodes. It does not.
    It returns the EDGES OF A CORRECTION, which form PATHS: an interior node
    of a path has degree 2 and is not in the syndrome at all. At d=3 the
    paths are short enough that this almost never showed; at d=7 it refused
    three of PyMatching's perfectly correct answers, and the refusal said
    "node 84 matched twice" about a node that was not even fired.

    The real invariant is the T-join condition: the correction's ODD-DEGREE
    vertices are exactly the fired detectors. Everything else may have any
    even degree. A checker built on the wrong model does not fail safe -- it
    accuses a correct decoder, which is the most expensive kind of wrong.
    """
    deg: dict[int, int] = {}
    # SUM EXACTLY, ROUND ONCE. Float addition is not associative, so
    # `primal += W[k]` made the reported weight depend on the ORDER of
    # `matched_edges`: on the surface-d3-5 conformance vector four edges give
    # two different doubles depending on the order they arrive in, and the
    # running-sum answer was the one that is NOT correctly rounded -- the
    # published receipt held the right value and the checker recomputed a
    # worse one. Accumulating in Fraction is exact and associative, so a
    # single rounding at the end is both order-independent and correctly
    # rounded, which is what makes the weight reproducible on any machine.
    exact_primal = Fraction(0)
    for (u, v) in cert.matched_edges:
        k = _edge_key(int(u), int(v))
        if k not in W:
            return None, CheckResult(
                False, f"correction edge {k} is not in the graph")
        exact_primal += Fraction(W[k])
        for endpoint in k:
            if endpoint != BOUNDARY:
                deg[endpoint] = deg.get(endpoint, 0) + 1
    primal = float(exact_primal)
    fired = {int(s) for s in syndrome}
    odd = {n for n, c in deg.items() if c % 2 == 1}
    if odd != fired:
        missing = sorted(fired - odd)
        extra = sorted(odd - fired)
        return None, CheckResult(
            False,
            f"NOT A VALID CORRECTION: odd-degree set != syndrome "
            f"(fired but even degree: {missing[:6]}; odd degree but quiet: "
            f"{extra[:6]})",
            primal_weight=primal,
            # THE ANSWER, AS A FIELD. The consumer used to recover this
            # by searching the sentence above for "odd-degree"; two
            # producers phrase it differently and both matched by luck.
            syndrome_consistent_declared=False)
    if cert.claimed_weight is not None:
        _cw, err = _finite(cert.claimed_weight, "claimed_weight")
        if err:
            return None, CheckResult(False, err, primal_weight=primal,
                                     syndrome_consistent_declared=True)
    if (cert.claimed_weight is not None
            and abs(cert.claimed_weight - primal) > 1e-6):
        return None, CheckResult(
            False,
            f"decoder claimed weight {cert.claimed_weight} but "
            f"its own edges sum to {primal}",
            primal_weight=primal,
            # PAST the T-join test: this correction DOES explain the
            # syndrome, it merely disagrees with its own claimed weight.
            syndrome_consistent_declared=True)
    return (primal, fired), None


def _moat_load(z, incident):
    """Dual load per edge from the odd-set (moat) terms.

    WALK EACH MOAT'S OWN BOUNDARY, DO NOT SCAN EVERY EDGE PER DUAL.

    The first version tested every (edge, dual) pair: O(nnz * |support(z)|).
    Measuring it -- as the milestone required, rather than asserting O(nnz) --
    showed why that matters. Cost was linear in nnz*|z| (exponent 0.83,
    R^2 0.997), but |z| ITSELF grows as nnz^0.86, so the product put the
    checker at ~nnz^1.86: very nearly quadratic, and the "linear-time
    checking" headline was wrong.

    The quadratic was in the implementation, not the mathematics. An edge is
    in delta(S) only if it touches S, so the work required is the sum of the
    moats' volumes -- the certificate's own size -- not the graph's size
    times the certificate's. Moats are local, so that is dramatically less.

    THE BOUNDARY IS A REAL VERTEX FOR CUT PURPOSES. Excluding it from every
    set gave the checker a DIFFERENT delta(S) from the producer's, and valid
    cut packings were reported DUAL INFEASIBLE on boundary edges 67 times in
    150 shots. It carries no potential, but it is inside or outside a set
    like any other vertex, and both sides must agree.
    """
    load: dict[tuple[int, int], float] = {}
    for S, zz in z.items():
        if zz == 0.0:
            continue
        for node in S:
            for k in incident.get(node, ()):
                other = k[1] if k[0] == node else k[0]
                if other not in S:          # exactly one endpoint inside
                    load[k] = load.get(k, 0.0) + zz
    return load


def _dual_feasible_edges(W, load, y, primal, tol):
    """Dual feasibility on EVERY edge (this is the O(nnz) core).

    An edge carrying no dual load can only violate its constraint through a
    node potential, so the full scan is needed only when potentials exist.
    They never do for a cut packing; the branch keeps the legacy dual form
    honest rather than quietly unchecked.
    """
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
    return None


def _complementary_slackness(matched_edges, W, y, z, primal):
    """Tight on every matched edge, or refuse."""
    for (u, v) in matched_edges:
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
    return None


def _exact_optimality(W, cert, y, z, fired):
    """Is the gap PROVABLY zero? Returns (proven, why).

    EXTERNAL AUDIT 2026-08-04, finding 4. Accepting on `gap <= 1e-6`
    certified provably suboptimal answers: with edges (0,1)=1.0000005 and
    0-B=1-B=0.5 the true optimum is 1.0, yet the direct edge certified
    with a feasible dual and gap 5e-7. Any epsilon > 0 admits a
    counterexample scaled to it, so shrinking the constant is not a fix.

    THE LATTICE ARGUMENT WAS TRUE BUT USELESS, and that is worth keeping
    because it looked like the answer for a day. Every weight here is an
    IEEE double, i.e. a DYADIC rational, so all of them lie on (1/D)Z with
    D the lcm of their denominators. Two distinct feasible objectives
    differ by at least 1/D, so a feasible dual with gap below 1/D has gap
    EXACTLY zero. Sound. But over double log-likelihood weights 1/D comes
    out near 4.4e-16 -- FINER THAN THE FLOAT ERROR IN THE DUAL ITSELF,
    which for a sum of a dozen terms near 11.4 is a couple of ulps. The
    rule could not certify the very certificates it was written for, and I
    concluded exactness was unavailable from float weights "by anyone".

    THAT CONCLUSION WAS WRONG, and the measurement that killed it is the
    useful part. The corrections were never suboptimal -- across the whole
    corpus `gap_vs_proven_optimum` is 0.0 with frac_zero 1.0. What is
    damaged is the DUAL, because a shipped dual is a float ROUNDING of an
    exact one, and rounding hurts it in BOTH directions:

        surface-d3-5   2.22e-16 SHORT of attaining
        surface-d3-4   4.44e-16 OVER on edge (6,12) -- infeasible, which is
                       why its objective exceeded its own primal and why the
                       producer's stated bound did not match its own dual sum

    Neither is suboptimality. Both are rounding damage, and rounding damage
    is REPAIRABLE. So this no longer asks whether the shipped dual happens
    to survive exact arithmetic; it REPAIRS the dual and then verifies the
    repair. Shed infeasibility (reducing a dual lowers load on every edge
    of its cut and raises it nowhere, so a cleared edge never regresses --
    which bounds the loop by the edge count as a proof, not a guess), then
    absorb the shortfall into a variable with slack across its whole cut,
    then re-verify feasibility, non-negativity and attainment from scratch.

    WHY THIS CANNOT BECOME A TOLERANCE IN DISGUISE. By LP duality a
    feasible dual attaining U exists IF AND ONLY IF U is optimal. So on a
    suboptimal primal the absorb step provably cannot find the slack -- it
    is forbidden by the theorem, not merely unlikely. The auditor's witness
    (edges (0,1)=1.0000005, 0-B=1-B=0.5, true optimum 1.0) is refused for
    exactly that reason, at every scale from 1e-9 to 1e9.

    What this buys over the fixed-point alternative: optimality is proven
    against the ORIGINAL double weights. Quantising the weights instead
    would have been sound but would have quietly changed the problem --
    weights stop being the true log-likelihoods and the claim degrades to
    "optimal with respect to our rounded weights". This proves the real
    one, needs no producer change, and cannot be applied at check time by
    rewriting the graph because `graph_sha256` binds the document to it.

    MY FIRST TWO ATTEMPTS WERE WRONG and the failures are worth keeping.
    Demanding exact float equality refused every genuine surface-code
    certificate. Then the lattice was computed over `Fraction(str(x))`,
    which converts the DECIMAL REPR -- str(0.1) -> 1/10 -- discarding the
    true binary value and producing a coarse fake lattice that also
    refused genuine work. `Fraction(x)` on a float is exact; `str` in the
    middle silently changes the number. Both broke the conformance
    vectors, which is how I found out.
    """
    from fractions import Fraction as F

    fired_set = {int(n) for n in fired}
    primal = sum((F(W[_edge_key(int(u), int(v))])
                  for (u, v) in cert.matched_edges), F(0))

    # Exact rational copies. Nothing past this line touches a float again.
    Y = {int(n): F(v) for n, v in y.items()}
    Z = {frozenset(int(x) for x in S): F(v) for S, v in z.items()}

    # MULTI-START. The repair is a greedy search and greedy order decides
    # whether it finds the dual: shedding can take from a variable that
    # absorbing then needs back, and no single spend order is right for
    # every graph. tools/exact_optimum_oracle.py proved 6/6 refused
    # hardware shots were EXACTLY optimal, so a feasible attaining dual is
    # guaranteed to exist by LP duality in those cases -- the search was
    # simply looking in one direction. Trying several deterministic orders
    # costs a few microseconds and cannot weaken anything: every attempt is
    # verified from scratch by step 3, so an ordering that produces garbage
    # is refused exactly like any other bad dual. Each attempt gets FRESH
    # copies, since the repair mutates in place.
    outcome = (False, "no spend ordering produced a feasible attaining dual")
    for order in range(_REPAIR_ORDERS):
        ok, why = _repair_and_verify(W, dict(Y), dict(Z), fired_set, primal,
                                     order)
        if ok:
            return True, why
        outcome = (ok, why)
    return outcome


_REPAIR_ORDERS = 4

# The widest common denominator O-2 will scale to, as a power of two.
# Surface-code weights are ordinary doubles near 1, so K lands around 52
# and the scaled integers are small. A subnormal would demand K ~ 1074,
# whose integers are slower than the Fractions they replace -- past this
# the repair keeps exact Fraction arithmetic instead. The bound decides
# WHICH exact representation is used, never whether the result is exact.
_MAX_SCALE_BITS = 256


def _spend_order(variables, pays, val, order):
    """WHICH variable pays when several could -- the choice that decides
    whether absorb can still close the gap afterwards.

    Order 0 is the original: free variables (potentials on unfired nodes,
    which cost load and buy nothing) first, then a total order on the key.
    The rest vary the tie-break. Every one of the four is a TOTAL order,
    so Python and JavaScript make identical repairs; a nondeterministic
    tie-break here would turn cross-language conformance from a check into
    a decoration.
    """
    def tie(vk):
        return str(sorted(vk[1])) if vk[0] == "z" else f"{vk[1]:012d}"

    if order == 1:
        return sorted(variables, key=lambda vk: (pays(*vk), vk[0], tie(vk)),
                      reverse=True)
    if order == 2:                        # biggest first: most room to give
        return sorted(variables,
                      key=lambda vk: (pays(*vk), -val(*vk), vk[0], tie(vk)))
    if order == 3:                        # smallest first: spread the loss
        return sorted(variables,
                      key=lambda vk: (pays(*vk), val(*vk), vk[0], tie(vk)))
    return sorted(variables, key=lambda vk: (pays(*vk), vk[0], tie(vk)))


def _to_common_scale(W, Y, Z, primal):
    """(W, Y, Z, primal, ZERO) in ONE exact domain -- integers when possible.

    WHY SCALING CANNOT CHANGE A VERDICT. `_repair_and_verify` uses only
    addition, subtraction, `min`, order comparison and equality against
    zero, and never multiplies or divides two unknowns. Every one of those
    commutes with scaling by a positive constant:

        a > b  <=>  Sa > Sb        a - b  <->  Sa - Sb
        min preserved              a == b <=>  Sa == Sb

    so multiplying every input by one S takes every branch the same way
    and reaches the same verdict. Scale-equivariant by construction, not
    "equivalent on the vectors we tried".

    AND THE SCALING IS ITSELF EXACT. `Fraction(x)` on a float is exact
    because every finite double IS a dyadic rational p/2^k -- the reason
    `_exact_optimality`'s docstring insists on it and forbids
    `Fraction(str(x))`. A common denominator 2^K, K the max over all
    inputs, turns every value into an exact integer with NOTHING rounded.
    This is the opposite of quantising the weights, which that docstring
    rejects as silently changing the problem: the same numbers, in
    different units. Python ints are far cheaper than Fraction, which
    normalises via gcd on every operation.

    FALLS BACK RATHER THAN FORCES. A subnormal weight would need a ~2^1074
    scale, whose integers are slower than the Fractions they replace, and a
    non-dyadic input has no power-of-two denominator at all. Either case
    keeps exact Fraction arithmetic, so the checker stays exact on inputs
    this optimisation cannot represent instead of rounding them to fit.

    Extracted from `_repair_and_verify` because O-1 and O-2 had grown it
    to 119 logic lines against a max of 90 -- the same standards finding
    that split that function out of `_exact_optimality`, on the same
    function, and length is where logic hides.
    """
    from fractions import Fraction as F

    def _dyadic_bits(fr):
        d = fr.denominator
        return None if (d & (d - 1)) else d.bit_length() - 1

    Wf = {e: F(W[e]) for e in W}
    Yf = {int(n): F(v) for n, v in Y.items()}
    Zf = {S: F(v) for S, v in Z.items()}
    pf = F(primal)
    vals = (*Wf.values(), *Yf.values(), *Zf.values(), pf)
    bits = [_dyadic_bits(v) for v in vals]

    scale = None
    if bits and all(b is not None for b in bits):
        k = max(bits)
        if k <= _MAX_SCALE_BITS:
            scale = 1 << k

    if scale is not None and all((v * scale).denominator == 1 for v in vals):
        return ({e: int(v * scale) for e, v in Wf.items()},
                {n: int(v * scale) for n, v in Yf.items()},
                {S: int(v * scale) for S, v in Zf.items()},
                int(pf * scale), 0)
    return Wf, Yf, Zf, pf, F(0)


def _load_tracker(edges, variables, in_cut, val, Y, Z, ZERO):
    """(load, put) -- each variable's cut computed ONCE, load kept by delta.

    `load(e)` used to sum over every dual variable for every edge, and the
    shed loop called it TWICE per edge per pass: O(passes x |E| x |vars|),
    rebuilt from scratch each pass. A variable's cut never changes -- y_n
    loads the edges incident to n, z_S the edges crossing S -- so it is
    computed once here and the load is maintained incrementally, paying
    O(|cut|) per change instead of O(|E| x |vars|) per pass. Measured 3.13x
    on refused shots, and the ratio grows with syndrome weight, which is
    what removing a quadratic term should look like.

    NOTHING ABOUT THE ANSWER CHANGES: the caller keeps the same spend
    order, the same max(over, edge) tie-break, the same visit order in
    absorb, and the same arithmetic. `put` mutates the caller's Y/Z exactly
    as the inline version did.

    Extracted for the same reason as `_to_common_scale`: this is
    BOOKKEEPING, and leaving it inline pushed the repair past the 90-line
    review limit, burying the shed/absorb/verify argument a reviewer is
    there to read.
    """
    cut = {vk: [e for e in edges if in_cut(vk[0], vk[1], e)]
           for vk in variables}
    load_of = {e: ZERO for e in edges}
    for vk in variables:
        v = val(*vk)
        if v:
            for e in cut[vk]:
                load_of[e] += v

    def load(e):
        return load_of[e]

    def put(kind, k, v):
        vk = (kind, k)
        delta = v - val(kind, k)
        (Y if kind == "y" else Z)[k] = v
        if delta:
            for e in cut[vk]:
                load_of[e] += delta

    return load, put


def _repair_and_verify(W, Y, Z, fired_set, primal, order=0):
    """SHED infeasibility, ABSORB the shortfall, VERIFY from scratch.

    Split out of `_exact_optimality` because the standards gate was right:
    at 127 logic lines the most security-critical function in this tree had
    grown too long to review, and length is where logic hides. The argument
    for why this is sound lives in `_exact_optimality`'s docstring.
    """
    # O-2 lives in `_to_common_scale`: one exact domain for W, Y, Z
    # and the primal, integers when the inputs allow it and exact
    # Fractions when they do not. Nothing outside this function
    # sees the representation -- the caller hands in fresh dict
    # copies per spend order and reads only (ok, why) back.
    W, Y, Z, primal, ZERO = _to_common_scale(W, Y, Z, primal)

    # A variable is (kind, key) and its CUT is the edges it loads: y_n loads
    # every edge incident to n, z_S every edge crossing S. Only y_n for FIRED
    # n and every z_S pay into the objective -- a potential on an unfired node
    # costs load and buys nothing, which is why shedding spends those first.
    def in_cut(kind, k, e):
        return (k in e) if kind == "y" else ((e[0] in k) != (e[1] in k))

    def pays(kind, k):
        return (k in fired_set) if kind == "y" else True

    def zkey(s):
        return sorted(s)

    variables = ([("y", n) for n in sorted(Y)]
                 + [("z", S) for S in sorted(Z, key=zkey)])

    def val(kind, k):
        return Y[k] if kind == "y" else Z[k]

    edges = sorted(W)

    # O-1 lives in `_load_tracker`: cuts computed once, load kept by
    # delta. `put` mutates the Y/Z handed in, exactly as before.
    load, put = _load_tracker(edges, variables, in_cut, val,
                              Y, Z, ZERO)

    def objective():
        return (sum((Y[n] for n in sorted(Y) if n in fired_set), ZERO)
                + sum((Z[S] for S in sorted(Z, key=zkey)), ZERO))
    # Deterministic spend order: free variables first, then a total order on
    # the key. Both checkers must make the SAME repair or conformance is a lie.
    spend = _spend_order(variables, pays, val, order)

    # --- 1. SHED infeasibility -----------------------------------------
    # Reducing a variable lowers the load on every edge of its cut and raises
    # it nowhere, so a feasible edge can never become violated. Each pass
    # clears at least the worst violated edge and none regress, so this
    # terminates within len(edges) passes: the bound is a proof, not a guess.
    for _ in range(len(edges) + 1):
        bad = [(e, load(e) - W[e]) for e in edges if load(e) > W[e]]
        if not bad:
            break
        e, over = max(bad, key=lambda t: (t[1], t[0]))
        for kd, k in spend:
            if over == 0:
                break
            if not in_cut(kd, k, e):
                continue
            cur = val(kd, k)
            # odd-set duals must stay >= 0; node potentials may go negative
            take = min(cur, over) if kd == "z" else over
            if take <= 0:
                continue
            put(kd, k, cur - take)
            over -= take
        if over > 0:
            return False, ("cannot restore dual feasibility without driving an "
                           "odd-set dual negative -- certificate is malformed")

    # --- 2. ABSORB the remaining shortfall -------------------------------
    short = primal - objective()
    if short < 0:
        return False, ("dual exceeds primal even after feasibility was restored "
                       "-- impossible under weak duality, so the certificate is "
                       "malformed")
    if short > 0:
        # SPREAD the shortfall. Requiring ONE variable to cover all of it
        # refused 77 of 9,841 hardware shots whose gap was measured at
        # EXACTLY 0.0 -- an attaining dual existed and the search simply
        # failed to build it. A certificate refused because the checker's
        # search was too weak is not the same as one that is unprovable,
        # and reporting the first as the second understates the result.
        # Each variable contributes up to the slack across its whole cut.
        for kd, k in variables:
            if short == 0:
                break
            if not pays(kd, k):
                continue
            cut = [e for e in edges if in_cut(kd, k, e)]
            if not cut:
                continue
            room = min(W[e] - load(e) for e in cut)
            take = room if room < short else short
            if take <= 0:
                continue
            put(kd, k, val(kd, k) + take)
            short -= take
        if short > 0:
            return False, ("no dual variable has the slack to close the gap, so "
                           "the shortfall is REAL and this correction is not "
                           "proven minimum-weight")

    # --- 3. VERIFY the repaired dual from scratch, exactly ---------------
    # THIS IS THE WHOLE GUARANTEE. Steps 1-2 are a SEARCH and are allowed to
    # be wrong; only this decides. And the search cannot launder a bad answer
    # even in principle: by LP duality a feasible dual attaining U exists IFF
    # U is optimal, so on a suboptimal primal step 2 provably cannot find the
    # slack. That is a theorem, not a heuristic that usually catches things.
    for e in edges:
        if load(e) > W[e]:
            return False, "repaired dual is infeasible"
    if any(Z[S] < 0 for S in Z):
        return False, "repaired dual has a negative odd-set term"
    if objective() != primal:
        return False, "repaired dual does not attain the primal"
    return True, ("a feasible dual ATTAINS the primal in exact arithmetic, "
                  "after exact repair of the float rounding in the shipped dual")


def _parse_duals(cert, primal, tol):
    """((y, z), error) -- every dual value read, checked FINITE, z >= 0.

    Split out of `check` when the C-3 refusal pushed it past the 90-line
    review limit. Length is where logic hides, and this is the most
    security-critical function in the tree, so the limit is not one to
    waive here of all places.

    `_finite` runs before any comparison sees a value: every comparison
    against NaN is False, so a non-finite dual walked through every
    refusal in this file and out the ACCEPT path -- fail-open in a
    checker whose whole purpose is to fail closed (external audit
    2026-08-04, finding 5).

    `y` is parsed and returned even though `check` now refuses a
    non-empty node_potentials outright: the parse is what makes a
    malformed potential a REFUSAL rather than a crash, and keeping it
    means the guard above is the only thing deciding policy.
    """
    y = {}
    for k, v in cert.node_potentials.items():
        fv, err = _finite(v, f"node potential y[{k}]")
        if err:
            return None, CheckResult(False, err, primal_weight=primal)
        y[int(k)] = fv
    z = {}
    for S, zz in cert.blossom_duals.items():
        fz, err = _finite(zz, "blossom dual z")
        if err:
            return None, CheckResult(False, err, primal_weight=primal)
        z[frozenset(int(x) for x in S)] = fz
    for S, zz in z.items():
        if zz < -tol:
            return None, CheckResult(False,
                                     f"negative blossom dual z={zz}",
                                     primal_weight=primal)
    return (y, z), None


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

    The four checks are four named functions, so each can be read against
    its own theorem instead of as one 131-line block. Each carries the
    failure history that produced it -- the d=7 T-join correction, the
    near-quadratic moat walk, the boundary-as-real-vertex fix -- because
    that history is why the code looks the way it does.
    """
    built, err = _collapse_graph(edges, tol)
    if err is not None:
        return err
    W, incident = built

    checks: dict = {}

    # --- 1. the correction is a valid T-JOIN: odd-degree nodes == syndrome.
    got, err = _tjoin_primal(cert, W, syndrome)
    if err is not None:
        return err
    primal, fired = got
    checks["t_join_valid"] = True
    checks["primal_weight"] = primal

    # --- 2. NODE POTENTIALS ARE NOT PART OF THE T-JOIN CUT DUAL.
    #
    # THIS IS THE FIX FOR THE CONFIRMED SOUNDNESS BREAK [audit C-3], and
    # it is a refusal rather than a repair because the break is in the
    # FORMULATION, not in the arithmetic.
    #
    # The module docstring above states the boundary: QEC decoding is a
    # minimum-weight T-JOIN, whose dual is a non-negative packing of
    # T-odd CUTS. Mixing node potentials y with odd-set duals z gives the
    # perfect-matching dual on the METRIC CLOSURE -- a strict relaxation,
    # in which a NEGATIVE potential absorbs slack on edges away from T.
    #
    # `_exact_optimality`'s load-bearing argument is "by LP duality a
    # feasible dual attaining U exists IF AND ONLY IF U is optimal". That
    # theorem is about the cut-packing dual. It does not hold for the
    # mixed system, so the exact repair INHERITED the break instead of
    # closing it: a forged certificate of weight 20 was accepted where
    # the true optimum was 2, and both the legacy check and the exact
    # tier accepted it.
    #
    # Refusing potentials collapses this checker onto exactly the
    # contract `matching_cert_strict.check_exact_cut` enforces, which
    # makes the theorem applicable again.
    #
    # THE BLAST RADIUS IS ZERO, measured rather than assumed: every
    # producer in this tree (certifying_decoder, the cert_format wire
    # format, forgeries, the prior-art experiments) passes `{}`, and
    # there is not one non-empty node_potentials across evidence/,
    # conformance/ or release/. The field is dead except to an attacker.
    if cert.node_potentials:
        return CheckResult(
            False,
            "node potentials are not part of the T-join cut dual; submit "
            "a pure T-odd cut packing (see matching_cert_strict). Mixing "
            "y with z is the metric-closure relaxation, in which a "
            "feasible dual can exceed the true T-join optimum.",
            primal_weight=primal)

    # --- 3. dual feasibility on EVERY edge (this is the O(nnz) core)
    got, err = _parse_duals(cert, primal, tol)
    if err is not None:
        return err
    y, z = got
    load = _moat_load(z, incident)
    err = _dual_feasible_edges(W, load, y, primal, tol)
    if err is not None:
        return err
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
    # EXTERNAL AUDIT 2026-08-04, finding 4 (CONFIRMED). This tolerance
    # made "OPTIMAL" mean "within 1e-6 of optimal". The auditor built the
    # witness: edges (0,1)=1.0000005 and 0-B=1-B=0.5, so the true optimum
    # is 1.0 via the boundary, yet the direct edge was ACCEPTED as optimal
    # with a feasible dual and gap 5e-7. A provably suboptimal correction,
    # certified. The claim "exit zero proves exact minimum weight" was
    # false on the float path, and no amount of shrinking the constant
    # fixes it -- any epsilon > 0 admits a counterexample scaled to it.
    #
    # ACCEPTANCE IS EXACT. `_exact_optimality` re-derives primal and dual in
    # exact rational arithmetic and must return proven=True; the float `gap`
    # below is an ADDITIONAL hurdle, never a substitute, so a generous
    # tolerance can only cause a REFUSAL and never an acceptance.
    #
    # TWO STALE COMMENT BLOCKS LIVED HERE UNTIL 2026-08-10, and they
    # contradicted each other in the sole accept path. One announced the fix
    # but named a helper that does not exist under that name (it is
    # `_exact_optimality`). The other, directly below it, still described the
    # audit finding as open and this checker as returning only
    # optimal-to-within-a-tolerance -- accurate before the fix, false after,
    # and an auditor reading it would have concluded this accept path is
    # float-tolerant when the code one line down requires `proven`.
    # A test now pins both the exact gate and the absence of that stale
    # wording: tests/test_exactness_invariants.py.
    #
    # The history is worth keeping, because it is why the exact gate exists:
    # accepting on `gap <= 1e-6` certified provably suboptimal answers.
    # External audit 2026-08-04, finding 4, with the witness -- edges
    # (0,1)=1.0000005 and 0-B=1-B=0.5, true optimum 1.0, yet the direct edge
    # certified with a feasible dual and gap 5e-7. Any epsilon > 0 admits a
    # counterexample scaled to it, so shrinking the constant was never a fix.
    proven, why = _exact_optimality(W, cert, y, z, fired)
    if gap > 1e-6 or not proven:
        return CheckResult(
            False,
            f"NOT PROVEN OPTIMAL: gap {gap:.9f} (primal {primal:.9f}, dual "
            f"{dual_obj:.9f}). The matching may well BE optimal -- this "
            f"certificate does not show it. A dual without odd-set (blossom) "
            f"terms cannot always close the gap.",
            primal_weight=primal, dual_objective=dual_obj, gap=gap,
            checks=checks,
            # *** THE ONLY REFUSAL WORTH ESCALATING A LADDER FOR. ***
            # The correction is valid and the dual is feasible; all that
            # is missing is a tighter packing, which a richer candidate
            # family can supply. `certifying_decoder` recognised this
            # case by searching this sentence for "NOT PROVEN OPTIMAL"
            # -- one rewording away from climbing the ladder three times
            # for a defect no rung can repair, or from not climbing it
            # at all when it would have helped.
            only_obstruction_is_gap=True)

    # --- 4. complementary slackness: tight on every matched edge
    err = _complementary_slackness(cert.matched_edges, W, y, z, primal)
    if err is not None:
        return err
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
        f"OPTIMAL: {why}, so weak duality makes both optimal -- proven "
        f"in exact dyadic-rational arithmetic, with no tolerance in the "
        f"acceptance decision, and without invoking any solver",
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
