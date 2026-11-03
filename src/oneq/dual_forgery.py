r"""THE PUBLISHED COUNTEREXAMPLES TO THE DUAL FORM, AS A LIBRARY.

*** THE CENTRAL CLAIM OF THIS REPOSITORY IS ONE IMPLICATION. *** "A
feasible dual attaining the primal proves the correction is
minimum-weight." That implication is about the *T-join cut dual*. The
legacy `matching_cert` object also carried arbitrary node potentials `y`
on original-graph vertices and added fired-node potentials to the dual
objective -- and edge-local feasibility of that MIXED object is not weak
duality for the T-join problem. It is the metric-closure relaxation, in
which a feasible dual can EXCEED the true optimum.

So there is a three-vertex forgery, and it is decisive:

    graph        0--1 = 20      0--2 = 1      2--1 = 1
    syndrome     T = {0, 1}
    truth        the T-join 0--2--1 costs 2
    forgery      y0 = 9, y1 = 9, y2 = -10, z_{ {0} } = 2

    edge 0--1 load  9 +  9 + 2 = 20   <= 20   feasible
    edge 0--2 load  9 - 10 + 2 =  1   <=  1   feasible
    edge 2--1 load  9 - 10     = -1   <=  1   feasible
    objective       9 +  9 + 2 = 20   == the claimed primal

Every local constraint holds and the objective meets the claimed weight,
yet the shipped correction costs 20 against a true optimum of 2. If the
checker accepts this, the decoder's optimality claim is worth nothing.

*** WHY THIS IS NOT A FORGERY-ONLY BATTERY. *** A checker that refused
EVERYTHING would pass a forgery-only test and be indistinguishable from
a correct one -- an instrument that cannot pass is as dark as one that
cannot fail. Every negative arm is paired with a POSITIVE CONTROL on the
same graph: the genuine weight-2 T-join with the exact cut packing
z_{ {0} } = z_{ {1} } = 1, which must be ACCEPTED at primal = dual = 2.
Refusals are matched on their REASON, because refusing the forgery
because the input looked malformed proves nothing about the dual form.

*** WHY IT LIVES IN `src/` AND NOT ONLY IN `challenge/`. *** A released
tarball should be able to attack its own checker, and `challenge/` is
not shipped. `challenge/matching_mixed_dual_forgery.py` is a thin runner
over this module.

Provenance: the forgery is from `research/max-campaign-20260817`
(2026-08-17, draft PR #1), which was never merged -- the fix
(`matching_cert_strict`) landed on main without the counterexample that
motivates it, leaving nothing to notice a regression. The positive
controls and the reason-matching are added here.
"""

from __future__ import annotations

from fractions import Fraction

from .matching_cert import MatchingCertificate, check
from .matching_cert_strict import ExactCutCertificate, check_exact_cut

__all__ = ["EDGES", "SYNDROME", "ARMS", "run", "legacy_mixed_forgery",
           "strict_positive_control", "strict_suboptimal_primal",
           "strict_infeasible_packing", "strict_not_a_tjoin",
           "PARITY_FORGERIES", "parity_forgeries_refused"]

#: 0--2--1 costs 2; the direct edge 0--1 costs 20.
EDGES = [(0, 1, 20.0), (0, 2, 1.0), (2, 1, 1.0)]
SYNDROME = [0, 1]


def legacy_mixed_forgery() -> tuple[bool, dict]:
    """The mixed y+z object. MUST be refused, and for the RIGHT reason."""
    cert = MatchingCertificate(
        matched_edges=((0, 1),),
        node_potentials={0: 9.0, 1: 9.0, 2: -10.0},
        blossom_duals={frozenset({0}): 2.0},
        claimed_weight=20.0,
    )
    got = check(edges=EDGES, syndrome=SYNDROME, cert=cert)
    # THE REASON IS PART OF THE ASSERTION. A refusal on "malformed
    # input" would satisfy a bare `not accepted` test while saying
    # nothing about whether the checker understands the T-join dual.
    named = "node potential" in got.reason.lower()
    return (not got.accepted) and named, {
        "accepted": got.accepted, "reason": got.reason,
        "reason_names_the_defect": named}


def strict_positive_control() -> tuple[bool, dict]:
    """The GENUINE optimum with an exact cut packing. MUST be accepted.

    Cuts {0} and {1} at z = 1 each. Loads: 0--1 sees both (2 <= 20),
    0--2 sees only {0} (1 <= 1, tight), 2--1 sees only {1} (1 <= 1,
    tight). Objective 2 == primal 2, so the packing PROVES optimality.
    """
    cert = ExactCutCertificate(
        matched_edges=((0, 2), (2, 1)),
        cut_duals={frozenset({0}): Fraction(1), frozenset({1}): Fraction(1)},
    )
    got = check_exact_cut(edges=EDGES, syndrome=SYNDROME, cert=cert)
    tight = (got.accepted and got.primal == Fraction(2)
             and got.dual == Fraction(2))
    return tight, {"accepted": got.accepted, "reason": got.reason,
                   "primal": str(got.primal), "dual": str(got.dual)}


def strict_suboptimal_primal() -> tuple[bool, dict]:
    """The weight-20 correction with an honest dual. MUST be refused.

    No T-odd cut packing can reach 20 here, so the honest object cannot
    close the gap -- which is exactly what the forgery faked.
    """
    cert = ExactCutCertificate(
        matched_edges=((0, 1),),
        cut_duals={frozenset({0}): Fraction(1), frozenset({1}): Fraction(1)},
    )
    got = check_exact_cut(edges=EDGES, syndrome=SYNDROME, cert=cert)
    return not got.accepted, {"accepted": got.accepted,
                              "reason": got.reason,
                              "primal": str(got.primal),
                              "dual": str(got.dual)}


def strict_infeasible_packing() -> tuple[bool, dict]:
    """An INFLATED z that breaks an edge constraint. MUST be refused.

    z_{ {0} } = 2 overloads edge 0--2 (load 2 > weight 1). The same
    arithmetic overreach the mixed object hid behind node potentials,
    now stated in the form the strict checker accepts.
    """
    cert = ExactCutCertificate(
        matched_edges=((0, 2), (2, 1)),
        cut_duals={frozenset({0}): Fraction(2), frozenset({1}): Fraction(1)},
    )
    got = check_exact_cut(edges=EDGES, syndrome=SYNDROME, cert=cert)
    return not got.accepted, {"accepted": got.accepted, "reason": got.reason}


def strict_not_a_tjoin() -> tuple[bool, dict]:
    """A correction whose odd set is not T. MUST be refused.

    Edge 0--2 alone leaves {0, 2} odd, not {0, 1}. A checker that only
    compared weights would take it, because it is CHEAPER than the
    truth -- a dual proving optimality of a correction that does not fix
    the syndrome is the worst available failure.
    """
    cert = ExactCutCertificate(
        matched_edges=((0, 2),),
        cut_duals={frozenset({0}): Fraction(1)},
    )
    got = check_exact_cut(edges=EDGES, syndrome=SYNDROME, cert=cert)
    return not got.accepted, {"accepted": got.accepted, "reason": got.reason}


#: THE R177 PARITY FORGERIES (found 2026-10-18 by an LP search over all
#: vertex subsets of random 3-5-vertex graphs; fixed in R177b). Each is a
#: PROVABLY SUBOPTIMAL correction whose dual "attains" its weight using sets
#: that are not T'-odd (T' = T u {BOUNDARY} when |T| is odd), so they are not
#: T-join constraints at all. Before R177b both Python checkers accepted
#: PB1/PB2 and `check` accepted PE1. Rows: (name, edges, T, F, Z, w(F), opt).
PARITY_FORGERIES = (
    # T={1}: 0-1, 0-B weighs 6; 1-B weighs 4. {B,0,1} and {B,1,2} are T-odd
    # by |S n T| but their boundary-free sides {2}, {0} hold no T vertex.
    ("PB1_boundary_side_parity", [(0, 1, 4.0), (0, -1, 2.0), (1, 2, 6.0), (1, -1, 4.0)],
     [1], [(0, 1), (0, -1)],
     {frozenset({1}): Fraction(3), frozenset({0, 1}): Fraction(1),
      frozenset({-1, 0, 1}): Fraction(1), frozenset({-1, 1, 2}): Fraction(1)},
     Fraction(6), Fraction(4)),
    ("PB2_boundary_side_parity",
     [(0, 1, 1.0), (0, 2, 5.0), (0, -1, 2.0), (1, 2, 4.0), (1, 3, 4.0), (1, -1, 2.0),
      (2, 3, 3.0)],
     [3], [(0, 1), (0, -1), (1, 3)],
     {frozenset({3}): Fraction(1, 2), frozenset({2, 3}): Fraction(5, 2),
      frozenset({-1, 3}): Fraction(1), frozenset({1, 2, 3}): Fraction(1, 2),
      frozenset({0, 1, 2, 3}): Fraction(1, 2), frozenset({-1, 0, 1, 3}): Fraction(3, 2),
      frozenset({-1, 1, 2, 3}): Fraction(1, 2)}, Fraction(7), Fraction(6)),
    # |T| even, sets with NO T vertex: 0-1, 0-2 weighs 5; 1-2 alone weighs 4.
    ("PE1_t_even_set_production_only",
     [(0, 1, 1.0), (0, 2, 4.0), (1, 2, 4.0), (2, -1, 1.0)],
     [1, 2], [(0, 1), (0, 2)],
     {frozenset({0}): Fraction(1, 2), frozenset({1}): Fraction(1, 2),
      frozenset({2}): Fraction(1, 2), frozenset({-1}): Fraction(1, 2),
      frozenset({0, 1}): Fraction(3)}, Fraction(5), Fraction(4)),
)


def parity_forgeries_refused() -> tuple[bool, dict]:
    """Every R177 parity forgery MUST be refused by BOTH Python checkers,
    and for the RIGHT reason (the set is T-even), never a gap or a crash."""
    rows = {}
    ok = True
    for name, edges, T, F, Z, _wF, _opt in PARITY_FORGERIES:
        s = check_exact_cut(edges=edges, syndrome=T,
                            cert=ExactCutCertificate(tuple(F), dict(Z)))
        p = check(edges=edges, syndrome=T,
                  cert=MatchingCertificate(tuple(F), {},
                                           {S: float(v) for S, v in Z.items()}))
        # read the checkers' structured T'-odd answer, never a phrase of their `reason`
        good = (not s.accepted and s.checks.get("t_odd") is False
                and not p.accepted and p.checks.get("dual_sets_t_odd") is False)
        ok = ok and good
        rows[name] = {"strict": s.reason, "production": p.reason, "refused": good}
    return ok, {"accepted": not ok,
                "reason": f"{sum(r['refused'] for r in rows.values())}/{len(rows)} "
                          f"parity forgeries refused as T-even by both checkers",
                "forgeries": rows}


#: Positive and negative in ONE list, so a checker stuck on either
#: answer fails at least one arm.
ARMS = (
    ("legacy mixed y+z forgery is REFUSED, naming the defect",
     legacy_mixed_forgery),
    ("genuine optimum with an exact cut packing is ACCEPTED",
     strict_positive_control),
    ("suboptimal primal with an honest dual is REFUSED",
     strict_suboptimal_primal),
    ("infeasible (overloaded) cut packing is REFUSED",
     strict_infeasible_packing),
    ("correction that is not a T-join is REFUSED", strict_not_a_tjoin),
    ("R177 parity forgeries (sets not T'-odd) are REFUSED by both checkers",
     parity_forgeries_refused),
)


def run() -> tuple[bool, list]:
    """(all_arms_correct, per-arm rows). Never raises on a wrong verdict."""
    rows = []
    ok = True
    for name, arm in ARMS:
        good, detail = arm()
        ok = ok and good
        rows.append({"arm": name, "correct": good, **detail})
    return ok, rows
