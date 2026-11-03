r"""The qLDPC CHECKER -- its own module, importing NO producer.

Moved here from qldpc_cert.py after an internal review caught the paper
claiming module-graph independence that the module layout did not deliver.
Now it does: this module contains the certificate types and check_qldpc
only, imports nothing from any producer, and the standards gate enforces
that by the import graph. The producers (feldman_dual, feldman_dual_lazy,
exact_milp) live in qldpc_cert.py and import the TYPES from here -- the
dependency arrow points producer -> checker, never back.

Everything check_qldpc consumes is re-derived from (H, w, s) and the
certificate's own listed support; the producer is never consulted.
Derivation notes inline; the relaxation and dual are stated in
qldpc_cert.py's module docstring and in the paper.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm


@dataclass(frozen=True)
class QldpcCertificate:
    """A Feldman-dual witness: sparse support only, plain data."""

    error_support: tuple[int, ...]              # the decoder's answer
    # (check index j -- an int, or a tuple of row indices for a REDUNDANT
    #  check; sorted subset F of N(j); y >= 0). y admits Fraction because
    # the exact tier REQUIRES exact rationals (check_qldpc_exact refuses
    # floats by type) and the exact dual polish emits them; `float` alone
    # was always wrong for that path, the annotation just never said so.
    facet_duals: tuple[tuple[int | tuple[int, ...], tuple[int, ...],
                             float | Fraction], ...]
    box_duals: tuple[tuple[int, float | Fraction], ...] = ()  # (i, mu >= 0)


@dataclass
class QldpcCheckResult:
    accepted: bool
    reason: str
    primal_weight: float | None = None
    dual_bound: float | None = None
    gap: float | None = None
    checks: dict = field(default_factory=dict)


def _validate_instance(checks, weights, syndrome, tol):
    """Step 0: is the PROBLEM well posed? Returns (parts, error).

    Nothing here concerns the certificate; it establishes that (H, w, s)
    is a legal instance at all, so every later step may assume it.
    """
    m = len(checks)
    supports, dup_err = _canonical_supports(checks)
    if dup_err:
        return None, QldpcCheckResult(False, dup_err)
    syn, syn_err = _binary_syndrome(syndrome)
    if syn_err:
        return None, QldpcCheckResult(False, syn_err)
    if len(syn) != m:
        return None, QldpcCheckResult(False,
                                      "syndrome length != number of checks")
    w = {int(i): float(v) for i, v in weights.items()}
    # NO TOLERANCE ON THE SIGN (mutation score, 2026-09-14). `v < -tol` let a
    # weight of -1e-10 through, and the exact gate behind this screen never
    # re-checks signs: a free variable at -1e-10 made a correction 1e-10 above
    # the optimum "OPTIMAL". A weight's sign is data, not a rounding artefact.
    if any(v < 0 for v in w.values()):
        return None, QldpcCheckResult(False, "negative variable weight")
    return (m, supports, syn, w), None


def _validate_correction(error_support, supports, syn, w, m):
    """Step 1: IS IT A CORRECTION? He = s, recomputed -- the only test
    production systems run today, and the baseline this certificate
    exceeds. Returns (e, primal, error)."""
    e = frozenset(int(i) for i in error_support)
    for i in e:
        if i not in w:
            return None, None, QldpcCheckResult(
                False, f"error uses unknown variable {i}")
    for j in range(m):
        if len(e & supports[j]) % 2 != syn[j]:
            return None, None, QldpcCheckResult(
                False, f"NOT A VALID CORRECTION: check {j} parity mismatch")
    return e, sum(w[i] for i in e), None


def _resolve_facet_check(j, supports, syn, m, primal):
    """Which constraint does a facet belong to, and what is its support
    and parity? Returns ((sup, par, jname, jdisp), error).

    A facet may belong to a SINGLE check j, or to a REDUNDANT check named
    as a tuple of row indices. A redundant check is the GF(2) sum of rows
    of H: every e with He = s satisfies it too, so its local polytope is
    an implied constraint and weak duality is untouched. CRITICALLY, the
    checker re-derives its support (symmetric difference of the named
    rows) and its parity (XOR of the named syndrome bits) from (H, s)
    alone -- the certificate names WHICH rows, never what they sum to, so
    there is nothing here to forge.

    `jdisp` exists to keep messages byte-identical: the single-check path
    canonicalises j to an int before any message can quote it, while the
    redundant path quotes whatever the certificate passed.
    """
    if isinstance(j, (tuple, list, frozenset)):
        rows = tuple(sorted({int(r) for r in j}))
        if len(rows) < 2 or any(not (0 <= r < m) for r in rows):
            return None, QldpcCheckResult(
                False, f"redundant check {rows} is not >=2 valid rows",
                primal_weight=primal)
        sup: frozenset = frozenset()
        par = 0
        for r in rows:
            sup = sup ^ supports[r]
            par ^= syn[r]
        return (sup, par, rows, j), None
    j = int(j)
    if not (0 <= j < m):
        return None, QldpcCheckResult(False, f"facet names check {j} of {m}",
                                      primal_weight=primal)
    return (supports[j], syn[j], j, j), None


def _accumulate_facet_duals(facet_duals, supports, syn, m, primal, tol):
    """Step 2: IS EVERY LISTED FACET REAL, WITH A NON-NEGATIVE DUAL?

    A facet is a pair (j, F) with F inside N(j) and |F| of the WRONG
    parity for s_j -- anything else is not a face of the relaxation, and
    admitting it would let the bound climb arbitrarily, the qLDPC
    analogue of the even-set moat.

    Returns ((load, obj, seen), error).
    """
    load: dict[int, float] = {}
    obj = 0.0
    seen: set[tuple] = set()
    for (j, F, y) in facet_duals:
        y = float(y)
        Fs = frozenset(int(i) for i in F)
        if y < -tol:
            return None, QldpcCheckResult(False, f"negative facet dual y={y}",
                                          primal_weight=primal)
        if y == 0.0:
            continue
        resolved, err = _resolve_facet_check(j, supports, syn, m, primal)
        if err is not None:
            return None, err
        sup, par, jname, jdisp = resolved
        if not Fs <= sup:
            return None, QldpcCheckResult(
                False, f"facet ({jname},{sorted(Fs)}) is not inside its check",
                primal_weight=primal)
        if len(Fs) % 2 == par:
            return None, QldpcCheckResult(
                False, f"facet ({jname},{sorted(Fs)}) has the RIGHT parity -- "
                       f"not a face of the relaxation", primal_weight=primal)
        key = (jname, Fs)
        if key in seen:
            return None, QldpcCheckResult(
                False, f"facet ({jdisp},{sorted(Fs)}) listed twice: ambiguous",
                primal_weight=primal)
        seen.add(key)
        # SIGNS, DERIVED AND THEN CAUGHT WRONG ANYWAY. For
        #     min w.x   s.t.  A x <= b,  0 <= x <= 1
        # the dual constraint at variable i is  -(A^T y)_i - mu_i <= w_i and
        # the bound is  -b.y - sum(mu). The facet row has coefficient +1 on
        # members of F and -1 on the rest, so the LOAD is -y on F-members and
        # +y on the others. The first implementation had exactly the opposite
        # -- while its own docstring stated the correct rule -- and the
        # brute-force self-test refused a tight certificate with DUAL
        # INFEASIBLE on trial 6. The docstring was right, the code was wrong,
        # and only a test with an independently known answer could tell them
        # apart.
        for i in sup:
            coef = -y if i in Fs else y
            load[i] = load.get(i, 0.0) + coef
        obj += y * (1.0 - len(Fs))
    return (load, obj, seen), None


def _accumulate_box_duals(box_duals, primal, tol):
    """The 0 <= x <= 1 box multipliers. Returns (mu, error)."""
    mu: dict[int, float] = {}
    for (i, v) in box_duals:
        i = int(i)
        v = float(v)
        if v < -tol:
            return None, QldpcCheckResult(False, f"negative box dual mu={v}",
                                          primal_weight=primal)
        mu[i] = mu.get(i, 0.0) + v
    return mu, None


def _check_dual_feasible(load, mu, w, primal, tol):
    """Step 3: DUAL FEASIBILITY per variable: load_i - mu_i <= w_i, i.e.
    the reduced cost of every variable stays non-negative. One pass over
    the certificate's touched variables; untouched variables have load 0
    and w_i >= 0 by step 0, so they cannot violate."""
    for i, ld in load.items():
        if ld - mu.get(i, 0.0) > w.get(i, 0.0) + tol:
            return QldpcCheckResult(
                False, f"DUAL INFEASIBLE at variable {i}: "
                       f"{ld - mu.get(i, 0.0):.9f} > w={w.get(i, 0.0):.9f}",
                primal_weight=primal)
    return None


def _q(x) -> Fraction:
    """Lossless widening to exact rational. An IEEE double IS a binary
    rational, so `Fraction(float)` rounds nothing: this changes the TYPE
    without moving the VALUE, which is what lets a float-weighted instance
    be re-decided exactly."""
    if isinstance(x, Fraction):
        return x
    try:
        return Fraction(x)
    except TypeError:                      # numpy scalar, Decimal, ...
        return Fraction(float(x))


def _exact_optimality(cert, supports, syn, w, e):
    """Step 5: RE-DECIDE THE WHOLE DUAL IN EXACT RATIONALS.

    WHY THIS EXISTS -- with the witness that forced it. Until 2026-08-10 the
    accept path was a bare float-tolerance refusal (gap over 1e-6, then
    accept) and nothing else, so acceptance
    [wording note: this sentence deliberately avoids quoting the code
    verbatim -- the equivalence gate's mutations anchor on the literal
    source line, and a docstring carrying the same bytes EATS the
    mutation: prove-sensitive went blind on accept-nonzero-gap because
    the replace hit this comment instead of the code]
    rested on a float tolerance ALONE. That is not a proof, it is a
    measurement, and this instance breaks it:

        checks {0,1} and {0,2}, syndrome (1,1)
        w = {0: 1.0000005, 1: 0.5, 2: 0.5}
        cert: error {0}, facet duals y=0.5 on each empty facet

    The dual is genuinely feasible and its bound is 1.0. The claimed error
    {0} weighs 1.0000005. But {1,2} is also a valid correction and weighs
    1.0, so {0} is NOT minimum weight -- and the old checker returned
    ACCEPTED with the reason "weak duality makes both optimal", because
    5e-7 slipped under 1e-6. It also reported `gap: 0.0`, a hardcoded
    constant in the accept branch, so the receipt CONCEALED the very gap
    that should have refused it. An auditor reading that artifact had no
    way to detect the defect. Same class as external audit 2026-08-04
    finding 4 against `matching_cert`, which was repaired there and left
    standing here.

    Two things must be exact, not one. The obvious one is closure. The
    other is FEASIBILITY: `_check_dual_feasible` allows `load - mu <= w +
    tol`, so a dual that is infeasible by a hair is admitted, and an
    infeasible dual's bound is not a bound at all -- it can be inflated to
    meet a suboptimal primal from below. Tolerance in either test is a way
    to accept a non-proof, so both are redone here over Fraction, where
    `<=` and `==` mean what they say.

    HOW, AND WHY THIS FUNCTION HOLDS NO RULES OF ITS OWN. The first draft
    re-derived facet validity, the loads and the sign conventions inline --
    which would have made this the THIRD copy of "what is a real facet" in
    a module whose own `_exact_facet_scan` docstring records killing the
    second one, and warns that a checker whose central definition exists
    twice is one edit away from certifying two different theorems. So the
    rule is not restated here. This widens the float instance to exact
    rationals and hands it to that single definition.

    The widening is lossless: an IEEE double IS a binary rational. Note
    `_exact_rational` refuses floats BY TYPE, on purpose -- that is the
    exact lane's guard against a tolerance walking in the front door -- so
    the conversion happens here, at the boundary, where it is visible.

    And the scan returns the SAFE bound
        L = sum_r y_r (1 - |F_r|) - sum_i max(0, t_i - w_i)
    whose second term derives box duals that make (y, z) feasible by
    construction. That is exactly the repair the ulp cases need: on this
    repo's corpus 22 of 24 float duals are already exactly feasible and
    exactly tight once widened, and the other 2 are infeasible by one ulp
    -- which is also precisely how much their raw bound OVERSHOOTS the
    primal. Absorbing the violation lowers the bound by the violation and
    lands them exactly on it. Since L <= OPT for ANY y >= 0 over valid
    facets, and the primal is a verified correction, `primal == L` forces
    primal = OPT. Nothing here can accept a bound that is too high.

    Returns (proven, why, gap, exact_weight) with `gap` the EXACT primal-dual
    difference and `exact_weight` the correction's weight as a rational,
    which on acceptance IS the bound as well.
    """
    wq = {i: _q(v) for i, v in w.items()}
    fd = [(j, F, _q(y)) for (j, F, y) in cert.facet_duals]
    L, why = _exact_safe_dual_bound(supports, syn, wq, fd)
    if L is None:
        return False, f"exact facet scan refused this dual: {why}", None, None

    primal = sum((wq[i] for i in e), Fraction(0))
    gap = primal - L
    if gap != 0:
        return False, (f"the safe exact bound does not EQUAL the weight: "
                       f"gap {gap!s} (weight {primal!s}, bound {L!s}); the "
                       f"float gap was inside tol, which proves nothing"), \
            gap, None
    return (True, "exact: safe dual bound equals the correction's weight",
            gap, primal)


def check_qldpc(*, checks, weights, syndrome, cert: QldpcCertificate,
                tol: float = 1e-9) -> QldpcCheckResult:
    """Accept only if the certificate PROVES the error is minimum-weight.

    `checks` is a list of variable-index tuples (the support of each row of
    H); `weights` maps variable index -> w_i >= 0; `syndrome` is the 0/1
    target per check. Nothing the decoder said is trusted: the syndrome of the
    claimed error is recomputed, every facet in the certificate is validated
    as a REAL facet (F inside N(j), wrong parity), every dual sign rule is
    re-derived, and the objective is summed from scratch.

    Each step below is a separate named function, so a reader can check
    one theorem at a time -- which is the entire point of a certifying
    algorithm, and something a 149-line block does not permit. The split
    is behaviour-preserving: tools/checker_equivalence_gate.py proves it
    against the pre-split bytes over a shape-varied corpus, comparing the
    verdict, the reason string, the primal weight and the dual bound.
    """
    parts, err = _validate_instance(checks, weights, syndrome, tol)
    if err is not None:
        return err
    m, supports, syn, w = parts

    e, primal, err = _validate_correction(cert.error_support, supports,
                                          syn, w, m)
    if err is not None:
        return err

    got, err = _accumulate_facet_duals(cert.facet_duals, supports, syn, m,
                                       primal, tol)
    if err is not None:
        return err
    load, obj, seen = got

    mu, err = _accumulate_box_duals(cert.box_duals, primal, tol)
    if err is not None:
        return err
    obj -= sum(mu.values())

    err = _check_dual_feasible(load, mu, w, primal, tol)
    if err is not None:
        return err

    # 4. DOES IT CLOSE? The dual bound never legitimately exceeds the weight
    #    of a FEASIBLE error it is shown; that would mean the relaxation or
    #    this checker is broken, and it is reported as impossible, not as
    #    excellent.
    gap = primal - obj
    if gap < -1e-6:
        return QldpcCheckResult(
            False, f"IMPOSSIBLE: dual bound {obj} exceeds the weight {primal} "
                   f"of a feasible correction", primal_weight=primal,
            dual_bound=obj, gap=gap)
    if gap > 1e-6:
        return QldpcCheckResult(
            False, f"NOT PROVEN OPTIMAL: gap {gap:.9f} "
                   f"(weight {primal:.9f}, bound {obj:.9f})",
            primal_weight=primal, dual_bound=obj, gap=gap)

    # 5. AND DOES IT CLOSE *EXACTLY*? A float tolerance may only ever
    #    REFUSE. Everything above is a fast screen; acceptance is decided
    #    in rational arithmetic, where feasibility and closure are proved
    #    rather than measured. See `_exact_optimality` for the witness that
    #    a tolerance alone certified as optimal.
    proven, why, _gap_q, exact_w = _exact_optimality(cert, supports, syn, w, e)
    if not proven:
        return QldpcCheckResult(
            False, f"NOT PROVEN OPTIMAL (exact): {why}",
            primal_weight=primal, dual_bound=obj, gap=gap)

    # AN ACCEPTED RECEIPT REPORTS THE EXACT VALUES, NOT THE SCREEN'S.
    # `primal` and `obj` above are running float sums, and float addition is
    # not associative: both depend on the ORDER their terms arrive in, and
    # neither is guaranteed to be the correctly rounded total. Publishing
    # them beside `gap=0.0` could therefore ship a receipt whose own two
    # numbers differ while it claims they are equal. The exact gate has just
    # proved weight == bound as rationals, so both fields are rendered from
    # that single value, rounded once. (The same defect, found the other way
    # round, made `matching_cert` recompute a weight one ulp worse than the
    # published receipt it was checking.)
    canon = float(exact_w)
    return QldpcCheckResult(
        True, "OPTIMAL: feasible Feldman dual attains the error's weight, so "
              "weak duality makes both optimal -- LP-dual <= LP-opt <= "
              "integer-opt, verified without any solver",
        # gap is a hardcoded 0.0 ONLY because the exact gate above has just
        # proved it is exactly zero. Before that gate existed this constant
        # was a lie that hid a 5e-7 gap from every artifact it wrote.
        primal_weight=canon, dual_bound=canon, gap=0.0,
        checks={"syndrome_ok": True, "facets_real": len(seen),
                "dual_feasible": True, "exact_gap_zero": True})


def _reduce_instance(checks, syn, S0: frozenset, S1: frozenset):
    """Exact reduced instance after fixing S0 -> 0, S1 -> 1.

    Fixed variables leave every check; each x_i = 1 flips the parity of
    the checks containing i. Pure integer logic -- nothing to round."""
    red_checks = []
    red_syn = []
    for sup, sj in zip(checks, syn):
        keep = tuple(i for i in sup if i not in S0 and i not in S1)
        flip = sum(1 for i in sup if i in S1) & 1
        red_checks.append(keep)
        red_syn.append((int(sj) ^ flip) & 1)
    return red_checks, red_syn


def _exact_facet_scan(supports, syn, w, facet_duals):
    """THE single definition of exact facet validity. Returns
    (L, facets_used, error_reason).

    There were two. This function's logic also existed, line for line but
    with TERSER refusal messages, inside check_qldpc_exact -- so the flat
    exact checker and the branch-and-bound leaf checker each carried
    their own copy of the rule "what is a real facet". They agreed only
    because someone kept them agreeing by hand, and a checker whose
    central definition exists twice is one edit away from certifying two
    different theorems. (This estate has the scar: format() once shipped
    four implementations that disagreed, three of them dropping the spec
    silently.) The duplicate is now gone and the fuller messages won,
    so a tree leaf and a flat certificate refuse an identical facet with
    identical words.

    Validates every facet against the (possibly reduced) instance,
    accumulates loads, derives the safe slacks, and returns L exactly:

        L = sum_r y_r (1 - |F_r|) - sum_i max(0, t_i - w_i)  <=  OPT

    for ANY y >= 0 over valid facets, because the derived z makes (y, z)
    dual-feasible by construction.
    """
    m = len(supports)
    load: dict[int, Fraction] = {}
    by = Fraction(0)
    seen: set[tuple] = set()
    for (j, F, y) in facet_duals:
        y = _exact_rational(y, "facet dual y")
        if y < 0:
            return None, 0, "negative facet dual"
        if y == 0:
            continue
        Fs = frozenset(int(i) for i in F)
        if isinstance(j, (tuple, list, frozenset)):
            rows = tuple(sorted({int(r) for r in j}))
            if len(rows) < 2 or any(not (0 <= r < m) for r in rows):
                return None, 0, (f"redundant check {rows} is not >=2 "
                                 f"valid rows")
            sup: frozenset = frozenset()
            par = 0
            for r in rows:
                sup = sup ^ supports[r]
                par ^= syn[r]
            jname: object = rows
        else:
            j = int(j)
            if not (0 <= j < m):
                return None, 0, f"facet names check {j} of {m}"
            sup, par, jname = supports[j], syn[j], j
        if not Fs <= sup:
            return None, 0, f"facet ({jname},{sorted(Fs)}) escapes its check"
        if len(Fs) % 2 == par:
            return None, 0, (f"facet ({jname},{sorted(Fs)}) has the RIGHT "
                             f"parity -- not a face of the relaxation")
        key = (jname, Fs)
        if key in seen:
            return None, 0, f"facet ({jname},{sorted(Fs)}) listed twice"
        seen.add(key)
        for i in sup:
            load[i] = load.get(i, Fraction(0)) + (-y if i in Fs else y)
        by += y * (1 - len(Fs))
    z_total = Fraction(0)
    for i, t in load.items():
        slack = t - w.get(i, Fraction(0))
        if slack > 0:
            z_total += slack
    return by - z_total, len(seen), None


def _exact_safe_dual_bound(supports, syn, w, facet_duals):
    """The exact safe-dual bound L for validated facets, or None+reason.

    Kept as a two-value wrapper so the branch-and-bound checker and
    certified_bnb call sites are untouched; the rule itself lives in
    _exact_facet_scan, which is the only copy.
    """
    L, _used, err = _exact_facet_scan(supports, syn, w, facet_duals)
    return L, err


def _bnb_target(w, error_support, supports, syn, m, step,
                bound_only_threshold):
    """The value U the leaves must cross. Returns (U, error).

    Two modes. Normally U is the weight of a candidate whose feasibility
    is re-verified here, and the tree proves nothing lighter exists. In
    BOUND-ONLY mode (frame determinacy / distance certificates) there is
    no candidate at all and none is claimed: the tree must prove that
    EVERY solution of this system weighs more than a threshold T. The
    leaf test below compares against U - step, so setting U := T + step
    makes that test exactly L + W1 > T.
    """
    if bound_only_threshold is not None:
        T = _exact_rational(bound_only_threshold, "threshold")
        return T + step, None
    e = frozenset(int(i) for i in error_support)
    for i in e:
        if i not in w:
            return None, QldpcCheckResult(False, f"unknown variable {i}")
    for j in range(m):
        if len(e & supports[j]) % 2 != syn[j]:
            return None, QldpcCheckResult(
                False, f"NOT A VALID CORRECTION: check {j}")
    return sum((w[i] for i in e), Fraction(0)), None


def _bnb_infeasible_leaf(node, red_sup, red_syn, m):
    """A leaf claiming no 0/1 completion exists below this branch.

    It names a single check, or a GF(2) COMBINATION of checks, whose
    reduced support XORs to EMPTY with ODD parity -- an equation reading
    0 = 1, which no assignment satisfies. The row tuple is named by the
    certificate; the combination is re-derived here, never trusted.
    Returns a refusal reason, or None if the leaf is genuine.
    """
    if len(node) != 2:
        return "malformed infeasible leaf"
    rows = (tuple(sorted({int(r) for r in node[1]}))
            if isinstance(node[1], (tuple, list, frozenset))
            else (int(node[1]),))
    if any(not (0 <= r < m) for r in rows):
        return f"infeasible leaf names bad rows {rows}"
    acc: frozenset = frozenset()
    par = 0
    for r0 in rows:
        acc = acc ^ red_sup[r0]
        par ^= red_syn[r0]
    if acc:
        return (f"claimed-infeasible rows {rows} XOR to "
                f"support {sorted(acc)}, not empty")
    if par == 0:
        return f"rows {rows} XOR to 0=0: feasible"
    return None                             # genuinely no completion


def _bnb_bound_leaf(node, red_sup, red_syn, w, S1, U, step):
    """A leaf carrying an exact safe dual for the REDUCED instance.

    The variables already fixed to 1 contribute W1 unconditionally, so
    the subproblem's optimum is at least L + W1. Crossing U - step
    exhausts the objective lattice below U and closes this subtree.
    Returns a refusal reason, or None if the leaf is genuine.
    """
    if len(node) != 2:
        return "malformed bound leaf"
    W1 = sum((w[i] for i in S1), Fraction(0))
    L, err = _exact_safe_dual_bound(red_sup, red_syn, w, node[1])
    if err is not None:
        return err
    if L + W1 > U - step:
        return None
    return (f"leaf bound {L}+{W1} does not cross "
            f"{U}-{step}")


def _bnb_walk(node, S0: frozenset, S1: frozenset, ctx):
    """Walk the branch tree, refusing anything that is not proven.

    Soundness of the WALK is separate from soundness of a LEAF, and this
    is the walk half: branching on i partitions the binary feasible set
    into the disjoint x_i = 0 and x_i = 1 subproblems, so proving both
    children proves the parent -- provided i is not already fixed on this
    path, which is the one way a forger could double-count a weight.
    Every reduced instance is re-derived from the path, never carried.

    ctx is (w, supports, syn, m, U, step, budget); budget is a one-element
    list so the node count is shared across the whole recursion.
    """
    w, supports, syn, m, U, step, budget = ctx
    budget[0] -= 1
    if budget[0] < 0:
        return "node budget exceeded"
    if not (isinstance(node, (tuple, list)) and node):
        return "malformed node"
    kind = node[0]
    if kind == "branch":
        if len(node) != 4:
            return "malformed branch"
        i = int(node[1])
        if i not in w or i in S0 or i in S1:
            return f"branch on invalid/refixed variable {i}"
        r0 = _bnb_walk(node[2], S0 | {i}, S1, ctx)
        if r0 is not None:
            return r0
        return _bnb_walk(node[3], S0, S1 | {i}, ctx)
    red_checks, red_syn = _reduce_instance(
        [tuple(s) for s in supports], syn, S0, S1)
    red_sup = [frozenset(c) for c in red_checks]
    if kind == "infeasible":
        return _bnb_infeasible_leaf(node, red_sup, red_syn, m)
    if kind == "bound":
        return _bnb_bound_leaf(node, red_sup, red_syn, w, S1, U, step)
    return f"unknown node kind {kind!r}"


def check_qldpc_bnb(*, checks, weights, syndrome, error_support,
                    tree, max_nodes: int = 100000,
                    bound_only_threshold=None) -> QldpcCheckResult:
    r"""CERTIFIED BRANCH-AND-BOUND: the solver tier made stranger-checkable.

    A flat dual sometimes cannot close (fractional pseudocodewords).
    Branching on one coordinate destroys the fractional face, and each
    child is AGAIN a syndrome-LP instance -- so the optimality proof for
    the residue is a TREE whose every leaf carries the same exact
    safe-dual certificate as tier 1, and whose checker needs fractions
    and recursion, no solver, no trust.

    tree node forms (plain tuples, no classes to trust):
      ("branch", i, child_zero, child_one)   -- split on variable i
      ("bound", facet_duals)                 -- leaf: exact dual on the
                                                REDUCED instance proves
                                                L + W1 > U - step
      ("infeasible", check_index)            -- leaf: reduced check has
                                                empty support and odd
                                                parity: no completion

    The checker re-derives every reduced instance from the branch path
    (fixing = integer parity flips), validates every leaf bound in
    exact arithmetic with the lattice rule, and verifies the split
    structure by construction of the recursion. Accept => the verified
    feasible candidate is minimum-weight. Every failure is a refusal,
    never a fallback.

    The two leaf rules live in _bnb_infeasible_leaf and _bnb_bound_leaf,
    so the recursion below is only the tree walk: branching is disjoint
    by construction, and what makes a LEAF sound is stated separately
    from what makes the WALK sound.
    """
    m = len(checks)
    supports, dup_err = _canonical_supports(checks)
    if dup_err:
        return QldpcCheckResult(False, dup_err)
    syn, syn_err = _binary_syndrome(syndrome)
    if syn_err:
        return QldpcCheckResult(False, syn_err)
    if len(syn) != m:
        return QldpcCheckResult(False, "syndrome length != checks")
    try:
        w = {int(i): _exact_rational(v, f"w[{i}]")
             for i, v in weights.items()}
        if any(v < 0 for v in w.values()):
            return QldpcCheckResult(False, "negative weight")
        D = lcm(*[v.denominator for v in w.values()]) if w else 1
        step = Fraction(1, D)
        U, err = _bnb_target(w, error_support, supports, syn, m, step,
                             bound_only_threshold)
        if err is not None:
            return err

        ctx = (w, supports, syn, m, U, step, [int(max_nodes)])
        try:
            fail = _bnb_walk(tree, frozenset(), frozenset(), ctx)
        except RecursionError:
            fail = "tree deeper than the recursion limit: refused"
        if fail is not None:
            return QldpcCheckResult(False, f"TREE REFUSED: {fail}",
                                    primal_weight=float(U))
        return QldpcCheckResult(
            True, "EXACT-OPTIMAL BY BRANCH DUALS: every leaf of the "
                  "branch tree crosses the lattice level below the "
                  "candidate in exact arithmetic -- minimum weight "
                  "proven with no solver and no trust",
            primal_weight=float(U),
            checks={"exact": True, "bnb": True,
                    "lattice_denominator": D,
                    "objective": {"num": U.numerator,
                                  "den": U.denominator}})
    except (TypeError, ValueError) as exc:
        return QldpcCheckResult(False, f"REJECTED MALFORMED INPUT: {exc}")


def _binary_syndrome(syndrome):
    """The syndrome is a vector over GF(2): every entry must be the integer
    0 or 1. Masking with & 1 would silently read 2 as 0 and -1 as 1, so a
    malformed instance is refused, never reinterpreted. Returns (syn, error)."""
    syn = []
    for j, b in enumerate(syndrome):
        try:
            v = operator.index(b)
        except TypeError:
            return None, f"syndrome[{j}] is not an integer"
        if v not in (0, 1):
            return None, f"syndrome[{j}] = {v} is not 0 or 1"
        syn.append(int(v))
    return syn, None


def _canonical_supports(checks):
    """Refuse duplicate indices inside any check support.

    The adversarial audit's one genuine hole: a duplicated index means
    'member once' under set semantics but CANCELS under GF(2) matrix
    semantics -- two different theorems from one input. Upstream
    producers emit clean supports; adversarial input is REFUSED, never
    silently reinterpreted. Returns (supports, None) or (None, error).
    """
    sups = []
    for j, c in enumerate(checks):
        t = [int(i) for i in c]
        fs = frozenset(t)
        if len(fs) != len(t):
            return None, (f"check {j} contains duplicate variable "
                          f"indices: ambiguous GF(2) semantics, refused")
        sups.append(fs)
    return sups, None


def _exact_rational(v, name: str) -> Fraction:
    """Admit int/Fraction ONLY. A float here is the tolerance smuggling
    itself back in through the front door -- refused by TYPE, not value."""
    if isinstance(v, bool) or isinstance(v, float):
        raise TypeError(f"{name} must be an exact rational, got {type(v)}")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, Fraction):
        return v
    raise TypeError(f"{name} must be int or Fraction, got {type(v)}")


def check_qldpc_exact(*, checks, weights, syndrome, cert: QldpcCertificate,
                      ) -> QldpcCheckResult:
    r"""EXACT-RATIONAL certification: no tolerance decides truth, anywhere.

    The external adversarial audit's correct objection to the float path:
    a floating dual checked with tolerances is evidence, not a proof. This
    path is the proof. All arithmetic is fractions.Fraction over
    arbitrary-precision integers; floats are rejected by type.

    The bound uses the SAFE-DUAL construction, which is strictly more
    permissive than requiring dual feasibility: for ANY y >= 0 over valid
    facets, with load t_i = sum_r y_r a_{r,i} (a = -1 on F, +1 off F
    within the support), the derived slack z_i = max(0, t_i - w_i) makes
    (y, z) dual-feasible by construction, so

        L = sum_r y_r (1 - |F_r|) - sum_i z_i  <=  OPT

    exactly. The producer's floats are rationalized upstream; nothing
    here trusts them. No monotonicity in y is claimed or needed:
    perturbing y moves b.y and the derived slacks in either direction,
    and soundness holds because EVERY exact nonnegative y is a valid
    dual candidate whose bound is recomputed from scratch here.

    Acceptance is lattice-strengthened: with every weight in (1/D)Z
    (D = lcm of weight denominators), any feasible objective lies on that
    lattice, so L > U - 1/D already pins U as the minimum. Equality is a
    special case. L > U for a verified-feasible candidate is reported as
    instrument failure, never as success.

    The facet rule itself is NOT written here -- it lives once, in
    _exact_facet_scan, shared with the branch-and-bound leaf checker.
    """
    m = len(checks)
    supports, dup_err = _canonical_supports(checks)
    if dup_err:
        return QldpcCheckResult(False, dup_err)
    syn, syn_err = _binary_syndrome(syndrome)
    if syn_err:
        return QldpcCheckResult(False, syn_err)
    if len(syn) != m:
        return QldpcCheckResult(False, "syndrome length != number of checks")
    try:
        w = {int(i): _exact_rational(v, f"w[{i}]")
             for i, v in weights.items()}
        if any(v < 0 for v in w.values()):
            return QldpcCheckResult(False, "negative variable weight")

        e = frozenset(int(i) for i in cert.error_support)
        for i in e:
            if i not in w:
                return QldpcCheckResult(
                    False, f"error uses unknown variable {i}")
        for j in range(m):
            if len(e & supports[j]) % 2 != syn[j]:
                return QldpcCheckResult(
                    False,
                    f"NOT A VALID CORRECTION: check {j} parity mismatch")
        U = sum((w[i] for i in e), Fraction(0))

        L, facets_used, err = _exact_facet_scan(supports, syn, w,
                                                cert.facet_duals)
        if err is not None:
            return QldpcCheckResult(False, err)

        return _exact_verdict(L, U, w, facets_used)
    except (TypeError, ValueError) as exc:
        return QldpcCheckResult(False, f"REJECTED MALFORMED INPUT: {exc}")


def _exact_verdict(L, U, w, facets_used):
    """Does the exact bound reach the last lattice level below U?

    With every weight in (1/D)Z, any feasible objective sits on that
    lattice, so L > U - 1/D leaves no room for a lighter feasible error
    and already pins U as the minimum. L > U cannot happen for a
    verified-feasible candidate and is reported as instrument failure.
    """
    D = lcm(*[v.denominator for v in w.values()]) if w else 1
    step = Fraction(1, D)
    if L > U:
        return QldpcCheckResult(
            False, f"IMPOSSIBLE: exact bound {L} exceeds the weight {U} "
                   f"of a verified-feasible correction",
            primal_weight=float(U), dual_bound=float(L))
    if L > U - step or L == U:
        return QldpcCheckResult(
            True, "EXACT-OPTIMAL: rational safe-dual bound crosses the "
                  "last objective lattice level below the candidate -- "
                  "minimum weight proven in exact arithmetic, no "
                  "tolerance anywhere",
            primal_weight=float(U), dual_bound=float(L),
            gap=float(U - L),
            checks={"exact": True, "lattice_denominator": D,
                    "facets_real": facets_used,
                    "bound": {"num": L.numerator, "den": L.denominator},
                    "objective": {"num": U.numerator,
                                  "den": U.denominator}})
    return QldpcCheckResult(
        False, f"NOT PROVEN: exact bound {L} does not reach the lattice "
               f"level below {U} (step 1/{D})",
        primal_weight=float(U), dual_bound=float(L), gap=float(U - L))


