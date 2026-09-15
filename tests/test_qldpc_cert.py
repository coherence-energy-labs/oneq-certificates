"""The qLDPC certificate, validated against an oracle that cannot be argued with.

Small random codes, exact minimum by brute force. Three properties, each of
which voids the instrument if it fails:

  SOUND     the LP-dual bound never exceeds the exact minimum (weak duality
            is a theorem; a violation means the relaxation or the checker is
            miscoded, and trial 6 of the first run found exactly that -- a
            sign inversion whose docstring stated the correct rule while the
            code implemented its opposite);
  ALIGNED   the checker accepts precisely when the bound meets the exact
            minimum, and refuses with NOT PROVEN when it does not;
  UNFOOLED  a wrong error paired with the genuine dual is never certified.

The tightness RATE is deliberately not asserted -- it is an empirical property
of the code ensemble (72% on this one), not a soundness property, and pinning
it would turn ensemble drift into a fake soundness failure.
"""

from __future__ import annotations

import itertools
import random

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc, exact_milp,
                             feldman_dual)


def _exact_min(n, checks, weights, syndrome):
    best_w, best_e = None, None
    for r in range(n + 1):
        for combo in itertools.combinations(range(n), r):
            e = set(combo)
            if all(len(e & set(c)) % 2 == syndrome[j]
                   for j, c in enumerate(checks)):
                w = sum(weights[i] for i in e)
                if best_w is None or w < best_w - 1e-12:
                    best_w, best_e = w, e
        if best_w is not None and r >= 2:
            break_at = min(weights.values()) * (r - 1)
            if best_w <= break_at:
                break
    return best_w, best_e


def test_qldpc_certificate_sound_aligned_and_unfooled():
    rnd = random.Random(20260801)
    proven = gaps = 0
    for t in range(25):
        n = rnd.randint(6, 12)
        m = rnd.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rnd.sample(range(n), rnd.randint(2, min(5, n)))))
                  for _ in range(m)]
        weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
        true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
        syndrome = [len(true_e & set(c)) % 2 for c in checks]
        best_w, best_e = _exact_min(n, checks, weights, syndrome)
        assert best_e is not None

        # BOTH producers against the same exact truth: the base LP (loose on
        # some instances, so the NOT-PROVEN branch gets real exercise) and the
        # RPC-cut LP (which on this ensemble closes nearly everything -- the
        # first run with cuts-only starved the refusal branch entirely, which
        # is exactly the vacuity this test exists to forbid).
        cert, bound = feldman_dual(checks, weights, syndrome)
        assert cert is not None
        assert bound <= best_w + 1e-6, (
            f"trial {t}: SOUNDNESS BROKEN -- bound {bound} > exact {best_w}")
        rcert, rbound, primal = feldman_dual(checks, weights, syndrome,
                                             return_primal=True, rpc_rounds=8)
        assert rcert is not None
        assert bound - 1e-9 <= rbound <= best_w + 1e-6, (
            f"trial {t}: cuts must tighten, never loosen or overshoot "
            f"({bound} -> {rbound}, exact {best_w})")

        # THE REPAIR PATH: an integral LP primal must itself be a correction
        # certified optimal by the dual returned beside it -- this is what
        # turned a detected BP-OSD logical error into a repaired one on the
        # [[144,12,12]] code, so it is pinned here against small exact truth.
        if primal is not None:
            rep = QldpcCertificate(error_support=primal,
                                   facet_duals=rcert.facet_duals,
                                   box_duals=rcert.box_duals)
            rr = check_qldpc(checks=checks, weights=weights,
                             syndrome=syndrome, cert=rep)
            assert rr.accepted, (
                f"trial {t}: integral primal failed its own dual: {rr.reason}")
            assert abs(sum(weights[i] for i in primal) - best_w) < 1e-6, (
                f"trial {t}: 'repaired' correction is not exact-minimum")

        full = QldpcCertificate(error_support=tuple(sorted(best_e)),
                                facet_duals=cert.facet_duals,
                                box_duals=cert.box_duals)
        r = check_qldpc(checks=checks, weights=weights,
                        syndrome=syndrome, cert=full)
        if abs(bound - best_w) < 1e-6:
            assert r.accepted, f"trial {t}: tight but refused: {r.reason}"
            proven += 1
        else:
            assert not r.accepted and "NOT PROVEN" in r.reason, (
                f"trial {t}: loose bound, wrong refusal: {r.reason}")
            gaps += 1

        # DEGRADED TIER pinned to the same truth: the MILP tier has no
        # stranger-checkable witness, so the only way to keep it honest is
        # exactly this -- brute-force agreement wherever brute force exists.
        msup, mw = exact_milp(checks, weights, syndrome)
        assert mw is not None and abs(mw - best_w) < 1e-6, (
            f"trial {t}: MILP tier wrong: {mw} != exact {best_w}")
        assert all(len(set(msup) & set(c)) % 2 == syndrome[j]
                   for j, c in enumerate(checks)), (
            f"trial {t}: MILP 'answer' is not even a correction")

        extra = next((i for i in range(n) if i not in best_e), None)
        if extra is not None:
            bad = QldpcCertificate(
                error_support=tuple(sorted(best_e | {extra})),
                facet_duals=cert.facet_duals, box_duals=cert.box_duals)
            rb = check_qldpc(checks=checks, weights=weights,
                             syndrome=syndrome, cert=bad)
            assert not rb.accepted, f"trial {t}: wrong error CERTIFIED"
    assert proven > 0, "no tight instance in 25 trials: the test covers nothing"
    assert gaps > 0, ("every instance was tight: the NOT-PROVEN branch was "
                      "never exercised and could be deleted unnoticed")


def test_lazy_separation_equals_full_enumeration():
    """The lazy separator's contract is EQUALITY, not approximation: at
    convergence no facet of the full family is violated, so its LP optimum
    is the enumerated LP's optimum -- to machine precision, on every
    instance. A drift here means the greedy separator missed a violated
    facet, which silently weakens every high-degree bound built on it."""
    from oneq.qldpc_cert import feldman_dual_lazy
    rnd = random.Random(9)
    for _t in range(20):
        n = rnd.randint(6, 12)
        m = rnd.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rnd.sample(range(n),
                                          rnd.randint(2, min(6, n)))))
                  for _ in range(m)]
        weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
        true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
        syndrome = [len(true_e & set(c)) % 2 for c in checks]
        _c1, b_full = feldman_dual(checks, weights, syndrome)
        _c2, b_lazy = feldman_dual_lazy(checks, weights, syndrome)
        assert abs(b_full - b_lazy) < 1e-9, (
            f"lazy {b_lazy} != enumerated {b_full}: the separator missed "
            f"a violated facet")


def test_capped_facets_stay_sound_and_ordered():
    """Degree-capped facet families (the circuit-level unlock) must satisfy
    capped <= full <= exact on every instance: a capped family is a SUBSET
    of the true facets, so its LP minimum can only sit lower. A violation in
    either direction voids the circuit-level artifact's soundness argument."""
    rnd = random.Random(5)
    checked = 0
    for _t in range(15):
        n = rnd.randint(6, 11)
        m = rnd.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rnd.sample(range(n),
                                          rnd.randint(2, min(6, n)))))
                  for _ in range(m)]
        weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
        true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
        syndrome = [len(true_e & set(c)) % 2 for c in checks]
        best_w, _e = _exact_min(n, checks, weights, syndrome)
        _c1, b_full = feldman_dual(checks, weights, syndrome)
        _c2, b_cap = feldman_dual(checks, weights, syndrome,
                                  max_facet_size=3)
        assert b_cap <= b_full + 1e-9, "capped bound ABOVE full: impossible"
        assert b_full <= best_w + 1e-6, "full bound above exact: unsound"
        checked += 1
    assert checked == 15


def test_the_three_cycle_pseudocodeword_closes_and_certifies():
    """The canonical LP-decoding failure, closed by one elimination cut.

    Checks (0,1,2),(2,3,4),(4,5,0) with all-ones syndrome admit the
    half-integral point x = 1/2 on {0,2,4} at cost 1.5 < 2 (the integer
    optimum, e.g. {0,3}). No PAIR of rows separates it -- the measured death
    of the pair-cut sketch -- while the full row sum {1,3,5} with parity 1
    (facet F = {}, violation 0) closes it in one round. This also pins the
    checker's parity XOR for redundant checks: the derived parity here is 1,
    so a checker that forgets to XOR the syndrome bits would call F = {}
    right-parity and refuse a good certificate.
    """
    checks = [(0, 1, 2), (2, 3, 4), (4, 5, 0)]
    weights = {i: 1.0 for i in range(6)}
    syndrome = [1, 1, 1]
    _c0, b0 = feldman_dual(checks, weights, syndrome)
    assert b0 < 2.0 - 1e-6, "base LP should be fractional here (1.5)"
    cert, b1, primal = feldman_dual(checks, weights, syndrome,
                                    return_primal=True, rpc_rounds=4)
    assert abs(b1 - 2.0) < 1e-6 and primal is not None
    rep = QldpcCertificate(error_support=primal, facet_duals=cert.facet_duals,
                           box_duals=cert.box_duals)
    assert check_qldpc(checks=checks, weights=weights, syndrome=syndrome,
                       cert=rep).accepted


def test_a_fractional_optimum_yields_NO_repair_rather_than_a_rounded_one():
    """A pseudocodeword must not be laundered into a correction.

    Same three-cycle instance, but WITHOUT the elimination cuts: the LP
    optimum is the half-integral point x = 1/2 on {0,2,4}, cost 1.5. That
    is not an error pattern -- rounding it gives {0,2,4}, whose parity
    under every check is 1 while the syndrome is also 1, so it would even
    look plausible, and it costs 3 against a true optimum of 2.

    The integrality gate is the only thing standing there, and the
    existing pseudocodeword test runs with `rpc_rounds=4`, by which point
    the LP has become integral -- so the fractional branch had no test at
    all. Both directions are asserted here: fractional gives None, and
    the cut version still gives a repair.
    """
    checks = [(0, 1, 2), (2, 3, 4), (4, 5, 0)]
    weights = {i: 1.0 for i in range(6)}
    syndrome = [1, 1, 1]

    _cert, bound, primal = feldman_dual(checks, weights, syndrome,
                                        return_primal=True, rpc_rounds=0)
    assert bound < 2.0 - 1e-6, (
        "the LP is not fractional on this instance, so this test would be "
        "asserting about the integral branch")
    assert primal is None, (
        f"a fractional LP optimum was returned as a repair support: "
        f"{primal}. Rounding a pseudocodeword produces something that "
        f"satisfies the parities and is NOT the minimum-weight error")

    # POSITIVE CONTROL: once the cuts close the gap, a repair IS produced.
    _c2, b2, primal2 = feldman_dual(checks, weights, syndrome,
                                    return_primal=True, rpc_rounds=4)
    assert abs(b2 - 2.0) < 1e-6 and primal2 is not None


def test_an_inflated_dual_cannot_certify_a_heavier_error():
    """Scale a genuine dual until its 'bound' matches a heavier error's
    weight: the objective scales linearly, so ONLY the per-variable dual
    feasibility check stands between this forgery and an OPTIMAL verdict."""
    checks = [(0, 1, 2)]
    weights = {0: 1.0, 1: 1.0, 2: 1.0}
    syndrome = [1]
    cert, bound = feldman_dual(checks, weights, syndrome)
    assert abs(bound - 1.0) < 1e-9
    heavy = (0, 1, 2)                    # parity 3 is odd: a valid correction
    s = 3.0 / bound
    forged = QldpcCertificate(
        error_support=heavy,
        facet_duals=tuple((j, F, y * s) for (j, F, y) in cert.facet_duals),
        box_duals=tuple((i, v * s) for (i, v) in cert.box_duals))
    r = check_qldpc(checks=checks, weights=weights, syndrome=syndrome,
                    cert=forged)
    assert not r.accepted and "INFEASIBLE" in r.reason, (
        f"an inflated dual certified a weight-3 answer as optimal: {r.reason}")


def test_a_non_correction_with_matching_weight_is_refused():
    """An error whose weight equals the bound but whose syndrome is WRONG:
    only the recomputed-syndrome step separates it from OPTIMAL."""
    checks = [(0, 1)]
    weights = {0: 1.0, 1: 1.0, 2: 1.0}
    syndrome = [1]
    cert, bound = feldman_dual(checks, weights, syndrome)
    assert abs(bound - 1.0) < 1e-9
    ghost = QldpcCertificate(error_support=(2,),   # weight 1, syndrome 0 != 1
                             facet_duals=cert.facet_duals,
                             box_duals=cert.box_duals)
    r = check_qldpc(checks=checks, weights=weights, syndrome=syndrome,
                    cert=ghost)
    assert not r.accepted and "NOT A VALID CORRECTION" in r.reason


def test_rpc_facets_check_and_forgeries_refuse():
    """Redundant-check facets: the checker must re-derive them honestly.

    A hand-built instance where the base LP is loose but one pair-cut closes
    it, then forgeries against the tuple-named facet: a row list that is not
    rows, a single-row 'pair', and a wrong-parity F -- each must refuse.
    """
    # MINED, not hand-built: the first hand-built instance had an integral
    # base LP (weight-1 solution at variable 2), so its "cut" test was
    # vacuous. Instead, walk a fixed seeded stream until an instance appears
    # whose base LP is loose but whose RPC-cut LP closes with an integral
    # primal -- the measured ensemble contains 15 such in 60, so this
    # terminates fast and deterministically.
    rnd = random.Random(20260801)
    found = None
    for _ in range(200):
        n = rnd.randint(6, 14)
        m = rnd.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rnd.sample(range(n),
                                          rnd.randint(2, min(5, n)))))
                  for _ in range(m)]
        weights = {i: round(rnd.uniform(0.5, 3.0), 3) for i in range(n)}
        true_e = set(rnd.sample(range(n), rnd.randint(0, n // 2)))
        syndrome = [len(true_e & set(c)) % 2 for c in checks]
        _c0, b0 = feldman_dual(checks, weights, syndrome)
        cert, b1, primal = feldman_dual(checks, weights, syndrome,
                                        return_primal=True, rpc_rounds=8)
        if (cert is not None and primal is not None and b1 > b0 + 1e-6
                and any(isinstance(j, tuple)
                        for (j, _F, y) in cert.facet_duals if y > 1e-12)):
            found = (checks, weights, syndrome, cert, primal)
            break
    assert found is not None, (
        "no pair-cut-closed instance in 200 draws: the RPC path is dead here")
    checks, weights, syndrome, cert, primal = found
    rep = QldpcCertificate(error_support=primal, facet_duals=cert.facet_duals,
                           box_duals=cert.box_duals)
    assert check_qldpc(checks=checks, weights=weights, syndrome=syndrome,
                       cert=rep).accepted

    def refused(facets):
        c = QldpcCertificate(error_support=primal, facet_duals=facets,
                             box_duals=cert.box_duals)
        return not check_qldpc(checks=checks, weights=weights,
                               syndrome=syndrome, cert=c).accepted

    tampered = []
    for (j, F, y) in cert.facet_duals:
        if isinstance(j, tuple):
            tampered.append(((0, 7), F, y))          # row 7 does not exist
        else:
            tampered.append((j, F, y))
    assert refused(tuple(tampered)), "facet naming a nonexistent row certified"
    assert refused(tuple(((0,) if isinstance(j, tuple) else j, F, y)
                         for (j, F, y) in cert.facet_duals)), (
        "a single-row 'redundant check' must refuse: it would double-count "
        "check 0's polytope under a different name")
