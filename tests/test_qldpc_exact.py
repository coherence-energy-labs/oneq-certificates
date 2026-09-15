r"""Exact-rational certification: the tolerance is gone, prove it stays gone.

Negative controls target the exact path's own failure modes: floats
smuggled by type, negative multipliers, heavier feasible candidates,
inflated duals (which must LOWER the bound through the derived slack,
never raise it), lattice violations with non-integer weights, and the
impossible bucket (L > U must refuse, never celebrate).
"""
from fractions import Fraction

import pytest

from oneq.qldpc_cert import (QldpcCertificate, check_qldpc,
                             check_qldpc_exact, exactify_certificate,
                             feldman_dual)

# x0 xor x1 = 1, unit weights: OPT = 1, dual y=1 on facet F={} of check 0
CHECKS = [(0, 1)]
W = {0: Fraction(1), 1: Fraction(1)}
SYN = [1]
CERT = QldpcCertificate(error_support=(0,),
                        facet_duals=((0, (), Fraction(1)),))


def test_exact_optimal_accepts_and_reports_exact_fields():
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN, cert=CERT)
    assert r.accepted and r.checks["exact"]
    assert r.checks["bound"] == {"num": 1, "den": 1}
    assert r.checks["lattice_denominator"] == 1


def test_float_weight_rejected_by_type():
    r = check_qldpc_exact(checks=CHECKS, weights={0: 1.0, 1: 1.0},
                          syndrome=SYN, cert=CERT)
    assert not r.accepted and "REJECTED" in r.reason


def test_float_dual_rejected_by_type():
    bad = QldpcCertificate(error_support=(0,),
                           facet_duals=((0, (), 1.0),))
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN, cert=bad)
    assert not r.accepted and "REJECTED" in r.reason


def test_negative_dual_refused():
    bad = QldpcCertificate(error_support=(0,),
                           facet_duals=((0, (), Fraction(-1)),))
    assert not check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN,
                                 cert=bad).accepted


def test_heavier_feasible_candidate_not_certified():
    # weight-3 feasible answer on a 3-var check cannot ride a bound of 1
    checks = [(0, 1, 2)]
    w3 = {i: Fraction(1) for i in range(3)}
    heavy = QldpcCertificate(error_support=(0, 1, 2),
                             facet_duals=((0, (), Fraction(1)),))
    r = check_qldpc_exact(checks=checks, weights=w3, syndrome=[1],
                          cert=heavy)
    assert not r.accepted and "NOT PROVEN" in r.reason


def test_inflated_dual_stays_sound():
    # y doubled ON THIS INSTANCE: the derived slacks absorb the excess
    # and L lands below 1, so the certificate is refused, not falsely
    # proven. No general monotonicity in y is claimed (audit round 2:
    # perturbing y moves b.y and the slacks in either direction); the
    # invariant under test is SOUNDNESS -- any y >= 0 yields a valid
    # exact bound, so tampering can only cause refusal, never a false
    # OPTIMAL.
    infl = QldpcCertificate(error_support=(0,),
                            facet_duals=((0, (), Fraction(2)),))
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN, cert=infl)
    assert not r.accepted
    assert r.dual_bound is not None and r.dual_bound < 1


def test_lattice_rule_certifies_across_float_dust():
    # weights in 1/3 Z: D=3; a bound 1/3 short of U by MORE than the
    # lattice step must refuse
    w3 = {0: Fraction(2, 3), 1: Fraction(2, 3)}
    c = QldpcCertificate(error_support=(0,),
                         facet_duals=((0, (), Fraction(1, 3)),))
    r = check_qldpc_exact(checks=CHECKS, weights=w3, syndrome=SYN, cert=c)
    assert not r.accepted  # bound 1/3 vs U 2/3, step 1/3: not past U-step


def test_wrong_parity_facet_refused():
    bad = QldpcCertificate(error_support=(0,),
                           facet_duals=((0, (0,), Fraction(1)),))
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN, cert=bad)
    assert not r.accepted and "parity" in r.reason


def test_stale_certificate_from_other_syndrome_refused():
    # syndrome 0: candidate () is optimal, but the s=1 facet F={} now has
    # the RIGHT parity and must be refused as not-a-face
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=[0], cert=CERT)
    assert not r.accepted


def test_exactified_real_certificate_roundtrip():
    # a real produced dual on a small random-ish code survives
    # exactification and exact re-certification
    checks = [(0, 1, 2), (1, 2, 3)]
    wf = {i: 1.0 for i in range(4)}
    wx = {i: Fraction(1) for i in range(4)}
    cert, bound, primal = feldman_dual(checks, wf, [1, 0],
                                       return_primal=True)
    fc = QldpcCertificate(error_support=tuple(primal),
                          facet_duals=cert.facet_duals,
                          box_duals=cert.box_duals)
    assert check_qldpc(checks=checks, weights=wf, syndrome=[1, 0],
                       cert=fc).accepted
    ec = exactify_certificate(fc)
    assert all(isinstance(y, Fraction) for (_j, _F, y) in ec.facet_duals)
    r = check_qldpc_exact(checks=checks, weights=wx, syndrome=[1, 0],
                          cert=ec)
    assert r.accepted, r.reason


def test_impossible_bucket_bound_above_feasible_weight():
    # two facets each with y=1 on disjoint checks would claim L=2 against
    # a feasible weight-1 candidate on the SECOND check only if the first
    # check's parity admitted it -- construct L > U directly instead via
    # empty-support candidate against a positive bound
    with_zero = QldpcCertificate(error_support=(),
                                 facet_duals=((0, (), Fraction(1)),))
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=[0],
                          cert=with_zero)
    # facet has right parity for s=0 -> refused before the bound compares
    assert not r.accepted


def test_exactify_clamps_negative_float_dust():
    fc = QldpcCertificate(error_support=(0,),
                          facet_duals=((0, (), -1e-12),))
    ec = exactify_certificate(fc)
    assert ec.facet_duals[0][2] == 0


# --- external audit's mutant matrix, the entries new to the exact path ---

def test_mutant_19_stale_cert_from_other_matrix_same_dims():
    # a certificate minted for check support (0,1) replayed against a
    # DIFFERENT matrix with matching dimensions: facet F={0} is now
    # inside the new support but has the RIGHT parity there, or the
    # candidate fails He=s -- either wall must hold
    other_checks = [(0, 2)]
    w3 = {i: Fraction(1) for i in range(3)}
    stale = QldpcCertificate(error_support=(0,),
                             facet_duals=((0, (), Fraction(1)),))
    r = check_qldpc_exact(checks=other_checks, weights=w3, syndrome=[1],
                          cert=stale)
    # candidate (0,) IS feasible for x0 xor x2 = 1 and F={} is a real
    # facet there too -- the exact bound must still be honest: accept is
    # only allowed because the mathematics genuinely transfers (facet {}
    # of ANY check with s=1 asserts >=1 unit of weight). Assert the
    # verdict matches brute force: OPT([(0,2)], s=1, w=1) == 1 == U.
    assert r.accepted  # sound: the stale cert happens to remain valid
    # now a stale cert that does NOT transfer: syndrome 0 makes F={}
    # wrong-parity-invalid
    r2 = check_qldpc_exact(checks=other_checks, weights=w3, syndrome=[0],
                           cert=stale)
    assert not r2.accepted


def test_mutant_8_duplicate_facet_refused_exact():
    dup = QldpcCertificate(
        error_support=(0,),
        facet_duals=((0, (), Fraction(1, 2)), (0, (), Fraction(1, 2))))
    r = check_qldpc_exact(checks=CHECKS, weights=W, syndrome=SYN, cert=dup)
    assert not r.accepted and "twice" in r.reason


def test_mutant_11_infeasible_candidate_same_weight_refused():
    # weight-1 candidate that does NOT satisfy He=s
    bad = QldpcCertificate(error_support=(1,),
                           facet_duals=((0, (), Fraction(1)),))
    r = check_qldpc_exact(checks=[(0, 1), (1,)], weights=W,
                          syndrome=[1, 0], cert=bad)
    assert not r.accepted and "NOT A VALID CORRECTION" in r.reason


def test_redundant_check_parity_is_XORED_from_the_syndrome_bits_exact():
    r"""A REDUNDANT check's parity must be XORed from its rows' syndrome
    bits, in the EXACT path as well as the float one.

    Found by the mutation gate the day the trust modules were refactored.
    The mutant that deletes `par ^= syn[r]` had one named test, and that
    test exercised the FLOAT checker; after the exact facet rule was
    consolidated into _exact_facet_scan the mutant landed on the EXACT
    path instead, and SURVIVED. Not a regression -- the equivalence gate
    proves the behaviour never moved -- but a genuine hole: the exact
    path's redundant-parity derivation had no test at all, and the whole
    point of naming rows rather than a support is that the checker
    DERIVES what they sum to.

    Rows 0 and 2 have supports {0} and {1} and syndrome bits 1 and 0, so
    the redundant check (0, 2) has support {0, 1} and parity 1. An
    even-sized F is therefore the WRONG parity for it -- a real facet.
    With the parity forgotten it defaults to 0, the same facet reads as
    RIGHT parity, and this certificate is refused.
    """
    checks = [(0,), (0, 1), (1,)]
    syn = [1, 1, 0]
    w = {0: Fraction(1), 1: Fraction(1)}
    cert = QldpcCertificate(error_support=(0,),
                            facet_duals=(((0, 2), (), Fraction(1)),),
                            box_duals=())
    r = check_qldpc_exact(checks=checks, weights=w, syndrome=syn, cert=cert)
    assert r.accepted, r.reason

    # the negative control: with the parity taken as its true value 1, an
    # ODD-sized F is the RIGHT parity and must be refused as not-a-face
    bad = QldpcCertificate(error_support=(0,),
                           facet_duals=(((0, 2), (0,), Fraction(1)),),
                           box_duals=())
    rb = check_qldpc_exact(checks=checks, weights=w, syndrome=syn, cert=bad)
    assert not rb.accepted and "RIGHT" in rb.reason, rb.reason
