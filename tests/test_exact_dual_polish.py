r"""The exact dual polish must close ROUNDING, and be unable to close TRUTH.

Context: the exact acceptance gate (2026-08-14) refuses certificates whose
float duals fall ulp-short of the correction's weight. That refusal is
correct but was landing on ~6% of integral-primal certificates -- pure dual
rounding, not pool insufficiency -- so the producer now re-solves its own
facet pool in exact rationals (`exact_dual_polish`, via the generalized
`exact_weighted_lp`) whenever the safe bound falls short of an integral
primal. Measured before wiring: 23/23 refused certificates closed.

The dangerous edit is the same one every repair risks: a polish that can
manufacture acceptance for a SUBOPTIMAL correction is a rubber stamp. By LP
duality the exact optimum over any facet pool is at most the true integer
optimum, so on a suboptimal primal the polished bound must fall short and
the checker must still refuse. The test for that direction is the one this
file exists for.
"""

from __future__ import annotations

import pathlib
import random
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from oneq.exact_lp import exact_cut_packing_lp, exact_weighted_lp  # noqa: E402
from oneq.qldpc_cert import exact_dual_polish, feldman_dual        # noqa: E402
from oneq.qldpc_check import QldpcCertificate, check_qldpc         # noqa: E402


def test_exact_weighted_lp_solves_a_known_instance_exactly():
    """max 3x + 2y s.t. x + y <= 4, x <= 3, x,y >= 0 -> (3,1), value 11."""
    z, val, _it = exact_weighted_lp([[1, 1], [1, 0]], [4, 3], [3, 2])
    assert z == [Fraction(3), Fraction(1)]
    assert val == Fraction(11)
    assert all(isinstance(v, Fraction) for v in z)


def test_uniform_delegation_is_the_same_simplex():
    """exact_cut_packing_lp now delegates; its answers must not move."""
    z, val, _it = exact_cut_packing_lp([[1, 1], [1, 0]], [4, 3])
    z2, val2, _it2 = exact_weighted_lp([[1, 1], [1, 0]], [4, 3], [1, 1])
    assert z == z2 and val == val2 == Fraction(4)


def test_polish_closes_a_dual_that_is_short_by_rounding():
    """A dual an ulp short on a genuinely optimal primal must close."""
    checks = [(0, 1), (0, 2)]
    weights = {0: 1.0, 1: 0.5, 2: 0.5}
    syn = [1, 1]
    ulp = 2.0 ** -53
    short = QldpcCertificate(
        error_support=(1, 2),
        facet_duals=((0, (), 0.5 - ulp), (1, (), 0.5 - ulp)),
        box_duals=())
    r0 = check_qldpc(checks=checks, weights=weights, syndrome=syn, cert=short)
    assert not r0.accepted, "the short dual should refuse before polish"
    p = exact_dual_polish(checks, weights, syn, short)
    assert p is not None
    r1 = check_qldpc(checks=checks, weights=weights, syndrome=syn, cert=p)
    assert r1.accepted, f"polish failed to close pure rounding: {r1.reason}"
    assert r1.checks.get("exact_gap_zero") is True


def test_polish_cannot_launder_a_suboptimal_correction():
    """THE load-bearing direction. The witness primal {0} weighs 1.0000005
    while {1,2} weighs 1.0; the exact LP optimum over ANY facet pool is at
    most the true optimum 1.0, so no polish can reach 1.0000005 and the
    checker must still refuse."""
    checks = [(0, 1), (0, 2)]
    weights = {0: 1.0000005, 1: 0.5, 2: 0.5}
    syn = [1, 1]
    sub = QldpcCertificate(
        error_support=(0,),
        facet_duals=((0, (), 0.5), (1, (), 0.5)),
        box_duals=())
    p = exact_dual_polish(checks, weights, syn, sub)
    for cand in (c for c in (sub, p) if c is not None):
        r = check_qldpc(checks=checks, weights=weights, syndrome=syn,
                        cert=cand)
        assert not r.accepted, (
            "THE POLISH LAUNDERED A SUBOPTIMAL CORRECTION -- the exact LP "
            f"bound must top out at 1.0, reason: {r.reason}")


def test_polish_refuses_malformed_facets_rather_than_repairing_them():
    """A facet naming a nonexistent check is a forgery-shaped input; the
    polish must return None, never a 'repaired' pool."""
    cert = QldpcCertificate(error_support=(0,),
                            facet_duals=((99, (), 0.5),), box_duals=())
    assert exact_dual_polish([(0, 1)], {0: 1.0, 1: 1.0}, [1], cert) is None


def test_every_integral_primal_certificate_now_verifies():
    """The restored producer-checker contract, on a fresh ensemble: every
    certificate feldman_dual emits beside an integral primal is accepted
    by the exact-gated checker. This is the property the producer
    equivalence gate enforces at 146/146; here it is a unit test with its
    own seed so a regression names the instance."""
    rng = random.Random(20260815)
    tried = 0
    for _t in range(40):
        n = rng.randint(6, 10)
        m = rng.randint(3, max(4, n // 2))
        checks = [tuple(sorted(rng.sample(range(n),
                                          rng.randint(2, min(4, n)))))
                  for _ in range(m)]
        weights = {i: round(rng.uniform(0.5, 3.0), 3) for i in range(n)}
        te = set(rng.sample(range(n), rng.randint(0, n // 2)))
        syn = [len(te & set(c)) % 2 for c in checks]
        cert, _b, primal = feldman_dual(checks, weights, syn,
                                        return_primal=True, rpc_rounds=4)
        if cert is None or primal is None:
            continue
        tried += 1
        r = check_qldpc(
            checks=checks, weights=weights, syndrome=syn,
            cert=QldpcCertificate(error_support=tuple(primal),
                                  facet_duals=cert.facet_duals,
                                  box_duals=cert.box_duals))
        assert r.accepted, (
            f"producer emitted a certificate its own checker refuses: "
            f"{r.reason}")
    assert tried >= 15, f"only {tried} instances exercised the contract"
