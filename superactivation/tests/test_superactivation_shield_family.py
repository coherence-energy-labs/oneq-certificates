"""Two-use superactivation with the shield-dimension family of private Horodecki PPT channels (R142).

Every link of the argument is checked here, and each check is shown able to fail:
  - the family is a PPT channel at p1 = 1/(sqrt d + 1), its d = 2 member is the paper's channel, and it is NOT PPT
    below p1;
  - one use of P_d (x) A_{2d} with the generalized Prop. 1 encoder gives N~_p: perfect recovery when A delivers
    (Knill-Laflamme), a binary symmetric channel with flip probability exactly p when A erases (Helstrom);
  - the two-use code {|++>, |-->} on N~_p has F = (7 - 6p + 2p^2)/8, above 3/4 iff p < (3 - sqrt 7)/2, i.e. d >= 22;
  - one use never exceeds 3/4.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SF = ROOT / "experiments" / "superactivation" / "shield_family"
sys.path.insert(0, str(SF))
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "compile"))
import effective_channel_check as ec  # noqa: E402
import onset_vs_p as ovp  # noqa: E402
import ppt_family as pf  # noqa: E402


def _u(d):
    return pf.hadamard2() if d == 2 else (pf.sylvester(int(math.log2(d))) if d in (4, 8) else pf.fourier(d))


def test_the_d2_member_is_the_papers_channel():
    import sclib
    rho = pf.choi_state(pf.hadamard2(), sclib.P_FLIP)
    ref = sclib.horodecki_choi_state()
    assert np.allclose(np.sort(np.linalg.eigvalsh(rho)), np.sort(np.linalg.eigvalsh(ref)), atol=1e-13)
    assert np.allclose(np.sort(np.linalg.eigvalsh(pf.partial_transpose_B(rho, 4, 4))),
                       np.sort(np.linalg.eigvalsh(pf.partial_transpose_B(ref, 4, 4))), atol=1e-13)


@pytest.mark.parametrize("d", [2, 3, 4, 5, 8])
def test_the_family_is_a_ppt_channel_at_the_boundary(d):
    is_channel, psd, pt = pf.check(d, _u(d), 1 / (math.sqrt(d) + 1))
    assert is_channel and psd > -1e-13 and pt > -1e-12


@pytest.mark.parametrize("d", [2, 3, 4, 8])
def test_the_family_is_NOT_ppt_below_the_boundary(d):
    """negative control: p1 = 1/(sqrt d + 1) is the least flip noise this family allows, not an arbitrary choice"""
    _, _, pt = pf.check(d, _u(d), 0.97 / (math.sqrt(d) + 1))
    assert pt < -1e-5


@pytest.mark.parametrize("d", [2, 3, 4])
def test_one_use_gives_the_flagged_channel_N_p(d):
    nk, kl, helstrom, p = ec.check(d, _u(d))
    assert nk > 0
    assert kl < 1e-12                          # kept branch: perfectly recoverable
    assert helstrom == pytest.approx(p, abs=1e-12)


def test_the_effective_channel_check_fails_without_the_shield_bell_pair():
    """negative control: sending fixed shields |0>|0> instead of Phi^d breaks perfect recovery in the kept branch"""
    d = 3
    D = 2 * d
    p = 1 / (math.sqrt(d) + 1)
    Ks = ec.kraus_from_choi_state(pf.choi_state(pf.fourier(d), p), D, D)
    V = np.zeros((D * D, 2), dtype=complex)
    for r in range(2):
        i = r * d                              # shield index 0 on both sides
        V[i * D + i, r] = 1
    W = np.concatenate([np.kron(K, np.eye(D)) @ V for K in Ks], axis=1)
    G = (W.conj().T @ W).reshape(len(Ks), 2, len(Ks), 2)
    deviation = max(np.abs(G[:, 0, :, 1]).max(), np.abs(G[:, 0, :, 0] - G[:, 1, :, 1]).max())
    assert deviation > 1e-3


@pytest.mark.parametrize("p", [0.0, 0.1, 1 / 6, 0.25, 1 / (1 + math.sqrt(2))])
def test_the_two_use_code_has_the_closed_form_fidelity(p):
    _, F = ovp.decode(ovp.explicit_code_n2(), 2, p, iters=4000)
    assert F == pytest.approx((7 - 6 * p + 2 * p * p) / 8, abs=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("d", [2, 16, 32])
def test_the_encoder_gradient_is_the_fidelitys_linear_functional(n, d):
    """F is linear in the encoder Choi X for fixed decoders, so Tr(X G) = F EXACTLY for G = adjoint_states(decs).
    Until 2026-10-18 apply_dual computed K^dag Y K^T: the identity missed by 0.008-0.24 on these nine cases (and
    by 0.013 at d = 32's optimum), and best()'s see-saw climbed the wrong functional."""
    p = 1 / (math.sqrt(d) + 1)
    rng = np.random.default_rng(10 * n + d)
    X = ovp.random_encoder(n, rng)
    decs, F = ovp.decode(X, n, p, iters=300)
    G = ovp.adjoint_states(decs, n, p)
    assert float(np.real(np.trace(X @ G))) == pytest.approx(F, abs=1e-12)


def test_the_repo_see_saw_now_leaves_the_closed_form_at_d_32():
    """the search the gradient feeds: at d = 32 one restart now beats {|++>, |-->} by > 5e-4 (the optimum is
    two_use_optimum.py's family A, +6.117e-4); with the K^T gradient every restart returned the closed form"""
    p = 1 / (math.sqrt(32) + 1)
    F, X = ovp.best(2, p, 1, 100, seed=5)
    assert F > (7 - 6 * p + 2 * p * p) / 8 + 5e-4
    assert np.linalg.eigvalsh(X)[-2] < 1e-8                     # rank 1: the optimum over CPTP encoders is an isometry


def test_the_two_use_threshold_is_d_22():
    pstar = (3 - math.sqrt(7)) / 2
    assert (7 - 6 * pstar + 2 * pstar ** 2) / 8 == pytest.approx(0.75, abs=1e-15)

    def F(d):
        p = 1 / (math.sqrt(d) + 1)
        return (7 - 6 * p + 2 * p * p) / 8
    assert F(21) < 0.75 < F(22)
    assert F(36) == pytest.approx(303 / 392, abs=1e-15)    # d = 36: p = 1/7 exactly
    assert F(64) == pytest.approx(515 / 648, abs=1e-15)    # d = 64: p = 1/9 exactly


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("f", [0.8, 1.0, 1.25])
def test_the_partial_transpose_has_the_analytic_spectrum(d, f):
    """PT eigenvalues p0/2d^2 +- p1/2d^(3/2) and p1/2d +- p0/2d^(3/2) (and 0): PSD iff p0/p1 = sqrt d exactly
    (derived in the independent pass, 2026-10-14). The minimum is negative on BOTH sides of the boundary."""
    p1 = f / (math.sqrt(d) + 1)
    p0 = 1 - p1
    ev = np.linalg.eigvalsh(pf.partial_transpose_B(pf.choi_state(_u(d), p1), 2 * d, 2 * d))
    a = p0 / (2 * d * d) - p1 / (2 * d ** 1.5)
    b = p1 / (2 * d) - p0 / (2 * d ** 1.5)
    allowed = {round(x, 12) for x in (p0 / (2 * d * d) + p1 / (2 * d ** 1.5), a, p1 / (2 * d) + p0 / (2 * d ** 1.5), b, 0.0)}
    assert {round(x, 12) for x in ev} <= allowed
    assert ev.min() == pytest.approx(min(a, b, 0.0), abs=1e-14)
    assert (ev.min() < -1e-4) == (f != 1.0)


CERTS = pathlib.Path(__file__).resolve().parents[1] / "evidence" / "superactivation" / "shield_family"


@pytest.mark.parametrize("d, crosses", [(20, True), (21, True), (22, True), (19, False)])
def test_the_two_use_certificates_are_bound_and_decide(d, crosses):
    import hashlib
    import json
    name = f"certificate_two_use_d{d}.json" if d != 22 else "certificate_two_use_d22_closed.json"
    cert = json.loads((CERTS / name).read_text(encoding="utf-8"))
    if d == 22:
        V = __import__("two_use_certificate").closed_code()
    else:
        raw = (CERTS / f"two_use_code_d{d}.npy").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == cert["code_sha256"]
        V = np.load(CERTS / f"two_use_code_d{d}.npy")
    w = np.concatenate([V[:, 0], V[:, 1]])
    _, F = ovp.decode(np.outer(w, w.conj()), 2, 1 / (1 + math.sqrt(d)), iters=6000)
    L = cert["certified_lower_bound_float"]
    assert 0 <= F - L < 1e-7                                  # the bound sits just below its own code
    assert cert["exceeds_three_quarters"] is crosses          # d = 19 is the case that does NOT cross
    assert (L > 0.75) is crosses


@pytest.mark.parametrize("p", [0.0, 0.15, 1 / (1 + math.sqrt(2))])
def test_one_use_never_exceeds_three_quarters(p):
    """F_c(N~_p) = 1/2 + (1 - p)/4 <= 3/4; no single-use encoder beats it (and the lemma bounds every PPT channel
    with a 50% erasure by 3/4 at n = 1)"""
    rng = np.random.default_rng(0)
    vals = [ovp.decode(ovp.random_encoder(1, rng), 1, p, iters=2000)[1] for _ in range(5)]
    _, F_id = ovp.decode(np.outer([1, 0, 0, 1.0], [1, 0, 0, 1.0]).astype(complex), 1, p, iters=4000)
    assert F_id == pytest.approx(0.5 + (1 - p) / 4, abs=1e-9)
    assert max(vals) <= 0.5 + (1 - p) / 4 + 1e-9          # no random encoder beats the identity code
    assert 0.5 + (1 - p) / 4 <= 0.75                         # and it never exceeds 3/4 (equality only at p = 0)
