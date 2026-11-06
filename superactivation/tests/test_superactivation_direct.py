"""Codes on the joint channel (P (x) A)^n directly (R141): the model, the lift of the paper's codes, and the certified
single-use separation F_c(P (x) A) >= 0.66275762870 > 0.64644660940 = F_c(N~), the paper's per-use reduction's optimum.
"""
import hashlib
import io
import json
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIRECT = ROOT / "experiments" / "superactivation" / "direct"
EV = ROOT / "evidence" / "superactivation" / "direct"
sys.path.insert(0, str(DIRECT))
import direct_certificate as dcert  # noqa: E402
import direct_codes as dc  # noqa: E402
from oneq import superactivation_certificate as sc  # noqa: E402


@pytest.mark.parametrize("n", [1, 2])
def test_the_encoder_gradient_is_the_adjoint_of_every_branch(n):
    rng = np.random.default_rng(n)
    SP = dc.p_superop()
    X = dc.random_encoder(n, rng)
    pats = dc.patterns(n)
    assert len(pats) == 2 ** n
    for z in pats:
        R = dc.bob_state(X, n, z, SP)
        D = rng.normal(size=R.shape) + 1j * rng.normal(size=R.shape)
        D = D @ D.conj().T
        assert np.trace(X @ dc.bob_adjoint(D, n, z, SP)) == pytest.approx(np.trace(R @ D), rel=1e-12)
        assert np.real(np.trace(R)) == pytest.approx(2.0, abs=1e-12)          # each branch is trace preserving


def test_a_wrong_adjoint_is_caught():
    """negative control: a dual that omits P breaks the identity (note: conjugating S is NOT a wrong dual -- for a
    Hermiticity-preserving map conj(S)[o,p,i,j] = S[p,o,j,i], the same operator)"""
    rng = np.random.default_rng(7)
    SP = dc.p_superop()
    X = dc.random_encoder(1, rng)
    R = dc.bob_state(X, 1, (0,), SP)
    D = rng.normal(size=R.shape) + 1j * rng.normal(size=R.shape)
    D = D @ D.conj().T
    wrong = dc.bob_adjoint(D, 1, (0,), dc.superop([np.eye(dc.DP, dtype=complex)]))
    assert abs(np.trace(X @ wrong) - np.trace(R @ D)) > 1e-3


def test_the_lifted_paper_code_scores_exactly_its_effective_channel_value():
    """E~ composed into the direct model reproduces N~ at one use: Prop. 1's decoder is optimal for its encoder"""
    X = dc.lift(np.outer([1, 0, 0, 1.0], [1, 0, 0, 1.0]).astype(complex), 1)
    _, F = dc.decode(X, 1, dc.p_superop(), iters=6000)
    p = 1 / (1 + math.sqrt(2))
    assert F == pytest.approx(0.5 + (1 - p) / 4, abs=1e-9)


def test_the_single_use_certificate_is_bound_and_beats_the_effective_channel():
    cert = json.loads((EV / "certificate_direct_n1.json").read_text(encoding="utf-8"))
    wraw = (EV / cert["encoder"]).read_bytes()
    draw = (EV / cert["decoders"]).read_bytes()
    assert hashlib.sha256(wraw).hexdigest() == cert["encoder_sha256"]
    assert hashlib.sha256(draw).hexdigest() == cert["decoders_sha256"]
    X = np.load(io.BytesIO(wraw), allow_pickle=False)
    with np.load(io.BytesIO(draw), allow_pickle=False) as z:
        decs = {pat: np.array(z["z" + "".join(map(str, pat))])[None] for pat in dc.patterns(1)}
    SP = dc.p_superop()
    F = sum(0.5 * float(np.real(np.einsum("bij,bji->", dc.bob_state(X, 1, pat, SP)[None], decs[pat]))) / 4
            for pat in dc.patterns(1))
    L = cert["certified_lower_bound_float"]
    assert 0 <= F - L < 1e-9                                  # the bound sits just below its own pair
    p = 1 / (1 + math.sqrt(2))
    assert L > 0.5 + (1 - p) / 4 + 0.016                      # the effective channel's single-use optimum
    assert L < 0.75                                           # and below the one-use ceiling


def test_the_certificate_runs_and_refuses_a_non_isometric_encoder():
    X = np.load(EV / "direct_encoder_n1.npy")
    decs, Fp = dc.decode(X, 1, dc.p_superop(), iters=3000)
    res = dcert.certify(1, X, decs, log=None)
    assert 0 <= Fp - res["certified_lower_bound_float"] < 1e-9
    with pytest.raises(sc.CertificateRefusal, match="rank 1"):
        dcert.certify(1, dc.random_encoder(1, np.random.default_rng(3), rank=2), decs, log=None)


def test_the_exact_kraus_operators_match_the_transcription():
    flint = sc._flint()
    PK = dcert.exact_p_kraus(flint)
    NZ = dcert.kraus_pattern_and_check(flint, PK)
    assert len(NZ) == 6 and all(len(col) <= 1 for K in NZ for col in K)    # at most one nonzero per column
    assert sum(len(col) for K in NZ for col in K) == 20                    # two of the six have two empty columns
