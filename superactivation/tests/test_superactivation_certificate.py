"""The superactivation certificate: tight when the witness is honest, refuses or shrinks when not.

Small-n witnesses (fast) carry the behavioural checks; the committed n = 17 witness carries the
claim itself -- certified fidelity > 3/4 for an exactly CPTP code.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from oneq import superactivation as sa
from oneq import superactivation_certificate as sc

ROOT = pathlib.Path(__file__).resolve().parents[1]
WITNESS = ROOT / "evidence" / "superactivation" / "witness_n17.npz"


@pytest.fixture(scope="module")
def small():
    n = 6
    X = sa.random_encoder(n, np.random.default_rng(5), rank=2)
    decs, F, _ = sa.decode_power(X, n, iters=40)
    return n, X, {k: list(v) for k, v in decs.items()}, F


def test_the_certified_bound_is_below_and_within_1e8_of_the_float_value(small):
    n, X, decs, F = small
    res = sc.certify(n, X, decs)
    L = res["certified_lower_bound_float"]
    assert L <= F + 1e-12
    assert F - L < 1e-8, (F, L)
    assert res["blocks_proved_positive_definite"] == sum(k + 1 for k in range(n + 1)) + 1


def test_a_decoder_that_is_not_completely_positive_is_refused(small):
    n, X, decs, _ = small
    bad = {k: [D.copy() for D in v] for k, v in decs.items()}
    d = n - 2 + 1
    bad[2][1] = bad[2][1] - 0.05 * np.eye(2 * d)          # a genuinely negative eigenvalue
    with pytest.raises(sc.CertificateRefusal, match=r"decoder \(k=2, b=1\)"):
        sc.certify(n, X, bad)


def test_an_inflated_encoder_certifies_less_not_more(small):
    n, X, decs, _ = small
    honest = sc.certify(n, X, decs)["certified_lower_bound_float"]
    # 1% too much trace: the normalization charge must absorb it, so the bound cannot rise
    cheat = sc.certify(n, 1.01 * X, decs)["certified_lower_bound_float"]
    assert cheat <= honest + 1e-12


def test_wrong_shapes_are_refused(small):
    n, X, decs, _ = small
    with pytest.raises(sc.CertificateRefusal):
        sc.certify(n + 1, X, decs)
    short = dict(decs)
    short[3] = short[3][:-1]
    with pytest.raises(sc.CertificateRefusal):
        sc.certify(n, X, short)
    resized = {k: list(v) for k, v in decs.items()}
    resized[2][0] = np.eye(4)                                        # the k = 2 decoders are 10 square
    with pytest.raises(sc.CertificateRefusal, match=r"decoder \(k=2, b=0\) must be 10 square"):
        sc.certify(n, X, resized)


def test_a_witness_that_is_not_a_map_is_refused_before_any_bound(small):
    n, X, decs, _ = small
    with pytest.raises(sc.CertificateRefusal, match="non-finite"):
        sc.certify(n, X * np.nan, decs)
    with pytest.raises(sc.CertificateRefusal, match="encoder not proved positive definite"):
        sc.certify(n, X - 3.0 * np.eye(X.shape[0]), decs)


def test_a_saved_witness_reloads_bit_for_bit_and_certifies_the_same(small, tmp_path):
    n, X, decs, _ = small
    path = tmp_path / "witness.npz"
    sc.save_witness(path, n, X, decs)
    n2, X2, decs2 = sc.load_witness(path)
    assert n2 == n and np.array_equal(X2, X)
    assert all(np.array_equal(a, b) for k in decs for a, b in zip(decs[k], decs2[k]))
    lines: list[str] = []
    res = sc.certify(n2, X2, decs2, log=lines.append)
    assert res["certified_lower_bound"] == sc.certify(n, X, decs)["certified_lower_bound"]
    assert len(lines) == n + 1                                       # one progress line per branch


@pytest.mark.parametrize("n", [17, 20, 24])
def test_each_committed_witness_is_the_one_its_certificate_names(n):
    """The certificate JSON records the witness's sha256; a swapped or edited witness fails here
    before any recertification is attempted."""
    import hashlib
    import json
    cert = json.loads((ROOT / "evidence" / "superactivation" / f"certificate_n{n}.json").read_text(encoding="utf-8"))
    wit = ROOT / "evidence" / "superactivation" / f"witness_n{n}.npz"
    assert cert["witness"] == wit.name
    assert hashlib.sha256(wit.read_bytes()).hexdigest() == cert["witness_sha256"]
    assert cert["exceeds_three_quarters"] is True


def test_the_committed_n20_witness_certifies_a_wider_margin():
    """n = 20, the last blocklength the authors tabulated (float 0.7506160): certified above it."""
    n, X, decs = sc.load_witness(ROOT / "evidence" / "superactivation" / "witness_n20.npz")
    assert n == 20
    res = sc.certify(n, X, decs)
    assert res["exceeds_three_quarters"] is True
    assert res["certified_lower_bound_float"] > 0.7506160          # their float value at n = 20
    assert res["certified_margin_lower_bound"] > 6.5e-4


def test_the_committed_n24_witness_certifies_a_margin_above_one_thousandth():
    """n = 24, four blocklengths past the authors' last tabulated point: margin >= 1.03e-3."""
    n, X, decs = sc.load_witness(ROOT / "evidence" / "superactivation" / "witness_n24.npz")
    assert n == 24
    res = sc.certify(n, X, decs)
    assert res["exceeds_three_quarters"] is True
    assert res["certified_margin_lower_bound"] > 1.03e-3
    assert res["blocks_proved_positive_definite"] == sum(k + 1 for k in range(n + 1)) + 1


def _curve():
    import json
    return json.loads((ROOT / "evidence" / "superactivation" / "curve_n17_n24.json").read_text(encoding="utf-8"))


def test_the_curve_record_names_the_committed_encoders_and_rises_with_n():
    import hashlib
    rows = _curve()
    assert [r["n"] for r in rows] == list(range(17, 25))
    for r in rows:
        raw = (ROOT / "evidence" / "superactivation" / r["encoder"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r["encoder_sha256"], r["n"]
    Fs = [r["F_exact_decoders"] for r in rows]
    assert all(b > a for a, b in zip(Fs, Fs[1:])), Fs
    assert Fs[0] > sa.THRESHOLD


@pytest.mark.parametrize("n", [21, 24])
def test_a_curve_point_rescores_from_its_encoder(n):
    """Power-iteration decoders are feasible, so they can only reach the recorded exact-decoder value
    from below; a swapped encoder or an edited record lands outside the 1e-6 window."""
    row = next(r for r in _curve() if r["n"] == n)
    X = np.load(ROOT / "evidence" / "superactivation" / row["encoder"])
    _, F, _ = sa.decode_power(X, n, iters=200)
    assert -1e-9 <= row["F_exact_decoders"] - F < 1e-6, (row["F_exact_decoders"], F)


def test_the_curve_codes_are_isometric_encoders():
    """The SDP returns a second eigenvalue ~1e-4 beside 2.000; truncating to the top eigenvector (an
    isometry C^2 -> Sym^n) does not lower the fidelity (measured: it RISES by 1-3e-9 at every n), so
    the edge over the authors' isometric-ansatz table is not encoder mixedness."""
    n = 24
    X = np.load(ROOT / "evidence" / "superactivation" / "encoder_n24_seesaw.npy")
    w, v = np.linalg.eigh(sa.herm(X))
    assert w[-1] > 1.999 and w[-2] < 1e-3
    top = v[:, -1:]
    X1 = sa.normalize_encoder(w[-1] * (top @ top.conj().T), n)
    _, F_full, _ = sa.decode_power(X, n, iters=300)
    _, F_iso, _ = sa.decode_power(X1, n, iters=300)
    assert F_iso > F_full - 1e-9, (F_full, F_iso)


def test_the_erasure_tolerance_of_the_certified_n17_code():
    """Held fixed, the n = 17 code stays above 3/4 only while the erasure probability q < 0.5004;
    at q = 1/2 the reweighting must reproduce the code's own fidelity exactly."""
    n, X, decs = sc.load_witness(WITNESS)
    Fk = sa.branch_fidelities(X, decs, n)
    assert Fk[0] > 0.99 and Fk[n] <= 0.5 + 1e-9                     # no flags: near-perfect; all flags: separable
    F = sa.fidelity(X, {k: np.stack(v) for k, v in decs.items()}, n)
    assert sa.fidelity_at_erasure(Fk, 0.5) == pytest.approx(F, abs=1e-12)
    grid = [0.5 + 0.001 * i for i in range(11)]
    vals = [sa.fidelity_at_erasure(Fk, q) for q in grid]
    assert all(b < a for a, b in zip(vals, vals[1:]))               # more erasure, less fidelity
    assert sa.fidelity_at_erasure(Fk, 0.5003) > 0.75 > sa.fidelity_at_erasure(Fk, 0.5005)
    with pytest.raises(sa.SuperactivationRefusal, match="not a probability"):
        sa.fidelity_at_erasure(Fk, 1.2)


def test_the_committed_n17_witness_certifies_superactivation():
    n, X, decs = sc.load_witness(WITNESS)
    assert n == 17
    res = sc.certify(n, X, decs)
    assert res["exceeds_three_quarters"] is True
    assert res["certified_margin_lower_bound"] > 1.9e-4
    # the float value of the same pair, for the record: the certificate must sit just under it
    F = sa.fidelity(X, {k: np.stack(v) for k, v in decs.items()}, n)
    assert 0 <= F - res["certified_lower_bound_float"] < 1e-7
