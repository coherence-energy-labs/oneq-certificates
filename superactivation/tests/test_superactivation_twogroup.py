"""R140: two-group (Sym^a x Sym^b) superactivation codes and their certificate.

The n = 15 and n = 16 certificates take hours of exact arithmetic, so the suite does not re-run them. It BINDS each
record to committed bytes instead, and every leg of that binding can fail:
  - the witness and the stored decoders hash to what the certificate says it proved;
  - the code that produced it is a git object, not a working copy that is gone;
  - the stored pair's float fidelity, recomputed here, sits just ABOVE the certified bound. A bound above its own
    pair certifies some other pair: that is the 2026-10-14 failure, where the first n = 15 record was 1.4e-9 above
    the pair the committed pipeline produced, because nothing it proved had been stored.
The SAME certificate machinery is exercised end to end at n = 4 (seconds) and must refuse a corrupted decoder.
"""
import copy
import hashlib
import io
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GC = ROOT / "experiments" / "superactivation" / "general_codes"
EV = ROOT / "evidence" / "superactivation" / "general_codes"
sys.path.insert(0, str(GC))
sys.path.insert(0, str(ROOT / "src"))
import twogroup as tg  # noqa: E402
import twogroup_certificate as tc  # noqa: E402
from oneq import superactivation_certificate as sc  # noqa: E402


CERTS = ["a2_b2", "a4_b4", "a7_b8", "a8_b8"]


def _cert(ab):
    return json.loads((EV / f"certificate_twogroup_{ab}.json").read_text(encoding="utf-8"))


def git_blob_id(data: bytes) -> str:
    """The object id git gives a blob: sha1 over the header `blob <size>`, a NUL byte, then the bytes."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + bytes([0]) + data).hexdigest()


def _blob_committed_or_shipped(blob: str) -> bool:
    if subprocess.run(["git", "cat-file", "-e", blob], cwd=ROOT, capture_output=True).returncode == 0:
        return True
    snap = EV / "producer_blobs" / f"{blob}.blob"
    return snap.is_file() and git_blob_id(snap.read_bytes()) == blob


def test_every_producer_blob_ships_and_is_the_object_it_is_named_for():
    """The release-side half of the binding, checked everywhere: each blob a record names has its bytes in
    producer_blobs/, and those bytes ARE that git object. A doctored snapshot cannot pass for the blob."""
    blobs = {b for ab in CERTS for b in _cert(ab)["provenance"]["code_git_blobs"].values()}
    assert len(blobs) >= 4
    for blob in blobs:
        data = (EV / "producer_blobs" / f"{blob}.blob").read_bytes()
        assert git_blob_id(data) == blob
        assert git_blob_id(data + b" ") != blob


def binding_problems(cert: dict) -> list:
    """everything that stops this record from being the certificate of the committed pair (empty = bound)"""
    probs = []
    wraw = (EV / cert["witness"]).read_bytes()
    draw = (EV / cert["decoders"]).read_bytes()
    if hashlib.sha256(wraw).hexdigest() != cert["witness_sha256"]:
        probs.append("witness bytes are not the ones certified")
    if hashlib.sha256(draw).hexdigest() != cert["decoders_sha256"]:
        probs.append("decoder bytes are not the ones certified")
    # a git OBJECT: in a clean clone only committed objects exist, so there this is "the producing code is committed".
    # A reader of the release has no git history: there the producing bytes ship as general_codes/producer_blobs/
    # <blob>.blob, and they must HASH to the blob as git hashes it -- the same object, re-derived from its bytes
    # (2026-11-03: the release rehearsal of paper 4's bundle failed all six binding tests on the missing history).
    for f, blob in cert["provenance"]["code_git_blobs"].items():
        if not _blob_committed_or_shipped(blob):
            probs.append(f"{f}: blob {blob} is in no commit and no shipped producer blob hashes to it -- the "
                         f"producing code is not in the repository")
    if probs:
        return probs
    X = np.load(io.BytesIO(wraw), allow_pickle=False)
    a, b, decs = tc.load_decoders(draw)
    if (a, b) != (cert["a"], cert["b"]):
        return [f"decoders are for ({a},{b}), the certificate for ({cert['a']},{cert['b']})"]
    F = tc.pair_fidelity(X, a, b, decs)
    L = cert["certified_lower_bound_float"]
    if abs(F - cert["pair_float_fidelity"]) > 1e-12:
        probs.append(f"stored pair scores {F!r} here, {cert['pair_float_fidelity']!r} in the record")
    if L > F + 1e-12:
        probs.append(f"bound {L!r} exceeds its own pair's fidelity {F!r}: it certifies some other pair")
    if F - L > 1e-7:
        probs.append(f"bound {L!r} sits {F - L:.2e} below its pair: not the certificate of these maps")
    return probs


@pytest.mark.parametrize("ab", CERTS)
def test_each_twogroup_certificate_is_bound_to_committed_bytes(ab):
    cert = _cert(ab)
    assert cert["witness"] == f"twogroup_witness_{ab}.npy" and cert["decoders"] == f"twogroup_decoders_{ab}.npz"
    assert binding_problems(cert) == []


@pytest.mark.parametrize("doctor, expect", [
    (lambda c: c.update(certified_lower_bound_float=c["pair_float_fidelity"] + 2e-9), "certifies some other pair"),
    (lambda c: c.update(decoders_sha256="0" * 64), "decoder bytes"),
    (lambda c: c.update(witness_sha256="0" * 64), "witness bytes"),
    (lambda c: c["provenance"]["code_git_blobs"].update({"twogroup.py": "f" * 40}), "in no commit"),
    (lambda c: c.update(pair_float_fidelity=c["pair_float_fidelity"] + 1e-9), "in the record"),
])
def test_the_binding_check_catches_a_stale_or_doctored_record(doctor, expect):
    """negative controls: each leg of the binding must be able to fail"""
    cert = copy.deepcopy(_cert("a2_b2"))
    doctor(cert)
    probs = binding_problems(cert)
    assert any(expect in p for p in probs), probs


@pytest.mark.parametrize("ab, n, kmax", [("a7_b8", 15, (8, 9)), ("a8_b8", 16, (9, 9))])
def test_the_certificates_record_superactivation_below_the_published_17(ab, n, kmax):
    cert = _cert(ab)
    assert cert["n"] == n == cert["a"] + cert["b"]
    assert cert["exceeds_three_quarters"] is True
    assert cert["certified_margin_lower_bound"] > 7.8e-5
    assert cert["blocks_proved_positive_definite"] == 1 + sum((k1 + 1) * (k2 + 1) for k1 in range(kmax[0])
                                                              for k2 in range(kmax[1]))


def test_stored_decoders_round_trip_to_the_same_certificate_inputs(tmp_path):
    """the packed (upper-triangle) storage is lossless for the certificate: exact_block of the rebuilt matrix equals
    exact_block of the original, because both read the same triangle of herm(D)"""
    flint = sc._flint()
    X = np.load(EV / "twogroup_witness_a2_b2.npy")
    decs, _ = tc.power_decoders(X, 2, 2, iters=200)
    _, _, back = tc.load_decoders(tc.save_decoders(tmp_path / "decs.npz", 2, 2, decs))
    assert sorted(back) == sorted(decs)
    for key in decs:
        for D, E in zip(decs[key], back[key], strict=True):
            assert sc.exact_block(flint, D, 2.0 ** -34) == sc.exact_block(flint, E, 2.0 ** -34)
            assert np.array_equal(E, E.conj().T)


def test_the_twogroup_certificate_runs_end_to_end_and_sits_just_under_the_float_value():
    X = np.load(EV / "twogroup_witness_a2_b2.npy")
    decs, F = tc.power_decoders(X, 2, 2, iters=1500)
    res = tc.certify_twogroup(2, 2, X, decs)
    assert 0 <= F - res["certified_lower_bound_float"] < 1e-7, (F, res["certified_lower_bound_float"])
    assert res["exceeds_three_quarters"] is False                     # n = 4 is far below 3/4
    assert res["blocks_proved_positive_definite"] == 1 + 36


def test_a_corrupted_decoder_is_refused():
    X = np.load(EV / "twogroup_witness_a2_b2.npy")
    decs, _ = tc.power_decoders(X, 2, 2, iters=200)
    bad = {k: [D.copy() for D in v] for k, v in decs.items()}
    bad[(1, 1)][0] = bad[(1, 1)][0] - 0.3 * np.eye(bad[(1, 1)][0].shape[0])
    with pytest.raises(sc.CertificateRefusal, match=r"decoder \(1,1,0,0\)"):
        tc.certify_twogroup(2, 2, X, bad)


def test_bareiss_and_sylvester_give_the_same_verdict():
    flint = sc._flint()
    rng = np.random.default_rng(3)
    for trial in range(20):
        d = int(rng.integers(2, 7))
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        H = A @ A.conj().T
        if trial % 2:
            w, v = np.linalg.eigh(H)
            w[0] = -abs(w).max() * 0.1
            H = (v * w) @ v.conj().T
        re, im = sc.exact_block(flint, H, 2.0 ** -34)
        assert sc.positive_definite(flint, re, im) == tc.positive_definite_bareiss(flint, re, im)


def test_the_twogroup_reduction_contains_the_symmetric_codes_exactly():
    """a Sym^n code embedded into Sym^a (x) Sym^b scores the same through the two-group reduction"""
    from oneq import superactivation as sa
    n, a, b = 6, 3, 3
    Xs = sa.random_encoder(n, np.random.default_rng(9), rank=1)
    M = np.kron(np.eye(2), tg.sym_embedding(a, b))
    _, Fs, _ = sa.decode_power(Xs, n, iters=300)
    _, Ft = tg.decode(M @ Xs @ M.T, a, b, iters=300)
    assert Ft == pytest.approx(Fs, abs=1e-9)
