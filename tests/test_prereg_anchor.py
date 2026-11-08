"""Anchored preregistrations (docs/PREREG_ANCHORING.md): the proof executor, the seed derivation and every refusal
of tools/prereg_anchor.py, offline.

The executor is held to REAL proofs: the four v2026.11.05 release proofs, whose twelve Bitcoin attestations must
execute to the merkle roots that blockstream.info and mempool.space both reported for blocks 974735, 974752 and
974757 on 2026-11-07. The tool is held to a synthetic proof built here, whose one attestation executes to a root the
fixture's two "explorers" report, so every link can be broken on purpose and must then refuse.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from oneq import prereg_seeds as PS  # noqa: E402
from oneq import ots_proof as O  # noqa: E402
import prereg_anchor as P  # noqa: E402

# merkle roots as block explorers print them; both explorers agreed on each (2026-11-07)
ROOTS = {974735: "ae9579d55d7362de2d66d09afc163aaaf97150317b5fd408c5f1d90905b51cea",
         974752: "5ee402a17f08e71b1b1fee8e5b4e116e1fdbbadee888051d53ee8ed0b107f783",
         974757: "6cf633634192596899a3787f08b641293e31d730e4ba571152eb0cb69fd51ff0"}
REAL = sorted((ROOT / "tests" / "data" / "ots_v2026.11.05").glob("*.v2026.11.05.ots"))   # = release/*.v2026.11.05.ots


# ----------------------------------------------------------------------------------------- the proof executor
def test_the_real_release_proofs_execute_to_their_blocks_merkle_roots():
    assert len(REAL) == 4, "the four v2026.11.05 proofs are the fixtures"
    hits = 0
    for p in REAL:
        covered, atts = O.parse_detached(p.read_bytes())
        target = p.name[: -len(".v2026.11.05.ots")]
        assert len(covered) == 32 and target
        btc = [a for a in atts if a.kind == "bitcoin"]
        assert sorted(a.height for a in btc) == sorted(ROOTS)
        for a in btc:
            assert a.merkle_root_display == ROOTS[a.height], (p.name, a.height)
            hits += 1
        assert all(a.uri and a.uri.startswith("https://") for a in atts if a.kind == "pending")
    assert hits == 12


def test_the_executor_agrees_with_the_release_verifiers_header_reader():
    import verify_release as V
    for p in REAL:
        raw = p.read_bytes()
        assert O.parse_detached(raw)[0].hex() == V.ots_covered_sha256(raw)
        assert sorted({a.height for a in O.bitcoin_attestations(raw)}) == V.ots_bitcoin_heights(raw)


def _varuint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _varbytes(b: bytes) -> bytes:
    return _varuint(len(b)) + b


def synthetic_proof(file_digest: bytes, height: int, salt: bytes = b"calendar") -> tuple[bytes, bytes]:
    """(proof bytes, the digest its one Bitcoin attestation commits to): append salt, sha256, attest."""
    body = bytes([0xF0]) + _varbytes(salt) + bytes([0x08]) + bytes([0x00]) + O.BITCOIN_TAG + _varbytes(_varuint(height))
    proof = O.MAGIC + _varuint(1) + bytes([0x08]) + file_digest + body
    return proof, hashlib.sha256(file_digest + salt).digest()


def test_a_flipped_argument_byte_moves_the_digest_and_a_damaged_proof_refuses():
    digest = hashlib.sha256(b"x").digest()
    proof, at = synthetic_proof(digest, 900000)
    assert O.bitcoin_attestations(proof)[0].digest == at
    flipped = bytearray(proof)
    flipped[proof.index(b"calendar")] ^= 1
    assert O.bitcoin_attestations(bytes(flipped))[0].digest != at
    with pytest.raises(O.ProofError, match="trailing"):
        O.parse_detached(proof + b"\x00")
    with pytest.raises(O.ProofError, match="truncated"):
        O.parse_detached(proof[:-3])
    with pytest.raises(O.ProofError, match="unknown operation"):
        O.parse_detached(O.MAGIC + _varuint(1) + bytes([0x08]) + digest + bytes([0x55]))
    with pytest.raises(O.ProofError, match="magic"):
        O.parse_detached(b"not a proof")


def test_a_fork_is_followed_down_every_branch():
    digest = hashlib.sha256(b"y").digest()
    branch_a = bytes([0x08]) + bytes([0x00]) + O.BITCOIN_TAG + _varbytes(_varuint(1))
    branch_b = bytes([0x00]) + O.PENDING_TAG + _varbytes(_varbytes(b"https://a.example"))
    proof = O.MAGIC + _varuint(1) + bytes([0x08]) + digest + bytes([0xFF]) + branch_a + branch_b
    _, atts = O.parse_detached(proof)
    assert [(a.kind, a.height, a.uri) for a in atts] == [("bitcoin", 1, None), ("pending", None, "https://a.example")]
    assert atts[0].digest == hashlib.sha256(digest).digest() and atts[1].digest == digest


# ----------------------------------------------------------------------------------------- seeds and rounds
R = "1c1ceb411d5f5b86825efc4c29a73e50ab4616b5d9f8375cbd6aaf5d5a45b35a"
H = hashlib.sha256(b"prereg").hexdigest()


def test_prereg_seeds_are_pinned_to_the_preregistration_and_the_count():
    a = PS.derive_prereg_seeds(R, prereg_sha256=H, n=5)
    assert a == PS.derive_prereg_seeds(R, prereg_sha256=H, n=5) and len(set(a)) == 5
    other = hashlib.sha256(b"prereg2").hexdigest()
    assert set(a).isdisjoint(PS.derive_prereg_seeds(R, prereg_sha256=other, n=5))
    assert set(a).isdisjoint(PS.derive_prereg_seeds(R, prereg_sha256=H, n=6))
    for bad in [dict(randomness_hex="ab", prereg_sha256=H, n=1), dict(randomness_hex=R, prereg_sha256="AB" * 32, n=1),
                dict(randomness_hex=R, prereg_sha256=H, n=0), dict(randomness_hex="zz" * 16, prereg_sha256=H, n=1)]:
        with pytest.raises(ValueError):
            PS.derive_prereg_seeds(**bad)


def test_round_time_is_drands_schedule():
    assert PS.round_time(1, genesis_time=1692803367, period_s=3) == 1692803367
    assert PS.round_time(33763471, genesis_time=1692803367, period_s=3) == 1692803367 + 33763470 * 3
    with pytest.raises(ValueError):
        PS.round_time(0, genesis_time=1692803367)


# ----------------------------------------------------------------------------------------- the tool, offline
INFO = {"hash": P.DRAND_CHAIN, "genesis_time": 1692803367, "period": 3, "public_key": "ab" * 48}
LATEST = 33_000_000
HEIGHT = 950_000


def _fixture(tmp: pathlib.Path, *, round_n: int, root: str, block_time: int, randomness=(R, R),
             published: bool = True) -> pathlib.Path:
    srcs = []
    for rnd in randomness:
        srcs.append({"drand_info": INFO, "drand_latest": LATEST,
                     "drand_rounds": {str(round_n): {"randomness": rnd, "signature": "cd" * 48}} if published else {},
                     "blocks": {str(HEIGHT): {"hash": "00" * 32, "merkle_root": root, "timestamp": block_time}}})
    path = tmp / "fixture.json"
    path.write_text(json.dumps({"sources": srcs}), encoding="utf-8")
    return path


@pytest.fixture
def anchored(tmp_path):
    """A declared anchor for a stand-in preregistration, its synthetic proof, and a consistent fixture."""
    prereg = b"# PREREG\nhypothesis, endpoints, analysis\n"
    decl = tmp_path / "decl.json"
    decl.write_text(json.dumps({"sources": [{"drand_info": INFO, "drand_latest": LATEST, "drand_rounds": {},
                                             "blocks": {}}] * 2}), encoding="utf-8")
    anchor = P.declare("docs/PREREG_test.md", sources=P.FixtureSources(decl), lead_hours=48, margin_hours=6,
                       stamp=False, prereg=(prereg, "0" * 40, "1" * 40), out=tmp_path / "PREREG_test.anchor.json")
    doc = json.loads(anchor.read_text(encoding="utf-8"))
    proof, at = synthetic_proof(hashlib.sha256(anchor.read_bytes()).digest(), HEIGHT)
    anchor.with_name(anchor.name + ".ots").write_bytes(proof)
    published = PS.round_time(doc["beacon"]["round"], genesis_time=INFO["genesis_time"], period_s=3)
    good = dict(round_n=doc["beacon"]["round"], root=at[::-1].hex(), block_time=published - 24 * 3600)
    return anchor, prereg, doc, good, tmp_path


def test_declare_names_a_round_far_enough_ahead_and_never_rewrites(anchored):
    anchor, prereg, doc, _, tmp = anchored
    assert doc["prereg_sha256"] == hashlib.sha256(prereg).hexdigest()
    assert doc["beacon"]["round"] == LATEST + 48 * 3600 // 3
    with pytest.raises(P.AnchorError, match="never rewritten"):
        P.declare("docs/PREREG_test.md", sources=P.FixtureSources(tmp / "decl.json"), lead_hours=48, margin_hours=6,
                  stamp=False, prereg=(prereg, "0" * 40, "1" * 40), out=anchor)
    with pytest.raises(P.AnchorError, match="twice the margin"):
        P.declare("docs/PREREG_test.md", sources=P.FixtureSources(tmp / "decl.json"), lead_hours=8, margin_hours=6,
                  stamp=False, prereg=(prereg, "0" * 40, "1" * 40), out=tmp / "other.anchor.json")


def test_every_link_holds_and_the_seeds_follow(anchored):
    anchor, prereg, doc, good, tmp = anchored
    rec = P.check(anchor, sources=P.FixtureSources(_fixture(tmp, **good)), prereg_bytes=prereg)
    assert rec["bitcoin"]["height"] == HEIGHT and rec["beacon"]["randomness"] == R
    assert rec["hours_between_block_and_round"] == 24.0
    assert rec["first_seeds"] == PS.derive_prereg_seeds(R, prereg_sha256=doc["prereg_sha256"], n=4)


@pytest.mark.parametrize("break_it, refusal", [
    ("prereg", "not the preregistration"),
    ("root", "merkle root"),
    ("late_block", "void"),
    ("unpublished", "not published yet"),
    ("relays_disagree", "disagree"),
    ("other_bytes", "other bytes"),
    ("calendar_only", "no Bitcoin attestation"),
])
def test_each_broken_link_refuses(anchored, break_it, refusal):
    anchor, prereg, doc, good, tmp = anchored
    kw = dict(good)
    data = prereg
    if break_it == "prereg":
        data = prereg + b"edited after the anchor\n"
    elif break_it == "root":
        kw["root"] = "ff" * 32
    elif break_it == "late_block":
        kw["block_time"] = good["block_time"] + 20 * 3600          # 4 h before the round, margin 6 h
    elif break_it == "unpublished":
        kw["published"] = False
    elif break_it == "relays_disagree":
        kw["randomness"] = (R, "00" + R[2:])
    elif break_it == "other_bytes":
        proof, _ = synthetic_proof(hashlib.sha256(b"another anchor").digest(), HEIGHT)
        anchor.with_name(anchor.name + ".ots").write_bytes(proof)
    elif break_it == "calendar_only":
        pending = (O.MAGIC + _varuint(1) + bytes([0x08]) + hashlib.sha256(anchor.read_bytes()).digest()
                   + bytes([0x00]) + O.PENDING_TAG + _varbytes(_varbytes(b"https://a.example")))
        anchor.with_name(anchor.name + ".ots").write_bytes(pending)
    with pytest.raises(P.AnchorError, match=refusal):
        P.check(anchor, sources=P.FixtureSources(_fixture(tmp, **kw)), prereg_bytes=data)


def test_verify_refuses_a_redemption_record_that_disagrees_with_the_sources(anchored, monkeypatch):
    anchor, prereg, doc, good, tmp = anchored
    src = P.FixtureSources(_fixture(tmp, **good))
    real_check = P.check
    monkeypatch.setattr(P, "check", lambda a, sources: real_check(a, sources=sources, prereg_bytes=prereg))
    rec = P.redeem(anchor, sources=src, upgrade=False)
    assert P.verify(anchor, sources=src) == rec
    assert P.seeds(anchor, 4) == rec["first_seeds"]
    doctored = dict(rec, first_seeds=[1, 2, 3, 4])
    P.redeemed_path(anchor).write_text(json.dumps(doctored), encoding="utf-8")
    with pytest.raises(P.AnchorError, match="does not match"):
        P.verify(anchor, sources=src)
