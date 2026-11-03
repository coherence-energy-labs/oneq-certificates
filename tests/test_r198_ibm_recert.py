"""R198 (2026-10-26): the 22 IBM gross-cycle time-shift certificates shipped in
evidence/circuit_distance_exact/ibm_recert_20261026/ -- every proof and model is the bytes the manifest names, every cell
was VERIFIED, and the COMMITTED checker re-verifies shipped proofs today and rejects a corrupted one. They are a
third-party-checkable route to IBM's d_circ >= 10 for every T (paper 3's gap 1: the campaign's own certificates were
checked by a working-tree checker never committed)."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LANE = ROOT / "experiments" / "circuit_distance_exact"
sys.path.insert(0, str(LANE))
import bb_circuits as bc  # noqa: E402
import tools_cde as tc  # noqa: E402

PKG = ROOT / "evidence" / "circuit_distance_exact" / "ibm_recert_20261026"


def _manifest() -> dict:
    return json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))


def test_every_shipped_file_is_the_bytes_the_manifest_names():
    m = _manifest()
    obs = m["observations"]
    assert m["verdict"] == "22/22 VERIFIED" and len(obs) == 22
    assert sorted((o["basis"], o["rounds"]) for o in obs) == sorted((b, t) for b in "ZX" for t in range(1, 12))
    for o in obs:
        assert hashlib.sha256((PKG / o["proof"]).read_bytes()).hexdigest() == o["proof_sha256"], o["proof"]
        assert hashlib.sha256((PKG / o["model"]).read_bytes()).hexdigest() == o["model_sha256"], o["model"]
        assert o["w"] == 9 and o["proof_magic"] == "CDP5"
        assert o["checker_verdict"].startswith("VERIFIED: no irreducible undetectable logical error of weight <= 9")


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    return tc.build_c(tmp_path_factory.mktemp("recert_bin"))


@pytest.mark.parametrize("cell", ["Z1", "X1"])
def test_the_committed_checker_reverifies_a_shipped_proof_and_rejects_a_corrupted_one(exe, cell, tmp_path):
    o = next(x for x in _manifest()["observations"] if f"{x['basis']}{x['rounds']}" == cell)
    model, proof = PKG / o["model"], PKG / o["proof"]
    ok = tc.run(exe["ccheck"], model, proof, bc.group_kind("ibm"))
    assert ok.returncode == 0 and ok.stdout.startswith("VERIFIED"), ok.stdout + ok.stderr
    bad = tmp_path / "bad.bin"
    raw = bytearray(proof.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                                   # one byte in the trace
    bad.write_bytes(bytes(raw))
    got = tc.run(exe["ccheck"], model, bad, bc.group_kind("ibm"))
    assert got.returncode != 0 and not got.stdout.startswith("VERIFIED")
