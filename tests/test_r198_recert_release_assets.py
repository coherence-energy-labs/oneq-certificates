"""Paper 3's large proofs as release assets (experiments/circuit_distance_exact/recert_assets.py, 2026-11-06).

External audit 3 found the thirty large re-certified proofs attached to release v2026.11.05 while paper 3 still said
they ship only as models and hashes, and no tracked file named the archives. These tests hold the shipped record to
the manifest, and the reader's verify/extract route to refusing every way an archive can be the wrong bytes. The real
archives (370 MB) are release assets, not repository files, so the route is exercised on small synthetic archives
built here; the shipped record is checked against the shipped manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tarfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_distance_exact"))
import recert_assets as ra  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------- the shipped record
def test_the_shipped_record_names_every_unshipped_proof_and_nothing_else():
    rec = json.loads(ra.RECORD.read_text(encoding="utf-8"))
    man = ra.load_manifest()
    cells = ra.unshipped(man)
    assert len(cells) == 30, len(cells)                    # a manifest with no unshipped cell would pass vacuously
    assert rec["schema"] == ra.SCHEMA and rec["manifest_sha256_lf"] == ra.lf_sha256(ra.MANIFEST)
    assert sorted(a["name"] for a in rec["archives"]) == sorted(ra.ARCHIVES)
    seen = []
    for a in rec["archives"]:
        assert a["families"] == list(ra.ARCHIVES[a["name"]])
        ra.check_members(a["name"], a["members"], cells)   # each member is its cell's path, size and sha256
        seen += [m["path"] for m in a["members"]]
    assert sorted(seen) == sorted(cells)
    assert rec["totals"] == {"archives": 2, "proofs": 30, "archive_bytes": sum(a["bytes"] for a in rec["archives"]),
                             "proof_bytes": sum(c["proof_bytes"] for c in cells.values())}
    for c in man["cells"]:                                 # the shipped ones are in the repository, byte for byte
        if c["shipped"]:
            assert ra.member_path(c) not in seen


def test_a_member_path_names_its_cell():
    c = {"family": "3c_t2", "kind": "cert", "circuit": "3c", "basis": "Z", "rounds": 2, "w": 10}
    p = ra.member_path(c)
    assert p == "recert_20261102/proofs/3c_t2/cert/proof_3c_Z2_w10.bin" and ra.MEMBER.fullmatch(p)


# ------------------------------------------------------------------------------------ synthetic archives and cells
def _cells() -> list[dict]:
    out = []
    for fam, kind, circ, basis, rounds, payload in (("3c", "shift_cert", "3c", "X", 1, b"proof-3c-X1"),
                                                     ("3c_t2", "cert", "3c", "Z", 2, b"proof-3c_t2-Z2"),
                                                     ("morph144", "cert", "morph144", "X", 1, b"proof-m144-X1")):
        out.append({"family": fam, "kind": kind, "circuit": circ, "basis": basis, "rounds": rounds, "w": 10,
                    "shipped": False, "proof_bytes": len(payload), "proof_sha256": _sha(payload), "_payload": payload})
    out.append({"family": "morph72", "kind": "cert", "circuit": "morph72", "basis": "X", "rounds": 1, "w": 6,
                "shipped": True, "proof_bytes": 3, "proof_sha256": _sha(b"abc")})
    return out


def _tar(path: pathlib.Path, members: list[tuple[str, bytes]], *, symlink: str | None = None) -> pathlib.Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        if symlink:
            ti = tarfile.TarInfo(symlink)
            ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tf.addfile(ti)
    return path


def _archives(d: pathlib.Path, cells: list[dict], **mod) -> dict[str, pathlib.Path]:
    by = {name: [(ra.member_path(c), c["_payload"]) for c in cells if not c["shipped"] and c["family"] in fams]
          for name, fams in ra.ARCHIVES.items()}
    for name, fn in mod.items():
        by[name] = fn(by[name])
    return {name: _tar(d / name, ms) for name, ms in by.items()}


def _strip(cells):
    return {"cells": [{k: v for k, v in c.items() if k != "_payload"} for c in cells]}


def test_a_record_built_from_the_right_archives_verifies_and_extracts(tmp_path):
    cells = _cells()
    arch = _archives(tmp_path, cells)
    rec = ra.build_record(arch, _strip(cells), "m" * 64)
    assert rec["totals"]["proofs"] == 3 and rec["totals"]["archives"] == 2
    out = ra.verify(tmp_path, rec, _strip(cells), manifest_sha256="m" * 64, extract_into=tmp_path / "root")
    assert out == {"archives": 2, "proofs": 3, "extracted": 3, "already_in_place": 0}
    for c in cells[:3]:
        assert (tmp_path / "root" / ra.member_path(c)).read_bytes() == c["_payload"]
    again = ra.verify(tmp_path, rec, _strip(cells), extract_into=tmp_path / "root")
    assert again["extracted"] == 0 and again["already_in_place"] == 3       # idempotent, nothing rewritten


@pytest.mark.parametrize("case, expect", [
    ("altered_bytes", "is not the proof its manifest cell names"),
    ("extra_member", "are extra"),
    ("missing_member", "are missing"),
    ("escaping_path", "outside the evidence directory"),
])
def test_an_archive_that_is_not_the_manifests_proofs_is_refused(tmp_path, case, expect):
    cells = _cells()
    mod = {
        "altered_bytes": {"oneq-recert-proofs-morph144.tar.gz": lambda ms: [(n, d + b"!") for n, d in ms]},
        "extra_member": {"oneq-recert-proofs-morph144.tar.gz":
                         lambda ms: ms + [("recert_20261102/proofs/morph144/cert/proof_morph144_Z9_w11.bin", b"x")]},
        "missing_member": {"oneq-recert-proofs-3c.tar.gz": lambda ms: ms[:1]},
        "escaping_path": {"oneq-recert-proofs-3c.tar.gz": lambda ms: ms + [("../../escape.bin", b"x")]},
    }[case]
    arch = _archives(tmp_path, cells, **mod)
    with pytest.raises(ra.AssetError, match=expect):
        ra.build_record(arch, _strip(cells), "m" * 64)


def test_a_link_member_is_refused(tmp_path):
    cells = _cells()
    arch = _archives(tmp_path, cells)
    _tar(arch["oneq-recert-proofs-3c.tar.gz"],
         [(ra.member_path(c), c["_payload"]) for c in cells if c["family"] in ("3c", "3c_t2")],
         symlink="recert_20261102/proofs/3c/shift_cert/link.bin")
    with pytest.raises(ra.AssetError, match="not a regular file"):
        ra.build_record(arch, _strip(cells), "m" * 64)


def test_a_download_that_is_not_the_recorded_archive_is_refused(tmp_path):
    cells = _cells()
    arch = _archives(tmp_path, cells)
    rec = ra.build_record(arch, _strip(cells), "m" * 64)
    with pytest.raises(ra.AssetError, match="written against another manifest"):
        ra.verify(tmp_path, rec, _strip(cells), manifest_sha256="n" * 64)
    _archives(tmp_path, cells, **{"oneq-recert-proofs-3c.tar.gz": lambda ms: list(reversed(ms))})   # same members,
    with pytest.raises(ra.AssetError, match="not the recorded archive"):                            # other bytes
        ra.verify(tmp_path, rec, _strip(cells))
    (tmp_path / "oneq-recert-proofs-3c.tar.gz").unlink()
    with pytest.raises(ra.AssetError, match="download it from the release"):
        ra.verify(tmp_path, rec, _strip(cells))


def test_extraction_never_overwrites_a_different_file(tmp_path):
    cells = _cells()
    arch = _archives(tmp_path, cells)
    rec = ra.build_record(arch, _strip(cells), "m" * 64)
    dest = tmp_path / "root" / ra.member_path(cells[0])
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"someone else's file")
    with pytest.raises(ra.AssetError, match="exists with other bytes"):
        ra.verify(tmp_path, rec, _strip(cells), extract_into=tmp_path / "root")
    assert dest.read_bytes() == b"someone else's file"


def test_check_refuses_when_no_proof_is_in_place(tmp_path):
    with pytest.raises(ra.AssetError, match="no unshipped proof is in place"):
        ra.check(ra.load_manifest(), threads=1, family=None, limit=None, root=tmp_path)


def test_a_verdict_is_compared_without_its_timing():
    rec = "VERIFIED: no irreducible undetectable logical error of weight <= 10 (36 anchors; 190.4s)"
    assert ra._verdict(rec) == ra._verdict(rec.replace("190.4s", "201.7s"))
    assert ra._verdict(rec) != ra._verdict(rec.replace("<= 10", "<= 9"))
