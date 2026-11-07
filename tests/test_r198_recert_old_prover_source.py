"""Where recert_old_prover.py takes the prover of an earlier commit from (external audit 4, R8, 2026-11-07).

The audit found that the script read the prover of --commit with `git show` only, so in a public clone -- whose
history is squashed and holds no development commit -- it failed, although the source ships as
old_provers/cdist_prove_<commit>.c. The script now falls back on the shipped copy, holds every route to the sha256 the
records pin for the commit, and says which route it took. These tests run the selection on a synthetic tree with no
git repository (git is pointed at a directory that does not exist), and once on this tree's own pinned commit.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "circuit_distance_exact"))
import recert_old_prover as rop  # noqa: E402

COMMIT = "0123abcd"
SOURCE_LF = b"int main(void) { return 0; }\n/* the prover of a commit nobody can git show */\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _tree(tmp_path: pathlib.Path, pin: str | None, shipped: bytes | None):
    """A tree with no git repository: records pinning `pin` for COMMIT, and the shipped copy `shipped`."""
    root = tmp_path / "tree"
    old = root / "experiments" / "circuit_distance_exact" / "old_provers"
    old.mkdir(parents=True)
    records = root / "recert_old_prover.jsonl"
    rows = [] if pin is None else [{"commit": COMMIT, "prover_source_sha256_git": pin, "family": "x"}]
    rows.append({"commit": "ffff0000", "prover_source_sha256_git": "0" * 64})     # another commit's pin, ignored
    records.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    if shipped is not None:
        (old / f"cdist_prove_{COMMIT}.c").write_bytes(shipped)
    return root, records, old


@pytest.fixture(autouse=True)
def _no_git(tmp_path, monkeypatch):
    # every `git show` in these tests fails as it does in a tree without the development history
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-such-git-directory"))


def test_without_git_the_shipped_copy_is_used_and_held_to_the_pinned_sha(tmp_path):
    # the shipped copy may arrive with CRLF line endings; it is read as LF, the bytes git stores and the record pins
    root, records, old = _tree(tmp_path, _sha(SOURCE_LF), SOURCE_LF.replace(b"\n", b"\r\n"))
    src, route = rop.prover_source(COMMIT, root=root, records=records, shipped=old)
    assert src == SOURCE_LF and _sha(src) == _sha(SOURCE_LF)
    assert route.startswith("shipped ") and route.endswith(f"cdist_prove_{COMMIT}.c")


def test_a_shipped_copy_whose_sha_is_not_the_pinned_one_is_refused(tmp_path):
    root, records, old = _tree(tmp_path, _sha(SOURCE_LF), SOURCE_LF + b"/* one byte more */\n")
    with pytest.raises(SystemExit, match="REFUSED: the prover source from shipped"):
        rop.prover_source(COMMIT, root=root, records=records, shipped=old)


def test_a_shipped_copy_that_no_record_pins_is_refused(tmp_path):
    root, records, old = _tree(tmp_path, None, SOURCE_LF)
    with pytest.raises(SystemExit, match="pins no source sha256"):
        rop.prover_source(COMMIT, root=root, records=records, shipped=old)


def test_neither_git_nor_a_shipped_copy_is_a_refusal_that_says_so(tmp_path):
    root, records, old = _tree(tmp_path, _sha(SOURCE_LF), None)
    with pytest.raises(SystemExit, match="neither in this tree's git history nor shipped"):
        rop.prover_source(COMMIT, root=root, records=records, shipped=old)


def test_an_explicit_source_is_held_to_the_pin_too(tmp_path):
    root, records, old = _tree(tmp_path, _sha(SOURCE_LF), None)
    good, bad = tmp_path / "good.c", tmp_path / "bad.c"
    good.write_bytes(SOURCE_LF)
    bad.write_bytes(b"/* not the pinned prover */\n")
    src, route = rop.prover_source(COMMIT, source=str(good), root=root, records=records, shipped=old)
    assert src == SOURCE_LF and route.startswith("--source ")
    with pytest.raises(SystemExit, match="REFUSED"):
        rop.prover_source(COMMIT, source=str(bad), root=root, records=records, shipped=old)


def test_this_trees_pinned_commit_resolves_to_its_pinned_source(monkeypatch):
    # the real pin: 72d03874, whose rows in records/recert_old_prover.jsonl all record one source sha256. Without git
    # (as in a public clone) the shipped copy must be that source; with git, when this tree has the development
    # history, `git show` must give the same bytes.
    pins = rop.pinned("72d03874")
    assert len(pins) == 1, pins
    src, route = rop.prover_source("72d03874")
    assert route.startswith("shipped ") and _sha(src) in pins
    monkeypatch.delenv("GIT_DIR")
    src2, route2 = rop.prover_source("72d03874")
    assert _sha(src2) in pins and src2 == src
    assert route2.startswith(("git show ", "shipped "))
