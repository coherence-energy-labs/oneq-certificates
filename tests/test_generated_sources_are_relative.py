"""Every generated file of the four manuscripts names its sources by repository-relative path (2026-11-07).

Paper 3's generator reused a variable and wrote one prover's ABSOLUTE path (C:\\Users\\...) as the source comment of
two macros. The file still typeset; only the trial rehearsal of v2026.11.07.1 caught it, because a sterile copy
regenerated a different comment and `generated_files_reproduce` failed. This holds the product: no line of any
committed generated file carries a drive-letter or home-directory path. The generator itself now refuses a source
that is not a relative name (make_numbers_distance.put).
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATED = sorted(p for p in (ROOT / "paper").glob("*/generated/*.tex"))
# a Windows drive path, or a POSIX home or temp directory, anywhere on a line
LOCAL_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home|tmp)/")


def local_paths(text: str) -> list[str]:
    return [line for line in text.splitlines() if LOCAL_PATH.search(line)]


def test_the_detector_sees_the_line_that_shipped_and_passes_a_relative_one():
    shipped = r"% C:\Users\Josh\Projects\ONE_Q\experiments\circuit_distance_exact\old_provers\cdist_prove_72d03874.c"
    assert local_paths(shipped) == [shipped]
    assert local_paths("% /home/someone/oneq/records/x.jsonl") != []
    assert local_paths("% recert_20261102/manifest.json") == []
    assert local_paths(r"\newcommand{\RecertOldSource}{\texttt{experiments/circuit\_distance\_exact/old\_provers/x.c}}") == []


def test_there_are_generated_files_to_check():
    assert {p.parent.parent.name for p in GENERATED} >= {"certifying-decoders", "qldpc-certificates", "circuit-distance"}


@pytest.mark.parametrize("path", GENERATED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_generated_line_names_a_local_path(path: pathlib.Path):
    bad = local_paths(path.read_text(encoding="utf-8"))
    assert not bad, f"{path.relative_to(ROOT).as_posix()}: {bad[:3]}"
