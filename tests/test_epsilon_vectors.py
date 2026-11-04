"""Standing conformance vectors for the epsilon claim, run against every implementation that decides it.

`conformance/epsilon_vectors.json` holds three cases of every class the differential fuzzer generates -- valid
outcomes on both sides of eps_max, and every malformed-input class that once split the implementations (2026-11-03:
20 classes, among them a fractional or boolean vertex id that the replay's loader coerced, a decimal-looking or
inexact hex literal that float.fromhex read as a different number, a self-loop and `true` in the syndrome that the
second Python verifier accepted, parallel edges that one reading collapsed and the other refused, a repeated set that
check_epsilon could not see). Each implementation must return the recorded verdict -- status, and the exact epsilon
and lower bound when it decides -- on every vector.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fuzz_epsilon_checkers as fz                         # noqa: E402

DOC = json.loads((ROOT / "conformance" / "epsilon_vectors.json").read_text(encoding="utf-8"))
VECTORS = DOC["vectors"]
ONCE_SPLIT = {
    "graph:parallel_edge_heavier", "graph:parallel_edge_lighter", "graph:self_loop", "hex:decimal_not_hex",
    "hex:json_number_not_string", "hex:not_binary64_extra_digits", "hex:overflow", "id:boolean_in_correction",
    "id:boolean_in_set", "id:boolean_in_syndrome", "id:fractional_in_correction", "id:fractional_in_set",
    "id:integral_float_in_correction", "id:string_in_correction", "id:unknown_vertex_in_set", "nonfinite:weight_inf",
    "nonfinite:weight_nan", "repeat:set_listed_twice_other_value", "repeat:set_listed_twice_permuted",
    "repeat:set_listed_twice_same_value"}


def _want(v):
    e = v["expected"]
    return (e["status"], e["epsilon"], e["lower_bound"])


def test_the_vectors_cover_every_class_that_once_split_the_implementations_and_both_verdict_kinds():
    classes = {v["class"] for v in VECTORS}
    assert ONCE_SPLIT <= classes, sorted(ONCE_SPLIT - classes)
    statuses = {v["expected"]["status"] for v in VECTORS}
    assert {"EXACT_OPTIMAL", "EPSILON_OPTIMAL", "NOT_PROVEN", "INVALID_INPUT"} <= statuses


@pytest.mark.parametrize("v", VECTORS, ids=[v["name"] for v in VECTORS])
def test_both_python_verifiers_return_the_recorded_verdict(v):
    a, b = fz.py_replay(v["case"])
    assert a == _want(v), ("check_epsilon", a)
    assert b == _want(v), ("verify_b", b)


def test_the_javascript_checker_returns_the_recorded_verdict_on_every_vector():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    got = fz.js_batch([v["case"] for v in VECTORS])
    wrong = [(v["name"], g, _want(v)) for v, g in zip(VECTORS, got) if g != _want(v)]
    assert not wrong, wrong[:5]


def test_a_vector_can_fail():
    """The harness is not dark: a vector whose recorded verdict is flipped is reported by every implementation."""
    v = next(x for x in VECTORS if x["expected"]["status"] == "EXACT_OPTIMAL")
    a, b = fz.py_replay(v["case"])
    assert a[0] == b[0] == "EXACT_OPTIMAL"
    bent = dict(v["case"], correction=v["case"]["correction"] + [v["case"]["correction"][0]] if v["case"]["correction"]
                else [[0, 0]])
    a, b = fz.py_replay(bent)
    assert a[0] == b[0] == "INVALID_INPUT"
    p = subprocess.run([shutil.which("node") or "node", str(fz.JS), "--batch"], capture_output=True, text=True,
                       input=json.dumps({"id": "x", **bent}) + "\n", encoding="utf-8") \
        if shutil.which("node") else None
    if p is not None:
        assert json.loads(p.stdout)["status"] == "INVALID_INPUT"
