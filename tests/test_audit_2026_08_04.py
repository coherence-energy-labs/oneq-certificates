r"""The six false-acceptance paths an external audit found, pinned.

Every one of these was ACCEPTED by a shipped checker on 2026-08-04. They
are regression tests in the strict sense: each fails if its defect
returns, and each is written from the auditor's own witness rather than
from my reconstruction of it.

The audit's verdict was RED and correct. These exist so that verdict can
never be earned twice by the same route.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.matching_cert import (BOUNDARY, MatchingCertificate,  # noqa: E402
                                check)

NAN_BITS = "0x7ff8000000000000"


def test_F4_a_provably_suboptimal_correction_is_REFUSED():
    """The auditor's witness, verbatim. True optimum 1.0 via the boundary;
    the direct edge weighs 1.0000005 and used to certify because the gap
    sat under a 1e-6 tolerance."""
    edges = [(0, 1, 1.0000005), (0, BOUNDARY, 0.5), (1, BOUNDARY, 0.5)]
    cert = MatchingCertificate(
        matched_edges=((0, 1),), node_potentials={},
        blossom_duals={frozenset({0}): 0.5, frozenset({1}): 0.5},
        claimed_weight=None, metric_closure_certified=True)
    r = check(edges=edges, syndrome=[0, 1], cert=cert)
    assert not r.accepted, (
        f"a correction heavier than the optimum by 5e-7 certified as "
        f"OPTIMAL: {r.reason}")


def test_F4_a_genuinely_tight_certificate_still_certifies():
    """The other direction, and the one that makes the fix worth having.
    Removing a tolerance is easy; removing it without refusing real work
    is the job."""
    edges = [(0, 1, 2.0)]
    cert = MatchingCertificate(
        matched_edges=((0, 1),), node_potentials={},
        blossom_duals={frozenset({0}): 2.0}, claimed_weight=None,
        metric_closure_certified=True)
    r = check(edges=edges, syndrome=[0, 1], cert=cert)
    assert r.accepted, r.reason


@pytest.mark.parametrize("where", ["cut_dual", "edge_weight"])
def test_F5_NaN_is_REFUSED_not_accepted(where):
    """Comparisons cannot catch NaN -- every one of them is False, so a
    NaN walked through every refusal and out the accept path.

    The "potential" arm became "cut_dual" when C-3 closed: a non-empty
    node_potentials is now refused by the FORMULATION guard, one step
    before finiteness is tested, so that arm could no longer reach the
    arithmetic it was written to probe. The dual value a certificate can
    still carry is z, so the NaN goes there. F5's finding is unchanged;
    only the field that can carry it moved."""
    nan = float("nan")
    edges = [(0, 1, nan if where == "edge_weight" else 1.0)]
    duals = {frozenset({0}): nan if where == "cut_dual" else 1.0}
    r = check(edges=edges, syndrome=[0, 1],
              cert=MatchingCertificate(
                  matched_edges=((0, 1),), node_potentials={},
                  blossom_duals=duals, claimed_weight=None,
                  metric_closure_certified=True))
    assert not r.accepted, f"NaN in the {where} certified: {r.reason}"
    assert "finite" in r.reason.lower(), r.reason


def test_F5_a_NaN_POTENTIAL_is_still_refused_just_earlier():
    """The arm C-3 displaced, kept rather than dropped.

    A dropped arm is a coverage hole that looks like a passing suite. The
    formulation guard refuses it before finiteness is reached, so the
    REASON differs -- but `accepted` must still be False, which is the
    property F5 was ever about."""
    nan = float("nan")
    r = check(edges=[(0, 1, 1.0)], syndrome=[0, 1],
              cert=MatchingCertificate(
                  matched_edges=((0, 1),),
                  node_potentials={0: nan, 1: 0.0},
                  blossom_duals={}, claimed_weight=None,
                  metric_closure_certified=True))
    assert not r.accepted, f"a NaN potential certified: {r.reason}"
    assert "node potentials" in r.reason, r.reason


def _tree_doc(**over):
    d = {"checks": [[0, 1]], "weights": [1, 1], "syndrome": [1],
         "candidate": [0], "tree": {"kind": "bound", "duals": []}}
    d.update(over)
    return d


def _run_tree(doc, tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools/external/tree_checker_b.py"),
         str(p)], capture_output=True, text=True, timeout=120)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_F6_non_integer_weights_are_REFUSED_not_truncated(tmp_path):
    """`int(1.9)` is 1. A weight vector of [1.9, 1.0] silently became
    [1, 1] and certified a candidate against weights nobody supplied."""
    rc, out = _run_tree(_tree_doc(weights=[1.9, 1.0]), tmp_path)
    assert rc != 0, f"1.9 was truncated to 1 and accepted: {out[:200]}"
    assert "not an integer" in out, out[:200]


def test_F6_negative_candidate_index_is_REFUSED(tmp_path):
    """Python's negative indexing is a gift to a forger: -1 means 'the
    last element', so an out-of-range index selects real data instead of
    being refused."""
    rc, out = _run_tree(_tree_doc(candidate=[-1]), tmp_path)
    assert rc != 0, f"index -1 was accepted as the last element: {out[:200]}"
    assert "out of range" in out, out[:200]


@pytest.mark.skipif(not (ROOT / "verifier-js").exists(),
                    reason="js verifier absent")
def test_F5_the_JS_checker_also_refuses_NaN():
    """The same defect existed in the second implementation. Two checkers
    sharing a blind spot is worse than one, because agreement between
    them is the evidence."""
    import shutil
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = ROOT / "verifier-js" / "verify-certificate.mjs"
    src = js.read_text(encoding="utf-8")
    assert "Number.isFinite" in src, (
        "the JS h2f no longer guards finiteness -- the NaN fail-open is "
        "back in the second implementation")
