"""Regression cases for the November 3 audit's untrusted-input contracts.

Each malformed specimen is one field away from an accepted control. The
reference executable is exercised separately; it imports no production code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from oneq.qldpc_check import (
    QldpcCertificate,
    check_qldpc,
    check_qldpc_bnb,
    check_qldpc_exact,
)

ROOT = Path(__file__).resolve().parents[1]
BAD_INDICES = [0.9, 0.0, "0", False, Fraction(0)]


def _three_results(*, checks=((0,),), weights=None, syndrome=(1,),
                   candidate=(0,), facets=((0, (), 1),)):
    weights = {0: 1} if weights is None else weights
    cert = QldpcCertificate(candidate, facets)
    return (
        check_qldpc(checks=checks, weights=weights, syndrome=syndrome, cert=cert),
        check_qldpc_exact(checks=checks, weights=weights, syndrome=syndrome, cert=cert),
        check_qldpc_bnb(checks=checks, weights=weights, syndrome=syndrome,
                        error_support=candidate, tree=("bound", facets)),
    )


def test_integer_control_is_accepted_by_all_production_lanes():
    for result in _three_results():
        assert result.accepted, result.reason


@pytest.mark.parametrize("bad", BAD_INDICES, ids=repr)
@pytest.mark.parametrize("field", ["check", "weight", "candidate", "facet_row", "facet_support"])
def test_integer_fields_are_never_silently_reinterpreted(field, bad):
    changes = {
        "check": {"checks": ((bad,),)},
        "weight": {"weights": {bad: 1}},
        "candidate": {"candidate": (bad,)},
        "facet_row": {"facets": ((bad, (), 1),)},
        # x=0, F={0} is a valid forbidden-set facet for an even check.
        "facet_support": {"syndrome": (0,), "candidate": (),
                          "facets": ((0, (bad,), 1),)},
    }
    for result in _three_results(**changes[field]):
        assert not result.accepted, (field, bad, result)


@pytest.mark.parametrize("field", ["candidate", "facet_support", "redundant_rows"])
def test_duplicate_set_fields_are_refused(field):
    changes = {
        "candidate": {"candidate": (0, 0)},
        "facet_support": {"syndrome": (0,), "candidate": (),
                          "facets": ((0, (0, 0), 1),)},
        "redundant_rows": {"checks": ((0,), ()), "syndrome": (1, 0),
                           "facets": (((0, 1, 1), (), 1),)},
    }
    for result in _three_results(**changes[field]):
        assert not result.accepted, (field, result)


@pytest.mark.parametrize("bad", BAD_INDICES, ids=repr)
@pytest.mark.parametrize("field", ["branch", "infeasible_row", "budget"])
def test_tree_integer_fields_are_validated(field, bad):
    tree = ("branch", 0, ("infeasible", 0), ("bound", ()))
    kwargs = {"checks": [(0,)], "weights": {0: 1}, "syndrome": [1],
              "error_support": (0,), "tree": tree}
    if field == "branch":
        kwargs["tree"] = ("branch", bad, tree[2], tree[3])
    elif field == "infeasible_row":
        kwargs["tree"] = ("branch", 0, ("infeasible", bad), tree[3])
    else:
        kwargs["max_nodes"] = bad
    result = check_qldpc_bnb(**kwargs)
    assert not result.accepted, (field, bad, result)


@pytest.mark.parametrize("budget", [True, 1.0, "1", Fraction(1)], ids=repr)
def test_positive_noninteger_node_budget_is_refused(budget):
    kwargs = {"checks": [(0,)], "weights": {0: 1}, "syndrome": [1],
              "error_support": (0,), "tree": ("bound", ((0, (), 1),))}
    control = check_qldpc_bnb(**kwargs, max_nodes=1)
    assert control.accepted, control.reason
    result = check_qldpc_bnb(**kwargs, max_nodes=budget)
    assert not result.accepted, (budget, result)


def _reference(document, tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(ROOT / "tools/external/tree_checker_b.py"), str(path)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=10,
    )


def _document():
    return {"checks": [[0]], "weights": [1], "syndrome": [1],
            "candidate": [0], "tree": ["bound", [[0, [], 1, 1]]]}


def test_reference_tree_valid_control(tmp_path):
    result = _reference(_document(), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("syndrome", [[2], [-2], [1.0], [True], ["1"]])
def test_reference_tree_refuses_nonbinary_and_noninteger_syndromes(syndrome, tmp_path):
    document = _document()
    if syndrome[0] in (2, -2):
        document.update(candidate=[], tree=["bound", []])
    document["syndrome"] = syndrome
    result = _reference(document, tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "syndrome" in result.stdout


@pytest.mark.parametrize("bad", [0.9, 0.0, "0", False], ids=repr)
@pytest.mark.parametrize("field", ["check", "candidate", "facet_row", "branch", "infeasible_row"])
def test_reference_tree_integer_fields_are_strict(field, bad, tmp_path):
    document = _document()
    if field == "check":
        document["checks"] = [[bad]]
    elif field == "candidate":
        document["candidate"] = [bad]
    elif field == "facet_row":
        document["tree"][1][0][0] = bad
    elif field == "branch":
        document["tree"] = ["branch", bad, ["infeasible", 0], ["bound", []]]
    else:
        document["tree"] = ["branch", 0, ["infeasible", bad], ["bound", []]]
    result = _reference(document, tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "not an integer" in result.stdout


@pytest.mark.parametrize("field", ["weight", "numerator", "denominator"])
@pytest.mark.parametrize("bad", [1.0, True, "1"])
def test_reference_tree_rational_integer_fields_are_strict(field, bad, tmp_path):
    document = _document()
    if field == "weight":
        document["weights"] = [bad]
    else:
        document["tree"][1][0][2 if field == "numerator" else 3] = bad
    result = _reference(document, tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize("threshold,lower_bound", [
    (0, 1), (Fraction(1, 2), 1), (1, 2),
])
def test_bound_only_proves_threshold_without_inventing_an_optimum(threshold, lower_bound):
    result = check_qldpc_bnb(
        checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(),
        tree=("bound", ((0, (), 2),)), bound_only_threshold=threshold,
    )
    assert result.accepted, result.reason
    assert result.reason.startswith("LOWER_BOUND_PROVEN")
    assert result.primal_weight is None
    assert result.dual_bound is None
    assert result.gap is None
    assert "objective" not in result.checks
    assert result.checks["status"] == "LOWER_BOUND_PROVEN"
    assert result.checks["feasibility"] == "not established"
    assert result.checks["lower_bound"] == {"num": lower_bound, "den": 1}
    expected = Fraction(threshold)
    assert result.checks["threshold"] == {"num": expected.numerator,
                                          "den": expected.denominator}


def test_bound_only_infeasible_instance_does_not_claim_a_finite_optimum():
    result = check_qldpc_bnb(
        checks=[()], weights={}, syndrome=[1], error_support=(),
        tree=("infeasible", 0), bound_only_threshold=0,
    )
    assert result.accepted and result.reason.startswith("LOWER_BOUND_PROVEN")
    assert result.primal_weight is None and "objective" not in result.checks


def test_refused_bound_only_tree_does_not_claim_a_candidate_weight():
    result = check_qldpc_bnb(
        checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(),
        tree=("bound", ((0, (), 2),)), bound_only_threshold=2,
    )
    assert not result.accepted
    assert result.primal_weight is None and "objective" not in result.checks


def test_attained_candidate_mode_still_reports_the_true_optimum():
    result = check_qldpc_bnb(
        checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(0,),
        tree=("bound", ((0, (), 2),)),
    )
    assert result.accepted and result.reason.startswith("EXACT-OPTIMAL")
    assert result.primal_weight == 2
    assert result.checks["objective"] == {"num": 2, "den": 1}
