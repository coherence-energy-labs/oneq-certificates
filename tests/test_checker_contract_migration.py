"""The historical differential gate must not excuse unrelated changes."""
from copy import deepcopy
from fractions import Fraction

import pytest

from oneq.qldpc_check import QldpcCheckResult, check_qldpc_bnb
from tools.checker_equivalence_gate import (
    _bound_only_transition,
    _duplicate_candidate_transition,
    _malformed_input_transition,
)


def _bound_control():
    old = (True, "EXACT-OPTIMAL BY BRANCH DUALS: historical payload", "1.0", None)
    current = check_qldpc_bnb(
        checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(),
        tree=("bound", ((0, (), 2),)), bound_only_threshold=0,
    )
    return old, current


def test_threshold_transition_requires_the_new_precise_claim():
    old, current = _bound_control()
    assert _bound_only_transition(old, current, 0, {0: 2})


@pytest.mark.parametrize("field,value", [
    ("accepted", False), ("primal_weight", 1), ("dual_bound", 0),
    ("gap", 0), ("reason", "EXACT-OPTIMAL"),
])
def test_threshold_transition_rejects_false_or_misleading_payloads(field, value):
    old, current = _bound_control()
    setattr(current, field, value)
    assert not _bound_only_transition(old, current, 0, {0: 2})


@pytest.mark.parametrize("key,value", [
    ("objective", {"num": 1, "den": 1}),
    ("threshold", {"num": 1, "den": 1}),
    ("relation", ">="), ("exact", False), ("claim", "optimal"),
    ("lattice_denominator", 2),
])
def test_threshold_transition_checks_all_contract_metadata(key, value):
    old, current = _bound_control()
    current = deepcopy(current)
    current.checks[key] = value
    assert not _bound_only_transition(old, current, 0, {0: 2})


def test_bound_transition_cannot_hide_a_changed_decision():
    _, current = _bound_control()
    old = (False, "TREE REFUSED: bound too weak", "1.0", None)
    assert not _bound_only_transition(old, current, 0, {0: 2})


def test_duplicate_transition_requires_a_duplicate_and_exact_refusal():
    result = (False, "REJECTED MALFORMED INPUT: candidate contains duplicate indices", None, None)
    assert _duplicate_candidate_transition((0, 0), result)
    assert not _duplicate_candidate_transition((0,), result)
    assert not _duplicate_candidate_transition((0, 0), (True, *result[1:]))
    assert not _duplicate_candidate_transition((0, 0), (False, "anything", None, None))


def test_boolean_transition_requires_the_raw_boolean_and_refusal():
    result = (False, "syndrome[0] is not an integer", None, None)
    assert _malformed_input_transition("bool_syndrome", [(0,)], {0: 1}, [True], (0,), result)
    assert not _malformed_input_transition("bool_syndrome", [(0,)], {0: 1}, [1], (0,), result)
    assert not _malformed_input_transition("bool_syndrome", [(0,)], {0: 1}, [True], (0,),
                                            (True, *result[1:]))


def test_ordinary_optimal_payload_is_never_a_threshold_transition():
    result = QldpcCheckResult(True, "EXACT-OPTIMAL", primal_weight=2,
                              checks={"objective": {"num": 2, "den": 1}})
    old, _ = _bound_control()
    assert not _bound_only_transition(old, result, Fraction(0), {0: 2})
