"""The equivalence gate must detect numerical and result-contract drift."""
from dataclasses import replace
from fractions import Fraction
import math
from types import SimpleNamespace

import pytest

from oneq.qldpc_check import QldpcCheckResult
from tools import checker_equivalence_gate as gate


def _control():
    return QldpcCheckResult(True, "controlled result", primal_weight=1.0,
                           dual_bound=1.0, gap=0.0,
                           checks={"status": "LOWER_BOUND_PROVEN",
                                   "lower_bound": {"num": 1, "den": 1}})


@pytest.mark.parametrize("field", ["primal_weight", "dual_bound", "gap"])
def test_sub_nanounit_numerical_change_is_visible(field):
    before = _control()
    old_value = getattr(before, field)
    after = replace(before, **{field: math.nextafter(old_value, math.inf)})
    assert round(old_value, 9) == round(getattr(after, field), 9)
    assert gate._agree(gate._norm(before), gate._norm(after)) == (False, False)


@pytest.mark.parametrize("checks", [
    {"status": "EXACT_OPTIMAL", "lower_bound": {"num": 1, "den": 1}},
    {"status": "LOWER_BOUND_PROVEN", "lower_bound": {"num": 2, "den": 1}},
    {"status": "LOWER_BOUND_PROVEN", "lower_bound": {"num": 1.0, "den": 1}},
    {"status": "LOWER_BOUND_PROVEN", "lower_bound": {"num": 1, "den": 1}, "new_claim": True},
])
def test_nested_metadata_and_return_schema_changes_are_visible(checks):
    before = _control()
    after = replace(before, checks=checks)
    assert gate._agree(gate._norm(before), gate._norm(after)) == (False, False)


def test_exact_fraction_change_is_visible_without_float_conversion():
    before = replace(_control(), checks={"exact": Fraction(2 ** 100 + 1, 2 ** 100)})
    after = replace(before, checks={"exact": Fraction(1)})
    assert float(before.checks["exact"]) == float(after.checks["exact"])
    assert gate._agree(gate._norm(before), gate._norm(after)) == (False, False)


def test_mapping_insertion_order_does_not_change_the_result():
    before = _control()
    after = replace(before, checks={"lower_bound": {"den": 1, "num": 1},
                                    "status": "LOWER_BOUND_PROVEN"})
    assert gate._agree(gate._norm(before), gate._norm(after)) == (True, False)


def test_comparator_errors_cannot_look_like_agreeing_checker_exceptions():
    result = replace(_control(), checks={"opaque": object()})
    implementation = SimpleNamespace(check=lambda: result)
    with pytest.raises(TypeError, match="unsupported checker result value"):
        gate._call(implementation, "check")


def test_checker_exception_reasons_are_part_of_comparison():
    def first():
        raise ValueError("first refusal")
    def second():
        raise ValueError("different refusal")
    before = gate._call(SimpleNamespace(check=first), "check")
    after = gate._call(SimpleNamespace(check=second), "check")
    assert before[0] == after[0] == "<raised>"
    assert gate._agree(before, after) == (False, False)
