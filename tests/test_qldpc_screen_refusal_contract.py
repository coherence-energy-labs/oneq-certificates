"""Direct float-screen API controls, independent of historical snapshots."""
import pytest

from oneq.qldpc_check import QldpcCertificate, check_qldpc


def _check(tolerance):
    return check_qldpc(
        checks=[(0,)], weights={0: 1.0}, syndrome=[1], tol=tolerance,
        cert=QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1.0),)),
    )


def test_valid_tolerance_retains_a_proven_optimal_control():
    result = _check(1e-9)
    assert result.accepted is True
    assert result.primal_weight == result.dual_bound == 1.0
    assert result.gap == 0.0
    assert result.checks["exact_gap_zero"] is True


@pytest.mark.parametrize("tolerance", [None, "1", [], {}, complex(1), 10 ** 1000],
                         ids=["none", "string", "list", "mapping", "complex", "overflow"])
def test_malformed_screen_input_is_a_refusal_without_invented_result_fields(tolerance):
    result = _check(tolerance)
    assert result.accepted is False
    assert result.reason.startswith("REJECTED MALFORMED INPUT: ")
    assert result.primal_weight is result.dual_bound is result.gap is None
    assert result.checks == {}
