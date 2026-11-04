"""Historical contract migrations are separate from current checker equivalence.

These are bounded executable regression vectors, not a requalification of any
old campaign. All preserved snapshots retain their original normalized hashes.
Current-equivalence comparisons never use a historical-transition allowlist.
"""
from dataclasses import asdict
from fractions import Fraction
import hashlib
from pathlib import Path

import pytest

from oneq import qldpc_check as live
from tools import checker_equivalence_gate as gate


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SHA256 = "2580ac1f14a97f54a2436b411dbe15609cc69b3bab4848d38a08d7dc7a22eee0"
PRE_ZERO_SHA256 = "59773707a8b85046f66149434aea9907b0990f7a70b873ab79be4234a1a51b25"
PRE_MERGE_SHA256 = "51940b3712a830895cbffa3516c81ba57836d470dbe52085219e581709c15ecc"
CURRENT_SHA256 = "c923c984c60a0888233cd00ccfea40995877cace05c572744f24dbf25a50b273"


@pytest.fixture(scope="module")
def snapshots():
    historical = ROOT / "tools/frozen/historical/qldpc_check_pre_input_contract_20261103.py"
    current = ROOT / "tools/frozen/qldpc_check.py"
    assert hashlib.sha256(gate._norm_bytes(historical)).hexdigest() == HISTORICAL_SHA256
    assert hashlib.sha256(gate._norm_bytes(current)).hexdigest() == CURRENT_SHA256
    return (gate._load(historical, "historical_input_contract_checker"),
            gate._load(current, "current_input_contract_checker"))


def _invoke(module, entry, *, checks=((0,),), weights=None, syndrome=(1,),
            candidate=(0,), facets=((0, (), 1),)):
    weights = {0: 1} if weights is None else weights
    kwargs = dict(checks=checks, weights=weights, syndrome=syndrome)
    if entry == "check_qldpc_bnb":
        kwargs.update(error_support=candidate, tree=("bound", facets))
    else:
        kwargs["cert"] = module.QldpcCertificate(candidate, facets)
    return getattr(module, entry)(**kwargs)


def test_current_gate_pins_the_reviewed_refreeze_and_has_no_transition_alias(snapshots):
    gate._verify_freeze()
    assert gate.FROZEN_SHA256["qldpc_check.py"] == CURRENT_SHA256
    assert HISTORICAL_SHA256 not in gate.FROZEN_SHA256.values()
    assert PRE_ZERO_SHA256 not in gate.FROZEN_SHA256.values()
    assert PRE_MERGE_SHA256 not in gate.FROZEN_SHA256.values()
    assert gate._REASON_ALIASES == []


@pytest.mark.parametrize("entry", ["check_qldpc", "check_qldpc_exact", "check_qldpc_bnb"])
def test_well_formed_control_remains_accepted_in_both_snapshots(snapshots, entry):
    historical, current = snapshots
    old, reference, result = [_invoke(module, entry) for module in (historical, current, live)]
    assert old.accepted is reference.accepted is result.accepted is True
    assert asdict(result) == asdict(reference)
    if entry == "check_qldpc":
        assert asdict(old) == asdict(result)
        assert gate._agree(gate._norm(old), gate._norm(result)) == (True, False)
    else:
        # The pre-input-contract snapshot predates the explicit status field.
        # This is a precise historical payload change, not current equivalence.
        expected = asdict(old)
        assert "status" not in expected["checks"]
        expected["checks"]["status"] = "EXACT_OPTIMAL"
        assert asdict(result) == expected
        assert gate._agree(gate._norm(old), gate._norm(result)) == (False, False)


@pytest.mark.parametrize("entry", ["check_qldpc", "check_qldpc_exact", "check_qldpc_bnb"])
@pytest.mark.parametrize("changed,expected_reason", [
    ({"candidate": (0, 0)}, "INPUT DOMAIN: candidate lists variable 0 twice (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md)"),
    ({"syndrome": (True,)}, "syndrome[0] is a boolean, not an integer"),
    ({"checks": (("0",),)}, "INPUT DOMAIN: check 0 entry '0' is not an integer (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md)"),
])
def test_historical_acceptance_now_refuses_only_the_malformed_specimen(snapshots, entry, changed, expected_reason):
    historical, current = snapshots
    old = _invoke(historical, entry, **changed)
    reference = _invoke(current, entry, **changed)
    result = _invoke(live, entry, **changed)
    assert old.accepted is True
    assert asdict(result) == asdict(reference)
    assert asdict(result) == dict(accepted=False, reason=expected_reason,
                                 primal_weight=None, dual_bound=None, gap=None, checks={})
    # A historical change is deliberately NOT current equivalence.
    assert gate._agree(gate._norm(old), gate._norm(result)) == (False, False)


@pytest.mark.parametrize("weight,exception,expected_reason", [
    ("heavy", ValueError, "INPUT DOMAIN: w[0] 'heavy' is not a real number (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md)"),
    (None, TypeError, "INPUT DOMAIN: w[0] None is not a real number (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md)"),
])
def test_historical_float_exception_is_a_precise_current_domain_refusal(snapshots, weight, exception, expected_reason):
    historical, current = snapshots
    with pytest.raises(exception):
        _invoke(historical, "check_qldpc", weights={0: weight})
    result = _invoke(live, "check_qldpc", weights={0: weight})
    assert asdict(result) == asdict(_invoke(current, "check_qldpc", weights={0: weight}))
    assert asdict(result) == dict(accepted=False, reason=expected_reason,
                                 primal_weight=None, dual_bound=None, gap=None, checks={})


@pytest.mark.parametrize("threshold", [Fraction(-1, 4), Fraction(0), Fraction(1, 4)])
@pytest.mark.parametrize("infeasible", [False, True])
def test_historical_pseudo_optimum_becomes_only_a_lattice_lower_bound(snapshots, threshold, infeasible):
    historical, current = snapshots
    kwargs = dict(checks=[()] if infeasible else [(0,)],
                  weights={0: Fraction(1, 2)}, syndrome=[1], error_support=(),
                  tree=("infeasible", 0) if infeasible else ("bound", ((0, (), Fraction(1, 2)),)),
                  bound_only_threshold=threshold)
    old, reference, result = [module.check_qldpc_bnb(**kwargs)
                              for module in (historical, current, live)]
    assert old.accepted is result.accepted is True
    assert old.reason.startswith("EXACT-OPTIMAL BY BRANCH DUALS:")
    assert old.primal_weight == float(threshold + Fraction(1, 2))
    assert asdict(result) == asdict(reference)
    assert result.primal_weight is result.dual_bound is result.gap is None
    lower = Fraction((threshold * 2).__floor__() + 1, 2)
    assert result.checks == {
        "status": "LOWER_BOUND_PROVEN", "exact": True, "bnb": True,
        "lattice_denominator": 2,
        "threshold": {"num": threshold.numerator, "den": threshold.denominator},
        "lower_bound": {"num": lower.numerator, "den": lower.denominator},
        "feasibility": "not established",
    }
    assert result.reason == (
        f"LOWER_BOUND_PROVEN BY BRANCH DUALS: every leaf crosses the threshold {threshold} "
        f"in exact arithmetic, so every solution of the system weighs at least {lower}; "
        "no candidate was checked, so no optimum, primal weight or feasibility is claimed")
    assert gate._agree(gate._norm(old), gate._norm(result)) == (False, False)


def test_current_bound_proof_refuses_equality_at_the_threshold(snapshots):
    _, current = snapshots
    kwargs = dict(checks=[(0,)], weights={0: Fraction(1, 2)}, syndrome=[1],
                  error_support=(), tree=("bound", ((0, (), Fraction(1, 2)),)),
                  bound_only_threshold=Fraction(1, 2))
    result = live.check_qldpc_bnb(**kwargs)
    assert result.accepted is False
    assert result.primal_weight is None
    assert asdict(result) == asdict(current.check_qldpc_bnb(**kwargs))


def test_current_comparison_refuses_a_roll_back_to_the_historical_checker(snapshots):
    historical, current = snapshots
    diffs, stats = gate.run_corpus(current, historical, cases=16, seed=20260804)
    assert diffs
    assert "contract_transitions" not in stats
    assert any(d["fn"] == "check_qldpc_bnb" for d in diffs)
    assert any(d["fn"] == "check_qldpc/abuse" for d in diffs)


@pytest.fixture(scope="module")
def pre_zero_snapshot():
    path = ROOT / "tools/frozen/historical/qldpc_check_pre_zero_multiplier_20261103.py"
    assert hashlib.sha256(gate._norm_bytes(path)).hexdigest() == PRE_ZERO_SHA256
    return gate._load(path, "historical_zero_multiplier_checker")


@pytest.mark.parametrize("entry", ["check_qldpc", "check_qldpc_exact", "check_qldpc_bnb"])
def test_valid_zero_multiplier_facet_preserves_optimality_and_counts_validation(
        snapshots, pre_zero_snapshot, entry):
    _, current = snapshots
    kwargs = dict(checks=((0,), (1,)), weights={0: 1, 1: 1}, syndrome=(1, 0),
                  facets=((0, (), 1), (1, (1,), 0)))
    old, reference, result = [
        _invoke(module, entry, **kwargs)
        for module in (pre_zero_snapshot, current, live)
    ]
    assert old.accepted is reference.accepted is result.accepted is True
    assert asdict(reference) == asdict(result)
    before, after = asdict(old), asdict(reference)
    if entry != "check_qldpc_bnb":
        # The old checker did not count the valid zero facet it skipped.
        # This is an explicit historical metadata change, not an alias in
        # the current-equivalence gate.
        assert before["checks"]["facets_real"] == 1
        assert after["checks"]["facets_real"] == 2
        before["checks"]["facets_real"] = 2
    assert before == after


@pytest.mark.parametrize("entry", ["check_qldpc", "check_qldpc_exact", "check_qldpc_bnb"])
@pytest.mark.parametrize("zero_facet,reason_fragment", [
    ((0, (0,), 0), "RIGHT parity"),
    ((0, (1, 2), 0), "check"),
    ((999, (), 0), "check 999"),
    (((0,), (), 0), "not >=2 valid rows"),
    ((0, (), 0), "listed twice"),
], ids=["wrong-parity", "outside-check", "unknown-row", "one-row-combination", "duplicate"])
def test_zero_multiplier_cannot_hide_a_malformed_facet(
        snapshots, pre_zero_snapshot, entry, zero_facet, reason_fragment):
    _, current = snapshots
    facets = ((0, (), 1), zero_facet)
    old, reference, result = [
        _invoke(module, entry, facets=facets)
        for module in (pre_zero_snapshot, current, live)
    ]
    assert old.accepted is True
    assert reference.accepted is result.accepted is False
    assert asdict(result) == asdict(reference)
    assert reason_fragment in result.reason
    # The historical acceptance is deliberately not current equivalence.
    assert gate._agree(gate._norm(old), gate._norm(result)) == (False, False)


@pytest.fixture(scope="module")
def before_merged_contract():
    path = ROOT / "tools/frozen/historical/qldpc_check_pre_merge_contract_20261104.py"
    assert hashlib.sha256(gate._norm_bytes(path)).hexdigest() == PRE_MERGE_SHA256
    return gate._load(path, "before_merged_api_contract")


@pytest.mark.parametrize("budget", [True, 1.0, 1.9, "1", Fraction(1)], ids=repr)
def test_pre_merge_budget_coercion_is_an_explicit_historical_change(
        before_merged_contract, snapshots, budget):
    _, current = snapshots
    kwargs = dict(checks=[(0,)], weights={0: 1}, syndrome=[1],
                  error_support=(0,), tree=("bound", ((0, (), 1),)), max_nodes=budget)
    old = before_merged_contract.check_qldpc_bnb(**kwargs)
    reference = current.check_qldpc_bnb(**kwargs)
    result = live.check_qldpc_bnb(**kwargs)
    assert old.accepted is True
    assert result.accepted is False
    assert result.reason.startswith("REJECTED MALFORMED INPUT: max_nodes ")
    assert result.primal_weight is result.dual_bound is result.gap is None
    assert asdict(reference) == asdict(result)
    assert gate._agree(gate._norm(old), gate._norm(result)) == (False, False)


@pytest.mark.parametrize("budget,accepted", [(1, True), (0, False), (-1, False)])
def test_pre_merge_integer_budget_boundaries_remain_unchanged(
        before_merged_contract, snapshots, budget, accepted):
    _, current = snapshots
    kwargs = dict(checks=[(0,)], weights={0: 1}, syndrome=[1],
                  error_support=(0,), tree=("bound", ((0, (), 1),)), max_nodes=budget)
    results = [m.check_qldpc_bnb(**kwargs) for m in (before_merged_contract, current, live)]
    assert all(r.accepted is accepted for r in results)
    assert all(asdict(r) == asdict(results[0]) for r in results)


@pytest.mark.parametrize("tolerance,exception", [
    (None, TypeError), ("1", TypeError), ([], TypeError), ({}, TypeError),
    (complex(1), TypeError), (10 ** 1000, OverflowError),
], ids=["none", "string", "list", "mapping", "complex", "overflow"])
def test_pre_merge_screen_exception_becomes_a_named_refusal(
        before_merged_contract, snapshots, tolerance, exception):
    _, current = snapshots
    kwargs = dict(checks=[(0,)], weights={0: 1}, syndrome=[1], tol=tolerance)
    def invoke(module):
        cert = module.QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1),))
        return module.check_qldpc(cert=cert, **kwargs)
    with pytest.raises(exception) as old:
        invoke(before_merged_contract)
    reference, result = invoke(current), invoke(live)
    assert asdict(reference) == asdict(result)
    assert asdict(result) == dict(accepted=False,
                                 reason=f"REJECTED MALFORMED INPUT: {old.value}",
                                 primal_weight=None, dual_bound=None, gap=None, checks={})


def test_current_corpus_detects_both_pre_merge_contract_gaps(before_merged_contract, snapshots):
    _, current = snapshots
    diffs, stats = gate.run_corpus(before_merged_contract, live, cases=0, seed=20261104)
    assert stats["api_contract"] == 22
    assert stats["budget_contract"] == 16
    assert stats["screen_refusal_contract"] == 6
    assert len(diffs) == 16
    assert sum(d["axis"].startswith("budget/") for d in diffs) == 10
    assert sum(d["axis"].startswith("screen-refusal/") for d in diffs) == 6
    assert stats["aliased"] == 0
    current_diffs, current_stats = gate.run_corpus(current, live, cases=0, seed=20261104)
    assert current_diffs == []
    assert current_stats["api_contract"] == stats["api_contract"]
    assert current_stats["aliased"] == 0


@pytest.fixture(scope="module")
def previous_public_checkers():
    history = ROOT / "tools/frozen/historical"
    paths = [
        ("qldpc_check_public_pre_contract_sync_20261103.py",
         "ed2f10cfb1669680bb0f721c78a2ebebe430817149ea52720f0fcc12e2073920"),
        ("matching_cert_epsilon_public_pre_contract_sync_20261103.py",
         "ac8ef04df05a6eb5cf40d6effc7f18f60f8348e5c84d90b37e2d87f13e401fec"),
    ]
    modules = []
    for filename, sha256 in paths:
        path = history / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
        modules.append(gate._load(path, "public_before_sync_" + path.stem))
    return tuple(modules)


@pytest.mark.parametrize("entry", ["check_qldpc", "check_qldpc_exact", "check_qldpc_bnb"])
def test_public_previous_checker_accepted_the_hidden_zero_facet(
        previous_public_checkers, entry):
    previous, _ = previous_public_checkers
    facets = ((0, (), 1), (999, (), 0))
    old = _invoke(previous, entry, facets=facets)
    current = _invoke(live, entry, facets=facets)
    assert old.accepted is True
    assert current.accepted is False and "check 999" in current.reason


@pytest.mark.parametrize("threshold", [Fraction(-1, 4), Fraction(0), Fraction(1, 4)])
@pytest.mark.parametrize("infeasible", [False, True])
def test_public_bound_only_metadata_migrates_without_inventing_a_candidate(
        previous_public_checkers, threshold, infeasible):
    previous, _ = previous_public_checkers
    kwargs = dict(checks=[()] if infeasible else [(0,)],
                  weights={0: Fraction(1, 2)}, syndrome=[1], error_support=(),
                  tree=("infeasible", 0) if infeasible else ("bound", ((0, (), Fraction(1, 2)),)),
                  bound_only_threshold=threshold)
    old = previous.check_qldpc_bnb(**kwargs)
    current = live.check_qldpc_bnb(**kwargs)
    assert old.accepted is current.accepted is True
    assert old.primal_weight is current.primal_weight is None
    assert old.dual_bound is current.dual_bound is None
    assert old.gap is current.gap is None
    assert old.checks["claim"] == "lower_bound" and old.checks["relation"] == ">"
    assert current.checks["status"] == "LOWER_BOUND_PROVEN"
    assert current.checks["feasibility"] == "not established"
    assert "claim" not in current.checks and "relation" not in current.checks
    expected = Fraction((threshold * 2).__floor__() + 1, 2)
    assert current.checks["lower_bound"] == {"num": expected.numerator, "den": expected.denominator}


def test_public_numpy_endpoint_domain_change_is_explicit(previous_public_checkers):
    from types import SimpleNamespace
    import numpy as np
    from oneq.matching_cert_epsilon import check_epsilon

    _, previous = previous_public_checkers
    certificate = SimpleNamespace(matched_edges=[np.array([0, -1], dtype=np.int64)],
                                  node_potentials={}, blossom_duals={frozenset({0}): 1.0})
    kwargs = dict(edges=[(0, -1, 1.0)], syndrome=[0], cert=certificate, eps_max=0)
    assert previous.check_epsilon(**kwargs).accepted is True
    current = check_epsilon(**kwargs)
    assert current.accepted is False and current.status == "INVALID_INPUT"
