"""The qLDPC checker's contract, pinned where an operator mutation score found it unpinned.

tools/mutation_score.py ran 320 mechanical mutants of src/oneq/qldpc_check.py
against the checker's tests (2026-09-14) and 70 survived. Eight compute the same
function as the original (evidence/mutation_score/qldpc_check.equivalents.json).
The rest were real gaps: refusals no test asserted, parity folding in branch
reductions that no tree exercised, lattice, budget and tolerance boundaries, and
one soundness defect in the floating-point screen, which admitted a weight of
-1e-10 on a free variable and then proved a non-optimal correction optimal.
Every test here fails on at least one of those mutants and passes on the checker
as specified.
"""
from __future__ import annotations

from fractions import Fraction as F

import pytest

from oneq.qldpc_check import QldpcCertificate as C
from oneq.qldpc_check import check_qldpc, check_qldpc_bnb, check_qldpc_exact


# --------------------------------------------------------------------------- the defect
def test_float_screen_refuses_a_tiny_negative_weight_that_would_hide_a_lighter_correction():
    # variable 1 is in no check: {0, 1} explains the syndrome and weighs 1 - 1e-10 < 1
    r = check_qldpc(checks=[(0,)], weights={0: 1.0, 1: -1e-10}, syndrome=[1],
                    cert=C(error_support=(0,), facet_duals=((0, (), 1.0),)))
    assert not r.accepted
    assert r.reason == "negative variable weight"


# --------------------------------------------------------------------------- float screen
ONE = dict(checks=[(0,)], weights={0: 1.0}, syndrome=[1])


def test_float_control_a_genuine_certificate_is_accepted_with_its_gate_record():
    r = check_qldpc(**ONE, cert=C(error_support=(0,), facet_duals=((0, (), 1.0),)))
    assert r.accepted, r.reason
    assert r.checks == {"syndrome_ok": True, "facets_real": 1, "dual_feasible": True,
                        "exact_gap_zero": True}


@pytest.mark.parametrize("kwargs, cert, prefix", [
    (dict(checks=[(0, 0)], weights={0: 1.0}, syndrome=[0]), C((), ()), "check 0 contains duplicate"),
    (dict(checks=[(0,)], weights={0: 1.0}, syndrome=[1, 0]), C((0,), ()), "syndrome length"),
    (dict(checks=[(0,)], weights={0: -1.0}, syndrome=[1]), C((0,), ()), "negative variable weight"),
    (ONE, C((7,), ()), "error uses unknown variable 7"),
    (ONE, C((0,), ((1, (), 1.0),)), "facet names check 1 of 1"),
    (ONE, C((0,), (((0, 1), (), 1.0),)), "redundant check (0, 1) is not >=2 valid rows"),
    (ONE, C((0,), ((0, (), -0.5),)), "negative facet dual"),
    (dict(checks=[(0,), (1,)], weights={0: 1.0, 1: 1.0}, syndrome=[1, 0]),
     C((0,), ((0, (1,), 1.0),)), "facet (0,[1]) is not inside its check"),
    (dict(checks=[(0, 1, 2)], weights={0: 1.0, 1: 1.0, 2: 1.0}, syndrome=[0]),
     C((), ((0, (0, 1), 0.25),)), "facet (0,[0, 1]) has the RIGHT parity"),
    (ONE, C((0,), ((0, (), 0.5), (0, (), 0.5))), "facet (0,[]) listed twice"),
    (ONE, C((0,), ((0, (), 1.0),), box_duals=((0, -0.5),)), "negative box dual"),
    (dict(checks=[(0, 1), (1, 2)], weights={0: 1.0, 1: 3.0, 2: 1.0}, syndrome=[1, 1]),
     C((0, 2), ((0, (), 1.0), (1, (), 1.0), ((0, 1), (0, 2), 0.5))),
     "facet ((0, 1),[0, 2]) has the RIGHT parity"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_float_screen_refusals_come_from_the_screen(kwargs, cert, prefix):
    r = check_qldpc(**kwargs, cert=cert)
    assert not r.accepted
    assert r.reason.startswith(prefix), r.reason


def test_float_dual_bound_above_the_weight_is_impossible():
    r = check_qldpc(checks=[(0,)], weights={0: 1e-6}, syndrome=[1], tol=1.0,
                    cert=C(error_support=(0,), facet_duals=((0, (), 4e-6),)))
    assert not r.accepted and r.reason.startswith("IMPOSSIBLE"), r.reason


def test_a_facet_dual_at_exactly_minus_tol_passes_the_screen_and_the_exact_gate_refuses_it():
    r = check_qldpc(checks=[(0,), (1,)], weights={0: 1.0, 1: 1.0}, syndrome=[1, 0],
                    cert=C(error_support=(0,), facet_duals=((0, (), 1.0), (1, (1,), -1e-9))))
    assert not r.accepted
    assert r.reason.startswith("NOT PROVEN OPTIMAL (exact)"), r.reason


def test_an_overload_of_exactly_tol_passes_the_screen_and_the_exact_safe_dual_accepts():
    r = check_qldpc(**ONE, cert=C(error_support=(0,), facet_duals=((0, (), 1.0 + 1e-9),)))
    assert r.accepted, r.reason


def test_a_box_dual_of_exactly_minus_tol_passes_the_screen():
    r = check_qldpc(**ONE, cert=C(error_support=(0,), facet_duals=((0, (), 1.0),),
                                  box_duals=((0, -1e-9),)))
    assert r.accepted, r.reason


def test_a_float_gap_of_exactly_minus_1e6_is_not_impossible():
    assert 1e-6 - 2e-6 == -1e-6 and 1e-6 + 1e-6 == 2e-6      # the construction is exact
    r = check_qldpc(checks=[(0,)], weights={0: 1e-6}, syndrome=[1], tol=1e-6,
                    cert=C(error_support=(0,), facet_duals=((0, (), 2e-6),)))
    assert r.accepted, r.reason


def test_a_float_gap_of_exactly_1e6_is_decided_by_the_exact_gate():
    assert 2e-6 - 1e-6 == 1e-6
    r = check_qldpc(checks=[(0,)], weights={0: 2e-6}, syndrome=[1],
                    cert=C(error_support=(0,), facet_duals=((0, (), 1e-6),)))
    assert not r.accepted
    assert r.reason.startswith("NOT PROVEN OPTIMAL (exact)"), r.reason


# --------------------------------------------------------------------------- exact path
EONE = dict(checks=[(0,)], weights={0: F(1)}, syndrome=[1])


@pytest.mark.parametrize("kwargs, cert, needle", [
    (dict(checks=[(0, 0)], weights={0: F(1)}, syndrome=[0]), C((), ()), "duplicate"),
    (dict(checks=[(0,)], weights={0: F(1)}, syndrome=[1, 0]), C((0,), ()), "syndrome length"),
    (dict(checks=[(0,)], weights={0: F(-1)}, syndrome=[1]), C((0,), ()), "negative variable weight"),
    (EONE, C((7,), ()), "unknown variable 7"),
    (EONE, C((0,), ((1, (), F(1)),)), "facet names check 1 of 1"),
    (EONE, C((0,), (((0, 1), (), F(1)),)), "is not >=2 valid rows"),
    (EONE, C((0,), (((0,), (), F(1)),)), "is not >=2 valid rows"),
    (dict(checks=[(0, 1, 2)], weights={0: F(1), 1: F(1), 2: F(1)}, syndrome=[0]),
     C((), ((0, (0, 1), F(1, 4)),)), "RIGHT"),
    (dict(checks=[(0, 1), (1, 2)], weights={0: F(1), 1: F(3), 2: F(1)}, syndrome=[1, 1]),
     C((0, 2), ((0, (), F(1)), (1, (), F(1)), ((0, 1), (0, 2), F(1, 2)))), "RIGHT"),
    (dict(checks=[(0,)], weights={0: True}, syndrome=[1]), C((0,), ((0, (), F(1)),)), "exact rational"),
    (dict(checks=[(0, 1)], weights={0: F(1)}, syndrome=[1]), C((0,), ((0, (), F(1)),)), "NOT PROVEN"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_exact_refusals(kwargs, cert, needle):
    r = check_qldpc_exact(**kwargs, cert=cert)
    assert not r.accepted
    assert needle in r.reason, r.reason


def test_exact_accepts_zero_and_fractional_weights():
    r = check_qldpc_exact(checks=[(0,)], weights={0: F(1, 2), 1: 0}, syndrome=[1],
                          cert=C(error_support=(0,), facet_duals=((0, (), F(1, 2)),)))
    assert r.accepted, r.reason


def test_exact_accepts_the_empty_instance():
    r = check_qldpc_exact(checks=[], weights={}, syndrome=[], cert=C(error_support=(), facet_duals=()))
    assert r.accepted, r.reason


def test_exact_lattice_rule_accepts_a_bound_above_the_last_level_below_the_weight():
    r = check_qldpc_exact(**EONE, cert=C(error_support=(0,), facet_duals=((0, (), F(1, 2)),)))
    assert r.accepted and r.reason.startswith("EXACT-OPTIMAL"), r.reason
    assert r.gap == 0.5 and r.checks["exact"] is True and r.checks["lattice_denominator"] == 1


def test_exact_not_proven_reports_the_true_gap():
    r = check_qldpc_exact(checks=[(0,)], weights={0: F(1), 1: F(1, 8)}, syndrome=[1],
                          cert=C(error_support=(0,), facet_duals=((0, (), F(1, 4)),)))
    assert not r.accepted and r.reason.startswith("NOT PROVEN"), r.reason
    assert r.gap == 0.75


def test_exact_accepts_a_facet_equal_to_the_whole_check():
    r = check_qldpc_exact(checks=[(0, 1, 2)], weights={0: F(1), 1: F(1), 2: F(1)}, syndrome=[0],
                          cert=C(error_support=(), facet_duals=((0, (0, 1, 2), F(1, 4)),)))
    assert r.accepted, r.reason


# --------------------------------------------------------------------------- branch and bound
W2 = {0: F(1), 1: F(5)}
GENUINE = ("branch", 0, ("branch", 1, ("infeasible", 0), ("bound", ())), ("bound", ()))
# claims the heavier candidate {1} optimal; the x0=1, x1=0 leaf is feasible (0 = 0)
FORGED = ("branch", 0, ("branch", 1, ("infeasible", 0), ("bound", ())),
          ("branch", 1, ("infeasible", 0), ("infeasible", 0)))


def test_bnb_a_genuine_tree_is_accepted_with_its_gate_record():
    r = check_qldpc_bnb(checks=[(0, 1)], weights=W2, syndrome=[1], error_support=(0,), tree=GENUINE)
    assert r.accepted, r.reason
    assert r.checks["exact"] is True and r.checks["bnb"] is True


def test_bnb_a_forged_infeasible_leaf_under_a_one_branch_is_refused():
    r = check_qldpc_bnb(checks=[(0, 1)], weights=W2, syndrome=[1], error_support=(1,), tree=FORGED)
    assert not r.accepted
    assert "0=0: feasible" in r.reason, r.reason


def test_bnb_a_combination_of_rows_is_folded_by_xor_not_or():
    # x5 = 1 and x5 = 1 are consistent: two odd rows with the same support sum to 0 = 0
    r = check_qldpc_bnb(checks=[(5,), (5,)], weights={5: F(1)}, syndrome=[1, 1], error_support=(5,),
                        tree=("infeasible", (0, 1)))
    assert not r.accepted and "0=0: feasible" in r.reason, r.reason


@pytest.mark.parametrize("kwargs, e, tree, needle", [
    (dict(checks=[(0, 0)], weights={0: F(1)}, syndrome=[0]), (), ("bound", ()), "duplicate"),
    (dict(checks=[(0,)], weights={0: F(1)}, syndrome=[1, 0]), (0,), ("bound", ()), "syndrome length"),
    (dict(checks=[(0,)], weights={0: F(-1)}, syndrome=[1]), (0,), ("bound", ()), "negative weight"),
    (EONE, (7,), ("bound", ()), "unknown variable 7"),
    (EONE, (0,), ("infeasible", 1), "bad rows"),
    (EONE, (0,), (), "malformed node"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_bnb_refusals(kwargs, e, tree, needle):
    r = check_qldpc_bnb(**kwargs, error_support=e, tree=tree)
    assert not r.accepted
    assert needle in r.reason, r.reason


def test_bnb_accepts_zero_and_fractional_weights():
    r = check_qldpc_bnb(checks=[(0,)], weights={0: F(1, 2), 1: 0}, syndrome=[1], error_support=(0,),
                        tree=("bound", ((0, (), F(1, 2)),)))
    assert r.accepted, r.reason


def test_bnb_accepts_the_empty_instance():
    r = check_qldpc_bnb(checks=[], weights={}, syndrome=[], error_support=(), tree=("bound", ()))
    assert r.accepted, r.reason


def _tree(n, depth=0):
    """A full binary tree of n (odd) nodes, branching on its depth."""
    if n == 1:
        return ("bound", ())
    a = (n - 1) // 2
    a += a % 2 == 0
    return ("branch", depth, _tree(a, depth + 1), _tree(n - 1 - a, depth + 1))


FREE = dict(checks=[], weights={i: F(0) for i in range(64)}, syndrome=[], error_support=())


def test_bnb_a_tree_of_exactly_max_nodes_is_within_budget():
    r = check_qldpc_bnb(**FREE, tree=_tree(99_999), max_nodes=99_999)
    assert r.accepted, r.reason


def test_bnb_the_default_budget_is_one_hundred_thousand_nodes():
    r = check_qldpc_bnb(**FREE, tree=_tree(100_001))
    assert not r.accepted and "node budget exceeded" in r.reason, r.reason


def test_float_screen_accepts_zero_and_fractional_weights():
    r = check_qldpc(checks=[(0,)], weights={0: 0.5, 1: 0.0}, syndrome=[1],
                    cert=C(error_support=(0,), facet_duals=((0, (), 0.5),)))
    assert r.accepted, r.reason
