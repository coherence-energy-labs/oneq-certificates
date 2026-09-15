r"""Certified branch-and-bound: the solver tier's replacement, gated.

The tree is only as trustworthy as the checker's refusals, so most of
this file is mutants: forged bounds, incomplete splits, refixed
variables, fake infeasibility combos, and the heavier-candidate attack.
The positive path is pinned by a brute-forced instance whose flat LP is
provably loose (the 6-cycle pseudocodeword).
"""
import itertools

from fractions import Fraction

from oneq.qldpc_cert import (QldpcCertificate, certified_bnb,
                             exactify_certificate, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb, check_qldpc_exact

# a brute-force-verified flat-LP-loose instance from the randomized
# validation harness (600-trial run, trial 93): root LP bound 8/3 while
# the integer optimum is 4 -- cuts do not close it; branching does
CHECKS = [(2, 3, 6, 7), (3, 5), (1, 4), (1, 3), (4, 5, 6),
          (1, 2, 3, 5), (4, 6, 7)]
SYN = [0, 0, 1, 1, 0, 0, 0]
N = 8
W = {i: 1.0 for i in range(N)}
WX = {i: Fraction(1) for i in range(N)}


def _brute():
    best, cand = None, None
    for bits in itertools.product((0, 1), repeat=N):
        if all(sum(bits[i] for i in c) % 2 == s
               for c, s in zip(CHECKS, SYN)):
            wt = sum(bits)
            if best is None or wt < best:
                best, cand = wt, tuple(i for i in range(N) if bits[i])
    return best, cand


def test_tree_certifies_a_flat_loose_instance():
    best, cand = _brute()
    assert best is not None
    tree, nodes = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    assert tree is not None
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=tree)
    assert r.accepted and r.checks["bnb"], r.reason


def test_the_TREE_path_recomputes_the_syndrome_too():
    r"""*** A GUARD ON ONE OF THREE PATHS IS A COINCIDENCE. ***

    `qldpc_check.py` recomputes the claimed error's syndrome in THREE
    places -- `_validate_correction`, `_bnb_target`, and
    `check_qldpc_exact` -- and the mutation registry aimed one mutant
    (`M48`) at the string, which the gate replaces at its FIRST
    occurrence. Deleting the parity test from either of the other two
    cost nothing, and `M298`/`M299` **SURVIVED** the first run that
    asked.

    The ghost here is the same WEIGHT as the optimum, deliberately: an
    over-weight ghost is refused by the tree bound before the syndrome
    is ever recomputed, so it would exercise the wrong guard and read
    like coverage. `(0,1,2,3)` weighs 4, the optimum weighs 4, and only
    the recomputed syndrome separates them.
    """
    best, _cand = _brute()
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    ghost = (0, 1, 2, 3)
    assert len(ghost) == best, "the ghost must not be separable by weight"
    assert any(len(set(ghost) & set(c)) % 2 != s
               for c, s in zip(CHECKS, SYN)), "the ghost IS a correction"
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=ghost, tree=tree)
    assert not r.accepted
    assert "NOT A VALID CORRECTION" in r.reason, (
        f"the tree path refused for the wrong reason: {r.reason}")


def test_the_EXACT_path_recomputes_the_syndrome_too():
    """The third of the three paths, and the one a stranger calls.

    Same shape as `test_a_non_correction_with_matching_weight_is_refused`
    in `test_qldpc_cert.py`, aimed at `check_qldpc_exact` rather than at
    the float entry point -- because that test reaches
    `_validate_correction` and nothing reached this one.
    """
    checks = [(0, 1)]
    weights = {0: Fraction(1), 1: Fraction(1), 2: Fraction(1)}
    syndrome = [1]
    ghost = QldpcCertificate(error_support=(2,),   # weight 1, syndrome 0
                             facet_duals=((0, (), Fraction(1)),))
    r = check_qldpc_exact(checks=checks, weights=weights,
                          syndrome=syndrome, cert=ghost)
    assert not r.accepted
    assert "NOT A VALID CORRECTION" in r.reason, r.reason


def test_heavier_candidate_refused_with_same_tree():
    # ANY strictly heavier feasible completion must be refused with the
    # optimal tree: its leaves prove > best-1, but a heavier candidate
    # raises U, so the same leaves no longer cross U-1. The first
    # version searched for exactly best+2 and SKIPPED on this instance
    # (whose only heavier completion is best+1) -- a dark gate; this
    # instance is brute-force-verified to HAVE a heavier completion, so
    # the search failing is now an assertion failure, never a skip.
    best, cand = _brute()
    heavier = None
    for bits in itertools.product((0, 1), repeat=N):
        if (sum(bits) > best
                and all(sum(bits[i] for i in c) % 2 == s
                        for c, s in zip(CHECKS, SYN))):
            heavier = tuple(i for i in range(N) if bits[i])
            break
    assert heavier is not None, \
        "instance lost its heavier completion: pick a new pinned instance"
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=heavier, tree=tree)
    assert not r.accepted


def test_mutant_empty_bound_leaf_refused():
    best, cand = _brute()
    fake = ("bound", ())        # claims bound 0 at the root: cannot cross
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=fake)
    assert not r.accepted and "does not cross" in r.reason


def test_mutant_branch_on_refixed_variable_refused():
    best, cand = _brute()
    bad = ("branch", 0, ("branch", 0, ("bound", ()), ("bound", ())),
           ("bound", ()))
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=bad)
    assert not r.accepted and "refixed" in r.reason


def test_mutant_fake_infeasible_leaf_refused():
    best, cand = _brute()
    # check 0 has nonempty support at the root: claiming it empty-odd lies
    fake = ("infeasible", 0)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=fake)
    assert not r.accepted


def test_mutant_fake_infeasible_row_combo_refused():
    best, cand = _brute()
    fake = ("infeasible", (0, 1))   # XOR of checks 0,1 = {0,2}, not empty
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=fake)
    assert not r.accepted and "not empty" in r.reason


def test_mutant_float_dual_in_leaf_rejected_by_type():
    best, cand = _brute()
    bad = ("bound", ((0, (), 1.0),))
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=bad)
    assert not r.accepted and "REJECTED" in r.reason


def test_mutant_infeasible_candidate_refused_before_tree():
    best, cand = _brute()
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=(0,), tree=tree)
    # weight-1 single-bit answer cannot satisfy this syndrome
    assert not r.accepted and "NOT A VALID CORRECTION" in r.reason


def test_node_budget_refuses_rather_than_accepts():
    best, cand = _brute()
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=cand, tree=tree, max_nodes=1)
    assert not r.accepted and "budget" in r.reason


def test_flat_exact_still_refuses_what_bnb_certifies():
    # the instance really is flat-LP-loose: the flat exact path must
    # refuse the optimal candidate, which is exactly why the tree exists
    best, cand = _brute()
    cert, _ = feldman_dual(CHECKS, W, SYN)
    ec = exactify_certificate(QldpcCertificate(
        error_support=cand, facet_duals=cert.facet_duals,
        box_duals=cert.box_duals))
    r = check_qldpc_exact(checks=CHECKS, weights=WX, syndrome=SYN, cert=ec)
    assert not r.accepted


# --- bound-only mode (frame determinacy / distance certificates) ---

def test_bound_only_proves_min_above_threshold():
    # the pinned instance's optimum is 4: a tree proving min > 3 must be
    # buildable and checkable with NO candidate
    best, _cand = _brute()
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=(), tree=tree,
                        bound_only_threshold=best - 1)
    assert r.accepted, r.reason


def test_bound_only_refuses_threshold_at_optimum():
    # min > 4 is FALSE (a weight-4 solution exists): the same tree's
    # leaves cross 3, not 4 -- refusal required
    best, _cand = _brute()
    tree, _ = certified_bnb(CHECKS, W, SYN, best, rpc_rounds=2)
    r = check_qldpc_bnb(checks=CHECKS, weights=WX, syndrome=SYN,
                        error_support=(), tree=tree,
                        bound_only_threshold=best)
    assert not r.accepted


def test_frame_determinacy_end_to_end_small():
    # augment with a logical row whose flip is infeasible below U+1;
    # frame_determinacy_trees must produce a tree the bound-only checker
    # accepts -- and a tampered empty tree must refuse
    from oneq.qldpc_cert import frame_determinacy_trees
    best, cand = _brute()
    lrow = (0, 1, 2)
    gstar = [sum(1 for i in cand if i in set(lrow)) % 2]
    trees, all_ok = frame_determinacy_trees(CHECKS, W, SYN, best,
                                            [lrow], gstar, rpc_rounds=2)
    ell, tree = trees[0]
    aug = CHECKS + [lrow]
    asyn = SYN + [gstar[0] ^ 1]
    if tree is not None:
        r = check_qldpc_bnb(checks=aug, weights=WX, syndrome=asyn,
                            error_support=(), tree=tree,
                            bound_only_threshold=best)
        assert r.accepted, r.reason
        # tamper: the empty-bound tree claims min > best with L=0
        rt = check_qldpc_bnb(checks=aug, weights=WX, syndrome=asyn,
                             error_support=(), tree=("bound", ()),
                             bound_only_threshold=best)
        assert not rt.accepted
    else:
        # honest refusal: verify the ambiguity is real by brute force
        import itertools as it
        amb = any(sum(bits) <= best
                  and all(sum(bits[i] for i in c) % 2 == s
                          for c, s in zip(aug, asyn))
                  for bits in it.product((0, 1), repeat=N))
        assert amb, "producer refused a provable frame bit"
