"""The syndrome is a GF(2) vector, and every checker entry point says so.

The checkers once read each entry as `int(b) & 1`, so a syndrome entry of 3
was silently read as 1 and -2 as 0: a malformed instance could be accepted as
if it were a different, well-formed one. Each case below is chosen so that the
old masking maps it onto the accepted instance -- the control proves the
certificate is otherwise good, so a refusal here can only come from the domain
check.
"""
from fractions import Fraction

import pytest

from oneq.qldpc_cert import QldpcCertificate, certified_bnb
from oneq.qldpc_check import check_qldpc, check_qldpc_bnb, check_qldpc_exact

CHECKS = [(0, 1)]
W_FLOAT = {0: 1.0, 1: 1.0}
W_EXACT = {0: Fraction(1), 1: Fraction(1)}
CERT = QldpcCertificate(error_support=(0,), facet_duals=((0, (), Fraction(1)),))

# every entry here reads as [1] under `& 1` (or as 1 under int())
BAD = [[3], [-1], [True + 2], [1.0], ["1"]]


def _tree():
    tree, _ = certified_bnb(CHECKS, W_FLOAT, [1], 1)
    assert tree is not None
    return tree


def _run_all(syndrome, tree):
    return [
        check_qldpc(checks=CHECKS, weights=W_FLOAT, syndrome=syndrome, cert=CERT),
        check_qldpc_exact(checks=CHECKS, weights=W_EXACT, syndrome=syndrome, cert=CERT),
        check_qldpc_bnb(checks=CHECKS, weights=W_EXACT, syndrome=syndrome,
                        error_support=(0,), tree=tree),
    ]


def test_control_binary_syndrome_is_accepted_by_all_three():
    tree = _tree()
    for r in _run_all([1], tree):
        assert r.accepted, r.reason


@pytest.mark.parametrize("syndrome", BAD, ids=repr)
def test_non_binary_syndrome_is_refused_by_all_three(syndrome):
    tree = _tree()
    for r in _run_all(syndrome, tree):
        assert not r.accepted
        assert "syndrome[0]" in r.reason, r.reason


def test_numpy_and_bool_entries_are_still_binary():
    np = pytest.importorskip("numpy")
    tree = _tree()
    for syn in ([np.uint8(1)], [np.int64(1)], [True]):
        for r in _run_all(syn, tree):
            assert r.accepted, r.reason
