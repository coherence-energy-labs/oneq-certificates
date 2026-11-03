"""Adversarial soundness of P2's safe dual bound: L <= OPT, attacked with an LP.

THE THEOREM P2 RESTS ON.  `_exact_facet_scan` returns
    L = sum_r y_r (1 - |F_r|) - sum_i max(0, t_i - w_i)
and the exact accept path certifies a correction as minimum weight when its weight
EQUALS L.  That is a proof only if  L <= OPT  for every y >= 0 over valid facets.

The mathematics holds by a short argument: a correction satisfying a check's parity
cannot equal the indicator of a wrong-parity facet on that check, so it obeys
sum_{S\\F} x_i - sum_F x_i >= 1 - |F|; weighting by y >= 0 and summing gives
sum_i t_i x_i >= sum_r y_r (1 - |F_r|); and x_i in [0, 1] then absorbs the slack
term exactly onto L <= weight(x).  XOR-of-rows facets are sound because a sum of
parity constraints is a parity constraint.

THIS FILE TESTS THE CODE, WHICH CAN DIFFER FROM THE MATH -- sign conventions in the
loads, the XOR construction, zero weights, zero duals.  Its strongest arm does not
guess duals: it SOLVES AN LP for the y that maximises L, which is the dual most
likely to expose an unsound bound, then re-evaluates that y exactly.  Measured over
two seeds and 4,000 such instances before this file was written, the adversary
reached L = OPT (margin 0) and never exceeded it -- so the attack genuinely reaches
the boundary rather than probing safely inside it.

It is also the first step of a Lean 4 formalisation of this theorem: a theorem worth
proving in a kernel is one that has first survived the strongest attack available.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

from oneq import qldpc_check as qc

scipy_optimize = pytest.importorskip("scipy.optimize")


def _brute_opt(n, supports, syn, w):
    best = None
    for bits in itertools.product((0, 1), repeat=n):
        if all((sum(bits[i] for i in S) & 1) == s for S, s in zip(supports, syn)):
            wt = sum((w[i] for i in range(n) if bits[i]), Fraction(0))
            best = wt if best is None or wt < best else best
    return best


def _valid_facets(supports, syn, max_xor=2):
    out = []
    m = len(supports)
    rowsets = [(j,) for j in range(m)]
    for k in range(2, max_xor + 1):
        rowsets += list(itertools.combinations(range(m), k))
    for rows in rowsets:
        sup, par = frozenset(), 0
        for r in rows:
            sup ^= supports[r]
            par ^= syn[r]
        if not sup:
            continue
        name = rows[0] if len(rows) == 1 else rows
        for size in range(len(sup) + 1):
            if size % 2 == par:
                continue
            for F in itertools.combinations(sorted(sup), size):
                out.append((name, frozenset(F), sup))
    return out


def _lp_maximising_duals(n, facets, w):
    R = len(facets)
    c = np.zeros(R + n)
    for r, (_, F, _) in enumerate(facets):
        c[r] = -(1 - len(F))
    c[R:] = 1.0
    A, b = [], []
    for i in range(n):
        row = np.zeros(R + n)
        for r, (_, F, sup) in enumerate(facets):
            if i in sup:
                row[r] = -1.0 if i in F else 1.0
        row[R + i] = -1.0
        A.append(row)
        b.append(float(w[i]))
    res = scipy_optimize.linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                                 bounds=[(0, None)] * (R + n), method="highs")
    assert res.status != 3, "L is UNBOUNDED above -- the bound cannot be sound"
    return res.x[:R] if res.success else None


def _instance(rng):
    n = rng.randint(3, 7)
    m = rng.randint(1, 4)
    supports = [frozenset(rng.sample(range(n), rng.randint(1, min(5, n))))
                for _ in range(m)]
    err = [rng.randint(0, 1) for _ in range(n)]
    syn = [sum(err[i] for i in S) & 1 for S in supports]
    w = {i: Fraction(rng.choice([0, 1, 1, 2, 3, 5]), rng.choice([1, 2, 3, 4]))
         for i in range(n)}
    return n, supports, syn, w


@pytest.mark.parametrize("seed", [20260922, 777])
def test_the_lp_maximising_dual_never_exceeds_the_true_optimum(seed):
    """The strongest adversary available: the dual that MAXIMISES L."""
    rng = random.Random(seed)
    reached_boundary = 0
    for _ in range(250):
        n, supports, syn, w = _instance(rng)
        opt = _brute_opt(n, supports, syn, w)
        assert opt is not None                 # syndrome drawn from a real error
        facets = _valid_facets(supports, syn)
        y = _lp_maximising_duals(n, facets, w)
        if y is None:
            continue
        duals = [(nm, sorted(F), Fraction(float(v)).limit_denominator(10**6))
                 for (nm, F, _), v in zip(facets, y) if v > 1e-12]
        L, _used, err = qc._exact_facet_scan(supports, syn, w, duals)
        assert err is None, f"a valid facet was refused: {err}"
        assert L <= opt, f"UNSOUND: L={L} exceeds OPT={opt}"
        reached_boundary += (L == opt)
    # the adversary must actually REACH the boundary, or it proves nothing
    assert reached_boundary > 0, "the attack never reached L == OPT; it is too weak"


@pytest.mark.parametrize("seed", [20260922, 777])
def test_random_nonnegative_duals_never_exceed_the_optimum(seed):
    rng = random.Random(seed + 1)
    checked = 0
    for _ in range(250):
        n, supports, syn, w = _instance(rng)
        opt = _brute_opt(n, supports, syn, w)
        facets = _valid_facets(supports, syn)
        if not facets:
            continue
        pick = rng.sample(facets, rng.randint(1, min(6, len(facets))))
        duals = [(nm, sorted(F), Fraction(rng.randint(0, 8), rng.randint(1, 4)))
                 for (nm, F, _) in pick]
        L, _u, err = qc._exact_facet_scan(supports, syn, w, duals)
        if err is None:
            checked += 1
            assert L <= opt, f"UNSOUND: L={L} exceeds OPT={opt}"
    assert checked > 100


def test_every_right_parity_facet_is_refused():
    """A right-parity facet's inequality is FALSE; accepting one is unsound."""
    rng = random.Random(4242)
    probed = 0
    for _ in range(150):
        n, supports, syn, w = _instance(rng)
        for j, S in enumerate(supports):
            for size in range(len(S) + 1):
                if size % 2 != syn[j]:
                    continue
                for F in itertools.combinations(sorted(S), size):
                    _L, _u, err = qc._exact_facet_scan(
                        supports, syn, w, [(j, list(F), Fraction(1))])
                    probed += 1
                    assert err is not None, f"ACCEPTED right-parity facet ({j},{F})"
    assert probed > 500


def test_a_facet_escaping_its_check_is_refused():
    supports = [frozenset({0, 1, 2})]
    syn = [1]
    w = {i: Fraction(1) for i in range(4)}
    _L, _u, err = qc._exact_facet_scan(supports, syn, w, [(0, [0, 3], Fraction(1))])
    assert err is not None and "escapes" in err


@pytest.mark.parametrize("checks, syn, w, true_opt", [
    # variable 2 is in no check: flipping it is free and lowers the weight
    ([(0, 1)], [1], {0: 1.0, 1: 1.0, 2: -1e-10}, 1.0 - 1e-10),
    # variable 2 and 3 share a check no facet touches: flipping both is free
    ([(0, 1), (2, 3)], [1, 0], {0: 1.0, 1: 1.0, 2: -1e-10, 3: -1e-10},
     1.0 - 2e-10),
])
def test_a_tolerated_negative_weight_cannot_certify_a_false_minimum(
        checks, syn, w, true_opt):
    """THE WITNESS (2026-09-22). check_qldpc admits w_i >= -tol, and the exact
    scan once summed the safe slack only over LOADED variables. An unloaded
    variable with w_i < 0 then lowered OPT without lowering L, and this
    certificate came back OPTIMAL with exact_gap_zero -- for a correction that
    is not minimum weight. The slack now runs over every weighted variable."""
    n = len(w)
    opt = _brute_opt(n, [frozenset(c) for c in checks], syn,
                     {i: Fraction(v) for i, v in w.items()})
    assert float(opt) == true_opt and opt < 1
    cert = qc.QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1.0),))
    r = qc.check_qldpc(checks=checks, weights=w, syndrome=syn, cert=cert)
    assert not r.accepted, f"false minimum certified: {r.reason}"
    # TWO INDEPENDENT FIXES, EITHER REFUSAL IS CORRECT. The admission screen
    # refuses any negative weight since 2026-09-14 (found by mutation score);
    # the scan below is sound for ANY weights since 2026-09-22 (found by
    # stating the Lean theorem). Defence in depth: each alone refuses this.
    assert ("negative variable weight" in r.reason
            or "NOT PROVEN OPTIMAL (exact)" in r.reason), r.reason
    # and the scan itself is below the true optimum, for this and any y
    L, _u, err = qc._exact_facet_scan(
        [frozenset(c) for c in checks], syn,
        {i: Fraction(v) for i, v in w.items()}, [(0, [], Fraction(1))])
    assert err is None and L <= opt


def test_the_same_certificate_with_honest_weights_is_still_accepted():
    """Control for the witness above: remove the negative weight and the
    identical certificate is a true minimum, so the fix must not refuse it --
    otherwise the witness test passes against a checker that refuses all."""
    cert = qc.QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1.0),))
    r = qc.check_qldpc(checks=[(0, 1)], weights={0: 1.0, 1: 1.0, 2: 0.0},
                       syndrome=[1], cert=cert)
    assert r.accepted, r.reason
    assert r.checks["exact_gap_zero"] is True


@pytest.mark.parametrize("seed", [20260922, 777])
def test_the_lp_adversary_with_negative_weights_never_exceeds_the_optimum(seed):
    """L <= OPT for ANY real weights -- the statement proved in Lean. The
    adversary draws weights that can be negative, on variables that are often
    in no facet at all, which is the regime the loaded-only sum got wrong."""
    rng = random.Random(seed + 2)
    evaluated = 0
    for _ in range(200):
        n, supports, syn, w = _instance(rng)
        n_extra = rng.randint(0, 2)             # variables in no check
        for k in range(n_extra):
            w[n + k] = Fraction(0)
        for i in w:
            if rng.random() < 0.3:
                w[i] = -Fraction(rng.randint(1, 5), rng.randint(1, 7))
        opt = _brute_opt(n + n_extra, supports, syn, w)
        facets = _valid_facets(supports, syn)
        y = _lp_maximising_duals(n + n_extra, facets, w)
        if y is None:
            continue
        duals = [(nm, sorted(F), Fraction(float(v)).limit_denominator(10**6))
                 for (nm, F, _), v in zip(facets, y) if v > 1e-12]
        L, _used, err = qc._exact_facet_scan(supports, syn, w, duals)
        assert err is None, f"a valid facet was refused: {err}"
        assert L <= opt, f"UNSOUND: L={L} exceeds OPT={opt}"
        evaluated += 1
    assert evaluated > 150


def test_the_bound_is_tight_somewhere_so_the_test_can_fail():
    """Negative-control shape: on a case with known OPT, the right dual MEETS it.

    Single check {0,1} with syndrome 1: the minimum correction is one variable,
    weight 1. The wrong-parity facets are the even subsets, including {} and
    {0,1}. A test that could only ever observe L far below OPT would pass an
    over-conservative checker and a sound one alike.
    """
    supports = [frozenset({0, 1})]
    syn = [1]
    w = {0: Fraction(1), 1: Fraction(1)}
    assert _brute_opt(2, supports, syn, w) == 1
    L, _u, err = qc._exact_facet_scan(supports, syn, w, [(0, [], Fraction(1))])
    assert err is None
    assert L == 1
