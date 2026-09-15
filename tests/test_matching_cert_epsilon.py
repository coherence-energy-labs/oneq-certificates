"""The epsilon checker enforces paper 1's claim and nothing weaker.

Soundness is tested against brute force: on random small graphs, for random
candidate corrections and random (often overloaded) packings over T-odd sets,
the reported epsilon must never be smaller than the candidate's true excess
over the exhaustively computed minimum T-join. Plus the external audit's named
cases: the tolerance-boundary witness, non-finite values, zero weights.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import random
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq import matching_cert_epsilon as mce  # noqa: E402
from oneq.matching_cert import MatchingCertificate  # noqa: E402

B = mce.BOUNDARY


def cert(edges, duals, potentials=None):
    return MatchingCertificate(matched_edges=tuple(edges), node_potentials=potentials or {},
                               blossom_duals={frozenset(S): z for S, z in duals.items()})


TRI = [(0, 1, 1.0), (0, B, 0.5), (1, B, 0.5)]


def test_exactly_tight_is_exact_optimal():
    v = mce.check_epsilon(edges=TRI, syndrome=[0, 1], cert=cert([(0, 1)], {(0,): 0.5, (1,): 0.5}),
                          eps_max=0)
    assert v.status == mce.EXACT_OPTIMAL and v.epsilon == 0 and v.accepted


def test_the_auditors_tolerance_witness_is_epsilon_not_exact():
    g = [(0, 1, 1.0000005), (0, B, 0.5), (1, B, 0.5)]
    c = cert([(0, 1)], {(0,): 0.5, (1,): 0.5})
    at_1e6 = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=c, eps_max=Fraction(1, 10**6))
    assert at_1e6.status == mce.EPSILON_OPTIMAL
    assert at_1e6.epsilon == Fraction(1.0000005) - 1
    at_1e7 = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=c, eps_max=Fraction(1, 10**7))
    assert at_1e7.status == mce.NOT_PROVEN and not at_1e7.accepted
    exact = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=c, eps_max=0)
    assert exact.status == mce.NOT_PROVEN


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_duals_and_weights_are_invalid_input(bad):
    v = mce.check_epsilon(edges=TRI, syndrome=[0, 1], cert=cert([(0, 1)], {(0,): bad, (1,): 0.5}),
                          eps_max=Fraction(1, 10**6))
    assert v.status == mce.INVALID_INPUT and not v.accepted
    g = [(0, 1, bad), (0, B, 0.5), (1, B, 0.5)]
    v = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=cert([(0, B), (1, B)], {(0,): 0.5}),
                          eps_max=Fraction(1, 10**6))
    assert v.status == mce.INVALID_INPUT


def test_non_finite_eps_max_is_refused_outright():
    with pytest.raises(ValueError):
        mce.check_epsilon(edges=TRI, syndrome=[0, 1], cert=cert([(0, 1)], {}), eps_max=math.nan)


@pytest.mark.parametrize("duals,potentials,why", [
    ({(0, 1): 0.5}, None, "not odd"),
    ({(0,): -0.1}, None, "negative dual"),
    ({(0,): 0.5}, {0: 0.25}, "node potentials"),
])
def test_inadmissible_packings_are_invalid_input(duals, potentials, why):
    v = mce.check_epsilon(edges=TRI, syndrome=[0, 1], cert=cert([(0, 1)], duals, potentials),
                          eps_max=Fraction(1))
    assert v.status == mce.INVALID_INPUT and why in v.reason


def test_a_correction_that_does_not_explain_the_syndrome_is_invalid_input():
    v = mce.check_epsilon(edges=TRI, syndrome=[0, 1], cert=cert([(0, B)], {(0,): 0.5}),
                          eps_max=Fraction(10))
    assert v.status == mce.INVALID_INPUT


def test_negative_weights_are_invalid_input():
    g = [(0, 1, -1.0), (0, B, 0.5), (1, B, 0.5)]
    v = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=cert([(0, 1)], {}), eps_max=Fraction(10))
    assert v.status == mce.INVALID_INPUT


def test_a_loaded_zero_weight_edge_gives_a_weak_bound_not_a_false_one():
    g = [(0, 1, 1.0), (0, B, 0.0), (1, B, 0.5)]
    v = mce.check_epsilon(edges=g, syndrome=[0, 1], cert=cert([(0, 1)], {(0,): 0.5, (1,): 0.5}),
                          eps_max=Fraction(1, 10**6))
    assert v.lower_bound == 0 and v.status == mce.NOT_PROVEN


def _d3_shots(n=60):
    import numpy as np
    import pymatching
    import stim
    p = 0.005
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), B if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0)))
             for e in m.edges()]
    det, _ = circ.compile_detector_sampler(seed=4).sample(n, separate_observables=True)
    shots = []
    for row in det:
        fired = [int(i) for i in np.flatnonzero(row)]
        if fired:
            shots.append((fired, [(int(a), B if b < 0 else int(b)) for a, b in m.decode_to_edges_array(row)]))
    return edges, shots


def test_producer_epsilon_mode_acceptances_reverify_independently():
    from oneq.certifying_decoder import CertifyingDecoder
    edges, shots = _d3_shots()
    emax = Fraction(1, 10**6)
    cd = CertifyingDecoder(edges, epsilon_max=emax)
    accepted = 0
    for fired, corr in shots:
        corr = tuple(cd.key(e) for e in corr)
        r = cd.certify(fired, corr)
        if r.accepted:
            accepted += 1
            assert r.stats["acceptance"] == "epsilon"
            v = mce.check_epsilon(edges=edges, syndrome=fired, cert=r.cert, eps_max=emax)
            assert v.accepted and v.epsilon <= emax and str(v.epsilon) == r.stats["epsilon"]
    assert accepted > 0


def test_producer_default_mode_is_still_the_exact_rule():
    from oneq.certifying_decoder import CertifyingDecoder
    edges, shots = _d3_shots(30)
    cd = CertifyingDecoder(edges)
    assert cd.epsilon_max is None
    for fired, corr in shots:
        r = cd.certify(fired, tuple(cd.key(e) for e in corr))
        assert "acceptance" not in r.stats


def _min_tjoin(edges, T):
    W = {}
    for u, v, w in edges:
        k = (min(u, v), max(u, v))
        W[k] = min(W.get(k, Fraction(w)), Fraction(w))
    keys = sorted(W)
    best = None
    for r in range(len(keys) + 1):
        for sub in itertools.combinations(keys, r):
            deg = {}
            for a, b in sub:
                for x in (a, b):
                    if x != B:
                        deg[x] = deg.get(x, 0) + 1
            if {x for x, n in deg.items() if n % 2} == set(T):
                wt = sum((W[k] for k in sub), Fraction(0))
                if best is None or wt < best[0]:
                    best = (wt, sub)
    return best


def test_soundness_against_brute_force_on_random_graphs():
    rng = random.Random(20260914)
    checked = accepted = 0
    for _ in range(150):
        nv = rng.randint(3, 5)
        pairs = [(a, b) for a in range(nv) for b in range(a + 1, nv)] + [(a, B) for a in range(nv)]
        es = rng.sample(pairs, rng.randint(nv, min(len(pairs), 9)))
        edges = [(a, b, rng.choice([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 0.1])) for a, b in es]
        T = sorted(rng.sample(range(nv), rng.choice([1, 2, 2, 3, 4][: nv])))
        opt = _min_tjoin(edges, T)
        if opt is None:
            continue
        # candidates: the optimum and a random feasible T-join
        keys = sorted({(min(a, b), max(a, b)) for a, b, _ in edges})
        cands = [opt[1]]
        for _ in range(30):
            sub = [k for k in keys if rng.random() < 0.4]
            deg = {}
            for a, b in sub:
                for x in (a, b):
                    if x != B:
                        deg[x] = deg.get(x, 0) + 1
            if {x for x, n in deg.items() if n % 2} == set(T):
                cands.append(tuple(sub))
                break
        Tp = set(T) | ({B} if len(T) % 2 else set())
        verts = list(range(nv)) + [B]
        for c in cands:
            duals = {}
            for _ in range(rng.randint(0, 4)):
                S = frozenset(rng.sample(verts, rng.randint(1, len(verts) - 1)))
                if len(S & Tp) % 2 == 1:
                    duals[S] = rng.choice([0.1, 0.25, 0.5, 1.0, 3.0])
            v = mce.check_epsilon(edges=edges, syndrome=T, cert=cert(c, duals), eps_max=Fraction(1, 2))
            if v.status == mce.INVALID_INPUT:
                continue
            assert v.status != mce.VERIFIER_FAILURE
            true_excess = sum((min(Fraction(w) for a2, b2, w in edges
                                   if (min(a2, b2), max(a2, b2)) == k) for k in c), Fraction(0)) - opt[0]
            assert v.epsilon >= true_excess            # the SOUNDNESS property
            checked += 1
            accepted += v.accepted
    assert checked > 150 and accepted > 0              # not vacuous


# ---------------------------------------------------------------------------
# Killers for the six non-equivalent survivors of tools/mutation_score.py on
# this module (2026-09-14): each input below is one the checker's contract
# refuses or accepts, and each is decided the other way by one mutant.

_E1 = [(0, 1, 1.0), (1, -1, 1.0), (0, -1, 1.0)]      # T = {0, 1}: J = {(0, 1)}, weight 1


def _cert1(duals=None, pots=None, corr=((0, 1),)):
    return MatchingCertificate(matched_edges=corr, node_potentials=pots or {},
                               blossom_duals=duals if duals is not None
                               else {frozenset([0]): 0.5, frozenset([1]): 0.5})


def test_control_the_small_certificate_is_exactly_optimal():
    v = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(), eps_max=0)
    assert v.status == mce.EXACT_OPTIMAL, v


@pytest.mark.parametrize("bad_edge", [(True, 2, 1.0), (2.0, 3, 1.0), (0, "x", 1.0)], ids=repr)
def test_a_graph_vertex_that_is_not_a_plain_integer_is_refused(bad_edge):
    v = mce.check_epsilon(edges=_E1 + [bad_edge], syndrome=[0, 1], cert=_cert1(), eps_max=0)
    assert v.status == mce.INVALID_INPUT, v


@pytest.mark.parametrize("syndrome", [[0, True], [0, 1.0]], ids=repr)
def test_a_syndrome_entry_that_is_not_a_plain_integer_is_refused(syndrome):
    v = mce.check_epsilon(edges=_E1, syndrome=syndrome, cert=_cert1(), eps_max=0)
    assert v.status == mce.INVALID_INPUT, v


def test_a_dual_set_member_that_is_not_a_plain_integer_is_refused():
    duals = {frozenset([0]): 0.5, frozenset([True]): 0.5}
    v = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(duals), eps_max=0)
    assert v.status == mce.INVALID_INPUT, v


def test_a_dual_value_of_exactly_one_counts_toward_the_bound():
    edges = [(0, 1, 2.0), (1, -1, 2.0), (0, -1, 2.0)]
    duals = {frozenset([0]): 1.0, frozenset([1]): 1.0}
    v = mce.check_epsilon(edges=edges, syndrome=[0, 1], cert=_cert1(duals), eps_max=0)
    assert v.status == mce.EXACT_OPTIMAL and v.lower_bound == 2, v


def test_zero_node_potentials_are_allowed_and_nonzero_ones_refused():
    ok = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(pots={0: 0.0, 1: 0}), eps_max=0)
    assert ok.status == mce.EXACT_OPTIMAL, ok
    bad = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(pots={0: 0.5}), eps_max=0)
    assert bad.status == mce.INVALID_INPUT, bad


def test_slack_exactly_equal_to_eps_max_is_accepted():
    duals = {frozenset([0]): 0.4, frozenset([1]): 0.5}
    loose = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(duals), eps_max=1)
    assert loose.status == mce.EPSILON_OPTIMAL and loose.epsilon > 0, loose
    at = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(duals), eps_max=loose.epsilon)
    assert at.status == mce.EPSILON_OPTIMAL and at.epsilon == at.eps_max, at
    below = mce.check_epsilon(edges=_E1, syndrome=[0, 1], cert=_cert1(duals),
                              eps_max=loose.epsilon - Fraction(1, 10**30))
    assert below.status == mce.NOT_PROVEN, below
