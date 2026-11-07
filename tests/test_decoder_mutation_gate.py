"""decoder_mutation_gate.mutants (paper 1 Sec. VII A): every seeded fault is an EDGE SET that differs from the decoder's.

Found 2026-11-07, revising paper 1 against external audit 4: the stealth mutant M7 appended one unused edge twice.
As an edge set that is the decoder's own correction; weighed as a list it looked heavier, so 601 cases were tallied
"defective" and "proven suboptimal" and the released corpus labelled them defective. These tests hold the rule that
replaced it: a mutant is returned as the edge set it denotes, one whose edge set equals the decoder's is never
returned, the new M7 adds a genuine even cycle disjoint from the correction, and the detection harness refuses a
correction that lists an edge twice instead of weighing it.
"""

from __future__ import annotations

import pathlib
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

import decoder_mutation_gate as dmg  # noqa: E402
from oneq.matching_cert import BOUNDARY  # noqa: E402

B = BOUNDARY


def test_edge_set_cancels_pairs_and_keeps_first_occurrence_order():
    assert dmg.edge_set([(1, 2), (3, 4), (1, 2)]) == [(3, 4)]
    assert dmg.edge_set([(5, 6), (1, 2), (5, 6), (5, 6)]) == [(5, 6), (1, 2)]
    assert dmg.edge_set([]) == []


def _ring():
    """A square 0-1-2-3-0 with a pendant edge to the boundary at every vertex, and a second square 4-5-6-7-4 that no
    correction below touches: the disjoint even cycle M7 must find."""
    W = {}
    for a, b in ((0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6), (6, 7), (4, 7), (3, 4)):
        W[(a, b)] = 1.0
    for v in range(8):
        W[(v, B)] = 2.5
    return W


def test_the_old_doubled_edge_construction_is_the_decoders_own_edge_set_and_is_never_returned():
    W = _ring()
    matched = ((0, 1), (2, 3))                    # fired {0, 1, 2, 3}
    alt = next(k for k in W if k not in matched and B not in k)
    assert sorted(dmg.edge_set(list(matched) + [alt, alt])) == sorted(matched)    # what M7 used to be
    own = frozenset(matched)
    out = dmg.mutants(W, matched, [0, 1, 2, 3], prev=None)
    assert out and all(frozenset(m) != own for _n, m in out)
    assert all(len(set(m)) == len(m) for _n, m in out), "a mutant lists an edge twice"


def test_m7_adds_a_disjoint_even_cycle_heavier_and_syndrome_consistent():
    W = _ring()
    matched = ((0, 1), (2, 3))
    m7 = [m for n, m in dmg.mutants(W, matched, [0, 1, 2, 3], prev=None) if n.startswith("M7")]
    assert len(m7) == 1
    m = m7[0]
    assert set(matched) < set(m) and len(set(m)) == len(m)
    assert dmg.odd_set(m) == {0, 1, 2, 3}
    assert sum(W[k] for k in m) > sum(W[k] for k in matched)
    assert not set(m) - set(matched) & set(matched)


def test_a_control_that_is_the_decoders_own_correction_is_not_returned():
    """M5 sends every fired detector to the boundary when that ties the correction's weight; when the decoder's
    correction already IS that, the "alternative" is the same edge set and must not be counted as one."""
    W = _ring()
    matched = ((0, B), (5, B))
    names = [n for n, _m in dmg.mutants(W, matched, [0, 5], prev=None)]
    assert not any(n.startswith("M5") for n in names)
    matched2 = ((0, 1),)                          # fired {0, 1}: all-boundary weighs 5.0, not a tie
    assert not any(n.startswith("M5") for n, _m in dmg.mutants(W, matched2, [0, 1], prev=None))


def test_no_mutant_on_real_shots_equals_the_decoders_edge_set():
    stim = pytest.importorskip("stim")
    pymatching = pytest.importorskip("pymatching")
    from oneq.certifying_decoder import CertifyingDecoder
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                  after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
                                  after_reset_flip_probability=0.005, before_round_data_depolarization=0.005)
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), B if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0))) for e in m.edges()]
    cd = CertifyingDecoder(edges)
    det, _ = circ.compile_detector_sampler(seed=7).sample(120, separate_observables=True)
    prev, seen, m7 = None, 0, 0
    for s in range(det.shape[0]):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        matched = tuple(cd.key((u, v if v >= 0 else B)) for u, v in m.decode_to_edges_array(det[s]))
        for name, mut in dmg.mutants(cd.W, matched, fired, prev):
            seen += 1
            assert frozenset(mut) != frozenset(matched), (s, name)
            assert len(set(mut)) == len(mut), (s, name)
            if name.startswith("M7"):
                m7 += 1
                assert dmg.odd_set(mut) == set(fired) and sum(cd.W[k] for k in mut) > cd.weight(matched)
        prev = matched
    assert seen > 0 and m7 > 0


def test_the_detection_harness_refuses_a_correction_that_lists_an_edge_twice():
    import detection_outcomes as do
    J = do.Judge([(0, 1, 1.0), (1, B, 1.0), (0, B, 1.5)])
    with pytest.raises(SystemExit, match="lists an edge twice"):
        J.outcome([0, 1], [(0, 1), (0, 1), (0, 1)], Fraction(1))
