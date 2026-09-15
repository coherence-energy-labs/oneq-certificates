"""The exact dual repair must never launder a suboptimal correction.

WHY THIS FILE EXISTS. The checker no longer accepts a shipped dual as-is: it
REPAIRS it (sheds infeasibility, absorbs the shortfall) and then verifies the
result in exact arithmetic. That is strictly stronger than what it replaced --
but it is also the single most dangerous edit in this repository, because a
repair step that is too generous turns the checker into a rubber stamp while
every existing test stays green.

So these tests attack the repair itself. The load-bearing claim is:

    by LP duality a feasible dual attaining U exists IFF U is optimal,

so on a SUBOPTIMAL primal the absorb step must provably fail to find the
slack. If any test here goes green by accepting, the whole programme is
broken and that is worth more than a passing suite.

The witness in `test_the_auditors_suboptimal_witness_is_still_refused` is the
external auditor's own (2026-08-04, finding 4), kept verbatim so a future
refactor cannot quietly re-open the hole they found.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.matching_cert import MatchingCertificate, check  # noqa: E402

# The auditor's graph: the direct edge is heavier than the boundary route, so
# the true optimum is 1.0 and any certificate claiming 1.0000005 is wrong.
AUDIT_EDGES = [(0, 1, 1.0000005), (0, -1, 0.5), (1, -1, 0.5)]


def _cert(**kw):
    kw.setdefault("metric_closure_certified", True)
    return MatchingCertificate(**kw)


def test_the_auditors_suboptimal_witness_is_still_refused():
    """gap 5e-7 is REAL suboptimality, not rounding. It must not be absorbed."""
    r = check(
        edges=AUDIT_EDGES,
        syndrome=[0, 1],
        cert=_cert(matched_edges=((0, 1),), node_potentials={},
                   blossom_duals={frozenset({0}): 0.5,
                                  frozenset({1}): 0.5}),
    )
    assert not r.accepted, (
        "THE REPAIR LAUNDERED A PROVABLY SUBOPTIMAL CORRECTION. The direct "
        f"edge weighs 1.0000005, the boundary route 1.0. Reason given: {r.reason}")
    assert "shortfall is REAL" in r.reason or "not proven" in r.reason.lower()


def test_a_dual_inflated_past_the_primal_is_refused():
    """Shedding must not be able to 'fix' a dual that was never feasible."""
    r = check(
        edges=AUDIT_EDGES,
        syndrome=[0, 1],
        cert=_cert(matched_edges=((0, 1),), node_potentials={},
                   blossom_duals={frozenset({0}): 0.9,
                                  frozenset({1}): 0.9}),
    )
    assert not r.accepted, f"accepted an inflated dual: {r.reason}"


def test_absorbing_cannot_exceed_available_slack():
    """A shortfall larger than every variable's slack must stay a refusal.

    This is the exact knob that would turn the repair into a tolerance: if
    the slack test were >= 0 instead of >= shortfall, this would accept.
    """
    r = check(
        edges=[(0, 1, 10.0), (0, -1, 0.5), (1, -1, 0.5)],
        syndrome=[0, 1],
        cert=_cert(matched_edges=((0, 1),), node_potentials={},
                   blossom_duals={frozenset({0}): 0.5,
                                  frozenset({1}): 0.5}),
    )
    assert not r.accepted, f"absorbed a gap of ~9.0: {r.reason}"


def test_a_genuine_certificate_still_certifies_after_the_repair():
    """Fail-closed is easy and worthless. The repair must still ACCEPT truth.

    Equal-weight edges, correction takes the boundary on both sides, and the
    dual attains exactly -- there is nothing to repair and nothing to refuse.
    """
    r = check(
        edges=[(0, 1, 2.0), (0, -1, 0.5), (1, -1, 0.5)],
        syndrome=[0, 1],
        cert=_cert(matched_edges=((0, -1), (1, -1)), node_potentials={},
                   blossom_duals={frozenset({0}): 0.5,
                                  frozenset({1}): 0.5}),
    )
    assert r.accepted, f"refused a genuinely optimal certificate: {r.reason}"


def test_every_recorded_conformance_verdict_still_holds():
    """All 56 vectors, including the 47 refusals and the forgery families."""
    from oneq.cert_format import graph_from_json, recheck
    vectors = json.loads(
        (ROOT / "conformance" / "cert_vectors.json").read_text(encoding="utf-8"))
    wrong = []
    for x in vectors["vectors"]:
        got, why = recheck(x["document"], graph_from_json(x["graph"]))
        if got != x["expected_accepted"]:
            wrong.append((x["name"], x["expected_accepted"], got, why[:60]))
    assert not wrong, f"verdict changed on {len(wrong)}: {wrong[:4]}"


def test_no_forgery_family_is_accepted():
    """The direction that matters: a forgery accepted is a refutation."""
    from oneq.cert_format import graph_from_json, recheck
    vectors = json.loads(
        (ROOT / "conformance" / "cert_vectors.json").read_text(encoding="utf-8"))
    accepted_forgeries = []
    for x in vectors["vectors"]:
        if x["name"].endswith("-genuine"):
            continue
        got, why = recheck(x["document"], graph_from_json(x["graph"]))
        if got:
            accepted_forgeries.append((x["name"], why[:60]))
    assert not accepted_forgeries, (
        f"ACCEPTED FORGERIES: {accepted_forgeries}")


def test_the_repair_terminates_on_a_heavily_overloaded_dual():
    """The shed loop is bounded by the edge count as a PROOF, not a guess.

    Reducing a dual lowers load on every edge of its cut and raises it
    nowhere, so a cleared edge never regresses. Pathological input must still
    return rather than spin.
    """
    n = 12
    edges = [(i, i + 1, 1.0) for i in range(n)] + [(0, -1, 1.0), (n, -1, 1.0)]
    r = check(
        edges=edges,
        syndrome=[0, n],
        cert=_cert(matched_edges=tuple((i, i + 1) for i in range(n)),
                   node_potentials={},
                   blossom_duals={frozenset({i}): 5.0
                                  for i in range(n + 1)}),
    )
    assert isinstance(r.accepted, bool)  # the point is that it RETURNED


# ---------------------------------------------------------------------------
# THE SAME HOLE, THE OTHER LANE.
#
# Everything above guards `matching_cert`. On 2026-08-10 the identical defect
# was found still open in `qldpc_check.check_qldpc`: it accepted on
# `gap <= 1e-6` with no exact gate whatsoever, and its accept branch reported
# a HARDCODED `gap: 0.0`, so the receipt concealed the very gap that should
# have refused it. The witness below is the analogue of the auditor's, ported
# to the qLDPC dual, and it was ACCEPTED with the reason "weak duality makes
# both optimal" before the exact gate was added. ~25 gate scripts, `api.py`
# and `coset_cert.py` all decide through this path.
#
# The lesson worth keeping: the repair was applied to one lane and the sibling
# lane was never checked. These tests exist so "fixed" means fixed everywhere.

# checks {0,1} and {0,2}, syndrome (1,1). The claimed error {0} weighs
# 1.0000005; {1,2} is also a valid correction and weighs 1.0.
QL_CHECKS = [(0, 1), (0, 2)]
QL_SYN = [1, 1]


def _ql(weights, support, duals):
    from oneq.qldpc_check import QldpcCertificate, check_qldpc
    return check_qldpc(
        checks=QL_CHECKS, weights=weights, syndrome=QL_SYN,
        cert=QldpcCertificate(error_support=support, facet_duals=duals,
                              box_duals=()))


def test_the_qldpc_lane_refuses_the_same_suboptimal_witness():
    """The dual here is GENUINELY feasible and its bound is a true 1.0 -- the
    defect was never a bad dual, it was accepting a primal that misses that
    bound by 5e-7, which is real suboptimality and not rounding."""
    r = _ql({0: 1.0000005, 1: 0.5, 2: 0.5}, (0,), ((0, (), 0.5), (1, (), 0.5)))
    assert not r.accepted, (
        "THE QLDPC LANE CERTIFIED A PROVABLY SUBOPTIMAL CORRECTION. Claimed "
        f"error {{0}} weighs 1.0000005; {{1,2}} weighs 1.0. Reason: {r.reason}")


def test_the_qldpc_lane_still_certifies_a_genuinely_tight_dual():
    """Fail-closed is easy and worthless -- calibrate the other way too."""
    r = _ql({0: 1.0, 1: 0.5, 2: 0.5}, (1, 2), ((0, (), 0.5), (1, (), 0.5)))
    assert r.accepted, f"refused a genuinely optimal certificate: {r.reason}"
    assert r.checks.get("exact_gap_zero") is True, (
        "accepted without recording that the EXACT gate ran; an artifact that "
        "cannot show which gate decided it is not evidence")


def test_a_qldpc_dual_inflated_past_feasibility_is_refused():
    """Shedding the infeasibility must not be able to 'repair' a dual that was
    never feasible into one that happens to land on the primal."""
    r = _ql({0: 1.0000005, 1: 0.5, 2: 0.5}, (0,), ((0, (), 0.9), (1, (), 0.9)))
    assert not r.accepted, f"accepted an inflated dual: {r.reason}"


def test_the_qldpc_accept_path_reports_a_gap_it_has_actually_proved():
    """The accept branch hardcodes `gap=0.0`. That constant was a lie until an
    exact gate stood in front of it; this pins that it still does."""
    src = (ROOT / "src" / "oneq" / "qldpc_check.py").read_text(encoding="utf-8")
    assert "def _exact_optimality" in src, "the exact gate is gone"
    i_gate = src.index(" = _exact_optimality(", src.index("def check_qldpc("))
    i_acc = src.index("OPTIMAL: feasible Feldman dual attains")
    assert i_gate < i_acc, (
        "the exact gate no longer runs before acceptance inside check_qldpc")
    assert "if not proven:" in src[i_gate:i_acc], (
        "the exact gate's verdict is computed but never REFUSES on it")


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_qldpc_suboptimality_is_refused_at_every_scale(scale):
    """Any epsilon-based rule fails at SOME scale; 5e-7 slips under 1e-6 at
    scale 1 and a scaled tolerance just moves where it slips."""
    r = _ql({0: 1.0000005 * scale, 1: 0.5 * scale, 2: 0.5 * scale}, (0,),
            ((0, (), 0.5 * scale), (1, (), 0.5 * scale)))
    assert not r.accepted, f"suboptimal accepted at scale {scale}: {r.reason}"


def test_a_receipt_may_not_state_a_weight_it_did_not_prove():
    """`f2h` stores a double as its exact 64-bit pattern so the number
    survives the round trip unchanged. `recheck` then compared it back with
    `abs(...) > 1e-9`, discarding that exactness and letting a receipt
    advertise a weight nothing was proven about. One ulp must be refused."""
    import math

    from oneq.cert_format import f2h, graph_from_json, h2f, recheck
    vectors = json.loads(
        (ROOT / "conformance" / "cert_vectors.json").read_text(encoding="utf-8"))
    x = next(v for v in vectors["vectors"]
             if v["name"].endswith("-genuine") and v["expected_accepted"])
    edges = graph_from_json(x["graph"])
    doc = x["document"]
    ok, why = recheck(doc, edges)
    assert ok, f"the honest published document was refused: {why}"

    drifted = dict(doc)
    moved = math.nextafter(h2f(doc["weight"]), math.inf)
    assert moved != h2f(doc["weight"]), "the drift did not actually move"
    drifted["weight"] = f2h(moved)
    ok2, why2 = recheck(drifted, edges)
    assert not ok2, (
        "a receipt stating a weight ONE ULP from the proven one was accepted; "
        "the bit-exact encoding is being compared with slack")
    assert "states weight" in why2, why2


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_suboptimality_is_refused_at_every_scale(scale):
    """Any epsilon-based rule fails at some scale. An exact one must not."""
    r = check(
        edges=[(0, 1, 1.0000005 * scale), (0, -1, 0.5 * scale),
               (1, -1, 0.5 * scale)],
        syndrome=[0, 1],
        cert=_cert(matched_edges=((0, 1),), node_potentials={},
                   blossom_duals={frozenset({0}): 0.5 * scale,
                                  frozenset({1}): 0.5 * scale}),
    )
    assert not r.accepted, f"suboptimal accepted at scale {scale}: {r.reason}"
