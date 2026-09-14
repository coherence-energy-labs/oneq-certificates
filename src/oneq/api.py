r"""The product surface: certify a decoder's answers in four lines.

    from oneq.api import MatchingCertifier

    cert = MatchingCertifier(edges)          # (u, v, weight); v may be BOUNDARY
    r = cert.certify(syndrome, correction)   # -> Receipt
    print(r.tier, r.bound, r.gap)            # "certified-optimal", ...
    doc = r.document(edges)                  # canonical JSON, stranger-checkable

THE TRUST MODEL, in one paragraph. Your decoder is never trusted: the
receipt carries a dual witness that an independent checker re-derives from
the graph, the syndrome, and arithmetic -- no solver, no shared code. When
the witness meets your answer's weight, the answer is PROVEN minimum-weight
(weak duality); when it cannot, you get a named lesser tier with a measured
gap instead of silence. A stranger verifies the document with
`oneq.api.verify(document, edges)` and needs nothing else from you.

Tiers, strongest first:
  "certified-optimal"   proven minimum-weight; checkable by anyone
  "degraded"            syndrome-consistent, weight <= W, gap <= g -- an
                        honest bound, not a proof

A DIFFERENT QUESTION, ON THE SAME SURFACE. Everything above certifies
MINIMUM WEIGHT: that a correction is the cheapest single explanation. The
decoding contract is the LOGICAL CLASS, and the most likely class is the
one whose whole coset carries the most mass -- which can differ, and this
repository has measured the difference. `enclosure_document(...)` produces
a per-shot certificate of the maximum-likelihood class, and
`verify_enclosure(document)` is its stranger's entry point:

    from oneq.api import enclosure_document, verify_enclosure

    doc = enclosure_document(mechanisms, syndrome, n_obs, beam=32)
    r = verify_enclosure(doc)                # replays; owns the verdict
    print(r.verdict, r.winner, r.posterior(r.winner))

The document carries the model and a REPLAY SCHEDULE, never a bound. The
checker recomputes both ends in exact integer arithmetic and refuses when
the bracket straddles. `verifier-js/verify-enclosure.mjs` reads the same
file. See `docs/COSET_ENCLOSURE_THEOREM.md`.

And those posteriors compose into something a logical error rate is not:

    shots, refused = from_check_results(results)
    sel = certified_select(shots, threshold, refused=refused, dem_level=3)
    sel.expected_misdecodes      # a BOUND, not an estimate
    sel.log_at_least(k)          # and the per-batch tail

`certified_select` REQUIRES the admitted DEM level and refuses below A3,
because every bound it reports is conditional on that model -- which is
also why its suite carries an arm that watches the bound BREAK when the
declared model is wrong.

`MatchingCertifier.certify` runs the fast ladder (milliseconds);
`certify_hard` adds the escalation and exact tiers (seconds to minutes) and
closes every shot whose answer is actually optimal. `QldpcCertifier` is the
same contract for sparse-check codes (NP-hard regime: certification rate is
empirical and honest, and the repair tier may return a STRICTLY BETTER
correction than the one you gave it -- take it).

Streaming: `MatchingCertifier.stream(shots)` yields ordered receipts with a
watermark; see oneq.streaming for the latency contract.

WHAT A RECEIPT'S SOUNDNESS RESTS ON, and how to check that too.
`soundness_block(receipt, ...)` grades a receipt's evidence and returns the
block that goes inside a signed passport. Its DEM floor caps a receipt at
REPLAY when the error model it was decoded against is graded below A3 --
Frechet says redundancy earns nothing, so the model's admission is upstream
of the decoder's assurance AS A THEOREM. That grade is not something you
take on trust either: `admitted_level(rungs)` re-derives it from the rungs,
`Rung` carries a digest or a test name for each one, and
`dem_admission_certificate(...)` is the record with its rungs spelled out
so you can recompute the level yourself.

*** WHAT IS DELIBERATELY NOT ON THIS SURFACE, AND WHY. *** Silence about an
omission is the one option that misleads, so:

  oneq.dem_cert       BUILDS a detector error model from a circuit. It is a
                      PRODUCER, and this module is the surface for checking
                      what a producer claims. Trusting our DEM builder is
                      exactly what `admitted_level` exists to avoid.
  oneq.abft_l0        a Python MODEL of a hardware fault-detection scheme.
                      L2' has synthesizable Verilog and a bit-exact
                      cross-language gate; L0 has neither, so putting it
                      here would lend it the standing of the things that do.
  oneq.capability     device negotiation -- what a backend can be asked for.
                      Nothing a stranger checking a document needs.

Import them directly if you are producing rather than verifying. They are
tested, gated and covered; they are just not part of the promise that you
"need nothing else from us".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from .contract_query import certify_q_max, refute_protection
from .dem_admission import NotAdmitted, Rung, admitted_level
from .dem_admission import certificate as dem_admission_certificate
from .matching_cert import BOUNDARY
from .matching_cert_strict import ExactCutCertificate, check_exact_cut
from .certifying_decoder import Certified, CertifyingDecoder
from .coset_enclosure import build_document as enclosure_document
from .coset_enclosure_check import check_document as verify_enclosure
# AUTHORIZE is a stage of the loop, so the shield is public
# surface, not an internal. `shield_admissible` is the filter a
# caller applies BEFORE its policy sees an action; the verdict
# and its independent re-derivation come with it, because a
# shield you cannot second-guess is a shield you must trust.
from .shield import Verdict as ShieldVerdict
from .shield import admissible as shield_admissible
from .shield import evaluate as shield_evaluate
from .shield_check import verify as verify_shield_verdict
# INFER is a stage of the loop too, and MODEL_STATUS = UNKNOWN is a
# first-class OUTPUT rather than an error, so the open-world floor
# and its ignorance certificate are public surface.
from .open_world import IgnoranceCertificate
from .open_world import OpenWorldPosterior
# The other half of INFER: open_world says the library is
# INCOMPLETE, identifiability says two live members are
# UNRESOLVABLE. Both feed the shield's ID leg.
from .identifiability import certify as certify_identifiability
from .identifiability import admissible_intersection
# MONITOR's twin at the reporting layer: a win nobody finishes is
# not a win, and P(complete) is a first-class output.
from .freeze_detector import judge as judge_policy_outcome
from .freeze_detector import best_win
# CHOOSE, the loop's fourth stage: value per unit cost in the
# control law's own units, with the honesty clause that a
# relabelling cannot create physical information.
from .dwell import dwell, gamma_binding, separation_orders
from .way import K_MAX as WAY_K_MAX
from .way import group as way_group
from .way import physical_window as way_physical_window
from .exchange_rate import Workload as ExchangeWorkload
from .exchange_rate import exchange_ratio, exchange_ratio_exact
from .rl_wrap import additivity as rl_wrap_additivity
from .rl_wrap import Device as RLDevice
from .policy_gap import CPOMDP
from .policy_gap import authority as policy_authority
from .policy_gap import delta as policy_gap_delta
from .checkability import Check as CheckabilityCheck
from .checkability import Suppression
from .checkability import advantage as checkability_advantage
from .checkability import operating_point
from .composition import Evidence as CompositionEvidence
from .composition import certified_p_vcc_lb
from .composition import posterior_mean_bound
from .shift_detector import SequentialGLR
from .shift_detector import calibrate_threshold as calibrate_shift_threshold
from .voi import select as select_by_value
from .voi import score as score_candidate
from .certified_selection import CertifiedShot, from_check_results
from .certified_selection import select as certified_select
from .soundness_budget import block_for_receipt

#: *** THE ENCLOSURE NAMES WERE IMPORTED AND NEVER EXPORTED. *** They
#: arrived under public aliases -- `enclosure_document`,
#: `verify_enclosure`, `certified_select` -- for exactly the reason this
#: module exists, the surface a stranger is pointed at. None of them
#: reached `__all__`, so `from oneq.api import *` did not give them and
#: every check that reads `__all__` reported a surface they were not on.
#: Found by `ruff` flagging five unused imports: the linter could see
#: that nothing consumed them and the intent could not.
__all__ = ["BOUNDARY", "CertifiedShot", "ExactCutCertificate",
           "MatchingCertifier", "NotAdmitted", "QldpcCertifier", "Receipt",
           "Rung", "admitted_level", "certified_select", "certify_q_max",
           "dem_admission_certificate", "enclosure_document",
           "from_check_results", "refute_protection", "soundness_block",
           "verify", "verify_enclosure", "verify_shield_verdict",
           "verify_strict",
           "ShieldVerdict", "shield_admissible", "shield_evaluate",
           "IgnoranceCertificate", "OpenWorldPosterior",
           "admissible_intersection", "certify_identifiability",
           "best_win", "judge_policy_outcome",
           "score_candidate", "select_by_value",
           "CompositionEvidence", "certified_p_vcc_lb",
           "posterior_mean_bound", "SequentialGLR",
           "calibrate_shift_threshold",
           "CheckabilityCheck", "Suppression",
           "checkability_advantage", "operating_point",
           "CPOMDP", "policy_authority", "policy_gap_delta",
           "rl_wrap_additivity", "RLDevice",
           "ExchangeWorkload", "exchange_ratio", "exchange_ratio_exact",
           "dwell", "gamma_binding", "separation_orders",
           "WAY_K_MAX", "way_group", "way_physical_window"]


@dataclass(frozen=True)
class Receipt:
    """One shot's outcome under the trust model. Immutable, plain data."""

    tier: str                    # "certified-optimal" | "degraded"
    accepted: bool
    weight: float                # the answer's weight (primal)
    bound: float                 # the CHECKED lower bound (dual)
    gap: float                   # weight - bound; 0 on the certified tier
    reason: str
    _certified: Certified | None = None

    def document(self, edges) -> dict[str, Any]:
        """The canonical, stranger-checkable JSON document for this shot."""
        from .cert_format import to_document
        c = self._certified
        return to_document(edges=edges, syndrome=self._syndrome,
                           cert=c.cert if c else None,
                           proven_optimal=self.accepted,
                           primal=self.weight, bound=self.bound,
                           reason=self.reason)

    _syndrome: tuple = ()


class MatchingCertifier:
    """Per-shot optimality receipts for matching-decoded codes.

    `edges`: iterable of (u, v, weight); use oneq.api.BOUNDARY for the
    boundary vertex. Weights are YOURS -- the receipt certifies min-weight
    conformance against the graph you declare, which is the honest contract
    (it is not maximum-likelihood over degenerate cosets, and says so).
    """

    def __init__(self, edges: Iterable[tuple]):
        self._cd = CertifyingDecoder(list(edges))
        self._edges = self._cd.edges

    @classmethod
    def from_pymatching(cls, matching) -> "MatchingCertifier":
        """Adopt a pymatching.Matching's graph (weights included)."""
        edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
                  float(e[2].get("weight", 1.0))) for e in matching.edges()]
        return cls(edges)

    @classmethod
    def from_dem(cls, dem) -> "MatchingCertifier":
        """Adopt a stim DetectorErrorModel via pymatching's loader."""
        import pymatching
        return cls.from_pymatching(
            pymatching.Matching.from_detector_error_model(dem))

    def _wrap(self, syndrome, r: Certified) -> Receipt:
        return Receipt(
            tier="certified-optimal" if r.accepted else "degraded",
            accepted=r.accepted, weight=r.primal, bound=r.bound,
            gap=max(0.0, r.primal - r.bound), reason=r.reason,
            _certified=r, _syndrome=tuple(int(s) for s in syndrome))

    def certify(self, syndrome: Sequence[int],
                correction: Iterable[tuple]) -> Receipt:
        """The fast ladder: milliseconds. ~95-99% certified on real data."""
        return self._wrap(syndrome, self._cd.certify(syndrome,
                                                     list(correction)))

    def certify_hard(self, syndrome: Sequence[int],
                     correction: Iterable[tuple]) -> Receipt:
        """Fast ladder -> escalation -> exact tier. Closes every shot whose
        answer is actually optimal; the exact tier costs minutes and is
        paid only on double refusal."""
        return self._wrap(syndrome, self._cd.certify_hard(syndrome,
                                                          list(correction)))

    def stream(self, shots: Iterable[tuple], **kw) -> Iterator:
        """Ordered streaming receipts; see oneq.streaming for the knobs."""
        from .streaming import StreamingCertifier
        return StreamingCertifier(self._edges, **kw).run(shots)


class QldpcCertifier:
    """The same contract for sparse-check (qLDPC) codes.

    `checks`: list of variable-index tuples (rows of H); `weights`: dict
    variable -> weight (log-likelihood ratios make min-weight = most-likely
    under independence). NP-hard regime: the certified RATE is empirical --
    measured 100% at operating noise on [[144,12,12]], stated per artifact,
    never promised by theory.
    """

    def __init__(self, checks, weights):
        self._checks = [tuple(int(i) for i in c) for c in checks]
        self._weights = {int(k): float(v) for k, v in weights.items()}

    def certify(self, syndrome: Sequence[int],
                error_support: Iterable[int], *,
                lazy: bool = True, rpc_rounds: int = 4) -> Receipt:
        """Certify a decoder's error support; on refusal with an integral
        LP the receipt's reason names the REPAIR -- a certified strictly
        better correction (take it)."""
        from .qldpc_cert import (QldpcCertificate, check_qldpc,
                                 feldman_dual, feldman_dual_lazy)
        syn = list(syndrome)
        sup = tuple(sorted(int(i) for i in error_support))
        if lazy:
            cert, bound, primal = feldman_dual_lazy(
                self._checks, self._weights, syn, return_primal=True)
        else:
            cert, bound, primal = feldman_dual(
                self._checks, self._weights, syn, return_primal=True,
                rpc_rounds=rpc_rounds)
        weight = sum(self._weights[i] for i in sup)
        if cert is None:
            return Receipt(tier="degraded", accepted=False, weight=weight,
                           bound=0.0, gap=weight,
                           reason="producer refused (degree/infeasible)",
                           _syndrome=tuple(syn))
        r = check_qldpc(checks=self._checks, weights=self._weights,
                        syndrome=syn,
                        cert=QldpcCertificate(error_support=sup,
                                              facet_duals=cert.facet_duals,
                                              box_duals=cert.box_duals))
        reason = r.reason
        if not r.accepted and primal is not None:
            r2 = check_qldpc(checks=self._checks, weights=self._weights,
                             syndrome=syn,
                             cert=QldpcCertificate(
                                 error_support=tuple(primal),
                                 facet_duals=cert.facet_duals,
                                 box_duals=cert.box_duals))
            if r2.accepted:
                reason = (f"REPAIR AVAILABLE: your answer is not minimum-"
                          f"weight, but the certified-optimal correction "
                          f"{tuple(primal)} is -- take it. ({r.reason})")
        return Receipt(tier="certified-optimal" if r.accepted
                       else "degraded",
                       accepted=r.accepted, weight=weight,
                       bound=bound, gap=max(0.0, weight - bound),
                       reason=reason, _syndrome=tuple(syn))


def verify(document: dict, edges) -> tuple[bool, str]:
    """THE STRANGER'S ENTRY POINT: re-check a receipt document from the
    edge list and arithmetic alone. Needs nothing from the producer.

    A malformed document is a REFUSED document, not a traceback: the
    stranger feeding a corrupted file gets a verdict with the reason,
    because an entry point that crashes on garbage teaches callers to
    wrap it in a bare except -- the exact silent-bypass the standards
    gate forbids in our own code."""
    from .cert_format import recheck
    try:
        return recheck(document, list(edges))
    except (ValueError, KeyError, TypeError) as e:
        return False, f"malformed document refused: {e}"


def verify_strict(cert: ExactCutCertificate, edges, syndrome
                  ) -> tuple[bool, str]:
    """THE SMALLER TRUST BOUNDARY: verify against the T-join cut dual
    alone, with no node potentials anywhere in the argument.

    WHY THERE ARE STILL TWO ENTRY POINTS, now that they agree.

    `verify` USED to accept a dual mixing node potentials y with odd-set
    terms z. That mixed object is NOT the dual of the T-join primal it
    claims optimality for -- potentials belong to the perfect-matching
    dual on the METRIC CLOSURE, a different problem -- and the
    consequence was demonstrated in tests/test_mixed_dual_soundness_gate:
    on a three-vertex graph it accepted a weight-20 correction as optimal
    when the true optimum was 2, because a NEGATIVE potential absorbed
    slack on the edges away from T while the potentials on T inflated the
    objective.

    CLOSED 2026-08-18. `matching_cert.check` now refuses a non-empty
    node_potentials outright, so both entry points enforce the same
    contract: a non-negative packing of T-odd cuts and nothing else. The
    difference that remains is the SERIALIZATION, not the mathematics --
    this one takes exact num/den pairs and admits no floats, while
    `verify` reads the shipped receipt format and converts.

    This entry point accepts one form only -- an exact non-negative
    packing of T-odd cuts, serialized as num/den pairs. Then weak
    duality does the entire job: a feasible primal equal to a feasible
    dual is already the certificate. No optimizer, no tolerance, no
    repair, no metric-closure flag to trust.

    RELEASED NUMBERS STAND, and now for a stronger reason than before.
    They already rested on a measurement -- the shipped producer does not
    construct exploiting duals, 601 suboptimal corrections and none
    accepted -- and no shipped document ever carried a node potential.
    That was a statement about the producer's behaviour; it is now a
    statement about the checker's contract, which is what a verification
    boundary is supposed to be.

    New evidence should still be minted in the strict form and checked
    here: `CertifyingDecoder.certify_strict` produces it, measured 30/30
    at d=5 against the mixed ladder's 29/30.
    """
    # AttributeError is in this list deliberately. An object that is not
    # an ExactCutCertificate fails on attribute access, not on a type
    # check, so omitting it makes the entry point CRASH on garbage --
    # the exact behaviour `verify` above exists to avoid, since a caller
    # who learns this can throw wraps it in a bare except and every
    # later refusal is swallowed with it.
    try:
        r = check_exact_cut(edges=list(edges), syndrome=list(syndrome),
                            cert=cert)
        return bool(r.accepted), str(r.reason)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        return False, f"malformed strict certificate refused: {e}"


def soundness_block(receipt: "Receipt", *, dem_level: int, claimed_level,
                    beta_ppb: int, exact_arithmetic: bool | None = None,
                    rho_exceeds_quantisation: bool | None = None) -> dict:
    """What soundness grade this receipt entitles a claim to, and the
    bound it carries there -- as a block for the SIGNED passport core.

    Verifying a document tells a stranger whether the answer was
    certified. It does not tell them what grade the surrounding claim may
    be PRESENTED at, nor what error bound holds at that grade, and those
    are the two things a passport asserts. This composes them and refuses
    to produce a block for a claim the evidence does not support --
    including when the DEM the receipt was decoded against has not itself
    been graded, since every decoder-assurance leg has the model in its
    ancestor set.

        blk = soundness_block(r, dem_level=3, claimed_level=L.PROVEN,
                              beta_ppb=500, exact_arithmetic=True)
        core = build_core(..., claimed_level=L.PROVEN, soundness=blk)
    """
    return block_for_receipt(
        receipt, dem_level=dem_level, claimed_level=claimed_level,
        beta_ppb=beta_ppb, exact_arithmetic=exact_arithmetic,
        rho_exceeds_quantisation=rho_exceeds_quantisation)
