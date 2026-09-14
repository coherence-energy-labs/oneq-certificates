r"""The certifying DECODER: decode, then prove it, escalating only when needed.

A certifying algorithm returns a solution AND a witness that an independent
checker can validate without trusting the solver. For QEC decoding the witness
is a feasible cut packing whose value equals the correction's weight; weak
duality then forces both to be optimal, and the argument holds no matter how
either was produced.

This is the same shape as `certify.py`'s distance router -- recognition gates,
cheapest lane first, fall closed with a priced reason -- applied to a different
question. There the lanes decide a code's distance; here the rungs try to prove
one shot's correction optimal. Neither ever averages lanes: the first one whose
verdict INDEPENDENTLY verifies decides.

WHY THERE IS A LADDER. Producing the packing is the expensive half, and the
candidate family it is drawn from can be enriched at roughly linear cost in
solver time. Richer families close more shots -- but on a 25-round d=5
space-time graph (601 nodes, 2742 edges, ~43 detectors lit per shot) the extra
candidates buy nothing past a point, while the LP keeps getting dearer:

    balls x4     122/200 certified    18.4 ms/shot
    balls x6     155/200 certified    23.5 ms/shot
    balls x8     168/200 certified    30.2 ms/shot
    balls x12    172/200 certified    56.4 ms/shot
    balls x24    172/200 certified   177.3 ms/shot     <- saturated at 86%

No single family reaches 90%. The ladder does, because its later rungs offer
the LP families the balls cannot generate at all.

*** THOSE TIMINGS PREDATE THE EXACT-REPAIR TIER AND NO LONGER DESCRIBE
    THE LADDER. RE-MEASURED 2026-08-18. ***

`_exact_optimality` was added in response to the 2026-08-04 audit's
finding 4 and sits on the hot path of every check; the table above was
never re-measured after it landed. On the SAME graph, rebuilt the way
tools/../checker_scaling.py builds it (stim surface_code:rotated_memory_z,
d=5, rounds=25 -> 600 nodes, 2742 edges), with corrections from
pymatching's decode_to_edges_array:

    CERTIFIED shots      52, 126, 270 ms      near the table above
    REFUSED shots      2291, 3022, 4808 ms    20-40x worse
    mean               1761.8 ms/shot         n = 6, mean |T| = 43.3

So the table is roughly right for shots that CERTIFY and badly wrong as
a per-shot figure, because the cost is concentrated in REFUSALS: the
exact repair runs all four spend orderings to completion and only then
fails. An independent profile puts ~72% of runtime in that repair's
inner loop plus Fraction overhead, against 2.7% in the LP -- and finds
only 10 of 141 repair calls succeed, so ~93% of that work is spent
proving nothing.

n = 6 here, enough to establish the order of magnitude and the
certified/refused split, not enough for a rate. Do not quote a single
ms/shot number from this paragraph without saying which half it is.

The optimisations that follow from it -- an incremental load vector, and
scaled integers in place of Fraction, both verdict-identical -- are
recorded in docs/OPEN_WORK.md rather than half-applied here.

AND DEEPER BALLS ARE NOT THE ANSWER, which took a measurement to establish. On
the shots the ladder could not close at d=7 over 25 rounds, raising the cap
from 8 to 32 and then to 128 -- sixteen times deeper -- closed the same 3 of 19
and left the same 0.42% residual gap. Balls are centred on SINGLE detectors;
when a shot lights ninety of them any useful ball swallows several, only the
T-ODD ones are admissible, and nothing that survives can express a moat around
a GROUP. Depth was never the missing ingredient.

The groups come from the packing's own slack. A set can only carry more weight
if every edge on its boundary still has some, so the sets worth adding are the
connected components of the TIGHT subgraph -- the edges this packing has
already saturated. Solve, read those components, add them, re-solve. The
objective is monotone because a richer family cannot lower a maximum, and the
loop halts as soon as a pass finds nothing new.

    d=7, 25 rounds, on the 19 shots the two-rung ladder refused
        cap 6,  6 rounds            11/19 closed  ->  93.3% overall
        cap 8, 20 rounds + growth   13/19 closed  ->  95.0% overall, gap 0.047%

    REAL WILLOW HARDWARE, same change
        d=7, 30 rounds   46.0% -> 77.3%
        d=5, 30 rounds   71.3% -> 90.7%
        d=7, 13 rounds   85.3% -> 93.3%

The rungs were chosen by measurement in two regimes, not by taste. They are a
default, not a law -- `ladder=` overrides them, and a caller who changes them
changes only cost and certification RATE, never soundness.

WHY ESCALATION CANNOT WEAKEN SOUNDNESS. Every rung is adjudicated by the same
independent checker, and the checker's verdict depends only on whether the
packing is feasible and meets the primal -- never on which rung produced it.
Escalating can turn a refusal into an acceptance only by finding a genuinely
tighter dual. It cannot turn a WRONG correction into a certified one, because
the correction is the one thing that does not change between rungs.

WHAT A REFUSAL MEANS. Not "the decoder is wrong". It means this answer was not
PROVEN optimal here, and the honest output is the degraded receipt:
syndrome-consistent, weight W, optimality gap at most W minus the best lower
bound found. That is strictly more than the silence every production decoder
offers today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .matching_cert import BOUNDARY, MatchingCertificate, check
from .moat_dual import MoatEngine
from .moat_growth import (build_adj, cs_family, matching_dual_candidates,
                          moat_growth, tight_components)

# (max_radii, use_growth, refine_rounds), cheapest first. Measured, not chosen.
#
# THE THIRD RUNG IS THE ANSWER TO THE RATE COLLAPSE. Balls cannot express a
# moat around a GROUP, and raising the radius cap sixteen-fold buys nothing
# (3/19 hard shots either way). The groups worth adding are the connected
# components of the TIGHT subgraph -- a set can only grow if no edge on its
# boundary is saturated -- so the LP's own slack names them. Solve, read the
# tight components, add them, re-solve. At d=7 over 25 rounds that takes the
# rate from 84.2% to 95.0%.
# The fourth field adds the COMPLEMENTARY-SLACKNESS family: sets grown from
# the correction's own tree-splits, absorbing foreign trees atomically so each
# cuts J exactly once. Only the shots rungs 1-2 refuse ever pay for it, and on
# exactly those shots it closed 9 of 17 at d=7 over 25 rounds -- with the
# JUICED variant (20 radii, 1500 pops, 40 refine rounds) closing FEWER (7/17):
# a bigger family blows the refinement cap and hurts the LP, so the knobs stay
# at the measured optimum rather than the intuitive maximum.
# The FIFTH field adds blossom-guided candidates: the metric-closure matching
# LP is solved with lazy odd-set cuts (it has only |T| ~ 100-170 nodes), and
# its DUALS name the optimal moat shapes -- y_t the depth each terminal's moat
# should stop at, each blossom the group whose union-moat should grow. Those
# shapes become candidates for the same max-form LP as every other family. On
# the 9 shots everything below refused at d=7 x 25 rounds it closed 7,
# taking the seed-99 rate to 148/150 (98.7%). Paid only by the shots rungs
# 1-3 refuse.
DEFAULT_LADDER: tuple[tuple, ...] = ((6, False, 0), (8, True, 20),
                                     (8, True, 12, True),
                                     (8, True, 12, True, True))

# The refusal tier's ladder: every budget the autopsy showed binding, raised
# together. Not the default because the default's caps ARE the million-shot
# memory envelope; a refused shot has earned the expensive retry.
ESCALATION_LADDER: tuple[tuple, ...] = ((12, True, 40),
                                        (16, True, 40, True),
                                        (16, True, 40, True, True))


@dataclass
class Certified:
    """The outcome for one shot."""

    accepted: bool
    reason: str
    rung: int                       # which rung closed it; -1 if none did
    primal: float
    bound: float
    gap: float
    cert: MatchingCertificate | None = None
    # The strongest packing found, EVEN WHEN THE SHOT WAS REFUSED. `cert` is
    # deliberately None on a refusal so no caller can mistake a rejected
    # attempt for a proof -- but the forgery battery needs a dual to build
    # attacks from, and conditioning the attacks on success means the hardest
    # shots, the ones where the bound was loose, are the ones never attacked.
    # That is exactly backwards, so the packing is kept separately under a name
    # that cannot be read as a verdict.
    best_dual: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    #: *** DOES THE CORRECTION ACTUALLY EXPLAIN THE SYNDROME? ***
    #:
    #: `None` means the producer did not say. The degraded receipt preserves
    #: that third state rather than inventing an answer from refusal prose.
    #:
    #: It exists because `degraded` decided this with
    #: `"odd-degree" not in self.reason` -- a substring search over
    #: English, gating a receipt whose own text claims "this correction
    #: explains the syndrome". Two different producers already phrase it
    #: differently (`matching_cert`: "odd-degree set != syndrome";
    #: `forgeries.NOT_A_CORRECTION`: "odd-degree set differs from the
    #: syndrome") and both happen to contain the hyphenated token. That
    #: is a coincidence, not a contract: drop the hyphen anywhere and a
    #: correction that does NOT explain the syndrome starts publishing
    #: `syndrome_consistent: true`.
    #:
    #: Same defect as `certify.py` reading `"exhaustive" in cert.note`,
    #: and in the same unsafe direction -- the ABSENCE of a phrase is
    #: read as good news.
    syndrome_consistent: bool | None = None

    @property
    def degraded(self) -> dict:
        """The honest receipt for a shot that was not proven optimal."""
        return {
            "syndrome_consistent": self.syndrome_consistent,
            "weight": self.primal,
            "certified_lower_bound": self.bound,
            "optimality_gap_at_most": self.gap,
            "claim": ("this correction explains the syndrome and its weight "
                      "exceeds a CHECKED lower bound by at most the stated "
                      "gap; it is NOT a proof of optimality"),
        }


class CertifyingDecoder:
    """Wraps a decoder's answers so each arrives with a checkable witness.

    Per-graph setup happens once; `certify` is then pure per-shot work sharing
    nothing but read-only structure, which is what makes the million-shot run
    embarrassingly parallel.
    """

    def __init__(self, edges, *, ladder=DEFAULT_LADDER,
                 max_family: int = 600, max_candidates: int = 3000,
                 blossom_rounds: int = 12, blossom_levels: int = 6,
                 l0_table=None, l0_masks=None):
        # *** L0: THE CONFIGURATION IS CHECKED BEFORE ANYTHING IS DERIVED
        # FROM IT. *** Every structure below -- the moat engine, the
        # adjacency, the weight map -- comes from ONE read of `edges`, so
        # this is the seam. A corrupted edge that reaches them decodes
        # against a graph nobody provisioned, and `H x_hat = s` cannot
        # see it: a flipped observable MASK changes no weight, so the
        # correction is unchanged and a check on the correction is
        # looking at an object the fault never touched (0 of 16,000
        # shots, measured).
        #
        # OPTIONAL, and absent by default, so no existing caller changes
        # behaviour. Supplying a table is what turns `abft_l0` from an
        # instrument into something a decode is refused by.
        if l0_table is not None:
            self._l0_verify(edges, l0_table, l0_masks)
        self.edges = [(int(u), int(v), float(w)) for u, v, w in edges]
        self.ladder = tuple(ladder)
        # a hard ceiling on candidates, so one pathological shot cannot
        # exhaust a worker and take the whole run down with it
        self.max_family = int(max_family)
        # total candidate rows the LP may see: balls (cap x |T|) plus extras.
        # The membership matrix is candidates x nodes and the crossing matrix
        # candidates x edges, so this is the real memory knob.
        self.max_candidates = int(max_candidates)
        self.blossom_rounds = int(blossom_rounds)
        self.blossom_levels = int(blossom_levels)
        self.engine = MoatEngine(self.edges)
        self.adj = build_adj(self.edges)
        self.W: dict[tuple[int, int], float] = {}
        for u, v, w in self.edges:
            k = self.key((u, v))
            if k not in self.W or w < self.W[k]:
                self.W[k] = w
        self._escalated: CertifyingDecoder | None = None

    def escalated(self) -> "CertifyingDecoder":
        """The refusal-tier decoder: same graph, budget dam removed.

        The autopsy that motivates this: at base caps the hardest hardware
        configuration refused ~5% of shots, and raising every budget --
        candidate caps, refinement rounds, the closure loop's cutting-plane
        rounds, emission density -- closed most of them. The caps exist for
        the million-shot memory envelope; a shot that already REFUSED has
        earned the expensive retry, and paying it only on refusals keeps the
        envelope for the other ~95%. Built lazily, cached, shares nothing
        mutable with the base decoder.
        """
        if self._escalated is None:
            self._escalated = CertifyingDecoder(
                self.edges, ladder=ESCALATION_LADDER,
                max_family=12000, max_candidates=60000,
                blossom_rounds=60, blossom_levels=999)
        return self._escalated

    def certify_exact(self, syndrome, correction, *, seed_sets=None) -> Certified:
        """The FINAL tier: T-odd cuts generated natively, no translation.

        Padberg-Rao separation in the decoding graph itself; by
        Edmonds-Johnson the converged bound equals the minimum T-join
        weight, so this tier certifies wherever the decoder's answer is
        optimal -- the ladder's ceiling becomes the decoder's own (audited)
        optimality. Cost is a Gomory-Hu tree per separation round; measured
        ~1-4 minutes on d7x30 hardware shots, which is why it is the LAST
        tier and not a rung. The generated sets go through the same engine
        packing and the same independent checker as every other candidate.
        """
        from . import tjoin_exact as _tj
        from .tjoin_exact import exact_tjoin_sets
        corr = tuple(self.key(e) for e in correction)
        primal = self.weight(corr)
        _tj.LAST_EXIT.update(reason="not_run", rounds=0)
        sets, _lp = exact_tjoin_sets(self.edges, syndrome,
                                     seed_sets=seed_sets)
        # Read immediately, before anything else can call the separator.
        exit_reason = dict(_tj.LAST_EXIT)
        z, obj = self.engine.dual(syndrome, max_radii=2,
                                  extra_sets=tuple(frozenset(S)
                                                   for S in sets))
        cert = MatchingCertificate(matched_edges=corr, node_potentials={},
                                   blossom_duals=z)
        r = check(edges=self.edges, syndrome=syndrome, cert=cert)
        if r.accepted:
            return Certified(True, r.reason + " [exact tier]", 5, primal,
                             obj, 0.0, cert, dict(z),
                             {"tier": "exact", **exit_reason})

        # EXACT ESCALATION. The float packing refused, and at this point the
        # interesting structure lives below float resolution: the pool may
        # admit a packing that attains the primal EXACTLY while every float
        # solve lands an ulp short on a wrong basis (measured: 775 d5
        # refusals, float gap 0.000000000, all rung -1). Solve the SAME pool
        # with the exact rational simplex and re-offer the improved dual to
        # THE SAME independent checker. The solver never accepts anything --
        # exact_lp's word is not evidence, the checker's verdict is -- so
        # this can only recover rate, never weaken soundness. And when even
        # the exact optimum falls short, the shortfall is now PROVEN over
        # the whole pool in rational arithmetic, which turns "not proven
        # optimal" into a measured statement about the pool or the
        # correction rather than about float luck.
        from .exact_lp import exact_cut_packing_lp
        pool = [frozenset(S) for S in sets]
        if pool:
            ekeys = sorted(self.W)
            rows = [[1 if ((k[0] in S) != (k[1] in S)) else 0 for S in pool]
                    for k in ekeys]
            zx, objx, _it = exact_cut_packing_lp(
                rows, [self.W[k] for k in ekeys])
            if zx is not None:
                zmap = {S: float(v) for S, v in zip(pool, zx) if v > 0}
                cert2 = MatchingCertificate(matched_edges=corr,
                                            node_potentials={},
                                            blossom_duals=zmap)
                r2 = check(edges=self.edges, syndrome=syndrome, cert=cert2)
                if r2.accepted:
                    return Certified(
                        True, r2.reason + " [exact-rational tier]", 5,
                        primal, float(objx), 0.0, cert2, dict(zmap),
                        {"tier": "exact", "exact_lp": True, **exit_reason})
                if float(objx) > obj:
                    obj, z = float(objx), zmap

        # A CONVERGED short bound and a STALLED short bound are opposite
        # findings: the first says no lighter T-join exists and the decoder
        # was suboptimal; the second says the separator gave up. Naming it
        # here keeps them out of one bucket.
        # *** THE SHORTFALL IS OUR SOLVER'S WORD, AND IT DOES NOT HAVE TO
        # BE. *** The exact simplex above computed the Lagrange
        # multipliers of its active constraints on the way to the
        # optimum, and threw them away. Those multipliers ARE a feasible
        # dual point for the packing LP, so `w^T u < primal` is a
        # certificate that NO dual over this pool attains the correction
        # -- weak duality, one line, re-checkable by a stranger in one
        # pass with no solver at all.
        #
        # Free: the same iteration, the same rationals, nothing extra
        # solved. It converts "our exact LP says the pool falls short"
        # from an assertion into an object, which is the only kind of
        # claim this repository is supposed to ship.
        #
        # Attached, never acted on here. A refusal is already the right
        # outcome; what changes is that the reason can now be verified.
        impossible = self._impossibility_certificate(pool, primal)
        return Certified(False, f"exact tier ({exit_reason['reason']}, "
                                f"{exit_reason['rounds']} rounds): bound "
                                f"{obj:.9f} short of primal {primal:.9f} "
                                f"({r.reason})", -1,
                         primal, obj, primal - obj, None, dict(z),
                         {"tier": "exact", **exit_reason,
                          "repair_impossible": impossible})

    def _impossibility_certificate(self, pool, primal):
        """A checkable proof that no dual over `pool` attains `primal`.

        Returns the certificate dict, or a short string saying why there
        is none. NEVER raises: this runs on the refusal path, where an
        exception would turn a correct refusal into a crash.
        """
        from fractions import Fraction as _F

        from .repair_impossible import IMPOSSIBLE, prove_impossible_exact
        if not pool:
            return "no pool"
        try:
            ekeys = sorted(self.W)
            rows = [[1 if ((k[0] in S) != (k[1] in S)) else 0 for S in pool]
                    for k in ekeys]
            cert = prove_impossible_exact(
                rows, [self.W[k] for k in ekeys], _F(primal),
                edge_keys=ekeys,
                variable_keys=[tuple(sorted(S)) for S in pool])
        except Exception as exc:                          # noqa: BLE001
            return f"not produced: {type(exc).__name__}: {exc}"[:160]
        if cert.verdict != IMPOSSIBLE:
            return cert.why
        return cert.as_dict()

    def certify_strict(self, syndrome, correction):
        """Emit and check a STRICT T-odd-cut certificate. [OPEN_WORK E2]

        Returns (accepted, reason, certificate_or_None).

        WHY THIS EXISTS. `certify_exact` already computes an EXACT rational
        cut packing with `exact_cut_packing_lp` -- and then throws the
        exactness away: `{S: float(v) for S, v in ...}` downgrades every
        dual to a double and hands it to the MIXED checker, whose soundness
        rests on the producer choosing not to construct an exploiting dual.
        The strict checker wants precisely what the producer already has.

        So this is not a new theorem, it is a path that stopped short. The
        packing is offered as exact numerator/denominator pairs to
        `check_exact_cut`, which admits ONE dual form -- a non-negative
        packing of T-odd cuts, the literal LP dual on the ORIGINAL graph.
        No node potentials, no metric-closure flag to trust, no repair
        search, no tolerance.

        NOTHING HERE DECIDES. The LP's word is not evidence; the checker's
        verdict is. A refusal is a refusal: this cannot recover a rate that
        the strict contract does not support, and it must not, because
        that contract is the point.
        """
        from fractions import Fraction

        from .exact_lp import exact_cut_packing_lp
        from .matching_cert_strict import ExactCutCertificate, check_exact_cut
        from .tjoin_exact import exact_tjoin_sets

        corr = tuple(self.key(e) for e in correction)
        sets, _lp = exact_tjoin_sets(self.edges, syndrome)
        pool = [frozenset(S) for S in sets]
        if not pool:
            return False, "no T-odd cut was separated: nothing to pack", None

        ekeys = sorted(self.W)
        rows = [[1 if ((k[0] in S) != (k[1] in S)) else 0 for S in pool]
                for k in ekeys]
        zx, _obj, _it = exact_cut_packing_lp(rows, [self.W[k] for k in ekeys])
        if zx is None:
            return False, "exact cut packing did not solve", None

        # EXACT ALL THE WAY THROUGH. Fraction in, Fraction out -- a float()
        # anywhere on this path would reintroduce the rounding the strict
        # schema exists to exclude, and it would do so silently.
        duals = {S: Fraction(v) for S, v in zip(pool, zx) if v > 0}
        cert = ExactCutCertificate(matched_edges=corr, cut_duals=duals)
        r = check_exact_cut(edges=self.edges, syndrome=syndrome, cert=cert)
        return bool(r.accepted), str(r.reason), cert

    def certify_hard(self, syndrome, correction) -> Certified:
        """The full ladder: base -> escalated -> exact. Best result wins.

        Each tier is paid only by shots the previous one refused, so the
        base memory envelope holds for the ~95%, the escalation seconds for
        the ~3%, and the exact tier's minutes for the remaining handful --
        whose certificates then close exactly (measured: 7/7 stubborn d7x30
        shots, gaps 0.095-1.237, all to 0.000)."""
        r = self.certify(syndrome, correction)
        if r.accepted:
            return r
        r2 = self.escalated().certify(syndrome, correction)
        if r2.accepted:
            return r2
        best = r2 if r2.bound > r.bound else r
        r3 = self.certify_exact(syndrome, correction,
                                seed_sets=list(best.best_dual) or None)
        return r3 if r3.accepted or r3.bound > best.bound else best

    def certify_or_repair(self, syndrome, correction) -> Certified:
        """The full ladder, then a certified REPAIR when refusal is earned.

        When certify_hard refuses, decode EXACTLY (exact_decode: Fraction
        Dijkstra + Fraction blossom, no float anywhere) and compare the
        exact weights. Three outcomes, each honest:

          repaired weight <  original:  the original correction was PROVABLY
              suboptimal -- the ulp-tie class measured on d5 hardware. The
              exactly-minimum correction is offered to the SAME ladder, and
              if it certifies, the receipt that ships is for the REPAIRED
              answer, flagged `repaired: True` with both weights recorded.
              The certificate need not explain the original decoder; it
              proves the answer that ships is optimal.
          repaired weight == original:  the original was already optimal and
              the refusal is a certificate-side limitation (pool-limited).
              No repair is possible by construction; the refusal stands,
              now annotated with that proof.
          exact decode refuses (negative weights, disconnected component):
              the refusal stands untouched.

        SOUNDNESS IS UNCHANGED BY CONSTRUCTION. The repair is a PRODUCER:
        its output passes reproduces_syndrome() here and then the same
        independent checker as every other correction. A wrong repair is
        refused like any other wrong answer, and `repaired` receipts are
        counted SEPARATELY downstream so no headline number silently
        blends proof-of-original with proof-of-repair.
        """
        r = self.certify_hard(syndrome, correction)
        if r.accepted:
            return r
        from fractions import Fraction
        from .exact_decode import exact_min_tjoin, reproduces_syndrome
        corr2, w2, why = exact_min_tjoin(self.edges, syndrome)
        if corr2 is None or not reproduces_syndrome(corr2, syndrome):
            r.stats["repair"] = f"unavailable ({why})"
            return r
        exact_primal = sum(
            (Fraction(self.W[self.key(e)]) for e in correction), Fraction(0))
        if w2 >= exact_primal:
            # Original already exactly optimal: the refusal is about the
            # certificate, not the correction, and that is now PROVEN.
            r.stats["repair"] = ("not_needed: original correction is exactly "
                                 "optimal; refusal is certificate-side")
            return r
        r2 = self.certify_hard(syndrome, corr2)
        if r2.accepted:
            r2.stats.update(
                repaired=True,
                original_weight=float(exact_primal),
                repaired_weight=float(w2),
                improvement=float(exact_primal - w2))
            r2.reason += (" [CERTIFIED REPAIR: original correction was "
                          f"provably suboptimal by {float(exact_primal - w2):.3e}]")
            return r2
        r.stats["repair"] = "repair found but did not certify"
        return r

    @staticmethod
    def _l0_verify(edges, table, masks) -> None:
        r"""Read every edge through the L0 accumulator, then commit.

        Raises `abft_l0.ConfigurationFault` rather than returning a
        verdict: a decoder built from a configuration that does not
        match its provisioned table must not exist, because anything
        holding one would be entitled to assume it decoded against the
        graph that was provisioned.

        The mask replica is checked IN FULL and is not optional here.
        `commit_check` accepts `check_mask_replica=False` to measure the
        vacuity of omitting it -- that switch is for the experiment that
        proves the P1 class is real, never for a decode path.

        *** BUT AT THIS CALL SITE THE REPLICA CHECK IS REDUNDANT, AND A
        READER SHOULD KNOW WHICH THING CATCHES P1 HERE. *** `Edge.tag()`
        hashes the MASK along with the weight and the endpoints, so a
        flipped mask fails the per-edge accumulator before the replica
        loop is reached -- and `masks_read` below is derived from the
        very edges that accumulator just verified. Measured: over 5,920
        configurations (every single-field weight and mask corruption of
        400 random graphs, plus their clean forms) enabling or disabling
        the replica changed 0 verdicts, which is why the aimed mutant for
        it was removed as equivalent rather than left to look guarded.
        It stays in the call because it is NOT redundant where it was
        designed for: a commit stage reading masks the per-edge
        accumulation never touched.
        """
        from .abft_l0 import (Accumulator, ConfigurationFault, commit_check,
                              edges_for)
        if masks is None:
            raise ConfigurationFault(
                "an L0 table was supplied without observable masks. The "
                "mask is the P1 field -- one bit at d=5 is 17.9x and "
                "distance-independent -- so verifying without it would "
                "cover every field except the one that matters most")
        read = edges_for(edges, masks)
        acc = Accumulator(table=table)
        for e in read:
            acc.read(e)
        ok, why = commit_check(acc, table,
                               masks_read={e.eid: e.mask for e in read},
                               check_mask_replica=True)
        if not ok:
            raise ConfigurationFault(
                f"L0 refused this decoder's configuration: {why}")

    @staticmethod
    def key(e) -> tuple[int, int]:
        u, v = int(e[0]), int(e[1])
        # BOUNDARY IS ALWAYS SECOND. Written boundary-first,
        # u = -1 makes `u <= v` already true, so the swap never
        # fired: (-1, 3) and (3, -1) both survived as distinct
        # keys, the same edge was counted twice, and a valid
        # document was refused with GRAPH MISMATCH -- the exact
        # outcome graph_fingerprint exists to avoid.
        if u == BOUNDARY:
            return (v, u)
        if v == BOUNDARY:
            return (u, v)
        return (u, v) if u <= v else (v, u)

    def weight(self, correction) -> float:
        return sum(self.W[self.key(e)] for e in correction)

    def certify(self, syndrome, correction) -> Certified:
        """Try to prove `correction` optimal for `syndrome`, cheapest rung first."""
        corr = tuple(self.key(e) for e in correction)
        primal = self.weight(corr)
        best_bound = 0.0
        best_reason = "no dual produced"
        best_z: dict = {}
        # `None`, not `True`: if no rung ever ran, the T-join test was
        # never performed and the honest state is "not asked". Starting
        # at `True` would publish `syndrome_consistent` on a shot whose
        # syndrome nothing looked at.
        best_consistent: bool | None = None
        tried = 0
        for rung, spec in enumerate(self.ladder):
            cap, use_growth = spec[0], spec[1]
            refine_rounds = spec[2] if len(spec) > 2 else 0
            use_cs = spec[3] if len(spec) > 3 else False
            use_blossom = spec[4] if len(spec) > 4 else False
            tried += 1
            extra: set = set()
            if use_blossom:
                # shapes dictated by the exact matching dual on the metric
                # closure -- consulted, translated, then re-optimized and
                # CHECKED like any other candidate. A producer may use any
                # oracle it likes; the checker cannot be talked into anything.
                extra |= set(matching_dual_candidates(
                    self.edges, syndrome, adj=self.adj,
                    max_rounds=self.blossom_rounds,
                    levels_per_set=self.blossom_levels))
            if use_cs:
                # sets that cut the CORRECTION exactly once -- the only shape
                # complementary slackness allows in an optimal dual's support
                extra |= set(cs_family(self.edges, corr, syndrome,
                                       adj=self.adj))
            if use_growth:
                zg, _og = moat_growth(self.edges, syndrome, adj=self.adj)
                # |= AND NOT =. The first wiring ASSIGNED here, silently
                # discarding the CS sets added two lines up -- so the new rung
                # behaved identically to the old one and the sanity run showed
                # exactly that: same 133/150, rung 2 never credited. A rung
                # that cannot differ from its predecessor is dead weight, and
                # the histogram is what caught it.
                # THE SEED NEEDS THE CAP TOO. The refinement loop was bounded
                # and the growth seed was not, so a shot lighting ~170
                # detectors handed the LP thousands of candidates on the FIRST
                # call and the membership matrix blew the worker's memory
                # before any refinement ran. A bound that only covers the path
                # you were thinking about is not a bound.
                #
                # Largest first: a moat carrying more dual weight is more
                # likely to be load-bearing, so if the family must be truncated
                # this keeps the part that matters.
                extra |= set(sorted(zg, key=lambda S: -zg[S])[:self.max_family])
            # THE BUDGET APPLIES TO EVERY SOLVE, NOT JUST REFINEMENT. The
            # candidate count of the INITIAL solve is cap x |T| balls plus the
            # extras, and on a 170-detector hardware shot with the CS family
            # that is ~2,300 columns against a 12k-edge graph -- the exact
            # allocation that killed 454 chunks, every one of them on the
            # largest configurations, while the refinement loop underneath sat
            # safely inside a budget the first call never saw. Shrink the ball
            # cap first (the extras are the informative family); truncate the
            # extras only if that is not enough, keeping emission order, which
            # is nested-first per split.
            nT = max(len(syndrome), 1)
            eff_cap = cap
            if eff_cap * nT + len(extra) > self.max_candidates:
                eff_cap = max(2, (self.max_candidates - len(extra)) // nT)
            extra_t = tuple(extra)
            if eff_cap * nT + len(extra_t) > self.max_candidates:
                extra_t = extra_t[:max(0, self.max_candidates - eff_cap * nT)]
            z, obj = self.engine.dual(syndrome, max_radii=eff_cap,
                                      extra_sets=extra_t)
            cap = eff_cap
            # ITERATIVE REFINEMENT. Each pass reads the components the current
            # packing has saturated and offers them back as candidates. The
            # objective is monotone -- a richer family cannot lower a maximum --
            # so this can only tighten the bound, and it stops as soon as a pass
            # discovers nothing new.
            for _ in range(refine_rounds):
                fresh = set(tight_components(self.edges, z, syndrome)) - extra
                if not fresh:
                    break
                # PRUNE TO THE SUPPORT BEFORE ADDING. Accumulating every
                # candidate ever generated made the family grow without bound,
                # and the LP's membership matrix is candidates x edges -- on a
                # d=7 30-round graph that reached tens of millions of booleans
                # per shot and the run died with std::bad_alloc inside HiGHS.
                #
                # A set the LP gave zero weight is not contributing; carrying it
                # forward costs memory and solver time to re-derive that it is
                # worthless. Keeping only the current support plus the new tight
                # components bounds the family by what is actually load-bearing,
                # and cannot lose ground: the pruned sets had value zero, so the
                # previous objective is still attainable.
                extra = {S for S in extra if z.get(S, 0.0) > 1e-12} | fresh
                if len(extra) > self.max_family:
                    break
                if cap * len(syndrome) + len(extra) > self.max_candidates:
                    # The BALL family scales with the syndrome too: cap radii
                    # per fired detector, so a heavy shot pays 8 x |T| rows
                    # before a single extra set is added. Budget the total.
                    break
                extra_r = tuple(extra)
                if cap * nT + len(extra_r) > self.max_candidates:
                    extra_r = extra_r[:max(0, self.max_candidates - cap * nT)]
                z2, obj2 = self.engine.dual(syndrome, max_radii=cap,
                                            extra_sets=extra_r)
                if obj2 <= obj + 1e-12:
                    break
                z, obj = z2, obj2
            if obj > best_bound or not best_z:
                best_bound, best_z = obj, z
            cert = MatchingCertificate(matched_edges=corr, node_potentials={},
                                       blossom_duals=z)
            r = check(edges=self.edges, syndrome=syndrome, cert=cert)
            if r.accepted:
                return Certified(True, r.reason, rung, primal, obj, 0.0, cert,
                                 dict(z), {"rungs_tried": tried},
                                 syndrome_consistent=r.syndrome_consistent)
            best_reason = r.reason
            # CARRY THE CHECKER'S OWN ANSWER FORWARD. Without this the
            # field on `Certified` is inert and `degraded` falls back to
            # the substring read it was added to replace -- a guard that
            # nothing sets is prose with a type annotation.
            best_consistent = r.syndrome_consistent
            # ONLY A GAP IS WORTH ESCALATING FOR. If the correction's
            # odd-degree set disagrees with the syndrome, or the packing came
            # back infeasible, no richer candidate family repairs it -- the
            # fault is in the correction or in the producer, not in the bound's
            # tightness. Climbing the ladder there pays three times for one
            # refusal and reports the same answer.
            if not r.only_obstruction_is_gap:
                break
        return Certified(False, best_reason, -1, primal, best_bound,
                         max(0.0, primal - best_bound), None, best_z,
                         {"rungs_tried": tried},
                         syndrome_consistent=best_consistent)
