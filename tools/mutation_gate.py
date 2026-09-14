#!/usr/bin/env python3
"""Mutation gate — proves the test suite can actually fail.

A suite that passes tells you nothing until you have watched it go red for the
right reason. This injects known defects into the source and asserts that a
SPECIFIC named test catches each one. If a mutant survives, the corresponding
check is decorative and the build fails.

The estate's own anti-false-pass discipline (glasshouse/tools/mutation_gate.py,
67/67 LIVE) applied to ONE-Q. Mutant 0 is the historical all-zeros-key defect.

    python tools/mutation_gate.py
    python tools/mutation_gate.py --json evidence/mutation_gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# The scoring rule, in its own module so that a mutant can be aimed at it
# without its find-string also appearing in the registry that holds the
# mutant. See `mutation_outcome` for why that matters.
from mutation_outcome import NO_TEST_RAN as _NO_TEST_RAN  # noqa: E402
from mutation_outcome import counts as _counts  # noqa: E402
from mutation_outcome import denominator as _denominator  # noqa: E402
from mutation_outcome import outcome as _outcome  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _process_may_observe_mutant(command: str) -> bool:
    """Whether a command may import source during an in-place mutation.

    The full wall and a full pytest run are consumers of the very files this
    tool temporarily replaces. They must be treated as conflicts in addition
    to the certificate-producing experiments that motivated the original
    interlock.
    """
    import re
    normal = command.lower().replace("\\", "/")
    if re.search(
            r"close_open_instances|close_all_reachable|queue_tail|"
            r"sat_open_probe|gate_all\.py", normal):
        return True
    return "pytest" in normal and (
        " tests/" in normal or normal.rstrip().endswith(" tests")
    )

_SOURCES = (
            # THE DUAL AS A PHYSICAL INSTRUMENT (2026-08-31). Three
            # modules that turn numbers this repository already computed
            # and then discarded into published claims: a proven interval
            # on the error a device actually made, a minimum-variance
            # shot weighting, and a test of whether the declared error
            # model is the device's. Each can be made quietly flattering
            # by a one-line change -- filter the hard shots out, weight
            # uniformly, judge on the p-value alone -- which is exactly
            # what a mutant is for.
            "certified_weight",
            "soft_estimator",
            "dem_calibration",
            # LOCATING CERTIFIED WEIGHT, 2026-09-03. The duals were
            # always there; the question is whether they say WHERE.
            # Mostly they do not, and every mutant here restores a
            # version of the map that answers anyway.
            "fragility_map",
            # DEVICE LAWS, 2026-09-03. ARC induces the rules of an
            # unseen game from play and kills a rule on one
            # counterexample. Ported, with the three-outcome verdict
            # that keeps silence from voting.
            "device_laws",
            # THE EXACT DUAL ENVELOPE, 2026-09-04. The optimal dual is a
            # polytope; this ranges a functional over the whole face and
            # certifies each extreme with an exact rational dual. Its
            # only product is a proof, so every mutant removes one.
            "dual_envelope",
            # THE ANNIHILATOR OF THE FACE, 2026-09-04: which functionals
            # of the dual are DETERMINED. Eight mutants, each making the
            # dual look more decided than it is.
            "face_annihilator",
            # THE EDGE-FAMILY OPERATOR, 2026-09-05: the arithmetic behind
            # the per-family re-weighting that cut logical errors 2-7x
            # on hardware (R14-R18) and the lane's declared calibration
            # layer. Three mutants, each a way to scale the wrong edges.
            "dem_families",
            # THE MODEL MEASURED FROM DETECTORS, 2026-09-06. R23: the
            # device's DEM is wrong DIFFERENTIALLY -- space-like
            # probabilities overstated 17-38x while time-like are within
            # 1.7x and sometimes understated -- and the two-point
            # estimator reads the truth off the detectors with no
            # calibration at all, beating the device's own model 8.47x on
            # held-out shots. Its formula is four symbols long and every
            # one of them is a way to publish a different model.
            "dem_measured",
            # SITE-TO-SITE NOISE CORRELATION, 2026-09-07. A decoherence-free
            # memory stores information in a relationship that shared noise
            # barely moves, and the whole decision turns on
            # Gamma = 2(Cii + Cjj - 2Cij) -- a variance of a DIFFERENCE. One
            # sign there inverts which pair a selector picks. Measured on
            # hardware the correlations come in ~40x BELOW the crossover, so
            # the selector must DECLINE, and a selector that cannot decline
            # is an advertisement rather than an instrument.
            "noise_correlation",
            # THE HIDDEN-STATE LEAKAGE PRIOR, 2026-09-08. A Baum-Welch fit
            # whose E-step or M-step is off by one normalisation still
            # converges, still returns a posterior, and still reprices a
            # graph -- the mutants here are those slips.
            "leakage_hmm",
            # THE i.i.d. ASSUMPTION, 2026-09-03. Every DEM in this field
            # assumes shots are independent and identically distributed,
            # and almost nobody tests it. This one can be tested with no
            # model at all -- under i.i.d. shot order is exchangeable, so
            # permuting gives an EXACT null. The failure mode is not a
            # wrong p-value; it is a BLIND test reporting "no drift
            # found", which reads identically to a stable device. So the
            # mutants aim at the sensitivity, at the refusals, and at the
            # temptation to combine four correlated block sizes into one
            # confident number.
            "exchangeability",
            # GA-11, 2026-08-24. The WAY readout floor. The whole theory
            # leg reduces to ONE number, K_MAX = 1.628332, and every
            # verdict about every device moves with it -- so the
            # stationarity condition that produces it carries a mutant,
            # as does the `eps^2 = 4p` conversion whose absence made the
            # model read as unphysical everywhere.
            "way",
            # ARR, 2026-09-08 (campaign 4.6). The band optimiser's answer
            # is now a closed form, ln(V - 1) / (2 mu), competing with a
            # grid. Two one-line edits make it wrong in the two
            # directions that matter: shift the crossover off V = 2, or
            # stop the closed form from competing and reinstate the grid
            # artefact the fix exists to remove.
            "arr",
            # RANK 5's INSTRUMENT, 2026-09-08 (campaign 2.5). The
            # deferred-Pauli propagator decides the document's kill rule
            # (median support at depth 100 vs 20 patches). Two one-line
            # edits make the verdict wrong in the flattering direction:
            # a T that never blocks (every boundary "deferrable"), and a
            # CX rule with control and target swapped (the support stops
            # growing where it should grow). The stim oracle sees the
            # second; only the direct rule test sees the first.
            "pauli_defer",
            # GT-CT, 2026-08-25. The gate turns entirely on whether
            # switching distance is cheaper than sitting still, so
            # the cost model is where a false PASS would come from:
            # drop the ancillas, shorten the seam, or protect the
            # seam at the LARGER distance and the switching arm wins
            # on arithmetic nobody checked.
            "code_trajectory",
            # E0/E1, 2026-08-25. Every number answers "how much
            # circuit runs before the decoder has to answer", and
            # there are exactly two ways to inflate it: count a gate
            # as transparent when it is not, or lose the correction
            # as it spreads. Both are mutated below, as are the
            # outcome table's own thresholds.
            "transparency",
            # E2, 2026-08-26. Every policy here can be made
            # to look good by not charging it for something,
            # so most of these mutants are an unpaid bill.
            "bakeoff",
            # THE HIGHEST-RISK MODULES THAT CARRIED NO
            # MUTANT, 2026-08-27. Ranked by size, branch
            # count, refusal count and wall reach -- 69
            # modules had none and 22 of those contain
            # refusals, which is the shape a decorative
            # check takes.
            "distance_portfolio",
            "latent_parity",
            "branch_certificate",
            "matching_cert_strict",
            "quotient_reduction",
            "git_pin",
            "capability_plan",
            # GA-0, 2026-08-24. The dwell law. Its headline is a claim
            # of ROBUSTNESS -- "no outcome kills the architecture" -- and
            # what makes that falsifiable is the break point: the gamma
            # at which latency would bind. Delete the break point and the
            # claim reverts to an assertion.
            "dwell",
            # GT-XR, 2026-08-24. The exchange-rate law. Its own worked
            # example is the anchor: a mis-transcribed factor of 4 in
            # `eps_syn/eps_QEC = n_rot C_T lnL / (4 n_ops d^2 ln2)` would
            # survive every review and change every allocation decision
            # downstream. Also carries the finding that the corpus's form
            # is the LARGE-d ASYMPTOTIC.
            "exchange_rate",
            # GA-14, 2026-08-24. Wrapping the RL calibrator. The stated
            # bar -- "gains additive to within 30%" -- is satisfied by
            # gA=gB=0 exactly, and cannot separate additive from
            # independent-multiplicative composition below g~0.6. Both
            # blind spots carry mutants.
            "rl_wrap",
            # GA-4b, 2026-08-23. The AUTHORITY rule -- whether the
            # kernel may actuate at all. Delta is bounded in [0,1] BY
            # CONSTRUCTION (the certified class is a subset of all
            # policies and a superset of the constant ones), and a live
            # run produced Delta = 1.0595, which is a set reported as
            # worse than its own subset. Both impossible directions now
            # refuse, and both carry mutants.
            "policy_gap",
            # GA-13, 2026-08-23. The FIRST of the three finish-line
            # conditions. Its mechanism is one line of calculus, so every
            # mutant here is aimed at what the calculus LEAVES OUT: the
            # false-accept exposure that more shots buy, the classes the
            # mechanism must refuse, and the baseline's right to use every
            # tool except a smaller distance.
            "checkability",
            # GA-12, 2026-08-23. The TERMINAL claim: the certified
            # bound never exceeds the measured P_VCC. Its own spec is
            # passed perfectly by a bound of 0, so the module carries
            # the non-vacuity instruments too -- and two of THOSE were
            # wrong first (a pooled rank correlation over dependent
            # clusters, and a max-over-ratios that selected whichever
            # denominator was noisiest). A silent survivor here closes
            # the gate the architecture has no successor for.
            "composition",
            # GA-12's complementary detector, 2026-08-23. Built because
            # the open-world posterior needs p95 448 shots to see an
            # out-of-library device, which priced the lag charge at 0.72
            # of the whole certified floor. The null is COMPOSITE: score
            # against one library member instead of the min over all of
            # them and every OTHER member's data reads as a false alarm.
            "shift_detector",
            # GA-3, 2026-08-23. Information gain is the WRONG
            # objective: VoI is scored in the control law's own value
            # units, so two models implying the same action are worth
            # nothing to separate. The gate FAILED on attribution and
            # the module is sound -- what is unproven is the empirical
            # claim, not the code, so it is mutated like anything else.
            "voi",
            # GA-5, 2026-08-23. RUL-grow(0.01) had the LOWEST logical
            # error rate in the entire study and completed 7.2% of
            # jobs. Optimising protection without a completion
            # constraint has a DEGENERATE OPTIMUM: a machine that
            # never answers, perfectly.
            "freeze_detector",
            # GA-1, 2026-08-23. A posterior over indistinguishable
            # models is a false-discrimination generator. Every unknown
            # here resolves to UNIDENTIFIABLE because a false SEPARABLE
            # makes the disjunction merge unconditional -- the machine
            # acts on a difference that is not there.
            "identifiability",
            # GA-2, 2026-08-23. The open-world floor: MODEL_STATUS
            # = UNKNOWN as a first-class signed output over mass
            # that CANNOT be renormalised away. No QEC stack has
            # an escape hatch -- every DEM-estimation paper
            # returns a best fit, however wrong the library is.
            "open_world",
            # GA-4a, 2026-08-23. The shield is what makes an
            # UNCERTIFIED policy admissible: it filters the action
            # set before the policy sees it, so a learned control
            # law can be arbitrarily wrong about value and still
            # cannot violate a threshold-theorem hypothesis. Every
            # adaptive gate in P3 is blocked behind it.
            "shield", "shield_check",
            # A REGISTRY GAP FOUND BY AN AUDIT (2026-08-22), not
            # by a red. Neither of these appeared in ANY target
            # list, so neither had ever been mutated. `core`
            # assembles the Fault-Tolerance Passport;
            # `device_dem.edges_from_pinned` is the VERIFIER's
            # entry point -- the function that exists so a
            # re-derivation is byte-stable on another machine.
            "core", "device_dem",
            "forgeries", "certifying_decoder", "moat_growth",
            "exact_lp", "exact_decode",
            "coset_oracle", "robustness",
            "bayes_coset",
            # Three Phase-1 modules carried tests but NO aimed mutant, while
            # the handoff asserted "every module has aimed mutants".
            # logical_cut is the worst of the three to leave unmutated: its
            # own docstring says a wrong `c` makes every downstream
            # certificate a proof about the wrong question, and nothing was
            # checking that its two-source agreement could still refuse.
            "logical_cut", "fixed_point_bridge", "decomposition",
            "coset_cert", "rehearsal", "beacon", "coset_matching",
            # Found by an audit of the registry against src/oneq: a
            # CHECKER module and the repair PRODUCER both had killing
            # tests but no aimed mutant.
            "cert_format",
            "passport", "isd", "gbb", "bz", "mitm", "satdist", "certify",
            "coverage", "ledger", "transversal", "survival", "lattice",
            "guard", "modelgate", "evidence", "admissibility",
            "matching_cert", "moat_dual", "qldpc_cert", "qldpc_check",
            "tjoin_exact",
            # The QX wave modules (2026-08-15). Same law that added the
            # Phase-1 three: a module whose tests have never been proven
            # able to fail is carrying vacuity risk in the exact place
            # the campaign can least afford it -- accept paths.
            "coset_lower_check", "coset_bnb_check", "logical_contrast",
            "robust_radius",
            # Organ V and the capability negotiation (2026-08-19).
            # Both arrived WITH their own aimed mutants rather than
            # acquiring them in a later registry audit, which is the
            # order those audits kept wishing for.
            "dem_cert", "capability", "capability_probe",
            # The enclosure tier (2026-08-22). It arrived WITH its aimed
            # mutants, which is the order every registry audit here has
            # wished for. Both bounds get one, because the tier is a
            # BRACKET and a defect that only widens it is invisible to a
            # suite that watches the verdict.
            "coset_enclosure", "coset_enclosure_check", "enclosure_forgery",
            # E7, ADDED 2026-09-10 BY THE ADVERSARIAL AUDIT. The module
            # carries the campaign's headline theorem and was 100% LINE
            # covered while being absent from this gate entirely -- so
            # "489/489 killed" never said anything about it. Line
            # coverage says a line RAN, not that anything would notice
            # if it were wrong.
            "coset_e7",
            "certified_selection",
            # The evidence algebra (2026-08-19): every mutant below
            # makes a bound look BETTER than its inputs support, which
            # is the only direction that ships an unearned claim.
            "evidence_algebra",
            # The DEM admission ladder (2026-08-19). Every mutant below
            # reports a HIGHER level than the rungs support, which is
            # what downstream composition reads.
            "dem_admission",
            # L0 ABFT (2026-08-19). The layer that covers the fault
            # class `H x_hat = s` provably cannot see, so both of the
            # section's named no-op constructions carry a mutant.
            "abft_l0",
            # The evidence FRONTIER (2026-08-19). Replaced the scalar
            # (level, beta) pair after an external review showed the
            # two could come from different legs.
            "evidence_frontier",
            # The WIRING (2026-08-19). Composes a receipt through the
            # frontier into the signed passport core, and refuses a
            # claim the evidence does not support.
            "soundness_budget",
            # THE JOIN (2026-08-22). `enclosure_budget` is where the
            # error budget meets the enclosure tier, so it is where a
            # bracket can silently stop being a bracket -- the one
            # failure neither module could have on its own.
            "enclosure_budget",
            # The exact per-shot error budget (2026-08-21).
            # `docs/HARDWARE_PREFLIGHT_GATES.md` adopts it as governing
            # and ends the section "IMPLEMENT + MUTATION-TEST before the
            # pilot" -- so it arrives WITH its mutants rather than
            # acquiring them in a later registry audit.
            "error_budget",
            # The published counterexamples to the dual form
            # (2026-08-21). Recovered from an unmerged 2026-08-17 draft
            # branch: the FIX shipped to main without the attack that
            # motivates it, so nothing would have noticed a regression.
            # A forgery arm that quietly stops being a forgery is the
            # failure mode, and it is what these mutants simulate.
            "dual_forgery")

_TOOL_SOURCES = (
                 # THE SCORER ITSELF, 2026-08-22. Every other
                 # mutant here is worth exactly what this
                 # mapping is worth: score a SURVIVOR as a
                 # kill and the gate reads 200/200 forever
                 # while the checks underneath rot. It is a
                 # separate module so the mutant's anchor is
                 # unique -- in `mutation_gate.py` the
                 # find-string would also appear in the
                 # registry entry that holds it.
                 "mutation_outcome",
                 # THE SANDBOX LOCK (2026-09-08). The guarantee that a
                 # second run never deletes the tree a first run is
                 # mutating; the last unmutated part of this tool.
                 "mutation_sandbox",
                 # THE WALL'S OWN REPORTER (2026-08-24). It told a
                 # reader that a `MemoryError: std::bad_alloc` traceback
                 # was "the verdict, not a crash report" -- a fact about
                 # free RAM presented as a finding about two distance
                 # closures. The classifier that separates a death from
                 # a decision is now the thing a mutant attacks.
                 "gate_all",
                 # THE LEDGER (2026-08-24). It derives every status in
                 # ROADMAP.md and had NO tests at all until the missing
                 # engineer-week was found in it. A tool that decides
                 # what is closed is the last one that should be
                 # unmutated.
                 "roadmap_gate",
                 "doc_claims_gate", "verify_closures",
                 "claims_provenance_gate",
                 # The REAL-shot lane (2026-08-21). Its verdict is that
                 # a planted wrong-DEM landed in the right cell and was
                 # not billed to the decoder, which is the claim the
                 # hardware spend turns on -- so the checks that produce
                 # that verdict carry aimed mutants, not just the module
                 # they read.
                 "error_budget_ledger",
                 # D2 (2026-08-21). Its sigma is the only thing that
                 # makes alpha* a number rather than a shape, and two of
                 # the ways to get it wrong -- forgetting the scale
                 # normalisation, or scoring a same-coset competitor --
                 # leave a plausible run with a meaningless answer.
                 "certificate_robustness_corpus",
                 # The PUBLISHER of the DEM identity (2026-08-21). If it
                 # stops recording which models it graded, every
                 # downstream binding degrades to "unnamed" with no
                 # error and no refusal, and the A3 grade floats free of
                 # any model again.
                 "dem_cert_run",
                 # THE RELEASE-BRANCH GUARD, 2026-08-23. It is the only
                 # check in this repository standing in front of an
                 # IRREVERSIBLE act: refs/heads/master is the tagged
                 # orphan v1.0.0 evidence release, and a source push
                 # aimed at it -- which happened -- would overwrite a
                 # published artifact. Every other gate protects a claim
                 # we can remake by re-running it. This one protects
                 # bytes that cannot be recovered, so a silent survivor
                 # here costs more than a survivor anywhere else.
                 "push_guard")

# Experiment code is source too: G4b's herald guard defends a number that was
# misread for two sessions, so it is held to the same standard as the engine.
_EXPERIMENT_SOURCES = {
                       # GA-3's REPRODUCTION CHECK (2026-08-24). The gate
                       # RAN AND FAILED and the failure is the evidence;
                       # it was pinned rather than recomputed because the
                       # row was priced at 30 CPU-h and it runs in 45
                       # seconds. The comparison that decides
                       # "reproduces" is the only thing standing between
                       # a recorded failure and silent drift in it.
                       "ga3_voi_gate":
                       ("experiments", "voi", "ga3_voi_gate.py"),
                       # E7's COST PROFILE (2026-09-11). The artifact that
                       # decides whether E7 can be integrated, and whose
                       # prose drifted 1.6x from the number beside it. The
                       # mutable part is the note BUILDER, so the producer
                       # is gated without regenerating the artifact.
                       "e7_cost_profile":
                       ("experiments", "enclosure", "e7_cost_profile.py"),
                       # The LIVE lane's observable (2026-08-21). Its
                       # cut names which logical the run measured, and
                       # the first version named the wrong boundary --
                       # a silent error that would have reported a
                       # logical error rate for a different observable
                       # than the one the device prepared.
                       "rep_code_cert":
                       ("experiments", "ibm_live", "rep_code_cert.py"),
                       "erasure_decoder":
                       ("experiments", "g4_hardware", "erasure_decoder.py"),
                       "ve_reweighted":
                       ("experiments", "g4_hardware", "ve_reweighted.py"),
                       "h1_reweighted":
                       ("experiments", "g4_hardware", "h1_reweighted.py"),
                       # A5's certification is what makes the heavy-hex
                       # overhead ratio mean anything, and A6's
                       # outside-the-rowspace check is what stops
                       # "symmetry helps" from silently becoming "a
                       # redundant row helps". Both defend a headline
                       # number -- the same reason G4b's herald guard is
                       # on this list.
                       "heavyhex_transpiler":
                       ("experiments", "m0_prior_art",
                        "qldpc_heavyhex_transpiler.py"),
                       "symmetry_decoding":
                       ("experiments", "m0_prior_art",
                        "qldpc_symmetry_decoding.py"),
                       # L2' is the layer that closes C2's measured hole,
                       # and its GF(2) solver already shipped wrong once.
                       # A solver that returns a plausible-but-wrong f
                       # yields a detector that silently misses faults --
                       # indistinguishable, from the outside, from clean
                       # hardware.
                       # GA-11 hardware IQ, 2026-08-26. It lives in
                       # experiments/, so a BARE name would resolve to
                       # src/oneq/ and point at nothing -- four mutants
                       # aimed at a missing file, every one of them a
                       # SKIP rather than a kill.
                       "ga11_hardware_iq":
                       ("experiments", "way", "ga11_hardware_iq.py"),
                       # The DEVICE LAMBDA (2026-08-30), and the same
                       # lesson twice in one session: it shipped with no
                       # test and no mutant days after `ga11_hardware_iq`
                       # was written up for exactly that. Its scorer
                       # ALSO sat inside the submission path, so the
                       # only way to execute it was to spend the
                       # owner's QPU budget -- it reached production
                       # unrun, and the bound being quoted from it was
                       # hand arithmetic. Both are pure functions now,
                       # and pure functions get mutants.
                       "hardware_lambda":
                       ("experiments", "ibm_live", "hardware_lambda.py"),
                                              "l2prime":
                       ("experiments", "g3_abft",
                        "c2_l2prime_class_attestation.py")}


def _targets(root: pathlib.Path) -> dict[str, pathlib.Path]:
    # TOOLS ARE MUTATED TOO. A gate is source like any other, and the one that
    # polices prose is exactly the kind that can rot into a no-op unnoticed --
    # it passes loudest when it finds nothing.
    t = {name: root / "src" / "oneq" / f"{name}.py" for name in _SOURCES}
    t.update({name: root / "tools" / f"{name}.py" for name in _TOOL_SOURCES})
    t.update({name: root.joinpath(*parts)
              for name, parts in _EXPERIMENT_SOURCES.items()})
    return t


TARGETS = _targets(ROOT)


# THE SANDBOX LOCK LIVES IN ITS OWN MODULE (2026-09-08) so it can be
# mutated: an anchor inside this file would also match the registry
# entry that holds it. Names are re-exported unchanged for the tests.
from mutation_sandbox import (SandboxBusy, _claim_sandbox,        # noqa: E402,F401
                              _pid_alive, _release_sandbox)


def _sandbox() -> pathlib.Path:
    """A disposable copy of the tree to mutate, so the live tree never breaks.

    THE ROOT FIX, replacing a wait. This gate writes DELIBERATELY BROKEN source
    to prove the suite can catch it. Doing that in the working tree meant any
    process STARTING inside the window -- a queued retry, a resumed class --
    could import a mutant and emit a 'closure' from a knowingly broken engine.
    The guard against that was a refusal to run at all while certificate work
    was live, which is correct and also meant the gate went unrun for as long as
    the campaign lasted: the exact stretch when the engine was changing most.

    Mutating a COPY removes the hazard instead of scheduling around it. No live
    process can import a path that no live process knows about, so the gate runs
    concurrently with proofs and the interlock becomes a fallback rather than
    the mechanism.
    """
    import shutil
    import stat
    base = os.environ.get("ONEQ_MUTATION_SANDBOX")
    root = pathlib.Path(base) if base else (
        pathlib.Path(os.environ.get("TEMP", "/tmp")) / "oneq_mutation_sandbox")
    _claim_sandbox(root)
    if root.exists():
        # ignore_errors=True HID A POISONED SANDBOX. Git object files are
        # read-only on Windows, so removing a sandbox that contains a nested
        # repository fails -- silently, under ignore_errors -- and leaves a
        # partial tree behind. The next run then dies mid-copy with a
        # permission error that looks like a filesystem problem and is really
        # last run's litter. Clear the read-only bit and delete for real; if it
        # still will not go, say so instead of building on the remains.
        def _force(func, path, _exc):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        # `onexc=` IS PYTHON 3.12+, AND THIS PROJECT DECLARES >=3.11.
        # On 3.11 the call raised TypeError before a single mutant ran --
        # so the gate was unrunnable on its own minimum supported
        # interpreter, and the only symptom was a traceback out of
        # cleanup code that reads like a filesystem problem. `onerror=`
        # is the 3.11 spelling and is deprecated rather than removed in
        # 3.12, so it works on both; the version test picks the
        # non-deprecated one where it exists.
        if sys.version_info >= (3, 12):
            shutil.rmtree(root, onexc=_force)
        else:
            shutil.rmtree(root, onerror=_force)
    # COPY THE TREE, DO NOT GUESS WHICH PARTS THE TESTS READ.
    #
    # Three separate baseline failures came from an allow-list that was one
    # directory short: first `experiments/` (a test imported exposure_atlas),
    # then `evidence/` (a test reads the independent-reproduction artifact).
    # Each time the gate printed "baseline: FAIL" and ran ZERO mutants -- the
    # exact silent-stop failure it exists to catch, caused by its own
    # isolation. An allow-list that must be edited whenever a test is added is
    # a latent outage; a deny-list of heavy, non-imported things is not.
    root.mkdir(parents=True, exist_ok=True)
    # `release/` IS COPIED (11 MB). It was excluded as bulk, and that made the
    # sandbox baseline FAIL: the doc-claims test derives its ground truth from
    # the release bundle, so without it the counts differ and a green tree
    # looks red inside the gate. Same failure the comment above describes --
    # an exclusion that is one directory short -- recurring for a directory
    # excluded on purpose. 11 MB is not bulk; `challenge/` (300 MB of forgery
    # copies) and `evidence/lrat_proofs/` (17 GB) are, and they are not read by
    # any test.
    skip = {".git", "__pycache__", ".pytest_cache", ".venv",
            "node_modules", "mut_sandbox", "challenge"}
    for item in ROOT.iterdir():
        if item.name in skip:
            continue
        dst = root / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(
                "__pycache__", "*.log", "*.lrat", "*.cnf", "*.pyc",
                "lrat_proofs", ".git"))
        else:
            shutil.copy2(item, dst)
    return root


@dataclass(frozen=True)
class Mutant:
    """One simulated defect, and the test that must catch it.

    *** AIMING `must_fail`: A MUTANT THAT MAKES A PREDICATE CONSTANT CAN
    ONLY BE CAUGHT BY A TEST ASSERTING THE OTHER VALUE. ***

    The obvious killer is usually the test whose NAME matches the
    mutant's subject, and for a predicate that is usually the wrong one.
    Measured, 2026-08-23: `M258` pinned `raw_unknown_max` at the floor so
    `is_dark()` returned True for EVERYTHING, and it was aimed at
    `..._is_DARK`, which asserts True and therefore still passed. The
    mutant SURVIVED. The killer had to be the `..._is_not_dark` test --
    the only direction the mutation can break.

    The companion trap is an assertion weaker than its own name: `M255`
    deleted the bootstrap resampling, making the lower bound EQUAL the
    point estimate, and a test named `..._is_BELOW_...` asserting `<=`
    stayed green. Both survivors were found by this gate rather than by
    review, which is the argument for the gate; both are the same
    underlying question -- **can this test observe this change at all?**

    Before registering: state what the mutated code now returns, then
    name a test that asserts something ELSE.
    """
    id: str
    why: str                 # the defect this simulates
    find: str
    replace: str
    must_fail: str           # the test that MUST catch it
    target: str = "passport" # which source file to mutate


MUTANTS: tuple[Mutant, ...] = (
    # ---- GA-3: the value-of-information selector ----
    Mutant(
        # THE MASK-FOLD'S CROSS TERM IS WRONG, SO FOLDED MECHANISMS MISPRICE
        "M506-e7-fold-cross-term",
        "the mask-fold uses a0*a where it must use a0*b, so any two "
        "mechanisms sharing a mask combine to the wrong odds",
        '            folded[mask] = (a0 * b + b0 * a, b0 * b + a0 * a)',
        '            folded[mask] = (a0 * a + b0 * b, b0 * b + a0 * a)',
        "test_the_FOLD_MATCHES_THE_UNFOLDED_TRANSFER_at_HIGH_MULTIPLICITY",
        target="coset_e7",
    ),
    Mutant(
        # THE FOLD KEEPS ONLY THE LAST MECHANISM PER MASK, DROPPING THE REST
        "M507-e7-fold-drops-all-but-last",
        "mechanisms sharing a mask overwrite instead of folding, so all but "
        "the last are silently dropped from the transfer",
        "        if mask in folded:\n"
        "            a0, b0 = folded[mask]\n"
        "            folded[mask] = (a0 * b + b0 * a, b0 * b + a0 * a)\n"
        "        else:\n"
        "            folded[mask] = (a, b)",
        "        folded[mask] = (a, b)",
        "test_the_FOLD_MATCHES_THE_UNFOLDED_TRANSFER_at_HIGH_MULTIPLICITY",
        target="coset_e7",
    ),
    Mutant(
        # THE WIDTH CAP STOPS REFUSING, SO AN OVER-WIDE COMPONENT IS ATTEMPTED
        "M508-e7-width-cap-inert",
        "component_mass stops refusing over-wide components, so a 2**n "
        "transfer is attempted instead of falling back to E6",
        '    if n > MAX_COMPONENT:',
        '    if False:',
        "test_the_WIDTH_CAP_still_REFUSES_rather_than_approximating",
        target="coset_e7",
    ),
    Mutant(
        # THE DIVISIBILITY ASSERTION GOES AWAY AND A FLOOR UNDER-PRICES
        "M509-e7-divisibility-unchecked",
        "price() stops checking that den divides phat_t, so the integer "
        "division silently FLOORS -- an UNDER-price, the one direction that "
        "breaks soundness",
        # Anchored on the two lines ABOVE it, because `if phat_t % den:`
        # occurs in BOTH price() and e6_price(): the registry's own guard
        # test caught the ambiguity, which would have mutated the first
        # site silently and proved nothing about the second.
        '        detail.append({"detectors": comp, "mechanisms": len(inner),\n'
        '                       "odd_mass": S, "total": A})\n'
        '    if phat_t % den:',
        '        detail.append({"detectors": comp, "mechanisms": len(inner),\n'
        '                       "odd_mass": S, "total": A})\n'
        '    if False:',
        # *** THIS NAMED THE ORACLE TEST AND THE MUTANT SURVIVED THE GATE. ***
        # The guard is UNREACHABLE on correct input by construction: `den` is
        # a product of (b + a) over a SUBSET of the remaining mechanisms and
        # `phat_t` is the product over ALL of them, so it always divides.
        # Every oracle model in the world passes with the check deleted. It is
        # a SAFETY NET for a bug elsewhere -- and a net is tested by throwing
        # something into it, which here is a contract-violating `phat_t`,
        # exactly what a caller passing the wrong normalisation supplies.
        "test_a_PHAT_NOT_DIVISIBLE_BY_DEN_IS_REFUSED_NOT_FLOORED",
        target="coset_e7",
    ),
    Mutant(
        # THE INDEXED PARTITION NEVER MERGES, SO EVERY COMPONENT IS A SINGLETON
        "M510-e7-partition-never-merges",
        "components_via_index examines no candidate mechanisms, so nothing "
        "ever merges and E7 silently degrades to one detector per component",
        '        for i in idx.get(c, ()):',
        '        for i in ():',
        "test_the_INDEXED_PARTITION_IS_IDENTICAL_to_the_scanning_one",
        target="coset_e7",
    ),
    Mutant(
        # THE INDEX RETURNS EVERY MECHANISM INSTEAD OF THE TOUCHING ONES
        "M511-e7-index-returns-everything",
        "touching_via_index ignores the component and returns the whole "
        "remaining list, so the transfer runs over mechanisms it must not",
        '    return [dets_and_odds[i] for i in sorted(ids)]',
        '    return list(dets_and_odds)',
        "test_the_INDEX_RETURNS_THE_SAME_LIST_ELEMENT_FOR_ELEMENT",
        target="coset_e7",
    ),
    Mutant(
        # A PERCENTAGE IN THE PROSE GOES BACK TO BEING TYPED BY HAND.
        # The defect this mutant restores is not hypothetical: the shipped
        # note read "the COMPONENT key serves ~37% within one shot" beside a
        # field computing 23.6%, overstating the only memo E7 integration
        # has left by 1.6x, and the number was quoted onward as measured.
        # The old assertion (`"NOT A CONSTANT" in note`) checked that a
        # WARNING was present, never that the numbers agreed.
        "M512-e7-profile-note-retypes-its-percentage",
        "the per-shot cache note hard-codes the component hit rate instead "
        "of interpolating it, so prose can outlive the measurement again",
        '        f"COMPONENT instead serves {comp_rate * 100:.1f}% within one shot, "',  # noqa: E501
        '        "COMPONENT instead serves ~37% within one shot, "',
        "test_the_NOTE_BUILDER_QUOTES_ITS_ARGUMENTS_and_not_a_frozen_number",
        target="e7_cost_profile",
    ),
    # ------------------------------------------------------------------
    # THE EXACT PATH'S OWN RULES (2026-09-12, PAPER2 pre-disclosure audit).
    # The checker already carried seven source mutants -- sign rule, float
    # dual feasibility, syndrome recomputation on all three paths, redundant
    # parity on both -- and none touched the four rules that only the EXACT
    # path has: facet parity in the exact scan, the safe-dual slack, the
    # duplicate-facet refusal, and the lattice acceptance rule. Those four
    # are what a stranger's exact receipt rests on, so each gets a mutant.
    Mutant(
        # THE EXACT SCAN STOPS CHECKING A FACET'S PARITY. A "facet" with the
        # RIGHT parity is not a face of the relaxation at all; accepting it
        # lets a certificate assert weight the true polytope never charges.
        # The killer asserts the REASON names parity, so a refusal for the
        # wrong reason (the bound happening to fall short) does not pass.
        "M513-qldpc-exact-facet-parity-unchecked",
        "the exact facet scan accepts a right-parity subset as a facet",
        '        if len(Fs) % 2 == par:\n'
        '            return None, 0, (f"facet ({jname},{sorted(Fs)}) has the RIGHT "',
        '        if False:\n'
        '            return None, 0, (f"facet ({jname},{sorted(Fs)}) has the RIGHT "',
        "test_wrong_parity_facet_refused",
        target="qldpc_check",
    ),
    Mutant(
        # THE SAFE-DUAL SLACK IS DELETED. z_i = max(0, t_i - w_i) is the
        # exact path's entire wall against an inflated multiplier: with it,
        # any y >= 0 yields a valid bound; without it the bound is y.b alone
        # and an inflated dual reports L > U -- the killer checks the bound
        # the checker REPORTS stays below the weight, not just the verdict.
        "M514-qldpc-exact-safe-dual-slack-deleted",
        "the exact bound omits the derived slacks, so y.b is reported raw",
        '        if slack > 0:\n'
        '            z_total += slack',
        '        if False:\n'
        '            z_total += slack',
        "test_inflated_dual_stays_sound",
        target="qldpc_check",
    ),
    Mutant(
        # A FACET LISTED TWICE COUNTS TWICE. Two half-multipliers on one facet
        # then sum to a full bound and a certificate that names nothing new
        # certifies by repetition.
        "M517-qldpc-exact-duplicate-facet-counted-twice",
        "the exact scan no longer refuses a facet listed twice",
        '        if key in seen:\n'
        '            return None, 0, f"facet ({jname},{sorted(Fs)}) listed twice"',
        '        if False:\n'
        '            return None, 0, f"facet ({jname},{sorted(Fs)}) listed twice"',
        "test_mutant_8_duplicate_facet_refused_exact",
        target="qldpc_check",
    ),
    Mutant(
        # THE LATTICE RULE IS LOOSENED BY ONE LEVEL. L > U - 1/D is exact:
        # every feasible objective sits on the lattice, so crossing the last
        # level below U pins U. Accepting L > U - 2/D admits a candidate one
        # lattice step heavier than the optimum -- a false OPTIMAL on
        # non-integer weights, which the killer builds in (1/3)Z.
        "M518-qldpc-exact-lattice-rule-loosened",
        "the exact verdict accepts a bound two lattice levels below U",
        '    if L > U - step or L == U:',
        '    if L > U - 2 * step or L == U:',
        "test_lattice_rule_certifies_across_float_dust",
        target="qldpc_check",
    ),
    Mutant(
        # THE GENERATOR-SET HONESTY CLAUSE STOPS REFUSING POST-PROCESSING RELABELLINGS
        "M240-a-relabelling-can-buy-information",
        "the generator-set honesty clause stops refusing post-processing relabellings",
        '    if not c.changes_physical and (c.evi > 0.0 or c.d_p_vcc > 0.0):',
        '    if False:',
        "test_a_relabelling_claiming_benefit_is_REFUSED_not_scored_low",
        target="voi",
    ),
    Mutant(
        # SELECT STOPS FILTERING ON ADMISSION, SO A RELABELLING WINS A WEAK FIELD
        "M241-a-refused-candidate-can-still-be-selected",
        "select stops filtering on admission, so a relabelling wins a weak field",
        '    ok = [s for s in scored if s.admitted]',
        '    ok = scored',
        "test_a_relabelling_cannot_win_when_the_honest_field_is_empty",
        target="voi",
    ),
    Mutant(
        # A ZERO-COST CANDIDATE IS PRICED INSTEAD OF REFUSED AS UNPRICED
        "M242-a-free-action-scores-infinity",
        "a zero-cost candidate is priced instead of refused as unpriced",
        '    if den <= 0.0:',
        '    if False:',
        "test_a_FREE_action_is_REFUSED_as_unpriced",
        target="voi",
    ),
    Mutant(
        # EVI IS COMPUTED WITH NO ALPHA-VECTORS, SO IT MEASURES ENTROPY NOT VALUE
        "M243-EVI-falls-back-to-nats-without-a-value-function",
        "EVI is computed with no alpha-vectors, so it measures entropy not value",
        '    if not alpha_vectors:',
        '    if False:',
        "test_EVI_without_alpha_vectors_is_REFUSED",
        target="voi",
    ),
    # ---- The release-branch guard: the only check in front of an
    # ---- IRREVERSIBLE act. Each of these four mutants leaves a guard
    # ---- that still PRINTS and still exits 0 on every ordinary push,
    # ---- and would be noticed only by the release it failed to save.
    Mutant(
        "M244-the-protected-table-is-empty",
        "no ref is protected, so the guard allows every push and still "
        "reports success on all of them",
        '    "refs/heads/master": (',
        '    "refs/heads/DISABLED": (',
        "test_source_history_aimed_at_the_release_ref_is_REFUSED",
        target="push_guard",
    ),
    Mutant(
        "M245-a-clobbering-push-is-allowed",
        "the ancestry test is inverted, so exactly the disjoint-history "
        "overwrite this exists to stop becomes the allowed case",
        '        if not is_ancestor(remote_sha, local_sha):',
        '        if is_ancestor(remote_sha, local_sha):',
        "test_source_history_aimed_at_the_release_ref_is_REFUSED",
        target="push_guard",
    ),
    Mutant(
        "M246-deleting-a-protected-ref-is-allowed",
        "the deletion branch falls through to the ancestry test, which a "
        "zero sha passes vacuously",
        '        if set(local_sha) == {"0"}:',
        '        if False:',
        "test_DELETING_the_release_ref_is_REFUSED",
        target="push_guard",
    ),
    Mutant(
        "M247-only-the-first-ref-line-is-judged",
        "a push carrying an innocent ref first is allowed wholesale -- the "
        "guard reports on a ref that was never the danger",
        '    for raw in lines:',
        '    for raw in list(lines)[:1]:',
        "test_every_ref_line_gets_its_own_verdict",
        target="push_guard",
    ),
    # ---- GA-0: the dwell law ----
    Mutant(
        "M286-the-break-point-loses-its-anchor",
        "gamma_binding stops dividing by the measured C3, so the exponent "
        "at which latency binds is computed against nothing and the "
        "robustness margin becomes arbitrary",
        '    return math.log(c_break / C3) / math.log(w / W3)',
        '    return math.log(c_break) / math.log(w / W3)',
        "test_the_break_point_is_far_below_any_plausible_exponent",
        target="dwell",
    ),
    Mutant(
        "M287-the-separation-is-measured-against-the-wrong-quantity",
        "separation is taken against rounds instead of seconds, so the "
        "syndrome cycle drops out and the orders-of-magnitude claim stops "
        "comparing two times",
        '    return math.log10(dwell(gamma, w).seconds / reaction_s)',
        '    return math.log10(dwell(gamma, w).rounds / reaction_s)',
        "test_the_stated_separation_is_slightly_optimistic_at_the_worst_corner",
        target="dwell",
    ),
    Mutant(
        "M288-an-ambiguous-printed-precision-is-guessed",
        "a trailing-zero integer stops being refused, so the gate invents "
        "a tolerance for '300' and reports a comparison whose bar was "
        "assumed rather than read",
        '    if float(published) == int(published) and int(published) % 10 == 0:',
        '    if False:',
        "test_a_trailing_zero_integer_is_REFUSED_not_guessed",
        target="dwell",
    ),
    Mutant(
        "M289-the-power-law-stops-being-anchored",
        "C(w) drops the measured anchor, so the curve no longer passes "
        "through C3 and the published table stops reproducing",
        '    return anchor_c * (w / anchor_w) ** gamma',
        '    return (w / anchor_w) ** gamma',
        # RE-AIMED 2026-08-24 AFTER IT SURVIVED. It was pointed at
        # `..._two_point_fit_reproduces_the_corpus`, which never calls
        # `c_at` -- `fit_two_points` computes gamma from two points and
        # the anchor is not on that path. Nothing in the suite asserted
        # an ABSOLUTE value of C, so dropping the anchor turned C(23) at
        # gamma=3.22 from 604 into 2603 with every test still green.
        "test_the_power_law_is_ANCHORED_at_the_measured_C3",
        target="dwell",
    ),
    # ---- GT-XR: the exchange-rate law ----
    Mutant(
        "M282-the-exchange-rate-loses-its-factor-of-4",
        "the denominator drops to 2 d^2 ln2, doubling every recommended "
        "synthesis budget -- the corpus's own worked 40x becomes 80x",
        '            / (4.0 * w.d * w.d * math.log(2.0)))',
        '            / (2.0 * w.d * w.d * math.log(2.0)))',
        "test_the_law_reproduces_the_corpus_worked_example",
        target="exchange_rate",
    ),
    Mutant(
        "M283-the-ratio-stops-depending-on-lnLambda",
        "the split becomes a constant instead of a TRAJECTORY quantity, "
        "which is the entire claim of Sec 3.2.7",
        '    return ((w.rot_fraction * w.c_t * math.log(w.lam))',
        '    return ((w.rot_fraction * w.c_t * 1.0)',
        "test_the_ratio_scales_as_ln_lambda",
        target="exchange_rate",
    ),
    Mutant(
        "M284-the-exact-law-collapses-to-the-asymptotic",
        "the 8d+2 correction is dropped, so the exact form stops "
        "differing from the asymptotic and the numerical calibration can "
        "no longer distinguish them",
        '    denom = 2.0 * math.log(2.0) * (6.0 * w.d * w.d + 8.0 * w.d + 2.0)',
        '    denom = 2.0 * math.log(2.0) * (6.0 * w.d * w.d)',
        "test_the_corpus_form_is_an_ASYMPTOTIC_and_is_high",
        target="exchange_rate",
    ),
    Mutant(
        "M285-a-non-suppressing-code-is-priced-anyway",
        "Lambda <= 1 stops being refused, so an exchange rate is quoted "
        "for a code that does not suppress and has no exchange to make",
        '        if self.lam <= 1.0:',
        '        if False:',
        "test_a_non_suppressing_code_is_REFUSED",
        target="exchange_rate",
    ),
    # ---- GA-14: wrapping the RL calibrator ----
    Mutant(
        "M278-two-inert-arms-are-scored-as-additive",
        "the non-vacuity floor is removed, so gA=gB=0 satisfies "
        "'additive to within 30%' EXACTLY and two controls that did "
        "nothing are reported as composing perfectly",
        '    if gain_a < min_gain or gain_b < min_gain:',
        '    if False:',
        "test_two_inert_arms_are_REFUSED_not_scored_as_additive",
        target="rl_wrap",
    ),
    Mutant(
        "M279-the-bar-is-reported-as-discriminating",
        "the gate stops saying that its +/-30% band contains the "
        "multiplicative prediction, so a pass that separates neither "
        "composition law is quoted as if it established additivity",
        '    discriminates = not (band_lo <= mul <= band_hi)',
        '    discriminates = True',
        "test_the_additive_band_does_NOT_discriminate_at_the_measured_gains",
        target="rl_wrap",
    ),
    Mutant(
        "M280-the-verdict-is-the-first-band-tested-not-the-nearest",
        "ADDITIVE is printed for a measurement closer to the "
        "multiplicative prediction and below BOTH -- the label then "
        "contradicts the numbers beside it",
        '    verdict = ("ADDITIVE" if abs(r_add) <= abs(r_mul)',
        '    verdict = ("ADDITIVE" if abs(r_add) <= 1e9',
        "test_the_verdict_is_the_NEAREST_reference_not_the_first_one_tested",
        target="rl_wrap",
    ),
    Mutant(
        "M281-the-multiplicative-reference-collapses-to-additive",
        "the interaction term is dropped, so independent non-interfering "
        "composition is scored as a shortfall it does not have",
        '    mul = gain_a + gain_b - gain_a * gain_b',
        '    mul = gain_a + gain_b',
        "test_independent_controls_land_on_the_MULTIPLICATIVE_prediction",
        target="rl_wrap",
    ),
    # ---- GA-4b: the certified policy gap ----
    Mutant(
        "M268-a-linear-policy-may-lose-to-static",
        "the Delta > 1 guard is removed, so a class can be reported as "
        "worse than the constant policies it contains -- exactly the "
        "1.0595 a live run produced",
        '    if j_linear + 1e-9 < j_static:',
        '    if False:',
        "test_a_linear_policy_losing_to_STATIC_is_REFUSED",
        target="policy_gap",
    ),
    Mutant(
        "M269-a-degenerate-denominator-still-yields-a-number",
        "Delta is computed where adaptation is worth nothing, so noise "
        "over noise can grant the kernel actuation authority",
        '    if headroom < min_headroom:',
        '    if False:',
        "test_a_degenerate_denominator_is_REFUSED",
        target="policy_gap",
    ),
    Mutant(
        "M270-a-truncated-alpha-set-is-silently-accepted",
        "capping instead of refusing makes J(alpha) a LOWER bound, which "
        "shrinks Delta and grants authority that was not earned",
        '        if len(gamma) > max_alphas:',
        '        if False:',
        "test_a_truncated_alpha_set_is_REFUSED_not_capped",
        target="policy_gap",
    ),
    Mutant(
        "M271-the-alpha-prune-drops-a-uniquely-optimal-vector",
        "pointwise domination becomes strict inequality on one coordinate, "
        "so vectors optimal in some belief region are discarded and "
        "J(alpha) falls",
        '        if any(all(a[i] <= b[i] + 1e-12 for i in range(len(a)))',
        '        if any(any(a[i] <= b[i] + 1e-12 for i in range(len(a)))',
        "test_alpha_vectors_agree_with_brute_force_belief_tree_search",
        target="policy_gap",
    ),
    # *** NO MUTANT IS AIMED AT THE CONSTANT-POLICY SEEDING, AND HERE IS
    # *** WHY. M272 tried, and SURVIVED: removing the seeding changes no
    # observable behaviour, because the theta grid ALWAYS contains 0
    # ("sigma >= 0" is always true, i.e. act always) and some (w, theta=1)
    # pair always expresses "never act" on a reachable belief set. The
    # seeding is defence-in-depth against a future grid parameterisation
    # that drops those corners -- it is not dead code, but it IS
    # currently unkillable.
    #
    # Registering a mutant for it anyway would have been a coverage claim
    # that is not real: a mutant that can never die reports the same
    # score whether the line works or not, which is the shape this whole
    # registry exists to catch. It was found by the gate, not by review.
    Mutant(
        "M273-TP2-reports-a-pass-by-ignoring-negative-minors",
        "violations are never recorded, so the structural precondition of "
        "the threshold theorem is asserted rather than decided",
        '            if minor < -tol:',
        '            if False:',
        "test_tp2_reports_WHERE_it_fails_not_merely_that_it_did",
        target="policy_gap",
    ),
    Mutant(
        "M274-a-demotion-is-raised-instead-of-returned",
        "Delta > 0.05 stops being a PASS-with-demotion and becomes an "
        "error, which is the asymmetry the corpus calls the honest part",
        '    granted = d <= bar',
        '    granted = True',
        "test_a_large_gap_DEMOTES_and_that_is_a_PASS_not_an_error",
        target="policy_gap",
    ),
    Mutant(
        "M275-every-verdict-is-reported-as-certified",
        "a DEMOTION stops being labelled provisional, so a failed search "
        "is dressed as a measured gap -- Delta is an UPPER bound, so only "
        "a grant actually follows from it",
        '        "CERTIFIED" if granted else "PROVISIONAL",',
        '        "CERTIFIED",',
        "test_a_grant_is_CERTIFIED_and_a_demotion_is_PROVISIONAL",
        target="policy_gap",
    ),
    Mutant(
        "M276-the-class-is-sampled-instead-of-enumerated",
        "pair enumeration returns only the constants, so the search falls "
        "back to a lattice covering 29% of the class -- which INFLATES "
        "Delta and demotes a kernel that may deserve authority",
        '    for i in range(n):',
        '    for i in range(0):',
        "test_pair_enumeration_finds_more_behaviours_than_a_grid",
        target="policy_gap",
    ),
    Mutant(
        "M277-the-winner-is-never-refined-on-the-scored-beliefs",
        "the final refinement is skipped, so a member tuned on a SUBSET "
        "is reported against the full set without being a local optimum "
        "of the objective it is scored on",
        '    step = 1.0 / max(2, grid)',
        '    step = 0.0',
        "test_the_union_result_is_a_LOCAL_OPTIMUM_on_the_scored_beliefs",
        target="policy_gap",
    ),
    # ---- GA-13: the checkability operating point ----
    Mutant(
        "M260-an-uncheckable-class-still-gets-an-operating-point",
        "C4-C6 stop being refused, so the gate reports an advantage for a "
        "workload with no check and no retry loop to move",
        '    if workload_class in UNCHECKABLE:',
        '    if False:',
        "test_an_UNCHECKABLE_class_is_REFUSED",
        target="checkability",
    ),
    Mutant(
        "M261-false-accept-exposure-ignores-the-shot-failure-rate",
        "delivered error stops scaling with (1-q), so running at q* costs "
        "nothing in exposure and the whole trade-off vanishes",
        '    r = (1.0 - q) * self.eps_fa / denom',
        '    r = self.eps_fa / denom',
        "test_running_at_q_star_multiplies_false_accept_exposure",
        target="checkability",
    ),
    Mutant(
        "M262-k-repetitions-stop-suppressing-the-error",
        "k agreeing acceptances are scored as one, so repetition costs "
        "shots and buys nothing",
        '        return r ** k',
        '        return r',
        "test_k_agreeing_acceptances_suppress_the_delivered_error",
        target="checkability",
    ),
    Mutant(
        "M263-a-false-accept-does-not-end-the-retry-loop",
        "shots are charged on q instead of on acceptance, overstating the "
        "low-q arm's shot count -- an error in OUR OWN FAVOUR",
        '    p_accept = q + (1.0 - q) * check.eps_fa',
        '    p_accept = q',
        "test_the_retry_loop_is_charged_on_ACCEPTANCE_not_on_success",
        target="checkability",
    ),
    Mutant(
        "M264-an-unreachable-target-returns-the-least-bad-point",
        "the refusal becomes a best-effort answer, so two arms delivering "
        "DIFFERENT correctness get compared as if they were equal",
        '    if best is None:',
        '    if False:',
        "test_an_unreachable_target_is_REFUSED_not_approximated",
        target="checkability",
    ),
    Mutant(
        "M265-suppression-accepts-an-undeclared-source",
        "the number that decides the whole result no longer has to say "
        "where it came from",
        # ANCHOR DISAMBIGUATED 2026-08-24. This exact line appears TWICE
        # in `checkability.py` -- in `Suppression.__post_init__` and in
        # `Check.__post_init__` -- and the gate replaces the FIRST
        # occurrence only. It happened to be the intended one, so this
        # mutant was aimed correctly BY ORDINAL LUCK, and the `Check`
        # guard underneath was covered by nothing at all (now M296).
        '        if not self.source.strip():\n'
        '            raise CheckabilityRefusal(\n',
        '        if False:\n'
        '            raise CheckabilityRefusal(\n',
        "test_a_suppression_with_no_declared_source_is_REFUSED",
        target="checkability",
    ),
    Mutant(
        # THE TWIN THE ORDINAL ANCHOR HID. `Check` carries the same
        # provenance guard as `Suppression`, and until now deleting it
        # would have cost nothing -- a guard on one of two paths is a
        # coincidence, not a policy.
        "M296-a-check-accepts-an-undeclared-source",
        "the CHECK's provenance guard is removed, so a false-alarm rate "
        "with no stated origin sets the operating point",
        '        if not self.source.strip():\n'
        '            raise CheckabilityRefusal("a check with no declared '
        'source")',
        '        if False:\n'
        '            raise CheckabilityRefusal("a check with no declared '
        'source")',
        "test_a_check_with_no_declared_source_is_REFUSED",
        target="checkability",
    ),
    # ---- the no_test_ran outcome: the newest safety-critical logic in
    # ---- the tool that polices whether every other check can fail ----
    Mutant(
        "M266-an-absent-suite-maps-back-to-killed",
        "the NO_TEST_RAN arm is removed, so pytest collecting NOTHING is "
        "scored as a mutant killed by a test that does not exist",
        '    if suite_green is NO_TEST_RAN:',
        '    if False:',
        "test_the_recorded_outcomes_are_exactly_what_the_tally_accepts",
        target="mutation_outcome",
    ),
    Mutant(
        "M267-a-vanished-killer-leaves-the-denominator",
        "no_test_ran drops out of the total, so a mutant whose killer "
        "could not be found RAISES the score -- the shrinking-denominator "
        "shape the skip rule already carries a scar for",
        '    return killed + survived + skipped + timed_out + no_test_ran',
        '    return killed + survived + skipped + timed_out',
        # AIMED AT THE FUNCTION THE MUTANT EDITS. The first version named
        # a `counts` test for a mutant of `denominator` -- a neighbour
        # with a similar subject, which never calls the mutated line, and
        # the mutant SURVIVED.
        "test_a_skipped_or_timed_out_mutant_stays_in_the_denominator",
        target="mutation_outcome",
    ),
    # ---- GA-12: the composition bound. Each of these leaves a bound
    # ---- that is still POSITIVE, still ordered, and still passes the
    # ---- spec's own soundness leg -- while being wrong.
    Mutant(
        "M248-unknown-mass-is-renormalised-away",
        "the open-world branch is given the library's average accuracy "
        "instead of zero, which is exactly what GA-2 forbids",
        '    val = (e.p_complete_lb * (1.0 - e.unknown_mass) * worst',
        '    val = (e.p_complete_lb * 1.0 * worst',
        "test_unknown_mass_is_SUBTRACTED_not_renormalised_away",
        target="composition",
    ),
    Mutant(
        "M249-the-credible-set-is-averaged-not-worst-cased",
        "a posterior-weighted mean over models the run cannot separate: "
        "a false SEPARABLE arriving as arithmetic",
        '    worst = min(e.accuracy_lb[m] for m in e.credible)',
        '    worst = max(e.accuracy_lb[m] for m in e.credible)',
        "test_the_credible_set_is_taken_at_its_WORST_not_its_mean",
        target="composition",
    ),
    Mutant(
        "M250-detector-latency-is-free",
        "the lag charge is dropped, so the bound trusts a posterior that "
        "is stale exactly when the adversary made it stale",
        '           * (1.0 - e.lag_damage) * (1.0 - e.stress_damage))',
        '           * 1.0 * (1.0 - e.stress_damage))',
        "test_detector_latency_is_CHARGED",
        target="composition",
    ),
    Mutant(
        "M251-an-UNKNOWN-library-still-certifies",
        "MODEL_STATUS = UNKNOWN no longer collapses the bound: the "
        "machine certifies accuracy for a device it has just said it "
        "cannot model",
        '    if e.model_status_unknown:',
        '    if False:',
        "test_MODEL_STATUS_UNKNOWN_collapses_the_bound_to_zero",
        target="composition",
    ),
    Mutant(
        "M252-a-noisy-inversion-is-called-fatal",
        "the significance test drops to the point estimate, so sampling "
        "noise is reported as the fatal class and the gate cries wolf",
        '        return self.bound > hi',
        '        return self.bound > self.p_hat',
        "test_a_strict_violation_can_be_explained_by_noise",
        target="composition",
    ),
    Mutant(
        "M253-a-confound-is-pooled-instead-of-stratified",
        "per-stratum correlations are replaced by one pooled figure, "
        "which measures the nuisance factor rather than the tracking",
        '    rhos = [spearman([x for x, _ in s], [y for _, y in s])',
        '    rhos = [spearman([x for t in strata for x, _ in t],\n'
        '                     [y for t in strata for _, y in t])',
        "test_stratification_recovers_a_signal_a_CONFOUND_destroys",
        target="composition",
    ),
    Mutant(
        "M254-a-dead-stratum-hides-behind-the-median",
        "the worst stratum is reported as the median, so one stratum "
        "that tracks nothing is invisible",
        '    return rhos, median, min(rhos)',
        '    return rhos, median, median',
        "test_the_WORST_stratum_is_returned_so_a_median_cannot_hide_it",
        target="composition",
    ),
    Mutant(
        "M255-the-bootstrap-ignores-cluster-structure",
        "resamples individual observations, treating 8 certificates from "
        "one scenario as 8 independent samples of a device",
        '        sample = [clusters[rng.randrange(n)] for _ in range(n)]',
        '        sample = list(clusters)',
        "test_the_bootstrap_lower_bound_is_STRICTLY_below_the_point_estimate",
        target="composition",
    ),
    # ---- GA-12's complementary detector ----
    Mutant(
        "M256-the-null-is-scored-against-one-model",
        "the composite min becomes a max, so data any library member "
        "explains still reads as a shift -- the false alarms that "
        "inflated the calibrated threshold from 9.75 to 37.1",
        '            s = min(per_model)',
        '            s = max(per_model)',
        "test_data_from_ANY_library_member_does_not_fire",
        target="shift_detector",
    ),
    Mutant(
        "M257-an-uncalibrated-detector-speaks-anyway",
        "a default threshold is invented, so a detector with no measured "
        "false-alarm rate is charged against by the certified bound",
        '        if self.threshold is None:',
        '        if False:',
        "test_an_UNCALIBRATED_detector_refuses_to_speak",
        target="shift_detector",
    ),
    # ---- GA-2's darkness witness ----
    Mutant(
        "M258-the-darkness-witness-records-the-post-floor-mass",
        "records unknown mass AFTER the floor, where 'M_? lost' and "
        "'M_? never competed' are numerically identical",
        '        self.raw_unknown_max = max(self.raw_unknown_max,\n'
        '                                   post[UNKNOWN_MODEL])',
        '        self.raw_unknown_max = max(self.raw_unknown_max,\n'
        '                                   float(self.alpha_floor))',
        # *** AIMED AT THE DIRECTION THE MUTANT ACTUALLY BREAKS. ***
        # This mutant pins `raw_unknown_max` at the floor, so `is_dark()`
        # returns True for EVERYTHING. The test that asserts a dark
        # posterior IS dark therefore still passes -- it was the obvious
        # killer and it was the wrong one. Only the NOT-dark direction
        # can fail here, and the gate caught the misaim by letting the
        # mutant survive.
        "test_a_posterior_whose_unknown_DID_compete_is_not_dark",
        target="open_world",
    ),
    Mutant(
        "M259-the-darkness-flag-does-not-travel-with-the-verdict",
        "the verdict is issued with the flag defaulted to False, so a "
        "caller reads KNOWN and is never handed the reason it may mean "
        "nothing -- the invariant reverts to advice",
        '            leader_mass=self._p[lead], detector_dark=self.is_dark())',
        '            leader_mass=self._p[lead])',
        "test_the_darkness_flag_TRAVELS_with_the_verdict",
        target="open_world",
    ),
    # ---- GA-5: the freeze detector ----
    Mutant(
        # THE COMPLETION BAR STOPS BEING CHECKED, SO PROTECTING HARDER ALWAYS WINS
        "M237-a-frozen-policy-is-scored-as-a-win",
        "the completion bar stops being checked, so protecting harder always wins",
        '    if outcome.p_complete < p_min:',
        '    if False:',
        "test_RUL_grow_is_REJECTED_despite_winning_on_the_metric",
        target="freeze_detector",
    ),
    Mutant(
        # THE TOP-SCORING CANDIDATE IS RETURNED EVEN WHEN NOTHING IS A WIN
        "M238-best-win-returns-the-best-scorer-regardless",
        "the top-scoring candidate is returned even when nothing is a win",
        '    return wins[0] if wins else None',
        '    return rank(outcomes, baseline, p_min)[0]',
        "test_best_win_returns_None_rather_than_the_best_scorer",
        target="freeze_detector",
    ),
    Mutant(
        # A FROZEN POLICY PRESENTED AS A WIN PASSES THE AUDIT SILENTLY
        "M239-the-reported-win-audit-finds-nothing",
        "a frozen policy presented as a win passes the audit silently",
        '    return [v for v in verdicts if v.is_win and v.p_complete < p_min]',
        '    return []',
        "test_the_audit_catches_a_frozen_policy_presented_as_a_win",
        target="freeze_detector",
    ),
    # ---- GA-1: the identifiability certificate ----
    Mutant(
        # A PAIR WITH NO MEASURED ENTRY CERTIFIES AS SEPARABLE
        "M232-an-unmeasured-pair-is-declared-separable",
        "a pair with no measured entry certifies as SEPARABLE",
        '            if row is None:',
        '            if False:',
        "test_a_pair_with_no_measured_entry_is_UNIDENTIFIABLE",
        target="identifiability",
    ),
    Mutant(
        # A RATE NOT MARKED MEASURED IS ACCEPTED, SO A CORRELATION STATISTIC CAN BE ESTIMATED
        "M233-a-guessed-KL-rate-enters-the-table",
        "a rate not marked measured is accepted, so a correlation statistic can be estimated",
        '        if not r.measured:',
        '        if False:',
        "test_a_GUESSED_rate_is_REFUSED_at_the_table",
        target="identifiability",
    ),
    Mutant(
        # TWO MODELS PREDICTING THE SAME THING GET A FINITE T_SEP
        "M234-a-zero-separation-rate-becomes-finite",
        "two models predicting the same thing get a finite T_sep",
        '    if kl_rate <= 0.0:',
        '    if False:',
        "test_a_zero_KL_rate_gives_INFINITY_not_a_big_number",
        target="identifiability",
    ),
    Mutant(
        # INSEPARABLE MODELS STOP CONSTRAINING EACH OTHER -- THE FATAL CLASS
        "M235-the-disjunction-merge-becomes-unconditional",
        "inseparable models stop constraining each other -- THE fatal class",
        '        for m in members[1:]:\n            acc &= actions_by_model[m]',
        '        for m in members[1:]:\n            acc |= actions_by_model[m]',
        "test_inseparable_models_constrain_you_to_their_INTERSECTION",
        target="identifiability",
    ),
    Mutant(
        # SILENCE ABOUT A PAIR PROMISED SEPARABLE IS READ AS SUCCESS
        "M236-a-promised-pair-that-never-separated-passes-the-audit",
        "silence about a pair promised SEPARABLE is read as success",
        '        if got is None or got > slack * v.t_sep:',
        '        if got is not None and got > slack * v.t_sep:',
        "test_a_pair_that_never_separated_at_all_is_CAUGHT",
        target="identifiability",
    ),
    # ---- GA-2: the open-world floor ----
    Mutant(
        # THE POSTERIOR ERASES M_? LIKE ANY OTHER UNLIKELY COMPONENT
        "M228-the-open-world-floor-is-renormalised-away",
        "the posterior erases M_? like any other unlikely component",
        '        u = max(post[UNKNOWN_MODEL], float(self.alpha_floor))',
        '        u = post[UNKNOWN_MODEL]',
        "test_overwhelming_evidence_cannot_renormalise_the_floor_away",
        target="open_world",
    ),
    Mutant(
        # M_?'S MAX-ENTROPY LIKELIHOOD CAN BE SUPPLIED FROM OUTSIDE
        "M229-the-caller-may-tune-the-open-world-component",
        "M_?'s max-entropy likelihood can be supplied from outside",
        '        if UNKNOWN_MODEL in loglik:',
        '        if False:',
        "test_a_caller_cannot_supply_the_unknown_likelihood",
        target="open_world",
    ),
    Mutant(
        # THE LOG-M DEPENDENCE PRICE IS DROPPED ON CORRELATED DETECTORS
        "M230-BY-degrades-to-BH-under-dependence",
        "the log-m dependence price is dropped on correlated detectors",
        '    harmonic = sum(1.0 / i for i in range(1, m + 1))',
        '    harmonic = 1.0',
        "test_BY_is_strictly_more_conservative_than_BH",
        target="open_world",
    ),
    Mutant(
        # TAU BECOMES THE FROZEN QUANTILE SPLIT-CONFORMAL CANNOT JUSTIFY
        "M231-the-adaptive-threshold-stops-adapting",
        "tau becomes the frozen quantile split-conformal cannot justify",
        '        if gamma <= 0.0:',
        '        if False:',
        "test_a_zero_gamma_threshold_is_REFUSED",
        target="open_world",
    ),
    Mutant(
        # THE INDEPENDENT CHECKER STOPS CHECKING. GA-4a's criterion has
        # two clauses and this is the second: a leg removal must be
        # caught by a checker that shares no code with the shield. If
        # `authorized_soundly` ignores soundness disagreements, the
        # shield becomes its own auditor -- and a checker built from the
        # producer's own predicates agrees with the bug by construction.
        "M227-the-independent-checker-accepts-a-forged-verdict",
        "shield_check stops refusing a verdict its own re-derivation contradicts",
        '    return not any(d.kind in ("SOUNDNESS", "STRUCTURE")\n'
        '                   for d in verify(verdict, action, machine))',
        "    return True",
        "test_the_checker_reports_a_SOUNDNESS_disagreement_by_name",
        target="shield_check",
    ),
    # ---- GA-4a: the shield's eight legs, one mutant each ----
    Mutant(
        # THE AS2 LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # a circuit distance below the declaration is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M219-shield-as2-leg-removed",
        "the shield stops enforcing distance (AS2)",
        "def leg_as2(a: Action, m: Machine) -> LegResult:",
        'def leg_as2(a: Action, m: Machine) -> LegResult:\n    return LegResult("AS2", "PASS", "MUTANT: leg removed")',
        "test_the_AS2_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE LEAK LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # an operator outside the epoch algebra is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M220-shield-leak-leg-removed",
        "the shield stops enforcing containment (LEAK)",
        "def leg_leak(a: Action, m: Machine) -> LegResult:",
        'def leg_leak(a: Action, m: Machine) -> LegResult:\n    return LegResult("LEAK", "PASS", "MUTANT: leg removed")',
        "test_the_LEAK_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE DEM LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # a schedule with no characterised model is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M221-shield-dem-leg-removed",
        "the shield stops enforcing model existence (DEM)",
        "def leg_dem(a: Action, m: Machine) -> LegResult:",
        'def leg_dem(a: Action, m: Machine) -> LegResult:\n    return LegResult("DEM", "PASS", "MUTANT: leg removed")',
        "test_the_DEM_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE PROG LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # a run that is observably not finishing is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M222-shield-prog-leg-removed",
        "the shield stops enforcing the freeze detector (PROG)",
        "def leg_prog(a: Action, m: Machine) -> LegResult:",
        'def leg_prog(a: Action, m: Machine) -> LegResult:\n    return LegResult("PROG", "PASS", "MUTANT: leg removed")',
        "test_the_PROG_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE ID LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # an action inadmissible under a live model class is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M223-shield-id-leg-removed",
        "the shield stops enforcing identifiability (ID)",
        "def leg_id(a: Action, m: Machine) -> LegResult:",
        'def leg_id(a: Action, m: Machine) -> LegResult:\n    return LegResult("ID", "PASS", "MUTANT: leg removed")',
        "test_the_ID_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE PAR LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # an unprotected data qubit is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M224-shield-par-leg-removed",
        "the shield stops enforcing the set cover (PAR)",
        "def leg_par(a: Action, m: Machine) -> LegResult:",
        'def leg_par(a: Action, m: Machine) -> LegResult:\n    return LegResult("PAR", "PASS", "MUTANT: leg removed")',
        "test_the_PAR_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE ANC LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # an unmeetable ancilla demand is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M225-shield-anc-leg-removed",
        "the shield stops enforcing supply counting (ANC)",
        "def leg_anc(a: Action, m: Machine) -> LegResult:",
        'def leg_anc(a: Action, m: Machine) -> LegResult:\n    return LegResult("ANC", "PASS", "MUTANT: leg removed")',
        "test_the_ANC_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # THE DL LEG REMOVED. Returning PASS unconditionally is
        # what "this leg is not there" means for a bitmask AND --
        # a decoder slower than the syndrome stream is authorised. GA-4a requires each removal to be caught
        # BOTH by the shield's own negative control AND by an
        # independent checker that shares no code with it.
        "M226-shield-dl-leg-removed",
        "the shield stops enforcing the backlog bound (DL)",
        "def leg_dl(a: Action, m: Machine) -> LegResult:",
        'def leg_dl(a: Action, m: Machine) -> LegResult:\n    return LegResult("DL", "PASS", "MUTANT: leg removed")',
        "test_the_DL_leg_is_load_bearing",
        target="shield",
    ),
    Mutant(
        # *** A SKIP COUNTED AS A KILL, AND NOTHING DOWNSTREAM CAN SEE
        # IT. *** Every declared invariant stays satisfied: killed ==
        # total, `mutants_unkilled` == 0, the ratchet holds. A mutant
        # whose anchor drifted -- which a refactor is enough to cause --
        # is written down as PROVEN CAUGHT. The registry keeps reporting
        # a perfect score for checks it never ran.
        "M218-every-outcome-is-tallied-as-a-kill",
        "the tally credits a kill for a skipped or surviving mutant",
        "        tally[o] += 1",
        '        tally["killed"] += 1',
        "test_a_skip_is_never_tallied_as_a_kill",
        target="mutation_outcome",
    ),
    Mutant(
        # *** THE DENOMINATOR, WHICH HAS ALREADY SHRUNK ONCE. *** Drop the
        # non-kills from the total and a gate that ran 201 mutants and
        # proved 199 reports `199/199 killed` -- a perfect score that
        # covers less than it did yesterday. Coverage falls, the number
        # rises, and nothing downstream can tell. The rule was a comment
        # above one expression until 2026-08-22; a comment cannot be
        # watched to fail.
        "M217-a-skipped-mutant-leaves-the-denominator",
        "the score rises as coverage falls: non-kills drop out of the total",
        "    return killed + survived + skipped + timed_out",
        "    return killed",
        "test_a_skipped_or_timed_out_mutant_stays_in_the_denominator",
        target="mutation_outcome",
    ),
    Mutant(
        # *** THE GATE SCORING ITS OWN VERDICTS. *** `run_tests` returns
        # True when the suite PASSED, i.e. when the named killing test did
        # NOT catch the mutant -- so True means SURVIVED. Collapse that to
        # a constant "killed" and every survivor is written down as a
        # kill: the headline stays at 200/200 while the registry becomes
        # decoration and every gate downstream inherits the lie. Nothing
        # verified this until 2026-08-22, because the gate was the one
        # source it never mutated.
        "M216-a-surviving-mutant-is-scored-as-killed",
        "the gate reports a kill for a mutant its test did not catch",
        '    return "survived" if suite_green else "killed"',
        '    return "killed"',
        "test_a_surviving_mutant_is_never_scored_as_killed",
        target="mutation_outcome",
    ),
    Mutant(
        # A BARE TOTAL WEARING A HYPHEN. `doc_claims_gate` recomputes the
        # front page's counts, and every one of its single-value patterns
        # expected a SPACE before the noun -- so `the 178-mutant registry`
        # sat on README.md against an artifact recording 199, a drift of
        # 21, in the section headed "A gate that cannot fail is not a
        # gate". Measured and fixed 2026-08-22; this keeps the shape
        # covered.
        "M209-the-doc-gate-cannot-see-a-hyphenated-total",
        "a stale `N-mutant` headline passes the doc-claims gate",
        '    (re.compile(r"\\b(\\d[\\d,]*)-mutants?\\b", re.I), "mutants_total"),',
        '    (re.compile(r"THIS PATTERN MATCHES NOTHING"), "mutants_total"),',
        "test_a_hyphenated_bare_total_is_checked_too",
        target="doc_claims_gate",
    ),
    # ---- two targets that were DECLARED mutatable and never mutated ----
    Mutant(
        # A SYNDROME NAMING A VERTEX THE GRAPH DOES NOT HAVE. The
        # Edmonds-Johnson reduction needs every defect to be a node it
        # can run Dijkstra from; without this the decode proceeds against
        # a graph that does not contain the syndrome it is explaining.
        # Found by an audit: `exact_decode` was in _SOURCES with nothing
        # aimed at it, so the registry read as wider than it was.
        "M207-an-out-of-graph-syndrome-is-decoded-anyway",
        "exact_decode stops refusing a syndrome its graph cannot contain",
        '    if any(t not in adj for t in T):\n        return None, None, '
        '"syndrome names a vertex absent from the graph"',
        '    if False:\n        return None, None, '
        '"syndrome names a vertex absent from the graph"',
        "test_unknown_vertex_is_refused",
        target="exact_decode",
    ),
    Mutant(
        # A PSEUDOCODEWORD LAUNDERED INTO A REPAIR. The Feldman LP's
        # optimum can be half-integral -- the canonical LP-decoding
        # failure -- and rounding it gives a support that satisfies every
        # parity while costing more than the true optimum. The
        # integrality gate is the only thing between that and a returned
        # "repair". The pre-existing pseudocodeword test runs with
        # rpc_rounds=4, by which point the LP is integral, so the
        # FRACTIONAL branch had no test at all until one was written for
        # this mutant.
        "M208-a-fractional-lp-optimum-is-returned-as-a-repair",
        "qldpc_cert rounds a pseudocodeword and calls it a correction "
        "(both LP paths: they now share one rule)",
        "    if np.max(np.abs(x - np.round(x))) >= INTEGRALITY_TOL:\n"
        "        return None",
        "    if False:\n"
        "        return None",
        "test_a_fractional_optimum_yields_NO_repair_rather_than_a_rounded_one",
        target="qldpc_cert",
    ),
    # ---- two modules that had never been mutated at all (2026-08-22) ----
    Mutant(
        # A LOOKUP KEYED ON None MATCHES THE WRONG CODE. `cid` comes from
        # `row["code_id"] or row["bliss_hash"]`, and `rec.get("code_id")
        # == None` is TRUE for every stored record that simply lacks the
        # key -- so a row with no identity was handed another code's
        # promoted certificate, with `from_artifact: true` and no error
        # anywhere. Not a crash: a wrong attribution that looks like
        # evidence. Found by closing mypy, which had been sitting at its
        # ratchet "best" of 52 and was never audited.
        "M205-a-promoted-lookup-runs-without-an-identity",
        "a passport is assembled from another code's certificate",
        "    if not cid:\n        raise ValueError(",
        "    if False:\n        raise ValueError(",
        "test_a_promoted_lookup_with_no_identity_is_REFUSED",
        target="core",
    ),
    Mutant(
        # THE KILLER IS AIMED AT THE MESSAGE, DELIBERATELY. Deleting this
        # guard does NOT stop a malformed key raising: every one of them
        # still reaches `int()`, which raises a ValueError of its own. A
        # bare `raises(ValueError)` would stay green with the guard gone
        # -- a killer aimed one layer past what it guards, which is how
        # M175/M178/M182 first survived. What the guard contributes is a
        # refusal that names the ARTIFACT, so that is what is asserted.
        "M206-a-malformed-pinned-edge-key-is-parsed-anyway",
        "the pinned-edge parser stops naming a malformed artifact",
        "    if not sep or not head or not tail:\n        raise ValueError(",
        "    if False:\n        raise ValueError(",
        "test_a_malformed_edge_key_is_REFUSED_BY_NAME",
        target="device_dem",
    ),
    # ---- the graph binding: what stops a certificate being REPLAYED ----
    Mutant(
        # cert_format is in standards_gate's CHECKER_MODULES and its own
        # docstring names the attack: "a genuine certificate for an easy
        # graph can be replayed as a certificate for a hard one and every
        # arithmetic check still passes". The defence is recomputing the
        # fingerprint instead of trusting the field. It had a killing
        # test and NO aimed mutant until a registry audit found it.
        "M86-cert-format-trusts-the-declared-graph",
        "graph fingerprint compared against itself: replay across graphs accepted",
        '    if doc.get("graph_sha256") != fp:',
        "    if False:",
        "test_a_certificate_cannot_be_replayed_against_a_different_graph",
        target="cert_format",
    ),
    # M87 REMOVED as EQUIVALENT UNDER SINGLE MUTATION, with evidence --
    # the same disposition M58/M59 got, and for the same reason rather
    # than because a red was inconvenient.
    #
    # The candidate was "exact_decode drops the XOR cancellation"
    # (`corr = tuple(sorted(k for k, c in count.items() if c % 2 == 1))`
    # -> `tuple(sorted(count))`). It SURVIVED, and the reason is not a
    # weak test: in an OPTIMAL matching over a metric closure the chosen
    # shortest paths cannot share an edge, because two paths sharing one
    # could be re-paired into a strictly cheaper matching (the standard
    # exchange argument). So every entry of `count` is 1 and the parity
    # filter is a no-op on every input the function can actually be
    # given. MEASURED: a mutated copy compiled alongside the real one
    # disagreed on 0 of 6,000 random graphs.
    #
    # The XOR stays -- it is correct, and it is what makes the union of
    # paths a T-join if the matching ever were not optimal -- but no
    # single mutation can make its absence observable, and a mutant that
    # cannot fail is not evidence.
    # ---- the parity-matching producer: two REAL bugs, now guarded ----
    Mutant(
        # Found by its own test. networkx adds nodes implicitly with
        # edges, so an UNREACHABLE terminal never appeared in the graph
        # and the perfect-matching check could not see it -- the producer
        # returned 0 for a syndrome with NO solution, a value BELOW the
        # truth. Unsoundness is the one failure this module may not have.
        "M84-matching-drops-unreachable-terminals",
        "terminals unregistered: an unpairable syndrome silently scores 0",
        "    for u in terms:",
        "    for u in []:",
        "test_a_disconnected_terminal_yields_no_bound_rather_than_a_wrong_one",
        target="coset_matching",
    ),
    Mutant(
        # Also found by its own test. A ZERO gap is a FREE parity flip,
        # not the absence of one; excluding it made a constructed tie
        # return None for the opposite coset -- refusing exactly the
        # ambiguity case the four-class chain turns on.
        "M85-matching-treats-a-free-flip-as-impossible",
        "zero-cost parity flips excluded, so exact ties lose a coset",
        "        if g is not None:",
        "        if g is not None and g > 0:",
        "test_it_reproduces_a_constructed_tie",
        target="coset_matching",
    ),
    # ---- THE BEACON: the ordering IS the blinding ----
    Mutant(
        # Committing to a round that is already published means committing
        # to a value we can already read -- the self-reference the beacon
        # exists to close. It would look identical in every artifact.
        "M81-beacon-accepts-a-current-round",
        "future-round guard dropped: a published round can be committed to",
        "    if lead_seconds <= 0:",
        "    if False:",
        "test_committing_to_a_current_or_past_round_is_REFUSED",
        target="beacon",
    ),
    Mutant(
        # Without the parameters bound into the stream, a schedule does not
        # pin its own shot count -- found for real: a 400-shot schedule
        # verified against a claimed 401 because shot 400 was not selected.
        "M82-beacon-parameters-not-bound",
        "declared parameters dropped from the derivation domain",
        '    return f"{domain}|shots={shots}|q={q.numerator}/{q.denominator}" \\\n           f"|mech={n_mechanisms}"',
        "    return domain",
        "test_every_declared_parameter_changes_the_whole_stream",
        target="beacon",
    ),
    Mutant(
        # A verifier that accepts whatever it is handed turns the whole
        # construction into a ceremony.
        "M83-beacon-verify-accepts-anything",
        "schedule verification stops comparing against the beacon",
        "    if got == expect:",
        "    if True:",
        "test_every_tamper_is_refused",
        target="beacon",
    ),
    # ---- PHASE 4: the seal a published dataset rests on ----
    Mutant(
        # RFC-6962 domain separation. Without the 0x00 prefix a leaf and
        # an internal node live in the same space, and a subtree can be
        # presented as a member of the committed set.
        "M75-merkle-no-domain-separation",
        "receipt leaves hashed without the RFC-6962 leaf prefix",
        "    return sha256_hex(_LEAF + canon(receipt))",
        "    return sha256_hex(canon(receipt))",
        "test_domain_separation_leaf_is_not_a_bare_hash",
        target="rehearsal",
    ),
    Mutant(
        # The odd tail is PROMOTED, not hashed against a copy of itself.
        # Duplicating it is the classic Merkle bug (CVE-2012-2459 shape):
        # two different leaf sets can produce the same root.
        "M76-merkle-odd-node-duplicated",
        "odd tail node duplicated instead of promoted",
        "            nxt.append(level[-1])          # odd node promoted, RFC-6962",
        "            nxt.append(_pair(level[-1], level[-1]))",
        "test_every_index_of_every_tree_size_proves_inclusion",
        target="rehearsal",
    ),
    Mutant(
        # The pre-registered threshold is the only thing standing between
        # an anomaly lane and a post-hoc story.
        "M77-anomaly-lane-ignores-threshold",
        "anomaly lane admits every model failure regardless of Delta_L",
        "        if Fraction(d) >= thr:",
        "        if True:",
        "test_anomaly_lane_threshold_is_inclusive_and_strict_below",
        target="rehearsal",
    ),
    Mutant(
        # A lane that fires on `search` or `ambiguity` is reporting
        # decoder misses and ties as physics anomalies.
        "M78-anomaly-lane-accepts-any-class",
        "anomaly lane no longer restricts itself to the model class",
        '        if r.get("class") != "model":',
        "        if False:",
        "test_anomaly_lane_selects_only_large_delta_model_failures",
        target="rehearsal",
    ),
    Mutant(
        # The analysis choices are what a post-hoc analyst would tune.
        # Dropping them from the sealed object makes the freeze bind the
        # data and leave the interpretation free.
        "M79-freeze-drops-the-analysis-pins",
        "freeze manifest omits the analysis section it exists to seal",
        '            "analysis": dict(sorted(analysis.items()))}',
        '            "analysis": {}}',
        "test_freeze_manifest_commitment_moves_when_the_threshold_moves",
        target="rehearsal",
    ),
    # ---- the H' reduction: the parity row IS the coset ----
    Mutant(
        # Drop the check that a supplied candidate lies in the coset it
        # claims and the tier will certify a witness from the WRONG side,
        # which is Phase 1's named adversarial case.
        "M80-coset-cert-accepts-wrong-coset-candidate",
        "H' tier accepts a candidate from the opposite coset",
        "        if sum(1 for i in candidate if i in set(cut_vars)) % 2 \\\n                != int(target_parity) % 2:",
        "        if False:",
        "test_wrong_coset_candidate_is_refused",
        target="coset_cert",
    ),
    # ---- THE DECOMPOSITION: the headline instrument's own classifier ----
    Mutant(
        # The tolerance this campaign was audited for. Restoring it credits
        # a provably suboptimal correction as `agree` whenever the excess
        # hides under 1e-12 -- p_search silently loses shots to p_model.
        "M67-decomposition-tolerance-restored",
        "search test uses a float tolerance instead of exact >",
        "    if w > u_same:",
        "    if float(w) - float(u_same) > 1e-12:",
        "test_sub_tolerance_excess_is_search_not_agree",
        target="decomposition",
    ),
    Mutant(
        # Two distinct optima that collapse to one double become a "tie",
        # so a certified logical decision is thrown away as ambiguity.
        # This is the divergence the four copies had already drifted into.
        "M68-decomposition-float-tie",
        "ambiguity test compares floats instead of exact Fractions",
        "    if u_same == u_opp:",
        "    if float(u_same) == float(u_opp):",
        "test_float_indistinguishable_optima_are_not_a_tie",
        target="decomposition",
    ),
    Mutant(
        # w_corr < u_same means the weight table and the certifier are
        # describing different graphs. Without the refusal the shot gets a
        # class anyway and the disagreement is laundered into a statistic.
        "M69-decomposition-invariant-dropped",
        "correction lighter than its own coset optimum is classified, not refused",
        "    if w < u_same:",
        "    if False:",
        "test_correction_lighter_than_its_own_optimum_refuses",
        target="decomposition",
    ),
    Mutant(
        # The certified sector is the LIGHTER coset's, which is the
        # decoder's parity only when the decoder won. Collapsing it to the
        # decoder's parity makes p_model unable to ever contradict it.
        "M70-decomposition-sector-is-decoder-parity",
        "certified sector taken from the decoder instead of the lighter coset",
        "    sector = corr_parity if u_same < u_opp else 1 - corr_parity",
        "    sector = corr_parity",
        "test_certified_sector_follows_the_lighter_coset_not_the_decoder",
        target="decomposition",
    ),
    # ---- THE LOGICAL CUT: if `c` is wrong, every certificate is too ----
    Mutant(
        # The whole point of extracting `c` from two sources. Without the
        # refusal, a decoder/noise-model disagreement about what the
        # logical operator IS passes silently into every receipt.
        "M71-logical-cut-agreement-not-enforced",
        "two-source cut disagreement accepted instead of refused",
        "    if a != b:",
        "    if False:",
        "test_disagreement_is_refused_loudly",
        target="logical_cut",
    ),
    Mutant(
        # The `^`-separator parse. Without the split, a decomposed
        # hyperedge becomes one >2-detector blob that is dropped, so the
        # dominant mechanism per edge is chosen over an incomplete set --
        # the exact bug that read 23 edges where the graph has 8.
        "M72-logical-cut-ignores-separators",
        "DEM separator components not split: dominant computed over an incomplete set",
        "            if t.is_separator():",
        "            if False:",
        "test_the_two_sources_agree_on_a_real_surface_code",
        target="logical_cut",
    ),
    # ---- THE BRIDGE: the transfer condition is the whole theorem ----
    Mutant(
        # eps_q < rho IS the lemma application. Forcing it true promotes
        # coarse-lattice decisions the raw objective does not support --
        # a claim about the wrong objective, silently.
        #
        # THIS MUTANT SURVIVED its first, obvious killer.
        # test_fragile_decision_is_refused_not_promoted allows the
        # transfer whenever the sector is still correct, and on its
        # instance eps_q ~ 3.3e-4 against rho pinned at the w_min/2 cap of
        # 0.5 -- at every scale -- so its refusal branch never executed
        # and hard-coding `transferred = True` passed it. The killer below
        # asserts the instance is fragile BEFORE asserting the refusal.
        "M73-bridge-transfers-without-margin",
        "fixed-point bridge promotes every decision regardless of the margin",
        "    transferred = eps_q < rho",
        "    transferred = True",
        "test_fragile_decision_is_never_promoted_constructed",
        target="fixed_point_bridge",
    ),
    Mutant(
        # A weight rounded to zero breaks the lemma's w_min > 0
        # precondition downstream; the clamp is what keeps the radius
        # meaningful, and the distortion is counted honestly.
        "M74-bridge-zero-weight-not-clamped",
        "quantization allows a weight to reach zero, voiding the lemma's precondition",
        "        if wq <= 0:",
        "        if False:",
        "test_zero_rounding_is_clamped_and_counted",
        target="fixed_point_bridge",
    ),
    Mutant(
        # THE PARITY CLASSIFIER. Drop the cut mask and every coset member
        # is classified by the parity of ALL its bits -- Z_0/Z_1 become a
        # partition of the wrong question and every Bayesian receipt lies
        # while summing to the correct total.
        "M65-bayes-parity-ignores-cut",
        "Bayesian tier classifies by total parity instead of cut parity",
        # ANCHOR DISAMBIGUATED 2026-08-24: this line appears twice, as
        # the SEED term and as the Gray-walk BODY, and the ordinal rule
        # aimed it at the seed -- the weaker of the two, and the one term
        # out of 2^k. The body carried no mutant at all (now M297).
        '    x = particular\n'
        '    z[bin(x & cut_mask).count("1") & 1] += mass(x)',
        '    x = particular\n'
        '    z[bin(x).count("1") & 1] += mass(x)',
        "test_matches_brute_force_on_random_instances",
        target="bayes_coset",
    ),
    Mutant(
        # THE 2^k - 1 TERMS THE ORDINAL ANCHOR NEVER TOUCHED. M65 mutates
        # the SEED; this mutates the Gray-code walk that accumulates
        # everything else. The two are the same defect at opposite ends
        # of the coverage, and only one of them was ever simulated.
        "M297-bayes-parity-ignores-cut-in-the-walk",
        "the Gray-code walk bins by TOTAL parity instead of cut parity, "
        "so every term but the seed lands in the wrong coset",
        '        x ^= kernel[changed.bit_length() - 1]\n'
        '        z[bin(x & cut_mask).count("1") & 1] += mass(x)',
        '        x ^= kernel[changed.bit_length() - 1]\n'
        '        z[bin(x).count("1") & 1] += mass(x)',
        "test_matches_brute_force_on_random_instances",
        target="bayes_coset",
    ),
    Mutant(
        # The scope guard. Without it a 2^29 walk does not refuse -- it
        # RUNS, for hours, and the tier silently becomes a hang instead of
        # an honest refusal naming the enclosure tier it needs.
        "M66-bayes-kernel-guard-dropped",
        "oversize kernels walked instead of refused",
        "if k > KERNEL_LIMIT:",
        "if False:",
        "test_kernel_guard_refuses_oversize",
        target="bayes_coset",
    ),
    Mutant(
        # THE PARITY DIMENSION ITSELF. Drop the XOR that threads path
        # parity through the pairing recursion and U_0/U_1 collapse into
        # the same table -- every Delta_L becomes 0, every receipt lies.
        "M63-oracle-drops-parity",
        "coset oracle ignores path parity: both cosets return the same optimum",
        "q = p ^ addp",
        "q = p",
        "test_matches_brute_force_on_every_syndrome_of_the_path_graph",
        target="coset_oracle",
    ),
    Mutant(
        # The derivation requires eps < w_min; the cap is what keeps the
        # emitted radius from promising a region where a weight reaches
        # zero and the loser-side bound goes vacuous.
        "M64-robustness-uncapped",
        "robustness radius emitted without the w_min/2 cap",
        "return min(rho, wm / 2)",
        "return rho",
        "test_refusals",
        target="robustness",
    ),
    Mutant(
        # THE SOLVER UNDER THE ACCEPTANCE DECISION. If the ratio test picks
        # the WRONG entering constraint the walk leaves the feasible region,
        # and an "optimum" from an infeasible walk is exactly the false
        # attainment the whole exact tier exists to prevent. The aimed test
        # verifies feasibility from scratch, so this must die there.
        "M59-exact-lp-ratio-test-max",
        "simplex ratio test takes the largest step instead of the smallest",
        "if best_t is None or t < best_t or (t == best_t and i < enter):",
        "if best_t is None or t > best_t or (t == best_t and i < enter):",
        "test_known_tiny_optimum",
        target="exact_lp",
    ),
    Mutant(
        # Bland guard: accepting a NEGATIVE multiplier as optimal stops the
        # simplex early, understating the optimum -- which would make the
        # auditor witness attain (it must stay SHORT by exactly 5e-7).
        "M60-exact-lp-stops-early",
        "optimality declared while a multiplier is still negative",
        "if lam[pos] < 0 and (leave_idx is None or i < leave_idx):",
        "if lam[pos] < -1 and (leave_idx is None or i < leave_idx):",
        "test_ulp_scale_optimum_is_exact",
        target="exact_lp",
    ),
    # M61 (XOR -> union in exact_decode) REMOVED as EQUIVALENT UNDER OPTIMAL
    # MATCHING, with the argument recorded rather than the mutant retried: an
    # edge-overlapping pairing costs +2x(shared weight) over its disjoint
    # exchange, so blossom never STRICTLY prefers overlap and the two
    # expressions agree on every optimal output with positive weights. The
    # XOR stays in the code as defense-in-depth for zero-weight ties (our
    # boundary copy-copy edges are exactly weight zero), where a tie-broken
    # overlapping pairing would put a cancelled edge into the correction and
    # break syndrome reproduction. Deleting the XOR to match the mutant's
    # visibility would be the regression; keeping an unkillable mutant would
    # overstate the corpus. reproduces_syndrome() remains the runtime guard.
    Mutant(
        # A repair that REPLACES the answer must be flagged as a repair.
        # Dropping the flag silently blends proof-of-repair into
        # proof-of-original -- the exact inflation the separate column
        # exists to prevent.
        "M62-repair-unflagged",
        "a certified repair shipping without repaired=True",
        "repaired=True,",
        "repaired=False,",
        "test_certify_or_repair_repairs_a_provably_suboptimal_correction",
        target="certifying_decoder",
    ),
    Mutant(
        "M0-weak-key-mint",
        "the historical defect: minting with a publicly-derivable key (bytes(32) seed)",
        "if pub_hex in WEAK_PUBKEYS and not allow_weak_key:",
        "if False:",
        "test_genesis_0_all_zeros_seed_is_refused_at_mint",
    ),
    Mutant(
        # THE GATE THAT POLICES PROSE, POLICED. It passes loudest when it finds
        # nothing, so a silent no-op in it is invisible: the docs go stale and
        # the run stays green. This is the shape that actually happens -- a
        # scanner that skips every line for a plausible-looking reason.
        "M28-doc-gate-skips-every-line",
        "the prose scanner silently matching nothing, so stale headlines pass",
        # RE-ANCHORED 2026-08-18. The old anchor was `if fenced:` followed
        # immediately by `continue`; the fenced branch then grew the
        # command-comment assertion logic, so the two-line anchor stopped
        # matching and the gate SKIPPED this mutant -- reporting that its
        # coverage was no longer proven. Exactly the drift the skip-is-a-
        # failure rule exists to surface, and it stayed invisible only
        # because the gate's baseline was separately red.
        #
        # The intent is unchanged: forcing the branch taken makes every
        # prose line skip the scanner, so a stale headline passes.
        "            if fenced:",
        "            if True:",
        "test_a_stale_headline_in_the_docs_is_actually_caught",
        "doc_claims_gate",
    ),
    Mutant(
        # The duplicate-replica detector. Its FIRST version conflated an empty
        # pivot list with a duplicated run and produced a false finding that was
        # published twice; the mutant now defends the corrected check, which
        # must still fire on genuinely identical non-empty pivots.
        "M29-tautology-detector-blinded",
        "a replica set of N identical runs no longer reported as a tautology",
        '    return "duplicate"',
        '    return "distinct"',
        "test_duplicate_replicas_are_flagged_but_missing_pivots_are_not",
        "verify_closures",
    ),
    Mutant(
        # The guard that turns a blind 1.000 into a refusal. If it stops firing
        # the lane silently returns to publishing a structural artifact that
        # reads exactly like an honest null result.
        "M30-blind-decoder-reports-anyway",
        "an inert-herald decoder emitting V_E = 1.000 instead of refusing",
        "    if not herald_influences:",
        "    if False:",
        "test_a_blind_decoder_refuses_instead_of_returning_one_point_zero",
        "erasure_decoder",
    ),
    Mutant(
        # The conditioning IS the result. If the fired-pattern lookup silently
        # degenerates, every bucket gets the marginal prior, the two arms
        # converge, and V_E pins to 1.0 -- indistinguishable from the two
        # sessions of nulls this lane just climbed out of.
        "M31-conditional-dem-ignores-the-herald",
        "the per-pattern DEM no longer conditions on which heralds fired",
        "ms = sorted({q for (rr, q) in sites if rr == r and fired.get((rr, q))})",
        "ms = sorted({q for (rr, q) in sites if rr == r and False})",
        "test_a_fired_erasure_becomes_a_free_edge_and_a_silent_one_vanishes",
        "ve_reweighted",
    ),
    Mutant(
        # THE ARM-MISMATCH CLASS, pinned. Blind arm at raw eps instead of the
        # 0.75*eps marginal is exactly the defect that once printed "ARR
        # REOPENS" from nothing -- one arm carrying more noise than the other.
        "M32-blind-arm-mispriced",
        "the blind arm decoding with eps instead of the 0.75*eps marginal",
        'out.append("DEPOLARIZE1", [q], 0.75 * eps)',
        'out.append("DEPOLARIZE1", [q], eps)',
        "test_the_blind_arm_gets_the_exact_marginal_not_the_raw_rate",
        "h1_reweighted",
    ),
    Mutant(
        # The channel gate IS the epistemic type system. If role enforcement
        # goes quiet, a prior walks into the observation channel again and the
        # two-session G4b defect is re-legalized -- silently, because nothing
        # else checks roles.
        "M33-epistemic-channels-unenforced",
        "check_channels admitting any Evidence into any channel",
        "        if ev.role is not need:",
        "        if False:",
        "test_a_herald_cannot_enter_the_observation_channel",
        "evidence",
    ),
    Mutant(
        # The contract's first check is the one the herald failed in history.
        # If the refusal path dies, an instrument with a provably unconsumed
        # input emits a clean 1.000 again.
        "M34-admissibility-emits-unconsumed",
        "emit() no longer refusing when side information is not consumed",
        '        if not m["A_causal_reachability"].satisfied:',
        '        if False:',
        "test_an_unconsumed_input_refuses_not_reports",
        "admissibility",
    ),
    Mutant(
        # THE C-3 REFUSAL. Re-permitting node potentials reopens the
        # CONFIRMED soundness break: a weight-20 correction certified
        # OPTIMAL on a graph whose true optimum is 2. Potentials are the
        # dual of a DIFFERENT primal (perfect matching on the metric
        # closure), and mixing them lets a negative potential absorb
        # slack on edges away from T while inflating the objective.
        #
        # The break survived for months because it looked like a caveat
        # rather than a bug, so the refusal gets an aimed mutant: a
        # one-line deletion must not be able to reopen it quietly.
        "M91-checker-readmits-node-potentials",
        "node potentials re-permitted: the mixed-dual break reopens and a "
        "weight-20 correction certifies as optimal against a true optimum of 2",
        "    if cert.node_potentials:",
        "    if False:",
        "test_the_legacy_checker_now_refuses_the_forgery",
        "matching_cert",
    ),
    Mutant(
        # THE CERTIFYING CHECKER'S CORE. If dual feasibility stops being
        # enforced, a fabricated dual certifies any correction and the whole
        # contribution becomes decorative -- a checker that cannot say no.
        "M36-checker-skips-dual-feasibility",
        "dual feasibility no longer tested, so an inflated dual certifies anything",
        "        if lhs > w + tol:",
        "        if False:",
        "test_refuses_an_infeasible_dual",
        "matching_cert",
    ),
    Mutant(
        # The T-JOIN invariant. Wrong here and the checker accuses correct
        # decoders (it did, at d=7) or accepts corrections that do not explain
        # the syndrome at all.
        "M37-checker-ignores-odd-degree-set",
        "odd-degree set no longer compared to the syndrome",
        "    if odd != fired:",
        "    if False:",
        "test_refuses_when_odd_degree_set_disagrees_with_the_syndrome",
        "matching_cert",
    ),
    Mutant(
        # Weak duality is the entire proof. Without the gap test the checker
        # accepts a feasible dual that does not reach the primal -- i.e. it
        # certifies optimality it has not shown.
        # RE-ANCHORED after the external audit of 2026-08-04. The
        # acceptance test moved from a bare `gap > 1e-6` to a conjunction
        # with the exact dyadic-lattice proof, so the old anchor stopped
        # matching. The gate SKIPPED it rather than passing it, which is
        # the only acceptable behaviour: a mutant whose anchor has
        # drifted proves nothing, and printing it killed would be a lie.
        # Now anchored on the exactness test, which is the part that
        # actually decides acceptance.
        "M38-checker-accepts-a-gap",
        "a positive primal-dual gap no longer refused",
        "    if gap > 1e-6 or not proven:",
        "    if False:",
        "test_refuses_a_feasible_dual_that_does_not_reach_the_primal",
        "matching_cert",
    ),
    Mutant(
        # T-ODD admissibility. An even set is not a dual variable; admitting
        # one lets the producer inflate the bound past the true optimum.
        "M39-moat-admits-even-sets",
        "balls with an EVEN intersection with T admitted as dual variables",
        "            if len(members & T) % 2 == 1:",
        "            if True:",
        "test_moat_producer_never_emits_an_even_set_dual",
        "moat_dual",
    ),
    Mutant(
        # THE FAST COPY MUST ENCODE THE SAME RULE AS THE REFERENCE. This is the
        # twin-copy trap: moat_dual is readable and slow, MoatEngine is the one
        # that actually runs at scale. If the fast copy drifts, the slow copy
        # keeps passing its own tests and the drift ships.
        "M40-engine-admits-even-sets",
        "MoatEngine drops the T-odd admissibility rule the reference enforces",
        '        odd = (memb[:, t_idx].sum(axis=1) % 2) == 1',
        '        odd = memb[:, t_idx].sum(axis=1) >= 0',
        "test_engine_agrees_with_the_reference_producer",
        "moat_dual",
    ),
    Mutant(
        # A GUARD THAT CANNOT FAIL IS AS BROKEN AS ONE THAT CANNOT PASS. Without
        # this branch the battery can quietly stop attacking -- scoring valid
        # certificates as forgeries -- and still report a clean sheet.
        "M41-battery-audit-cannot-say-no",
        "assert_all_false stops reporting an attack that is not actually heavier",
        '                bogus.append((name, "not actually heavier than a known correction"))',
        '                pass',
        "test_the_battery_detects_a_bogus_attack",
        "forgeries",
    ),
    Mutant(
        # D, G and H are the only attacks that test the WEIGHT rule; every other
        # one breaks the odd-degree set and is caught by a check that every
        # decoder already performs. Losing them guts the battery while leaving
        # the headline number intact.
        "M42-battery-drops-the-stealth-attacks",
        "the strictly-suboptimal (syndrome-consistent) attacks stop being generated",
        '        if _odd(cand) == T and sum(W[k] for k in cand) > primal + 1e-9:',
        '        if False:',
        "test_the_battery_still_contains_the_hardest_attack_class",
        "forgeries",
    ),
    Mutant(
        # Escalation is safe ONLY because the checker adjudicates every rung. If a
        # rung could report success on its own, the LADDER would be deciding --
        # which is precisely what a certifying algorithm must never allow.
        "M43-ladder-accepts-without-checking",
        "a ladder rung reports success without the checker agreeing",
        '            if r.accepted:',
        '            if True:',
        "test_escalating_the_ladder_never_certifies_a_worse_correction",
        "certifying_decoder",
    ),
    Mutant(
        # The CS family's entire value is the cut-once property: it is what
        # makes sum(z) telescope to the primal. Absorbing a foreign tree
        # PARTIALLY -- exactly what this mutant does by dropping the atomic
        # propagation -- produces sets that slice trees in half and cut J
        # several times. The LP still runs, the checker still refuses bad
        # packings, and the rung silently degrades to noise: invisible
        # everywhere except a test of the property itself.
        "M44-cs-family-splits-foreign-trees",
        "atomic tree absorption dropped; emitted sets slice foreign trees",
        "            if x in tree_of and tree_of[x] != home and x not in A:",
        "            if False:",
        "test_cs_family_sets_cut_the_correction_exactly_once",
        "moat_growth",
    ),
    Mutant(
        # The boundary-crossing moats are the deepest ones on the hardest
        # shots, and dropping them is invisible everywhere except a property
        # test: the LP still runs, the checker still refuses bad packings,
        # and the rung silently loses exactly the candidates the shots that
        # reach it need. This mutant reverts to the v1 behaviour -- refuse
        # the set instead of emitting its complement -- which is precisely
        # the lossiness that was diagnosed and fixed.
        "M45-blossom-drops-boundary-moats",
        "boundary-containing moats dropped instead of emitted as complements",
        "        if BOUNDARY in S:\n            S = allv - S",
        "        if BOUNDARY in S:\n            return",
        "test_blossom_candidates_express_boundary_crossing_moats",
        "moat_growth",
    ),
    Mutant(
        # THE HISTORICAL DEFECT, verbatim: the first check_qldpc implemented
        # the opposite of its own docstring's sign rule and refused a tight
        # certificate on trial 6 of the brute-force self-test. Only a test
        # with an independently known answer can tell a docstring from its
        # code, so that test is the killer.
        "M46-qldpc-dual-sign-inverted",
        "the qLDPC dual load computed with inverted facet signs",
        "            coef = -y if i in Fs else y",
        "            coef = y if i in Fs else -y",
        "test_qldpc_certificate_sound_aligned_and_unfooled",
        "qldpc_check",  # moved in the checker module split
    ),
    Mutant(
        # The dual-feasibility pass is the ONLY thing separating a linearly
        # inflated dual (scale every y until the bound equals any weight you
        # like) from an OPTIMAL verdict. Delete it and forged certificates
        # certify arbitrary answers.
        "M47-qldpc-feasibility-unchecked",
        "per-variable dual feasibility skipped: inflated duals certify anything",
        "        if ld - mu.get(i, 0.0) > w.get(i, 0.0) + tol:",
        "        if False:",
        "test_an_inflated_dual_cannot_certify_a_heavier_error",
        "qldpc_check",  # moved in the checker module split
    ),
    Mutant(
        # Step 1 is the recomputed syndrome. Without it a 'correction' that
        # corrects nothing but whose weight happens to meet the bound is
        # OPTIMAL -- the checker would bless an answer to a different problem.
        "M48-qldpc-syndrome-not-recomputed",
        "claimed error's syndrome taken on faith instead of recomputed",
        # ANCHOR DISAMBIGUATED 2026-08-24. THREE paths in `qldpc_check.py`
        # recompute the claimed error's syndrome -- `_validate_correction`,
        # the branch-and-bound target, and `check_qldpc_exact` itself --
        # and the ordinal rule mutated only the first. Deleting the parity
        # test from either of the other two cost nothing. A guard on one
        # of three paths is a coincidence three times over (M298, M299).
        "        if len(e & supports[j]) % 2 != syn[j]:\n"
        "            return None, None, QldpcCheckResult(",
        "        if False:\n"
        "            return None, None, QldpcCheckResult(",
        "test_a_non_correction_with_matching_weight_is_refused",
        "qldpc_check",  # moved in the checker module split
    ),
    Mutant(
        "M298-qldpc-bnb-target-takes-the-syndrome-on-faith",
        "the branch-and-bound target stops recomputing the claimed "
        "error's syndrome, so the SEARCH accepts a non-correction even "
        "where the final validator would have caught it",
        "        if len(e & supports[j]) % 2 != syn[j]:\n"
        "            return None, QldpcCheckResult(",
        "        if False:\n"
        "            return None, QldpcCheckResult(",
        # RE-AIMED 2026-08-24 AFTER IT SURVIVED. The killer it was given
        # reaches `_validate_correction` through the FLOAT entry point
        # and never touches the tree path at all -- the same "wrong
        # organ" mistake the ordinal anchor made, one level up.
        "test_the_TREE_path_recomputes_the_syndrome_too",
        "qldpc_check",
    ),
    Mutant(
        "M299-qldpc-exact-takes-the-syndrome-on-faith",
        "`check_qldpc_exact` -- the entry point a stranger calls -- "
        "stops recomputing the syndrome of the error it is handed",
        "            if len(e & supports[j]) % 2 != syn[j]:\n"
        "                return QldpcCheckResult(",
        "            if False:\n"
        "                return QldpcCheckResult(",
        # RE-AIMED after it SURVIVED: same reason -- the killer
        # exercised the float path, not `check_qldpc_exact`, which is
        # the entry point a stranger actually calls.
        "test_the_EXACT_path_recomputes_the_syndrome_too",
        "qldpc_check",
    ),
    Mutant(
        # A redundant check's parity is the XOR of its rows' syndrome bits.
        # Forget the XOR and every odd-parity combined check (the 3-cycle's
        # {1,3,5} among them) has its true facets called right-parity and
        # refused -- the cut rung silently dies while everything else stays
        # green.
        # RE-ANCHORED after the trust-module refactor, and it caught a real
        # hole doing it. The anchors here are INDENTATION-COUPLED text, so
        # extracting a function moves them. This mutant used to sit in the
        # float checker; once the exact facet rule was consolidated into
        # _exact_facet_scan the same anchor matched the EXACT path instead,
        # where no test covered redundant-check parity at all -- so it
        # SURVIVED. The behaviour never moved (both equivalence gates prove
        # that); what had moved was which copy of the rule this mutant
        # tested, and the new copy was untested. Both are now covered, by
        # indentation, with a test each.
        "M49-qldpc-redundant-parity-forgotten-exact",
        "redundant-check parity not XORed from syndrome bits (exact path)",
        "            for r in rows:\n                sup = sup ^ supports[r]\n                par ^= syn[r]",
        "            for r in rows:\n                sup = sup ^ supports[r]",
        "test_redundant_check_parity_is_XORED_from_the_syndrome_bits_exact",
        "qldpc_check",
    ),
    Mutant(
        # The SAME rule in the float checker's _resolve_facet_check, at its
        # own indentation. Two copies of a rule need two mutants, or
        # deleting one of them reads as green.
        "M49b-qldpc-redundant-parity-forgotten-float",
        "redundant-check parity not XORed from syndrome bits (float path)",
        "        for r in rows:\n            sup = sup ^ supports[r]\n            par ^= syn[r]",
        "        for r in rows:\n            sup = sup ^ supports[r]",
        "test_the_three_cycle_pseudocodeword_closes_and_certifies",
        "qldpc_check",
    ),
    Mutant(
        # The exact tier's separation: the minimum T-odd cut among Gomory-Hu
        # tree cuts. Flip the parity test and separation only ever offers
        # T-EVEN sides -- inadmissible, so the pool never grows, the bound
        # stays at whatever the seeds gave, and the tier silently degrades
        # to a no-op that still returns cleanly. The oracle-equality test is
        # the only thing that can see it.
        # RE-ANCHORED: the Padberg-Rao scan moved into _min_odd_cut when
        # exact_tjoin_sets was split, losing four spaces of indentation.
        # The gate reported it SKIPPED rather than passing it, which is the
        # only acceptable behaviour -- a mutant whose anchor has drifted
        # proves nothing, and printing it as killed would be the lie.
        "M50-exact-tier-separates-even-cuts",
        "Padberg-Rao scan admits T-even sides; the exact tier goes inert",
        "        if len(set(side) & Tp) % 2 == 1:",
        "        if len(set(side) & Tp) % 2 == 0:",
        "test_exact_tier_equals_the_oracle_and_survives_the_repack",
        "tjoin_exact",
    ),
    Mutant(
        # canon(): a Gomory-Hu side containing the boundary defines the same
        # cut as its complement, which the packing can use. Throw such sides
        # away instead and every boundary-adjacent moat the optimum needs is
        # silently unavailable -- the bound stops short on exactly the
        # instances with boundary structure, which is most hardware graphs.
        "M51-exact-tier-drops-boundary-sides",
        "boundary-containing GH sides discarded instead of complemented",
        "        if BOUNDARY in S:\n            S = allv - S\n        if not S or BOUNDARY in S:\n            return None\n        return S",
        "        if BOUNDARY in S:\n            return None\n        if not S:\n            return None\n        return S",
        "test_exact_tier_equals_the_oracle_and_survives_the_repack",
        "tjoin_exact",
    ),
    Mutant(
        # The loop can exit right after appending a set the LP never saw;
        # without the final re-solve the duals are stale against the pool and
        # the zip silently drops the newest set's contribution. (An earlier
        # M52 candidate -- loosening the convergence threshold to 0.5 -- was
        # MEASURED EQUIVALENT across 400 adversarial odd-cycle trials: when a
        # half-valued cut exists a smaller violated cut nearly always
        # coexists and Gomory-Hu returns the minimum. A mutant whose killer
        # cannot fire is a dead gate; it was replaced, not shipped.)
        "M52-exact-tier-stale-duals",
        "final re-solve deleted: duals stale against the pool on early exit",
        "    if pool and len(duals) != len(pool):",
        "    if False:",
        "test_round_capped_run_still_resolves_its_final_pool",
        "tjoin_exact",
    ),
    Mutant(
        # The provenance sweep, blinded the way such scanners actually die:
        # every token quietly treated as already-traced. The gate still
        # prints its banner, still exits 0, still LOOKS like diligence --
        # the same shape as M28 for the doc gate, guarded the same way: a
        # test that plants an orphan and demands the red.
        "M53-provenance-sweep-blinded",
        "every number treated as traced; orphan claims sail through",
        "                if tok in truth or tok in wl:\n                    continue\n                orphans.append((rel, ln, tok, line.strip()[:90]))\n            for m in RATE.finditer(line):",
        "                if True:\n                    continue\n                orphans.append((rel, ln, tok, line.strip()[:90]))\n            for m in RATE.finditer(line):",
        "test_provenance_gate_catches_a_planted_orphan",
        "claims_provenance_gate",
    ),
    Mutant(
        "M1-weak-key-verify",
        "accepting a passport signed with a publicly-derivable key",
        'checks["key_not_weak"] = pub not in WEAK_PUBKEYS',
        'checks["key_not_weak"] = True',
        "test_genesis_0_weak_signed_passport_is_rejected_at_verify",
    ),
    Mutant(
        "M2-no-provenance",
        "the old schema: provenance outside the signed core, so a result can be relabelled",
        '"provenance": provenance.to_core(),',
        "",
        "test_swapping_the_backend_breaks_the_digest",
    ),
    Mutant(
        "M3-lattice-violation",
        "letting ACHIEVED exceed CLAIMED (breaks ACHIEVED = min(CLAIMED, REVERIFIABLE))",
        "achieved_final = max(0, min(int(claimed), int(achieved))) if valid else 0",
        "achieved_final = max(0, int(achieved)) if valid else 0",
        "test_claiming_less_than_you_can_prove_is_honoured",
    ),
    Mutant(
        "M4-digest-not-checked",
        "accepting a tampered core because the digest is not recomputed",
        'checks["digest"] = recomputed == passport["digest"]',
        'checks["digest"] = True',
        "test_tampering_with_the_claim_breaks_the_digest",
    ),
    Mutant(
        "M6-false-proof",
        "a certified lower bound that exceeds the true distance -- a FALSE PROOF",
        "if best_w is not None and lb >= best_w:\n            wit =",
        "if True:\n            wit =",
        "test_certified_bound_never_exceeds_true_distance",
        "isd",
    ),
    Mutant(
        "M7-stabilizer-as-logical",
        "INVERTING the logical/stabilizer filter -- returns a stabilizer as the witness",
        "keep = ~ls.is_stabilizer_batch(V)",
        "keep = ls.is_stabilizer_batch(V)",
        "test_proof_agrees_with_ibm_on_exact_codes",
        "isd",
    ),
    Mutant(
        "M8-reconstruction-unchecked",
        "returning a code whose k disagrees with the published catalogue",
        "if expect_k is not None and code.k != expect_k:",
        "if False:",
        "test_reconstruction_is_faithful_to_the_catalogue",
        "gbb",
    ),
    Mutant(
        "M9-deficiency-ignored",
        "charging nothing for rank deficiency -- inflates the Zimmermann bound into a FALSE PROOF",
        "info.contribution = max(0, p + 1 - info.deficiency)",
        "info.contribution = max(0, p + 1)",
        "test_certified_bound_never_exceeds_a_distance_ibm_proved",
        "bz",
    ),
    Mutant(
        "M10-enumeration-incomplete",
        "dropping the Y option per qubit -- the enumeration misses operators it claims to have covered",
        "t = np.indices((3,) * p).reshape(p, -1).T if p else np.zeros((1, 0), int)",
        "t = np.indices((2,) * p).reshape(p, -1).T if p else np.zeros((1, 0), int)",
        "test_enumeration_is_exhaustive_at_its_stated_depth",
        "bz",
    ),
    Mutant(
        "M11-weight-ignores-Z",
        "counting only the X half as qubit weight -- every weight the engine reports is wrong",
        "return popcount(V[..., :nwh] | V[..., nwh:], xp)",
        "return popcount(V[..., :nwh], xp)",
        "test_split_packing_roundtrips_and_weighs_correctly",
        "bz",
    ),
    Mutant(
        "M12-keeps-stabilizers",
        "inverting the stabilizer filter -- the certificate's witness is a stabilizer, not a logical",
        "keep = par.astype(bool)",
        "keep = ~par.astype(bool)",
        "test_certificate_is_self_contained_and_the_witness_is_logical",
        "bz",
    ),
    Mutant(
        "M13-no-catchup",
        "letting a set contribute a bound without enumerating its shallow depths",
        "todo = [(i, q) for i in active for q in range(swept[i] + 1, p + 1)]",
        "todo = [(i, q) for i in active for q in range(p, p + 1)]",
        "test_a_set_only_contributes_a_bound_it_enumerated_every_level_for",
        "bz",
    ),
    # ---- the verdict-producing engines added after the first 14: MITM, the SAT
    # lane, and the certify() router all decide closures and had NO mutant
    # proving their checks can fail. Ground truth for these is brute force
    # (tests/test_cross_engine.py), which shares no machinery with any engine.
    Mutant(
        "M14-mitm-drops-a-split",
        "skipping a (h1,h2) split -- the collision sweep silently stops being exhaustive",
        # Anchored on the condition ALONE. The previous anchor carried the
        # following line and the exact indentation, so a trailing comment added
        # later silently turned this mutant into a SKIP -- a gate quietly
        # dropping a check while still reporting a clean sweep.
        "if h1 + h2 > W + 1:",
        "if h1 + h2 > W + 1 or h2 == 1:",
        "test_mitm_withdraws_its_negative_claim_if_any_split_went_unswept",
        "mitm",
    ),
    Mutant(
        "M15-mitm-skips-stabilizer-rejection",
        "accepting a stabilizer as a logical -- MITM reports a distance that is not one",
        "        if ls.is_stabilizer(pack(v[None, :])[0]):\n            return None",
        "        if False:\n            return None",
        "test_mitm_returns_a_genuine_logical_not_a_stabilizer",
        "mitm",
    ),
    Mutant(
        "M16-sat-one-anchor-only",
        "checking a single translation anchor -- UNSAT no longer covers every operator",
        "    for case in (0, 1):\n        cnf, xv, zv = _build_instance(code, W, case)\n        with Solver(",
        "    for case in (0,):\n        cnf, xv, zv = _build_instance(code, W, case)\n        with Solver(",
        "test_sat_unsat_is_refused_unless_both_anchor_cases_were_decided",
        "satdist",
    ),
    Mutant(
        "M17-sat-drops-cardinality",
        "omitting the weight bound -- SAT answers a different question than the one asked",
        "    card = CardEnc.atmost(lits=wv, bound=W, vpool=pool, encoding=EncType.seqcounter)",
        "    card = CardEnc.atmost(lits=wv, bound=len(wv), vpool=pool, encoding=EncType.seqcounter)",
        "test_sat_matches_brute_force_both_directions",
        "satdist",
    ),
    Mutant(
        "M18-router-accepts-unwitnessed-target",
        "certifying a bound with no verified witness -- the anti-tautology guard removed",
        'dec.refusal = "no verified witness at the claimed bound -- refusing a target without evidence"',
        'dec.witness_weight = upper_bound; witness = np.zeros(2 * n, dtype=np.uint8)',
        "test_router_refuses_a_target_it_cannot_witness",
        "certify",
    ),
    Mutant(
        "M5-floats-allowed",
        "permitting floats in the signed core (cross-language formatting divergence)",
        'raise TypeError(\n            "floats are not permitted',
        'return obj\n    if False:\n        raise TypeError(\n            "floats are not permitted',
        "test_floats_are_refused_in_the_core",
    ),
    # ---- M19..M27: the modules built AFTER the original gate. Every test
    # written for them had never been proven able to fail -- the exact vacuity
    # the coverage contract exists to prevent, sitting inside the code that
    # implements it.
    Mutant(
        "M19-coverage-always-admissible",
        "the Negative Claim Admission Contract rubber-stamping every claim",
        "admissible = complete and corpus_can_trigger_target is not False",
        "admissible = True",
        "test_a_missing_partition_refuses_the_claim",
        "coverage",
    ),
    Mutant(
        "M20-coverage-empty-schedule-passes",
        "an empty required set counted as complete coverage -- the all([]) vacuity",
        "complete = (not omitted) and (not a_missing) and bool(req or a_req)",
        "complete = (not omitted) and (not a_missing)",
        "test_an_empty_schedule_cannot_support_a_negative_claim",
        "coverage",
    ),
    Mutant(
        "M21-spectrum-always-complete",
        "multiplicity declared complete below its own weight -- the retracted 18x",
        "return max(0, int(self.completed_depth))",
        "return 10**9",
        "test_spectrum_below_its_own_distance_is_partial_not_complete",
        "coverage",
    ),
    Mutant(
        "M22-ledger-unswept-set-contributes",
        "crediting a bound to an information set that was never enumerated",
        "return sum(max(0, int(depths.get(i, -1)) + 1 - int(d))",
        "return sum(max(0, int(depths.get(i, 10**9)) + 1 - int(d))",
        "test_an_unenumerated_set_earns_nothing",
        "ledger",
    ),
    Mutant(
        "M23-transversal-never-refuses",
        "a transversal certifier that certifies everything, witness or not",
        "if _is_logical(code, v, rankH, Hs):",
        "if False:",
        "test_it_REFUSES_when_a_logical_fits_inside_the_hot_set",
        "transversal",
    ),
    Mutant(
        "M24-survival-ignores-amplitude",
        "declaring a mixture observable from the SPECTRUM alone -- the overclaim "
        "a reviewer caught: at dwell 100 lambda2/lambda1 is 0.98 while the second "
        "amplitude is 1.4e-3 and the hazard is flat to 0.7%",
        "observable_mixture=bool(variation > 1.05 and amp > 1e-3),",
        "observable_mixture=True,",
        "test_FAST_switching_is_NOT_observable_despite_a_two_moded_spectrum",
        "survival",
    ),
    Mutant(
        "M25-lattice-permits-overclaim",
        "letting a claim be presented ABOVE its evidence -- the forgery, unblocked",
        "return int(claimed) > int(reverifiable)",
        "return False",
        "test_the_forgery_as_arithmetic",
        "lattice",
    ),
    Mutant(
        "M26-guard-layer-c-blind",
        "Layer C accepting a replayed syndrome from another round",
        "if sd != expected.syndrome_digest:",
        "if False:",
        "test_layer_c_catches_a_replayed_syndrome_that_A_and_B_both_pass",
        "guard",
    ),
    Mutant(
        # A FIRST ATTEMPT AT THIS MUTANT SURVIVED, and the lesson is about the
        # mutant, not the test. It flipped `if not a.admitted: return a`, which
        # is a REDUNDANT branch -- a refused admission falls through the expiry
        # loop and returns the same object anyway. It changed no behaviour, so
        # nothing could catch it. A mutant that alters nothing tests nothing,
        # and counting it as a survivor would have condemned a sound test.
        "M27-modelgate-defaults-to-admit",
        "an unregistered (model, quantity) pair silently ADMITTED -- the real "
        "rubber-stamp risk, since refusal-by-default IS the contract",
        'f"no admission on record for ({model}, {quantity}) "\n                             f"-- default is REFUSE, not \'probably fine\'")',
        'f"no admission on record") if False else Admission(model, quantity, True, "STAMPED")',
        "test_an_unknown_pair_is_refused_not_defaulted",
        "modelgate",
    ),
    Mutant(
        # Set EQUALITY is the whole certification. Weaken it to a subset
        # test and a block measuring MORE than its stabilizer -- a route
        # through a foreign data qubit -- certifies happily, and the
        # published overhead ratio describes a circuit that destroys the
        # codeword it claims to measure.
        "M55-a5-certifies-a-superset",
        "A5 accepts a block measuring more than its stabilizer",
        "    if measured != set(terminals):",
        "    if not set(terminals) <= measured:",
        "test_routing_through_a_FOREIGN_data_qubit_is_refused",
        "heavyhex_transpiler",
    ),
    Mutant(
        # Drop the scratch check and a block that leaves routing qubits
        # entangled certifies: it measures the right thing ONCE and
        # poisons every check scheduled after it.
        "M56-a5-ignores-dirty-scratch",
        "A5 accepts a block that leaves scratch qubits holding data",
        "        elif val[q] & data:",
        "        elif False:",
        "test_a_block_that_leaves_scratch_dirty_is_refused",
        "heavyhex_transpiler",
    ),
    Mutant(
        # A conserved quantity the stabilizers already imply carries no
        # information. Skip the rowspace test and A6's claim quietly
        # becomes "an extra redundant row helps" -- which its own q=0.5
        # control shows is false.
        "M57-a6-accepts-a-stabilizer-as-the-symmetry",
        "A6 picks a 'conserved quantity' inside the stabilizer group",
        "        if gf2_in_rowspace(xb, xp, v, n):",
        "        if False:",
        "test_conserved_quantity_lies_outside_the_stabilizer_group",
        "symmetry_decoding",
    ),
    # M58/M59 REMOVED as EQUIVALENT UNDER SINGLE MUTATION, not because
    # they were inconvenient. L2's solver carries two independent
    # soundness checks -- an inconsistency return during elimination, and
    # a full re-verification of f against every mechanism afterwards --
    # and they COVER EACH OTHER. Disable the elimination return and the
    # verification still rejects the bad f; disable the verification and
    # the elimination return still catches the inconsistent system. So no
    # single mutation can make either observable, and a mutant that
    # cannot fail is not evidence.
    #
    # That mutual redundancy is deliberate (the solver already shipped
    # wrong once), and the honest record is to say so rather than to
    # weaken a test until a survivor turns green.

    Mutant(
        # The complementarity control must actually perturb something. A
        # no-op corrupt_weight would make "L2' is 0% on weight faults"
        # true for the wrong reason and the layer would look complementary
        # without being tested at all.
        "M215-weight-corruption-is-a-no-op",
        "the complementarity control stops being a control",
        "            args[0] = min(0.49, args[0] * factor)",
        "            args[0] = args[0]",
        "test_weight_corruption_actually_changes_a_weight",
        "l2prime",
    ),

    # ---- M61..M65: the QX wave (2026-08-15). Every mutant sits on a
    # VERDICT line of a new accept path; each was chosen so the named
    # test fails through the mutated line itself, not a neighbor.
    Mutant(
        # The transplant forgery the portable receipt exists to refuse:
        # skip admissibility and a feasible packing for a different
        # (T, p) sails through with an honestly recomputed binding.
        "M210-receipt-skips-admissibility",
        "verify_receipt accepting a packing over rows this coset never satisfied",
        "    if not check_rows_admissible(syndrome, rows, p):",
        "    if False:",
        "test_forgery_wrong_parity_with_honest_rebinding",
        "coset_lower_check",
    ),
    Mutant(
        # The bnb's minimality half: without the attain check, a tree of
        # valid prunes certifies a value nothing achieves.
        "M211-bnb-value-need-not-be-attained",
        "verify_bnb accepting a claimed optimum no closed leaf reaches",
        "    if value not in attained:",
        "    if False:",
        "test_forged_value_is_refused",
        "coset_bnb_check",
    ),
    Mutant(
        # The bnb's optimality half: a prune whose bound stops covering
        # the claimed value must refuse, or the tree is a rubber stamp.
        "M212-bnb-prune-always-holds",
        "verify_bnb accepting a subtree bound weaker than the claim",
        "    if forced + bound < value:",
        "    if False:",
        "test_a_too_weak_prune_is_refused",
        "coset_bnb_check",
    ),
    Mutant(
        # ProofRank receipts re-derive the rank from the span; trusting
        # the stated number instead readmits forged proof progress.
        "M213-proofrank-trusts-the-stated-rank",
        "verify_proofrank accepting a rank the span does not support",
        "    if receipt.get(\"proof_rank\") != rank:",
        "    if False:",
        "test_proofrank_receipt_round_trips_and_refuses_forgeries",
        "logical_contrast",
    ),
    Mutant(
        # A zero sigma-sum with positive margin means INVULNERABLE, not
        # divisible: drop the guard and the radius computation crashes
        # (or worse, a future refactor returns a wrong finite number).
        "M214-radius-divides-by-zero-uncertainty",
        "fixed_witness_radius mishandling an invulnerable competitor",
        "    if denom == 0:",
        "    if False:",
        "test_no_uncertainty_on_the_difference_means_invulnerable",
        "robust_radius",
    ),
    # ---- Organ V: DEM-CERT, the twelve classes Sec 3.7.3 asks for
    Mutant(
        # The DEM is the shared ancestor of every decoder-assurance leg, so a
        # wrong signature here is not one wrong number -- it silently
        # invalidates everything computed downstream of the model.
        "M100-cx-propagates-x-but-not-z",
        "CX drops its Z rule: every Z fault before a CX loses its partner",
        'A[:, t] ^= A[:, c]\n                A[:, nq + c] ^= A[:, nq + t]',
        'A[:, t] ^= A[:, c]',
        "test_every_signature_matches_injected_ground_truth",
        "dem_cert",
    ),
    Mutant(
        # The direction is the whole content of the rule and a swap still
        # type-checks, still runs, and still produces a plausible DEM.
        "M101-cx-z-rule-points-the-wrong-way",
        "CX Z rule reversed: control and target exchanged",
        'A[:, nq + c] ^= A[:, nq + t]',
        'A[:, nq + t] ^= A[:, nq + c]',
        "test_every_signature_matches_injected_ground_truth",
        "dem_cert",
    ),
    Mutant(
        # CZ is never reached by rotated_memory_z, so this line shipped
        # unexercised until a case was aimed at it.
        "M102-cz-couples-a-qubit-to-itself",
        "CZ reading its own column: the two-qubit correlation disappears",
        'A[:, nq + b] ^= A[:, a]',
        'A[:, nq + b] ^= A[:, b]',
        "test_each_unexercised_gate_agrees_with_the_injection_oracle",
        "dem_cert",
    ),
    Mutant(
        # The table is HAND-DERIVED so stim is not the ancestor of both the
        # propagator and its ground truth. That is only safe if a typo in it
        # cannot survive.
        "M103-h-stops-exchanging-x-and-z",
        "H as identity on the Pauli frame",
        '"H": ("swap",), "H_XZ": ("swap",),',
        '"H": (), "H_XZ": ("swap",),',
        "test_the_single_qubit_table_matches_stims_own_definition",
        "dem_cert",
    ),
    Mutant(
        # The two ops are visually symmetric and either one alone looks
        # right; only the comparison against stim's tableau separates them.
        "M104-s-and-sqrt-x-swap-their-column-ops",
        "S given SQRT_X's action: a plausible, wrong Clifford table",
        '"S": ("z^=x",), "SQRT_Z": ("z^=x",),',
        '"S": ("x^=z",), "SQRT_Z": ("z^=x",),',
        "test_the_single_qubit_table_matches_stims_own_definition",
        "dem_cert",
    ),
    Mutant(
        # Backward, the reset must be undone BEFORE the measurement is read.
        # Getting that order wrong deletes every fault before an MR, which
        # reads downstream as a cleaner circuit.
        "M105-mr-stops-resetting",
        "measure-and-reset losing its reset: faults leak across the boundary",
        '        if name in _MEASURE_RESET:\n            # Backward order',
        '        if False:\n            # Backward order',
        "test_every_signature_matches_injected_ground_truth",
        "dem_cert",
    ),
    Mutant(
        # A partial reset is the hardest version to see: most signatures stay
        # correct and only the ones crossing the reset go wrong.
        "M106-mr-resets-only-the-x-component",
        "half a reset: the Z component survives a fresh qubit",
        '            A[:, q] = 0\n            A[:, nq + q] = 0\n        for r in rec_rows',
        '            A[:, q] = 0\n        for r in rec_rows',
        "test_every_signature_matches_injected_ground_truth",
        "dem_cert",
    ),
    Mutant(
        # This IS the defect found by building DEM-CERT. p/m leaves a 4.7e-4
        # relative gap against the compiler's own DEM -- small enough that a
        # tolerance would have swallowed it and certified the approximation.
        "M107-merge-convention-reverts-to-p-over-m",
        "depolarizing channels split as p/m instead of 1-(1-p)^(1/m)",
        'return 1.0 - (1.0 - p) ** (1.0 / m)',
        'return p / m',
        "test_the_merge_convention_is_not_p_over_m",
        "dem_cert",
    ),
    Mutant(
        # The Y branch is unreachable in every surface-code circuit here, so
        # it is exactly the kind of line that ships wrong and silent.
        "M108-y-basis-measurement-loses-its-z-component",
        "MY treated as MX: half of a Y-basis record disappears",
        '            if basis in ("Z", "Y"):',
        '            if basis in ("Z",):',
        "test_each_unexercised_gate_agrees_with_the_injection_oracle",
        "dem_cert",
    ),
    Mutant(
        # Skipping drops the faults that instruction carries. The certificate
        # then covers fewer faults than it claims and still verifies -- an
        # uncovered code path is an uncovered fault class.
        "M109-unknown-instruction-skipped-not-refused",
        "an unrecognised gate silently ignored, shrinking |F|",
        '        raise Refused(_why_refused(name))',
        '        pass',
        "test_an_unknown_instruction_is_refused_rather_than_ignored",
        "dem_cert",
    ),
    Mutant(
        # The same positional-desync class that produced a 795-second
        # branching certificate its own author could not verify.
        "M110-fault-order-and-signature-order-desync",
        "faults reversed but signatures not: every leaf names the wrong fault",
        '    faults.reverse()\n    sigs.reverse()',
        '    faults.reverse()',
        "test_every_signature_matches_injected_ground_truth",
        "dem_cert",
    ),
    Mutant(
        # The root IS the product -- a stranger compares 32 bytes and nothing
        # else. A root that survives a corrupted leaf is a constant.
        "M111-every-leaf-hashes-to-a-constant",
        "the Merkle leaf ignoring its content: one root for every input",
        '        stack.append((0, leaf_hash(leaf)))',
        '        stack.append((0, sha256_hex(_NODE)))',
        "test_the_root_changes_when_any_leaf_changes",
        "dem_cert",
    ),
    Mutant(
        # THE DEFECT THAT ALREADY SHIPPED HERE. run_g4.py recorded whether
        # QISKIT built a circuit and reported it as a provider capability;
        # IBM's own two documentation pages disagree about IBM in that gap.
        "M112-sdk-evidence-admitted-as-hardware",
        "the source filter removed: documentation admits plans",
        'if v["source"] == HARDWARE}',
        'if v["source"] in (HARDWARE, SDK)}',
        "test_an_sdk_sourced_field_can_never_be_admitted",
        "capability",
    ),
    Mutant(
        # A capability manifest expires exactly like a calibration; without
        # the check it is a permanent claim about a device that changes.
        "M113-capability-manifest-never-expires",
        "a capability measured in March admitted in September",
        'if now > manifest["probe_expiry"]:',
        'if False:',
        "test_an_expired_manifest_is_refused_like_a_stale_calibration",
        "capability",
    ),
    Mutant(
        # The direction is encoded in the field table precisely so a call
        # site cannot get it backwards; the table itself still can.
        "M114-a-floor-compared-as-a-ceiling",
        "at_least inverted: an unusable backend admitted silently",
        '    if direction == "at_least":\n        return (have >= want,',
        '    if direction == "at_least":\n        return (have <= want,',
        "test_requirement_directions_are_not_reversible",
        "capability",
    ),
    Mutant(
        # Asserted against the SAME manifest as M114 so one swapped
        # comparison cannot satisfy both.
        "M115-a-ceiling-compared-as-a-floor",
        "at_most inverted: a slow backend admitted for a latency-bound plan",
        '    if direction == "at_most":\n        return (have <= want,',
        '    if direction == "at_most":\n        return (have >= want,',
        "test_requirement_directions_are_not_reversible",
        "capability",
    ),
    Mutant(
        # A value with no probe is a documentation claim wearing a
        # manifest's clothes.
        "M116-a-value-may-carry-no-probe",
        "a manifest entry with no probe behind it accepted as measured",
        'if not isinstance(entry["probe"], str) or not entry["probe"]:',
        'if False:',
        "test_a_value_with_no_probe_named_is_malformed",
        "capability",
    ),
    Mutant(
        # One bool cannot express operand width, nesting, measure- and
        # reset-in-conditional, or arithmetic. A field that silently does
        # nothing is how the SDK/hardware confusion got in the first time.
        "M117-the-deleted-boolean-returns",
        "supports_dynamic_control_flow readmitted as an ignored key",
        'for banned, why in BANNED.items():',
        'for banned, why in {}.items():',
        "test_the_deleted_boolean_cannot_come_back",
        "capability",
    ),
    Mutant(
        # Nothing can distinguish a measured 63 us from a remembered one --
        # but a hardware claim that cannot name the job that produced it was
        # not executed, and that IS checkable.
        "M118-a-hardware-claim-needs-no-job",
        "hardware_executable accepted without a job id: a TYPED manifest",
        'if entry["source"] == HARDWARE and not entry.get("job_id"):',
        'if False:',
        "test_a_hardware_claim_that_cannot_name_its_job_is_malformed",
        "capability",
    ),
    Mutant(
        # A capability cannot be executed by a laptop. This mutant is the
        # single edit that would turn the whole free half into theatre.
        "M119-offline-probe-labels-itself-as-hardware",
        "the free SDK run claiming hardware_executable for every field",
        '"source": SDK,\n                                   "probe"',
        '"source": HARDWARE,\n                                   "probe"',
        "test_the_offline_manifest_ACCEPTS_NOTHING",
        "capability_probe",
    ),
    Mutant(
        # THE FRECHET COROLLARY, DELETED. Every decoder-assurance leg has the
        # DEM in its ancestor set, so no union of them bounds the failure
        # below the model's own error. Without this cap the ladder in
        # dem_admission goes back to being a grade nobody consulted.
        "M150-dem-floor-never-caps",
        "a receipt certified PROVEN against a model nobody graded",
        '    if dem_level >= min_dem_level:\n        return frontier',
        '    if True:\n        return frontier',
        "test_a_DEM_below_A3_caps_the_receipt_at_REPLAY",
        "soundness_budget",
    ),
    Mutant(
        # Understating is free and overstating is the forgery this repository
        # was audited for. This is the line that makes lattice.py's rule
        # an exception on the minting path rather than a sentence in a
        # docstring.
        "M151-passport-block-admits-an-overclaim",
        "a block issued for a claim the frontier does not support",
        '    if supported is None or claimed_level > supported:',
        '    if False:',
        "test_a_DEM_below_A3_caps_the_receipt_at_REPLAY",
        "soundness_budget",
    ),
    Mutant(
        # A caller can construct the block by hand, so the producer having
        # checked is not a reason for the signed core not to. The core is the
        # LAST place a claim can be inflated.
        "M152-minting-trusts-the-producers-block",
        "build_core accepting a hand-built block that overclaims",
        '        if int(claimed_level) > supported:',
        '        if False:',
        "test_the_minting_path_REFUSES_an_overclaim_even_from_a_handbuilt_block",
        "passport",
    ),
    Mutant(
        # Two numbers for one claim is how a claim gets inflated between
        # them -- and whichever a reader happens to look at is the one they
        # will believe.
        "M153-two-claims-may-disagree",
        "the block and the passport allowed to claim different levels",
        '        if claimed_in_block != int(claimed_level):',
        '        if False:',
        "test_two_numbers_for_one_claim_are_refused",
        "passport",
    ),
    Mutant(
        # Defaulting an unknown tier to the weakest grade LOOKS conservative.
        # It is how an unknown producer becomes a trusted one, because the
        # grade it gets is still a grade.
        "M154-unknown-tier-defaults-instead-of-refusing",
        "an unrecognised producer tier silently graded",
        '    if tier not in TIER_EVIDENCE:',
        '    if False:',
        "test_an_unrecognised_tier_is_REFUSED_not_defaulted",
        "soundness_budget",
    ),
    Mutant(
        # UNKNOWN IS NOT WEAK. With this branch gone, a receipt whose
        # robustness nobody computed keeps the PROVEN grade its tier would
        # have had in exact arithmetic -- which is the one case where the
        # float check can accept a marginally infeasible dual.
        "M155-missing-robustness-margin-treated-as-present",
        "a float dual with no eps_q < rho answer graded PROVEN anyway",
        '        if rho_exceeds_quantisation is None:',
        '        if False:',
        "test_a_float_dual_with_NO_ANSWER_about_the_margin_is_refused",
        "soundness_budget",
    ),
    Mutant(
        # Silently composing one leg is SOUND -- one leg's bound is never
        # tighter than the disjunction's -- so no verdict test catches it,
        # and the receipt simply stops getting credit for the tiers it
        # earned. A conservative bug is still a bug.
        "M156-only-the-first-leg-composes",
        "every leg after the first dropped from the frontier",
        '    for leg in legs[1:]:\n        out = out.alt(Frontier.leg(leg.level, leg.beta_ppb))',
        '    for leg in []:\n        out = out.alt(Frontier.leg(leg.level, leg.beta_ppb))',
        "test_the_legs_are_composed_by_FRECHET_and_the_block_says_so",
        "soundness_budget",
    ),
    Mutant(
        # THE ARITHMETIC GATE ONLY BITES WHERE IT CAN CHANGE THE GRADE.
        # Delete the early return and `degraded` -- REPLAY whatever the
        # arithmetic -- is refused again for failing to settle the
        # exactness of a proof it never claimed. A gratuitous refusal is
        # not a soundness hole, which is exactly why no verdict test sees
        # it and why it survived until the flag was made optional.
        "M157-replay-tier-still-demands-a-margin",
        "a tier that is not an optimality proof refused over eps_q < rho",
        '    if level is not L.PROVEN:',
        '    if False:',
        "test_a_tier_that_is_not_an_optimality_proof_is_not_asked_at_all",
        "soundness_budget",
    ),
    Mutant(
        # *** RE-AIMED, AND THE FIRST AIM IS THE FINDING. *** It replaced
        # `known` with True, meaning to catch "a tier whose arithmetic
        # nobody decided is graded PROVEN by default". It SURVIVED, and
        # not because the test was weak: the branch above raises when
        # `known is None`, so this line is only reached when `known` is
        # already True. `= True` and `= known` are THE SAME VALUE on
        # every reachable path -- an EQUIVALENT mutant, unkillable by
        # construction, and a test written to kill it could only have
        # been a test of something else.
        #
        # Re-aimed at the direction that is actually reachable: a
        # derivation that returns the WRONG value. False costs the
        # capability the derivation exists to provide -- a
        # certified-optimal receipt with the flag omitted is refused for
        # want of a margin it does not need. Conservative, sound, and
        # exactly the class the ratchet was built for.
        #
        # The dangerous direction -- deriving EXACT for a float-checked
        # tier -- is unreachable while every determined entry is True,
        # which is the same statement M159's killer makes explicitly.
        "M158-derivation-returns-the-wrong-arithmetic",
        "the derived arithmetic wrong, so a proven tier is refused",
        '        exact_arithmetic = known',
        '        exact_arithmetic = False',
        "test_the_arithmetic_is_DERIVED_not_asked_of_the_caller",
        "soundness_budget",
    ),
    Mutant(
        # The one direction that inflates: a caller asserting EXACT about
        # a tier this module knows is float-checked. Nothing in the
        # shipped vocabulary can reach it, so the killer builds the tier
        # -- a guard for a case that does not exist yet still has to be
        # shown to work, or it is decoration.
        "M159-caller-may-assert-exact-over-a-float-tier",
        "a float-checked tier graded on the caller's word that it is exact",
        '    elif exact_arithmetic and known is False:',
        '    elif False:',
        "test_the_overclaim_branch_names_what_would_make_it_live",
        "soundness_budget",
    ),
    Mutant(
        # THE FRECHET COROLLARY'S TEETH, PULLED. With the re-derivation
        # gone, an admission certificate is believed at its own headline
        # -- exactly the habit the function exists to break -- and a
        # record claiming A5 over A0 rungs grades a model from nothing.
        "M160-admission-certificate-believed-not-re-derived",
        "a DEM certificate's own headline trusted over its rungs",
        '        if stated is not None and int(stated) != int(level):',
        '        if False:',
        "test_a_certificate_that_contradicts_its_own_rungs_is_REFUSED",
        "soundness_budget",
    ),
    Mutant(
        # An empty ladder re-derives to nothing. Grading a model from
        # nothing is precisely what the ladder exists to stop, and the
        # refusal is the only thing standing between the two.
        "M161-empty-ladder-grades-a-model",
        "a certificate with no rungs accepted as an admission",
        '        if not rungs:',
        '        if False:',
        "test_an_empty_ladder_cannot_grade_a_model",
        "soundness_budget",
    ),
    Mutant(
        # VERIFIED AND UNVERIFIED MUST NOT SHARE A NAME. `evidence_verified`
        # false means nobody opened the artifacts the rungs name -- four
        # fictional paths once graded to A3 on exactly that. If both
        # report the same binding the signed block cannot tell them apart.
        "M162-unverified-rungs-reported-as-verified",
        "a re-derivation from unopened artifacts labelled verified",
        '        if dem_level.get("evidence_verified"):',
        '        if True:',
        "test_UNVERIFIED_rungs_are_named_differently_from_verified_ones",
        "soundness_budget",
    ),
    Mutant(
        # The FLOOR must be applied at the re-derived level. Compute it
        # from the caller's number instead and a certificate can be handed
        # over for the record while the cap still comes from a friendlier
        # figure -- the artifact says re_derived and the maths does not.
        "M163-floor-computed-from-the-callers-number",
        "the DEM cap taken from the asserted level, not the re-derived one",
        'passport_block(dem_floor(compose(legs), dem_int),',
        'passport_block(dem_floor(compose(legs), 5),',
        "test_a_re_derived_LOW_grade_makes_the_floor_bite",
        "soundness_budget",
    ),
    Mutant(
        # beta must be NON-DECREASING in the verification grade: a pickier
        # verifier admits FEWER legs, so it cannot support a tighter bound.
        # A dipping frontier claims a stronger checker can prove MORE, which
        # is the scalar defect this object was built to make inexpressible --
        # reintroduced one layer down.
        "M144-frontier-monotonicity-unchecked",
        "a frontier that TIGHTENS with grade accepted as well-formed",
        '            if hi < lo:',
        '            if False:',
        "test_a_frontier_that_TIGHTENS_with_grade_is_malformed",
        "evidence_frontier",
    ),
    Mutant(
        # The second half of the reported soundness bug: the scalar form let
        # a leg declared INVALID donate the winning beta, because validity
        # was a third scalar that took no part in the arithmetic. Here it is
        # arithmetic -- an invalid leg IS the constant no-bound -- and this
        # mutant turns it back into a flag nobody consults.
        "M145-invalid-leg-still-contributes",
        "an INVALID leg donating a bound instead of contributing nothing",
        '        if not valid:\n            return cls.nothing()',
        '        if False:\n            return cls.nothing()',
        "test_an_INVALID_leg_contributes_nothing_anywhere",
        "evidence_frontier",
    ),
    Mutant(
        # THE REPORTED DEFECT, at its source. alt(PROVEN@1000, REPLAY@1)
        # returned PROVEN@1 because the REPLAY leg's bound was available at
        # every grade. The step function is what forbids it; flatten the step
        # and the whole repair is undone while every scalar still looks right.
        "M146-leg-supports-above-its-own-grade",
        "a leg lending its bound to grades it cannot be checked at",
        '        return cls(tuple(beta_ppb if lv <= level else PPB for lv in LEVELS))',
        '        return cls(tuple(beta_ppb for lv in LEVELS))',
        "test_a_leg_supports_NOTHING_above_its_own_grade",
        "evidence_frontier",
    ),
    Mutant(
        # Discarding it is SOUND -- the re-synthesised step is never tighter
        # than the truth -- which is exactly why no verdict test catches it.
        # It silently throws away the tighter bounds available at lower
        # grades, so a chain of compositions degrades for no reason. A
        # conservative bug is still a bug.
        "M147-composed-frontier-discarded",
        "a composed frontier thrown away and re-synthesised from scalars",
        '        if self.composed is not None:\n            return self.composed',
        '        if False:\n            return self.composed',
        "test_the_composed_frontier_survives_a_CHAIN",
        "evidence_algebra",
    ),
    Mutant(
        # Both arguments default that way, so the DEFAULT call performed no
        # mask check at all and still reported 'verified against the
        # provisioned table' -- in the layer whose entire purpose is the
        # class that check covers.
        "M148-mask-check-may-silently-not-run",
        "a requested mask-replica check skipped for want of masks",
        '    if check_mask_replica and masks_read is None:',
        '    if False:',
        "test_a_REQUESTED_mask_check_that_cannot_run_is_a_REFUSAL",
        "abft_l0",
    ),
    Mutant(
        # A sticky comparator holds its reference. Committing against some
        # other table compares two things that never met, and accepting it
        # would make the reference a parameter rather than the authority.
        "M149-accumulator-bound-to-any-table",
        "a commit accepted against a table the accumulator never read",
        '    if acc.table is not table:',
        '    if False:',
        "test_an_accumulator_built_against_a_DIFFERENT_table_is_refused",
        "abft_l0",
    ),
    Mutant(
        # THE ONE THAT MATTERS. Frechet's min is the entire reason DEM
        # admission is upstream of decoder assurance; replacing it with the
        # product is the intuition everyone starts with, and on two checks
        # sharing the DEM it claims PERFECT soundness from two imperfect
        # ones.
        "M120-or-multiplies-instead-of-frechet",
        "OR takes the naive product: redundancy becomes free",
        '        return Frontier(tuple(min(a, b)\n                              for a, b in zip(self.beta, other.beta)))',
        '        return Frontier(tuple(min(PPB, (a * b) // PPB)\n                              for a, b in zip(self.beta, other.beta)))',
        "test_redundant_decoder_checks_earn_NOTHING_over_the_better_one",
        "evidence_frontier",
    ),
    Mutant(
        # Covering one of two shared ancestors and multiplying the rest is
        # exactly the correlation the witness was supposed to rule out. The
        # caller believes they bought independence and did not.
        "M121-partial-independence-witness-accepted",
        "a witness covering part of the shared ancestry silently believed",
        '    elif witness is not None and shared:',
        '    elif False:',
        "test_a_PARTIAL_witness_is_refused_rather_than_pro_rated",
        "evidence_algebra",
    ),
    Mutant(
        # Reporting the larger number is not conservative here -- it discards
        # a bound the inputs already prove, and a witness whose arithmetic
        # lands above Frechet is wrong about something.
        "M122-a-witness-may-raise-a-bound",
        "a wrong witness allowed to report a WEAKER bound than Frechet",
        '            out.append(min(frechet, candidate))',
        '            out.append(candidate)',
        "test_a_witness_may_LOWER_a_bound_and_never_RAISE_it",
        "evidence_frontier",
    ),
    Mutant(
        # The composed claim holds in the INTERSECTION of the domains, which
        # is empty. HyperBlossom's 4.8x is code-capacity and has been quoted
        # as though it were circuit-level.
        "M123-domain-intersection-ignored-on-and",
        "a code-capacity result chained onto a circuit-level one",
        '    if a.noise_class != b.noise_class:\n        raise Unsound(\n            f"cannot chain',
        '    if False:\n        raise Unsound(\n            f"cannot chain',
        "test_composing_across_noise_classes_is_refused",
        "evidence_algebra",
    ),
    Mutant(
        # The blindness law. `H x_hat = s` on a syndrome-consistent decoder
        # has PROVABLY ZERO coverage -- measured 0/3,000 and 0/16,000 -- and
        # without the cap it composes as though it proved something.
        "M124-blind-leg-not-capped",
        "a check with no construction_witness presented as PROVEN",
        '        if self.construction_witness:\n            return self.level\n        return min(self.level, L.REPLAY)',
        '        return self.level',
        "test_a_leg_with_no_construction_witness_is_CAPPED_not_dropped",
        "evidence_algebra",
    ),
    Mutant(
        # The assurance-ceiling theorem, deleted. Without it a leg that saw
        # 3,000 injections can assert 1e-9, and the reason DEM-CERT is
        # exhaustive rather than sampled evaporates.
        "M125-rule-of-three-floor-dropped",
        "a measured leg claiming below 3/n with zero misses",
        '        return max(self.beta_ppb, self.floor_ppb)',
        '        return self.beta_ppb',
        "test_a_measured_leg_carries_the_rule_of_three_floor",
        "evidence_algebra",
    ),
    Mutant(
        # A bound that rounds down is not a bound. The error is one ppb and
        # it is in the unsafe direction, which is the only size that matters.
        "M126-bridge-rounds-down",
        "the bridge truncating instead of rounding up",
        '    return min(PPB, -(-num // p_a_ppb))',
        '    return min(PPB, num // p_a_ppb)',
        "test_the_bridge_rounds_UP_because_a_bound_that_rounds_down_is_not_one",
        "evidence_algebra",
    ),
    Mutant(
        # Integers only. A float inside a composed bound is a value two
        # languages can disagree about while both believing they agree.
        "M127-beta-accepts-a-float",
        "a float admitted into a composed bound",
        '        if not isinstance(self.beta_ppb, int) or isinstance(self.beta_ppb, bool):',
        '        if False:',
        "test_beta_must_be_an_integer_count_of_parts_per_billion",
        "evidence_algebra",
    ),
    Mutant(
        # THE TEMPTING ONE-LINER, and the whole reason the ladder exists.
        # `max(levels)` reports A5 for a model whose only other rung is A0.
        # A level that is not earned from below is a list of adjectives.
        "M128-ladder-takes-the-max-not-the-run",
        "a gap stepped over: A0+A3 reported as A3",
        '    level = 0\n    while level + 1 <= MAX_LEVEL and (level + 1) in have:\n        level += 1',
        '    level = max(have)',
        "test_a_gap_CAPS_the_ladder_rather_than_being_stepped_over",
        "dem_admission",
    ),
    Mutant(
        # 67.7% of undecomposed mechanisms at d=11 are TRUE HYPEREDGES, so a
        # graphlike DEM is a different model rather than a view of the same
        # one. The cap is honest because the error is a finite calculation.
        "M129-graphlike-cap-removed",
        "a graphlike decomposition admitted at A3 with no certified error",
        '    if decomposition == "graphlike" and decomposition_error_log10_micro is None:',
        '    if False:',
        "test_graphlike_is_capped_at_A2_without_a_certified_error",
        "dem_admission",
    ),
    Mutant(
        # A restriction that promotes an A1 model to A2 is an upgrade wearing
        # a cap's name, and it fires exactly on the models with the least
        # evidence.
        "M130-cap-allowed-to-raise-a-level",
        "the decomposition cap rounding UP to its own ceiling",
        '        capped = min(level, 2)',
        '        capped = 2',
        "test_the_cap_never_RAISES_a_level",
        "dem_admission",
    ),
    Mutant(
        # A level whose evidence cannot be named is a level nobody can check,
        # which is the same as no level at all -- except that it reads as one.
        "M131-a-rung-may-name-no-artifact",
        "a level claimed with nothing a stranger could open",
        '        if not self.artifact:',
        '        if False:',
        "test_a_rung_with_no_artifact_is_refused",
        "dem_admission",
    ),
    Mutant(
        # A0 is the rung everything else stands on. Without it the walk starts
        # at 0 regardless, and reports A0 for a model that has not been read.
        "M132-unparsed-dem-still-graded",
        "a DEM that has not been shown to parse graded anyway",
        '    if 0 not in have:',
        '    if False:',
        "test_a_dem_that_does_not_even_parse_admits_nothing",
        "dem_admission",
    ),
    Mutant(
        # It cannot be capped correctly if nobody knows what it is, so passing
        # it through grants the undecomposed ceiling to a model nobody has
        # characterised.
        "M133-unknown-decomposition-passed-through",
        "an unrecognised decomposition neither capped nor refused",
        '    elif decomposition not in ("undecomposed", "graphlike"):',
        '    elif False:',
        "test_an_unrecognised_decomposition_is_refused_not_passed_through",
        "dem_admission",
    ),
    Mutant(
        # A4 and A5 are fitted to data taken on a day and a device drifts.
        # Without the cap, a model admitted in March keeps claiming A5
        # forever -- which is the failure mode a calibration expiry exists
        # for, moved one layer up.
        "M140-device-fit-never-expires",
        "a held-out fit taken in March still admitting in September",
        '    stale = device_rungs_expired(rungs, now)\n    if stale and level > CIRCUIT_LEVEL_CEILING:',
        '    stale = device_rungs_expired(rungs, now)\n    if False:',
        "test_an_expired_device_rung_caps_the_ladder_at_the_circuit_ceiling",
        "dem_admission",
    ),
    Mutant(
        # A0-A3 are mathematics about a FIXED circuit; they do not go stale.
        # An expiry there is theatre, and theatre is precisely how a real
        # expiry gets switched off by the next person who trips over it.
        "M141-circuit-rungs-made-to-expire",
        "an exhaustive fault-signature proof discarded because a date passed",
        '        if r.level > CIRCUIT_LEVEL_CEILING and r.expires_on',
        '        if r.expires_on',
        "test_an_expiry_on_a_CIRCUIT_level_rung_is_inert",
        "dem_admission",
    ),
    Mutant(
        # An expiry that always fires is a deletion with a date on it, and
        # one that never fires is a comment. The inverted comparison is both
        # at once, and it still type-checks.
        "M142-expiry-comparison-inverted",
        "the date comparison reversed: in-date rungs expire and stale ones do not",
        '                  and now > r.expires_on)',
        '                  and now < r.expires_on)',
        "test_a_device_rung_still_in_date_admits_normally",
        "dem_admission",
    ),
    Mutant(
        # `now=None` means nobody asked what time it is. Returning [] for
        # every call collapses that into 'it is fine', which is the same
        # answer for 'not checked' and 'checked and current'.
        "M143-missing-clock-treated-as-in-date",
        "a caller who supplied no clock silently given a pass",
        '    if now is None:\n        return []',
        '    if True:\n        return []',
        "test_an_expired_device_rung_caps_the_ladder_at_the_circuit_ceiling",
        "dem_admission",
    ),
    Mutant(
        # THE HIGHEST-CONSEQUENCE FIELD. One bit in one edge's observable
        # mask at d=5 takes p_L from 1.4e-4 to 2.5e-3 -- 17.9x, and
        # distance-independent. A tag that omits it still passes every
        # weight-only test.
        "M134-mask-dropped-from-the-edge-tag",
        "the observable mask left out of H(e || w || mask || endpoints)",
        'body = (f"{self.eid}|{self.weight}|{self.mask}|"',
        'body = (f"{self.eid}|{self.weight}|"',
        # SURVIVED ITS FIRST RUN against
        # test_a_single_bit_flip_in_any_field_is_caught, and the
        # survival was the finding: that test's mask case is
        # satisfied by the dedicated REPLICA, so the tag's mask_e
        # term had no test of its own and could be deleted in
        # silence. The killer below drives the tag with the replica
        # switched off, which is the only way to isolate it.
        "test_the_TAG_ALONE_covers_the_mask",
        "abft_l0",
    ),
    Mutant(
        # Sec 3.7.4: without the replica the check is VACUOUS against exactly
        # the class it targets. Measured, not quoted: 32/32 caught with it,
        # 0/32 without.
        "M135-mask-replica-never-consulted",
        "the dedicated replica check skipped: the P1 class goes dark",
        '    if check_mask_replica and masks_read is not None:',
        '    if False:',
        "test_WITHOUT_the_mask_replica_the_check_is_vacuous_on_masks",
        "abft_l0",
    ),
    Mutant(
        # An empty accumulator agrees with an empty reference. That is the
        # vacuous pass an XOR scheme falls into first, and it is the state a
        # crashed decode leaves behind.
        "M136-empty-active-set-passes",
        "a decode that read no edge committing successfully",
        '    if not acc.touched:',
        '    if False:',
        "test_reading_no_edge_at_all_is_a_REFUSAL_not_a_pass",
        "abft_l0",
    ),
    Mutant(
        # P6: a stale binary reading a graph the provisioned table does not
        # describe. Skipping the edge removes it from BOTH sides and passes.
        "M137-unprovisioned-edge-silently-skipped",
        "an edge outside the provisioned table dropped from the reference",
        '        if ref is None or edge.tag() != ref:',
        '        if ref is not None and edge.tag() != ref:',
        "test_an_edge_outside_the_provisioned_table_is_refused",
        "abft_l0",
    ),
    Mutant(
        # THIS MUTANT CHANGED DIRECTION WHEN THE MODULE WAS REBUILT,
        # and re-aiming it from memory got it backwards. Against the
        # old PARITY accumulator, deleting the XOR made the check
        # REFUSE EVERY INPUT, so the clean-run test was the killer.
        # Against the STICKY COMPARATOR, pinning the bit to False
        # makes it ACCEPT EVERYTHING, so the killer is a corruption
        # test. Same id, same one-line edit, opposite failure -- a
        # re-aim is a measurement, not an edit, and this one survived
        # once before being measured.
        "M138-detector-never-fires",
        "the sticky fault bit pinned False: every corruption commits",
        '            self.fault_seen = True',
        '            self.fault_seen = False',
        "test_EVERY_single_bit_flip_on_the_active_subset_is_caught",
        "abft_l0",
    ),
    # ---- the exact error budget: every term charged to the WRONG stage ----
    Mutant(
        # THE MISTAKE THE OLD p_search/p_model COUNTING MADE, restored.
        # `A != E` charges search for disagreeing with the exact
        # optimum -- but a heuristic can miss its own objective and be
        # accidentally RIGHT, and charging it there sends a model bug
        # to the decoder team. Only L_A - L_E charges what search
        # CAUSED, and it is negative on a lucky shot.
        "M164-search-charged-for-disagreement-not-causation",
        "delta_S counts A != E, so a heuristic that got lucky is billed",
        '        return {"delta_S": self.loss_a - self.loss_e,',
        '        return {"delta_S": int(self.a != self.e),',
        "test_search_is_NOT_charged_for_being_accidentally_right",
        "error_budget",
    ),
    Mutant(
        # r_B is the risk NO decoder can remove. Reading q for it
        # inflates the irreducible term above 1/2 whenever the model is
        # confident, which makes a miscalibrated model look like
        # unavoidable physics -- the one direction that excuses a
        # defect instead of finding it.
        "M165-irreducible-risk-reads-q-not-min",
        "R_I = q instead of min(q, 1-q): confidence read as ambiguity",
        '        return min(self.q, 1 - self.q)',
        '        return self.q',
        "test_R_I_is_min_q_1_minus_q_and_is_derived_not_accepted",
        "error_budget",
    ),
    Mutant(
        # The per-shot sum is the ONLY place a zip that drifted by one
        # is visible: all four terms stay in range and every marginal
        # still looks sane. Disarming the check leaves a budget that
        # adds up to the wrong loss and says nothing.
        "M166-telescoping-identity-never-checked",
        "assert_identity cannot raise: terms from different shots pass",
        '    if total != shot.loss_a:',
        '    if False:',
        "test_terms_that_came_from_DIFFERENT_shots_are_caught_by_the_sum",
        "error_budget",
    ),
    Mutant(
        # A model that called something impossible and watched it
        # happen is refuted by ONE observation. Not counting it turns
        # the strongest evidence in the instrument into a Z score that
        # declines to run.
        "M167-certain-and-wrong-not-counted",
        "a zero-risk shot that was WRONG stops being a refutation",
        '            if s.loss_b:',
        '            if False:',
        "test_a_model_that_called_it_impossible_and_was_wrong_is_REFUTED",
        "error_budget",
    ),
    Mutant(
        # Certain shots have zero predicted variance. Letting one into
        # the sample does not make Z wrong -- it makes Z REFUSE, taking
        # every informative shot down with it, which reads as "no
        # signal" rather than "one shot was excluded".
        "M168-zero-variance-shots-enter-the-statistic",
        "a certain shot is counted as a Z_M observation",
        '            continue                 # zero variance: not a Z observation',
        '            pass',
        "test_certain_shots_do_not_DESTROY_Z_M_for_everything_else",
        "error_budget",
    ),
    Mutant(
        # 16 cells, not 8. Collapsing Y loses the joint distribution
        # the cube exists for: the marginals are recoverable from the
        # cube and the cube is not recoverable from the marginals.
        "M169-cube-collapses-the-true-class",
        "the cube drops Y: 8 cells reported as the contingency cube",
        '    for a in (0, 1) for e in (0, 1) for b in (0, 1) for y in (0, 1))',
        '    for a in (0, 1) for e in (0, 1) for b in (0, 1) for y in (0,))',
        "test_all_sixteen_cells_are_present_even_at_zero",
        "error_budget",
    ),
    Mutant(
        # A slice whose Z REFUSED has no |Z| to compare. Ranking a
        # refutation by its Z instead of ahead of every Z puts the one
        # observation that ENDS the question behind a large well-behaved
        # region -- and a model that called something impossible and
        # watched it happen is not a matter of degree.
        "M172-a-refutation-is-ranked-as-if-it-were-a-Z",
        "worst_slice stops promoting an outright refutation",
        '    if refuted:',
        '    if False:',
        "test_worst_slice_ranks_an_OUTRIGHT_REFUTATION_above_any_Z",
        "error_budget",
    ),
    Mutant(
        # Conditioning that puts every shot in one slice is the global
        # statistic wearing a slice's name, and it looks exactly like a
        # working slice until a fault is LOCAL -- which is the only case
        # slicing exists for.
        "M173-every-shot-lands-in-one-slice",
        "by() ignores the key: conditioning silently becomes global",
        '        groups.setdefault(key(s), []).append(s)',
        '        groups.setdefault(None, []).append(s)',
        "test_a_LOCAL_fault_is_louder_in_its_slice_than_globally",
        "error_budget",
    ),
    # ---- the join: a budget past the exact tier's ceiling ----
    Mutant(
        # r_B = min(q, 1-q) IS NOT MONOTONE. Evaluating it at the two
        # endpoints brackets a monotone function and understates this
        # one: on an interval straddling 1/2 the maximum is 1/2, in the
        # INTERIOR. Understating the irreducible risk is the one
        # direction that makes a decoder look better than the model
        # permits, and the arithmetic still looks like a bracket.
        "M202-r-B-bracketed-at-its-endpoints",
        "the non-monotone peak at q=1/2 is missed",
        '    top = HALF if q_lo <= HALF <= q_hi else max(ends)',
        '    top = max(ends)',
        "test_r_B_over_a_STRADDLING_interval_peaks_in_the_INTERIOR",
        "enclosure_budget",
    ),
    Mutant(
        # SUBTRACTION CROSSES THE ENDS. delta_M = L_B - r_B, so its LOW
        # end pairs with r_B's HIGH end. Pairing them the same way round
        # is the ordinary interval-arithmetic slip and reports a
        # narrower bracket than the evidence supports.
        "M203-delta-M-bracket-does-not-cross",
        "the delta_M interval is narrowed by pairing ends the same way",
        '        out.delta_M_lo += l_b - r_hi\n        out.delta_M_hi += l_b - r_lo',
        '        out.delta_M_lo += l_b - r_lo\n        out.delta_M_hi += l_b - r_hi',
        "test_delta_M_pairs_its_LOW_end_with_r_B_s_HIGH_end",
        "enclosure_budget",
    ),
    Mutant(
        # AN UNDECIDABLE CLASS IS NOT A CLASS. Counting the shot anyway
        # would need a Bayes call the enclosure explicitly refused to
        # make, and the budget would be built on a guess wearing a
        # certificate's type.
        "M204-an-undecidable-shot-is-budgeted-anyway",
        "a straddling enclosure is treated as decided",
        '        if b is None:\n            out.undecided += 1\n            continue',
        '        if b is None:\n            b = 0',
        "test_an_UNDECIDED_shot_is_counted_and_EXCLUDED",
        "enclosure_budget",
    ),
    # ---- a disjunction across domains, on the path that MINTS ----
    Mutant(
        # THE ALGEBRA GOES BACK TO BEING AN INSTRUMENT. `Frontier.alt`
        # is arithmetic and returns a frontier for a code-capacity leg
        # ORed with a circuit-level one, because a min over two integers
        # cannot know they are about different worlds. Skipping the
        # route leaves the refusal implemented, tested, gated -- and
        # never reached by the function that feeds `passport_block`.
        "M200-compose-never-routes-through-the-algebra",
        "the minting path composes without the domain refusal",
        '    if len(legs) > 1 and all(d is not None for d in declared):',
        '    if False:',
        "test_a_CROSS_DOMAIN_or_is_REFUSED_on_the_minting_path",
        "soundness_budget",
    ),
    Mutant(
        # THE OR's OWN DOMAIN CHECK. M123 aims at the AND ("cannot
        # chain"); the OR's ("cannot OR") had no mutant at all -- found
        # by reading the registry rather than by a red, which is the
        # only way a missing aim is ever found.
        "M201-the-OR-accepts-two-different-domains",
        "alt() stops refusing a disjunction across noise classes",
        '    if a.noise_class != b.noise_class:\n        raise Unsound(\n            f"cannot OR',
        '    if False:\n        raise Unsound(\n            f"cannot OR',
        "test_a_CROSS_DOMAIN_or_is_REFUSED_on_the_minting_path",
        "evidence_algebra",
    ),
    # ---- L0 on a decode path: three ways to guard nothing ----
    Mutant(
        # THE GUARD NEVER RUNS. Every structure the decoder derives
        # comes from one read of the edge list, so skipping the check
        # there decodes against a graph nobody provisioned -- and the
        # syndrome side cannot see it, which is the entire reason this
        # layer exists.
        "M198-the-l0-guard-is-never-invoked",
        "a supplied L0 table is accepted and never checked",
        '        if l0_table is not None:\n            self._l0_verify(edges, l0_table, l0_masks)',
        '        if False:\n            self._l0_verify(edges, l0_table, l0_masks)',
        "test_a_flipped_OBSERVABLE_MASK_refuses_the_decoder",
        "certifying_decoder",
    ),
    # THE L0 MASK-REPLICA MUTANT, REMOVED as EQUIVALENT UNDER SINGLE
    # MUTATION, with evidence -- originally numbered M185, renumbered
    # out of the way when the enclosure branch merged and its own M185
    # took that slot. Named by what it does, not by a number two
    # registries both wanted --
    # the same disposition M58/M59/M87 got, and for the same reason
    # rather than because a red was inconvenient.
    #
    # The candidate was "the decode path skips the mask replica"
    # (`check_mask_replica=True` -> `False` in
    # `CertifyingDecoder._l0_verify`). It SURVIVED, and the reason is not
    # a weak test: `Edge.tag()` hashes the MASK along with the weight and
    # endpoints, so a flipped mask fails the per-edge accumulator before
    # the replica loop is ever reached -- and `_l0_verify` derives
    # `masks_read` from the very edges the accumulator just verified, so
    # the loop compares bytes already proven equal.
    #
    # MEASURED: over 5,920 configurations -- every single-field weight
    # and mask corruption of 400 random graphs, plus their clean forms --
    # the two settings disagreed on 0.
    #
    # The replica check stays in the call, because it is not redundant
    # WHERE IT WAS DESIGNED FOR: a commit stage that reads masks from
    # somewhere the per-edge accumulation never touched. It is redundant
    # only at this call site, and `test_the_tag_COVERS_the_mask` pins the
    # precondition that makes it so -- if `Edge.tag()` ever stops hashing
    # the mask, that test fails and this note becomes wrong out loud.
    Mutant(
        # THE VERDICT IS COMPUTED AND DISCARDED. The most expensive
        # possible no-op: the full check runs, costs what it costs, and
        # nothing acts on it -- the shape of a gate whose output is the
        # same whether or not it ran.
        "M199-the-l0-commit-verdict-is-ignored",
        "commit_check runs and its refusal is dropped",
        '        if not ok:\n            raise ConfigurationFault(',
        '        if False:\n            raise ConfigurationFault(',
        "test_a_changed_WEIGHT_refuses_the_decoder",
        "certifying_decoder",
    ),
    # ---- WHICH DEM does the grade belong to? ----
    Mutant(
        # A GRADE EARNED BY A DIFFERENT MODEL. The rung below this closed
        # "an int is an int" by re-deriving the level from the rungs. It
        # left open the layer above: a legitimately-A3 record for model X
        # capping a receipt decoded against model Y, with every field in
        # the block true. Disarm the membership test and that is exactly
        # what happens -- silently.
        "M181-a-grade-for-another-model-is-accepted",
        "dem_binding stops checking WHICH model the certificate graded",
        '        if str(decoded_against) not in covers:',
        '        if False:',
        "test_a_grade_earned_by_a_DIFFERENT_model_is_REFUSED",
        "soundness_budget",
    ),
    Mutant(
        # A CERTIFICATE THAT NAMES NO MODEL MUST NOT READ AS BOUND.
        # Reporting the unnamed case as bound is worse than refusing it:
        # the block would say `certificate_bound` for a record whose
        # `dem_root` is null, which is the exact shape of "declared !=
        # independent".
        "M182-an-unnamed-dem-reads-as-bound",
        "the no-root case is reported as a binding",
        "    return (\"certificate_unnamed_dem\",\n"
        "            \"the certificate records no `dem_root`, so WHICH model it \"",
        "    return (\"certificate_bound\",\n"
        "            \"the certificate records no `dem_root`, so WHICH model it \"",
        "test_a_certificate_with_NO_root_cannot_be_bound_to_anything",
        "soundness_budget",
    ),
    Mutant(
        # THE PRODUCER MUST PUBLISH THE IDENTITY. If `dem_cert_run`
        # stops recording the roots, every downstream binding degrades
        # to `certificate_unnamed_dem` -- no error, no refusal, and the
        # A3 grade floats free of any model again.
        "M183-the-published-admission-names-no-model",
        "dem_cert_run stops recording which models it graded",
        '    return ([c["fault_root"] for c in certs],',
        '    return (None,',
        "test_the_PRODUCER_of_the_identity_is_what_the_test_calls",
        "dem_cert_run",
    ),
    # ---- D2: the sigma that makes alpha* a number, not a shape ----
    Mutant(
        # DROP THE SCALE NORMALISATION and sigma becomes ~4x too large,
        # every alpha* ~4x too small, and the corpus is reported as far
        # more brittle than it is -- with no error, no refusal, and a
        # perfectly plausible histogram. Matching is invariant under a
        # positive rescale, so the whole inflation is a unit artifact.
        "M179-sigma-keeps-the-log-likelihood-scale",
        "normalised() stops dividing by the mean",
        '    m = sum(vals) / len(vals)',
        '    m = 1.0',
        "test_normalised_weights_have_mean_one",
        "certificate_robustness_corpus",
    ),
    Mutant(
        # A TIE IS NOT A CERTIFICATE. Scoring one gives alpha* = 0 --
        # the declared objective did not determine the sector, so there
        # is nothing for a radius to be a radius of, and folding zeros
        # in reports a corpus that is mostly ambiguous as mostly
        # brittle. 25 of 231 shots here are ties.
        "M180-ties-are-scored-as-zero-radius-certificates",
        "a tie between the two cosets is treated as a proof",
        '        if w0 == w1:\n            ties += 1\n            continue',
        '        if False:\n            ties += 1\n            continue',
        "test_the_corpus_run_produces_radii_and_no_flip_at_the_measured_sigma",
        "certificate_robustness_corpus",
    ),
    # ---- the live lane's observable: which logical did it measure? ----
    Mutant(
        # THE CUT AND THE TRUTH COLUMN MUST NAME ONE END. Flip the cut to
        # the other boundary and the lane still runs, still prints a
        # logical error rate, and the number describes a DIFFERENT
        # observable than the one the device prepared. The first version
        # of this file had exactly this defect and only the two-sided
        # check found it.
        "M177-the-logical-cut-names-the-other-boundary",
        "logical_cut_edges returns the opposite end from truth_column",
        '    return frozenset(_lc_key(t * (d - 1) + (d - 2), BOUNDARY)\n'
        '                     for t in range(r + 1))',
        '    return frozenset(_lc_key(t * (d - 1), BOUNDARY)\n'
        '                     for t in range(r + 1))',
        "test_an_error_on_the_LAST_data_qubit_flips_the_declared_cut",
        "rep_code_cert",
    ),
    Mutant(
        # delta_S = 0 is a THEOREM about certified shots. Reporting it
        # for a refused shot converts "E is unknown here" into "search
        # cost nothing", which is the strongest claim in the artifact
        # and the one it would not have earned.
        "M178-delta-S-reported-zero-for-refused-shots",
        "an unknown E is counted as a certified zero",
        '            if res.accepted:\n'
        '                obs_stats["delta_S_certified_zero"] += 1\n'
        '            else:\n'
        '                obs_stats["delta_S_unknown"] += 1',
        '            obs_stats["delta_S_certified_zero"] += 1',
        # THE FIRST KILLER BRANCHED ON THE COUNTER THIS MUTANT ZEROES,
        # so the mutant chose the test's own passing branch and SURVIVED.
        # The killer now FORCES a refusal and asserts delta_S is None.
        "test_delta_S_is_UNDETERMINED_when_any_scored_shot_was_refused",
        "rep_code_cert",
    ),
    # ---- the REAL-shot lane: a verdict that stops being a verdict ----
    Mutant(
        # A CHECK THAT ONLY LOOKS AT THE GUILTY CELL CANNOT SEE A
        # FALSE ACCUSATION. Dropping the clean-cell arm leaves a lane
        # that would pass while every configuration screamed -- which is
        # what a broken exact tier or a broken mass computation would
        # look like, not a localized model fault.
        "M174-a-clean-cell-can-never-be-found-guilty",
        "check_localization stops testing the configurations it did not plant",
        '        elif abs(share["delta_M"]) > tol["delta_M"]:',
        '        elif False:',
        "test_the_check_FAILS_when_the_fault_is_not_where_it_is_CLAIMED",
        "error_budget_ledger",
    ),
    Mutant(
        # THE WHOLE POINT IS THAT A MODEL BUG IS NOT A DECODER BUG.
        # Without this arm the lane would accept a run in which the
        # planted wrong-DEM moved delta_S -- i.e. in which the budget
        # billed the heuristic for the model, which is the failure the
        # four signed terms exist to prevent.
        "M175-a-model-fault-may-be-billed-to-the-decoder",
        "the mismatched cell stops checking that delta_S and delta_O stay put",
        '            for k in ("delta_S", "delta_O"):',
        '            for k in ():',
        # THE KILLER NAMED HERE FIRST WAS ONE LAYER AWAY AND THE MUTANT
        # SURVIVED. `test_the_planted_fault_is_NOT_billed_to_the_decoder`
        # asserts a property of `budget(...).totals` and never calls
        # `check_localization`, so removing an arm of the check could
        # not fail it. The killer now builds a cell with BOTH faults and
        # asserts the CHECK objects.
        "test_the_CHECK_objects_when_a_model_fault_reaches_delta_S",
        "error_budget_ledger",
    ),
    Mutant(
        # THE MODEL MUST NEVER SEE THE TRUE RATE. `p_model` is what the
        # decoder and the Bayes tier are told; `p_true` is what reality
        # used. Passing the true rate to the masses makes every planted
        # mismatch vanish -- a lane that always looks calibrated,
        # reporting that no model fault exists because it was handed the
        # answer.
        "M176-the-bayes-tier-is-told-the-true-rate",
        "coset masses computed under p_true instead of the declared p_model",
        '        z0, z1, st = coset_masses(checks, [p_model] * n_vars, syn_bits,',
        '        z0, z1, st = coset_masses(checks, [p_true] * n_vars, syn_bits,',
        "test_a_planted_MODEL_fault_lands_in_delta_M_in_the_right_cell",
        "error_budget_ledger",
    ),
    # ---- the dual forgery battery: an arm that stops being an attack ----
    Mutant(
        # THE FAILURE MODE OF A RED-TEAM SUITE IS NOT A WRONG VERDICT,
        # IT IS AN ARM THAT QUIETLY STOPS ATTACKING. Strip the node
        # potentials and the "forgery" is no longer the mixed-dual
        # counterexample at all; the checker still refuses it, the
        # battery still prints PASS, and the one published attack on
        # the repository's central claim is no longer being run.
        "M170-forgery-arm-stops-being-a-forgery",
        "the mixed y+z object loses its node potentials",
        '        node_potentials={0: 9.0, 1: 9.0, 2: -10.0},',
        '        node_potentials={},',
        "test_the_forgery_is_refused_BY_NAME_not_incidentally",
        "dual_forgery",
    ),
    Mutant(
        # The positive control is what stops a checker stuck on "no"
        # from passing every negative arm. Weakening its tightness
        # check leaves the battery unable to tell a working checker
        # from a broken one.
        "M171-positive-control-stops-proving-optimality",
        "the accepted control no longer has to attain the primal",
        '    tight = (got.accepted and got.primal == Fraction(2)',
        '    tight = (got.accepted and got.primal == Fraction(20)',
        "test_the_challenge_can_ACCEPT_so_a_stuck_checker_cannot_pass",
        "dual_forgery",
    ),
    Mutant(
        # THE TERM E2 RESTS ON. Stop charging for discarded states and the
        # upper bound becomes the truncated value -- i.e. the LOWER bound
        # -- so the tier reports a point estimate wearing a bracket's
        # clothes and decides every shot. `tools/enclosure_gate.py
        # --prove-sensitive` deletes the same line and measures 373
        # containment violations; this pins the suite to notice too.
        "M184-enclosure-discard-is-free",
        "discarded states are dropped without being priced",
        "                if r:\n                    slack += val * r",
        "                if False:\n                    slack += val * r",
        "test_the_bracket_contains_the_truth_on_random_models",
        target="coset_enclosure",
    ),
    Mutant(
        # E4's residual demand. Price every discard at k = 0 and the
        # bound is (discarded mass) * prod(1 + lam) -- still SOUND, and
        # useless. A defect that only widens a bracket is invisible to
        # any test that watches the verdict, so the killer is the one
        # that watches the tier still DECIDE.
        "M185-enclosure-residual-demand-ignored",
        "every discard priced as if it owed nothing",
        '        k = demand(cov, bin(dem).count("1"))',
        "        k = 0",
        # The killer has to exercise the CHECKER. The first version named
        # a producer test, which never imports this module -- an aimed
        # mutant pointed one module away from its own defect, which is
        # the failure mode the whole registry exists to prevent, found
        # by running it.
        "test_the_checker_reproduces_the_producer_through_the_wire",
        target="coset_enclosure_check",
    ),
    Mutant(
        # E6's disjointness. Take the odd-parity product over EVERY owed
        # detector instead of a mechanism-disjoint family and the
        # factorisation is no longer exact -- the bound can fall BELOW
        # the true future value and the enclosure stops enclosing.
        "M186-enclosure-parity-family-not-disjoint",
        "odd-parity factors multiplied over overlapping mechanism sets",
        "    C = _independent(demand, adj) if demand else []",
        "    C = [i for i in range(len(adj)) if demand >> i & 1]",
        "test_the_bracket_contains_the_truth_on_random_models",
        target="coset_enclosure",
    ),
    Mutant(
        # THE SYNDROME IS ENFORCED WHERE A DETECTOR CLOSES, AND NOWHERE
        # ELSE. Accept both parities at closure and every class collects
        # the mass of configurations that do not reproduce the syndrome
        # at all -- an enclosure of the wrong question that still looks
        # like a bracket.
        "M187-enclosure-closure-stops-enforcing",
        "a closing detector no longer has to match the syndrome",
        "                if (dmask & close) != want:",
        "                if False:",
        "test_the_bracket_contains_the_truth_on_random_models",
        target="coset_enclosure",
    ),
    Mutant(
        # A detector no mechanism touches can never be flipped, so a
        # syndrome demanding a flip there is INFEASIBLE. Drop the check
        # and the tier reports positive mass -- and a confident verdict
        # -- on a provably impossible shot. This is not hypothetical: it
        # is the defect the first prototype of the module shipped.
        "M188-enclosure-untouched-detector-ignored",
        "an unreachable syndrome bit is no longer infeasible",
        "        if c not in last and syn[c]:",
        "        if False:",
        "test_an_untouched_detector_demanding_a_flip_is_INFEASIBLE",
        target="coset_enclosure",
    ),
    Mutant(
        # THE VERDICT BELONGS TO THE REPLAY. Accept every cell comparison
        # and the checker stops recomputing the verdict at all -- a
        # producer certifying itself, the exact separation this whole
        # repository exists to keep.
        "M189-enclosure-checker-stops-comparing",
        "the checker endorses a winner without beating the rivals",
        "        if all(c == best or LO[best] > LO[c] + slack for c in cells):",
        "        if True:",
        "test_the_checker_reproduces_the_producer_through_the_wire",
        target="coset_enclosure_check",
    ),
    Mutant(
        # A tie is p_ambiguity, not a decision. Relax the comparison to
        # >= and every exactly-balanced shot is certified for whichever
        # class sorts first.
        "M190-enclosure-tie-becomes-a-decision",
        "a straddling bracket accepted on equality",
        "            if c != best and not LO[best] > HI[c]:",
        "            if c != best and not LO[best] >= HI[c]:",
        "test_a_tie_is_an_AMBIGUITY_not_a_decision",
        target="coset_enclosure",
    ),
    Mutant(
        # The battery's own audit. If a forgery stops being false, "0
        # accepted" is reported by an attack that was never an attack --
        # the fake-alarm failure `forgeries.py` recorded twice.
        "M191-enclosure-forgery-stops-being-false",
        "the digest-break forgery no longer edits the model",
        '        d["mechanisms"][0]["p"] = "1/2"',
        "        pass",
        # NAMING every attack is not the same as every attack being
        # FALSE. The vacuity test only counts tags, so it passed with the
        # forgery neutered; the killer is the one that demands each
        # attack actually be refused.
        "test_every_forgery_is_refused_and_the_battery_says_which",
        target="enclosure_forgery",
    ),
    Mutant(
        # S1 IS A SHORTFALL, NOT A CONFIDENCE. Sum p_lo instead of
        # (1 - p_lo) and the "bound" on misdecodes becomes the number of
        # shots you were confident about -- large, monotone in the right
        # direction, and reported with a straight face.
        "M192-selection-bound-sums-confidence",
        "the batch bound sums p_lo instead of its shortfall",
        # RE-ANCHORED. The line acquired `_up(...)` when the sum started
        # rounding outward, and the mutant went SKIP -- which the gate
        # correctly refuses to score as a pass: a mutant whose anchor has
        # drifted proves nothing, and it drifts exactly when the code it
        # aims at is the code being changed.
        "        return sum((_up(1 - s.p_lo) for s in self.accepted), Fraction(0))",
        "        return sum((_up(s.p_lo) for s in self.accepted), Fraction(0))",
        "test_S1_sums_the_shortfalls_and_rounds_OUTWARD",
        target="certified_selection",
    ),
    Mutant(
        # THE ADMISSION FLOOR. Every bound in that module is a statement
        # about a model; reporting one for a model the ladder has not
        # admitted launders exactly the authority `dem_admission` exists
        # to withhold.
        "M193-selection-ignores-the-admission-floor",
        "a bound reported for a model below the DEM floor",
        "    if dem_level < min_dem_level:",
        "    if False:",
        "test_a_model_below_the_admission_floor_is_REFUSED",
        target="certified_selection",
    ),
    Mutant(
        # A REFUSAL IS NOT A LOW-CONFIDENCE ACCEPTANCE. Fold the refused
        # shots into the accepted set and the batch bound covers shots
        # that carry no posterior at all -- evidence invented for the
        # only quantity the module consumes.
        "M194-selection-scores-the-refusals",
        "shots the enclosure refused are counted as accepted",
        "        (sel.accepted if s.p_lo >= t else sel.rejected).append(s)",
        "        sel.accepted.append(s)",
        "test_a_threshold_moves_the_bound_and_the_discard_rate_together",
        target="certified_selection",
    ),
    Mutant(
        # A FLOAT POSTERIOR IS A ROUNDING WEARING A BOUND'S CLOTHES. The
        # whole tier exists so no bound rests on one.
        "M195-selection-accepts-a-float-posterior",
        "a float p_lo admitted instead of an exact Fraction",
        "        if not isinstance(self.p_lo, Fraction):",
        "        if False:",
        "test_a_float_or_out_of_range_posterior_is_REFUSED",
        target="certified_selection",
    ),
    Mutant(
        # A CONTRACT WITH ONE CELL DECIDES NOTHING and said "ok" anyway --
        # the "beats every rival" quantifier is vacuously true, so the
        # tier returned a winner nothing could contradict, including one
        # whose certified lower bound was zero. Found by reading the
        # comparison adversarially rather than by a failing test, which
        # is why it gets a mutant: the next reader will not re-derive it.
        "M196-enclosure-one-cell-contract-decides",
        "a single-cell contract accepted instead of refused",
        "    if len(cells) < 2:",
        "    if False:",
        "test_a_ONE_CELL_contract_is_REFUSED_because_it_decides_nothing",
        target="coset_enclosure",
    ),
    Mutant(
        # THE WORK GUARD ON UNTRUSTED INPUT. The E3 tables are quadratic
        # in the mechanism count, so a document declaring a large model
        # asks the checker for a gigabyte. Refusing is the only safe
        # answer; discovering it by being killed is not.
        "M197-enclosure-work-guard-removed",
        "an unbounded model accepted instead of refused",
        "    if len(mechs) > MAX_MECHANISMS:",
        "    if False:",
        "test_an_UNTRUSTED_document_cannot_ask_for_a_gigabyte",
        target="coset_enclosure",
    ),
    # ---- THE LEDGER'S PREREQUISITE ARITHMETIC ----
    #
    # `E1` was priced at "minutes" while the `E0` trace it reads --
    # "~1 engineer-week" -- was not a row in the ledger at all. A cost
    # column that omits a precondition understates the board by exactly
    # the work nobody can see, and the omission was undetectable because
    # the missing item had no slot to be missing from.
    #
    # AIMING NOTE, all six: the obvious killer here is
    # `test_E1_cannot_be_priced_without_E0`, and for four of them it is
    # the WRONG one -- it asserts the composed string is long, which
    # several of these mutations leave true. Each must_fail below is the
    # test asserting the direction its own mutation breaks.
    Mutant(
        "M290-a-prerequisite-outside-the-ledger-stops-being-refused",
        "closure is dropped, so a gate may name work that has no row and "
        "therefore no cost -- the exact shape of the defect this field "
        "was added for, reintroduced one level up",
        "            if req not in index:",
        "            if req in index and False:",
        "test_a_prerequisite_that_is_not_a_row_is_REFUSED",
        target="roadmap_gate",
    ),
    Mutant(
        "M291-a-prerequisite-cycle-loops-instead-of-refusing",
        "A-needs-B-needs-A recurses until the interpreter stops it, "
        "instead of naming the cycle -- a ledger that prices nothing "
        "while appearing to compute",
        "            if req in stack:",
        "            if False:",
        "test_a_prerequisite_cycle_is_REFUSED_not_looped",
        target="roadmap_gate",
    ),
    Mutant(
        # NOT aimed at the E1 test: with finished work still charged, E1
        # still reads "engineer-week + minutes" and that test stays
        # GREEN. Only a test asserting a CLOSED prerequisite drops out
        # can see this.
        "M292-finished-work-is-charged-forever",
        "the CLOSED filter is dropped, so the ready cost bills every "
        "prerequisite ever completed and overstates the remaining board "
        "as badly as the original omission understated it",
        '            if done.get(req) != "CLOSED" and req not in seen:',
        "            if req not in seen:",
        "test_a_CLOSED_prerequisite_drops_out_of_the_ready_cost",
        target="roadmap_gate",
    ),
    Mutant(
        "M293-the-ready-cost-stops-composing",
        "every gate reports its ISOLATED price again, so `E1` reads "
        "'minutes' with an engineer-week in front of it",
        "    if not chain:",
        "    if True:",
        "test_E1_cannot_be_priced_without_E0",
        target="roadmap_gate",
    ),
    Mutant(
        # THE CHECK GOING DARK IS THE WORST OUTCOME HERE: it reports
        # zero findings, which reads exactly like a clean document.
        "M294-the-document-check-inspects-no-rows",
        "every row is skipped, so the prerequisite check returns no "
        "findings whether the document names its preconditions or not",
        '        if not r["unmet_prerequisites"]:',
        "        if True:",
        "test_the_document_check_FAILS_when_the_row_hides_its_prerequisite",
        target="roadmap_gate",
    ),
    Mutant(
        "M295-the-rendered-table-prints-the-isolated-cost",
        "the ledger a reader actually reads reverts to the number that "
        "hides the prerequisite, while the JSON stays correct -- drift "
        "between the artifact and the document it is supposed to fix",
        '            cost = r["ready_cost"]',
        '            cost = r["cost"]',
        "test_the_rendered_row_names_the_open_prerequisite",
        target="roadmap_gate",
    ),
    # ---- GA-11: the WAY floor, and the units that nearly became a result
    #
    # AIMING NOTE. The obvious killer for four of these is
    # `test_the_window_is_bounded_at_BOTH_ends`, and it is the wrong one
    # for three: several of these mutations leave a two-ended window in
    # place and only move it. Each must_fail names the test asserting the
    # direction its own mutation actually breaks.
    Mutant(
        # THE UNITS ERROR, REINSTATED. `eps^2 = 4p` is what separates
        # "the model is physical on a window two orders wide" from "the
        # model is unphysical everywhere". Dropping the /4 is the
        # smallest edit that reintroduces the first draft's defect.
        "M300-the-rms-to-probability-conversion-loses-its-factor",
        "eps^2 = 4p becomes eps^2 = p, so the floor is quadrupled and "
        "the comparison drifts back toward the units error that made the "
        "model look unphysical at every photon number",
        "    return eps * eps / 4.0",
        "    return eps * eps",
        "test_the_units_conversion_is_not_cosmetic",
        target="way",
    ),
    Mutant(
        # NOT aimed at the window test: with n_0 entering linearly the
        # window still exists and still has two ends. Only the test that
        # compares two routes to the same k can see it.
        "M301-n0-enters-the-group-linearly",
        "the square root is dropped from k = C*SNR_0/sqrt(n_0), so two "
        "devices the reduction says are identical stop agreeing and the "
        "one-group claim silently becomes false",
        "    return abs(commutator) * snr_0 / math.sqrt(n_0)",
        "    return abs(commutator) * snr_0 / n_0",
        "test_n_0_enters_only_through_the_square_root",
        target="way",
    ),
    Mutant(
        # THE BOUND THAT CANNOT FAIL. A commuting observable given a
        # floor of 0.0 is satisfied by every measurement ever made.
        "M302-a-commuting-observable-is-floored-at-zero",
        "the QND refusal is removed, so [sigma_z, N] = 0 yields a WAY "
        "floor of 0.0 -- a bound no measurement can violate, sitting "
        "under a gate whose job is to fail when the mechanism does not "
        "work",
        "    if commutator == 0.0:\n        raise WayRefusal(\n"
        '            "|<[A,N]>| = 0: the observable commutes with the '
        'conserved "\n'
        '            "quantity, so WAY prices NOTHING here (the QND '
        'dispersive "\n'
        '            "case). Refusing rather than returning a floor of '
        '0")',
        "    if False:\n        raise WayRefusal(\n"
        '            "|<[A,N]>| = 0: the observable commutes with the '
        'conserved "\n'
        '            "quantity, so WAY prices NOTHING here (the QND '
        'dispersive "\n'
        '            "case). Refusing rather than returning a floor of '
        '0")',
        "test_a_COMMUTING_observable_is_REFUSED_not_floored_at_zero",
        target="way",
    ),
    Mutant(
        # A WINDOW REPORTED WHERE NONE EXISTS. The refusal is the only
        # thing between `physical_window` and a bracket with no root.
        "M303-a-window-is-reported-above-K_MAX",
        "the no-window refusal is removed, so a device whose group "
        "exceeds K_MAX gets an interval anyway -- an unphysical model "
        "handed a validity domain",
        "    if gap(MU_STAR) <= 0.0:",
        "    if False:",
        "test_a_group_above_K_MAX_REFUSES_rather_than_returning_an_interval",
        target="way",
    ),
    Mutant(
        # K_MAX ITSELF. `mu*` wrong makes G_MAX wrong makes K_MAX wrong,
        # and every verdict about every device moves with it.
        "M304-mu-star-solves-the-wrong-stationarity-condition",
        "the factor of 2 in 2*Phi(-mu) = mu*phi(mu) is dropped, so mu* "
        "is no longer the maximiser of g and K_MAX -- the single number "
        "GA-11 reduces to -- is wrong for every device",
        "        return 2.0 * _phi(-m) - m * _pdf(m)",
        "        return _phi(-m) - m * _pdf(m)",
        "test_mu_star_is_the_stationary_point_of_g",
        target="way",
    ),
    Mutant(
        # THE DARK BISECTION. An interval-halving that reports its
        # midpoint whether or not a root is inside.
        "M305-the-bisection-accepts-a-bracket-with-no-root",
        "the sign-change guard is removed, so bisection returns a "
        "midpoint from a bracket containing no root and reports it as an "
        "edge of the physical window",
        "    if (flo > 0) == (fhi > 0):",
        "    if False:",
        "test_a_bracket_with_no_sign_change_is_REFUSED",
        target="way",
    ),
    Mutant(
        # THE RELATIVE RULE. Its sign is the claim that photons buy
        # precision; flipping it inverts the policy the corpus names as
        # the successor to a loose floor.
        "M306-the-marginal-price-loses-its-sign",
        "d p_W / d n_bar becomes positive, so the pricing rule says "
        "photons BUY error -- and the elasticity that is supposed to be "
        "exactly -1 becomes +1",
        "    return -floor_probability(commutator, n_bar) / n_bar",
        "    return floor_probability(commutator, n_bar) / n_bar",
        "test_the_marginal_price_is_negative_and_matches_a_finite_difference",
        target="way",
    ),
    # ---- THE ARTIFACT'S SUBJECT, AND A RECORDED FAILURE ----
    Mutant(
        # THE PROXY CHECK GOING DARK IS THE WORST OUTCOME: zero findings
        # reads exactly like a document that declares every proxy.
        "M307-the-proxy-check-inspects-no-rows",
        "every row is skipped, so a gate fed by a stand-in for its own "
        "subject can sit in the ledger with the criterion's experiment "
        "implied and never run",
        '        if not r.get("proxy"):',
        "        if True:",
        "test_the_proxy_check_FAILS_when_the_row_hides_it",
        target="roadmap_gate",
    ),
    Mutant(
        "M308-a-proxy-row-passes-without-declaring-it",
        "the row is inspected and the demand dropped, so a proxy "
        "declared in the registry never has to appear in the document "
        "anyone reads",
        '                if "PROXY" not in line:',
        "                if False:",
        "test_the_proxy_check_FAILS_when_the_row_hides_it",
        target="roadmap_gate",
    ),
    Mutant(
        # A RECORDED FAILURE THAT CANNOT BE SEEN TO CHANGE. 141 failures
        # becoming 200 would leave the pin, the row and every sentence
        # about the finding intact.
        "M309-a-changed-failure-count-reproduces",
        "the per-arm comparison is dropped, so GA-3's recorded failure "
        "reproduces no matter what the arms now do",
        '        if got != r["failures_total"]:',
        "        if False:",
        "test_a_CHANGED_failure_count_is_caught_and_NAMED",
        target="ga3_voi_gate",
    ),
    Mutant(
        # NOT aimed at the failure-count test: the arms still compare, so
        # that one stays GREEN. Only the verdict/attribution test sees it.
        "M310-a-flipped-verdict-reproduces",
        "the headline comparison is dropped, so a FAILED gate whose "
        "verdict silently flips to passed still reports that it "
        "reproduces its recorded result",
        "        if not same:",
        "        if False:",
        "test_a_FLIPPED_verdict_is_caught",
        target="ga3_voi_gate",
    ),
    Mutant(
        "M311-a-vanished-headline-is-treated-as-agreement",
        "a key absent from the fresh run stops being a difference, so "
        "deleting the number instead of matching it reads as a "
        "reproduction",
        "        if b is None:",
        "        if False:",
        "test_a_MISSING_headline_key_is_caught_not_treated_as_equal",
        target="ga3_voi_gate",
    ),
    Mutant(
        # THE ROADMAP POSITION. The front page drifted 8 gates and a
        # whole row while the sentence around the number promised it was
        # "held there by a standing gate". Deleting the bold-tolerance
        # makes the pattern match nothing again -- silently.
        "M312-the-roadmap-position-pattern-matches-nothing",
        "the asterisk tolerance is dropped, so the BOLDED spelling the "
        "README actually uses stops matching and the position claim goes "
        "unchecked exactly as it did for eight gates",
        r'    (re.compile(r"\b(\d+)\s+of\s+(\d+)\*{0,2}\s+roadmap\s+gates\b", re.I),',
        r'    (re.compile(r"\b(\d+)\s+of\s+(\d+)\s+roadmap\s+gates\b", re.I),',
        "test_the_roadmap_POSITION_is_checked_and_the_pattern_can_FIRE",
        target="doc_claims_gate",
    ),
    Mutant(
        "M313-an-absent-roadmap-artifact-licenses-any-position",
        "the shape guard is removed, so a roadmap.json missing its "
        "counts raises instead of reporting UNCHECKABLE -- or worse, "
        "a partial artifact supplies a position nothing derived",
        '    if "closed" not in d or "gates" not in d:',
        "    if False:",
        # AIMED AT THE GUARD, NOT AT THE PATTERN. The obvious killer --
        # `..._can_FIRE` -- builds its own truth dict and never calls
        # `_roadmap_position`, so it would have stayed GREEN with this
        # guard deleted. Only the malformed-artifact test reaches it.
        "test_a_MALFORMED_roadmap_artifact_yields_UNCHECKABLE_not_a_position",
        target="doc_claims_gate",
    ),
    Mutant(
        # THE EXEMPTION GOING BACK TO ONE SPELLING. It is not a
        # loosening that breaks here -- it is a TIGHTENING that makes
        # the exemption unreachable for `N of M` claims, so a paragraph
        # correctly quoting a superseded figure beside the true one gets
        # flagged, and the restamping pass then rewrites the quotation.
        # A gate that forces the record to be falsified is worse than no
        # gate.
        "M314-the-superseded-figure-exemption-loses-a-spelling",
        "the `N of M` form is dropped from the exemption, so a "
        "correctly-qualified historical citation is flagged and the "
        "next restamp destroys the record to satisfy the gate",
        '                            or re.search(rf"\\b{tn}\\s+of\\s+{td}\\b", line)):',
        "                            or False):",
        "test_the_superseded_figure_exemption_speaks_BOTH_spellings",
        target="doc_claims_gate",
    ),
    # ---- THE WALL'S OWN VERDICT-vs-DEATH CLASSIFIER ----
    Mutant(
        # THE DEFECT AS IT ACTUALLY OCCURRED. With the classifier blind,
        # a MemoryError traceback is announced as "the verdict, not a
        # crash report" -- and a reader believes it, because the wall
        # says the tool decided.
        "M315-a-crash-is-announced-as-the-tools-own-verdict",
        "the death check is dropped from the verdict branch, so an "
        "exit 1 carrying a traceback is reported as a documented "
        "refusal -- a fact about free RAM read as a soundness finding",
        "    if not is_pytest and 0 < code < 126 and not looks_like_a_death(output):",
        "    if not is_pytest and 0 < code < 126:",
        "test_a_CRASH_that_exits_1_is_not_reported_as_a_verdict",
        target="gate_all",
    ),
    Mutant(
        # THE OPPOSITE FAILURE, WHICH IS ALSO REAL. Calling every exit 1
        # a death sends a reader to check RAM and Defender for a gate
        # that said exactly what was wrong -- the misdiagnosis this
        # reporter was built to stop.
        "M316-every-refusal-is-reclassified-as-a-death",
        "`looks_like_a_death` returns True for any output, so a "
        "deliberate refusal with a legible message is reported as a "
        "crash and the reader is sent somewhere confidently wrong",
        "    return any(mark in output for mark in _DEATH_MARKS)",
        "    return True",
        "test_the_SAME_exit_code_with_a_clean_refusal_is_still_a_VERDICT",
        target="gate_all",
    ),
    Mutant(
        "M317-an-absent-output-is-treated-as-a-crash",
        "the empty-output guard is removed, so `run()`'s streamed nodes "
        "-- which capture nothing -- would every one be classified from "
        "a None it cannot read",
        "    if not output:",
        "    if False:",
        "test_the_death_marks_are_reachable_and_do_not_match_ordinary_prose",
        target="gate_all",
    ),
    # ---- THE COST COLUMN vs THE ARTIFACT'S OWN CLOCK ----
    Mutant(
        # THE CHECK GOING DARK. Four ratios between 589x and 98,182x sat
        # unremarked because nothing compared the cost column against
        # the number inside the evidence it points at.
        "M318-a-cost-column-may-contradict-its-own-artifacts-clock",
        "the ratio test is dropped, so a gate priced at 40 CPU-h whose "
        "artifact records 1.7 seconds passes unremarked and a reader "
        "takes the campaign to have been run",
        "                    and declared / seconds > COST_RATIO_LIMIT",
        "                    and False",
        "test_a_cost_column_that_contradicts_the_artifacts_own_clock_is_CAUGHT",
        target="roadmap_gate",
    ),
    Mutant(
        # THE OPPOSITE FAILURE. A cost naming no duration -- "$0",
        # "owner act" -- cannot contradict a runtime, and a parser that
        # invented one would flag every such row forever.
        "M319-a-cost-naming-no-duration-is-given-one",
        "`declared_seconds` stops returning None for a cost with no "
        "time unit, so rows claiming no duration are compared against "
        "a number nobody wrote",
        "    if not m:\n        return None",
        "    if not m:\n        return 1.0",
        "test_a_cost_that_names_no_TIME_cannot_be_contradicted",
        target="roadmap_gate",
    ),
    Mutant(
        # A BOOL IS NOT A DURATION. `True` reads as 1.0 second and would
        # manufacture a 144,000x ratio out of a flag.
        "M320-a-boolean-is-read-as-a-runtime",
        "the bool guard is removed from `recorded_seconds`, so a field "
        "like `passed: true` is read as 1.0 second and invents a "
        "contradiction with any hourly cost",
        "        if k in _TIME_KEYS and isinstance(v, (int, float)) \\\n"
        "                and not isinstance(v, bool):",
        "        if k in _TIME_KEYS and isinstance(v, (int, float)):",
        "test_recorded_seconds_reads_nested_and_ignores_non_numbers",
        target="roadmap_gate",
    ),
    # ---- GT-CT: the cost model that decides the whole gate ----
    Mutant(
        "M321-qubit-rounds-forgets-the-ancillas",
        "physical qubits drops to d^2, halving every arm's bill and "
        "flattering whichever arm sits longest at large d -- which is "
        "the switching arm under test",
        "    return 2 * d * d - 1",
        "    return d * d",
        "test_qubit_count_includes_the_ANCILLAS",
        target="code_trajectory",
    ),
    Mutant(
        # THE SINGLE CHEAPEST WAY TO MANUFACTURE THE RESULT.
        "M322-the-seam-is-shortened-below-the-corpus-floor",
        "the seam lasts min(d_lo,d_hi) rounds instead of max, so the "
        "hole in the protection closes sooner than the corpus says it "
        "can and switching looks affordable",
        "    seam_rounds = hi * cost.seam_slack",
        "    seam_rounds = lo * cost.seam_slack",
        "test_the_seam_is_protected_at_the_SMALLER_distance",
        target="code_trajectory",
    ),
    Mutant(
        "M323-a-sub-floor-seam-slack-is-accepted",
        "the seam_slack floor is removed, so a caller can price a "
        "switch with a seam shorter than the corpus's own minimum",
        "        if self.seam_slack < 1.0:",
        "        if False:",
        "test_a_SEAM_SHORTER_than_the_corpus_floor_is_REFUSED",
        target="code_trajectory",
    ),
    Mutant(
        "M324-the-posterior-underflows-on-a-long-run",
        "the log-sum shift is dropped, so a long run underflows to zero "
        "mass and the posterior becomes uniform exactly when the "
        "evidence is strongest",
        "        w = [moved[j] * math.exp(lls[j] - top) for j in range(k)]",
        "        w = [moved[j] * math.exp(lls[j]) for j in range(k)]",
        "test_a_long_run_does_not_underflow_to_a_uniform_posterior",
        target="code_trajectory",
    ),
    Mutant(
        "M325-an-impossible-syndrome-record-steers-the-policy",
        "more fires than detectors stops being refused, so a corrupted "
        "feed becomes evidence the trigger acts on",
        "        if n < 0 or c < 0 or c > n:",
        "        if False:",
        # The guard moved into `FilterState.update` when `forward_filter`
        # was collapsed to delegate to it -- there is one copy now, and
        # the anchor still identifies it uniquely.
        "test_an_IMPOSSIBLE_observation_is_REFUSED",
        target="code_trajectory",
    ),
    Mutant(
        # SAFE IS NOT FREE. Removing this gate is what made the policy
        # pay two seams before its first informative observation.
        "M326-an-up-switch-needs-no-evidence-at-all",
        "the economic gate on a monotone-SAFE up-switch is removed, so "
        "the uniform prior alone drives 3 -> 5 -> 7 in the first two "
        "rounds, paying seams for protection nothing justified",
        "        return want if posterior[best] >= up_confidence else current",
        "        return want",
        "test_SAFE_is_not_the_same_as_FREE",
        target="code_trajectory",
    ),
    Mutant(
        "M327-protection-is-stripped-on-no-evidence",
        "the down-switch hysteresis is removed, so one noisy round can "
        "strip a patch of distance -- and the seam means it pays twice "
        "to put it back",
        "        return want if posterior[best] >= hysteresis else current",
        "        return want",
        "test_moving_DOWN_requires_evidence",
        target="code_trajectory",
    ),
    Mutant(
        "M328-a-one-pair-bootstrap-reports-an-interval",
        "the pair-count guard is removed, so a degenerate interval "
        "around a mean of nothing can satisfy the CI clause",
        "    if n < 2:",
        "    if False:",
        "test_a_bootstrap_over_one_pair_is_REFUSED",
        target="code_trajectory",
    ),
    Mutant(
        "M329-the-cost-bar-drops-to-zero",
        "the 15% cost-saving bar becomes 0%, so any saving at all -- "
        "including noise -- clears the criterion's second clause",
        "COST_BAR = 0.15",
        "COST_BAR = 0.0",
        "test_advantage_reports_BOTH_bars_and_the_disjunction",
        target="code_trajectory",
    ),
    Mutant(
        # THE MAP THAT FAILED ANTI-VACUITY. Picking the LARGEST
        # sufficient distance instead of the smallest is exactly the
        # hand-written map that made Arm B pay seams the arithmetic
        # never justified.
        "M330-the-band-map-over-provisions",
        "the derived map takes the first distance that FAILS the budget "
        "rather than the first that meets it, so every band asks for "
        "more protection than it needs and every switch is a seam paid "
        "for nothing",
        "            if -math.log(1.0 - 2.0 * e) <= budget:",
        "            if -math.log(1.0 - 2.0 * e) >= budget:",
        "test_the_map_picks_the_SMALLEST_sufficient_distance",
        target="code_trajectory",
    ),
    Mutant(
        "M331-the-per-round-budget-ignores-the-horizon",
        "the error budget stops being divided by the number of rounds, "
        "so a long trajectory is allowed the same per-round error as a "
        "short one and the map under-provisions everywhere",
        "    budget = -math.log(2.0 * target - 1.0) / rounds",
        "    budget = -math.log(2.0 * target - 1.0)",
        "test_the_map_tightens_as_the_horizon_lengthens",
        target="code_trajectory",
    ),
    Mutant(
        # THE COIN-FLIP FLOOR. Ranking an arm that cannot reach the
        # target is how a broken arm won at every band and every length.
        "M332-an-arm-that-cannot-reach-the-target-is-ranked-anyway",
        "the target filter is dropped, so the CHEAPEST arm wins even "
        "when it never delivers the required correctness -- the defect "
        "that made d=3 optimal out to 8,000 rounds by scoring its "
        "failures as half-successes",
        "    ok = {k: v for k, v in candidates.items() if v[0] >= target}",
        "    ok = dict(candidates)",
        "test_an_arm_that_cannot_REACH_the_target_is_excluded_not_ranked",
        target="code_trajectory",
    ),
    Mutant(
        "M333-a-coin-flip-counts-as-a-completion",
        "the floor check accepts a target at or below 0.5, so a fully "
        "decohered qubit's 50% guess rate satisfies the criterion",
        "    return p > COIN_FLIP",
        "    return True",
        "test_a_COIN_FLIP_is_not_a_verified_completion",
        target="code_trajectory",
    ),
    # ---- E0/E1: the transparency census ----
    Mutant(
        # THE ERROR THAT INFLATES EVERYTHING. An unrecognised gate that
        # is neither Clifford nor blocking is silently transparent.
        "M334-an-unclassified-gate-is-silently-transparent",
        "the unknown-gate refusal is removed, so a gate whose "
        "commutation nobody declared is counted as transparent and "
        "every run containing it is too long",
        "        if self.name not in CLIFFORD | NON_CLIFFORD | MEASURE:",
        "        if False:",
        "test_an_UNCLASSIFIED_gate_is_REFUSED_not_assumed_transparent",
        target="transparency",
    ),
    Mutant(
        # THE CORRECTION STOPS SPREADING. A T on a qubit the pending
        # Pauli has already reached becomes invisible.
        "M335-the-correction-cone-stops-growing",
        "a two-qubit Clifford no longer spreads the pending correction, "
        "so a blocking gate on the coupled qubit is missed and the "
        "transparent run runs past the point it must stop",
        "            cone.update(g.qubits)          # the correction spreads",
        "            pass",
        "test_the_correction_SPREADS_through_two_qubit_cliffords",
        target="transparency",
    ),
    Mutant(
        "M336-R_eff-counts-every-boundary-not-the-blocked-ones",
        "R_eff stops distinguishing blocked boundaries from transparent "
        "ones, so the reaction-depth KILL is keyed on a count that has "
        "nothing to do with whether the decoder was ever waited on",
        "    return sum(1 for b in trace if b.blocked_by is not None) / n_t",
        "    return len(trace) / n_t",
        # RE-AIMED after it SURVIVED: the first killer drove a
        # synthetic where EVERY boundary blocked, so "count all" and
        # "count blocked" agreed and the mutation was invisible. The
        # adder has 7 boundaries and NONE block.
        "test_R_eff_counts_only_the_BLOCKED_boundaries",
        target="transparency",
    ),
    Mutant(
        # N_T IS A DENOMINATOR OF A KILL THRESHOLD.
        "M337-an-arbitrary-rotation-is-counted-as-free",
        "RZ stops counting as magic, understating N_T -- the very "
        "denominator the reaction-depth KILL divides by",
        "    return sum(1 for g in gates if g.is_blocking)",
        "    return sum(1 for g in gates if g.name in ('T', 'TDG'))",
        "test_RZ_counts_as_magic",
        target="transparency",
    ),
    Mutant(
        "M338-the-PROMOTE-threshold-drops-to-zero",
        "f(63) >= 0.80 becomes f(63) >= 0.0, so every circuit PROMOTES "
        "and the pre-registered table's best outcome is unconditional",
        "    if f63 >= 0.80:",
        "    if f63 >= 0.0:",
        # RE-AIMED after it SURVIVED: the first killer computed the
        # bands in a LOCAL HELPER instead of calling `census`, so it
        # would have stayed green with the real threshold deleted.
        "test_a_MID_BAND_circuit_does_NOT_promote",
        target="transparency",
    ),
    Mutant(
        "M339-the-greedy-baseline-stops-counting-blocks",
        "the published greedy partition stops starting a block at each "
        "non-Clifford, so the baseline the DAG-reordering KILL compares "
        "against is not the published rule",
        "        if g.is_blocking:\n            blocks += 1",
        "        if False:\n            blocks += 1",
        "test_the_greedy_baseline_starts_a_block_at_every_non_clifford",
        target="transparency",
    ),
    Mutant(
        # A STRUCTURAL ZERO WEARING A MEASURED ZERO'S CLOTHES.
        "M340-a-structural-P_stall-zero-reads-as-measured",
        "the too-few-blocked guard is removed, so P_stall = 0.0 from a "
        "trace where NOTHING could have stalled reads as a measured "
        "rate clearing the conditional row's <= 0.02 bar",
        "    if len(blocked) < 2:",
        "    if False:",
        "test_P_stall_of_zero_from_TOO_FEW_BLOCKED_says_it_is_structural",
        target="transparency",
    ),
    # ---- the roadmap ledger's four-state verdict ----
    Mutant(
        # THE ORIGINAL DEFECT, RE-CREATED EXACTLY. One bit for two
        # questions: a recorded KILL becomes indistinguishable from a
        # gate that was never run.
        "M341-a-failed-artifact-is-reported-as-absent",
        "verdict_of collapses FAILED into ABSENT, so a gate that RAN "
        "and recorded a kill falls through to BUILT-UNMEASURED and the "
        "roadmap says we wrote some code and never measured it",
        "        return FAILED",
        "        return ABSENT",
        "test_a_FAILED_artifact_is_FAILED_and_NOT_absent",
        target="roadmap_gate",
    ),
    Mutant(
        # THE SOFTEST WORD, MADE FREE AGAIN.
        "M342-a-partial-need-not-say-what-blocks-it",
        "the why_not_closed refusal is removed, so any failing gate can "
        "be demoted to RAN-PARTIAL by adding one word to its artifact, "
        "and the ledger goes on deriving the softer status honestly",
        '    if not str(doc.get("why_not_closed", "")).strip():',
        "        if False:",
        "test_a_PARTIAL_with_NO_REASON_is_REFUSED",
        target="roadmap_gate",
    ),
    Mutant(
        # A KILL SOFTENED BY THE ARTIFACT BESIDE IT.
        "M343-a-partial-outranks-a-failure",
        "the FAILED branch stops firing, so a gate holding both a "
        "tested-and-failed criterion and a blocked one reports the "
        "blocked one -- the weaker fact",
        "        if any(x == FAILED for x in v.values()):",
        "        if False:",
        "test_a_FAILURE_beside_a_PARTIAL_reports_the_FAILURE",
        target="roadmap_gate",
    ),
    Mutant(
        # AN ARTIFACT WITH NO VERDICT IS EVIDENCE, NOT A FAILURE.
        "M344-a-corpus-with-no-passed-field-reads-as-failed",
        "an artifact that renders no verdict -- a corpus, a catalogue, a "
        "receipt -- stops counting as evidence, which would demote every "
        "gate closed by one of them",
        '    if not isinstance(doc, dict) or "passed" not in doc:',
        "    if False:",
        "test_an_artifact_with_NO_VERDICT_is_evidence_not_a_failure",
        target="roadmap_gate",
    ),
    # ---- Q_j and the censoring correction ----
    Mutant(
        # THE DEFAULT THAT WAS THE BUG, PUT BACK.
        "M345-an-undeclared-correction-target-defaults-to-the-measured-qubit",
        "correction_support stops refusing an undeclared measurement and "
        "falls back to the qubit that was MEASURED -- which in a "
        "measurement-based uncomputation is the one place the correction "
        "is guaranteed not to land, and is what made every routing width "
        "read 1.0 and left E1's conditional row unresolvable",
        "    if g.fixup is UNDECLARED:",
        "    if False:",
        "test_an_UNDECLARED_correction_target_is_REFUSED_not_defaulted",
        target="transparency",
    ),
    Mutant(
        # THE CONE ROOTED BACK AT THE MEASURED QUBIT.
        "M346-the-cone-is-seeded-from-the-measured-qubit-again",
        "the causal cone is seeded from the qubits the measurement acted "
        "on rather than the qubits its correction lands on, so a blocking "
        "gate on the corrected qubit is invisible and one on the measured "
        "qubit blocks spuriously",
        "    cone = set(q_j)",
        "    cone = set(gates[start].qubits)",
        "test_the_CONE_IS_SEEDED_FROM_THE_FIXUP_not_the_measured_qubit",
        target="transparency",
    ),
    Mutant(
        # A READOUT COUNTED AS A BOUNDARY PADS THE DENOMINATOR f(k)
        # IS A FRACTION OVER.
        "M347-a-terminal-readout-is-counted-as-a-boundary",
        "a measurement that conditions no correction is admitted as a "
        "boundary, padding the denominator of every f(k) with records "
        "whose segment length means nothing",
        "            if g.is_measurement and correction_support(g)]",
        "            if g.is_measurement]",
        "test_a_TERMINAL_READOUT_is_not_a_boundary",
        target="transparency",
    ),
    Mutant(
        # THE CENSORING CORRECTION, DELETED.
        "M348-right-censored-boundaries-are-counted-as-at-risk",
        "f_of_k_at_risk stops dropping boundaries that sit closer to the "
        "end of the circuit than k gates, so it becomes the naive "
        "statistic it exists to correct -- and f(63) goes back to "
        "measuring circuit LENGTH",
        "        at_risk = [b for b in trace if n_gates - 1 - b.index >= k]",
        "        at_risk = list(trace)",
        "test_the_AT_RISK_estimate_drops_boundaries_that_COULD_NOT_run_k",
        target="transparency",
    ),
    Mutant(
        # AN EMPTY DENOMINATOR REPORTED AS A NUMBER.
        "M349-an-empty-at-risk-denominator-returns-zero-not-None",
        "an at-risk fraction over an empty denominator returns 0.0 "
        "instead of None, so a band nobody sampled reads as a measured "
        "zero -- and the 'f(63) < 0.50 -> NOT A KILL' row fires on it",
        "            out[k] = (None, 0)",
        "            out[k] = (0.0, 0)",
        "test_an_EMPTY_at_risk_denominator_is_None_not_a_number",
        target="transparency",
    ),
    Mutant(
        # THE SIBLING STATISTIC, LEFT WITH THE ORIGINAL FLAW.
        "M350-epsilon-is-normalised-by-the-whole-circuit-again",
        "epsilon_available divides each boundary by the WHOLE circuit "
        "instead of the gates actually after it, so it collapses back "
        "into the naive epsilon it exists to correct and both readings "
        "are depressed by boundary POSITION rather than by blocking",
        "    shares = [b.segment / (n_gates - 1 - b.index)",
        "    shares = [b.segment / n_gates",
        "test_EPSILON_has_the_SAME_censoring_flaw_and_the_SAME_correction",
        target="transparency",
    ),
    # ---- the third circuit family ----
    Mutant(
        # WITHOUT THE SWAP THIS IS JUST A LONGER ADDER.
        "M351-the-modular-rounds-stop-reusing-their-registers",
        "the swap between rounds is dropped, so each round is "
        "independent and the family stops exhibiting the ONE property "
        "it exists to test -- a correction pending from one round "
        "reaching the next round's magic",
        "            out.append(Gate(\"SWAP\", (a[i], b[i])))",
        "            pass",
        "test_the_MODULAR_family_REUSES_its_registers",
        target="transparency",
    ),
    Mutant(
        # THE WIDE CORRECTION, NARROWED BACK TO ONE QUBIT.
        "M352-the-sign-bits-correction-shrinks-to-the-sign-qubit",
        "the sign measurement declares its correction on the qubit that "
        "was measured rather than on the register the conditional "
        "addition acts upon, collapsing |R| and returning this family "
        "to the mis-rooting that made E1's conditional row unresolvable",
        "    out.append(Gate(\"MZ\", (sign,), fixup=tuple(b)))",
        "    out.append(Gate(\"MZ\", (sign,), fixup=(sign,)))",
        "test_the_SIGN_BIT_correction_spans_the_whole_target_register",
        target="transparency",
    ),
    Mutant(
        # A SUBSTITUTE THAT STOPS SAYING IT IS ONE.
        "M353-the-substituted-family-claims-the-papers-variant",
        "the standard modular-exponentiation block stops declaring that "
        "it is NOT arXiv:2505.15917's optimised variant, so every number "
        "computed on it silently becomes a claim about a construction "
        "nobody in this repository can check",
        '        "built as the STANDARD square-and-multiply block '
        '(Beauregard\'s "',
        '        "the modular exponentiation block of arXiv:2505.15917, "',
        "test_the_SUBSTITUTED_family_never_claims_the_paper",
        target="transparency",
    ),
    Mutant(
        # THE SHARED CARRY CHAIN, SILENTLY SHORTENED.
        "M354-the-shared-carry-chain-drops-its-last-position",
        "_ripple_add stops emitting the final CX, changing BOTH the "
        "plain adder and every modular adder built on it -- the exact "
        "drift that factoring the chain was meant to make impossible",
        '    out.append(Gate("CX", (a[-1], b[-1])))',
        "    pass",
        "test_the_ADDER_gate_lists_are_PINNED_across_the_refactor",
        target="transparency",
    ),
    # ---- the mixture decomposition ----
    Mutant(
        # THE BIMODAL POPULATION, AVERAGED BACK AWAY.
        "M355-the-mixture-second-reading-stops-firing",
        "a width spread past the limit no longer produces a second "
        "reading, so one f(63) is reported for two classes that read "
        "1.000 and 0.000 -- and the only engineering conclusion here, "
        "WHICH corrections stay transparent, disappears into a mean",
        "    if spread is not None and spread > MIXTURE_SPREAD_LIMIT:",
        "    if False:",
        "test_the_MIXTURE_is_REPORTED_not_averaged_away",
        target="transparency",
    ),
    Mutant(
        # EVERY BOUNDARY IN ONE BUCKET.
        "M356-the-width-breakdown-stops-splitting-by-width",
        "boundaries are bucketed by a constant rather than by |Q_j|, so "
        "the breakdown has one class, no spread is ever detectable and "
        "the detector reports a single population on every circuit",
        "        w = len(b.qubits)",
        "        w = 1",
        "test_the_aggregate_f63_HIDES_TWO_OPPOSITE_populations",
        target="transparency",
    ),
    Mutant(
        # A CLASS TOO SMALL TO RESOLVE, VOTING ANYWAY.
        "M357-an-unresolvable-width-class-votes-in-the-spread",
        "the minimum-at-risk filter is dropped, so a class of two or "
        "three boundaries -- standard error ~0.35 -- can set the spread "
        "and report sampling noise as a bimodal population",
        "            if r[\"at_risk\"] >= MIXTURE_MIN_AT_RISK",
        "            if True",
        "test_an_UNRESOLVABLE_class_is_EXCLUDED_from_the_spread",
        target="transparency",
    ),
    Mutant(
        # ONE CLASS REPORTED AS NO SPREAD RATHER THAN NO ANSWER.
        "M358-a-single-width-class-reports-a-spread-of-zero",
        "with fewer than two resolvable classes the spread returns 0.0 "
        "instead of None, so 'nothing could be compared' reads as "
        "'measured, and the population is uniform'",
        "    return max(vals) - min(vals) if len(vals) >= 2 else None",
        "    return max(vals) - min(vals) if len(vals) >= 2 else 0.0",
        "test_a_SINGLE_population_reports_NO_spread",
        target="transparency",
    ),
    # ---- the sixth row: D1+D1.5 dead fraction ----
    Mutant(
        # A BLOCKED CORRECTION COUNTED AS DEAD.
        "M359-a-blocked-correction-is-counted-as-dead",
        "the never-blocked clause is dropped, so a correction the "
        "decoder WAS waited on counts toward the dead fraction -- the "
        "one number that is supposed to mean 'never had to be resolved'",
        "               if b.blocked_by is None and not obs.intersection(b.cone))",
        "               if not obs.intersection(b.cone))",
        "test_a_BLOCKED_correction_is_never_counted_dead",
        target="transparency",
    ),
    Mutant(
        # THE OBSERVABLES GUESSED INSTEAD OF DECLARED.
        "M360-the-dead-fraction-guesses-the-observables",
        "the undeclared-observables refusal is removed, so the row "
        "reports a fraction derived from a guess about what the circuit "
        "is FOR -- and 'none' would report every correction dead",
        "    if observables is None:\n        raise ObservablesUndeclared(",
        "    if False:\n        raise ObservablesUndeclared(",
        "test_the_DEAD_row_REFUSES_to_guess_the_observables",
        target="transparency",
    ),
    Mutant(
        # A LOWER BOUND USED TO REFUTE A `>=` CLAIM.
        "M361-a-lower-bound-below-the-bar-is-reported-as-a-failure",
        "the sub-bar branch stops saying UNRESOLVED, so a bound that "
        "undercounts BY CONSTRUCTION is used to refute the claim it "
        "cannot refute -- 'I could not see it' reported as 'it is not "
        "there'",
        '                 f"UNRESOLVED: the computable LOWER BOUND is {dead:.3f}, "',
        '                 f"FAILED: the dead fraction is {dead:.3f}, "',
        "test_the_DEAD_row_reports_UNRESOLVED_not_FAILED_below_the_bar",
        target="transparency",
    ),
    Mutant(
        # THE ROW REMOVED AGAIN.
        "M362-the-sixth-row-goes-missing-again",
        "the D1 row stops being appended when no observables are "
        "declared, so the table silently answers five questions while "
        "claiming six -- and an absent row reads as a row that passed",
        '             "UNMEASURED: no final observables were declared for this "',
        '             "" if True else "',
        "test_a_census_with_NO_observables_says_UNMEASURED_not_zero",
        target="transparency",
    ),
    # ---- E2: the six-policy bake-off ----
    Mutant(
        # THE COST THAT DECIDES ABORT-RETRY.
        "M363-abort-retry-does-not-repay-the-magic-it-redoes",
        "a retried block stops recharging its T gates, so abort-retry "
        "buys a lower logical error with time alone and dominates every "
        "policy for a reason that is an accounting omission",
        "    return _axes(\"abort_retry\", p_L_kept, (w.rounds + stall) * attempts,\n"
        "                 w.t_count * attempts,",
        "    return _axes(\"abort_retry\", p_L_kept, (w.rounds + stall) * attempts,\n"
        "                 w.t_count,",
        "test_abort_retry_is_charged_for_the_MAGIC_it_redoes",
        target="bakeoff",
    ),
    Mutant(
        # SPECULATION MADE FREE.
        "M364-D7-speculates-without-paying-magic",
        "the expected magic charge stops accumulating, so D7 buys time "
        "for nothing and strictly dominates plain deferral -- the exact "
        "free lunch the T-count axis exists to price",
        "            charge += expected",
        "            pass",
        "test_D7_pays_MAGIC_for_the_time_it_saves",
        target="bakeoff",
    ),
    Mutant(
        # THE RULE BECOMES A HABIT.
        "M365-D7-speculates-regardless-of-the-posterior",
        "the speculate/stall comparison is removed, so D7 speculates on "
        "a coin-flip posterior at a platform where the magic costs more "
        "than the stall it avoids",
        "        if expected * magic < stall_saved:",
        "        if True:",
        "test_D7_declines_to_speculate_when_the_posterior_is_BAD",
        target="bakeoff",
    ),
    Mutant(
        # THE ASYMMETRY THAT DECIDED THE FRONT.
        "M366-undo-fixup-redo-rolls-back-a-fixed-shallow-depth",
        "the rollback depth stops being the decoder latency, so "
        "Undo-Fixup-Redo is billed for undoing less than it executed "
        "and never-waiting looks nearly free",
        "    rollback = blocked * p_mis_mean * w.rollback_rounds",
        "    rollback = blocked * p_mis_mean",
        "test_the_ROLLBACK_DEPTH_is_the_decoder_latency_not_the_block_length",
        target="bakeoff",
    ),
    Mutant(
        # A TIE REPORTED AS A WIN.
        "M367-a-tie-counts-as-strict-dominance",
        "dominance stops requiring a strict improvement on any axis, so "
        "two identical policies each dominate the other and the Pareto "
        "front empties",
        "    return any(u < v - DOMINANCE_EPS for u, v in zip(xa, xb))",
        "    return True",
        "test_strict_dominance_needs_BETTER_on_one_and_NOT_WORSE_on_any",
        target="bakeoff",
    ),
    Mutant(
        # THE PARTIAL ORDER MADE TOTAL.
        "M368-dominance-ignores-the-axes-a-policy-is-worse-on",
        "the worse-on-any-axis veto is dropped, so a policy that is "
        "cheaper in spacetime dominates one that is cheaper in magic -- "
        "an exchange rate nobody declared, applied silently",
        "    if any(u > v + DOMINANCE_EPS for u, v in zip(xa, xb)):",
        "    if False:",
        "test_the_front_keeps_INCOMPARABLE_policies",
        target="bakeoff",
    ),
    Mutant(
        # INV-2 STOPS COUNTING.
        "M369-INV2-accepts-a-speculation-with-no-D7-record",
        "the speculation/record count comparison is removed, so a "
        "policy may depend on a live bit with no D7 record at all -- "
        "which is precisely what INV-2 forbids",
        "        if spec and c.policy == \"deferral_d7\" and len(c.d7_records) != spec:",
        "        if False:",
        "test_INV2_catches_a_speculation_with_NO_RECORD",
        target="bakeoff",
    ),
    Mutant(
        # THE KILL THAT CANNOT DISTINGUISH BURSTY FROM UNBOUNDED.
        "M370-repair-debt-is-judged-by-its-mean-not-its-minima",
        "the debt test compares window MEANS instead of MINIMA, so a "
        "queue that drains fully every few rounds but bursts higher "
        "reads as growing without bound",
        "    return min(debt_by_round[half:]) > min(debt_by_round[:half]) + tol",
        "    return (sum(debt_by_round[half:]) / len(debt_by_round[half:])\n"
        "            > sum(debt_by_round[:half]) / len(debt_by_round[:half]) + tol)",
        "test_the_REPAIR_DEBT_kill_is_reachable_and_calibrated_both_ways",
        target="bakeoff",
    ),
    # ---- GA-11: the hardware readout analysis ----
    Mutant(
        # THE UNITS MISTAKE THAT ONCE MADE THE WAY MODEL LOSE EVERYWHERE.
        "M371-eps-is-reported-as-a-probability-not-an-RMS-error",
        "the eps = 2 sqrt(p) conversion is dropped, so a probability is "
        "reported in the units of A -- the exact error that made the WAY "
        "model read 'unphysical at every scale' the first time",
        "        \"eps_empirical\": 2.0 * math.sqrt(p_emp),",
        "        \"eps_empirical\": p_emp,",
        "test_eps_is_an_RMS_ERROR_IN_UNITS_OF_A_not_a_probability",
        target="ga11_hardware_iq",
    ),
    Mutant(
        # THE FINDING, DELETED.
        "M372-the-empirical-error-is-replaced-by-the-model-one",
        "the counted misassignments are replaced by the Gaussian "
        "prediction, so the 527x disagreement that IS this module's "
        "result becomes exactly 1.0 and the device looks textbook",
        "    p_emp = (wrong0 + wrong1) / (n0 + n1)",
        "    p_emp = p_model",
        "test_a_NON_GAUSSIAN_TAIL_is_what_the_ratio_detects",
        target="ga11_hardware_iq",
    ),
    Mutant(
        # A DEAD READOUT REPORTED AS A PERFECT ONE.
        "M373-a-qubit-that-did-not-read-out-is-not-refused",
        "identical centroids stop being refused, so a dead readout "
        "divides by zero separation or reports an error of zero -- an "
        "instrument that says a broken qubit is a flawless one",
        "    if sep == 0:",
        "    if False:",
        "test_analyse_REFUSES_a_qubit_that_did_not_read_out",
        target="ga11_hardware_iq",
    ),
    Mutant(
        # A HANDFUL OF SHOTS REPORTED AS A MEASUREMENT.
        "M374-too-few-shots-are-scored-anyway",
        "the minimum-shot refusal is removed, so a discrimination error "
        "with no resolution is computed and written into an artifact",
        "    if iq0.size < 32 or iq1.size < 32:",
        "        if False:",
        "test_analyse_REFUSES_too_few_shots",
        target="ga11_hardware_iq",
    ),
    # ---- distance_portfolio: the refusals a portfolio must not soften ----
    Mutant(
        # A PORTFOLIO THAT VOTES LAUNDERS A WRONG ANSWER.
        "M375-engine-disagreement-is-resolved-instead-of-raised",
        "two engines returning different answers for the SAME instance "
        "stops being a refusal, so the portfolio silently picks one -- "
        "which is exactly the laundering the module's docstring says "
        "must never happen: one of those engines is broken and that is "
        "the finding, not the vote",
        "    if len(answers) > 1:",
        "    if False:",
        "test_a_SPLIT_is_REFUSED_and_never_voted_on",
        target="distance_portfolio",
    ),
    Mutant(
        # A YES WHOSE WITNESS NOBODY RE-CHECKED.
        "M376-a-YES-with-an-unverifiable-witness-is-accepted",
        "the portfolio-boundary re-check is removed, so an engine may "
        "answer YES with a witness that does not validate -- the gap "
        "the second check exists to close, and it costs microseconds",
        "        if not ok:\n            raise Disagreement(",
        "        if False:\n            raise Disagreement(",
        "test_a_YES_with_an_UNVERIFIABLE_witness_is_REFUSED",
        target="distance_portfolio",
    ),
    Mutant(
        # A BUDGET THAT CANNOT BE ENFORCED, REPORTED AS ENFORCED.
        "M377-an-uninterruptible-solver-is-given-a-budget-anyway",
        "a solver that cannot be interrupted stops being refused, so "
        "`budget_s` becomes a comment and any bound applied to it is a "
        "claim about a timeout that never fired",
        "        raise BudgetNotEnforceable(",
        "        pass  # noqa",
        "test_a_SOLVER_THAT_CANNOT_BE_STOPPED_is_REFUSED_not_humoured",
        target="distance_portfolio",
    ),
    # ---- latent_parity + branch_certificate: declared, now mutated ----
    Mutant(
        # A RATE OUTSIDE [0,1] IS NOT A PROBABILITY.
        "M378-a-mechanism-rate-outside-zero-one-is-accepted",
        "the probability-range refusal is dropped, so a 'rate' of 1.7 or "
        "-0.2 enters the parity model and every downstream likelihood is "
        "computed from a number that is not a probability",
        '            raise ValueError("a mechanism rate outside [0, 1]")',
        "            pass",
        # *** THE KILLER NAMED HERE COULD NOT SEE THIS LINE. ***
        # It was `test_a_boundary_prior_is_refused`, which refuses a
        # Beta(0.5,0.5) PRIOR inside `fit_em` and never constructs a
        # Model at all. Plausible name, same file, same word
        # "refused" -- and the mutant SURVIVED the first run that
        # actually executed it. A mutant is only as good as the test
        # named beside it.
        "test_a_rate_OUTSIDE_zero_one_is_refused",
        target="latent_parity",
    ),
    Mutant(
        # A SHAPE MISMATCH SILENTLY BROADCAST.
        "M379-a-rate-vector-of-the-wrong-length-is-broadcast",
        "the rates-vs-mechanisms shape check is removed, so numpy "
        "broadcasts a wrong-length rate vector and the model fits a "
        "different number of mechanisms than it was given",
        "            raise ValueError(f\"{p.shape} rates for {A.shape[0]} mechanisms\")",
        "            pass",
        # Same wrong killer, same survival. See M378.
        "test_a_rate_vector_of_the_WRONG_LENGTH_is_refused_not_broadcast",
        target="latent_parity",
    ),
    Mutant(
        # A BOUND CERTIFIED WITHOUT THE SEARCH THAT EARNS IT.
        "M380-the-node-budget-is-never-enforced",
        "the branch-and-bound node ceiling stops raising, so a search "
        "that ran out of budget returns a certificate for a bound it "
        "never finished proving -- an unbounded claim from a bounded "
        "search",
        '            raise RuntimeError(f"exceeded {max_nodes} nodes")',
        "            pass",
        # *** THE KILLER NAMED HERE CLOSED AT THE ROOT. ***
        # `test_it_REFUSES_to_certify_a_false_bound` uses an
        # instance that certifies in ONE node, so it never came
        # near the ceiling and the mutant SURVIVED. The replacement
        # drives a search that genuinely branches.
        "test_the_NODE_BUDGET_is_enforced_and_refuses_rather_than_certifying",
        target="branch_certificate",
    ),
    Mutant(
        # A MALFORMED CERTIFICATE ACCEPTED.
        "M381-a-node-with-no-branch-and-no-dual-is-accepted",
        "a tree node carrying neither a branching variable nor a dual "
        "stops being refused, so a certificate with an unjustified step "
        "checks out -- the step that proves nothing is the one a forger "
        "would insert",
        '            raise ValueError("no branching variable and no dual: the "',
        "            pass  # noqa",
        "test_the_checker_refuses_malformed_documents",
        target="branch_certificate",
    ),
    # ---- batch 2: refusals that nothing proved could fire ----
    Mutant(
        # A FLOAT DUAL IN AN EXACT CERTIFICATE.
        "M382-a-float-dual-is-accepted-into-an-exact-certificate",
        "the exact-pair type refusal is dropped, so a floating-point "
        "dual enters a certificate whose whole value is that it is "
        "EXACT -- and weak duality checked in floats is an inequality "
        "about rounding, not about the cut",
        '    raise TypeError("dual values must be exact {num,den} pairs, not floats")',
        "    return Fraction(0, 1)",
        # THE RIGHT KILLER ALREADY EXISTED AND WAS NOT NAMED.
        # The mutant pointed at a test that passes only well-formed
        # duals, so deleting the float refusal changed nothing it
        # could see, and it SURVIVED.
        "test_exact_cut_forbids_float_dual_values",
        target="matching_cert_strict",
    ),
    Mutant(
        # A ZERO DENOMINATOR SILENTLY ADMITTED.
        "M383-a-zero-denominator-becomes-a-fraction",
        "the zero-denominator refusal stops firing, so a dual value with "
        "den = 0 is constructed and every comparison downstream is "
        "against an undefined quantity",
        '            raise ValueError("zero denominator")\n        return Fraction(int(value["num"]), den)',
        '            pass\n        return Fraction(int(value["num"]), den or 1)',
        # No test existed for this branch at all; the named killer
        # passes only well-formed duals. Written for it.
        "test_a_ZERO_DENOMINATOR_is_refused_in_the_MAPPING_form",
        target="matching_cert_strict",
    ),
    Mutant(
        # THE REDUCTION BUILT ON A DIFFERENT SUBSPACE.
        "M384-a-disagreeing-certified-rank-is-accepted",
        "the certified-rank agreement check is removed, so the reduction "
        "is built on a span the closure certificate does not describe -- "
        "and every question below it becomes a question about a "
        "different subspace, which the module's own message says",
        "    if expect_certified_rank is not None and rank_S != expect_certified_rank:",
        "    if False:",
        "test_a_DISAGREEING_certified_rank_is_REFUSED",
        target="quotient_reduction",
    ),
    Mutant(
        # A RANK THAT EXCEEDS ITS OWN BOUND.
        "M385-a-rank-exceeding-r-is-not-refused",
        "`rank_S > r` stops being refused, so the reduction proceeds "
        "with more independent directions than the codimension admits "
        "and the question count it derives is meaningless",
        "    if rank_S > r:",
        "    if False:",
        # *** THE RIGHT KILLER WAS IN THE FILE ALL ALONG. ***
        # `test_the_question_count_is_2_to_the_codim_minus_one`
        # counts quotient classes and never approaches this branch,
        # so the mutant SURVIVED. `test_a_rank_ABOVE_r_is_REFUSED`
        # reaches it by stubbing `_rank` -- the honest way to test a
        # guard whose precondition cannot fail for real input.
        "test_a_rank_ABOVE_r_is_REFUSED",
        target="quotient_reduction",
    ),
    # ---- batch 3: a pin that cannot say what it is FOR ----
    Mutant(
        # A PIN NAMING BYTES THAT DO NOT EXIST IN THE COMMIT.
        "M386-a-pin-may-name-a-file-absent-from-its-commit",
        "the not-in-commit refusal is dropped, so a pin can name bytes "
        "that were never in the immutable commit it cites -- and a pin "
        "to bytes nobody can retrieve is a hash with no referent",
        "        raise NotTracked(\n            f\"{path} is not in {commit[:8]}; a pin may only name bytes \"",
        "        return b\"\"  # noqa\n        raise NotTracked(\n            f\"{path} is not in {commit[:8]}; a pin may only name bytes \"",
        "test_a_pin_naming_a_commit_that_lacks_the_file_is_refused",
        target="git_pin",
    ),
    Mutant(
        # A BINDING THAT CANNOT STATE ITS ROLE.
        "M387-a-pin-need-not-say-what-it-is-for",
        "the empty-role refusal stops firing, so a pin is created that "
        "cannot say what it binds -- and a binding with no stated role "
        "cannot state what its own staleness would MEAN, which is the "
        "only thing a pin is for",
        "    if not role or not role.strip():",
        "    if False:",
        # *** OFF BY ONE TEST. *** The killer named here was the
        # test immediately BELOW the right one in the same file:
        # `test_an_UNKNOWN_byte_domain_is_refused` exercises the
        # DOMAIN guard, not the role guard, so deleting this line
        # changed nothing it could see and the mutant SURVIVED.
        "test_a_pin_must_say_what_it_is_FOR",
        target="git_pin",
    ),
    Mutant(
        # NOT A REPOSITORY, PINNED ANYWAY.
        "M388-a-non-repository-still-yields-a-head-commit",
        "the not-a-repository refusal is removed, so `head_commit` "
        "returns whatever the failed git call left in stdout and every "
        "pin below it cites a commit that does not exist",
        '        raise NotTracked("not a git repository, so nothing can be pinned")',
        "        return \"0\" * 40",
        "test_blob_bytes_and_head_commit_refuse_outside_a_repository",
        target="git_pin",
    ),
    # ---- batch 4: a requirement that can never be refused ----
    Mutant(
        # A FIELD THE MANIFEST HAS NO VOCABULARY FOR.
        "M389-a-plan-may-require-a-field-that-does-not-exist",
        "the unknown-field refusal is dropped, so a plan can require "
        "something the manifest cannot express -- and a requirement that "
        "can NEVER be refused for lacking it is a requirement that does "
        "nothing, which is the module's own wording",
        "        if n.field not in FIELDS:",
        "        if False:",
        "test_a_requirement_must_name_a_REAL_field",
        target="capability_plan",
    ),
    Mutant(
        # A REQUIREMENT WITH NO STATED REASON.
        "M390-a-requirement-need-not-state-why",
        "the empty-reason refusal stops firing, so a requirement enters "
        "with no rationale -- and only a requirement whose reason is "
        "stated can ever be safely DROPPED, which is the entire purpose "
        "of a plan-specific set",
        "        if not n.because.strip():",
        "        if False:",
        "test_a_requirement_must_state_WHY",
        target="capability_plan",
    ),
    Mutant(
        # AN ENUM VALUE OUTSIDE ITS OWN DOMAIN.
        "M391-an-enum-requirement-outside-its-domain-is-accepted",
        "the enum-domain check is removed, so a plan demands a value the "
        "field can never take and the resulting verdict is a comparison "
        "against a string nothing produces",
        "        if kind == \"enum\" and n.want not in ENUM_DOMAIN[n.field]:",
        "        if False:",
        "test_an_ENUM_requirement_outside_its_domain_is_refused",
        target="capability_plan",
    ),
    Mutant(
        # A STRICT MAJORITY MADE NON-STRICT.
        # *** THIS SLOT HELD AN EQUIVALENT MUTANT AND SURVIVED. ***
        # It swapped `>` for `>=`. With `d` odd, `2*sum` is even, so
        # equality is UNREACHABLE and the two comparisons cannot differ
        # on any input -- unkillable, and correctly reported SURVIVED by
        # this gate. The response was not to weaken the killer but to
        # enforce the precondition that made it equivalent (the function
        # now refuses an even `d` instead of trusting argparse to) and
        # to aim at a difference that is real on every input.
        "M392-the-majority-vote-direction-is-inverted",
        "the decode reports a logical error whenever the majority reads "
        "ZERO, so a clean run is scored as an almost total failure -- "
        "p_L 2.6e-4 becomes 0.99974, and Lambda is a ratio of two "
        "numbers that both mean their own opposite",
        "    majority = (counts_fin.sum(axis=1) * 2 > d)",
        "    majority = (counts_fin.sum(axis=1) * 2 < d)",
        "test_the_MAJORITY_DIRECTION_is_not_reversible",
        target="hardware_lambda",
    ),
    Mutant(
        # AN EMPTY SAMPLE SCORED AS A PERFECT ONE.
        "M393-zero-shots-are-scored-as-a-zero-error-rate",
        "the empty-sample refusal is dropped, so a run that returned no "
        "shots at all reports p_L = 0 -- a perfect logical error rate "
        "recorded in the evidence for a measurement that never happened",
        '        raise ValueError("no shots to score")',
        "        pass",
        "test_ZERO_shots_is_REFUSED_not_scored_as_zero",
        target="hardware_lambda",
    ),
    Mutant(
        # A ZERO RATE REPORTED AS A PRECISE ONE.
        "M394-a-zero-error-rate-claims-zero-relative-error",
        "the infinite relative error on a zero-count rate becomes 0.0, "
        "so a run that saw nothing reads as the TIGHTEST measurement in "
        "the artifact rather than the least informative one",
        '            "rel_stderr": (se / p) if p > 0 else float("inf"),',
        '            "rel_stderr": (se / p) if p > 0 else 0.0,',
        "test_a_ZERO_rate_reports_INFINITE_relative_error_not_zero",
        target="hardware_lambda",
    ),
    Mutant(
        # A COUNT-BASED RESOLUTION TEST MADE VACUOUS.
        "M395-any-number-of-events-counts-as-resolved",
        "the >= 25 event floor for calling a rate resolved is removed, "
        "so the ONE logical error seen at d=5 would promote the point "
        "estimate of Lambda from UNRESOLVED to 26 -- a ten-fold "
        "overstatement of the device's suppression, published as a "
        "measurement",
        '            "resolved": bool(errors >= 25)}',
        '            "resolved": True}',
        "test_the_RESOLUTION_is_reported_and_gates_on_the_COUNT",
        target="hardware_lambda",
    ),
    Mutant(
        # A ONE-SIDED BOUND BUILT ON AN UNRESOLVED NUMERATOR.
        "M396-a-lambda-bound-is-computed-from-an-unresolved-numerator",
        "the requirement that the NUMERATOR be resolved is dropped, so "
        "a ratio of two unresolved rates -- noise over a confidence "
        "limit -- is published as a 95% lower bound on Lambda",
        '        if lo["resolved"]:',
        "        if True:",
        "test_a_BOUND_needs_a_RESOLVED_numerator",
        target="hardware_lambda",
    ),
    Mutant(
        # AN EXACT LIMIT REPLACED BY THE APPROXIMATION IT EXISTS TO AVOID.
        "M397-the-poisson-limit-becomes-a-normal-approximation",
        "the exact Garwood limit is replaced by k + 1.96*sqrt(k), which "
        "at the k <= 5 this experiment actually lands on understates "
        "the upper limit by more than half (2.96 against 4.74 at k=1) "
        "and therefore OVERSTATES the lower bound on Lambda -- the "
        "error points in the flattering direction",
        "    return float(0.5 * chi2.ppf(0.95, 2 * (k + 1)))",
        "    return float(k + 1.96 * (k ** 0.5))",
        "test_the_TABLE_calibrates_the_COMPUTATION",
        target="hardware_lambda",
    ),
    Mutant(
        # A ZERO NUMERATOR TURNED INTO INFINITE SUPPRESSION.
        "M398-a-zero-numerator-rate-is-allowed-to-form-a-ratio",
        "the zero-numerator refusal is dropped, so a distance that saw "
        "no errors at all -- nothing to suppress -- yields a ratio "
        "rather than a stated reason, and a division by the smaller "
        "rate reports suppression the data cannot show",
        '        if lo["p_L"] <= 0:',
        "        if False:",
        "test_a_ZERO_numerator_is_REFUSED_with_its_reason",
        target="hardware_lambda",
    ),
    Mutant(
        # THE STRONGEST BOUND REPLACED BY THE WEAKEST.
        "M399-the-reported-lambda-bound-is-the-weakest-step",
        "the aggregate lower bound takes min instead of max across "
        "steps, discarding the strongest claim the data supports -- "
        "each step's bound is independently valid, so reporting the "
        "smallest understates what was measured",
        "            max(bounds) if bounds else None)",
        "            min(bounds) if bounds else None)",
        "test_EVERY_STEP_is_reported_and_the_bound_is_the_STRONGEST",
        target="hardware_lambda",
    ),
    Mutant(
        # A KILL ROW'S REACHABILITY TEST INVERTED.
        "M400-an-unreachable-kill-row-reports-itself-reachable",
        "the reachability test flips to > 0.5, so a KILL whose "
        "threshold no posterior can reach -- p_mis cannot exceed one "
        "half -- is reported as a live check, and E2's sensitivity "
        "table says the opposite of what it measures at every lambda",
        "    return d7_decline_threshold(d, t_d_rounds, lam) < 0.5",
        "    return d7_decline_threshold(d, t_d_rounds, lam) > 0.5",
        "test_a_THRESHOLD_ABOVE_ONE_HALF_is_a_KILL_THAT_CANNOT_FIRE",
        target="bakeoff",
    ),
    Mutant(
        # THE THRESHOLD AND THE VERDICT MADE TO DISAGREE.
        "M401-the-decline-threshold-inverts-its-own-lambda-dependence",
        "lambda moves from the denominator to the numerator of the "
        "decline threshold, so MORE suppression would demand MORE "
        "confidence to decline -- the sign of the whole surcharge "
        "argument reversed, while every individual verdict still "
        "returns a plausible boolean",
        # SINGLE-LINE anchor: a multi-line one has to carry its own
        # newline through this file's string literal, and the first
        # attempt put a REAL newline here and broke the module.
        "            / (lam * D7_MAGIC_PER_SPECULATION"
        " * magic_cost_qubit_rounds(d)))",
        "            * lam / (D7_MAGIC_PER_SPECULATION"
        " * magic_cost_qubit_rounds(d)))",
        "test_the_threshold_FALLS_as_lambda_RISES",
        target="bakeoff",
    ),
    Mutant(
        # A SENSITIVITY TABLE THAT PICKS A WINNER.
        "M402-the-sensitivity-table-keeps-only-the-reachable-rows",
        "rows whose KILL cannot fire are filtered out of the "
        "sensitivity, so the table shows only the lambdas that make the "
        "row look live -- the simulated value, which is the one E2 "
        "actually used and the one that makes it dark, disappears from "
        "the artifact entirely",
        '        out.append({"source": name, "lambda": lam,',
        '        if thr >= 0.5:\n            continue\n'
        '        out.append({"source": name, "lambda": lam,',
        "test_the_SENSITIVITY_reports_every_candidate_and_never_picks_one",
        target="bakeoff",
    ),
    Mutant(
        # THE FALSE CERTIFICATION, RESTORED.
        "M403-the-router-goes-back-to-grepping-the-lane-s-note",
        "the MITM lane's proof obligation is read out of English again "
        "instead of from its boolean. The lane's TIME-OUT note says "
        "'time budget exhausted; NO exhaustive claim at any W', which "
        "CONTAINS the substring -- so a sweep that ran out of budget "
        "and proved nothing certifies the distance, and the router "
        "reports decided=True on a bound nothing established",
        "        if cert.exhaustive:",
        '        if "exhaustive" in cert.note:',
        "test_a_TIMED_OUT_mitm_lane_MUST_NOT_CERTIFY_THE_BOUND",
        target="certify",
    ),
    Mutant(
        # A SWEEP THAT FOUND SOMETHING CLAIMS TO HAVE FOUND NOTHING.
        "M404-a-completed-sweep-always-claims-exhaustiveness",
        "the exhaustiveness flag stops depending on whether a lighter "
        "operator was FOUND, so a sweep that found one -- which "
        "improves the bound rather than proving it -- is published as "
        "a completed negative result",
        "                           exhaustive=not found)",
        "                           exhaustive=True)",
        "test_mitm_matches_brute_force_both_directions",
        target="mitm",
    ),
    Mutant(
        # AN INVALID CORRECTION REPORTS ITSELF CONSISTENT.
        "M405-a-failed-T-join-test-declares-the-syndrome-consistent",
        "the odd-degree refusal declares syndrome_consistent=True, so a "
        "correction whose odd-degree set does NOT equal the syndrome is "
        "published in a degraded receipt whose own text claims the "
        "correction explains the syndrome -- a false statement in an "
        "artifact, from the one place that actually knows the answer",
        "            syndrome_consistent_declared=False)",
        "            syndrome_consistent_declared=True)",
        "test_the_T_JOIN_verdict_is_a_FIELD_not_a_phrase_in_the_reason",
        target="matching_cert",
    ),
    Mutant(
        # "NOT ASKED" COLLAPSES INTO "ANSWERED YES".
        "M406-an-unasked-T-join-question-answers-itself-yes",
        "the third state is removed: an input rejected BEFORE the T-join "
        "test ever ran reports syndrome_consistent=True rather than "
        "None, so a malformed certificate the checker never got far "
        "enough to evaluate is credited with explaining its syndrome",
        '        got = self.checks.get("t_join_valid")\n'
        "        return bool(got) if got is not None else None",
        "        return True",
        "test_a_MALFORMED_input_leaves_the_verdict_UNASKED",
        target="matching_cert",
    ),
    Mutant(
        # CONTROL FLOW MUST NOT BE PARSED OUT OF A REFUSAL SENTENCE.
        "M407-ladder-escalation-goes-back-to-grepping-refusal-prose",
        "the structured only_obstruction_is_gap result is discarded and "
        "the decoder again searches the human-readable refusal reason; a "
        "wording-only edit can then stop a useful escalation or repeat an "
        "expensive one whose actual obstruction cannot be repaired",
        "            if not r.only_obstruction_is_gap:",
        '            if "NOT PROVEN OPTIMAL" not in r.reason:',
        "test_ladder_escalation_is_a_field_not_a_refusal_phrase",
        target="certifying_decoder",
    ),
    Mutant(
        # THE CONSERVATIVE DEFAULT INVERTED.
        "M408-a-refusal-that-never-set-the-flag-claims-to-be-gap-only",
        "the escalation flag defaults to True, so a refusal that never "
        "set it -- a malformed document, an odd-degree mismatch -- is "
        "read as an unclosed gap and the ladder pays three rungs for a "
        "defect no candidate family can repair. A default must be the "
        "answer that is safe when nobody spoke",
        "    only_obstruction_is_gap: bool = False",
        "    only_obstruction_is_gap: bool = True",
        "test_the_gap_flag_DEFAULTS_to_the_non_escalating_state",
        target="matching_cert",
    ),
    Mutant(
        # THE PRECONDITION AN EQUIVALENT MUTANT EXPOSED.
        "M409-an-even-distance-is-scored-instead-of-refused",
        "the even-distance refusal is dropped, so a repetition code with "
        "no majority is decoded anyway and every tie is broken silently "
        "in one direction. This guard exists because a mutant swapping "
        "> for >= was EQUIVALENT under odd d and could not be killed; "
        "the precondition is what makes the comparison's strictness a "
        "property rather than an accident of parity",
        "    if d % 2 == 0:",
        "    if False:",
        "test_an_EVEN_distance_has_no_majority_and_is_REFUSED",
        target="hardware_lambda",
    ),
    Mutant(
        # AN IMPOSSIBLE DUAL CLAMPED INSTEAD OF REFUSED.
        "M410-a-dual-above-its-primal-is-clamped-not-refused",
        "weak duality forbids a dual above its primal, so seeing one "
        "means the packing is infeasible or the two sides disagree "
        "about the edge weights -- this repository caught exactly that "
        "at d=7. Clamping turns a producer defect into a quietly "
        "optimistic measurement: wrong, and silent",
        "    if bound > primal + 1e-9:",
        "    if False:",
        "test_a_dual_ABOVE_its_primal_is_REFUSED_not_clamped",
        target="certified_weight",
    ),
    Mutant(
        # THE SURVIVORSHIP FILTER, WHICH IMPROVES AS THE DECODER WORSENS.
        "M411-the-device-floor-averages-only-the-CERTIFIED-shots",
        "refused shots are dropped from the average. They are the HARD "
        "shots -- more error, weaker packing -- so the reported floor "
        "falls, and it falls FURTHER every time the decoder gets worse. "
        "A measurement that improves when the machinery degrades is the "
        "survivorship error in its purest form",
        "        ivs.append(iv)",
        "        if iv.certified:\n            ivs.append(iv)",
        "test_REFUSED_shots_are_counted_and_their_exclusion_would_FLATTER",
        target="certified_weight",
    ),
    Mutant(
        # A BOUND WHOSE COVERAGE IS UNKNOWN, PUBLISHED AS 95%.
        "M412-a-left-skewed-sample-still-claims-a-reliable-95-percent",
        "the shape check is removed, so a sample whose measured coverage "
        "is 0.640 against a nominal 0.95 is published beside a 95% "
        "label. MEASURED at 400 trials: left-skewed weights under-cover "
        "at every sample size, and neither the bootstrap nor the normal "
        "approximation repairs it",
        "    reliable = skew is None or skew > LEFT_SKEW_UNRELIABLE",
        "    reliable = True",
        "test_a_LEFT_SKEWED_sample_marks_the_population_bound_UNRELIABLE",
        target="certified_weight",
    ),
    Mutant(
        # THE UNITS QUIETLY ASSUMED TO BE THE FLATTERING ONES.
        "M413-an-undeclared-weight-convention-assumes-log-likelihood",
        "an undeclared convention defaults to the likelihood reading, so "
        "a run under UNIFORM weights -- which is what the live hardware "
        "lane actually used -- publishes a statement about PROBABILITY "
        "computed from a count of edges",
        "                 convention: str = UNDECLARED,",
        "                 convention: str = LOG_LIKELIHOOD,",
        "test_an_UNDECLARED_convention_is_UNCHECKABLE_not_assumed",
        target="certified_weight",
    ),
    Mutant(
        # THE DEBIAS OFF BY THE FACTOR THAT MATTERS.
        "M414-the-reliability-drops-the-factor-of-two",
        "reliability becomes 1 - p instead of 1 - 2p, so every shot is "
        "debiased by the wrong amount and the estimate is biased "
        "towards zero by a factor no number of shots removes",
        "    return min(math.tanh(gap / 2.0), MAX_RELIABILITY)",
        "    return min(1.0 - 1.0 / (1.0 + math.exp(gap)), MAX_RELIABILITY)",
        "test_the_reliability_is_EXACTLY_one_minus_twice_the_error_rate",
        target="soft_estimator",
    ),
    Mutant(
        # THE WEIGHTING MADE DECORATIVE.
        "M415-every-shot-is-weighted-the-same",
        "the inverse-variance weight becomes a constant, so the "
        "estimator reports a variance reduction it did not achieve and "
        "every error bar downstream is too small",
        "    return (a * a) / (1.0 - a * a)",
        "    return 1.0",
        # *** THE OLD KILLER STRUCTURALLY COULD NOT SEE THIS. ***
        # `test_weighting_BEATS_equal_weighting_and_the_gain_is_REAL`
        # compares `equal_weight_variance` -- built from the
        # RELIABILITIES -- against `variance`, built from the WEIGHTS.
        # Mutate only the weights and the ratio can still clear its bar,
        # because half of the comparison never saw the mutation. It
        # SURVIVED. The replacement asserts what weighting is FOR: a
        # near-certain shot must outvote a near-coin-flip.
        "test_a_CONFIDENT_shot_OUTVOTES_a_doubtful_one",
        target="soft_estimator",
    ),
    Mutant(
        # A COIN FLIP GIVEN A VOTE.
        # *** THIS SLOT HELD AN EQUIVALENT MUTANT AND SURVIVED. ***
        # It replaced the guard's condition with `False` -- but at gap 0
        # the reliability IS 0, so `a^2/(1-a^2)` evaluates to zero with
        # or without the branch. Nothing could kill it because nothing
        # changed. The mutation now hands a coin-flip shot a full vote,
        # which is the behaviour the guard actually prevents.
        "M416-a-zero-confidence-shot-is-given-a-full-vote",
        "a shot the decoder had no opinion about is handed unit weight "
        "instead of zero, so pure noise joins the average with the same "
        "authority as a near-certain shot",
        "    if a <= MIN_RELIABILITY:\n        return 0.0",
        "    if a <= MIN_RELIABILITY:\n        return 1.0",
        "test_a_ZERO_CONFIDENCE_shot_contributes_NOTHING_to_the_answer",
        target="soft_estimator",
    ),
    Mutant(
        # JENSEN, RESTORED.
        "M417-the-predicted-rate-is-evaluated-at-the-bins-mean-gap",
        "the per-shot predicted rate is replaced by the rate at the "
        "bin's MEAN gap. The logistic is convex, so this under-predicts "
        "the error rate -- MEASURED at +0.0070 on the [2,4) bin, a third "
        "of the materiality bar -- and every honestly calibrated run "
        "reads as OVERCONFIDENT, a systematic false positive in exactly "
        "the direction this module exists to flag",
        '        pred = c["pred_sum"] / c["n"]        # mean of p(g), not p(mean g)',
        '        pred = predicted_rate(mean_gap)',
        "test_an_HONEST_model_is_reported_CONSISTENT",
        target="dem_calibration",
    ),
    Mutant(
        # SIGNIFICANCE MISTAKEN FOR SIZE.
        "M418-calibration-is-judged-on-the-p-value-alone",
        "the effect-size bar is dropped, so a chi-square on a large run "
        "calls a deviation in the fourth decimal MISCALIBRATED. True, "
        "useless, and the fastest way to get a real gate switched off",
        "    elif significant and material:",
        "    elif significant:",
        "test_a_HUGE_sample_does_not_call_a_TINY_deviation_MISCALIBRATED",
        target="dem_calibration",
    ),
    Mutant(
        # A THIN CELL ALLOWED TO DECIDE.
        "M419-bins-too-thin-to-tell-a-rate-from-noise-are-scored-anyway",
        "the minimum-per-bin floor is removed, so a cell holding three "
        "shots contributes its 0-or-1 rate to the chi-square and the "
        "verdict is decided by sampling noise",
        '        if c["n"] < MIN_PER_BIN:',
        "        if False:",
        "test_THIN_bins_are_EXCLUDED_and_the_exclusion_is_REPORTED",
        target="dem_calibration",
    ),
    Mutant(
        # THE DIAGNOSIS THROWN AWAY.
        "M420-the-calibration-error-loses-its-sign",
        "the signed error becomes an absolute one, so the report can say "
        "the model and the device disagree but not WHICH WAY. "
        "Overconfident means every downstream logical error rate is "
        "optimistic; pessimistic means performance is being left "
        "unclaimed. Without the sign there is nowhere to look",
        '        signed_num += c["n"] * (obs - pred)',
        '        signed_num += c["n"] * abs(obs - pred)',
        "test_a_PESSIMISTIC_model_is_caught_with_the_OTHER_sign",
        target="dem_calibration",
    ),
    Mutant(
        # THE GUARD BLINDED IN THE ONE DIRECTION THAT MATTERS.
        "M421-calibration-power-counts-only-what-the-model-EXPECTED",
        "power is granted on expected events alone. Expected events are "
        "computed FROM the model, and an overconfident model understates "
        "exactly that -- so a bin with five REAL errors against 1.25 "
        "expected is scored powerless and the run reads UNCHECKABLE. "
        "MEASURED on ibm_marrakesh: that is the bin showing the device "
        "four times noisier than its own calibration claims",
        "            or c[\"wrong\"] >= MIN_EXPECTED_EVENTS)",
        "            or False)",
        "test_power_counts_OBSERVED_events_not_only_EXPECTED_ones",
        target="dem_calibration",
    ),
    Mutant(
        # ZERO ERRORS READ AS CERTAINTY.
        "M422-zero-observed-errors-becomes-a-zero-error-rate",
        "the Jeffreys debias rate becomes the MLE, so a bin with no "
        "observed errors claims p = 0 -- certainty derived from having "
        "seen nothing go wrong -- and hands itself unbounded weight. "
        "MEASURED: the ibm_marrakesh top bin held 3,896 clean shots, "
        "which support p <= 7.7e-4 and not zero",
        "        p_hat = (k + 0.5) / (n + 1.0)                 # Jeffreys",
        "        p_hat = k / n",
        "test_ZERO_observed_errors_does_NOT_become_certainty",
        target="soft_estimator",
    ),
    Mutant(
        # THE CONSERVATIVE RATE REPLACED BY THE POINT RATE.
        "M423-the-weights-use-the-point-rate-instead-of-the-upper-limit",
        "the weights stop using the one-sided upper limit and use the "
        "point estimate, so confidence is overstated and every error bar "
        "downstream is too small. The debias and the weights want "
        "OPPOSITE errors, and collapsing them to one rate is how a "
        "clamped model produced +1.3033 on hardware",
        "        w = (a_cons * a_cons) / (1.0 - a_cons * a_cons)",
        "        w = (a_pt * a_pt) / (1.0 - a_pt * a_pt)",
        "test_the_DEBIAS_and_the_WEIGHTS_use_DIFFERENT_rates_on_purpose",
        target="soft_estimator",
    ),
    Mutant(
        # CONFIDENCE ALLOWED TO RISE WITH DOUBT.
        "M424-the-measured-calibration-map-stops-being-monotone",
        "the running maximum is dropped, so sampling noise in a thin bin "
        "can give a HIGH-gap bin a worse measured error rate than a "
        "low-gap one -- inverting the weighting the estimator exists to "
        "apply, on exactly the shots it trusts most",
        "        running_hat = max(running_hat, out[b][\"p_hat\"])",
        "        running_hat = out[b][\"p_hat\"]",
        "test_the_map_is_MONOTONE_because_confidence_cannot_rise_with_doubt",
        target="soft_estimator",
    ),
    Mutant(
        # A RANGE CHECK THAT CRIES WOLF.
        "M425-the-out-of-range-check-goes-back-to-an-absolute-epsilon",
        "the self-falsification check stops scaling with the error bar, "
        "so an estimate 0.28 sigma past the boundary -- a value pinned "
        "at the edge of its own range -- is flagged exactly like one 22.5 "
        "sigma out. A check that cannot tell sampling noise from a broken "
        "model fires on every honest run and gets switched off",
        "    return abs(value) > 1.0 + OUT_OF_RANGE_SIGMA * se + 1e-12",
        "    return abs(value) > 1.0 + 1e-12",
        "test_out_of_range_is_RELATIVE_to_the_error_bar",
        target="soft_estimator",
    ),

    # ============================================================
    # THE i.i.d. ASSUMPTION (2026-09-03). A permutation test cannot
    # really get the p-value wrong -- the null is exact by
    # construction. What it CAN do is be blind, and report its
    # blindness as stability. Every mutant below produces a test that
    # still runs, still prints a p-value, and has stopped being able
    # to say anything.
    # ============================================================
    Mutant(
        # THE SENSITIVITY, DELETED.
        "M426-a-negative-result-stops-reporting-what-it-could-have-seen",
        "the 3-sigma sensitivity becomes zero, so 'no order effect "
        "found' no longer carries the size of the effect that would "
        "have been invisible. MEASURED on ibm_marrakesh: r1=+0.022 "
        "against a sensitivity of 0.047, so the honest sentence is "
        "'nothing above 0.05 was found' and the mutant's sentence is "
        "'the device is stable' -- a claim 4,000 shots cannot support",
        "sensitivity=3.0 * sd,",
        "sensitivity=0.0,",
        "test_every_result_REPORTS_the_smallest_effect_it_could_have_SEEN",
        target="exchangeability",
    ),
    Mutant(
        # A CONSTANT SAMPLE CALLED UNCORRELATED.
        "M427-an-undefined-correlation-is-returned-as-zero",
        "a sample whose observations are all identical returns 0.0 "
        "instead of refusing. Zero is a real answer meaning 'no order "
        "dependence', and it is indistinguishable from 'the "
        "denominator was zero and nothing was measured' -- the dark "
        "gate shape, on the statistic every other reading here feeds "
        "from",
        '        raise ExchangeabilityRefusal(\n            "every observation is identical',
        '        return 0.0\n        raise ExchangeabilityRefusal(\n            "every observation is identical',
        "test_a_CONSTANT_sample_is_REFUSED_not_reported_as_uncorrelated",
        target="exchangeability",
    ),
    Mutant(
        # THE PERMUTATION, REMOVED.
        "M428-the-null-is-built-without-shuffling",
        "the shuffle is dropped, so every 'permutation' recomputes the "
        "statistic on the ORIGINAL order. The null collapses to a "
        "point at the observed value, its sd is zero, and the test can "
        "never reject anything -- while still printing a p-value and a "
        "null mean that look entirely ordinary",
        "        rng.shuffle(work)",
        "        pass",
        "test_it_FINDS_a_drift_that_is_really_there",
        target="exchangeability",
    ),
    Mutant(
        # ONE-SIDED WHERE TWO-SIDED WAS ASKED FOR.
        "M429-the-two-sided-test-only-counts-the-upper-tail",
        "a two-sided request is scored as one-sided, so a device whose "
        "shots ANTI-correlate -- a decoder alternating, a duty cycle, a "
        "readout that alternates with its neighbour -- passes with a "
        "p-value near 1.0 while its order dependence is as strong as "
        "any drift",
        "        hits = sum(1 for x in null if abs(x - mean) >= abs(observed - mean))",
        "        hits = sum(1 for x in null if x >= observed)",
        "test_it_FINDS_ANTI_correlation_that_a_ONE_SIDED_test_would_miss",
        target="exchangeability",
    ),
    Mutant(
        # A SAMPLE TOO SHORT TO SAY ANYTHING, SCORED ANYWAY.
        "M430-the-minimum-sample-floor-is-removed",
        "a permutation null over three points is reported as a test. "
        "With so few distinct orderings the statistic can take only a "
        "handful of values, so the p-value is quantised into near "
        "uselessness -- and an impossible measurement is dressed as a "
        "passed one",
        "    if len(v) < MIN_OBSERVATIONS:",
        "    if False:",
        "test_a_TOO_SHORT_sample_is_REFUSED",
        target="exchangeability",
    ),
    Mutant(
        # FOUR VIEWS OF ONE DATASET, SOLD AS FOUR EXPERIMENTS.
        "M431-the-block-scan-offers-a-combined-p-value",
        "the refusal to combine correlated block sizes is replaced by a "
        "Fisher-style product, as though the four were independent "
        "experiments. On ibm_marrakesh all four leaned over-dispersed "
        "and none cleared its own bar; multiplied together they would "
        "read as a finding. This is how a non-result becomes a "
        "publication",
        '            "no_combined_p_value": (',
        '            "combined_p_value": min(1.0, 4 * min(\n                [r["p_value"] for r in out if "p_value" in r] or [1.0])),\n            "no_combined_p_value": (',
        "test_the_block_scan_REFUSES_to_combine_correlated_tests",
        target="exchangeability",
    ),
    Mutant(
        # THE SEED, MADE DECORATIVE.
        "M432-the-permutation-seed-stops-reaching-the-generator",
        "the seeded generator is replaced by an unseeded one, so the "
        "published p-value moves between runs of identical data. Every "
        "hash covering this artifact breaks, and the `--from-artifact` "
        "verifier -- which recomputes 18 published claims from stored "
        "shots -- starts reporting disagreements that are pure RNG",
        "    rng = random.Random(seed)",
        "    rng = random.Random()",
        "test_the_same_seed_gives_the_SAME_p_value",
        target="exchangeability",
    ),
    Mutant(
        # A VARIANCE OVER ONE BLOCK.
        "M433-block-dispersion-accepts-a-single-block",
        "a block size that yields one block is scored instead of "
        "refused. The variance of a single mean is zero by "
        "construction, so any sample too short to test reports PERFECT "
        "stability -- the most confident possible answer from the least "
        "possible evidence",
        "    if k < 2:",
        "    if k < 1:",
        "test_block_dispersion_needs_at_least_two_blocks",
        target="exchangeability",
    ),

    # ============================================================
    # LOCATING CERTIFIED WEIGHT (2026-09-03). This module's main
    # product is a REFUSAL: relabelling detectors i -> N-1-i, a pure
    # renaming, left the certified total at 2952.828 exactly while
    # moving 4.61% of it between rounds and reordering the top five.
    # Every mutant below restores a version of the overclaim that
    # first shipped -- a map that names a worst qubit the duals
    # cannot identify. Each one produces a confident, plausible,
    # wrong hardware finding.
    # ============================================================
    Mutant(
        # THE OVERCLAIM, RESTORED.
        "M434-dominance-compares-floors-to-floors-instead-of-ceilings",
        "a detector is called dominant when its floor merely beats the "
        "others' floors, ignoring that their ceilings tower over it. On "
        "ibm_marrakesh this names detector 26 the worst on the device "
        "when a pure renaming reorders the leaderboard -- the exact "
        "false finding this module was rewritten to refuse",
        "        if all(s.lower > o.upper for o in shares "
        "if o.detector != s.detector))",
        "        if all(s.lower >= o.lower for o in shares "
        "if o.detector != s.detector))",
        "test_OVERLAPPING_intervals_identify_NO_worst_detector",
        target="fragility_map",
    ),
    Mutant(
        # ONE LABELLING, WHICH WAS THE ARTIFACT.
        "M435-an-envelope-over-a-single-labelling-is-accepted",
        "the envelope stops requiring a second labelling, so it returns "
        "one solver's tie-breaking dressed in the language of "
        "invariance. One labelling is precisely the reading that was "
        "MEASURED to be an artifact",
        "    if len(maps) < 2:",
        "    if len(maps) < 1:",
        "test_ONE_labelling_is_REFUSED_because_one_labelling_was_the_artifact",
        target="fragility_map",
    ),
    Mutant(
        # A SHARED CUT SET ALLOWED TO RAISE A FLOOR.
        "M436-a-multi-detector-cut-raises-every-members-floor",
        "weight proven ONCE across a cut is credited in full to each "
        "detector the cut touches, so a 3-detector set adds its "
        "multiplier three times. Floors then exceed the certified total "
        "they are meant to partition, and the hottest detector is "
        "whichever appears in the most crowded cuts",
        "            out[det] = (lo + z if solo else lo, hi + z,",
        "            out[det] = (lo + z, hi + z,",
        "test_a_MULTI_detector_cut_raises_every_ceiling_and_no_floor",
        target="fragility_map",
    ),
    Mutant(
        # THE BOUNDARY, BACK IN THE MAP.
        "M437-the-virtual-boundary-is-placed-on-the-chip",
        "the decoding graph's boundary node is attributed like a real "
        "detector, which is what produced a `round -1` carrying 3.45 "
        "units on the first run. It is part of the proven bound and "
        "sits at no qubit; giving it a location puts weight on hardware "
        "that does not exist",
        "            if det == boundary:",
        "            if False:",
        "test_the_BOUNDARY_is_kept_out_of_the_physical_map",
        target="fragility_map",
    ),
    Mutant(
        # A RANGE GUARD WATCHING ONE END.
        "M438-the-detector-range-guard-stops-checking-the-low-end",
        "the collapse guard checks only the upper bound again, so any "
        "negative index -- the boundary, or a map built for another "
        "circuit -- lands in a bucket instead of being refused. This is "
        "the exact half-guard that let `round -1` through",
        "    if lowest < 0 or seen >= expected:",
        "    if seen >= expected:",
        "test_a_detector_OUTSIDE_the_circuit_is_REFUSED_at_BOTH_ends",
        target="fragility_map",
    ),
    Mutant(
        # DIFFERENT PROBLEMS, COMPARED ANYWAY.
        "M439-labellings-that-disagree-on-the-total-are-combined",
        "the invariance check is dropped, so maps of DIFFERENT problems "
        "are merged into one envelope. Relabelling cannot change the "
        "certified total, so a disagreement there means something other "
        "than the labels moved -- and every per-detector comparison "
        "afterwards is between two different experiments",
        "    if spread > total_tol * max(1.0, abs(totals[0])):",
        "    if False:",
        "test_labellings_that_DISAGREE_on_the_total_are_REFUSED",
        target="fragility_map",
    ),
    Mutant(
        # AN ABSENT DETECTOR GIVEN A FLOOR IT NEVER EARNED.
        "M440-a-detector-absent-from-one-labelling-keeps-its-floor",
        "a detector that some labelling never named keeps the floor "
        "another labelling gave it, so the envelope reports as forced "
        "exactly the attribution the labellings disagreed about most. "
        "Contributing nothing under a labelling IS a floor of zero",
        "            ranges[det] = (0.0, hi)",
        "            pass",
        "test_a_detector_ABSENT_from_one_labelling_gets_a_floor_of_ZERO",
        target="fragility_map",
    ),

    # ============================================================
    # DEVICE LAWS (2026-09-03). Ported from arc_prize_2026's
    # `mechanism_lab.AlgebraLaw` -- induce a rule from play, kill it
    # on one counterexample -- with ONE change: an observation that
    # could not have refuted a law is counted separately from one
    # that tested it and agreed. Every mutant here collapses that
    # back to a two-outcome tally, which is how a law verifies on
    # silence. MEASURED: 3,424 of 4,000 hardware shots have an empty
    # syndrome, and detector 26's honest confidence of 0.667 reads
    # 0.995 once they count as support.
    # ============================================================
    Mutant(
        # SILENCE, PROMOTED TO EVIDENCE.
        "M441-observations-that-could-not-refute-are-counted-as-support",
        "NOT_TESTED is tallied as support, restoring the two-outcome "
        "count this module exists to replace. 3,424 empty-syndrome shots "
        "then vouch for every law about which detectors co-fire, and a "
        "detector that breaks its law a third of the time it is examined "
        "publishes at 0.995 confidence",
        "        else:\n            untested += 1",
        "        else:\n            support += 1",
        "test_a_law_NOTHING_could_refute_is_VACUOUS_not_verified",
        target="device_laws",
    ),
    Mutant(
        # A LAW NOTHING TESTED, CALLED TRUE.
        "M442-a-vacuous-law-stops-being-told-from-a-thinly-tested-one",
        "a law NOTHING put at risk stops reporting VACUOUS and reports "
        "UNDERPOWERED, collapsing 'the data cannot speak to this' into "
        "'the data spoke and said little'. The first is a fact about a "
        "sample that never asked the question and calls for a different "
        "experiment; the second only asks for more shots. NOTE: the "
        "obvious mutation -- deleting the vacuity clause from "
        "`verified` -- is EQUIVALENT, because a vacuous law has zero "
        "support and already fails MIN_SUPPORT. That clause was dead "
        "code and was removed; `status` is where vacuity decides",
        "        if self.vacuous:\n            return \"VACUOUS\"",
        "        if False:\n            return \"VACUOUS\"",
        "test_a_law_NOTHING_could_refute_is_VACUOUS_not_verified",
        target="device_laws",
    ),
    Mutant(
        # CONFIDENCE DILUTED BY SILENCE.
        "M443-confidence-is-taken-over-every-observation-not-the-risked-ones",
        "confidence divides by all observations rather than the ones "
        "that tested the law, so it becomes a function of sample size. "
        "MEASURED: detector 26's 0.667 reads 0.995 under this, and the "
        "arithmetic is not wrong about any single shot -- the "
        "denominator simply never asked the question",
        "        return self.support / self.opportunities if self.opportunities else 0.0",
        "        n = self.support + self.refutations + self.not_tested\n"
        "        return (self.support + self.not_tested) / n if n else 0.0",
        "test_confidence_is_over_RISKED_observations_only",
        target="device_laws",
    ),
    Mutant(
        # A RULE FROM ONE EVENT.
        "M444-a-single-observation-is-enough-to-establish-a-law",
        "the minimum support drops to one, so a rule induced from a "
        "single event is published as a device law. That is a "
        "description of the event, and the operator this was ported "
        "from uses 2 for exactly this reason",
        "MIN_SUPPORT = 2",
        "MIN_SUPPORT = 1",
        "test_silence_CANNOT_carry_a_law_to_verified",
        target="device_laws",
    ),
    Mutant(
        # THE WITNESS, OVERWRITTEN.
        "M445-the-last-counterexample-replaces-the-first",
        "each refutation overwrites the stored witness, so a law killed "
        "by shot 7 reports shot 900 instead. A kill is a claim, and the "
        "SMALLEST witness is the one a reader can check by hand",
        "            if witness is None:",
        "            if True:",
        "test_the_FIRST_witness_is_kept_because_the_smallest_is_the_useful_one",
        target="device_laws",
    ),
    Mutant(
        # A SIGHTING, DRESSED AS A PROOF.
        "M446-every-refutation-is-marked-certified",
        "a counterexample that carries no proof is labelled certified, "
        "erasing the one thing this module has that a game agent does "
        "not: the difference between an observation that might be noise "
        "and a witness an independent checker re-verifies",
        "                certified = bool(w and w.get(\"proof\"))",
        "                certified = True",
        "test_a_kill_that_carries_a_PROOF_is_marked_as_certified",
        target="device_laws",
    ),
    Mutant(
        # AN UNTESTED EDGE, CALLED AVOIDABLE.
        "M447-an-edge-the-correction-never-used-counts-as-support",
        "a shot whose optimal correction never touched the edge is "
        "scored as evidence the edge is unnecessary. It is proof for "
        "THAT shot and no test of the law, and with 68 edges over 576 "
        "shots such shots dominate -- so every edge verifies on "
        "arithmetic about edges that were never on the table",
        "        if edge not in set(used):\n            # Proven avoidable here, but the law was never at risk.\n            return Verdict.NOT_TESTED",
        "        if edge not in set(used):\n            return Verdict.UPHELD",
        "test_an_edge_the_correction_NEVER_USED_is_not_counted_as_support",
        target="device_laws",
    ),
    Mutant(
        # TWO CLAIMS, ONE NAME.
        "M448-two-different-laws-may-share-a-name",
        "the duplicate-name guard is removed, so a REFUTED law and a "
        "VERIFIED one can be cited under one name and a reader cannot "
        "tell which was meant",
        "    if len(set(names)) != len(names):",
        "    if False:",
        "test_two_laws_under_ONE_NAME_are_REFUSED",
        target="device_laws",
    ),
    Mutant(
        # SURVIVING A COMPARISON THAT NEVER HAPPENED.
        "M449-relabeling-comparison-accepts-two-disjoint-law-sets",
        "the shared-name guard goes, so comparing two labellings with no "
        "law in common reports that every law survived relabelling. "
        "Vacuously true, and it certifies exactly the solver-dependence "
        "the comparison exists to catch",
        "    if not shared:",
        "    if False:",
        "test_comparing_labellings_with_NO_shared_laws_is_REFUSED",
        target="device_laws",
    ),

    # ============================================================
    # THE EXACT DUAL ENVELOPE (2026-09-04). The optimal dual is a
    # polytope and the engine returned one vertex; this module ranges
    # a functional over the WHOLE face and certifies each extreme.
    # Every mutant below keeps a number flowing while removing the
    # proof underneath it -- which, for a module whose only product is
    # a proof, is the same as removing the module.
    # ============================================================
    Mutant(
        # A POINT OFF THE FACE, REPORTED AS A RANGE END.
        "M450-a-perturbed-optimum-off-the-face-is-accepted",
        "the on-face check is dropped, so the first perturbed solve is "
        "reported whether or not 1'z == W*. The maximality argument "
        "holds ONLY on the face; off it the value is a point of some "
        "other problem, and it is published as the true extreme",
        "            if total == wstar:",
        "            if True:",
        "test_never_landing_on_the_face_REFUSES_after_bounded_halvings",
        target="dual_envelope",
    ),
    Mutant(
        # THE SOLVER'S WORD, TAKEN.
        "M451-the-returned-point-is-not-re-verified-feasible",
        "feasibility re-verification is skipped, so an infeasible point "
        "-- one that violates a packing constraint -- becomes a range "
        "end. exact_lp's own docstring says its word is never the "
        "acceptance evidence; this mutant makes it exactly that",
        "            if not _verify_feasible(rows, rhs, z):\n"
        "                raise EnvelopeRefusal(\"solver returned an infeasible point\")",
        "            if False:\n"
        "                raise EnvelopeRefusal(\"solver returned an infeasible point\")",
        "test_an_INFEASIBLE_point_from_the_solver_is_REFUSED",
        target="dual_envelope",
    ),
    Mutant(
        # MAXIMALITY ASSERTED, NOT PROVEN.
        "M452-the-dual-certificate-of-the-perturbed-optimum-is-not-checked",
        "the weak-duality certificate is skipped, so 'max over the face' "
        "rests on the simplex having stopped rather than on u >= 0, "
        "A^T u >= c, w'u == c'z. A solver that stalls one pivot early "
        "would publish a range that is too narrow -- and narrow is the "
        "flattering direction",
        "            if not _verify_optimal(rows, rhs, c, z, u):",
        "            if False:",
        "test_a_FORGED_dual_certificate_is_REFUSED",
        target="dual_envelope",
    ),
    Mutant(
        # AN UNDEFINED FACE, RANGED ANYWAY.
        "M453-a-pool-exceeding-the-claimed-optimum-is-not-refused",
        "the total > W* guard is removed, so a caller passing the wrong "
        "W* gets halvings until 'never landed' instead of the true "
        "diagnosis. The face {1'z = W*} does not exist when W* is not "
        "the optimum, and a refusal that names the wrong cause sends "
        "the reader to the wrong fix",
        "            if total > wstar:",
        "            if False:",
        "test_a_pool_that_EXCEEDS_the_claimed_optimum_is_REFUSED",
        target="dual_envelope",
    ),
    Mutant(
        # DOMINANCE BY A TIE.
        "M454-exact-dominance-accepts-a-floor-that-merely-EQUALS-a-ceiling",
        "the dominance test becomes >=, so a detector whose floor equals "
        "a rival's ceiling is named worst. Equality means some feasible "
        "attribution has them level, which is the opposite of 'no "
        "re-attribution can dislodge it'",
        "        if all(floors[d] > ceils[o] for o in detectors if o != d)))",
        "        if all(floors[d] >= ceils[o] for o in detectors if o != d)))",
        "test_aggregate_SUMS_ranges_EXACTLY_because_faces_are_independent",
        target="dual_envelope",
    ),
    Mutant(
        # AN EVEN BALL, ADMITTED AS A DUAL VARIABLE.
        "M455-the-T-odd-filter-on-candidate-cuts-is-removed",
        "even cuts enter the pool. An even cut is not a dual variable of "
        "the T-join LP, so the 'optimum' over the pool stops being a "
        "lower bound on anything; measured on the tied pair it lifts W* "
        "above the true 2.0. The conformance test against the engine's "
        "bound exists for exactly this",
        "            if len(S & T) % 2 == 1:\n                seen.add(S)",
        "            if True:\n                seen.add(S)",
        "test_a_TIED_pair_gives_the_FULL_interval_at_each_endpoint",
        target="dual_envelope",
    ),
    Mutant(
        # THE BOUNDARY, LEFT OUT OF T.
        "M456-an-odd-syndrome-no-longer-adds-the-boundary-to-T",
        "for an odd number of fired detectors the boundary stops joining "
        "T, so the ball {boundary} vanishes from the pool and the fired "
        "detector's attribution collapses from [0, 5] to the point [5, "
        "5]. Narrower, confident, and wrong. NOTE the scope: the boundary "
        "is ONE source of non-uniqueness, not the source -- MEASURED on "
        "148 even-|T| marrakesh shots with no boundary in T, 0 had a "
        "unique dual; nested balls around any centre trade freely",
        "    if len(T) % 2 == 1:\n        T = T | {BOUNDARY}\n    adj = build_adj(edges)",
        "    adj = build_adj(edges)",
        "test_an_ODD_syndrome_adds_the_boundary_and_the_range_shows_it",
        target="dual_envelope",
    ),
    Mutant(
        # A CUT COUNTED ONCE PER GROUP INSTEAD OF ONCE PER MEMBER.
        "M457-a-group-functional-counts-a-cut-once-instead-of-per-member",
        "phi_G(S) becomes [S & G non-empty] instead of |S & G|, so a cut "
        "holding two detectors of one round contributes half. The group "
        "range then no longer equals the sum of member attributions at "
        "any vertex, and the engine's own value falls OUTSIDE it -- the "
        "'exact per-round range' would be ranging a different quantity "
        "from the per-detector table it is compared against",
        "        phi = [len(S & G) for S in pool]",
        "        phi = [1 if (S & G) else 0 for S in pool]",
        "test_at_ONE_vertex_a_group_equals_the_SUM_of_its_members",
        target="dual_envelope",
    ),
    # ============================================================
    # THE ANNIHILATOR OF THE FACE (2026-09-04). R5 said the attribution
    # is a range; this module says EXACTLY which functionals are not --
    # the implied equalities of the optimal face, their row space, and
    # the detector combinations inside it. Each mutant below makes the
    # dual look more decided than it is, or drops the check that would
    # notice; every killer names a face built by hand whose answer is
    # derivable on paper.
    Mutant(
        # TWO READERS THAT DISAGREE, AND NEITHER IS TRUSTED -- UNLESS THIS.
        "M458-the-cross-check-against-the-envelope-is-dropped",
        "the per-detector agreement between the algebraic verdict "
        "(phi in the row space of the implied equalities) and the ranged "
        "verdict (the envelope's is_point) is no longer enforced, so a "
        "wrong implied-equality set publishes 'determined' detectors the "
        "ranges contradict",
        "        if bad:",
        "        if False:",
        "test_the_cross_check_refuses_a_disagreeing_envelope",
        target="face_annihilator",
    ),
    Mutant(
        # THE FREE EQUALITY IS NO LONGER FREE.
        "M459-complementary-slackness-no-longer-marks-a-tight-row",
        "a row with u_e > 0 is no longer taken as tight on the whole face; "
        "it falls through to the exact range test and is still found, so "
        "the answer is the same and only the accounting moves -- which is "
        "exactly what the killer observes: free_by_slackness and "
        "range_tests on the segment",
        "        if uu[i] > 0:                    # complementary slackness: tight on F",
        "        if False:                        # complementary slackness: tight on F",
        "test_the_segment_equalities_come_from_slackness_not_from_ranges",
        target="face_annihilator",
    ),
    Mutant(
        # A ROW TIGHT SOMEWHERE, CALLED TIGHT EVERYWHERE.
        "M460-the-tight-row-test-reads-the-maximum-instead-of-the-minimum",
        "a row is an implied equality iff its MINIMUM over the face equals "
        "the right-hand side; reading the maximum calls every row that is "
        "tight at the simplex's vertex an equality, adds it to M, and "
        "collapses a segment to a point -- the flattering direction",
        "            if optimal_face_range(rows, rhs, rows[i], wstar).lo == rhs[i]:",
        "            if optimal_face_range(rows, rhs, rows[i], wstar).hi == rhs[i]:",
        "test_a_cap_tight_at_the_vertex_but_not_on_the_face_is_not_an_equality",
        target="face_annihilator",
    ),
    Mutant(
        # A VARIABLE ZERO SOMEWHERE, CALLED ZERO EVERYWHERE.
        "M461-the-zero-variable-test-reads-the-minimum-instead-of-the-maximum",
        "a variable is zero on the face iff its MAXIMUM there is 0; reading "
        "the minimum (always 0 at the vertex where it is zero) declares it "
        "an equality, and the endpoint segment becomes a point with the "
        "later endpoint 'determined' at zero",
        "            if optimal_face_range(rows, rhs, e_j, wstar).hi == 0:",
        "            if optimal_face_range(rows, rhs, e_j, wstar).lo == 0:",
        "test_a_segment_determines_the_sum_and_not_the_split",
        target="face_annihilator",
    ),
    Mutant(
        # MEMBERSHIP INVERTED.
        "M462-row-space-membership-is-inverted",
        "'determined' and 'undetermined' swap: every floating detector is "
        "published as fixed and every fixed one as floating",
        "    return rref(list(rows) + [list(phi)])[0] == rank",
        "    return rref(list(rows) + [list(phi)])[0] != rank",
        "test_in_rowspace_both_ways",
        target="face_annihilator",
    ),
    Mutant(
        # THE FREE LOCATION DIMENSIONS, ZEROED.
        "M463-the-undetermined-location-dimension-is-reported-as-zero",
        "loc_free is fixed at 0, so k_loc reads as the full detector count "
        "on every shot: the dual appears to determine everything about "
        "location, which is the precise claim R5 refuted",
        "        loc_free = rref(phiN)[0]",
        "        loc_free = 0",
        "test_a_segment_determines_the_sum_and_not_the_split",
        target="face_annihilator",
    ),
    Mutant(
        # THE INVISIBLE DIRECTIONS, ZEROED.
        "M464-invisible-face-directions-are-not-counted",
        "face directions that move weight between cuts without changing "
        "any detector's attribution are reported as 0, so dim F and "
        "loc_free are published as equal on every shot",
        "invisible_dirs=dim_face - loc_free,",
        "invisible_dirs=0,",
        "test_face_directions_no_detector_sees_are_counted_as_invisible",
        target="face_annihilator",
    ),
    Mutant(
        # A SINGLE IS NOT A COMBINATION.
        "M465-single-detectors-are-published-as-combination-facts",
        "the support threshold drops to 1, so every individually fixed "
        "detector is also listed as a 'combination' and the count of "
        "facts no member determines is inflated",
        "            if len(sup) >= 2:",
        "            if len(sup) >= 1:",
        # NOT the point-face test: a point face short-circuits before the
        # threshold line and cannot observe it (measured by the pre-flight:
        # SURVIVED). A never-charged detector on a segment reaches it.
        "test_a_never_charged_detector_is_a_single_not_a_combination",
        target="face_annihilator",
    ),
    # ============================================================
    # THE BURST MODEL (2026-09-05, R21). One constant, no fit, 710 -> 147
    # logical errors on hardware and far worse on synthetic shots. Each
    # mutant below turns it back into something it is not.
    Mutant(
        "M472-the-burst-price-becomes-a-scale-again",
        "the time-like weight is MULTIPLIED by the cost instead of replaced, "
        "so the per-round probability still sets the price -- which is the "
        "very assumption a burst violates, and the model quietly becomes the "
        "surrogate it was built to replace",
        "    out = [(u, v, float(cost) if edge_family(u, v, positions) == \"time-like\" else w)",
        "    out = [(u, v, float(cost) * w if edge_family(u, v, positions) == \"time-like\" else w)",
        "test_the_burst_cost_is_a_price_not_a_scale",
        target="dem_families",
    ),
    Mutant(
        "M473-burst-edges-are-priced-by-their-span",
        "the cost grows with the span, so a long chain is dear again and the "
        "one thing the model asserts -- a burst costs about one event whatever "
        "its length -- is gone while the edges remain",
        "    return [(t * positions + pos, t2 * positions + pos, float(cost))",
        "    return [(t * positions + pos, t2 * positions + pos, float(cost) * (t2 - t))",
        "test_burst_edges_span_more_than_one_round_at_each_position",
        target="dem_families",
    ),
    Mutant(
        "M474-the-burst-model-touches-the-data-families",
        "space-like and boundary weights are re-priced too, so the edges that "
        "carry the logical observable are flattened along with the harmless "
        "route -- the asymmetry the whole mechanism rests on disappears",
        "    out = [(u, v, float(cost) if edge_family(u, v, positions) == \"time-like\" else w)",
        "    out = [(u, v, float(cost))",
        "test_the_burst_model_REPLACES_time_like_weights_and_leaves_data_alone",
        target="dem_families",
    ),
    # ============================================================
    # THE EDGE-FAMILY OPERATOR (2026-09-05). A family mis-assigned is a
    # scale applied to the wrong edges: the decoder gets a different
    # model from the one the artifact declares.
    Mutant(
        # A TIME-LIKE EDGE, MISFILED.
        "M466-time-like-edges-are-recognised-one-round-off",
        "the time-like test compares the index gap to positions + 1, so no "
        "edge is time-like and every measurement/idle edge falls to 'other' "
        "and is never scaled",
        "    if abs(u - v) == positions:",
        "    if abs(u - v) == positions + 1:",
        "test_edge_families_by_index_arithmetic",
        target="dem_families",
    ),
    Mutant(
        # AN UNSCALED FAMILY, ZEROED.
        "M467-an-unnamed-family-is-scaled-by-zero-instead-of-one",
        "a family absent from the scales gets factor 0, so a partial scale "
        "set erases whole families of edges instead of leaving them alone",
        "        s = scales.get(edge_family(u, v, positions), 1)",
        "        s = scales.get(edge_family(u, v, positions), 0)",
        "test_scales_touch_only_their_family_and_keep_exact_weights_exact",
        target="dem_families",
    ),
    Mutant(
        # A MISSING FAMILY, SILENTLY 1.0.
        "M468-a-scale-set-missing-a-family-is-accepted",
        "the completeness check is dropped, so 'time-like=0.5' parses as a "
        "complete set and the other two families are silently unscaled -- "
        "a set that reads as complete and is not",
        "    if missing:",
        "    if False:",
        "test_parse_scales_requires_every_family_and_only_known_ones",
        target="dem_families",
    ),
    # ============================================================
    # THE SCALE'S MECHANISM (2026-09-05). Its two verdicts are the reason
    # R19's claim is about the DEVICE. A verdict that cannot say REFUTED
    # is a decoration.
    Mutant(
        "M469-the-synthetic-control-cannot-refute",
        "the control returns UPHELD whenever the optimum is not tiny, so a "
        "small scale that WINS on a model that is right by construction -- "
        "the outcome that would say the gain is our decoder's, not the "
        "device's -- is published as confirmation",
        "    if curve[small] <= at_one:",
        "    if False:",
        "test_the_control_REFUTES_a_small_scale_that_merely_TIES",
        target="dem_families",
    ),
    Mutant(
        "M470-the-degeneracy-kill-accepts-half-its-witness",
        "the kill fires on the flip count alone, so an arm that stops "
        "predicting flips for some OTHER reason is recorded as the same "
        "degeneracy -- and the claim that the cut operator specifically "
        "destroys the decoder loses its second witness",
        "    if flip_predictions == 0 and errors == true_flips:",
        "    if flip_predictions == 0:",
        "test_the_degeneracy_kill_needs_BOTH_of_its_witnesses",
        target="dem_families",
    ),
    Mutant(
        "M471-the-run-statistic-counts-edges-not-chains",
        "the run length resets are dropped, so every time-like edge at a "
        "position counts toward one run and an isolated pair reads as a "
        "burst -- the signature that distinguishes a multi-round mechanism "
        "from ordinary measurement noise goes dark",
        "            cur = cur + 1 if b == a + 1 else 1",
        "            cur = cur + 1",
        "test_longest_timelike_run_counts_consecutive_rounds_at_one_position",
        target="dem_families",
    ),
    # ============================================================
    # THE MODEL MEASURED FROM DETECTORS (2026-09-06, R23). This estimator
    # is the only thing in the campaign that reads a device's error model
    # WITHOUT its calibration -- and it is four symbols of algebra. Each
    # mutant below leaves a working pipeline that publishes a different
    # model, which is exactly the failure that would be believed.
    Mutant(
        "M475-the-radicand-loses-half-its-covariance",
        "the two-point formula carries 2C where the derivation gives 4C, so "
        "every probability comes back systematically wrong while the graph "
        "still builds and the decoder still runs -- a calibration-free model "
        "that is quietly miscalibrated, which is worse than none",
        "    v = 1 - 4 * c / den",
        "    v = 1 - 2 * c / den",
        "test_estimator_recovers_a_model_it_was_not_told",
        target="dem_measured",
    ),
    Mutant(
        "M476-the-covariance-becomes-a-second-moment",
        "C = <x_i x_j> + <x_i><x_j> instead of minus, so the estimator no "
        "longer measures CORRELATION at all -- it reports a large p for "
        "every pair of detectors that merely fire often, and an "
        "anti-correlated pair stops being rejected",
        "    c = x_ij - x_i * x_j",
        "    c = x_ij + x_i * x_j",
        "test_an_anticorrelated_pair_estimates_below_the_floor_and_is_not_added",
        target="dem_measured",
    ),
    Mutant(
        # THE 2026-09-05 REGRESSION, RE-ARMED AS A MUTANT.
        "M477-a-sub-floor-code-edge-is-dropped-again",
        "a code edge the data prices below the floor is deleted rather than "
        "priced dear. That removes a logical cut and the observable node with "
        "it, and the decoder refuses the shots array outright -- the exact "
        "failure this module was written around",
        "            if (i, j) not in code_pairs:",
        "            if True:",
        "test_a_code_edge_the_data_prices_at_nothing_is_floored_not_dropped",
        target="dem_measured",
    ),
    Mutant(
        "M478-a-negative-estimate-reaches-the-logarithm",
        "only None is rejected, so the small NEGATIVE p that sampling noise "
        "produces on an uncorrelated pair reaches log((1-p)/p), which has no "
        "value there. The floor branch is what makes the estimator safe on "
        "real data and this removes it",
        # ANCHORED WITH THE COMMENT THAT FOLLOWS IT. The identical guard
        # exists inside `graph_likeness`, and the registry's own
        # exactly-once gate caught the ambiguity the moment that function
        # landed -- an anchor matching twice mutates the first site
        # silently and proves nothing about the second.
        "        if p is None or p <= pfloor:\n"
        "            # A code edge is PRICED at the floor",
        "        if p is None:\n"
        "            # A code edge is PRICED at the floor",
        "test_a_code_edge_the_data_prices_at_nothing_is_floored_not_dropped",
        target="dem_measured",
    ),
    Mutant(
        "M479-the-boundary-edge-becomes-the-raw-marginal",
        "each boundary edge is priced at the detector's own firing rate "
        "instead of the residue the marginal constraint leaves, so every "
        "pair mechanism is counted twice -- once as its edge and again as a "
        "boundary edge at both ends",
        # Anchored with the line that follows, for the same reason as M478:
        # `graph_likeness` solves the identical expression.
        "            v = (1 - (1 - 2 * float(means[i])) / prod[i]) / 2\n"
        "            if v > pfloor:",
        "            v = float(means[i])\n"
        "            if v > pfloor:",
        "test_the_marginal_constraint_recovers_the_planted_boundary",
        target="dem_measured",
    ),
    Mutant(
        "M480-the-multi-round-candidates-stop-one-span-short",
        "the span range excludes max_span, so the longest-range correlation "
        "the model is willing to represent is silently one round shorter "
        "than the constant declares -- the artifact says span 6 and the "
        "graph carries 5",
        "            for s in range(2, max_span + 1):",
        "            for s in range(2, max_span):",
        "test_candidate_pairs_span_exactly_two_through_max_span",
        target="dem_measured",
    ),
    # THE SELF-REFUTATION'S OWN GUARDS. R23's decoding win is real and its
    # generative reading is false, and the only thing standing between the
    # two is `graph_likeness`. A silent version of it would let the next
    # person read the 8.47x as a measurement of the hardware.
    # ============================================================
    # THE SITE-CORRELATION INSTRUMENT AND ITS SELECTOR (2026-09-07).
    # Every one of these leaves a working selector that recommends the
    # WRONG STORAGE, which is the failure nobody would notice.
    Mutant(
        "M483-the-difference-code-stops-cancelling-shared-noise",
        "Gamma_- adds the covariance instead of subtracting it, so the code "
        "that PROTECTS against shared noise is scored as though shared noise "
        "hurt it. The selector then prefers the individually quiet independent "
        "pair over the correlated one -- precisely the mistake the physics "
        "exists to prevent",
        "        return 2.0 * (cii + cjj - 2.0 * cij)",
        "        return 2.0 * (cii + cjj + 2.0 * cij)",
        "test_logical_rates_are_the_published_algebra",
        target="noise_correlation",
    ),
    Mutant(
        "M484-the-crossover-claims-encoding-pays-when-it-never-does",
        "the no-solution branch returns the low end of the sweep instead of "
        "None, so a regime where four operations spend the entire error budget "
        "before storage even begins reports a threshold of -0.999 and every "
        "correlation on earth looks good enough",
        "    if gain(hi) <= 0:\n        return None",
        "    if gain(hi) <= 0:\n        return lo",
        "test_the_crossover_is_where_encoding_STARTS_to_pay",
        target="noise_correlation",
    ),
    Mutant(
        "M485-the-selector-can-no-longer-decline-to-encode",
        "the single-site option stops being the starting incumbent, so the "
        "selector must return a PAIR however bad it is -- and on the measured "
        "hardware correlations, which sit ~40x below the crossover, that is "
        "always the wrong answer",
        "    best = best_single",
        "    best = (0.0, (\"difference\", 0, 1))",
        "test_the_selector_DECLINES_when_the_correlation_is_not_there",
        target="noise_correlation",
    ),
    Mutant(
        "M486-the-correlation-becomes-a-coincidence-rate",
        "the product of means is not subtracted, so two sites that merely fire "
        "OFTEN read as strongly correlated. Every pair on a noisy device would "
        "then clear the crossover and the selector would encode on noise that "
        "is not shared at all",
        # *** THIS MUTANT SURVIVED ITS FIRST KILLER AND THAT WAS THE POINT. ***
        # It was pointed at `test_site_correlation_diagonal_is_exactly_one`,
        # which cannot see it: rho is cov / sqrt(diag * diag), so removing the
        # mean subtraction changes numerator and denominator together and the
        # diagonal stays exactly 1.0. It also slipped the rep-code fixture,
        # where the off-diagonal only reaches sqrt(p_i p_j) ~ 0.02 at p = 0.02,
        # under a 0.05 bound. The killer is now a case built to observe it:
        # independent columns at p = 0.4, where the coincidence reading is
        # ~0.4 against an honest covariance of ~0.
        "        cov += (xt.T @ xt) / shots - np.outer(m, m)",
        "        cov += (xt.T @ xt) / shots",
        "test_a_COVARIANCE_is_not_a_COINCIDENCE_RATE",
        target="noise_correlation",
    ),
    Mutant(
        "M487-the-extra-gate-error-is-charged-subtractively-again",
        "the encode/decode cost comes off the error budget instead of "
        "multiplying the fidelity. That is the convention an independent "
        "implementation does NOT use; it shifts every horizon by 4e-3, and "
        "adopting it silently breaks agreement with the published curve this "
        "module was validated against",
        "    fac = (1.0 - err_per_gate) ** extra_gates",
        "    fac = 1.0 - err_per_gate * extra_gates",
        "test_the_extra_gate_error_is_charged_MULTIPLICATIVELY",
        target="noise_correlation",
    ),
    Mutant(
        "M481-the-graph-likeness-check-goes-silent",
        "an unsatisfiable marginal stops being counted, so the check reports "
        "0 on every run including the four where it should report 108 of 112 "
        "-- and the result reads as 'we measured the device's error model', "
        "which the same data refutes",
        "            if v <= pfloor:\n                unsatisfiable += 1",
        "            if False:\n                unsatisfiable += 1",
        "test_graph_likeness_DETECTS_a_three_detector_mechanism",
        target="dem_measured",
    ),
    Mutant(
        "M482-damping-scales-the-code-edges-too",
        "the damping factor is applied before the code/added split, so the "
        "consistency arm also cheapens the DEM's own edges -- and the "
        "trade-off curve between generative accuracy and decoding stops "
        "being about the multi-round edges at all",
        "            p = p * damping\n            if p <= pfloor:\n                continue\n            added += 1",
        "            added += 1",
        "test_damping_removes_the_added_edges_and_nothing_else",
        target="dem_measured",
    ),
    # THE HIDDEN-STATE LEAKAGE PRIOR, 2026-09-08 (PREREG_leakage_hmm). A
    # two-state chain per detector position fit by Baum-Welch; the mutants
    # are the E-step / M-step slips that leave a fitter that still runs and
    # still returns numbers, and the pricing slips that leave a decoder
    # that still decodes.
    Mutant(
        "M488-the-leak-rate-is-normalised-by-the-leaked-mass",
        "lambda's M-step divides by the LEAKED occupancy instead of the healthy "
        "one, so a rare leak reads as near-certain; a fit that cannot recover "
        "a planted 0.03 has no business pricing a device",
        "        lam = xi[:, :, 0, 1].sum() / max(g0[:, :-1].sum(), EPS)",
        "        lam = xi[:, :, 0, 1].sum() / max(g1[:, :-1].sum(), EPS)",
        "test_the_fitter_recovers_planted_parameters",
        target="leakage_hmm",
    ),
    Mutant(
        "M489-the-leaked-emission-is-normalised-by-the-healthy-mass",
        "ell's M-step divides by the HEALTHY occupancy, so the leaked state's "
        "firing rate collapses toward zero and the state stops meaning "
        "anything",
        "        ell = _clip((g1 * xf).sum() / max(g1.sum(), EPS))",
        "        ell = _clip((g1 * xf).sum() / max(g0.sum(), EPS))",
        "test_the_fitter_recovers_planted_parameters",
        target="leakage_hmm",
    ),
    Mutant(
        "M490-the-implied-round-profile-forgets-that-qubits-leak-in",
        "the forward marginal only lets leakage decay, never accumulate, so "
        "the model's own round profile is flat while the data (and PAEMS) "
        "rise with the round -- H3 could never be read",
        "        z = z * (1.0 - p.mu) + (1.0 - z) * p.lam",
        "        z = z * (1.0 - p.mu)",
        "test_the_implied_round_profile_matches_the_data",
        target="leakage_hmm",
    ),
    Mutant(
        "M491-the-mixture-pricing-keeps-the-whole-healthy-probability",
        "p' = p + g p_L instead of (1-g) p + g p_L: a fully leaked node "
        "prices its edges ABOVE the leaked probability, and the prereg's "
        "rule is not the rule that ran",
        "        pp = _clip((1.0 - g) * p + g * pl)",
        "        pp = _clip(p + g * pl)",
        "test_reweighting_is_the_prereg_mixture",
        target="leakage_hmm",
    ),
    Mutant(
        "M492-an-edge-takes-the-LESS-leaked-endpoint",
        "g is the minimum endpoint posterior, so an edge into a leaked node "
        "from a healthy one is never repriced -- exactly the edges a leaked "
        "ancilla fires",
        "        g = max(gamma_shot[n] for n in (u, v) if n >= 0)",
        "        g = min(gamma_shot[n] for n in (u, v) if n >= 0)",
        "test_reweighting_is_the_prereg_mixture",
        target="leakage_hmm",
    ),
    Mutant(
        "M493-fifty-leaked-cells-are-enough-to-estimate-an-edge",
        "the cell floor becomes 'any', so p_L is read off a handful of shots "
        "and the fallback the prereg wrote down never fires",
        "        if n_cells >= min_cells and len(ends) == 2:",
        "        if n_cells > 0 and len(ends) == 2:",
        "test_a_few_leaked_cells_below_the_floor_still_fall_back",
        target="leakage_hmm",
    ),
    Mutant(
        "M494-time-like-means-two-rounds-apart",
        "the time-like test asks for a two-round gap, so the consecutive-round "
        "edge -- the measurement-error edge the leaked state should price at "
        "0.5 -- is treated as space-like and keeps its healthy probability",
        "        and abs(u - v) == positions",
        "        and abs(u - v) == 2 * positions",
        "test_timelike_is_same_position_consecutive_round",
        target="leakage_hmm",
    ),
    Mutant(
        "M495-the-marginal-band-is-ten-times-wider",
        "H2's 3-SE band becomes a 30-SE band, and a model whose implied rate "
        "is six times the data passes as generative",
        # anchored from the v1 `implied_rate` call: the same line exists in
        # marginal_check3, and an anchor matching twice mutates the first
        # occurrence silently (test_every_anchor_occurs_EXACTLY_ONCE)
        "        pred = implied_rate(p, n_rounds)\n        for t in range(n_rounds):\n            n = t * positions + i\n            obs = float(det_test[:, n].mean())\n            se = math.sqrt(max(obs * (1.0 - obs), EPS) / S)\n            ok = abs(pred[t] - obs) <= k_se * se",
        "        pred = implied_rate(p, n_rounds)\n        for t in range(n_rounds):\n            n = t * positions + i\n            obs = float(det_test[:, n].mean())\n            se = math.sqrt(max(obs * (1.0 - obs), EPS) / S)\n            ok = abs(pred[t] - obs) <= 10.0 * k_se * se",
        "test_marginal_check_FAILS_a_model_that_is_wrong",
        target="leakage_hmm",
    ),
    Mutant(
        "M496-em-stops-after-one-iteration",
        "the convergence test is always true, so the 'fit' is the first "
        "M-step from the initialiser -- the parameters returned are the "
        "guess, dressed as an estimate",
        # anchored from the v1 HMMParams construction: fit_hmm3 has the same
        # convergence line
        "                      ell=ell, loglik=total, iterations=it)\n        if prev > -math.inf and abs(total - prev) <= tol * max(1.0, abs(prev)):",
        "                      ell=ell, loglik=total, iterations=it)\n        if it >= 1:",
        "test_the_fitter_recovers_planted_parameters",
        target="leakage_hmm",
    ),
    Mutant(
        "M497-the-posterior-is-the-healthy-probability",
        "posterior() returns P(healthy), so every downstream 'leaked' reading "
        "is inverted: the reweighting prices HEALTHY nodes as leaked",
        "    g1, _xi, _ll = forward_backward(np.asarray(x, dtype=np.uint8), p)\n    return g1",
        "    g1, _xi, _ll = forward_backward(np.asarray(x, dtype=np.uint8), p)\n    return 1.0 - g1",
        "test_the_posterior_separates_leaked_cells_from_healthy_ones",
        target="leakage_hmm",
    ),
    Mutant(
        "M498-the-sampler-swaps-leak-and-return",
        "the generative check's transition draw uses mu to leak and lambda to "
        "return, so the answer key the fitter is validated against is a "
        "different process from the one it is told it is",
        "        z = np.where(z == 1, (flip >= p.mu).astype(np.uint8),\n                     (flip < p.lam).astype(np.uint8))",
        "        z = np.where(z == 1, (flip >= p.lam).astype(np.uint8),\n                     (flip < p.mu).astype(np.uint8))",
        "test_the_fitter_recovers_planted_parameters",
        target="leakage_hmm",
    ),
    # THE SANDBOX LOCK, 2026-09-08 (campaign 4.2). Three ways the lock
    # becomes decorative, each with the sibling-run test that sees it.
    Mutant(
        "M499-a-live-owner-no-longer-refuses-the-second-run",
        "the liveness test is dropped, so a second mutation run starts beside "
        "a live one and deletes the tree it is mutating -- the 2026-08-25 "
        "fifty-minute loss, reinstated",
        "        if owner > 0 and _pid_alive(owner):",
        "        if False:",
        "test_a_LIVE_owner_makes_the_second_run_REFUSE",
        target="mutation_sandbox",
    ),
    Mutant(
        "M500-release-drops-a-lock-we-do-not-own",
        "the ownership check on release is short-circuited, so a finishing "
        "run deletes a lock a later run wrote and hands the sandbox to a third",
        "                lock.read_text(encoding=\"utf-8\").strip() == str(os.getpid()):",
        "                True:",
        "test_release_drops_only_OUR_lock",
        target="mutation_sandbox",
    ),
    Mutant(
        "M501-our-own-lock-is-refused",
        "the same process re-entering refuses its own lock as if it were a "
        "sibling's, so a legitimate re-claim inside one run wedges the gate",
        "        if owner == os.getpid():\n            return",
        "        pass",
        "test_RECLAIMING_our_OWN_lock_is_not_a_refusal",
        target="mutation_sandbox",
    ),
    # THE BAND OPTIMUM'S CLOSED FORM, 2026-09-08 (campaign 4.6).
    Mutant(
        # ln(V) instead of ln(V - 1). SURVIVED its first killer (the
        # V = 2 neutrality test) on 2026-09-08: the `<= 2.0` guard
        # returns 0 before the mutated line runs, so the crossover does
        # NOT move -- the optimum above 2 is displaced upward instead
        # (ln 2.5 = 0.92 against ln 1.5 = 0.41 at V = 2.5). Only a test
        # that checks the closed form IS the stationary point sees it;
        # the aim was at the value the mutation cannot reach.
        "M502-the-optimum-above-two-is-displaced",
        "the closed form reads ln(V) / (2 mu), so every band recommended "
        "above the crossover is too wide by ln(V) - ln(V - 1) over 2 mu",
        "    return math.log(erasure_value - 1.0) / (2.0 * mu)",
        "    return math.log(erasure_value) / (2.0 * mu)",
        "test_the_closed_form_is_a_STATIONARY_point_and_a_MINIMUM",
        target="arr",
    ),
    Mutant(
        # The closed form is computed for the report and never competes,
        # so `best` is the nearest grid point again -- the 2026-08 grid
        # artefact, reinstated with the fix's own field still reading
        # correctly. Only a test that requires `best` to BE the analytic
        # row can see it.
        "M503-the-analytic-candidate-does-not-compete",
        "optimal_band is reported but excluded from the candidates, so the "
        "returned optimum is the grid's and band 0 reappears on (2, 2.1)",
        "    if not explicit and analytic is not None:",
        "    if False:",
        "test_a_MUTANT_that_drops_the_analytic_candidate_is_caught",
        target="arr",
    ),
    # THE DEFERRED-PAULI PROPAGATOR, 2026-09-08 (campaign 2.5 / R34).
    Mutant(
        "M504-a-T-never-blocks",
        "the X-component check is dropped, so a pending X passes a T and every "
        "boundary reads deferrable: the blocked fraction R34 reports as the "
        "mechanism's binding cost becomes zero",
        "            for q in g.qubits:\n                if x[q]:\n                    raise Blocked(index, q)\n            return                              # diagonal: Z components pass",
        "            return                              # diagonal: Z components pass",
        "test_a_T_BLOCKS_an_X_component_and_PASSES_a_Z_component",
        target="pauli_defer",
    ),
    Mutant(
        "M505-CX-control-and-target-swapped",
        "x[t] ^= x[c] becomes x[c] ^= x[t], so an X on the control no longer "
        "spreads to the target and the support curve under-counts",
        # THE `elif` LINE IS PART OF THE ANCHOR, and has to be: CY's rule
        # is CX conjugated by S on the target, so it CONTAINS these two
        # lines verbatim and the bare pair matches twice. The registry's
        # uniqueness test caught it before the run could mutate whichever
        # site came first.
        "    elif g.name == \"CX\":\n        c, t = g.qubits\n        x[t] ^= x[c]\n        z[c] ^= z[t]",
        "    elif g.name == \"CX\":\n        c, t = g.qubits\n        x[c] ^= x[t]\n        z[c] ^= z[t]",
        "test_a_WRONG_rule_would_be_caught_by_the_oracle",
        target="pauli_defer",
    ),
)


#: Wall-clock ceiling for ONE mutant's killing test. A mutation can turn a
#: guard into an unbounded computation -- M66 removes the kernel-dimension
#: refusal, and its killing test then walks 2^29 cosets in exact rational
#: arithmetic -- and with no ceiling the gate simply never returns. That is
#: the worst failure mode available to a gate: not a wrong answer, but no
#: answer, which reads as "still running" forever and quietly stops anyone
#: from running it at all. A timeout is NOT scored as a kill; it is
#: reported as TIMEOUT and fails the gate, because "the suite hung" is not
#: the same evidence as "a test caught it".
#: 40 minutes. Chosen against the MEASURED worst case rather than a round
#: number: M66 is the expensive one even after its killing test was moved
#: to the boundary (2**23 coset members instead of 2**29, a 64x cut), and
#: on a contended box that still ran past ten minutes. A ceiling below the
#: slowest honest mutant would turn this gate red for being slow, which
#: teaches people to raise the ceiling rather than read it.
MUTANT_TIMEOUT_S = 2400
BASELINE_TIMEOUT_S = 3600


#: Tests that are (a) seconds-cheap, (b) deterministic, and (c) the ones
#: that go red first when the repository is mid-edit. They are a SUBSET of
#: the baseline, never a substitute for it.
PREFLIGHT: tuple[str, ...] = (
    "tests/test_doc_claims_totals.py",
    "tests/test_mutation_registry_coverage.py",
    # ADDED 2026-08-26, after it cost a 13-minute baseline. Wiring `E2`
    # into the ledger gave it a `proxy=` declaration that
    # `test_the_proxy_audit_has_not_gone_stale` had not been told
    # about -- the audit working exactly as designed. But it lives
    # here, was not in this list, and so collected LATE: the gate ran
    # the whole suite to reach a red it could have reached in eleven
    # seconds. Every gate wired into the ledger trips this file, which
    # makes it the most edit-sensitive cheap test in the repository.
    "tests/test_roadmap_ledger.py",
    # And the one that catches a collapsed heredoc before anything runs
    # the file it broke: 241 files parsed in under two seconds.
    "tests/test_every_source_file_parses.py",
)


def preflight(root: pathlib.Path) -> int:
    """*** 45 MINUTES TO LEARN THAT A PROSE COUNT WAS STALE. ***

    Measured 2026-08-24. The baseline runs the whole suite under `-x`,
    and the failure that stopped it -- a README test-count one integer
    behind the tree -- collects **late**. So the gate spent 603 seconds
    reaching a verdict it could have reached in six, printed
    `baseline: FAIL`, ran **ZERO** mutants, and (because the invocation
    was piped) reported exit 0 from `tail`.

    Nothing here weakens the baseline: `PREFLIGHT` names tests the
    baseline still runs, in the same sandbox, and a green preflight
    decides nothing. It only moves the cheapest, most edit-sensitive
    reds to the front so the refusal costs seconds instead of an hour.

    Returns a non-zero exit code to propagate, or 0 to continue.
    """
    present = [t for t in PREFLIGHT if (root / t).is_file()]
    if not present:
        # A PREFLIGHT THAT SELECTS NOTHING IS NOT A CLEAN PREFLIGHT. The
        # same defect as a `--select` matching no mutant: silence that
        # reads like success.
        print("REFUSING: the preflight names no test that exists in the "
              f"sandbox -- expected {list(PREFLIGHT)}", file=sys.stderr)
        return 3
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *present, "-q", "--no-header",
         "-p", "no:randomly",
         # THE SAME CHICKEN-AND-EGG THE BASELINE ALREADY DECLARES. This
         # test compares the README's mutant total against the artifact
         # THIS RUN writes; including it here would make the gate refuse
         # to produce the very evidence that would satisfy it.
         "--deselect", "tests/test_doc_claims_totals.py::"
                       "test_the_live_readme_hyphenated_total_matches_"
                       "the_artifact"],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    if proc.returncode == 0:
        print(f"preflight: PASS ({len(present)} cheap file(s), first)")
        return 0
    if proc.returncode == 5:
        print("REFUSING: the preflight collected NOTHING -- an absent "
              "check is not a passing one", file=sys.stderr)
        return 3
    print("preflight: FAIL -- refusing before the 10-minute baseline",
          file=sys.stderr)
    print((proc.stdout or "") + (proc.stderr or ""), file=sys.stderr)
    return 2


def run_tests(only: str | None = None,
              root: pathlib.Path | None = None) -> tuple[bool, str]:
    root = root or ROOT
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "-x",
           # -rf so a red baseline SUMMARISES what failed. Without it the
           # only record is the progress dots, and the gate refuses
           # without saying why.
           "-rf", "--tb=line", "-p", "no:randomly"]
    if only:
        cmd += ["-k", only]
    else:
        # THE BASELINE EXCLUDES slow_exhaustive TESTS, AND SAYS SO. The
        # 900-second enumeration sweep is borderline under any load, and a
        # baseline it fails runs ZERO mutants -- twice it took the whole gate
        # down while every mutant-relevant test was green. No mutant names it,
        # so excluding it from the baseline costs no coverage; excluding it
        # SILENTLY would be a dark gate, hence the registered marker, this
        # comment, and the printout.
        # *** AND release_staleness, WHICH IS A FALSE-KILL SOURCE. ***
        #
        # `test_released_bundle_matches_source` compares a PUBLISHED
        # tarball's copy of each module against source. Mutate ANY
        # bundled module -- matching_cert, certifying_decoder, qldpc_check
        # -- and that test fails, so the mutant is scored KILLED by a
        # check that never looked at its behaviour. Every such kill is a
        # mutant whose real killer was never established, and the ones
        # aimed at the checker are exactly the mutants this gate exists
        # to prove are caught.
        #
        # So excluding it makes the mutation result mean MORE, not less.
        # It is also the reason the baseline could not be green while a
        # re-mint is pending: a stale published artifact is an OWNER
        # action outstanding, not a broken suite, and holding mutation
        # testing hostage to it would be the "gate one layer from the
        # defect" failure in the other direction.
        cmd += ["-m", "not slow_exhaustive and not release_staleness"]
        print("baseline excludes -m slow_exhaustive (declared, not silent)")
        print("baseline excludes -m release_staleness: it compares a "
              "PUBLISHED artifact to source, so mutating any bundled "
              "module would score a FALSE KILL (declared, not silent)")
        # AND THE ONE TEST THAT READS THIS GATE'S OWN OUTPUT. [F3]
        #
        # doc_claims_gate checks HANDOFF.md's "90/90 mutants killed"
        # against evidence/mutation_gate.json -- which THIS TOOL writes.
        # So a run that legitimately reports a lower number (an anchor
        # drifted, a mutant skipped) makes the next run's BASELINE red,
        # and the gate can no longer refresh its own evidence. That
        # happened: recovery meant restoring the artifact by hand.
        #
        # Excluding it costs no mutant coverage -- M28 is killed by
        # test_a_stale_headline_in_the_docs_is_actually_caught, a
        # different test, and mutants run under `-k <must_fail>` anyway
        # -- and the check still runs in the ordinary suite and in CI,
        # which is where a real doc drift must be caught. What it stops
        # is a gate that cannot recover from its own honest report.
        cmd += ["--deselect", "tests/test_end_to_end.py::"
                              "test_every_countable_headline_in_the_docs"
                              "_matches_the_evidence"]
        print("baseline deselects the headline-vs-artifact test: it reads "
              "this gate's own output (declared, not silent)")
        # AND THE REGISTRY-vs-ARTIFACT TESTS, FOR EXACTLY THE SAME
        # REASON -- caught before it bit, 2026-08-21.
        #
        # `test_mutation_registry_matches_its_artifact` compares the
        # registry to evidence/mutation_gate.json, which THIS TOOL
        # writes. Registering a new mutant makes it red until the gate
        # runs; if the baseline ran it, the gate could never run, so the
        # very act of adding a mutant would permanently deadlock the
        # tool that measures mutants. That is the same trap [F3]
        # describes above, and it would have been introduced by a gate
        # written to close a real gap.
        #
        # It costs no mutant coverage -- no mutant names it -- and the
        # check still runs in the ordinary suite and on the wall, which
        # is where an unrun mutant must be caught.
        cmd += ["--deselect",
                "tests/test_mutation_registry_matches_its_artifact.py::"
                "test_the_artifact_counted_EVERY_mutant_the_registry_holds",
                "--deselect",
                "tests/test_mutation_registry_matches_its_artifact.py::"
                "test_the_artifact_describes_the_SAME_mutants_not_merely_"
                "as_many"]
        print("baseline deselects the registry-vs-artifact tests: they "
              "read this gate's own output too (declared, not silent)")
        # AND THE THIRD OF THE SAME FAMILY, 2026-08-22.
        #
        # `test_the_live_readme_hyphenated_total_matches_the_artifact`
        # compares README's `N-mutant` headline against
        # evidence/mutation_gate.json -- which THIS TOOL writes. Adding
        # M209 made it red until the gate ran, and the gate could not run
        # while the baseline included it: the same chicken-and-egg the two
        # deselects above already name.
        #
        # It was split OUT of `test_a_hyphenated_bare_total_is_checked_too`
        # precisely so that one stays pure and can remain M209's killer
        # inside the baseline. This half costs no mutant coverage -- no
        # mutant names it -- and still runs in the ordinary suite and on
        # the wall, which is where a real doc drift must be caught.
        cmd += ["--deselect",
                "tests/test_doc_claims_totals.py::"
                "test_the_live_readme_hyphenated_total_matches_the_artifact"]
        print("baseline deselects the README-vs-artifact mutant total: "
              "same reason again (declared, not silent)")
    env_path = str(root / "src")
    try:
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": env_path},
            timeout=BASELINE_TIMEOUT_S if only is None else MUTANT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # Distinct from a pass AND from a kill: the caller must be able to
        # tell "no test caught this" from "we never found out".
        return None, (f"TIMEOUT after "
                      f"{BASELINE_TIMEOUT_S if only is None else MUTANT_TIMEOUT_S}s")
    if proc.returncode == 5:
        # *** PYTEST COLLECTED NOTHING. *** Neither a pass nor a fail, and
        # emphatically not a kill: `5 != 0` used to read as suite_green =
        # False, which this gate maps to KILLED. A `must_fail` that has
        # been renamed or deleted therefore scored its mutant as caught
        # by a test that does not exist -- forever, silently, with the
        # headline still reading N/N.
        return _NO_TEST_RAN, (
            f"pytest COLLECTED NOTHING for -k {only!r} (exit 5). The named "
            f"test does not exist: a mutant cannot be killed by a test "
            f"that was never run. " + proc.stdout[-300:])
    # *** A REFUSAL THAT CANNOT NAME ITS CAUSE. ***
    #
    # This returned `proc.stdout[-400:]`, and under `-q` the last 400
    # characters of a 2,600-test run are progress dots. So a red
    # baseline printed a wall of dots, refused, and left the operator to
    # find the failing test themselves -- twice in one session.
    #
    # The FAILED lines are what the reader needs, and they are not at
    # the tail. Pull them from the whole stream, and keep a tail as
    # well for the case where pytest died before summarising.
    failed = [ln for ln in proc.stdout.splitlines()
              if ln.startswith(("FAILED", "ERROR"))
              or " - " in ln and ln.lstrip().startswith("tests")]
    report = chr(10).join(failed[:20])
    if not report:
        report = proc.stdout[-1200:]
    return proc.returncode == 0, report


def _conflicting_processes() -> list[str]:
    """Any live process that might IMPORT a mutated source file.

    THE HAZARD, found while auditing rather than after a bad result: this gate
    temporarily writes DELIBERATELY BROKEN source into src/oneq. A long-running
    proof already holds its imports in memory and is safe -- but any process
    that STARTS during the window (a queued retry, a resumed class) would import
    the mutant and could emit a 'closure' computed by a knowingly broken engine.
    Certificates are the one artifact that must never be produced under a
    mutation window, so refuse to open one while such work is live or armed.

    RETURNS None WHEN IT COULD NOT LOOK -- see the caller, which refuses.

    THE DEFECT THIS REPLACES. The probe was PowerShell-only and wrapped
    in `except Exception: return []`. An empty list means "nothing
    conflicting is live", so on Linux and macOS, and on Windows whenever
    PowerShell was absent, blocked by policy, or slower than the
    timeout, the interlock reported ALL CLEAR without having looked.
    That is a skip read as a pass, guarding the one window in which a
    knowingly broken engine could mint a certificate.

    So "no conflicts" and "could not determine" are now DIFFERENT
    values. Portability alone would not have fixed it; it would have
    moved the silence to the next environment nobody tested.
    """
    import subprocess
    me = os.getpid()

    def _hit(pid: int, name: str, cmdline: str) -> str | None:
        # EXCLUDE THIS PROCESS. The patterns match tool names, so
        # `mutation_gate.py --json queue_tail.json` would match the
        # command doing the asking and the gate would refuse to run
        # because of itself.
        if pid == me or not cmdline or not _process_may_observe_mutant(
                cmdline):
            return None
        return f"{pid} {name}"

    # 1. psutil, if it happens to be installed. Not a declared dependency,
    #    so its absence is ordinary rather than an error.
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            found = []
            for pr in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmd = " ".join(pr.info.get("cmdline") or ())
                except Exception:
                    continue
                h = _hit(pr.info["pid"], pr.info.get("name") or "", cmd)
                if h:
                    found.append(h)
            return found
        except Exception:
            pass

    # 2. Linux /proc -- no subprocess, nothing to be blocked by policy.
    proc = pathlib.Path("/proc")
    if proc.is_dir():
        try:
            found = []
            for d in proc.iterdir():
                if not d.name.isdigit():
                    continue
                try:
                    cmd = (d / "cmdline").read_bytes().replace(
                        b"\0", b" ").decode("utf-8", "replace")
                    name = (d / "comm").read_text(
                        encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue        # the process exited; not our concern
                h = _hit(int(d.name), name, cmd)
                if h:
                    found.append(h)
            return found
        except Exception:
            pass

    # 3. POSIX ps, then Windows PowerShell.
    for argv in (["ps", "-eo", "pid=,comm=,args="],
                 ["powershell", "-NoProfile", "-Command",
                  "Get-CimInstance Win32_Process | ForEach-Object { "
                  "$_.ProcessId.ToString() + ' ' + $_.Name + ' ' + "
                  "$_.CommandLine }"]):
        try:
            out = subprocess.run(argv, capture_output=True, text=True,
                                 timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode != 0:
            continue
        found = []
        for ln in out.stdout.splitlines():
            parts = ln.strip().split(None, 2)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            h = _hit(int(parts[0]), parts[1],
                     parts[2] if len(parts) > 2 else "")
            if h:
                found.append(h)
        return found

    # NOTHING COULD ANSWER. Not the same as "all clear".
    return None


def _interlock_verdict(busy: list[str] | None,
                       force: bool) -> tuple[bool, str]:
    """(refuse, message) -- the interlock DECISION, separated to be tested.

    This lived inline in `main`, which meant the only way to check it was
    to assert on the source text or to run the whole gate. Text checks
    have already produced false findings in this repository, and running
    the gate to test its own precondition is not a test. It is a pure
    function of two values, so it is one here.

    THREE CASES, and the middle one is the defect this replaced:
      None -> could not look          REFUSE (was: reported all-clear)
      [..] -> conflicting work live   REFUSE
      []   -> looked, nothing live    PROCEED

    A false positive -- some unrelated process whose command line merely
    mentions one of the tool names -- refuses a run that would have been
    safe. That is the correct direction to be wrong in, and `--force` is
    the documented way past it.
    """
    if force:
        return False, ""
    if busy is None:
        return True, (
            "REFUSING to open an in-place mutation window: could not "
            "determine whether certificate-producing work is live.\n"
            "No process probe was available (psutil, /proc, ps, PowerShell "
            "all failed). An interlock that cannot look must not report\n"
            "all-clear -- that is how a mutation window opens under live "
            "work. Drop --in-place to run in a sandbox, install psutil,\n"
            "or --force if you have checked by hand.")
    if busy:
        return True, (
            "REFUSING to open an in-place mutation window: "
            "certificate-producing work is live\n"
            + "\n".join(f"  {b}" for b in busy)
            + "\nA process starting during the window would import a "
              "deliberately broken engine and could emit a bogus closure.\n"
              "Drop --in-place to run in a sandbox instead, or wait, or "
              "--force.")
    return False, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    # *** THE DEFAULT USED TO BE "PRODUCE NO EVIDENCE". ***
    # A forty-minute run that records nothing is not a cheap mistake: it
    # is the whole run, spent, with the artifact three separate tests
    # compare the registry against left stale. README documents
    # `python tools/mutation_gate.py` with no flag, so the DOCUMENTED
    # invocation was the one that produced no evidence -- and a stale
    # artifact reads as a smaller registry, never as an absent run.
    # Writing is now the default and silence is the opt-in.
    ap.add_argument("--json", type=str, default="evidence/mutation_gate.json",
                    help="write the evidence artifact here (default); "
                         "--no-json to suppress")
    ap.add_argument("--no-json", action="store_true",
                    help="run without recording the artifact")
    ap.add_argument("--force", action="store_true",
                    help="run even if certificate-producing work is live (unsafe)")
    ap.add_argument("--in-place", action="store_true",
                    help="mutate the WORKING TREE instead of a sandbox copy (unsafe)")
    ap.add_argument("--select", type=str, default=None,
                    help="comma-separated mutant id prefixes to run (a partial run; "
                         "the evidence artifact is NOT written, because a subset "
                         "score is not the gate's score)")
    args = ap.parse_args()
    if args.no_json:
        args.json = None
    selected = ([s.strip() for s in args.select.split(",") if s.strip()]
                if args.select else None)

    # *** A SELECTOR THAT MATCHES NOTHING IS A DARK RUN, NOT A CLEAN ONE.
    # *** Measured: `--select "M15[789]"` -- a regex, where this takes
    # comma-separated PREFIXES -- matched no mutant and the gate printed
    # "MUTATION GATE: 0/0 killed" and exited 0. Read quickly that is
    # indistinguishable from a passing partial run, and it is the same
    # defect `tools/gate_all.py` already guards against with
    # --strict-markers: a selector that silently selects nothing looks
    # exactly like a check that had nothing to complain about.
    if selected is not None:
        matched = sorted({m.id for m in MUTANTS
                          for p in selected if m.id.startswith(p)})
        if not matched:
            print(f"FATAL: --select {args.select!r} matched no mutant. This "
                  f"takes comma-separated ID PREFIXES, not a regex or a "
                  f"glob. Nothing would have run, and '0/0 killed' would "
                  f"have looked like a pass.", file=sys.stderr)
            # SHOW REAL IDS. The first version printed the part before
            # the first '-', which is "M0, M1, M10, M100 ..." -- true,
            # useless, and it tells a reader nothing about the shape they
            # got wrong.
            print(f"  {len(MUTANTS)} mutants; the last few are:",
                  file=sys.stderr)
            for m in MUTANTS[-4:]:
                print(f"    {m.id}", file=sys.stderr)
            return 4
        print(f"--select matched {len(matched)} mutant(s): "
              f"{', '.join(matched[:8])}"
              f"{' ...' if len(matched) > 8 else ''}")

    if args.in_place:
        # The old behaviour, kept only because a sandbox that silently diverged
        # from the tree would be worse than no gate. Still interlocked.
        refuse, msg = _interlock_verdict(_conflicting_processes(), args.force)
        if refuse:
            print(msg, file=sys.stderr)
            return 3
        work = ROOT
    else:
        # A REFUSAL IS NOT A CRASH. Letting `SandboxBusy` propagate would
        # print a traceback, and an exit code cannot tell a crash from a
        # refusal -- the operator would go looking for a broken gate
        # instead of the sibling run that is the whole message.
        try:
            work = _sandbox()
        except SandboxBusy as busy:
            print(f"REFUSED: {busy}", file=sys.stderr)
            return 2
        print(f"sandbox: {work}  (the working tree is never mutated)")

    targets = _targets(work)
    # FAIL BEFORE THE BASELINE, NOT AFTER THE 16TH KILL. A mutant whose target
    # file is not registered crashed the gate mid-run with a KeyError -- after
    # 43 minutes of kills, with no artifact written, leaving the PREVIOUS
    # run's 43/43 on disk looking current while a 44th mutant existed that it
    # had never counted. That is the stale-artifact shape this gate exists to
    # prevent, produced by the gate itself. Validate every declared target
    # up front and name the offender.
    unknown = sorted({m.target for m in MUTANTS} - set(targets))
    if unknown:
        print(f"FATAL: mutant target(s) not registered in _SOURCES/"
              f"_TOOL_SOURCES: {unknown}", file=sys.stderr)
        return 4
    # A KILLING TEST THAT DOES NOT EXIST is a mutant that can never die --
    # the registry looks guarded while the gate would report the mutant
    # unkillable (or worse, a typo'd name silently matches nothing). Check
    # the name is defined somewhere under tests/ before any run.
    test_src = "\n".join(f.read_text(encoding="utf-8", errors="replace")
                         for f in (ROOT / "tests").glob("test_*.py"))
    ghosts = sorted({m.must_fail for m in MUTANTS
                     if f"def {m.must_fail}" not in test_src})
    if ghosts:
        print(f"FATAL: killing test(s) not found under tests/: {ghosts}",
              file=sys.stderr)
        return 4
    # *** BYTES, NOT TEXT, ON BOTH SIDES OF THE ROUND TRIP. ***
    # `write_text` translates a newline to os.linesep on Windows, so
    # read-modify-write silently converts an LF file to CRLF. In the
    # sandbox that is invisible (the copy is disposable); under
    # `--in-place` it rewrites every mutated file in the WORKING TREE
    # -- and `git diff` hides it, because autocrlf normalizes on read,
    # while `git status` reports the file modified. Porcelain-dirty
    # and content-identical is the hardest kind of change to explain.
    #
    # MEASURED 2026-08-24: a helper doing exactly this round trip left
    # `bayes_coset.py` and `qldpc_check.py` modified after a restore
    # that was supposed to be byte-exact.
    #
    # *** AND THE FIRST VERSION OF THIS FIX BLINDED FIVE MUTANTS. ***
    # `read_text` performs universal-newline translation; `read_bytes`
    # then `.decode()` does NOT. So every anchor spanning a newline
    # stopped matching on any CRLF file, and the gate printed
    # `SKIP (anchor not found -- source drifted)` -- M265, M296, M258,
    # M208 and M53, all still perfectly matched in the source. A fix for
    # a byte-fidelity defect that silently removed five checks.
    #
    # THE TWO NEEDS ARE SEPARATE, so they are met separately: `originals`
    # holds the RAW BYTES, and only the restore path uses them, which is
    # what makes the restore exact. Matching and writing the mutant both
    # happen on NEWLINE-NORMALISED text -- the mutated file is temporary
    # and need only be valid source, never byte-identical to anything.
    originals = {k: v.read_bytes() for k, v in targets.items()}
    normalised = {k: v.decode("utf-8").replace("\r\n", "\n")
                  for k, v in originals.items()}
    results = []

    # *** EVERY ANCHOR IS CHECKED BEFORE ANY MUTANT RUNS. ***
    # A drifted anchor prints SKIP mid-run, an hour in, in a wall of
    # kills -- and a SKIP proves nothing while reading like progress.
    # Five were blinded at once by a line-ending change; the run went on
    # for 209 mutants before anyone looked. This is decidable in
    # milliseconds and is a REFUSAL, so the registry cannot silently
    # shrink.
    drifted = sorted(m.id for m in MUTANTS
                     if not (selected and not any(
                         m.id.startswith(p) for p in selected))
                     and m.find not in normalised[m.target])
    if drifted:
        print(f"FATAL: {len(drifted)} anchor(s) match nothing in their "
              f"target, so those mutants would SKIP and their checks "
              f"would go unproven: {drifted}", file=sys.stderr)
        return 4

    rc = preflight(work)
    if rc:
        return rc

    baseline_ok, baseline_out = run_tests(root=work)
    # *** THE SENTINEL IS TRUTHY, AND `if not baseline_ok` IS NOT ENOUGH. ***
    # `NO_TEST_RAN` is a non-empty string, so the obvious guard below reads
    # it as a PASS -- the exact false pass this sentinel was introduced to
    # remove, reintroduced one line away by the fix itself. A three-valued
    # return demands a three-valued test at EVERY call site, and the
    # boolean-shaped one is the one that looks fine.
    if baseline_ok is _NO_TEST_RAN:
        print("baseline: NO TEST RAN", file=sys.stderr)
        print("the baseline suite collected NOTHING. That is not a green "
              "suite, it is an absent one, and every mutant scored against "
              "it would be scored against nothing.", file=sys.stderr)
        print(baseline_out, file=sys.stderr)
        return 2
    print(f"baseline: {'PASS' if baseline_ok else 'FAIL'}")
    if not baseline_ok:
        # SAY WHY. A gate that refuses without a reason costs a debugging round
        # trip every time, and the reason is already in hand.
        print("the suite must be green before mutation testing means anything",
              file=sys.stderr)
        print("--- sandbox baseline output ---", file=sys.stderr)
        print(baseline_out, file=sys.stderr)
        return 2

    # NO PARALLEL COUNTERS. What each mutant did is recorded once, in
    # `results`, and the totals are derived from that record after the
    # loop (see `_counts`). Keeping `+= 1` tallies beside the list they
    # were appending to made the headline and the artifact two
    # independent accounts of one run, with nothing comparing them.
    try:
        for m in MUTANTS:
            if selected and not any(m.id.startswith(p) for p in selected):
                continue
            tgt = targets[m.target]
            # NORMALISED text for the anchor; the raw bytes are used
            # only by the restore in the `finally` below.
            original = normalised[m.target]
            if m.find not in original:
                print(f"  {m.id:22s} SKIP  (anchor not found -- source drifted)")
                results.append({**asdict(m), "outcome": "skip"})
                continue
            tgt.write_bytes(
                original.replace(m.find, m.replace, 1).encode("utf-8"))
            # `suite_green` TRUE means the named killing test still PASSED
            # against broken source -- the mutant SURVIVED. It was called
            # `caught`, which reads as the exact opposite of what it
            # holds, in the lines that decide this gate's whole score.
            # The mapping now lives in `mutation_outcome`, so the one
            # tool that polices whether a check can fail is itself
            # checkable -- and mutated (M216).
            suite_green, why = run_tests(only=m.must_fail, root=work)
            tgt.write_bytes(original.encode("utf-8"))
            verdict = _outcome(suite_green)
            if verdict == "timeout":
                print(f"  {m.id:22s} TIMEOUT  ({why}) -- NOT scored as a kill")
            elif verdict == "survived":
                print(f"  {m.id:22s} SURVIVED  <-- {m.must_fail} did NOT catch it")
            else:
                print(f"  {m.id:22s} killed by {m.must_fail}")
            results.append({**asdict(m), "outcome": verdict})
    finally:
        for k, v in targets.items():
            v.write_bytes(originals[k])

    # A SKIP MUST NOT LEAVE THE DENOMINATOR.
    #
    # A mutant whose anchor drifted printed SKIP on one line and then vanished
    # from the total, so the headline read "29/29 killed" while thirty mutants
    # were declared and one had never run. That is this gate's own silent-stop
    # shape turned on itself: coverage FELL and the score went UP, because the
    # denominator shrank along with the numerator. Refactoring a source file is
    # enough to trigger it -- which is precisely when coverage matters most.
    # A TIMEOUT STAYS IN THE DENOMINATOR for the same reason a skip does:
    # it is a mutant that was declared and not proven caught, and letting
    # it leave the total would raise the score while lowering coverage.
    # The rule, and the scar behind it, live in `denominator` --
    # where a test can watch it fail and M217 can break it. The counts
    # themselves come from the RECORD (`_counts`, M218), so the headline
    # and the artifact cannot be two accounts of one run.
    tally = _counts(r["outcome"] for r in results)
    killed, survived = tally["killed"], tally["survived"]
    skipped, timed_out = tally["skip"], tally["timeout"]
    no_test_ran = tally["no_test_ran"]
    total = _denominator(killed, survived, skipped, timed_out, no_test_ran)
    print(f"\nMUTATION GATE: {killed}/{total} killed"
          + (f"  ({skipped} SKIPPED -- anchor drifted, coverage NOT proven)"
             if skipped else "")
          + (f"  ({timed_out} TIMED OUT -- coverage NOT proven)"
             if timed_out else "")
          + (f"  ({no_test_ran} NO TEST RAN -- the named killer does not "
             f"exist, coverage NOT proven)" if no_test_ran else ""))
    if args.json and selected:
        # A SUBSET SCORE IS NOT THE GATE'S SCORE. Writing "2/2 killed" into the
        # evidence artifact would overwrite a full run's 39/39 with a number
        # that looks perfect and covers almost nothing -- the same
        # shrinking-denominator shape the SKIP rule above exists to prevent.
        print("--select given: refusing to write the evidence artifact from a "
              "partial run", file=sys.stderr)
    elif args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"targets": [str(v.relative_to(work)) for v in targets.values()],
             "mutated": "working-tree" if args.in_place else "sandbox-copy",
             "total": total, "killed": killed, "survived": survived,
             "skipped": skipped, "timed_out": timed_out,
             "results": results}, indent=2), encoding="utf-8")
        print(f"evidence -> {out}")

    if skipped:
        # A DRIFTED ANCHOR IS A FAILURE, NOT A FOOTNOTE. It means a declared
        # mutant never ran, so the check it was written to defend is unproven
        # again -- indistinguishable, from the exit code, from never having
        # written it. Refactoring is what causes drift, and refactoring is
        # exactly when you need the coverage.
        print(f"FAILED: {skipped} mutant(s) SKIPPED because their anchors no "
              f"longer appear in the source. Re-anchor them; a skipped mutant "
              f"proves nothing.", file=sys.stderr)
    if timed_out:
        # Same verdict as a skip, different cause: the mutant ran and we
        # stopped waiting. Either its killing test is too expensive under
        # mutation (fix the TEST, at the boundary) or the mutation really
        # does hang the suite (which is a finding, not a pass).
        print(f"FAILED: {timed_out} mutant(s) TIMED OUT. A mutant we stopped "
              f"waiting for is not a mutant a test caught -- shrink the "
              f"killing test to the boundary, or record the hang as the "
              f"finding it is.", file=sys.stderr)
    if survived:
        print("FAILED: a surviving mutant means that check is decorative.", file=sys.stderr)
    # `no_test_ran` is fatal like the rest: a mutant whose killer could
    # not be found is the furthest thing there is from proven caught.
    return 1 if (survived or skipped or timed_out or no_test_ran) else 0


if __name__ == "__main__":
    raise SystemExit(main())
