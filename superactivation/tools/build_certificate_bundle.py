#!/usr/bin/env python3
r"""Assemble the M0.5 re-runnable bundle: everything a stranger needs, nothing they must trust.

A bundle is a promise that someone else can reach our conclusions without us.
That promise breaks in two ways, and this project has made both:

  THE DESCRIPTION DISAGREES WITH THE BYTES. Four releases in a row shipped an
  artifact contradicting its own manifest, every one because different parts of
  the pipeline disagreed about which files the release consists of. The fix
  lives in `mint_release.payload()` -- ONE file list, from which the manifest,
  the archive and the tree pin all derive. This tool IMPORTS that function
  rather than reimplementing it, because a second copy of the rule is how the
  bug came back the last four times.

  THE BUNDLE PROVES ITS OWN CLAIMS. A bundle that ships only our verifier
  checking our certificates against our graph is a closed loop. So the bundle
  carries TWO checkers in different languages, the conformance vectors that
  pin them to each other, the FORGERIES (most of the vectors are documents
  that must be REFUSED; how many is counted from conformance/cert_vectors.json
  by `conformance_vector_counts` and printed in the bundle README, never typed
  here), and the seeded-fault corpus -- so a reader can score their own
  detector against the same faults instead of taking our two numbers on faith.

THE IN-TOTO PREDICATE. Emitted as an in-toto Statement v1 rather than a bespoke
format, because the milestone says to propose a predicate type rather than
invent a wrapper. Subjects are the evidence artifacts with their digests; the
predicate records what was run, on what, and -- the part that matters -- what
each number does NOT establish.

WHAT SIGNING DOES. It attests who assembled this. It says nothing about whether
the certificates inside are valid; that is what re-running the checker is for.
The two are kept in separate calls throughout this codebase for exactly that
reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mint_release import TEXT, build_archive, normalize, write_manifest  # noqa: E402

CERT_VECTORS = "conformance/cert_vectors.json"


def conformance_vector_counts(path: pathlib.Path | None = None) -> tuple[int, int]:
    """(vectors, refusals) COUNTED from the conformance vectors, for every sentence that states them.

    The README said "56 vectors ... 48 of them are refusals" and "56/56" while the file held 59 and 51 (2026-11-03):
    three typed copies of one count, none of them read from the file they describe. A vector whose expected verdict
    is not a boolean is a refusal to count, not a guess.
    """
    doc = json.loads((path or ROOT / CERT_VECTORS).read_text(encoding="utf-8"))
    flags = [v.get("expected_accepted") for v in doc.get("vectors", [])]
    if not flags or not all(isinstance(f, bool) for f in flags):
        raise ValueError(f"{CERT_VECTORS}: no vectors, or a vector without a boolean expected_accepted")
    return len(flags), flags.count(False)


def _cert_vectors_note() -> str:
    # The builder ships in paper 4's bundle too, which carries no conformance vectors; importing it there (paper 4's
    # generator does) must not need them. A decoder build without the file still refuses: it is declared in CONTENTS.
    if not (ROOT / CERT_VECTORS).is_file():
        return "the conformance vectors (absent from this tree)"
    n, refusals = conformance_vector_counts()
    return f"{n} vectors, {refusals} of them REFUSALS (counted from the file)"


# What goes in. Paths are repo-relative; a missing one is an ERROR, not a
# silent omission -- a bundle that quietly ships without its forgery vectors
# still looks complete.
CONTENTS = [
    ("schemas/decoding-certificate-1.schema.json", "the certificate format"),
    (CERT_VECTORS, _cert_vectors_note()),
    ("verifier-js/verify-certificate.mjs", "second-language checker, zero deps"),
    ("src/oneq/matching_cert.py", "the checker"),
    # THE SOUND CHECKER MUST SHIP WITH THE BROKEN ONE. An external audit
    # found the bundle carrying matching_cert.py -- which accepts a
    # weight-20 correction on a graph whose optimum is 2 -- and NOT
    # matching_cert_strict.py, the repair. An offline auditor therefore
    # received the confirmed break and no access to the fix, while
    # api.py told them "a stranger verifies with verify() and needs
    # nothing else from you". Adding verify_strict to api.py satisfied
    # the reachability gate; it did not put the module in the box a
    # stranger opens.
    ("src/oneq/matching_cert_strict.py", "the strict T-odd cut checker"),
    ("src/oneq/__init__.py", "the package init the held-out producer loaded; evidence/producer_identity_willow.json "
                             "pins its bytes, so paper 1's generator re-hashes it (its imports are lazy)"),
    ("src/oneq/cert_format.py", "serialisation, graph binding, exact floats"),
    ("src/oneq/moat_dual.py", "the dual producer (reference + vectorised)"),
    ("src/oneq/moat_growth.py", "primal-dual moat growth"),
    ("src/oneq/certifying_decoder.py", "the escalation ladder"),
    ("src/oneq/tjoin_exact.py", "the exact tier: native Padberg-Rao duals"),
    ("src/oneq/streaming.py", "ordered watermarked receipt stream"),
    ("src/oneq/qldpc_cert.py", "qLDPC certificates: Feldman duals + repair"),
    # EXTERNAL AUDIT 2026-08-04, finding 3. AUDIT.md names this as THE
    # 685-line central checker to read, and it was not in the archive --
    # qldpc_cert imports it, so the central qLDPC result could not be
    # executed or fully reviewed from the release. Shipping the producer
    # without its checker is shipping the half a reader must trust.
    # The commands AUDIT.md advertises must BE here. Finding 2: they were
    # not, so the three-command audit failed on a sterile extraction --
    # "it's in the repo" is no defence to someone who has only the tarball.
    ("tools/verify_release.py", "step 0: signatures, digests, manifests"),
    ("tools/recheck_all.py", "step 1: COMPLETE manifest-pinned replay of every receipt"),
    ("tools/replay_receipts.py", "the complete replay: fails on missing, duplicate, extra or altered"),
    ("tools/receipt_manifest.py", "builds the receipt manifest, refusing disagreeing artifacts"),
    ("tools/export_receipt_instances.py", "persisted instances; --verify rebuilds and must match"),
    ("experiments/m0_prior_art/qldpc_dev_refsched_trees.py", "the 4 development circuit trees"),
    ("experiments/m0_prior_art/qldpc_strong_baseline_v2.py", "strong baseline with receipts on every answer"),
    ("experiments/m0_prior_art/qldpc_strong_baseline_v3.py",
     "strong baseline with every proof RETAINED, content-addressed, and every decision naming its proof"),
    ("tools/replay_strong_baseline.py",
     "rebuilds the strong-baseline tables from records and proofs alone: no producer, no solver, no BP-OSD"),
    ("tools/reproduce_heldout_public.py",
     "the public-only route to re-run the frozen held-out campaign from the source snapshot, no git history"),
    ("experiments/dual_observable/certified_measured_lane.py",
     "paper 1's measured-model section: the certified accept path run against a decoding graph "
     "estimated from each run's own first half, held out, with the campaign's scorer beside it"),
    # ITS IMPORT CLOSURE. The bundle refuses to ship a script it cannot run (the check that caught
    # this), and the closure is worth carrying on its own account: `dem_measured` IS the two-point
    # estimator the section's model comes from, so a reader auditing that claim needs it in hand.
    ("experiments/dual_observable/measured_model_lane.py",
     "the estimated-model lane the certified one is scored against: families, tails and the "
     "held-out decode comparison"),
    ("experiments/dual_observable/hardware_noise_floor.py",
     "the campaign's own scorer, the second decoder the certified lane must agree with per record"),
    ("src/oneq/dem_measured.py",
     "the two-point estimator itself: pairwise probabilities from detector correlations alone, no "
     "calibration input, no logical outcomes"),
    ("src/oneq/dem_families.py", "the time-like / space-like family split and the burst price"),
    ("src/oneq/dem_calibration.py", "the device-calibration model the estimate is compared against"),
    ("src/oneq/certified_weight.py", "the certified weight interval the noise floor reads"),
    ("src/oneq/soft_estimator.py", "the soft-output estimate reported beside the certified one"),
    ("src/oneq/exchangeability.py", "the exchangeability screen the noise floor applies"),
    ("docs/PREREG_measured_model_program_2026-09-16.md",
     "the preregistration that fixed predictions A1-A4 before that lane was run, with its receipts"),
    ("tools/bposd_circuit_outcome.py",
     "BP-OSD's own observable outcome on the held-out circuit shots, derived from the records and the pinned DEM (audit F-18)"),
    ("docs/EXTERNAL_AUDIT_2026-09-14.md",
     "the register of the 2026-09-14 external audits' findings, verdicts and closures, with the commit that closed each"),
    ("docs/EXTERNAL_AUDIT_2026-09-15.md",
     "the register of the 2026-09-15 follow-up audit's findings, verdicts and closures"),
    ("tools/compare_heldout.py", "compare a re-run: science, not wall-clock"),
    ("src/oneq/qldpc_check.py",
     "THE qLDPC CHECKER: flat, exact-rational and branch-dual tiers"),
    ("tools/frozen/qldpc_check.py",
     "pre-refactor snapshot the equivalence gate compares against"),
    ("tools/frozen/matching_cert.py", "pre-refactor snapshot"),
    ("tools/frozen/qldpc_cert.py", "pre-refactor snapshot"),
    ("tools/frozen/tjoin_exact.py", "pre-refactor snapshot"),
    ("tests/test_audit_2026_08_04.py",
     "the six false-acceptance paths the audit found, pinned"),
    ("src/oneq/forgeries.py", "the audited attack battery"),
    ("src/oneq/passport.py", "canonical bytes + Ed25519, reused not rebuilt"),
    # `passport.build_core` gained a `model_audit` block (ledger R94/R96a), and it
    # RE-CHECKS that block before signing it, so the module is reachable from the
    # signed core and the bundle cannot ship the passport without it. Seven files
    # import it and the mint was refusing each of them by name.
    ("src/oneq/model_audit.py",
     "is the model this certificate binds to structurally misspecified"),
    ("experiments/m0_prior_art/million_shot_gate.py", "the scale run"),
    ("experiments/m0_prior_art/google_corpus_gate.py", "the hardware run"),
    ("experiments/m0_prior_art/corpus_chunk_runner.py", "chunk-resumable atom"),
    ("tools/corpus_chunk_driver.py", "chunk-resumable driver + merge"),
    ("experiments/m0_prior_art/qldpc_bb_gate.py", "the [[144,12,12]] gate"),
    ("experiments/m0_prior_art/qldpc_bb_phenom.py", "phenomenological gate"),
    ("experiments/m0_prior_art/qldpc_circuit_gate.py", "circuit-level gate"),
    ("experiments/m0_prior_art/qldpc_bb_circuit_gate.py", "the full stack"),
    ("experiments/m0_prior_art/qldpc_ensembles_gate.py", "ensemble artifacts"),
    ("experiments/m0_prior_art/qldpc_exact_gate.py",
     "exact-rational re-certification + independent cross-validation"),
    ("experiments/m0_prior_art/qldpc_paired_stress_gate.py",
     "paired 2x2 + McNemar with per-shot records"),
    ("experiments/m0_prior_art/qldpc_exact_all_gate.py",
     "the canonical exact datasets: every non-circuit rung"),
    ("experiments/m0_prior_art/qldpc_scale_probe.py",
     "quantization robustness of the fixed-point objective"),
    ("experiments/m0_prior_art/qldpc_bb_refsched_gate.py",
     "THE PRIMARY circuit experiment: published schedule, exactified"),
    ("experiments/m0_prior_art/qldpc_bnb_residue_gate.py",
     "branch-dual closure of the solver tier"),
    ("experiments/m0_prior_art/qldpc_frame_demo_gate.py",
     "FRAME-CERTIFIED receipts: validated + demonstrated"),
    ("experiments/m0_prior_art/qldpc_heldout_flat_certs.py",
     "the 2,002 held-out flat certificates, regenerated from the shipped "
     "records (no new sampling) and STORED with identical_to_run per "
     "certificate; recheck_all.py re-checks every one from this payload"),
    ("experiments/m0_prior_art/qldpc_heldout_gate.py",
     "THE run of record: frozen held-out campaign"),
    ("docs/HELD_OUT_PREDECLARATION.md",
     "the immutable preregistration record"),
    ("docs/HELD_OUT_RESULTS.md", "observed results + disclosures"),
    ("docs/PRIOR_ART_REGISTER.md",
     "the standing register of what is and is not ours, row by row, with each row's nearest found "
     "work and the verbatim query that looked for it; both manuscripts' novelty wording is gated "
     "on it, and paper 1's measured-model section points a reader here"),
    ("docs/PAPER1_LITERATURE_SEARCH.md",
     "paper 1's literature searches: databases, dates, verbatim queries, what was not searched"),
    ("docs/PAPER2_LITERATURE_SEARCH.md",
     "paper 2's literature searches: databases, dates, verbatim queries, what was not searched"),
    ("docs/REVIEW_PACKET.md",
     "the reviewer's guide: each paper's claims, their evidence class, and the artifact or command that re-derives each"),
    ("tools/external/tree_checker_b.py",
     "second branch-dual tree checker, from the written schema"),
    ("experiments/m0_prior_art/qldpc_refsched_fault_xval.py",
     "circuit transcription + single-fault signature cross-validation"),
    ("experiments/m0_prior_art/qldpc_circuit_distance_cert.py",
     "certified circuit-distance bounds (and the method's limit)"),
    ("experiments/m0_prior_art/qldpc_bb_family_gate.py",
     "the ladder across the published BB family, constructor self-checked"),
    ("experiments/m0_prior_art/qldpc_bposd_sensitivity.py",
     "decoder-configuration sensitivity: tier1 moves, certification does not"),
    ("experiments/m0_prior_art/qldpc_tree_corpus_gate.py",
     "the complete 60-tree corpus: both checkers, full cost table"),
    ("experiments/m0_prior_art/qldpc_family_circuit_gate.py",
     "the published schedule generalized to any BB code, at circuit level"),
    ("experiments/m0_prior_art/qldpc_frame_campaign_gate.py",
     "FRAME-CERTIFIED as a measured per-rung rate"),
    ("experiments/b1_analog/tcert_gate.py",
     "B1-alpha: the certified analog horizon, offline"),
    ("experiments/m0_prior_art/qldpc_strong_baseline_gate.py",
     "the decisive test: pipeline vs a deliberately strong BP-OSD"),
    ("experiments/m0_prior_art/qldpc_dem_mechanism_audit.py",
     "the DEM leg: mechanism list == enumerated physical faults"),
    # paper 2's post-hoc analyses of the 2026-11-02 review round, each named by the manuscript
    ("experiments/m0_prior_art/qldpc_strong_baseline_scaled.py",
     "post hoc: the strong baseline's A and B with min-sum scaling 0.625, classified by the stored proofs"),
    ("experiments/m0_prior_art/qldpc_heldout_scaled_posthoc.py",
     "post hoc: the held-out syndromes re-decoded with scaling, after reproducing the campaign"),
    ("experiments/m0_prior_art/qldpc_bposd_scaling_sensitivity.py",
     "post hoc: BP-OSD against the declared rate and the scaling factor"),
    ("experiments/m0_prior_art/qldpc_minsum_rate_invariance.py",
     "post hoc: why unscaled min-sum depends on the declared rate -- exact ties"),
    ("experiments/m0_prior_art/qldpc_minsum_rate_scan.py",
     "post hoc: every arm at every level across 16 declared rates exact arithmetic cannot tell apart"),
    ("experiments/m0_prior_art/qldpc_root_dual_classes.py",
     "how many trees are a single dual, re-decided by the exact checker"),
    # paper 1's re-measured certification cost and its preregistered held-out Willow sample
    ("experiments/m0_prior_art/certify_cost.py", "paper 1's certification cost, conditions recorded"),
    ("experiments/m0_prior_art/producer_identity_willow.py",
     "paper 1: every development and held-out chunk re-run, receipts compared byte for byte"),
    ("experiments/m0_prior_art/producer_identity_diff.py",
     "paper 1: the differing development chunks diffed shot by shot; today's development verdicts"),
    ("experiments/m0_prior_art/fresh_seed_sim.py", "paper 1: the simulation on fresh seeds (post hoc)"),
    ("experiments/ibm_live/ibm_epsilon_rerun.py",
     "paper 1: the IBM run re-certified under the epsilon-checker from its committed raw registers"),
    ("experiments/m0_prior_art/heldout_willow.py",
     "paper 1's held-out Willow sample (docs/PREREG_heldout_willow_2026-11-02.md)"),
    ("docs/PREREG_heldout_willow_2026-11-02.md", "the held-out sample's preregistration"),
    ("tools/checker_equivalence_gate.py",
     "every checker verdict AND reason, vs the pre-refactor bytes"),
    ("tools/producer_equivalence_gate.py",
     "producer yield, tiers and trees, vs the pre-refactor bytes"),
    ("experiments/m0_prior_art/streaming_deadline_sweep.py",
     "the DEFERRED tier: fires, resolves on `late`, loses nothing"),
    ("experiments/m0_prior_art/qldpc_symmetry_decoding.py",
     "A6: a conserved quantity as an extra check row, with a q=0.5 control"),
    ("experiments/m0_prior_art/qldpc_heavyhex_transpiler.py",
     "A5: [[144,12,12]] compiled onto heavy-hex, 144/144 blocks certified"),
    ("experiments/g3_abft/c2_mask_flip_scaling.py",
     "C2: the mask-flip multiplier GROWS with distance; L1 and L2 blind"),
    ("experiments/g3_abft/c2_l2prime_class_attestation.py",
     "L2': the missing layer -- 100% on mask flips, 0 false positives"),
    ("experiments/overhead/h4_hetero_ratio_verify.py",
     "H-4: 2.31x independently re-derived; the 3.747x trap refused"),
    ("experiments/overhead/b1_bias_ceiling.py",
     "B-1: ABSTAIN 6/6, robust across three ceiling definitions"),
    ("rtl/l2prime_detector.v",
     "L2' as synthesizable Verilog: AND-reduce + XOR tree, no state"),
    ("rtl/tb_l2prime.v",
     "testbench driven by PYTHON-produced expected bits, not recomputed"),
    ("experiments/overhead/l2prime_synth_gate.py",
     "L2' synthesized: real cell counts; LUT mapping NOT completed"),
    ("experiments/overhead/l2prime_rtl_gate.py",
     "L2' RTL vs Python, bit-exact per vector across 7 masks"),
    ("experiments/overhead/h5_gidney_crosscheck.py",
     "H-5: Gidney's 897,864-qubit layout reproduced exactly, line by line"),
    ("experiments/overhead/h1_sensitivity_exact.py",
     "H-1: 252 faults, 0 discrepancies against stim.Tableau AND a "
     "hand-written symplectic propagator"),
    ("experiments/m0_prior_art/qldpc_ablation_exact_gate.py",
     "exactification of both circuit ablations"),
    ("experiments/m0_prior_art/qldpc_paired_bd_closure.py",
     "branch-dual closure of the paired-stress solver tier"),
    ("tools/external/exact_lp_certificate_reference.py",
     "independently authored exact checker, pinned"),
    ("experiments/ibm_live/rep_code_cert.py", "the live-IBM lane"),
    ("src/oneq/api.py", "the product surface"),
    ("src/oneq/pq_receipt.py", "hybrid PQ envelopes"),
    ("experiments/m0_prior_art/streaming_gate.py", "streaming throughput gate"),
    ("tools/merge_million_shot.py", "sim-supplement merge + legitimacy"),
    ("experiments/m0_prior_art/decoder_mutation_gate.py", "the two numbers"),
    ("experiments/m0_prior_art/gap_distribution.py", "the gap distribution"),
    ("experiments/m0_prior_art/certification_semantics_audit.py",
     "what the pre-audit acceptance rule proved, in exact arithmetic"),
    ("experiments/m0_prior_art/google_generation_history.py",
     "the Willow generation table from the objects each generation wrote"),
    ("experiments/m0_prior_art/false_positive_hunt.py", "the adversarial hunt"),
    ("experiments/m0_prior_art/checker_scaling.py", "the O(nnz) measurement"),
    ("tools/zenodo_range.py", "read the 5.72 GB corpus without downloading it"),
    ("tools/fetch_google_corpus.py", "fetch the working set"),
    ("tools/make_cert_vectors.py", "regenerate the vectors"),
    ("src/oneq/dual_forgery.py", "the parity forgeries make_cert_vectors.py turns into vectors (R229)"),
    ("tools/doc_claims_gate.py", "the gate that polices the prose"),
    ("tools/mutation_gate.py", "the 53-defect wall"),
    ("tools/derive_claims.py", "derived headlines, regenerated not hand-summed"),
    ("tools/claims_provenance_gate.py", "every number must trace"),
    ("tools/gate_all.py", "one command, every gate"),
    # THE CHECKER WHOSE SOUNDNESS IS A THEOREM. `Cert.accept_sound` is proved
    # about the executable `accept` that `oneqcert-check` runs; with the
    # exported certificates below, a stranger re-judges every P2 exact
    # certificate without trusting any of this repository's Python.
    ("lean/OneqCert/README.md", "what is proved, and what is trusted"),
    ("lean/OneqCert/lakefile.toml", "Lean project"),
    ("lean/OneqCert/lean-toolchain", "pinned Lean version"),
    ("lean/OneqCert/lake-manifest.json", "pinned mathlib revision"),
    ("lean/OneqCert/OneqCert.lean", "library root"),
    ("lean/OneqCert/OneqCert/SafeBound.lean", "the odd-set facet inequality"),
    ("lean/OneqCert/OneqCert/WeakDuality.lean", "weak duality of the safe bound"),
    ("lean/OneqCert/OneqCert/Checker.lean", "the checker + accept_sound"),
    # imported by OneqCert.lean and AxiomCheck.lean since 79e25a06 (2026-10-23) and hashed by the Lean gate, but not
    # listed here until 2026-11-07: in the bundles minted after it (v2026.11.07 checked) the shipped lake project could
    # not build and check_lean_certificate.py --check stopped on the missing file (tests/test_bundle_carries_the_
    # papers_evidence.py now requires every file the Lean gate reads)
    ("lean/OneqCert/OneqCert/Statements.lean", "accept_sound's statement, pinned"),
    ("lean/OneqCert/OneqCert/AxiomCheck.lean", "prints every theorem's axioms"),
    ("lean/OneqCert/Main.lean", "oneqcert-check: the binary"),
    ("tools/check_lean_certificate.py", "build, run, and gate the Lean receipt"),
    ("tools/export_p2_certificates.py", "regenerate P2's certificates"),
    ("tools/pathrel.py", "path display that cannot raise"),
    # 2026-09-15: FOUR DOCUMENTED COMMANDS FAILED FROM THE PUBLIC TREE, each of them passing
    # here because this repository supplied what the bundle lacked
    # (tools/rehearse_public_release.py reconstructs the reader's situation and found them).
    # verify_release imports sign_release's release_core; REPRODUCE.md runs
    # pack_epsilon_receipts; certifying_decoder imports the ladder's exact, impossibility and
    # configuration-integrity tiers lazily, rep_code_cert the calibration model, and the
    # frozen qldpc_cert the frozen exact_lp. The import-closure guard in main() now refuses a
    # bundle whose shipped code reaches a module it does not carry.
    ("tools/sign_release.py",
     "release_core, which verify_release recomputes; signing needs a key this file does not hold"),
    # AND THE REGISTRY IT PINS AGAINST. With sign_release shipped, the rehearsal of the SIGNED tree
    # reached the next check and verify_release refused: "NO TRUSTED ISSUER REGISTRY". It reads
    # ISSUERS.json at the tree root and fails closed without it, so no reader could ever have
    # authenticated a release from this tree. Public keys only. The pin is only as independent as
    # the place a reader takes this file from: the published repository, not the archive it checks.
    ("ISSUERS.json", "the pinned issuer registry verify_release checks the signature against; public keys only"),
    ("tools/pack_epsilon_receipts.py", "paper 1's receipt archives: pinned hashes, then every member"),
    ("src/oneq/exact_decode.py", "the ladder's exact decoding tier"),
    ("src/oneq/repair_impossible.py", "the ladder's proofs that repair cannot succeed"),
    ("src/oneq/abft_l0.py", "the ladder's configuration-integrity check"),
    ("src/oneq/logical_cut.py", "which decoding-graph edges flip an observable"),
    ("src/oneq/device_dem.py", "the IBM lane's decoding model from device calibration"),
    ("src/oneq/exact_lp.py", "exact rational LP for cut-packing duals"),
    ("tools/frozen/exact_lp.py", "pre-refactor snapshot the frozen qldpc_cert imports"),
    ("conftest.py", "puts src/ on the path for the shipped tests"),
    # A released tree with no licence grants nobody anything, whatever the
    # README says (2026-09-14: the paper said MIT, the repository is Apache-2.0,
    # and the bundle carried no LICENSE at all).
    ("LICENSE", "Apache License 2.0"),
    ("experiments/m0_prior_art/stale_calibration_fault.py",
     "the upstream stale-calibration fault experiment"),
    # Added for the 2026-09-14 audit fixes: paper 1's checker, the rerun
    # that uses it, the tools that replay its receipts and score its tests.
    ("src/oneq/matching_cert_epsilon.py",
     "THE epsilon checker: exact rationals, declared eps_max, five outcome states"),
    ("experiments/m0_prior_art/epsilon_rerun.py",
     "paper 1's scale runs under the epsilon checker, every receipt kept"),
    ("tools/replay_epsilon_receipts.py",
     "replays every epsilon receipt with two checkers, plus boundary attacks"),
    ("experiments/m0_prior_art/detection_outcomes.py",
     "detection with base rates, outcome states and discordant logical cells"),
    ("experiments/m0_prior_art/willow_residual_shots.py",
     "the two Willow shots generation 6 left open, re-examined"),
    ("tools/mutation_score.py", "operator mutation score of a checker's tests"),
    ("verifier-js/verify-qldpc-flat.mjs",
     "qLDPC flat receipts: a checker in a second language, exact BigInt rationals"),
    ("tools/qldpc_js_agreement.py", "every flat receipt and planted defect through the JavaScript checker"),
    ("verifier-js/verify-qldpc-tree.mjs",
     "qLDPC branch-dual trees: a checker in a second language, written from the schema"),
    ("tools/qldpc_tree_js_agreement.py", "every stored tree and planted defect through the JavaScript tree checker"),
    ("tools/revision_snapshots.py", "R-02: source snapshots of the producing revisions, pinned by commit and sha256"),
    # THE INSTALL ROUTE. REPRODUCE.md told a reader of the bundle to `pip install -e ".[...]"`, and the bundle carries no
    # pyproject.toml, so the first command of the reproduction failed (2026-11-03). Pinned to the environment of record.
    ("requirements.txt", "the pinned Python dependencies: python -m pip install -r requirements.txt"),
    # ---- paper 1, revised against the 2026-11-03 external audits (docs/reviews/EXTERNAL1/2_2026-11-03.md)
    ("tools/fuzz_epsilon_checkers.py", "paper 1: the differential fuzzer over every epsilon-checker"),
    ("tools/write_cert_conformance.py", "writes evidence/cert_conformance.json from a run, never by hand"),
    ("verifier-js/verify-epsilon.mjs", "paper 1: the epsilon-checker in a second language, written from the spec"),
    ("conformance/epsilon_vectors.json", "paper 1: the epsilon-checker conformance vectors"),
    ("docs/PREREG_gap_distribution_2026-11-03.md", "paper 1: the gap run's preregistration"),
    ("docs/PREREG_heldout_d7_2026-11-03.md", "paper 1: the held-out d = 7 sample's preregistration"),
    ("docs/PREREG_heldout_d7_2026-11-03_DEVIATIONS.md", "paper 1: its deviations, each committed before it took effect"),
    ("experiments/dual_observable/certified_measured_receipts.py",
     "paper 1: the measured-model section's receipts and instances (evidence/certified_measured_receipts/)"),
    ("experiments/m0_prior_art/gap_distribution_original_rescore.py", "paper 1: the original gap run, rescored"),
    ("experiments/m0_prior_art/gap_distribution_rejudge.py", "paper 1: the preregistered gap run, rejudged"),
    ("experiments/m0_prior_art/pymatching_native_dual.py", "paper 1: R177 reproduced, PyMatching's own dual"),
    ("third_party/pymatching_return_duals.patch",
     "the patch that exports PyMatching's dual; evidence/pymatching_native_dual.json pins its sha256"),
    ("experiments/m0_prior_art/stale_false_alarm_shots.py", "paper 1: which shots the stale-calibration alarms fell on"),
    # named by paper 1's final-read revision (2026-11-03); a build refuses until the commit that writes them holds them
    ("experiments/m0_prior_art/gap_distribution_logical.py", "paper 1: the gap run's logical failures"),
    ("experiments/m0_prior_art/pymatching_native_dual_arms.py", "paper 1: PyMatching's own dual, per arm"),
    ("experiments/m0_prior_art/heldout_willow_d7.py", "paper 1: the held-out d = 7 sample, as preregistered"),
    ("experiments/m0_prior_art/heldout_willow_d7_interleaved.py", "paper 1: the held-out d = 7 arms run interleaved"),
    ("experiments/m0_prior_art/heldout_d7_partial.py", "paper 1: the held-out d = 7 report when its rule stops it"),
    ("tests/test_heldout_d7_partial.py", "paper 1: that report's tests"),
    ("experiments/m0_prior_art/heldout_d7_cost_calibration.py", "paper 1: the held-out d = 7 cost calibration"),
    ("experiments/m0_prior_art/heldout_d7_watchdog.py", "paper 1: the held-out d = 7 core-hour watchdog"),
    ("experiments/m0_prior_art/heldout_d7_babysitter.py", "paper 1: the held-out d = 7 run's babysitter (restarts a dead "
                                                          "driver; named by the deviations record)"),
    ("tests/test_heldout_d7_babysitter.py", "paper 1: the babysitter's tests"),
    # named by paper 1's revision against external audit 3 (docs/reviews/EXTERNAL3_2026-11-06.md)
    ("experiments/m0_prior_art/gap_distribution_2x2_posthoc.py",
     "paper 1: the post hoc 2x2 of Sec. VIII's BP-OSD arms (representation x min-sum scaling)"),
    ("tests/test_gap_distribution_2x2_posthoc.py", "paper 1: the 2x2's scoring, reproduction gate and artifact"),
    # named by paper 1's revision against external audit 4 (docs/reviews/EXTERNAL4_2026-11-07.md, items 1.1 and R4)
    ("experiments/m0_prior_art/decode_call_context.py",
     "paper 1 Sec. IX: the same PyMatching call timed in three contexts, why the two per-call figures differ"),
    ("tests/test_decode_call_context.py", "paper 1: that run's arithmetic and its record"),
    ("tests/test_certified_measured_receipts.py",
     "paper 1 Sec. XI: re-decides a sample of Sec. VI E's tracked receipts with both verifiers in the reader's tree"),
    ("tests/test_decoder_mutation_gate.py",
     "paper 1 Sec. VII A: every seeded fault is an edge set that differs from the decoder's correction (2026-11-07)"),
    # ---- paper 2, revised against the same audits
    ("docs/PREREG_qldpc_scaling_2026-11-03.md", "paper 2: the scaling comparison's preregistration"),
    ("docs/PREREG_qldpc_scaling_2026-11-03_DEVIATIONS.md", "paper 2: its deviations"),
    ("docs/QLDPC_RECEIPT_INPUT_DOMAIN.md", "paper 2: the one input domain every qLDPC checker enforces"),
    ("conformance/qldpc_input_domain_vectors.json", "paper 2: that domain's conformance vectors"),
    ("tests/test_qldpc_input_domain.py", "paper 2: every qLDPC checker held to the input domain"),
    ("tests/test_qldpc_adjudication_evidence.py", "paper 2: the adjudication evidence re-checked"),
    # read by that test's moved-pin check (every moved pin rests on a fresh re-verification record); the import-closure
    # guard cannot see a JSON read, and the composite-tree rehearsal found it missing (2026-11-05)
    ("docs/evidence_annotations.json", "the evidence annotations paper 2's moved pins are checked against"),
    ("experiments/m0_prior_art/qldpc_heldout_scaled_posthoc_all8.py", "paper 2: the scaled post-hoc on all eight"),
    ("experiments/m0_prior_art/qldpc_refsched_p3_equality.py", "paper 2: P2 and P3 on one mechanism set"),
    ("experiments/m0_prior_art/qldpc_tier_by_receipt.py", "paper 2: tiers by what each receipt proves"),
    ("experiments/m0_prior_art/qldpc_scale_probe_trees.py", "paper 2: quantization robustness of the trees"),
    ("experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py", "paper 2: the preregistered scaling campaign"),
    ("experiments/m0_prior_art/qldpc_receipt_reverification.py", "paper 2: every receipt re-verified"),
    # the campaign's archived inputs (evidence/qldpc_scaling_prereg/inputs/) and its costs (.../costs/) travel with the
    # campaign's directory in EVIDENCE below; these are what wrote, measure and replay them (external audit 3, 2026-11-06)
    ("experiments/m0_prior_art/qldpc_scaling_prereg_export_inputs.py",
     "paper 2: writes the campaign's archived inputs, the one place its model-building code is called"),
    ("experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py",
     "paper 2: the campaign re-run with per-stage timers; its production costs, and a byte-for-byte reproduction"),
    ("tools/replay_qldpc_scaling_campaign.py",
     "paper 2: replays the preregistered campaign from its archived inputs and records; no producer"),
    ("tests/test_replay_qldpc_scaling_campaign.py",
     "paper 2: that archive held to build(), and that replay held to its negative controls and to no producer"),
    ("tests/test_qldpc_scaling_prereg_timed_rerun.py",
     "paper 2: the cost records held to their own per-shot times and to the committed record files"),
    ("tests/test_qldpc_generator_rounding.py",
     "paper 2: the generator's one rounding rule, held on the numbers it prints (external audit 4, X6)"),
    # ---- paper 3
    ("docs/PAPER3_LITERATURE_SEARCH.md",
     "paper 3's literature searches: databases, dates, verbatim queries, what was not searched"),
]

# THE MANUSCRIPTS TRAVEL WITH THEIR RECEIPTS. Every paper is typeset from
# macros generated out of the evidence shipped here, so a reader who holds the
# bundle can regenerate every number and rebuild every PDF. Enumerated from the
# tree (sources only: no build output, no caches) so a new figure or table
# cannot be left out by forgetting to list it.
# Paper 4 (superactivation) is NOT a decoder paper and travels in its own bundle (SUPERACTIVATION_CONTENTS below, the
# owner's decision of 2026-11-03): its manuscript and generator are excluded here, and the shared files it inputs
# (owner.tex, release.tex, refs.bib, build.py) ship in both.
P4_PAPER = ("paper/superactivation/", "paper/tools/make_numbers_superactivation.py")
P4_SHARED_PAPER = ("paper/owner.tex", "paper/release.tex", "paper/refs.bib", "paper/tools/build.py")
_PAPER_SOURCES = sorted(
    p.relative_to(ROOT).as_posix()
    for p in (ROOT / "paper").rglob("*")
    if p.is_file() and "build" not in p.parts and "__pycache__" not in p.parts
)
CONTENTS += [(rel, "manuscript source, regenerable from the evidence")
             for rel in _PAPER_SOURCES if not rel.startswith(P4_PAPER)]

# PAPER 3 (circuit-level distances) TRAVELS WITH ITS LANE. Its numbers are generated from the lane's records, its lower
# bounds are certificates its own checker decides, and its checker's calculus is a Lean development, all of which lived
# only in the private development repository until 2026-11-02 (the paper-3 referee's finding: the paper pointed at "the
# repository" and no reader could reach it). Enumerated from the tree like the manuscripts, so a new record or proof
# cannot be left out by forgetting to list it. Excluded: the later decoder lanes R207-R223 that share the directory but
# are not paper 3's (their scripts and evidence), build output and caches.
_P3_ROOTS = ("experiments/circuit_distance_exact", "evidence/circuit_distance_exact", "lean/CertR198")
_P3_SKIP = re.compile(r"(^|/)r2(0[7-9]|1\d|2\d)_|^evidence/circuit_distance_exact/r2(0[7-9]|1\d|2\d)|"
                      r"(^|/)(\.lake|__pycache__|build)(/|$)|\.(exe|o|pyc)$")
_OWN_CHECKOUT: list[bool] = []


def _own_checkout() -> bool:
    """Is ROOT the top of its own git work tree?

    A bundle extracted INSIDE another checkout (a trial mint under .build/, the tracked release/ copy) answers git
    commands with the enclosing repository: `git ls-files -- experiments/superactivation` from there lists nothing --
    the enclosing tree ignores .build/ -- and exits 0, so the lane enumerations below came back empty and paper 4's
    generator, which reads them, would have said no release carries its lane (found 2026-11-03 preparing the next
    mint). Only a checkout whose top is ROOT may answer for ROOT's files."""
    if not _OWN_CHECKOUT:
        try:
            top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout.strip()
            _OWN_CHECKOUT.append(bool(top) and pathlib.Path(top).resolve() == ROOT.resolve())
        except (OSError, subprocess.CalledProcessError):
            _OWN_CHECKOUT.append(False)
    return _OWN_CHECKOUT[0]


def _tree_files(*specs: str) -> list[str]:
    """The files the commit holds under each directory in `specs`, or matching each glob (`*` within one path
    segment). A directory walk also finds git-ignored run output (the campaign's out/ work files, CRLF scratch models
    that shipped records pin by digest), which no clone has; the first bundle with paper 3's lane shipped them and its
    normalisation then broke those pins (2026-11-02). Outside its own git checkout -- a sterile copy of the bundle,
    or one extracted inside another repository -- every file present is the tree's."""
    globbed = [s for s in specs if any(c in s for c in "*?[")]
    if _own_checkout():
        pathspecs = [f":(glob){s}" if s in globbed else s for s in specs]
        out = subprocess.run(["git", "ls-files", "-z", "--", *pathspecs], cwd=ROOT, capture_output=True,
                             check=True).stdout
        return sorted(x for x in out.decode("utf-8").split("\0") if x)
    found = set()
    for s in specs:
        paths = ROOT.glob(s) if s in globbed else (ROOT / s).rglob("*") if (ROOT / s).is_dir() else [ROOT / s]
        found |= {p.relative_to(ROOT).as_posix() for p in paths if p.is_file()}
    return sorted(found)


_P3_SOURCES = sorted(
    rel for top in _P3_ROOTS for rel in _tree_files(top)
    if (ROOT / rel).is_file() and not _P3_SKIP.search(rel)
) + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("test_r198_*.py")) + [
    "docs/RESEARCH_R205_CDP5.md",       # make_numbers_distance.py reads its prior-art sentence
    "docs/reviews/PAPER3_REFEREE_2026-10-29.md",
    "tools/roster_digest.py",           # lean_receipt.py's source digest
]
CONTENTS += [(rel, "paper 3: lane code, records, certificates or the Lean development")
             for rel in _P3_SOURCES if rel not in {c[0] for c in CONTENTS}]

# THE REVIEW ROUND THE REVISED MANUSCRIPTS ANSWER: the two external audits of 2026-11-03, every referee report of the
# 2026-11-02/03 round and the brief the JavaScript epsilon-checker was written from. A revision whose reviews the reader
# cannot see asks to be trusted on what it fixed. Paper 4's referee reports travel with paper 4.
_REVIEWS = _tree_files("docs/reviews/*_2026-11-0[23].md")
CONTENTS += [(rel, "the 2026-11-02/03 review round the revised manuscripts answer")
             for rel in _REVIEWS if rel not in {c[0] for c in CONTENTS} and "/PAPER4_" not in rel]
# the third external audit (of v2026.11.05, all four manuscripts), which the next revision answers; it ships in both
# bundles, since it reviews paper 4 too
CONTENTS += [("docs/reviews/EXTERNAL3_2026-11-06.md", "the external audit of v2026.11.05 the revised manuscripts answer")]
# and the fourth (also of v2026.11.05, all four manuscripts), which v2026.11.07.1 answers; both bundles, as the third
CONTENTS += [("docs/reviews/EXTERNAL4_2026-11-07.md", "the external audit of v2026.11.05 the revised manuscripts answer")]

# =====================================================================================================================
# THE SECOND BUNDLE: PAPER 4 (superactivation in two channel uses), a separate quantum-capacity line.
#
# It ships in the same release tag as the decoder bundle, signed and stamped on its own (owner's decision, 2026-11-03):
# the decoder bundle stays about decoders and well under GitHub's 104,857,600-byte file limit, which paper 4's lane
# (972 tracked files, 664 MB raw) would have brought within 4 MB of. Enumerated from the commit like paper 3's lane
# (never a directory walk), so a new certificate, record or test cannot be left out by forgetting to list it; paper 4's
# generator reads this list to decide which release carries the lane (make_numbers_superactivation.py, `lane_release`).
# What else it carries is what its guards demand: the import closure, the generator's inputs, the scripts its documents
# name, and the means to verify a release (verify_release, its signing core, the pinned issuers).
P4_LANE_ROOTS = ("experiments/superactivation", "evidence/superactivation")
_P4_HW = ("src/oneq/hw/__init__.py", "src/oneq/hw/campaign.py", "src/oneq/hw/contract_synth.py", "src/oneq/hw/cz_channel.py", "src/oneq/hw/device.py", "src/oneq/hw/dynamic_exact.py", "src/oneq/hw/dynamic_timing.py", "src/oneq/hw/engine.py", "src/oneq/hw/frame.py", "src/oneq/hw/freeze.py", "src/oneq/hw/iq.py", "src/oneq/hw/layout.py", "src/oneq/hw/noise.py", "src/oneq/hw/pauli_frame.py", "src/oneq/hw/premise.py", "src/oneq/hw/randomization.py", "src/oneq/hw/sensitivity.py", "src/oneq/hw/snapshot.py", "src/oneq/hw/stats.py", "src/oneq/hw/submit.py", "src/oneq/hw/timing.py", "src/oneq/hw/twirl_mode.py", "src/oneq/hw/value.py", "src/oneq/hw/window.py")
_P4_SKIP = re.compile(r"(^|/)(__pycache__|build)(/|$)|\.(exe|o|pyc)$")
_P4_SOURCES = [rel for rel in _tree_files(*P4_LANE_ROOTS, "src/oneq/superactivation*.py", "tests/test_superactivation*.py",
                                          "tests/test_r142_*.py", "tests/test_nuse_certificate.py",
                                          # the modules of the hardware engine the lane imports (the import-closure
                                          # guard names any it reaches and this list lacks), and two tools it imports
                                          *_P4_HW, "tools/analyze_r158_independent.py",
                                          "tools/calibration_window.py", "docs/PAPER4_LITERATURE_SEARCH.md", "docs/reviews/PAPER4_REFEREE*_2026-11-03.md",
                                          "docs/reviews/EXTERNAL3_2026-11-06.md", "docs/reviews/EXTERNAL4_2026-11-07.md")
               if not _P4_SKIP.search(rel)]
SUPERACTIVATION_CONTENTS = [(rel, "paper 4: lane code, records, certificates or tests") for rel in _P4_SOURCES]
SUPERACTIVATION_CONTENTS += [(rel, "paper 4's manuscript source, regenerable from the evidence")
                             for rel in _PAPER_SOURCES if rel.startswith(P4_PAPER) or rel in P4_SHARED_PAPER]
SUPERACTIVATION_CONTENTS += [
    ("tools/verify_r142_rational_witness.py",
     "the producer-free exact-rational checker the generator re-runs on the d = 20, 21, 22 bounds"),
    # loaded by FILE PATH (importlib), which the import-closure guard cannot see; the bundle's rehearsal found it
    # missing when tests/test_r142_physical_decoder_witness.py failed to collect in a sterile copy (2026-11-03)
    ("tools/verify_r142_two_use_independent.py", "the independent two-use checker test_r142_physical_decoder_witness loads"),
    # the generator imports this builder to read what this bundle declares; with it, what it imports
    ("tools/build_certificate_bundle.py", "the builder of both bundles: what each carries, and every guard on it"),
    ("tools/mint_release.py", "the one file list each manifest, archive and tree pin derive from"),
    ("tools/revision_snapshots.py", "the import resolver the builder's closure guard uses"),
    # step 0 of a reader's audit: is this the signed release
    ("tools/verify_release.py", "verifies this bundle's signature, statement and timestamps"),
    ("tools/sign_release.py", "release_core, which verify_release recomputes; signing needs a key this file does not hold"),
    ("ISSUERS.json", "the pinned issuer registry verify_release checks the signature against; public keys only"),
    # what verify_release, sign_release and the builder's signing path import (the import-closure guard named them)
    ("src/oneq/passport.py", "canonical bytes + Ed25519: the signed core's digest and signature"),
    ("src/oneq/model_audit.py", "re-checked by the passport before it signs"),
    ("src/oneq/cert_format.py", "sign_document, the builder's signing path"),
    ("src/oneq/matching_cert.py", "imported by cert_format"),
    ("tools/roster_digest.py", "the source digest the lane's Lean receipts (r203_lean_receipt, r203_float_boundary) pin"),
    ("LICENSE", "Apache License 2.0"),
    ("conftest.py", "puts src/ on the path for the shipped tests"),
]

#: The bundles one release carries. Each has its own SHA256SUMS, AUDIT.md count, in-toto statement, README and
#: requirements.txt; `top` maps a top-level file of the bundle to the repository file it is copied from.
BUNDLES = {
    "certifying-decoder": {
        "dir": "oneq-certifying-decoder",
        "top": {"AUDIT.md": "AUDIT.md", "REPRODUCE.md": "REPRODUCE.md", "REFREEZE_PROTOCOL.md": "docs/REFREEZE_PROTOCOL.md"},
    },
    "superactivation": {
        "dir": "oneq-superactivation",
        "top": {"AUDIT.md": "packaging/superactivation/AUDIT.md",
                "requirements.txt": "packaging/superactivation/requirements.txt"},
    },
}

EVIDENCE = [
    # THE INTEGER PROOFS FOR THE d=8 CLOSURES. A portable proof
    # object a stranger never receives is not portable, and these
    # are the only machine-checkable objects for those bounds --
    # the LP dual cannot certify them, because the gain OISC
    # exploits IS the integrality gap.
    "evidence/phase7/branch_cert_phase2_64.json",
    "evidence/phase7/branch_cert_phase2_65.json",
    # DEM-CERT. One 32-byte root per circuit certifying every elementary
    # fault's signature -- and the leaf vector the JavaScript checker
    # reads, because a canonical serialization a second language has
    # never been handed is a canonical serialization of one language.
    "evidence/dem_certificates.json",
    "conformance/dem_leaves_d3.json",
    # The admission LEVEL, shipped next to the certificate, because
    # a root without a grade invites the reader to supply their own.
    "evidence/dem_admission.json",
    # THE NULL-FAULT CATALOGUE. Faults that move no detector and no
    # observable: provably invisible to any decoder reading this
    # syndrome, computed as a byproduct of a pass that had to run
    # anyway, and -- per the blueprint -- never published anywhere.
    "evidence/null_fault_catalogue_d3.json",
    "evidence/null_fault_catalogue_d5.json",
    "evidence/null_fault_catalogue_d7.json",
    "evidence/null_fault_catalogue_d9.json",
    "evidence/null_fault_catalogue_d11.json",
    "evidence/null_fault_catalogue_d15.json",
    "evidence/null_fault_catalogue_d21.json",
    # 51,797 provably invisible faults at d=25 -- 5.04% of |F|,
    # against the 5.0% the blueprint records.
    "evidence/null_fault_catalogue_d25.json",
    # The evidence algebra's falsification and its PAIRED negative
    # control. Shipping the positive alone would be shipping the
    # half that cannot fail.
    "evidence/gt_a0_evidence_algebra.json",
    "evidence/gt_a0_negative_control.json",
    # L0 configuration integrity, with the vacuity arm alongside it:
    # 96/96 mask corruptions caught with the dedicated replica and
    # 0/96 without, on the same graph.
    "evidence/l0_config_integrity.json",
    # The capability manifest, shipped precisely because it ADMITS
    # NOTHING: it is the record of what the client library will express,
    # which is not evidence about any device, and a reader who cannot see
    # that distinction in the bundle will assume the other one.
    "evidence/capability_manifest_sdk.json",
    "evidence/million_shot_gate_d5.json",
    "evidence/million_shot_gate_d5_supplement.json",
    "evidence/million_shot_gate_d5_supplement2.json",
    "evidence/million_shot_gate_d5_supplement3.json",
    "evidence/million_shot_gate_d5_merged.json",
    "evidence/cert_conformance.json",
    "evidence/gap_distribution.json",
    "evidence/gap_distribution_exact_rule.json",
    # Quoted by paper 1 (Sec. VII B and VI D) and read by make_numbers.py;
    # absent until 2026-09-14, so a reader holding only the bundle could not
    # regenerate the manuscript. tests/test_bundle_carries_the_papers_evidence.py
    # now holds every artifact the generators read.
    "evidence/stale_calibration_d3.json",
    "evidence/stale_calibration_d5.json",
    "evidence/stale_calibration_d7.json",
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.certified.json",
    # the same run re-certified under the epsilon-checker from its raw registers, every receipt kept (2026-11-02)
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.shots.json",
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.epsilon.json",
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.epsilon_receipts.jsonl.gz",
    "evidence/certification_semantics_audit.json",
    "evidence/google_generation_history.json",
    "evidence/decoder_mutation_gate_d3.json",
    "evidence/decoder_mutation_gate_d5.json",
    "evidence/decoder_mutation_gate_d7.json",
    "evidence/seeded_fault_corpus_d3.json",
    "evidence/seeded_fault_corpus_d5.json",
    "evidence/seeded_fault_corpus_d7.json",
    "evidence/false_positive_hunt_d3.json",
    "evidence/growth_vs_lp.json",
    "evidence/mutation_gate.json",
    "evidence/decode_vs_certify.json",
    "evidence/derived_claims.json",
    "evidence/lean_safe_bound.json",
    "evidence/lean_safe_bound_mutants.json",    # the Lean gate's mutant drill, which --check reads (2026-11-07)
    "evidence/p2_certificates.jsonl.gz",
    "evidence/p2_certificates_export.json",
    # read by paper/tools/make_numbers*.py since the 2026-09-14 revisions
    "evidence/detection_outcomes/d3.json",
    "evidence/detection_outcomes/d5.json",
    "evidence/detection_outcomes/d7.json",
    "evidence/willow_residual_shots.json",
    "evidence/mutation_score/matching_cert_epsilon.json",
    "evidence/mutation_score/matching_cert_epsilon.equivalents.json",
    "evidence/mutation_score/qldpc_check.json",
    "evidence/mutation_score/qldpc_check.equivalents.json",
    "evidence/qldpc_dev_refsched_trees.json",
    "evidence/qldpc_strong_baseline_v2.json",
    "evidence/qldpc_strong_baseline_v3.json",
    "evidence/strong_baseline_v3_replay.json",
    "evidence/bposd_circuit_outcome.json",
    # paper 2's post-hoc analyses (2026-11-02 review round); the scaled arms' records travel through
    # _strong_baseline_scaled_payloads, the held-out re-analysis re-reads the shipped per-shot records
    "evidence/qldpc_heldout_scaled_posthoc.json",
    "evidence/qldpc_bposd_scaling_sensitivity.json",
    "evidence/qldpc_minsum_rate_invariance.json",
    "evidence/qldpc_minsum_rate_scan.json",
    "evidence/qldpc_root_dual_classes.json",
    # paper 1's certification cost, re-measured with its conditions recorded
    "evidence/certify_cost_d5_r25.json",
    # paper 1's preregistered held-out Willow sample, pooled; each configuration's summary, chunk index, instance and
    # complete replay travel through _epsilon_rerun_payloads, its receipts as release assets
    "evidence/epsilon_rerun/heldout_willow_r50.json",
    "evidence/producer_identity_willow.json",
    "evidence/producer_identity_willow_diff.json",
    # paper 1's measured-model section: the generator reads this artifact's `totals`, so a reader
    # who cannot open it cannot re-derive those numbers
    "evidence/certified_measured_lane.json",
    "evidence/freeze_identity.json",
    "evidence/receipt_manifest.json",
    "evidence/qldpc_js_agreement.json",
    "evidence/qldpc_tree_js_agreement.json",
    "evidence/revision_snapshots.json",
    # ---- paper 1, revised against the 2026-11-03 external audits: every artifact its generator and text read
    "evidence/epsilon_checker_fuzz.json",
    "evidence/epsilon_checker_fuzz_after_fix.json",
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.epsilon_2026-11-03.json",
    "evidence/ibm_live/d9nt2imij12s73fu9ihg.epsilon_receipts_2026-11-03.jsonl.gz",
    "evidence/producer_identity_willow_2026-11-03.json",
    "evidence/producer_identity_willow_diff_2026-11-03.json",
    "evidence/pymatching_native_dual.json",
    "evidence/pymatching_native_dual_arms.json",     # written by pymatching_native_dual_arms.py (P1's final read)
    # the same script's 2026-11-03 run, before the exact check was timed (external audit 3), kept byte for byte: the
    # generator holds the rerun equal to it in every field that is not a time and typesets its ratio beside
    "evidence/pymatching_native_dual_arms_run_2026-11-03.json",
    # the post hoc 2x2 of Sec. VIII's BP-OSD arms (external audit 3), named here as well as by the gap_distribution_v4
    # tree below, so the bundle carries it before and after the commit that adds it
    "evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/summary.json",
    "evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/per_shot_d3.jsonl.gz",
    "evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/per_shot_d5.jsonl.gz",
    "evidence/gap_distribution_v4/posthoc_2x2_2026-11-06/per_shot_d7.jsonl.gz",
    "evidence/stale_false_alarm_shots.json",
    "evidence/google_corpus_manifest_heldout_d7.json",
    # NAMED BY PAPER 1'S PREREGISTRATIONS AND SEC. VI E, AND NOT SHIPPED until external audit 4 (R6, 2026-11-07): both
    # Willow preregistrations name the corpus manifests (the first is byte-identical to the data/google_corpus/
    # MANIFEST.json this builder already ships under the other name), and the measured-model lane certifies these six
    # raw IBM records, which its record and its preregistration name
    "evidence/google_corpus_manifest.json",
    "evidence/google_corpus_manifest_heldout_r50.json",
    "evidence/rep_d3_r13_kingston.json",
    "evidence/rep_d3_r40_kingston.json",
    "evidence/hardware_noise_floor_fez_d3_20k.json",
    "evidence/rep_d3_r40_fez.json",
    "evidence/rep_d5_r13_fez.json",
    "evidence/rep_d3_r13_fez_nested.json",
    # paper 1 Sec. IX: the PyMatching call timed in three contexts on the cost run's shots (external audit 4, 1.1)
    "evidence/decode_call_context_d5_r25.json",
    # ---- paper 2, revised against the same audits
    "evidence/qldpc_heldout_scaled_posthoc_all8.json",
    "evidence/qldpc_refsched_p3_equality.json",
    "evidence/qldpc_tier_by_receipt.json",
    "evidence/qldpc_scale_robustness_trees.json",
    "evidence/qldpc_receipt_reverification_2026-11-03.json",
    "evidence/qldpc_receipt_reverification_2026-11-03b.json",
]
# Whole runs, enumerated from the commit so a record added to one cannot be left behind: paper 1's measured-model
# receipts, its preregistered gap run (per-shot records the figure reads), the 2026-11-03 replays of every epsilon
# configuration, and paper 2's preregistered scaling campaign.
EVIDENCE += [rel for rel in _tree_files("evidence/certified_measured_receipts", "evidence/gap_distribution_v4",
                                        "evidence/epsilon_rerun/replay_2026-11-03", "evidence/qldpc_scaling_prereg",
                                        # the held-out d = 7 sample's stop records, logs and dispatch markers, and
                                        # the partial-plan replay of every chunk it completed (2026-11-05)
                                        "evidence/epsilon_rerun/heldout_d7_stopped_2026-11-04",
                                        "evidence/epsilon_rerun/replay_heldout_d7_*.json")
             if rel not in EVIDENCE]

# Present only if the run that makes them has finished. Recorded as MISSING
# rather than skipped, so a reader can tell an absent artifact from one nobody
# mentioned.
OPTIONAL = [
    # paper 1's held-out d = 7 sample: the complete run, or the report its preregistered core-hour rule writes when it
    # stops the run; until one exists the generator holds HoDSevenState pending and the release build refuses
    "evidence/epsilon_rerun/heldout_d7.json",
    "evidence/epsilon_rerun/heldout_d7_partial.json",
    "evidence/heldout_public_reproduction.json",
    "evidence/google_corpus_gate.json",
    "evidence/google_corpus_gate_gen4.json",
    "evidence/corpus_escalated_d7X30.json",
    "evidence/corpus_escalated_d7Z30.json",
    "evidence/qldpc_bb_gate.json",
    "evidence/qldpc_bb_gate_p01.json",
    "evidence/qldpc_bb_gate_p03.json",
    "evidence/qldpc_bb_gate_p05_rpc.json",
    "evidence/qldpc_bb_gate_p02_hetero.json",
    "evidence/qldpc_bb_phenom.json",
    "evidence/qldpc_bb_phenom_p02.json",
    "evidence/qldpc_circuit_gate.json",
    "evidence/qldpc_circuit_gate_lazy.json",
    "evidence/qldpc_bb_circuit_gate.json",
    "evidence/qldpc_xsector_gate.json",
    "evidence/qldpc_ensembles.json",
    "evidence/ibm_live_certified.json",
    "evidence/qldpc_claims_backing.json",
    "evidence/qldpc_exact_cert.json",
    "evidence/qldpc_paired_stress.json",
    "evidence/qldpc_exact_all.json",
    "evidence/qldpc_scale_robustness.json",
    "evidence/environment.json",
    "evidence/qldpc_circuit_binding.json",
    "evidence/claims_registry.json",
    "evidence/qldpc_bb_refsched_gate.json",
    "evidence/qldpc_bnb_residue.json",
    "evidence/qldpc_frame_demo.json",
    "evidence/qldpc_heldout.json",
    "evidence/heldout_qldpc_bb_refsched_gate.json",
    "evidence/qldpc_heldout_addendum.json",
    # the 2,002 held-out FLAT certificates themselves (2026-09-12): before
    # this the bundle carried their hashes only, so a stranger could re-check
    # the 17 trees and merely REPRODUCE the flat receipts by running the
    # producer. recheck_all.py re-checks every one from this payload.
    "evidence/qldpc_heldout_flat_certs.json",
    "evidence/refsched_layer_manifest.json",
    "evidence/qldpc_refsched_fault_xval.json",
    "evidence/qldpc_circuit_distance.json",
    "evidence/qldpc_bb_family.json",
    "evidence/qldpc_bposd_sensitivity.json",
    "evidence/qldpc_tree_corpus.json",
    "evidence/qldpc_family_circuit.json",
    "evidence/qldpc_frame_campaign.json",
    "evidence/b1_tcert.json",
    "evidence/qldpc_strong_baseline.json",
    "evidence/qldpc_dem_mechanism_audit.json",
    "evidence/checker_equivalence.json",
    "evidence/producer_equivalence.json",
    "evidence/streaming_deadline_sweep.json",
    "evidence/qldpc_symmetry_decoding.json",
    "evidence/qldpc_heavyhex_transpiler.json",
    "docs/AUDIT_DEBT.md",
    "evidence/c2_mask_flip_scaling.json",
    "evidence/c2_l2prime_attestation.json",
    "evidence/h4_hetero_ratio_verify.json",
    "evidence/b1_bias_ceiling.json",
    "evidence/h1_sensitivity_exact.json",
    "evidence/h5_gidney_crosscheck.json",
    "evidence/l2prime_rtl.json",
    "evidence/l2prime_synth.json",
    "evidence/qldpc_bb_phenom_r8.json",
    "evidence/qldpc_bb_phenom_r12.json",
    "evidence/qldpc_bb_phenom_r16.json",
    "evidence/qldpc_ablation_exact.json",
    "evidence/qldpc_paired_bd_closure.json",
    "evidence/corpus_gen6_d7X30.json",
    "evidence/corpus_gen6_d7Z30.json",
    "evidence/ibm_live_job.json",
    "evidence/streaming_gate_d3_at_q4_5_X_r13.json",
    "evidence/streaming_gate_d5_at_q8_7_X_r30.json",
    "evidence/exact_oracle_audit_d3.json",
    "evidence/exact_oracle_audit_d5.json",
    "evidence/exact_oracle_audit_d7.json",
    "evidence/checker_scaling.json",
    "data/google_corpus/MANIFEST.json",
]

README = """# Certifying QEC decoders — re-runnable bundle

Everything here can be checked without trusting the people who made it.

This release accompanies three manuscripts by Josh Philbrick (Coherence Energy
Labs), whose LaTeX sources are in `paper/`:

* *Certifying quantum error-correction decoders: independently checkable
  near-optimality receipts at scale* — `paper/certifying-decoders/`
* *Exact, independently checkable optimality receipts for BP-OSD on the
  [[144,12,12]] bivariate-bicycle code* — `paper/qldpc-certificates/`
* *Machine-checkable lower-bound certificates for the circuit-level distances
  of bivariate-bicycle syndrome-extraction circuits* — `paper/circuit-distance/`,
  with its lane in `experiments/circuit_distance_exact/`, its records and
  certificates in `evidence/circuit_distance_exact/`, and the Lean development
  of its checker's calculus in `lean/CertR198/`

The same release tag carries a second, separately signed bundle,
`oneq-superactivation`: a manuscript on superactivation of quantum capacity,
a different line of work that shares no evidence with this one. In the public
repository it is the subdirectory `superactivation/`, a complete tree of its
own with its own `SHA256SUMS`, `AUDIT.md` and `README.md`; its commands run
from inside it (`cd superactivation`), and nothing in this tree's `SHA256SUMS`,
tests or commands reaches into it.

Install the pinned Python dependencies first:

    python -m pip install -r requirements.txt

Every number in all three papers is a macro generated from the evidence in this
tree. Regenerate them and rebuild the PDFs (a TeX distribution with REVTeX 4.2
is also needed):

    python paper/tools/make_numbers.py
    python paper/tools/make_numbers_qldpc.py
    python paper/tools/make_numbers_distance.py
    python paper/tools/make_figures.py
    python paper/tools/build.py certifying-decoders
    python paper/tools/build.py qldpc-certificates
    python paper/tools/build.py circuit-distance

Paper 3's certificates re-check with its own checker (a C compiler is needed):
`python -m pytest tests/test_r198_ibm_recert.py tests/test_r198_paper3_review.py`
re-verifies shipped proofs and refuses a corrupted one. The 30 certificates of the
three-colour and large morphing circuits, too large for git, are release assets
(`oneq-recert-proofs-3c.tar.gz`, `oneq-recert-proofs-morph144.tar.gz`, listed in
`evidence/circuit_distance_exact/recert_20261102/release_assets.json`): download
both into DIR, then
`python experiments/circuit_distance_exact/recert_assets.py verify --archives DIR --extract`
checks every archive and proof against that record and the manifest and puts the
proofs in place, and `... recert_assets.py check --threads 8` runs the committed
checker on them (minutes to over an hour per proof: the manifest's `checker_s`,
per cell). Each also regenerates bit for
bit from the committed prover (`experiments/circuit_distance_exact/recert_campaign.py`) under
the hash in `evidence/circuit_distance_exact/recert_20261102/manifest.json`,
which then pins the proof the checker decides. Where that proof differs from the
original record, the prover of the record's day, compiled from git, reproduces
the recorded proof (`recert_old_prover.py`, `records/recert_old_prover.jsonl`).

## What runs from this tree, checked from a clean copy

    node verifier-js/verify-certificate.mjs --conformance conformance/cert_vectors.json
    python tools/external/exact_lp_certificate_reference.py --self-test
    python tools/recheck_all.py
    python tools/replay_epsilon_receipts.py --jobs 8
    python -m pytest tests/ -q

`recheck_all.py` is a complete replay: it checks every one of the 2,042
receipts named by the pinned `evidence/receipt_manifest.json` (2,002 flat
certificates and 40 branch-dual trees) with two separately written
checkers, against the persisted instances and integer weight vectors, and
FAILS if any receipt or bound file is missing, duplicated, unlisted or
altered. It uses no solver, no producer and no rebuilt input. `replay_epsilon_receipts.py`
does the same for paper 1's rerun, with a second verifier in scaled-integer
arithmetic and attacks at the acceptance boundary; it needs the receipt archive
extracted under `data/epsilon_receipts/`. Re-running the original experiments needs
the code of the revisions that produced them: the release assets
`oneq-source-<commit>.tar` carry it (the producing scripts and every module they
import), each named with its sha256 in `evidence/revision_snapshots.json`; check a
download with `python tools/revision_snapshots.py --verify --no-git --out <dir>`.
What does NOT run from this tree: the full development test suite, the mutation
gate and `tools/gate_all.py`, and the commands in `AUDIT.md` and `REPRODUCE.md`
that read git history, which need the full development repository. `tools/verify_release.py`
checks a signed release against the issuer pinned in `ISSUERS.json`: put the three
`oneq-certifying-decoder` release assets in `release/` first.

## The one-minute check

    node verifier-js/verify-certificate.mjs --conformance conformance/cert_vectors.json

@N_VECTORS@ vectors must agree. **@N_REFUSALS@ of them are refusals** — documents that a sound
checker must reject. A suite where everything verifies would not notice a
checker that accepts everything, which is the failure that matters, because
such a checker is worthless and looks perfect.

Then corrupt something and watch it fail. Change one hex float in any `dual`
entry of an accepting vector; the count must drop. A harness that cannot fail
is not evidence.

## What each number means, and does not

| file | claim | what it does NOT say |
|---|---|---|
| `million_shot_gate_d5_merged.json` | 10⁶ shots merged from four disjoint-seed runs, 997,594 (99.75%, rounded down) accepted under the pre-audit tolerance rule, 6,000,002 forgeries, 0 accepted | nothing about hardware noise; and "certified" here means the pre-audit acceptance rule (float gap ≤ 1e-6) — see the next row |
| `certification_semantics_audit.json` | what that rule proved: the producing code of each headline artifact, re-run on seeded samples, every accepted certificate analysed in exact rational arithmetic — all admissible, largest proven slack far below 1e-6 | not the per-shot slack of the full runs, whose certificates were not stored; and not EXACT optimality, which today's stricter checker measures separately |
| `google_generation_history.json` | the Willow certified rate by producer generation, read from the objects each generation wrote | generations 2 and 3 ran partial coverage; their aggregates weight configurations unequally |
| `gap_distribution_exact_rule.json` | the gap experiment re-run under the exact checker, with per-shot gaps | a different certified subset from `gap_distribution.json`; both are shipped so the difference is visible |
| `cert_conformance.json` | two independent checkers agree @N_VECTORS@/@N_VECTORS@ | agreement is not correctness — it is the absence of a *shared* misunderstanding |
| `decoder_mutation_gate_d*.json` | syndrome consistency catches 0 of the syndrome-consistent fault class; the certificate catches all of it | output mutation models wrong answers, not crashes, hangs or memory errors |
| `epsilon_rerun/<config>/summary.json` | the scale runs repeated under the epsilon checker (exact rationals, eps_max = 1e-6) with every receipt kept: proven-within-epsilon, exact and not-proven counts, largest proven slack, forgeries | the receipts are a separate archive; `replay_<config>.json` records a complete two-verifier replay of them |
| `detection_outcomes/d*.json` | seeded faults and a stale calibration judged against the exact optimum: outcome states, false-alarm base rates, discordant logical cells | the stale-calibration result is objective mismatch against a supplied reference model; it cannot say which model describes a device |
| `mutation_score/*.json` | operator mutation scores of both checkers' tests, every surviving mutant declared equivalent with its reason | a score measures the tests against mechanical defects, not the absence of defects |
| `willow_residual_shots.json` | the two Willow shots generation 6 left open: PyMatching's correction is exactly optimal and the exact tier stopped at its round cap | two shots, not a rate |
| `gap_distribution.json` | BP-OSD's per-shot optimality gap against a checkable bound | a gap is **not** a defect: BP-OSD maximises likelihood over the hypergraph, not matching weight |
| `seeded_fault_corpus_d*.json` | the faults themselves | released so you can score *your* detector; the two numbers are only comparable if the corpus is shared |
| `lean_safe_bound.json` | a checker PROVED sound in Lean 4 (`Cert.accept_sound`, standard axioms only) accepted all 1,851 P2 exact certificates in `p2_certificates.jsonl.gz` and refused all 11,106 forgeries of them | the JSON reader and the Lean compiler are trusted, not proved; see `lean/OneqCert/README.md` |

## The claim in one paragraph

Minimum-weight decoding is a minimum-weight T-join. Its dual is a cut packing:
vertex sets with non-negative weights, no edge over-subscribed. Any feasible
packing lower-bounds the optimal correction **by weak duality, regardless of
how it was produced** — so a checker need not trust or even know the producer.
When the packing's value equals the correction's weight, both are optimal, and
verifying that takes time proportional to the packing's size and the edges it
touches, with no solver.

## Two things the authors got wrong, kept here on purpose

An "attack" was accepted 16 times and looked program-ending. The 16 were
**equal-weight alternative optima** — minimum-weight corrections are not
unique, and accepting a different optimum is correct. Later a second attack was
accepted on 4 of 147 shots; an audit outside the checker's own code path found
zero negative duals, zero violated constraints and objective equal to primal
within 9e-16. It had degenerated into a **valid certificate**.

Both times the checker was right and the attack was wrong. That is why every
attack in `forgeries.py` now carries the property making its claim false, and
why that property is re-derived independently before any acceptance is counted.
Without that audit, "0 forgeries accepted" could mean the battery had quietly
stopped attacking.
"""


SUPERACTIVATION_README = """# Superactivation in two channel uses — re-runnable bundle

This is a SEPARATE line of work from the decoder certificates released beside
it: quantum capacity, not error-correction decoding. It accompanies the
manuscript by Josh Philbrick (Coherence Energy Labs), whose LaTeX source is in
`paper/superactivation/`:

* *Superactivation in two channel uses for private Horodecki channels with
  large shields: exact certificates*

The same release tag carries the decoder bundle (`oneq-certifying-decoder`,
three manuscripts on certifying decoders). Each bundle is signed and
time-stamped on its own, and neither needs the other. In the public repository
the decoder bundle is the root of the tree and this bundle is the subdirectory
`superactivation/`; every command below runs from INSIDE this directory
(`cd superactivation`), and its `SHA256SUMS` lists this directory's files only.

## What is here

* `experiments/superactivation/` — the lane: the code families, the n-use
  certifiers, the reductions and the hardware designs.
* `evidence/superactivation/` — every certificate, witness, code and decoder
  the manuscript's numbers come from, each certificate bound by sha256 to the
  bytes it proved.
* `src/oneq/superactivation*.py` — the channel model and the certificate
  checker (exact integers and ball arithmetic; no solver).
* `tests/` — the lane's tests, including the recomputation of stored pairs.
* `docs/PAPER4_LITERATURE_SEARCH.md` and `docs/reviews/` — the literature
  record the novelty wording is held to, and the referee reports.

## Reproduce

    python -m pip install -r requirements.txt
    python paper/tools/make_numbers_superactivation.py
    python paper/tools/build.py superactivation
    python tools/verify_r142_rational_witness.py evidence/superactivation/shield_family/certificate_two_use_d20.json evidence/superactivation/shield_family/rational_recovery_d20.json
    python -m pytest tests/ -q

(and the same for d21 and d22_closed with their rational_recovery_d21/d22
witnesses). `make_numbers_superactivation.py` re-proves every n-use certificate exactly,
recomputes the closed forms and their thresholds, and refuses when a
certificate no longer binds to its bytes; every number in the manuscript is a
macro it writes. `verify_r142_rational_witness.py` re-checks the d = 20, 21, 22
two-use bounds in exact rationals with no producer code. The two-group and
symmetric certificates took hours of exact arithmetic and are not re-run by the
generator: each is bound by sha256 to its witness and decoder bytes, and the
tests recompute the stored pairs. Building the PDF needs a TeX distribution
with REVTeX 4.2. The see-saw producers that found the codes need an SDP solver
(cvxpy) that is not pinned here: their outputs are what is checked.

`python tools/verify_release.py --bundle superactivation` checks a signed
release against the issuer pinned in `ISSUERS.json`: put the three
`oneq-superactivation` release assets in `release/` first.
"""


def _conformance_agreement() -> str:
    """"N/M agree" read from the conformance artifact, never typed.

    Falls back to a REFUSAL string rather than a plausible number: a
    predicate that cannot read its evidence must not assert a count.
    """
    import json as _json
    f = ROOT / "evidence" / "cert_conformance.json"
    try:
        d = _json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return "conformance agreement UNREAD (evidence missing)"
    n = d.get("vectors")
    txt = str(d.get("agree", ""))
    if "agree" in txt:
        return txt.split(":", 1)[-1].strip() if ":" in txt else txt
    return f"{n}/{n} agree" if n else "conformance agreement UNREAD"


def render_readme(template: str = README, root: pathlib.Path = ROOT) -> str:
    """The bundle README with its conformance counts COUNTED from conformance/cert_vectors.json.

    And the recorded conformance run must be a run over those vectors: the README's "@N_VECTORS@/@N_VECTORS@" row
    describes evidence/cert_conformance.json, so a run over another vector set (or one that disagreed) is a refusal,
    not a sentence that quotes today's count beside yesterday's evidence.
    """
    n, refusals = conformance_vector_counts(root / CERT_VECTORS)
    conf = json.loads((root / "evidence" / "cert_conformance.json").read_text(encoding="utf-8"))
    if (conf.get("vectors"), conf.get("expected_refuse"), conf.get("all_agree")) != (n, refusals, True):
        raise ValueError(f"evidence/cert_conformance.json records {conf.get('vectors')} vectors "
                         f"({conf.get('expected_refuse')} refusals, all_agree={conf.get('all_agree')}) and "
                         f"{CERT_VECTORS} holds {n} ({refusals} refusals): re-run tools/write_cert_conformance.py")
    text = template.replace("@N_VECTORS@", f"{n:,}").replace("@N_REFUSALS@", f"{refusals:,}")
    left = re.findall(r"@[A-Z_]+@", text)
    if left:
        raise ValueError(f"README placeholders with no value: {sorted(set(left))}")
    return text


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _place(src, rel, dest, subjects, placed):
    """Copy one payload file, PRESERVING its repository path.

    CONTENTS has always done this; EVIDENCE and OPTIONAL did not, and
    flattened to evidence/<basename>. Two entries collided --
    evidence/per_shot/MANIFEST.json and data/google_corpus/MANIFEST.json
    -- so one silently overwrote the other, per_shot/MANIFEST.json was
    absent from the bundle entirely, and step 2 of verify_release could
    not run for anyone holding only the tarball. The signed statement
    meanwhile carried that subject name twice with a digest belonging to
    neither file.

    `subject.name` is now the repo-relative path too, so the digest
    lookup cannot collide either.

    AND A COLLISION IS A REFUSAL, not a last-writer-wins. A packaging
    step that can silently drop a file is exactly the class the comment
    above CONTENTS already names.
    """
    import shutil

    # `dest / rel`, exactly as CONTENTS does -- NOT dest/evidence/rel.
    # Joining under evidence/ doubled the prefix into
    # evidence/evidence/per_shot/, which is not where verify_release
    # step 2 looks: it reads evidence/per_shot/MANIFEST.json, the
    # repository path. A bundle whose layout differs from the
    # repository is a bundle whose own audit commands fail from a
    # sterile extraction -- the finding that produced this rule.
    target = dest / rel
    key = target.resolve()
    if key in placed and placed[key] != rel:
        raise SystemExit(
            f"REFUSING: {rel} and {placed[key]} both land on "
            f"{target.relative_to(dest)} -- a packaging step that "
            f"silently drops a file is the defect this check exists "
            f"for")
    if key in placed:
        # Already placed at this very path (a file both bound by the receipt
        # manifest and listed in OPTIONAL): nothing new to copy, and attesting
        # it a second time is exactly the duplicate subject verify_release refuses.
        return
    placed[key] = rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    subjects.append({"name": rel,
                     "digest": {"sha256": sha256(src)}})


def _strong_baseline_payloads() -> list[str]:
    """The strong-baseline manifest and EVERY file it binds (external audit 2026-09-15, finding 3).

    tools/replay_strong_baseline.py refuses when any bound file is absent or altered, so the
    bundle carries exactly what the manifest names -- derived from it, never listed by hand.
    """
    rel = "evidence/strong_baseline_v3/MANIFEST.json"
    doc = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    files = list(doc["files"]) + [doc["instance_z"]["file"], doc["instance_x"]["file"]]
    for f in files:
        if not (ROOT / f).exists():
            raise RuntimeError(f"strong-baseline manifest binds a file that does not exist: {f}")
    return [rel] + files


def _strong_baseline_scaled_payloads() -> list[str]:
    """Paper 2's post-hoc scaled arms and every record file their summary binds by hash.

    make_numbers_qldpc.py re-hashes each bound record before it typesets a scaled count, so the bundle carries
    exactly what the summary names -- derived from it, never listed by hand.
    """
    rel = "evidence/qldpc_strong_baseline_scaled.json"
    files = list(json.loads((ROOT / rel).read_text(encoding="utf-8"))["files"])
    for f in files:
        if not (ROOT / f).exists():
            raise RuntimeError(f"scaled strong-baseline summary binds a file that does not exist: {f}")
    return [rel] + files


def _receipt_manifest_payloads() -> list[str]:
    """The receipt manifest and EVERY file it binds (external audit R-01/R-03).

    tools/replay_receipts.py fails a complete replay when any bound file is
    absent, so the bundle must carry exactly what the manifest names --
    derived from the manifest, never listed by hand beside it.
    """
    rel = "evidence/receipt_manifest.json"
    doc = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    files = list(doc["files"])
    for f in files:
        p = pathlib.PurePosixPath(f)
        if p.is_absolute() or ".." in p.parts or not f.startswith("evidence/"):
            raise ValueError(f"unsafe receipt-manifest path: {f!r}")
    return [rel] + [f for f in files if f not in EVIDENCE]


def _epsilon_rerun_payloads() -> list[str]:
    """Every COMPLETED epsilon-rerun configuration: its summary, its chunk index
    and its persisted instance (external audit A-01). The receipts themselves are
    a separate archive; with these files beside them, replay_epsilon_receipts.py
    runs. A pilot merge (fewer chunks than the plan) is not a result and is left
    out, so the bundle never ships a partial run as if it were the run."""
    out = []
    for summ in sorted((ROOT / "evidence" / "epsilon_rerun").glob("*/summary.json")):
        doc = json.loads(summ.read_text(encoding="utf-8"))
        lo, hi = doc["spec"]["seeds"]
        if doc["chunks"] != hi - lo + 1:
            continue
        name = summ.parent.name
        out += [f"evidence/epsilon_rerun/{name}/summary.json",
                f"evidence/epsilon_rerun/{name}/chunks.jsonl",
                f"evidence/instances/matching_{name}.json"]
        # the replay the manuscript's numbers are gated on, and the archive pin
        for extra in (f"evidence/epsilon_rerun/replay_{name}.json",
                      f"evidence/epsilon_rerun/{name}/archive.json"):
            if (ROOT / extra).exists():
                out.append(extra)
    return out


def _stopped_rerun_payloads() -> list[str]:
    """Every arm of a sample its preregistered rule STOPPED, as the partial report binds it (2026-11-05: the held-out
    d = 7 sample, docs/PREREG_heldout_d7_2026-11-03.md). Such an arm has no summary.json, so _epsilon_rerun_payloads
    passes it by; what paper 1 reads instead is the report, each arm's chunk index and failures, its persisted
    instance, the archive pin the receipts travel under, the partial-plan replay, and the stop records. Derived from
    the report, never listed by hand beside it."""
    out = []
    for rep in sorted((ROOT / "evidence" / "epsilon_rerun").glob("*_partial.json")):
        doc = json.loads(rep.read_text(encoding="utf-8"))
        if doc.get("schema") != "oneq-heldout-willow-d7-partial/1":
            continue
        out.append(f"evidence/epsilon_rerun/{rep.name}")
        for arm in sorted(doc["arms"]):
            if not re.fullmatch(r"[A-Za-z0-9_]+", arm):
                raise ValueError(f"unsafe arm name in {rep.name}: {arm!r}")
            out.append(f"evidence/epsilon_rerun/{arm}/chunks.jsonl")
            out.append(f"evidence/instances/matching_{arm}.json")
            for extra in (f"evidence/epsilon_rerun/{arm}/failures.jsonl", f"evidence/epsilon_rerun/{arm}/archive.json"):
                if (ROOT / extra).exists():
                    out.append(extra)
        # the stop records the report's head pins (head.records.dir), every file of them
        rec_dir = (doc.get("head") or {}).get("records", {}).get("dir")
        if isinstance(rec_dir, str) and rec_dir.startswith("evidence/epsilon_rerun/"):
            out += _tree_files(rec_dir)
    return list(dict.fromkeys(out))


def required_evidence_files() -> list[str]:
    """Every evidence file the bundle must carry: the declared list and every derived closure. One function, so the
    builder and tests/test_bundle_carries_the_papers_evidence.py cannot disagree about what ships."""
    return (list(EVIDENCE) + _per_shot_payloads() + _receipt_manifest_payloads() + _strong_baseline_payloads()
            + _strong_baseline_scaled_payloads() + _epsilon_rerun_payloads() + _stopped_rerun_payloads()
            + _mutation_score_payloads())


def _mutation_score_payloads() -> list[str]:
    """Every operator mutation score, with the checker and the killing tests it was measured on.

    paper/tools/make_numbers*.py refuse to typeset a score whose checker or tests changed
    since it was measured, so the score is checkable -- and the papers buildable -- only from
    a tree that carries those tests. Derived from the score artifacts, never listed by hand
    beside them. 2026-09-15: the tests were not shipped, and both number generators failed in
    the public tree.
    """
    out = []
    for p in sorted((ROOT / "evidence" / "mutation_score").glob("*.json")):
        out.append(f"evidence/mutation_score/{p.name}")
        if p.name.endswith(".equivalents.json"):
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        for f in (doc["target"], *doc["tests"]):
            q = pathlib.PurePosixPath(f)
            if q.is_absolute() or ".." in q.parts or not f.startswith(("src/", "tests/")):
                raise ValueError(f"unsafe mutation-score path: {f!r}")
            out.append(f)
    return list(dict.fromkeys(out))


#: Shipped files whose imports reach modules the bundle deliberately does not carry, each with
#: its reason. A declaration that no longer holds -- the file is not shipped, or everything it
#: imports now is -- is itself a refusal, so this list cannot outlive its reasons.
IMPORTS_BEYOND_THE_BUNDLE = {
    "src/oneq/api.py": ("the product surface reaches the whole package through lazy imports; the "
                        "bundle carries the checkers and the escalation ladder, not the package"),
    "src/oneq/__init__.py": ("imports nothing when it runs (importlib, typing); the public surface (passport, api) "
                             "loads only on attribute access, which no shipped file makes. It ships because "
                             "evidence/producer_identity_willow.json pins its bytes"),
    "tools/claims_provenance_gate.py": "a development gate that rebuilds this bundle, so it needs the builder",
    "tools/doc_claims_gate.py": "a development gate over the repository's own prose and package",
    "tools/gate_all.py": "the development wall, which README says does not run from this tree",
    "tools/mutation_gate.py": "the development mutation gate, which README says does not run from this tree",
}
#: The same declaration for paper 4's bundle: nothing it ships may reach beyond it.
SUPERACTIVATION_IMPORTS_BEYOND: dict[str, str] = {}


def bundle_spec(name: str) -> dict:
    """Everything that differs between the two bundles, in one place, so main() and the tests read the same lists."""
    if name == "certifying-decoder":
        return {**BUNDLES[name], "contents": CONTENTS, "evidence": required_evidence_files, "optional": OPTIONAL,
                "readme": render_readme, "beyond": IMPORTS_BEYOND_THE_BUNDLE}
    if name == "superactivation":
        return {**BUNDLES[name], "contents": SUPERACTIVATION_CONTENTS, "evidence": lambda: [], "optional": [],
                "readme": lambda: SUPERACTIVATION_README, "beyond": SUPERACTIVATION_IMPORTS_BEYOND}
    raise ValueError(f"no bundle named {name!r}; the bundles are {sorted(BUNDLES)}")

_NAMESPACE_INIT = "src/oneq/__init__.py"
_DIGEST = re.compile(rb"[0-9a-f]{64}")


def dirty_sources(root: pathlib.Path, rels) -> list[str]:
    """Bundled source files whose bytes are NOT the ones the bound commit names.

    The signature and the in-toto statement bind the archive to `git rev-parse HEAD`;
    a reader who checks that commit out and rebuilds must get the same bytes. On
    2026-09-16 a mint bound to 87b3b97 carried two evidence files restored in the
    working tree and never committed, so the commit the signature named did not
    reproduce the archive it signed. Reported as "<XY> <path>" from `git status`
    (`??` = untracked): tracked-and-modified, staged, and untracked all count.

    The whole-tree status is intersected with `rels` rather than passed the path
    list: 1,366 paths exceed the Windows command-line limit, and a check that
    crashed on its own argument list would be a check that never ran.
    """
    wanted = {pathlib.PurePosixPath(r).as_posix() for r in rels}
    r = subprocess.run(["git", "status", "--porcelain", "-z", "--no-renames",
                        "--untracked-files=all"], cwd=root, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("git status failed: " + r.stderr.decode("utf-8", "replace")[:300])
    out = []
    for entry in r.stdout.decode("utf-8", "replace").split(chr(0)):
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if path in wanted:
            out.append(f"{code.strip() or '??'} {path}")
    return sorted(out)


def advertised_scripts_missing(dest: pathlib.Path) -> list[str]:
    """Scripts the bundle's top-level documents tell a reader to run that the bundle does not carry.

    2026-09-15: REPRODUCE.md named tools/pack_epsilon_receipts.py and the bundle did not ship
    it. The comment above CONTENTS already said the commands AUDIT.md advertises must be
    here; nothing held that for any document, AUDIT.md included.
    """
    missing = set()
    for doc in sorted(dest.glob("*.md")):
        words = doc.read_text(encoding="utf-8").replace("`", " ").split()
        for cmd, arg in zip(words, words[1:]):
            if (cmd in ("python", "python3", "node") and arg.endswith((".py", ".mjs", ".js"))
                    and not (dest / arg).is_file()):
                missing.add(f"{doc.name}: {cmd} {arg}")
    return sorted(missing)


#: The repository directories a top-level document's backticked path can point into.
_DOC_PATH_TOPS = ("docs/", "evidence/", "experiments/", "tools/", "src/", "tests/", "paper/", "verifier-js/",
                  "conformance/", "lean/", "third_party/", "schemas/")


def advertised_paths_missing(dest: pathlib.Path) -> list[str]:
    """Backticked repository paths in the bundle's top-level documents that the bundle does not carry.

    2026-11-07 (external audit 4, R8): AUDIT.md and REPRODUCE.md named seven files and directories the bundle never
    shipped (old paper drafts, an evidence directory, two tools, an integration document), and nothing checked any
    path that was not a `python x.py` command. A line that says the path is in the development repository is
    exempt: it tells the reader where the thing is instead of promising it. Placeholders (<config>, DIR, *) match a
    path segment.
    """
    shipped = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    missing = set()
    for doc in sorted(dest.glob("*.md")):
        for line in doc.read_text(encoding="utf-8").splitlines():
            if "development repository" in line:
                continue
            for raw in re.findall(r"`([^`\s]+)`", line) + re.findall(r"\]\(([^)\s#]+)", line):
                q = raw.rstrip(".,;:)")
                if not q.startswith(_DOC_PATH_TOPS):
                    continue
                if any(c in q for c in "*<>{}"):
                    rx = re.sub(r"\\\*|<[^>]*>|\\\{[^}]*\\\}", "[^/]*", re.escape(q))
                    hit = any(re.fullmatch(rx + "(/.*)?", f) for f in shipped)
                elif q.endswith("/"):
                    hit = any(f.startswith(q) for f in shipped)
                else:
                    hit = q in shipped or any(f.startswith(q + "/") for f in shipped)
                if not hit:
                    missing.add(f"{doc.name}: {q}")
    return sorted(missing)


def import_closure_gaps(dest: pathlib.Path, repo_root: pathlib.Path, repo_files: set[str],
                        declared: dict[str, str]) -> tuple[dict[str, list[str]], list[str]]:
    """Shipped Python files whose imports reach repository modules the bundle does not carry.

    2026-09-15: verify_release.py imports sign_release inside a try block, sign_release was
    not shipped, and so the command that establishes a release's authenticity could not run
    from the release -- while it passed here, where the repository supplied the module.

    Resolved statically with revision_snapshots' resolver against the repository's files, so
    an import inside a function counts: it fails for a reader the first time that path runs.
    The package __init__ is not followed: the bundle ships `oneq` as a namespace package, and
    `import oneq.x` there never executes it. Returns (undeclared gaps, stale declarations).
    """
    from revision_snapshots import _imports

    shipped = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    memo: dict[str, set[str]] = {}

    def deps(path: str) -> set[str]:
        if path not in memo:
            base = dest if path in shipped else repo_root
            memo[path] = set(_imports(path, (base / path).read_bytes(), repo_files)) - {_NAMESPACE_INIT}
        return memo[path]

    gaps: dict[str, list[str]] = {}
    stale = set(declared) - shipped
    for entry in sorted(p for p in shipped if p.endswith(".py")):
        todo, seen = [entry], set()
        while todo:
            p = todo.pop()
            if p not in seen:
                seen.add(p)
                todo.extend(deps(p) - seen)
        missing = sorted(seen - shipped)
        if missing and entry not in declared:
            gaps[entry] = missing
        elif not missing and entry in declared:
            stale.add(entry)
    return gaps, sorted(stale)


def pinned_before_normalisation(dest: pathlib.Path, text_suffixes) -> set[pathlib.Path]:
    """CRLF text files whose current digest another shipped file states: they ship as they are.

    `normalize` rewrites CRLF to LF so SHA256SUMS survives git. A digest recorded over the
    CRLF bytes then names bytes no reader receives, and a reader who checks it with sha256sum
    sees what looks like tampering. Measured 2026-09-15 on one mint: the held-out
    preregistration pinned two files by their CRLF digests, the checker-equivalence record
    pinned a frozen snapshot, and the receipt manifest pinned a CRLF artifact.
    in-toto-statement.json is not read: at this point it holds source digests by design, and
    it is rewritten from the shipped bytes after normalisation.
    """
    # EVERY file with a pinned digest, not one per digest: two shipped copies of the same bytes (paper 3's T = 2 model
    # in recert_20261102/ and in r205_cdp5/, 2026-11-03) share a digest, and a one-file map kept only the last-seen
    # copy and let normalisation rewrite the other under its pins
    crlf: dict[bytes, list[pathlib.Path]] = {}
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix.lower() in text_suffixes:
            data = f.read_bytes()
            if bytes([13, 10]) in data:
                crlf.setdefault(hashlib.sha256(data).hexdigest().encode(), []).append(f)
    keep: set[pathlib.Path] = set()
    if crlf:
        for f in dest.rglob("*"):
            if f.is_file() and f.relative_to(dest).as_posix() != "in-toto-statement.json":
                stated = set(_DIGEST.findall(f.read_bytes())) & set(crlf)
                keep.update(g for d in stated for g in crlf[d] if g != f)
    return keep


def pins_broken_by_normalisation(before: dict[str, str], dest: pathlib.Path) -> list[str]:
    """Shipped files that still state the pre-normalisation digest of a file that was rewritten."""
    old = {digest.encode(): rel for rel, digest in before.items()
           if (dest / rel).is_file() and sha256(dest / rel) != digest}
    broken = set()
    if old:
        for f in dest.rglob("*"):
            if f.is_file():
                for d in set(_DIGEST.findall(f.read_bytes())) & set(old):
                    broken.add(f"{f.relative_to(dest).as_posix()} pins {old[d]}")
    return sorted(broken)


def head_pins_unshipped(dest: pathlib.Path) -> list[str]:
    """Files that a shipped evidence artifact pins in its provenance block (`head.sha256`: the bytes that judged or
    produced it) but that the bundle does not carry.

    The papers' generators re-hash those files and refuse an artifact judged by other bytes, so a pin to a file that
    does not ship fails every reader's regeneration. 2026-11-03: the producer-identity record pinned
    src/oneq/__init__.py, which the bundle had left out, and paper 1's generator failed in the release rehearsal.
    """
    missing = []
    for f in sorted((dest / "evidence").rglob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        head = doc.get("head") if isinstance(doc, dict) else None
        pins = head.get("sha256") if isinstance(head, dict) else None
        if isinstance(pins, dict):
            missing += [f"{f.relative_to(dest).as_posix()} pins {p}" for p in pins if not (dest / p).is_file()]
    return missing


def audit_sums_count_mismatch(dest: pathlib.Path, n_files: int) -> str | None:
    """AUDIT.md's first command tells a reader how many files SHA256SUMS checks; it must be this bundle's.

    It said 180 while the bundle held 353 (2026-09-15): a count a reader compares by eye is a
    claim, and nothing held it.
    """
    audit = dest / "AUDIT.md"
    if not audit.is_file():
        return "AUDIT.md is not in the bundle"
    for line in audit.read_text(encoding="utf-8").splitlines():
        if line.startswith("sha256sum -c SHA256SUMS"):
            words = line.partition("#")[2].split()
            if len(words) >= 2 and words[1] == "files" and words[0].replace(",", "").isdigit():
                stated = int(words[0].replace(",", ""))
                return None if stated == n_files else f"AUDIT.md says {stated} files; SHA256SUMS lists {n_files}"
            return "AUDIT.md's sha256sum line does not state how many files it checks"
    return "AUDIT.md has no `sha256sum -c SHA256SUMS` line"


def _per_shot_payloads() -> list[str]:
    """Return every record/weight subject declared by the per-shot manifest.

    The verifier advertises these as release evidence, so shipping the
    manifest without the subjects it names is a build refusal, not an
    optional partial bundle.
    """
    manifest_rel = "evidence/per_shot/MANIFEST.json"
    path = ROOT / manifest_rel
    if not path.is_file():
        raise ValueError(f"required manifest missing: {manifest_rel}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    sections = [doc.get("rungs")]
    if "heldout" in doc:
        held = doc.get("heldout")
        sections.append(held.get("rungs") if isinstance(held, dict) else None)
    out = [manifest_rel]
    for rungs in sections:
        if not isinstance(rungs, dict) or not rungs:
            raise ValueError("per-shot manifest contains an empty rung section")
        for rung in rungs.values():
            if not isinstance(rung, dict):
                raise ValueError("per-shot manifest contains a malformed rung")
            for field, prefix, suffix in (
                    ("records_file", "evidence/per_shot/", ".jsonl"),
                    ("weights_file", "evidence/weights/", "_weights.json")):
                rel = rung.get(field)
                p = pathlib.PurePosixPath(rel) if isinstance(rel, str) else None
                if (p is None or p.is_absolute() or p.as_posix() != rel
                        or ".." in p.parts or not rel.startswith(prefix)
                        or not rel.endswith(suffix)):
                    raise ValueError(f"unsafe {field}: {rel!r}")
                out.append(rel)
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=sorted(BUNDLES), default="certifying-decoder",
                    help="which of the release's bundles to build (default: the decoder bundle)")
    ap.add_argument("--out", type=str, default=None, help="default: release/<the bundle's directory>")
    ap.add_argument("--sign", type=str, default=None,
                    help="path to a 32-byte Ed25519 seed OUTSIDE the repo")
    a = ap.parse_args()
    spec = bundle_spec(a.bundle)
    a.out = a.out or f"release/{spec['dir']}"
    if pathlib.PurePath(a.out).name != spec["dir"]:
        # the archive is named after the directory and verify_release reads it under that root; a bundle built into
        # another bundle's name would be signed and verified as the wrong one
        print(f"REFUSING: --out must end in {spec['dir']} for the {a.bundle} bundle, not {a.out}", file=sys.stderr)
        return 2
    contents, top = spec["contents"], spec["top"]

    # THE CORPUS MANIFEST IS SHIPPED FROM A GIT-IGNORED DIRECTORY. On 2026-10-31 data/google_corpus/ was emptied, and a
    # later fetch of other configurations rewrote its MANIFEST.json: a bundle built then would have shipped the wrong
    # corpus description, or none. The tracked copy is the authority; the local file must equal it byte for byte.
    local_m, tracked_m = ROOT / "data" / "google_corpus" / "MANIFEST.json", ROOT / "evidence" / "google_corpus_manifest.json"
    if a.bundle == "certifying-decoder" and (not local_m.is_file() or local_m.read_bytes() != tracked_m.read_bytes()):
        print(f"REFUSED: {local_m.relative_to(ROOT).as_posix()} is missing or differs from the tracked "
              f"{tracked_m.relative_to(ROOT).as_posix()}; restore it with `python tools/fetch_google_corpus.py` "
              f"(the default configurations) before minting.")
        return 2

    dest = ROOT / a.out
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    missing, subjects = [], []
    for rel, _why in contents:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        # PRESERVE THE REPOSITORY PATH. EXTERNAL AUDIT 2026-08-04,
        # finding 2: the bundle flattened everything to basenames while
        # the scripts and AUDIT.md still assumed repo layout, so all
        # three advertised audit commands failed from a sterile
        # extraction. Worse, flattening made basenames COLLIDE --
        # tools/frozen/qldpc_check.py and src/oneq/qldpc_check.py map to
        # the same name, and one silently overwrote the other, so a file
        # we had just added to fix finding 3 disappeared without a word.
        # A packaging step that can drop a file without failing is the
        # same class of defect as a checker that accepts without proving.
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    if missing:
        print(f"REFUSING: {len(missing)} declared file(s) absent: {missing}",
              file=sys.stderr)
        return 2

    # exist_ok: paper 3's lane ships its records under evidence/ as CONTENTS (2026-11-02), so the directory can exist
    (dest / "evidence").mkdir(exist_ok=True)
    placed: dict = {}
    absent = []
    try:
        required_evidence = spec["evidence"]()
    except Exception as exc:
        print(f"REFUSING: per-shot release closure is invalid: {exc}",
              file=sys.stderr)
        return 2
    for rel in dict.fromkeys(required_evidence):
        src = ROOT / rel
        if not src.exists():
            absent.append(rel)
            continue
        _place(src, rel, dest, subjects, placed)
    if absent:
        print(f"REFUSING: required evidence absent: {absent}", file=sys.stderr)
        return 2

    pending = []
    for rel in spec["optional"]:
        src = ROOT / rel
        if src.exists():
            _place(src, rel, dest, subjects, placed)
        else:
            pending.append(rel)

    try:
        readme = spec["readme"]()
    except (OSError, ValueError, KeyError) as exc:
        print(f"REFUSING: the README's conformance counts cannot be read from the vectors: {exc}", file=sys.stderr)
        return 2
    (dest / "README.md").write_text(readme, encoding="utf-8")

    # ENTRY-POINT DOCS GO AT THE TOP LEVEL, beside README. AUDIT.md was
    # landing in evidence/ because it had been added to OPTIONAL, which
    # routes everything there -- so the one file an auditor should open
    # first was buried among 91 JSON artifacts, and REPRODUCE.md was not
    # shipped at all. An offline auditor has no repository to fall back
    # on: if it is not in the tarball, it does not exist for them.
    for name, doc in top.items():
        src = ROOT / doc
        if src.exists():
            if (dest / name).exists():
                print(f"REFUSING: top-level {name} would overwrite a shipped file", file=sys.stderr)
                return 2
            shutil.copy2(src, dest / name)
            subjects.append({"name": name,
                             "digest": {"sha256": sha256(src)}})

    # THE COMMIT THE SIGNATURE NAMES MUST REPRODUCE THE ARCHIVE IT SIGNS. Every source the
    # bundle copied is checked against the working tree's git status; a signed mint refuses
    # any difference, an unsigned (dry-run) mint records it in the statement instead of
    # hiding it (see dirty_sources).
    sources = ([rel for rel, _why in contents if (ROOT / rel).exists()]
               + list(placed.values())
               + [doc for doc in top.values() if (ROOT / doc).exists()])
    dirty_unknown = None
    try:
        dirty = dirty_sources(ROOT, sources)
    except Exception as exc:                    # noqa: BLE001
        # *** "NO GIT HERE" IS NOT "DIRTY", and the first version conflated them. *** A sterile
        # copy has no repository: the mutation sandbox, an extracted tarball, a reviewer's
        # unzipped download. Refusing there means the bundle can be rebuilt only inside a clone,
        # which is the opposite of what this bundle is for, and the mutation gate caught it within
        # hours of the check landing (2026-09-16). A SIGNED mint still refuses, because a
        # signature naming a commit asserts the tree matches it and that assertion cannot be
        # checked without the repository.
        if a.sign:
            print(f"REFUSING to sign: cannot compare the bundle's sources with the commit the "
                  f"signature would name: {exc}", file=sys.stderr)
            return 2
        dirty, dirty_unknown = [], str(exc).splitlines()[0][:200]
        print(f"  UNSIGNED mint: the source-vs-commit comparison could not run here "
              f"({dirty_unknown}); recorded as unknown, not as clean")
    if dirty and a.sign:
        print(f"REFUSING to sign: {len(dirty)} bundled source file(s) differ from HEAD, so the "
              f"commit the signature would name does not reproduce this archive: {dirty}",
              file=sys.stderr)
        return 2
    if dirty:
        print(f"  UNSIGNED mint from a dirty tree; recorded in the statement: {dirty}")

    # WHAT A SHIPPED DOCUMENT TELLS A READER TO RUN MUST RUN FROM THE BUNDLE, imports included.
    # 2026-09-15: a script REPRODUCE.md names was not shipped, and verify_release could not
    # import sign_release. Both passed here, where the repository fills every gap.
    missing_scripts = advertised_scripts_missing(dest)
    if missing_scripts:
        print(f"REFUSING: shipped documents name scripts the bundle does not carry: {missing_scripts}",
              file=sys.stderr)
        return 2
    missing_paths = advertised_paths_missing(dest)
    if missing_paths:
        print(f"REFUSING: shipped documents name paths the bundle does not carry: {missing_paths}", file=sys.stderr)
        return 2
    listing = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if listing.returncode == 0:
        repo_files = {f for f in listing.stdout.split(chr(0)) if f and (ROOT / f).is_file()}
    else:
        # A STERILE TREE STILL GETS THE CHECK, from the filesystem. This refused outright, so the
        # bundle could not be built anywhere without a repository -- the mutation sandbox, an
        # extracted tarball, a reviewer's unzipped download. The import closure is a property of
        # the FILES, and the files are right here; git was only ever a convenient index of them.
        # (Found 2026-09-16 by the mutation gate's own baseline, which builds a bundle in a
        # sandbox it deliberately strips of .git.)
        print("  no git index here; resolving the bundle's imports from the filesystem instead")
        repo_files = set()
        for p in ROOT.rglob("*.py"):
            if any(part in (".git", "__pycache__", "node_modules") for part in p.parts):
                continue
            if p.is_file():
                repo_files.add(p.relative_to(ROOT).as_posix())
    gaps, stale = import_closure_gaps(dest, ROOT, repo_files, spec["beyond"])
    for entry, miss in gaps.items():
        print(f"REFUSING: {entry} imports what the bundle does not carry: {miss}", file=sys.stderr)
    for entry in stale:
        print(f"REFUSING: IMPORTS_BEYOND_THE_BUNDLE declares {entry}, which is not shipped or no "
              f"longer reaches beyond the bundle", file=sys.stderr)
    if gaps or stale:
        return 2

    # in-toto Statement v1. A proposed predicate type, not a bespoke wrapper.
    versions = {}
    for mod in ("stim", "pymatching", "numpy", "scipy", "ldpc"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:                       # noqa: BLE001
            versions[mod] = "absent"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True,
                                timeout=30).stdout.strip()
    except Exception:                           # noqa: BLE001
        commit = "unknown"

    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://one-q.org/DecodingCertification/v0.1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://one-q.org/certifying-decoder-run/v0.1",
                "externalParameters": {
                    "corpus": "10.5281/zenodo.13273331 (read by HTTP range; "
                              "the 5.72 GB archive is never downloaded)",
                    "simulated": "stim surface_code:rotated_memory_z, p=5e-3"},
                "resolvedDependencies": [
                    {"uri": "git+https://one-q.org/ONE_Q", "digest": {"gitCommit": commit}}],
                # Empty for every signed mint (a signed mint refuses otherwise); an unsigned
                # dry run from an edited tree names the files the commit does not reproduce.
                # `null` means CHECKED AND CLEAN; a string means the check could not run here
                # and the empty list above must not be read as clean.
                "sourcesDifferingFromCommit": dirty,
                "sourceComparisonUnavailable": dirty_unknown,
                "internalParameters": {"library_versions": versions,
                                       "python": sys.version.split()[0]},
            },
            "runDetails": {
                "builder": {"id": "https://one-q.org/local"},
                "metadata": {"invocationId": commit},
            },
            "claims": {
                "soundness": ("no forged certificate was accepted; the count of "
                              "attacks that failed to falsify is reported "
                              "alongside, because a battery that stops "
                              "attacking also reports zero"),
                # DERIVED, NOT TYPED. This read "55/55" while the
                # artifact said 56/56 -- a hardcoded count in a SIGNED
                # predicate drifts silently the moment a vector is
                # added, and the signature then attests the stale
                # number. Reading it from the conformance result means
                # the claim cannot disagree with the evidence beside it.
                "independence": (
                    f"two checkers, two languages, no shared code, "
                    f"{_conformance_agreement()} on vectors that are "
                    f"mostly refusals"),
            },
            "explicitlyNotClaimed": [
                "real-time decoding: certifying costs ~10^5x decoding",
                "completeness: ~11% of shots receive a degraded receipt, not a proof",
                "novelty of the dual: the lane is adjacent-occupied "
                "(arXiv:2508.04969); the contribution is the assurance engineering",
                "that agreement between two checkers implies correctness -- it "
                "implies the absence of a SHARED misunderstanding, which is less",
            ],
            "pendingArtifacts": pending,
        },
    }
    if a.bundle == "superactivation":
        # A SEPARATE CLAIM, so a separate predicate: nothing in paper 4's bundle is a decoding certificate, and a
        # statement carrying the decoder bundle's claims would attest to evidence this archive does not hold.
        pred = statement["predicate"]
        statement["predicateType"] = "https://one-q.org/SuperactivationCertification/v0.1"
        pred["buildDefinition"]["buildType"] = "https://one-q.org/superactivation-certificates/v0.1"
        pred["buildDefinition"]["externalParameters"] = {
            "channels": "a family of PPT channels P_d paired with the 50% erasure channel A_2d, and Smith and Yard's "
                        "pair (arXiv:0807.4935); no hardware input is needed to check any certificate"}
        sa_versions = {}
        for mod in ("numpy", "scipy", "flint", "sympy", "qiskit"):
            try:
                sa_versions[mod] = __import__(mod).__version__
            except Exception:                   # noqa: BLE001
                sa_versions[mod] = "absent"
        pred["buildDefinition"]["internalParameters"]["library_versions"] = sa_versions
        pred["claims"] = {
            "certificates": ("every n-use lower bound is an exact certificate in Q(sqrt d) or exact rationals, bound by "
                             "sha256 to the code and decoder bytes it proved, and re-proved by the shipped generator"),
        }
        pred["explicitlyNotClaimed"] = [
            "that any device superactivates: the hardware runs are an outlook, and the recorded device run is below 3/4",
            "the onset for Smith and Yard's pair: a certified code bounds n* from above, it does not fix it",
            "that the see-saw producers re-run from this bundle: they need an SDP solver it does not pin; their "
            "outputs are what is checked",
        ]
    (dest / "in-toto-statement.json").write_text(
        json.dumps(statement, indent=2), encoding="utf-8")

    # A FILE ANOTHER SHIPPED FILE PINS BY DIGEST SHIPS BYTE FOR BYTE (see
    # pinned_before_normalisation). LF normalisation had made four pins in one mint name bytes
    # no reader receives, two of them in the immutable held-out preregistration.
    keep = pinned_before_normalisation(dest, TEXT)
    before_normalisation = {p.relative_to(dest).as_posix(): sha256(p)
                            for p in dest.rglob("*") if p.is_file()}
    n_norm = normalize(dest, keep=keep)
    if keep:
        print(f"  shipped byte for byte because a shipped file pins their digest: "
              f"{sorted(p.relative_to(dest).as_posix() for p in keep)}")

    # EXTERNAL AUDIT 2026-08-04, finding 1: the signed statement listed 93
    # subjects of which only 10 matched the archived bytes. Cause: the
    # digests were taken from the SOURCE files, and `normalize` rewrites
    # line endings in the bundle AFTERWARDS -- so the statement attested
    # bytes that were never shipped. Same root cause as the frozen-snapshot
    # CRLF defect found this morning: hash what you SHIP, not what you have.
    #
    # Recomputed here, after normalisation, from the files actually in the
    # bundle. A subject that cannot be located is dropped rather than left
    # attesting a digest nobody can check.
    # LOOK THE SUBJECT UP BY ITS PATH, WHICH IS WHAT ITS NAME IS.
    #
    # This resolved subjects through a BASENAME map while `_place` had
    # already switched subject.name to the repo-relative path (see its
    # docstring). So 90 of 93 subjects failed to resolve and were
    # silently dropped: the shipped statement attested AUDIT.md,
    # REPRODUCE.md and REFREEZE_PROTOCOL.md, and NOT ONE line of the
    # checker it exists to attest. The build printed "93" and exited 0.
    #
    # The basename map was also unsound on its own terms:
    # src/oneq/matching_cert.py and tools/frozen/matching_cert.py share a
    # basename, so `setdefault` bound whichever rglob reached first and a
    # matching subject would have been digested against the WRONG FILE --
    # the precise collision _place refuses upstream.
    _fixed, _dropped = [], []
    for s in statement["subject"]:
        f = dest / s["name"]
        if not f.is_file():
            _dropped.append(s["name"])
            continue
        s["digest"]["sha256"] = sha256(f)
        _fixed.append(s["name"])

    # AND AN UNLOCATABLE SUBJECT IS A REFUSAL, NOT A DROP. Every subject
    # here was placed into this bundle moments ago by _place, so failing
    # to find one means the packaging and the attestation disagree. The
    # old behaviour -- drop it and carry on -- is what let the statement
    # shrink to 3 subjects without anyone noticing. Shipping less than we
    # claim must cost an exit code.
    if _dropped:
        print(f"REFUSING: {len(_dropped)} in-toto subject(s) are not in the "
              f"bundle they attest: {_dropped[:5]}"
              + (" ..." if len(_dropped) > 5 else ""), file=sys.stderr)
        return 2
    statement["subject"] = [s for s in statement["subject"]
                            if s["name"] in set(_fixed)]

    # ATTEST WHAT WE SHIP, WHICH INCLUDES THE CHECKER.
    #
    # The subject list was built only from EVIDENCE, so it covered 93 of
    # 184 shipped files and NOT ONE line of source: not matching_cert.py
    # (the file carrying the confirmed mixed-dual soundness break), not
    # the tools, not the conformance vectors, not the README making the
    # claims. A statement titled "here is an independent checker you can
    # audit" that attests the measurements but never the checker leaves
    # the reader to assume a coverage it does not have.
    #
    # Bundle integrity did not depend on this -- SHA256SUMS covers every
    # file and rides under the archive signature -- so this widens the
    # ATTESTATION to match the manifest, rather than closing a hole.
    #
    # TWO FILES CANNOT BE SUBJECTS OF THIS STATEMENT, and are declared
    # instead of quietly missing: the statement cannot contain its own
    # digest, and SHA256SUMS is generated afterwards (it is what covers
    # the statement, so the direction is deliberate).
    _unattestable = ("in-toto-statement.json", "SHA256SUMS")
    _have = {s["name"] for s in statement["subject"]}
    for f in sorted(dest.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(dest).as_posix()
        if rel in _unattestable or rel in _have:
            continue
        statement["subject"].append(
            {"name": rel, "digest": {"sha256": sha256(f)}})
    # A SUBJECT LISTED TWICE IS A REFUSAL, here and not only in the verifier.
    # 2026-09-15: files bound by the receipt manifest were also in OPTIONAL, so
    # both loops placed them and the signed statement named two of them twice;
    # verify_release caught it, but the producer should never write such a
    # statement in the first place.
    _seen_names: dict = {}
    for s in statement["subject"]:
        _seen_names[s["name"]] = _seen_names.get(s["name"], 0) + 1
    _dup_names = sorted(n for n, c in _seen_names.items() if c > 1)
    if _dup_names:
        print(f"REFUSING: in-toto subjects listed more than once: {_dup_names}", file=sys.stderr)
        return 2
    statement["predicate"]["unattestableFiles"] = {
        "files": list(_unattestable),
        "why": ("a statement cannot carry its own digest; SHA256SUMS is "
                "written after the statement and is what covers it"),
    }
    (dest / "in-toto-statement.json").write_text(
        json.dumps(statement, indent=2), encoding="utf-8")
    print(f"  in-toto: {len(_fixed)} subjects digested from the SHIPPED "
          f"bytes" + (f"; {len(_dropped)} unlocatable and dropped"
                      if _dropped else ""))
    broken = pins_broken_by_normalisation(before_normalisation, dest)
    if broken:
        print(f"REFUSING: normalisation rewrote files that shipped files pin by their old digest: "
              f"{broken}", file=sys.stderr)
        return 2
    n_norm += 1
    unshipped = head_pins_unshipped(dest)
    if unshipped:
        print(f"REFUSING: shipped evidence pins the bytes of files the bundle does not carry, so no reader can "
              f"check them: {unshipped}", file=sys.stderr)
        return 2
    n_files = write_manifest(dest)
    stale_count = audit_sums_count_mismatch(dest, n_files)
    if stale_count:
        # NAME THE CAUSE. A clean clone built 1,405 files against AUDIT.md's 1,406 (2026-10-19) and the refusal said
        # only that; the missing file was an OPTIONAL input this checkout had never fetched
        why = (f" -- OPTIONAL inputs absent from this checkout: {pending}"
               + ("; data/google_corpus is fetched by `python tools/fetch_google_corpus.py`"
                  if any(p.startswith("data/google_corpus/") for p in pending) else "")) if pending else ""
        print(f"REFUSING: {stale_count}{why}", file=sys.stderr)
        return 2
    tar = dest.parent / f"{dest.name}.tar.gz"
    digest = build_archive(dest, tar)

    print(f"bundle: {n_files} files ({n_norm} newline-normalised)")
    # COUNT THE STATEMENT, NOT THE INTENT. This printed len(subjects) --
    # the list as first assembled -- while the statement written to disk
    # had been filtered down to 3. The build reported 93 and the artifact
    # carried 3, so the one number an auditor reads was the one number
    # nothing checked.
    print(f"  subjects in the in-toto statement: "
          f"{len(statement['subject'])}")
    if pending:
        print(f"  PENDING (recorded, not silently skipped): {pending}")
    # --out may name a directory outside the repository (a dry-run mint);
    # the report must not crash after the archive is already written.
    shown = tar.relative_to(ROOT) if tar.is_relative_to(ROOT) else tar
    print(f"  archive {shown}  sha256 {digest}")

    if a.sign:
        seed = pathlib.Path(a.sign)
        if ROOT in seed.resolve().parents:
            print("REFUSING to read a signing key from inside the repository",
                  file=sys.stderr)
            return 3
        from oneq.cert_format import sign_document
        signed = sign_document(statement, seed.read_bytes())
        (dest.parent / f"{dest.name}.statement.signed.json").write_text(
            json.dumps(signed, indent=2), encoding="utf-8")
        print(f"  signed by {signed['pubkey'][:16]}...  "
              f"(attests authorship ONLY -- re-run the checker for truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
