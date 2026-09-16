# External adversarial audits of both manuscripts — 2026-09-14 — findings register

Two independent audits of *Certifying quantum error-correction decoders* (P1) and *Exact,
independently checkable optimality receipts for BP-OSD on [[144,12,12]]* (P2), against the public
release `coherence-energy-labs/oneq-certificates` at `ac9370f`. Source files (owner's Downloads):
`QEC_Adversarial_Audit_2026-09-14.md` (IDs A-/B-/R-/C-), `audit_philbrick_qec_receipts_2026-09-14.md`
(IDs B1-B4/M/m), `qec_audit_independent_checks.py` + results.

**Every finding below was re-verified here against source and artifacts before being accepted.**
Verdicts: CONFIRMED (defect is real), RESOLVED (not a defect; explanation needed), QUALIFIES (claim
must narrow), OPEN (needs an experiment to decide).

## What survived

- **Mathematics: no counterexample.** The auditor's independent script was re-run on this machine
  and reproduced its output file exactly (4,000 safe-dual bounds vs exhaustive, 419 + 500 lattice
  acceptances, 1,000 T-join packings incl. zero-weight edges, BB construction rank 66 / 66 / k=12;
  0 violations). T-odd cut packing, the scaling lemma, syndrome facets, the safe dual, the strict
  lattice rule and the branch partition all stand.
- Arithmetic reconciles, including the two items one audit flagged:
  4,924 = 4,861 defective + 62 controls + 1 non-defective d=3 stale-cache mutation (169 attempts,
  168 defective); 6,000,002 = 6 per shot plus one 7th applicable attack in each of two runs.

## Register

| ID | Verdict | Finding (verified) | Acceptance condition |
|---|---|---|---|
| B-02 / P2 | CONFIRMED | `qldpc_strong_baseline_gate.py`: when the checked LP path gives no answer the reference is the MILP support with **no receipt check**, and a MILP returning nothing becomes an **empty correction** that is still scored. One fixed reference is reused across all BP-OSD configs, so its error count is constant by construction. | Every reference answer carries a verified flat or branch-dual receipt or the shot is reported unresolved; a second arm runs the real tier-1-preserving pipeline per config; per-shot reference type, ties and frames logged; rerun. |
| B-01 / P2 §VI D | CONFIRMED (math) | "The pipeline's error count does not depend on the configuration because its answer is the certified minimum-weight correction" is false: tied optima can differ by a logical (ZZ / X1 vs X2). The "floor" language contradicts Table II (32 vs 34). | Invariance claim removed; replaced by what the experiment fixed; "floor" removed. |
| R-01 | CONFIRMED | `tools/recheck_all.py` exits 0 when the flat-certificate file is absent; tree lists default to empty; no expected cardinalities. | Manifest-pinned complete replay: missing / duplicate / extra / count-mismatched records FAIL; a subset mode is explicitly named. Planted-omission tests. |
| R-03 | CONFIRMED | Replay rebuilds weights via float `log` + rounding and rebuilds H, instead of reading the shipped hash-bound `evidence/weights/*.json` and binding records to hashes. | Replay consumes stored weights (vector hash verified) and hash-bound per-shot records; rebuild is a separate exact-agreement test. |
| R-04 / M8 | CONFIRMED | 4 development circuit trees are "re-derivable", not shipped; P2 says "36 stored trees" and "every one of the 23 closed by a tree". | Ship the 4 trees with a replay path, or narrow the text. |
| B-06 | CONFIRMED | `qldpc_exact_all_gate.run_rung` writes `syndrome_valid: True` without computing it. | Computed from stored records (He = s); code fixed for future runs. Frozen campaign NOT re-run. |
| A-01 / B1 / B3 | CONFIRMED (claim) | P1's 99.76% / 99.69% are historical tolerance-rule acceptances without retained witnesses; the NaN hole means acceptance ≠ admissible witness; 2.6e-13 is a sample result only. The released checker enforces exact optimality (842/1,588). Forgery zero never probed the tolerance boundary, non-finite values or graph substitution. | Short term: relabel as historical tolerance-based acceptance. Real fix: a sound rational-ε checker (Lemma 1, exact θ and ε, accept iff ε ≤ declared ε_max) as released default; rerun headline configs storing every receipt and ε; forgery battery rerun with boundary / non-finite / substitution classes. |
| A-02 | QUALIFIES | Rule-level ε estimate divides by min weight (zero weights allowed); summation error unquantified. | Per-receipt exact ε replaces the estimate. |
| B4 | OPEN | Two Willow d=7×30 shots uncertified after the exact tier (gaps 0.80, 0.11); artifacts do not record which shots or the tier's exit reason. | Locate the shots in chunk records; run an independent exact oracle; report converged-vs-capped. |
| M5 / A-06 | OPEN | 74,250 shots lost to OOM (265 chunks) are recoverable from seeds. | Rerun the lost chunks, or show death preceded syndrome-dependent work; report per-configuration rates and aggregation policy. |
| M6 / A-03 | QUALIFIES | Detection reported without false-alarm base rate; refusal ≠ proven suboptimal. | Clean-output refusal rate beside sensitivity; outcome states EXACT / EPSILON(ε) / PROVEN_SUBOPTIMAL / NOT_PROVEN / INVALID_INPUT / VERIFIER_FAILURE. |
| M7 / A-04 | QUALIFIES | Stale-calibration "detection" needs an independent reference model. | Reframed as objective-mismatch detection against an independently supplied reference; define drift factor; report discordant cells, not "rate unchanged". |
| M8 (P1) | OPEN | BP-OSD gap scoring objective (hypergraph vs decomposed graph) must be stated; confirm no negative gaps. | Stated; `frac_below_zero` = 0 already recorded — cite it. |
| M9 (P1) | QUALIFIES | Equal-weight different-coset tie-break change passes the certificate. | Listed under "not claimed" (or frame check). |
| M10 (P1) | QUALIFIES | Willow result is PyMatching on Google's DEM, not Google's decoder; 99.69% mixes gen-4 and gen-6. | Abstract says so; per-generation split reported. |
| B2 (P1) | QUALIFIES | PyMatching 2.4.0 discretises weights internally; exact-checker refusals mix decoder inexactness with producer slack. | State the decoder objective; split the column. |
| B-03 | CONFIRMED | Circuit rung is 150 raw shots, not a nontrivial target. | Text corrected. |
| B-04 / M7 / m17 | QUALIFIES | Commit-derived seeds are not an external beacon; freeze is self-controlled. | Wording; external timestamp for future freezes. |
| B-05 | QUALIFIES | 99.15% includes 222 repairs; directly certified BP-OSD = 1,780 (88.16%). | Split shown wherever the rate appears. |
| B-07 | QUALIFIES | Circuit agreement is one declared observable; 0/149 ⇒ one-sided 95% upper ≈ 1.99%. | Stated. |
| M1 / M6 (P2) | QUALIFIES | Abstract leads with the stated-baseline advantage; "because min-weight ≠ ML" at p=0.07 unsupported (32 vs 34). | Lead with the tier-1 instrument; drop the "because". |
| M2 / M3 (P2) | QUALIFIES | Independence self-attested (same author, AI assistance, same workstation); 8 hand-picked mutants ≠ a mutation score. | Precise independence statement; automated mutation score; consider different-language checker. |
| M4 (P2) | QUALIFIES | 58 non-byte-identical flats: "ruled out" overclaims. | Wording. |
| M5 (P2) | OPEN | BP-OSD's own logical outcome on the 149 circuit shots not reported. | Report from records. |
| R-02 | CONFIRMED | Historical revisions (freeze `aba2aec`, `d4a81ee`, …) are not in the public release. | Publish a git bundle / archive of required revisions; separate receipt replay from experiment reproduction. |
| C-01..C-03 | QUALIFIES | "One pass over edges" needs certificate size; forgery counts and checker agreement are not universal guarantees; add Sahay–Williamson–Brown (arXiv:2602.22770) and sharpen Shen–Zhong / Krishnamoorthy comparisons. | Wording; citation fetched and verified. |
| m20 | RESOLVED | "Repo not public": it is public (unauthenticated HTTP 200 since 2026-09-14); not yet search-indexed. | — |

## Order of work

1. Release/experiment contracts (no heavy compute): R-01, R-03, R-04, B-06, B-02 (+ rerun), B4.
2. Sound rational-ε checker as default + forgery battery with boundary classes.
3. Compute-bound reruns: lost chunks; headline configs under the ε checker with retained receipts.
4. Manuscript revision against this register; release rebuilt; historical-revision archive.

## Status — 2026-09-14, 21:40

| ID | Status | Evidence |
|---|---|---|
| B-02 / B-01 (P2) | RESOLVED | `qldpc_strong_baseline_v2.py`: every reference answer receipted (flat or branch-dual tree), the real tier-1-preserving pipeline run per configuration. Pipeline errors change with the configuration (p=0.05: 9 / 9 / 8; p=0.07: 34 / 34 / 36) because tied optima differ in logical frame; invariance and "floor" withdrawn from P2 (`fe2d610`). |
| R-01 | RESOLVED | `tools/replay_receipts.py`: pinned manifest, complete replay of 2,042 receipts, missing / duplicate / extra / altered all fail; 9 planted-defect tests. |
| R-03 | RESOLVED | Persisted hash-bound instances and stored integer weights consumed by replay; rebuild is a separate exact-agreement check. |
| R-04 / M8 | RESOLVED | The 4 development circuit trees shipped and replayed (`qldpc_dev_refsched_trees.py`). |
| B-06 | RESOLVED | Validity computed from records: 3,740 checkable records all valid; 299 circuit records store no syndrome. Code fixed. |
| NEW (found while fixing) | RESOLVED | `qldpc_check` read syndrome entries as `int(b) & 1`, so 3 read as 1. Every entry point now refuses non-binary entries; the new tests fail 5/5 against the old checker (`fe2d610`). |
| A-01 / B1 / B3 | IN PROGRESS | Sound rational-epsilon checker (`8a222cc`). Rerun with every receipt retained: simulation d=5x25 at 3,450 / 3,992 chunks; the twelve Willow configurations queued behind it. Replay with a second, integer-arithmetic verifier plus boundary (refused at ~1e-15 past eps_max), inside-boundary and graph-substitution attacks (`3eabe6e`). |
| A-02 | IN PROGRESS | Per-receipt exact epsilon replaces the rule-level estimate once the rerun merges. |
| B4 | RESOLVED | Both residual d=7x30 shots located (X shot 26714, Z shot 5302): PyMatching's correction is exactly optimal per the exact oracle; today's exact tier stops at its 200-round cap on both. Producer slack at a budget, not decoder error (`3eabe6e`). |
| M5 / A-06 | IN PROGRESS | The epsilon rerun covers every chunk of the original seed plan, retried until complete; the historical "independent of syndrome content" sentence is removed with the results rewrite. |
| M6 / A-03 | RESOLVED | `detection_outcomes.py` (runs of record reproduced): 4,861 defective = 3,549 INVALID + 1,312 PROVEN_SUBOPTIMAL, none certified, none NOT_PROVEN; base rate on 770 unmodified corrections 1 refusal (base ladder) / 0 (full ladder); 62 equal-weight controls all certified. |
| M7 / A-04 | RESOLVED | Reframed as objective mismatch against an independently supplied reference; drift factor defined; 547 suboptimal all proven, 2 false alarms in 6,147; discordant cells reported (d=3, drift 3.0: 7 vs 7 errors hide 1 stale-only and 1 reference-only failure). |
| M8 (P1) | RESOLVED | Gap objective stated in P1; the generator refuses any negative gap. |
| M9 (P1) | RESOLVED | Tied optima listed under what is not claimed. |
| M10 (P1) | RESOLVED | Split reported: 99.38% over 99,384 shots (ten configurations, generation 4) and 99.998% over 100,000 (d=7x30, generation 6). Abstract wording lands with the epsilon rewrite. |
| B2 (P1) | RESOLVED | 14 of 770 unmodified PyMatching corrections exceed the exact optimum by a positive amount at most 1e-6; stated beside the exact-checker refusals. |
| B-03, B-04, B-05, B-07, M1 / M6 (P2), M4 (P2), M5 (P2), C-01..C-03 | RESOLVED | Text corrected in P2 (`fe2d610`); P1 complexity wording corrected. |
| M2 / M3 (P2) | PARTIAL | Independence stated precisely and the eight mutants described as a targeted set, not a score. An automated mutation score and a different-language checker for P2 are not done. |
| R-02 | OPEN | Historical revisions not yet published as an archive. |

## Status — 2026-09-14, 22:30

| ID | Status | Evidence |
|---|---|---|
| A-01 / B1 / B3 | IN PROGRESS | Simulation d=5x25 complete under the sound checker (`80d16e1`): 998,145 of 1,000,000 (99.81%) proven within eps_max = 1e-6, 8,628 of them exactly, 1,855 not proven; largest proven slack 6.2e-12; 8,000,001 forgeries in nine classes, none accepted. Complete two-verifier replay with boundary battery running. Willow: d=3x13 X complete (9,841 of 9,841, the run of record's nontrivial count); eleven configurations queued. Paper 1 typesets a rerun only once its complete replay exists (`560cd75`). |
| M5 / A-06 | RESOLVED (simulation) | Every chunk of the original seed plan is present; no shot was lost, so the selection concern no longer applies to the reported rate. |
| M2 / M3 (P2) | RESOLVED (score) | `tools/mutation_score.py`, eight standard operators at every eligible site: `qldpc_check.py` 313 of 321 killed, the other 8 declared equivalent with reasons and verified (none killed, none stale); `matching_cert_epsilon.py` 93 of 97, 4 equivalent. Getting there took new tests for 6 (epsilon) and 62 (qLDPC) real gaps. A checker in a second language for P2 is still not done. |
| NEW (found by the mutation score) | RESOLVED | The float screen `check_qldpc` admitted negative weights within its tolerance and then proved a non-optimal correction optimal (weight -1e-10 on a free variable). Fixed in `9867322`; the defect test fails on the previous checker. The exact and branch-dual checkers, which decide every receipt, always refused negative weights; the 2,042-receipt replay passes. |
| NEW (commit hygiene) | NOTED | `560cd75` was committed after a piped test run reported failures (the pipe's status was `tail`'s). The failures were uncommitted paper-2 work in the tree, fixed in the next commit; test runs now gate on pytest's own exit status. |

## Correction — 2026-09-14, 23:00

Commit `3bd2441` stated d=3x30 X as 9,994 of 9,994 proven and d=3x13 Z as 9,560 of
9,560. The summaries say 9,993 of 9,994 and 9,559 of 9,560: one shot in each was
not proven. The error was reading the replay tool's count of receipts re-decided
(which includes not-proven receipts) as a count of receipts proven. Paper 1 is
unaffected: its numbers are generated from the summaries. Both shots were accepted
by the original tolerance rule and are refused by the sound checker, which is the
distinction finding A-01 is about.

| ID | Status | Evidence |
|---|---|---|
| M2 / M3 (P2) | RESOLVED | Operator mutation scores (`918bea2`) and a checker in a second language: `verifier-js/verify-qldpc-flat.mjs` accepts all 2,002 flat receipts and refuses all 104 planted defects in 13 classes, each for its own reason, with the lattice boundary decided exactly; ten broken copies of it all make the harness fail. Trees have no checker outside Python. |

## Status — 2026-09-15

| ID | Status | Evidence |
|---|---|---|
| R-02 | RESOLVED | `tools/revision_snapshots.py` builds `oneq-source-<commit>.tar` for the five producing revisions (`aba2aec`, `d4a81ee`, `cce1e91`, `06344ae`, `f57e87d`): each entry script and its import closure, read from that revision's git objects and pinned in `evidence/revision_snapshots.json`; `tests/test_revision_snapshots.py` rebuilds them byte for byte and refuses a tampered one. Published as assets of release v2026.09.15 (https://github.com/coherence-energy-labs/oneq-certificates/releases/tag/v2026.09.15). |
| A-01 / B1 / B3 | IN PROGRESS | The simulation and ten of the twelve Willow configurations are complete, replayed by both verifiers, archived under pinned sha256 and published as assets of the same release. Willow d=7x30 X and Z are running on three workers, 200 chunks each: X had 7 chunks after 190 minutes (about 27 minutes each while this session's test suites shared the machine), so roughly 70 to 90 hours per basis, X then Z. |
| NEW (found by rehearsing the public tree) | RESOLVED | A copy of the minted bundle with nothing beside it failed four of its own documented commands, each passing in the development repository only because the repository supplied what the bundle lacked: `verify_release.py` could not import `sign_release`; `recheck_all.py` found a bound file altered, because `qldpc_heldout_flat_certs.py` wrote CRLF on Windows and the bundle normalised the shipped copy to LF; `pack_epsilon_receipts.py`, named in REPRODUCE.md, was not shipped; `make_numbers*.py` read killing tests that were not shipped. Fixed at the producer and in the bundle contents, and the builder now refuses each class (`9ad8862`). `tools/rehearse_public_release.py` runs every documented command on such a copy: 5 of 19 steps failed on the defective mint, and all passed on the fixed one. |
| NEW (line endings) | RESOLVED | LF normalisation made five pins in one mint name bytes no reader receives, two of them in the immutable held-out preregistration (`evidence/environment.json` and `qldpc_bb_refsched_gate.py`, pinned by their CRLF digests) and two in the mutation score's record of its killing tests. A file another shipped file pins by digest now ships byte for byte, and the builder refuses a normalisation that breaks a pin. |
| NEW (evidence ratchet) | RESOLVED | `detection_outcomes.py` and `willow_residual_shots.py` wrote evidence with no test naming them, and the repository's ratchet refused them. Tests added (`2609e44`): the real Judge on a graph known by hand with the soundness tripwire shown firing, and the residual-shot artifact bound to its declared targets, each clause refusing an altered copy. |
| NEW (wall) | BLOCKED | `tools/gate_all.py` refuses to start below its memory floor (10 GB available against 2.7 GB on this machine, held by other sessions), so no current wall census exists. REPRODUCE.md, which ships, states no gate or skip count until a wall runs. The full suite ran instead on two workers with the wall's selection, at `411b8bd` under coverage: 4,403 collected then, 4,399 passed and none skipped. Three of the four failures read state this release replaced (coverage artifacts uncommitted while the suite ran, and the release minted before the fixes) and pass after it (`786cb1c`, `e2243f0`); the fourth is the headline census, which needs a wall run. |
| Release | DONE | Bundle re-minted and signed by the pinned issuer (`e2243f0`; archive sha256 `35f99f02bb0ce21cd061fdb0243af6f3ba7cd5058bbfbbcf8952a7ff70c6e6fe`, 373 subjects, bound to `17bedcf742af`); `verify_release.py` passes, the signature verifies independently of the release tools, and the public rehearsal passes on the signed tree and on a fresh clone of the public repository. |
| NEW (found by the full suite) | RESOLVED | Four reds introduced by this session's own fixes, found by the whole suite on two workers (4,402 tests; `b1c1bc3`). The revision-snapshot manifest pinned past revisions' files without naming the revision, so the hash-pin audit checked them against HEAD (41 mismatches); each entry now names its commit and blob and verifies there. The mutation scores pinned two killing tests by CRLF working-tree bytes no clone holds, so a fresh clone would have refused to typeset paper 2's score; pins are now over CRLF-normalised bytes, migrated only after proving each file is the measured bytes and, normalised, the blob at HEAD. Five derived numbers in the generated macros traced to no artifact; tools/derive_claims.py now derives them. A path lint caught the bundle builder. |
| NEW (lint ratchet) | RESOLVED | The coverage run's ratchet measured ruff 36 against a best of 28 and mypy 1 against 0. The eight new ruff findings were statements joined by semicolons in the new `tools/replay_receipts.py`, now split. The mypy error was `float()` of an epsilon acceptance's bound and slack, typed as possibly empty: `certifying_decoder` now refuses such an acceptance instead of casting it away, and a test shows the refusal firing (`411b8bd`). |
| M2 / M3 (P2), last part | RESOLVED | The trees now have a checker outside Python: `verifier-js/verify-qldpc-tree.mjs`, written from the serialized-tree schema and the manuscript's tier BD, accepts all 40 stored trees and refuses 55 planted defects in 15 classes, each for its own reason, with the production checker agreeing on every planted input (`tools/qldpc_tree_js_agreement.py`, `evidence/qldpc_tree_js_agreement.json`). No stored tree closes a leaf by infeasibility, sits on the lattice boundary, or overloads a multiplier, so a hand-built instance whose x3 = 0 branch is provably infeasible exercises those rules both ways; the first harness let two of eight broken copies of the checker pass (the boundary and overload rules were dark), and all eight now make it fail. |
| NEW (rerun died) | RESOLVED | The 15:09 reboot killed the Willow d=7x30 epsilon rerun at X chunk 41 of 200 and nothing restarted it (found 19:00). Relaunched detached (`%LOCALAPPDATA%\oneq_rerun\run_d7x30.ps1`, X then Z, 3 workers, resuming from the chunk index); log `%LOCALAPPDATA%\oneq_rerun\d7x30.log`. |
| Release v2026.09.15.1 | DONE | Re-minted with the tree checker, both literature logs and the review packet (380 files; archive sha256 `f80e1c5d997e3b81209f8abb0813e86961674ecd994a0e3f76c3b74916ffe70b`, signed by the pinned issuer, bound to `2c29f78`); `verify_release` passes and the 19-step rehearsal passes on the signed tree. Public tree pushed as `5f5673e`; https://github.com/coherence-energy-labs/oneq-certificates/releases/tag/v2026.09.15.1 with 20 assets (receipt archives and source snapshots unchanged from v2026.09.15, re-attached). |
| NEW (rerun rate) | NOTED | The relaunched d=7x30 rerun measured a median of 3,676 s per chunk at 3 workers on the loaded box, not the 27 min quoted from the earlier session; restarted at 5 workers (21:03), losing the three in-flight chunks. |
| NEW (issuer registry) | RESOLVED | With `sign_release` shipped, the rehearsal of the signed tree reached the next check and `verify_release` refused: NO TRUSTED ISSUER REGISTRY. The bundle had never carried `ISSUERS.json`, so no reader could ever have authenticated a release from the public tree; the missing import found first had hidden this check. `ISSUERS.json` (public keys only) now ships at the root, and a test holds the name `verify_release` reads to the bundle contents (`17bedcf`). |
