# Referee report, first round, 2026-11-02: paper 1, *Certifying quantum error-correction decoders*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: post after fixes. Do not post as-is.**

The central maths holds up. Lemma 1 is correct, and `check_epsilon` (`src/oneq/matching_cert_epsilon.py`) implements it in exact `Fraction` arithmetic. It refuses NaN, ±inf, negative values, even sets and corrections that don't explain the syndrome. It handles the boundary parity correctly. It checks the correction is a T-join before computing the bound, so no set with an empty boundary can inflate the bound. I found no soundness hole. What fails is the abstract: one unsupported number, one aggregate that mixes two producers, a detection result that is a theorem rather than a measurement, receipts nobody else can get, and a related-work section that understates the three closest papers.

## Findings, most severe first

**1. FATAL: the abstract's cost figure ("about 6,610 times batch decoding"); also line 242 (`\CertifyMsPerShot`, `\CostRatioBatch`).**
- **Problem:** The source file `evidence\decode_vs_certify.json` records no noise level, shot count, commit, producer or machine. Its only keys are schema, rows, the ms value, the two ratios, "correction" and "which_ratio_to_quote".
- **Evidence:** The repo's own ledger says the number reproduces only at p=0.002. At line 5142 of `docs/RESEARCH_LEDGER.md` ("CORRECTION TO R84"): "at p=0.002 the live CertifyingDecoder costs 113.47 ms/shot against the recorded 106.344". A p=0.005 run "gave 1,275 ms". The paper attaches the figure to "the distance-5, 25-round graph", whose only experiment in the paper runs at p=0.005. The same ledger entry says the artifact "should record its noise model, shot count and defect distribution". That was never done, and `numbers.tex` (10-26) still reads it.
- **Also hidden:** the cost is a tail. Refused shots cost 7.4 times an accepted shot and take 39% of total time.
- **What survives:** "offline, not real-time", which only gets stronger.
- **Smallest fix:** time certification inside the ε rerun at p=0.005, on the same shots. Record the conditions, and report median, p99 and refused-shot cost.

**2. MAJOR: the "detection power" results cannot fail.** This covers the abstract, intro bullet 4 (line 48), line 200 and line 215.
- **Problem:** "Defective" is defined by an exact oracle: `excess > EPS_MAX` (`experiments/m0_prior_art/detection_outcomes.py:104`). "Proven suboptimal" means "the exact oracle's" lighter correction (lines 24–26). By Lemma 1, any sound checker refuses every correction whose excess exceeds ε_max, for any packing. So 1,312/1,312 and 547/547 are regression tests of the checker, not power measurements. "Syndrome checking detects none" holds by construction.
- **What is actually measured:** only the false-alarm counts.
  - 1/770: 95% Clopper–Pearson interval 0.003% to 0.72%.
  - 0/770: at most 0.48%.
  - 2/6,147.
- **Smallest fix:** state detection as a corollary of Lemma 1. Say the lighter correction comes from the oracle. Report the false alarms with intervals as the measurement.

**3. MAJOR: the 99.80% hardware headline is a mixture of two producers** (abstract, line 45).
- **Problem:** 100,000 of the 199,384 shots (50.2%) are the hardest configuration (distance 7, 30 rounds). They ran at five times the per-configuration sampling, with the full ladder including the exact tier at "tens of seconds per shot". The other ten configurations use the base ladder and reach 99.60% (`make_numbers.py:91-102`, 250-271).
- **Size of the effect:** if distance 7 over 30 rounds were run with the base ladder at 10k shots per basis, the 12-configuration aggregate would be about 98.7%. This is my estimate, using the generation-4 rates of 93.76/94.71 as a proxy.
- **Wording:** the intro credits the 99.80% to "a producer".
- **Smallest fix:** give the base-ladder and full-ladder rates separately in the abstract, or a configuration-balanced rate.

**4. MAJOR: the hardware rates are measured on the shots the producer was developed on** (lines 150, 166).
- **Problem:** each generation was introduced "against the shots its predecessor left open". The rerun is "on the same shots". There is no held-out hardware sample. The paper also doesn't say whether the base ladder was tuned on the simulation's seed plan.
- **Smallest fix:** certify a fresh sample with the ladder frozen (unused Willow shots or rounds, or new simulation seeds) and make that the headline.

**5. MAJOR: the receipts are not released, though the title and abstract rest on them** ("independently checkable … receipts", "we keep every receipt", "All receipts replay").
- **Evidence:** `data/epsilon_receipts/` is 1.2 GB and is git-ignored (`.gitignore:55`). Section 9's release list (line 260) omits it. No third party can replay the headline.
- **Smallest fix:** deposit the receipts on Zenodo with the per-chunk SHA-256 manifest (`chunks.jsonl`) and cite the DOI.

**6. MAJOR: the second implementation corroborates a different checker** (line 89, "The two agree on all 56 conformance vectors").
- **Which checker the vectors record:** `conformance\cert_vectors.json` holds the verdicts of `oneq.matching_cert.check` (`tools/make_cert_vectors.py:34,57`). `check_epsilon`'s own docstring (lines 3–8) identifies that as the exact-optimality checker, not the ε-checker the results use.
- **The ε-checker's only second implementation** is "verifier B", inside the replay harness (`tools/replay_epsilon_receipts.py`). It is in the same language and the same file, and implements the same lemma.
- **Shared parsing path:** both verifiers receive certificates through `oneq.matching_cert.MatchingCertificate` (line 47). That is exactly the failure class in the paper's own duplicate-set story.
- **No recorded tallies:** the replay JSONs record no per-verifier verdict counts. Agreement is implied only because the harness raises on disagreement (lines 208, 307).
- **I also found no JS file containing `blossom_duals` in the tree,** outside `.build`.
- **Smallest fix:** generate ε-vectors from `check_epsilon` and port the JS checker to them (or say which checker it implements). Have verifier B parse the raw bytes itself.

**7. MAJOR: the related-work section misdescribes the closest papers** (lines 250–252, plus the real-time sentence in the abstract). I checked each against the authors' own LaTeX.
- **Wu et al. (arXiv:2508.04969):** "By producing a certified proximity bound for each syndrome it decodes" and "guarantees optimality for … simple graphs". That is a decoder that emits a certificate for every shot while it decodes. The 6,610× is a property of re-deriving the dual afterwards. The repo's own ledger entry R84 says so in its title: "the producer recovers a dual sparse blossom already computes". The paper never compares against exporting a primal–dual decoder's own dual.
- **Shen & Zhong (arXiv:2609.07035):** the paper says theirs is aimed "at libraries rather than individual shots". Their paper actually offers "An oracle-free suboptimality certificate … Two feasible corrections of different total weight prove the heavier one non-minimal … 639 certificates". Its abstract also says it "catches every one we injected". That is the same witness as this paper's "proven suboptimal".
- **Krishnamoorthy et al. (arXiv:2608.25545):** the paper says their certificates are "internal to a sampler". Their source reports "machine-checkable optimality proofs" and "deterministic certificates, at 272 s per decode".
- **Fix:** rewrite that paragraph accurately and narrow the novelty claim.

**8. MAJOR: "Checking … is cheap" (line 242, Fig. 1) is not reconciled with the repo's own measurement.**
- The ledger's same-run measurement (`RESEARCH_LEDGER.md:5142`) is 23.74 ms per check against 113.47 ms to certify. That makes the checker about 21% of the cost and about 1,400 times a decode.
- `checker_scaling.json` implies about 1.3 ms per check on the same graph, an 18× gap.
- Which checker Fig. 1 times is not recorded. Fix: time `check_epsilon` in the same run as the certification timing.

**9. MAJOR: the measured-model section (lines 188–190) uses certification semantics the paper doesn't name.**
- It claims "certificate[s] of exact minimum weight" through `CertifyingDecoder` (`experiments/ibm_live/rep_code_cert.py:52`). That contradicts Section 4's "we do not claim it" without explanation.
- It doesn't say whether the receipts were kept, replayed, or put through the forgery battery.
- It reports an unexplained 89× drop in logical errors (1,517 to 17), which invites the suspicion that the baseline model is badly built.
- "Is known to violate the detector marginals" (line 192) has no citation.

**10. MAJOR (statistics): no uncertainty is stated anywhere.** The small-count claims need intervals:
- false alarms: 1/770, 0/770, 2/6,147;
- controls: 0/62;
- moat growth against the LP: 31/120 versus 113/120.
- The stale-calibration logical-error comparison has only 10 discordant shots (8 against 2, exact two-sided p ≈ 0.11). So "barely register" partly reflects a small sample; the paper should say so.

**Minor**
- **11. Terminology (line 41):** both McConnell's certifying algorithms and Blum–Kannan checkers require completeness. This pipeline refuses correct outputs (0.19% of simulated shots), so "in the sense of" should be "in the spirit of".
- **12. Misattribution (line 130):** equal-rate moat growth that "picks the wrong pairs" is the Goemans–Williamson-style approximation, not Edmonds' blossom algorithm (`edmonds1965paths`), which is exact.
- **13. Forgery count:** 9.6M is mostly eight deterministic mutation classes applied to every shot, 2M of them non-finite substitutions. The most relevant class, `A_suboptimal`, ran on 1 shot (`evidence\epsilon_rerun\sim_d5_r25\summary.json`). The paper itself concedes the count proves nothing, so the abstract shouldn't lead with it.
- **14. Non-proof results still presented:** the IBM 2,296/2,296 run and the generations and gaps tables use the tolerance rule that Section 4 calls "not a proof". Move them to an appendix or rerun under the ε-checker.
- **15. Parallel edges:** `_graph` collapses them to the minimum weight (lines 87–100). On a multigraph, w(J) would then not be the weight of the decoder's actual edges. I didn't check whether any instance has parallel edges.
- **16. Naming:** "Today's stricter checker" (Table I caption) clashes with "the checker the results use". Three checker names are in play. The title says "optimality" where the claim is ε-optimality.
- **17. Wrong source file for two macros:** `\SimDetectors` and `\SimEdges` are read from the cost artifact (`make_numbers.py:77-80`), not from the rerun's instance file.

## Attacks that failed (evidence in the paper's favour)
- **Lemma and checker:** Lemma 1 and the checker's soundness hold, per the code walk above.
- **Numbers:** the macro-to-artifact trace works; the generator stops with an error on mismatches. Simulation: 998,145/1,000,000 accepted, largest ε 6.18e-12. Audit table: 1,588 / 842 / 53% re-add correctly.
- **Citations:** I checked 8 arXiv IDs against the arXiv API (2508.04969, 2608.25545, 2609.07035, 2605.16523, 2606.11496, 2601.16123, 2303.15933, 2408.13687). Authors, author order and titles are all correct. Takou et al.'s source supports "including on IBM hardware", and Stein et al.'s supports the 11.1× figure (their text says 11.11×). The classical entries (McConnell, Blum–Kannan, Steffy–Wolter, Cheung et al., Eifler–Gleixner) look right.
- **Release URL:** the GitHub address returns HTTP 200.
- **Prior audits:** the old JavaScript findings (F-1, F-8 in `docs/EXTERNAL_AUDIT_2026-09-15.md`) are closed. The cost-artifact defect the ledger recorded in September is still open.

## Could not check
- Whether each receipt's correction is PyMatching's own edge set or a path reconstruction.
- Which Willow error model variant was used.
- Which checker Fig. 1 times.
- What the public repository contains, and where the JS checkers live.
- Exactly how verifier B parses its input (I saw only the import).
- Whether the base ladder was tuned on the simulation's seed plan.
- PDF-to-source consistency (the PDF is newer than `numbers.tex`, but I didn't open it).
- The exact format of Krishnamoorthy et al.'s deterministic certificates.

## Memory lines (not written: your task made ONE_Q read-only)
- WORKED on "every fault refused": the weak-duality tautology makes the positive arm undecidable as power; the witness came from the oracle (`detection_outcomes.py:24,104`).
- WORKED on an aggregate rate: decomposing it showed 99.80% = 99.60% base ladder + the 100k-shot full-ladder configuration.
- WORKED on "second implementation": tracing which checker the vectors record (`make_cert_vectors.py:34`, the exact checker, not ε).
- WORKED on an abstract cost figure: checking artifact provenance; `decode_vs_certify.json` has no p, shots or commit (ledger line 5142).
- INSTRUMENT: replay JSONs carry no per-verifier tallies, both verifiers share one parser, and the receipts are git-ignored (`.gitignore:55`).
- FAILED to kill: `check_epsilon` soundness and Lemma 1, and the 8 arXiv citations.

The downloaded preprint sources are in `<reviewer's local scratch path>`, one folder per arXiv ID (for example `ep_2609.07035`).

Sources:
- [arXiv:2508.04969](https://arxiv.org/abs/2508.04969)
- [arXiv:2609.07035](https://arxiv.org/html/2609.07035)
- [arXiv:2608.25545](https://arxiv.org/pdf/2608.25545)
- [arXiv:2405.07433](https://arxiv.org/pdf/2405.07433), which per the search summary uses per-shot matching duals for soft outputs; I did not verify this in its source.
