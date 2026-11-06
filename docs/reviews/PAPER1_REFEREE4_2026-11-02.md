# Referee report, fourth (final read) round, 2026-11-02: paper 1, *Certifying quantum error-correction decoders*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: NOT READY.** Two things block posting: the PDF cites a release that does not exist, and one statistical bound in the defect paragraph is not supported by the data. A third fix belongs in the same edit: the names of the checkers in that paragraph. No headline number is affected. Every headline rate, interval and count I recomputed reproduces.

I edited no files. That includes agent memory, following the round-3 referee's reading of "read-only". Line numbers refer to `paper/certifying-decoders/main.tex`.

## Blocker (from round 3, still not fixed)

**B1. The release is still UNSET** (Sec. XI, line 270).
- `paper/release.tex:4` still holds `\ReleaseTag{UNSET}`.
- The built PDF, page 10, reads: "oneq-certificates (release UNSET). Every receipt of Sec. VI … is attached to that release". That sentence is false today.
- `main.tex` also has uncommitted edits. Its last commit, `c794dcf0`, is from 17:49; the file was modified at 21:10.
- **Fix:** mint the release with every receipt archive, set the tag, rebuild with `build.py --release` (it refuses while the tag is UNSET), and commit the manuscript.

## MAJOR

**M1. The audit does not support "below 0.19% (one-sided 95%)"** (line 129: "which bounds the inadmissible share among those runs' acceptances below 0.19% (one-sided 95%)").
- **The arithmetic is right; the inference is not.** 0.19% is 1 − 0.05^(1/1588) (`make_numbers.py:758`). That bound only holds if the 1,588 certificates are a simple random sample of the population. They are not.
- **Most of the sample is outside "those runs".** Table I shows 1,000 of the 1,588 (230 + 392 + 378) are the whole gap experiment of Sec. VIII. "Those runs" are the original runs whose rates Sec. VI keeps.
- **The rest covers three strata only:**
  - the simulation, 399 of its 997,594 acceptances;
  - Willow d=5, 13 rounds, X basis: 149;
  - Willow d=7, 30 rounds, X basis: 40.
- **Never sampled:** generations 1–3 and 5, ten of the twelve Willow configurations (every Z basis among them), and the IBM run.
- **Counterexample:** suppose 0.5% of the simulation's acceptances were inadmissible and everything else was clean.
  - The chance the audit sees 0 of 399 is 0.995^399 = 0.135, so the data do not exclude this at 95%.
  - The inadmissible share would then be 4,988 / 1,198,660 = 0.42% at final generations, or about 0.25% over all generations. Both exceed 0.19%.
  - Even pooling only the 588 certificates from Sec. VI runs gives 0.51%.
- **What survives:** none of the 1,588 was inadmissible. The gap experiment was audited in full, so for Table IV the share is exactly 0. Per sampled stratum, the one-sided 95% bounds are 0.75% (simulation, n = 399), 2.0% (Willow d=5, 13 rounds, X) and 7.2% (Willow d=7, 30 rounds, X). Everything not sampled is not bounded by the audit.
- **Fix:** replace the clause with that sentence.

## MINOR

**m1. Checker names: a reader cannot tell which checker decided what. Fix before posting.**
- **One function, five names.** `matching_cert.check` appears as:
  - "the exact checker" (Table I caption, lines 127, 196 and 240, Fig. 3);
  - "the rational-repair checker" (line 147);
  - the conformance suite's "exact checker" or "Python checker" (lines 92 and 129);
  - "the production checker" (line 129 only);
  - "the tolerance-rule checker", before 2026-08-04.
- **Evidence for the mapping:**
  - `tools/make_cert_vectors.py:34` and `:149-153`: the conformance vectors and the three forgeries are recorded with this `check`.
  - `src/oneq/matching_cert.py:453, 834, 940`: its rational repair matches line 127's description.
- **"The strict checker"** is `matching_cert_strict.check_exact_cut`. It appears once (line 129) and is never tied to any number.
- **The natural misreading is "strict = exact".** That reading gives the checker behind Table I's exact columns, Sec. VIII, Fig. 3, Sec. VI-E's 40,963 certificates and the 59 vectors the milder defect (it accepted two forgeries). In fact that checker accepted all three. Under the misreading, "the two implementations had disagreed on these inputs" is false for the third forgery; under the true mapping it is correct.
- Line 117's "The released checker" is also ambiguous: the release contains three Python checkers.
- **Fix:**
  - At line 127, name it: "(`matching_cert.check`, the production checker; before 2026-08-04 it applied the tolerance rule)".
  - At line 129, gloss the strict checker and say what, if anything, it decided.
  - At line 117, write "The ε-checker".

**m2. The abstract's forgery claim is unscoped** (line 31). "Every forged certificate tried was refused" conflicts with line 129, where the production checker accepted three forged certificates and the strict checker two. **Fix:** "…tried against them was refused".

**m3. The forgery count is outside its sentence's scope** (line 163). The sentence counts each configuration "at its final generation", but the macro is `forgeries_attempted_gen4_gen5_gen6` (`make_numbers.py:112`): 751,060 + 4 × 300,000. **Fix:** "…tried at generations 4–6…".

**m4. Line 155 misdescribes how Table III was measured.** It says each generation was "measured against the shots its predecessor left open". But generation 4 ran 9,560–10,000 nontrivial shots per configuration, including d=3 at 13 rounds, where generation 3 left about 1 shot open. Lines 139 and 173 correctly say "introduced against". **Fix:** "…introduced against the shots its predecessor left open and measured on a sample of all shots".

**m5. Two different exact counts for the same shots.**
- Line 240 gives 227 / 383 / 253 certified on the gap shots under the exact checker (`gap_distribution_exact_rule.json`, commit `624d41f2`).
- Table I's last column gives 228 / 388 / 292 for the same shots and checker (`certification_semantics_audit.json`, commit `0052b997`). That audit's base-ladder counts are 227 / 380 / 246, so 253 matches neither ladder.
- **Fix:** name the producer and ladder used at line 240, and say why its counts differ from Table I.

**m6. An unlabelled tolerance-rule rate** (line 135). The 31/120 against 113/120 comes from `growth_vs_lp.json`, committed 2026-07-31. That predates the exact repair (2026-08-04) and the ε-checker (2026-09-14). Line 117 promises such rates are labelled wherever they appear. **Fix:** add "(tolerance rule, development)".

**m7. Sec. III step (iv) does not match the Lemma or the code** (line 82). It says the checker scales by "the largest factor that makes every edge constraint hold". The Lemma and `src/oneq/matching_cert_epsilon.py:159-163` cap that factor at 1. **Fix:** "the largest factor not exceeding 1 (θ of Sec. IV)".

**m8. Cost-section figures** (line 252).
- "129 µs per call" is the median (`make_numbers.py:700`), but the sentence says "mean cost of decoding". The mean is 131.6 µs.
- The typeset 53 ms / 16 µs gives 3,312, while the text says about 3,200. The true ratio is 52.60 / 16.35 = 3,217.
- Neither changes a ratio at two significant figures. **Fix:** typeset 132 µs (mean) and 16.4 µs.

**m9. The abstract's 99.76% is a rounded-down 99.770%** (19,952 of 19,998). A reader can reconstruct 99.770% from Table II's 25 + 21 not proven, so it looks like an error. Round 3 raised this as m5 and it is still open. **Fix:** say "rounded down" once in Sec. VI, or typeset 99.77.

**m10. Intro section references** (lines 49–50). The scaled-integer verifier is described in Sec. IV, and the reruns are reported in Sec. VI. **Fix:** cite "(Secs. III, IV, VI)" for bullet 2 and "(Secs. IV, VI)" for bullet 3.

**m11. A novelty claim without kept witnesses** (line 200). The paragraph presents a "third-party-checkable witness attached to a model estimated from the shots" as new, while line 196 says this section's certificates were not kept or replayed. **Fix:** add "this section produces such witnesses but did not keep them".

**m12. Wording.**
- Line 173: "two qubit placements" reads as "two-qubit placements".
- Line 147: "after the review" is ambiguous after three rounds; give the date.

## What I checked that held

**Internal consistency:**
- **Intervals recomputed exactly (Clopper–Pearson):** base-ladder false alarms (1/770), 0.00329–0.721%; full ladder (0/770), at most 0.478%; equal-weight control (0/62), at most 5.78%; stale calibration (2/6,147), 0.00394–0.117%; held-out (91/20,000), 99.44–99.63%; fresh simulation (19/10,000), 99.70–99.89%.
- **Other statistics:** the binomial p for 8 against 2 is 0.109; the per-round rates are 1.77, 2.53 and 1.82 × 10⁻⁴.
- **Table I:** 1,588 accepted, 842 exact, largest slack 2.6e-13.
- **Table II:** N sums to 99,384 + 100,000; not proven to 395 + 2 = 397; exact to 44,157; development shots to 79,384.
- **Table V:** 2,049 shots per drift; 271 suboptimal at drift 3.0; discordant 8 against 2; 7 and 7 logical errors.
- **Other sums:** 998,145 + 1,855; 19,909 and 91; 8,000,001 forgeries = 8 × 10⁶ + 1; 98,772 + 99,998 = 198,770; 4,861 + 1 + 62 = 4,924 cases; 93 + 4 = 97 mutants.
- **Percentages:** 99.81, 99.80, 99.60, 99.998, 99.09, 99.54, 99.69, 99.38, 53% and 50%.
- **Cost:** the ratios recompute from `certify_cost_d5_r25.json`.
- **Graph size:** 600 detectors is right for a d=5, 25-round memory experiment.

**Claims and scoping:** apart from M1, m1, m2 and m11, nothing else overclaims.
- Fig. 1 was measured 2026-07-31, before the ε-checker existed (first commit 2026-09-14), so its caption holds.
- The node-potential refusal is real (`matching_cert.py:1088`).
- The defect paragraph's forgery weights and dates agree with round 3.

**Carelessness:**
- Every `\ref` resolves to the intended section.
- There is no "??" anywhere in the PDF.
- The PDF (21:10:58) is newer than `main.tex` and `numbers.tex`.
- `build.py` enforces the 1,920-character abstract limit.

**Not checked:**
- which function decided Sec. VI-E's counts (its evidence file records no checker);
- the JavaScript checker's behaviour (not run);
- whether 74,250 lost shots over 265 chunks (280.2 per chunk) reflects mixed chunk sizes;
- the visual layout of tables and figures.

**What makes it READY:** mint the release, rebuild, commit, and rewrite the two line-129 sentences (M1 and m1). Everything else is a short wording edit.

```
STRONGEST SURVIVING RESULT  Every headline number, interval and count reproduces from its macro or evidence;
                            the only false inference is the pooled 0.19% bound
EVIDENCE CLASS              MEASURED (recomputed) + DERIVED (counterexample population)
HOW IT COULD STILL BE WRONG Sec. VI-E's checker identity and the JavaScript checker were not run or verified
WHAT IT LICENSES            Posting after B1, M1 and m1
KILLED                      "audit bounds inadmissible share among those runs' acceptances < 0.19%" ->
                            0.995^399 = 0.135 (0.5% bad in the simulation not excluded) -> per-stratum bounds 0.75% / 2.0% / 7.2%
OPEN BRANCHES               the producer used for Sec. VIII's exact-rule counts (one evidence read);
                            the checker behind Sec. VI-E (read its lane's call site)
RECOMMENDATION              SHIP after B1 + M1 + m1
```

Memory lines I would have saved:
- Pooled Clopper–Pearson bound over a stratified sample: check the sampling frame (63% of this sample was a census from another section).
- Name mapping: one function, `matching_cert.check`, is called "exact", "production", "rational-repair" and "Python" checker.
- A sum macro's key name (`gen4_gen5_gen6`) can contradict the scope of the sentence it sits in.
- `release.tex` was still UNSET at the 21:10 build.
