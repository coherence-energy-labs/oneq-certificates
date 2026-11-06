# Referee report, third round, 2026-11-02: paper 2, *Exact, independently checkable optimality receipts for BP-OSD*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

I found no fatal problem. There is one major one: the paper's own invariance result undermines its reading of configuration B, and that was round 1's finding 1, explained but not fixed. The four new parts survived, with corrections to wording and mechanism.

I did not touch the repository. Python ran with `-I -B` so it wrote no `.pyc`, and I did not write agent memory. The scratch script and its output are `<reviewer's local scratch path>` and `...\review_p2c\jitter_confound.json`. The script runs in 3.1 s. It reproduces the published A, B and A_s hit counts exactly (p=0.05: 84 / 87 / 197; p=0.07: 24 / 171), so the attacks below run on the paper's own decoder.

## Verdict: post after fixes

| Focus item | Result |
|---|---|
| 1. Post-hoc held-out re-analysis | **Confirmed.** The replay control is real and can fail; the numbers and McNemar values are right. |
| 2. Why unscaled min-sum depends on the declared rate | The exact-arithmetic argument is **confirmed**. "Breaking exact ties removes most of the difference" survives two confound controls I added. The tie statistic quoted as the mechanism does not tell a failing decoder from a working one. The paper's own result **refutes** its p=0.05 vs p=0.07 reading of B (MAJOR 1). |
| 3. Tier-1 and one-node-tree accounting | **Confirmed**: counts and both producers' rules. One wording overclaim. |
| 4. 0.625 as the Sinter default | **Confirmed.** |

## FATAL
None.

## MAJOR

**M1. The B column measures the floating-point form of the declared error rate, not iterations, OSD order or noise level.**

- **Where:**
  - main.tex:98: "at p=0.05, 100 iterations and OSD-CS of order 10 raise its optimum hits from 84 to 87 … at p=0.07 both matter, from 24 to 171".
  - main.tex:141: "scaling alone repairs A and B, which at p=0.05 commit 114 and 109 errors unscaled", and "Scaling is not uniformly better: … B 200/200/199, B_s 199/197/198 … B_s commits 1 logical error where B commits 0".
  - The B columns of Tables `tab:strong` and `tab:instrument`.
- **What is wrong:**
  - The paper itself argues (main.tex:141) that unscaled min-sum + OSD-CS cannot depend on the declared rate in exact arithmetic. So declaring 0.05 or 0.07 gives the same decoder, and which of B's two behaviours appears depends only on how the prior log-likelihood ratio rounds in floating point.
  - Declared 0.05 falls in a "bad" group of rates and 0.07 in a "good" one. The authors' own scan shows this: `evidence/qldpc_bposd_scaling_sensitivity.json`, p=0.05, B gives 109 failures when told 0.05 and 12 when told 0.07.
  - The p=0.07 level was never scanned, so this artifact could not show up there.
- **Witness:** `python -I -B <reviewer's local scratch path>`, run on the stored v3 syndromes and scored against each shot's certified optimum:
  ```
  p0.05 B a=1 plain @0.05: hits 87  fails 109 conv 81    <- paper's B
  p0.05 B a=1 plain @0.07: hits 196 fails 12  conv 177
  p0.05 B a=1 plain @0.03: hits 194 fails 10  conv 179
  p0.07 B a=1 plain @0.07: hits 171 fails 41  conv 121   <- paper's B
  p0.07 B a=1 plain @0.05: hits 20  fails 175 conv 18    (same at 0.035 and 0.08)
  p0.07 A a=1 plain, every rate 0.03-0.08: hits 21-24
  ```
  - Swapping two rates that are equivalent in exact arithmetic reverses the conclusion at both levels:
    - At p=0.05, more iterations and a higher OSD order take B to 196 hits, about B_s's 197.
    - At p=0.07, they take it to 20, below A's 24.
  - The low-p "scaling is not uniformly better" contrast is the same effect. In the sensitivity JSON, B on the p=0.02 shots commits 17 errors when told 0.05 and 0 when told 0.02.
  - So this sentence is the round-1 finding-1 anomaly again, and the round-2 minor 4 ("24→171") that the revision adopted was itself this artifact.
- **What survives:**
  - Configuration A is poor at every declared rate tried: 84–88 of 200 hits at p=0.05 and 21–24 at p=0.07.
  - Scaling 0.625 (197 / 196 hits) or a 10⁻⁶ per-bit prior jitter (185 at p=0.05) recovers almost all of it.
  - The held-out attribution, which uses configuration A, therefore stands.
  - The effect of iterations and OSD order at scaling 1, and how it changes with p, cannot be identified from these runs.
- **Fix (in how the arms are measured):**
  - Report every scaling-1 arm as a distribution over exact-arithmetic-equivalent declared rates. Extend the existing 16-rate scan to every level, and count optimum hits as well as failures. Alternatively, report the tie-broken decoder.
  - Delete "at p=0.07 both matter" and "Scaling is not uniformly better" unless they hold across that scan, and footnote the B columns.
  - Add a check that refuses any scaling-1 comparison whose sign flips across the scan.

## MINOR

1. **The quoted mechanism statistic does not separate the arms (main.tex:141).** "143.5 of the 144 soft outputs share their magnitude exactly with another bit (71.5 at 0.625)". But the scaled decoder ties on 71.5 bits per shot and fails only 7 times.
   - The variables that do separate the arms (same script) are BP convergence and exact-zero soft values:

     | p=0.05 | plain (scaling 1) | jittered | scaled 0.625 |
     |---|---|---|---|
     | A, BP converged | 81 | 133 | 172 |
     | B, BP converged | 81 | 181 | 174 |
     | A, shots with an exact-zero soft value | 125 | 0 | 0 |

   - Every logical failure in every arm is on a shot where BP did not converge; the only exception is a single failure for jittered B.
   - On non-converged shots, A fails on 114/119 plain, 20/67 jittered and 7/28 scaled. So the jitter works mainly by letting BP converge, and secondarily through OSD on the remaining shots.
   - The macro `\InvZerosPerShotUnscaled` is generated but never used.
   - Fix: report convergence and exact zeros instead of the tie count.
2. **Poulin & Chung citation.** It fits the idea: perturb the prior to break symmetric degeneracy so BP converges, which is what the convergence data show. But it is attached to the wrong statistic. Their procedure, from memory and not verified, is a random perturbation of qubits on unsatisfied checks, re-run on failure; this paper uses one fixed global relative 10⁻⁶ jitter. Say "in the spirit of".
3. **"Declared rates whose ratios differ by a power of two" (main.tex:141) is misworded.** It is the prior log-likelihood ratios that are 1, 2, 4 and 8. The rates (0.269, 0.119, 0.0180, 0.000335) do not have power-of-two ratios.
4. **Jitter wording overclaims for B.** "Independent of the declared rate (… 180 for B)" and "cuts … B's to 8–15 at every declared rate":
   - 20 of B's shots still depend on the rate under jitter, so exact ties are not the only route.
   - At the "good" rates, jitter raises B's failures (told 0.03: 10→14; told 0.1: 9→11).
5. **"The remaining 17 needed branching" (main.tex:141).** The stored trees branched, but necessity is not shown. Say "were closed by branching trees".
6. **Abstract "traced this to the default min-sum scaling factor, 1".** This is defensible, but the paper's own jitter result (unscaled A with a 10⁻⁶ tie-break reaches 185/200) supports "unscaled min-sum, whose messages tie exactly" as the more precise cause.
7. **Rationale for 0.625 is worded inconsistently.** main.tex:139 says "taken for that reason and not tuned"; `evidence/qldpc_heldout_scaled_posthoc.json` says "fixed on the strong-baseline data". Harmonize them.
8. **Companion manuscripts (main.tex:36, :123)** have a title but no citation or identifier. This round-1 minor is only partly fixed; I did not find the bib file.
9. **Tier 1 under the stated rule is given only for C at p=0.07.** `qldpc_root_dual_classes.json` already has the other configurations (B 20; A_s/B_s 33 at p=0.07 and 10 at p=0.05), and `\StrongFiveTierOneRuleC` is unused.

## Checks that found no problem

- **Held-out replay control.**
  - The script runs the campaign decoder at scaling 1 and refuses to write if any regenerated syndrome, stored candidate or recorded logical flag differs (`qldpc_heldout_scaled_posthoc.py`:114–117).
  - The artifact records 300/470/200/300 reproduced on all four controls.
- **Logical outcomes and McNemar.**
  - The counts are consistent: scaled errors 8 = 3 + 5 both wrong, and pipeline errors 5 = 0 + 5.
  - The two-sided McNemar for (3, 0) is 2·2⁻³ = 0.25; the recorded (170, 0) is 2·2⁻¹⁷⁰ = 1.34×10⁻⁵¹.
- **Held-out tier totals, from the per-shot records.**
  - Tier 1: 1,780 = 116+299+150+149+122+300+450+194.
  - Tier 2: 222 = 31+1+164+20+6, and in every one BP-OSD's recorded objective is strictly heavier. The 31 circuit records carry no validity flag.
  - Trees: 17 = 2+1+14, of which 8 are one-node (all on the stress dataset, none with BP-OSD optimal).
  - Derived figures check: 2,010 needed no branching (99.55%); 1,783 = 1,780+3 BP-OSD answers proven optimal (88.31%).
  - For C at p=0.07: 50 optimal answers on tree shots, 33 accepted by a one-node tree's root dual, so 146+33 = 179 (89.5%).
- **Tier-1 rules.**
  - The held-out producer (`qldpc_exact_all_gate.py`:150–158) issues tier 1 whenever the shot's dual accepts BP-OSD's answer. Integrality only gates tier 2 (`elif primal is not None`), and `feldman_dual` returns the dual either way (`qldpc_cert.py`:342, :401). So the held-out tier 1 is the stated rule.
  - The strong-baseline producer (`qldpc_strong_baseline_v3.py`:181–183, :241) needs a stored flat proof, which exists only for an integral primal. The text is accurate.
- **Sinter default.** `ldpc/sinter_decoders/sinter_bposd_decoder.py`:41 sets `ms_scaling_factor=0.625` (also the LSD and belief-find wrappers). The `BpOsdDecoder` class default is 1.0 (`_bposd_decoder.pyx`:53).
- **Invariance argument.**
  - Min-sum updates, the hard decision and the OSD ordering are all homogeneous. The library decides a bit as an error when its soft value is ≤ 0; I measured that on 200/200 shots, and that rule does not depend on scale.
  - The power-of-two test gives 200/200 identical answers.
- **Jitter confound controls.** Only the per-bit jitter changes anything (116 and 114 answers changed for A and B), so tie-breaking is what the intervention does:
  - Passing uniform probabilities through the same `error_channel` code path the jitter uses gives answers identical to the plain decoder, 200/200, for A and B.
  - Shifting every prior by the same relative 10⁻⁶ (new low-order bits, ties kept) also gives identical answers, 200/200.
- **Other quoted numbers** all match the JSON: 10/109, 9–109, 85 shots with one answer, 9–11 failures at scaling 0.9 / 0.75 / 0.625, and the jitter figures.
- **Earlier rounds, fixed in the text:**
  - Round 1: F2, F3, F4, plus the import-list, abstract noise-model, preregistered-label and circuit-distance minors.
  - Round 2: M1, M2, and minors 1, 2, 5, 6, 7, 8.
  - Not fixed: round-1 F1 and round-2 minors 3 and 4, which were "fixed" with sentences that are themselves the artifact in M1.

## Could not check

- The Poulin–Chung text, so the description of their procedure is from memory.
- The ldpc OSD-CS candidate cost: the installed package ships no C++ source. If it ranks candidates by Σlog(1/p_i), the jittered decoder's OSD stage is not exactly homogeneous. This is immaterial in practice (A is 200/200 independent of the rate under jitter).
- I did not re-run the held-out post-hoc script itself; the round-2 referee re-decoded it.
- Whether config A is insensitive to the declared rate on the held-out syndromes themselves; I checked only on strong-baseline shots.
- The circuit, heterogeneous and phenomenological datasets under scaling; the contents of `tab_*.tex`; whether round 1's suggested per-weight consistency check was added; and `make_numbers` reading its pass/fail flags from the artifact it checks.

**Memory lines, not written because the repository is read-only:**
- Paper 2's B-column reading (p=0.05 vs p=0.07, "scaling not uniformly better") is refuted: the rounding of the declared prior decides B, while A is insensitive to the declared rate. Witness: `review_p2c/jitter_confound.py`.
- The jitter claim is confirmed against the two controls; it acts mainly through BP convergence.
- Attack that worked: use the paper's own invariance argument to swap inputs that are equivalent in exact arithmetic.
- The declared-rate scan existed only at p=0.02 and p=0.05, so nothing could have flagged the p=0.07 comparison.
