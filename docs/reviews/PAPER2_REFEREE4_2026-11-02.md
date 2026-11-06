# Referee report, fourth (final read) round, 2026-11-02: paper 2, *Exact, independently checkable optimality receipts for BP-OSD*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: READY.** Every number I recomputed matches its macro and evidence file. No headline claim fails. All seven findings are wording or consistency fixes; none needs a new computation. I would still make the first two before upload, because a careful reader will notice both.

## 1. Internal consistency: nothing wrong with any number

What I recomputed, all of it matching:
- **Table I sums:** N = 2,019; tier 1 = 1,780; tier 2 = 222; BD = 17; flat = 2,002 (99.15%). No branching = 2,010 (99.55%). Proven optimal = 1,783 (88.31%). Stress misses 178 = 164 + 14. Not re-decoded tier 2 = 1 + 31 = 32.
- **Clopper–Pearson (1-sided 95%):** n=300 gives 99.006, n=470 gives 99.365, n=150 gives 98.023, n=200 gives 98.513, n=149 gives 98.0095. All are floored correctly. The 0/149 upper bound is 1.9905%, printed as 2.00, which is the conservative direction.
- **McNemar:** 2·2⁻¹⁷⁰ = 1.34e-51 and 2·2⁻³ = 0.25. The paired cells sum correctly (300 shots; 175 and 5 errors; post hoc 8 and 5).
- **"At least 87%" recovery:** (185−84)/(197−84) = 89.4%, (175−24)/(196−24) = 87.8%, (285−122)/(299−122) = 92.1%. It stays at or above 87.8% whichever end of A's range is used as the baseline.
- **Tables II and III against `evidence/qldpc_minsum_rate_scan.json`:** every A, B, A_s and B_s entry and range matches. So do the tier-1-rule counts (179 = 146+33, 163, 178, 196 = 187+9) and the percentages.
- **The build is current:** the PDF and log (21:11) are newer than main.tex, numbers.tex and refs.bib, and the log has no undefined references.

## 2. Min-sum scaling story

The core holds:
- **Homogeneity argument:** correct, and the implementation passes the control. The prior LLR is read back bit-exactly at 1, 2, 4 and 8 (`qldpc_minsum_rate_invariance.py:83-92`), and answers are identical on 200/200 shots at both scaling factors.
- **B not well defined:** B's optimum hits run 87–197 at p=0.05 and 20–178 at p=0.07.
- **Jitter:** relative size below 10⁻⁶, no exact zeros left, A's answers rate-independent on every dataset, convergence 133 versus 172.
- **0.625:** the installed ldpc 2.4.1 sets it as the default in `sinter_bposd_decoder.py:41`, and "not tuned" matches the post-hoc JSON. "Every factor below 1" is B only, three factors: 10/11/9 failures at p=0.05 and 39/36/36 at p=0.07.

**F1 (MINOR; the one to fix first): A's answers depend on the declared rate as much as B's, but only B's figure is printed.**
- **Where:** line 150 ("only 85 of the 200 shots at p=0.05 keep one answer. B at scaling 1 is therefore not a well-defined arm"); line 111 ("The campaign's misses do not depend on the floating-point form of its declared rate"); the B-only daggers in the captions at lines 126 and 133.
- **Evidence (rate-scan JSON):**
  - Shots keeping one answer across the 16 rates: A 88/200 at p=0.05, 30/200 at p=0.07, 129/300 held out; B 85, 18 and 115.
  - The macro `\InvOneAnswerA`=88 is generated but never printed.
  - A's failure counts also move with the declared rate: 9–16 at p=0.02, 31–37 at 0.03, 172–177 at 0.07. Table II prints 9, 31 and 172 with no marker.
  - Line 111's own range (175–178) shows the miss count does depend on the rate, slightly.
- **Fix:**
  - Print A's figure beside B's.
  - Say the reason for excluding B is how far its counts swing (87–197) against A's (84–88).
  - Reword line 111 to "the miss count is not an artifact of…".
  - Mark Table II's A entries as as-run too.
  - In the line 133 caption, say "except the comparison in Sec. VI D", to match line 150.

**F3 (MINOR): line 152, "What goes wrong at scaling 1 is that the soft outputs tie exactly."** As the cause of the poor results, this is refuted by the paper's own control arm.
- **Evidence (invariance JSON, p=0.05):** tied soft bits are 28,707 at scaling 1 and still 14,305 at 0.625, where the decoder works well. The statistic that separates the two arms is exact zeros: 6,312 against 0.
- Breaking ties recovers 87–92% of the optimum-hit gap but only 41–57% of the convergence gap (133 vs 172; 59 vs 119; 196 vs 263).
- Referee 3's item 1 fix (report zeros and convergence) is in the text, but this lead sentence still names ties.
- **Fix:** "Much of what goes wrong is exact cancellation: soft values land exactly on zero (6,312 at p=0.05, none at 0.625)…"
- Also "125 of them" points back to the 81 converged shots, which cannot hold 125. Change it to "on 125 of the 200".

**F4 (MINOR): line 150, "Scaled, A_s and B_s fail 7 and 9 times at every declared rate."** It does not say p=0.05. These numbers, and the 9–11 at scaling factors 0.9 / 0.75 / 0.625, come from `qldpc_bposd_scaling_sensitivity.json`. Its script is never named; line 152 names only the rate-scan and invariance scripts. The statement does hold at every level (35/36 at p=0.07, 8/8 held out), so add "at p=0.05" and name the script.

## 3. Crocker and Pan (line 123): accurate in substance

- **Pan, v2.0, `release/arxiv/main.tex`:** "The IBM circuit … has circuit distance exactly 10 under SD6 for R≤3, by exact enumeration", with no certificate for that value.
- **Crocker, Zenodo v1, 2026-10-18:** "12 noisy syndrome cycles followed by 2 noiseless ones", both memory bases, exhaustive branch-and-bound, "There is no machine-checkable certificate".

Both quotes came through WebFetch summaries, not the raw files.

**F6 (MINOR): three precision problems in that sentence.**
- **(a) Noise model.** Crocker credits Bravyi et al. for "the fault model" and never names SD6, yet the sentence attaches Crocker's value to SD6. The companion paper's Table I caption is careful here ("each source's own noise model").
- **(b) "Established for every number of rounds … by checkable certificates."** The companion's own abstract says the every-round statements "rest on two hand proofs" whose finite hypotheses are machine-checked. It also says its fault placement matches SD6's mechanism set only on the circuits and round counts compared. Fix: "certificates and a hand-proved extension".
- **(c) Which companion.** This companion is unnamed, while Sec. I names a different one. Name it: *Machine-checkable lower-bound certificates for the circuit-level distances of bivariate-bicycle syndrome-extraction circuits*.

## 4. Careless items

- **F2 (MINOR, a late-edit error):** line 113 says the after-the-fact trees "are included in the tree statistics below". Those statistics are above, in line 111. Change to "above".
- **F5 (MINOR):** line 111 says "all four code-capacity datasets" and that the heterogeneous dataset uses "other matrices and weights". The producer describes it as "heterogeneous code capacity, per-qubit rates 0.5-2x 0.02" (`qldpc_exact_all_gate.py:8`) and builds it on `checks_z` (line 370), so a reader counts five code-capacity rows in Table I. Fix: "four uniform-weight code-capacity datasets". I did not confirm that the uniform datasets also use `checks_z`.
- **F7 (MINOR, older text):** line 42, "certified 299 / 299 / 299 shots", gives three values but names only two scale factors.
- Every `\ref` resolves. I found no other broken sentences.

## What survives, and a better tool for the paper

Every result survives. The min-sum story survives in a narrower form:
- In exact arithmetic the decoder cannot depend on the declared rate (proven, and the implementation passes the check).
- In floating point, the dependence comes from rounding breaking exact ties (measured).
- What separates scaling 1 from 0.625 is exact zeros, not ties in general.

A better way to define the unscaled arms: at p=0.05 the power-of-two runs fail exactly as often as the as-run arms (A 114, B 109). Six of the 16 declared rates also reproduce the as-run answers on every shot. Running the unscaled arms at a power-of-two prior LLR, where floating-point scaling is exact, would make B a defined arm (109 failures at p=0.05) instead of an excluded one. Two caveats: the JSON records only counts, not whether each power-of-two answer equals the as-run answer, and exact integer arithmetic over 100 iterations is unverified.

**What I could not test:**
- I re-ran no decoder; every min-sum number was traced from JSON through `make_numbers_qldpc.py:332-488`.
- The jitter is a single draw (seed 20261102), so how 185 / 175 / 285 vary with the seed is untested.
- I did not view the PDF pages; I relied on the tex and the log.

Files:
- `paper/qldpc-certificates/main.tex`
- `evidence/qldpc_minsum_rate_scan.json`
- `evidence/qldpc_minsum_rate_invariance.json`
- `evidence/qldpc_bposd_scaling_sensitivity.json`
- `paper/tools/make_numbers_qldpc.py`
- `paper/circuit-distance/main.tex`
- Script I used to summarise the evidence: `<reviewer's local scratch path>`
- Memory entry written: `.claude/agent-memory/adversary/p2_final_read_2026-11-02.md`
