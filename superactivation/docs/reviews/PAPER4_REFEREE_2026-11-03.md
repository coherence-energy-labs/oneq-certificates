# Referee report, first round, 2026-11-03: paper 4, *Superactivation in two channel uses: exact certificates for a family of PPT channels*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

# Referee report: "Superactivation in two channel uses" (`paper/superactivation/main.tex`)

**Bottom line.** The headline result holds at exactly the scope stated, and the certificates reproduce under arithmetic I wrote independently. I found no FATAL defect. The problems are:
- two evidence-class labels that claim more than the evidence carries,
- two sentences that assert something does not exist, backed only by a local search,
- one attribution of a load-bearing step that I could not verify,
- a list of minor items.

Before reading, I checked that the PDF is current: it was built at 09:57:25, after `main.tex` (09:45:14) and `numbers.tex` (09:56:05). The tree is clean, and page 1 renders the same macro values as `numbers.tex`.

## 1. Headline: n* = 2 for d ≥ 20 at threshold 3/4

**Adversary verdict on this claim: CONFIRMED.**

- **Definition.** Definition 1 (l.91–95) matches arXiv:2604.27042v2: each channel alone is "≤ 1−ε for every m", the joint channel is "> 1−ε" at n, and n* is the least such n.
- **The chain.**
  - Theorem 2: P_d is PPT, so F_c(P_d^{⊗m}) ≤ 1/2.
  - A_{2d} is antidegradable, so F_c(A_{2d}^{⊗m}) ≤ 3/4.
  - Lemma 4 rules out n = 1.
  - Theorem 3 with Lemma 2 gives n = 2.
- **Scope sentences.** Every sentence that states the result holds it to threshold 3/4 and to the larger channels: l.49, 66, 244, 250, 258, 298, 326.

**F3 (MAJOR if it cannot be resolved, otherwise MINOR): where "Prop. 1 and its general form" comes from.** Locations: l.104–112, Lemma 2 (l.206–210, labelled "Imported"), l.318 ("we did not re-prove it"), and the abstract (l.49, "We prove … two uses of P_d … exceed 3/4").
- **What is wrong.** In arXiv v2, Prop. 1 is stated only for Smith and Yard's channel, with p = 1/(1+√2).
- **What I could see.** The compiled supplement defines general-K shifted private states (Eqs. S3.5–S3.10) and has a general-K dephasing lemma (Lemma 4). The HTML text ends inside Lemma 5, before §S3.2, and I found no general form of Prop. 1 in it.
- **The authors' own records call the generalisation theirs:**
  - `docs/RESEARCH_LEDGER.md:9347-9357` (R142 item 2: "PROVEN, and MEASURED end to end", decoder "from the adversary");
  - every certificate's `imported_theorems` field: "Prop. 1 (generalized, R142)" (`certificate_nuse_d20_n2.json` l.26).
- **The statement itself is true.** I re-derived it for P_d with any complex Hadamard twist.
  - Delivered branch: Bob measures the parity of A's key copy and P's key output, which gives ℓ without disturbing the logical coherence. He applies U^{i,ℓ†} to the two shields, controlled on A's key copy, then a CNOT. Fidelity is 1.
  - Erased branch: A's copy forces dephasing, and the key output is r⊕ℓ with ℓ ~ (p0, p1), which is X^{p1}∘Δ.
- **Fix.** Either cite the exact label in the compiled supplement, or turn the sketch after Lemma 2 into a proof and relabel the lemma "Proven".

**F4 (MINOR): Imported result 1 (l.98–101) is misquoted.**
- arXiv v2 Result 1 states max{F_c(P^{⊗m},2), F_c(A^{⊗m},2)} ≤ 3/4 for all m, and F_c((P⊗A)^{⊗17},2) > 3/4.
- The "≤ 1/2" for P comes from the proof's PPT argument, not from the statement.
- The Introduction (l.60) quotes it correctly. The same misquote appears in the certificate metadata (`certificate_nuse_d20_n2.json` l.27).
- Fix: quote the statement as written and attribute the 1/2 to the PPT argument.

**F5 (MINOR): Lemma 3 (l.212–216) is labelled "Imported".** Result 1 as stated does not cover P_d; the paper applies a standard argument to a new channel. Label it "Proven (standard argument)".

**F6 (MINOR): Definition 1 is a specialisation.** It is Parentin et al.'s definition restricted to qubit messages; their F_c(·, d) has a general message dimension d, while d here is the shield dimension. Say "specialised to qubits".

**F11 (MINOR): the Introduction's framing (l.58).** Smith and Yard's own witness is the coherent information of a single joint use (Q^{(1)}(P⊗A) ≥ ½P^{(1)}(P)). The sentence "How many uses are needed before the effect can be seen at all" should be scoped to the fidelity-threshold notion.

## 2. Proofs given in full: all re-derived, all hold

- **Theorem 2 (PPT if and only if p1 = 1/(√d+1)).**
  - Sector S0: M is a phased permutation, so it is unitary, and the eigenvalues are α ± β.
  - Sector S1: N equals u† on span{|aa⟩}, and the eigenvalues are p1/(2d) ± p0/(2d^{3/2}).
  - Both sectors are non-negative only when p0 = √d·p1.
- **P_d is a channel.** Every twisted shield has marginal 1/d, so Tr_B = 1/(2d). U^{1,1} is unitary exactly when u/√d is. The source remark's "unitary with unimodular entries" is indeed impossible for d ≥ 2.
- **Lemma 4 (one use ≤ 3/4).** Correct. The flag is classical, and the erased branch is PPT, so it contributes at most 1/2.
- **Closed form (7 − 6p + 2p²)/8.** All three branches check out.
  - The crossing is at p < (3−√7)/2 = 0.177124, i.e. d > 11 + 4√7 = 21.58.
  - I decided the sign exactly in Q(√d): negative at d = 19, 20, 21 and positive at d = 22, 23.
  - At d = 64 the value is 515/648. At d = 22 it is (3259 − 130√22)/3528 = 0.7509200542015, which equals the stored certificate.
  - With decoders I optimised independently, the |++⟩, |−−⟩ encoder reproduces the closed form at d = 20 and d = 22 to 1e-12.
- **Family A.**
  - F_A > 3/4 if and only if d + √d > 24, i.e. d > (49 − √97)/2.
  - F_A meets the closed form at d = 21 + 12√3 = (3 + 2√3)².
  - At d = 20, my independent search over Δ gives 0.7507996106 against the formula's 0.7507996270.
- **Theorem 4's upper bound.** Correct: ‖N0‖ + ‖N1‖ = 1 + (1−2p)(λmax − λmin) ≤ 2 − 2p.

## 3. Certificates

**What "certified" means here, and what is trusted.**
- Stored entries are rounded to the 2^−96 grid, plus a 2^−34·I shift for the symmetric and two-group certifiers.
- Positivity is exact: either by construction (Gram matrices and KK†) or by Sylvester's criterion with exact integer determinants.
- Normalisation uses an exact Gershgorin bound plus a completely positive completion. The argument is given in the text and is correct.
- The fidelity is computed either in 256-bit ball arithmetic or exactly in Q(√d).
- Trusted and not proved in the paper: the two imports, the hand-derived symmetric and two-group reductions, and the flint/arb implementation.

**F1 (MAJOR): the class of the d = 2 headline is overstated.**
- **Locations.** Table I row 1 says "certified, given Prop. 1 and Result 1". Theorem 1 (l.145–149) and l.79 ("Every statement … rests on the two imported results") say the same.
- **What is wrong.** The n* ≤ 15 result also rests on the two-group reduced representation. The paper itself says that reduction is "derived by hand and tested … at small n" and that the certifier "trusts it" (l.135, l.320). By the paper's own definition of *Derived* (l.79), that link is Derived.
- **Why the binding test does not help.** The test that recomputes the stored pair's float fidelity uses the same reduced representation, so it cannot catch an error in the reduction at n = 15.
- **Fix.** Either add the reduction as a proved lemma (within each group, permutation invariance makes the Hamming weight of the flagged bits a sufficient statistic, and the unflagged qubits stay in Sym), or relabel the row, l.79 and Theorem 1. In addition, add a dense cross-check of an a ≠ b code at the largest feasible n (2^n ≤ 4096).

**F8 (MINOR): "certified below 3/4" (l.267 †, l.273).** A lower-bound certifier cannot certify that a value is below anything. I re-checked this case exactly: a + b√8 = 0.7457644678 (sign −1). Reword to "its certificate fails: certified lower bound 0.74576".

**F15 (MINOR): the decoder storage convention is undocumented.** The npz files hold 8×r Choi factors in out⊗in order, complex-conjugated. Of the TP-valid readings, this is the only one that reproduces the JSON branch values; the unconjugated reading gives a z00 branch of 0.1215 instead of 0.25. State the convention in the paper or the JSON.

**Spot checks and recomputation.**
- **All 16 stored n-use certificates.**
  - a + b√d, recomputed from the stored rationals, equals the stated bound to ≤ 1e-40.
  - The exact sign agrees with the stored verdict in every case, including the d = 8, n = 3 negative.
  - The SHA-256 hashes of the code and decoder files match.
- **The d = 20 two-use certificate.** I re-evaluated the stored code and decoders with my own implementation of the two-use reduced channel, built from Eq. (nt) alone. I self-tested that implementation three ways (decoder form, encoder form, Kraus path) and they agree to 1e-15.
  - Result: 0.7508849257776. All four branch values match the JSON to 1e-13.
  - The certified bound is 0.75088492577764. The margin above 3/4 is 8.8e-4; the normalisation slack is 3e-15.
- **Two-group certificates at n = 15 and 16.** The witness and decoder bytes match their hashes. The gap between float value and certified bound is 2.15e-9 and 2.42e-9.
- **Other certificates.** The two-use ball-arithmetic certificates at d = 19, 20, 21, the d = 22 closed-form certificate and the family-A d = 20 certificate all equal their macros.

## 4. Smith and Yard's pair (d = 2)

This is handled correctly.
- l.65 and l.160 concede that non-symmetric codes beating symmetric ones is not new. That is consistent with v2's Fig. 2, which shows a standard see-saw curve and calls the symmetric restriction "suboptimal in general". I read the figure text, not the plot.
- The paper claims only a code below 17 uses. v2 reports no crossing below 17.

**F17 (MINOR): "found numerically" (abstract).** Parentin et al. say their methods "can certify" n* ≤ 17 through an explicit protocol, and the paper's own check at l.164 corroborates their stored blocks to 3.8e-14. Rephrase so the published bound does not read as doubtful; the real distinction is floating-point versus rigorous evaluation.

## 5. Onset table and "the reduction is lossy"

The table separates float, certified and proven entries correctly, and presents its entries as upper bounds from a local search. One exception:

**F2 (MAJOR): unsupported claims that no code exists.**
- **Locations.** The Table III caption (l.267): "At d=5 no code with n≤4 crosses". Section VI (l.273): "For d≤6 no code with n≤4 crosses".
- **What is wrong.** These are claims of non-existence. The paper has no upper bound on F_c beyond n = 1 (l.294), and different seeds differ by up to 3.5e-3 (l.264). The paper hedges correctly at d = 19 (l.256) and at n = 14 (l.158), but not here. Ledger line 12275 has the same wording.
- **Fix.** Write "no code found by our search".

**F9 (MINOR): "the reduction is strictly lossy".** In the abstract, the Introduction (item 5) and the title of Section VII, this is shown at one use only. Add "at one use".

## 6. Related work and novelty

**Descriptions that check out.**
- Zhu–Wang v3 is dated 5 Oct 2026 ("added quantum superactivation results"), and its abstract matches the paper's description.
- 2609.10520 and 2609.20133 concern private capacity; 2610.02176 has no superactivation result; 2606.10744 is a different mechanism (superposed channels).
- The novelty paragraph is hedged and held to the search log.

**F12 (MINOR): related-work gaps.**
- Wu–Wang (2510.04527) say in their abstract that they "give an alternative proof of superactivation" over a broader parameter regime. The paper does not mention this.
- The paper says neither Wu–Wang nor Zhu–Wang states anything about a finite number of uses. The search log (`docs/PAPER4_LITERATURE_SEARCH.md`) has no Wu–Wang entry, so that half of the sentence is not covered by the logged search. From the abstract I found only asymptotic results, so the statement is probably right, but it should be logged.
- arXiv:1107.4385 was returned by the logged search (log l.14) and is neither cited nor dismissed. I believe it is Brandão, Oppenheim and Strelchuk, but I have not checked.
- Smith, Smolin and Yard's Gaussian superactivation is not cited.

## 7. Internal consistency

- **F7 (MINOR).** l.162 says "Three are certified" but lists two. The n = 20 value, 0.7506549819, appears only in Table II.
- **F10 (MINOR).** l.181 identifies P_2 with Smith and Yard's channel using only the spectra of the Choi state and its partial transpose, which is necessary but not sufficient. Theorem 4 depends on that identification. Compare entrywise against the supplement's Kraus operators (Eqs. S2.20–S2.27).
- **F13 (MINOR).** In the hardware section (l.304), the measured 0.615 ± 0.009 against the model's 0.657 is a 4.7σ over-prediction (0.042). That is larger than the projected ion-trap margins of +0.017 to +0.024 (ledger l.12131–12134). Say so next to "clear 3/4 (simulated)".
- **F14 (MINOR).** `\LaneRelease` is "[pending]" (`numbers.tex` l.82) and renders in bold at l.332. That placeholder cannot go into a posted manuscript.
- **Optional.** The PPT-relaxation values in Section VIII (0.750 and 0.875) are exactly 3/4 and 7/8 = 1 − ¼·½. A two-line proof could replace "measured".

All margins, truncations and closed-form constants I recomputed match.

## Attacks that told me nothing

My unrestricted two-use see-saw, with 24 restarts at d = 19, reached only 0.733 at d = 20, where a 0.7509 code exists. Because it could not find a known code, its result at d = 19 (best 0.719) says nothing, and I discarded it. Whether two uses can cross at d = 19 remains open, as the paper says.

## Output contract

- **STRONGEST SURVIVING RESULT.** At threshold 3/4, n* = 2 exactly for P_d ⊗ A_{2d}, for every integer d ≥ 20 and every complex Hadamard twist. This uses the reduction (re-derived here) and the PPT and two-extendibility caps. For Smith and Yard's pair, n* ≤ 15, given additionally the derived two-group reduction.
- **EVIDENCE CLASS.**
  - PROVEN: d ≥ 22, Theorem 2 and Lemma 4.
  - CERTIFIED and independently reproduced: d = 20; d = 21 is certified, and I re-checked it exactly but did not re-evaluate it in floating point.
  - The d = 2 bound is certified inside a DERIVED reduction.
- **HOW IT COULD STILL BE WRONG.** The two-group reduction at n = 15. Test: a dense evaluation of an a ≠ b code at n ≈ 12, or a proof.
- **WHAT IT LICENSES.** Posting, after F1–F4 are fixed.
- **KILLED.**
  - "No code with n ≤ 4 crosses for d ≤ 6": there is no upper bound, and seed spread is 3.5e-3. What survives is "none found".
  - The Result 1 quote: what survives is "max ≤ 3/4, with the 1/2 from the PPT argument".
  - "Certified below 3/4": what survives is "the certificate fails at 0.74576".
- **OPEN BRANCHES.**
  1. Read the compiled supplement §S3.2 in the PDF; that decides whether Lemma 2 is "Imported" or "Proven".
  2. Making n* exact for d = 9–19 needs an upper bound below 3/4 on two *joint* uses. Bounds on the reduced channel are not enough, because Section VII shows the reduction is lossy. The joint two-use input dimension is about 2·10^6 at d = 19, so the instrument is a symmetry-reduced extendibility SDP.
  3. d = 19 at n = 2, with a search that first reproduces 0.75088 at d = 20.
- **RECOMMENDATION: SHIP** after the required edits.

I wrote no agent memory, as instructed.

**Journal verdict: MINOR REVISION.** Every result survives the attacks above, the proofs given in full are correct, and the stored certificates reproduce under independent exact and float arithmetic. Required before posting: F1 (relabel, or prove the reduction), F2 (two sentences), F3 (cite the exact statement, or prove Lemma 2) and F4. The rest is recommended. None of these changes a result.

**Files:**
- Manuscript: `paper/superactivation/main.tex`
- Macros: `paper/superactivation/generated/numbers.tex`
- Certificates: `evidence/superactivation/shield_family/certificate_nuse_d20_n2.json` and `evidence/superactivation/general_codes/certificate_twogroup_a7_b8.json`
- Ledger: `docs/RESEARCH_LEDGER.md` (lines 9347–9386 and 12251–12276)
- Search log: `docs/PAPER4_LITERATURE_SEARCH.md`
- My recomputation script (reads the evidence, writes nothing): `<reviewer's local scratch path>`
