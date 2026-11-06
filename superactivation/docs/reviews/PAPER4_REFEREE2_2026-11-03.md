# Referee report, second round, 2026-11-03: paper 4, *Superactivation in two channel uses: exact certificates for a family of PPT channels*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

# Referee report, round 2: "Superactivation in two channel uses" (`paper/superactivation/main.tex`, revision 3df91497)

**Verdict: MINOR REVISION.** The new mathematics holds:
- Lemma 2's proof is complete for both certifiers as they are implemented.
- Lemma 4's proof is correct, and the citation to the general theorem in the supplement's Sec. S3.2 is exact.
- P_2 equals Smith and Yard's channel exactly.
- The dense check does what the paper says, and it can fail.

I found nothing FATAL or MAJOR. Seven MINOR items remain: six are one-sentence fixes, and one is the round-1 placeholder, which is still there. None changes a result.

## Adversary verdicts on what is new

| Claim | Verdict |
|---|---|
| Lemma 2 is complete, and it is exactly what both ball-arithmetic certifiers compute | CONFIRMED |
| The dense check tests what l.184 says, can fail, and is not circular with the certifier | CONFIRMED, with the scope notes in N6 and N7 |
| Lemma 4 is correct, and the compiled v2 supplement's S3.2 states the general form | CONFIRMED |
| Lemma 5; P_2 equals Smith–Yard's channel entry by entry | CONFIRMED; "exactly" is narrower than written for the Choi-state half (N3) |
| Abstract: "Every step is proven or certified here" | REFUTED (N2) |

## Findings (all MINOR)

**N1. l.143 and l.71 describe the lemma's domain wrongly.**
- **What is wrong.** "Codes whose encoder output is invariant under permutations" and "codes with permutation symmetry" describe a larger class than Lemma 2 covers. The lemma needs the encoder output to be *supported on* R⊗Sym^{m_1}⊗…, and invariance does not imply that.
- **Witness.** Take one group of m = 2 qubits and ω = ½𝟙_R⊗|Ψ⁻⟩⟨Ψ⁻|. It is swap-invariant. But |Ψ⁻⟩ is orthogonal to Sym², so neither the splitting identity nor Eq. (5) applies to it.
- **This is a regression.** The previous draft said "with the encoder supported on R⊗Sym^n"; this revision removed that wording.
- **Impact.** No result is affected. The certified encoders are 2·Π(m_g+1)-square matrices on the Dicke basis, so they lie in Sym by construction.
- **Fix.** "Codes whose encoder output lies in R⊗Sym^{m_1}⊗…⊗Sym^{m_G}."

**N2. l.56, the abstract's last two sentences.**
- **What is wrong.** "A PPT-assisted relaxation cannot bound n* from below… Every step is proven or certified here." The PPT sentence is measured: Table I's last row says "measured (a kill)", and l.375 says "(measured)". So the closing sentence is false for a statement in the same abstract.
- **Also overstated.** "Cannot bound n* from below" goes beyond the table's own wording, "cannot certify a lower bound on n* above 2".
- **Fix.** "…cannot show n*>2: it exceeds 3/4 at two uses (measured). Every other step is proven or certified." The better fix is to prove the 7/8 value (the round-1 optional suggestion) and keep the sentence as written.

**N3. l.206: "exactly" for the Choi state.**
- **What is wrong.** The text says P_2 matches "entry by entry and exactly… its six Kraus operators… and its Choi state (Eqs. S2.4–S2.17)". The exact sympy comparison covers only the Kraus side: `exact_comparison` compares the family with the Kraus-derived Choi matrix alone (`p2_kraus_comparison.py` l.82-145). The Choi-state formula is compared only in floating point (`choi_formula`, l.66-79; 4.2e-17 in the JSON).
- **Fix.** Either add the Choi-state formula to the sympy comparison (about 10 lines), or write "exactly against the Kraus operators, and to 4.2e-17 against the Choi state".

**N4. l.343 contradicts the Table III caption (l.346) and `tab_onset.tex`.**
- **What is wrong.** l.343 says "for larger n (at d=2,3,4,6) from symmetric codes". At d = 2 the entry is the two-group 7+8 code.
- **Fix.** "At d=3,4,6 from symmetric codes; at d=2 the two-group code of Thm. 7."

**N5. l.415: the round-1 finding F14 is unresolved.**
- **What is wrong.** `\LaneRelease` still renders as bold "[pending]" (`numbers.tex` l.98).
- **Fix.** Mint the release, or cite the public commit hash. The generator should refuse the placeholder at posting time.

**N6. l.184: the negative controls.**
- **What is wrong.** "Changing one reduction amplitude, or p, by 10⁻⁶" describes relative changes:
  - The amplitude change is `S[0,1] *= 1 + 1e-6` (`verify_reduction_dense.py` l.299). The absolute change is √(1/5)·10⁻⁶ = 4.47e-7, which is exactly the reported "deviation 4.4e-7".
  - The p change is `p = P*(1+1e-6)` (l.303).
- **Also.** The amplitude control perturbs the identity comparator's expected matrix (l.295-301), not the module that produces the states. Only the p control shows that the state comparator can fail.
- **Fix.** Say "by a relative 10⁻⁶".

**N7. l.182, l.184 and l.399: the dense check validates the float twin, not the certifier's own arithmetic.**
- **What is wrong.** `verify_reduction_dense.py` imports only `twogroup.sel/coeffs/bob_states` and `superactivation.selection/bob_states` (l.49-52). The certificates are computed by separate ball-arithmetic code: `_selection2` (`twogroup_certificate.py` l.61-73), and `_selection` and `_coeff` (`superactivation_certificate.py` l.134-152).
- **What I verified by reading.** These are the same formulas:
  - `_coeff(k,b,a)` is C(k,a) times the same sum as `flip_transition`.
  - The selection matrices are 𝟙⊗P, with the same amplitude √(C(m−k,i)/C(m,i+a)).
  - The weights are C(a,k₁)C(b,k₂)/2ⁿ, the functional is Tr(ρD)/(4(1+e_D)), and the whole sum is divided by (1+e_E).
- **Two further things the check does not test.**
  - Both arms use the same functional Tr(ρD)/4 with the same decoders, so the fidelity agreement follows from the state agreement.
  - The dense arm never applies the dephase-and-flip channel; it builds that step in.
- **Not circular, but disjoint.** The check shares no code with the certifier; it also does not exercise the certifier's arithmetic.
- **Fix.** Add one sentence saying so, or have the check compare the midpoints of `_selection2` and `_coeff` against its dense slices and per-string law.

**Nits**
- l.121: the theorem is stated in S3.2; its proof occupies S3.3–S3.5 (source l.1321, 1411, 1763). Write "states (S3.2) and proves (S3.3–S3.5)".
- The script's docstring says "about 10 min, under 2 GB". The JSON records 259.8 s, and the paper says about 4 min.

## Checked and held

- **Lemma 2, every step re-derived.**
  - The splitting identity, including its normalization.
  - Placement: U_π fixes Sym pointwise, and the channel under flags π(z) is U_π N_z(U_π†·U_π) U_π†.
  - The dephase-and-flip formula with T(s|t).
  - The sum over strings factorizes over groups and depends on t only through its weight, giving Pr_k(b|a) with multiplicity C(k_g,a_g).
  - The support claim.
  - The physical decoder through the Dicke isometry, with a completion outside Sym.
  - The weight 2⁻ⁿΠC(m_g,k_g).
  - The certifier's functional: Tr(XJ)/4 = ⟨Φ|(id⊗𝒟)(X/2)|Φ⟩ for J the Choi matrix of 𝒟† with Tr_R J = 𝟙.
  - The encoder's completion may leave Sym; its contribution to the fidelity is still ≥ 0, so the bound survives.
- **No disagreeing code exists within the hypotheses.** A disagreement needs an encoder that is not supported on Sym (the N1 witness), and the reduced representation cannot hold such an encoder.
- **My independent physical check agrees.** It differs from the repo's check in three ways: the channel is applied by its Kraus operators to the full density matrix; the decoders are physical Kraus maps applied through the isometry adjoint with a completion; and encoders and decoders are random and complex.
  - Coverage: all 2ⁿ flag patterns, for one group at n = 6, 7; two groups (1,1), (1,3), (2,3), (3,3), (3,5); and a rank-1 encoder at (3,4).
  - States agree to ≤ 1.7e-16, nothing leaks outside Sym (≤ 1.4e-17), and the fidelities agree to ≤ 3.6e-15.
  - Control, p×(1+1e-6): states move by 3.9e-8 and F by 1.7e-10.
  - Control, decoder Kraus taken without the complex conjugate: F moves by 7.2e-4. This confirms the reading at l.413.
  - The first run had a bug of mine: the normalizer used `einsum("risj")` instead of the partial trace, which blew up the rank-1 case. I fixed and re-ran it. The identity is linear in X and J, so the other rows were unaffected.
- **The dense-check JSON matches the macros**: (5,7); 4,096 patterns; 7.8e-15; 2.3e-16; controls 4.47e-7 and 1.77e-8; 259.8 s.
  - The recorded code blobs equal `git hash-object` of all five files in the current tree, and the tree is clean.
  - The generator refuses on a verdict or blob mismatch (`make_numbers_superactivation.py` l.307-313).
  - The n = 15/16/17 dense fidelities equal the recorded pair values to ≤ 2.2e-16.
- **Lemma 4, re-derived.**
  - P_d(|r,s⟩⟨r',s'|) as stated.
  - Delivered branch: the CNOT gives |ℓ,r⟩ and keeps the coherence; the controlled U^{i,ℓ†} undoes the twist.
  - Erased branch: δ_{rr'}Σ_ℓ p_ℓ|r⊕ℓ⟩⟨r⊕ℓ|, which is 𝒳^{p₁}∘Δ̄.
- **The S3.2 citation is exact.**
  - The e-print's SHA-256 is a5c80fa2…, matching the paper.
  - The supplement is `\input` at the arXiv `main.tex` l.111.
  - `\thesubsection` is S⟨section⟩.⟨subsection⟩ (supplement l.25). S3.2 is "Proof of Effective Channel" (l.1283).
  - `thm:effective_channel_general` is at l.1306-1307, outside both comment blocks (l.1256-1279 and l.1917-1925).
  - Its hypothesis is a Choi state that decomposes into K shifted private states, with an erasure channel of input dimension K·d_s and general q. Its encoder is the paper's V, so Lemma 4 is the instance K = 2, d_s = d, q = 1/2.
- **P_2 equals Smith–Yard's channel.**
  - The Kraus transcription matches source l.371-387 verbatim, including the key-first factor order and the minus sign in M_1.
  - The Choi-state transcription matches l.333-357.
  - Equation numbers: with the per-section reset (l.31), counting gives S2.4 for the Choi state, S2.17 for p, S2.20–25 for K_i and S2.26–27 for M_i. The citations are exact.
  - The sympy arm rebuilds the family from Eq. (gamma) itself, independently of `ppt_family.py`.
  - The source's U^{1,1} is SWAP on span{|01⟩,|10⟩}, while the paper's is the identity there. This is irrelevant, because σ¹ is supported on span{|aa⟩}.
- **Lemma 5.**
  - PPT part: T∘P is completely positive, tensor powers stay PPT, and local post-processing preserves PPT, giving 1/2.
  - Erasure part: the dilation is swap-symmetric and any encoder environment stays with Alice, so D⊗D gives a symmetric extension and the bound 3/4.
- **Numbers.** Abstract, body, `tab_dtwo`, `tab_onset` and the macros all agree.
- **PDF.** It is current (built 11:06 after main.tex 11:03 and numbers.tex 11:03), and pages 1-3 render correctly.
- **Novelty paragraph.** It matches `docs/PAPER4_LITERATURE_SEARCH.md` l.26-35, including the "existential" wording quoted from Zhu–Wang.
- **Round-1 findings.** All are addressed except F14 (N5).

## What I did not test
- I did not re-run the ball certifiers (2.1 h and 4.6 h); I relied on reading the code.
- I did not read the test that checks the bound sits just below the float value; I saw only the 1e-12 float recompute at `tests/test_superactivation_twogroup.py:61`.
- My physical check used random codes with n ≤ 8, where F ≈ 1/4. The high-fidelity codes at n = 15–17 are covered only by the repo's slice-based check.
- I read the supplement's theorem statement and encoder, not its full proof in S3.3–S3.5.

## Output contract
- **STRONGEST SURVIVING RESULT:** For the Smith–Yard pair, n* ≤ 15, certified through a reduction that is now proven. At threshold 3/4, n* = 2 for every d ≥ 20.
- **EVIDENCE CLASS:** PROVEN (Lemmas 2, 4, 5 and 6; d ≥ 22) and CERTIFIED (d = 2, 20, 21).
- **HOW IT COULD STILL BE WRONG:** the ball-arithmetic certifier diverging from its float twin; this is closed by inspection and by the binding window (N7).
- **KILLED:** "Every step is proven or certified" (witness: the PPT-relaxation sentence). What survives: "every other step".
- **RECOMMENDATION: SHIP** after N1–N7.

Per your instruction I edited no repository file and wrote no agent memory. The only files I wrote are in my scratchpad:
- Check script: `<reviewer's local scratch path>`
- arXiv source (read, not executed): `<reviewer's local scratch path>`
