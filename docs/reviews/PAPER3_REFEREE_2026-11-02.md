# Referee report, first (of this round) round, 2026-11-02: paper 3, *Machine-checkable lower-bound certificates for the circuit-level distances of bivariate-bicycle syndrome-extraction circuits*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

No headline number is refuted. Three findings are MAJOR and nine are MINOR. In order of severity: the novelty framing is out of date, the supplementary hand proof contradicts the main text on the evidence behind the IBM result, and the certificates for two of the three headline families are not available to readers.

## Findings

**1. MAJOR: the novelty framing misses a public, certified result on the same three-colour circuit.**
- **Where:** abstract (main.tex:39, "for circuits on the [[144,12,12]] gross code the published values are upper bounds"), main.tex:53, and related work at main.tex:301-302 ("the sources we found report upper bounds only").
- **What is out there:** Guancheng Pan's repository `GavinNull/gross-code-distance-12`. The GitHub API gives: repository created 2026-10-13, release v1.0 on 2026-10-20, v2.0 on 2026-10-24, last push 2026-11-01. The README states it proves that the Strikis–Browne–Beverland Fig. 12 circuit "has circuit distance exactly 11", under SD6 noise, in both bases, for any number of rounds. That is your headline 2.
  - Its release notes add interleaved gross-code circuits at distance 12 and [[90,8,10]] at 10.
  - Its proofs are certified: DRAT checked by drat-trim and cake_lpr, VeriPB in v2.0, plus a round-splitting argument for every number of rounds.
  - I did not check Pan's proofs, and I could not date when the "exactly 11" sentence first appeared.
- **Why it matters for you:** SD6 also puts noise on idle ancillas, which your three-colour placement leaves out (main.tex:110). If Pan's transcription matches yours, his lower bound implies your three-colour result.
- **Also missing:**
  - The obvious certified-SAT alternative: Stim's SAT export with DRAT/LRAT/VeriPB proofs and a verified checker such as cake_lpr. The paper compares itself only with integer-programming branch-and-bound certificates.
  - Veri-QEC (Huang, Zhou, Fang, Zhao, Ying; arXiv:2504.07732).
  - arXiv:2610.03214, which has code-parameter certificates checked by the Lean kernel.
- **Smallest fix:**
  - Cite Pan with dates and replace the "upper bounds" sentences with a dated, scoped statement.
  - Compare noise models and circuits. If your branch-and-bound certificates and his SAT certificates agree on the same circuit, that would be the strongest cross-check in the paper.
  - Add one paragraph on branch-and-bound certificates versus DRAT/VeriPB.
- **What remains new:** IBM's cycle at 10 for every T ≥ 2 with shipped proofs, the morphing results (6 and 12 for every T), Table II, and the checker bugs that the Lean formalization found.

**2. MAJOR: the supplementary proof still cites the IBM evidence the paper says it no longer relies on.**
- **Where:** `WINDOW_PROOF.md` §4.2, lines 247-253 (and header line 34), against main.tex:172 and main.tex:255.
- **IBM:** Corollary 3 gets the lower bound from `records/ibm_proofs_w9.jsonl` (rule W). main.tex:172 says of exactly these campaign certificates that they are "therefore not the evidence for IBM's cycle". The route the paper actually uses, rule S plus the `ibm_recert_20261026` certificates, appears only as Lemma W(iii); no corollary applies it.
- **Three-colour, three stale passages:**
  - Line 255 still says "no solution", where it should say "no irreducible solution" (the Lean finding F2).
  - Lines 259-260 say "MEASURED ... until its 24 certificates are checked".
  - Line 318 again says "MEASURED for T = 3..12".
  - All three contradict the file's own header (lines 35-38) and the paper.
- **Previous review:** its fix for finding #2 is real in main.tex but never reached this file, which its fix for finding #5 made part of the paper.
- **Smallest fix:** rewrite Corollary 3 as Lemma W(iii) plus the re-certification, citing the manifest hashes. Scope the three-colour statement to "irreducible" and delete the stale sentences. Add a `need()` check to the number generator that refuses while the supplement cites `ibm_proofs_w9.jsonl`.
- **The mathematics holds:** I read Lemma W (i)-(iii) and it is sound given Corollary 1. The thresholds check out: w+2 ≥ T₁ is 11 ≥ 5 for IBM and 12 ≥ 7 for the three-colour cycle.

**3. MAJOR: for the three-colour and morphing results, "certified" means a verdict line, not a checkable certificate.**
- **Where:** main.tex:39, main.tex:172, main.tex:308.
- **What ships and what does not:** only IBM's 22 CDP5 proofs, plus two more, are in the repository. For the following, the record keeps only a hash and the checker's verdict line:
  - three-colour at T = 2 (479 and 536 MB) and its 24 time-shift certificates
  - the 14 morph72 time-shift certificates
  - morph144 at T = 1 and T = 2 (185-516 MB)
  - Table II
- **Which checker:** per main.tex:172, many of these verdicts came from checker versions that predate the fixes. None came from the version the Lean model follows (`ccheck.c` at 493facf7). I confirmed the committed file has that version's hash, 4fc5540e203b45c4.
- **Consequence:** a reader can re-verify headline 2 only by re-running the prover, about 23 h for the time-shift set.
- **Smallest fix:** re-certify all of these in CDP5 with the committed checker and ship them, as was done for IBM. Table III shows an 18-21× size reduction. Otherwise, say in the abstract that only IBM's certificates are public.

**MINOR**
4. **main.tex:99, "exactly 6 meta-checks" for "every circuit here".** The macro is read only from the morph72/morph144 records. For these balanced two-block codes the count should be k/2 per type: 6 for the k=12 codes, but 3, 4 and 2 for Table II's [[42,6,6]], [[30,8,5]] and [[48,4,8]]. That assumes equal check counts and ranks per type, which I did not verify.
5. **Provenance of the 14 circuits.** The abstract says "we re-found from the parameters of their 2026 follow-up" and Table II's caption says "built from". But main.tex:274 says "whether the paper's own circuits are among ours cannot be decided". Say "with the code parameters reported in".
6. **Theorem 1 states only the projected model** (with T₀ = 1+U). Section VI then uses "the full formula" (main.tex:257, main.tex:261). Its start index j_s, and T₀ = 4 for the full three-colour model, are defined only in the supplement.
7. **"The Lean kernel checks the distance itself"** (abstract, Table II caption). The kernel theorem is about the exported projected model's distance. The circuit distance also needs a witness lift that happens outside Lean.
8. **Non-standard noise placements, justified only on [[72,12,6]]:**
   - IBM has noiseless data initialization and no data idle noise during reset, Hadamard or measurement.
   - In the three-colour cycle, idle ancillas are not faulted.
   - Check that SD6 gives the same mechanism set on the gross-code IBM and three-colour models at T = 2 and 3; it is a cheap test. If it does, the results carry over to SD6 and the "attained" and "conjectured" comparisons become exact. It also lines you up with Pan's SD6 claim.
9. **Possible citation error (low confidence).** The Tesseract authors' own LaTeX byline is "Laleh Aghababaie Beni". The bib renders "L. A. Beni", and the text cites "Beni, Higgott and Shutty" (main.tex:53, main.tex:301). Confirm the family name and protect it with braces.
10. **Table I, last row:** "no record of it is kept". Either record it or drop the row.
11. **Wording of "Proven" and the title.** "Proven" here means a hand proof audited once by an AI agent. The every-T statements also rest on reading the builder code. Lean does not cover rule S across models, or anything upstream of the detector error model (SPEC.md G9/G10). The title's "Certified" should not cover the every-T part.
12. **The schedule-gate subsection (main.tex:280) shows only that the gate rejects:** of 240 proposals, 12 were valid, 11 kept the baseline distance and 1 was rejected. Cut it or move it to the supplement.

## Attacks that failed to find a problem
- **Re-checking a shipped IBM proof:** I compiled the committed `ccheck.c` and ran it on `proof_ibm_Z1_w9.bin` with its model, whose hashes match the manifest. It printed VERIFIED. Flipping one byte in the middle gave REJECTED (reason 4), and dropping the last byte gave REJECTED (truncated).
- **Citations, checked against the authors' own LaTeX:**
  - Tesseract (lines 440-441): MaxSAT verified only [[72,12,6]] = 6 and [[90,8,10]] = 8; the gross code is "at most 10". Your novelty claim against Tesseract holds.
  - Strikis–Browne–Beverland: "we conjecture ... d_circ = 11". Accurate.
  - Webster et al.: Gurobi finished within 8 h only on the first circuit of the family. Accurate.
  - Shaw–Terhal 2026: Prop. 1 is the two-round result. Accurate.
  - Shaw–Terhal 2024: the morphing bound "≤ 12" comes from BP-OSD. Accurate.
  - Bravyi et al., Webster et al., Strikis–Browne–Beverland, Shaw–Terhal 2026: authors and titles match.
- **Definition versus computation:** counting mechanisms equals counting faults for the stated channels. The quantity computed — distinct Stim symptoms, the projection as a relaxation, witnesses checked on the full model — is the quantity defined.
- **Previous review:**
  - Fix #1 is real.
  - Fix #2 is real in the main text only (finding 2).
  - Fix #4 is adequately scoped.
  - Fix #5 is partial (finding 2).
  - Fix #13 is incomplete (finding 1).

## Verdict
Nothing I found is false at the level of the headline numbers in the stated fault placement. The IBM lower bound is the best supported: its proofs ship, I re-checked one, and the hand proof's mathematics holds. The three-colour and morphing results rest on certificates readers cannot obtain, checked by older checker versions, plus hand proofs that earlier AI-agent passes confirmed. The paper's distances stand in its stated noise placements. Its positioning does not: Pan's public, certified claim on the same circuit, released before this draft, is uncited. Recommendation: major revision before posting. Every fix is a text change, one re-certification run, or a cheap noise-model equality test.

## What I could not check
- Pan's proofs and circuits, and when the "exactly 11" sentence first appeared.
- Whether Shaw–Terhal 2026 names its groups and polynomials, and the exact wording of their App. C.4 (a match exists at line 1373 of their main.tex).
- The section number 5.5.3 in Webster et al.
- Automated arXiv metadata checks: the export API returned no entries on both attempts, so a "no mismatch" result from it would have meant nothing.
- `FOLD_PROOF.md`, which I did not re-read; I relied on the earlier adversary pass that confirmed it.
- Whether Strikis, Browne and Beverland released machine-readable circuits.

## Summary in the research-protocol format

| | |
|---|---|
| **Strongest surviving result** | IBM depth-8 cycle: d_circ = 10 for every T ≥ 2, both bases, in the stated placement. Shipped CDP5 proofs, re-checked here. |
| **Evidence class** | Lower bound machine-checked at T = 1..11 (rule S); every-T extension proven by hand (Lemma W(iii) and Theorem 1). |
| **How it could still be wrong** | The noise placement differs from the cited sources' models. Test: compare mechanism sets under SD6. |
| **Killed** | "Published values for gross-code circuits are upper bounds." Smallest counterexample: Pan's public release of 2026-10-20/24. What survives: the statement holds for the arXiv/journal record only. |
| **Open branches** | (1) SD6 mechanism-set equality test; (2) CDP5 re-certification of the three-colour and morphing results; (3) exchange circuits with Pan. |
| **Recommendation** | RESEARCH FURTHER for positioning; IMPLEMENT the three fixes above. |

I did not write agent memory, because the brief makes the ONE_Q tree read-only. Lines to persist:
- **Worked:** GitHub API alongside arXiv (`api.github.com/repos/GavinNull/gross-code-distance-12/releases`) found concurrent certified work.
- **Worked:** reading the supplement against the main text found a fix that never propagated.
- **Did not find a problem:** shipped IBM proof re-check (VERIFIED; 1-byte XOR rejected with reason 4; truncation rejected).
- **Dark instrument:** the arXiv export API returned no entries without any error.

Files are in `<reviewer's local scratch path>`:
- `ccheck.exe`
- `mut.bin`
- `trunc.bin`
- `src_*\` — the downloaded arXiv sources

Sources:
- [Pan, gross-code-distance-12 repository](https://github.com/GavinNull/gross-code-distance-12)
- [Pan, release v1.0](https://github.com/GavinNull/gross-code-distance-12/releases/tag/v1.0)
- [Pan, release v2.0](https://github.com/GavinNull/gross-code-distance-12/releases/tag/v2.0)
- [Tesseract, arXiv:2503.10988](https://arxiv.org/html/2503.10988v1)
- [Strikis, Browne and Beverland, arXiv:2603.05481](https://arxiv.org/pdf/2603.05481)
- [Ehatamm et al., arXiv:2605.16523](https://arxiv.org/abs/2605.16523)
- [Veri-QEC, arXiv:2504.07732](https://arxiv.org/abs/2504.07732)
- [arXiv:2610.03214](https://arxiv.org/html/2610.03214)
