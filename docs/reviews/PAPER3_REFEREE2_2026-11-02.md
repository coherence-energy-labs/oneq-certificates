# Referee report, second (of this round) round, 2026-11-02: paper 3, *Machine-checkable lower-bound certificates for the circuit-level distances of bivariate-bicycle syndrome-extraction circuits*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Paper 3 (circuit-distance), second-round referee report, 2026-11-02**

**Verdict: post after fixes, but not until the re-certification campaign is packaged.** As committed, `main.tex` uses ten `\Recert*` macros that `generated/numbers.tex` does not define (0 of 205 macros), so it does not compile. No headline distance is refuted. The re-certification argument holds as a soundness argument. Two things around it do not hold yet:
- The evidence cited for "regenerates bit for bit" is a control that cannot fail.
- One prior-work attribution is not supported by the two pages the paper cites.

Campaign state now (`C:\lq\recert*\records.jsonl`):
- 268 of 280 cells are done, all VERIFIED, every model byte-identical to its record.
- 12 three-colour cells are missing: X7–X11 and Z1–Z7.
- 2 of the 21 cells whose proofs differ from their records (3c Z8, Z9) have no old-prover reproduction yet.
- `recert_20261102/` does not exist yet.

## FATAL
None.

## MAJOR

**M1. "The prover is deterministic … the rest regenerate bit for bit" is not supported by the control it cites.**
- **Where:** `main.tex:40` (abstract), `:175`, `:307`; `recert_package.py:119-129`; `make_numbers_distance.py:529-530`.
- **The determinism run cannot fail.** All 14 cells in `C:\lq\recert_det\records.jsonl` are morph72 proofs with 0 trace bits (912–1592 bytes).
  - A CDP5 file is `"CDP5"`, a 16-byte header, then per anchor a 1-byte root code and 4-byte child count, then per task an 8-byte length and its bits.
  - morph72 Z1 has 12 anchors and 106 tasks: 20 + 60 + 8·106 = 928 bytes, exactly the recorded size. No thread schedule can change those bytes.
- **"At another thread count" cannot be traced.** No row in any of the four records files has a `threads` field.
- **Half the regenerate-only cells have been produced once.**
  - morph144: 4 cells, all differing from their records, none in the determinism run.
  - 3c Z1–Z10: the records are CDP3 (116–373 MB) and the regenerated proofs are CDP5 (15,157,966 B for Z8–Z10).
  - So 14 of the 28 regenerate-only cells have exactly one CDP5 production.
- **"RecertProofEqual byte-identical" overstates the evidence.**
  - Of the 247 equal proofs so far, 198 have zero trace bits and 40 have at most 94.
  - Only 9 are substantial: 3c X1–X6, X12, Z11, Z12 (6.6–34 MB, up to 2.7×10⁸ bits).
- **The claim is probably true, and the strongest evidence is not cited:**
  - Determinism follows from the code: each task writes its own buffer, and the buffers are written in task order after the parallel loop (`cdist_prove.c:343-361`).
  - The checker's sort comparators are total orders (`ccheck.c:45`, `:79-80`), so derived orders do not depend on the platform.
  - The 9 large 3c proofs were reproduced byte for byte.
  - The 19 CDP3 proofs (up to 372.7 MB) were reproduced by a separately built prover.
- **Fix:**
  - Re-run morph144 Z1/X1 (~2 min to prove) and one 3c Z cell at a different `OMP_NUM_THREADS`.
  - Record `threads` and a binary hash in every row.
  - Make the packager refuse a determinism set with zero trace bits or a single thread count.
  - Rewrite the sentence around the code argument and the large reproductions. Otherwise say "expected to regenerate; N of M regenerated twice", or put the ~0.7 GB on Zenodo.

**M2. Prior-work attribution to Pan is not supported by the cited source.**
- **Where:** `main.tex:54` ("Pan proved … that IBM's circuits have dc=10 for every number of rounds"), `:65` (Table I caption: Crocker and Pan "give 10 for IBM's cycle"), `:260` ("Pan's SAT-certified result for every number of rounds under SD6"), `:299` ("DRAT, LRAT and VeriPB").
- **What I found (fetched today):**
  - The bib target, release v2.0, makes no distance claim about IBM's cycle. IBM's depth-8 circuit appears only as a logical-error-rate comparison, "without idle noise".
  - It names DRAT (drat-trim, cake_lpr) "for the lifts" and VeriPB "for the extended-code distances and list completeness". Those concern Pan's new circuits.
  - The README claims SBB Fig. 12 = 11 under SD6, both bases, any number of rounds, and names no proof system. LRAT appears in neither page.
- **Caveat:** WebFetch returns a model summary, so "not found" is weaker than "absent". Pan's manuscript in the repository may contain the claims.
- **Fix:** cite file and section in Pan's tagged manuscript for each attributed claim. If the IBM claim is not there, attribute IBM = 10 to Crocker only (12 noisy rounds, no certificate) and adjust "agreeing with both earlier results".

**M3. The supplements were not updated to the re-certification. This is the same failure as the 11-02 review's finding 2.**
- **Where:**
  - `WINDOW_PROOF.md:35-38` and `:261-267`: the 3c lower bound rests on `records/3c_shift_certificates_w10.jsonl` "all VERIFIED". Those are checker-of-their-day verdicts; Z1–Z10 are CDP3 proofs, and the generator counts verdicts that lack the word "irreducible" (`TcShiftPreFix`, `make_numbers_distance.py:624-625`).
  - `FOLD_PROOF.md:400-410` cites the original morph144 records.
  - The morph72 bullet there says the T=1 shift certificate "excludes every solution". It excludes every irreducible one, which is the exact distinction `main.tex:242` reports as a Lean finding. The conclusion d ≥ 6 still follows.
- **Why it recurs:** only the IBM corollary is gated (`make_numbers_distance.py:328`).
- **Fix:**
  - Cite the `recert_20261102` manifest in both Corollary 3s.
  - Change "every solution" to "every irreducible, hence every minimum-weight, solution".
  - Add `need()` gates like line 328 for the 3c and morph corollaries.

## MINOR
1. **Proof sizes come from the old records** (`main.tex:264`). The sentence says the proofs are "TcShiftMBLo to TcShiftMBHi MB, all verified by the committed checker". The generator reads those sizes from the original records (`:628-630`), giving 6.6–372.8 MB. The committed checker never saw a 372.7 MB proof; the re-certified Z proofs are 15.2 MB. Take the sizes from the manifest.
2. **Two phrases in `main.tex:175` overstate.**
   - "In one campaign": IBM's re-certification is a separate campaign (2026-10-26), and the 11-02 run was three processes plus a determinism run.
   - "Every re-exported model is byte-identical to the model its original search ran on", read over IBM, conflicts with `:258`: only `IbmRecertSame` = 2 of 22 IBM models are.
3. **"The commit of its day" is not what was shown** (`:175`). The test shows that the prover at 72d03874 reproduces the records. The commit message of 9c858cac says its proofs are byte-identical, so the test does not identify which commit wrote them.
4. **The generator's re-certification block accepts any VERIFIED line.**
   - It checks only `checker.startswith("VERIFIED")` (`:515`). It does not check the weight, the "earliest round ≤ m" scope, or the irreducible form, which the IBM block does check (`:454-456`).
   - The coverage key (`:511`) omits `kind` and `maxround`.
   - Current data are clean: 0 of 268 verdicts name a weight other than the cell's. The gate still cannot catch a mis-parameterized re-run.
5. **The records do not bind binaries.**
   - Rows hash the working-tree source at row-write time, not the compiled binary (`recert_campaign.py:106-112`).
   - The packager reads the proof magic from a file it does not re-hash for regenerate-only cells (`recert_package.py:145`).
   - I checked the three campaign binaries: they differ only at bytes 136-137 and 216-217 (PE timestamp and checksum).
6. **The 3c and morph144 models do not ship.** `main.tex:307` says the regenerate-only cells "ship as their models". But `recert_package.py:43` and `:153-162` copy models only for morph72 and morph2026. Ship the 28 models; they are small.
7. **SD6 equality is claimed beyond what was compared.** "Which is our model" (`:260`) extends SD6 equality past T ≤ 4, which contradicts `:113`. Either apply Theorem 1 to the SD6 transform (compare S′, I′, End′ with S, I, End) or scope the sentence. Separately, the documented command (`sd6_equivalence.py:3`, defaults at `:136-137`) does not regenerate the committed record (4 circuits × T = 1..4).
8. **Rule W's stated precondition is stronger than needed and false in one case.**
   - `main.tex:192` says both rules need P1–P3 on every consecutive pair, but P3 fails at (1,2) in memory-Z (`:200`).
   - Lemma W(ii) needs only the windows of models with T ≥ T₁.
   - State rule W's actual requirement.
9. **Most morph certificates carry no trace bits.**
   - All 14 morph72 proofs and 198 of the 238 morph2026 proofs are header-only, so the checker's verified terminal procedure re-runs the whole search.
   - Say so; "about one bit per node" and "the proof ships" suggest otherwise.
10. **The trusted base omits the detector round labels and the exporter's time ordering**, which rule S's scope depends on. They are partly cross-checked by window_uniform BIND at the listed T values.

## Earlier findings
**11-02 review:**

| # | Status |
|---|---|
| 1 | Fixed in text (Pan and Crocker cited, no priority claim), but see M2 |
| 2 | IBM part fixed and gated; the 3c and morph parts recur (M3) |
| 3 | In progress (the campaign); see M1 and minor 6 |
| 4, 5, 7, 9, 11, 12 | Fixed |
| 6 | Adequately addressed (`main.tex:203`) |
| 8 | Fixed (SD6 comparison), with the over-reach in minor 7 |
| 10 | Disclosed, not recorded |

**10-29 review:** #1, #2, #3, #10 and #12 verified in the text.

## Checks that found no problem
- **Checker and prover versions:**
  - HEAD `ccheck.c` LF sha256 is 263a54bfa254, the same as at 493facf7, the version SPEC.md models. Every campaign row has that checker hash and prover hash 2339dad4478e (HEAD).
  - Neither file has changed since 2026-10-22.
  - The CRLF/LF pin change is consistent, and the generator ties both forms to the file (`:461-472`).
- **Old-prover reproductions:** 19 of 19 reproduce the record (CDP3, exit 0). Packager and generator refuse a differing cell without one, including when `proof_equal` is None.
- **Packager and generator refusal logic otherwise can fail:** coverage, verification, model identity, source hash and CDP5 magic are all checked.
- **Theorem window:** matches WINDOW_PROOF Theorem 1 (j_s = 1). I re-read the Lemma W(iii) chaining and embedding: sound.
- **Theorem fold:** matches FOLD_PROOF Theorem 1 (A)(B)(C). The morph72 upper bound via (A) is correct.
- **SD6:** the transform implements the stated SD6. The record covers 4 circuits × 2 bases × T = 1..4, all equal, and every control changes the set. The generator gate (`:311-319`) can fail.
- **Crocker citation is accurate:** Zenodo v1, 2026-10-18; both bases; 12 noisy + 2 noiseless rounds; the same three reductions; "There is no machine-checkable certificate".
- **Pan's README:** morphing work is "in progress", not claimed, so "new here" for the morphing circuits survives against Pan.

## What I could not check
- Pan's manuscript in the repository.
- Whether anyone else has published exact morphing-circuit distances; I did not re-read Webster et al. or Shaw–Terhal 2026.
- The 12 missing cells, the Z1–Z9 reproductions, and the packaged manifest.
- The generator's re-certification block on real data, and whether a negative test exists for it.
- The Lean sources and the C-to-Lean correspondence.
- FOLD_PROOF §§1–3; I relied on the 10-22 audit.
- Thread counts of any run.

## Research contract

| | |
|---|---|
| **Strongest surviving result** | All distances in the stated placement. The re-certification logic is sound: a new CDP5 proof, the committed Lean-modelled checker, and byte-identical models. |
| **Killed** | "Regenerates bit for bit, shown by a second run at another thread count." Witness: 14 header-only proofs, thread count unrecorded. What survives: determinism follows from the code, plus 9 large CDP5 and 19 CDP3 byte-for-byte reproductions. |
| **Open branches** | (1) A non-trivial determinism run (minutes); (2) Pan's manuscript text; (3) the SD6 every-T extension via Theorem 1. |
| **Recommendation** | IMPLEMENT the fixes, then SHIP after the campaign packages. |

Memory was not written, because the tree is read-only. Lines worth persisting:
- **Dark-control witness:** a determinism check on a CDP5 proof with zero trace bits cannot fail; require trace bits > 0 and a differing thread count.
- **Recurring class:** a fix to the main text that never reaches the supplement. Only a `need()` gate stops it.

Scripts are in `<reviewer's local scratch path>`:
- `a1_recert.py`
- `a2_more.py`

Run each with `python -I -B`; both are read-only and take under a minute.
