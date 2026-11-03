# Paper 3 (`paper/circuit-distance`): first independent review, 2026-10-29

**Reviewer.** An AI agent (the `adversary` type) that had not seen the manuscript or its drafting. It was given the
manuscript, the evidence paths and the instruction to act as a hostile referee: trace every headline number, check
soundness against `lean/CertR198/SPEC.md`, run a fresh literature search and spot-check citations. It worked
read-only and ran no proof search, Lean build or test suite. **This is not third-party peer review.**

**Verdict as given:** "Post only after the fixes above." 1 FATAL, 8 MAJOR, 6 MINOR. The core results traced cleanly.
The defects were in the claims made around them.

**Checked before acting.** The FATAL finding's quotations were confirmed in the authors' own LaTeX source:
- arXiv 2503.10988, `main.tex` line 440: "We verified with a MaxSAT solver [...] that the circuit distance of the
  [[72,12,6]] code is 6 [...] and the circuit distance of the [[90,8,10]] code is 8";
- arXiv 2603.22532, `main.tex` line 2512: Gurobi "completed within 8 hours only for the first member of the family";
- arXiv 2606.12445 computes code distances, not circuit distances;
- arXiv 2603.00228 reports circuit-level distances for its own codes, by an ILP decoder, with upper bounds where the
  ILP did not finish.

## Findings and what was done

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | FATAL | Abstract: "published values are upper bounds from heuristic searches". False for smaller BB circuits (Tesseract App. D: [[72,12,6]] = 6, [[90,8,10]] = 8, MaxSAT). | **Fixed.** The abstract, introduction, Table I (the IBM-style row now shows "6 (exact)" with the citation) and related work all say that exact solvers settled the small circuits and that the gross-code values were upper bounds. Each number is a macro read from `RESULTS.md`, whose arXiv ids the generator ties to the bibliography entries. |
| 2 | MAJOR | IBM headline rests on an unpreserved checker; the committed 22/22 re-certification is uncited. | **Fixed.** Sec. VI A now rests on the rule-S re-certification. The generator re-hashes every shipped proof and model and checks the checker hash against SPEC.md's. **New evidence:** `ibm_recert_models.py` re-exports all 22 models from the cycle's builder, and all 22 are byte-identical to the shipped files (`records/ibm_recert_reexport.jsonl`). |
| 3 | MAJOR | "the two sets are equal on all 66 models" omits the 2026 circuits. | **Fixed by measurement.** `dem_independent_morph2026.py` runs the unchanged comparison on every (target, basis, T) the campaign used, including both models of each window pair: 210 models, 180 to 12,624 mechanisms, all equal (`records/dem_independent_morph2026.jsonl`). The generator requires exact coverage. |
| 4 | MAJOR | "tight", "exact" and "the value they conjectured" are used without comparing fault models. | **Scoped.** Each now reads "in our fault placement"/"attained". Sec. VII says these compare numbers across conventions and states when a lower bound transfers. **Open:** the per-source comparison of noise models. |
| 5 | MAJOR | The every-T theorems' proofs are not in the paper; Theorem 1's notation. | **Partly fixed.** The proofs ship as supplementary files (`WINDOW_PROOF.md`, `FOLD_PROOF.md`) and the paper says so. Theorem 1's map is renamed μ, and h, e are defined as the least integers satisfying the identities. **Open, deliberate:** a typeset LaTeX appendix (about 1,000 lines of Markdown; Markdown ancillary files were judged an acceptable supplement for a first posting). |
| 6 | MAJOR | The generic-solver passage has no raw record and is contradicted by prior art. | **Withdrawn.** It is deleted from the paper. `RESULTS.md` marks it WITHDRAWN, and the generator refuses if that mark goes. |
| 7 | MAJOR | Table II's tree rows are indistinguishable, and "degree" is undefined. | **Fixed.** The caption lists every tree schedule from `morph2026_targets.json` and defines degree. The generator refuses if the trees stop sharing A, B and case, or if two trees share a schedule. |
| 8 | MAJOR | The title and abstract overstate the scope ("published circuits", "exact"). | **Fixed.** New title: "Certified circuit-level distances of syndrome-extraction circuits for bivariate-bicycle and two-block group-algebra codes". The abstract states the fault-placement scope, "as we rebuilt it" and "re-found". |
| 9 | MAJOR (low risk) | "no model of the campaign contained one" has no record. | **Fixed mechanically.** The independent derivation writes symptoms as sets and equals the exporter's list on every certified model (66 + 210), so no certified mechanism repeats a detector, and the projection keeps distinct detectors distinct. |
| 10 | MINOR | "A certificate is bound to the SHA-256 of the model": the checker reads no hash. | **Reworded.** The binding is in the run records, and the checker does not read a hash. |
| 11 | MINOR | "Every rejection path has a test": 8 named, at least 17 rejection sites exist. | **Narrowed** to the tests that exist. It adds the repeated-detector test (`test_a_mechanism_repeating_a_detector_is_refused_by_every_program`). |
| 12 | MINOR | "no C code trusted" holds only for the kernel check. | **Scoped:** "of the exported model ... no C code trusted in that check"; the model's own provenance is stated. |
| 13 | MINOR | `TODO(ref)` placeholders and unengaged prior art. | **Fixed.** Lean 4 and Mathlib are cited (Crossref-verified). Tesseract, Stairway codes and Chen–Jafari–Lai are cited and engaged. The Tseitin, Glucose and RC2 mentions went with finding 6. `verify_refs.py --fetch`: 58 of 58 agree. |
| 14 | MINOR | The schedule gate's "none improved it" is vacuous: the baseline is already at the code distance. | **Fixed.** The paper says none could, so this instance shows only rejections. The generator holds the premise (baseline = code distance = the published exact value). **Open:** a demonstration on a baseline that loses distance. |
| 15 | MINOR | The term "affine". | **Defined** after Lemma 2. |

## What the reviewer could not check (still true)

- No proof was re-checked by the reviewer; `tests/test_r198_ibm_recert.py` re-checks two shipped IBM proofs and
  rejects a corrupted one.
- The hand proofs were not re-audited in this round.
- The source papers' noise models were not compared (finding 4, open).
- The Lean sources were not read; the committed receipt was relied on.
