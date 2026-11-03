# Paper 3 (circuit-distance): notes for the owner

**Status: partial.** This is a draft. It builds with no failures in draft and release mode
(`python paper/tools/build.py circuit-distance [--release]`): 9 pages including references, arXiv
abstract 1,806 of 1,920 characters, no undefined references, no overfull boxes, all floats placed
before the final section. It has not had an independent review, and five preprints still need
verified bibliography entries (see "References" below).

Files (all new; nothing existing was modified):

- `paper/circuit-distance/main.tex` is the manuscript.
- `paper/tools/make_numbers_distance.py` writes `generated/{numbers,tab_summary,tab_morph,tab_costs}.tex`.
  It defines 189 macros, and each one is preceded in `numbers.tex` by a comment naming its source file.
- `paper/circuit-distance/NOTES.md` is this file.

## How the numbers are bound

Every number in `main.tex` is a macro from the generator. The only bare digits left are definitional
constants of the rules (leaf budget 2, terminal budget 3, rounds 0/1 of rule S), labels (P1-P3), tool
names (RC2, SHA-256) and table spacing. The generator refuses (`SystemExit: REFUSED: <reason>`) when:

- `results.json` differs from what `collect_results.py` computes from `records/` today;
- the Lean receipt does not certify the current sources (`lean_receipt.check()`);
- a distance derived from the records (certified weight w, witness weight w + 1) disagrees with the
  claim string;
- a certificate is not VERIFIED on the very model hash its search ran on;
- a shipped proof or model does not hash to its manifest;
- a prose sentence it parses has changed shape.

Negative controls I ran on copies of the inputs: a missing record, a doctored checker verdict, a summary
that drifted from its records, a tampered shipped proof, and a certificate on another model. All five
refuse, each for its own reason, and the untouched copy passes. The written files equal a fresh
generation. The repo's provenance gate (`tools/claims_provenance_gate.py --docs <the new files>`)
passes on all five new .tex files.

## Claims and their evidence

Classes: PROVEN = hand proof in the repo, audited once by an AI agent that did not write it.
MACHINE-CHECKED = a certificate accepted by `ccheck`, or a finite hypothesis checked by a script whose
negative controls fail. KERNEL-CHECKED = a Lean theorem on the standard axioms (a subclass of
machine-checked). MEASURED = an uncertified computation. HYPOTHESIS = an untested explanation.
DERIVED = a short argument given in the paper.

| # | claim (as the paper states it) | class | backing files |
|---|---|---|---|
| 1 | IBM depth-8 cycle, gross code: d_circ = 10 for every T >= 2, memory-Z and memory-X | PROVEN from MACHINE-CHECKED premises | `evidence/circuit_distance_exact/results.json` (ibm_gross); `records/ibm_proofs_w9.jsonl` (22 VERIFIED, w <= 9, T = 1..11); `records/ibm_witnesses.jsonl`; `experiments/circuit_distance_exact/WINDOW_PROOF.md` Thm 1; `records/window_uniform_run.jsonl`, `window_review_*.jsonl` |
| 1a | ... its lower bound for T <= 11; its witness verified directly at T = 2..6 | MACHINE-CHECKED | same records; `window_uniform_run.jsonl` WITNESS |
| 1b | at T = 1 only d_circ >= 10 is known | MACHINE-CHECKED (lower bound only) | `ibm_proofs_w9.jsonl` |
| 2 | 3-colour cycle (2603.05481): d_circ = 11 for every T >= 2, both bases; the conjectured value holds for the circuit as rebuilt and under our noise placement | PROVEN from MACHINE-CHECKED premises | results.json (three_colour, three_colour_every_T); `records/3c_T2_certificates_w10.jsonl`, `3c_shift_certificates_w10.jsonl` (24 VERIFIED), `3c_sweep_shift_w10.jsonl`, `3c_witnesses_w11.jsonl`, `3c_witness_embedding.jsonl`; WINDOW_PROOF.md |
| 3 | morphing [[72,12,6]] (2407.16336): d_circ = 6 for every T >= 1, both bases; MACHINE-CHECKED for T <= 7 | PROVEN from MACHINE-CHECKED premises | results.json (morph72); `records/morph72_*`; `FOLD_PROOF.md` Thm 1; `records/fold_uniform_hom_x.jsonl`, `fold_review_cor2.jsonl` |
| 4 | morphing [[144,12,12]]: d_circ = 12 for every T >= 1, both bases (no loss to the circuit); MACHINE-CHECKED for T <= 13 | PROVEN from MACHINE-CHECKED premises | results.json (morph144); `records/morph144_*`, `fold_lemma.jsonl`; FOLD_PROOF.md |
| 5 | 14 circuits built from 2604.09797: values in Table II; 10 standard schedules hold for every T, 4 tree circuits MACHINE-CHECKED to T = 6 | PROVEN from MC premises (standard); MACHINE-CHECKED (trees) | results.json (morph2026); `records/morph2026_*`; `experiments/circuit_distance_exact/morph2026_targets.json` |
| 5a | these are the paper's own circuits | not claimed (the paper names no groups or polynomials) | `records/morph2026_codes_N42.jsonl`, `_N30.jsonl` |
| 6 | c42a (6/6) and t30a (5/5) one-round distances are Lean theorems | KERNEL-CHECKED | `lean_r198_receipt.json`; results.json kernel_checked_T1 |
| 7 | the checker's certificate calculus is sound (cert_sound and 39 others, standard axioms) | KERNEL-CHECKED (Lean) | `lean_r198_receipt.json`; `lean/CertR198/SPEC.md` |
| 8 | ccheck.c implements that calculus | argued, not proved; brute-force oracle MEASURED (4,798 queries, 9 mutants caught) | SPEC.md table; RESULTS.md (prose only) |
| 9 | the formalization found false-VERIFIED inputs (repeated detector), a prover crash and an overstated verdict, none affecting a recorded result | MEASURED (exposure: 825 model files, 0 repeated detectors) | SPEC.md (prose only) |
| 10 | stim's fault set equals an independent backward Pauli pass on 66 models | MEASURED | `records/dem_independent.jsonl`; results.json trusted_base |
| 11 | the projection is exact for these circuits | MEASURED (test on 3 two-round models); PROVEN for morphing (Thm 2 part C) | `tests/test_r198_circuit_distance_exact.py`; FOLD_PROOF.md |
| 12 | min over mechanisms = min over faults | DERIVED | paper Sec. II A; FOLD_PROOF.md section 1.5 |
| 13 | every decoder fails at order p^ceil(d/2) | DERIVED (standard argument) | paper Sec. I |
| 14 | Glucose stalled at d >= 4 after 70 s; RC2 did not finish [[72,12,6]] in ~10 min | MEASURED, prose only, no raw record | RESULTS.md |
| 15 | Tseitin hardness explains the stall | HYPOTHESIS (stated as "plausible") | none |
| 16 | CDP5 proof 18.3x smaller (IBM Z, T = 2, w <= 9); 21.3x (3-colour X, w <= 8) | MEASURED | `r205_cdp5/manifest.json` |
| 17 | schedule gate: 240 proposals, 12 valid, 11 keep d = 6, 0 improve, 1 drops to 5 (two-round distances) | MACHINE-CHECKED (at T = 2) | `r206_schedule_discovery/registry.json` |
| 18 | memory-X time-shift proofs stop growing in T from T = 5 | MEASURED | `3c_shift_certificates_w10.jsonl` |
| 19 | on the [[72,12,6]] IBM cycle (T = 2, memory-Z), single-qubit noise adds no mechanism | MEASURED (test only) | `tests/test_r198_circuit_distance_exact.py` |
| 20 | literature values: IBM <= 10 by BP-OSD search; 11 conjectured by heuristic search; morphing <= 6 / <= 12 by BP-OSD; 2604.09797 did not check circuit distance (App. C.4); Def. 10, Prop. 1, Fig. 25; Figs. 10-12 of 2603.05481 | taken from the repo's own documents; NOT checked against the papers' text (the 2026-10-29 reviewer confirmed Strikis et al.'s conjecture of 11 and Shaw-Terhal's Prop. 1 from search-result text only) | RESULTS.md; results.json claims; `bb_circuits.py` comment |
| 21 | prior EXACT values for smaller BB circuits: [[72,12,6]] = 6 and [[90,8,10]] = 8 (MaxSAT, 2503.10988 App. D); Gurobi within 8 h only for the first BB circuit (2603.22532 Sec. 5.5.3) | CHECKED in the authors' own LaTeX source, 2026-10-29 | RESULTS.md (the question) |

## What I could not verify, and gaps

1. **The checker version behind the 22 IBM certificates is not in version control.** Their records show 4
   bytes per trace entry, which is the `CDP1` format (int32 entries). The first committed checker
   (`e9f540fc`) reads only `CDP2`. So the proofs were written and checked by a working-tree version that was
   never committed. Both T = 2 models were later re-certified in CDP5 (`r205_cdp5/manifest.json`,
   `r206_performance.json`, model hashes equal). On 2026-10-26 I compiled the current `ccheck.c` (sha256
   `4fc5540e...` over this checkout's CRLF bytes, `263a54bf...` as git stores it; the Lean-modelled
   version) and re-checked both shipped CDP5 proofs: both VERIFIED. A
   flipped byte and a wrong group were both REJECTED. That run is not a committed artifact. The paper
   states the gap (Sec. III F).
2. **Most certificates predate the Lean-driven fixes** (part 14, `f37072d2`). By commit order these are the
   IBM certificates, the 3-colour T = 2 certificates, the morph72 and morph144 certificates, and 8 of the
   24 3-colour time-shift certificates (their verdicts lack the word "irreducible"). The exposure measurement
   (825 models, 0 repeated detectors) covers this; the paper says so.
3. **The campaign's proofs are not in the repository** (only hashes and verdict lines). A third party can
   re-check only the two shipped CDP5 proofs.
4. **Noise placement.** The paper's distances are for the lane's fault sets. I did not compare them with the
   noise models of Bravyi et al., Strikis-Browne-Beverland or Shaw-Terhal, and the repo does not record that
   comparison. Under a placement with more fault locations, a lower bound need not transfer. The paper
   states this (Sec. VII). "Settles the conjecture" is therefore worded as "holds for the circuit as
   rebuilt, under our noise placement".
5. **Prose-only numbers.** These macros are parsed from Markdown, not from a machine record: generic-solver
   stalls (RESULTS.md), the brute-force oracle (4,798 queries, 20 models, 9 mutants, 44-1,941 violations),
   Lean kernel cost (141-624 s, 2.6 GB), the exposure count 825 (SPEC.md), the fold components 5.7e6
   (FOLD_PROOF.md), and figure, appendix and definition numbers of the cited papers. Each is read by a strict
   pattern that refuses if its sentence changes. RESULTS.md's incidence count (42,480) had no evidence-JSON
   home, and the provenance gate flagged it, so it is used only as an internal consistency check and is not
   typeset.
6. **The model behind the Glucose run** is not named explicitly in RESULTS.md (the sentence sits beside the
   gross-code model). The paper avoids naming it.
7. **Stale lane documents, as found.** `RESULTS.md`'s status header still calls the 24 time-shift certificates
   the open item, and its results-table row says "= 11 in memory-Z with our own witness" (an X witness
   exists). The ledger's R198 heading still says "reaches 11 at T = 2". All three are superseded by the later
   sections and by `results.json`. The atlas card says "4 soundness bugs", while RESULTS.md and SPEC.md say
   three plus an overstated verdict. I did not edit any of them.
8. **Lean was not rebuilt.** I relied on the committed receipt, and `lean_receipt.check()` confirms it
   certifies the current sources. The kernel modules were not re-run.
9. ~~No independent review of this manuscript.~~ **First independent review 2026-10-29** (an AI agent that had not
   seen the draft; not third-party peer review): 1 FATAL, 8 MAJOR, 6 MINOR. Report and the disposition of each
   finding: `docs/reviews/PAPER3_REFEREE_2026-10-29.md`.

**Status of the gaps above after that review.**
- Gap 1: the IBM claim now rests on the 22/22 rule-S re-certification, with every proof and model shipped. Each model
  is byte-identical to a fresh export of the cycle (`ibm_recert_models.py`). The campaign's unpreserved-checker
  certificates are history.
- Gap 2: closed mechanically. Independent DEMs equal the exporter's on every certified model, so no certified
  mechanism repeats a detector.
- Gap 5: the generic-solver numbers are WITHDRAWN.
- Gap 6: no longer applies (the Glucose passage is withdrawn).
- Gap 4 (noise placement) and row 20 (source noise models) stay open; the paper now scopes every comparison to "in
  our fault placement".

## References

The shared `paper/refs.bib` may not be edited by this draft. Cited from it (all already fetch-verified):
`bravyi2024high`, `gidney2021stim`, `panteleev2021degenerate`, `roffe2020decoding`,
`cheung2017verifying`, `eifler2023computational`, `mcconnell2011certifying`, `blumkannan1995`,
`ehatamm2026end`.

Five preprints are cited inline by arXiv identifier, each read from the repo. `TODO(ref)` comments mark
them in `main.tex`. Repo-verbatim details:

| arXiv id | repo states | where |
|---|---|---|
| 2603.05481 | Strikis, Browne & Beverland | RESULTS.md |
| 2407.16336 | Shaw & Terhal; PRL 134, 090602 | RESULTS.md, `bb_circuits.py` comment |
| 2604.09797 | Shaw & Terhal, "Optimising Quantum Error Correction Using Morphing Circuits" | FOLD_PROOF.md section 6.4 |
| 2603.22532 | Webster, Jacob and Higgott, "Distance-Finding Algorithms for Quantum Codes and Circuits" | docs/RESEARCH_R205_CDP5.md |
| 2601.12509 | Liu et al., "AlphaSyndrome: ...", ASPLOS 2026 | docs/RESEARCH_R206_PROOF_GUIDED_DISCOVERY.md |

On 2026-10-26 I read arXiv's citation metadata with the repo's own `verify_refs.fetch_arxiv`, read-only.
For all five, the family names the repo gives match arXiv's author list in order. Nothing was written to
`refs_verified.json`.

~~Still uncited, with placeholder comments: Tseitin / resolution lower bounds, Glucose, RC2, Lean 4,
Mathlib.~~ **Resolved 2026-10-29.** Lean 4 (`demoura2021lean4`) and Mathlib (`mathlib2020`) are cited and
Crossref-verified. Tseitin, Glucose and RC2 left with the withdrawn generic-solver passage. Added after the review:
`beni2025tesseract`, `jacoby2026stairway`, `chen2026satqldpc`. `verify_refs.py --fetch`: 58 of 58 agree.

~~Owner action: add the five entries to `refs.bib`, run `python paper/tools/verify_refs.py --fetch`, then
replace the inline identifiers with `\cite`.~~ **DONE 2026-10-26 (review session).**
- The five entries are in `refs.bib`, every field from arXiv's citation metadata.
- `verify_refs.py --fetch`: all 53 entries agree. The fetch also corrected `cao2026differentiable`, which had lost an
  author, Zixuan Lu, in a later arXiv revision.
- The inline identifiers are now `\cite`.

## Review (2026-10-26, the session that commissioned this draft)

- The abstract's three-colour sentence now says "as rebuilt from their figures", matching Sec. VII.
- Two `table*` floats moved to just after their first reference. The release build failed on floats after the final
  section; it now passes, 9 pages.
- Registered in `tests/test_paper_build_tools.py`.
- The provenance gate passes repo-wide. Its five orphans were paper 1's new sums, now derived in
  `tools/derive_claims.py`.
- Still no INDEPENDENT review: this session commissioned the draft and only checked it.

## Open items

- ~~Register paper 3~~ **done** (above).
- **IBM re-certification: DONE 2026-10-26, 22/22 VERIFIED** (gap 1 has a second route now).
  - `evidence/circuit_distance_exact/ibm_recert_20261026/` holds the rule-S time-shift certificates for T = 1..11,
    both bases, w = 9, proved and checked by the COMMITTED prover and the Lean-modelled `ccheck.c`.
  - Every CDP5 proof (3.4 MB) and model is shipped, with a manifest of their hashes.
  - `tests/test_r198_ibm_recert.py` re-verifies two proofs with today's checker and rejects a corrupted one.
  - Rule S needs time-ordered exports, so 20 of the 22 models are byte-different serializations of the campaign's
    circuits. This is a second route to the same lower bound, not a re-check of the original files.
  - ~~Next revision of the paper: cite this route~~ **done 2026-10-29.** It is the primary IBM evidence in Sec. VI A.
- **Submission (owner):** attach `experiments/circuit_distance_exact/WINDOW_PROOF.md` and `FOLD_PROOF.md` as arXiv
  ancillary files (`anc/`). The paper calls them supplementary material.
- A typeset LaTeX appendix for the two proofs, replacing the Markdown supplement (review finding 5).
- A schedule-gate demonstration on a baseline that loses distance (review finding 14).
- Compare the noise placements with the source papers (gap 4).
- Formal (Lean) versions of the window and fold theorems; kernel-checking gross-code models.
- ~~Repo-wide provenance gate~~ **passes** (above).
- The generator imports the lane's `collect_results.py` and `lean_receipt.py`, so it needs stim and numpy
  (about 3 s to run).
- The author messages drafted in the ledger (`bf14e811`) are unsent. The paper does not mention them.
