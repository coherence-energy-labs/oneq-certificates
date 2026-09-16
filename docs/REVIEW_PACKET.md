# Review packet — both manuscripts, 2026-09-15

**Evidence class:** [PROJECT] a reader's guide to what each paper claims, where the evidence for each
claim lives, and the commands that re-derive it. Nothing here is evidence itself; every line points at
an artifact or a command.

Two manuscripts, one repository, one public release
(https://github.com/coherence-energy-labs/oneq-certificates, release v2026.09.16):

| | Paper 1 | Paper 2 |
|---|---|---|
| Title | Certifying quantum error-correction decoders: independently checkable optimality receipts at scale | Exact, independently checkable optimality receipts for BP-OSD on the [[144,12,12]] bivariate-bicycle code |
| Source | `paper/certifying-decoders/main.tex` | `paper/qldpc-certificates/main.tex` |
| Long-form record | `docs/PAPER_DRAFT.md` (the typeset manuscript governs where they differ) | `docs/PAPER2_QLDPC_DRAFT.md` (same rule) |
| Numbers | `paper/certifying-decoders/generated/*.tex`, from `paper/tools/make_numbers.py` | `paper/qldpc-certificates/generated/*.tex`, from `paper/tools/make_numbers_qldpc.py` |
| State on 2026-09-16 | **Draft-complete except 16 numbers** that wait on the Willow d=7×30 rerun (below); everything else builds and traces | **Complete after the 2026-09-15 follow-up audit** (`docs/EXTERNAL_AUDIT_2026-09-15.md`: 20 findings, 17 closed, 3 open and named: BP-OSD's outcome on the 149 circuit shots, timestamping future freezes, paper 1's rerun-cost sentence); builds clean, every number traces, every proof retained and replayed |
| Length | 9 pages, abstract 1,527 characters | 7 pages, abstract 1,887 characters (arXiv limit 1,920) |

## How to read them in the order that catches the most

1. **Build the PDFs yourself**, so you are reading what the artifacts say and not a stale copy:
   `python paper/tools/build.py certifying-decoders` and `python paper/tools/build.py qldpc-certificates`.
   The builder fails on any undefined reference, any BibTeX warning, any overfull line, a float
   placed after the last section, and an abstract over arXiv's limit. Output lands in
   `build/main.pdf` under each paper's directory. Paper 1 typesets its pending numbers as **[pending]** in bold and
   `build.py --release` refuses while any remain.
2. **Regenerate every number**: `python paper/tools/make_numbers.py`, `python paper/tools/make_numbers_qldpc.py`,
   `python paper/tools/make_figures.py`, then `git status paper/` — nothing should change. Each
   generator refuses to typeset a claim its artifact contradicts (it exits with the sentence it would
   have typeset).
3. **Check the references**: `python paper/tools/verify_refs.py` — 48 entries against sources fetched
   2026-09-14 (`paper/refs_verified.json`). Two citation defects were found and fixed on 2026-09-13
   (an invented author list and an invented venue; ledger R57); that is why this check exists.
4. **Replay the receipts** rather than trusting the counts. Paper 2: `python tools/replay_receipts.py`
   replays all 2,042 stored receipts (2,002 flat, 40 trees) by two Python checkers against a pinned
   manifest in about a minute; `python tools/qldpc_js_agreement.py` and
   `python tools/qldpc_tree_js_agreement.py` decide the same receipts with two JavaScript checkers
   written from the specification and refuse 104 + 55 planted defects, each for its own reason.
   Paper 1: `tools/replay_epsilon_receipts.py` re-decides every retained ε-receipt of the rerun with a
   second, integer-arithmetic verifier and runs the boundary, non-finite and graph-substitution
   attacks (receipts are release assets; sha256 pinned in `evidence/epsilon_rerun/*.json`).
5. **Then read the text against the registers.** `docs/EXTERNAL_AUDIT_2026-09-14.md` and
   `docs/EXTERNAL_AUDIT_2026-09-15.md` hold every finding three external audits raised against both
   papers, its verdict after re-verification here, and its closure with the commit that closed it. A referee's likeliest objections are already rows there; the useful review
   is the objection that is not.

## Paper 1 — the claims, and where each one lives

| Claim in the text | Class | Artifact / command |
|---|---|---|
| Any feasible T-odd cut packing lower-bounds every correction; scaling by θ makes any non-negative packing feasible with a proven slack ε (the lemma) | PROVEN | Sec. IV; the auditor's independent script reproduced 1,000 packings with 0 violations (register, "What survived") |
| 99.81% of 10⁶ simulated d=5×25 shots proven within 10⁻⁶ of optimal, 8,628 exactly, largest slack 6.2×10⁻¹²; 8,000,001 forgeries in nine classes, none accepted | MEASURED | `evidence/epsilon_rerun/sim_d5_r25/summary.json`, replay `evidence/epsilon_rerun/replay_sim_d5_r25.json` |
| Willow: ten configurations at generation 4 complete under the ε-checker; d=7×30 X and Z **pending** | MEASURED / PENDING | `evidence/epsilon_rerun/willow_*/summary.json` and `evidence/epsilon_rerun/replay_willow_*.json`; the d=7×30 rerun log is `%LOCALAPPDATA%\oneq_rerun\d7x30.log` |
| The original tolerance-rule rates (99.76% / 99.69%) are historical; the audit of 1,588 accepted certificates found none inadmissible and a largest slack of 2.6×10⁻¹³; the exact checker accepts 842 of them | MEASURED | `evidence/certification_semantics_audit.json`; Table I |
| Seeded faults: 3,549 invalid + 1,312 proven suboptimal of 4,861; base rate 1 refusal in 770 (base ladder), 0 (full); 62 equal-weight controls certified | MEASURED | `evidence/detection_outcomes/` |
| Stale calibration: 547 suboptimal all proven, 2 false alarms in 6,147, discordant cells at d=3 | MEASURED | `evidence/stale_calibration*.json`, Table II |
| Both residual d=7×30 shots: decoder exactly optimal, exact tier capped at 200 rounds | MEASURED | `evidence/willow_residual_shots.json` |
| Certification costs ~6,610× batch decoding | MEASURED | `evidence/decode_vs_certify.json` |
| Literature: no forgery battery, upstream-fault experiment or hardware certification rates found elsewhere | OBSERVED (search of 2026-09-15) | `docs/PAPER1_LITERATURE_SEARCH.md` |

**What is pending and why.** The 16 `[pending]` macros are the Willow totals (`EpsHw*`), the forgery
total, the largest slack over all reruns, and the boundary-attack counts of the complete replay. They
wait on the d=7×30 X and Z reruns (200 chunks of 250 shots each per basis, full ladder with the exact
tier). The reboot of 2026-09-15 15:09 killed that run at X chunk 41 of 200; it was relaunched detached
at 19:03 and resumes from its chunk index. Measured on the relaunched run (first chunks, 3 workers, a
shared machine under load): a median of 3,676 s per chunk, the slowest 6,359 s. At that rate X needs
about two days more and Z about two days after it, then the merge, the two-verifier replay and
`make_numbers.py`; an idle machine or more workers shortens it. Until then the paper reads correctly with those
16 numbers shown as pending; nothing else in it depends on them.

**What the paper does not claim** is collected in Sec. IX: not real-time; not complete; ε-optimality,
not exact optimality at scale; optimality for the given weights only; blind to tied optima.

## Paper 2 — the claims, and where each one lives

| Claim in the text | Class | Artifact / command |
|---|---|---|
| Safe-dual lemma, lattice rule, branch partition, infeasibility witness | PROVEN | Secs. II–III; auditor's independent script: 4,000 safe-dual bounds vs exhaustive, 419 + 500 lattice acceptances, 0 violations |
| Adaptive separation rule is exact (Taghavi–Siegel), restated for syndromes | PROVEN + MEASURED | Sec. V; `qldpc_claims_backing.json:separation_bruteforce` (8,000 instances, 0 suboptimal) |
| Preregistered campaign: 2,019 nontrivial shots, 2,002 flat + 17 trees, 1,780 tier-1, 222 repairs | MEASURED, preregistered | `docs/HELD_OUT_PREDECLARATION.md` (tag `held-out-freeze`), `evidence/qldpc_heldout.json`, Table I |
| Every stored receipt replays under two Python checkers and a JavaScript checker per receipt kind | MEASURED | `tools/replay_receipts.py`; `evidence/qldpc_js_agreement.json`; `evidence/qldpc_tree_js_agreement.json` |
| Circuit level under the published depth-8 schedule: 149/149, transcription validated at fault level (0 mismatches) | MEASURED | `evidence/qldpc_refsched_fault_xval.json`; Sec. VI B |
| Strong baseline: pipeline errors change with the BP-OSD configuration; the advantage is over the stated configuration, not BP-OSD as such | MEASURED, every proof retained | `evidence/qldpc_strong_baseline_v3.json`, Table II; the 955 proofs under `evidence/strong_baseline_v3/`; `python tools/replay_strong_baseline.py` rebuilds every number from records and proofs alone (`evidence/strong_baseline_v3_replay.json`) |
| The instrument, two ways: tier-1 (the flat route proves BP-OSD's answer) and optimum-hit (BP-OSD's answer feasible at the certified optimum) at all five levels; at p=0.07 under C, 146 vs 196 of 200 | MEASURED | Table III, same artifact and replay |
| The frozen held-out campaign re-runs from the public source snapshot alone, no git history | MEASURED | `python tools/reproduce_heldout_public.py --work <dir>`; `evidence/heldout_public_reproduction.json` (8 rungs, 14 fields, all record files byte-identical, 155 s) |
| Mutation score of the checker's tests: 313 of 321 killed, 8 equivalent with reasons | MEASURED | `evidence/mutation_score/qldpc_check.json` |
| 58 of 2,002 regenerated flat certificates differ from the run of record while remaining exact; cause open | OBSERVED | Sec. VII; ledger R56b |
| "Under a logged search protocol we did not find a prior report of exact, independently re-checkable, minimum-weight receipt coverage for BP-OSD on this code" | OBSERVED (search of 2026-09-15) | `docs/PAPER2_LITERATURE_SEARCH.md`; nearest: arXiv:2608.25545, cited and distinguished in Sec. VIII |

**Every shot is simulated** and Sec. VII says so first; no device has run this code.

## What a reviewer should know about the review history

- Every review round before today was AI-assisted (`docs/reviews/`, `docs/PAPER2_QLDPC_DRAFT.md` header)
  or an external written audit answered in `docs/EXTERNAL_AUDIT_2026-09-14.md`. **No human peer
  review has occurred.** That is what this packet is for.
- The manuscripts' own numbers cannot drift from their artifacts: the generators refuse, the claims
  gate (`python tools/claims_provenance_gate.py`) traces every material number in the documents to the
  evidence tree, and `tests/test_paper_*.py`, `tests/test_doc_claims_*.py`,
  `tests/test_bundle_carries_the_papers_evidence.py` hold the builder, the bibliography and the bundle.
- What cannot be checked from the release alone: re-running the sampling and production needs the
  code at the freeze commits, shipped as source snapshots in release v2026.09.16
  (`evidence/revision_snapshots.json`); the 17.3 GB of ε-receipts for paper 1 are release assets, not
  repository files.

## Submission notes (for the owner, not the reviewer)

- Each paper cites the other by title as a companion; on submission, cite the arXiv identifiers.
- `paper/owner.tex` holds the author block; `build.py --release` refuses placeholders and pending
  numbers, and writes `build/arxiv_metadata.txt` with the plain-text abstract.
- Paper 1 should not be released until the d=7×30 reruns merge and its complete two-verifier replay
  exists; the generator enforces this.
