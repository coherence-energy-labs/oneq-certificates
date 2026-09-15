# Certifying QEC decoders — re-runnable bundle

Everything here can be checked without trusting the people who made it.

This release accompanies two manuscripts by Josh Philbrick (Coherence Energy
Labs), whose LaTeX sources are in `paper/`:

* *Certifying quantum error-correction decoders: independently checkable
  optimality receipts at scale* — `paper/certifying-decoders/`
* *Exact, independently checkable optimality receipts for BP-OSD on the
  [[144,12,12]] bivariate-bicycle code* — `paper/qldpc-certificates/`

Every number in both papers is a macro generated from the evidence in this
tree. Regenerate them and rebuild the PDFs (needs Python and a TeX
distribution with REVTeX 4.2):

    python paper/tools/make_numbers.py
    python paper/tools/make_numbers_qldpc.py
    python paper/tools/make_figures.py
    python paper/tools/build.py certifying-decoders
    python paper/tools/build.py qldpc-certificates

## What runs from this tree, checked from a clean copy

    node verifier-js/verify-certificate.mjs --conformance conformance/cert_vectors.json
    python tools/external/exact_lp_certificate_reference.py --self-test
    python tools/recheck_all.py
    python tools/replay_epsilon_receipts.py --jobs 8
    python -m pytest tests/ -q

`recheck_all.py` is a complete replay: it checks every one of the 2,042
receipts named by the pinned `evidence/receipt_manifest.json` (2,002 flat
certificates and 40 branch-dual trees) with two independently written
checkers, against the persisted instances and integer weight vectors, and
FAILS if any receipt or bound file is missing, duplicated, unlisted or
altered. It uses no solver, no producer and no rebuilt input. `replay_epsilon_receipts.py`
does the same for paper 1's rerun, with a second verifier in scaled-integer
arithmetic and attacks at the acceptance boundary; it needs the receipt archive
extracted under `data/epsilon_receipts/`. Re-running the original experiments needs
the code of the revisions that produced them: the release assets
`oneq-source-<commit>.tar` carry it (the producing scripts and every module they
import), each named with its sha256 in `evidence/revision_snapshots.json`; check a
download with `python tools/revision_snapshots.py --verify --no-git --out <dir>`.
What does NOT run from this tree: the full development test suite, the mutation
gate and `tools/gate_all.py`, and the commands in `AUDIT.md` and `REPRODUCE.md`
that read git history, which need the full development repository. `tools/verify_release.py`
checks a signed release against the issuer pinned in `ISSUERS.json`: put the three
`oneq-certifying-decoder` release assets in `release/` first.

## The one-minute check

    node verifier-js/verify-certificate.mjs --conformance conformance/cert_vectors.json

56 vectors must agree. **48 of them are refusals** — documents that a sound
checker must reject. A suite where everything verifies would not notice a
checker that accepts everything, which is the failure that matters, because
such a checker is worthless and looks perfect.

Then corrupt something and watch it fail. Change one hex float in any `dual`
entry of an accepting vector; the count must drop. A harness that cannot fail
is not evidence.

## What each number means, and does not

| file | claim | what it does NOT say |
|---|---|---|
| `million_shot_gate_d5_merged.json` | 10⁶ shots merged from four disjoint-seed runs, 99.76% accepted under the pre-audit tolerance rule, 6,000,002 forgeries, 0 accepted | nothing about hardware noise; and "certified" here means the pre-audit acceptance rule (float gap ≤ 1e-6) — see the next row |
| `certification_semantics_audit.json` | what that rule proved: the producing code of each headline artifact, re-run on seeded samples, every accepted certificate analysed in exact rational arithmetic — all admissible, largest proven slack far below 1e-6 | not the per-shot slack of the full runs, whose certificates were not stored; and not EXACT optimality, which today's stricter checker measures separately |
| `google_generation_history.json` | the Willow certified rate by producer generation, read from the objects each generation wrote | generations 2 and 3 ran partial coverage; their aggregates weight configurations unequally |
| `gap_distribution_exact_rule.json` | the gap experiment re-run under the exact checker, with per-shot gaps | a different certified subset from `gap_distribution.json`; both are shipped so the difference is visible |
| `cert_conformance.json` | two independent checkers agree 56/56 | agreement is not correctness — it is the absence of a *shared* misunderstanding |
| `decoder_mutation_gate_d*.json` | syndrome consistency catches 0 of the syndrome-consistent fault class; the certificate catches all of it | output mutation models wrong answers, not crashes, hangs or memory errors |
| `epsilon_rerun/<config>/summary.json` | the scale runs repeated under the epsilon checker (exact rationals, eps_max = 1e-6) with every receipt kept: proven-within-epsilon, exact and not-proven counts, largest proven slack, forgeries | the receipts are a separate archive; `replay_<config>.json` records a complete two-verifier replay of them |
| `detection_outcomes/d*.json` | seeded faults and a stale calibration judged against the exact optimum: outcome states, false-alarm base rates, discordant logical cells | the stale-calibration result is objective mismatch against a supplied reference model; it cannot say which model describes a device |
| `mutation_score/*.json` | operator mutation scores of both checkers' tests, every surviving mutant declared equivalent with its reason | a score measures the tests against mechanical defects, not the absence of defects |
| `willow_residual_shots.json` | the two Willow shots generation 6 left open: PyMatching's correction is exactly optimal and the exact tier stopped at its round cap | two shots, not a rate |
| `gap_distribution.json` | BP-OSD's per-shot optimality gap against a checkable bound | a gap is **not** a defect: BP-OSD maximises likelihood over the hypergraph, not matching weight |
| `seeded_fault_corpus_d*.json` | the faults themselves | released so you can score *your* detector; the two numbers are only comparable if the corpus is shared |

## The claim in one paragraph

Minimum-weight decoding is a minimum-weight T-join. Its dual is a cut packing:
vertex sets with non-negative weights, no edge over-subscribed. Any feasible
packing lower-bounds the optimal correction **by weak duality, regardless of
how it was produced** — so a checker need not trust or even know the producer.
When the packing's value equals the correction's weight, both are optimal, and
verifying that takes time proportional to the packing's size and the edges it
touches, with no solver.

## Two things the authors got wrong, kept here on purpose

An "attack" was accepted 16 times and looked program-ending. The 16 were
**equal-weight alternative optima** — minimum-weight corrections are not
unique, and accepting a different optimum is correct. Later a second attack was
accepted on 4 of 147 shots; an audit outside the checker's own code path found
zero negative duals, zero violated constraints and objective equal to primal
within 9e-16. It had degenerated into a **valid certificate**.

Both times the checker was right and the attack was wrong. That is why every
attack in `forgeries.py` now carries the property making its claim false, and
why that property is re-derived independently before any acceptance is counted.
Without that audit, "0 forgeries accepted" could mean the battery had quietly
stopped attacking.
