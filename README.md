# Certifying QEC decoders — re-runnable bundle

The shipped certificates can be replayed against their stated models. The checker
implementations and execution environment remain part of the trust boundary;
missing proof files are reported explicitly below.

## Corrections and new replayable evidence (2026-11-03)

The historical decoder-gap comparisons in `evidence/gap_distribution.json`
and `evidence/gap_distribution_exact_rule.json` overcount repeated DEM
components that map to the same graph edge. Their numerical comparisons are
**superseded**, including the corresponding tables and figure in the historical
matching manuscript. The retained historical files are unchanged. The corrected
comparison projects selected components modulo two, sums canonical graph-edge
costs as exact rationals, and reports a TRAIN-selected BP-OSD baseline first.
It uses an independent graphlike-component approximation; it does not implement
maximum likelihood on the full correlated hypergraph. See
[the corrected protocol](docs/CORRECTED_GAP_BENCHMARK.md) and
[the measured evidence](evidence/corrected_gap_20261103/README.md).

New native Sparse Blossom tooling exports the dual from the same matching solve,
with a separate standard-library checker. All 256 retained integer-graph proofs
verify. Original IEEE-weight claims require additional exact reconciliation and
can miss the requested epsilon. The measured native production costs differ
substantially from the historical post-hoc Python ladder; an absent Python
dual-export API does not establish an inherent cost of certification. Build,
license, objective and exploratory timing details are in
[the native package](tools/native_matching_duals/README.md).

Replay both new evidence sets from this checkout:

```bash
python tools/recheck_corrected_gaps_current.py evidence/corrected_gap_20261103/summary.json
python -I -S tools/native_matching_duals/replay_native.py
```

The corrected-gap command explicitly rechecks the unchanged packet under the
current stricter matching-endpoint contract and reports its source difference.
The original source-pinned replay remains available at the measured revision
and still refuses changed checker bytes.

The native replay requires only the Python standard library. It rejects missing
or duplicated shots, recomputes every proof and original-weight bound, and checks
retained summary arithmetic; it cannot reproduce or attest observed machine
timings. [The checker contract update](CHECKER_CONTRACT_UPDATE_2026-11-03.md)
separately documents stricter malformed-input refusals and the distinction
between a threshold lower bound and a primal optimality claim. These additions
are a new reviewable revision, outside the historical signed release. They do
not establish logical-error improvement, physical model adequacy, real-time
deployment, or outside human reproduction.

The November 4 circuit-distance replay verified all 304 manifested receipts:
62 global finite-model bounds and 242 prefix irreducible exclusions. This was
a project-run replay, not outside human reproduction. The checkout contains
274 proof blobs; the other 30 were recovered into a separate ZIP that is not
included in Git or the historical release. See
[proof availability and replay](DISTANCE_PROOF_REPLAY.md) for the commands,
required archive, current access limitation and statement scopes.

## Current checker input contracts

The follow-up [contract synchronization](CHECKER_CONTRACT_SYNC_2026-11-03.md)
validates zero-multiplier facets, aligns the standalone readers, and uses a
current equivalence baseline with no historical transition allowances. The
previous public checker sources and historical evidence remain preserved.
This is an intentional API tightening: epsilon certificate IDs must be Python
integers, bound-only qLDPC results use the explicit `LOWER_BOUND_PROVEN` payload,
and JSON qLDPC integer fields require integer tokens. Use Node 24 for the
JavaScript readers. The linked guide identifies the migration and focused
commands; it makes no claim of full backward behavioral compatibility.

## Historical release

This release accompanies three manuscripts by Josh Philbrick (Coherence Energy
Labs), whose LaTeX sources are in `paper/`:

* *Certifying quantum error-correction decoders: independently checkable
  near-optimality receipts at scale* — `paper/certifying-decoders/`
* *Exact, independently checkable optimality receipts for BP-OSD on the
  [[144,12,12]] bivariate-bicycle code* — `paper/qldpc-certificates/`
* *Machine-checkable lower-bound certificates for the circuit-level distances
  of bivariate-bicycle syndrome-extraction circuits* — `paper/circuit-distance/`,
  with its lane in `experiments/circuit_distance_exact/`, its records and
  certificates in `evidence/circuit_distance_exact/`, and the Lean development
  of its checker's calculus in `lean/CertR198/`

Every number in all three papers is a macro generated from the evidence in this
tree. Regenerate them and rebuild the PDFs (needs Python and a TeX
distribution with REVTeX 4.2; paper 3's generator also needs Stim and NumPy):

    python paper/tools/make_numbers.py
    python paper/tools/make_numbers_qldpc.py
    python paper/tools/make_numbers_distance.py
    python paper/tools/make_figures.py
    python paper/tools/build.py certifying-decoders
    python paper/tools/build.py qldpc-certificates
    python paper/tools/build.py circuit-distance

Paper 3's certificates re-check with its own checker (a C compiler is needed):
`python -m pytest tests/test_r198_ibm_recert.py tests/test_r198_paper3_review.py`
re-verifies shipped proofs and refuses a corrupted one. Every certificate of the
three-colour and large morphing circuits was reported as regenerated from the
committed prover (`experiments/circuit_distance_exact/recert_campaign.py`) under
the hash in `evidence/circuit_distance_exact/recert_20261102/manifest.json`,
which then pins the proof the checker decides. Where that proof differs from the
original record, the prover of the record's day, compiled from git, reproduced
the recorded proof (`recert_old_prover.py`, `records/recert_old_prover.jsonl`).
These historical generation records remain unchanged. The November 4 review
update linked above separately records the completed 304-receipt replay;
the historical release still omits the 30 recovered payloads.

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

59 vectors must agree. **51 of them are refusals** — documents that a sound
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
| `gap_distribution_exact_rule.json` | historical exact-reference rerun, retained unchanged | its graph-cost comparisons share the duplicate-component scoring error and are superseded by `corrected_gap_20261103/` |
| `cert_conformance.json` | historical conformance run; the current vector set independently replays 59/59 | agreement on the exercised cases does not establish correctness or rule out a shared misunderstanding |
| `decoder_mutation_gate_d*.json` | syndrome consistency catches 0 of the syndrome-consistent fault class; the certificate catches all of it | output mutation models wrong answers, not crashes, hangs or memory errors |
| `epsilon_rerun/<config>/summary.json` | the scale runs repeated under the epsilon checker (exact rationals, eps_max = 1e-6) with every receipt kept: proven-within-epsilon, exact and not-proven counts, largest proven slack, forgeries | the receipts are a separate archive; `replay_<config>.json` records a complete two-verifier replay of them |
| `detection_outcomes/d*.json` | seeded faults and a stale calibration judged against the exact optimum: outcome states, false-alarm base rates, discordant logical cells | the stale-calibration result is objective mismatch against a supplied reference model; it cannot say which model describes a device |
| `mutation_score/*.json` | operator mutation scores of both checkers' tests, every surviving mutant declared equivalent with its reason | a score measures the tests against mechanical defects, not the absence of defects |
| `willow_residual_shots.json` | the two Willow shots generation 6 left open: PyMatching's correction is exactly optimal and the exact tier stopped at its round cap | two shots, not a rate |
| `gap_distribution.json` | historical tolerance-rule comparison, retained unchanged | its graph-cost comparisons are superseded; the implemented BP-OSD input treats graphlike components independently, and a graph-weight gap is not a logical error |
| `corrected_gap_20261103/summary.json` | exact XOR graph costs, a TRAIN-selected primary BP-OSD arm, common held-out draws and complete retained denominators | exact-optimum gaps use only 877 checked exact references; the other 170 nontrivial references provide lower-bound comparisons only |
| `tools/native_matching_duals/benchmark_final.json` | 256 checked integer-graph native witnesses, separate original-IEEE bounds, and 12 nontrivial same-shot timing comparisons per cell | integer exactness is not original-weight exactness; small local timings do not establish a general speed or live-control guarantee |
| `seeded_fault_corpus_d*.json` | the faults themselves | released so you can score *your* detector; the two numbers are only comparable if the corpus is shared |
| `lean_safe_bound.json` | a checker PROVED sound in Lean 4 (`Cert.accept_sound`, standard axioms only) accepted all 1,851 P2 exact certificates in `p2_certificates.jsonl.gz` and refused all 11,106 forgeries of them | the JSON reader and the Lean compiler are trusted, not proved; see `lean/OneqCert/README.md` |

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
