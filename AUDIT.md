# What to audit, and where to attack it

`REPRODUCE.md` tells you how to re-run this. This file tells you where
it is most likely to be wrong, which is the more useful document.

**Commit:** see `git rev-parse HEAD`; this file no longer pins one, because a hardcoded hash here went stale within a day (external audit 2026-08-04, finding 8)
**Archive sha256:** regenerated on every build — read it from `release/oneq-certifying-decoder.tar.gz.sig.json`, never from this file (`release/oneq-certifying-decoder.tar.gz`, signature beside it)

---

## The claim, in one sentence

A decoder can emit a **per-shot certificate that its correction is
minimum-weight**, and a checker sharing no code with any solver can
verify that certificate in exact arithmetic — so "trust the decoder" is
replaced by "check the receipt".

Everything else in this repository is either evidence for that, a gate
defending it, or an honest record of where it stops.

---

## 15 minutes: the shortest path to a verdict

**From the signed tarball alone** — no repository, no network:

```
sha256sum -c SHA256SUMS                                    # 374 files
node verifier-js/verify-certificate.mjs      --conformance conformance/cert_vectors.json           # 56/56, 48 refusals
python tools/external/exact_lp_certificate_reference.py --self-test
python tools/recheck_all.py                                # complete pinned replay: 2,042 receipts
python tools/replay_epsilon_receipts.py --jobs 8           # paper 1's rerun receipts (needs the archives)
```

**With the full development repository**, `REPRODUCE.md` §3 adds the wall:
every gate, the whole test suite and the mutation gate, with counts that
`tools/doc_claims_gate.py` holds to the artifacts. None of it is needed for a
verdict. The first block is the one that matters: it needs nothing from us.

If any of those fails on a clean clone, stop and tell us — that failure
is worth more than anything below. It has happened: the last audit pass
was done by cloning this repository and following its own instructions,
and **five defects** turned up that made the package work only on the
author's machine (see `git log 9851c28 b36db59`).

---

## What you actually have to read: 1,725 lines

The entire trust argument rests on code you can read in an afternoon.
Everything else can be wrong without the central claim being wrong.

| file | lines | why it matters |
|---|---|---|
| `src/oneq/qldpc_check.py` | 685 | the qLDPC checker: flat, exact, and branch-dual tiers |
| `src/oneq/matching_cert.py` | 441 | the MWPM checker |
| `tools/external/exact_lp_certificate_reference.py` | 418 | independent checker, written from the written spec |
| `tools/external/tree_checker_b.py` | 181 | independent tree checker, written from the schema |

**The property to test while reading:** the checkers import no producer,
and re-derive everything from `(H, w, s)` and the certificate's own
listed support. `tools/standards_gate.py` enforces the import direction
by module graph. If you find any path where a checker consults a
producer's opinion, the claim is broken.

---

## Where to attack it

These are the places we think are weakest. Attacking elsewhere is
welcome, but start here.

**1. Forge a certificate.** `challenge/` contains the forgery
corpus and a self-testing verifier. 7,951,062 forged certificates have
been rejected and 0 accepted, so a single accepted forgery is a
publishable refutation of the whole programme. The soundness argument is
weak duality: any feasible dual is a lower bound. Attack the *checker's*
implementation of it — sign conventions, the facet-parity rule, and above
all **the dual repair** in `matching_cert._repair_and_verify`, which is
the newest and most dangerous code here: it MUTATES the shipped dual
before deciding. If you can make the shed/absorb steps construct a
feasible attaining dual for a primal that is NOT optimal, the matching
lane is broken. (We think you cannot — LP duality says such a dual exists
only when the primal is optimal, and step 3 re-verifies from scratch —
but that is exactly the kind of confidence worth attacking.)

**2. Make the equivalence gates vacuous.** `tools/frozen/` holds
byte-identical pre-refactor snapshots; two gates compare live code
against them. Both have `--prove-sensitive`, which mutates the live
checker and requires the gate to catch it. **If you can make a
behavioural change that neither gate sees, the refactor guarantee is
theatre.** One such hole was found this way and closed (a mutant that
had drifted onto weaker coverage while still "passing").

**3. Break a negative control.** Every headline result carries one, and
they are the load-bearing part:
- A6 symmetry decoding: at `q=0.5` the side-channel bit is pure noise
  and the result is **13.5× worse** than baseline. If that column can be
  made to look good, the claim is an artifact of the extra constraint
  rather than of information.
- C2 mask flips: L2′ is 0% on *weight* faults by design. If it fires
  there, it is a second copy of L2, not a complementary layer.

**4. The paper-2 receipt corpus is complete, and replay fails if it is not.**
`recheck_all.py` replays every one of the 2,042 receipts named by the pinned
`evidence/receipt_manifest.json` (2,002 flat certificates and 40 branch-dual
trees: 17 held-out, 23 development) through two independently written
checkers, against persisted instances and integer weight vectors, and fails on
any missing, duplicated, unlisted or altered receipt or bound file. External
audit 2026-08-04, finding 7, and 2026-09-14, R-01/R-04: earlier releases
re-checked 36 embedded trees and exited cleanly when files were absent. Tree
records elsewhere in the evidence that keep only tier and outcome cannot be
re-checked from this bundle and support no claim in either paper.

**5. Check the numbers we refused to produce.** Absence is a claim too:
- C2's ~5,400 LUT figure is **unverified** — ABC will not complete in the
  WASM yosys build (three flows tried). Run native yosys and tell us.
- B-1's `ETA_MAX` formula is not stated in our spec, so we computed the
  ceiling under three definitions rather than fitting one to reproduce
  3.56. If a fourth defensible definition crosses the bar of 10, the
  ABSTAIN is not robust and we want to know.

**6. The preregistration.** `docs/HELD_OUT_PREDECLARATION.md` was
written before the results. Seeds derive from the freeze commit sha at
runtime. **The freeze commitment is self-referential** — our own sha, not
an external public beacon — and that is disclosed as finding F4 in
`docs/HELD_OUT_RESULTS.md`. It is the weakest link in the
preregistration and we are not going to pretend otherwise.

---

## READ THIS BEFORE QUOTING ANY CERTIFIED RATE

Paper 1's rates are acceptances of `src/oneq/matching_cert_epsilon.py`, which
decides a DECLARED claim: it scales the packing to feasibility, computes the
correction's proven slack exactly in rational arithmetic, and accepts iff that
slack is at most eps_max = 1e-6, returning EXACT_OPTIMAL, EPSILON_OPTIMAL,
NOT_PROVEN, INVALID_INPUT or VERIFIER_FAILURE. The external audit of 2026-09-14
showed that the original runs' tolerance rule (float gap <= 1e-6, no NaN guard)
is not a proof; those runs were repeated under the epsilon checker with every
receipt kept (`evidence/epsilon_rerun/`), and their original rates are
historical. The exact checker described next is a separate, stricter
measurement.

The external audit of 2026-08-04 found that the MWPM checker accepted on
a 1e-6 tolerance, which certified provably suboptimal corrections. That
is fixed. **`matching_cert.check` now accepts only exact optimality, with
no epsilon anywhere in the decision**, and it is exact with respect to the ORIGINAL IEEE double
weights — nothing is quantised and no fixed-point vector is declared.

**A previous version of this section said the opposite and was wrong.**
It claimed exact optimality was "unprovable in principle from float
weights, by anyone", because the dyadic lattice step over doubles
(~4.4e-16) is finer than the float error in the dual itself (~1e-15).
The lattice argument really is useless here. The conclusion drawn from
it was not: what is damaged is the shipped DUAL, which is a float
rounding of an exact one, and rounding damage is REPAIRABLE.

    surface-d3-5   2.22e-16 SHORT of attaining
    surface-d3-4   4.44e-16 OVER on edge (6,12) -- infeasible, which is why
                   its objective exceeded its own primal

So the checker now REPAIRS the dual and verifies the repair: shed any
infeasibility (reducing a dual lowers load on every edge of its cut and
raises it nowhere), absorb the shortfall into variables with slack across
their whole cuts, then re-verify feasibility, non-negativity and
attainment from scratch in exact rational arithmetic.

**Why this cannot be a tolerance in disguise, which is the thing to
attack.** By LP duality a feasible dual attaining U exists **if and only
if** U is optimal. On a suboptimal primal the absorb step therefore
provably cannot find the slack — forbidden by the theorem, not merely
unlikely. The auditor's own witness (edges (0,1)=1.0000005, 0-B=1-B=0.5,
true optimum 1.0) is refused at every scale from 1e-9 to 1e9. Attack this
first; the repair is the most dangerous code in the tree, because a
too-generous version stays green on every other test.

### The rates, re-measured under the exact rule (2026-08-09)

`evidence/corpus_exact/` -- every artifact carries its own sample size,
tier stack (`oneq_escalate`), and forgery count. There is deliberately NO
single aggregate: the table is the claim.

| config | certified | rate | n | tier |
|---|---|---|---|---|
| d3_X_r13 | 9838/9841 | 99.97% | 10000 | full |
| d3_X_r30 | 9989/9994 | 99.95% | 10000 | full |
| d3_Z_r13 | 9558/9560 | 99.98% | 10000 | full |
| d3_Z_r30 | 9986/9991 | 99.95% | 10000 | full |
| d5_X_r13 | 9447/10000 | 94.47% | 10000 | full |
| d5_X_r30 | 194/250 | 77.60% | 250* | full |
| d5_Z_r13 | 9420/9998 | 94.22% | 10000 | full |
| d5_Z_r30 | 276/375 | 73.60% | 375* | full |
| d7_X_r13 | 437/500 | 87.40% | 500 | full |
| d7_X_r30 | 406/5000 | 8.12% | 5000 | **BASE floor** |
| d7_Z_r13 | 422/500 | 84.40% | 500 | full |
| d7_Z_r30 | 76/1000 | 7.60% | 1000 | **BASE floor** |

Forgeries: **0 accepted of 412,812 attempted**, at every size and tier.
(*) surviving shots after memory-killed chunks; losses recorded in-band.

**Read the two BASE-floor rows carefully -- they are the exhibit, not an
embarrassment.** The full exact stack on d7xr30 graphs (~8,000 detectors)
exceeded this workstation's compute at every sample size tried (10k, 2k,
500, 150 -- all documented timeouts), so those rows show the BASE tier
alone: 7.8%. The repo's gen-5 evidence reached 98%+ there with the full
ladder -- the ladder is worth ~90 points at the hardest scale, and the
exact tier's measured cost wall is the quantified motivation for the
fixed-point bridge tier (quantized certification + certified robustness
transfer), designed and scheduled.

**Certify-or-repair, measured separately** (d5_X_r13, 1,000 shots): 94.8%
certified as-decoded + 3.6% CERTIFIED REPAIRS (the decoder's answer
provably suboptimal; the exactly-minimum correction shipped with proof,
both weights on the receipt) = **98.40% cert-or-repaired**. Repairs are a
separate column by construction -- blending them into "certified" would
credit the decoder with answers it did not produce.

**A known coverage gap, disclosed rather than found by you.** On
`d3_at_q4_5_X_r13`, 24 of 9,841 nontrivial shots (0.24%) are refused even
though the correction **is** exactly optimal. These are FALSE REFUSALS:
`tools/exact_optimum_oracle.py` computes the true minimum-weight T-join
in exact rational arithmetic — exact Dijkstra plus exhaustive pairing
enumeration, sharing no code with the checker — and finds primal ==
optimum. By LP duality an attaining dual therefore exists and the
checker's greedy repair search fails to find it. Four deterministic spend
orders are tried; the principled fix is exact re-optimisation over the
dual support, which is not done.

**The direction of that failure is the point.** A false refusal costs
COVERAGE. A false ACCEPTANCE would refute the programme. Every defect
found on 2026-08-05 pushed the checker toward refusing, and 0 of
7,951,062 forgeries have ever been accepted. If you find the reverse —
anything accepted that should not be — that is the finding that matters
and we want it immediately.

---

## What we already know is not done

Believing any of this depends on knowing what it excludes.

- **No independent execution.** Everything ran on one workstation. Your
  run is the first that changes that.
- **The hardware leg WAS at zero; it is now measured, and the model is wrong by family.** Whether the *declared* noise model matches real hardware was untested until the 2026-09 campaign (ledger R19-R29): on seven IBM Heron repetition-code runs across two devices and three distances the calibration-derived DEM overstates the space-like family 7-38x, its logical errors are 89.8% spurious flips, and a model estimated from the syndromes themselves cuts its held-out errors 10.1x (474 -> 47). Google's shipped SI1000 prior on the public Willow corpus does NOT show the per-family overstatement (0.31-0.37), so it is IBM's pipeline as read by this instrument (R29). The remaining hardware gap is the A/E/B/Y error-budget ledger, which has still never been pointed at device shots.
- **Benchmarks are recorded, not judged** — this machine measured 500%
  timing spread, so every performance number is unjudged by our own gate.
- **`ONE_Q_MAX`'s OS layer is design**, not code.
- Prior art is adjacent and real: Wu et al. (arXiv:2508.04969) publish a
  "Certifying Optimum" theorem for MWPM. What is new here is the qLDPC
  ladder, the exact-rational tier, and the checkers.

---

## What we are asking you to conclude

Only what you verify. Every paper claim is indexed by artifact in
`docs/PAPER2_QLDPC_DRAFT.md` §11 and `docs/PAPER_DRAFT.md` §9. **If a
number in a paper does not fall out of a command in `REPRODUCE.md`, that
is a defect and we want it named** — the provenance gate is supposed to
make it impossible, and it has failed before.

A finding that kills a claim is worth more to us than a green run.
