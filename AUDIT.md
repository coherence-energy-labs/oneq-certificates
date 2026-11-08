# What to audit, and where to attack it

`REPRODUCE.md` tells you how to re-run this. This file tells you where it is most likely to be wrong, which is the
more useful document. It covers the three manuscripts of this bundle (papers 1-3); paper 4 has its own bundle in
`superactivation/`, with its own `AUDIT.md`.

**Commit:** see `git rev-parse HEAD`; this file no longer pins one, because a hardcoded hash here went stale within a day (external audit 2026-08-04, finding 8)
**Archive sha256:** regenerated on every build — read it from `release/oneq-certifying-decoder.tar.gz.sig.json`, never from this file (`release/oneq-certifying-decoder.tar.gz`, signature beside it)

---

## The claim, in one sentence

A decoder's correction, or a lower bound on a circuit's distance, can come with a **certificate** that a checker
sharing no code with the producer verifies in exact arithmetic, so "trust the solver" is replaced by "check the
receipt".

Everything else in this bundle is either evidence for that, a gate defending it, or a record of where it stops.

---

## 15 minutes: the shortest path to a verdict

**From the signed tarball alone** — no repository, no network:

```
sha256sum -c SHA256SUMS                                    # 2526 files
python -m pip install -r requirements.txt
node verifier-js/verify-certificate.mjs      --conformance conformance/cert_vectors.json           # 59/59, 51 refusals
python tools/external/exact_lp_certificate_reference.py --self-test
python tools/recheck_all.py                                # paper 2: complete pinned replay, 2,042 receipts
python tools/replay_qldpc_scaling_campaign.py              # paper 2: the preregistered campaign, no producer
python tools/replay_epsilon_receipts.py --jobs 8           # paper 1: needs the receipt archives (release assets)
```

`python tools/verify_release.py --bundle certifying-decoder` checks the signature, the statement and the timestamps
once the release files sit in `release/`. Paper 3's large proofs are release assets too:
`python experiments/circuit_distance_exact/recert_assets.py verify --archives DIR --extract` checks them, and
`... recert_assets.py check` runs the committed checker on them (hours). `REPRODUCE.md` gives every command.

`REPRODUCE.md` §3 also describes the whole wall (every gate, the full test suite, the mutation gate). It needs the
project's development repository, which is private; none of it is needed for a verdict. The first block is the one
that matters: it needs nothing from us.

If any of those fails on a clean copy, stop and tell us: that failure is worth more than anything below. It has
happened. Auditors who cloned the public repository and followed its instructions have found commands that worked
only on the author's machine, and each fix is recorded in the release notes of the release that made it.

---

## What you actually have to read

The trust argument rests on the checkers, not on the producers. These are the files whose correctness the papers'
claims depend on:

| file | what it decides |
|---|---|
| `src/oneq/matching_cert_epsilon.py` | paper 1: the ε-checker (`check_epsilon`), the decision behind every rate |
| `tools/replay_epsilon_receipts.py` (`verify_b`) | paper 1: the second, separately written ε-verifier every receipt is replayed under |
| `verifier-js/verify-epsilon.mjs` | paper 1: a third ε-checker in JavaScript, written by a separate agent from the paper and a logged brief |
| `src/oneq/matching_cert.py` | paper 1: the exact checker, which repairs and re-verifies a dual |
| `src/oneq/qldpc_check.py` | paper 2: the flat, exact and branch-dual tiers |
| `tools/external/exact_lp_certificate_reference.py` | paper 2: the separately written flat checker, from the written specification |
| `tools/external/tree_checker_b.py` | paper 2: the separately written tree checker, from the schema |
| `experiments/circuit_distance_exact/ccheck.c` | paper 3: the certificate checker, whose calculus is proven sound in `lean/CertR198/` |

"Separately written" is the claim, not "independent": the authors of these files are one project, and paper 2 says
so. **The property to test while reading:** a checker imports no producer and re-derives everything from its inputs
and the certificate. In the development repository a gate enforces the import direction by module graph; from this
bundle, `tests/test_replay_qldpc_scaling_campaign.py` checks it for paper 2's campaign replay in a fresh
interpreter. If you find a path where a checker consults a producer's opinion, the claim is broken.

---

## Where to attack it

**1. Forge a certificate.** The forgery batteries are `src/oneq/forgeries.py` and the battery inside
`tools/replay_epsilon_receipts.py`. `evidence/derived_claims.json` counts 7,951,062 forgeries attempted against the
matching and qLDPC lanes and 9,629,829 against paper 1's ε-receipts, none accepted. Most forgeries are refused before
any bound is computed, as malformed or invalid input; the ones that test weak duality are those that reach the bound,
and the strict replay's records count them (`evidence/epsilon_rerun/replay_2026-11-03/`: 59,224 forgeries, 10,034
refused by the bound, none accepted). The soundness argument is weak duality: any feasible dual is a lower bound.
Attack the *checker's* implementation of it: sign conventions, the T-oddness test (the battery's parity forgeries are
accepted by parity-blind copies of both verifiers and refused by the real ones), and above all **the dual repair** in
`matching_cert._repair_and_verify`, which mutates the shipped dual before deciding. If you can make its shed and
absorb steps construct a feasible attaining dual for a correction that is NOT optimal, the exact lane is broken. (LP
duality says such a dual exists only when the correction is optimal, and step 3 re-verifies from scratch, but that
is exactly the kind of confidence worth attacking.)

**2. Make the equivalence gates vacuous.** `tools/frozen/` holds byte-identical pre-refactor snapshots;
`tools/checker_equivalence_gate.py` and `tools/producer_equivalence_gate.py` compare live code against them, and
`--prove-sensitive` mutates the live checker and requires the gate to catch it. **If you can make a behavioural
change that neither gate sees, the refactor guarantee is theatre.** One such hole was found this way and closed (a
mutant that had drifted onto weaker coverage while still "passing").

**3. Break a negative control.** Every replay and attack carries one that must succeed: the parity-blind copies must
accept the parity forgeries, a correction with one edge dropped must be told apart from the certified one, a swapped
input file must be refused by paper 2's replay for its own reason. A control that cannot fail would make its
check dark; find one.

**4. Paper 2's receipt corpus is complete, and replay fails if it is not.** `recheck_all.py` replays every one of the
2,042 receipts named by the pinned `evidence/receipt_manifest.json` (2,002 flat certificates and 40 branch-dual
trees: 17 held-out, 23 development) through two separately written checkers, against persisted instances and integer
weight vectors, and fails on any missing, duplicated, unlisted or altered receipt or bound file (external audit
2026-08-04, finding 7, and 2026-09-14, R-01/R-04: earlier releases re-checked 36 embedded trees and exited cleanly
when files were absent).

**5. The preregistrations.** `docs/HELD_OUT_PREDECLARATION.md`, `docs/PREREG_heldout_willow_2026-11-02.md`,
`docs/PREREG_gap_distribution_2026-11-03.md`, `docs/PREREG_heldout_d7_2026-11-03.md`,
`docs/PREREG_qldpc_scaling_2026-11-03.md` and `docs/PREREG_measured_model_program_2026-09-16.md`, each with its
deviations file where there is one. They were committed and pushed to the project's development repository before
their runs. That repository is private and the public history is squashed, so **a reader cannot check those
timestamps, and git dates are set by the author anyway**: the claim rests on our word until a preregistration is
anchored externally before its run, which none of these was. The held-out seeds derive from the freeze commit's sha,
which is self-referential rather than an external beacon (finding F4 in `docs/HELD_OUT_RESULTS.md`). This is the
weakest link of every preregistered result, and we are not going to pretend otherwise.

---

## READ THIS BEFORE QUOTING ANY CERTIFIED RATE

Paper 1's rates are acceptances of `src/oneq/matching_cert_epsilon.py`, which decides a DECLARED claim: it scales the
packing to feasibility, computes the correction's proven slack exactly in rational arithmetic, and accepts iff that
slack is at most eps_max = 1e-6, returning EXACT_OPTIMAL, EPSILON_OPTIMAL, NOT_PROVEN, INVALID_INPUT or
VERIFIER_FAILURE. The external audit of 2026-09-14 showed that the original runs' tolerance rule (float gap <= 1e-6,
no NaN guard) is not a proof; those runs were repeated under the ε-checker with every receipt kept
(`evidence/epsilon_rerun/`), and their original rates are historical. The exact checker described next is a
separate, stricter measurement.

The external audit of 2026-08-04 found that the matching checker accepted on a 1e-6 tolerance, which certified
provably suboptimal corrections. That is fixed: **`matching_cert.check` accepts only exact optimality, with no epsilon
anywhere in the decision**, exactly with respect to the original IEEE double weights. What is damaged is the shipped
DUAL, a float rounding of an exact one, and rounding damage is repairable: the checker sheds any infeasibility
(reducing a dual lowers the load on every edge of its cut and raises it nowhere), absorbs the shortfall into variables
with slack across their whole cuts, then re-verifies feasibility, non-negativity and attainment from scratch in exact
rational arithmetic.

**Why this cannot be a tolerance in disguise, which is the thing to attack.** By LP duality a feasible dual attaining
the correction's weight exists **if and only if** the correction is optimal. On a suboptimal correction the absorb
step therefore cannot find the slack: forbidden by the theorem, not merely unlikely. The 2026-08-04 auditor's own
witness (edges (0,1) = 1.0000005, 0-B = 1-B = 0.5, true optimum 1.0) is refused at scales 1e-9, 1 and 1e9
(`tests/test_exact_dual_repair.py`; at scale 1 also `tests/test_audit_2026_08_04.py`), three points rather than every
scale, which is what an exact rule should make unnecessary and an epsilon rule could not pass.

**The direction of the remaining failure is the point.** The exact checker also refuses some corrections that are
exactly optimal, when its repair search does not find an attaining dual; paper 1's Sec. IV and Table I give how often.
A false refusal costs coverage. A false acceptance would refute the programme. Every defect found so far pushed the
checker toward refusing, and no forgery has ever been accepted. If you find the reverse, anything accepted that
should not be, that is the finding that matters and we want it immediately.

---

## What we already know is not done

Believing any of this depends on knowing what it excludes.

- **No independent execution, and no human specialist.** Everything ran on one workstation. The reviews so far are
  AI agents that did not write the work under review and four external written audits, the last two of which state
  that they were AI-assisted (`docs/reviews/`). No human expert has re-run the evidence or read paper 3's two
  every-round hand proofs. Your run is the first that changes that.
- **A certificate is about its declared model.** Whether a declared error model matches a device is a separate,
  empirical question; paper 1's Sec. VI E measures one route (models estimated from the device's own syndromes).
- **The preregistration timestamps are private** (attack 5 above).
- Prior art is adjacent and real: Wu et al. (arXiv:2508.04969) give a decoder that emits a proximity bound with every
  solution and certifies optimality when their certifiability condition holds. Each paper's related-work section says
  what it adds and what it does not.

---

## What we are asking you to conclude

Only what you verify. Every claim of papers 1-3 is indexed by artifact and command in `docs/REVIEW_PACKET.md`. **If
a number in a paper does not fall out of a command there or in `REPRODUCE.md`, that is a defect and we want it
named**: the provenance gate is supposed to make it impossible, and it has failed before.

A finding that kills a claim is worth more to us than a green run.
