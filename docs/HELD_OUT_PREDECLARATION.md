# Frozen held-out confirmatory campaign — predeclaration (DRAFT)

**Status: FROZEN at the commit tagged `held-out-freeze`. No code,
weight, circuit, or analysis change is permitted until the results
section is complete; any change voids the campaign and requires a
fresh freeze.**

The audit's requirement, verbatim: freeze the producer, checker, BP-OSD
settings, circuits, weight vectors, receipt schema, and claims script;
publish their hashes BEFORE examining new outcomes; run fresh unseen
seeds; generate all tables automatically from per-shot records; make no
changes after opening the held-out data.

## Freeze set (hashes to be pinned at freeze time)

| item | file(s) | sha256 |
|---|---|---|
| producers | src/oneq/qldpc_cert.py | 825804bb86d5248a5be68213620c1888b9e44d6ec5c62a4026537bafab37b2c9 |
| checkers | src/oneq/qldpc_check.py | eba3f2d6280a30613323ca8bb1fd66ae81670a5ddfcf76c9b560b395ac5e8f34 |
| reference checker (pinned) | tools/external/exact_lp_certificate_reference.py | f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd |
| campaign runner | experiments/m0_prior_art/qldpc_heldout_gate.py (+ run_rung in qldpc_exact_all_gate.py: 0666dc982a6db17c…) | 60559ce0abd44686b7c4b8dc848508bbc4f50f7fbb2bff60d77a150363ed0391 |
| BP-OSD config | in-runner constants (min-sum, parallel, 30 iters code-capacity / 40 circuit, OSD-CS 5) | covered by runner hash |
| circuits | reference-schedule builder (qldpc_bb_refsched_gate.py) | 589a9ed05a6304a6ef8315fc1eb21ce5875a8b0e55c97468c945787f3359a870 |
| weight rule | floor(1000·llr + 1/2); uniform w=1 | covered by producer hash |
| environment | evidence/environment.json (numpy 2.3.5, scipy 1.17.0, ldpc 2.4.1, stim 1.16.0) | 484386376bc1ebc563dd018f471bfb1a89cd3a7b1810ba21d01fc51e46fa9064 |

## Predeclared protocol

- **Seeds**: drawn AFTER freeze from a self-referential commitment
  (see F4 in the disclosures below):
  seed_rung = SHA256("oneq-heldout-2026" || rung_name || freeze_commit
  sha) truncated to 63 bits. No other seed source is permitted.
- **Rungs and sample sizes** (same protocols as the canonical
  datasets, fresh seeds): uniform p=0.01 (300 nontrivial), uniform
  p=0.02 (470), heterogeneous p=0.02 fixed-point (300; fresh channel
  from the same seed rule), phenomenological r=4 p=pm=0.01 and 0.02
  (150 each), X-sector p=0.02 (200), stress p=0.05 (300),
  reference-schedule circuit r=3 p=0.002 (150).
- **Pipeline per shot**: tier 0 / tier 1 / tier 2 flat exact / tier BD
  branch-dual closure; every exact certificate through the pinned
  reference checker; per-shot records + MANIFEST as in the canonical
  runs; solver used only as producer heuristic.

## Predeclared endpoints and table structure

Primary endpoint per rung: exact-certified count / nontrivial count
(conditional), with tier decomposition (t0/t1/t2/BD) and all-shot
totals, exact Clopper–Pearson bounds floored. Stress adds: the paired
2×2 vs BP-OSD from per-shot records with exact two-sided McNemar.
Secondary: component timings (median/p95/max), certificate sizes,
independent-checker agreement count, branch-tree node counts.

Predeclared success criteria (stated before data): each rung's
held-out exact-certified conditional rate is reported AS OBSERVED; no
re-runs, no seed changes, no exclusions. Discordance with the canonical
rates is a finding, not a defect to repair. The held-out rates become
the papers' headline rates; canonical rates become development data.

## Timing

Freeze happens only after: gen-6b landed (DONE), the full mutation gate
green with zero skips (DONE: 53/53), both papers' totals stable (DONE),
and the runner script committed and reviewed. The waived long-function
refactors are DEFERRED to after this campaign, deliberately: the frozen
code should be the code that survived adversarial testing (~28k hostile
trees, 53/53 mutants), not a fresh refactor of it; the waivers remain
visible debt in the standards gate. Owner may trigger the freeze
by asking for it; the run itself is ~30–60 minutes of compute.

---

*Results and post-campaign disclosures live in a separate record: `docs/HELD_OUT_RESULTS.md` — the preregistration file above is the immutable protocol document (audit round 6: prereg and results must be separate records).*
