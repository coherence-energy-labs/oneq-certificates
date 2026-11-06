# Referee report, third round, 2026-11-02: paper 1, *Certifying quantum error-correction decoders*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: post after fixes.** Nothing I found makes a headline number false. Two things block posting:

- **The release does not exist yet.** The PDF prints "release UNSET".
- **The new defect disclosure understates the defect.** The same missing parity test was in the checker that decided every number from the original runs, not only in "the exact checkers".

## FATAL (blocking, mechanical)

**F1. The paper cites a release that has not been made.**
- **Where:** `paper/release.tex` holds `\ReleaseTag{UNSET}`. `main.tex:267` typesets it and says every receipt "is attached to that release".
- **Witness:** `analyze.py B` printed `release.tex ReleaseTag UNSET: True`, and the PDF text reads "`oneq-certificates (release UNSET). Every receipt of Sec. VI…`".
- `build.py --release` refuses in this state (`build.py:138`).
- `git branch -r --contains a70c2877` returns nothing, so the latest commit is on no remote.
- **Fix:** make the release with every receipt archive, including the held-out ones and `fresh_sim_d5_r25`. Set the tag, rebuild with `--release`, and check the asset SHA-256s against `chunks.jsonl` from a clean download.

## MAJOR

**M1. The parity hole is described as a defect of "the exact checkers". It was in the production checker from its first version, including the tolerance-rule checker behind every original-run number.**
- **Where:**
  - `main.tex:113`: the tolerance-rule paragraph lists only the tolerance and the missing NaN guard.
  - `main.tex:127`: "The exact checkers of the first public release…" and "every rate of Sec. results is an acceptance of the ε-checker".
  - `main.tex:186`: the IBM run.
- **Witness:** `analyze.py A` extracts `src/oneq` at each commit and runs PB1, PB2 and PE1. I brute-forced the optima: PB1 weighs 6 against an optimum of 4, PB2 7 against 6, PE1 5 against 4.
  - `matching_cert.check` **accepts all three**, answering "OPTIMAL: feasible dual attains the primal weight…", at every commit the audit pins as a producer: cce1e91 (07-31, gap distribution), d4a81ee (08-01, million-shot run), 06344ae (08-01, Willow generation 4), f57e87d (08-03, Willow generation 6).
  - It also accepts all three at 10e2a75 (08-04, the last tolerance-rule version).
  - The fixed versions refuse all three, naming the set as T-even.
- **Consequences:**
  - (a) "Every rate of Sec. results is an acceptance of the ε-checker" is false. That section reports 997,594 of 10⁶, Table II, Fig. 2, the generation 5 and 6 rates, 198,770 of 199,384 and IBM's 2,296 of 2,296, all accepted by this checker.
  - (b) Line 113 implies the tolerance rule admits only "a correspondingly small suboptimality". In fact it accepted a correction 25% heavier than the optimum.
  - (c) The IBM claim rests on a checker that accepts forged optimality proofs. Its artifact was last committed 2026-08-10, before the fix; it is not in the audit's six configurations, and its certificates were not kept.
  - (d) The only evidence that tolerance-rule acceptances were admissible is the audit's 0 inadmissible of 1,588. That bounds the inadmissible share at ≤0.23% (two-sided 95% Clopper–Pearson interval). This is about the same size as the differences the paper discusses, for example 99.76% against 99.81%.
- **Fix:**
  - Add the missing parity test as the third failing of the tolerance rule at line 113.
  - At line 127, say that the production checker, in every version before 2026-10-18 including the tolerance-rule checker of the original runs, never tested T′-oddness.
  - Change "every rate of Sec. results" to "every headline rate".
  - State the audit bound wherever tolerance-rule rates are used.
  - Rerun IBM under `check_epsilon` (the records are small and preregistered), or label it as accepted by a checker later found unsound.

## MINOR

- **m1. Notation (`main.tex:58` against `:127`).** Section 2 already defines T to include the boundary when |T| is odd. In that notation T′ = T, so "tested against T instead of T′" contradicts it. Introduce a name such as T₀ for the fired set.
- **m2. "First public release" (`:127`) and "first release" (`:90`) are undefined.**
  - v1.0.0 (07-29) contains no `matching_cert*`.
  - The strict checker was created 08-17 (8aa2bcac), after the release bundle's first commit (9bc160cf, 08-04).
  - The paper's own release is UNSET.
  - The fix landed in 617da3af, whose message is "WIP: lane files in progress".
  - **Fix:** give dates and cite the commit.
- **m3. The defect sentence at `:127` is incomplete, and one phrase overclaims.**
  - The production checker accepted all three forgeries, including PB2 (7 against 6); the strict checker accepted PB1 and PB2.
  - "The three forged certificates" refers to a third one never described.
  - "Disagreed on exactly these inputs" claims more than is known; say "on these inputs".
- **m4. `:90` "It implements the exact-optimality checker of the first release"** is now contradicted by `:127`, which says the two disagreed. Say it implements that checker's specification.
- **m5. Per-round wording (`:171`).** The development data reject proportional accrual: the per-shot-round not-proven rate is 1.43 times higher at 30 rounds than at 13 (p = 0.036).
  - The held-out rate is *below* the 30-round rate: ratio 0.72, p = 0.012. Proportional accrual from the 30-round rate predicts 98.74%; the held-out sample gave 99.09%.
  - Say instead: "equal to the 13-round rate, below the 30-round rate; no sign of overfitting".
  - The abstract's 99.76% is the floor of 99.770% (19,952 of 19,998); the abstract does not say rates are rounded down.
- **m6. The fresh-seed artifact has no commit or source hashes.** `summary.json` has no `head` block, the same gap the last review's M2 closed for the held-out sample. The text also does not say the fresh shots used the current producer, while the 10⁶ rerun it is compared with used the September producer.
- **m7. `producer_identity_willow.py` claims to pin "every module" the producer and checker import, but pins 9 of 16.** The static import closure also contains `abft_l0`, `exact_decode`, `exact_lp`, `logical_cut`, `matching_cert_strict`, `repair_impossible` and `tjoin_exact`. Whether the 11 changed packings come from code or from library versions is not established.
- **m8. "Corrupting one dual value drops the agreement to 58" is arithmetic, not a measurement.** It is `total - 1` (`make_numbers.py:698`). The matching test (`tests/test_cert_conformance.py:73-81`) can skip.
- **m9. The forgery battery still has no T-even-set class.** The fresh run's classes are E, F, I, D, G, H, NaN and +inf. The last review asked for one.
- **m10. Carried over:** Fig. 1 still does not say which checker it times, and the title still says "optimality" where the claim is ε-optimality.

## Earlier findings: fixed or reworded

**Round 1**

| Finding | Status |
|---|---|
| 1 (cost figure) | Fixed |
| 2 (detection cannot fail) | Fixed |
| 3 (99.80% mixes two producers) | Fixed |
| 4 (rates on development shots) | Fixed: held-out sample |
| 7 (related work) | Fixed |
| 8 (checking cost) | Fixed |
| 9 (measured-model semantics) | Fixed |
| 10 (no uncertainty) | Fixed |
| 11 ("in the spirit of") | Fixed |
| 12 (Goemans–Williamson attribution) | Fixed |
| 13 (forgery count in abstract) | Fixed |
| 15 (parallel edges) | Fixed in text |
| 5 (receipts not released) | **Not fixed**: F1 |
| 6 (second implementation) | Reworded only; no per-verifier tallies |
| 14 (non-proof results presented) | Labelled, not rerun, and now known to rest on a checker with the parity hole (M1) |
| 16 (naming, title) | "today's" is gone; the title is unchanged |
| 17 (`SimDetectors`/`SimEdges` source) | Not checked |

**Round 2**

| Finding | Status |
|---|---|
| M1 (exact-checker holes) | **Partly fixed.** The re-judged artifacts changed only in their `seconds` fields; heads at 624d41f2 and 0052b997 with clean `src/oneq`; pins enforced. But the disclosure is mis-scoped (my M1) and there is no forgery class (m9) |
| M2 (same producer) | **Genuinely fixed** by measurement |
| M3 (abstract comparison) | Fixed; see m5 for the per-round wording |
| M4 (median against mean) | Fixed |
| M5 ("preregistered") | Reworded honestly; the commit is still local only |
| m1, m6, m7, m9, m10 | Fixed |
| m2 (cost section) | Fixed |
| m3 (no parity forgery class) | Not fixed |
| m4, m5 (tallies, `int()` coercion) | Deferred to OPEN_WORK |
| m8 (title, "today's") | Half fixed: wording yes, title no |

## Checks that found no problem

- **Abstract:** 1,905 of 1,920 characters, using `build.py`'s own de-TeX. Every number in it is a macro, and distance 5 is compared with distance 5.
- **Fix diff (617da3af):** matches the description. Brute-force optima confirm each forgery's weight and optimum.
- **Older checkers:**
  - `check_epsilon` has had the T′ test since its first commit (8a222cc4, line 139), and `tests/test_r229_paper_reviews.py:175` runs it on the three forgeries.
  - The JavaScript checker's first commit (7a5a3cff) builds T′ and tests sets against it. I checked this by reading the code; I did not run it.
- **Release copies:** the release bundle's `matching_cert.py`, `matching_cert_strict.py` and `tools/frozen` copy are fixed. The release's JavaScript checker is identical to `verifier-js`.
- **Producer identity:**
  - 80/80 held-out chunks and 310/320 development chunks are byte-identical.
  - 11 shots differ; 1 changed verdict (X r30 chunk 10, shot 87: ε from 1.2e-13 to 0.205).
  - All 9 pins equal the git blobs at 624d41f2.
  - Receipts went to an empty scratch directory, and there is no disk cache.
  - The test can fail, and it did.
- **Restated development rates:** 99.99, 99.99, 99.76, 99.24 and the per-round figures 1.8, 2.5, 1.8 all reproduce.
- **Fresh-seed simulation:**
  - 9,981 of 10,000 proven; Clopper–Pearson interval 99.703–99.886, typeset rounded outward.
  - Same graph as the main simulation (`edges_sha256` equal).
  - Replay complete on 10,000 receipts; 80,000 forgeries, none accepted.
  - No other recorded seed is within 4.4×10⁹ of the fresh range.
  - Consistent with the million-shot rate (Fisher exact test, p = 0.91).
- **Replay boundary battery:** in every replay, the number of attempts equals the number refused.
- **Detection oracle:** it is re-verified inside the detection script, not by the defective checkers.

## Could not check

- Whether the GitHub release exists or carries the receipts.
- The JavaScript checker's behaviour by running it.
- Whether the 11 changed packings come from code or from library and environment changes.
- Whether the full tolerance-rule runs ever accepted a T-even set. Their certificates were not kept.
- The linear-programming search script that found the forgeries.
- Round 1's finding 17.

Read-only was respected, so I wrote no agent memory. Lines I would have saved:
- WORKED on "defect scoped to component Y": run the published witnesses at every pinned producing commit.
- INSTRUMENT: `ConfVectorsMinusOne` is `total - 1`.
- INSTRUMENT: the identity record pins 9 of 16 modules.
- FAILED to kill: the producer byte-identity test, the fresh-seed interval and seed claim, and the abstract length.

```
STRONGEST SURVIVING RESULT  check_epsilon is sound; the headline rates and the 80/80 same-producer held-out comparison hold
EVIDENCE CLASS              PROVEN (Lemma 1) + MEASURED (identity rerun, replays)
HOW IT COULD STILL BE WRONG tolerance-rule rates and IBM rest on a checker that accepts forged proofs; only a 1,588-certificate sample bounds that
WHAT IT LICENSES            posting once the release exists and the disclosure is re-scoped
KILLED                      "every rate of Sec. results is an ε-checker acceptance" -> cce1e91 accepts PE1 (5 vs 4) -> "every headline rate"
OPEN BRANCHES               rerun IBM under check_epsilon; rerun 0314438a on the 10 chunks to attribute the drift
RECOMMENDATION              SHIP after F1 + M1
```

Files are in <reviewer's local scratch path>
- analyze.py (run with argument A, B, C or E)
- child_forgery.py
