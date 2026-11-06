# Referee report, second round, 2026-11-02: paper 1, *Certifying quantum error-correction decoders*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: post after fixes.** Nothing I found makes a headline number false. Every headline rate (99.81% simulated, 99.60% / 99.998% Willow, 99.54% held-out) traces to its artifact, is decided by `check_epsilon`, and its receipts replay completely. `check_epsilon` survived my attacks. What fails is wording in the abstract, one comparison that was never controlled, and an undisclosed soundness history.

## FATAL
None.

## MAJOR

**M1. The paper never mentions the soundness holes found in the exact checkers on 2026-10-18, and several numbers were decided by the faulty versions.**
- **Where:** `main.tex` line 80 (step ii), line 90 (the JavaScript checker), line 125, line 191 (measured model), line 235 and the Fig. 3 caption (exact-rule gaps), Table I.
- **What happened:** `matching_cert.check` never tested that a dual set is T'-odd. `matching_cert_strict` tested |S∩T| instead of |S∩T'|. I read all 271 lines of the manuscript; neither hole nor the fix is mentioned. Step (ii) describes the T-odd test as a property of "the checker", which was false for the first release's exact checker.
- **Witness:** `git show 617da3af -- src/oneq/matching_cert.py src/oneq/matching_cert_strict.py`. The fix comments say "forgery PE1 certified a weight-5 correction whose optimum is 4" and "forgery PB1 certified a weight-6 correction whose optimum is 4".
- **Numbers decided before the fix** (last commit dates of their evidence):

  | Paper numbers | Evidence file | Date | Status |
  |---|---|---|---|
  | `CertMeas` (40,963 of 40,979) | `certified_measured_lane.json` | 09-16 | Certificates not kept, so they cannot be re-decided |
  | `GapExactRule*` and Fig. 3 | `gap_distribution_exact_rule.json` | 09-13 | Not re-checked |
  | `AuditExactSame` (842) | `certification_semantics_audit.json` | 09-13 | Being rerun now; so far only the `judge_seconds` / `export_seconds` timing fields differ |
  | `ConfVectors` | `cert_vectors.json` | 08-18 | 56 vectors, none for set parity |

- **Why the conformance agreement says nothing here:** the fix comment says the JavaScript verifier already used the correct rule. So the two implementations disagreed on exactly this case, and no vector ever exercised it.
- **Fix:**
  - Disclose the hole and the fix in the checker-semantics section (Sec. 4).
  - Rerun the measured-model and exact-rule gap experiments under the fixed checker and under `check_epsilon`, keeping the receipts. Both are small graphs, so this takes minutes.
  - Add PE1 and PB1 as conformance vectors and as a forgery class.

**M2. "The same base ladder" in the held-out comparison is not the same code, and the held-out artifacts do not record which code ran.**
- **Where:** line 169.
- **Code drift:** the development rates come from the 2026-09-14 rerun (`git log -1 -- evidence/epsilon_rerun/willow_d5_at_q8_7_X_r30/summary.json` gives 0314438a 2026-09-14). Since 09-15, five commits changed the producer modules (`certifying_decoder`, `moat_dual`, `branch_certificate`, `moat_growth`):
  - 83fb4bbb (10-18), whose message says "three defects ... fixed at the root";
  - 3ede8251, 127a192e, 490b70ec and 82718d82 (all 10-21), including "owned native graph kernel".
- **No code binding:** the held-out `summary.json`, `chunks.jsonl` and `archive.json` carry no commit and no source hashes, unlike `certify_cost_d5_r25.json`. Nothing in the artifacts shows the run used f9074d55, and the run took place in a shared working tree.
- **Partial mitigation:** the cost run found that the current producer reproduced the stored September verdicts on 2,000 simulated shots.
- **Fix:** either:
  - re-certify the development d3/d5 shots at 13 and 30 rounds (79,384 shots) with the frozen producer and quote those rates; or
  - re-certify a few held-out chunks from a clean checkout of f9074d55 and compare the chunk SHA-256s.

  In both cases, record commit and source hashes per chunk.

**M3. The abstract's held-out comparison is not like-for-like.**
- **Where:** line 29.
- **What is wrong:** the abstract sets 99.54% beside 99.60%. But the held-out sample has no distance 7 (excluded in advance) and is half distance 3 (100%). The 99.60% development figure includes distance 7 at 13 rounds (about 99.0%).
- **Witness** (`stats_heldout.py`):
  - The development rate on the same placements is 79,185 / 79,384 = 99.749%.
  - The held-out not-proven rate is 0.455% against 0.251% in development: 1.82 times higher (z = 4.79, p = 1.7e-6).
  - Per round, the not-proven rate is flat: 1.77e-4 at 13 rounds, 2.52e-4 at 30, 1.82e-4 at 50. The drop is explained by longer runs. That is good news, but the abstract hides it rather than stating it.
- **Fix:** in the abstract, put the distance-5 held-out rate (99.09%, 95% interval 98.88–99.27%) beside the development rates of 99.76% and 99.24%, and say that distance 7 was excluded. The body's wording "consistent with that trend" is fair; at distance 5, 50 rounds against 30 gives z = 1.41, p = 0.16.

**M4. The abstract pairs a median with a ratio computed from the mean.**
- **Where:** line 29.
- **What is wrong:** "a median of 30 ms per shot, about 3,200 times batch decoding". The 3,200 is mean over mean.
- **Witness** (from `certify_cost_d5_r25.json`):
  - median certify / batch decode = 1,836.1;
  - mean certify / batch decode = 3,216.8;
  - 30 ms / 3,200 = 9.4 µs, but the recorded batch decode is 16.35 µs.
- **Further problem:** 30 ms is the cheapest configuration (distance 5, 25 rounds, simulated). A held-out distance-5, 50-round chunk took 104 s per 250 shots, battery included.
- **Fix:** "median 30 ms (mean 53 ms) at distance 5 over 25 rounds; the mean is about 3,200 times batch decoding."

**M5. "Preregistered" rests on a local, unpushed, self-timestamped commit.**
- **Where:** line 29 and line 169.
- **Witness:**
  - `git branch -r --contains f9074d55` returns nothing.
  - The last push to `origin/main` was 2026-10-29 22:58.
- **What does hold:** the order is as claimed. The commit is at 12:29:19, the first held-out file (`failures.jsonl`) appeared at 12:30:05, and no `src/` file changed in any commit before 0052b997.
- **What does not hold:** every one of those timestamps is local and set by the author.
- **Same weakness elsewhere:** the manifest cited as proof that no generation read 50-round data was itself created in the preregistration commit. That said, no `.py` file at 0d26863a references r50.
- **Fix:** say "a protocol fixed in a commit made before the run (local timestamps)". In future, push the commit publicly or timestamp it externally before running.

## MINOR
- **m1. Preregistered reporting not fully in the paper.** The preregistration promised per-configuration rows, exactly-optimal counts and the largest ε. These are in the JSON but not the paper: exactly optimal is 7,778 of 10,000 at d3 against 206 of 10,000 at d5, and the largest ε is 1.97e-12. The confidence interval also needs its population stated: it covers a 50/50 mix of two placements, taken as the first 5,000 consecutive shots of each run.
- **m2. Cost section mixes statistics and overstates the load.**
  - The refused-shot "13 times" is a median over n = 4; the mean ratio is 7.6. The 24% for checking is median over median; mean over mean is 14%.
  - "At least 70% busy" conflicts with the recorded `cpu_percent_before` of 69.5, and the load is two 2-second samples taken before and after the run, not during it.
  - The recorded commit 0d26863a does not contain `certify_cost.py`. The recorded SHA-256s do match the current files.
- **m3. Forgery battery still misses the class that broke the sibling checkers.** It has no T-even or LP-searched dual class, and `dual_forgery` is never run against `check_epsilon`.
- **m4. Replay records still carry no per-verifier tallies.** Prior finding 6 was reworded in the text, not instrumented in the evidence.
- **m5. `matching_cert_epsilon.py:121` coerces correction endpoints with `int()`.** An edge written (2.9, 3) is checked as (2, 3). This is not a soundness hole for the parsed correction, but syndrome entries and set members get strict type checks and endpoints do not.
- **m6. Abstract false-alarm figure lacks context.** "1 of 770" has no interval and does not name the ladder (the full ladder gave 0 of 770). Since false alarms are a subset of not-proven shots, the 10^6-shot not-proven rate bounds the false-alarm rate far more tightly.
- **m7. Abstract drops "with current producers".** Checking alone already costs about 432 times batch decoding (median), which independently supports "not in the real-time loop".
- **m8. Wording left from the prior review.** The title still says "optimality" where the claim is ε-optimality (prior finding 16). "Today's producer" and "today's exact tier" remain (lines 120 and 159).
- **m9. Simulation seed plan.** The paper still does not say whether the producer was developed on the simulation's seed plan. A fresh-seed held-out simulation would be cheap.
- **m10. The audit rerun is still writing `certification_semantics_audit.json`.** Regenerate `numbers.tex` after it finishes, before posting.
- **m11. Related work spot check (arXiv abstracts only).**
  - Shen & Zhong (2609.07035): the abstract matches the paper ("two feasible corrections of different weight prove the heavier is not minimum-weight", 639 certificates, no lower bounds). It says "standalone verifier re-derives from first principles", not "imports nothing".
  - Krishnamoorthy et al. (2608.25545): the abstract says "certifying the bulk of its decisions" and "elimination clusters". "Mini-bucket" and "machine-checkable" are not in the abstract.

## Checks that found no problem
- **Run matched the preregistration.**
  - Sample: shots [0, 5000) per configuration (`epsilon_rerun.py:134`, seeds 1–20).
  - Producer and checker: method `certify`, `eps_max` 1/1000000, no chunk retries.
  - Forgeries: 169,999 attempted, none accepted.
  - Replays: all four COMPLETE, 5,000 receipts re-checked each.
  - `make_numbers.py` lines 288–320 refuse a mismatch.
- **Clopper–Pearson interval is right.** My own tail bisection, scipy's `binomtest` and the recorded value all give [0.9944165, 0.9963351].
- **Detection is stated correctly.** It is a corollary of Lemma 1, so a theorem; the false-alarm rate is a measurement. The intervals agree with mine: 1/770 gives 0.00329–0.721%, and 0/770 gives at most 0.478%.
- **`check_epsilon` implements Lemma 1 exactly.**
  - Code walk: the T'-odd test is at lines 139 and 151, and the T-join is validated before the bound is computed.
  - Differential fuzz against brute-force optimum: 12,000 random small graphs, 3,949 acceptances, 0 unsound acceptances, 0 lower bounds above the optimum.
- **Cost section numbers match the JSON**, and the timing run used the same shots as the stored receipts.
- **Prior findings 2, 3, 7, 8, 10, 11, 12 and 13 are fixed** in both text and evidence. Finding 1 is fixed in evidence, but the new abstract mismatch in M4 replaced it.

## Could not check
- Whether the GitHub release actually carries the held-out receipts.
- Whether the development and held-out runs used the same Willow error-model variant.
- Which checker Fig. 1 times (the prior review found an 18× gap).
- Wu et al. (not re-fetched).
- The JavaScript verifier's source.
- The decisive test for M2: whether f9074d55 reproduces the September verdicts on the Willow development shots.

## Output contract
- **Strongest surviving result:**
  - `check_epsilon` is sound per Lemma 1. Evidence class: PROVEN by the lemma, plus MEASURED by the fuzz.
  - The held-out sample gives 19,909 of 20,000, 95% interval 99.44–99.63%. Evidence class: MEASURED.
  - The per-round not-proven rate is unchanged from development to held-out. Evidence class: DERIVED.
- **How it could still be wrong:**
  - The held-out runs used producer code other than f9074d55 (the M2 test settles this).
  - The measured-model certificates included T-even sets (unverifiable without a rerun).
- **Killed claims and what survives:**
  - "The same base ladder" → "the producer at f9074d55".
  - "A median of 30 ms, about 3,200 times batch decoding" → median about 1,800 times; mean about 3,200 times.
  - "Preregistered" → "fixed in a local commit made before the run".
- **Recommendation:** fix the wording, then two cheap reruns: the measured-model section under the fixed checker, and the development d3/d5 shots with the frozen producer.

Proposed memory lines, not written because ONE_Q is read-only for this task:
- WORKED: comparing a pooled held-out rate with a pooled development rate; the mix of configurations decided it.
- INSTRUMENT: the held-out artifacts carry no commit or source hash.
- INSTRUMENT: 56 conformance vectors with no parity case could not see the exact checker's T-odd hole.
- FAILED to kill: `check_epsilon` (12k-trial brute-force fuzz) and the held-out Clopper–Pearson interval.

Scratch scripts:
- <reviewer's local scratch path>
- <reviewer's local scratch path>
