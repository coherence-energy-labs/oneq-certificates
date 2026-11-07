# Deviations from `docs/PREREG_heldout_d7_2026-11-03.md`

The preregistration and its runner (`experiments/m0_prior_art/heldout_willow_d7.py`) were committed in 03b03e66 and
pushed with origin/main 2008c81f before any of its shots was certified. The run was then started exactly as committed:
`heldout_willow_d7.py --workers 3`, at 2026-11-03 14:35:29 EST, in a `git archive` snapshot of 2008c81f (so that
`src/oneq` was clean while other lanes edit the working tree). This file records every departure from the
preregistration, when it happened, why, and what it changes. The preregistration itself is not edited.

## Deviation 1 (2026-11-03, 16:18 EST): the arms are interleaved

**What happened.** The preregistration fixed N from a cost calibration on 20 development shots at 30 rounds and
estimated 4 to 21 core-hours for the whole run, inside its 40 core-hour ceiling. In per-chunk terms (50 shots) its
figures give 118 s at 50 rounds for the lower estimate (1.27 s per shot x 1.86) and 542 s for the upper (3.87 s per
shot x (50/30)^2). The first chunks at 50 rounds took:

| chunk (heldout_d7_X_r50) | seconds | source |
|---|---|---|
| 1 | 150.65 | the index line's `seconds` |
| 3 | 1,098.13 | the index line's `seconds` |
| 4 | about 5,070 | worker process start (14:38) to the next worker's start (16:02:32) |
| 5 | about 4,940 | worker process start (14:53:50) to the next worker's start (16:16:09) |
| 2 | still running after 6,096 CPU-seconds when the run was stopped | the worker's CPU time |

Chunks 4 and 5 took about 9 times the upper estimate, and chunk 2 had already run more than 11 times as long. At the stop (16:18:10, 5.13 core-hours after the start) 4 of the
240 chunks were complete, all in the first arm.

**Diagnosis.** The preregistration already notes that 20 shots cannot measure the exact tier, which engaged on 1.8%
of development shots at 30 rounds. Stack samples of two running workers (`py-spy dump`, 15:58) showed both in the
exact tier's rational linear program, `oneq.exact_lp.exact_weighted_lp`, called from
`CertifyingDecoder.certify_exact`, one of them through `oneq.repair_impossible.prove_impossible_exact`. At 50 rounds
that tier is reached more often or costs more per shot than the calibration allowed for. The calibration could not
have shown this.

**Why the order had to change.** The committed runner runs the six arms one after another: X 50 rounds, X 10 rounds,
same-producer development X 30 rounds, then the same three in Z. At about 1.9 completed chunks per wall-clock hour,
the 40 core-hour stop at 2026-11-04 03:55:29 would have fallen at about chunk 25 of the first arm's 40. No other arm
would have started. The preregistered conditional test, and the interpretation rule that reads it against the
10-round bracket, both need the same-producer development arm, so they could not have been computed. The arm order is
a property of the runner. The preregistration does not fix it.

**Nobody had read any result.** When this deviation was decided and written, no one had read a chunk's verdict or
status.
* The run was stopped by a script that counted index lines and seed numbers only. The watchdog that enforced the
  stopping rule until then did the same.
* No receipt file and no failures file was opened.
* Disclosed in full, the only fields of any chunk record anyone displayed are:
  * `seconds` of chunks 1 and 3 (above);
  * their `epsilon_max_accepted` (the largest accepted epsilon, about 6.6e-13 and 5.2e-13), printed with the
    `seconds` at 15:57 while the record's `status` and forgery fields were masked;
  * their `battery_not_false` (0 and 0).
* None of these fields says how many shots were not proven.

**The new order (the only change).**
* The driver `experiments/m0_prior_art/heldout_willow_d7_interleaved.py` cycles through the six arms in their
  committed order, X r50, X r10, samedev X r30, Z r50, Z r10, samedev Z r30, and again.
* Each arm takes its next uncompleted chunk in seed order.
* Chunks 1, 3, 4 and 5 of X r50 keep their place as that arm's first chunks.
* Chunks 2, 6 and 7 were being computed when the run was stopped. Their work was lost. They are X r50's next chunks
  and are recomputed first in that arm's turns.

**What does not change.**
* The producer is the frozen snapshot of 2008c81f. The driver imports `heldout_willow_d7` from it unmodified, which
  registers the six arms (shots, chunk size, seeds, `certify_hard`).
* Every chunk is computed by `epsilon_rerun.run_chunk` and indexed exactly as `epsilon_rerun.run` indexes it: one line
  per completed chunk, failures retried for up to six passes, a fresh worker per chunk.
* The receipts, the checker, the forgery battery, the analysis and the interpretation are as preregistered.
* The stopping rule keeps its deadline. 40 core-hours on 3 workers from the original start at 14:35:29 EST is
  13 h 20 min of wall time, so the deadline stays **2026-11-04 03:55:29 EST**. Three things count against it:
  * the lost in-flight work, 7,135 CPU-seconds (about 2.0 core-hours);
  * the interval between the stop and the restart;
  * the run itself.

**One addition outside the chunk.**
* When a worker starts a chunk the driver writes a dispatch marker in
  `evidence/epsilon_rerun/heldout_d7_dispatch/` (arm, seed, worker pid, start time). The stopping-rule watchdog
  (`experiments/m0_prior_art/heldout_d7_watchdog.py`) uses the markers to name exactly which chunks are being
  computed when it stops the run.
* Markers carry no verdict.
* The watchdog counts index lines and seeds and never reads a status.

**How a stopped run is reported, per arm.** If the stop comes before every chunk of every arm is complete,
`experiments/m0_prior_art/heldout_d7_partial.py` reports each arm separately, with that arm's own sample size, in three
views. It then pools them per round count as the preregistration pools them.
* **Completed**: every completed chunk, with the preregistered exact one-sided 95% Clopper-Pearson bound. This is what
  the stopping rule names.
* **Worst case**: every chunk handed to a worker, with every nontrivial shot of an unfinished chunk counted as not
  proven. Completion order depends on cost, and cost tracks the tiers that refusals come from, so the completed set
  alone could flatter the rate; this bound cannot be flattered. The nontrivial counts are read from the corpus and say
  nothing about a verdict.
* **Prefix**: the longest run of completed chunks from seed 1.

The preregistered conditional tests are computed only between arms that both have completed chunks, and are labelled
partial. If every chunk completes, `heldout_willow_d7.py --merge` reports the run as preregistered.

**Records.** The stop for this deviation is recorded in `d7_amendment_stop_record.json`:
* the time, 16:18:10, and 5.134 core-hours spent;
* the completed seeds, X r50 1, 3, 4, 5;
* the in-flight seeds 2, 6, 7, with their workers' CPU time;
* the `taskkill` output, with no process left;
* that the index did not change during the stop.

The record, the first watchdog's log, and the restarted run's log are kept with the run's results in
`evidence/epsilon_rerun/`.

**Bytes.** SHA-256 of the bytes as git stores them (LF).

| file | sha256 |
|---|---|
| `experiments/m0_prior_art/heldout_willow_d7.py` (committed, unchanged) | 5887bca8b2268e4974006357bec481c664532f9ab763c0238707ee00ca59b20a |
| `experiments/m0_prior_art/heldout_willow_d7_interleaved.py` (deviation 1) | 84f076c437a8df865ede89ef1e97db5ead6295f76bf06b7cb77539b17c8cae86 |
| `experiments/m0_prior_art/heldout_d7_watchdog.py` (deviation 1) | a34d19ab7c6d03663171c91bf09ea548bf0b460e927df550bb7a8b67c3c6dac4 |
| `experiments/m0_prior_art/heldout_d7_partial.py` (the per-arm partial report, as committed with deviation 1 in 4030f88d) | 9dd066bec2f9f1cb55f79fe35d4719e6d60562f05e1156de18a6ae3d521c94ff |
| `experiments/m0_prior_art/heldout_d7_partial.py` (as run on the stopped sample, 2026-11-05: adds a head that pins the producer's modules to 2008c81f and the driver to 4030f88d, hashes the run records, verifies the copy against the run's tree, and reports the recorded 2026-09-17 development rate beside; the analysis is unchanged; as committed in 90863bb8) | b140804c442953fe76f684074a6d1d6774b0e2d38c9c810d58fa5d9174dedaec |
| `experiments/m0_prior_art/heldout_d7_partial.py` (re-run on the same stopped sample, 2026-11-06, note 1 below: adds Fisher's exact one-sided p beside the preregistered test and names both in the record; every other field of the report is unchanged but the report's own commit and sha256) | a244b2586313a51166f051f87b6871e48d3c5e9ed8849c6fc22b429b5e76e051 |

## Record of the stop (2026-11-04)

The watchdog stopped the run at **2026-11-04 03:55:29 EST**, the 40 core-hours from the original start, with no
not-proven count read by anyone before the stop record was written. The index did not change during the stop. At the
stop, chunks complete per arm (of 40 planned): held-out X 50 rounds 10 (seeds 1-8, 10, 11), held-out X 10 rounds 7,
development X 30 rounds 7, held-out Z 50 rounds 6, held-out Z 10 rounds 7, development Z 30 rounds 6: 43 chunks,
2,150 nontrivial shots. Three chunks were being computed and are not in the sample: held-out X 50 rounds seed 9,
held-out Z 50 rounds seed 7, development Z 30 rounds seed 7. The run's records (both stop records, the watchdog,
babysitter and run logs, the 42 dispatch markers: 46 chunks dispatched over the whole run less the 4 the committed
runner completed before the deviation, which it dispatched without markers) are kept in
`evidence/epsilon_rerun/heldout_d7_stopped_2026-11-04/`. Beside the watchdog (`heldout_d7_watchdog.py`,
a34d19ab7c6d03663171c91bf09ea548bf0b460e927df550bb7a8b67c3c6dac4) a supervisor ran from 2026-11-03 17:10 to the stop,
`experiments/m0_prior_art/heldout_d7_babysitter.py` (committed 2026-11-05 as the bytes that ran, sha256
0c9f042a49244c57590553335d8731a15bc6ff5f35dd37498e21d33f61b697ec): every 60 s it checked the stop record, completion,
the deadline and the driver's pid, and would have restarted the committed driver and a new watchdog with the same
START_EPOCH had the driver died (the preregistration's resume rule); it read index lines only and never acted;
the arm indices, receipts and instances were copied from the snapshot into their preregistered paths and verified
byte for byte; every completed receipt was replayed by both verifiers with the forgery battery in a partial-plan mode
added to `tools/replay_epsilon_receipts.py` for the purpose (`evidence/epsilon_rerun/replay_heldout_d7_2026-11-05.json`);
the per-arm report is `evidence/epsilon_rerun/heldout_d7_partial.json`. The preregistered interpretation rule is
applied by the paper's number generator (`paper/tools/make_numbers.py`, `heldout_d7_macros`) and nowhere else.

## Notes (2026-11-06, external audit 3): labels and reading, not deviations of execution

Nothing below changes what was run, which chunks are in the sample, or any count. The audit is archived at
`docs/reviews/EXTERNAL3_2026-11-06.md` (its "Paper B" is paper 1).

**Note 1: the conditional test's label.** The preregistration's test ("the exact one-sided conditional binomial test
... success probability n_h/(n_h+n_d)") allocates the pooled not-proven count binomially, k_h ~ Binomial(k_h + k_d,
n_h/(n_h + n_d)). That distribution is the exact conditional distribution for two POISSON counts. For two binomial
samples of fixed size, conditioning on the total gives a HYPERGEOMETRIC k_h: Fisher's exact test. So the word "exact"
in the preregistration's name for its test is wrong for this design. The preregistered test stays the preregistered
one: its p-value is reported as before (`p_one_sided`), named "conditional binomial". Fisher's one-sided exact p, with
the alternative that the held-out rate is larger, is reported beside it (`fisher_exact_p_one_sided`), labelled as
not preregistered. On this sample both are 1 for both comparisons (no not-proven shot in any arm), so no reading
changes. `heldout_d7_partial.py` was re-run on 2026-11-06 on the same copied sample with the same arguments; diffed
leaf by leaf against the 2026-11-05 record, it adds the two tests' Fisher p-values and their names, and changes only
the report's own commit and sha256 (Bytes table above). The committed runner `heldout_willow_d7.py` keeps its bytes,
and with them its docstring's old label: its bytes are the preregistration's.

**Note 2: the permitted sentence is not made.** The preregistration says that if the held-out 50-round bound is below
1%, paper 1 "may say the full ladder proved at least (100 - bound)% of held-out 50-round corrections". The bound is
0.374% (800 of 800 completed shots proven), and it is still reported, as the preregistration requires, as the
preregistered computation. Paper 1 does not make the permitted sentence. The bound's coverage guarantee holds for a
sample whose inclusion does not depend on the outcome, and this sample's does: the stop kept the chunks that finished
inside the budget, and a chunk finishes late when its shots need the ladder's expensive tiers, which is where refusals
come from. The paper reports what was observed instead: all 800 completed primary shots proven, and 100 more primary
shots (held-out X 50 rounds seed 9, held-out Z 50 rounds seed 7) unfinished at the stop, apart from the development
chunk (development Z 30 rounds seed 7, 50 shots) also unfinished. The worst case and the prefix view are reported as
descriptive sensitivity analyses, not as bounds with coverage. A coverage claim would need a cohort frozen
independently of completion, a defined per-shot computational budget, and an outcome recorded for every shot,
timeouts included (`docs/FUTURE_WORK.md`).
