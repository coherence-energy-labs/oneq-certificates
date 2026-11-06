# Preregistration: a held-out distance-7 Willow sample for paper 1's full ladder (2026-11-03)

**Why.** Paper 1's abstract reports the full ladder proving 99.998% of 100,000 Willow corrections at distance 7 over
30 rounds, "shots the producer was developed on": generations 5 and 6 (the escalation and exact tiers) were introduced
against exactly those shots, and the 2026-11-02 held-out sample excluded distance 7 for cost. External review 1
(2026-11-03) asked for an out-of-sample check of that figure. This file fixes it before any of its shots is read; it
is committed, with its runner, ahead of the run, and the commit's timestamp is the record.

**What exists (checked 2026-11-03 from the archive's ZIP central directory and each configuration's
`metadata.json`, Zenodo 10.5281/zenodo.13273331, `google_105Q_surface_code_d3_d5_d7.zip`).** Distance 7 was run at
ONE placement, `d7_at_q6_7`, in both bases, at rounds 1, 10, 13, 30, 50, 70, ..., 250, with 50,000 shots per
configuration. Paper 1 used 13 rounds (generation 1 read all 50,000 shots per basis) and 30 rounds (generations 5 and
6 read all 50,000 per basis). **No unused distance-7 shot exists at 30 rounds**, at any placement. The nearest unused
round counts are 10 and 50, equally near. The primary sample is **50 rounds**: it is the harder direction (more
detectors, larger graphs, the regime where the escalation and exact tiers engage), so it is the stricter test of the
tiers fitted at 30 rounds; it matches the 2026-11-02 held-out protocol for distances 3 and 5 (also 50 rounds); and a
producer fitted to its development shots would be expected to do worse, not better, on harder unseen data. Because
rounds differ from development, a change in rate cannot by itself separate overfitting from the change of
configuration; that limit is stated with the result, and the bracketing arm below exists to narrow it.

**Data (never used in development).** `d7_at_q6_7`, bases X and Z, 50 rounds (PRIMARY) and 10 rounds (BRACKET), with
the experimenters' graphlike model `correlated_matching_decoder_with_si1000_prior`, fetched by
`tools/fetch_google_corpus.py --rounds 50 10 --distances 7` and pinned by SHA-256 in
`evidence/google_corpus_manifest_heldout_d7.json` before any shot is decoded (about 17 MB by HTTP range: the archive lists
the model for both round counts). The corpus fetched for paper 1 held 13
and 30 rounds only (`evidence/google_corpus_manifest.json`).

**Sample (fixed rule).** In each held-out configuration, the first N shots, `[0, N)`, all of them, in chunks of 50; no
exclusion, no re-run, every chunk retried until it completes. N is set by the measured cost below:
**N = 2,000 per basis** at 50 rounds and 2,000 per basis at 10 rounds (8,000 held-out shots).

**Same-producer development arm.** The development rate (99.998%) was measured on 2026-09-17, before the producer
optimisations of 2026-10-21, so it is not the frozen producer's rate. The frozen producer therefore also re-certifies
the first 2,000 shots of `d7_at_q6_7` X and Z at 30 rounds (development shots, `[0, 2000)`), so that the comparison
held-out vs development is made by one producer, and is reported beside the recorded 2026-09-17 rate.

**Producer (frozen).** `CertifyingDecoder(edges, epsilon_max=1e-6).certify_hard` -- the full ladder (base, escalation,
exact separation tier with its round cap), exactly the call `experiments/m0_prior_art/epsilon_rerun.py` makes for the
two distance-7 30-round configurations -- at the commit that carries this file, with `src/oneq` clean. Decoder:
PyMatching on the experimenters' model.

**Checker.** `matching_cert_epsilon.check_epsilon`, eps_max = 1e-6, exact rationals; every receipt kept
(`data/epsilon_receipts/heldout_d7_*/`, per-chunk SHA-256 in `evidence/epsilon_rerun/heldout_d7_*/`), replayed by
both verifiers (`tools/replay_epsilon_receipts.py --battery 1`) and by `verifier-js/verify-epsilon.mjs`. The forgery
battery (`oneq.forgeries` at the frozen commit) runs on every shot with a packing.

**Cost, measured on development shots only** (`evidence/heldout_d7_cost_calibration.json`,
`experiments/m0_prior_art/heldout_d7_cost_calibration.py`): on 20 development shots at 30 rounds (shots 17 + 5000 i,
i = 0..9, in each basis) the frozen producer, check and battery took a mean of 1.27 CPU-seconds per shot (median 0.70,
maximum 6.03; certification 1.02, battery 0.22), all 20 proven within eps_max, peak working set 221 MB. That is about
14 times faster than the same configuration's 2026-09-17 rerun (17-19 s per shot), the producer having been optimised
on 2026-10-21. Twenty shots cannot measure the exact tier, which engaged on 1.8% of development shots (918 and 895 of
50,000 per basis in generation 6), so the upper estimate takes generation 6's recorded 3.87 core-seconds per shot at 30
rounds. Rounds 30 to 50 are scaled by 1.86 (the measured ratio at distance 5 for the base ladder with the battery,
0.248 to 0.462 s per shot) for the lower estimate and by (50/30)^2 = 2.8 for the upper; 10 rounds are taken at no more
than the 30-round cost. For 2,000 shots per basis in each of the three arms (12,000 shots): **4 to 21 core-hours**,
inside the 40 core-hour ceiling, so N = 2,000; with 3 workers about 1.5 to 7 hours of wall time and an expected peak
under 1.5 GB in all.

**Reported, whatever it shows.** Per configuration (3 arms x 2 bases) and pooled per arm: nontrivial shots, proven
within eps_max, exactly optimal, not proven, the largest accepted epsilon; the not-proven rate with its **exact
one-sided 95% Clopper-Pearson upper bound**; and, for held-out 50 rounds against the same-producer development arm,
the exact one-sided conditional binomial test of H0 "the held-out not-proven rate is at most the development rate"
(the number of not-proven shots in the held-out arm given the total in both arms, success probability n_h/(n_h+n_d)),
with its p-value. The recorded 2026-09-17 development rate (2 not proven of 100,000) is reported beside it, labelled
as a different producer. No per-round normalisation is used for inference (external audit 2, Sec. 8): the rate is
reported per shot, and rounds are named with every rate.

**Interpretation, fixed now.** If the held-out 50-round not-proven rate's one-sided 95% upper bound is below 1%, paper
1 may say the full ladder proved at least (100 - bound)% of held-out 50-round corrections; it will say what the
sample size can and cannot exclude (with no refusal among the 4,000 pooled 50-round shots the bound is
1 - 0.05^(1/4000) = 0.075%, which bounds the rate but cannot confirm 99.998%). If the held-out 50-round rate is
significantly higher than the same-producer development rate (p < 0.05), the paper reports that as consistent with
fitting to the development shots OR with the harder configuration, and uses the bracket to say which the data favour:
a 10-round held-out rate significantly above the 30-round development rate (fewer rounds, yet more refusals) points
to fitting; a 10-round rate at or below it, with the 50-round rate above, is what a configuration effect predicts;
anything else is reported as undecided. Whatever the outcome, the 99.998% development figure is labelled in the
abstract as a development-sample rate.

**Stopping rule.** All chunks of all arms are run; nothing is stopped on a result. If the run is interrupted, it
resumes from its chunk index. If after 40 core-hours the run has not completed, it stops at a chunk boundary,
every completed chunk is reported as a partial sample with the reason, and no chunk is re-selected or replaced.

**Runner.** `python experiments/m0_prior_art/heldout_willow_d7.py --workers 3` then `--merge`, writing
`evidence/epsilon_rerun/heldout_d7.json` (the runner registers the six arms above in the epsilon rerun; it is committed with
this file, before the run).
