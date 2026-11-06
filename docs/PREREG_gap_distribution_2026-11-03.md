# Preregistration: paper 1's per-shot optimality gaps, re-measured with a correct scorer (2026-11-03)

**Why.** Paper 1 Sec. VIII (Table V, Fig. 3) compared BP-OSD and belief-matching against PyMatching's certified
correction. External audit 2 (2026-11-03, F01) showed its scorer charged a graph edge once per selected component
instead of once per edge of the projected correction; re-running the published experiment bit for bit, 158 of the
1,000 Table V shots and 149 of the 863 Fig. 3 shots were overcounted, by 7.6 to 146.3 weight units each. Fixing the
scorer alone changes the section's conclusion (the "up to 148" maximum becomes 21, and BP-OSD's 99th percentile falls
below belief-matching's at d=5 and d=7), and the adjudication found the remaining gaps depend strongly on how BP-OSD is
configured (its input model and unscaled min-sum; audit 2, F02). The released numbers were measured on a seed the
whole development history had used and on a reference subset chosen by the exact checker, which keeps easier shots
(audit 1). This file fixes a fresh measurement before any of its shots is decoded; it is committed ahead of the run,
and the commit's timestamp is the record.

**Data (never used).** Stim `surface_code:rotated_memory_z`, the paper's configuration: distance d in {3, 5, 7},
rounds = d, `after_clifford_depolarization = before_measure_flip_probability = after_reset_flip_probability =
before_round_data_depolarization = 0.005`, the decomposed detector error model (`decompose_errors=True`). Sampler
`compile_detector_sampler(seed=4108100 + d)`, i.e. **seeds 4108103 (d=3), 4108105 (d=5), 4108107 (d=7)**: none
appears in any tracked file of the repository on 2026-11-03 (`git grep -w`), and the seeds of every earlier gap run
(31337) and of every other recorded run differ. **2,000 shots per distance**, all of them; every nontrivial shot is
reported, none excluded. Stim's seeded sampler is not bit-identical across platforms, so the per-shot records keep
every syndrome and the analysis below is a function of those records.

**Decoders (fixed here, read back from the decoder objects into the record).**

| arm | input model | settings |
|---|---|---|
| `bposd_merged` (PRIMARY) | one column per matching-graph edge; probability = the XOR-combination `p <- p(1-q) + q(1-p)` of every graphlike component on that edge (this reproduces PyMatching's merged weight to 5e-16 relative at d=3, 5, 7) | ldpc 2.4.1 `BpOsdDecoder`: `bp_method="ms"`, `ms_scaling_factor=0.625`, `max_iter=30`, `schedule="parallel"`, `osd_method="osd_cs"`, `osd_order=5` |
| `bposd_component` (SECONDARY, input-model sensitivity) | the released construction: one column per graphlike component of each decomposed instruction, carrying the instruction's probability (components of one instruction treated as independent) | the released settings: as above but `ms_scaling_factor` left at the library default (1.0) |
| `belief_matching` | `BeliefMatching(dem, max_bp_iters=20)`, its applied edge set recovered as in the released script | as released |
| `pymatching` | `Matching.from_detector_error_model(dem)` | the reference decoder |

**Scoring.** Every candidate's selection is projected onto graph edges mod 2 (a correction is an edge set), its
odd-degree set compared with the syndrome (a mismatch is reported as syndrome-inconsistent, never scored), and its
cost computed as the sum of `Fraction(w_e)` over the projected edges -- the binary64 weights the certificate checker
reads. No tolerance enters any comparison.

**Reference (PRIMARY: the epsilon subset).** PyMatching's correction J is certified by
`CertifyingDecoder(edges, epsilon_max=1e-6).certify` (the base ladder) and judged by `check_epsilon`, eps_max = 1e-6.
On every shot it accepts, OPT lies in [w(J) - eps(J), w(J)], so a candidate K's excess lies in
`[max(0, g), g + eps(J)]` with `g = w(K) - w(J)` exact. Per candidate the record holds the interval.
**SECONDARY reference: the exact subset**, J accepted by the exact checker (`CertifyingDecoder(edges).certify`), on
which the excess is exactly g. It is reported because the original section used it, with the statement that it
selects easier shots.

**Reported, whatever it shows**, per distance and per arm (`bposd_merged`, `bposd_component`, `belief_matching`):

1. nontrivial shots; reference shots under each rule; syndrome-inconsistent corrections;
2. on the epsilon subset: **proven epsilon-optimal** (upper end <= 1e-6; the primary hit measure), proven exact hit
   (upper end = 0), possible exact hit (lower end = 0), proven miss beyond eps (lower end > 1e-6), undecided, each
   with its rate and an exact two-sided 95% Clopper-Pearson interval; the median, 99th percentile (NumPy linear
   interpolation) and maximum of the lower and of the upper end of the gap interval;
3. on the exact subset: exact hit rate (g = 0 exactly), the median, 99th percentile and maximum of g;
4. the number of shots on which the arm selected more than one column of an edge (cancelled copies).

The conclusion is whatever these numbers say, including any ordering of the three arms; in particular nothing here
presumes that belief-matching lands closer to the optimum than BP-OSD, or the reverse, and the input-model arm is
reported as a sensitivity, not as "the" BP-OSD. Paper 1 will state which configuration each number belongs to and
will not call BP-OSD a maximum-likelihood decoder.

**What this measurement is not.** Weight, not homology class; a positive gap says K is heavier than the
minimum-weight correction under the matching graph's weights, not that K is a logical error. BP with finitely many
iterations and bounded-order OSD is a heuristic on either input model.

**Runner.** `python experiments/m0_prior_art/gap_distribution.py --seed 4108100 --shots 2000 --tag prereg_2026-11-03`
(writes `evidence/gap_distribution_v4/prereg_2026-11-03/summary.json` and `per_shot_d{3,5,7}.jsonl.gz`), at the commit
that carries this file, with `src/oneq` and `experiments/m0_prior_art` clean. Expected cost on the workstation that
produced the original artifacts: under 30 minutes, under 1 GB. The earlier artifacts
(`evidence/gap_distribution.json`, `evidence/gap_distribution_exact_rule.json`) are kept unchanged as the record of
the defective measurement.
