# Corrected graph-gap comparison

The November 3 audit found that `gap_distribution.py` counted the weight of
selected graphlike DEM columns before combining duplicate graph edges modulo
two. The historical JSON and paper assets are retained unchanged. New runs
use schema `oneq-gap-distribution/4` and a separate evidence directory.

## Implemented contract

- Convert selected columns to a canonical XOR edge set, including the virtual
  boundary convention. Refuse off-graph edges and malformed binary vectors.
- Check the projected syndrome and evaluate the graph's original binary64
  weights as exact `Fraction` values. Exact zero means equality to zero;
  floating quantiles are display statistics only.
- Use an independently checked exact rational lower bound on each comparable
  shot. Restrict optimum-gap claims to retained witnesses accepted by the
  exact checker. Report certificate refusal separately from decoder failure.
- Freeze a BP scaling factor using a separate TRAIN seed and syndrome-only
  graph costs over the declared scale grid. The primary BP comparison is
  `bp_osd_train_selected`; `bp_osd_unscaled` is a secondary diagnostic.
  Both run on the same HELDOUT syndromes as MWPM.
- Retain selected component columns, projected corrections, syndromes,
  observed logical bits (not used for selection), cut witnesses, exact costs,
  failures, source hashes and local nanosecond timings in compressed JSONL.
  Trivial shots are counted and retained but excluded from the graph-gap
  distribution; all coverage denominators are explicit.

This model treats graphlike components independently and drops correlations
between components of the original event. It is not full correlated-hypergraph
ML decoding. The tuning objective is graph correction weight, not logical
failure probability. No superiority claim about logical decoding follows.

## Reproduce

Use the versions recorded in the resulting artifact. The tested environment
uses Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Stim 1.16.0, PyMatching 2.4.0 and
ldpc 2.4.1. Optional belief-matching is explicitly disabled in this comparison.

```bash
python experiments/m0_prior_art/gap_distribution.py \
  --distances 3 5 7 --shots 400 --train-shots 128 \
  --seed 31337 --train-seed 131337 --bp-scales 0.5 0.625 0.75 1.0 \
  --no-belief-matching \
  --out .build/corrected_gap_rerun/summary.json
python tools/replay_corrected_gaps.py .build/corrected_gap_rerun/summary.json
python tools/analyze_corrected_gap_cost.py .build/corrected_gap_rerun/summary.json \
  --out .build/corrected_gap_rerun/operating_cost.json
python -m pytest tests/test_corrected_gap_distribution.py -q
```

The replay implementation does not import the experiment, Stim, or BP-OSD. It
independently reconstructs the projection, exact costs, TRAIN selection,
classification counts and summary statistics, and invokes the mathematical
checkers on every retained reference witness. It verifies timing aggregation,
not the physical truth of a clock reading or general performance. Source
hashes bind the checker and producer implementation; a future edited checkout
must use the pinned implementation to reproduce the same source check.

The fixture `tests/fixtures/gap_counterexample_20261103.json` preserves the
original audit's explicit `[7,9,10]` syndrome and attaining witness. The fixed
scorer returns 7.543721817379714 instead of 44.3621835074505; the projected
correction is exactly optimal. Tests also reject resealed cost, projection,
proof-verdict, model-binding, denominator, zero-count and TRAIN-selection
mutations. They deliberately cover meaningful semantic failures, rather than
relying on a checksum mismatch.

New samples, library versions and tuning choices do not retroactively correct
the historical paper tables. Replacing those tables requires documenting the
new experiment and its sampling protocol in the manuscript.

## Retained measured run

The [measured results](../evidence/corrected_gap_20261103/README.md) record
400 HELDOUT and 128 TRAIN draws at each distance. The original source-pinned
replay is
`python tools/replay_corrected_gaps.py evidence/corrected_gap_20261103/summary.json`
at the measured source revision. The later matching-endpoint contract hardening
changes one pinned checker file, so that command deliberately refuses the new
checkout. Recheck the unchanged packet against the current checker with
`python tools/recheck_corrected_gaps_current.py evidence/corrected_gap_20261103/summary.json`.
It reports both source identities and every difference; it does not regenerate
or relabel the measurement, nor requalify unrelated historical scale campaigns.
Use a new output directory to regenerate it; the producer refuses to overwrite
existing evidence. The cost analyzer reports retrospective availability curves
and accepted/refused timing separately; it does not implement timeouts.
