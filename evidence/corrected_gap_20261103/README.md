# Corrected graph-gap measurements, November 3, 2026

**The TRAIN-selected BP baseline improves graph cost relative to unscaled BP
in this held-out sample, with increased decoding time. MWPM remains the strong
baseline for this graph objective. This experiment establishes no logical-error
advantage or speedup over MWPM.**

The scoring/protocol implementation was committed before this measured run at
`282ad3b38f724428582d1102fcfcaf4d2c0966d4`; the source was clean. Versioned
source hashes, dependency versions, model graphs, original Stim circuits/DEMs,
syndromes, selected component columns, projected edge sets, exact rational
costs, cut witnesses and measured wall times are retained in the accompanying
summary and compressed trace.

## Primary tuned comparison

Four min-sum scales `{0.5, 0.625, 0.75, 1.0}` were compared on 128 separate
TRAIN draws per distance. Selection minimizes invalid answers, then exact total
graph weight, with a fixed scale tie-break. It uses neither logical labels nor
HELDOUT observations. The selected scale is frozen before 400 common HELDOUT
draws per distance. Circuit rounds equal distance; all noise probabilities are
0.005. Seeds are 131337 (TRAIN) and 31337 (HELDOUT).

| Distance | TRAIN-selected scale | Nontrivial HELDOUT | Exact references / nontrivial | Tuned BP exact / exact-reference subset | Mean exact gap | p99 exact gap |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.5 | 250 | 249/250 (99.60%) | 239/249 (95.98%) | 0.023993 | 0.856393 |
| 5 | 0.5 | 397 | 386/397 (97.23%) | 352/386 (91.19%) | 0.069132 | 1.589115 |
| 7 | 0.75 | 400 | 242/400 (60.50%) | 207/242 (85.54%) | 0.074763 | 1.449130 |

All 1,200 HELDOUT draws are retained. The 153 trivial syndromes are explicitly
excluded from this gap distribution, leaving 1,047 nontrivial comparisons.
Reference optimality was proved on 877; 170 retained safe lower bounds without
an exact-optimality claim. All candidate outputs in this run were valid and
syndrome-consistent. There were no negative rational gaps against checked
bounds. “Exact” means rational equality, including differences smaller than
any usual floating-point tolerance. Quantiles displayed above are approximate.

The tuned BP answers have 79 strictly positive exact graph gaps among the 877
certified-reference shots. These are deviations from the minimum graph cost,
not counts of logical failures. The experiment does not estimate the effect of
replacing those answers on logical failure probability.

## Secondary unscaled diagnostic

| Distance | Unscaled BP exact / exact-reference subset | Mean exact gap | p99 exact gap |
|---|---:|---:|---:|
| 3 | 223/249 (89.56%) | 0.120089 | 2.634550 |
| 5 | 336/386 (87.05%) | 0.132100 | 1.830769 |
| 7 | 198/242 (81.82%) | 0.109488 | 1.689954 |

The selected BP configuration is the primary baseline. The unscaled comparison
illustrates sensitivity to decoder configuration; it must not carry the main
performance conclusion.

## Measured cost and certificate refusal

Times are local wall-clock observations over all nontrivial HELDOUT attempts,
including refused certificates. They do not establish general performance or
a hardware control deadline. Decoder, exact projection, proof generation and
recheck timings are retained separately in `summary.json` and the trace.

| Distance | Tuned BP decode mean ms | Unscaled BP decode mean ms | MWPM decode mean ms | Reference certify mean ms | Retained-witness recheck mean ms | TRAIN total s |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.960 | 0.159 | 0.050 | 5.689 | 0.626 | 0.165 |
| 5 | 28.416 | 12.400 | 0.092 | 11.610 | 3.263 | 12.221 |
| 7 | 276.499 | 257.327 | 0.143 | 97.693 | 9.509 | 145.983 |

“Certify” includes the certifier's internal checker calls. Recheck is the
additional retained-witness check used by this benchmark. TRAIN cost is kept
separate and is not treated as free. The fixed serial decoder ordering and a
single local timing run do not support a broad speed comparison.

The full reference pipeline below comprises MWPM decoding, exact projection,
certification and retained-witness rechecking. Model setup, training, sampling
and artifact serialization are excluded. The availability values are
**retrospective**: a proof counted only if it succeeded within the recorded
time. The experiment did not interrupt workers at the budgets shown.

| Distance | Mean accepted pipeline ms | Mean refused pipeline ms | Exact proof within 25 ms / all nontrivial | Within 100 ms / all nontrivial |
|---|---:|---:|---:|---:|
| 3 | 6.336 (249 cases) | 17.298 (1 case) | 248/250 (99.20%) | 248/250 (99.20%) |
| 5 | 13.841 (386 cases) | 55.606 (11 cases) | 356/397 (89.67%) | 386/397 (97.23%) |
| 7 | 63.625 (242 cases) | 174.529 (158 cases) | 52/400 (13.00%) | 189/400 (47.25%) |

The distance-7 result exposes a practical limitation: exact coverage is 60.5%
in this configuration, and refusals consume more time on average than accepted
proofs. A budget-aware runtime must retain those refusals and their cost.

## Verification and reproduction

The standalone replay passed on all 1,047 retained nontrivial references,
including all 170 exact-proof refusals. It independently reconstructs XOR
projection, rational costs, training selection, denominators, exact zero
classification, quantiles and timing aggregation, and runs the mathematical
checkers on the retained witnesses. It does not import the experiment, Stim or
BP-OSD. Replaying a clock reading cannot attest hardware performance or prove
which decoder produced an answer; those limits remain explicit.

```bash
python tools/replay_corrected_gaps.py evidence/corrected_gap_20261103/summary.json
python -m pytest tests/test_corrected_gap_distribution.py -q
```

The 19 focused tests passed. They include the frozen nine-copy scoring
counterexample, all 64 selections of a small duplicated-column model, exact
equality below floating tolerance, malformed selections, and seven semantic
mutations whose checksums were recomputed. All seven semantic forgeries were
refused for their substantive defect.

To regenerate a fresh run and operating-cost analysis, see
[`docs/CORRECTED_GAP_BENCHMARK.md`](../../docs/CORRECTED_GAP_BENCHMARK.md).

## Scope

BP-OSD here uses independent graphlike components, not the original correlated
DEM events. It is a finite BP/OSD heuristic. The common scoring objective is the
matching graph's weight, and it does not establish logical-coset ML or logical
failure performance. Pure logical events and original component correlations
are not represented by this decoder model. Observed logical bits are retained
but neither used for tuning nor promoted to a logical-performance result.

These are fresh measurements on the stated software and sampling protocol.
They do not retroactively repair the historical paper tables, and the original
`gap_distribution.json`, `gap_distribution_exact_rule.json`, LaTeX tables and
figures remain unchanged in this branch.
