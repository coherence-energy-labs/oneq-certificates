# Native Sparse Blossom dual export and portable verification

This implementation exports the terminal primal-dual state already computed by
PyMatching's Sparse Blossom. It snapshots `GraphFillRegion.radius` and nested
`blossom_children` before normal matching extraction frees them. The original
SearchFlooder then expands matching pairs into an actual XOR graph correction
from that same solve. Upstream matching code is unchanged.

## Mathematical contract

For terminal set T, a leaf supplies a singleton potential and every nontrivial
blossom supplies a nonnegative multiplier of an odd terminal-set cut. The
independent checker recomputes exact shortest-path distances on the declared
nonnegative integer graph, then checks every terminal-pair and boundary dual
inequality. It checks pair coverage, primal-dual attainment, the actual graph
correction, and its syndrome. This proves minimum T-join weight on that graph.

These are metric-closure constraints: they must not be confused with putting
unrestricted potentials on nonterminal vertices of the original sparse graph.
That would recreate the old mixed-dual soundness defect. No nonterminal is
permitted in the native witness's terminal-set supports.

The public PyMatching Python 2.4.0 API exposes no dual-export method. That is an
API omission, not a mathematical or practical impossibility. This bridge uses
14 existing C++ implementation files and a small extra translation unit; it does
not run a second LP or a second matching algorithm to discover the witness.

## Original floating-point weights

PyMatching internally rounds input weights onto an integer lattice, then doubles
them so collision times remain integral (`user_graph.h` and `user_graph.cc`).
Exactness on those integers does not imply exactness for the input IEEE doubles.

`reconcile_original.py` recomputes original shortest-path distances on the exact
binary64 rational lattice. It scales the native dual uniformly by the largest
factor obeying every original metric constraint. This gives a valid original-
model lower bound and an exact rational epsilon for the candidate correction.
It labels exact optimality only when epsilon is exactly zero. A caller-requested
epsilon of 1e-9 is checked exactly; failure is reported honestly. There is no
claim that this simple scaling recovers the tightest possible original dual.

## Build and test

Requirements: Python 3.12 or newer, a C++20 g++ toolchain, and the pinned upstream source.
The bridge and portable verifier need no Stim/PyMatching Python installation.
Only the surface-code benchmark needs numpy, stim 1.16.0 and pymatching 2.4.0,
plus the existing ONE_Q certificate stack and its SciPy dependencies.

From the repository root:

```bash
git clone https://github.com/oscarhiggott/PyMatching.git /tmp/PyMatching
git -C /tmp/PyMatching checkout 6f63b2b9474ba0fa7e511fe52bffdce858a06984
mkdir -p .build
python tools/native_matching_duals/build_native.py --upstream /tmp/PyMatching --out .build/native_dual.so
python tools/native_matching_duals/validate_native.py --library .build/native_dual.so
python tools/native_matching_duals/validate_adapter.py --library .build/native_dual.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/native_matching_duals/benchmark_native.py \
  --oneq . --library .build/native_dual.so --shots 64 --ladder-shots 12 \
  --output .build/native_benchmark_rerun.json
```

`native_dual.manifest.json` records upstream commit, all C++ sources and headers,
bridge/build-script hashes, compiler, build command, binary hash and whether the
upstream checkout was modified. UPSTREAM_LICENSE and NOTICE.txt travel with it.
The retained manifest describes the measured native binary; a new build writes
its own manifest next to the output library. No binary is committed here.
Building under `.build/` preserves the retained measured manifest. The optional
`--library` arguments affect only how the already-built extension is located;
they do not change the measured algorithm.

On Windows with MinGW g++, use a `.dll` output. The builder copies the compiler's
runtime DLLs beside that output and records their hashes in the new build
manifest. This works with Python's restricted DLL search without changing the
machine-wide search path. A fresh Windows source checkout can be prepared with
`git clone --no-checkout`, followed by `git config core.longpaths true` inside
that checkout and checkout of the pinned commit, to avoid upstream benchmark
filenames exceeding the default Windows Git path limit. The Windows build has
been loaded and exercised with Python 3.14; those observations do not replace
the retained Linux benchmark timings.

### Independently replay the retained proofs

From the repository root, this command checks all 256 retained proofs and their
original-weight bounds without installing PyMatching, Stim, SciPy, or ONE_Q:

```bash
python -I -S tools/native_matching_duals/replay_native.py
```

This reproduces mathematical verdicts and exact rational bounds. It does not
pretend to reproduce the recorded machine timings. The packet retains the full
integer and original graphs, every syndrome, dual, correction, and result.
The replay requires every declared shot identity exactly once, checks the graph
digests and quantization, and recomputes the summary counts, bounds and timing
aggregates. An omitted shot or altered summary fails rather than reducing the
reported denominator. The individual recorded clock observations and ladder
verdicts remain measurements, not claims proved by the native checker.

## Runtime adapter

```python
from fractions import Fraction
from tools.native_matching_duals.native_metric_producer import NativeMetricProducer
from tools.native_matching_duals.native_metric_verify import verify_document

edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0),
         (0, -1, 10.0), (1, -1, 10.0), (2, -1, 10.0)]
syndrome = [0, 1, 2]
producer = NativeMetricProducer(edges, library_path=".build/native_dual.so")
try:
    document = producer.produce(syndrome)
    checked = verify_document(document, edges, syndrome, epsilon=Fraction(0))
    print(checked)
finally:
    producer.close()
```

The producer may instead certify a supplied candidate edge set with
`produce(syndrome, correction=edges_of_existing_candidate)`. Both paths bind
the original graph's exact IEEE bits and the syndrome. The independent verifier
imports only the Python standard library and the two local checker files, never
the producer or native extension. Its result separates `bound_valid` from
`accepted` under the requested epsilon. A valid bound missing epsilon must not
be relabelled exact or silently admitted by an operating system.

## Verification and benchmark scope

The small-graph validation compares against an independent exhaustive syndrome
dynamic program: 160 random graphs, 1,930 feasible syndromes, 498 infeasible
refusals, 615 actual blossom proofs and 5,541 malformed/overstated proof refusals.

The benchmark uses the same syndrome and integer-weight objective for the native
proof and the existing certificate ladder. Each cell has 64 Stim shots and 12
nontrivial ladder comparisons. It includes refused ladder outcomes. Native
mode microbenchmarks use six counterbalanced same-syndrome repetitions after
warm-up; Python/native export and independent verification are separately timed.
These are local exploratory timings, not a preregistered general speed claim.

Retained run, identical integer objective and correction for both producers:

| Distance / rounds | Native decode + export + independent check, median ms | Existing ladder certificate only, median ms | Native / ladder accepted on compared shots |
|---|---:|---:|---:|
| 3 / 3 | 0.1064 | 3.6912 | 12 / 12; 12 / 12 |
| 5 / 5 | 0.8323 | 8.5594 | 12 / 12; 12 / 12 |
| 7 / 7 | 5.0263 | 12.2831 | 12 / 12; 11 / 12 |
| 5 / 25 | 11.4847 | 22.6696 | 12 / 12; 11 / 12 |

There are 12 nontrivial compared shots per cell. The native integer proof passes
on all 64 shots per cell (256 total). Decoder-only/export measurements use
counterbalanced repetitions; the exact checker and ladder are timed once per
shot. Tail outliers occur on the shared host. The ladder times include refusals;
the native column includes decoding work in addition to certificate production.

Original IEEE weights, using the exact uniform metric lift:

| Distance / rounds | Within epsilon 1e-9, all shots | Within epsilon 1e-9, nontrivial shots | Largest certified epsilon | Reconciliation median ms, nontrivial shots |
|---|---:|---:|---:|---:|
| 3 / 3 | 60 / 64 | 19 / 23 | 2.9681e-7 | 0.2226 |
| 5 / 5 | 26 / 64 | 15 / 53 | 1.3337e-6 | 1.6030 |
| 7 / 7 | 4 / 64 | 3 / 63 | 3.1718e-6 | 8.2790 |
| 5 / 25 | 0 / 64 | 0 / 64 | 4.0502e-6 | 20.6092 |

Reconciliation time is additional to integer verification. A positive epsilon
can reflect slack in this lift and does not by itself prove that the native
correction is suboptimal for the original graph.

The original-weight epsilon result is a separate question from the integer
comparison. Integer proof success is not counted as original exact success.
No end-to-end real-time or physical logical-success claim is made.

Supported scope: nonnegative simple matching graphs, optional boundary edges,
ordinary uncorrelated Sparse Blossom. Negative-edge preprocessing, correlation-
conditioned reweighting, degenerate logical-coset likelihoods and hardware
calibration are outside this prototype's acceptance contract.

## Primary source pin

Upstream: `oscarhiggott/PyMatching`, commit
`6f63b2b9474ba0fa7e511fe52bffdce858a06984`.

- `sparse_blossom/flooder/graph_fill_region.h`: region radii and nested blossom state.
- `sparse_blossom/driver/mwpm_decoding.cc`: event loop, destructive extraction, and original path expansion.
- `sparse_blossom/driver/user_graph.h` and `.cc`: input rounding, normalization, and doubled integer weights.

All paths are under upstream `src/pymatching`. This integration exports existing
solver state; it does not claim a new matching algorithm or a new duality theorem.
