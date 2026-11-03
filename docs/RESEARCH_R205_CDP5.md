# R205: Proof-carrying circuit-distance search with verified terminal decisions

**Status:** BUILT / VERIFIED ON SPECIFIED MODELS. **Novelty claim:** original integration to be assessed against prior work; no claim of new physical code distance, quantum advantage, or new mathematical theorem. **Branch:** research/r205-distance-bounds-20261021. **Base:** origin/main at cc1a10f413390651067bbe7355dff0d94ee227d9.

## The scientific problem

The R198 exact circuit-distance prover produces a bit for every interior search decision when at least three faults remain. The independent checker replays those decisions and verifies all prunes. On hard exported projected detector-error models this yields megabyte-scale certificates even where the last few levels are decidable directly by an exact hash-based completion oracle. Producing a smaller proof is not the same as making a quantum computer intrinsically better.

R205 combines (a) an allowed-orbit-aware independent detector packing bound, (b) an exact-integer feasible dual-ascent lower bound, and (c) **a trace-free, exact three-fault terminal decision**. The proof language advances to **CDP5**. The independent checker accepts CDP3/4/5 while applying the correct permitted rules to each version. Existing CDP3 witnesses remain readable, and old checkers reject CDP5.

## Lemma A: orbit-filtered detector packing

At an anchored search state with orbit floor o, every additional mechanism must lie in orbit >= o. Let H_o be the hypergraph whose vertices are firing detectors and whose hyperedges are intersections with syndromes of allowed mechanisms. If U is a subset of firing detectors with no allowed mechanism touching more than one member of U, then any fault completion requires at least |U| mechanisms. Every firing detector is toggled an odd (therefore positive) number of times.

The prover builds suffix-union co-touch masks CT[o,d] and greedily obtains U. The independent checker reconstructs masks from the model's own orbit labels after independently checking the declared orbit structure. Self-edges keep packing well-defined when no allowed mechanism touches a detector. This is an admissible lower bound, not necessarily a maximal packing.

## Lemma B: exact-integer dual ascent

Fix a residual syndrome s, orbit floor o and Q=2520. For each firing detector d, choose an integer y_d >= 0 such that for every allowed mechanism m:

    sum_{d in s intersect syndrome(m)} y_d <= Q.

For any set F of mechanisms that realizes s under XOR, every firing detector appears in at least one member of F, therefore

    sum_{d in s} y_d
       <= sum_{m in F} sum_{d in s intersect syndrome(m)} y_d
       <= Q * |F|.

Thus ceil(sum_d y_d/Q) is a valid lower bound.

Initialize c_d as the largest size of s intersect syndrome(m) over allowed mechanisms m containing d. The assignment y_d = floor(Q/c_d) is feasible: for any mechanism touching k firing detectors, c_d >= k for each of those k detectors, hence their total initial weight is <= k floor(Q/k) <= Q. If c_d=0, no allowed mechanism can cover d, and the state cannot be completed.

Now visit firing detectors in deterministic increasing index order. Raise each y_d by at most the minimum slack of any incident allowed mechanism. Every constraint remains feasible, by induction, using integer arithmetic only. R205 invokes this stronger bound only when the inexpensive integer lower bound is exactly one away from pruning. A different, independently written implementation checks every newly accepted prune in CDP4/5.

This is an application of established LP-duality/dual-feasibility reasoning, not a claim of inventing dual ascent.

## Lemma C: complete three-fault subtree without a trace bit

Let s be a nonzero residual detector syndrome, and suppose a solution exists using at most three additional mechanisms all of orbit >= o. Choose a firing detector d. Since the xor of chosen mechanism syndromes equals s, the number of chosen mechanisms incident to d is odd and nonzero. At least one allowed mechanism m therefore touches d.

Enumerate every allowed such m. For each one, compute s' = s xor syndrome(m) and logical parity l' = l xor logical(m).

- If s'=0 and l' is nonzero, this is a solution.
- Otherwise invoke the already exact <=2-fault completion procedure on (s',l').
- If all cases fail, no solution using <=3 additional mechanisms exists.

The checker independently chooses its detector, regenerates candidates from its own model representation, computes parity and uses its separately implemented one/two-fault checker. This is **an exact decision** under the model and orbit hypotheses, not a heuristic bound. CDP5 omits all bits for these subtrees, so no producer-provided decision can falsely prune them.

The grouping/orbit preconditions and upper-bound witness handling are inherited from R198. Tests of the circuit exporter, the original full-DEM mapping, and any time-window lemma remain separate obligations. Nothing here promotes a projected-model certificate directly into a full hardware claim.

## Measurement (same compiler, environment, modeled circuits)

Baseline is compiled directly from origin/main cc1a10f4. The R205 prototype was compiled from the isolated worktree; C prover and C checker are separate executables. MSYS2 GCC 16.1.0, Windows, Python 3.14.3, Stim 1.16.0, 6 OpenMP threads. Exported models are the same bytes within each pair.

| Instance | Baseline trace bits | CDP4 trace bits | CDP5 trace bits | Baseline proof bytes | CDP5 proof bytes | Check result |
|---|---:|---:|---:|---:|---:|---|
| IBM-style BB 144-qubit projected model, Z, T=2, w<=9 | 13,511,629 | 13,123,392 | **726,208** | 1,690,546 | **92,370** | VERIFIED |
| Three-colour projected model, X, T=2, w<=8 | 7,305,549 | 7,305,549 | **296,777** | 919,225 | **43,084** | VERIFIED |

The byte reductions are 94.54% and 95.31% respectively (rounded to 0.01%). They are **proof-size** improvements, not physical-qubit, error-rate, or total computation-time improvements. Baseline/CDP4 paired timings used two runs and showed noisy modest prover changes and negligible checker changes. CDP5 has only preliminary timing; a controlled repeated timing campaign is needed before any throughput claim.

Trace validation: IBM model SHA256 5f29c8ddcf2db55fb83fed3a5e59924ddc9684471bed2a5dc548352250ef2fa3, CDP5 proof SHA256 58ce6e482444c1f00ef4251b22441c5de761b145ba9da5e7bb1ceecb5cd9f0fe. Three-colour model SHA256 28875332778f11e52fd6d38756617fe237e94a696f8b9419fe922129220485db, CDP5 proof SHA256 9b51371e52c6c889950d8ce2925192f400642c40d6dd9f35acc0e0c97e2954ca. The provenance is in the corresponding run directories and the evidence registry.

## Verification / negative controls

1. Independently compiled C checker validates all two CDP5 hard-model certificates.
2. A Python fault-subset oracle exhaustively enumerates every subset up to w on 24 seeded synthetic detector models, checking both solver FOUND and NONE behavior against an independent source-level implementation. The oracle does not import prover or checker algorithms.
3. The R198 adversarial certificate tests exercise forged root pruning, wrong weight, bad child counts, truncated and trailing bytes, and a deliberately lying prover. CDP5's legitimate empty terminal traces require adversarial tests to forge nonempty streams instead of assuming every proof must contain interior bits.
4. Old checkers must reject CDP5; current checkers must still accept older valid certificates.
5. A nontrivial speed claim requires paired runs with no other heavy jobs on the machine, equal compiler flags and thread counts, measurement of end-to-end time, and confidence intervals.

## Reproduction

    python experiments/circuit_distance_exact/run_lane.py certify --circuit ibm --basis Z --l 12 --rounds 2 --w 9 --threads 6
    python experiments/circuit_distance_exact/run_lane.py certify --circuit 3c --basis X --l 12 --rounds 2 --w 8 --threads 6
    python -m pytest -q tests/test_r198_circuit_distance_exact.py tests/test_r205_distance_dual.py

The proof is valid only for the exact attached model and declared symmetry. A proof generated for a modified model, another fault parameter, another orbit structure, or a stronger distance claim must be checked afresh.

## Prior art / what remains new to establish

Relevant 2026 external work includes Webster, Jacob and Higgott, *Distance-Finding Algorithms for Quantum Codes and Circuits* (arXiv:2603.22532), which benchmarks exact and heuristic distance algorithms; and existing exact-decoding and certified-distance work. Branch-and-bound, LP-dual lower bounds, exact small-tail enumeration, and proof certificates have substantial prior art. R205's measured contribution is the particular **versioned, independently replayable combination** applied to ONE_Q's circuit-distance search, plus its adversarial testing and proof-size reduction. A publication's novelty statement needs a fuller reproducible prior-art comparison before submission.

## Next algorithmic fronts

- Implement a proof-DAG with typed memo references and an independently checked acyclic dependency order, so repeated residual (syndrome, logical, budget, orbit) states are verified once.
- Compete with SAT/MaxSAT, BZ, cluster and MILP baselines on identical exported models, with externally reproducible time, memory, and proof-size measurements.
- Introduce proof-preserving learned branching only as an untrusted proposer; the checker must independently reconstruct any changed branch coverage.
- Derive independent claims over full circuit DEMs, not merely same-type projections. Test on additional geometries, larger rounds and non-symmetric models, reporting refusals and timeouts.
- Pursue actual scientific improvements in QEC constructions or decoding error rates as a separate research objective. Compressed certificates alone do not constitute such a breakthrough.
