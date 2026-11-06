"""oneq.hw -- one pipeline from a protocol's circuits to a certified hardware verdict, shared by every experiment.

The rule that makes it trustworthy: a protocol supplies CIRCUITS and a per-shot SCORE function, and the SAME score
function grades the simulated prediction and the real counts. Prediction and verdict cannot drift apart.

  snapshot   DeviceSnapshot: per-qubit T1/T2, the ASYMMETRIC readout split (p01, p10), sx error, per-edge CZ error,
             durations, the calibration timestamp, and a content hash a record can name. Read-only from any backend.
  noise      the calibrated error model: per-edge CZ error x a MEASURED in-circuit factor (R133: 2.29x the vendor
             number in consistent units, twirled), single-qubit error, Pauli-twirled relaxation over the ALAP
             schedule (a qubit parked in |0> is free), asymmetric readout
  engine     the expected score, EXACT where structure allows: every Pauli error is classified once per circuit as
             REACHABLE (it stays a Pauli to the measurements: a fixed bit-flip pattern, summed in closed form by a
             Walsh-Hadamard characteristic function) or BLOCKED (a later non-Clifford gate makes it state-dependent:
             sampled). A Clifford circuit's noisy expectation is exact with zero variance. Validated against a
             density-matrix oracle (to rounding) and against qiskit-aer's own backend noise model
             (evidence/hw/engine_vs_aer.json, bound to these sources)
  layout     one pinned layout per run, ranked by the engine's own noise model on the live calibration, refusing
             any routed circuit that touches a qubit or edge outside the policy
  stats      certified decisions from counts: weighted Hoeffding for ENUMERATED designs, two-stage Hoeffding for
             SAMPLED ones (the choice is explicit; treating sampled circuits as the whole design under-covers),
             empirical Bernstein, controls, completeness
  freeze     the ISA circuits are the registered object: QPY + hashes + sources; the run loads, never re-transpiles
  submit     the only path to hardware: authorization, preregistration hash, frozen-bytes check, one-run rule,
             quota read first, live health; a conservative QPU-cost model fitted to this repository's recorded jobs

Ported from the hardware-validated experiments/tw_teleport/tw2 (layout, predict, analysis, freeze), whose idle-aware
predictor read 0.6695 against a measured 0.6595 on 43-CZ circuits on ibm_kingston (R138). First client:
experiments/superactivation/shield_family/hw_two_use.py.
"""
