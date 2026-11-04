# Checker input contract synchronization

This review update follows public commit `bc962616512b6b0ce5a07e08fa1e72768846f7a2`.
It changes the declared input and result contracts. It does not claim identical
behavior to that revision on all inputs. Historical measured evidence, receipt
manifests and signed releases retain their original bytes and source pins.

## What callers must change

| Interface | Current requirement or result |
| --- | --- |
| Epsilon checker vertex IDs | Python `int`, excluding booleans, with `-1` as the sole boundary ID. NumPy scalar IDs are refused at the certificate boundary. |
| Model-owned decoder arrays | Convert valid integral arrays at the producing boundary. `CertifyingDecoder.key` already produces Python integer correction endpoints; a regression test verifies that path. |
| Epsilon graph | A simple graph: no loops and no duplicate edge in either orientation. Every correction endpoint and dual-set member belongs to the graph or its boundary. |
| Serialized epsilon values | Finite, nonnegative binary64 values encoded by exact hexadecimal literals. Decimal-looking strings and literals that require rounding are refused. |
| Serialized packing | Repeated set keys are refused before construction of the Python mapping. |
| qLDPC integer fields and `max_nodes` | Python `operator.index` values excluding Python and NumPy booleans; JSON integer literals. Floating, fractional and string indices or budgets are refused. |
| qLDPC witness | Every listed facet is validated before a zero multiplier is skipped. A combined-row list has at least two distinct rows. |
| qLDPC exact candidate result | `checks.status == "EXACT_OPTIMAL"` accompanies the verified candidate weight and exact objective metadata. |
| qLDPC bound-only result | `checks.status == "LOWER_BOUND_PROVEN"`, exact threshold and lattice-rounded `checks.lower_bound`; feasibility is explicitly not established. There is no primal objective, attained optimum, `checks.claim`, or `checks.relation` field. |

The full qLDPC parsing and claim contract is in
[docs/QLDPC_RECEIPT_INPUT_DOMAIN.md](docs/QLDPC_RECEIPT_INPUT_DOMAIN.md).
For a valid zero-multiplier facet, flat-checker diagnostics now count that
validated facet in `facets_real`; the zero contributes nothing to the bound.

## Current equivalence and historical changes

Current checker equivalence uses a pinned versioned contract snapshot and
requires identical current verdicts, reasons and numerical results on every
exercised case. This finite test does not prove agreement for all inputs. Historical
transition allowances have been removed from that gate. Previous snapshots,
including both previous public entry points, are preserved byte-for-byte under
`tools/frozen/historical/`. Separate bounded regression tests demonstrate the
deliberate input and result changes. They do not requalify historical campaigns.

The [current freeze record](docs/CHECKER_CONTRACT_FREEZE_2026-11-04.md) explains
the preserved `51940b...` snapshot, the strict node-budget boundary, and the
new deterministic comparison cases. Initial source equality between a newly
versioned snapshot and production is intentional. That snapshot is a regression
reference, not an independent proof or an outside human review.
The comparator includes every declared result field and nested metadata, with
exact binary64 bit patterns and rational values. Earlier reports used a weaker
comparator; they are preserved and do not acquire this stronger claim.

Synthetic qLDPC and epsilon vectors exercise the shared parsing contract and
the acceptance/refusal rules across the implementations. The epsilon fuzzer
exits nonzero when any input class disagrees, including malformed input classes.

## Changing a frozen baseline

An unexpected pin or equivalence failure requires investigation; do not copy
live source over a frozen snapshot just to make the comparison pass. For an
intentional contract correction, preserve the previous snapshot byte-for-byte,
add explicit changed-input controls and unchanged valid controls, document the
API change, and review the proposed replacement snapshot and pin together.
Run the current comparison without transition allowances after that change.
Historical measured results retain their original checker and corpus hashes.

## Run the focused checks

Use Python 3.11+ with the existing scientific dependencies and Node 24. The
JavaScript qLDPC readers require `JSON.parse` source-text access so that `1`
and `1.0` are distinguished before interpretation; an unsupported engine is
refused rather than silently weakening the parser.

```bash
PYTHONPATH=src python -m pytest tests/test_checker_contract_migration.py tests/test_checker_equivalence_exact_result.py tests/test_checker_integer_contract.py tests/test_matching_endpoint_contract.py tests/test_qldpc_input_domain.py tests/test_epsilon_vectors.py tests/test_epsilon_fuzzer_gate.py -q
PYTHONPATH=src python tools/checker_equivalence_gate.py --cases 400
PYTHONPATH=src python tools/fuzz_epsilon_checkers.py --cases 40 --seed 20261103
PYTHONPATH=src python tools/replay_receipts.py
```

The public receipt-manifest pin remains unchanged. Replays under this revision
must be reported separately with the current checker hashes; they do not replace
the historical source-qualified results. No private campaign data or manuscript
revision is part of this contract update.

## Qualification of this update

The [current-checker qualification record](evidence/checker_contract_reverification_20261104.json)
binds the tested source files and retained artifacts. On Windows with Python
3.14, the complete public suite passed **809 tests with no skips**. The current
comparison passed 6,702 exact full-result comparisons with no divergence or
reason alias, and the sensitivity run detected all 24 behavioral mutations.
Both Python implementations replayed all 2,042 manifested receipts. The
JavaScript readers accepted those receipts and refused all 210 planted defects.
The epsilon differential run agreed on 18,007 specimens, including 2,963
valid-domain specimens. These are finite tests, not a universal soundness proof.

The original corrected-gap replay deliberately refuses the changed epsilon
source. To check the 1,047 retained witnesses under the identified current
checkers while preserving their historical measurement and timing scope, run:

```bash
PYTHONPATH=src python tools/recheck_corrected_gaps_current.py evidence/corrected_gap_20261103/summary.json
```

The record lists the measured and current source hashes separately and confirms
that the historical summary and trace are unchanged. It does not call this a
new benchmark measurement or a requalification of the old scale campaigns.
