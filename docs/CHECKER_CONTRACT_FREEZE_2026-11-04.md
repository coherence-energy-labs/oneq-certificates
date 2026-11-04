# Versioned checker contract snapshot: 2026-11-04

The merged production qLDPC checker remains byte-identical to SHA-256
`c923c984c60a0888233cd00ccfea40995877cace05c572744f24dbf25a50b273`.
This update makes that already reviewed contract the current frozen snapshot.
The initial source equality is intentional: the snapshot records the behavior
that subsequent edits must preserve. It is not an independent implementation
or a mathematical proof of the checker.

## Why another snapshot was necessary

The previous `51940b3712a830895cbffa3516c81ba57836d470dbe52085219e581709c15ecc`
snapshot includes the input-domain and zero-multiplier-facet corrections, but
still converts `max_nodes` with `int()`. On a valid one-node proof it accepts
`True`, `1.0`, `1.9`, `"1"` and `Fraction(1)` after coercion. The merged
checker explicitly refuses those values. Ordinary integer budgets retain
their precise node limits; NumPy integer scalars remain valid qLDPC budgets.

The merged floating-screen wrapper also turns malformed-input `TypeError`,
`ValueError` and `OverflowError` exceptions into explicit refusals. Named
controls exercise malformed tolerance values that previously escaped.

The preceding 400-case corpus did not vary `max_nodes`, so its 6,680 agreeing
comparisons did not establish agreement on this input boundary. The current
gate adds 22 deterministic contract comparisons: 16 node-budget cases and
six screen-refusal cases, including accepted integer controls and exact budget
exhaustion. Historical transition tests demonstrate both changes. Sensitivity
mutations explicitly reintroduce budget coercion and escaped type errors.

## Preserved history and current checks

The previous snapshot is preserved byte-for-byte at
`tools/frozen/historical/qldpc_check_pre_merge_contract_20261104.py`.
Earlier `2580ac...` and `597737...` snapshots remain intact. Historical replay
records, campaign evidence, manifests and source pins are not rewritten.
The pre-input-contract `2580ac...` snapshot also predates the explicit
`checks.status == "EXACT_OPTIMAL"` field on accepted exact flat/tree results.
The historical controls pin that one added field while preserving every other
field of their valid-control results; current comparison has no such exception.

Current equivalence has no transition allowlist or reason aliases. Every
exercised case must have an identical complete result. The comparator now
preserves every dataclass field, including gap and nested diagnostic metadata,
and compares binary64 values by exact bit pattern. The preceding comparator
rounded two objective fields to nine decimal places and omitted other fields;
its earlier reports do not support this stronger result-comparison claim.
Controlled tests change a single float ULP, an exact fraction, a nested claim
or result field, and require the new comparator to detect the change. An
unsupported comparison value now fails loudly instead of looking like two
matching checker exceptions. This is finite
regression evidence; it does not prove equality for all possible inputs.
Historical changed-input comparisons are reported separately and cannot
turn a current disagreement into a pass.

```bash
PYTHONPATH=src python -m pytest tests/test_checker_contract_migration.py tests/test_checker_integer_contract.py tests/test_checker_equivalence_exact_result.py -q
PYTHONPATH=src python tools/checker_equivalence_gate.py --cases 400
PYTHONPATH=src python tools/checker_equivalence_gate.py --cases 400 --prove-sensitive
```
