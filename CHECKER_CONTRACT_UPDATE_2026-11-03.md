# Checker contract and portable model serialization update

> Historical update note: the subsequent [checker contract synchronization](CHECKER_CONTRACT_SYNC_2026-11-03.md)
> supersedes the endpoint, bound-only metadata and current-equivalence descriptions below.
> The original note is retained to describe the earlier review revision.

Base: public `main` commit `451d773bbb58325a3d6eeed361a2313f2673b7c8`.
This update repairs the November 3 audit's input and return-value findings.
It does not rewrite the released manuscripts, historical receipts, frozen
checker sources, release manifests, or their pinned hashes.

## Strict integer fields

Production check supports, weight-map indices, candidate supports, facet
supports/row identifiers, branch identifiers, infeasible-row identifiers,
node budgets, and syndrome bits reject lossy coercion. Actual integer
values are required. The Python API accepts `numbers.Integral`, including
NumPy integer scalars; booleans, NumPy booleans, strings, floats (including
`1.0`), and `Fraction` objects used as indices are refused. A syndrome entry
must additionally equal 0 or 1. Set-valued supports reject duplicate indices.

The independently implemented JSON tree checker has its own plain-integer
validation and now applies it to every consumed index and rational field.
It no longer masks syndrome entries modulo two. JSON booleans and floats
are refused. The independent flat reference already enforced this domain
and was left unchanged.

Compatibility: callers that explicitly supplied Python boolean syndrome
vectors must convert actual boolean bits to integers before constructing
the certificate instance. That conversion belongs at a documented data
adapter, not inside the trust boundary. Valid retained JSON receipts
already contain integer fields. Historical evidence is not migrated.

## Bound-only output is a different claim

`check_qldpc_bnb(bound_only_threshold=T)` still returns the same acceptance
decision for valid proofs. Acceptance states `LOWER_BOUND_PROVEN`: every
feasible solution has weight strictly greater than T. This statement is
also true when the instance is infeasible. It establishes neither existence
of a feasible candidate nor an attained optimum.

Accepted threshold results expose exact `checks.threshold`,
`checks.relation == ">"`, and `checks.claim == "lower_bound"`.
`primal_weight`, `dual_bound`, and `gap` remain `None`, and `checks.objective`
is absent. Refused threshold proofs also have no purported primal weight.
Candidate mode retains its `EXACT-OPTIMAL` reason, verified candidate weight,
and exact objective metadata. Existing frame/distance consumers inspect
`accepted`; they require no behavioral change.

## Historical differential checking

`tools/frozen/` and its SHA-256 manifest are unchanged. The differential
gate still demands identical historical outputs for unchanged-contract
inputs. Three narrowly checked transitions are counted separately:
duplicate candidate refusals, specified malformed-input refusals, and
threshold-only payloads. A transition cannot permit a new acceptance of
malformed input, hide a changed bound-only verdict, or retain an invented
objective. Regression tests mutate these transition checks themselves.
The structured differential report is version 2 and no longer claims
blanket byte-for-byte result equivalence across the API correction.

## Matching receipt endpoint contract

`matching_cert_epsilon.check_epsilon` now requires each correction edge to
contain exactly two integral, nonboolean endpoint values before canonicalizing
it. Fractional floats, integral-valued floats, strings, booleans and malformed
edge shapes return `INVALID_INPUT`; they are never silently truncated into a
different edge. Python and NumPy integer endpoints retain exact semantics.
Fourteen focused regression cases cover the malformed values and shapes and
the two supported integer representations.

The measured corrected-gap packet predates this endpoint-only hardening and
pins the previous checker bytes. Its source-pinned replay must still refuse
an edited checkout. A separately named compatibility command rechecks every
retained witness under the current checker and reports every source difference:

```bash
python tools/recheck_corrected_gaps_current.py evidence/corrected_gap_20261103/summary.json
```

The historical summary, source hashes, timings and original replay result stay
unchanged. The new result is a current-checker replay of this finite packet,
not a new measurement or qualification of the earlier large Willow or mutation
campaigns. Those campaigns retain their historical source pins until their
full qualification runs are repeated.

## Exact byte format for circuit models

Both circuit model exporters now specify `encoding="ascii"` and
`newline="\r\n"`. This matches the existing released CRLF model files on
both Linux and Windows, preserving their historical SHA-256 identities.
No hash check is removed or relaxed. A future change to canonical model
bytes should use an explicit format/release version.

## Reproduction

Use the release's installed scientific dependencies and Python 3.11+:

```bash
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python tools/replay_receipts.py
PYTHONPATH=src python tools/checker_equivalence_gate.py --cases 400
PYTHONPATH=src python tools/checker_equivalence_gate.py --cases 400 --prove-sensitive
PYTHONPATH=src python tools/qldpc_js_agreement.py --no-write
PYTHONPATH=src python tools/qldpc_tree_js_agreement.py --no-write
```

These commands do not restamp historical evidence. The complete manifested
receipt replay contains 2,002 held-out flat proofs and 40 tree proofs across
the held-out and development cohorts; it invokes both independent Python
implementations. Timing observations are environment-specific.
