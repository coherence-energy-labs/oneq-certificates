# Circuit-distance proof availability and replay

The two archived receipt manifests name **304 proof/model pairs**: 22 IBM
receipts and 282 other circuit-distance receipts. This checkout supplies 274
proof blobs. The other 30 blobs, totaling **732,193,669 bytes**, are not tracked
in Git or included in the published release.

The completed November 4 replay report is recorded at review revision
`15a0c5fd2fc991d7a96583d90acd5ff1c9377fde`, after the historical
`v2026.11.03` release. Use this review checkout for the commands below.

**Recovery and replay update, November 4, 2026.** All 30 omitted payloads have been recovered
and matched to their exact manifest hashes. The
[recovery record](evidence/final_integration_validation_20261104/distance_complete_replay_20261104/historical_sources/executed_history/copy_manifest.json)
and [archive record](evidence/final_integration_validation_20261104/distance_complete_replay_20261104/historical_sources/executed_history/proof_archive.json)
bind those bytes and a separate delivery ZIP. The
[completed full-catalog replay](evidence/final_integration_validation_20261104/distance_complete_replay_20261104/run/distance_receipt_replay_20261104_recovered.json)
verified all 304 unique receipts with checker exit status 0: 62 global
finite-model bounds and 242 prefix irreducible exclusions. Its
[qualification binding](evidence/final_integration_validation_20261104/distance_complete_replay_20261104/run/distance_receipt_replay_20261104_recovered.binding.json)
records the exact sources and inputs and their stability throughout the run.
This project-run replay does not establish outside human reproduction or
replay the window and fold arguments.

The dated November 3 missing-proof inventory, including every expected hash, size,
model, weight and prefix, is
[`missing_proofs_20261103.json`](evidence/circuit_distance_exact/missing_proofs_20261103.json).
It is an availability snapshot, not a new mathematical verdict.

| Receipt family | Manifested | Proofs tracked in Git | Proofs not tracked in Git | Untracked proof bytes |
| --- | ---: | ---: | ---: | ---: |
| IBM recertification | 22 | 22 | 0 | 0 |
| morph72 | 14 | 14 | 0 | 0 |
| morph2026 | 238 | 238 | 0 | 0 |
| 3c prefix | 24 | 0 | 24 | 540,004,497 |
| 3c full T=2 | 2 | 0 | 2 | 44,202,298 |
| morph144 | 4 | 0 | 4 | 147,986,874 |
| **Total** | **304** | **274** | **30** | **732,193,669** |

## Replay the proofs shipped in Git

From the repository checkout, without a received-proof directory:

```sh
python -I -S tools/distance_receipt_inventory.py --output output/distance_receipt_replay.json
```

This requires Python 3.10 or newer and GCC or Clang. It imports only the Python
standard library, compiles the reviewed C checker, hashes every model/proof
pair, and runs the actual checker for each available proof. OpenMP is used when
the compiler supports it, with four threads by default; compilation falls back
to a serial checker when OpenMP is unavailable. `--threads 8` selects eight
threads. Compilation and per-receipt check times are recorded separately.

Without `--received`, the expected result is **274 fresh checks, 30 missing
proofs, exit status 2**. It must not report 304 verified receipts. The report keeps an entry
for every manifested cell, including missing, refused and timed-out cells.
There is no option that turns a partial replay into a complete success.

For a quick availability and hash inventory without compiling or running C:

```sh
python -I -S tools/distance_receipt_inventory.py --inventory-only --output output/distance_receipt_inventory.json
```

That mode labels available evidence `PRESENT_NOT_REPLAYED` and reports zero
freshly verified receipts. It cannot produce a mathematical success status.

| Exit status | Meaning |
| ---: | --- |
| 0 | All 304 manifested receipts passed fresh mathematical replay. |
| 1 | A binding, input, compilation or mathematical check failed, or a check timed out. |
| 2 | Evidence is missing; the receipt set is incomplete. |
| 3 | All files are available, but only an inventory was requested. |

## Replay a received proof set

The recovered proofs are packaged as `oneq-distance-proofs-20261104.zip`
(370,620,735 bytes), with SHA-256
`3d08b51b1325777d987f6a7c5ac82ed29d0c2d5ba547658bf4cfe43ad2579f52`.
**This guide does not supply a public download URL for that ZIP.**
Obtain it from the author before attempting the complete replay. The linked
archive record identifies the bytes; its local `.build/` path is not a download.

Keep the repository checkout alongside the ZIP: the archive contains the
30 recovered proofs and their inventory/README, while the checkout supplies
the other 274 proofs, model files, receipt manifests, CLI and C checker.
After verifying the ZIP hash, extract it and run:

```sh
python -m zipfile -e oneq-distance-proofs-20261104.zip received-proofs
python -I -S tools/distance_receipt_inventory.py --received received-proofs/oneq-distance-proofs-20261104/proofs --timeout-seconds 14400 --threads 8 --output output/distance_receipt_replay.json
```

For separately supplied raw `.bin` files, point `--received` at their directory.

`--received` can be supplied more than once. Files are matched by SHA-256,
regardless of their filenames. The expected hashes are the manifests'
`proof_sha256` fields; some older campaign proofs have different hashes in
`recorded_proof_sha256`, which are not interchangeable with the recertified
receipts. Supplied files whose hashes do not occur in the catalog are listed
separately. A damaged shipped file is reported as damaged, even when a good
received copy exists, so receipt corruption cannot disappear from the report.

The default timeout is 600 seconds **per proof**. The larger recovered proofs can
take substantially longer; the example above allows four hours per check.
Timeouts remain explicit unverified failures. This command does not search for
proofs outside the requested directories, regenerate proofs, or contact anyone.
The earlier campaign records include long prover and checker runs. The
completed November 4 project replay is linked above; this command lets an
outside reviewer perform a separate replay of the same manifested inputs.

## What the checked statements actually say

An unprefixed proof establishes a lower bound for the **specific finite model
file** supplied to the checker. For example, excluding weights up to 5 gives
a global distance lower bound of 6 for that model. It does not by itself cover
other numbers of rounds, an unprojected hardware model, or a device.

A prefix proof establishes only an **irreducible logical-error exclusion** for
its stated earliest-detector-round prefix. The checker explicitly makes this
distinction. A reducible error can contain an early undetectable component and
a later logical component; a prefix exclusion does not exclude every such
error. Therefore prefix receipts are reported as
`PREFIX_IRREDUCIBLE_EXCLUSION`, with `irreducible_only: true` and
`global_distance_lower_bound: null`.

Of the 274 shipped receipts, **218 are prefix exclusions and 56 are global
finite-model bounds**. The 22 IBM receipts are all in the prefix category.
Any promotion to a global or every-round-count statement also needs the
appropriate window theorem and its model-generation preconditions. This
command does not replay that theorem and always reports
`prefix_to_global_window_theorems_replayed: false` and
`every_round_count_proven: false`. It makes no hardware-adequacy claim.

## Trust and portability

The CLI pins both full receipt manifests, so deleting missing rows or changing
a stated scope cannot redefine what counts as a complete receipt set. It also
pins the current reviewed C source to SHA-256
`263a54bfa2545cdfcf6ebff4973d6cb7c4ac10d6f001eb3794157ba98e877504`.
It compiles a snapshot of those checked source bytes, and the checker receives
snapshots of the exact model/proof bytes whose hashes were validated.

Source and JSON-manifest text pins normalize CRLF to LF to support Windows
checkouts. **Model and proof hashes never normalize bytes.** The circuit
exporters explicitly preserve the archived ASCII/CRLF model serialization.
Historical checker/source hashes and archived manifests remain unchanged;
the output identifies the current checker used for this replay.

The trusted mathematical component remains the reviewed C checker, plus the
compiler and execution environment. The CLI does not independently prove the
checker implementation, attest runtime measurements, or establish authorship
of a report. Independent external human reproduction remains unestablished
until someone actually performs and records it.

## Focused regression coverage

The regression tests exercise real global and IBM-prefix proofs, a proof
mutation whose new hash is supplied to ensure the C checker itself refuses it,
model/proof mismatches, missing evidence, a shortened manifest, CRLF manifest
portability, received-proof matching, timeout accounting, and the reviewed
source pin:

```sh
python -m pytest tests/test_distance_receipt_inventory.py -q
```
