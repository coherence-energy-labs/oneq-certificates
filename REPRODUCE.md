# Independent reproduction — what a stranger should run

This is the third-party execution package for the three manuscripts of this bundle (papers 1-3; paper 4's is in
`superactivation/`). It exists so that "independently reproduced" can replace "reported by the authors". Nothing
here needs a conversation with us; if a step fails, that failure is the finding.

**Environment of record:** Python 3.14, numpy 2.3.5, scipy 1.17.0, ldpc 2.4.1, stim 1.16.0 (see
`evidence/environment.json`). Any recent versions should work; version differences that change results are
themselves worth reporting.

```
python -m pip install -r requirements.txt
```

`requirements.txt` pins the environment of record and ships in the release bundle, which carries no
`pyproject.toml`: the line this replaced, `pip install -e ".[dev,qpu,sat,science]"`, could not run from the bundle
(2026-11-03). In the development repository the editable install still works and is what CI runs. The Qiskit SDK is
needed for offline frozen-circuit and fake-backend tests; installing it does not authorize or execute a provider
job. An earlier instruction here named five packages by hand and omitted `pymatching`, `sinter` and `networkx`, so a
stranger following it got an installation whose exact tier raised `ImportError` on first use: a reproduction
instruction that disagrees with CI is one nobody has run.

Post-quantum receipt tests require both the declared `liboqs-python` wrapper and the native liboqs shared library
with ML-DSA-65 enabled. Follow the
[upstream installation instructions](https://github.com/open-quantum-safe/liboqs-python#installation) for the native
library; the wrapper can build it automatically when absent. Check it before collecting tests:
`python -c "import oqs; assert 'ML-DSA-65' in oqs.get_enabled_sig_mechanisms()"`. An environment missing this
backend can silently omit the receipt-sealing test modules during collection, so a smaller green suite is
insufficient evidence.

Some of this project's work (the magic-state pipeline, the hardware lanes, the full wall of gates) is not part of
this release; it lives in the project's development repository, which is private.

## 0. Verify you have what we shipped (30 seconds)

Put the bundle's three release files (`oneq-certifying-decoder.tar.gz`, its `.sig.json` and
`.statement.signed.json`) and their `.ots` in `release/`, then:

```
python tools/verify_release.py --bundle certifying-decoder   # signature, statement, bound evidence, timestamps
```

Checks: the release archive's sha256 matches its signature file, the signing key is not weak, the signed statement
covers every shipped file, every per-shot record file matches the hash recorded in `evidence/per_shot/MANIFEST.json`,
the pinned external checkers match their recorded sha256, every published weight vector hashes to its declared value,
and each timestamp proof covers the bytes it names.

## 1. Re-check the certificates you did not produce (minutes)

Papers 1 and 2 each have a second, separately written checker that shares no code with the producers; paper 3 has
one checker, whose calculus is proven sound in Lean (`lean/CertR198/`). None needs an LP or MIP solver.

```
python tools/external/exact_lp_certificate_reference.py --self-test
python tools/recheck_all.py                     # paper 2: the complete pinned replay
python tools/replay_qldpc_scaling_campaign.py   # paper 2: the preregistered scaling campaign, from archived inputs
node verifier-js/verify-certificate.mjs --conformance conformance/cert_vectors.json
```

`recheck_all.py` replays every one of the 2,042 receipts named by the pinned `evidence/receipt_manifest.json` (2,002
flat certificates and 40 branch-dual trees), each through two separately written checkers, against the persisted
instances and integer weight vectors, and fails if any receipt or bound file is missing, duplicated, unlisted or
altered. `replay_qldpc_scaling_campaign.py` re-checks the 6,500 receipts of the preregistered campaign with both of
their checkers from `evidence/qldpc_scaling_prereg/inputs/`, loading no producer.

For paper 1, obtain the receipt archives (release assets, one `<config>.tar` per configuration; each is an
uncompressed tar of gzip chunks, and `evidence/epsilon_rerun/<config>/archive.json` pins its sha256, size and
members), place them in `data/epsilon_receipts_archive/`, and from the bundle's root run

```
python tools/pack_epsilon_receipts.py --verify       # archive hashes, then every member
tar -xf data/epsilon_receipts_archive/<config>.tar   # once per configuration
python tools/replay_epsilon_receipts.py --jobs 8     # two verifiers and the forgery battery, every receipt unpacked
# the held-out d=7 arms were stopped by their preregistered budget and have no summary; replay what they completed:
python tools/replay_epsilon_receipts.py --partial --jobs 8 --config heldout_d7_X_r50 heldout_d7_X_r10 samedev_d7_X_r30 heldout_d7_Z_r50 heldout_d7_Z_r10 samedev_d7_Z_r30
```

For paper 3, the proofs that ship in the tree re-check with `python -m pytest tests/test_r198_ibm_recert.py
tests/test_r198_paper3_review.py` (a C compiler is needed). Its 30 largest proofs are release assets
(`oneq-recert-proofs-3c.tar.gz`, `oneq-recert-proofs-morph144.tar.gz`); with both in a directory DIR:

```
python experiments/circuit_distance_exact/recert_assets.py verify --archives DIR --extract
python experiments/circuit_distance_exact/recert_assets.py check --threads 8     # the committed checker; hours
```

## 2. Re-run paper 2's frozen held-out campaign end to end (hours of solver time)

The campaign was frozen at a commit of the development repository; that commit ships as the source snapshot
`oneq-source-aba2aec.tar` (a release asset, pinned in `evidence/revision_snapshots.json`). From the public release
alone:

```
python tools/revision_snapshots.py --verify --no-git --out <dir-with-the-snapshots>
python tools/reproduce_heldout_public.py --work <a fresh directory>
```

It extracts the frozen snapshot into the fresh directory, supplies the freeze identity
(`evidence/freeze_identity.json`) without editing a byte of the frozen code, re-runs the campaign there, and compares
every non-timing number with the shipped `evidence/qldpc_heldout.json`, rung by rung, and every regenerated per-shot
record with the shipped one by sha256.

**Two things this file used to get wrong, both found by cloning the repository and following it.** *It is not
byte-exact, and cannot be:* the artifact embeds wall-clock timings, so the comparison excludes the timing keys by name
(not by pattern, which would silently excuse a future field) and requires every other field to match. *The
preregistration still holds although the checker was refactored after the freeze:* the claim was always about
behaviour, and `tools/checker_equivalence_gate.py` (verdicts and reason strings) and
`tools/producer_equivalence_gate.py` (certificates, bounds, tiers, trees) compare live code against byte-identical
pre-refactor snapshots in `tools/frozen/`. The empirical check is the re-run above.

Known caveat, disclosed: the freeze commitment is *self-referential* (seeds derive from our own commit sha), not an
external public beacon, and the freeze commit lives in the private development repository; see
`docs/HELD_OUT_RESULTS.md` finding F4 and `AUDIT.md`, attack 5.

## 3. Re-run the whole wall (development repository only)

`tools/gate_all.py`, the full test suite and `tools/mutation_gate.py` ship, but the wall they make reads the
development repository's history and files the bundle does not carry, so it runs only there. Nothing in a verdict
depends on it.

## 4. Re-run the science (optional, longer)

```
python experiments/m0_prior_art/qldpc_exact_all_gate.py        # canonical rungs
python experiments/m0_prior_art/qldpc_bb_refsched_gate.py      # reference circuit
python experiments/m0_prior_art/qldpc_refsched_fault_xval.py   # fault signatures
python experiments/m0_prior_art/qldpc_bnb_residue_gate.py      # branch-dual closures
python experiments/m0_prior_art/qldpc_frame_demo_gate.py       # frame determinacy
```

## What we are asking you to conclude

Only what you verify. Every claim of papers 1-3 is indexed by artifact and command in `docs/REVIEW_PACKET.md`. If a
number in a paper does not fall out of a command there or above, that is a defect and we want it named: the
provenance gate is supposed to make that impossible, and it has failed before.

## What independence we do and do not claim

- Separately written checkers: **achieved** (a reference flat checker written during adversarial review from the
  written specification, a tree checker from the schema, and a JavaScript ε-checker written by a separate agent from
  paper 1 and a logged brief). They are separately written, not independent: one project wrote them.
- Separate execution environment: **not achieved** by us; everything ran on one workstation. Your run is the first
  that changes this.
- Independent third-party reproduction: **not achieved** until you do it. That is the point of this file.
