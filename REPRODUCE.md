# Independent reproduction — what a stranger should run

This is the third-party execution package. It exists so that
"independently reproduced" can replace "reported by the authors" in
both manuscripts. Nothing here needs a conversation with us; if a step
fails, that failure is the finding.

**Environment of record:** Python 3.14, numpy 2.3.5, scipy 1.17.0,
ldpc 2.4.1, stim 1.16.0 (see `evidence/environment.json`). Any recent
versions should work; version differences that change results are
themselves worth reporting.

```
pip install -e ".[dev,sat,science]"
```

That is the same line CI runs, plus the two optional-solver extras. The
previous instruction here named five packages by hand and omitted
`pymatching`, `sinter` and `networkx` — so a stranger following it got
an installation whose exact tier raised `ImportError` on first use, and
seven SAT tests that failed rather than skipped. A reproduction
instruction that disagrees with CI is a reproduction instruction nobody
has run.

## 0. Verify you have what we shipped (30 seconds)

```
python tools/verify_release.py          # signature, digests, manifests
```

Checks: the release archive's sha256 matches its signature file, the
signing key is not weak, every per-shot record file matches the hash
recorded in `evidence/per_shot/MANIFEST.json`, the pinned external
checkers match their recorded sha256, and every published weight
vector hashes to its declared value.

## 1. Re-check the certificates you did not produce (minutes)

Both proof systems have a second checker that shares no code with the
producers. Neither needs an LP or MIP solver.

```
python tools/external/exact_lp_certificate_reference.py --self-test
python tools/recheck_all.py             # every stored certificate + tree
```

`recheck_all.py` walks the evidence tree and re-verifies, in exact
rational arithmetic: every stored branch-dual tree (held-out and
development) through BOTH the production checker and the
separately-implemented `tools/external/tree_checker_b.py`, and the
sampled flat certificates it can rebuild. It prints counts and exits
nonzero on any disagreement.

## 2. Re-run the frozen held-out campaign end to end (~2 minutes)

```
git verify-tag held-out-freeze 2>/dev/null; git rev-parse held-out-freeze
git diff held-out-freeze..HEAD -- src/oneq/qldpc_cert.py \
    src/oneq/qldpc_check.py \
    experiments/m0_prior_art/qldpc_heldout_gate.py \
    experiments/m0_prior_art/qldpc_bb_refsched_gate.py
python experiments/m0_prior_art/qldpc_heldout_gate.py
```

The seeds derive from the freeze commit sha at runtime, so a correct
re-run reproduces the same seeds, the same 2,019 records and the same
tier splits.

```
python tools/compare_heldout.py         # compares the SCIENCE, not the clock
```

**Two corrections to what this file used to say, both found by cloning
this repository and following it.**

*It is not byte-exact, and cannot be.* The artifact embeds wall-clock
timings, so a byte comparison fails on any machine whose load differs
from ours — which is every machine. `compare_heldout.py` excludes the
timing keys **by name** (not by pattern, which would silently excuse a
future field) and requires every other field to match. On a clean clone
of HEAD it reports all 8 rungs identical.

*The `git diff` is no longer empty, and the preregistration still
holds.* `qldpc_cert.py` and `qldpc_check.py` were refactored after the
freeze — ~1,350 lines — so the old instruction ("the diff must be empty")
would now tell you the preregistration is void. It is not. The claim was
always about BEHAVIOUR, and the behaviour is proven unchanged
mechanically rather than assumed:

```
python tools/checker_equivalence_gate.py    # verdicts AND reason strings
python tools/producer_equivalence_gate.py   # certificates, bounds, tiers, trees
```

Both compare live code against byte-identical pre-refactor snapshots in
`tools/frozen/`. If you would rather not take that on trust, the
empirical check is the one above: re-run the campaign and compare. It
reproduces.

Known caveat, disclosed: the freeze commitment is *self-referential*
(seeds derive from our own commit sha), not an external public beacon.
See `docs/HELD_OUT_RESULTS.md` finding F4.

## 3. Re-run the whole wall (~10 minutes)

```
python tools/gate_all.py                # all 105 gates incl. provenance, registry,
                                        # both equivalence arbiters and their
                                        # self-proofs, and the overhead gates
python -m pytest tests/ -q              # 4,249 tests, 0 skips expected
python tools/mutation_gate.py           # 500 mutants, all must be killed
```

## 4. Re-run the science (optional, longer)

```
python experiments/m0_prior_art/qldpc_exact_all_gate.py        # canonical rungs
python experiments/m0_prior_art/qldpc_bb_refsched_gate.py      # reference circuit
python experiments/m0_prior_art/qldpc_refsched_fault_xval.py   # 12,816 fault signatures
python experiments/m0_prior_art/qldpc_bnb_residue_gate.py      # branch-dual closures
python experiments/m0_prior_art/qldpc_frame_demo_gate.py       # frame determinacy
```

## What we are asking you to conclude

Only what you verify. The papers' claims are indexed by artifact in
`docs/PAPER2_QLDPC_DRAFT.md` §11 and `docs/PAPER_DRAFT.md` §9. If a
number in a paper does not fall out of a command above, that is a
defect and we want it named — the provenance gate is supposed to make
that impossible, and it has failed before.

## What independence we do and do not claim

- Separate implementation of both checkers: **achieved** (one written
  during adversarial review from the written specification, one from
  the tree schema).
- Separate execution environment: **not achieved** by us — everything
  ran on one workstation. Your run is the first that changes this.
- Independent third-party reproduction: **not achieved** until you do
  it. That is the point of this file.
