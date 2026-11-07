# Deviations from `docs/PREREG_qldpc_scaling_2026-11-03.md`

The preregistration was committed and pushed (parent of origin/main 70b3c4d8) before any of its syndromes was sampled.
Its runner was then started exactly as committed, one dataset per invocation, in the preregistered order. This file
records every departure from it, when it happened, why, and what it changes. The preregistration itself is not edited.

## Deviation 1 (2026-11-03): the phenomenological datasets use the lazy-separation LP

**What happened.** The first four datasets completed under the committed runner, with these runtimes:

| dataset | seconds |
|---|---|
| `uniform_p01` | 16 |
| `uniform_p02` | 24 |
| `hetero_p02` | 62 |
| `phenom_p01` | 117 |

The fifth, `phenom_p02`, did not finish within the 3,000 s the operator's wrapper allowed. The wrapper killed it, and
the runner wrote no record, because a dataset's records and summary are written only when it completes.

**Diagnosis.** The diagnosis ran the committed runner's own loop on the same seeds with its producer calls wrapped in
timers, and printed timings only, no scores (scratch script `diag_phenom.py`; nothing written in the repository).
The receipt producer of the committed runner (`feldman_dual` with 6 redundant-parity rounds) took 18.6 s, 42.9 s and
206.5 s on three of the first 81 shots of the 360 x 864 spacetime system, against well under a second on the others.
* The residue path (MILP, branch-dual tree) was never reached.
* The cost is the enumeration of facets of redundant-parity cuts. Sec. V of paper 2 names it as the reason the
  circuit dataset uses lazy separation.
* The preregistration fixed `feldman_dual_lazy` for the circuit dataset only.

**The fix.** The phenomenological datasets use `feldman_dual_lazy`, as the circuit dataset already did. Nothing else
changes: seeds, samplers, decoders, the receipt rule, the checkers, the scoring and the analysis are as
preregistered. This is runner line 233 (the LP choice), marked `DEVIATION 1` there.

**What it changes.**
* The FORM of the receipts on a phenomenological dataset run after the change may change: lazy separation without
  redundant-parity cuts can leave more shots to the MILP and a branch-dual tree. Only one phenomenological dataset,
  `phenom_p02`, ran after the change (see "Which datasets"); measured, all 500 of its receipts are flat and none is
  unresolved.
* It does not change the certified optimum of any resolved shot, because the optimum is a property of the syndrome,
  not of the proof. So it changes no score. It could change the number of unresolved shots, which is reported.

**Which datasets.**
* `phenom_p01` had already completed under the committed runner. It is kept as it ran; its summary entry carries no
  per-dataset code hash and ran under the summary-level `code_sha256`, the committed bytes.
* `phenom_p02` and the three datasets after it (`xsector_p02`, `stress_p05`, `bb_refsched`) run under the changed
  runner. The change affects only phenomenological datasets.
* Each of these four summary entries records its own `code_sha256` and a `deviation` field.

**Bookkeeping fix made in the same edit.** The runner also now writes each dataset's own code hashes into its summary
entry, and never overwrites the summary-level hashes. Before, every dataset rewrote them, which would have relabelled
the earlier datasets with the changed runner's bytes.

**Clarification (final referee of paper 2, 2026-11-03).** Exactly one dataset received its receipts from the lazy
LP because of this deviation: `phenom_p02`. `phenom_p01` ran under the committed runner, with the six-round
redundant-parity LP, before the change. `xsector_p02` and `stress_p05` ran under the changed runner, but the change does
not touch their LP, and `bb_refsched` used the lazy LP under the committed runner too. The runtimes in the table above
are the operator wrapper's wall-clock for each invocation; the runner's own `seconds` in the summary are 14.4, 22.4,
59.6 and 115.4.

**Bytes.**

| file | committed (preregistered) | after deviation 1 |
|---|---|---|
| `experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py` | 842b465f085f735e805c9a9e1609b6ca625294d3021ba7ea87d83ce9e5a6d632 | 1da3e1c05777b3e7e12afe870cb0aeceb44979ec190676cb822db9456168ef74 |

The checkers and the producer module are unchanged.

## Note (2026-11-06): additions after the results, not deviations

External audit 3 of release v2026.11.05 asked for the campaign's inputs to be archived and for its costs. Three
things were added after the run. None changes a seed, a sample, a receipt, a score, the analysis or any file the run
wrote: `summary.json` and the eight record files are byte-for-byte as the run left them.

* **An archive of the normative inputs** (`evidence/qldpc_scaling_prereg/inputs/`, written by
  `experiments/m0_prior_art/qldpc_scaling_prereg_export_inputs.py`): what the runner's `build()` fixed for each
  dataset, plus every nontrivial shot's sampled fault (or, at circuit level, sampled observable bit), which the
  records do not carry. The samples were re-drawn from the preregistered seeds and kept only because every re-drawn
  syndrome equals the recorded one. A manifest binds each input to its record file.
* **A producer-free replay** of all receipts from that archive (`tools/replay_qldpc_scaling_campaign.py`).
* **A timed re-run** of the runner on the preregistered seeds, under the bytes each dataset ran under
  (`experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py`), written to `evidence/qldpc_scaling_prereg/costs/`.
  It is a measurement of cost and a reproduction check; its records are compared with the run of record and
  discarded, never substituted for it. The run of record's per-shot production times were not recorded.
