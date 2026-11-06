# Preregistration: BP-OSD at min-sum scaling 1.0 and 0.625, on fresh syndromes, scored by exact receipts (2026-11-03)

**Why.** Paper 2's preregistered held-out campaign decoded every syndrome with ldpc 2.4.1's `BpOsdDecoder` at its
class-default min-sum scaling factor, 1.0. The receipts then showed BP-OSD missing the certified optimum on 178 of the
300 stress syndromes. Two external audits (`docs/reviews/EXTERNAL1_2026-11-03.md`, `EXTERNAL2_2026-11-03.md`) and
their adjudication asked whether that measures BP-OSD or a configuration choice.

The re-decodes so far are post hoc, run on syndromes whose results were known:
* the strong-baseline arms A_s and B_s;
* `qldpc_heldout_scaled_posthoc.py`;
* `qldpc_heldout_scaled_posthoc_all8.py` → `evidence/qldpc_heldout_scaled_posthoc_all8.json`.

The last of these found a scaling factor of 0.625 raising optimum hits on the stress, uniform p=0.02, X-sector and
circuit datasets. It found 0.625 lowering them on the heterogeneous dataset and both phenomenological datasets, and it
found shots where each setting hits and the other misses.

This file fixes a confirmatory comparison of the two settings on an equal footing **before any of its syndromes is
sampled**. It is committed with its runner, and the commit's timestamp is the record.

**The question, fixed.** On each of the eight dataset families of paper 2, how often does BP-OSD return a certified
optimum at `ms_scaling_factor` 1.0, and at 0.625? Do the two settings differ on the same syndromes?

## Datasets

The models are those of the held-out campaign (`experiments/m0_prior_art/qldpc_heldout_gate.py`). The check structure
is taken from the persisted, hash-bound instances.

| dataset | check matrix (rows x columns) | fault model | decoder prior | objective (integer weights) | nontrivial shots |
|---|---|---|---|---|---|
| `uniform_p01` | H_Z, 72 x 144 (Z checks; X errors) | i.i.d. X, p = 0.01 | `error_rate=0.01` | unit | 1,000 |
| `uniform_p02` | H_Z, 72 x 144 | i.i.d. X, p = 0.02 | `error_rate=0.02` | unit | 1,000 |
| `hetero_p02` | H_Z, 72 x 144 | independent X, p_i = 0.02 U(0.5, 2), drawn from this file's channel seed | `error_channel=p_i` | floor(1000 log((1-p_i)/p_i) + 1/2) | 1,000 |
| `phenom_p01` | spacetime H_Z, 4 noisy rounds + 1 perfect: 360 x 864 (576 data, 288 measurement) | i.i.d. p = p_m = 0.01 | `error_channel=[0.01]*864` | floor(1000 log((1-p)/p) + 1/2), uniform | 500 |
| `phenom_p02` | as above | p = p_m = 0.02 | `error_channel=[0.02]*864` | as above, p = 0.02 | 500 |
| `xsector_p02` | H_X, 72 x 144 (X checks; Z errors) | i.i.d. Z, p = 0.02 | `error_rate=0.02` | unit | 1,000 |
| `stress_p05` | H_Z, 72 x 144 | i.i.d. X, p = 0.05 | `error_rate=0.05` | unit | 1,000 |
| `bb_refsched` | DEM of the published depth-8 cycle, memory-Z, 3 rounds, p = 0.002 (`build_refsched_circuit(3, 0.002)`, `decompose_errors=False`): 288 x 2,304, one declared observable | its circuit-level noise family: no idle faults; the mechanism set equals paper 3's IBM model's (`evidence/qldpc_refsched_p3_equality.json`) | `error_channel` = DEM priors | floor(1000 llr + 1/2) | 500 |

Notes on the table:
* Persisted instances: `evidence/instances/bb144_{z,x}_code_capacity.json`, `bb144_phenom_r4.json` and
  `bb144_refsched_r3_p002.json`.
* The code-capacity faults are sampled column by column with NumPy's `default_rng(seed).random(n) < p`, exactly as in
  the campaign. The circuit dataset is sampled with Stim's `compile_sampler(seed)`, in batches of 1,000.
* **A shot is nontrivial when its syndrome is not zero.** Sampling continues until the stated number of nontrivial
  shots is reached, for every dataset, the circuit one included. Trivial shots are counted, not used. Nothing is
  excluded and nothing is re-run.

## Seeds (never used before)

`seed(d) = int(SHA256("oneq-qldpc-scaling-prereg-2026-11-03||" + d)[:16], 16) & (2^63 - 1)`, reduced modulo
`2^31 - 1` for `bb_refsched`. The domain string is new: no seed below occurs in any tracked file of this repository at
the time of writing (`git grep` for each value, 0 hits).

| name | seed |
|---|---|
| `uniform_p01` | 7388119315214903760 |
| `uniform_p02` | 2358906921348550124 |
| `hetero_p02` (shots) | 2499531330981900147 |
| `hetero_p02_channel` (the channel p_i) | 2661926806025368101 |
| `phenom_p01` | 5791936648250179652 |
| `phenom_p02` | 2997173273735429858 |
| `xsector_p02` | 339979895904008218 |
| `stress_p05` | 9135337716753873663 |
| `bb_refsched` | 458719843 |

The runner refuses to run unless it derives exactly these values, and every one of them appears in this file.
* Dry runs (`--smoke`) use a different domain, `oneq-qldpc-scaling-SMOKE-never-the-preregistered-seeds`, and refuse
  to write under `evidence/`.
* The runner was exercised only that way, on 4 shots per dataset and 40 stress shots.
* None of the preregistered syndromes has been sampled.

## The two configurations, on an equal footing

Both are ldpc 2.4.1 `BpOsdDecoder` with `bp_method="ms"`, `schedule="parallel"`, `osd_method="osd_cs"`, `osd_order=5`
and `max_iter=30` (40 for `bb_refsched`), with the decoder prior above. They differ in one parameter only:

* **`ms_scaling_factor=1.0`**, the class default of `BpOsdDecoder` (`ldpc/bposd_decoder/_bposd_decoder.pyx:53`,
  forwarded at `ldpc/bp_decoder/_bp_decoder.pyx:94`);
* **`ms_scaling_factor=0.625`**, the default of ldpc's own `SinterBpOsdDecoder`
  (`ldpc/sinter_decoders/sinter_bposd_decoder.py:41`; that file's docstring says 1.0, its code says 0.625).

Neither value is tuned on any data here. Every syndrome is decoded by both configurations.

## Receipts: one per syndrome, independent of both answers

* **The producer.**
  * The LP is `feldman_dual` with 6 redundant-parity rounds (`feldman_dual_lazy` at circuit level), rationalised
    once.
  * The first of (the 1.0 answer, the 0.625 answer, the LP primal) that `check_qldpc_exact` accepts with that single
    dual fixes the optimum U.
  * Otherwise the MILP proposes a candidate, `certified_bnb` builds a branch-dual tree, and `check_qldpc_bnb` accepts
    it.
* **The second checker.** Every receipt is also decided by the second checker: the reference flat checker
  (`tools/external/exact_lp_certificate_reference.py`) or tree checker B (`tools/external/tree_checker_b.py`).
* **Unresolved shots.** A shot with no receipt, or with a receipt either checker refuses, is reported as unresolved,
  per dataset, and is in neither configuration's hit count.
* **What is stored.** Every receipt is kept in the per-shot records.
* **The input domain.** All checkers enforce the input-domain contract `docs/QLDPC_RECEIPT_INPUT_DOMAIN.md`.

## The analysis (fixed now; reported whatever it shows)

**Per dataset, separately, on the resolved shots:**
1. **Optimum hits per configuration.** A hit is a syndrome-valid answer whose integer objective equals the certified
   optimum. Each hit is split into:
   * accepted by the shot's single dual (tier 1, Sec. III's rule: attaining or lattice);
   * at the optimum on a branching tree.

   Misses are split into heavier answers and answers that are not corrections.
2. **The primary comparison.** The exact two-sided McNemar test of optimum hit, on the same syndromes: hits at 1.0
   only against hits at 0.625 only.
3. **Secondary**, where the dataset defines it.
   * Code-capacity datasets: logical errors per configuration against the sampled fault, and their exact two-sided
     McNemar test, with the logical errors of the receipt's own answer beside them.
   * Circuit dataset: agreement of the declared observable with the sampled truth, per configuration.
   * Phenomenological datasets: no logical outcome, as in the campaign.

**What is not done.**
* No pooled comparison is computed or claimed.
* No dataset is selected after the fact.
* No winner is declared from the eight p-values. They are reported as eight descriptive tests, uncorrected, and named
  as such.

**What the conclusion will be.** The conclusion is what the counts show, dataset by dataset. That may be that one
setting finds more optima on some datasets and fewer on others, or that neither setting dominates. That outcome is a
result, not a failure.

**What is not concluded from this experiment.**
* That either setting is the better BP-OSD in general.
* Maximum-likelihood decoding over degenerate cosets.
* Any claim about the block's 12 logical qubits.

## Runner, outputs, cost

* **Runner.** `python experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py --dataset <name>`, one dataset per
  invocation, in the table's order. It writes `evidence/qldpc_scaling_prereg/<dataset>.jsonl.gz` (deterministic gzip)
  and appends the dataset's analysis to `evidence/qldpc_scaling_prereg/summary.json`. It refuses to overwrite a
  dataset's records.
* **Pinned at this commit (sha256 of the bytes):** see the table below. The runner records the same hashes in its
  summary, and a run under any other bytes is a deviation, reported as one.
* **Cost.** Measured on the smoke seeds while another three-process job shared the CPU: about 0.03 s per shot on the
  four 144-column datasets, 0.3 s on the phenomenological ones, 0.25 s at circuit level, and 1.7 s on stress, where
  the receipts are slowest.
  * Estimate for the whole run: about 35 to 45 minutes of one core, under 1 GB of memory.
  * The stress dataset is the longest single invocation, about 30 minutes; the others take under 6 minutes each.

| file | sha256 |
|---|---|
| `experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py` | 842b465f085f735e805c9a9e1609b6ca625294d3021ba7ea87d83ce9e5a6d632 |
| `src/oneq/qldpc_check.py` | 59773707a8b85046f66149434aea9907b0990f7a70b873ab79be4234a1a51b25 |
| `src/oneq/qldpc_cert.py` | 6e95ffa13e8373a1a94d459823503286ebba6760796b444bbabcfda91f3e1e3c |
| `tools/external/exact_lp_certificate_reference.py` | f260294977aa2459110e0551d9c8b8bea3d5e13e84859b373a64cee26e21b0cd |
| `tools/external/tree_checker_b.py` | 1e250b841568973afcb8ad092a4dac0434794b6f8b0133a3c15824bf68bcb64f |
| `evidence/instances/bb144_z_code_capacity.json` | 96d052a100562bc33bae3b5be1cd71dc8b06fab74e62b5373aea04fcb4f24b9a |
| `evidence/instances/bb144_x_code_capacity.json` | ba61df4c6c3b2375ccbdd8bf1d94b960bd12aa79bf11f9fa1446d52f3cd81235 |
| `evidence/instances/bb144_phenom_r4.json` | e9130581b81661b017d0c262a570c60b5c07d554e97e0293db3a564daf67e4a5 |
| `evidence/instances/bb144_refsched_r3_p002.json` | 3f7379de381b5c10d851530919f57610b796af92f771551ce986f9af2e6bb9eb |

Environment of record: Python 3.14.3, NumPy 2.3.5, SciPy 1.17.0 (HiGHS), Stim 1.16.0, ldpc 2.4.1, Windows 11 x86-64.

**Paper 2 reports** this campaign beside the original held-out campaign, not in place of it, and says which is which.
