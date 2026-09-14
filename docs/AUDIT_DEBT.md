# Audit debt — named, not silent

**Evidence class:** [ESTATE-BUILT] each debt named below is measured against real coverage of a module that is built.


**The rule this file enforces:** a module at 0% test coverage is either
retired, tested, or listed HERE with its reason and its disposition. A dead
module nobody would notice going wrong is not neutral — it is where the next
silent failure lives. `tools/gate_all.py` floors the campaign-core modules;
everything below is the measured residue (coverage re-run 2026-08-19, suite
at 84.7% total, up from 69%). The five modules named have not changed in
that time and their dispositions still hold; only the surrounding total
moved, which is the number a reader would otherwise carry away.

## 0%-coverage modules (src/oneq/), triaged

> **Read the coverage column, not the heading.** This table was written on
> 2026-08-02 when every row was at 0%. Rows that have since risen say so
> inline ("now covered NN%"), and `tools/audit_debt_gate.py` refuses a row
> that describes a covered module as 0% without that note (the forward
> direction, added 2026-09-08 after a month of exactly that drift).

| module | import audit (2026-08-02, full-tree grep) | disposition |
|---|---|---|
| `coset_floor.py` | **NO importers anywhere** | **DELETED** in `e0d95d7`. |
| `algebraic.py` | **NO importers anywhere** | **DELETED** in `e0d95d7`. (Re-verified 2026-08-04: the ~36 tree-wide hits for this name are the WORD 'algebraic' in prose about a dead Delsarte approach, not imports.) |
| `mitm_fast.py` | **NO importers, no dynamic references** — the GPU parity test lives in `test_bz.py` against `bz`, not this module | **DELETED** in `e0d95d7`; if GPU MITM returns it returns with a test. |
| `qec.py` | `experiments/g1_saturation_wall`, `g2_arr`, `g4_hardware` (legacy gate lanes) | KEEP as experiment-support: covered by those lanes' own gates, not unit tests. Not campaign-core; excluded from floors. **now covered 97% (2026-09-08, coverage.json; the 0% reading is the 2026-08-02 audit's).** |
| `distance.py` | `experiments/gate0b_ibm825` sweeps | KEEP as experiment-support (historical sweeps that produced signed closures — deleting would orphan their provenance). **now covered 98% (2026-09-08, coverage.json; the 0% reading is the 2026-08-02 audit's).** |
| `arr.py` | `experiments/g2_arr` | KEEP as experiment-support. **now covered 100% (2026-09-08, coverage.json; the 0% reading is the 2026-08-02 audit's).** |
| `abft.py` | `experiments/g3_abft` | KEEP as experiment-support. **NOTE 2026-08-19:** this module is the **L1** layer (`Hx=s`), which ONE_Q_MAX 3.7.4 scopes to fault class P3 only and which catches an observable-mask flip in 0 of 16,000 shots. The **L0** layer that covers that class is a separate module, `abft_l0.py`, at a 100% floor. Reading `abft.py`'s 0% as "ABFT is untested" would be wrong in both directions. **now covered 95% (2026-09-08, coverage.json; the 0% reading is the 2026-08-02 audit's).** |
| `peeling.py` | `experiments/g4_hardware/peeling_ve.py` | KEEP as experiment-support. **now covered 100% (2026-09-08, coverage.json; the 0% reading is the 2026-08-02 audit's).** |

**Disposition is work**: each row is evidence-based (import audit above).
The three DELETEs **executed** in `e0d95d7`; the KEEPs are
experiment-support whose correctness gates are their experiments' own
artifacts. Nothing here is permission to stay unexamined.

**This register is now GATED** (`tools/audit_debt_gate.py`, in
`gate_all`). It had drifted, in the direction that is easy to miss: all
three DELETEs were already done and the file still described them as
queued, so finished work read as outstanding. The gate enforces the rule
at the top of this file in BOTH directions — every row names a module
that exists (or says DELETED), and every 0%-coverage module has a row.
The second direction is the one the rule was written for: a module that
quietly falls to 0% and is never listed becomes exactly the dead module
this file exists to prevent.

## Known-uncovered spans in covered modules (from `--cov-report=term-missing`)

- `moat_growth.py` 70%: uncovered spans include the separation-stall jitter
  branch (fires only on degenerate faces) and parts of the uncrossing
  emission — both exercised by hardware probes but not by unit tests. Add a
  pinned degenerate-face instance as a unit test.
- `certifying_decoder.py` 79%: degraded-path branches of `certify_exact` and
  parts of the escalated ladder spec parsing.
- `certify.py` 58%, `satdist.py` 55%, `isd.py` 57%: distance-foundry lanes;
  their end-to-end gates live in `experiments/` and the closure challenge —
  unit coverage is thin by design but should still rise.

## Optimization ledger (what "optimized?" currently means, per hot path)

Measured and tracked in `evidence/bench_ledger.jsonl` (`tools/bench_gate.py`):

| path | state | named upgrade if needed |
|---|---|---|
| checker | measured nnz^1.076; never the bottleneck | — |
| base certify | measured across 6 generations; caps tuned by OOM history | — |
| streaming | 315 shots/s @ 16 ms p50 (d3); recycle knob bounded both sides | p99 recycle spikes: pre-spawned standby workers |
| feldman (qLDPC) | **cache LANDED**: 20.5 → 9.3 ms/shot on [[144,12,12]] (2.2×), bounds bit-identical; phenom re-timing queued for a quiet box | — |
| exact tier | 46–258 s/shot, final tier only | Gomory–Hu rebuilt per round → warm-start / incremental trees |

**The law:** an optimization is either measured in the ledger or it is a
guess. Upgrades are named here BEFORE they are built, so the bench entry
that follows them has a hypothesis to confirm.
