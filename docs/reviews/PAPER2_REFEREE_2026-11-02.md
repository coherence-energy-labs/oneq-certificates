# Referee report, first round, 2026-11-02: paper 2, *Exact, independently checkable optimality receipts for BP-OSD*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

I recommend fixing this before it is posted. The core receipt claim held up under every attack I ran. The BP-OSD baseline layer did not: one column of the strong-baseline tables is internally inconsistent, and the BP setting that most likely drives the headline comparisons is never stated. Related work also misses decoders that provably return minimum-weight corrections on this exact code.

## Findings, most severe first

**1. MAJOR: configuration B's results at p=0.05 cannot come from a single fixed decoder.**
- **Where:** Table `tab:strong`, rows 0.05 and 0.07 (BP-OSD errors A/B/C = 114/109/9, then 172/41/32). Table `tab:instrument`, B column (tier 1 86 → 143, optimum hit 87 → 171). B is defined at main.tex l.138.
- **What's wrong:** with noise rate fixed across bits, an error of a given weight is equally likely to be any pattern of that weight, whatever p is. Min-sum with a constant scaling factor gives the same answers whatever uniform p it is told, so B's failure rate at a given true error weight must match across p. It does not:
  - B, true weight 7–10: **80/90 failures at p=0.05, 0/93 at p=0.07** (one-sided Fisher p = 3.3e-41).
  - B is also clean at p=0.03 (weights 6–8: 0/41).
  - Control arm A, also min-sum, is consistent across p (weight 8: 24/26 vs 26/31).
  - B's p=0.05 answers are not copies of A's: none of the non-optimal answers are identical, and B is heavier than A on 46 shots and lighter on 52. B's 113 non-optimal answers average weight 19.8, against a mean true weight of 7.1.
- **Why it wasn't caught:** `tools/replay_strong_baseline.py` recounts these columns from the stored records "with no BP-OSD", so it cannot detect this kind of defect.
- **Reproduce:** from the ONE_Q root, `python -I <reviewer's local scratch path> evidence/strong_baseline_v3`
- **Mechanism undecided:** either the p=0.05 run did not use configuration B, or ldpc 2.4.1 is unstable at that prior. In either case the B column is not a measurement of "min-sum, 100 iterations, OSD-CS 10".
- **Smallest fix:** re-decode the 200 stored p=0.05 syndromes with B at error_rate 0.05 and 0.07 (about 400 decodes) and diff against the stored answers. Then either explain the result or regenerate the column. Add this per-weight consistency check as a gate.

**2. MAJOR: the min-sum scaling factor is never stated, and the code uses unscaled min-sum (α = 1.0).**
- **Where:** l.97 and l.138, the abstract, Table `tab:heldout` tier 2 (222), and the McNemar test at l.114.
- **Evidence:**
  - `experiments/m0_prior_art/qldpc_strong_baseline_gate.py:53-61,134` calls `BpOsdDecoder(csr_matrix(HZ), error_rate=p, **cfg)` without `ms_scaling_factor`.
  - ldpc 2.4.1 `bp_decoder/_bp_decoder.pyx:94` defaults it to 1.0.
  - "scaling" appears 0 times in main.tex.
- **Symptoms of a mis-tuned decoder:**
  - At p=0.05, configuration A's tier-2 answers average 11.5 heavier than the certified optimum (maximum 22).
  - At p=0.01 the one tier-2 answer is 10 heavier on an error of weight about 2.
  - The paper's own discovery instance is weight 22 against an optimum of 6.
- **Impact:** the 164/300 tier-2 rate at stress and "McNemar 170:0, p = 1.3e-51" are measured against unscaled min-sum. The paper attributes the gap to iteration count, OSD order and product-sum instead. A web-search summary of the cited Krishnamoorthy et al. (2608.25545) says BP+OSD is within statistical error of maximum likelihood on this code; I did not confirm that in their source.
- **Fix:** state α = 1.0, and add a scaled or adaptive min-sum arm to Sec. `sec:strong`.

**3. MAJOR: related work omits prior art directly on this code (stale since 2025).**
- **Ott, Hetényi, Beverland, arXiv:2502.16408** (confirmed in their LaTeX source): the "height-bound decoder provides provable minimum-weight corrections", MaxSAT decoders give "guaranteed min-weight corrections", and both are applied to "the gross code [Bravyi et al.]". Their BP-DTD beats BP-OSD on the gross code under circuit noise. Not cited.
- **Tesseract, arXiv:2503.10988:** a most-likely-error search decoder benchmarked on [[144,12,12]] circuits against integer-programming decoders. Not cited. The project's own `docs/reviews/PAPER3_REFEREE_2026-10-29.md:23,35` found this work for paper 3, but paper 2 was never updated.
- **Wu et al. (HyperBlossom), 2508.04969:** its source says a dual "with the same objective value as the weight of a parity factor E certifies the optimality of E". That is the same kind of certificate as tier 1 with L = U. l.163 undersells it as "certifiable proximity bounds".
- **What survives:** the novelty sentence at l.165 still holds in its narrow form. Fix by citing these and stating the exact difference: exact rationals, the lattice rule, separately written checkers, branch trees, and the BP-OSD wrapper.

**4. MAJOR: the abstract contradicts the body on what the tier-1 rate measures.**
- The abstract says "the tier-1 rate is a measurement of decoder quality".
- l.140 says the gap between tier 1 and optimum hit is "the producer's certification cost, not the decoder's". Configuration C at p=0.07 shows it: tier 1 73.0% vs optimum hit 98.0%, with 50 optimal answers the flat route did not close.
- **Fix:** headline the optimum-hit rate, or call tier 1 a lower bound.

**5. MINOR issues.**
- **"Strongest configuration tested" (l.138, abstract) is not uniformly true.** At p=0.02, C has 3 logical errors and tier 1 197, while B has 0 and 200.
- **The checker's import list is wrong (l.77).** It claims imports of only `fractions`, `math` and `dataclasses`, but `src/oneq/qldpc_check.py:19` also has `import operator`.
- **The abstract implies the published noise model.** The circuit noise omits idle faults, uses 3 rounds and one observable. The body discloses this; the abstract should too.
- **The strong baseline sits next to "preregistered" in the abstract but is not preregistered.** It is a post-hoc run with seed `default_rng(int(10000p)+31337)`, gate dated 2026-08-03; label it so.
- **The pooled Clopper–Pearson bound of 98.73% (l.110) has no population behind it.** It pools eight different noise families weighted by arbitrary sample sizes. Drop it or call it what it is.
- **The circuit-distance paragraph is out of date (l.122).** It gives "circuit distance at most 12". From memory, not re-verified: Bravyi et al. report at most 10 for this cycle, and project records (R191) report exact BB circuit distances. Reconcile, or say the bound covers one observable without idle faults.
- **The companion manuscript at l.35 has no citation or identifier.**

## Attacks that failed (evidence for the paper)
- **Numbers:** I ran `make_numbers_qldpc.build()` in memory, without writing files. All four generated files came out byte-identical, and the evidence tree is clean. The 2026-09-23 evidence commit only changed line endings.
- **Lemma and facets:** the safe-dual lemma and the validity of the forbidden-set facets are correct.
- **Lattice rule:** the checker computes D from the weights itself (`qldpc_check.py:675`) and refuses negative weights (l.71, l.673).
- **Tier 2 is strict:** across 388 strong-baseline tier-2 records, BP-OSD's answer is always at least 2 heavier, with zero ties.
- **Statistics:** the per-dataset all-success bound 0.05^(1/n), the McNemar value 2·2⁻¹⁷⁰, and the 2.00% upper bound are all correct.
- **Citations:** 12 arXiv IDs checked against the arXiv API; titles and authors match the bib entries.

## Verdict
**Post after fixes.** Findings 1–4 are mandatory: finding 1 needs a roughly one-hour re-decode, and findings 2–4 are wording and citation changes plus one extra baseline arm. The certification layer survived every attack I ran. The baseline layer has an anomaly visible in a printed table and an unstated parameter that likely accounts for most of the tier-2 and McNemar numbers.

**Could not check:**
- Finding 1's cause, which needs a BP-OSD run (forbidden here).
- Per-shot tier-2 strictness on the held-out records.
- The JavaScript checkers and the full replay.
- What the cited 2026 papers actually claim, beyond their titles.
- The flat exact checker's code path end to end (grep only).
- Bravyi et al.'s "at most 10" and Krishnamoorthy et al.'s BP+OSD-near-ML result.

```
STRONGEST SURVIVING RESULT   Exact receipts are sound (lemma, lattice rule, strict tier 2) and the macros reproduce from committed evidence
EVIDENCE CLASS               DERIVED (soundness) + MEASURED (provenance)
HOW IT COULD STILL BE WRONG  The baseline columns rest on BP-OSD records that no replay re-derives; re-decoding B at p=0.05 decides it
WHAT IT LICENSES             Posting the receipt system; NOT the decoder-quality comparisons as currently framed
KILLED                       "B column measures min-sum 100 / OSD-CS 10" -> true weight 7-10: 80/90 failures vs 0/93 -> A and C columns survive
OPEN BRANCHES                (1) re-decode B @0.05 to find the cause; (2) scaled min-sum arm; (3) confirm Krishnamoorthy's near-ML result in source
RECOMMENDATION               RESEARCH FURTHER (branch 1), then SHIP
```

**Memory not written:** you made ONE_Q read-only and my memory directory is inside it. Lines to record:
- Witness: strong_baseline_v3, B @0.05 vs @0.07, true weight 7–10: 80/90 vs 0/93; reproduce with `strat.py` in the scratch folder.
- Attack that worked on decoder baselines: stratify failures by true error weight. A decoder that ignores uniform p must give the same conditional failure rate at every p.
- Instrument blind spot: `replay_strong_baseline.py` cannot detect BP-OSD record provenance.
- Prior art for minimum-weight decoding on the gross code: arXiv:2502.16408.

Sources: [Ott/Hetényi/Beverland 2502.16408](https://arxiv.org/abs/2502.16408), [Tesseract 2503.10988](https://arxiv.org/abs/2503.10988), [HyperBlossom 2508.04969](https://arxiv.org/abs/2508.04969), [Krishnamoorthy et al. 2608.25545](https://arxiv.org/abs/2608.25545), [Tesseract repo](https://github.com/quantumlib/tesseract-decoder)
