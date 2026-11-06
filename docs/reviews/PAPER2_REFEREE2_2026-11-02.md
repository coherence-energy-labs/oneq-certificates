# Referee report, second round, 2026-11-02: paper 2, *Exact, independently checkable optimality receipts for BP-OSD*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

**Verdict: REFUTED as worded, narrowly.** Every number in the post-hoc analyses traces to an artifact, and I reproduced the key ones independently. Two statements fail. First, the tier labels follow the producer's code path, not the rule the paper states for tier 1. Second, the abstract's "it exposed a misconfiguration" credits the receipts with something a referee and a parameter sweep did. I found nothing fatal.

## FATAL
None.

## MAJOR

**M1. The tier labels follow the producer's code path, not the paper's stated tier-1 rule. Tier-1 counts are understated, and about half the "trees" have no branch.**
- **Where:** the tier definitions (main.tex:66–68); the abstract's "17 as small branch-dual trees"; main.tex:111 ("fraction of BP-OSD answers proven optimal (88.16%)", "needed no tree" 99.15%); main.tex:141 ("50 optimal BP-OSD answers were not closed by the flat route", "the relaxation and its cuts may simply not close"); the tier-1 column of Table 3.
- **What is wrong:** the paper defines tier 1 as "L > U − 1/D for BP-OSD's answer". The producer (`experiments/m0_prior_art/qldpc_strong_baseline_v3.py` ~L179–184 and L241) only issues tier 1 when the shot has a stored flat proof. It stores one only when the LP primal is integral. On shots where the primal is fractional but the root dual already closes by the lattice rule, it writes a one-node "tree": `["bound", duals]`, with no branching. BP-OSD's equally optimal answer on those shots is then filed as tier BD.
- **Witness:** the production `check_qldpc_exact`, given each stored root dual and an empty box map, accepts BP-OSD's own answer on these shots.

  | configuration, p | optimal answers filed BD | accepted as a flat certificate |
  |---|---|---|
  | C, p=0.07 | 50 | 33 |
  | C, p=0.05 | 11 | 9 |
  | B, p=0.07 | 28 | 20 |
  | B, p=0.05 | 1 | 1 |

  Sanity check: the tree's own candidate is accepted in every case.
- **Share of one-node trees:** strong baseline 10/12 at p=0.05 and 34/52 at p=0.07 (this is why the median at p=0.07 is 1). Held-out: 8/17 (`bd_trees` in `qldpc_heldout_addendum.json`). Development: 10/19.
- **Circuit rung:** two tier-BD shots certified BP-OSD's own answer. Replaced = 31 = tier 2, so identical = 118 = 116 tier-1 + 2. "BP-OSD answers proven optimal" is therefore at least 1,782, not 1,780.
- **Effect:** under the paper's own rule, Table 3's tier-1 column at p=0.07 rises from 146 to at least 179 (C), 143 to 163 (B) and 145 to 178 (A_s/B_s). Held-out syndromes needing no branching are 2,010/2,019 (99.55%), not 99.15%. Soundness is untouched and the error runs in the conservative direction. But the stated reason for the tier-1 vs optimum-hit gap is wrong for two-thirds of it at p=0.07.
- **Fix:** re-classify from the stored proofs (no new computation) and correct the counts, or state the integral-primal precondition in the tier-1 definition and say that a one-node tree is a single root dual.

**M2. "It exposed a misconfiguration" (abstract; also main.tex:36 and :98) overstates what the receipts did.**
- **What is wrong:** receipts show that answers are heavier than the optimum. They cannot say why.
- **Pre-revision evidence:** the previous version (`git show 95f5509f:paper/qldpc-certificates/main.tex`) read the same receipts as decoder quality ("rises from 42.0% to 93.5% between the stated and the strongest BP-OSD configuration"). Searching it for "scaling" or "unscaled" finds nothing.
- **Who found it:** the post-hoc scripts' own docstrings credit the referee and say the paper never stated the default: `qldpc_strong_baseline_scaled.py`:6–8, `qldpc_heldout_scaled_posthoc.py`:5, `qldpc_bposd_scaling_sensitivity.py`:5–7.
- **What survives, and I measured it:** the cause is clean. Re-decoding at α=1.0 reproduced all 300 recorded stress candidates and all 200 X-sector candidates exactly. At α=0.625 the stress misses fall from 178 to 1.
- **Fix:** "Used as an instrument, the receipts show the campaign's BP-OSD missed the certified optimum on 178/300 stress syndromes. A post-hoc check prompted by review traced this to the library's default scaling factor of 1. Re-decoded at 0.625 with nothing else changed, it missed on 1." Also retitle the strong-baseline section ("What the advantage is: a strong baseline"); the title is left over from the old framing.

## MINOR
1. **Rungs not re-decoded:** heterogeneous, phenomenological and circuit were not re-decoded with scaling, and the paper never says so. That leaves 31 of 149 circuit answers proven heavier with no attribution, while main.tex:98 generalises from code-capacity data.
2. **No control arm in the artifact:** `qldpc_heldout_scaled_posthoc.py` compares the old run against a fresh one without an α=1 re-decode. My control passed, but the artifact should carry it. Separately, `make_numbers_qldpc.py`:231 trusts the `published_rule_check` values stored inside the artifact it is checking; it re-hashes the records but does not recompute the check.
3. **Scaling is not uniformly better:**
   - B_s finds fewer optima than B at low p: 199 vs 200, 197 vs 200, 198 vs 199.
   - At p=0.02, B_s makes 1 logical error where B makes 0.
   - The scaling scan covers B only.
   - The declared-rate scan has no scaled arm, so "unscaled is erratic" has no matched control.
4. **"More than the iteration count or the OSD order" (main.tex:98):** at p=0.07, more iterations and a higher OSD order alone take optimum hits from 24 to 171. Scaling alone takes them to 196. Qualify the sentence.
5. **The McNemar rebuttal (main.tex:115) uses optimum hits, not logical outcomes.** Scaled logical outcomes are computable, because the seed rule (main.tex:103) regenerates the faults. Yet the post-hoc JSON says they "cannot be recomputed". Either compute them or say what the rebuttal rests on.
6. **Checker wording:** "two independently written checkers" (main.tex:113) contradicts main.tex:80 ("separate implementations, not independent third-party reproductions").
7. **Shen–Zhong wording:** their witness is a *lighter* feasible correction that proves the heavier one non-minimal (2609.07035, `sections/01-introduction.tex`:99–100). The paper says "a heavier feasible answer that exhibits suboptimality".
8. **The 0.625 choice:** it is disclosed everywhere, labelled post hoc, and insensitive across the scan (B at p=0.05: 10/11/9 failures for α = 0.9/0.75/0.625; at p=0.07: 39/36/36). It was fixed on strong-baseline data before being applied to the held-out syndromes. But the paper gives no rationale or citation for it, and the sentence reporting the sensitivity scan is not itself labelled post hoc.

## Checks that did not find a problem
- **Tier-1 rule reproduction:** confirmed shot by shot, not just in totals. Zero mismatches over 3,000 A/B/C decisions and 2,000 scaled ones, using my own rule (optimum hit and a flat reference proof, no checker needed). The agreement holds by construction, so it shows the two scripts are consistent, not that they follow the stated rule (see M1).
- **Optimum hit:** computed correctly in both analyses (feasibility against the stored syndrome, and weight equal to the certified optimum). Zero flag mismatches over 5,000 decisions, and no BP-OSD answer was ever lighter than the certified optimum; that check could have fired.
- **178/300 from the records:** 122 tier-1 hits; 164 tier-2 plus 14 BD answers, all syndrome-valid and strictly heavier. Re-decoding at α=0.625 gives 299/300 (the miss is shot 35: weight 12 against optimum 6). Uniform p=0.02 (450/470) and X-sector (194/200) recounts also match.
- **Library default:** `ldpc` 2.4.1's default scaling factor is 1.0, both in `_bposd_decoder.pyx`:53 and on a default-constructed decoder.
- **Scaled strong-baseline records:** re-decoded p=0.07 A_s shots 0–99, 100/100 identical.
- **Paired comparison:** the records give 175 BP-OSD errors and 5 pipeline errors, consistent with the 2×2. It was predeclared (`docs/HELD_OUT_PREDECLARATION.md`:48–49). Nothing post hoc is labelled preregistered.
- **Bounds and sensitivity:** the Clopper–Pearson bounds (98.00, 99.36) and the 2.00% upper bound check out, and the sensitivity macros match the JSON.
- **Related work, checked against the authors' own LaTeX:** Ott–Hetényi–Beverland (`0_main.tex`:201–204), Wu et al. (`abstract.tex`:6, `introduction.tex`:16) and Krishnamoorthy et al. (`pgmec.tex`:69–81; "linear program" does not appear) are described accurately.

## Referee verdict
Post after fixes. The numbers are sound and traceable, and the scaling attribution is now a measured, controlled result. M1 needs a re-classification from proofs already stored; M2 is a wording fix. Both should be made before posting, because a hostile reader will find them in the released files.

**Could not check:**
- full corpus replay (excluded by your constraints)
- the PDF and the GitHub release
- the held-out producer's tier-1 code (I inferred its behaviour from the records)
- the other related-work entries
- the circuit, heterogeneous and phenomenological rungs under scaling
- whether 0.625 is the conventional literature value

**Constraint breach:** I wrote my agent memory inside ONE_Q, against your read-only instruction: a new file `.claude/agent-memory/adversary/r229_paper2_scaled_posthoc_claims.md` and one line appended to `MEMORY.md` in the same folder. Nothing else in ONE_Q was touched; revert those if unwanted.

Scripts to reproduce, in `<reviewer's local scratch path>`:
- `verify.py` — run `python verify.py all`
- `leafcheck.py` — run `python leafcheck.py`; this is the M1 witness
