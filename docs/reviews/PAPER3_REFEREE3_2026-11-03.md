# Referee report, third (final read) round, 2026-11-03: paper 3, *Machine-checkable lower-bound certificates for the circuit-level distances of bivariate-bicycle syndrome-extraction circuits*

Written by an AI agent (Claude, briefed as a hostile referee) that did not write the manuscript, with read-only access to the repository. It is not third-party peer review. Reproduced verbatim except that local file paths are replaced by a placeholder. What was changed in response is recorded in the commit that archives it and in RESEARCH_LEDGER.md (R229).

---

# Paper 3 final read: NOT READY, two MAJOR findings, both text-only fixes

The re-certification bookkeeping holds up exactly. Two abstract-level statements are false as written. Fixing them means rewording about six sentences; no number, table or certificate changes.

**The manuscript changed while I was reading it.** Commit 73d1072d (03:19) edited three lines of main.tex after my first read:
- "two reviews" became "four reviews" (l.309), which is correct.
- The 2.8 h longest-cell time is now scoped to "the original campaign" (l.266). The re-certification took 4.5 h for that cell.
- l.107 gained "their~Fig.".

I re-anchored everything to HEAD 73d1072d; the tree is clean and the PDF was built from it. All line numbers below are for HEAD.

## MAJOR

**M1. Pan's three-colour result is described as certificate-backed, and Pan says it is not.** The earlier fix (87d215b7) corrected the IBM half; the three-colour half is still wrong.
- **Where:**
  - Abstract l.40: "by SAT refutations with checked proofs (Pan)".
  - Sec. I l.54: "with every infeasibility claim backed by a proof checked with DRAT or VeriPB tools".
  - Sec. VI B l.264: "as Pan also found under SD6 with SAT refutations".
  - Sec. VIII l.299: "Pan's lower bounds for the three-colour cycle are SAT refutations, checked with drat-trim … and VeriPB", and "two independent certificate systems agree on the same SD6 quantity … the strongest check either can offer the other".
- **Evidence:** I read the raw text of Pan's `release/arxiv/main.tex` at the cited tag v2.0 (identical on main):
  - l.248: "Every infeasibility claim used in the proofs of **both theorems** is accompanied by a checked certificate." The sentences that follow list only Pan's new circuits (period-11 gross, period-11 [[90,8,10]], period-16).
  - l.260–265: the three-colour result is a separate Corollary: "The Fig.~12 circuit of SBB has dcirc=11 in both bases for every R … This is SBB's Prop. 3 and Cor. 1 combined with our value dext=11 for its orders, **from two independent exhaustive searches**."
  - Supplement l.824–826: "dext≥9 for the Fig.~12 order class is certified; dext=11 and the list of 4224 weight-11 logicals rest on two independent exhaustive searches."
  - The only DRAT proofs tied to Fig. 12 (l.460–467) show that retimings keeping the orders cannot reach 12. They are not the lower bound of 11.
- **Consequence:** the sentence at l.299 is false, and the comparison runs the wrong way. Among the cited works, this paper's 24 time-shift certificates and two T=2 certificates are the only ones I found that certify 11 for the three-colour cycle. Don't call that "first" without a fresh literature search.
- **Smallest fix:**
  - l.40: "…for the three-colour cycle at every number of rounds, by combining a reduction of Strikis, Browne and Beverland with an extended-code distance found by exhaustive search (Pan)".
  - l.54: "…from Strikis, Browne and Beverland's Prop. 3 and Cor. 1 and an extended-code distance of 11 from two independent exhaustive searches (Pan certifies ≥9); Pan's DRAT and VeriPB certificates cover the new circuits".
  - l.264: "…as Pan also found under SD6, by exhaustive search and the reduction of Ref. [2]".
  - l.299: reword the Pan sentence the same way, and delete the "two independent certificate systems agree…" sentence.
  - l.40 and l.56: "through a different proof system" no longer contrasts with anything; "machine-checkable certificates for these values" is accurate.

**M2. "Every certificate was re-checked by the committed checker" is false as written.**
- **Where:** abstract l.40 (the whole sentence, including "small proofs ship and the rest regenerate bit for bit"), Sec. I l.58, Sec. VII l.291. Line 175 already has the correct scope: "Every certificate the results rest on".
- **Recorded certificates not in the manifest (your item 1):** `records/` holds 304 VERIFIED certificate rows. 282 map one-to-one by proof hash into `recert_20261102/manifest.json`.
- **The 22 exceptions:** the IBM campaign certificates in `records/ibm_proofs_w9.jsonl` (T=1–11, Z and X, w≤9, rule W, 10.8–820.0 MB) are in neither manifest.
  - The IBM re-certification proved a different statement (rule S), on 20 re-serialized models.
  - Its own manifest says the originals were "checked by a working-tree checker never committed".
  - They don't ship, and the committed prover (CDP5) cannot regenerate them.
  - The paper still reports them: l.258 gives their size and time ranges, and Table III rows 1–2 are two of them (T=2 Z: 54,046,792 B, 14.5 s, 78.9 s).
- **Possible further exception:** Table III row 4 (IBM X, CDP5, 168 KB) was checked by the R206-era checker, whose ccheck.c differs from HEAD's (blob d7ed6cc1 vs 263a54bf). I couldn't establish that the committed checker re-checked it. Row 3 is re-checked at HEAD by `tests/test_r205_distance_dual.py:89–106`.
- **Fix:** add "the results rest on" at l.40, l.58 and l.291. In the Table III caption, say rows 1–2 are original campaign certificates whose checker was not preserved.

## MINOR

1. **Rounding up overstates two figures.**
   - l.264 says "33 h"; the manifest sums to 32.18 h, and Table III's own cells add to 32.2 h.
   - l.307 says "0.8 GB"; the actual total is 732,193,669 B, i.e. 0.73 GB.
   - Both come from `math.ceil` in `make_numbers_distance.py:582–583`. Fix: round to nearest.
2. **Table III caption (l.160) is ambiguous on provenance.**
   - "the earlier trace format of Sec. III F" could mean either earlier format described there. Rows 1–2 are the 4-bytes-per-node one: 54,046,792 B is about 4 × 13,511,651 entries.
   - "later re-certifications" reuses the campaign's term for R205/R206 benchmark runs. Those were made with older sources (ccheck.c blobs f66ac182 and d7ed6cc1, not HEAD's).
3. **"the recorded hash confirms" (l.307) is ambiguous.** Line 175 uses "recorded" to mean the original records. For 16 of the 30 regenerate-only proofs, the originally recorded hash is a CDP3 hash the committed prover will never reproduce. Fix: say the hash is the one in `recert_20261102/manifest.json`.
4. **Artifact: the manifest's regeneration recipe misses one family.** It says `--families 3c morph144`, but `recert_campaign.py:38` treats `3c_t2` as a separate family. That command regenerates 28 of the 30 proofs; it needs `3c_t2` added.
5. **Bit-for-bit regeneration never observed for 14 cells.** These are 3c Z1–Z10, 3c_t2 Z/X and morph144 Z2/X2; they have never been produced twice by the committed prover. For them, "regenerate bit for bit" rests on the code argument plus the 4-cell sample. About 1 h of proving would cover every cell kind.
6. **l.309, "independent reviews in Secs. IV and V".** Section V never calls anything an independent review. The l.250 cross-check comes from one, but the text doesn't say so.
7. **bb72 (last row of Table I).** Its certificate is regenerated by the test suite, but `bb72.jsonl` records no proof hash, so "bit for bit" has nothing to compare against.
8. **l.40, "at a fixed number of rounds".** Pan covers R≤3 and Crocker 12+2, so "at fixed numbers of rounds".
9. **Generator check that cannot fail.** `\MtwoMaxLoss` (maximum loss against the end-cycle code) is printed as "CNOT order alone changes the distance by up to". The two numbers agree today only through c42a vs c42a2 (6 vs 4); every other reorder pair differs by 1 or 0. Have the generator compute the reorder difference itself.
10. **Artifact nit.** `recert_old_prover.jsonl` has `committed_prover_proof_sha256: null` for both 3c_t2 rows. The manifest has the hashes, so no claim is affected.

## Clean (recomputed from raw cells and files, not from the generator)

- **Item 1, re-certification paragraph (l.175, l.307):**
  - Counts: 282 cells, all VERIFIED, all CDP5, all models byte-identical; 252 proofs identical and 30 differing. Of the identical ones, 54 carry trace bits, up to 34 MB. 212 of 256 morph proofs carry no bits. 252 ship and 30 regenerate.
  - Hashes: 0 recorded or model hashes missing from the original records. 0 mismatches on re-hashing all 534 shipped files. The `proof_equal` flag matches actual hash equality in both directions.
  - Determinism: 4 proofs regenerated at 4 threads came back identical to the 8-thread campaign proofs.
  - Old prover: its 30 rows are exactly the 30 differing cells, and all reproduce their records. The shipped `old_provers/cdist_prove_72d03874.c` matches the git blob at 72d03874.
  - Sources: HEAD sources match the manifest's hashes. The IBM re-certification's pins are the CRLF hashes of the same files, so one prover/checker version served both campaigns.
- **Item 2, Table III numbers:** every row matches its source to rounding. The provenance issues are MINOR 2 and M2.
- **Item 3:** Crocker's attribution matches the earlier source check. Pan's IBM claim matches Pan v2.0 ("R≤3, by exact enumeration"). Both references render correctly.
- **Item 4:** the counts agree across abstract, body and tables.
  - 14 circuits = 4 + 4 + 6.
  - The distances (two at 4, five at 5, five at 6, two at 7) reproduce the manifest's per-weight cell counts exactly.
  - 22, 24 and 14 certificates are 11, 12 and 7 round counts times two bases.
  - Four kernel-checked models.
- **Item 5:** the PDF is newer than every source, the log has no undefined references and no overfull boxes, and the review count is now correct.

**Not covered:** I checked Pan's statements, not his proofs. I ran no prover or checker. The Lean receipt does not pin ccheck.c, so "the version the Lean model follows" stays argued (as l.250 says). I did not test whether `recert_old_prover.py` runs without the `C:/lq` directories. Anything edited after 73d1072d is unreviewed.

**Adversary verdicts:**
- The abstract's "Every certificate…" sentence: **REFUTED**. Witness: the 22 rows of `ibm_proofs_w9.jsonl`. "Every certificate the results rest on" is confirmed.
- "Pan's three-colour bound is SAT refutations with checked proofs": **REFUTED**. Witness: Pan v2.0 l.248, l.260–265 and l.824–826.
- The re-certification counts and bindings: **CONFIRMED**.

I wrote no project files and no agent memory. The two cross-check scripts are in the scratchpad:
- <reviewer's local scratch path>
- <reviewer's local scratch path>

**Verdict: NOT READY.** It becomes READY once the M1 and M2 sentences are fixed; of the MINORs I'd do 1, 3 and 4 before posting.
