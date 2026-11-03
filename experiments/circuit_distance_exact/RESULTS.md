# R198 — exact circuit distance with checkable certificates

**Status: partial** — the IBM-cycle result is complete and machine-checked; the 3-colour result is
d_circ = 11 for every T >= 2 in both bases, PROVEN from machine-checked premises (the 24 time-shift certificates
were VERIFIED on 2026-10-23, the row below; this header said they were open until 2026-10-26); the 72-qubit
morphing circuit is complete (d_circ = 6) and so is the 144-qubit one (d_circ = 12, the code distance): both for
EVERY T >= 1 in both bases, PROVEN from machine-checked premises -- the 1-round certificates and `FOLD_PROOF.md`
Theorem 1 (d_circ(T) = d_circ(1); a hand proof that survived an independent audit, hypotheses machine-checked).
Certificate
status is whatever `evidence/circuit_distance_exact/results.json` records.

## The question
A syndrome-extraction circuit with a Pauli fault model compiles to a detector error model (DEM): D (detectors x
mechanisms) and L (observables x mechanisms). The circuit distance is
`d_circ = min{|X| : D X = 0, L X != 0}`, the fewest fault events that fire no detector and flip a logical.
For the gross code [[144,12,12]] the literature had only UPPER bounds: <= 10 for IBM's depth-8 cycle (BP-OSD
search), and 11 conjectured by Strikis, Browne & Beverland (arXiv 2603.05481) for a 3-colour tiled cycle from a
heuristic search over one QEC cycle.
Exact values WERE published for smaller BB circuits (found by paper 3's referee on 2026-10-29; each sentence checked
in the authors' own LaTeX source): Aghababaie Beni, Higgott & Shutty (arXiv 2503.10988, App. D) verified with stim's
shortest_error_sat_problem and a MaxSAT Evaluation 2024 solver that the circuit distance of the [[72,12,6]] code is 6
and of the [[90,8,10]] code is 8; Webster, Jacob & Higgott (arXiv 2603.22532, Sec. 5.5.3) report that Gurobi
completed within 8 hours only for the first member of their BB circuit family. For the gross-code circuits both give,
or cite, upper bounds only. (Their noise models were not compared with ours.)

## Results
| circuit | claim | evidence |
|---|---|---|
| IBM depth-8 gross cycle (Bravyi et al. 2024) | **d_circ = 10 for every number of rounds T >= 2, memory-Z and memory-X** | 2026-10-26, a second route that a third party can check: 22/22 rule-S time-shift certificates (T = 1..11, both bases, w <= 9) proved and checked by the COMMITTED prover and the Lean-modelled checker, every CDP5 proof shipped in `evidence/circuit_distance_exact/ibm_recert_20261026/` (`tests/test_r198_ibm_recert.py`). The original campaign: lower bound MACHINE-CHECKED for T = 1..11: 22/22 certificates (both bases, "no logical error of weight <= 9") VERIFIED by the independent checker; every T by `WINDOW_PROOF.md` Theorem 1 (P1-P3 for EVERY consecutive pair from finite hypotheses, machine-checked by `window_uniform.py`; audited 2026-10-22) -- PROVEN from machine-checked premises; upper bound: lifted full-DEM weight-10 witnesses in both bases |
| 3-colour cycle of 2603.05481 (Fig. 12) | **d_circ = 11 at T = 2 in both bases** (lifted weight-11 witnesses in memory-Z and memory-X): the conjecture holds at T = 2, for the circuit as rebuilt from their figures | engine search NONE <= 10: memory-Z 2,919 s / 2.34e9 nodes, memory-X 1,792 s / 2.49e9 nodes (20-22 threads); a lifted full-DEM weight-11 witness for memory-Z (61.6 s search); memory-X witness, every-T campaign and certificate status in `results.json` |
| 3-colour cycle, every number of rounds | **d_circ = 11 for every T >= 2, both bases** (every T by `WINDOW_PROOF.md` Theorem 1) | time-shift sweep: no solution of weight <= 10 starting in round 0 or 1 for T = 1..12 (Z 13,464 s, X 26,442 s, 12 threads); time-shift preconditions machine-checked on every consecutive pair; both weight-11 witnesses embed unchanged into every T = 2..12 and lift to verified full DEMs (`embed_witness.py`). **24/24 time-shift certificates VERIFIED** by the independent checker (2026-10-23; "no irreducible undetectable logical error of weight <= 10 whose earliest round is <= 1", T = 1..12, both bases, 23.1 h of prove + check): PROVEN for every T >= 2 from machine-checked premises |
| [[72,12,6]] with the IBM-style cycle | d_circ = 6 at T = 2 (memory-Z) | NONE <= 5 (engine; certificate in the test suite; an independent CDCL program agrees at <= 4), lifted weight-6 witness |
| Morphing circuit of Shaw & Terhal (2407.16336), [[72,12,6]] end-cycle code (mid-cycle [[144,12,12]], homomorphism x) | **d_circ = 6 for every T >= 1, memory-Z and memory-X**: their BP-OSD bound (<= 6) is exact | 14/14 time-shift certificates VERIFIED (weight <= 5, T = 1..7, both bases); space-time window preconditions machine-checked on every pair; weight-6 witnesses lift, survive the meta-checks and embed into every T = 2..7. MACHINE-CHECKED for T <= 7; for every T >= 1, PROVEN from machine-checked premises (`FOLD_PROOF.md` Theorem 1 + the T = 1 certificates) |
| Morphing circuit, [[144,12,12]] end-cycle code (mid-cycle [[288,12,18]]) | **d_circ = 12 for every T, memory-Z and memory-X** -- the end-cycle code's own distance: an ancilla-free, degree-5 circuit that loses NO distance (IBM's cycle loses 2, the 3-colour cycle 1). **MACHINE-CHECKED** for T = 1..13: weight-11 certificates VERIFIED at T = 1 (prove 230 / 210 s, check 801 / 812 s, Z / X, 12 threads) and T = 2 (prove 763 / 696 s, check 2,925 / 2,517 s), the fold lemma checked for every T = 2..13, and weight-12 witnesses (memory-X found in the 1-round model, 1,368 s) verified in every T = 1..13; for every T >= 1, PROVEN from machine-checked premises (`FOLD_PROOF.md` Theorem 1, audited 2026-10-22, + the T = 1 certificates). History of the evidence: | engine search NONE <= 11 (1,054 s, 4 threads); our weight-12 witness (found in the 1-round model, 545 s) lifts to a verified full DEM and survives the meta-checks. **Every T = 1..13 (memory-Z):** the 1-round model has NONE <= 11 (492 s), the fold lemma d_proj(T) >= d_proj(1) holds for every T = 1..13, and the witness embeds and verifies in every T = 1..13 -- d_circ = 12 for every T, DERIVED beyond 13. Until the certificates (2026-10-21 03:03) this rested on one uncertified cdist search per model -- the adversarial pass's open point, now closed |

The circuits are the published ones: the IBM cycle's term maps are re-derived in the tests from IBM's own matrix
construction (github.com/sbravyi/BivariateBicycleCodes, decoder_setup.py); the 3-colour cycle is rebuilt from
the paper's figures (geometry derived algebraically and matched to Fig. 10's long-edge directions), stim accepts
every detector in both bases, and its hook residuals reproduce all six patterns of Fig. 11. The landed exporter
reproduces byte-identical the models every recorded search and proof ran on (tested).

## Why generic solvers stall (measured, prose only -- WITHDRAWN from paper 3 on 2026-10-29)
No raw record of these runs was kept, the solvers had short budgets, and the [[72,12,6]] instance RC2 did not finish
in ~10 min was solved exactly by Beni, Higgott & Shutty with a MaxSAT Evaluation 2024 solver (above). Kept as
history, not as a claim.
Each detector is a parity over ~148 mechanisms (gross code, 2 rounds: 42,480 incidences, 8,568 mechanisms).
Parity over an expander is the worst case for resolution (Tseitin) — the proof system of CDCL SAT and MaxSAT.
Glucose with XOR chains proves d >= 4 in 70 s and stalls at 4; stim's exact MaxSAT export solved by RC2 does not
finish [[72,12,6]] in ~10 min. The expansion that makes qLDPC codes good makes their distance proofs hard for
resolution.

## The method, rule by rule (each with its soundness argument)
- **A Anchoring.** The translation group acts freely on mechanisms, preserves the detector structure and maps
  observables AFFINELY, obs(g m) = M obs(m) + K syn(m) — translated logical operators pick up stabilizer terms,
  so per-mechanism linearity is false, but on undetectable sets obs(gX) = M obs(X), and with g^-1 in the group
  "flips some logical" is invariant. Any solution can be translated to contain an orbit representative.
- **I Ordered anchoring.** With o the minimal orbit in X, translate so an element of orbit o is that orbit's
  representative: the search from anchor o uses only orbits >= o (4x fewer nodes, measured).
- **B Toggling semantics.** State = (syndrome, logical); successors depend only on the syndrome, so a
  transposition table keyed on the full state is sound (the prover omits it).
- **C Canonical branching.** Any completion has an odd number of members touching a firing detector, so
  branching over the allowed mechanisms touching it is complete; a firing detector with none is infeasible.
- **D/G Bounds.** Firing detectors no single mechanism co-touches each need their own mechanism; y_d = 1/c_d
  (c_d = max over allowed m touching d of |syn m & s|) is dual-feasible for the cover LP. Exact integers.
- **E** A state with empty syndrome and no logical flip has no branching.
- **L Radius-2 perimeter.** Completions of weight <= 2 decided exactly: single lookup (syndromes are distinct,
  checked at load), overlapping pairs from a table (every hit re-verified), disjoint pairs by the subset rule.
- **P Same-type projection.** Memory-Z keeps Z detectors: every full solution projects to a projected one of
  no larger weight with the same observables, so d(projected) <= d_circ. 8,568 -> 1,584 mechanisms on the gross
  code — the representation change that made weight 10 reachable. Witnesses are lifted back (`run_lane.py lift`).
- **W Time window.** Each mechanism touches <= 2 consecutive detector rounds, so a minimum-weight solution
  occupies a contiguous block of <= w rounds; with interior rounds time-translates and boundary windows
  independent of T (P1-P3, machine-checked by `tools_cde.window_preconditions`), NONE at w for T = 1..w+2 gives
  NONE at w for all T.

- **S Time-shift anchoring.** With orbits numbered in round order (`export_model.py --time-order`; the engine and
  checker refuse otherwise), a solution whose first round is r0 >= 2 is a time-translate (P2, and P3 for the end
  window, chained over consecutive models) of a solution starting in round 1 of the (T - r0 + 1)-round model. So
  "no solution starting in round 0 or 1" for every T = 1..w+2 covers every T, and each model needs anchors in two
  rounds only (`cdist ... MAXROUND=1`). On the IBM cycle it re-proves T = 1..11 at weight 9 in 0.2-6.1 s per model.

- **M Morphing circuits (2407.16336).** No ancillas: the 2lm qubits of a mid-cycle BB code alternate between two
  contraction circuits, ending each round in an lm-qubit end-cycle code (weight-9 checks, degree-5 graph). The
  schedule is a port of the authors' builder, compared gate by gate with it (`morph_vs_authors.py`: gates,
  measurement lists and every stabilizer's contraction round identical, both codes, all three homomorphisms;
  a schedule with two layers swapped is caught). The DETECTORS are derived, not copied: a stabilizer carried
  through round r's contraction is (measured part) x (rest on the data); its value at the next mid-cycle point
  is its value now plus the measured part, and where the rest is empty that value is a measurement -- a detector
  wherever the value is known. Only one round of the symmetry is spatial: round r + 1 is round r translated by
  x, so the anchoring group is the kernel {a even} (36 / 72 translations) and the window lemma becomes a
  SPACE-TIME lemma: windows are compared after undoing r0 * x (`window_preconditions(..., drift)`), and since a
  translation carries logical representatives to other representatives, the observables must transform
  AFFINELY between the paired windows, in both directions (`_shift_obs_affine`; one flipped interior observable
  breaks it -- tested).
- **F Fold lemma (morphing circuits, every T from ONE round).** Shaw & Terhal prove d_circ(T) = d_circ(1) for
  two-round morphing circuits (arXiv 2604.09797, Prop. 1, by time-reversal folding, in their convention).
  `fold_lemma.py` re-derives the step for this lane's experiments and CHECKS it per T: around each measurement
  the circuit is a palindrome (F_r, measure | reset, F_r reversed) with nothing measured between a mid-cycle point
  and the measurements, so every fault equals a Pauli at a mid-cycle point; each fault of the T-round circuit is
  sent to the 1-round fault with the same (projected) mid-cycle Pauli, and the projected effect there must be a
  GF(2)-linear function of the effect in the T-round model, observables included. Then every undetectable
  logical set folds to one in the 1-round model, no heavier: d_proj(1) <= d_proj(T). The Pauli argument only
  proposes the map; the linear check alone carries the conclusion (a wrong map can only fail it). Controls (each
  makes the check fail, tested): no propagation (conflicts), the other homomorphism's table (missing twins), one
  doctored observable (inconsistency). It holds for both morphing circuits, both bases, T = 1..13 (52 rows, of which
  the 4 at T = 1 fold a model onto itself and cannot fail: 48 carry information), and turns
  the [[72,12,6]] result into two independent routes (space-time windows + 14 certificates; fold + the 1-round
  certificate) and the [[144,12,12]] every-T question into a single 1-round model.
- **Completeness of the detectors.** `tools_cde.deterministic_gap` measures what the detectors and observables
  leave out of the circuit's deterministic parities (stim's tableau counts the random outcomes exactly; shots
  span the rest). For every circuit here the gap is exactly the 6 = n - rank(H) redundancy meta-checks of the
  other type's first-round outcomes -- standard memory circuits omit them. A lower bound is unaffected (more
  detectors only remove solutions). A witness must also survive them: `witness_survives_gap` re-derives the DEM
  with the meta-checks added and solves the choice of faults per merged mechanism exactly. Every recorded witness
  survives (IBM, 3-colour, morphing); a toy witness that flips one does not (tested). For these CSS circuits the
  survival is automatic, so it adds no evidence here (adversarial pass 2, below).

Engine speed: the work is split into (anchor, first child) tasks — the anchor-0 subtree (every orbit allowed) had
serialised the runs (the overnight checker averaged ~4.4 busy cores of 20). Gross T = 2 weight 9: 13.1 s on 20
threads before, 6.1 s on 4 after.

Certificate speed (2026-10-21, measured, not estimated): the checker was 3.5-4.2x slower than the prover
(3-colour Z T = 8 weight 10: prove 1,163 s, check 4,874 s), and a leaf-stub experiment showed why -- on a valid
proof every leaf answers "no completion", so a checker whose leaf returns 0 walks the same tree: 0.8 s instead of
50.8 s, i.e. 98% of the time was the weight-<=2 leaves (116M leaves, 3.9 billion candidate mechanisms at weight
8). Both programs now carry a LINEAR syndrome hash (per-detector keys; each its own keys and tables) and |s| down
the walk, test candidates with three-word span masks in degree-sorted lists that stop at the first candidate too
small to complete, filter probes with an in-cache bitmap, and confirm every hash hit exactly. On 3-colour Z T = 3
weight 9, 20 threads, alternating runs: prove 32.1 -> 23.4 s, check 98 -> 23 s (4.3x), 2.8x per certificate,
and the proof is BYTE-IDENTICAL to the original prover's. The optimized checker agreed with the original one on
423 mutated proofs over three models (flipped bits, truncations, extra bytes; 0 disagreements), and its fast leaf
path agrees with its exact sparse path (CCHECK_SPARSE=1) on mutated proofs (tested). Measured and kept out:
-march=native (no gain) and bound-pruning at remaining 3 (prunes 46% of those nodes, but the bound costs the
prover more than the leaves it saves; the proof language stays the original one).

## The certificate
`cdist_prove.c` writes one bit per node with remaining budget >= 3 and nonzero syndrome: expanded, or pruned by
the bound (format CDP3: per anchor a root record — pruned, or expanded with one bitstream per first-level child —
so prover and checker both run the children in parallel; IBM T = 2 weight 9: prove 2.9 s, check 11.0 s). `ccheck.c` re-derives everything else from the model: the symmetry (invariance, the affine
observable map by GF(2) elimination, free action, orbits, anchors), the branching detector by the deterministic
rule (any firing detector is sound; a prover that branched elsewhere desynchronises and is rejected), the child
lists, every bound in exact integers, and every leaf with its own procedure (its own index over a sorted array,
not the prover's tables). Every rejection path is tested to fire: a raised weight claim, a flipped bit,
truncation, trailing bytes, a fake bound prune, a lying prover that ignores solutions, a model that is not
invariant, orbit ids that disagree. Format CDP2 is ~1 bit per node: the IBM T = 2 weight-9 proof is 1.69 MB.
R205/R206 added CDP4 (orbit-filtered packing, the dual raise) and CDP5 (also a trace-free weight-3 terminal):
the same proof is 92 KB (`docs/RESEARCH_R205_CDP5.md`); the checker reads all three formats.

Trusted base: the circuits' instruction lists (built by `bb_circuits.py`, parsed by stim); the projection (a few
lines); the arguments above; `ccheck.c`. stim's error ANALYSIS is not in it: `dem_independent.py` derives every
model's fault set a second time by a backward Pauli-sensitivity pass (per qubit, the outputs an X or a Z at that
point would flip; CX, H, resets and measurements propagate them; each noise component contributes its Pauli's
mask) and it equals the list the exporter projects from on all 66 models — [[72,12,6]], the IBM cycle T = 1..11,
the 3-colour cycle T = 1..12 and the morphing circuits (72: T = 1..7, 144: T = 1..2), both bases, 432 to 66,912
mechanisms (`results.json` → `trusted_base`). The
equality can fail: a circuit copy with one noise line dropped or one CX reversed breaks it, and an instruction
the derivation does not model is refused (tested).

**The checker's calculus is PROVEN sound in Lean** (`lean/CertR198`, Lean 4 + mathlib v4.34.0; `SPEC.md` there).
29 theorems, no `sorry`, standard axioms only: branching on any firing detector loses no completion; the packing
and fractional-cover bounds in the checker's exact integers; CDP4/CDP5's orbit-filtered packing and dual raise
(every raise keeps the dual feasible; feasible weights bound every completion); CDP5's weight-3 terminal as a valid
tree; the radius-2 leaves; anchoring with an affine observable map; and `cert_sound` -- a valid certificate for w
means every undetectable logical error has weight > w -- with its scoped form for rule S (irreducible solutions
only). Two concrete models run the chain (d = 3; and a prune only the dual raise makes, on which `ccheck.c`
agrees). `lean_receipt.py` pins the proved sources and every theorem's axioms; the gate needs no Lean and refuses
a stale source, a `sorryAx`, a non-standard axiom or a changed theorem list (tested). NOT proved: that `ccheck.c`
implements the Lean predicate -- argued line by line in `SPEC.md` -- and the trace decoding, the leaf index, the
elimination, parsing and the compiler. The formalization found three soundness bugs in the C (a mechanism listing
a detector twice gave a false VERIFIED, F1/F5, and crashed the prover, F4) and an overstated scoped verdict (F2);
all four are closed (part 14).

## Independent review of R205/R206's checker rules (2026-10-22)
R205/R206 (another session) added three rules to the trusted checker: packing over the co-touch graph of the
mechanisms ALLOWED at the anchor's orbit (a suffix union), an exact-integer dual raise of the fractional cover,
and CDP5's trace-free weight-3 terminal (four pairwise non-co-touching firing detectors, else every allowed first
member followed by the pair leaf). Reviewed here by something that did not build them: each rule's argument
checked by hand, then formalized (the Lean theorems above), then held against brute force --
`ccheck_oracle.c` compiles `ccheck.c` itself and answers with its own `bound_ok` / `leaf_exists3` /
`leaf_exists`; on 4,798 queries over 20 random models (breadth-first search over (syndrome, observable), no shared
code) no prune ever cut a completion within budget and no terminal missed one, under the CDP3 and the CDP5 rules.
Nine one-token mutants, one per rule (obstruction at three, no suffix union, a raise that forgets its own weight
or skips orbit o, a terminal that ignores singles, skips orbit o or drops pairs, either bound at equality), each
produce violations (44 to 1,941), so the comparison is not dark (tested). Verdict: sound; nothing refuted.

## Independent adversarial pass (2026-10-20)
An adversary agent that built none of this attacked all three claims and refuted none. It re-derived the DEMs
with its own backward Pauli-sensitivity pass (no stim error-model code) — byte-identical to stim's for the IBM
cycle (8,568 / 8,640 mechanisms) and the 3-colour cycle (7,800); its own projections equal the exported models;
its own GF(2) checks confirm the symmetry (72 / 24 translations, affine observables, free action); it re-verified
the witnesses (no 9-subset is logical; they stay valid shifted to T = 3..5); its own SAT encoding (CaDiCaL,
totalizer, own model) settles [[72,12,6]] at 6 in both bases. Its findings, all fixed here: the exporter now
verifies the affine observable map itself (it was only in the checker); the window lemma's proof now states the
"w mechanisms occupy <= w rounds" step; "every T" is MACHINE-CHECKED only up to the largest T run and DERIVED
beyond; "exactly 11" needed a witness in this model (now exhibited for memory-Z). Its heuristic BP-OSD searches
never reached even the known minima (15/12 for d = 10), so they carry no evidence either way.

## Independent adversarial pass 2 (2026-10-21, on the morphing / fold results)
An adversary that built none of it attacked "the [[144,12,12]] morphing circuit has d_circ = 12 for every T":
verdict CANNOT-DECIDE, nothing refuted. It re-exported the 1-round model (hash matches the search record),
rebuilt it its own way (all 864 mechanisms set-equal), read the fold check and judged it sound (images are real
mechanisms and depend only on the mechanism; zero-effect components drop out; coincident images only cancel; the
observable condition is linear), named the fold map explicitly (stabilizer g's detector in any round -> its single
1-round detector: 0 mismatches at T = 2, 14 and 15, so the reason does not depend on T), found no completeness
bug in `cdist.c`, and confirmed the upper bound. Its own CDCL SAT encoding could not decide weight 11 (orbit 0 at
weight <= 8 took 113 s; its weight-12 positive control did not finish), so **the whole lower bound rests on one
uncertified cdist search of the 1-round model** -- the certificate (`cdist_prove` + `ccheck`) is queued. Narrower
findings, all taken: the part-9 commit title states "every T" without its class (MEASURED, single engine) -- the
claim and this file carry it; "52/52" fold rows include 4 that cannot fail; and two things about projection in
these CSS circuits with depolarizing noise: every projected mechanism has a pure-X (memory-Z) / pure-Z (memory-X)
fault component, which flips no other-type record. So (i) the same-type projection is EXACT here, d_proj = d_circ
(a projected solution lifts by those components at the same weight), and (ii) the meta-check survival test can
never fail here -- it is a real test in general (the toy control fails it) but carries no evidence for these
memory experiments.

## The window lemma for every T (2026-10-22): the IBM cycle and the 3-colour cycle
Rules W and S needed the window preconditions P1-P3 on every consecutive pair (T, T + 1); they were machine-checked
on the computed models (T <= 11 / 12) and rested beyond on "the builder's time-periodicity" (DERIVED).
`WINDOW_PROOF.md` replaces that with Theorem 1: for both circuits and both bases the T-round projected model has
the exact closed form M_T = S u (T - j_s - U time-shifted copies of one interior set I) u tau_T(End) for EVERY
T >= T_0, with S, I, End finite -- from a template C_T = P . F . B^(T-1) . E . R (the 3-colour cycle is PIPELINED:
a round's X checks finish in the next block, so it has a first block F and a 5-step tail E; it is not
prep . body^T . readout), a labelling rule, and two matrix identities on the body block's frame maps (FIR, END).
So P1-P3 hold for every pair (Corollary 1; in memory-Z the pair (1, 2) in its time-shift form), and with the
lane's finite premises: IBM d_circ = 10 and 3-colour d_circ = 11 for every T >= 2 (Corollary 3).
`window_uniform.py` checks the hypotheses (60 negative controls, all failing) and the closed form against stim's
DEM at T = 16, 24, 32 (equal in 20 of 20). An independent audit (`WINDOW_PROOF.md` section 8) rebuilt the closed
form from stim alone at another reference and compared it at T up to 27 (102 comparisons, 0 differences, every
control failing), checked Lemmas 2-3 at the frame level, and refuted nothing; its two narrower notes are fixed.
`results.json`: the IBM claim is now PROVEN for every T >= 2 from machine-checked premises; the 3-colour every-T
step is PROVEN too, and its T <= 12 premises became machine-checked when the last time-shift certificates landed
(next section).

## The 3-colour cycle's 24 time-shift certificates (2026-10-23): d_circ = 11 for every T, machine-checked premises
Each of the 24 (basis, T) cells, T = 1..12 in memory-Z and memory-X, has a certificate the independent checker
VERIFIED: no irreducible undetectable logical error of weight <= 10 whose earliest round is <= 1, on exactly the
projected model the time-shift sweep searched (`finish_3c_certs.py` refuses a certificate on any other model
hash). Translation group 24, anchors 36-60 per model, 864 to 8,784 mechanisms. Cost: memory-Z 9,560 s prove +
28,535 s check, memory-X 26,102 s + 18,958 s, 23.1 h in all. The longest cell was X T = 8, 2.8 h, 8 threads under
load. Rule S turns "no solution starting in round 0 or 1" into "no solution at all", and `WINDOW_PROOF.md` Theorem 1
supplies its preconditions on every consecutive pair. The weight-11 witnesses give the upper bound. So the 3-colour
cycle of 2603.05481 has d_circ = 11 for EVERY T >= 2, both bases. `results.json` now states this as PROVEN from
machine-checked premises (the gate requires all 24 cells, each on its swept model). The authors conjectured 11; this
settles it for their circuit as rebuilt from Figs 10-12. The proofs themselves (6.6-373 MB each) are deleted after
their check; the record keeps each proof's sha256 and the checker's verdict line
(`records/3c_shift_certificates_w10.jsonl`).

## The fold lemma for every T (2026-10-22): both morphing circuits, every T >= 1
`FOLD_PROOF.md` proves, for the two-round morphing memory experiments, d_circ(T) = d_proj(T) = d_proj(1) =
d_circ(1) for EVERY T >= 1 under six hypotheses on finite, T-independent data (involutive layers, all right
qubits measured and reset, trackability, the palindrome, the boundaries, pure parts). The proof is an explicit
formula, not an induction on a pattern: a fault whose mid-cycle image is P has projected symptom
(iota_t(sigma_S(P)), sigma_L(P)), and summing each generator's detectors undoes every iota_t at once. So the
T-round projected model is the union over t of iota_t(the 1-round model) (Corollary 2). `fold_uniform.py` checks
the six hypotheses (both circuits, both bases, homomorphisms x, y, xy) with 20 negative controls that must and do
fail, and identifies the builder's circuit with the theorem's template at T up to 32 (5.7 million components, 0
mismatches); `fold_twin.py` sends the weight-12 witnesses to T = 32. An independent adversarial audit
(2026-10-22, `FOLD_PROOF.md` section 8) attacked every lemma, checked Corollary 2 against stim's DEM ALONE at 18
fresh T (both controls disagree), and ran 8 small bivariate-bicycle morphing codes the proof never saw (distances
constant in T; breaking one hypothesis makes them drop): CONFIRMED, nothing refuted; its four narrower notes are
fixed in the proof. With the 1-round certificates this gives both circuits' distances for every T
(`results.json`: PROVEN from machine-checked premises; the per-T fold check, T = 1..13, stays as a second route).
Tests: the hypotheses live, the twin map, Corollary 2 against stim with both controls, and the claim falling back
to its per-T route when any premise record breaks.

## Shaw & Terhal 2026 (arXiv 2604.09797): circuit distance of the Abelian-2BGA morphing circuits (2026-10-22)
The paper searches two-round morphing circuits over weight-5 and weight-6 Abelian 2BGA codes and reports END-cycle
parameters only -- "we did not check the circuit-level distance" (its App. C.4) -- and names neither the groups nor
the polynomials. `morph2026_codes.py` re-finds its [[42,6,6]] degree-4 example (Z_42: 4 standard-schedule circuits,
two codes x two CNOT orders) and shows its [[30,8,5]] degree-5 example is NOT reachable by the standard schedule;
`morph2026_trees.py` searches its Def. 10 tree circuits and finds 12 with end-cycle [[30,8,5]] (degree 5 or 6).
`morph2026.py` builds each as a memory experiment in this lane's exact conventions (the generalized builder
reproduces `bb_circuits.morphing_cycle` byte for byte on the lane's codes); the standard schedules equal the authors'
own builder gate by gate (`morph2026_vs_authors.py`; a swapped-layer control fails).

Per target and basis (`results.json` -> `morph2026`): the T = 1 search finds weight d with a full-DEM-verified,
minimal witness (every single deletion fails); the same weight in every T = 2..w+2 (w = d - 1); certificates
"no solution of weight <= w" VERIFIED at T = 1 and 2 on exactly the searched models; time-shift certificates and the
space-time window preconditions on every T = 1..w+2. For every standard schedule, `fold_uniform.py` checks
FOLD_PROOF.md's six hypotheses (all hold, all 18 controls fail; template agrees up to T = 16), so its Theorem 1 gives
the distance for EVERY T.

| end-cycle code (degree) | circuit | d_circ (Z / X) | every T |
|---|---|---|---|
| [[42,6,6]] (4) | Z_42, A = 1 + x + x^10, B = 1 + x^7, standard schedule (c42a) | **6 / 6 -- loses nothing** | PROVEN |
| [[42,6,6]] (4) | the same code, other CNOT order (c42a2); A = 1 + x^3 + x^9 (c42b, c42b2) | 4 / 4; 5 / 5; 4 / 4 | PROVEN |
| [[48,4,8]] (4) | the six standard-schedule circuits on Z_48 (the paper's Fig. 25 code family) | best **7 / 7** (two of six), the rest 6 | PROVEN |
| [[30,8,5]] (5) | the four Def. 10 tree circuits on Z_30 (A = 1 + x + x^4, B = 1 + x^6 + x^13) | **5 / 5 -- loses nothing** | MACHINE-CHECKED for T <= 6, DERIVED beyond |

So: the paper's degree-4 [[42,6,6]] example keeps its full distance in one of its four CNOT orders and loses up to 2
in the others -- the order is not a detail; its degree-5 [[30,8,5]] tree circuits keep their distance; and no
standard-schedule circuit for the [[48,4,8]] family reaches 8 (cyclic Z_48 only; non-cyclic groups and deeper trees
not searched). Upper bounds are our own verified witnesses; every lower bound is a checker-VERIFIED certificate.

## Circuit distances as theorems the Lean kernel checks (part 18, 2026-10-22)
The trusted checker `ccheck.c` is argued, not proved, to implement the Lean calculus (gap G1). For small models
that gap is now closed outright. `lean/CertR198/CertR198/Exec.lean` defines a Lean PROGRAM: an exhaustive anchored
search over the exported projected model (bitmasks, an incidence table that is data and is checked for
completeness). It proves that the program's acceptance implies "every undetectable logical error has weight > w"
(`cert2_sound_exec`, through the same `cert_sound` the certificate calculus uses, with the trivial group).
`lean_kernel_proof.py` writes one module per model, and the KERNEL runs the program there: one `decide +kernel`
theorem per anchor, combined by rewriting, so no single declaration grows (an uncapped single `decide` reached 20 GB).
The modules ship four results as Lean theorems on the standard axioms only, trusting no C code and no compiler:

| model (1-round projected, the campaign's certified export) | theorem | distance |
|---|---|---|
| 2604.09797 [[42,6,6]] degree 4 (c42a), memory-Z / memory-X | `Kernel.C42aZ1.distance`, `Kernel.C42aX1.distance` | **6 / 6** |
| 2604.09797 [[30,8,5]] degree-5 tree circuit (t30a), memory-Z / memory-X | `Kernel.T30aZ1.distance`, `Kernel.T30aX1.distance` | **5 / 5** |

Each module's model literal is regenerated by the gate from a fresh export whose sha256 must equal the campaign
record's, so the theorem is bound to the certified model. For c42a, FOLD_PROOF.md Theorem 1 then carries the
distance to every T. Kernel time 141-624 s per model at <= 2.6 GB resident (Mathlib's baseline is ~1.5 GB). Larger
models -- the gross-code cycles' hundreds to thousands of mechanisms at w = 9-10 -- would need the bound rules and
the symmetry in the kernel program. That is the next step, not a claim.

## Measured nulls
- Patch pattern database with detector-share cost partitioning (2x2 tiles): 443K vs 441K nodes.
- Orbit re-ordering by mechanism degree: 15.5M (descending) / 18.4M (ascending) vs 9.6M nodes (circuit order).

## Reproduce
```
python run_lane.py search  --circuit ibm --basis Z --rounds 2 --wmin 1 --wmax 10
python run_lane.py sweep   --circuit ibm --basis X --tmax 11 --w 9
python run_lane.py certify --circuit 3c  --basis Z --rounds 2 --w 10
python run_lane.py lift    --circuit ibm --basis Z --rounds 2 --ids 702 1513 1512 1207 1527 1552 1080 1560 1562 1579
python run_lane.py window  --circuit ibm --basis Z --t 10
python dem_independent.py   --circuit 3c --basis Z --tmin 1 --tmax 12     # the fault set without stim's analysis
python run_lane.py sweep   --circuit morph72 --basis Z --tmin 1 --tmax 7 --w 5 --maxround 1
python run_lane.py window  --circuit morph72 --basis X --t 3              # space-time window, drift x
python -I morph_vs_authors.py --clone <clone of github.com/Mac-Shaw/morphing_qec_circuits>
python collect_results.py
```

## Open (see docs/OPEN_WORK.md, R198)
- (Done: the window lemma for every T, `WINDOW_PROOF.md`, and the fold lemma's T-uniform proof, `FOLD_PROOF.md`,
  2026-10-22; the 3-colour cycle's 24 time-shift certificates, 2026-10-23. Every every-T claim in this lane now
  rests on machine-checked premises plus an audited hand proof.)
- The C => Lean step for `ccheck.c` (gap G1 in `lean/CertR198/SPEC.md`): argued line by line and cross-checked by
  the brute-force oracle, not proved. For small models it is bypassed outright -- the kernel runs a proved-sound
  Lean checker (part 18). For the gross-code cycles that needs the bound rules and the symmetry inside the kernel
  program.
