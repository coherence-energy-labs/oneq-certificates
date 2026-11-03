# lean/CertR198: soundness of the R198 certificate calculus (`ccheck.c`)

A Lean 4 + mathlib formalization of the rules `ccheck.c` applies to a CDP3, CDP4 or CDP5
certificate, and the theorem that an accepted certificate means what it claims.

**Modelled version.** `experiments/circuit_distance_exact/ccheck.c` at `493facf7` (441 lines,
sha256 `263a54bfa254...7504`): the R198 part-14 checker with R205/R206's CDP4/CDP5 rules. The hash is
of the file as git stores it, LF line endings; a Windows checkout made before the repository pinned
C sources to LF holds CRLF bytes, sha256 `4fc5540e203b...0af3`, the form
`evidence/circuit_distance_exact/ibm_recert_20261026/manifest.json` records -- the same text. Every line
number below refers to that version. The CDP3 calculus was first formalized against the 260-line
checker of `0bc51a64` (2026-10-21); the CDP4/CDP5 rules (`prune5`, `Valid.terminal3`) were added
on 2026-10-22, after an independent review of R205/R206 (the brute-force oracle and its nine
mutants in `tests/test_r198_circuit_distance_exact.py` are the executable half of that review).

**Status: partial.** The calculus is PROVEN sound (no `sorry`, standard axioms only:
`lean_receipt.py` pins this). The step from "`ccheck.c` printed VERIFIED" to "the Lean hypotheses
hold" is an ARGUED correspondence (the table below), not a proof. The places where it failed were
found and reproduced on the real checker (F1, F2, F5) and are closed in the C (part 14, `f37072d2`).

## Building

The producer builds a copy at a short path and writes the receipt:

```
python experiments/circuit_distance_exact/lean_receipt.py --build C:/lq/CertR198
python experiments/circuit_distance_exact/lean_receipt.py          # the Lean-free gate
```

Toolchain `leanprover/lean4:v4.34.0`, mathlib `v4.34.0` (rev `5ed2965`), the same pin as
`lean/OneqCert`. When `ONEQ_MATHLIB_PACKAGES` names an already-built `.lake/packages`, the producer
links to it (a directory junction), so nothing is downloaded and no mathlib module is rebuilt. On a fresh machine: `lake exe cache get`, then `lake build`. **Caution:**
never `lake update` in a copy whose packages are that junction, and remove the link itself
(`cmd /c rmdir .lake\packages`) before deleting a copy with a tool that follows junctions. A clean
build of the 8 modules took 5 min 31 s on 2026-10-22, beside a 16-thread certificate run. `CertR198/AxiomCheck.lean` prints the axioms of every
theorem below: `[propext, Classical.choice, Quot.sound]` for all 29.

## What is proved

| theorem | file | content |
|---|---|---|
| `DEM.synSum_apply_eq_card` | Model | toggling: detector `d` of an XOR-sum is the parity of the members touching `d` |
| **`DEM.branch_odd`** (a) | Model | a firing detector of `synSum Y` is touched by an ODD number of members of `Y` |
| **`DEM.branch_complete`** (a) | Model | for ANY firing `v`, some member `m` of a completion `Y` is an allowed child at `v`, and `Y \ {m}` completes the child's state |
| **`DEM.greedy_le_card`** (b) | Bounds | rule D: the greedy independent-set count is at most the weight of any set touching every remaining firing detector, in any detector order and for any over-approximating `CT` |
| **`DEM.fracNum_le`** (b) | Bounds | rule G in exact integers: `Σ ⌊K / c_d⌋ ≤ K · card Y` for every allowed completion `Y` |
| **`DEM.bound_sound`** (b) | Bounds | `bound_ok` under CDP3 (independent set, `c_d = 0`, `⌈num / K⌉`) implies `rem < card Y` for every allowed completion |
| `DEM.greedy_le_card_of` (b') | Bounds | rule D': the same for a set of mechanisms satisfying `P` and a `ct` containing their co-touch -- with `P = (orbit ≥ o)` this is the packing over `CT_ORBIT[o]` |
| **`DEM.Feasible.raise1`** (b') | Bounds | rule G', one step: raising `y_d` by its allowance (the least slack of an allowed candidate of `d`) keeps every allowed mechanism's load `≤ K` |
| `DEM.Feasible.raiseList` (b') | Bounds | the same for any sequence of raises (the checker's order, or a prefix of it at an early exit) |
| `DEM.feasible_init` (b') | Bounds | the initial `⌊K / c_d⌋` are dual-feasible |
| **`DEM.sum_le_of_feasible`** (b') | Bounds | LP weak duality in integers: dual-feasible weights give `Σ_{firing d} y_d ≤ K · card Y` for every allowed completion |
| **`DEM.bound5_sound`** (b') | Bounds | `bound_ok` under CDP4/CDP5 (packing over `CT_ORBIT[o]`, `c_d = 0`, the raised cover sum above `K · rem`) implies `rem < card Y` |
| `DEM.leaf_sound` | Tree | rule L: the leaf condition excludes every completion of weight `≤ rem ≤ 2` by ANY mechanisms |
| `Valid.sound` | Tree | every completion of an accepted state within the budget CONTAINS a solution |
| `Valid.obs_eq_zero_of_syn_zero` | Tree | at a zero syndrome only rule E applies, so a valid tree forces `l = 0` (finding F1) |
| `root_sound` | Tree | the per-anchor tree (root codes 0/1/2) with budget `w - 1` |
| **`Valid.terminal3`** | Tree | CDP5's weight-3 terminal: branching on a firing `v` with every allowed child either `(0, 0)` or passing the radius-2 leaf is a valid tree at budget 3 |
| `obs_kernel_of_affine` | Anchor | `obs(g m) = M obs(m) + K syn(m)` implies the kernel form the checker decides |
| `Symmetry.isSol_image` | Anchor | a translate of a solution is a solution (uses `g⁻¹`: the group property) |
| **`Symmetry.anchoring`** (c) | Anchor | rules A + I: every solution has a same-weight translate containing the anchor of its minimal orbit `o` and using only orbits `≥ o`. Freeness is not a hypothesis. |
| `no_covered_irreducible_solution` | Main | core: no irreducible solution of weight `≤ w` with a member in a covered orbit |
| **`cert_sound`** (d) | Main | **a valid full certificate for `w` implies every undetectable logical error has weight `> w`** |
| `cert_sound_scoped` | Main | rule S, single model: no IRREDUCIBLE solution of weight `≤ w` touching a round `≤ maxround` |
| `Example.symmetry`, `Example.example_distance` | Example | non-vacuity end to end on a concrete model with a non-trivial group (`Z_3`) and only-affine observables: certificate + `cert_sound` give `d > 2`, and a weight-3 solution gives `d = 3` |
| `Example.scoped_claim_needs_irreducibility` | Example | a VALID rule-S certificate and a reducible weight-6 solution touching round 0: `cert_sound_scoped` is false without irreducibility (finding F2) |
| `ExampleDual.raise_prunes`, `plain_does_not_prune` | ExampleDual | the raise is not vacuous: two copies of a 3-mechanism gadget, all 10 detectors firing, `rem = 4`; packing counts 2 and the plain cover is exactly `2520 · 4`, the raise adds 1680 and prunes. `ccheck.c` agrees (CDP3 rules: no prune; CDP5 rules: prune) |
| `ExampleDual.completion_heavier`, `all_six_complete` | ExampleDual | so every completion has more than 4 mechanisms, and the 6 mechanisms are one |

`cert_sound` covers CDP4 and CDP5 certificates because `Valid` has the `prune5` constructor and
CDP5's terminal is a `Valid` tree (`Valid.terminal3`). Stated but unproved: nothing. Rule S is
covered as the single-model claim; its cross-model half (G9) is not.

### `cert_sound`, exactly

```lean
theorem cert_sound (hT : T.Sound E) (hS : Symmetry E G anchor) {norb : ℕ}
    (horb : ∀ m, E.orb m < norb) (hC : CertValid E T anchor w norb) :
    ∀ X : Finset M, E.IsSol X → w < X.card
```

`CertValid E T anchor w nanch := ∀ o < nanch, Valid E T o (E.syn (anchor o)) (E.obs (anchor o)) (w - 1)`.

## The model, mapped to `ccheck.c` (`493facf7` line numbers)

| Lean | `ccheck.c` | notes |
|---|---|---|
| `DEM.syn m : D → ZMod 2` | `S[m]`, lines 278-284 | the XOR of the `DET[m]` list; a repeated detector is refused (line 283), so `DEG[m] = |S[m]| ≥ 1` (line 279) |
| `DEM.obs m : Ob → ZMod 2` | `OB[m]`, line 285 | `Ob = Fin NO`, `NO ≤ 32` (line 277) |
| `DEM.orb m` | `ORB[m]`, line 285 | equal to the recomputed orbits, line 344 |
| `DEM.IsSol X` | the VERIFIED claim, lines 436-439 | `X` is a SET: weight = distinct mechanisms |
| `firing s` | the set bits of `s` | |
| `Tables.ct d` | `CT_ORBIT[0][d]`, lines 352-367 | the union over every orbit = the full co-touch relation, used under CDP3 (line 196) |
| `Tables.cto o d` | `CT_ORBIT[o][d]`, lines 352-367 | per-orbit co-touch (354-360), suffix union over orbits `≥ o` (361-365), `d` itself (366-367) |
| `Tables.cand d` | the mechanisms in `alist[d]`, lines 346-349 | sorted by (orbit, id) |
| `Tables.Sound` | by construction, lines 346-367 | the tables CONTAIN the true incidence |
| `greedy ct dl r` | `bound_ok` lines 191-198; `leaf_exists3` lines 167-174 | `dl` = firing detectors in increasing index; the loop takes the lowest bit still set |
| `DEM.cval s m`, `DEM.cmax T o s d` | `cover`, `max_cover`, lines 207-214 | over the allowed candidates (`ORB[m] < o` skipped, line 209) |
| `DEM.fracNum K T o s` | `sum`, lines 217-218 | `weight[d] = 2520 / max_cover`, C integer division = `ℕ` floor |
| `DEM.BoundOK 2520 T dl o s rem` | `bound_ok` under CDP3 (`modern_trace = 0`), lines 190-222 | disjuncts: line 199 / line 215 / line 221 (`sum > 2520 · rem` is `⌈sum / 2520⌉ > rem`) |
| `DEM.used`, `DEM.allowance` | lines 225-235 | `used` sums `weight[v]` over firing `v` of `m`; `allowance` starts at 2520 and takes the min over allowed candidates |
| `DEM.raise1`, `DEM.raiseList` | lines 223-241 | `weight[d] += allowance` in increasing index order; an early exit (line 239) raised a prefix |
| `DEM.BoundOK5 2520 T dl rl o s rem` | `bound_ok` under CDP4/CDP5 (`modern_trace ≥ 1`), lines 190-243 | disjuncts: line 199 with `CT_ORBIT[o]` / line 215 / line 221 (`rl = []`) or line 239 (`rl` the raised prefix). The `rem ≥ 4` and near-threshold trigger (line 222) only make the checker accept less |
| `DEM.LeafOK s l rem` | `leaf_exists`, lines 125-144 (negated), `single_hit` 113-121 | singles over all mechanisms (126); pairs `a` touching `d`, the lowest firing (128-142), `c ≠ a` (116) |
| `Valid.done` | line 249 | `s = 0`, `l = 0`; `s = 0`, `l ≠ 0` is rejection 1 |
| `Valid.spent` | line 251 | `rem <= 0` |
| `Valid.leaf` | line 252 | |
| `Valid.prune` | CDP3: line 262 (bit 0), line 264, root code 0 (lines 394-396) | the C also demands `rem >= 4` (`w > 3` at the root); not needed |
| `Valid.prune5` | the same lines under CDP4/CDP5; CDP5's terminal obstruction, lines 253-254 with 167-174 | four pairwise non-co-touching firing detectors at `rem = 3`: greedy count `≥ 4 > 3` |
| `Valid.expand v` | lines 263-269; root code 1, lines 397-401 and 410 | ANY firing `v`; the C recomputes `pick_c` (149-152), not needed for soundness; children = `alist[v]` with `ORB ≥ o` |
| `Valid.terminal3` | CDP5, `rem == 3`: lines 253-255, `leaf_exists3` lines 175-188 | branch on `pick_c`; an empty child with a flip is rejection 2 (185), with none it is `done`; any other child must fail `leaf_exists(…, 2)` (186) -- `leaf` at budget 2 |
| `CertValid E T anchor w nanch` | lines 390-428 | root codes 0/1/2 all denote one tree at depth 1, budget `w - 1`: code 2 runs `node` at depth 1 (line 422), code 1 runs each child at depth 2 (lines 423-424) |
| `Symmetry.syn_equiv` | lines 310-318 | the coordinate translation of detectors, invariance of the mechanism set; `img[m][g]` is `g • m` |
| `Symmetry.obs_kernel` | lines 304-336 | elimination on rows `[obs m \| syn m \| obs img]`: every zero-input combination has zero output |
| `Symmetry.orb_inv`, `anchor_mem` | lines 337-344 | |
| `G` a group | lines 295-302 | the declared translations form a subgroup of `Z_GL × Z_GM` |
| `horb : ∀ m, orb m < norb` | lines 343-344 | |
| `hscope` (rule S) | lines 376-387 | `want = 1 +` the last orbit whose anchor's first round is `<= maxround` |
| `rnd`, `hrnd` (rule S) | line 314 | translation keeps the round coordinate |

## Findings (2026-10-21; all closed in the C)

**F1: a zero-syndrome anchor under root code 1 was accepted unchecked.** A `DET` list that repeats a
detector XORs to a zero syndrome; `pick_c` then returns `-1`, `nch = 0` and the anchor got no task.
Witness `witnesses/zero_triv.txt` (one mechanism `2 0 0 1 0`, itself a weight-1 undetectable logical
error) with a 25-byte crafted proof: the checker printed VERIFIED. In Lean:
`Valid.obs_eq_zero_of_syn_zero` -- no valid tree exists at `(0, l ≠ 0)`. **Closed** (`f37072d2`):
every program refuses a repeated detector at load (`ccheck.c` line 283).

**F2: the rule-S verdict overstated what is checked.** Rule E ends a branch at `s = 0, l = 0`
although the remaining completion may be a lighter solution on its own; with only a prefix of
anchors covered, that solution may sit in an uncovered orbit. The checker proves "no IRREDUCIBLE
solution of weight `≤ w` whose earliest round is `≤ maxround`" (`cert_sound_scoped`). Witness
`witnesses/tri_z3.txt` (the model of `Example.lean`). **Closed:** the verdict (lines 436-437) and
RESULTS.md rule S now say "irreducible" (every minimum-weight solution is).

**F3: checks the checker makes that soundness does not need:** free action (line 340), anchor = first
member (341), orbit count (343), time-ordering of orbits (384), `rem >= 4` before a prune (262), the
deterministic pick (149-152, 263), trace length exactness (425), and the constant 2520 (any `K > 0`).
Harmless; they keep prover and checker in step. Freeness was load-bearing as input validation
before line 283 existed.

**F4: the part-13 working-tree prover crashed** (SIGSEGV) on `witnesses/zero_triv.txt`. **Closed** by
the same load-time refusal.

**F5: the part-13 leaf rewrite assumed `DEG[m] = |S[m]|`** (in `pop_after`, `single_hit`'s
`DEG[c] != pr`, and the span filters). Witness `witnesses/rep_z2.txt` (a group of 2, so freeness
passes) with a crafted proof: VERIFIED for a false claim. **Closed** by line 283, which makes the
assumption hold.

**Exposure (MEASURED, 2026-10-21):** the 825 model files then in the campaign directories had
457,272 mechanism lines and 0 repeated detectors, so F1 and F5 never affected a recorded result.

## Gap G1 closed for search certificates: an executable checker, run by the kernel (`Exec.lean`, `Kernel/`)

`Exec.lean` defines `cert`, a Lean PROGRAM: an exhaustive anchored search over a `CModel` (the exported projected DEM
as bitmasks), with no bounds and no trace. It also defines `cert2`, the same walk through an incidence table that
is supplied as data and checked for completeness (`incOK`). Both are proved sound end to end:

| theorem | content |
|---|---|
| `Exec.search_valid`, `Exec.search2_valid` | an accepting walk is a `Valid` tree (rules E, L, C + I; no prune) |
| `Exec.leafOK_sound`, `Exec.leafOK2_sound` | the decided leaf is `LeafOK`; v2 looks a pair's second member up through the remainder's lowest bit |
| `Exec.incOK_complete` | a checked table lists every mechanism under every detector it touches (all a "for all" check needs) |
| **`Exec.cert_sound_exec`, `Exec.cert2_sound_exec`** | `cert c w = true` (resp. `cert2`) implies every undetectable logical error of `c` has weight `> w` -- via `cert_sound` with the trivial group (`Exec.Triv`; orbit = index, every mechanism its own anchor) |

`Kernel/<Name>.lean` (generated by `gen_kernel_proof.py`) applies it to one real model: the model literal and its
table, `incOK` and one `search2` theorem per anchor, each `decide +kernel` in its own declaration (the kernel's
reduction cache is per declaration, so memory stays bounded), combined by `simp only` with the per-anchor lemmas (no
re-evaluation), then `cert2_sound_exec` and a witness at the next weight. The result is `distance`: the model's
minimum undetectable logical weight is exactly `d`, a Lean theorem on the standard axioms. No C code and no
compiler is trusted. What stays outside is the model: that the literal is the exported projected DEM of the
circuit (G10; the file's docstring pins the export's sha256, which the lane test compares with the campaign record).

Not used here, and why: bounds and symmetry (the full calculus). With them the walk shrinks ~20x (`cdist_prove`'s
proofs), but a kernel run would then evaluate `BoundOK5` and the symmetry hypotheses, more kernel work per node.
For models of a few hundred mechanisms at `w <= 5` the bare search is cheaper. Measured on c42a (210 mechanisms,
w = 5): ~12 s and ~1.1 GB above Mathlib's 1.5 GB baseline for the heaviest anchor. An uncapped single `decide` over
all anchors grew to 20 GB (2026-10-22), which is why the proofs are split per anchor.

## Not covered: the gap list

The Lean theorem is about `CertValid` and the hypotheses `Tables.Sound`, `Symmetry`, `horb` (and
`hscope`, `hrnd` for rule S). What is NOT proved:

- **G1. C ⇒ Lean.** Nothing proves that `ccheck.c` printing VERIFIED implies `CertValid` and the
  hypotheses. The table above is a line-by-line argument. Its executable cross-checks:
  `test_cdp5_rules_hold_against_brute_force_and_every_mutant_is_caught` holds the C's own `bound_ok`,
  `leaf_exists3` and `leaf_exists` against brute force (4,798 queries, both rule sets, nine mutants
  caught), and `ExampleDual` is one instance where Lean and the C agree that only the raise prunes.
- **G2. The trace decoding.** Bits in DFS preorder, one per node with `rem >= 3` (CDP3/CDP4) or
  `rem >= 4` (CDP5) and `s ≠ 0`; per-anchor root codes; per-child streams (lines 369-414). `Valid` is
  a tree, not a bitstream.
- **G3. The leaf lookups.** The Zobrist index with exact sparse confirmation (lines 56-71, 113-121),
  the popcount bookkeeping and the span filters (73-103) must find EVERY mechanism with a given
  syndrome; `LeafOK` assumes they do. Fuzzed against the sparse path
  (`test_checker_fast_and_sparse_leaf_paths_agree_on_mutated_proofs`) and against brute force (G1).
- **G4. Table construction.** `alist[d]` must be sorted by orbit (line 349) for `allowed_count`'s
  binary search (147-148). `CT_ORBIT[o]` must contain `d` (366-367), or the greedy loops would not
  terminate.
- **G5. The elimination.** That lines 325-335 decide exactly `Symmetry.obs_kernel`. Not formalized.
- **G6. The group action.** That `img[m][g]` is a group action of the declared subgroup on
  mechanisms. Needs distinct syndromes (duplicates are rejected at the orbit stage,
  `witnesses/dup_syn.txt`), nonzero syndromes (line 283) and unique coordinates (line 294).
- **G7. Arithmetic and parsing.** `long sum` cannot overflow (`ND · 2520 < 2^63`); `weight[d]` is an
  `unsigned short` and stays `≤ 2520` (a raised `d` lies in an allowed candidate whose load includes
  `weight[d]`). The `int32` header and `fscanf` parse are unmodelled; Lean uses `ℕ`.
- **G8. Execution.** The OpenMP task loop (417-429), the `volatile int bad` flag, and the compiler.
- **G9. Rule S across models.** Time-translation between `T`-round models and the "every T"
  conclusion (the window and fold lemmas). Only the single-model claim is formalized.
- **G10. Everything upstream of the DEM.** The circuits, the DEM derivation (stim /
  `dem_independent.py`), the same-type projection, the window lemma, the fold lemma, and witness
  lifting.

## Reproduce the witnesses

From this directory, against any revision `V` of the lane (`V = f37072d2` or later refuses F1/F4/F5's
inputs at load; `V = 0bc51a64` reproduces F1 and F2, the part-13 working tree F5):

```
R=<this repository>; P=experiments/circuit_distance_exact
git -C $R show V:$P/ccheck.c > ccheck.c; git -C $R show V:$P/cdist_prove.c > prove.c
gcc -O2 -o ccheck.exe ccheck.c && gcc -O2 -o prove.exe prove.c
./prove.exe witnesses/tri_z3.txt 6 s0.bin 0 && ./ccheck.exe witnesses/tri_z3.txt s0.bin full   # F2: VERIFIED, scoped to irreducible
./prove.exe witnesses/tri_z3.txt 6 f.bin                                                       # control: FOUND
python -I witnesses/craft.py zero4.bin 4 1 1 1 && ./ccheck.exe witnesses/zero_triv.txt zero4.bin full  # F1: refused at load (was VERIFIED)
python -I witnesses/craft.py rep3.bin 3 3 3 2 && ./ccheck.exe witnesses/rep_z2.txt rep3.bin full       # F5: refused at load
python -I witnesses/craft.py d.bin 4 2 2 1 && ./ccheck.exe witnesses/dup_syn.txt d.bin full    # REJECTED: orbit structure
```
