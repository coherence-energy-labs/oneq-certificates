import CertR198.Bounds

/-!
# R198 certificate calculus -- the search tree: rules E, L, D/G, C as one inductive predicate

`ccheck.c` `node(s, l, depth, w, o)` (lines 102-118) walks one subtree of the DFS trace.
`Valid E T o s l r` is the predicate "the subtree at state `(s, l)` with remaining budget
`r = w - depth`, searched from anchor orbit `o`, is accepted". Its constructors are the
checker's acceptance paths:

| constructor | `ccheck.c`                    | rule |
|-------------|-------------------------------|------|
| `done`      | line 105, `s = 0` and `l = 0` | E    |
| `spent`     | line 107, `rem <= 0`          | --   |
| `leaf`      | line 108, `rem <= 2`, `leaf_exists` false | L |
| `prune`     | line 111 (bit 0), line 113, root code 0 (line 219) | D/G |
| `prune5`    | the same under a CDP4/CDP5 header (R205/R206), and CDP5's terminal obstruction | D'/G' |
| `expand`    | lines 112-117, root code 1 (lines 221-234) | C + I |

CDP5's weight-3 terminal (`leaf_exists3`, no trace bit) needs no constructor of its own: its
obstruction is a `prune5` (greedy count `≥ 4 > 3`), and its enumeration is an `expand` whose
children are `done` or `leaf` (`Valid.terminal3`).

The predicate is deliberately weaker than the checker (fewer side conditions: the checker also
demands `rem >= 4` before a prune, the deterministic pick for `v`, the trace bit, ...); every
extra C condition only makes the checker reject more, so it cannot hurt soundness.

`Valid.sound`: every completion of an accepted state within the budget CONTAINS a solution.
(It does not say "there is no completion": rule E stops at `s = 0, l = 0` although the
remaining completion `Y` may be a lighter solution on its own. That is sound for the
distance -- `Y` is lighter -- and is why the scoped theorem (rule S) needs irreducibility.)
-/

open Finset

namespace CertR198

variable {M D Ob : Type*}

namespace DEM

/-- What `leaf_exists(s, l, rem)` (lines 64-76) refutes, as a property of the model:
no single mechanism completes `(s, l)`, and (when `rem ≥ 2`) for some firing detector `d`
(the checker uses the lowest) no pair `a ≠ b` with `a` touching `d` does. Both lookups range
over ALL mechanisms, not only allowed ones (the checker's `alist[d]` and its sorted-syndrome
index are unrestricted). -/
def LeafOK (E : DEM M D Ob) (s : D → ZMod 2) (l : Ob → ZMod 2) (rem : ℕ) : Prop :=
  (∀ m, E.syn m = s → E.obs m = l) ∧
  (2 ≤ rem → ∃ d, s d ≠ 0 ∧
    ∀ a b, E.syn a d ≠ 0 → a ≠ b → E.syn a + E.syn b = s → E.obs a + E.obs b = l)

variable {E : DEM M D Ob}

/-- **Rule L.** At remaining budget `≤ 2` with a nonzero syndrome, `LeafOK` excludes every
completion (by ANY mechanisms) within the budget. A pair `{a, b}` with `syn a + syn b = s`
has exactly one member touching a firing `d`, so trying every `a` touching `d` is complete. -/
theorem leaf_sound [DecidableEq M] {s : D → ZMod 2} {l : Ob → ZMod 2} {rem : ℕ} (hs0 : s ≠ 0)
    (hr : rem ≤ 2) (h : E.LeafOK s l rem) {Y : Finset M} (hs : E.synSum Y = s)
    (hl : E.obsSum Y ≠ l) (hc : Y.card ≤ rem) : False := by
  obtain ⟨hone, htwo⟩ := h
  have hc' : Y.card = 0 ∨ Y.card = 1 ∨ Y.card = 2 := by omega
  rcases hc' with h0 | h1 | h2
  · rw [Finset.card_eq_zero] at h0
    subst h0
    exact hs0 (by rw [← hs, synSum_empty])
  · obtain ⟨m, rfl⟩ := Finset.card_eq_one.mp h1
    simp only [synSum, obsSum, Finset.sum_singleton] at hs hl
    exact hl (hone m hs)
  · obtain ⟨a, b, hab, rfl⟩ := Finset.card_eq_two.mp h2
    simp only [synSum, obsSum, Finset.sum_pair hab] at hs hl
    obtain ⟨d, hd, hpair⟩ := htwo (by omega)
    by_cases had : E.syn a d = 0
    · have hbd : E.syn b d ≠ 0 := by
        intro hb
        apply hd
        rw [← hs, Pi.add_apply, had, hb, add_zero]
      apply hl
      rw [add_comm]
      exact hpair b a hbd (Ne.symm hab) (by rw [add_comm]; exact hs)
    · exact hl (hpair a b had hab hs)

end DEM

variable [Fintype D] [DecidableEq D]

/-- The checker's scale for the fractional bound (`ccheck.c` lines 96, 98). -/
def scale : ℕ := 2520

/-- **The valid-tree predicate.** `Valid E T o s l r`: the subtree at search state `(s, l)`
with remaining budget `r`, anchor orbit `o`, is accepted by the checker's rules. -/
inductive Valid (E : DEM M D Ob) (T : Tables M D) (o : ℕ) :
    (D → ZMod 2) → (Ob → ZMod 2) → ℕ → Prop
  /-- Rule E (line 105): empty syndrome and no logical flip -- the branch ends. (Empty
  syndrome WITH a flip is the checker's rejection 1: a solution was reached.) -/
  | done (r : ℕ) : Valid E T o 0 0 r
  /-- Line 107: no budget left and the syndrome is not yet zero. -/
  | spent {s : D → ZMod 2} {l : Ob → ZMod 2} : s ≠ 0 → Valid E T o s l 0
  /-- Rule L (line 108): remaining budget at most 2, decided by the leaf procedure. -/
  | leaf {s : D → ZMod 2} {l : Ob → ZMod 2} {r : ℕ} :
      s ≠ 0 → r ≤ 2 → E.LeafOK s l r → Valid E T o s l r
  /-- Rules D/G (line 111 with trace bit 0; line 113; root code 0 at line 219). The greedy
  pass may meet the detectors in any order `dl`. -/
  | prune {s : D → ZMod 2} {l : Ob → ZMod 2} {r : ℕ} (dl : List D) :
      E.BoundOK scale T dl o s r → Valid E T o s l r
  /-- Rules D'/G' (R205/R206): a prune under a CDP4/CDP5 header -- packing over `CT_ORBIT[o]`,
  the cover with the detectors `rl` raised -- and CDP5's terminal obstruction (four firing
  detectors pairwise outside `CT_ORBIT[o]` at remaining budget 3). -/
  | prune5 {s : D → ZMod 2} {l : Ob → ZMod 2} {r : ℕ} (dl rl : List D) :
      E.BoundOK5 scale T dl rl o s r → Valid E T o s l r
  /-- Rule C with rule I (lines 112-117; root code 1 at lines 221-234): branch on ANY firing
  detector `v`; every mechanism of orbit `≥ o` touching `v` is a child, one deeper. -/
  | expand {s : D → ZMod 2} {l : Ob → ZMod 2} {r : ℕ} (v : D) : s v ≠ 0 →
      (∀ m, o ≤ E.orb m → E.syn m v ≠ 0 →
        Valid E T o (s + E.syn m) (l + E.obs m) (r - 1)) →
      Valid E T o s l r

variable {E : DEM M D Ob} {T : Tables M D}

/-- **Node soundness.** Every completion `Y` (allowed mechanisms, `synSum Y = s`,
`obsSum Y ≠ l`) of an accepted state within the budget contains a solution `Z ⊆ Y`. -/
theorem Valid.sound [DecidableEq M] (hT : T.Sound E) {o : ℕ} {s : D → ZMod 2}
    {l : Ob → ZMod 2} {r : ℕ} (h : Valid E T o s l r) :
    ∀ Y : Finset M, E.IsCompletion o s l Y → Y.card ≤ r → ∃ Z ⊆ Y, E.IsSol Z := by
  induction h with
  | done r =>
    intro Y hY _
    exact ⟨Y, Finset.Subset.refl Y, hY.2.1, hY.2.2⟩
  | spent hs0 =>
    intro Y hY hc
    rw [Nat.le_zero, Finset.card_eq_zero] at hc
    subst hc
    exact absurd (by rw [← hY.2.1, DEM.synSum_empty]) hs0
  | leaf hs0 hr hleaf =>
    intro Y hY hc
    exact (DEM.leaf_sound hs0 hr hleaf hY.2.1 hY.2.2 hc).elim
  | prune dl hb =>
    intro Y hY hc
    exact absurd hc (not_le.mpr (DEM.bound_sound hT (by decide) hb hY.1 hY.2.1))
  | prune5 dl rl hb =>
    intro Y hY hc
    exact absurd hc (not_le.mpr (DEM.bound5_sound hT hb hY.1 hY.2.1))
  | expand v hv _ ih =>
    intro Y hY hc
    obtain ⟨m, _, hmo, hmv, hY', hc'⟩ := DEM.branch_complete hY hv
    obtain ⟨Z, hZ, hZs⟩ := ih m hmo hmv (Y.erase m) hY' (by omega)
    exact ⟨Z, hZ.trans (Finset.erase_subset m Y), hZs⟩

omit [DecidableEq D] in
@[simp] theorem firing_zero : firing (0 : D → ZMod 2) = ∅ := by
  ext d
  simp

omit [Fintype D] in
theorem greedy_empty (ct : D → Finset D) (dl : List D) : greedy ct dl ∅ = 0 := by
  induction dl with
  | nil => rfl
  | cons d ds ih => simp [greedy, ih]

/-- **At a zero syndrome only rule E applies.** A valid tree at state `(0, l)` forces
`l = 0`: the bound cannot prune (every term is zero), there is no firing detector to branch
on, and leaves need a nonzero syndrome. `ccheck.c`'s root code 1 (lines 221-225) does NOT
enforce this: for an anchor with zero syndrome `pick_c` returns `-1`, `nch = 0`, and the
anchor is accepted with no check at all -- finding F1 in SPEC.md. -/
theorem Valid.obs_eq_zero_of_syn_zero {o : ℕ} {l : Ob → ZMod 2} {r : ℕ}
    (h : Valid E T o 0 l r) : l = 0 := by
  cases h with
  | done => rfl
  | spent hs => exact absurd rfl hs
  | leaf hs _ _ => exact absurd rfl hs
  | prune dl hb =>
    exfalso
    rcases hb with hb | ⟨d, hd, _⟩ | hb
    · rw [firing_zero, greedy_empty] at hb
      exact Nat.not_lt_zero _ hb
    · simp at hd
    · simp only [DEM.fracNum, firing_zero, Finset.sum_empty, zero_add] at hb
      simp [scale] at hb
  | prune5 dl rl hb =>
    exfalso
    rcases hb with hb | ⟨d, hd, _⟩ | hb
    · rw [firing_zero, greedy_empty] at hb
      exact Nat.not_lt_zero _ hb
    · simp at hd
    · simp only [firing_zero, Finset.sum_empty] at hb
      exact Nat.not_lt_zero _ hb
  | expand v hv _ => exact absurd rfl hv

/-- **CDP5's weight-3 terminal** (R205, `ccheck.c` `leaf_exists3`, no trace bit). Short of the
obstruction (a `prune5`), the checker branches on a firing `v` -- its own pick -- and decides
every allowed child itself: a child with an empty syndrome must carry no flip (rule E; a flip is
rejection 2), any other must pass the radius-2 leaf (rule L, the checker's `leaf_exists`). That
acceptance is a valid tree at remaining budget 3, so `cert_sound` covers CDP5 certificates. -/
theorem Valid.terminal3 {o : ℕ} {s : D → ZMod 2} {l : Ob → ZMod 2} (v : D) (hv : s v ≠ 0)
    (h : ∀ m, o ≤ E.orb m → E.syn m v ≠ 0 →
      (s + E.syn m = 0 ∧ l + E.obs m = 0) ∨
        (s + E.syn m ≠ 0 ∧ E.LeafOK (s + E.syn m) (l + E.obs m) 2)) :
    Valid E T o s l 3 := by
  refine Valid.expand v hv fun m hm hmv => ?_
  rcases h m hm hmv with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · rw [h1, h2]
    exact Valid.done _
  · exact Valid.leaf h1 (by decide) h2

/-- **The root.** The search for anchor `a` starts at depth 1 in state `(syn a, obs a)` with
budget `w - 1` (root codes 0/1/2, lines 214-248). If that tree is valid, every solution of
weight `≤ w` that contains `a` and uses only orbits `≥ o` has a solution among `X \ {a}`. -/
theorem root_sound [DecidableEq M] (hT : T.Sound E) {o w : ℕ} {a : M}
    (h : Valid E T o (E.syn a) (E.obs a) (w - 1)) {X : Finset M} (hX : E.IsSol X)
    (ha : a ∈ X) (hXo : ∀ m ∈ X, o ≤ E.orb m) (hw : X.card ≤ w) :
    ∃ Z ⊆ X.erase a, E.IsSol Z := by
  refine h.sound hT (X.erase a) ⟨fun m hm => hXo m (Finset.mem_of_mem_erase hm), ?_, ?_⟩ ?_
  · have h0 := hX.1
    simp only [DEM.synSum] at h0 ⊢
    rw [sum_erase_vec X E.syn ha, h0, zero_add]
  · have h0 := hX.2
    simp only [DEM.obsSum] at h0 ⊢
    rw [sum_erase_vec X E.obs ha]
    intro h'
    exact h0 (add_right_cancel (h'.trans (zero_add _).symm))
  · rw [Finset.card_erase_of_mem ha]
    omega

end CertR198
