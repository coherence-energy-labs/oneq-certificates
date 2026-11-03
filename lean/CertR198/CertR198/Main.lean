import CertR198.Anchor
import Mathlib.Data.Nat.Find

/-!
# R198 certificate calculus -- the main theorems

A CDP3 certificate for weight `w` covers the anchors `o < nanch` (`nanch = NORB`, or the
rule-S prefix). For each covered anchor the checker accepts one search tree rooted at depth 1
in state `(syn a, obs a)`, `a = ANC[o]`, with budget `w - 1` -- root code 0 (pruned), 1
(expanded, one stream per child) or 2 (one stream) all denote that one tree (lines 214-248).
`CertValid` is the statement that every covered tree is `Valid`.

* `no_covered_irreducible_solution` -- the core: no IRREDUCIBLE solution of weight `≤ w`
  whose minimal orbit is covered.
* `cert_sound` -- **theorem (d)**: with every orbit covered, NO solution of weight `≤ w`:
  the circuit distance exceeds `w`. (A minimum-weight solution is irreducible.)
* `cert_sound_scoped` -- rule S as the checker states it in one model: no irreducible
  solution of weight `≤ w` touching a detector of round `≤ maxround`. Irreducibility CANNOT be
  dropped: `Example.lean` gives a model and a valid scoped certificate with a reducible early
  solution of weight `≤ w`.
-/

open Finset

namespace CertR198

variable {M D Ob : Type*} [Fintype D] [DecidableEq D] [DecidableEq M]

/-- The certificate for weight `w` covering anchors `o < nanch` is accepted. -/
def CertValid (E : DEM M D Ob) (T : Tables M D) (anchor : ℕ → M) (w nanch : ℕ) : Prop :=
  ∀ o < nanch, Valid E T o (E.syn (anchor o)) (E.obs (anchor o)) (w - 1)

variable {E : DEM M D Ob} {T : Tables M D} {G : Type*} [Group G] [MulAction G M]
  [MulAction G D] {anchor : ℕ → M} {w nanch : ℕ}

/-- **Core.** A valid certificate leaves no irreducible solution of weight `≤ w` with a member
in a covered orbit (equivalently, whose minimal orbit is covered). -/
theorem no_covered_irreducible_solution (hT : T.Sound E) (hS : Symmetry E G anchor)
    (hC : CertValid E T anchor w nanch) {X : Finset M} (hX : E.IsSol X)
    (hirr : E.Irreducible X) (hw : X.card ≤ w) (hcov : ∃ m ∈ X, E.orb m < nanch) : False := by
  obtain ⟨o, g, -, hmin, ha, hXo, hcard, hsol⟩ := hS.anchoring hX
  obtain ⟨m, hm, hmlt⟩ := hcov
  have ho : o < nanch := lt_of_le_of_lt (hmin m hm) hmlt
  obtain ⟨Z, hZ, hZs⟩ := root_sound hT (hC o ho) hsol ha hXo (by rw [hcard]; exact hw)
  have hZeq := hS.irreducible_image g hirr Z (hZ.trans (Finset.erase_subset _ _)) hZs
  have hnot : anchor o ∉ Z := fun h => Finset.notMem_erase (anchor o) _ (hZ h)
  rw [hZeq] at hnot
  exact hnot ha

omit [Fintype D] [DecidableEq D] [DecidableEq M] in
/-- Every minimum-weight solution is irreducible; so if any solution has weight `≤ w`, an
irreducible one does. -/
theorem exists_irreducible (E : DEM M D Ob) {X : Finset M} (hX : E.IsSol X) :
    ∃ Y : Finset M, E.IsSol Y ∧ E.Irreducible Y ∧ Y.card ≤ X.card := by
  classical
  have hex : ∃ n, ∃ Y : Finset M, E.IsSol Y ∧ Y.card = n := ⟨X.card, X, hX, rfl⟩
  obtain ⟨Y, hY, hYc⟩ := Nat.find_spec hex
  have hmin : ∀ Z : Finset M, E.IsSol Z → Nat.find hex ≤ Z.card :=
    fun Z hZ => Nat.find_min' hex ⟨Z, hZ, rfl⟩
  refine ⟨Y, hY, fun Z hZ hZs => Finset.eq_of_subset_of_card_le hZ ?_, ?_⟩
  · rw [hYc]; exact hmin Z hZs
  · rw [hYc]; exact hmin X hX

/-- **Theorem (d): soundness of the full certificate.** If the model satisfies the checked
symmetry hypotheses, the tables contain the true incidence, every orbit id is below `norb`,
and the certificate covering all `norb` anchors is valid for weight `w`, then every
undetectable logical error has weight `> w`. -/
theorem cert_sound (hT : T.Sound E) (hS : Symmetry E G anchor) {norb : ℕ}
    (horb : ∀ m, E.orb m < norb) (hC : CertValid E T anchor w norb) :
    ∀ X : Finset M, E.IsSol X → w < X.card := by
  intro X hX
  by_contra hlt
  obtain ⟨Y, hY, hirr, hYX⟩ := exists_irreducible E hX
  obtain ⟨m, hm⟩ := hY.nonempty
  exact no_covered_irreducible_solution hT hS hC hY hirr (hYX.trans (not_lt.mp hlt))
    ⟨m, hm, horb m⟩

/-- **Rule S (time-shift scope), the single-model claim.** Detector rounds `rnd` are invariant
under the group (the translations keep the round coordinate, line 158), and every anchor
that touches a detector of round `≤ maxround` is covered -- what the checker's `want`
computation guarantees (lines 203-211: `nanch = 1 +` the last such anchor; the
time-ordering check there is NOT needed for this statement). Then no IRREDUCIBLE solution of
weight `≤ w` touches a detector of round `≤ maxround`. -/
theorem cert_sound_scoped (hT : T.Sound E) (hS : Symmetry E G anchor) (rnd : D → ℕ)
    (hrnd : ∀ (g : G) (d : D), rnd (g • d) = rnd d) {maxround : ℕ}
    (hscope : ∀ (o : ℕ) (d : D), E.syn (anchor o) d ≠ 0 → rnd d ≤ maxround → o < nanch)
    (hC : CertValid E T anchor w nanch) {X : Finset M} (hX : E.IsSol X)
    (hirr : E.Irreducible X) (hw : X.card ≤ w)
    (hearly : ∃ m ∈ X, ∃ d, E.syn m d ≠ 0 ∧ rnd d ≤ maxround) : False := by
  obtain ⟨m, hm, d, hmd, hd⟩ := hearly
  obtain ⟨g, hg⟩ := hS.anchor_mem m
  have h1 : E.syn (anchor (E.orb m)) (g • d) ≠ 0 := by
    rw [← hg, hS.syn_equiv]
    exact hmd
  have h2 : rnd (g • d) ≤ maxround := by
    rw [hrnd]
    exact hd
  exact no_covered_irreducible_solution hT hS hC hX hirr hw ⟨m, hm, hscope _ _ h1 h2⟩

end CertR198
