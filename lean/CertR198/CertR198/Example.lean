import CertR198.Main
import Mathlib.Data.ZMod.Defs
import Mathlib.Tactic.Ring

/-!
# A concrete model: the hypotheses are satisfiable, and rule S needs irreducibility

Six detectors `0..5` (round 0: `0,1,2`; round 1: `3,4,5`) and six mechanisms forming two
triangles: `p_i` (`0,1,2`) with syndromes `{0,1}, {1,2}, {0,2}` and no observable, `q_i`
(`3,4,5`) with syndromes `{3,4}, {4,5}, {3,5}`, and `q_0` flips the one observable. `Z_3`
rotates each triangle. `{q_0, q_1, q_2}` is an undetectable logical error, so `d = 3`.

The observables are NOT invariant under the rotation (`q_0` flips it, its image `q_1` does
not) -- only AFFINE, `obs (g m) = obs m + K_g (syn m)`, which is the situation RESULTS.md
describes for translated logical operators. `symmetry` checks all four hypotheses.

* `example_distance`: a valid full certificate for `w = 2` (two anchors, leaves at the root)
  and `cert_sound` give `d > 2`; with the weight-3 solution, `d = 3` exactly.
* `scoped_claim_needs_irreducibility`: the rule-S certificate for `w = 6`, `maxround = 0`
  (anchor `p_0` only -- the orbit of `q_0` starts in round 1) is VALID, and the whole set of
  six mechanisms is an undetectable logical error of weight `6 ≤ w` touching round 0. It is
  reducible (it contains `{q_0, q_1, q_2}`). So `cert_sound_scoped` is false without its
  irreducibility hypothesis, and the checker's scoped verdict "no undetectable logical error of
  weight <= w whose earliest round is <= maxround" (ccheck.c line 256) overstates what it
  proves. `ccheck.c` itself prints VERIFIED on this model (see SPEC.md, finding F2).
-/

open Finset

namespace CertR198.Example

/-- Incidence: mechanism `m` (triangle `m / 3`, position `i = m % 3`) touches the detectors
of its triangle at positions `i` and `i + 1 mod 3`: `p_0 = {0,1}`, `p_1 = {1,2}`,
`p_2 = {2,0}`, `q_0 = {3,4}`, `q_1 = {4,5}`, `q_2 = {5,3}`. -/
def touchB (m d : Fin 6) : Bool :=
  m.val / 3 == d.val / 3 && (d.val % 3 == m.val % 3 || d.val % 3 == (m.val % 3 + 1) % 3)

/-- The model. -/
def E : DEM (Fin 6) (Fin 6) (Fin 1) where
  syn := fun m d => if touchB m d then 1 else 0
  obs := fun m _ => if m = 3 then 1 else 0
  orb := fun m => if m.val < 3 then 0 else 1

instance (X : Finset (Fin 6)) : Decidable (E.IsSol X) := by
  unfold DEM.IsSol; infer_instance

/-- The exact tables. -/
abbrev T : Tables (Fin 6) (Fin 6) := Tables.canonical E

/-- `ANC = [p_0, q_0]`. -/
def anchor (o : ℕ) : Fin 6 := if o = 0 then 0 else 3

/-- Rotation of each triangle by `g`. -/
def rot (g : ZMod 3) (i : Fin 6) : Fin 6 :=
  if i.val < 3 then ⟨(i.val + g.val) % 3, by omega⟩
  else ⟨3 + (i.val - 3 + g.val) % 3, by omega⟩

/-- The group: `Z_3`, written multiplicatively. -/
abbrev G := Multiplicative (ZMod 3)

instance : Fintype G := inferInstanceAs (Fintype (ZMod 3))
instance : DecidableEq G := inferInstanceAs (DecidableEq (ZMod 3))

instance : MulAction G (Fin 6) where
  smul g i := rot (Multiplicative.toAdd g) i
  one_smul := by decide
  mul_smul := by decide

theorem smul_def (g : G) (i : Fin 6) : g • i = rot (Multiplicative.toAdd g) i := rfl

/-- `K_g` of the affine observable map: `obs (g • m) = obs m + K_g (syn m)`. -/
def Kg (g : G) : (Fin 6 → ZMod 2) →+ (Fin 1 → ZMod 2) where
  toFun s := fun _ =>
    if Multiplicative.toAdd g = 1 then s 3 else if Multiplicative.toAdd g = 2 then s 3 + s 5 else 0
  map_zero' := by funext; simp
  map_add' x y := by
    funext
    simp only [Pi.add_apply]
    split_ifs <;> ring

theorem obs_affine_apply :
    ∀ (g : G) (m : Fin 6) (i : Fin 1),
      E.obs (g • m) i = (AddMonoidHom.id _ (E.obs m) + Kg g (E.syn m)) i := by
  decide

theorem obs_affine (g : G) (m : Fin 6) :
    E.obs (g • m) = AddMonoidHom.id _ (E.obs m) + Kg g (E.syn m) :=
  funext (obs_affine_apply g m)

/-- All four symmetry hypotheses hold, with a non-trivial group and non-invariant
observables. -/
theorem symmetry : Symmetry E G anchor where
  syn_equiv := by decide
  obs_kernel g := obs_kernel_of_affine g (AddMonoidHom.id _) (Kg g) (obs_affine g)
  orb_inv := by decide
  anchor_mem := by decide

theorem horb (m : Fin 6) : E.orb m < 2 := by revert m; decide

/-! ## A full certificate for `w = 2`, and `d = 3` -/

theorem cert_w2 : CertValid E T anchor 2 2 := by
  intro o ho
  obtain rfl | rfl : o = 0 ∨ o = 1 := by omega
  · exact Valid.leaf (by decide) (by decide) ⟨by decide, by decide⟩
  · exact Valid.leaf (by decide) (by decide) ⟨by decide, by decide⟩

/-- The weight-3 solution `{q_0, q_1, q_2}`. -/
theorem sol_q : E.IsSol {3, 4, 5} := by decide

/-- **End to end on a concrete model:** the certificate and `cert_sound` give `d > 2`; the
weight-3 solution gives `d = 3`. -/
theorem example_distance :
    (∀ X : Finset (Fin 6), E.IsSol X → 2 < X.card) ∧ ∃ X, E.IsSol X ∧ X.card = 3 :=
  ⟨cert_sound (Tables.canonical_sound E) symmetry horb cert_w2, {3, 4, 5}, sol_q, rfl⟩

/-! ## The scoped (rule S) certificate for `w = 6`, `maxround = 0` -/

/-- Detector rounds. -/
def rnd (d : Fin 6) : ℕ := if d.val < 3 then 0 else 1

theorem rnd_inv (g : G) (d : Fin 6) : rnd (g • d) = rnd d := by revert g d; decide

/-- The search states reachable from `p_0` along `p`-mechanisms: the three `p` syndromes. -/
def Tri (s : Fin 6 → ZMod 2) : Prop := s = E.syn 0 ∨ s = E.syn 1 ∨ s = E.syn 2

instance (s : Fin 6 → ZMod 2) : Decidable (Tri s) := by unfold Tri; infer_instance

theorem tri_ne_zero {s : Fin 6 → ZMod 2} (hs : Tri s) : s ≠ 0 := by
  rcases hs with rfl | rfl | rfl <;> decide

theorem tri_leaf {s : Fin 6 → ZMod 2} (hs : Tri s) (r : ℕ) : E.LeafOK s 0 r := by
  rcases hs with rfl | rfl | rfl
  · exact ⟨by decide, fun _ => ⟨0, by decide, by decide⟩⟩
  · exact ⟨by decide, fun _ => ⟨1, by decide, by decide⟩⟩
  · exact ⟨by decide, fun _ => ⟨0, by decide, by decide⟩⟩

/-- Branching at a firing detector of a `Tri` state leads to `0` or another `Tri` state, and
only `p`-mechanisms (no observable) are children. -/
theorem tri_step {s : Fin 6 → ZMod 2} (hs : Tri s) :
    ∃ v, s v ≠ 0 ∧ ∀ m, E.syn m v ≠ 0 →
      (s + E.syn m = 0 ∧ E.obs m = 0) ∨ (Tri (s + E.syn m) ∧ E.obs m = 0) := by
  rcases hs with rfl | rfl | rfl
  · exact ⟨0, by decide, by decide⟩
  · exact ⟨1, by decide, by decide⟩
  · exact ⟨0, by decide, by decide⟩

/-- Every `Tri` state with no accumulated flip is accepted, at every budget. -/
theorem tri_valid (r : ℕ) : ∀ s, Tri s → Valid E T 0 s 0 r := by
  induction r with
  | zero => intro s hs; exact Valid.spent (tri_ne_zero hs)
  | succ r ih =>
    intro s hs
    by_cases hr : r + 1 ≤ 2
    · exact Valid.leaf (tri_ne_zero hs) hr (tri_leaf hs (r + 1))
    · obtain ⟨v, hv, hstep⟩ := tri_step hs
      refine Valid.expand v hv fun m _ hmv => ?_
      rw [Nat.add_sub_cancel, zero_add]
      rcases hstep m hmv with ⟨h0, hobs⟩ | ⟨htri, hobs⟩
      · rw [h0, hobs]
        exact Valid.done r
      · rw [hobs]
        exact ih _ htri

/-- The rule-S certificate: weight 6, anchors `o < 1`. -/
theorem cert_w6_scoped : CertValid E T anchor 6 1 := by
  intro o ho
  have h0 : o = 0 := by omega
  subst h0
  have hl : E.obs (anchor 0) = 0 := by decide
  rw [hl]
  exact tri_valid 5 _ (Or.inl rfl)

/-- The scope condition the checker's `want` computation guarantees for `maxround = 0`. -/
theorem scope0 : ∀ (o : ℕ) (d : Fin 6), E.syn (anchor o) d ≠ 0 → rnd d ≤ 0 → o < 1 := by
  intro o d h1 h2
  by_cases ho : o = 0
  · omega
  · exfalso
    have ha : anchor o = 3 := by simp [anchor, ho]
    rw [ha] at h1
    revert d
    decide

/-- **Rule S needs irreducibility.** Every hypothesis of `cert_sound_scoped` holds for the
full set of six mechanisms except irreducibility, and it IS an undetectable logical error of
weight `≤ 6` touching round 0. -/
theorem scoped_claim_needs_irreducibility :
    CertValid E T anchor 6 1 ∧
    (∀ (o : ℕ) (d : Fin 6), E.syn (anchor o) d ≠ 0 → rnd d ≤ 0 → o < 1) ∧
    ∃ X : Finset (Fin 6), E.IsSol X ∧ X.card ≤ 6 ∧
      (∃ m ∈ X, ∃ d, E.syn m d ≠ 0 ∧ rnd d ≤ 0) ∧ ¬ E.Irreducible X := by
  refine ⟨cert_w6_scoped, scope0, univ, by decide, by decide, ⟨0, mem_univ _, 0, by decide, by decide⟩, ?_⟩
  intro hirr
  have := hirr {3, 4, 5} (subset_univ _) sol_q
  revert this
  decide

end CertR198.Example
