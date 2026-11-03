import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Data.Finset.Card

/-!
# R198 certificate calculus -- the model and rule C

`ccheck.c` (ONE_Q `experiments/circuit_distance_exact`) loads a detector error model (DEM):
mechanisms `m < NM`, each with a detector list `DET[m]` that is XORed into the bit-vector
syndrome `S[m]` (lines 127-129), an observable mask `OB[m]` (a `uint32`, line 129) and an orbit
id `ORB[m]` (line 129).

Here a syndrome is a GF(2)-vector `D → ZMod 2` and an observable mask a GF(2)-vector
`Ob → ZMod 2`; XOR is `+`. A set of mechanisms `X` is an *undetectable logical error*
(`IsSol`) when the XOR of its syndromes is zero and the XOR of its observables is not.
The circuit distance is the least `card X` over solutions.

A mechanism *touches* detector `d` when `syn m d ≠ 0` (bit `d` of `S[m]` is set).
-/

open Finset

namespace CertR198

/-- GF(2): every element is its own negative. -/
theorem zmod2_add_self (a : ZMod 2) : a + a = 0 := by
  revert a; decide

/-- GF(2)-vectors: `x + x = 0`. -/
theorem vec_add_self {α : Type*} (x : α → ZMod 2) : x + x = 0 :=
  funext fun i => zmod2_add_self (x i)

/-- Removing one member from an XOR-sum is XORing it in again. -/
theorem sum_erase_vec {M α : Type*} [DecidableEq M] (Y : Finset M) (f : M → α → ZMod 2)
    {m : M} (hm : m ∈ Y) : ∑ x ∈ Y.erase m, f x = (∑ x ∈ Y, f x) + f m := by
  rw [← Finset.add_sum_erase Y f hm, add_comm (f m), add_assoc, vec_add_self, add_zero]

/-- A detector error model, as `ccheck.c` holds it after loading (lines 124-133):
`syn m` is `S[m]`, `obs m` is `OB[m]`, `orb m` is `ORB[m]`. -/
structure DEM (M D Ob : Type*) where
  /-- `S[m]`: the syndrome of mechanism `m` over GF(2). -/
  syn : M → D → ZMod 2
  /-- `OB[m]`: the observables mechanism `m` flips. -/
  obs : M → Ob → ZMod 2
  /-- `ORB[m]`: the orbit id of mechanism `m` (checked against the recomputed orbits). -/
  orb : M → ℕ

namespace DEM

variable {M D Ob : Type*}

section defs

variable (E : DEM M D Ob)

/-- The XOR of the syndromes of a set of mechanisms. -/
def synSum (Y : Finset M) : D → ZMod 2 := ∑ m ∈ Y, E.syn m

/-- The XOR of the observable masks of a set of mechanisms. -/
def obsSum (Y : Finset M) : Ob → ZMod 2 := ∑ m ∈ Y, E.obs m

/-- `X` is an undetectable logical error: `D X = 0` and `L X ≠ 0`. -/
def IsSol (X : Finset M) : Prop := E.synSum X = 0 ∧ E.obsSum X ≠ 0

/-- `X` is a solution none of whose proper subsets is a solution. Every minimum-weight
solution is irreducible; rule E (an empty syndrome with no logical flip ends a branch) is
sound only relative to this notion, see `Valid.sound`. -/
def Irreducible (X : Finset M) : Prop := ∀ Z ⊆ X, E.IsSol Z → Z = X

/-- A *completion* of the search state `(s, l)` at anchor orbit `o` (rule I): a set of
mechanisms of orbit `≥ o` whose XOR completes the syndrome to zero (`synSum Y = s`) and whose
observables do not cancel the accumulated flip (`obsSum Y ≠ l`). The search state of a path
`P` is `(synSum P, obsSum P)`; `P ∪ Y` (disjoint) is then a solution. -/
def IsCompletion (o : ℕ) (s : D → ZMod 2) (l : Ob → ZMod 2) (Y : Finset M) : Prop :=
  (∀ m ∈ Y, o ≤ E.orb m) ∧ E.synSum Y = s ∧ E.obsSum Y ≠ l

end defs

variable {E : DEM M D Ob}

@[simp] theorem synSum_empty : E.synSum (∅ : Finset M) = 0 := by simp [synSum]

@[simp] theorem obsSum_empty : E.obsSum (∅ : Finset M) = 0 := by simp [obsSum]

theorem not_isSol_empty : ¬ E.IsSol (∅ : Finset M) := fun h => h.2 obsSum_empty

theorem IsSol.nonempty {X : Finset M} (h : E.IsSol X) : X.Nonempty := by
  rw [Finset.nonempty_iff_ne_empty]
  rintro rfl
  exact not_isSol_empty h

/-! ## Rule C: canonical branching -/

/-- Toggling semantics: detector `d` of an XOR-sum is the parity of the number of members
touching `d`. -/
theorem synSum_apply_eq_card (Y : Finset M) (d : D) :
    E.synSum Y d = ((Y.filter fun m => E.syn m d ≠ 0).card : ZMod 2) := by
  rw [Finset.natCast_card_filter, synSum, Finset.sum_apply]
  refine Finset.sum_congr rfl fun m _ => ?_
  generalize E.syn m d = a
  revert a; decide

/-- **Rule C, parity (theorem (a)).** If detector `v` fires in the XOR of `Y`, then an ODD
number of members of `Y` touch `v`. -/
theorem branch_odd {Y : Finset M} {v : D} (h : E.synSum Y v ≠ 0) :
    Odd (Y.filter fun m => E.syn m v ≠ 0).card := by
  rw [synSum_apply_eq_card] at h
  exact ZMod.natCast_ne_zero_iff_odd.mp h

/-- In particular at least one member of `Y` touches a firing detector. -/
theorem exists_mem_touching {Y : Finset M} {v : D} (h : E.synSum Y v ≠ 0) :
    ∃ m ∈ Y, E.syn m v ≠ 0 := by
  have hc : (Y.filter fun m => E.syn m v ≠ 0).card ≠ 0 := by
    intro h0
    apply h
    rw [synSum_apply_eq_card, h0, Nat.cast_zero]
  obtain ⟨m, hm⟩ := Finset.card_ne_zero.mp hc
  exact ⟨m, (Finset.mem_filter.mp hm).1, (Finset.mem_filter.mp hm).2⟩

/-- **Rule C, completeness (theorem (a)).** Let `Y` be a completion of the state `(s, l)` at
anchor orbit `o`, and let `v` be ANY firing detector of `s`. Then some member `m` of `Y`
is an allowed mechanism touching `v` -- i.e. one of the children the checker enumerates at
`v` (`ccheck.c` lines 115-117) -- and `Y.erase m` is a completion of that child's state
`(s + syn m, l + obs m)`, one lighter. So branching over the allowed mechanisms touching `v`
loses no completion, whichever firing `v` the pick rule chose. -/
theorem branch_complete [DecidableEq M] {o : ℕ} {s : D → ZMod 2} {l : Ob → ZMod 2}
    {Y : Finset M} (hY : E.IsCompletion o s l Y) {v : D} (hv : s v ≠ 0) :
    ∃ m ∈ Y, o ≤ E.orb m ∧ E.syn m v ≠ 0 ∧
      E.IsCompletion o (s + E.syn m) (l + E.obs m) (Y.erase m) ∧
      (Y.erase m).card + 1 = Y.card := by
  obtain ⟨hYo, hs, hl⟩ := hY
  obtain ⟨m, hmY, hmv⟩ := exists_mem_touching (E := E) (Y := Y) (v := v) (by rw [hs]; exact hv)
  refine ⟨m, hmY, hYo m hmY, hmv, ⟨fun x hx => hYo x (Finset.mem_of_mem_erase hx), ?_, ?_⟩, ?_⟩
  · simp only [synSum] at hs ⊢
    rw [sum_erase_vec Y E.syn hmY, hs]
  · simp only [obsSum] at hl ⊢
    rw [sum_erase_vec Y E.obs hmY]
    intro h
    exact hl (add_right_cancel h)
  · rw [Finset.card_erase_of_mem hmY]
    have := Finset.card_pos.mpr ⟨m, hmY⟩
    omega

end DEM

end CertR198
