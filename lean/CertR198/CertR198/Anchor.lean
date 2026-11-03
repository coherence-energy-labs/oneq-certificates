import CertR198.Tree
import Mathlib.Algebra.Group.Action.Basic

/-!
# R198 certificate calculus -- rules A + I: anchoring by a symmetry group

`ccheck.c` section (1) (lines 136-188) checks, for every translation `g` of the declared
subgroup of `Z_GL × Z_GM`:

* **detector structure.** `g` maps each detector by its `(site, type, round)` coordinates and
  the image of every mechanism's syndrome is again a mechanism's syndrome (lines 157-162);
  `img[m][g]` is that mechanism. Here: `G` acts on mechanisms and on detectors, and
  `syn (g • m) (g • d) = syn m d` (`syn_equiv`).
* **affine observables.** By GF(2) elimination on the rows `[obs m | syn m | obs (g m)]`
  (lines 152-180): every combination of rows with zero input part has zero output part. That
  is exactly: for every set `Y`, `obsSum Y = 0 ∧ synSum Y = 0 → Σ_{m ∈ Y} obs (g • m) = 0`
  (`obs_kernel`). It is implied by the RESULTS.md form `obs(g m) = M obs(m) + K syn(m)`
  (`obs_kernel_of_affine`).
* **orbits and anchors.** Orbits are recomputed from the images and must equal `ORB`;
  each anchor `ANC[o]` is a member of orbit `o` (lines 181-188). Here: `orb` is invariant
  (`orb_inv`) and every mechanism is a translate of its orbit's anchor (`anchor_mem`).

The checker ALSO checks that the action is free (line 184), that the anchor is the orbit's
first member (line 185) and the orbit count (line 187). None of these is needed for
soundness: `Symmetry` omits them and the theorems hold without them.
-/

open Finset

namespace CertR198

variable {M D Ob : Type*}

/-- The symmetry hypotheses `ccheck.c` section (1) establishes, for a group `G` acting on
mechanisms and detectors, and the anchor map `anchor o = ANC[o]`. -/
structure Symmetry (E : DEM M D Ob) (G : Type*) [Group G] [MulAction G M] [MulAction G D]
    (anchor : ℕ → M) : Prop where
  /-- The action preserves the detector structure (lines 157-162). -/
  syn_equiv : ∀ (g : G) (m : M) (d : D), E.syn (g • m) (g • d) = E.syn m d
  /-- Observables transform affinely: the elimination check of lines 152-180. -/
  obs_kernel : ∀ (g : G) (Y : Finset M), E.synSum Y = 0 → E.obsSum Y = 0 →
    ∑ m ∈ Y, E.obs (g • m) = 0
  /-- Orbit ids are invariant (lines 181-188). -/
  orb_inv : ∀ (g : G) (m : M), E.orb (g • m) = E.orb m
  /-- Every mechanism is a translate of its orbit's anchor (lines 185, 188). -/
  anchor_mem : ∀ m : M, ∃ g : G, g • m = anchor (E.orb m)

variable {E : DEM M D Ob} {G : Type*} [Group G] [MulAction G M] [MulAction G D]
  {anchor : ℕ → M}

omit [MulAction G D] in
/-- The RESULTS.md statement of the affine observable map, `obs(g m) = M obs(m) + K syn(m)`
with `M`, `K` additive (GF(2)-linear), implies the kernel form the checker decides. -/
theorem obs_kernel_of_affine (g : G) (Mg : (Ob → ZMod 2) →+ (Ob → ZMod 2))
    (Kg : (D → ZMod 2) →+ (Ob → ZMod 2))
    (h : ∀ m : M, E.obs (g • m) = Mg (E.obs m) + Kg (E.syn m)) (Y : Finset M)
    (hs : E.synSum Y = 0) (ho : E.obsSum Y = 0) : ∑ m ∈ Y, E.obs (g • m) = 0 := by
  simp only [h]
  rw [Finset.sum_add_distrib, ← map_sum, ← map_sum]
  simp only [DEM.synSum, DEM.obsSum] at hs ho
  rw [hs, ho, map_zero, map_zero, add_zero]

variable [DecidableEq M]

theorem sum_image_smul {α : Type*} [AddCommMonoid α] (g : G) (Y : Finset M) (f : M → α) :
    ∑ m ∈ Y.image (g • ·), f m = ∑ m ∈ Y, f (g • m) :=
  Finset.sum_image fun _ _ _ _ hxy => MulAction.injective g hxy

theorem image_smul_inv (g : G) (Y : Finset M) : (Y.image (g • ·)).image (g⁻¹ • ·) = Y := by
  rw [Finset.image_image]
  simp [Function.comp_def]

namespace Symmetry

theorem synSum_image_apply (hS : Symmetry E G anchor) (g : G) (Y : Finset M) (d : D) :
    E.synSum (Y.image (g • ·)) d = E.synSum Y (g⁻¹ • d) := by
  simp only [DEM.synSum, Finset.sum_apply]
  rw [sum_image_smul g Y (fun m => E.syn m d)]
  refine Finset.sum_congr rfl fun m _ => ?_
  rw [← hS.syn_equiv g m (g⁻¹ • d), smul_inv_smul]

theorem synSum_image_eq_zero (hS : Symmetry E G anchor) (g : G) {Y : Finset M}
    (hY : E.synSum Y = 0) : E.synSum (Y.image (g • ·)) = 0 := by
  funext d
  rw [hS.synSum_image_apply, hY]
  rfl

/-- A translate of a solution is a solution. Undetectability is carried by `syn_equiv`;
"flips some logical" is carried by `obs_kernel` applied to `g⁻¹` -- this is where the
declared translations must form a GROUP (the checker's subgroup check, line 143). -/
theorem isSol_image (hS : Symmetry E G anchor) (g : G) {X : Finset M} (hX : E.IsSol X) :
    E.IsSol (X.image (g • ·)) := by
  refine ⟨hS.synSum_image_eq_zero g hX.1, fun h0 => hX.2 ?_⟩
  have h1 := hS.obs_kernel g⁻¹ (X.image (g • ·)) (hS.synSum_image_eq_zero g hX.1) h0
  rw [sum_image_smul g X (fun m => E.obs (g⁻¹ • m))] at h1
  simp only [inv_smul_smul] at h1
  exact h1

/-- A translate of an irreducible solution is irreducible. -/
theorem irreducible_image (hS : Symmetry E G anchor) (g : G) {X : Finset M}
    (hX : E.Irreducible X) : E.Irreducible (X.image (g • ·)) := by
  intro Z hZ hZs
  have h1 : Z.image (g⁻¹ • ·) ⊆ X := by
    intro x hx
    obtain ⟨z, hz, rfl⟩ := Finset.mem_image.mp hx
    obtain ⟨y, hy, rfl⟩ := Finset.mem_image.mp (hZ hz)
    rw [inv_smul_smul]
    exact hy
  have h2 := hX _ h1 (hS.isSol_image g⁻¹ hZs)
  rw [← h2, Finset.image_image]
  simp [Function.comp_def]

/-- **Rules A + I (theorem (c)).** Every solution `X` has a translate `g • X` that is a
solution of the same weight, contains the anchor of `X`'s MINIMAL orbit `o`, and uses only
orbits `≥ o`. So searching from anchor `o` over mechanisms of orbit `≥ o` (rule I) covers it.
Freeness of the action is not used. -/
theorem anchoring (hS : Symmetry E G anchor) {X : Finset M} (hX : E.IsSol X) :
    ∃ (o : ℕ) (g : G), (∃ m ∈ X, E.orb m = o) ∧ (∀ m ∈ X, o ≤ E.orb m) ∧
      anchor o ∈ X.image (g • ·) ∧ (∀ m ∈ X.image (g • ·), o ≤ E.orb m) ∧
      (X.image (g • ·)).card = X.card ∧ E.IsSol (X.image (g • ·)) := by
  obtain ⟨m0, hm0, hmin⟩ := Finset.exists_min_image X E.orb hX.nonempty
  obtain ⟨g, hg⟩ := hS.anchor_mem m0
  refine ⟨E.orb m0, g, ⟨m0, hm0, rfl⟩, hmin, ?_, ?_, ?_, hS.isSol_image g hX⟩
  · rw [← hg]
    exact Finset.mem_image_of_mem _ hm0
  · intro m hm
    obtain ⟨x, hx, rfl⟩ := Finset.mem_image.mp hm
    rw [hS.orb_inv]
    exact hmin x hx
  · exact Finset.card_image_of_injective X (MulAction.injective g)

end Symmetry

end CertR198
