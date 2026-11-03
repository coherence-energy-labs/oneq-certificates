import Mathlib.Data.Finset.Card
import Mathlib.Tactic.Tauto
import Mathlib.Tactic.Linarith

/-!
  ONE_Q, P2: soundness of the safe dual bound, in the Lean kernel.

  The P2 checker (`src/oneq/qldpc_check.py`, `_exact_facet_scan`) certifies that a
  decoded correction is minimum weight when its weight EQUALS

      L = sum_r y_r (1 - |F_r|) - sum_i max(0, t_i - w_i)

  That is a proof only if L <= weight(e') for every correction e' consistent with the
  syndrome. This file proves exactly that, so the accept path's trust root becomes the
  kernel rather than an independent checker.

  Milestone 1a is `facet_violation`, the combinatorial heart. Milestone 1b is
  `safe_bound_le_weight`, the weighted weak duality that wraps it.

  The correction is modelled as a FINITE SET `e` of flipped variables rather than a
  0/1 vector: the parity condition becomes a statement about `(S ∩ e).card`, and the
  facet inequality becomes a statement about set differences, with no parity of a
  rational sum anywhere.

  Imports are targeted rather than the `Mathlib` umbrella: the prebuilt cache for this
  mathlib revision lacks the umbrella's own olean, and building it risks silently
  recompiling mathlib from source. Importing only what the proof uses is also what
  keeps the kernel's trusted input small.
-/

open Finset

namespace OneqCert

variable {ι : Type*} [DecidableEq ι]

/-- **Milestone 1a.** A correction satisfying a check's parity cannot coincide, on that
check, with a facet of the WRONG parity. So it must either flip something outside the
facet or leave something inside it unflipped.

This is the whole combinatorial content of the FWK odd-set inequality
`sum_{S \ F} x_i - sum_F x_i >= 1 - |F|`, stated over sets. -/
theorem facet_violation (S F e : Finset ι) (s : ℕ)
    (hFS : F ⊆ S) (hpar : F.card % 2 ≠ s) (he : (S ∩ e).card % 2 = s) :
    1 ≤ ((S \ F) ∩ e).card + (F \ e).card := by
  by_contra h
  have h1 : ((S \ F) ∩ e).card = 0 := by omega
  have h2 : (F \ e).card = 0 := by omega
  rw [Finset.card_eq_zero] at h1 h2
  -- If both are empty, the correction agrees with F exactly on the check.
  have hSe : S ∩ e = F := by
    ext i
    constructor
    · intro hi
      rw [Finset.mem_inter] at hi
      by_contra hiF
      have hmem : i ∈ (S \ F) ∩ e := by
        rw [Finset.mem_inter, Finset.mem_sdiff]
        exact ⟨⟨hi.1, hiF⟩, hi.2⟩
      rw [h1] at hmem
      simp at hmem
    · intro hi
      rw [Finset.mem_inter]
      refine ⟨hFS hi, ?_⟩
      by_contra hie
      have hmem : i ∈ F \ e := by
        rw [Finset.mem_sdiff]
        exact ⟨hi, hie⟩
      rw [h2] at hmem
      simp at hmem
  -- ...but then the correction's parity on the check is F's parity, which is wrong.
  rw [hSe] at he
  exact hpar he

/-- The facet inequality in the additive form the dual uses: the signed count over the
check is at least `1 - |F|`. Derived from `facet_violation` by counting, since
`|F \ e| = |F| - |F ∩ e|` for `F ∩ e ⊆ F`. -/
theorem facet_inequality (S F e : Finset ι) (s : ℕ)
    (hFS : F ⊆ S) (hpar : F.card % 2 ≠ s) (he : (S ∩ e).card % 2 = s) :
    (1 : ℤ) - F.card ≤ ((S \ F) ∩ e).card - (F ∩ e).card := by
  have hv := facet_violation S F e s hFS hpar he
  have hsplit : (F \ e).card + (F ∩ e).card = F.card := by
    rw [← Finset.card_union_of_disjoint]
    · congr 1
      ext i
      simp only [Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter]
      tauto
    · rw [Finset.disjoint_left]
      intro i hi hi'
      rw [Finset.mem_sdiff] at hi
      rw [Finset.mem_inter] at hi'
      exact hi.2 hi'.2
  omega

end OneqCert
