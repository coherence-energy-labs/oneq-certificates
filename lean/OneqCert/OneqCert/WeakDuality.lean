import OneqCert.SafeBound
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.SymmDiff

/-!
  ONE_Q, P2: milestone 1b -- the weighted weak duality behind the safe dual bound.

  `src/oneq/qldpc_check.py`, `_exact_facet_scan`, returns

      L = sum_{r in R} y_r (1 - |F_r|) - sum_{i in V} max(0, t_i - w_i)
      t_i = sum_{r in R} y_r * (-1 if i in F_r, +1 if i in S_r \ F_r, 0 otherwise)

  with V = (loaded variables) ∪ (weighted variables). `safe_bound_le_weight` proves
  L <= weight(e) for every correction e inside V that meets every facet's parity, for
  ANY rational weights and ANY y >= 0. No sign condition on w is needed: that is the
  point of summing the slack over all of V, and it is why the scan was changed on
  2026-09-22 (before, the sum ran over loaded variables only, which is sound only when
  every weight is non-negative -- and check_qldpc admitted weights down to -tol).

  A facet r here is a check support `S r` with its required parity `s r`; a redundant
  check (an XOR of rows) is just another such pair, and `xor_parity` shows the XOR of
  two satisfied rows is a satisfied row, so the scan's XOR facets meet the hypothesis.
-/

open Finset

namespace OneqCert

variable {ι ρ : Type*} [DecidableEq ι]

/-- The scan's per-variable coefficient of one facet: `-1` inside `F`, `+1` on the
rest of the check, `0` off the check. Exactly the `(-y if i in Fs else y)` load. -/
def coef (S F : Finset ι) (i : ι) : ℚ :=
  if i ∈ F then -1 else if i ∈ S then 1 else 0

/-- The load `t_i` the scan accumulates. -/
def load (R : Finset ρ) (S F : ρ → Finset ι) (y : ρ → ℚ) (i : ι) : ℚ :=
  ∑ r ∈ R, y r * coef (S r) (F r) i

/-- The safe bound exactly as the scan computes it. -/
def safeBound (R : Finset ρ) (S F : ρ → Finset ι) (y : ρ → ℚ) (w : ι → ℚ)
    (V : Finset ι) : ℚ :=
  (∑ r ∈ R, y r * (1 - ((F r).card : ℚ))) -
    ∑ i ∈ V, max 0 (load R S F y i - w i)

/-- Summing one facet's coefficients over a correction gives its signed count. -/
theorem sum_coef (S F e : Finset ι) (hFS : F ⊆ S) :
    ∑ i ∈ e, coef S F i = (((S \ F) ∩ e).card : ℚ) - ((F ∩ e).card : ℚ) := by
  have hsplit : ∀ i ∈ e, coef S F i =
      (if i ∈ S \ F then (1 : ℚ) else 0) - (if i ∈ F then (1 : ℚ) else 0) := by
    intro i _
    unfold coef
    by_cases hF : i ∈ F
    · have hS : i ∈ S := hFS hF
      simp [hF]
    · by_cases hS : i ∈ S <;> simp [hF, hS, Finset.mem_sdiff]
  rw [Finset.sum_congr rfl hsplit, Finset.sum_sub_distrib, Finset.sum_boole,
    Finset.sum_boole, Finset.filter_mem_eq_inter, Finset.filter_mem_eq_inter,
    Finset.inter_comm e, Finset.inter_comm e]

/-- One facet, in the rational form the dual sums: `1 - |F| <= sum_{i in e} coef`. -/
theorem facet_sum_le (S F e : Finset ι) (s : ℕ)
    (hFS : F ⊆ S) (hpar : F.card % 2 ≠ s) (he : (S ∩ e).card % 2 = s) :
    1 - (F.card : ℚ) ≤ ∑ i ∈ e, coef S F i := by
  rw [sum_coef S F e hFS]
  have h := facet_inequality S F e s hFS hpar he
  exact_mod_cast h

/-- **Milestone 1b.** The safe dual bound never exceeds the weight of a correction
that satisfies every facet's check. -/
theorem safe_bound_le_weight (R : Finset ρ) (S F : ρ → Finset ι) (s : ρ → ℕ)
    (y : ρ → ℚ) (w : ι → ℚ) (V e : Finset ι)
    (hy : ∀ r ∈ R, 0 ≤ y r)
    (hFS : ∀ r ∈ R, F r ⊆ S r)
    (hpar : ∀ r ∈ R, (F r).card % 2 ≠ s r)
    (he : ∀ r ∈ R, (S r ∩ e).card % 2 = s r)
    (heV : e ⊆ V) :
    safeBound R S F y w V ≤ ∑ i ∈ e, w i := by
  unfold safeBound
  -- (1) weight the facet inequalities by y >= 0 and sum them
  have h1 : (∑ r ∈ R, y r * (1 - ((F r).card : ℚ))) ≤
      ∑ r ∈ R, y r * ∑ i ∈ e, coef (S r) (F r) i := by
    apply Finset.sum_le_sum
    intro r hr
    exact mul_le_mul_of_nonneg_left
      (facet_sum_le (S r) (F r) e (s r) (hFS r hr) (hpar r hr) (he r hr)) (hy r hr)
  -- (2) exchange the sums: the right side is the total load on e
  have h2 : (∑ r ∈ R, y r * ∑ i ∈ e, coef (S r) (F r) i) =
      ∑ i ∈ e, load R S F y i := by
    unfold load
    simp_rw [Finset.mul_sum]
    exact Finset.sum_comm
  -- (3) each load is at most its weight plus its slack
  have h3 : (∑ i ∈ e, load R S F y i) ≤
      ∑ i ∈ e, (w i + max 0 (load R S F y i - w i)) := by
    apply Finset.sum_le_sum
    intro i _
    have := le_max_right 0 (load R S F y i - w i)
    linarith
  -- (4) slack over e is at most slack over V, since every slack is >= 0
  have h4 : (∑ i ∈ e, max 0 (load R S F y i - w i)) ≤
      ∑ i ∈ V, max 0 (load R S F y i - w i) := by
    apply Finset.sum_le_sum_of_subset_of_nonneg heV
    intro i _ _
    exact le_max_left _ _
  rw [Finset.sum_add_distrib] at h3
  linarith

/-- The scan's XOR facets are sound: the XOR of two rows the correction satisfies is
a row it satisfies, with the XOR of the two parities. -/
theorem xor_parity (A B e : Finset ι) :
    (symmDiff A B ∩ e).card % 2 = ((A ∩ e).card + (B ∩ e).card) % 2 := by
  have hdist : symmDiff A B ∩ e = symmDiff (A ∩ e) (B ∩ e) := by
    ext i
    simp only [Finset.mem_inter, Finset.mem_symmDiff]
    tauto
  rw [hdist]
  have hunion := Finset.card_union_add_card_inter (A ∩ e) (B ∩ e)
  have hsd : symmDiff (A ∩ e) (B ∩ e) = ((A ∩ e) ∪ (B ∩ e)) \ ((A ∩ e) ∩ (B ∩ e)) := by
    ext i
    simp only [Finset.mem_symmDiff, Finset.mem_sdiff, Finset.mem_union, Finset.mem_inter]
    tauto
  have hcard : (symmDiff (A ∩ e) (B ∩ e)).card + 2 * ((A ∩ e) ∩ (B ∩ e)).card =
      (A ∩ e).card + (B ∩ e).card := by
    rw [hsd, Finset.card_sdiff_of_subset Finset.inter_subset_union]
    have hle := Finset.card_le_card (Finset.inter_subset_union (s := A ∩ e) (t := B ∩ e))
    omega
  omega

end OneqCert
