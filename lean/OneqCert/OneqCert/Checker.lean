import OneqCert.WeakDuality
import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp

/-!
  ONE_Q, P2: milestone 2 -- an EXECUTABLE certificate checker whose soundness is a theorem.

  Milestone 1 proved the mathematics of the safe dual bound, and left one link to trust:
  that the Python checker computes it. This file removes that link for anyone who runs
  this checker instead. `accept` is ordinary executable Lean -- it is what the compiled
  binary runs -- and `accept_sound` is a kernel-checked theorem ABOUT THAT FUNCTION:

      accept c = true  ->  the candidate satisfies every check, and no correction that
                           satisfies every check weighs less.

  The certificate format mirrors the one P2's independent reference checker reads:
  check supports, a syndrome, rational weights, the candidate's support, and facets
  (a row set naming a single check or an XOR of checks, a subset F, a multiplier y).

  Acceptance is lattice-strengthened, as in P2 section 3: with every weight in
  (1/D)Z, every correction's weight is in (1/D)Z, so L > U - 1/D together with
  L <= OPT forces OPT >= U. `lattice_step` proves that step; nothing is assumed.
-/

open Finset

namespace OneqCert

/-- A facet as the certificate states it. -/
structure RawFacet where
  rows : List ℕ
  F : List ℕ
  y : ℚ
deriving Inhabited, Repr

/-- A certificate: the instance and the claimed witness, all in one document. -/
structure Cert where
  checks : List (List ℕ)
  syn : List ℕ
  w : List ℚ
  e : List ℕ
  facets : List RawFacet
  D : ℕ
deriving Inhabited, Repr

namespace Cert

variable (c : Cert)

/-- Support of check `j`. -/
def supp (j : ℕ) : Finset ℕ := (c.checks.getD j []).toFinset

/-- Required parity of check `j`, as a bit. -/
def synBit (j : ℕ) : ℕ := c.syn.getD j 0 % 2

/-- Weight of variable `i`. -/
def wt (i : ℕ) : ℚ := c.w.getD i 0

/-- Weight of a correction. -/
def weight (e : Finset ℕ) : ℚ := ∑ i ∈ e, c.wt i

/-- A correction satisfies every check of the instance. -/
def Consistent (e : Finset ℕ) : Prop :=
  ∀ j < c.checks.length, (c.supp j ∩ e).card % 2 = c.synBit j

end Cert

/-- Support of an XOR of rows. -/
def xorSupp (sp : ℕ → Finset ℕ) : List ℕ → Finset ℕ
  | [] => ∅
  | r :: rs => symmDiff (sp r) (xorSupp sp rs)

/-- Parity of an XOR of rows. -/
def xorPar (sb : ℕ → ℕ) : List ℕ → ℕ
  | [] => 0
  | r :: rs => (sb r + xorPar sb rs) % 2

/-- A correction meeting each row's parity meets the XOR row's parity. -/
theorem xor_list_parity (sp : ℕ → Finset ℕ) (sb : ℕ → ℕ) (e : Finset ℕ) :
    ∀ rows : List ℕ, (∀ r ∈ rows, (sp r ∩ e).card % 2 = sb r) →
      (xorSupp sp rows ∩ e).card % 2 = xorPar sb rows
  | [], _ => by simp [xorSupp, xorPar]
  | r :: rs, h => by
    have ih := xor_list_parity sp sb e rs (fun r' hr' => h r' (List.mem_cons_of_mem r hr'))
    have hr := h r List.mem_cons_self
    have hx := xor_parity (sp r) (xorSupp sp rs) e
    simp only [xorSupp, xorPar]
    omega

namespace Cert

variable (c : Cert)

/-- The facet's support, re-derived from the named rows -- never taken from the certificate. -/
def fS (f : RawFacet) : Finset ℕ := xorSupp c.supp f.rows

/-- The facet's required parity, re-derived from the named rows. -/
def fPar (f : RawFacet) : ℕ := xorPar c.synBit f.rows

/-- A facet is valid: named rows exist, F lies in the re-derived support with the WRONG
parity, and the multiplier is non-negative. -/
def facetOk (f : RawFacet) : Bool :=
  !f.rows.isEmpty && f.rows.all (· < c.checks.length) &&
    decide (f.F.toFinset ⊆ c.fS f) &&
    decide (f.F.toFinset.card % 2 ≠ c.fPar f) &&
    decide (0 ≤ f.y)

/-- Facet `r` of the certificate. -/
def fac (r : ℕ) : RawFacet := c.facets.getD r default

/-- The safe dual bound of the certificate, over every variable of the instance. -/
def bound : ℚ :=
  safeBound (Finset.range c.facets.length) (fun r => c.fS (c.fac r))
    (fun r => (c.fac r).F.toFinset) (fun r => (c.fac r).y) c.wt (Finset.range c.w.length)

/-- THE CHECKER. Everything the soundness theorem needs is checked here, from the
certificate's own data; nothing a producer says is trusted. -/
def accept : Bool :=
  (c.syn.length == c.checks.length) &&
    decide c.e.Nodup &&
    c.e.all (· < c.w.length) &&
    (List.range c.checks.length).all
      (fun j => (c.supp j ∩ c.e.toFinset).card % 2 == c.synBit j) &&
    c.facets.all c.facetOk &&
    decide (0 < c.D) &&
    c.w.all (fun q => (q * c.D).den == 1) &&
    decide (c.weight c.e.toFinset - 1 / (c.D : ℚ) < c.bound)

end Cert

/-- On the lattice (1/D)Z, `a - 1/D < b` already forces `a <= b`. -/
theorem lattice_step (D : ℕ) (hD : 0 < D) (a b : ℚ) (ma mb : ℤ)
    (ha : a * D = ma) (hb : b * D = mb) (h : a - 1 / (D : ℚ) < b) : a ≤ b := by
  have hDq : (0 : ℚ) < D := by exact_mod_cast hD
  have h1 : a * D - 1 < b * D := by
    have := mul_lt_mul_of_pos_right h hDq
    rwa [sub_mul, one_div, inv_mul_cancel₀ (ne_of_gt hDq)] at this
  rw [ha, hb] at h1
  have h2 : ma - 1 < mb := by exact_mod_cast h1
  have h3 : (ma : ℚ) ≤ mb := by exact_mod_cast (show ma ≤ mb by omega)
  rw [← ha, ← hb] at h3
  exact le_of_mul_le_mul_right h3 hDq

/-- A sum of lattice points is a lattice point. -/
theorem sum_on_lattice (D : ℕ) (f : ℕ → ℚ) (s : Finset ℕ)
    (h : ∀ i ∈ s, (f i * D).den = 1) : ∃ m : ℤ, (∑ i ∈ s, f i) * D = m := by
  refine ⟨∑ i ∈ s, (f i * D).num, ?_⟩
  rw [Finset.sum_mul]
  push_cast
  exact Finset.sum_congr rfl (fun i hi => (Rat.coe_int_num_of_den_eq_one (h i hi)).symm)

theorem getD_mem {α : Type*} (l : List α) (r : ℕ) (d : α) (h : r < l.length) :
    l.getD r d ∈ l := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  exact List.getElem_mem h

namespace Cert

variable (c : Cert)

/-- **Milestone 2.** An accepted certificate proves its candidate is a correction and
that no correction over the instance's variables weighs less. -/
theorem accept_sound (h : c.accept = true) :
    c.Consistent c.e.toFinset ∧ c.e.toFinset ⊆ Finset.range c.w.length ∧
      ∀ e' ⊆ Finset.range c.w.length, c.Consistent e' →
        c.weight c.e.toFinset ≤ c.weight e' := by
  simp only [accept, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨_hlen, _hnd⟩, he_rng⟩, hcons⟩, hfac⟩, hD⟩, hlat⟩, hcmp⟩ := h
  have hEV : c.e.toFinset ⊆ Finset.range c.w.length := by
    intro i hi
    rw [List.mem_toFinset] at hi
    exact Finset.mem_range.mpr (he_rng i hi)
  have hconsE : c.Consistent c.e.toFinset := by
    intro j hj
    exact hcons j (List.mem_range.mpr hj)
  refine ⟨hconsE, hEV, ?_⟩
  intro e' he'V hcons'
  -- facet by facet, the hypotheses of the weak duality theorem
  have hok : ∀ r ∈ Finset.range c.facets.length, c.facetOk (c.fac r) = true := by
    intro r hr
    exact hfac _ (getD_mem _ _ _ (Finset.mem_range.mp hr))
  have hL : c.bound ≤ c.weight e' := by
    unfold bound weight
    refine safe_bound_le_weight _ _ _ (fun r => c.fPar (c.fac r)) _ _ _ _
      ?_ ?_ ?_ ?_ he'V
    · intro r hr
      have := hok r hr
      simp only [facetOk, Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.2
    · intro r hr
      have := hok r hr
      simp only [facetOk, Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.1.1.2
    · intro r hr
      have := hok r hr
      simp only [facetOk, Bool.and_eq_true, decide_eq_true_eq] at this
      exact this.1.2
    · intro r hr
      have := hok r hr
      simp only [facetOk, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at this
      exact xor_list_parity c.supp c.synBit e' _
        (fun r' hr' => hcons' r' (this.1.1.1.2 r' hr'))
  -- the lattice: both weights are in (1/D)Z
  have hlatV : ∀ i ∈ Finset.range c.w.length, (c.wt i * c.D).den = 1 := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    unfold wt
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi', Option.getD_some]
    exact hlat _ (List.getElem_mem hi')
  obtain ⟨ma, hma⟩ := sum_on_lattice c.D c.wt c.e.toFinset
    (fun i hi => hlatV i (hEV hi))
  obtain ⟨mb, hmb⟩ := sum_on_lattice c.D c.wt e' (fun i hi => hlatV i (he'V hi))
  exact lattice_step c.D hD _ _ ma mb hma hmb (lt_of_lt_of_le hcmp hL)

end Cert

end OneqCert
