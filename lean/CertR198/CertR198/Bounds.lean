import CertR198.Model
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# R198 certificate calculus -- rules D/G: the two lower bounds of `bound_ok`

`ccheck.c` `bound_ok(s, o, rem)` (lines 85-99) prunes a node when

1. **(D) independent firing detectors.** A greedy pass (lines 86-89): take the lowest
   remaining firing detector `d`, count it, delete `CT[d]` (every detector co-touched with `d`
   by some mechanism, `d` included) and repeat. If the count exceeds `rem`, prune.
2. **infeasible detector.** Some firing `d` has `c_d = 0` (line 95): no allowed mechanism touches it.
3. **(G) fractional cover.** `c_d` = max over allowed candidates `m` of `d` of `|syn m & s|`
   (lines 92-94); `num = Σ_{firing d} ⌊2520 / c_d⌋` (line 96); prune if `⌈num / 2520⌉ > rem`
   (line 98), computed as `(num + 2519) / 2520`.

All three are modelled here *exactly as computed*, in natural-number arithmetic (`/` is floor
division, as in C), over the checker's two tables:

* `ct : D → Finset D` is `CT` (line 195),
* `cand : D → Finset M` is the set of mechanisms in `alist[d]` (line 193).

The C tables are built from the detector LISTS `DET[m]`; when a list repeats a detector the
tables over-approximate the true incidence. So the theorems assume only `Tables.Sound`: the
tables CONTAIN the true co-touch / touch relation. `Tables.canonical` is the exact table.

`bound_sound` (theorem (b)): if `BoundOK` holds at `(s, o, rem)` then every set `Y` of allowed
mechanisms with `synSum Y = s` has `card Y > rem` -- the bounds are valid lower bounds on the
weight of any completion. The constant `2520` plays no role: any `K > 0` works.
-/

open Finset

namespace CertR198

variable {M D Ob : Type*}

/-- The checker's derived tables: `CT` and the touching lists `alist`. -/
structure Tables (M D : Type*) where
  /-- `CT[d]` (`ccheck.c` line 195). -/
  ct : D → Finset D
  /-- The mechanisms listed in `alist[d]` (`ccheck.c` line 193). -/
  cand : D → Finset M
  /-- `CT_ORBIT[o][d]` (R205, CDP4/CDP5): the detectors co-touched with `d` by some mechanism of
  orbit `≥ o` -- a suffix union over orbits, plus `d` itself. -/
  cto : ℕ → D → Finset D

/-- The tables contain the true incidence of the model (they may contain more). -/
structure Tables.Sound (T : Tables M D) (E : DEM M D Ob) : Prop where
  ct_sound : ∀ m d d', E.syn m d ≠ 0 → E.syn m d' ≠ 0 → d' ∈ T.ct d
  cand_sound : ∀ m d, E.syn m d ≠ 0 → m ∈ T.cand d
  /-- `CT_ORBIT[o]` contains the co-touch of every mechanism ALLOWED at orbit floor `o`. -/
  cto_sound : ∀ m d d' o, o ≤ E.orb m → E.syn m d ≠ 0 → E.syn m d' ≠ 0 → d' ∈ T.cto o d

/-- The firing detectors of a syndrome. -/
def firing [Fintype D] (s : D → ZMod 2) : Finset D := univ.filter fun d => s d ≠ 0

@[simp] theorem mem_firing [Fintype D] {s : D → ZMod 2} {d : D} : d ∈ firing s ↔ s d ≠ 0 := by
  simp [firing]

/-- The exact tables of a finite model (what `ccheck.c` builds when no `DET` list repeats a
detector); they satisfy `Tables.Sound`, so the hypotheses of every theorem are satisfiable. -/
def Tables.canonical [Fintype M] [Fintype D] (E : DEM M D Ob) : Tables M D where
  ct d := univ.filter fun d' => ∃ m, E.syn m d ≠ 0 ∧ E.syn m d' ≠ 0
  cand d := univ.filter fun m => E.syn m d ≠ 0
  cto o d := univ.filter fun d' => ∃ m, o ≤ E.orb m ∧ E.syn m d ≠ 0 ∧ E.syn m d' ≠ 0

theorem Tables.canonical_sound [Fintype M] [Fintype D] (E : DEM M D Ob) :
    (Tables.canonical E).Sound E where
  ct_sound m d d' h h' := by simp only [Tables.canonical, mem_filter, mem_univ, true_and]; exact ⟨m, h, h'⟩
  cand_sound m d h := by simp only [Tables.canonical, mem_filter, mem_univ, true_and]; exact h
  cto_sound m d d' o ho h h' := by
    simp only [Tables.canonical, mem_filter, mem_univ, true_and]; exact ⟨m, ho, h, h'⟩

/-- The greedy independent-set count of `bound_ok` (lines 86-88), over a list of detectors:
a listed detector still in the remaining set `r` is counted and `ct d` is removed from `r`.
`ccheck.c` runs it on the firing detectors in increasing index order; soundness holds for
ANY order (`greedy_le_card`). -/
def greedy [DecidableEq D] (ct : D → Finset D) : List D → Finset D → ℕ
  | [], _ => 0
  | d :: ds, r => if d ∈ r then greedy ct ds (r \ ct d) + 1 else greedy ct ds r

namespace DEM

section defs

variable (E : DEM M D Ob) [Fintype D]

/-- `popcount(S[m] & s)` (line 94). -/
def cval (s : D → ZMod 2) (m : M) : ℕ := ((firing s).filter fun d => E.syn m d ≠ 0).card

/-- `c_d` (lines 92-94): the max of `cval s` over the candidates of `d` with orbit `≥ o`
(`0` when there are none). -/
def cmax (T : Tables M D) (o : ℕ) (s : D → ZMod 2) (d : D) : ℕ :=
  ((T.cand d).filter fun m => o ≤ E.orb m).sup (E.cval s)

/-- `num` (line 96): `Σ_{firing d} ⌊K / c_d⌋`. -/
def fracNum (K : ℕ) (T : Tables M D) (o : ℕ) (s : D → ZMod 2) : ℕ :=
  ∑ d ∈ firing s, K / E.cmax T o s d

/-- `bound_ok(s, o, rem)` (lines 85-99), with scale `K` (`2520` in the checker) and the order
`dl` in which the greedy pass meets detectors. The three disjuncts are lines 89, 95, 98. -/
def BoundOK [DecidableEq D] (K : ℕ) (T : Tables M D) (dl : List D) (o : ℕ)
    (s : D → ZMod 2) (rem : ℕ) : Prop :=
  rem < greedy T.ct dl (firing s) ∨ (∃ d ∈ firing s, E.cmax T o s d = 0) ∨
    rem < (E.fracNum K T o s + (K - 1)) / K

end defs

variable {E : DEM M D Ob}

/-- **Rule D.** The greedy count never exceeds the size of any set `Y` that touches every
detector still in `r`: the detectors it counts are pairwise never touched by one mechanism
(each was outside `ct` of every earlier pick), so each needs its own member of `Y`. -/
theorem greedy_le_card [DecidableEq M] [DecidableEq D] {T : Tables M D} (hT : T.Sound E)
    (dl : List D) :
    ∀ (r : Finset D) (Y : Finset M), (∀ d ∈ r, ∃ m ∈ Y, E.syn m d ≠ 0) →
      greedy T.ct dl r ≤ Y.card := by
  induction dl with
  | nil => intro r Y _; simp [greedy]
  | cons d ds ih =>
    intro r Y h
    simp only [greedy]
    split_ifs with hd
    · obtain ⟨m, hmY, hmd⟩ := h d hd
      have h' : ∀ d' ∈ r \ T.ct d, ∃ m' ∈ Y.erase m, E.syn m' d' ≠ 0 := by
        intro d' hd'
        rw [Finset.mem_sdiff] at hd'
        obtain ⟨m', hm'Y, hm'd'⟩ := h d' hd'.1
        refine ⟨m', Finset.mem_erase.mpr ⟨?_, hm'Y⟩, hm'd'⟩
        rintro rfl
        exact hd'.2 (hT.ct_sound _ d d' hmd hm'd')
      have h1 := ih (r \ T.ct d) (Y.erase m) h'
      rw [Finset.card_erase_of_mem hmY] at h1
      have h2 : 0 < Y.card := Finset.card_pos.mpr ⟨m, hmY⟩
      omega
    · exact ih r Y h

variable [Fintype D]

/-- An allowed mechanism touching `d` is among the candidates `c_d` maximises over. -/
theorem cval_le_cmax {T : Tables M D} (hT : T.Sound E) {o : ℕ} {s : D → ZMod 2} {m : M} {d : D}
    (ho : o ≤ E.orb m) (hmd : E.syn m d ≠ 0) : E.cval s m ≤ E.cmax T o s d :=
  Finset.le_sup (f := E.cval s) (Finset.mem_filter.mpr ⟨hT.cand_sound m d hmd, ho⟩)

theorem cval_pos {s : D → ZMod 2} {m : M} {d : D} (hd : s d ≠ 0) (hmd : E.syn m d ≠ 0) :
    0 < E.cval s m :=
  Finset.card_pos.mpr ⟨d, Finset.mem_filter.mpr ⟨mem_firing.mpr hd, hmd⟩⟩

/-- **Rule G, exact integers.** For every completion `Y` (allowed mechanisms, `synSum Y = s`),
`num ≤ K * card Y`. Each firing `d` is touched by some member of `Y`; a member `m` with
`k = |syn m & s|` meets `k` firing detectors, each with `c_d ≥ k`, so it carries at most
`k * ⌊K / k⌋ ≤ K`. (The real-valued statement is LP weak duality for `y_d = 1/c_d`; the floor
only lowers each term, so no divisibility of `K` is needed.) -/
theorem fracNum_le [DecidableEq D] {T : Tables M D} (hT : T.Sound E) (K : ℕ) {o : ℕ}
    {s : D → ZMod 2} {Y : Finset M} (hY : ∀ m ∈ Y, o ≤ E.orb m) (hs : E.synSum Y = s) :
    E.fracNum K T o s ≤ K * Y.card := by
  have step1 : ∀ d ∈ firing s, K / E.cmax T o s d ≤
      ∑ m ∈ Y, (if E.syn m d ≠ 0 then K / E.cmax T o s d else 0) := by
    intro d hd
    obtain ⟨m, hmY, hmd⟩ :=
      exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd)
    have := Finset.single_le_sum
      (f := fun m' => if E.syn m' d ≠ 0 then K / E.cmax T o s d else 0)
      (fun _ _ => Nat.zero_le _) hmY
    rw [ite_eq_left hmd] at this
    exact this
  have step2 : ∀ m ∈ Y,
      ∑ d ∈ firing s, (if E.syn m d ≠ 0 then K / E.cmax T o s d else 0) ≤ K := by
    intro m hmY
    rw [← Finset.sum_filter]
    have hk : ∀ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0),
        K / E.cmax T o s d ≤ K / E.cval s m := by
      intro d hd
      rw [Finset.mem_filter] at hd
      exact Nat.div_le_div_left (cval_le_cmax hT (hY m hmY) hd.2)
        (cval_pos (mem_firing.mp hd.1) hd.2)
    calc ∑ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0), K / E.cmax T o s d
        ≤ ∑ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0), K / E.cval s m :=
          Finset.sum_le_sum hk
      _ = E.cval s m * (K / E.cval s m) := by rw [Finset.sum_const, smul_eq_mul]; rfl
      _ ≤ K := Nat.mul_div_le K (E.cval s m)
  calc E.fracNum K T o s = ∑ d ∈ firing s, K / E.cmax T o s d := rfl
    _ ≤ ∑ d ∈ firing s, ∑ m ∈ Y, (if E.syn m d ≠ 0 then K / E.cmax T o s d else 0) :=
        Finset.sum_le_sum step1
    _ = ∑ m ∈ Y, ∑ d ∈ firing s, (if E.syn m d ≠ 0 then K / E.cmax T o s d else 0) :=
        Finset.sum_comm
    _ ≤ ∑ m ∈ Y, K := Finset.sum_le_sum step2
    _ = K * Y.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- The ceiling `(num + K - 1) / K` of line 98 is at most `n` when `num ≤ K * n`. -/
theorem ceil_div_le {K a n : ℕ} (hK : 0 < K) (h : a ≤ K * n) : (a + (K - 1)) / K ≤ n := by
  have : a + (K - 1) < (n + 1) * K := by rw [add_mul, one_mul, mul_comm n K]; omega
  exact Nat.lt_succ_iff.mp ((Nat.div_lt_iff_lt_mul hK).mpr this)

/-- **Rules D/G (theorem (b)).** If `bound_ok` holds at `(s, o, rem)`, every completion of
`s` by allowed mechanisms is heavier than `rem`; in particular the node has no completion
within its budget. -/
theorem bound_sound [DecidableEq M] [DecidableEq D] {T : Tables M D} (hT : T.Sound E) {K : ℕ}
    (hK : 0 < K) {dl : List D} {o : ℕ} {s : D → ZMod 2} {rem : ℕ}
    (h : E.BoundOK K T dl o s rem) {Y : Finset M} (hY : ∀ m ∈ Y, o ≤ E.orb m)
    (hs : E.synSum Y = s) : rem < Y.card := by
  rcases h with h | ⟨d, hd, h0⟩ | h
  · exact lt_of_lt_of_le h (greedy_le_card hT dl _ Y fun d hd =>
      exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd))
  · obtain ⟨m, hmY, hmd⟩ :=
      exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd)
    have h1 := cval_le_cmax (s := s) hT (hY m hmY) hmd
    have h2 := cval_pos (E := E) (mem_firing.mp hd) hmd
    omega
  · exact lt_of_lt_of_le h (ceil_div_le hK (fracNum_le hT K hY hs))

/-! ### CDP4/CDP5 (R205/R206): orbit-filtered packing and the exact-integer dual raise

Under a CDP4 or CDP5 header `ccheck.c`'s `bound_ok` (R206 lines 190-243) changes in two ways:

1. **(D')** the greedy packing runs on `CT_ORBIT[o]` -- the co-touch of the mechanisms ALLOWED at
   orbit floor `o` -- instead of `CT`;
2. **(G')** after the plain cover test `Σ ⌊K / c_d⌋ > K · rem`, when `rem ≥ 4` and the sum is
   within `K` of the threshold, each firing `d` in increasing index order is RAISED by its
   allowance, the least slack `K - used m` over the allowed candidates `m` of `d`, and the node
   is pruned as soon as the sum exceeds `K · rem`.

The raise keeps every allowed mechanism's load at most `K` (`Feasible.raise1`), the initial
weights already satisfy that (`feasible_init`), and any such weights bound every completion
(`sum_le_of_feasible`, LP weak duality in integers). The C test order and the `rem ≥ 4` trigger
only make the checker accept less, so `BoundOK5` drops them. -/

omit [Fintype D] in
/-- **Rule D'.** The greedy count is at most the size of any set of mechanisms satisfying `P`
that touches every remaining detector, as soon as `ct` contains the co-touch of every mechanism
satisfying `P` (`greedy_le_card` is the case `P = ⊤`). -/
theorem greedy_le_card_of [DecidableEq M] [DecidableEq D] {ct : D → Finset D} (P : M → Prop)
    (hct : ∀ m, P m → ∀ d d', E.syn m d ≠ 0 → E.syn m d' ≠ 0 → d' ∈ ct d) (dl : List D) :
    ∀ (r : Finset D) (Y : Finset M), (∀ m ∈ Y, P m) → (∀ d ∈ r, ∃ m ∈ Y, E.syn m d ≠ 0) →
      greedy ct dl r ≤ Y.card := by
  induction dl with
  | nil => intro r Y _ _; simp [greedy]
  | cons d ds ih =>
    intro r Y hP h
    simp only [greedy]
    split_ifs with hd
    · obtain ⟨m, hmY, hmd⟩ := h d hd
      have h' : ∀ d' ∈ r \ ct d, ∃ m' ∈ Y.erase m, E.syn m' d' ≠ 0 := by
        intro d' hd'
        rw [Finset.mem_sdiff] at hd'
        obtain ⟨m', hm'Y, hm'd'⟩ := h d' hd'.1
        refine ⟨m', Finset.mem_erase.mpr ⟨?_, hm'Y⟩, hm'd'⟩
        rintro rfl
        exact hd'.2 (hct _ (hP _ hmY) d d' hmd hm'd')
      have h1 := ih (r \ ct d) (Y.erase m) (fun m' hm' => hP m' (Finset.mem_of_mem_erase hm')) h'
      rw [Finset.card_erase_of_mem hmY] at h1
      have h2 : 0 < Y.card := Finset.card_pos.mpr ⟨m, hmY⟩
      omega
    · exact ih r Y hP h

variable (E) in
/-- `used` (lines 228-232): the load `Σ_{v firing, m touches v} y v` of mechanism `m`. -/
def used (s : D → ZMod 2) (y : D → ℕ) (m : M) : ℕ :=
  ∑ v ∈ (firing s).filter (fun v => E.syn m v ≠ 0), y v

variable (E) in
/-- `allowance` (lines 225-235): the least slack `K - used m` over the allowed candidates `m`
of `d`, starting from `K` (C: `int allowance = 2520`, then `min`). -/
def allowance (K : ℕ) (T : Tables M D) (o : ℕ) (s : D → ZMod 2) (y : D → ℕ) (d : D) : ℕ :=
  ((T.cand d).filter fun m => o ≤ E.orb m).fold min K fun m => K - E.used s y m

variable (E) in
/-- One raise (lines 236-240): `y d += allowance d`. -/
def raise1 [DecidableEq D] (K : ℕ) (T : Tables M D) (o : ℕ) (s : D → ZMod 2) (y : D → ℕ)
    (d : D) : D → ℕ :=
  Function.update y d (y d + E.allowance K T o s y d)

variable (E) in
/-- The raises over a list of detectors, in order (C: the firing detectors by increasing index;
an early exit raised a prefix of that list). -/
def raiseList [DecidableEq D] (K : ℕ) (T : Tables M D) (o : ℕ) (s : D → ZMod 2)
    (dl : List D) (y : D → ℕ) : D → ℕ :=
  dl.foldl (E.raise1 K T o s) y

variable (E) in
/-- Dual feasibility: every allowed mechanism carries at most `K`. -/
def Feasible (K o : ℕ) (s : D → ZMod 2) (y : D → ℕ) : Prop :=
  ∀ m, o ≤ E.orb m → E.used s y m ≤ K

variable (E) in
/-- `bound_ok` under a CDP4/CDP5 header: packing over `CT_ORBIT[o]` (in the order `dl`), an
infeasible detector, or the cover sum after raising the detectors `rl` above `K · rem`
(`rl = []`: the plain test `Σ ⌊K / c_d⌋ > K · rem`). -/
def BoundOK5 [DecidableEq D] (K : ℕ) (T : Tables M D) (dl rl : List D) (o : ℕ)
    (s : D → ZMod 2) (rem : ℕ) : Prop :=
  rem < greedy (T.cto o) dl (firing s) ∨ (∃ d ∈ firing s, E.cmax T o s d = 0) ∨
    K * rem < ∑ d ∈ firing s, E.raiseList K T o s rl (fun d => K / E.cmax T o s d) d

theorem used_update_of_not_mem [DecidableEq D] {s : D → ZMod 2} {y : D → ℕ} {m : M} {d : D}
    (x : ℕ) (hd : d ∉ (firing s).filter (fun v => E.syn m v ≠ 0)) :
    E.used s (Function.update y d x) m = E.used s y m := by
  unfold DEM.used
  refine Finset.sum_congr rfl fun v hv => ?_
  rw [Function.update_of_ne (by rintro rfl; exact hd hv)]

theorem used_update_of_mem [DecidableEq D] {s : D → ZMod 2} {y : D → ℕ} {m : M} {d : D}
    (a : ℕ) (hd : d ∈ (firing s).filter (fun v => E.syn m v ≠ 0)) :
    E.used s (Function.update y d (y d + a)) m = E.used s y m + a := by
  unfold DEM.used
  rw [Finset.sum_update_of_mem hd, ← Finset.add_sum_erase _ y hd, Finset.sdiff_singleton_eq_erase]
  omega

/-- The allowance never exceeds the slack of an allowed candidate. -/
theorem allowance_le {K : ℕ} {T : Tables M D} {o : ℕ} {s : D → ZMod 2} {y : D → ℕ} {d : D}
    {m : M} (hm : m ∈ (T.cand d).filter fun m => o ≤ E.orb m) :
    E.allowance K T o s y d ≤ K - E.used s y m :=
  (Finset.fold_min_le _).mpr (Or.inr ⟨m, hm, le_rfl⟩)

/-- **Rule G', one step.** A raise keeps the weights dual-feasible: a mechanism touching `d` is
an allowed candidate of `d` (the tables are sound), so its load grows by at most its slack. -/
theorem Feasible.raise1 [DecidableEq D] {T : Tables M D} (hT : T.Sound E) {K o : ℕ}
    {s : D → ZMod 2} {y : D → ℕ} (hy : E.Feasible K o s y) (d : D) :
    E.Feasible K o s (E.raise1 K T o s y d) := by
  intro m hm
  unfold DEM.raise1
  by_cases hd : d ∈ (firing s).filter (fun v => E.syn m v ≠ 0)
  · rw [used_update_of_mem _ hd]
    have ha := allowance_le (K := K) (T := T) (y := y) (s := s)
      (Finset.mem_filter.mpr ⟨hT.cand_sound m d (Finset.mem_filter.mp hd).2, hm⟩)
    have := hy m hm
    omega
  · rw [used_update_of_not_mem _ hd]
    exact hy m hm

theorem Feasible.raiseList [DecidableEq D] {T : Tables M D} (hT : T.Sound E) {K o : ℕ}
    {s : D → ZMod 2} (dl : List D) :
    ∀ {y : D → ℕ}, E.Feasible K o s y → E.Feasible K o s (E.raiseList K T o s dl y) := by
  induction dl with
  | nil => intro y hy; exact hy
  | cons d ds ih =>
    intro y hy
    show E.Feasible K o s (E.raiseList K T o s ds (E.raise1 K T o s y d))
    exact ih (hy.raise1 hT d)

/-- The initial weights `⌊K / c_d⌋` are dual-feasible (the inner step of `fracNum_le`). -/
theorem feasible_init {T : Tables M D} (hT : T.Sound E) (K o : ℕ) (s : D → ZMod 2) :
    E.Feasible K o s (fun d => K / E.cmax T o s d) := by
  intro m hm
  unfold DEM.used
  have hk : ∀ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0),
      K / E.cmax T o s d ≤ K / E.cval s m := by
    intro d hd
    rw [Finset.mem_filter] at hd
    exact Nat.div_le_div_left (cval_le_cmax hT hm hd.2) (cval_pos (mem_firing.mp hd.1) hd.2)
  calc ∑ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0), K / E.cmax T o s d
      ≤ ∑ d ∈ (firing s).filter (fun d => E.syn m d ≠ 0), K / E.cval s m :=
        Finset.sum_le_sum hk
    _ = E.cval s m * (K / E.cval s m) := by rw [Finset.sum_const, smul_eq_mul]; rfl
    _ ≤ K := Nat.mul_div_le K (E.cval s m)

/-- **LP weak duality, in integers.** Dual-feasible weights bound every completion `Y` by
allowed mechanisms: `Σ_{firing d} y d ≤ K · card Y`. -/
theorem sum_le_of_feasible {K o : ℕ} {s : D → ZMod 2} {y : D → ℕ} (hy : E.Feasible K o s y)
    {Y : Finset M} (hY : ∀ m ∈ Y, o ≤ E.orb m) (hs : E.synSum Y = s) :
    ∑ d ∈ firing s, y d ≤ K * Y.card := by
  have step1 : ∀ d ∈ firing s, y d ≤ ∑ m ∈ Y, (if E.syn m d ≠ 0 then y d else 0) := by
    intro d hd
    obtain ⟨m, hmY, hmd⟩ :=
      exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd)
    have := Finset.single_le_sum (f := fun m' => if E.syn m' d ≠ 0 then y d else 0)
      (fun _ _ => Nat.zero_le _) hmY
    rw [ite_eq_left hmd] at this
    exact this
  have step2 : ∀ m ∈ Y, ∑ d ∈ firing s, (if E.syn m d ≠ 0 then y d else 0) ≤ K := by
    intro m hmY
    rw [← Finset.sum_filter]
    exact hy m (hY m hmY)
  calc ∑ d ∈ firing s, y d
      ≤ ∑ d ∈ firing s, ∑ m ∈ Y, (if E.syn m d ≠ 0 then y d else 0) := Finset.sum_le_sum step1
    _ = ∑ m ∈ Y, ∑ d ∈ firing s, (if E.syn m d ≠ 0 then y d else 0) := Finset.sum_comm
    _ ≤ ∑ m ∈ Y, K := Finset.sum_le_sum step2
    _ = K * Y.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- **Rules D'/G' (CDP4/CDP5).** If `BoundOK5` holds at `(s, o, rem)`, every completion of `s`
by allowed mechanisms is heavier than `rem`. -/
theorem bound5_sound [DecidableEq M] [DecidableEq D] {T : Tables M D} (hT : T.Sound E) {K : ℕ}
    {dl rl : List D} {o : ℕ} {s : D → ZMod 2} {rem : ℕ} (h : E.BoundOK5 K T dl rl o s rem)
    {Y : Finset M} (hY : ∀ m ∈ Y, o ≤ E.orb m) (hs : E.synSum Y = s) : rem < Y.card := by
  rcases h with h | ⟨d, hd, h0⟩ | h
  · exact lt_of_lt_of_le h (greedy_le_card_of (fun m => o ≤ E.orb m)
      (fun m hm d d' h1 h2 => hT.cto_sound m d d' o hm h1 h2) dl _ Y hY fun d hd =>
        exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd))
  · obtain ⟨m, hmY, hmd⟩ :=
      exists_mem_touching (E := E) (Y := Y) (v := d) (by rw [hs]; exact mem_firing.mp hd)
    have h1 := cval_le_cmax (s := s) hT (hY m hmY) hmd
    have h2 := cval_pos (E := E) (mem_firing.mp hd) hmd
    omega
  · exact Nat.lt_of_mul_lt_mul_left
      (lt_of_lt_of_le h (sum_le_of_feasible ((feasible_init hT K o s).raiseList hT rl) hY hs))

end DEM

end CertR198
