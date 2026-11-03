import CertR198.Main

/-!
# An executable checker whose acceptance IS the theorem (gap G1, for search certificates)

`ccheck.c` is argued, not proved, to implement the Lean calculus (`SPEC.md`, gap G1). This module removes that gap
for one kind of certificate. `cert` is a Lean program, an exhaustive anchored search, and `cert_sound_exec` proves
that `cert c w = true` implies every undetectable logical error of the model `c` has weight `> w`. Evaluated by the
kernel (`decide +kernel`), it makes a circuit-distance bound a Lean theorem about the model's own numbers. No C
code is trusted, and no axiom beyond the standard three is needed.

* **The model.** A `CModel` is the exported projected DEM: per mechanism, its syndrome as a bitmask over `nd`
  detectors and its observable mask over `no` observables. `toDEM` reads it as a `DEM` (bit `d` set: `syn m d = 1`)
  with orbit = index: the trivial symmetry group, so every mechanism is its own anchor.
* **The search.** `search c o s l r` is the checker's walk with no bounds and no trace. A zero syndrome must carry
  no flip (rule E). A nonzero one ends the budget (spent), or is decided by the radius-2 leaf (rule L: singles over
  all mechanisms, pairs through the lowest firing detector), or branches over EVERY mechanism of index `≥ o`
  touching that detector (rules C + I). Each accepting path is one constructor of `Valid` (`search_valid`).
* **No bounds.** Pruning is what makes the C prover fast, and every prune needs a bound proof. Without prunes the
  tree is the full search: exponential in the weight, but exact, and small enough for small models.
-/

open Finset

namespace CertR198.Exec

/-- An exported model: per mechanism `(syndrome mask, observable mask)`. -/
structure CModel where
  nd : ℕ
  no : ℕ
  mech : List (ℕ × ℕ)

/-- `a` and `b` agree on their low `k` bits. -/
def eqOn (k a b : ℕ) : Bool := (a ^^^ b) % 2 ^ k == 0

/-- The lowest set bit of `s` among bits `i .. i + fuel - 1`, else `i + fuel`. -/
def firstBit (s : ℕ) : ℕ → ℕ → ℕ
  | 0, i => i
  | fuel + 1, i => if s.testBit i then i else firstBit s fuel (i + 1)

/-- Rule L, decided: no single mechanism completes `(s, l)`, and (`rem ≥ 2`) no pair whose first member touches
the firing detector `v` does. -/
def leafOK (c : CModel) (s l v rem : ℕ) : Bool :=
  c.mech.all (fun p => !eqOn c.nd p.1 s || eqOn c.no p.2 l) &&
  (decide (rem < 2) || (c.mech.zipIdx).all (fun a => !a.1.1.testBit v ||
      (c.mech.zipIdx).all (fun b => a.2 == b.2 || !eqOn c.nd (a.1.1 ^^^ b.1.1) s || eqOn c.no (a.1.2 ^^^ b.1.2) l)))

/-- The subtree at `(s, l)` with remaining budget `r`, anchor orbit `o` (orbit = index). -/
def search (c : CModel) (o : ℕ) (s l : ℕ) : ℕ → Bool
  | 0 => !(s % 2 ^ c.nd == 0) || l % 2 ^ c.no == 0
  | r + 1 =>
    if s % 2 ^ c.nd == 0 then l % 2 ^ c.no == 0
    else
      decide (firstBit s c.nd 0 < c.nd) &&
        (if r + 1 ≤ 2 then leafOK c s l (firstBit s c.nd 0) (r + 1)
         else (c.mech.zipIdx).all fun p => decide (p.2 < o) || !p.1.1.testBit (firstBit s c.nd 0) ||
                search c o (s ^^^ p.1.1) (l ^^^ p.1.2) r)

/-- Every anchor's tree with budget `w - 1`: no undetectable logical error of weight `≤ w`. -/
def cert (c : CModel) (w : ℕ) : Bool := (c.mech.zipIdx).all fun p => search c p.2 p.1.1 p.1.2 (w - 1)

/-! ## The model as a `DEM` -/

/-- Bit `d` of `x` as an element of GF(2). -/
def bitv {k : ℕ} (x : ℕ) (d : Fin k) : ZMod 2 := if x.testBit d.val then 1 else 0

/-- The mechanisms are `Fin c.mech.length`; orbit = index (the trivial group). -/
def CModel.toDEM (c : CModel) : DEM (Fin c.mech.length) (Fin c.nd) (Fin c.no) where
  syn m := bitv (c.mech[m]).1
  obs m := bitv (c.mech[m]).2
  orb m := m.val

theorem bitv_xor {k : ℕ} (a b : ℕ) : (bitv (a ^^^ b) : Fin k → ZMod 2) = bitv a + bitv b := by
  funext d
  simp only [bitv, Pi.add_apply, Nat.testBit_xor]
  cases a.testBit d.val <;> cases b.testBit d.val <;> decide

theorem bitv_ne_zero {k : ℕ} {x : ℕ} {d : Fin k} : bitv x d ≠ 0 ↔ x.testBit d.val = true := by
  unfold bitv
  cases x.testBit d.val <;> decide

theorem bitv_eq_zero_iff {k : ℕ} (s : ℕ) : (bitv s : Fin k → ZMod 2) = 0 ↔ s % 2 ^ k = 0 := by
  constructor
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_mod_two_pow, Nat.zero_testBit]
    by_cases hi : i < k
    · have h1 : bitv s ⟨i, hi⟩ = 0 := congrFun h ⟨i, hi⟩
      have h2 : ¬ s.testBit i = true := fun ht => bitv_ne_zero.mpr ht h1
      simp [hi, h2]
    · simp [hi]
  · intro h
    funext d
    have := congrArg (fun x => x.testBit d.val) h
    simp only [Nat.testBit_mod_two_pow, Nat.zero_testBit, d.isLt, decide_true, Bool.true_and] at this
    simp [bitv, this]

theorem pi_add_eq_zero_iff {α : Type*} (x y : α → ZMod 2) : x + y = 0 ↔ x = y := by
  constructor
  · intro h
    funext d
    have := congrFun h d
    simp only [Pi.add_apply, Pi.zero_apply] at this
    revert this
    generalize x d = p
    generalize y d = q
    revert p q
    decide
  · rintro rfl
    exact vec_add_self _

theorem eqOn_iff {k : ℕ} (a b : ℕ) : eqOn k a b = true ↔ (bitv a : Fin k → ZMod 2) = bitv b := by
  rw [eqOn, beq_iff_eq, ← bitv_eq_zero_iff, bitv_xor, pi_add_eq_zero_iff]

theorem firstBit_spec (s : ℕ) : ∀ fuel i, firstBit s fuel i < i + fuel → s.testBit (firstBit s fuel i) = true := by
  intro fuel
  induction fuel with
  | zero => intro i h; simp [firstBit] at h
  | succ n ih =>
    intro i h
    unfold firstBit at h ⊢
    cases hs : s.testBit i
    · simp only [hs, Bool.false_eq_true, ↓reduceIte] at h ⊢
      exact ih (i + 1) (by omega)
    · simp [hs]

theorem mem_zipIdx_get {α : Type*} (L : List α) (m : Fin L.length) : (L[m], m.val) ∈ L.zipIdx := by
  rw [List.mem_iff_getElem]
  exact ⟨m.val, by simp, by simp [List.getElem_zipIdx]⟩

/-! ## Soundness of the walk -/

variable (c : CModel)

theorem leafOK_sound {s l v rem : ℕ} (hv : v < c.nd) (hsv : s.testBit v = true)
    (h : leafOK c s l v rem = true) : c.toDEM.LeafOK (bitv s) (bitv l) rem := by
  simp only [leafOK, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun m hm => ?_, fun hrem => ⟨⟨v, hv⟩, bitv_ne_zero.mpr hsv, fun a b ha hab hs => ?_⟩⟩
  · have := h1 (c.mech[m]) (List.mem_iff_getElem.mpr ⟨m.val, m.isLt, rfl⟩)
    have he : eqOn c.nd (c.mech[m]).1 s = true := (eqOn_iff _ _).mpr hm
    simp only [he, Bool.not_true, Bool.false_or] at this
    exact (eqOn_iff _ _).mp this
  · have hr : ¬ rem < 2 := by omega
    simp only [hr, decide_false, Bool.false_or, List.all_eq_true] at h2
    have hx := h2 _ (mem_zipIdx_get c.mech a)
    have ha' : (c.mech[a]).1.testBit v = true := bitv_ne_zero.mp ha
    simp only [ha', Bool.not_true, Bool.false_or, List.all_eq_true] at hx
    have hy := hx _ (mem_zipIdx_get c.mech b)
    have hne : (a.val == b.val) = false := by
      rw [beq_eq_false_iff_ne]
      exact fun h' => hab (Fin.ext h')
    have he : eqOn c.nd ((c.mech[a]).1 ^^^ (c.mech[b]).1) s = true := by
      rw [eqOn_iff, bitv_xor]
      exact hs
    simp only [hne, he, Bool.not_true, Bool.false_or] at hy
    have := (eqOn_iff _ _).mp hy
    rw [bitv_xor] at this
    exact this

theorem search_valid (o : ℕ) : ∀ (r s l : ℕ), search c o s l r = true →
    Valid c.toDEM (Tables.canonical c.toDEM) o (bitv s) (bitv l) r := by
  intro r
  induction r with
  | zero =>
    intro s l h
    by_cases hs : s % 2 ^ c.nd = 0
    · have hl : l % 2 ^ c.no = 0 := by simpa [search, hs] using h
      rw [(bitv_eq_zero_iff s).mpr hs, (bitv_eq_zero_iff l).mpr hl]
      exact Valid.done 0
    · exact Valid.spent (fun h0 => hs ((bitv_eq_zero_iff s).mp h0))
  | succ r ih =>
    intro s l h
    by_cases hs : s % 2 ^ c.nd = 0
    · have hl : l % 2 ^ c.no = 0 := by simpa [search, hs] using h
      rw [(bitv_eq_zero_iff s).mpr hs, (bitv_eq_zero_iff l).mpr hl]
      exact Valid.done _
    · have hs' : (s % 2 ^ c.nd == 0) = false := by simpa using hs
      simp only [search, hs', Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨hv, hrest⟩ := h
      have hsv : s.testBit (firstBit s c.nd 0) = true := firstBit_spec s c.nd 0 (by omega)
      have hs0 : (bitv s : Fin c.nd → ZMod 2) ≠ 0 := fun h0 => hs ((bitv_eq_zero_iff s).mp h0)
      by_cases hr : r + 1 ≤ 2
      · simp only [hr, ↓reduceIte] at hrest
        exact Valid.leaf hs0 hr (leafOK_sound c hv hsv hrest)
      · simp only [hr, ↓reduceIte, List.all_eq_true] at hrest
        refine Valid.expand ⟨firstBit s c.nd 0, hv⟩ (bitv_ne_zero.mpr hsv) fun m hmo hmv => ?_
        have hx := hrest _ (mem_zipIdx_get c.mech m)
        have hlt : ¬ m.val < o := by
          have : o ≤ m.val := hmo
          omega
        have hbit : (c.mech[m]).1.testBit (firstBit s c.nd 0) = true := bitv_ne_zero.mp hmv
        simp only [hlt, decide_false, hbit, Bool.not_true, Bool.false_or] at hx
        have hv' := ih _ _ hx
        rw [bitv_xor, bitv_xor] at hv'
        exact hv'

/-! ## The trivial group, and the end-to-end theorem -/

/-- The trivial group, acting trivially on everything. -/
structure Triv : Type where

instance : Group Triv where
  mul _ _ := ⟨⟩
  one := ⟨⟩
  inv _ := ⟨⟩
  mul_assoc _ _ _ := rfl
  one_mul _ := rfl
  mul_one _ := rfl
  inv_mul_cancel _ := rfl

instance (α : Type*) : MulAction Triv α where
  smul _ a := a
  one_smul _ := rfl
  mul_smul _ _ _ := rfl

/-- Anchor of orbit `o`: the mechanism with index `o`. -/
def anchor (h0 : 0 < c.mech.length) (o : ℕ) : Fin c.mech.length :=
  if h : o < c.mech.length then ⟨o, h⟩ else ⟨0, h0⟩

theorem symmetry (h0 : 0 < c.mech.length) : Symmetry c.toDEM Triv (anchor c h0) where
  syn_equiv _ _ _ := rfl
  obs_kernel _ Y _ ho := ho
  orb_inv _ _ := rfl
  anchor_mem m := ⟨⟨⟩, by simp [anchor, CModel.toDEM, m.isLt]; rfl⟩

theorem cert_valid (h0 : 0 < c.mech.length) (w : ℕ) (h : cert c w = true) :
    CertValid c.toDEM (Tables.canonical c.toDEM) (anchor c h0) w c.mech.length := by
  intro o ho
  simp only [cert, List.all_eq_true] at h
  have := search_valid c o (w - 1) _ _ (h _ (mem_zipIdx_get c.mech ⟨o, ho⟩))
  simpa [anchor, ho, CModel.toDEM] using this

/-- **The executable checker is sound.** If `cert c w` evaluates to `true`, every undetectable logical error of
the model `c` has more than `w` mechanisms. -/
theorem cert_sound_exec (h0 : 0 < c.mech.length) (w : ℕ) (h : cert c w = true) :
    ∀ X : Finset (Fin c.mech.length), c.toDEM.IsSol X → w < X.card :=
  cert_sound (Tables.canonical_sound c.toDEM) (symmetry c h0) (norb := c.mech.length) (fun m => m.isLt)
    (cert_valid c h0 w h)

/-! ## v2: the same walk through an incidence table (supplied as data, verified once)

The kernel spends nearly all of `cert`'s time scanning every mechanism at every leaf and branch. `cert2` takes an
incidence table `tbl` (per detector, the `((syndrome, observable), index)` entries of the mechanisms touching it)
and scans only the detector's own list. The table is DATA: `incOK` checks that it lists every mechanism under every
detector it touches (completeness), and that is all soundness needs -- every use is a "for all" check, so an extra
entry can only make the checker reject more. -/

/-- A table entry: `((syndrome mask, observable mask), mechanism index)`, as `List.zipIdx` makes them. -/
abbrev Entry := (ℕ × ℕ) × ℕ

/-- Detector `v`'s list. -/
def incAt (tbl : List (List Entry)) (v : ℕ) : List Entry := tbl.getD v []

/-- The table lists every mechanism under every detector (below `nd`) it touches. -/
def incOK (tbl : List (List Entry)) : Bool :=
  (c.mech.zipIdx).all fun p => (List.range c.nd).all fun v => !p.1.1.testBit v || (incAt tbl v).contains p

/-- Rule L through the table: singles must touch `v`; a pair's second member touches the lowest bit `u` of what
the first leaves (a zero remainder falls back to every mechanism). -/
def leafOK2 (tbl : List (List Entry)) (s l v rem : ℕ) : Bool :=
  (incAt tbl v).all (fun p => !eqOn c.nd p.1.1 s || eqOn c.no p.1.2 l) &&
  (decide (rem < 2) || (incAt tbl v).all (fun a =>
      if (s ^^^ a.1.1) % 2 ^ c.nd == 0 then
        (c.mech.zipIdx).all (fun b => a.2 == b.2 || !eqOn c.nd (a.1.1 ^^^ b.1.1) s || eqOn c.no (a.1.2 ^^^ b.1.2) l)
      else
        decide (firstBit (s ^^^ a.1.1) c.nd 0 < c.nd) &&
          (incAt tbl (firstBit (s ^^^ a.1.1) c.nd 0)).all
            (fun b => a.2 == b.2 || !eqOn c.nd (a.1.1 ^^^ b.1.1) s || eqOn c.no (a.1.2 ^^^ b.1.2) l)))

/-- `search` through the table. -/
def search2 (tbl : List (List Entry)) (o : ℕ) (s l : ℕ) : ℕ → Bool
  | 0 => !(s % 2 ^ c.nd == 0) || l % 2 ^ c.no == 0
  | r + 1 =>
    if s % 2 ^ c.nd == 0 then l % 2 ^ c.no == 0
    else
      decide (firstBit s c.nd 0 < c.nd) &&
        (if r + 1 ≤ 2 then leafOK2 c tbl s l (firstBit s c.nd 0) (r + 1)
         else (incAt tbl (firstBit s c.nd 0)).all fun p => decide (p.2 < o) ||
                search2 tbl o (s ^^^ p.1.1) (l ^^^ p.1.2) r)

/-- The table is complete, and every anchor's tree passes. -/
def cert2 (tbl : List (List Entry)) (w : ℕ) : Bool :=
  incOK c tbl && (c.mech.zipIdx).all fun p => search2 c tbl p.2 p.1.1 p.1.2 (w - 1)

theorem incOK_complete {tbl : List (List Entry)} (h : incOK c tbl = true) (m : Fin c.mech.length) {v : ℕ}
    (hv : v < c.nd) (hb : (c.mech[m]).1.testBit v = true) : (c.mech[m], m.val) ∈ incAt tbl v := by
  simp only [incOK, List.all_eq_true] at h
  have := h _ (mem_zipIdx_get c.mech m) v (List.mem_range.mpr hv)
  simp only [hb, Bool.not_true, Bool.false_or, List.contains_iff_mem] at this
  exact this

theorem leafOK2_sound {tbl : List (List Entry)} (ht : incOK c tbl = true) {s l v rem : ℕ} (hv : v < c.nd)
    (hsv : s.testBit v = true) (h : leafOK2 c tbl s l v rem = true) :
    c.toDEM.LeafOK (bitv s) (bitv l) rem := by
  simp only [leafOK2, Bool.and_eq_true, List.all_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  refine ⟨fun m hm => ?_, fun hrem => ⟨⟨v, hv⟩, bitv_ne_zero.mpr hsv, fun a b ha hab hs => ?_⟩⟩
  · -- a single completing s touches v, so it is in v's list
    have hbit : (c.mech[m]).1.testBit v = true := by
      have := congrFun hm ⟨v, hv⟩
      exact bitv_ne_zero.mp (by rw [show c.toDEM.syn m = bitv (c.mech[m]).1 from rfl] at this
                                rw [this]; exact bitv_ne_zero.mpr hsv)
    have := h1 _ (incOK_complete c ht m hv hbit)
    have he : eqOn c.nd (c.mech[m]).1 s = true := (eqOn_iff _ _).mpr hm
    simp only [he, Bool.not_true, Bool.false_or] at this
    exact (eqOn_iff _ _).mp this
  · have hr : ¬ rem < 2 := by omega
    simp only [hr, decide_false, Bool.false_or, List.all_eq_true] at h2
    have ha' : (c.mech[a]).1.testBit v = true := bitv_ne_zero.mp ha
    have hx := h2 _ (incOK_complete c ht a hv ha')
    have hne : (a.val == b.val) = false := by
      rw [beq_eq_false_iff_ne]
      exact fun h' => hab (Fin.ext h')
    have he : eqOn c.nd ((c.mech[a]).1 ^^^ (c.mech[b]).1) s = true := by
      rw [eqOn_iff, bitv_xor]
      exact hs
    -- syn b = s + syn a
    have hbs : (bitv (c.mech[b]).1 : Fin c.nd → ZMod 2) = bitv (s ^^^ (c.mech[a]).1) := by
      rw [bitv_xor, ← hs]
      funext d
      simp only [Pi.add_apply]
      have := zmod2_add_self (c.toDEM.syn a d)
      calc bitv (c.mech[b]).1 d = (c.toDEM.syn a d + c.toDEM.syn a d) + bitv (c.mech[b]).1 d := by
            rw [this, zero_add]
        _ = c.toDEM.syn a d + (c.toDEM.syn a d + c.toDEM.syn b d) := by rw [add_assoc]; rfl
        _ = c.toDEM.syn a d + c.toDEM.syn b d + c.toDEM.syn a d := by
            rw [add_comm (c.toDEM.syn a d) (c.toDEM.syn a d + c.toDEM.syn b d)]
    by_cases hz : (s ^^^ (c.mech[a]).1) % 2 ^ c.nd = 0
    · have hz' : ((s ^^^ (c.mech[a]).1) % 2 ^ c.nd == 0) = true := by simpa using hz
      simp only [hz', ↓reduceIte, List.all_eq_true] at hx
      have hy := hx _ (mem_zipIdx_get c.mech b)
      simp only [hne, he, Bool.not_true, Bool.false_or] at hy
      have := (eqOn_iff _ _).mp hy
      rw [bitv_xor] at this
      exact this
    · have hz' : ((s ^^^ (c.mech[a]).1) % 2 ^ c.nd == 0) = false := by simpa using hz
      simp only [hz', Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq,
        List.all_eq_true] at hx
      obtain ⟨hu, hx⟩ := hx
      have hru : (s ^^^ (c.mech[a]).1).testBit (firstBit (s ^^^ (c.mech[a]).1) c.nd 0) = true :=
        firstBit_spec _ c.nd 0 (by omega)
      have hbu : (c.mech[b]).1.testBit (firstBit (s ^^^ (c.mech[a]).1) c.nd 0) = true := by
        have := congrFun hbs ⟨_, hu⟩
        exact bitv_ne_zero.mp (by rw [this]; exact bitv_ne_zero.mpr hru)
      have hy := hx _ (incOK_complete c ht b hu hbu)
      simp only [hne, he, Bool.not_true, Bool.false_or] at hy
      have := (eqOn_iff _ _).mp hy
      rw [bitv_xor] at this
      exact this

theorem search2_valid {tbl : List (List Entry)} (ht : incOK c tbl = true) (o : ℕ) : ∀ (r s l : ℕ),
    search2 c tbl o s l r = true → Valid c.toDEM (Tables.canonical c.toDEM) o (bitv s) (bitv l) r := by
  intro r
  induction r with
  | zero =>
    intro s l h
    by_cases hs : s % 2 ^ c.nd = 0
    · have hl : l % 2 ^ c.no = 0 := by simpa [search2, hs] using h
      rw [(bitv_eq_zero_iff s).mpr hs, (bitv_eq_zero_iff l).mpr hl]
      exact Valid.done 0
    · exact Valid.spent (fun h0 => hs ((bitv_eq_zero_iff s).mp h0))
  | succ r ih =>
    intro s l h
    by_cases hs : s % 2 ^ c.nd = 0
    · have hl : l % 2 ^ c.no = 0 := by simpa [search2, hs] using h
      rw [(bitv_eq_zero_iff s).mpr hs, (bitv_eq_zero_iff l).mpr hl]
      exact Valid.done _
    · have hs' : (s % 2 ^ c.nd == 0) = false := by simpa using hs
      simp only [search2, hs', Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at h
      obtain ⟨hv, hrest⟩ := h
      have hsv : s.testBit (firstBit s c.nd 0) = true := firstBit_spec s c.nd 0 (by omega)
      have hs0 : (bitv s : Fin c.nd → ZMod 2) ≠ 0 := fun h0 => hs ((bitv_eq_zero_iff s).mp h0)
      by_cases hr : r + 1 ≤ 2
      · simp only [hr, ↓reduceIte] at hrest
        exact Valid.leaf hs0 hr (leafOK2_sound c ht hv hsv hrest)
      · simp only [hr, ↓reduceIte, List.all_eq_true] at hrest
        refine Valid.expand ⟨firstBit s c.nd 0, hv⟩ (bitv_ne_zero.mpr hsv) fun m hmo hmv => ?_
        have hbit : (c.mech[m]).1.testBit (firstBit s c.nd 0) = true := bitv_ne_zero.mp hmv
        have hx := hrest _ (incOK_complete c ht m hv hbit)
        have hlt : ¬ m.val < o := by
          have : o ≤ m.val := hmo
          omega
        simp only [hlt, decide_false, Bool.false_or] at hx
        have hv' := ih _ _ hx
        rw [bitv_xor, bitv_xor] at hv'
        exact hv'

/-- **v2 is sound**: `cert2 c tbl w = true` implies every undetectable logical error of `c` has weight `> w`,
whatever the table holds (its completeness is part of what `cert2` checks). -/
theorem cert2_sound_exec (h0 : 0 < c.mech.length) (tbl : List (List Entry)) (w : ℕ) (h : cert2 c tbl w = true) :
    ∀ X : Finset (Fin c.mech.length), c.toDEM.IsSol X → w < X.card := by
  simp only [cert2, Bool.and_eq_true] at h
  obtain ⟨ht, h⟩ := h
  refine cert_sound (Tables.canonical_sound c.toDEM) (symmetry c h0) (norb := c.mech.length) (fun m => m.isLt) ?_
  intro o ho
  simp only [List.all_eq_true] at h
  have := search2_valid c ht o (w - 1) _ _ (h _ (mem_zipIdx_get c.mech ⟨o, ho⟩))
  simpa [anchor, ho, CModel.toDEM] using this

end CertR198.Exec
