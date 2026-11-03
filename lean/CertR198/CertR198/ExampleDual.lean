import CertR198.Tree
import Mathlib.Data.ZMod.Defs

/-!
# The CDP5 dual raise is not vacuous: a prune only the raise can make

Two disjoint copies of one gadget: detectors `0..4` with mechanisms `{0,2}`, `{0,1,3}`, `{0,4}`,
and the same on `5..9`. Every mechanism is its own orbit (`orb m = m`, the trivial group), and
the state is anchor orbit `0` with all ten detectors firing and remaining budget `4` -- the
checker's raise needs `rem ≥ 4`.

* the packing over `CT_ORBIT[0]` counts 2 (one per copy), and the plain cover
  `Σ ⌊2520 / c_d⌋ = 10080 = 2520 · 4` does not exceed `2520 · 4`: `plain_does_not_prune`;
* raising every detector in index order adds `420` at detectors `2, 4, 7, 9` (the mechanisms
  `{0,2}` and `{0,4}` carry `2100 < 2520`): `11760 > 10080`, `raise_prunes`;
* `bound5_sound` turns that into "every completion has more than 4 mechanisms", and the six
  mechanisms together are a completion of weight 6.

`ccheck.c` agrees on this instance: `bound_ok` returns 0 under a CDP3 header and 1 under a CDP5
one (`tests/test_r198_circuit_distance_exact.py`, through `ccheck_oracle.c`).
-/

open Finset

namespace CertR198.ExampleDual

/-- Mechanism `m` (copy `m / 3`, base detector `5 · (m / 3)`, kind `m % 3`) touches `{b, b+2}`,
`{b, b+1, b+3}` or `{b, b+4}`. -/
def touchB (m : Fin 6) (d : Fin 10) : Bool :=
  let b := 5 * (m.val / 3)
  d.val == b || (m.val % 3 == 0 && d.val == b + 2) ||
    (m.val % 3 == 1 && (d.val == b + 1 || d.val == b + 3)) || (m.val % 3 == 2 && d.val == b + 4)

/-- The model (observables play no part in a bound). -/
def E : DEM (Fin 6) (Fin 10) (Fin 1) where
  syn := fun m d => if touchB m d then 1 else 0
  obs := fun _ _ => 0
  orb := fun m => m.val

/-- The exact tables, `CT_ORBIT` included. -/
abbrev T : Tables (Fin 6) (Fin 10) := Tables.canonical E

/-- All ten detectors fire. -/
def s : Fin 10 → ZMod 2 := fun _ => 1

/-- The raise prunes at remaining budget 4 (every detector raised, in index order). -/
theorem raise_prunes : E.BoundOK5 scale T (List.finRange 10) (List.finRange 10) 0 s 4 := by
  unfold DEM.BoundOK5
  decide +kernel

/-- Neither the packing nor the plain cover prunes there. -/
theorem plain_does_not_prune : ¬ E.BoundOK5 scale T (List.finRange 10) [] 0 s 4 := by
  unfold DEM.BoundOK5
  decide +kernel

/-- So every completion of `s` has more than 4 mechanisms ... -/
theorem completion_heavier (Y : Finset (Fin 6)) (hs : E.synSum Y = s) : 4 < Y.card :=
  DEM.bound5_sound (Tables.canonical_sound E) raise_prunes (fun _ _ => Nat.zero_le _) hs

/-- ... and the six mechanisms are one. -/
theorem all_six_complete : E.synSum univ = s := by
  unfold DEM.synSum
  decide +kernel

end CertR198.ExampleDual
