import OneqCert.Checker
import Lean.Data.Json

/-!
  `oneqcert-check FILE.jsonl` -- runs the PROVED checker `OneqCert.Cert.accept` on every
  certificate in a JSON-lines file and prints one verdict per line.

  What is proved and what is not: `accept_sound` is a theorem about `accept`, the exact
  function called below. The JSON parsing in this file is NOT proved; it is kept small
  and total, refuses what it cannot read (a zero or negative denominator is refused, not
  read as zero), and every field it reads maps one-to-one onto a `Cert` field.

  Line format:
    {"id": str, "checks": [[var, ...], ...], "syn": [bit, ...],
     "w": [[num, den], ...], "e": [var, ...],
     "facets": [{"rows": [row, ...], "F": [var, ...], "y": [num, den]}, ...], "D": int}
-/

open Lean OneqCert

def ratOf (j : Json) : Except String ℚ := do
  let l ← fromJson? (α := List Int) j
  match l with
  | [n, d] => if d > 0 then pure ((n : ℚ) / (d : ℚ)) else throw "non-positive denominator"
  | _ => throw "a rational must be [num, den]"

def facetOf (j : Json) : Except String RawFacet := do
  let rows ← j.getObjValAs? (List Nat) "rows"
  let F ← j.getObjValAs? (List Nat) "F"
  let y ← ratOf (← j.getObjVal? "y")
  pure ⟨rows, F, y⟩

def certOf (j : Json) : Except String (String × Cert) := do
  let id ← j.getObjValAs? String "id"
  let checks ← j.getObjValAs? (List (List Nat)) "checks"
  let syn ← j.getObjValAs? (List Nat) "syn"
  let w ← (← j.getObjValAs? (Array Json) "w").toList.mapM ratOf
  let e ← j.getObjValAs? (List Nat) "e"
  let facets ← (← j.getObjValAs? (Array Json) "facets").toList.mapM facetOf
  let D ← j.getObjValAs? Nat "D"
  pure (id, ⟨checks, syn, w, e, facets, D⟩)

def main (args : List String) : IO UInt32 := do
  let [path] := args
    | IO.eprintln "usage: oneqcert-check FILE.jsonl"
      return 2
  let lines ← IO.FS.lines path
  let mut acc := 0
  let mut refd := 0
  let mut bad := 0
  for ln in lines do
    if ln.trimAscii.isEmpty then continue
    match Json.parse ln >>= certOf with
    | .error msg =>
      IO.println s!"MALFORMED {msg}"
      bad := bad + 1
    | .ok (id, c) =>
      if c.accept then
        IO.println s!"{id} ACCEPT"
        acc := acc + 1
      else
        IO.println s!"{id} REFUSE"
        refd := refd + 1
  IO.println s!"SUMMARY accepted={acc} refused={refd} malformed={bad}"
  return 0
