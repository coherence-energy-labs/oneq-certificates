import OneqCert.Checker

/-!
# The headline statement, pinned

`Cert.accept_sound` is what the P2 claim rests on: an accepted certificate proves its candidate is a
minimum-weight correction. This file restates the theorem's type in full and elaborates it from the theorem.
If `accept_sound` is ever weakened while its name and axioms stay the same (for example, a broken proof
"fixed" by a weaker statement), the build fails here.

`tools/check_lean_certificate.py` pins this file's text and requires `accept_sound_pinned` in the receipt.
Its `--mutants` drill records Lean refusing weakened forms of `accept_sound` AT this file. (Found 2026-10-22,
when an adversary passed the R211 gate in this repo with a weakened headline that built on standard axioms.)
-/

namespace OneqCert.Statements

open OneqCert

/-- An accepted certificate's candidate satisfies every check and lies on the instance's variables, and no
correction over those variables that satisfies every check weighs less. -/
theorem accept_sound_pinned (c : Cert) (h : c.accept = true) :
    c.Consistent c.e.toFinset ∧ c.e.toFinset ⊆ Finset.range c.w.length ∧
      ∀ e' ⊆ Finset.range c.w.length, c.Consistent e' →
        c.weight c.e.toFinset ≤ c.weight e' :=
  Cert.accept_sound c h

end OneqCert.Statements
