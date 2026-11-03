# OneqCert — a decoder-optimality checker whose soundness is a Lean theorem

P2 certifies that a decoded quantum error correction is **minimum weight** by shipping a
dual certificate. This project proves, in the Lean 4 kernel, that such a certificate
means what it claims. It also provides the checker as ordinary executable Lean, so the
theorem is about the code that runs.

## The headline theorem

`OneqCert.Cert.accept_sound` (`OneqCert/Checker.lean`):

> If `accept c = true`, then the candidate satisfies every check of the instance, and
> **no correction over the instance's variables that satisfies every check weighs
> less.**

`accept` is the function the `oneqcert-check` binary calls on every certificate
(`Main.lean`). It re-derives everything from the certificate's own data: the syndrome
of the candidate, the support and parity of every facet (single checks and XORs of
checks), the facet's wrong parity, y ≥ 0, the weight lattice, and the safe bound.
Nothing a producer asserts is trusted.

| theorem | file | content |
|---|---|---|
| `facet_violation`, `facet_inequality` | `SafeBound.lean` | the odd-set inequality `1 − card F ≤ card((S∖F)∩e) − card(F∩e)` |
| `sum_coef`, `facet_sum_le` | `WeakDuality.lean` | the ±1 load, summed over a correction, is that signed count |
| `safe_bound_le_weight` | `WeakDuality.lean` | L ≤ weight(e) for any rational weights and any y ≥ 0 |
| `xor_parity`, `xor_list_parity` | `WeakDuality.lean`, `Checker.lean` | XOR facets meet the parity hypothesis |
| `lattice_step`, `sum_on_lattice` | `Checker.lean` | with weights in (1/D)ℤ, L > U − 1/D and L ≤ OPT force OPT ≥ U |
| **`Cert.accept_sound`** | `Checker.lean` | **accepted ⇒ minimum-weight correction** |

Every theorem rests only on `propext`, `Classical.choice` and `Quot.sound`.
`AxiomCheck.lean` prints them, and `tools/check_lean_certificate.py` refuses anything
else, `sorryAx` included.

## The headline's statement is pinned (2026-10-22)

Sources, axioms and a theorem list do not bind what a theorem SAYS. In this repo's R211
gate, an adversary weakened the headline (bound replaced by `≤ 1`). It still built on the
standard axioms, and the rebuilt receipt passed. The same was possible here: a broken proof
"fixed" by weakening `accept_sound` would have kept its name, its axioms and a green gate.

- `OneqCert/Statements.lean` restates `accept_sound`'s full type as
  `accept_sound_pinned` and elaborates it from the theorem. Any weakening fails to build
  there.
- The gate pins that file's reviewed text (`STATEMENTS_SHA256`) and requires both names in
  `AxiomCheck.lean`. A deliberate statement change therefore has to edit the pinned file
  and the gate's constant.
- `check_lean_certificate.py --mutants DIR` applies two weakenings Lean accepts in
  `Checker.lean`: the conclusion replaced by `True`, and the minimality claim kept as text
  but disjoined with `True`. It records that each fails at `Statements.lean`
  (`evidence/lean_safe_bound_mutants.json`). The gate refuses a missing or stale drill, and
  a mutant that built or failed anywhere else.

## Applied to the paper

`tools/export_p2_certificates.py` regenerates P2's exact certificates by re-running the
paper's own producer, and writes them to `evidence/p2_certificates.jsonl.gz`. The
producer's output root is redirected, so no committed evidence is touched. Every
regenerated per-shot record must equal the committed one in every field except the
certificate hash: instance, candidate, tier, verdict, answer, objective. Hash drift (a
different dual witness for the same claim) is listed shot by shot in
`evidence/p2_certificates_export.json`. The compiled checker then judges every exported
certificate, and the verdict counts go into `evidence/lean_safe_bound.json`.

## What is trusted, stated plainly

- **The Lean kernel and compiler.** The theorem is about `accept` as Lean code. The
  binary is Lean's compilation of it, so compiler correctness is assumed, as it is for
  every compiled checker.
- **The JSON reader in `Main.lean`.** It is not proved. It is small, maps each field
  one-to-one onto `Cert`, and refuses what it cannot read: a zero or negative
  denominator is refused, not read as zero.
- **Scope of "minimum".** The minimum is over corrections supported on the instance's
  variables `0 … n−1`, which is the decoding problem's domain.

## Building

The build directory must sit on a short path. Mathlib's package tree exceeds Windows'
260-character limit under a deep checkout.

    python tools/check_lean_certificate.py --build C:/Users/<you>/lq/OneqCert
    python tools/check_lean_certificate.py --mutants C:/Users/<you>/lq/OneqCert

The first command copies these sources, builds the library and the `oneqcert-check`
binary, prints the axioms, runs the binary on every P2 certificate, and writes the
receipt. The second runs the statement-pin drill on that build and restores it.

The wall runs `--check`, which needs no Lean. It refuses:
- a receipt whose proved sources or judged certificates changed;
- any non-standard axiom;
- any refused or unparsed certificate;
- a changed pinned statement;
- a missing headline theorem;
- a missing, stale or unrefused drill.
