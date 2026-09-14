# Re-freezing a trust module

**Evidence class:** [ESTATE-BUILT] the protocol for re-freezing a trust module that is built and shipping in tools/frozen/.


`tools/frozen/` holds byte-identical snapshots of the checker and
producer modules. Two gates compare live code against them on every
`gate_all` run:

| gate | what it forbids |
|---|---|
| `tools/checker_equivalence_gate.py` | any change to a **verdict**, a **reason string**, a primal weight or a dual bound |
| `tools/producer_equivalence_gate.py` | any change to an emitted **certificate**, **bound**, **tier** or **branch tree** |

This is deliberate friction. A checker's value is its verdicts, and a
verdict that moves silently turns every downstream receipt into a lie.

**But sometimes a verdict SHOULD move** — a genuine bug fix in a checker
changes what it accepts, and that is the whole point of fixing it. Being
blocked with no instructions is how a careful person concludes the gate
is broken and deletes it. So:

---

## If the gate goes red

**First, assume you did not mean to change behaviour.** That is true
almost every time. Read the divergence it printed: it names the case, the
frozen verdict and the live one. Refactors that "obviously cannot change
anything" are exactly what this catches — three of the six holes found
while building it were of that shape.

**If the change is intentional**, pick the smallest option that fits.

### Option 1 — the reason string improved, the verdict did not

Add an entry to `_REASON_ALIASES` in the checker gate: the frozen
pattern, the live pattern, and *why*. The gate still requires the
accepted flag and every numeric field to be identical, so an aliased
change cannot hide a verdict move. An alias that goes **unused** also
fails, because a standing permission for a case the corpus cannot reach
is somewhere a real divergence could later hide.

### Option 2 — a verdict genuinely changes

This is a re-freeze, and it is a **reviewable act**, not a chore:

1. **Land the behaviour change and its test first**, in its own commit,
   with the gate red. A red gate in history is the record that a verdict
   moved on purpose.
2. **Write down which inputs decide differently and why.** "Refactor" is
   not an answer here; if you cannot name the class of inputs whose
   verdict changed, you do not yet know what you changed.
3. Copy the module into `tools/frozen/` (same filename — the package
   layout is what lets `tjoin_exact`'s relative import resolve untouched).
4. Update the `FROZEN_SHA256` entry in the gate that reads it.
5. **Re-run `--prove-sensitive` on both gates.** A new baseline is a new
   corpus target, and a mutation that was caught against the old bytes
   can be invisible against the new ones.
6. Commit the re-freeze **separately**, quoting the review from step 2.

### Option 3 — you are adding a module to the baseline

Snapshot it, add its sha256, and add at least one **behavioural mutation**
for it to the gate's mutation table. A frozen module with no mutation
proving the corpus reaches it is decoration: it will compare equal
forever whether or not anything is actually being tested.

---

## What the snapshots are not

They are **not maintained code**. Do not fix a bug in `tools/frozen/`.
Do not lint it, reformat it, or let a tool rewrite its line endings — the
sha256 check fails closed and will tell you, but the point is that
editing the reference is how a differential gate quietly becomes a
comparison of a thing against itself.

If a snapshot is corrupted, restore it from git history rather than
regenerating it from live source. Regenerating from live source makes the
gate pass by definition and proves exactly nothing.
