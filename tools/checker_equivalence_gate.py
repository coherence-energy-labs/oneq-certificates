#!/usr/bin/env python3
r"""DIFFERENTIAL EQUIVALENCE: the refactored checker decides exactly what
the frozen checker decided, on every input, for the same stated reason.

WHY THIS EXISTS. Eight functions across the trust modules carried a
standards waiver for being long -- 1,339 lines, the longest 247. Length
in a CHECKER is not a style complaint: the entire claim of a certifying
algorithm is that a third party can READ the checker and believe it, so
an unreadable checker weakens the result it is supposed to establish.
But refactoring a checker is the most dangerous edit in this repository.
A checker's value is its verdicts, and a verdict that moves silently
turns every downstream receipt into a lie.

"The test suite still passes" is NOT evidence the refactor was safe. The
suite covers the inputs somebody thought of. What is needed is evidence
over inputs nobody thought of, including the malformed ones -- because a
checker's rejection path is exactly as load-bearing as its acceptance
path, and it is the path least likely to be covered.

So: tools/frozen/ holds byte-identical snapshots of the trust modules as
they stood before the refactor (sha256 re-verified here on every run).
This gate runs both the frozen and the live checker over a shape-varied
corpus and requires the FULL result to agree -- accepted, reason string,
primal weight, dual bound -- on every single case. Not "both rejected":
rejected FOR THE SAME STATED REASON.

CORPUS SHAPE, deliberately non-uniform. A corpus that is uniform in
shape hides defects even at high coverage (this estate has a scar from
exactly that: a false merge by address collision survived 382 tests, 90%
coverage and 97% mutation because every case had the same shape). So the
generator varies, independently and adversarially:

  * instance    -- m and n, check degree (including degree-0 and
                   degree-1 rows), overlapping vs disjoint supports
  * weights     -- uniform 1.0, random floats, ints, Fractions,
                   zero weights, and NEGATIVE weights (must reject)
  * syndrome    -- all-zero, random, all-ones, wrong length
  * certificate -- genuine (from the real producers), and then PERTURBED
                   along each axis independently: negated dual, dual on a
                   non-facet (even-parity F), facet naming a nonexistent
                   check, F not a subset of N(j), support with an unknown
                   variable, support with a duplicate, empty support
  * type abuse  -- strings, None, floats where exact demands rationals,
                   numpy scalars, ragged nesting  (the REJECT path)
  * tier BD     -- valid trees, trees with an unproven leaf, trees whose
                   branch variable repeats, bound_only thresholds

SENSITIVITY IS PROVEN, NOT ASSUMED. An equivalence gate that compares a
module against itself passes trivially and forever. `--prove-sensitive`
injects behavioural mutations into a COPY of the live checker (flip an
inequality, widen a tolerance, drop a validation, soften a reason
string) and requires this gate to catch each one. A mutation this
corpus cannot see is a hole in the corpus, and the gate says so.

    python tools/checker_equivalence_gate.py
    python tools/checker_equivalence_gate.py --prove-sensitive
    python tools/checker_equivalence_gate.py --cases 4000 --json evidence/...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import pathlib
import random
import re
import sys
import tempfile
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# The freeze manifest. These are the bytes the refactor must not change
# the behaviour of. Recorded here so a silent edit to the "reference"
# cannot quietly make the comparison vacuous -- the classic way a
# differential gate rots into a no-op.
FROZEN_SHA256 = {
    # RE-FROZEN 2026-08-14 (protocol option 2; behaviour commit 8cf6863).
    # The class of inputs whose verdict moved: certificates whose float
    # dual gap is under the 1e-6 screen but whose EXACT safe bound falls
    # short of the exact primal (ulp-scale shortfalls, observed 1-5 *
    # 2^-55 on 3 of 1500 corpus cases) -- previously ACCEPTED on the
    # tolerance alone, now REFUSED as NOT PROVEN (exact). The witness
    # that forced this is pinned in tests/test_exact_dual_repair.py:
    # a genuinely feasible dual with bound 1.0 certified a 1.0000005
    # correction while {1,2} weighed 1.0, and the receipt said gap 0.0.
    "qldpc_check.py":
        "1de3881e541a2b2031e938f11482b9aaf2109398f27083fcda80d8ae0b081b82",
    "tjoin_exact.py":
        "9d326865e47f16556ab64cd7469cafd2939e85abebec785f71321f3c9717e9ac",
    "matching_cert.py":
        "8743421b7e1eb1def0ad63eba66d95bf14d33c67a31a43216624b9a1b1009fcd",
}


def _norm_bytes(p: pathlib.Path) -> bytes:
    """Content with line endings normalised, so the pin is PORTABLE.

    Found by X-1: the manifest was pinned to the bytes in MY checkout,
    which git had converted to CRLF. A fresh clone gets LF, so the
    sha256 differed and BOTH equivalence gates failed for everyone but
    me -- "FROZEN SNAPSHOT ALTERED" on a snapshot nobody had touched.
    That is precisely the green-only-in-your-tree failure X-1 exists to
    catch, in a gate built to catch drift.

    Hashing normalised content pins the CONTENT rather than a checkout
    convention, and is stable across platforms and core.autocrlf
    settings.
    """
    return p.read_bytes().replace(b"\r\n", b"\n")


def _verify_freeze() -> None:
    d = ROOT / "tools" / "frozen"
    for name, want in FROZEN_SHA256.items():
        p = d / name
        if not p.exists():
            raise SystemExit(f"FROZEN SNAPSHOT MISSING: {p}")
        got = hashlib.sha256(_norm_bytes(p)).hexdigest()
        if got != want:
            raise SystemExit(
                f"FROZEN SNAPSHOT ALTERED: {name}\n  expected {want}\n"
                f"  found    {got}\nThe reference is the whole point; it "
                f"is not editable. Restore it from git history -- "
                f"regenerating it from live source makes this gate pass "
                f"by definition. See docs/REFREEZE_PROTOCOL.md.")


# ---------------------------------------------------------------- corpus

def _rand_instance(rng: random.Random):
    """A syndrome-LP instance whose SHAPE varies, not just its values."""
    style = rng.choice(("dense", "sparse", "lowdeg", "degenerate",
                        "overlap", "disjoint"))
    if style == "disjoint":
        n = rng.randint(4, 14)
        m = rng.randint(1, max(1, n // 2))
        idx = list(range(n))
        rng.shuffle(idx)
        checks, at = [], 0
        for _ in range(m):
            k = rng.randint(1, 3)
            checks.append(tuple(sorted(idx[at:at + k])))
            at += k
            if at >= n:
                break
    else:
        n = rng.randint(3, 16)
        m = rng.randint(1, 8)
        if style == "lowdeg":
            lo, hi = 0, 2
        elif style == "dense":
            lo, hi = max(1, n // 2), n
        elif style == "degenerate":
            lo, hi = 1, 2
        else:
            lo, hi = 1, min(n, 5)
        checks = []
        for _ in range(m):
            k = rng.randint(lo, max(lo, min(hi, n)))
            checks.append(tuple(sorted(rng.sample(range(n), k))))
    checks = [c for c in checks] or [()]
    n = max([max(c) + 1 for c in checks if c] + [1])
    return checks, n


def _maybe_duplicate_indices(rng: random.Random, checks):
    """Plant a DUPLICATE variable index inside a check support.

    Found as a corpus hole by --prove-sensitive: nothing in the generated
    corpus ever produced one, so the `_canonical_supports` refusal -- the
    adversarial audit's one genuine hole, where an index means 'member
    once' under set semantics but CANCELS under GF(2) -- was never
    compared between frozen and live at all. A branch the corpus never
    reaches is a branch the equivalence claim does not cover.
    """
    if rng.random() < 0.12 and checks and any(checks):
        k = rng.randrange(len(checks))
        c = list(checks[k])
        if c:
            c.append(rng.choice(c))
            checks = list(checks)
            checks[k] = tuple(c)
    return checks


def _synthetic_cert(rng: random.Random, cert_cls, checks, syn, n):
    """Build a certificate DIRECTLY, with facet duals guaranteed non-empty.

    The second corpus hole --prove-sensitive found. On small random
    instances the real producer usually returns no certificate at all, so
    `cert.facet_duals` was empty in most cases and the entire facet
    validation section of the checker -- negative dual, parity, subset,
    duplicate facet, redundant-row naming -- went uncompared. Synthesising
    duals from the instance forces those branches to be exercised on
    every case, whether or not any of them is a VALID certificate: the
    checker must agree with itself about rejections too.
    """
    m = len(checks)

    # e FIRST, then s := He, then the facets against THAT syndrome. The
    # order matters: a facet's validity is defined by the parity of s_j,
    # so facets built against a syndrome the certificate does not carry
    # would be invalid for a reason the corpus did not intend.
    sup_e = tuple(sorted(rng.sample(range(n), rng.randint(0, min(n, 4)))))
    es = set(sup_e)
    syn = [len(es & set(c)) % 2 for c in checks]

    def valid_facet():
        """(j, F, y) that PASSES every facet rule: j in range, F a subset
        of N(j) with the WRONG parity, y > 0. The later branches of the
        checker are only reachable behind one of these."""
        if not m:
            return None
        for _ in range(8):
            j = rng.randrange(m)
            sup = sorted(set(checks[j]))
            par = syn[j] if j < len(syn) else 0
            sizes = [s for s in range(len(sup) + 1) if s % 2 != par]
            if sizes:
                F = tuple(sorted(rng.sample(sup, rng.choice(sizes))))
                return (j, F, round(rng.uniform(0.05, 1.0), 3))
        return None

    # ONE FLAW PER SPECIMEN. The checker returns on the FIRST violation,
    # so a certificate carrying several defects only ever exercises the
    # earliest one -- stacking them made `drop-duplicate-facet` and
    # `drop-redundant-row-validation` alternately invisible, each
    # shadowing the other depending on insertion order. So: build a body
    # of VALID facets, then introduce exactly one deliberate flaw, at a
    # known position. A fail-fast checker needs one probe per specimen.
    flaw = rng.choice(("clean", "clean", "duplicate", "bad_rows",
                       "wrong_parity", "not_subset", "negative_dual",
                       "huge_dual", "zero_dual"))
    fd = [f for f in (valid_facet() for _ in range(rng.randint(1, 3)))
          if f is not None]

    if flaw == "duplicate" and fd:
        fd.insert(rng.randrange(len(fd) + 1), fd[0])
    elif flaw == "bad_rows" and m:
        rows = rng.choice([(rng.randrange(m),), (),
                           (rng.randrange(m), m + 7), (m + 3, m + 4)])
        fd.insert(0, (rows, (), round(rng.uniform(0.1, 1.0), 3)))
    elif flaw == "wrong_parity" and fd:
        j, F, y = fd[0]
        nb = sorted(set(checks[j]) - set(F))
        if nb:
            fd[0] = (j, tuple(sorted(set(F) | {nb[0]})), y)
    elif flaw == "not_subset" and fd:
        j, F, y = fd[0]
        fd[0] = (j, tuple(sorted(set(F) | {n + 11})), y)
    elif flaw == "negative_dual" and fd:
        j, F, _y = fd[0]
        fd[0] = (j, F, -0.75)
    elif flaw == "huge_dual" and fd:
        j, F, _y = fd[0]
        fd[0] = (j, F, 5e5)
    elif flaw == "zero_dual" and fd:
        j, F, _y = fd[0]
        fd[0] = (j, F, 0.0)

    # a VALID redundant-check facet too, so the GF(2) row-sum branch is
    # reached on its success path and not only on its refusal path.
    if m >= 2 and flaw == "clean" and rng.random() < 0.35:
        rows = tuple(sorted(rng.sample(range(m), 2)))
        sup: set = set()
        par = 0
        for r in rows:
            sup ^= set(checks[r])
            par ^= (syn[r] if r < len(syn) else 0)
        sizes = [s for s in range(len(sup) + 1) if s % 2 != par]
        if sup and sizes:
            F = tuple(sorted(rng.sample(sorted(sup), rng.choice(sizes))))
            fd.append((rows, F, round(rng.uniform(0.05, 0.8), 3)))
    bd = [(rng.randrange(n), round(rng.uniform(0.0, 1.5), 3))
          for _ in range(rng.randint(0, 2))] if n else []

    # THE SYNDROME IS DERIVED FROM THE ERROR (computed above), not drawn
    # independently. This was the dominant shadow and the deepest corpus
    # hole: step 1 of the checker recomputes the syndrome of the claimed
    # error and rejects on mismatch BEFORE any facet is examined. With an
    # independently drawn syndrome a random support essentially never
    # matched, so the whole facet-validation section -- the part the
    # equivalence claim is mostly about -- was reached only by accident,
    # which is why `drop-redundant-row-validation` was caught on two
    # seeds and missed on a third.
    return cert_cls(error_support=sup_e, facet_duals=tuple(fd),
                    box_duals=tuple(bd)), syn


def _rand_weights(rng: random.Random, n: int, exact: bool):
    kind = rng.choice(("uniform", "int", "float", "frac", "zero", "neg"))
    if kind == "uniform":
        v = [Fraction(1) if exact else 1.0] * n
    elif kind == "int":
        v = [Fraction(rng.randint(1, 5)) if exact else float(rng.randint(1, 5))
             for _ in range(n)]
    elif kind == "float":
        v = [Fraction(rng.randint(1, 40), 8) if exact
             else round(rng.uniform(0.2, 5.0), 3) for _ in range(n)]
    elif kind == "frac":
        v = [Fraction(rng.randint(1, 9), rng.randint(1, 7)) if exact
             else rng.randint(1, 9) / rng.randint(1, 7) for _ in range(n)]
    elif kind == "zero":
        v = [Fraction(0) if exact else 0.0] * n
    else:                                   # NEGATIVE: must be rejected
        v = [Fraction(1) if exact else 1.0] * n
        v[rng.randrange(n)] = Fraction(-1) if exact else -1.0
    return {i: v[i] for i in range(n)}


def _rand_syndrome(rng: random.Random, m: int):
    kind = rng.choice(("zero", "rand", "ones", "short", "long"))
    if kind == "zero":
        return [0] * m
    if kind == "ones":
        return [1] * m
    if kind == "short":
        return [rng.randint(0, 1) for _ in range(max(0, m - 1))]
    if kind == "long":
        return [rng.randint(0, 1) for _ in range(m + 1)]
    return [rng.randint(0, 1) for _ in range(m)]


def _perturb_cert(rng: random.Random, cert_cls, cert, checks, n):
    """Bend a genuine certificate along ONE axis. The rejection path is
    as load-bearing as the acceptance path and far less covered."""
    axis = rng.choice((
        "none", "negate_dual", "even_parity_F", "bad_check_index",
        "F_not_subset", "unknown_var", "dup_var", "empty_support",
        "huge_dual", "extra_box", "negative_box", "drop_dual",
        "halved_duals"))
    fd = list(cert.facet_duals)
    bd = list(cert.box_duals)
    sup = list(cert.error_support)
    if axis == "negate_dual" and fd:
        k = rng.randrange(len(fd))
        j, F, y = fd[k]
        fd[k] = (j, F, -abs(y) - 0.5)
    elif axis == "even_parity_F" and fd:
        k = rng.randrange(len(fd))
        j, F, y = fd[k]
        # j may name a SINGLE check (int) or a REDUNDANT one (tuple of
        # rows); flipping parity has to work for both.
        rows = j if isinstance(j, (tuple, list)) else (j,)
        nb: set = set()
        for r in rows:
            if isinstance(r, int) and 0 <= r < len(checks):
                nb ^= set(checks[r])
        alt = tuple(sorted(set(F) ^ ({min(nb)} if nb else set())))
        fd[k] = (j, alt, y)
    elif axis == "bad_check_index" and fd:
        k = rng.randrange(len(fd))
        _j, F, y = fd[k]
        fd[k] = (len(checks) + 7, F, y)
    elif axis == "F_not_subset" and fd:
        k = rng.randrange(len(fd))
        j, F, y = fd[k]
        fd[k] = (j, tuple(sorted(set(F) | {n + 3})), y)
    elif axis == "unknown_var":
        sup.append(n + 5)
    elif axis == "dup_var" and sup:
        sup.append(sup[0])
    elif axis == "empty_support":
        sup = []
    elif axis == "huge_dual" and fd:
        k = rng.randrange(len(fd))
        j, F, y = fd[k]
        fd[k] = (j, F, 1e9)
    elif axis == "extra_box":
        bd.append((rng.randrange(n), 3.5))
    elif axis == "negative_box":
        bd.append((rng.randrange(n), -2.0))
    elif axis == "drop_dual" and fd:
        fd.pop(rng.randrange(len(fd)))
    elif axis == "halved_duals" and fd:
        # A FEASIBLE dual that is LOOSE -- the shape this corpus lacked.
        # Discovered by prove-sensitive going blind on accept-nonzero-gap
        # after the exact gate landed: the mutation's historical catches
        # were verdict flips (tolerance-only acceptance), and once the
        # exact gate refused those same cases no corpus case reached the
        # float screen at all. Halving every y keeps feasibility
        # unconditionally (a halved positive load shrinks; a halved
        # negative load stays <= 0 <= w) and, on an accepted base with
        # bound == primal, lands the new bound at or below primal/2 --
        # a gap far above 1e-6 and never past the primal, so it exercises
        # exactly the NOT-PROVEN screen and nothing else. Halving a float
        # is exact, so no rounding enters the corpus.
        fd = [(j, F, y / 2) for (j, F, y) in fd]
    return cert_cls(error_support=tuple(sup), facet_duals=tuple(fd),
                    box_duals=tuple(bd)), axis


_TYPE_ABUSE = (
    "str_weight", "none_weight", "str_syndrome", "none_cert_support",
    "float_in_exact", "nested_ragged", "bool_syndrome", "str_check_index",
)


def _abuse(rng: random.Random, kind, checks, weights, syn, sup):
    """Malformed input. A checker MUST refuse it, and the refactor must
    refuse it identically -- same reason string, not merely same bool."""
    checks = [tuple(c) for c in checks]
    weights = dict(weights)
    syn = list(syn)
    sup = tuple(sup)
    if kind == "str_weight" and weights:
        weights[next(iter(weights))] = "heavy"
    elif kind == "none_weight" and weights:
        weights[next(iter(weights))] = None
    elif kind == "str_syndrome" and syn:
        syn[0] = "1"
    elif kind == "none_cert_support":
        sup = (None,) + sup
    elif kind == "float_in_exact" and weights:
        weights[next(iter(weights))] = 0.1
    elif kind == "nested_ragged" and checks:
        checks[0] = (checks[0], 1) if checks[0] else ((),)
    elif kind == "bool_syndrome" and syn:
        syn[0] = True
    elif kind == "str_check_index" and checks and checks[0]:
        checks[0] = ("0",) + tuple(checks[0][1:])
    return checks, weights, syn, sup


def _break_tree(rng: random.Random, tree):
    """Damage a branch-dual tree the way a forger would.

    Node forms are plain tuples: ("branch", i, child0, child1) and
    ("bound", facet_duals). A checker that walks a tree must refuse a
    damaged one for a NAMED reason, and the refactor has to keep naming
    the same one -- so the corpus has to contain damaged trees.
    """
    nodes = []

    def walk(t, path):
        if isinstance(t, tuple) and t and t[0] == "branch":
            nodes.append((t, path))
            walk(t[2], path + [2])
            walk(t[3], path + [3])
        elif isinstance(t, tuple) and t and t[0] == "bound":
            nodes.append((t, path))

    walk(tree, [])
    if not nodes:
        return tree
    target, path = nodes[rng.randrange(len(nodes))]
    if target[0] == "bound":
        mode = rng.choice(("empty_duals", "drop_one", "inflate"))
        duals = list(target[1]) if len(target) > 1 and target[1] else []
        if mode == "empty_duals" or not duals:
            new = ("bound", ())
        elif mode == "drop_one":
            duals.pop(rng.randrange(len(duals)))
            new = ("bound", tuple(duals))
        else:
            j, F, y = duals[0]
            duals[0] = (j, F, y * 1000 if isinstance(y, (int, float)) else y)
            new = ("bound", tuple(duals))
    else:
        mode = rng.choice(("swap_children", "bad_var", "truncate"))
        if mode == "swap_children":
            new = ("branch", target[1], target[3], target[2])
        elif mode == "bad_var":
            new = ("branch", target[1] + 9999, target[2], target[3])
        else:
            new = ("bound", ())

    def rebuild(t, path):
        if not path:
            return new
        i = path[0]
        parts = list(t)
        parts[i] = rebuild(t[i], path[1:])
        return tuple(parts)

    return rebuild(tree, path)


def _forced_tree_cases(rng: random.Random):
    """Instances and trees built to REACH the tree walker's every branch.

    Profiling the producer showed why four BD mutations were invisible:
    on random small instances the flat LP closes immediately, so 150 of
    152 generated trees were a single `bound` leaf at depth 0. Branch
    nodes, `infeasible` leaves, GF(2) row combinations and a non-empty
    S1 simply never occurred, and the code validating them was never
    compared between frozen and live at all.

    Getting here took one wrong turn worth recording. The first attempt
    built an instance whose rows CONTRADICTED the claimed error, so
    every case died at step 1 ("NOT A VALID CORRECTION") and the walker
    never ran -- the tree logic stayed just as dark, while the corpus
    looked bigger. The requirement is subtler: the syndrome must be
    consistent with e (so step 1 passes), and the infeasibility must
    arise only in the REDUCED instance after variables are fixed.

    The construction that satisfies both, with e = {i}:

        row 0 = (i,)    s=1     pins x_i = 1
        row 1 = (i, j)  s=1
        row 2 = (j,)    s=0

    Fixing x_j = 0 reduces row 1 to support {i} -- non-empty, so an
    `infeasible` leaf naming it is a lie the checker must name. Fixing
    x_j = 0 reduces row 2 to empty support with EVEN parity: feasible,
    another distinct lie. Fixing x_j = 1 makes rows 0 and 1 reduce to the
    same support with OPPOSITE parity, so together they XOR to empty with
    odd parity -- a GENUINE infeasibility that no single row shows, which
    is exactly what the row-combination branch exists for. And w_j is
    chosen far larger than U so that the 1-side bound leaf crosses ONLY
    because the fixed weight W1 is added.

    Yields (checks, weights, syndrome, error_support, U, tree, label).
    """
    i, j, k = 0, 1, 2
    wj = rng.randint(4, 9)          # >> U, so W1 is what makes it cross
    w = {i: 1, j: wj, k: rng.randint(1, 3)}
    checks = [(i,), (i, j), (j,)]
    syn = [1, 1, 0]                 # = |e & row| mod 2 for e = {i}
    e = (i,)
    U = 1                           # the weight of e

    # a leaf that legitimately crosses with S1 empty: one facet on row 0,
    # whose reduced support is {i} with odd parity, so F = {} is a real
    # facet and y = w_i = 1 gives L = 1 > U - step = 0.
    good = ("bound", ((0, (), 1),))

    yield (checks, w, syn, e, U, good, "bound-leaf-crosses")
    yield (checks, w, syn, e, U,
           ("branch", j, good, ("infeasible", (0, 1))),
           "infeasible-row-combination")
    yield (checks, w, syn, e, U,
           ("branch", j, good, ("bound", ())), "fixed-weight-crossing")
    yield (checks, w, syn, e, U,
           ("branch", i, ("branch", i, good, good), good),
           "rebranch-on-fixed-var")
    yield (checks, w, syn, e, U,
           ("branch", j, ("infeasible", (1,)), good),
           "infeasible-xor-nonempty")
    yield (checks, w, syn, e, U,
           ("branch", j, ("infeasible", (2,)), good),
           "infeasible-even-parity")
    yield (checks, w, syn, e, U,
           ("branch", len(w) + 40, good, good), "branch-unknown-var")
    yield (checks, w, syn, e, U,
           ("branch", j, ("infeasible", (99,)), good),
           "infeasible-bad-row-index")
    yield (checks, w, syn, e, U, ("bound", ()), "bound-leaf-uncrossed")

    # INVALID FACETS INSIDE A BOUND LEAF. Distinct from an invalid facet
    # in a flat certificate: the leaf path runs the exact scan, whose
    # refusals the tree checker re-emits as "TREE REFUSED: ...". Nothing
    # in the corpus reached it before -- the broken-tree mutator damages
    # bounds and branches, never facet VALIDITY -- so the exact leaf rule
    # was going uncompared while the gate printed green.
    for duals, label in (
            (((0, (i,), 1),), "leaf-facet-right-parity"),
            (((0, (j,), 1),), "leaf-facet-escapes-check"),
            (((0, (), 1), (0, (), 1)), "leaf-facet-listed-twice"),
            ((((0,), (), 1),), "leaf-redundant-rows-invalid"),
            ((((0, 99), (), 1),), "leaf-redundant-rows-out-of-range"),
            (((99, (), 1),), "leaf-facet-unknown-check"),
            (((0, (), -1),), "leaf-facet-negative-dual"),
            (((0, (), 0),), "leaf-facet-zero-dual-skipped")):
        yield (checks, w, syn, e, U, ("bound", duals), label)
        yield (checks, w, syn, e, U,
               ("branch", j, ("bound", duals), ("infeasible", (0, 1))),
               f"branched-{label}")

    # malformed shapes, each named separately by the walker
    for bad, label in ((("branch", i, good), "malformed-branch"),
                       (("bound",), "malformed-bound"),
                       (("infeasible",), "malformed-infeasible"),
                       (("nonsense", 1), "unknown-node-kind"),
                       ((), "empty-node")):
        yield (checks, w, syn, e, U, bad, label)


# DECLARED REASON CHANGES. Consolidating the duplicated exact facet rule
# into _exact_facet_scan meant one of the two wordings had to win, and
# the fuller one did -- so three tree-leaf refusals now read better and
# read IDENTICALLY to the flat exact checker's, which is the point: one
# rule, one sentence, everywhere.
#
# These are enumerated rather than waved through. The invariant this gate
# exists to protect is that no VERDICT ever moves; a reason string is
# allowed to improve only when it is named here, with its justification,
# and only when the accepted flag and every numeric field are unchanged.
# A reason change that is not on this list is still a hard failure.
_REASON_ALIASES = [
    (r"^OPTIMAL: feasible dual attains the primal weight, so weak duality "
     r"makes both optimal -- verified without invoking any solver$",
     r"^OPTIMAL: .* so weak duality makes both optimal -- proven in exact "
     r"dyadic-rational arithmetic, with no tolerance in the acceptance "
     r"decision, and without invoking any solver$",
     "external audit 2026-08-04 finding 4: acceptance moved from a 1e-6 "
     "tolerance to an exact dyadic-lattice proof. On these cases the "
     "VERDICT and both numeric fields are unchanged -- the dual attained "
     "the primal exactly all along -- and only the sentence describing "
     "WHY changed, from 'verified' to 'proven'. The cases whose verdict "
     "genuinely moved are the suboptimal-by-tolerance and NaN ones, which "
     "this alias cannot excuse: it requires accepted and every number to "
     "match, so a verdict change still fails here."),
    (r"^decoder claimed weight [0-9.e-]+ but its own edges sum to "
     r"[0-9.e-]+$",
     r"^decoder claimed weight [0-9.e-]+ but its own edges sum to "
     r"[0-9.e-]+$",
     "2026-08-10: the recomputed sum in this message moved by at most one "
     "ulp because `matching_cert` now sums the correction's edge weights "
     "in exact rational arithmetic and rounds ONCE, instead of a running "
     "float sum whose value depended on edge ORDER. On the surface-d3-5 "
     "conformance vector the running sum was the value that was NOT "
     "correctly rounded -- the published receipt held the right one -- so "
     "the canonical sum is the fix, and the message repr shifts on the "
     "~0.5% of cases where association error reached an ulp. The VERDICT "
     "is decided by |claimed - primal| > 1e-6, nine orders above an ulp, "
     "and this alias still requires the accepted flag and both numeric "
     "fields (compared at 9 dp) to be identical, so a real verdict or "
     "value change cannot hide behind it."),
    (r"^TREE REFUSED: facet \(.*\) has RIGHT parity$",
     r"^TREE REFUSED: facet \(.*\) has the RIGHT parity -- "
     r"not a face of the relaxation$",
     "the tree leaf now states WHY the parity is wrong, in the same "
     "words the flat checker already used"),
    (r"^TREE REFUSED: duplicate facet$",
     r"^TREE REFUSED: facet \(.*\) listed twice$",
     "names WHICH facet was duplicated instead of only that one was"),
    (r"^TREE REFUSED: redundant check \(.*\) invalid$",
     r"^TREE REFUSED: redundant check \(.*\) is not >=2 valid rows$",
     "states the rule that was violated rather than the word 'invalid'"),
]


def _agree(a, b):
    """Do the two checkers agree? Returns (agree, aliased).

    Agreement is exact by default. A declared alias may excuse the REASON
    only -- never the accepted flag, never a numeric field, never an
    exception where the other returned a verdict.
    """
    if a == b:
        return True, False
    if len(a) != len(b) or a[0] != b[0]:
        return False, False
    if len(a) < 2:
        return False, False
    if a[2:] != b[2:]:              # primal weight / dual bound moved
        return False, False
    for old, new, _why in _REASON_ALIASES:
        if re.match(old, str(a[1])) and re.match(new, str(b[1])):
            return True, True
    return False, False


BOUNDARY = -1


def _matching_valid_case(rng: random.Random, cert_cls):
    """A MWPM certificate that genuinely ACCEPTS, built from LP theory.

    The dark-leg guard caught this: 1,000 random matching certificates
    produced ZERO acceptances, so the entire accept path -- complementary
    slackness, the zero-gap branch, the final verdict -- was uncompared
    while the leg looked covered. Random data reaches rejections only.

    Construction, in T-ODD CUTS rather than node potentials. Take
    disjoint matched edges (u, v, W) and set z_{u} = W on the SINGLETON
    set {u}. That set is T-odd (|S n T| = 1) and delta({u}) contains the
    matched edge, so the edge is exactly TIGHT and contributes W to both
    the primal and the dual. The odd-degree set is precisely the matched
    endpoints, so it equals the syndrome; the dual objective sums z and
    lands exactly on the primal, giving gap 0 for free. Extra unmatched
    edges are weighted above z_{a} + z_{b} -- a singleton {u} separates
    a from b iff exactly one of them is u -- so feasibility still holds.

    THIS USED TO BUILD y. It was ported because C-3's fix refuses a
    non-empty `node_potentials`, and a corpus built on potentials would
    have gone all-refusals under it: the accept path this function exists
    to reach would have gone dark while the gate still reported
    agreement. The construction is otherwise the same argument, written
    in the dual the checker is actually supposed to verify.

    Boundary edges are included: y_u = W with the BOUNDARY endpoint
    contributing nothing is the same argument, and boundary handling is
    where MWPM implementations classically go wrong.
    """
    npairs = rng.randint(1, 3)
    edges, matched, fired = [], [], []
    cuts: dict = {}          # singleton T-odd cuts: {u} -> z
    nid = 0
    for _ in range(npairs):
        W = round(rng.uniform(0.5, 4.0), 3)
        if rng.random() < 0.35:                  # boundary edge
            u = nid
            nid += 1
            edges.append((u, BOUNDARY, W))
            matched.append((u, BOUNDARY))
            cuts[u] = W
            fired.append(u)
        else:
            u, v = nid, nid + 1
            nid += 2
            edges.append((u, v, W))
            matched.append((u, v))
            cuts[u] = W                          # {u} alone is T-odd
            fired.append(u)
            fired.append(v)
    # unmatched edges, weighted so dual feasibility still holds
    for _ in range(rng.randint(0, 3)):
        if nid < 2:
            break
        a, b = rng.sample(range(nid), 2)
        need = cuts.get(a, 0.0) + cuts.get(b, 0.0)
        edges.append((a, b, round(need + rng.uniform(0.1, 2.0), 3)))
    label = "valid"
    blossoms: dict = {frozenset({u}): z for u, z in cuts.items()}
    # BLOSSOM DUALS reached on a certificate that SURVIVES the earlier
    # checks. Attaching these to random graphs did not work: the blossom
    # loop sits after the odd-degree-set test, which random data almost
    # never passes, so `mwpm-accept-negative-blossom` was caught on one
    # seed and missed on the next. Hung off the valid construction they
    # are reached every time.
    # 0.35 -> 0.6. At the old rate `mwpm-accept-negative-blossom` landed
    # ~7 divergences in 300 cases and FLICKERED: it passed standalone and
    # failed inside gate_all on the same code, one seed apart. Tuning the
    # seed until it passes would be dishonest; raising the rate until the
    # margin is real is the fix. A thin margin and a corpus hole look
    # identical from the outside, which is the whole reason this pass
    # exists.
    if nid >= 3 and rng.random() < 0.6:
        S = frozenset(rng.sample(range(nid), 3))
        kind = rng.choice(("negative", "even_set", "huge", "small"))
        if kind == "negative":
            blossoms[S] = -round(rng.uniform(0.1, 1.0), 3)
            label = "blossom-negative"
        elif kind == "even_set" and nid >= 4:
            blossoms[frozenset(rng.sample(range(nid), 2))] = 0.4
            label = "blossom-even-set"
        elif kind == "huge":
            blossoms[S] = 1e6
            label = "blossom-huge"
        else:
            blossoms[S] = round(rng.uniform(0.01, 0.2), 3)
            label = "blossom-small"

    # A FEASIBLE BUT SUBOPTIMAL dual: lowering a cut only relaxes every
    # load(e) <= w_e constraint, so the dual stays feasible and the
    # syndrome is untouched, but the objective no longer reaches the
    # primal. That is the one shape that reaches the "NOT PROVEN
    # OPTIMAL: gap" branch -- valid certificates close the gap exactly
    # and malformed ones are refused earlier, so without this the gap
    # test was invisible to the corpus.
    if rng.random() < 0.3 and cuts:
        k = rng.choice(sorted(cuts))
        if cuts[k] > 0:
            blossoms[frozenset({k})] = round(cuts[k] * 0.5, 3)
            label = "slack-dual"

    # A WRONG claimed weight on a certificate that otherwise PASSES. The
    # random generator had this flaw too, but its certificates die at the
    # odd-degree test long before the claimed-weight comparison, so
    # `mwpm-ignore-claimed-weight` flickered between seeds on a single
    # divergence. Hung off the valid construction it is reached every run.
    true_w = round(sum(w for (_u, _v, w) in edges[:len(matched)]), 6)
    roll = rng.random()
    if roll < 0.2:
        claimed = round(true_w + rng.uniform(1.0, 5.0), 6)
        label = "wrong-claimed-weight"
    elif roll < 0.6:
        claimed = true_w
    else:
        claimed = None
    cert = cert_cls(matched_edges=tuple(matched), node_potentials={},
                    blossom_duals=blossoms, claimed_weight=claimed,
                    metric_closure_certified=rng.random() < 0.5)
    return edges, sorted(fired), cert, label


def _matching_cases(rng: random.Random, cert_cls):
    """Decoding graphs, syndromes and MWPM certificates, shape-varied.

    matching_cert.check is the OTHER headline checker, and refactoring it
    needs the same arbiter the qLDPC checker got. Its rejection surface is
    wide -- negative edge weight, a correction edge absent from the graph,
    odd syndrome, a blossom dual on an even set, a dual exceeding an
    edge's slack, a claimed weight that does not match the matching -- and
    none of it is reached by a corpus of only well-formed inputs.

    BOUNDARY (-1) edges are included deliberately: half of QEC matching
    graphs are boundary-heavy, and an off-by-one in boundary handling is
    the classic MWPM bug.
    """
    # THE ABSORB-ORDER CASE. Random duals never reach the absorb step with
    # a shortfall it can actually close, so `repair-absorb-on-any-slack`
    # was invisible to this corpus -- the gate reported it as a hole and it
    # WAS one. This shape makes it observable: the first paying variable
    # (y_0) is exactly TIGHT on its boundary edge, so a correct absorb must
    # skip it and take the shortfall out of y_1, which has precisely enough
    # slack. A mutant that absorbs on ANY non-negative slack takes y_0
    # instead, overloads (0,-1), and gets caught by step 3 -- accept
    # becomes refuse, which is the divergence.
    #
    # It also mirrors the real defect: `surface-d3-5` shipped a dual 2.22e-16
    # short of attaining, which is exactly this, at ulp scale.
    if rng.random() < 0.08:
        eps = 2.0 ** -30
        return ([(0, BOUNDARY, 1.0), (1, BOUNDARY, 1.0), (0, 1, 4.0)],
                [0, 1],
                cert_cls(matched_edges=((0, BOUNDARY), (1, BOUNDARY)),
                         node_potentials={},
                         blossom_duals={frozenset({0}): 1.0,
                                        frozenset({1}): 1.0 - eps},
                         claimed_weight=2.0,
                         metric_closure_certified=True),
                "absorb_order")
    style = rng.choice(("chain", "boundary_heavy", "dense", "disjoint"))
    n = rng.randint(2, 9)
    edges = []
    if style == "chain":
        for u in range(n - 1):
            edges.append((u, u + 1, round(rng.uniform(0.5, 3.0), 3)))
        edges.append((0, BOUNDARY, round(rng.uniform(0.5, 3.0), 3)))
        edges.append((n - 1, BOUNDARY, round(rng.uniform(0.5, 3.0), 3)))
    elif style == "boundary_heavy":
        for u in range(n):
            edges.append((u, BOUNDARY, round(rng.uniform(0.2, 2.0), 3)))
        for _ in range(rng.randint(0, n)):
            u, v = rng.sample(range(n), 2) if n >= 2 else (0, 0)
            edges.append((u, v, round(rng.uniform(0.2, 4.0), 3)))
    elif style == "disjoint":
        for u in range(0, n - 1, 2):
            edges.append((u, u + 1, round(rng.uniform(0.5, 2.0), 3)))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.6:
                    edges.append((u, v, round(rng.uniform(0.2, 4.0), 3)))
        if not edges and n >= 2:
            edges.append((0, 1, 1.0))
    if rng.random() < 0.10 and edges:            # must be REFUSED
        k = rng.randrange(len(edges))
        u, v, _w = edges[k]
        edges[k] = (u, v, -abs(round(rng.uniform(0.1, 2.0), 3)))

    fired = [u for u in range(n) if rng.random() < 0.5]
    if rng.random() < 0.25 and fired:            # deliberately odd/even flip
        fired = fired[:-1] if len(fired) % 2 == 0 else fired

    # the matching: sometimes real edges, sometimes an edge not in the
    # graph at all, sometimes empty
    pool = [(u, v) for (u, v, _w) in edges]
    take = rng.randint(0, min(3, len(pool)))
    matched = tuple(rng.sample(pool, take)) if pool else ()
    flaw = rng.choice(("none", "none", "ghost_edge", "bad_cut",
                       "even_blossom", "negative_blossom", "huge_blossom",
                       "wrong_claimed_weight"))
    if flaw == "ghost_edge":
        matched = matched + ((n + 5, n + 6),)

    # RANDOM SINGLETON CUTS, not potentials. `bad_potential` became
    # `bad_cut` for the same reason the valid case was ported: under
    # C-3's refusal a potential-based corpus refuses uniformly and stops
    # discriminating. An over-large cut breaks edge feasibility, which is
    # the same defect this flaw always aimed at.
    blossoms = {frozenset({u}): round(rng.uniform(0.0, 2.0), 3)
                for u in range(n) if rng.random() < 0.5}
    if flaw == "bad_cut" and n >= 1:
        blossoms[frozenset({rng.randrange(n)})] = 1e6
    if flaw == "even_blossom" and n >= 2:
        blossoms[frozenset(rng.sample(range(n), 2))] = 0.5
    elif flaw == "negative_blossom" and n >= 3:
        blossoms[frozenset(rng.sample(range(n), 3))] = -0.5
    elif flaw == "huge_blossom" and n >= 3:
        blossoms[frozenset(rng.sample(range(n), 3))] = 1e6
    elif rng.random() < 0.3 and n >= 3:
        blossoms[frozenset(rng.sample(range(n), 3))] = round(
            rng.uniform(0.0, 1.0), 3)

    claimed = None
    if flaw == "wrong_claimed_weight":
        claimed = round(rng.uniform(50.0, 100.0), 3)
    elif rng.random() < 0.4:
        wmap = {(min(u, v) if v != BOUNDARY else u,
                 max(u, v) if v != BOUNDARY else BOUNDARY): w
                for (u, v, w) in edges}
        claimed = round(sum(wmap.get((min(a, b) if b != BOUNDARY else a,
                                      max(a, b) if b != BOUNDARY else BOUNDARY),
                                     0.0) for (a, b) in matched), 6)

    cert = cert_cls(matched_edges=matched, node_potentials={},
                    blossom_duals=blossoms, claimed_weight=claimed,
                    metric_closure_certified=rng.random() < 0.5)
    return edges, fired, cert, flaw


def _norm(res):
    """Compare the WHOLE verdict, not just the boolean. A refactor that
    accepts the same things for different reasons has changed what the
    certificate MEANS, and that must show up as a failure."""
    if res is None:
        return ("<none>",)
    def q(v):
        if v is None:
            return None
        try:
            return round(float(v), 9)
        except (TypeError, ValueError):
            return str(v)
    # The qLDPC result names its bound `dual_bound`; the MWPM result names
    # the same quantity `dual_objective`. Reading only the first made
    # _norm raise AttributeError on EVERY matching call -- identically on
    # both sides, so nothing looked wrong: the leg silently compared
    # exception names instead of verdicts. Caught only because the
    # dark-leg guard demands the matching checker ACCEPT something.
    bound = getattr(res, "dual_bound", None)
    if bound is None:
        bound = getattr(res, "dual_objective", None)
    return (bool(res.accepted), str(res.reason), q(res.primal_weight),
            q(bound))


def _call(mod, fn, **kw):
    """Run one checker call, turning any escape into a comparable value.
    A checker that RAISES where the other returns False is a behaviour
    change -- one of them fails open under a malformed input."""
    try:
        return _norm(getattr(mod, fn)(**kw))
    except RecursionError:
        return ("<recursion>",)
    except Exception as exc:                     # noqa: BLE001 -- deliberate
        return ("<raised>", type(exc).__name__)


# ------------------------------------------------------------------ run

def run_corpus(frozen, live, cases: int, seed: int, verbose: bool = False,
               live_match=None):
    """Both checkers, same inputs, full-verdict comparison."""
    from oneq.qldpc_cert import certified_bnb, feldman_dual
    import importlib as _il
    fz_match = _il.import_module("tools.frozen.matching_cert")
    lv_match = live_match or _il.import_module("oneq.matching_cert")

    rng = random.Random(seed)
    diffs, stats = [], {"exercised": 0, "accepted": 0, "rejected": 0,
                        "raised": 0, "bnb": 0, "abuse": 0, "exact": 0,
                        "bnb_accepted": 0, "forced": 0, "aliased": 0, "matching": 0,
                        "matching_accepted": 0}

    for c in range(cases):
        exact = (c % 3 == 0)
        checks, n = _rand_instance(rng)
        checks = _maybe_duplicate_indices(rng, checks)
        weights = _rand_weights(rng, n, exact)
        syn = _rand_syndrome(rng, len(checks))

        # a genuine certificate when the instance admits one; the
        # producer is allowed to fail -- a checker must handle the
        # certificate it is HANDED, including a poor one.
        cert = None
        try:
            wf = {i: float(v) for i, v in weights.items()}
            if all(v >= 0 for v in wf.values()) and len(syn) == len(checks):
                got, _b, primal = feldman_dual(checks, wf, syn,
                                               return_primal=True)
                if got is not None:
                    cert = live.QldpcCertificate(
                        error_support=tuple(primal or ()),
                        facet_duals=got.facet_duals,
                        box_duals=got.box_duals)
        except Exception:                        # noqa: BLE001
            cert = None
        if cert is None:
            cert = live.QldpcCertificate(
                error_support=tuple(sorted(rng.sample(
                    range(n), rng.randint(0, min(n, 4))))),
                facet_duals=(), box_duals=())

        # ---- float checker, genuine and perturbed, plus a SYNTHETIC
        #      certificate whose facet duals are guaranteed non-empty so
        #      the facet-validation section is always compared.
        for variant in range(3):
            syn_v = syn
            if variant == 2:
                base, syn_v = _synthetic_cert(rng, live.QldpcCertificate,
                                              checks, syn, n)
            else:
                base = cert
            pc, axis = _perturb_cert(rng, live.QldpcCertificate, base,
                                     checks, n)
            if variant == 2:
                axis = f"synthetic/{axis}"
            fc = frozen.QldpcCertificate(error_support=pc.error_support,
                                         facet_duals=pc.facet_duals,
                                         box_duals=pc.box_duals)
            wf = {i: float(v) for i, v in weights.items()}
            a = _call(frozen, "check_qldpc", checks=checks, weights=wf,
                      syndrome=syn_v, cert=fc)
            b = _call(live, "check_qldpc", checks=checks, weights=wf,
                      syndrome=syn_v, cert=pc)
            stats["exercised"] += 1
            if a[0] is True:
                stats["accepted"] += 1
            elif a[0] == "<raised>":
                stats["raised"] += 1
            else:
                stats["rejected"] += 1
            ok, aliased = _agree(a, b)
            stats['aliased'] += int(aliased)
            if not ok:
                diffs.append({"case": c, "fn": "check_qldpc", "axis": axis,
                              "frozen": list(map(str, a)),
                              "live": list(map(str, b)),
                              "checks": [list(x) for x in checks],
                              "syndrome": list(map(str, syn_v))})

        # ---- exact checker (rationals; floats must be REFUSED by type)
        if exact:
            pc, axis = _perturb_cert(rng, live.QldpcCertificate, cert,
                                     checks, n)
            fc = frozen.QldpcCertificate(error_support=pc.error_support,
                                         facet_duals=pc.facet_duals,
                                         box_duals=pc.box_duals)
            a = _call(frozen, "check_qldpc_exact", checks=checks,
                      weights=weights, syndrome=syn, cert=fc)
            b = _call(live, "check_qldpc_exact", checks=checks,
                      weights=weights, syndrome=syn, cert=pc)
            stats["exercised"] += 1
            stats["exact"] += 1
            ok, aliased = _agree(a, b)
            stats['aliased'] += int(aliased)
            if not ok:
                diffs.append({"case": c, "fn": "check_qldpc_exact",
                              "axis": axis, "frozen": list(map(str, a)),
                              "live": list(map(str, b)),
                              "checks": [list(x) for x in checks],
                              "syndrome": list(map(str, syn))})

        # ---- tier BD: the tree walker, valid trees and broken ones.
        #
        # THIS LEG WAS DARK. The first full run reported `tier-BD 0`: the
        # producer takes a REQUIRED positional U (the candidate weight)
        # and returns (tree, nodes), and calling it without U raised on
        # every case, which the surrounding try/except swallowed into
        # tree=None. So check_qldpc_bnb -- one of the very functions
        # being refactored -- was compared exactly zero times while the
        # gate printed green. A dark leg is not coverage; it is the
        # skip-is-never-a-pass failure wearing a different hat.
        if c % 2 == 0 and 1 <= len(checks) <= 6:
            # integer weights so the lattice step D is 1, and a syndrome
            # DERIVED from a planted error so a feasible U exists.
            wi = {i: rng.randint(1, 3) for i in range(n)}
            e_sup = tuple(sorted(rng.sample(range(n),
                                            rng.randint(1, min(n, 3)))))
            es = set(e_sup)
            syn_b = [len(es & set(cc)) % 2 for cc in checks]
            U = sum(wi[i] for i in e_sup)
            try:
                tree, _nodes = certified_bnb(checks, wi, syn_b, U,
                                             max_nodes=300, max_depth=12)
            except Exception:                    # noqa: BLE001
                tree = None
            if tree is not None:
                for broken in (False, True):
                    t = _break_tree(rng, tree) if broken else tree
                    ka = dict(checks=checks, weights=wi, syndrome=syn_b,
                              error_support=e_sup, tree=t)
                    a = _call(frozen, "check_qldpc_bnb", **ka)
                    b = _call(live, "check_qldpc_bnb", **ka)
                    stats["exercised"] += 1
                    stats["bnb"] += 1
                    if a[0] is True:
                        stats["bnb_accepted"] += 1
                    ok, aliased = _agree(a, b)
                    stats['aliased'] += int(aliased)
                    if not ok:
                        diffs.append({"case": c, "fn": "check_qldpc_bnb",
                                      "axis": f"broken={broken}",
                                      "frozen": list(map(str, a)),
                                      "live": list(map(str, b))})
                # bound-only mode: leaves must prove min > T, the shape
                # frame determinacy and the distance certificate rely on.
                ka = dict(checks=checks, weights=wi, syndrome=syn_b,
                          error_support=e_sup, tree=tree,
                          bound_only_threshold=U - 1)
                a = _call(frozen, "check_qldpc_bnb", **ka)
                b = _call(live, "check_qldpc_bnb", **ka)
                stats["exercised"] += 1
                stats["bnb"] += 1
                ok, aliased = _agree(a, b)
                stats['aliased'] += int(aliased)
                if not ok:
                    diffs.append({"case": c, "fn": "check_qldpc_bnb",
                                  "axis": "bound_only",
                                  "frozen": list(map(str, a)),
                                  "live": list(map(str, b))})

        # ---- FORCED tree shapes: branch nodes, infeasible leaves,
        #      GF(2) row combinations, non-empty S1. The producer almost
        #      never emits these on easy instances, so they are built.
        if c % 3 == 0:
            for (fchecks, fw, fsyn, fe, fU, ftree,
                 flabel) in _forced_tree_cases(rng):
                ka = dict(checks=fchecks, weights=fw, syndrome=fsyn,
                          error_support=fe, tree=ftree)
                a = _call(frozen, "check_qldpc_bnb", **ka)
                b = _call(live, "check_qldpc_bnb", **ka)
                stats["exercised"] += 1
                stats["bnb"] += 1
                stats["forced"] += 1
                if a[0] is True:
                    stats["bnb_accepted"] += 1
                ok, aliased = _agree(a, b)
                stats['aliased'] += int(aliased)
                if not ok:
                    diffs.append({"case": c, "fn": "check_qldpc_bnb",
                                  "axis": f"forced/{flabel}",
                                  "frozen": list(map(str, a)),
                                  "live": list(map(str, b))})

        # ---- the MWPM checker. Same arbiter, its own corpus: the
        #      matching certificate has a completely different rejection
        #      surface (boundary edges, odd syndrome, blossom duals on
        #      even sets, slack violations) and shares no code with the
        #      qLDPC path, so qLDPC coverage says nothing about it.
        for _rep in range(2):
            gen = (_matching_valid_case if rng.random() < 0.45
                   else _matching_cases)
            edges, fired, mcert, mflaw = gen(
                rng, lv_match.MatchingCertificate)
            fcert = fz_match.MatchingCertificate(
                matched_edges=mcert.matched_edges,
                node_potentials=mcert.node_potentials,
                blossom_duals=mcert.blossom_duals,
                claimed_weight=mcert.claimed_weight,
                metric_closure_certified=mcert.metric_closure_certified)
            a = _call(fz_match, "check", edges=edges, syndrome=fired,
                      cert=fcert)
            b = _call(lv_match, "check", edges=edges, syndrome=fired,
                      cert=mcert)
            stats["exercised"] += 1
            stats["matching"] += 1
            if a[0] is True:
                stats["matching_accepted"] += 1
            ok, aliased = _agree(a, b)
            stats['aliased'] += int(aliased)
            if not ok:
                diffs.append({"case": c, "fn": "matching_cert.check",
                              "axis": f"matching/{mflaw}",
                              "frozen": list(map(str, a)),
                              "live": list(map(str, b))})

        # ---- type abuse: the REJECT path, compared by reason
        kind = _TYPE_ABUSE[c % len(_TYPE_ABUSE)]
        ck, wt, sy, sp = _abuse(rng, kind, checks,
                                {i: float(v) for i, v in weights.items()},
                                syn, cert.error_support)
        fc = frozen.QldpcCertificate(error_support=sp,
                                     facet_duals=cert.facet_duals,
                                     box_duals=cert.box_duals)
        lc = live.QldpcCertificate(error_support=sp,
                                   facet_duals=cert.facet_duals,
                                   box_duals=cert.box_duals)
        a = _call(frozen, "check_qldpc", checks=ck, weights=wt,
                  syndrome=sy, cert=fc)
        b = _call(live, "check_qldpc", checks=ck, weights=wt,
                  syndrome=sy, cert=lc)
        stats["exercised"] += 1
        stats["abuse"] += 1
        ok, aliased = _agree(a, b)
        stats['aliased'] += int(aliased)
        if not ok:
            diffs.append({"case": c, "fn": "check_qldpc/abuse",
                          "axis": kind, "frozen": list(map(str, a)),
                          "live": list(map(str, b))})

        if verbose and c and c % 200 == 0:
            print(f"    {c} cases, {stats['exercised']} calls, "
                  f"{len(diffs)} divergences")

    return diffs, stats


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------- sensitivity mutations

# Each is (label, find, replace). They are BEHAVIOURAL: every one changes
# what some input decides. If the corpus cannot see one, the corpus is
# incomplete and this gate must say so rather than pass.
# A MARGIN, NOT A COIN TOSS. A mutation caught by ONE divergence is not
# meaningfully caught. While this gate was being built the same code
# passed standalone and failed inside gate_all one seed apart, three
# separate times, and each time the honest diagnosis was "thin margin"
# rather than "corpus hole" -- but those are indistinguishable from the
# outside. A floor turns a flaky red into an actionable one: it fires
# while the coverage is merely weak, instead of when it has already
# vanished on some future seed.
MIN_MARGIN = 5

_MUTATIONS = [
    ("accept-nonzero-gap", "if gap > 1e-6:", "if False:"),
    ("widen-tolerance", "tol: float = 1e-9", "tol: float = 1e9"),
    ("drop-parity-check", "if len(Fs) % 2 == par:", "if False:"),
    ("drop-subset-check", "if not Fs <= sup:", "if False:"),
    ("drop-negative-dual", "if y < -tol:", "if False:"),
    ("drop-duplicate-facet", "if key in seen:", "if False:"),
    # ANCHORED ON THE FLOAT PATH, which every case exercises three times.
    # The refactor turned check_qldpc's version into `return None,
    # QldpcCheckResult(...)`, so this anchor stopped matching it and fell
    # through to the BD checker's copy instead -- a path the corpus
    # reaches far less often, and the mutation went MISSED on one seed in
    # three. The anchor drifted onto thinner coverage without ever
    # failing to match, which is the quietest way a mutant stops testing.
    ("soften-dup-reason",
     "        return None, QldpcCheckResult(False, dup_err)",
     '        return None, QldpcCheckResult(False, "rejected")'),
    ("skip-unknown-var", 'f"error uses unknown variable {i}"', '"ok"'),
    ("drop-redundant-row-validation",
     "if len(rows) < 2 or any(not (0 <= r < m) for r in rows):",
     "if False:"),

    # --- tier BD, the tree walker. Sensitivity proven for the FLOAT
    # checker says nothing about the tree checker, and check_qldpc_bnb is
    # one of the functions being refactored.
    ("bd-accept-uncrossed-leaf", "if L + W1 > U - step:", "if True:"),
    ("bd-allow-refixed-branch-var",
     "if i not in w or i in S0 or i in S1:", "if False:"),
    ("bd-accept-nonempty-infeasible-xor", "if acc:", "if False:"),
    ("bd-accept-even-parity-infeasible", "if par == 0:", "if False:"),
    ("bd-ignore-fixed-weight", "W1 = sum((w[i] for i in S1), Fraction(0))",
     "W1 = Fraction(0)"),

    # --- the EXACT checker. Its whole point is refusing floats by TYPE;
    # a refactor that lets one through reintroduces the tolerance it was
    # built to eliminate.
    ("exact-admit-floats",
     "if isinstance(v, bool) or isinstance(v, float):", "if False:"),
]

# PROVEN-EQUIVALENT MUTANTS. Not every surviving mutant is a corpus hole;
# some are unreachable by construction, and calling those "holes" would
# push me to weaken the corpus definition until it went green. Each entry
# here carries the ARGUMENT for unreachability, and the gate reports them
# separately rather than counting them as either caught or missed.
#
# Deleting the branch instead would be the cheaper green and the wrong
# move: a defensive assertion that never fires is exactly what you want
# guarding an invariant, and removing a check to buy a clean number is a
# regression, not a simplification.
_EQUIVALENT = [
    # `_repair_and_verify` step 3 re-verifies feasibility, non-negativity and
    # attainment AFTER steps 1-2 have constructed exactly those properties.
    # Under SINGLE-fault mutation that makes all three unreachable: shedding
    # clears every violated edge, shedding clamps odd-set duals at zero, and
    # absorbing only fires when the slack covers the shortfall exactly. So a
    # mutation of step 3 alone cannot be observed, and reporting it as a
    # corpus hole would be false -- there is no input that distinguishes it.
    #
    # STATED PLAINLY BECAUSE IT IS UNCOMFORTABLE: step 3 therefore carries no
    # TESTED value under this gate. It is kept because it is the entire
    # guarantee if steps 1-2 are ever wrong, and steps 1-2 are a search over
    # floating-point damage that I have already had to correct twice. A
    # verifier that re-derives its own conclusion is worth more than a clean
    # mutation score, and deleting it to buy one would be the regression.
    # (Not listed as entries: this list is scanned against the qLDPC checker
    # source, so a matching_cert.py pattern reads here as PATTERN ABSENT --
    # which is a false failure, not an equivalence. The argument above stands
    # on its own and the three step-3 checks stay in the code.)
    ("hide-impossible", "if gap < -1e-6:",
     "Unreachable given the two checks that precede it. Step 1 has "
     "already verified that the claimed error REPRODUCES the syndrome, "
     "so it is integer-feasible with weight `primal`; step 3 has "
     "already verified dual feasibility (load_i - mu_i <= w_i for every "
     "touched variable, untouched ones trivially). Weak duality then "
     "gives obj <= LP-opt <= primal, so gap = primal - obj >= 0 for "
     "every input that reaches line 4. The branch is an internal "
     "consistency assertion against a broken relaxation or a broken "
     "checker -- by design it cannot be triggered from outside."),
]


# The MWPM checker's own mutations. Sensitivity proven on qldpc_check
# says nothing about matching_cert -- they share no code, and
# matching_cert.check is the other function being refactored.
_MATCH_MUTATIONS = [
    ("mwpm-accept-negative-edge", "if w < -tol:", "if False:"),
    ("mwpm-ignore-syndrome-mismatch", "if odd != fired:", "if False:"),
    ("mwpm-accept-negative-blossom", "if zz < -tol:", "if False:"),
    ("mwpm-accept-dual-infeasible", "if lhs > w + tol:", "if False:"),
    # RE-ANCHORED AGAIN, and this time the drift was silent for longer. The
    # audit fix appended `or not proven` to this condition, so the pattern
    # `if gap > 1e-6:` stopped matching and the mutation went invisible --
    # reported as PATTERN ABSENT only once gate_all ran the sensitivity pass
    # in full. A mutation whose anchor has drifted is not a passing test, it
    # is an absent one, and it looks identical from the summary line.
    ("mwpm-accept-nonzero-gap", "if gap > 1e-6 or not proven:", "if False:"),
    # THE REPAIR'S OWN VERIFICATION. `_repair_and_verify` sheds infeasibility
    # and absorbs the shortfall before deciding, so the three checks below are
    # now the entire acceptance decision for the matching lane. Without these
    # mutations a bug in the repair would be invisible to this corpus -- and
    # the repair is the most dangerous code in the tree precisely because a
    # too-generous version stays green on every existing test.
    # The tolerance-in-disguise knob: absorbing on ANY slack rather than
    # ENOUGH slack turns the exact rule back into an epsilon one. Not
    # equivalent -- it diverges whenever the first paying variable is tight
    # and a later one carries the slack, which `_absorb_order_case` builds.
    # RE-ANCHORED, and I caused the drift MYSELF one commit after adding the
    # anchor-drift check that catches it. Generalising absorb to spread the
    # shortfall replaced the `>= short` guard with room/take, so the old
    # pattern went absent. That is the third instance of this class in one
    # day and the lesson is not "be careful": a mutation's anchor must be
    # re-verified by the gate on every run, which is why --prove-sensitive
    # now belongs in gate_all rather than in someone's memory.
    ("mwpm-repair-take-more-than-room",
     "take = room if room < short else short",
     "take = short"),
    # RE-ANCHORED. The matching_cert refactor reformatted this condition
    # across two lines and the anchor drifted -- silently, because the
    # sensitivity pass was not yet part of gate_all. Its first automated
    # run found this, which is precisely the argument for automating it:
    # a mutation nobody re-runs is a mutation that quietly stops testing.
    ("mwpm-ignore-claimed-weight",
     "if (cert.claimed_weight is not None",
     "if (False and cert.claimed_weight is not None"),
]


def _prove_sensitive(frozen, cases: int, seed: int) -> int:
    """Mutate a COPY of the live checker; require this gate to catch it."""
    live_src = (ROOT / "src" / "oneq" / "qldpc_check.py").read_text(
        encoding="utf-8")
    caught, missed = [], []
    with tempfile.TemporaryDirectory() as td:
        for label, find, repl in _MUTATIONS:
            if find not in live_src:
                missed.append(f"{label} (PATTERN ABSENT: {find!r})")
                continue
            mutated = live_src.replace(find, repl, 1)
            assert mutated != live_src
            p = pathlib.Path(td) / f"mut_{abs(hash(label))}.py"
            p.write_text(mutated, encoding="utf-8")
            mod = _load(p, f"oneq_mutant_{abs(hash(label))}")
            diffs, _st = run_corpus(frozen, mod, cases, seed)
            if diffs:
                caught.append((label, len(diffs)))
                print(f"  CAUGHT  {label:22s} {len(diffs)} divergences")
            else:
                missed.append(label)
                print(f"  MISSED  {label:22s} <-- corpus hole")
    # the MWPM checker, mutated in its own module
    match_src = (ROOT / "src" / "oneq" / "matching_cert.py").read_text(
        encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        for label, find, repl in _MATCH_MUTATIONS:
            if find not in match_src:
                missed.append(f"{label} (PATTERN ABSENT: {find!r})")
                continue
            p = pathlib.Path(td) / f"mm_{abs(hash(label))}.py"
            p.write_text(match_src.replace(find, repl, 1), encoding="utf-8")
            mod = _load(p, f"oneq_mmutant_{abs(hash(label))}")
            live_ok = importlib.import_module("oneq.qldpc_check")
            diffs, _st = run_corpus(frozen, live_ok, cases, seed,
                                    live_match=mod)
            if diffs:
                caught.append((label, len(diffs)))
                print(f"  CAUGHT  {label:32s} {len(diffs)} divergences")
            else:
                missed.append(label)
                print(f"  MISSED  {label:32s} <-- corpus hole")

    # The proven-equivalent mutants run too, and the requirement INVERTS:
    # each must SURVIVE. If the corpus catches one, my unreachability
    # argument is wrong -- the branch is live, the entry is a lie, and
    # that is a finding about the checker, not a pass.
    broken_proof = []
    with tempfile.TemporaryDirectory() as td:
        for label, find, why in _EQUIVALENT:
            if find not in live_src:
                broken_proof.append(f"{label}: pattern absent ({find!r})")
                continue
            p = pathlib.Path(td) / f"eq_{abs(hash(label))}.py"
            p.write_text(live_src.replace(find, "if False:", 1),
                         encoding="utf-8")
            mod = _load(p, f"oneq_equiv_{abs(hash(label))}")
            diffs, _st = run_corpus(frozen, mod, cases, seed)
            if diffs:
                broken_proof.append(
                    f"{label}: REACHED in {len(diffs)} case(s) -- the "
                    f"unreachability argument is WRONG")
                print(f"  PROOF BROKEN {label}: reached {len(diffs)}x")
            else:
                print(f"  equivalent (survives, as argued)  {label}")

    thin = [(lab, k) for (lab, k) in caught if k < MIN_MARGIN]
    if thin and not missed:
        print(f"\nMARGIN TOO THIN ({len(thin)}): caught, but barely -- "
              f"this is the state that flips to MISSED on another seed")
        for lab, k in thin:
            print(f"    {lab}: {k} divergence(s), floor {MIN_MARGIN}")
        print("Make the corpus reach the branch at a guaranteed rate; do "
              "not tune the seed until it passes.")
        return 1

    if missed:
        print(f"\nSENSITIVITY FAILED: {len(missed)} mutation(s) invisible "
              f"to this corpus:")
        for mlab in missed:
            print(f"    {mlab}")
        print("A mutation the corpus cannot see is a hole in the corpus, "
              "not a pass.")
        return 1
    if broken_proof:
        print("\nEQUIVALENCE ARGUMENT FALSIFIED:")
        for b in broken_proof:
            print(f"    {b}")
        return 1
    print(f"\nSENSITIVITY PROVEN: {len(caught)}/"
          f"{len(_MUTATIONS) + len(_MATCH_MUTATIONS)} "
          f"behavioural mutations caught, {len(_EQUIVALENT)} proven-"
          f"unreachable branch(es) survive as argued. This gate can fail.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--prove-sensitive", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    _verify_freeze()
    print("frozen snapshots verified against the pinned sha256 manifest")

    frozen = importlib.import_module("tools.frozen.qldpc_check")
    live = importlib.import_module("oneq.qldpc_check")

    if args.prove_sensitive:
        # Enough cases per mutation that CAUGHT is not luck. At cases//6
        # the marginal mutations landed 1-3 divergences and flickered
        # between seeds -- the gate was reporting noise, and a corpus
        # hole and a power shortage look identical from the outside.
        per = max(200, args.cases // 2)
        print(f"\nSENSITIVITY PASS "
              f"({len(_MUTATIONS) + len(_MATCH_MUTATIONS)} behavioural "
              f"mutations, {per} cases each)")
        return _prove_sensitive(frozen, per, args.seed)

    print(f"\nEQUIVALENCE: frozen vs live, {args.cases} shape-varied cases")
    diffs, stats = run_corpus(frozen, live, args.cases, args.seed,
                              verbose=True)
    print(f"  calls compared : {stats['exercised']}")
    print(f"  accepted / rejected / raised : "
          f"{stats['accepted']} / {stats['rejected']} / {stats['raised']}")
    print(f"  exact {stats['exact']}   tier-BD {stats['bnb']} "
          f"(accepting {stats['bnb_accepted']})   "
          f"malformed {stats['abuse']}")
    print(f"  matching-cert {stats['matching']} "
          f"(accepting {stats['matching_accepted']})")

    print(f"  reason improved via a DECLARED alias: {stats['aliased']} "
          f"(verdict and every number identical in all of them)")

    # The declared aliases must actually be USED. An alias nobody reaches
    # is a permission granted for a case the corpus cannot produce, which
    # would let a later real divergence hide behind it unnoticed.
    if stats["aliased"] == 0 and _REASON_ALIASES:
        print("DECLARED ALIASES UNUSED: the corpus never reaches the "
              "reason changes this gate was told to permit -- either the "
              "permission is stale and should be deleted, or the corpus "
              "no longer covers the leaf facet rule")
        return 1

    # A corpus that never accepts, or never rejects, proves nothing about
    # the branch it never took. Vacuity is a failure, not a pass.
    if stats["accepted"] == 0 or stats["rejected"] == 0:
        print("VACUOUS: the corpus did not exercise both verdicts")
        return 1
    # A LEG THAT NEVER RAN IS NOT A LEG THAT PASSED. The first run of
    # this gate printed `tier-BD 0` and still said EQUIVALENT, because a
    # swallowed TypeError in the producer call made every tree None.
    # Each leg now has to show up, and the tree walker has to have
    # ACCEPTED something -- a corpus of only-broken trees would exercise
    # the refusal path and silently skip the proof path.
    for leg, need in (("exact", 1), ("bnb", 1), ("abuse", 1),
                      ("bnb_accepted", 1), ("forced", 1),
                      ("matching", 1), ("matching_accepted", 1)):
        if stats[leg] < need:
            print(f"DARK LEG: '{leg}' was exercised {stats[leg]} times -- "
                  f"the equivalence claim does not cover it, so this is a "
                  f"FAILURE, not a pass")
            return 1

    out = {"schema": "oneq-checker-equivalence/1",
           "cases": args.cases, "seed": args.seed,
           "calls_compared": stats["exercised"],
           "accepted": stats["accepted"], "rejected": stats["rejected"],
           "raised": stats["raised"], "exact_calls": stats["exact"],
           "bnb_calls": stats["bnb"],
           "bnb_accepting": stats["bnb_accepted"], "malformed_calls": stats["abuse"],
           "matching_calls": stats["matching"],
           "matching_accepting": stats["matching_accepted"],
           "divergences": len(diffs),
           "declared_reason_aliases_used": stats["aliased"],
           "declared_reason_aliases": [{"frozen": a, "live": b, "why": c}
                                       for a, b, c in _REASON_ALIASES],
           "frozen_sha256": FROZEN_SHA256,
           "claim": ("the refactored trust modules return the identical "
                     "verdict, reason string, primal weight and dual "
                     "bound as the pre-refactor bytes, on every case")}
    if args.json:
        d = pathlib.Path(args.json)
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"evidence -> {d}")

    if diffs:
        print(f"\nDIVERGENCE: {len(diffs)} case(s) decide differently")
        print("  If this was NOT intentional, it is the defect this "
              "gate exists for. If it WAS, see "
              "docs/REFREEZE_PROTOCOL.md -- a verdict may move, but it "
              "moves as a reviewed act.")
        for d in diffs[:8]:
            print(f"  case {d['case']} {d['fn']} [{d['axis']}]")
            print(f"    frozen: {d['frozen']}")
            print(f"    live  : {d['live']}")
        return 1
    print("\nEQUIVALENT: every verdict, reason, weight and bound agrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
