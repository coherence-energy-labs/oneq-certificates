#!/usr/bin/env python3
"""Second branch-dual tree checker -- separately implemented from the schema.

Written from the WRITTEN SPECIFICATION of the branch-dual receipt (paper
SS2 tier BD + the serialized-tree schema of qldpc_heldout_addendum.json),
without importing or consulting the production checker (qldpc_check.py).
Shares no code, no sign-convention derivation, and no data structures
with it; arithmetic is fractions over Python ints only.

Serialized tree schema (JSON):
  ["branch", k, child_for_xk_eq_0, child_for_xk_eq_1]
  ["bound",  [[rows, F, num, den], ...]]     # facet duals on the REDUCED
                                             # instance at this node
  ["infeasible", rows]                       # single row index or list:
                                             # reduced supports XOR to
                                             # empty with odd parity

Instance document (JSON):
  {"checks": [[i,...],...], "weights": [int,...], "syndrome": [0/1,...],
   "candidate": [i,...], "tree": <tree>}

Acceptance := candidate feasible AND every root-to-leaf path closes:
each leaf either exhibits a parity-infeasible row combination of its
reduced instance, or carries facet duals whose safe-dual bound L
satisfies W_fixed1 + L > U - 1/D (D = lcm of weight denominators; here
weights are ints so D = 1). Exit 0 accept, 2 refuse, 1 malformed.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import gcd


def refuse(msg):
    print(f"REFUSE: {msg}")
    raise SystemExit(2)


def _exact_int(v, what: str) -> int:
    """An integer, or a refusal. NEVER a silent truncation.

    EXTERNAL AUDIT 2026-08-04, finding 6 (CONFIRMED and fixed here).
    This file parsed untrusted evidence with bare `int()`, which is a
    LOSSY COERCION dressed as a cast: `int(1.9)` is 1, so a weight vector
    of [1.9, 1.0] became [1, 1] and a genuinely suboptimal candidate
    certified against weights nobody supplied. Booleans coerce too, and
    numeric strings.

    The mathematics in this file was sound. The failure was entirely at
    the untrusted-input boundary -- which is the only boundary an
    independent checker has, and therefore the one place it cannot afford
    to be relaxed.
    """
    if isinstance(v, bool):
        refuse(f"{what} is a boolean, not an integer: {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            refuse(f"{what} is not finite: {v!r}")
        if v != int(v):
            refuse(f"{what} is not an integer: {v!r} -- refusing rather "
                   f"than truncating to {int(v)}")
        return int(v)
    refuse(f"{what} is not a number: {v!r}")


def _index(v, n: int, what: str) -> int:
    """A VALID index in [0, n). Python's negative indexing is a gift to
    a forger: index -1 silently means 'the last element', so an
    out-of-range value that should be refused instead selects real data.
    """
    i = _exact_int(v, what)
    if not (0 <= i < n):
        refuse(f"{what} out of range: {i} not in [0, {n})")
    return i


def main(path: str) -> int:
    doc = json.loads(open(path, encoding="utf-8").read())
    nvar = len(doc["weights"])
    checks = [tuple(_index(i, nvar, f"check {j} index") for i in c)
              for j, c in enumerate(doc["checks"])]
    for j, c in enumerate(checks):
        if len(set(c)) != len(c):
            refuse(f"duplicate index in check {j}")
    w = [_exact_int(v, f"weight[{i}]") for i, v in
         enumerate(doc["weights"])]
    if any(v < 0 for v in w):
        refuse("negative weight")
    syn = [_exact_int(b, f"syndrome[{i}]") & 1 for i, b in
           enumerate(doc["syndrome"])]
    if len(syn) != len(checks):
        refuse("syndrome length mismatch")
    cand = sorted(_index(i, nvar, "candidate index")
                  for i in doc["candidate"])
    cs = set(cand)
    if len(cs) != len(cand):
        refuse("duplicate candidate index")
    for j, c in enumerate(checks):
        if sum(1 for i in c if i in cs) % 2 != syn[j]:
            refuse(f"candidate violates check {j}")
    U = sum(w[i] for i in cand)
    step = Fraction(1, 1)               # integer weights

    def reduced(j, fix):                # (support tuple, parity) at node
        sup = tuple(i for i in checks[j] if i not in fix)
        par = (syn[j] ^ (sum(1 for i in checks[j]
                             if fix.get(i) == 1) & 1)) & 1
        return sup, par

    def leaf_bound(duals, fix):
        load: dict[int, Fraction] = {}
        by = Fraction(0)
        seen = set()
        for entry in duals:
            if not (isinstance(entry, list) and len(entry) == 4):
                refuse("malformed dual entry")
            rows, F, num, den = entry
            if not isinstance(num, int) or not isinstance(den, int) \
                    or den <= 0:
                refuse("non-integer rational")
            y = Fraction(num, den)
            if y < 0:
                refuse("negative multiplier")
            if y == 0:
                continue
            rlist = ([int(r) for r in rows] if isinstance(rows, list)
                     else [int(rows)])
            if len(set(rlist)) != len(rlist):
                refuse("duplicate row in combination")
            if any(not (0 <= r < len(checks)) for r in rlist):
                refuse("row out of range")
            sup: set = set()
            par = 0
            for r in rlist:
                s2, p2 = reduced(r, fix)
                sup ^= set(s2)
                par ^= p2
            Fs = set(int(i) for i in F)
            if len(Fs) != len(F):
                refuse("duplicate F index")
            if not Fs <= sup:
                refuse("F escapes reduced support")
            if len(Fs) % 2 == par:
                refuse("F has satisfiable parity: not a facet")
            key = (tuple(sorted(rlist)), tuple(sorted(Fs)))
            if key in seen:
                refuse("duplicate facet")
            seen.add(key)
            for i in sup:
                load[i] = load.get(i, Fraction(0)) + \
                    (-y if i in Fs else y)
            by += y * (1 - len(Fs))
        z = Fraction(0)
        for i, t in load.items():
            excess = t - w[i]
            if excess > 0:
                z += excess
        return by - z

    budget = [200000]

    def walk(node, fix):
        budget[0] -= 1
        if budget[0] < 0:
            refuse("node budget")
        if not (isinstance(node, list) and node):
            refuse("malformed node")
        kind = node[0]
        if kind == "branch":
            if len(node) != 4:
                refuse("malformed branch")
            k = int(node[1])
            if k in fix or not (0 <= k < len(w)):
                refuse(f"bad branch variable {k}")
            walk(node[2], {**fix, k: 0})
            walk(node[3], {**fix, k: 1})
            return
        if kind == "infeasible":
            rlist = ([int(r) for r in node[1]]
                     if isinstance(node[1], list) else [int(node[1])])
            sup: set = set()
            par = 0
            for r in rlist:
                if not (0 <= r < len(checks)):
                    refuse("infeasible row out of range")
                s2, p2 = reduced(r, fix)
                sup ^= set(s2)
                par ^= p2
            if sup:
                refuse("claimed-infeasible combo has support")
            if par == 0:
                refuse("combo is 0=0: feasible")
            return
        if kind == "bound":
            if len(node) != 2:
                refuse("malformed bound leaf")
            W1 = sum(w[i] for i, a in fix.items() if a == 1)
            L = leaf_bound(node[1], fix)
            if not (W1 + L > U - step):
                refuse(f"leaf bound {W1}+{L} fails vs {U}-{step}")
            return
        refuse(f"unknown node kind {kind}")

    try:
        walk(doc["tree"], {})
    except RecursionError:
        refuse("recursion limit")
    print(f"ACCEPT: candidate weight {U} proven minimum by branch duals")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: tree_checker_b.py <document.json>")
        raise SystemExit(1)
    try:
        raise SystemExit(main(sys.argv[1]))
    except SystemExit:
        raise
    except Exception as e:                          # malformed input
        print(f"MALFORMED: {type(e).__name__}: {e}")
        raise SystemExit(1)
