#!/usr/bin/env python3
r"""PRODUCER EQUIVALENCE: the refactored producers emit the same evidence.

A producer and a checker fail differently, and the difference decides what
a gate has to prove.

A CHECKER that changes behaviour is a soundness catastrophe: it can accept
something false, and every receipt downstream inherits the lie. That is why
tools/checker_equivalence_gate.py compares the full verdict, reason string
included, on a shape-varied corpus.

A PRODUCER cannot do that. Nothing it emits is trusted; its output is
re-derived from (H, w, s) by a checker that consults it for nothing. A
producer that breaks therefore fails CLOSED -- the checker refuses, the
receipt is not issued, the tier drops. The soundness of the system does
not depend on the producer being correct at all.

What a broken producer CAN do is quietly certify LESS. If a refactor makes
feldman_dual return None on instances it used to solve, or makes
certified_bnb build a tree the checker no longer accepts, nothing goes red
-- the pipeline simply falls to a weaker tier, and a headline rate like
"tier-1 on 93.5%" silently becomes a smaller number that still looks like
a result. That is the failure this gate exists to catch, and it is
invisible to the checker gate by construction.

So the claim here is YIELD-AND-CONTENT equivalence:

  * the same instances produce a certificate (never None where the frozen
    producer returned one)
  * the bound is numerically identical
  * the primal support is identical
  * the facet dual list is identical, element for element
  * certified_bnb builds a structurally identical tree
  * exact_milp reports the same support, weight and tier status

and, because a producer's whole job is to be checkable, every certificate
that either side emits is handed to the LIVE checker and must still be
accepted. A producer refactor that keeps its output identical but no
longer verifies would be caught by that leg alone.

    python tools/producer_equivalence_gate.py
    python tools/producer_equivalence_gate.py --prove-sensitive
    python tools/producer_equivalence_gate.py --cases 400 --json evidence/...
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import pathlib
import random
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

FROZEN_SHA256 = {
    "tjoin_exact.py":
        "9d326865e47f16556ab64cd7469cafd2939e85abebec785f71321f3c9717e9ac",
    # RE-FROZEN 2026-08-15 (protocol option 2; behaviour commits 8cf6863 +
    # the exact-dual-polish commit). The class of outputs that moved:
    # certificates beside an INTEGRAL primal whose float duals fell
    # ulp-short of the weight in exact arithmetic (20 of 146 on this
    # corpus) now carry exact-rational duals re-solved from the same
    # facet pool; bounds are identical, only the dual values move. The
    # verify leg is the proof: 126/146 -> 146/146.
    "qldpc_cert.py":
        "a21c6b89f4b21a13d3e14167c1b6143c9c4e44543251eb068f8d0c12c9aa43fb",
    # option 3: new module the re-frozen producer imports for the polish;
    # its reach mutation is fd-skip-exact-polish below.
    "exact_lp.py":
        "97a88f6b70034dce3059e4fae0a6022ee773069a7d9ae0f3db0d855d77d4c3cc",
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
    for name, want in FROZEN_SHA256.items():
        p = ROOT / "tools" / "frozen" / name
        if not p.exists():
            raise SystemExit(f"FROZEN SNAPSHOT MISSING: {p}")
        got = hashlib.sha256(_norm_bytes(p)).hexdigest()
        if got != want:
            raise SystemExit(
                f"FROZEN SNAPSHOT ALTERED: {name}\n  expected {want}\n"
                f"  found    {got}\nThe reference is not editable. "
                f"See docs/REFREEZE_PROTOCOL.md.")


def _instance(rng: random.Random):
    """A syndrome-LP instance the producers can actually work on.

    Shape varies -- degree, density, overlap, weight kind -- for the same
    reason the checker corpus does: a corpus uniform in shape hides
    defects at any coverage level.
    """
    style = rng.choice(("sparse", "dense", "lowdeg", "overlap"))
    n = rng.randint(4, 18)
    m = rng.randint(2, 9)
    if style == "lowdeg":
        lo, hi = 1, 2
    elif style == "dense":
        lo, hi = 3, min(n, 7)
    elif style == "overlap":
        lo, hi = 2, min(n, 5)
    else:
        lo, hi = 1, min(n, 4)
    checks = [tuple(sorted(rng.sample(range(n), rng.randint(lo, max(lo, hi)))))
              for _ in range(m)]
    n = max(max(c) + 1 for c in checks if c) if any(checks) else 2
    # "skewed" exists to reach the BOX DUALS. A box multiplier is only
    # non-zero when a variable saturates at its upper bound, which almost
    # never happens with uniform weights -- so `fd-drop-box-duals` was
    # caught by 2 divergences and tripped the margin floor. Near-free
    # variables get driven to 1 and make the bound bind.
    kind = rng.choice(("uniform", "int", "float", "skewed", "skewed"))
    if kind == "uniform":
        w = {i: 1.0 for i in range(n)}
    elif kind == "int":
        w = {i: float(rng.randint(1, 4)) for i in range(n)}
    elif kind == "skewed":
        w = {i: (0.01 if rng.random() < 0.4 else round(rng.uniform(1.0, 4.0), 3))
             for i in range(n)}
    else:
        w = {i: round(rng.uniform(0.3, 3.0), 3) for i in range(n)}
    # a syndrome DERIVED from a planted error, so the instance is
    # feasible and the producers have something to find
    e = set(rng.sample(range(n), rng.randint(0, min(n, 4))))
    syn = [len(e & set(c)) % 2 for c in checks]
    return checks, w, syn


def _norm_cert(got):
    """A comparable rendering of whatever a producer returned."""
    if got is None:
        return None
    cert = got
    if isinstance(got, tuple):
        cert = got[0]
    if cert is None:
        return None
    fd = tuple(sorted(
        (str(j), tuple(sorted(int(i) for i in F)), round(float(y), 9))
        for (j, F, y) in getattr(cert, "facet_duals", ())))
    bd = tuple(sorted((int(i), round(float(v), 9))
                      for (i, v) in getattr(cert, "box_duals", ())))
    return (tuple(sorted(int(i) for i in getattr(cert, "error_support", ()))),
            fd, bd)


def _norm_tree(tree):
    if tree is None:
        return None
    if isinstance(tree, (tuple, list)):
        return tuple(_norm_tree(x) for x in tree)
    if isinstance(tree, float):
        return round(tree, 9)
    return tree


def _r(v, nd=9):
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


# INSTANCES THAT ACTUALLY BRANCH, pinned rather than sampled.
#
# The branching path is rare: at the default cut strength only ~4% of
# random trees contain a branch node, so a --fast run of 40 cases found
# NONE and the dark-leg guard correctly refused to call that a pass.
# Weakening the guard would have been the cheap green and the wrong move
# -- it exists precisely because "the branch code never ran" and "the
# branch code is fine" print the same way.
#
# These three were found by search (31 trials) and verified to produce
# 1, 1 and 2 branch nodes with rpc_rounds=0. They run on EVERY
# invocation regardless of --cases, so coverage of the branch path is a
# property of the gate rather than a property of the sample size.
_BRANCHING_INSTANCES = [
    ([(2, 4), (3, 4, 6, 7), (4, 5), (3, 7)], (6, 7)),
    ([(4, 5, 6), (1, 4, 5, 6), (0, 5, 7), (1, 5, 7), (2, 5), (2, 4),
      (2, 3, 4)], (1, 4, 6)),
    ([(0, 2, 4, 5), (0, 3, 4), (3, 4, 5), (0, 1, 3, 5)], (0, 1, 4)),
]


def run(frozen, live, live_check, cases: int, seed: int, verbose=False,
        live_tjoin=None):
    import importlib as _il
    fz_tjoin = _il.import_module("tools.frozen.tjoin_exact")
    lv_tjoin = live_tjoin or _il.import_module("oneq.tjoin_exact")
    rng = random.Random(seed)
    diffs = []
    st = {"instances": 0, "fd_cert": 0, "lazy_cert": 0, "milp": 0,
          "bnb_tree": 0, "verified": 0, "verify_fail": 0,
          "bnb_branching": 0, "tjoin": 0,
          "tjoin_bound": 0}

    for c in range(cases):
        checks, w, syn = _instance(rng)
        st["instances"] += 1

        # ---- feldman_dual, with and without RPC rounds
        for rpc in (0, 4):
            try:
                a = frozen.feldman_dual(checks, w, syn, return_primal=True,
                                        rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                a = ("<raised>", type(exc).__name__)
            try:
                b = live.feldman_dual(checks, w, syn, return_primal=True,
                                      rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                b = ("<raised>", type(exc).__name__)
            na = (_norm_cert(a), _r(a[1]) if isinstance(a, tuple)
                  and len(a) > 1 else None)
            nb = (_norm_cert(b), _r(b[1]) if isinstance(b, tuple)
                  and len(b) > 1 else None)
            if na[0] is not None:
                st["fd_cert"] += 1
            if na != nb:
                diffs.append({"case": c, "fn": f"feldman_dual(rpc={rpc})",
                              "frozen": str(na)[:400], "live": str(nb)[:400]})

        # ---- feldman_dual_lazy
        try:
            a = frozen.feldman_dual_lazy(checks, w, syn, return_primal=True)
        except Exception as exc:                     # noqa: BLE001
            a = ("<raised>", type(exc).__name__)
        try:
            b = live.feldman_dual_lazy(checks, w, syn, return_primal=True)
        except Exception as exc:                     # noqa: BLE001
            b = ("<raised>", type(exc).__name__)
        na, nb = _norm_cert(a), _norm_cert(b)
        if na is not None:
            st["lazy_cert"] += 1
        if na != nb:
            diffs.append({"case": c, "fn": "feldman_dual_lazy",
                          "frozen": str(na)[:400], "live": str(nb)[:400]})

        # ---- exact_milp: support, weight and TIER STATUS
        try:
            a = frozen.exact_milp(checks, w, syn, return_status=True)
            a = (tuple(sorted(int(i) for i in a[0])) if a[0] is not None
                 else None, _r(a[1]), str(a[2]))
        except Exception as exc:                     # noqa: BLE001
            a = ("<raised>", type(exc).__name__)
        try:
            b = live.exact_milp(checks, w, syn, return_status=True)
            b = (tuple(sorted(int(i) for i in b[0])) if b[0] is not None
                 else None, _r(b[1]), str(b[2]))
        except Exception as exc:                     # noqa: BLE001
            b = ("<raised>", type(exc).__name__)
        st["milp"] += 1
        if a != b:
            diffs.append({"case": c, "fn": "exact_milp",
                          "frozen": str(a)[:400], "live": str(b)[:400]})

        # ---- certified_bnb: the TREE must be structurally identical
        wi = {i: int(round(v)) or 1 for i, v in w.items()}
        e = set(rng.sample(range(len(wi)), rng.randint(1, min(len(wi), 3))))
        syn_b = [len(e & set(cc)) % 2 for cc in checks]
        U = sum(wi[i] for i in e)
        # BOTH cut regimes. With the default rpc_rounds the flat dual
        # closes almost everything -- measured: 3 of 75 trees contained a
        # single branch node -- so the BRANCHING code was effectively
        # unreachable and `bnb-worst-branch-var` was invisible. The RPC
        # cuts are precisely what suppresses branching, so running with
        # rpc_rounds=0 on both sides exposes it (24 of 75 branch), while
        # the default run keeps the regime production actually uses.
        for rpc in (6, 0):
            try:
                ta, _na = frozen.certified_bnb(checks, wi, syn_b, U,
                                               max_nodes=250, max_depth=10,
                                               rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                ta = ("<raised>", type(exc).__name__)
            try:
                tb, _nb = live.certified_bnb(checks, wi, syn_b, U,
                                             max_nodes=250, max_depth=10,
                                             rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                tb = ("<raised>", type(exc).__name__)
            if ta is not None:
                st["bnb_tree"] += 1
                if _norm_tree(ta) and str(_norm_tree(ta)).count("branch"):
                    st["bnb_branching"] += 1
            if _norm_tree(ta) != _norm_tree(tb):
                diffs.append({"case": c, "fn": f"certified_bnb(rpc={rpc})",
                              "frozen": str(_norm_tree(ta))[:400],
                              "live": str(_norm_tree(tb))[:400]})

        # ---- exact_tjoin_sets: the T-join cut-packing producer. It
        #      lives in a different module and shares no code with the
        #      qLDPC path, so nothing above says anything about it.
        #      Compared on the CONVERGED BOUND (the number the ladder
        #      consumes) and the canonical cut family.
        if c % 2 == 0:
            nv = rng.randint(3, 8)
            tedges = [(u, u + 1, round(rng.uniform(0.4, 3.0), 3))
                      for u in range(nv - 1)]
            tedges.append((0, -1, round(rng.uniform(0.4, 3.0), 3)))
            tedges.append((nv - 1, -1, round(rng.uniform(0.4, 3.0), 3)))
            for _ in range(rng.randint(0, 3)):
                a, b = rng.sample(range(nv), 2)
                tedges.append((a, b, round(rng.uniform(0.4, 4.0), 3)))
            T = sorted(rng.sample(range(nv), rng.randint(1, min(nv, 4))))
            # BOTH round budgets. The final re-solve -- the fix that keeps
            # the duals aligned with the FINAL pool -- guards the case
            # where the loop exits immediately after admitting a cut the
            # LP never saw. With the default max_rounds=200 that never
            # happens on graphs this size, so `tjoin-skip-final-resolve`
            # was invisible: the gate could not tell whether the last
            # admitted cut was being silently dropped from the family.
            for rounds in (200, 2):
                try:
                    sa, ba = fz_tjoin.exact_tjoin_sets(tedges, T,
                                                       max_rounds=rounds)
                    sa = (tuple(sorted(tuple(sorted(S)) for S in sa)),
                          _r(ba, 7))
                except Exception as exc:             # noqa: BLE001
                    sa = ("<raised>", type(exc).__name__)
                try:
                    sb, bb = lv_tjoin.exact_tjoin_sets(tedges, T,
                                                       max_rounds=rounds)
                    sb = (tuple(sorted(tuple(sorted(S)) for S in sb)),
                          _r(bb, 7))
                except Exception as exc:             # noqa: BLE001
                    sb = ("<raised>", type(exc).__name__)
                st["tjoin"] += 1
                if sa[0] != "<raised>" and sa[1]:
                    st["tjoin_bound"] += 1
                if sa != sb:
                    diffs.append({"case": c,
                                  "fn": f"exact_tjoin_sets(r={rounds})",
                                  "frozen": str(sa)[:400],
                                  "live": str(sb)[:400]})

        # ---- and every LIVE certificate must still VERIFY. A producer
        # whose output is unchanged but no longer checkable would pass
        # every comparison above and still have broken the pipeline.
        try:
            cert, _bound, primal = live.feldman_dual(
                checks, w, syn, return_primal=True, rpc_rounds=4)
            if cert is not None and primal is not None:
                res = live_check.check_qldpc(
                    checks=checks, weights=w, syndrome=syn,
                    cert=live.QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals))
                if res.accepted:
                    st["verified"] += 1
                else:
                    st["verify_fail"] += 1
        except Exception:                            # noqa: BLE001
            pass

        if verbose and c and c % 50 == 0:
            print(f"    {c} instances, {len(diffs)} divergences")

    # the pinned branching instances, every run
    for (bchecks, be) in _BRANCHING_INSTANCES:
        nvar = max(max(c) for c in bchecks) + 1
        bw = {i: 1 for i in range(nvar)}
        bes = set(be)
        bsyn = [len(bes & set(cc)) % 2 for cc in bchecks]
        bU = sum(bw[i] for i in be)
        for rpc in (0, 6):
            try:
                ta, _ = frozen.certified_bnb(bchecks, bw, bsyn, bU,
                                             max_nodes=300, max_depth=12,
                                             rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                ta = ("<raised>", type(exc).__name__)
            try:
                tb, _ = live.certified_bnb(bchecks, bw, bsyn, bU,
                                           max_nodes=300, max_depth=12,
                                           rpc_rounds=rpc)
            except Exception as exc:                 # noqa: BLE001
                tb = ("<raised>", type(exc).__name__)
            if ta is not None:
                st["bnb_tree"] += 1
                if "branch" in str(_norm_tree(ta)):
                    st["bnb_branching"] += 1
            if _norm_tree(ta) != _norm_tree(tb):
                diffs.append({"case": -1,
                              "fn": f"certified_bnb/pinned(rpc={rpc})",
                              "frozen": str(_norm_tree(ta))[:400],
                              "live": str(_norm_tree(tb))[:400]})

    return diffs, st


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Behavioural mutations of the PRODUCERS. Each makes the producer emit
# something different -- a weaker bound, a missing cut, a truncated tree.
# A producer mutation that this gate cannot see is a yield regression it
# would not have caught either.
# Same floor as the checker gate, for the same reason: a mutation caught
# once is a coin toss wearing the word "caught".
MIN_MARGIN = 3

_MUTATIONS = [
    # RPC cut generation: skipping it costs tier-1 yield on exactly the
    # fractional pseudocodewords the cuts exist to kill.
    ("fd-no-rpc-rounds", "for _round in range(int(rpc_rounds)):",
     "for _round in range(0):"),
    ("fd-stop-cutting-early", "if frac.max() < 1e-7:", "if True:"),
    # the box multipliers are part of the dual the checker sums
    ("fd-drop-box-duals", "box_duals=boxes)", "box_duals=())"),
    # the exact dual polish (2026-08-15): disabling it ships ulp-short
    # float duals again, which the frozen producer now polishes -- so the
    # corpus's shortfall instances diverge, and the verify leg would read
    # PRODUCED-BUT-UNCHECKABLE besides. This mutation is also the reach
    # proof REFREEZE option 3 demands for the frozen exact_lp.py: a
    # divergence here means the corpus genuinely exercises the polish,
    # and the polish is the only caller of exact_weighted_lp.
    # (replace-first hits the lazy producer's hook; both hooks share the
    # line, and the lazy path is the production one.)
    ("fd-skip-exact-polish",
     "cert = _polish_if_short(checks, weights, syndrome, cert, primal)",
     "cert = cert"),
    # branch-variable choice is a heuristic, but a WORSE one yields
    # bigger trees and more budget exhaustion -- i.e. fewer BD receipts
    ("bnb-worst-branch-var",
     "pick = min(live, key=lambda i: abs(xd.get(i, 0.0) - 0.5))",
     "pick = max(live, key=lambda i: abs(xd.get(i, 0.0) - 0.5))"),
    # never emit a bound leaf: every subtree must branch to exhaustion
    ("bnb-never-bound",
     "if err is None and L is not None and L + W1 > Ux - step:",
     "if False:"),
    ("lazy-one-round", "max_rounds: int = 60", "max_rounds: int = 1"),
]


_TJOIN_MUTATIONS = [
    ("tjoin-one-round", "for _round in range(max_rounds):",
     "for _round in range(1):"),
    ("tjoin-stop-separating",
     "if best_val is None or best_val >= 1.0 - 1e-7:", "if True:"),
    ("tjoin-wrong-cut-parity", "if len(set(side) & Tp) % 2 == 1:",
     "if len(set(side) & Tp) % 2 == 0:"),
    # the re-solve that keeps duals aligned with the FINAL pool; without
    # it the last admitted cut is silently dropped from the family
    ("tjoin-skip-final-resolve", "if pool and len(duals) != len(pool):",
     "if False:"),
]


def _prove_sensitive(frozen, live_check, cases, seed) -> int:
    src = (ROOT / "src" / "oneq" / "qldpc_cert.py").read_text(encoding="utf-8")
    caught, missed = [], []
    with tempfile.TemporaryDirectory() as td:
        for label, find, repl in _MUTATIONS:
            if find not in src:
                missed.append(f"{label} (PATTERN ABSENT: {find!r})")
                print(f"  MISSED  {label:24s} <-- pattern absent")
                continue
            d = pathlib.Path(td) / label
            d.mkdir()
            (d / "__init__.py").write_text("", encoding="utf-8")
            # the mutant needs its sibling checker to import
            (d / "qldpc_check.py").write_text(
                (ROOT / "src" / "oneq" / "qldpc_check.py").read_text(
                    encoding="utf-8"), encoding="utf-8")
            (d / "qldpc_cert.py").write_text(src.replace(find, repl, 1),
                                             encoding="utf-8")
            sys.path.insert(0, str(d.parent))
            try:
                mod = importlib.import_module(f"{label}.qldpc_cert")
            except Exception as exc:                 # noqa: BLE001
                missed.append(f"{label} (IMPORT FAILED: {exc})")
                print(f"  MISSED  {label:24s} <-- import failed")
                continue
            finally:
                sys.path.pop(0)
            diffs, _st = run(frozen, mod, live_check, cases, seed)
            if diffs:
                caught.append((label, len(diffs)))
                print(f"  CAUGHT  {label:24s} {len(diffs)} divergences")
            else:
                missed.append(label)
                print(f"  MISSED  {label:24s} <-- corpus hole")

    # The T-join packing producer lives in its own module and shares no
    # code with the qLDPC path, so nothing above says anything about it.
    tj_src = (ROOT / "src" / "oneq" / "tjoin_exact.py").read_text(
        encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        for label, find, repl in _TJOIN_MUTATIONS:
            if find not in tj_src:
                missed.append(f"{label} (PATTERN ABSENT: {find!r})")
                print(f"  MISSED  {label:26s} <-- pattern absent")
                continue
            d = pathlib.Path(td) / label
            d.mkdir()
            (d / "__init__.py").write_text("", encoding="utf-8")
            (d / "matching_cert.py").write_text(
                (ROOT / "src" / "oneq" / "matching_cert.py").read_text(
                    encoding="utf-8"), encoding="utf-8")
            (d / "tjoin_exact.py").write_text(tj_src.replace(find, repl, 1),
                                              encoding="utf-8")
            sys.path.insert(0, str(d.parent))
            try:
                mod = importlib.import_module(f"{label}.tjoin_exact")
            except Exception as exc:                 # noqa: BLE001
                missed.append(f"{label} (IMPORT FAILED: {exc})")
                print(f"  MISSED  {label:26s} <-- import failed")
                continue
            finally:
                sys.path.pop(0)
            live_ok = importlib.import_module("oneq.qldpc_cert")
            diffs, _st = run(frozen, live_ok, live_check, cases, seed,
                             live_tjoin=mod)
            if diffs:
                caught.append((label, len(diffs)))
                print(f"  CAUGHT  {label:26s} {len(diffs)} divergences")
            else:
                missed.append(label)
                print(f"  MISSED  {label:26s} <-- corpus hole")

    if missed:
        print(f"\nSENSITIVITY FAILED: {len(missed)} invisible:")
        for m in missed:
            pass
    thin = [(lab, k) for (lab, k) in caught if k < MIN_MARGIN]
    if thin and not missed:
        print(f"\nMARGIN TOO THIN ({len(thin)}): caught, but barely -- "
              f"the state that flips to MISSED on another seed")
        for lab, k in thin:
            print(f"    {lab}: {k} divergence(s), floor {MIN_MARGIN}")
        return 1
    if missed:
        for m in missed:
            print(f"    {m}")
        return 1
    print(f"\nSENSITIVITY PROVEN: {len(caught)}/{len(_MUTATIONS) + len(_TJOIN_MUTATIONS)} producer "
          f"mutations caught. This gate can fail.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--prove-sensitive", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    _verify_freeze()
    print("frozen producer snapshot verified against the pinned sha256")

    frozen = importlib.import_module("tools.frozen.qldpc_cert")
    live = importlib.import_module("oneq.qldpc_cert")
    live_check = importlib.import_module("oneq.qldpc_check")

    if args.prove_sensitive:
        per = max(40, args.cases // 3)
        print(f"\nSENSITIVITY ({len(_MUTATIONS) + len(_TJOIN_MUTATIONS)} producer mutations, "
              f"{per} instances each)")
        return _prove_sensitive(frozen, live_check, per, args.seed)

    print(f"\nPRODUCER EQUIVALENCE: frozen vs live, {args.cases} instances")
    diffs, st = run(frozen, live, live_check, args.cases, args.seed,
                    verbose=True)
    print(f"  instances            : {st['instances']}")
    print(f"  feldman_dual certs   : {st['fd_cert']}")
    print(f"  lazy certs           : {st['lazy_cert']}")
    print(f"  exact_milp calls     : {st['milp']}")
    print(f"  certified_bnb trees  : {st['bnb_tree']} "
          f"(with branch nodes: {st['bnb_branching']})")
    print(f"  tjoin packings       : {st['tjoin']} "
          f"(nonzero bound: {st['tjoin_bound']})")
    print(f"  live certs VERIFIED  : {st['verified']} "
          f"(failed {st['verify_fail']})")

    # Yield legs that never fired prove nothing about yield.
    for leg in ("fd_cert", "lazy_cert", "bnb_tree", "verified",
                "bnb_branching", "tjoin", "tjoin_bound"):
        if st[leg] == 0:
            print(f"DARK LEG: '{leg}' never happened -- this gate would not "
                  f"notice that producer regressing, so it is a FAILURE")
            return 1
    if st["verify_fail"]:
        print(f"PRODUCED-BUT-UNCHECKABLE: {st['verify_fail']} live "
              f"certificates were emitted and then REFUSED by the live "
              f"checker -- the producer and checker disagree")
        return 1

    out = {"schema": "oneq-producer-equivalence/1",
           "cases": args.cases, "seed": args.seed,
           "divergences": len(diffs), "stats": st,
           "frozen_sha256": FROZEN_SHA256,
           "claim": ("the refactored producers emit byte-identical "
                     "certificates, bounds, tiers and branch trees, and "
                     "every certificate they emit still verifies")}
    if args.json:
        p = pathlib.Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"evidence -> {p}")

    if diffs:
        print(f"\nDIVERGENCE: {len(diffs)} producer output(s) differ")
        print("  A producer regression is a YIELD loss, not a soundness "
              "failure -- and invisible without this gate. Intentional? "
              "docs/REFREEZE_PROTOCOL.md.")
        for d in diffs[:6]:
            print(f"  case {d['case']} {d['fn']}")
            print(f"    frozen: {d['frozen'][:200]}")
            print(f"    live  : {d['live'][:200]}")
        return 1
    print("\nEQUIVALENT: same certificates, same bounds, same tiers, "
          "same trees -- and all of them still verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
