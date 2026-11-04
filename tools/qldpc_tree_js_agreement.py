r"""Paper 2's branch-dual trees, decided by a checker in a second language.

Both checkers that replay paper 2's trees are Python. This exports every tree
receipt named by the pinned receipt manifest (held-out, development and the
development circuit trees), with its hash-bound instance and integer weight
vector, adds planted inputs derived from real trees, runs
verifier-js/verify-qldpc-tree.mjs (JavaScript, exact BigInt rationals, written
from the manuscript's statement of tier BD and the serialized-tree schema), and
requires

  * every tree accepted;
  * every planted defect refused FOR ITS OWN REASON -- a defect refused by some
    other rule would not show that the rule it targets is alive;
  * the rules no stored tree exercises -- a leaf closed by infeasibility, the
    lattice boundary itself, an overloaded multiplier -- decided on a hand-built
    instance whose one branch is provably infeasible, both ways.

Every planted input is also given to the production checker, which must reach
the same verdict, so a mutation that leaves a true claim true cannot count as a
defect.

    python tools/qldpc_tree_js_agreement.py              # writes evidence/qldpc_tree_js_agreement.json
    python tools/qldpc_tree_js_agreement.py --js other.mjs --no-write
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from oneq.qldpc_check import check_qldpc_bnb                      # noqa: E402
from replay_receipts import PINNED_MANIFEST_SHA256, _deser_tree   # noqa: E402

MANIFEST = ROOT / "evidence" / "receipt_manifest.json"
ADDENDUM = ROOT / "evidence" / "qldpc_heldout_addendum.json"
REFSCHED = ROOT / "evidence" / "qldpc_dev_refsched_trees.json"
JS = ROOT / "verifier-js" / "verify-qldpc-tree.mjs"
OUT = ROOT / "evidence" / "qldpc_tree_js_agreement.json"
TREE_COHORTS = ("heldout_tree", "dev_tree", "dev_refsched_tree")

# the refusal each planted defect must receive (a substring of the checker's reason)
REASON = {
    "candidate_extra_variable": "does not satisfy check",
    "syndrome_bit_dropped": "does not satisfy check",
    "empty_leaf_on_open_branch": "leaf not proven",
    "leaf_duals_zeroed": "leaf not proven",
    "branch_on_fixed_variable": "already fixed",
    "branch_missing_child": "malformed branch",
    "unknown_node_kind": "unknown node kind",
    "facet_on_fixed_variable": "outside its reduced check",
    "facet_wrong_parity": "wrong parity",
    "negative_multiplier": "negative multiplier",
    "row_out_of_range": "out of range",
    "infeasible_leaf_support_not_empty": "do not XOR to empty",
    "infeasible_leaf_even_parity": "even parity",
    "lattice_boundary_leaf": "leaf not proven",
    "overloaded_multiplier": "leaf not proven",
    # a malformed facet appended to a bound leaf at multiplier 0 (final referee, 2026-11-03): read and refused
    "zero_multiplier_row_out_of_range": "out of range",
    "zero_multiplier_facet_on_fixed_variable": "outside its reduced check",
    "zero_multiplier_facet_wrong_parity": "wrong parity",
    "zero_multiplier_facet_listed_twice": "listed twice",
}


class AgreementError(Exception):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_batch() -> dict:
    raw = MANIFEST.read_bytes()
    if _sha(raw) != PINNED_MANIFEST_SHA256:
        raise AgreementError("receipt manifest does not match the pinned hash")
    man = json.loads(raw)
    for rel, h in man["files"].items():
        if rel.startswith("evidence/instances/") and _sha((ROOT / rel).read_bytes()) != h:
            raise AgreementError(f"{rel} does not match its manifest hash")
    add = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    trees = {f"heldout_tree/{t['rung']}/{t['shot_index']}": t for t in add["bd_trees"]}
    trees.update({f"dev_tree/{t['rung']}/{t['shot_index']}": t for t in add["dev_bd_trees"]})
    trees.update({f"dev_refsched_tree/{t['rung']}/{t['shot_index']}": t
                  for t in json.loads(REFSCHED.read_text(encoding="utf-8"))["trees"]})
    instances = {k: json.loads((ROOT / v["file"]).read_text(encoding="utf-8")) for k, v in man["instances"].items()}
    batch = {"instances": {k: {"n": v["n_variables"], "checks": v["checks"]} for k, v in instances.items()},
             "weights": {}, "receipts": []}
    for m in man["receipts"]:
        if m["cohort"] not in TREE_COHORTS:
            continue
        t = trees[m["id"]]
        wf = m["weights_file"]
        if wf not in batch["weights"]:
            vec = [int(x) for x in json.loads((ROOT / wf).read_text(encoding="utf-8"))["vector"]]
            if _sha(json.dumps(vec).encode()) != m["weights_vector_sha256"]:
                raise AgreementError(f"{wf} does not match its manifest hash")
            batch["weights"][wf] = vec
        batch["receipts"].append({"id": m["id"], "instance": m["instance"], "weights": wf,
                                  "syndrome_support": t["syndrome_support"], "candidate": t["candidate"],
                                  "tree": t["tree"], "expect": "accept", "kind": "receipt"})
    want = sum(man["expected_counts"][c] for c in TREE_COHORTS)
    if len(batch["receipts"]) != want:
        raise AgreementError("the batch does not hold every tree receipt the manifest names")
    return batch


def _python_accepts(batch: dict, r: dict) -> bool:
    """The production checker's verdict on the same input (unconstructible input is a refusal)."""
    inst = batch["instances"][r["instance"]]
    checks = [tuple(c) for c in inst["checks"]]
    syn = [0] * len(checks)
    try:
        for j in r["syndrome_support"]:
            syn[int(j)] = 1
        tree = _deser_tree(r["tree"])
        wx = {i: Fraction(w) for i, w in enumerate(batch["weights"][r["weights"]])}
        return bool(check_qldpc_bnb(checks=checks, weights=wx, syndrome=syn,
                                    error_support=tuple(int(i) for i in r["candidate"]), tree=tree).accepted)
    except Exception:  # noqa: BLE001  a malformed input is a refusal, whatever the exception
        return False


def _leftmost_bound(tree):
    """Path (list of child indices) to the first bound leaf reached by always taking x_k = 0."""
    path, node = [], tree
    while node[0] == "branch":
        path.append(2)
        node = node[2]
    return (path, node) if node[0] == "bound" else (None, None)


def _at(tree, path):
    node = tree
    for k in path:
        node = node[k]
    return node


def _find_infeasible(tree, path=()):
    if tree[0] == "infeasible":
        return list(path)
    if tree[0] == "branch":
        return _find_infeasible(tree[2], path + (2,)) or _find_infeasible(tree[3], path + (3,))
    return None


def _reduced_support(inst, rows, fixed):
    count: dict[int, int] = {}
    for r in rows:
        for i in inst["checks"][r]:
            if i not in fixed:
                count[i] = count.get(i, 0) ^ 1
    return {i for i, odd in count.items() if odd}


def planted(batch: dict) -> list[dict]:
    """Defects derived from the first tree of each rung."""
    by_rung: dict[str, list[dict]] = {}
    for r in batch["receipts"]:
        by_rung.setdefault(r["id"].split("/", 1)[1].rsplit("/", 1)[0], []).append(r)
    out = []
    for rung, rs in sorted(by_rung.items()):
        base = rs[0]
        inst = batch["instances"][base["instance"]]
        n, m = inst["n"], len(inst["checks"])

        def variant(cls, mutate, src=base):
            r = copy.deepcopy(src)
            r["id"] = f"planted/{cls}/{rung}"
            r["expect"] = "refuse"
            r["kind"] = cls
            mutate(r)
            out.append(r)

        free = next(i for i in range(n) if i not in set(base["candidate"]))
        variant("candidate_extra_variable", lambda r: r["candidate"].append(free))
        if base["syndrome_support"]:
            variant("syndrome_bit_dropped", lambda r: r["syndrome_support"].pop(0))
        variant("unknown_node_kind", lambda r: r["tree"].__setitem__(0, "bounds"))

        if base["tree"][0] == "branch":
            k = base["tree"][1]
            # the branch's x_k = 0 child replaced by a bound leaf with no duals: nothing fixed, L = 0
            variant("empty_leaf_on_open_branch", lambda r: r["tree"].__setitem__(2, ["bound", []]))
            variant("branch_on_fixed_variable",
                    lambda r: r["tree"].__setitem__(2, ["branch", k, r["tree"][2], copy.deepcopy(r["tree"][2])]))
            variant("branch_missing_child", lambda r: r["tree"].pop())

        path, leaf = _leftmost_bound(base["tree"])
        if path is not None:
            fixed = set()
            node = base["tree"]
            for _ in path:
                fixed.add(node[1])
                node = node[2]
            duals = leaf[1]
            pos = next((q for q, f in enumerate(duals) if f[2] > 0), None)

            def set_leaf_duals(r, value):
                _at(r["tree"], path)[1] = value

            variant("leaf_duals_zeroed", lambda r: set_leaf_duals(r, [[f[0], f[1], 0, 1] for f in duals]))
            if pos is not None:
                f0 = duals[pos]
                rows = f0[0] if isinstance(f0[0], list) else [f0[0]]
                support = _reduced_support(inst, rows, fixed)

                def set_facet(r, q, value):
                    _at(r["tree"], path)[1][pos][q] = value

                if fixed:
                    variant("facet_on_fixed_variable", lambda r: set_facet(r, 1, sorted(set(f0[1]) | {min(fixed)})))
                inside = next((i for i in sorted(support) if i not in f0[1]), None)
                if f0[1] or inside is not None:
                    variant("facet_wrong_parity", lambda r: set_facet(r, 1, f0[1][:-1] if f0[1] else [inside]))
                variant("negative_multiplier", lambda r: set_facet(r, 2, -f0[2]))
                variant("row_out_of_range", lambda r: set_facet(r, 0, m))
                # ZERO MULTIPLIERS (final referee, 2026-11-03): a malformed facet APPENDED to the leaf at multiplier
                # 0 changes no number, so only a checker that reads it before skipping it can refuse it
                row0 = copy.deepcopy(f0[0])

                def append_facet(r, facet):
                    _at(r["tree"], path)[1].append(facet)

                variant("zero_multiplier_row_out_of_range", lambda r: append_facet(r, [m, [], 0, 1]))
                if fixed:
                    variant("zero_multiplier_facet_on_fixed_variable",
                            lambda r: append_facet(r, [copy.deepcopy(row0), sorted(set(f0[1]) | {min(fixed)}), 0, 1]))
                if f0[1] or inside is not None:
                    variant("zero_multiplier_facet_wrong_parity",
                            lambda r: append_facet(r, [copy.deepcopy(row0), f0[1][:-1] if f0[1] else [inside], 0, 1]))
                variant("zero_multiplier_facet_listed_twice",
                        lambda r: append_facet(r, [copy.deepcopy(row0), list(f0[1]), 0, 1]))

        ipath = next((p for r in rs if (p := _find_infeasible(r["tree"])) is not None), None)
        if ipath is not None:
            src = next(r for r in rs if _find_infeasible(r["tree"]) == ipath)
            fixed = set()
            node = src["tree"]
            for step in ipath:
                fixed.add(node[1])
                node = node[step]
            wide = next(j for j in range(m) if _reduced_support(inst, [j], fixed))
            variant("infeasible_leaf_support_not_empty", lambda r: _at(r["tree"], ipath).__setitem__(1, wide), src=src)
    out += synthetic(batch)
    return out


def synthetic(batch: dict) -> list[dict]:
    """A hand-built instance with an infeasible branch.

    No stored tree closes a leaf by infeasibility, so on the real corpus that rule of a checker is
    never exercised. Over GF(2) the checks x0+x1 = s0, x1+x2 = s1, x0+x2+x3 = s2 sum to x3 = s0+s1+s2:
    with s = (1, 0, 0) the branch x3 = 0 has no solution, and on x3 = 1 the reduced check x0+x1 = 1
    bounds every completion by 1, so the candidate {x0, x3} of weight 2 is optimal. With s = (1, 0, 1)
    the same rows XOR to empty with EVEN parity on x3 = 0, so a leaf that claims infeasibility there
    is false and must be refused.
    """
    batch["instances"]["synthetic_x3"] = {"n": 4, "checks": [[0, 1], [1, 2], [0, 2, 3]]}
    batch["weights"]["synthetic_unit"] = [1, 1, 1, 1]
    base = {"instance": "synthetic_x3", "weights": "synthetic_unit"}
    good = dict(base, id="synthetic/infeasible_branch", syndrome_support=[0], candidate=[0, 3],
                tree=["branch", 3, ["infeasible", [0, 1, 2]], ["bound", [[0, [], 1, 1]]]],
                expect="accept", kind="synthetic_receipt")
    not_empty = copy.deepcopy(good)
    not_empty.update(id="planted/infeasible_leaf_support_not_empty/synthetic", expect="refuse",
                     kind="infeasible_leaf_support_not_empty")
    not_empty["tree"][2] = ["infeasible", 0]
    even = dict(base, id="planted/infeasible_leaf_even_parity/synthetic", syndrome_support=[0, 2], candidate=[0],
                tree=["branch", 3, ["infeasible", [0, 1, 2]], ["infeasible", [0, 1, 2]]],
                expect="refuse", kind="infeasible_leaf_even_parity")
    # the lattice boundary, decided exactly: on the x3 = 1 leaf the committed weight is 1 and the
    # facet's multiplier y is the whole bound, so y = 0 lands on U - 1 exactly (refuse) and y = 1/2
    # above it (accept); y = 3 overloads x0 and x1 by 2 each, which charged leaves nothing (refuse)
    # and uncharged would prove the leaf
    def with_multiplier(rid, num, den, expect, kind):
        r = copy.deepcopy(good)
        r.update(id=rid, expect=expect, kind=kind)
        r["tree"][3] = ["bound", [[0, [], num, den]]]
        return r
    boundary = with_multiplier("planted/lattice_boundary_leaf/synthetic", 0, 1, "refuse", "lattice_boundary_leaf")
    above = with_multiplier("synthetic/lattice_boundary_accept", 1, 2, "accept", "lattice_boundary_accept")
    overloaded = with_multiplier("planted/overloaded_multiplier/synthetic", 3, 1, "refuse", "overloaded_multiplier")
    return [good, not_empty, even, boundary, above, overloaded]


def run(js: pathlib.Path = JS, write: bool = True) -> dict:
    if shutil.which("node") is None:
        raise AgreementError("node is not installed")
    batch = load_batch()
    extra = planted(batch)
    wrong_truth = [x["id"] for x in extra if _python_accepts(batch, x) != (x["expect"] == "accept")]
    if wrong_truth:
        raise AgreementError(f"planted inputs the production checker decides the other way: {wrong_truth[:5]}")
    batch["receipts"] += extra
    with tempfile.TemporaryDirectory() as tmp:
        bp = pathlib.Path(tmp) / "batch.json"
        bp.write_text(json.dumps(batch), encoding="utf-8")
        proc = subprocess.run(["node", str(js), str(bp)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=3600)
    verdict = {}
    for ln in proc.stdout.splitlines():
        if ln and not ln.startswith("SUMMARY"):
            parts = ln.split("\t")
            verdict[parts[0]] = (parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")
    missing = [x["id"] for x in batch["receipts"] if x["id"] not in verdict]
    if missing:
        raise AgreementError(f"no verdict for {len(missing)} inputs (node exit {proc.returncode}): {missing[:3]}")
    wrong = []
    for x in batch["receipts"]:
        got, why = verdict[x["id"]]
        want = "ACCEPT" if x["expect"] == "accept" else "REFUSE"
        if got != want or (want == "REFUSE" and REASON[x["kind"]] not in why):
            wrong.append({"id": x["id"], "verdict": got, "reason": why})
    receipts = [x for x in batch["receipts"] if x["kind"] == "receipt"]
    defects = [x for x in extra if x["expect"] == "refuse"]
    synth = [x for x in extra if x["expect"] == "accept"]
    doc = {
        "schema": "oneq-qldpc-tree-js-agreement/1",
        "checker": "verifier-js/verify-qldpc-tree.mjs",
        "checker_sha256": _sha(pathlib.Path(js).read_bytes()),
        "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
        "manifest_sha256": PINNED_MANIFEST_SHA256,
        "trees": len(receipts),
        "trees_by_cohort": dict(Counter(x["id"].split("/")[0] for x in receipts)),
        "trees_accepted": sum(1 for x in receipts if verdict[x["id"]][0] == "ACCEPT"),
        "synthetic_receipts_accepted": sum(1 for x in synth if verdict[x["id"]][0] == "ACCEPT"),
        "planted_defects": len(defects),
        "planted_defects_refused_for_their_reason": sum(
            1 for x in defects if verdict[x["id"]][0] == "REFUSE" and REASON[x["kind"]] in verdict[x["id"]][1]),
        "planted_by_class": dict(Counter(x["kind"] for x in defects)),
        "disagreements": wrong[:50],
        "command": "python tools/qldpc_tree_js_agreement.py",
    }
    if write:
        OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    if wrong:
        raise AgreementError(f"{len(wrong)} disagreements; first: {wrong[:3]}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--js", default=str(JS))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    try:
        doc = run(pathlib.Path(a.js), write=not a.no_write)
    except AgreementError as exc:
        print(f"TREE JS AGREEMENT FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: doc[k] for k in ("trees", "trees_by_cohort", "trees_accepted", "planted_defects",
                                          "planted_defects_refused_for_their_reason", "node")}))
    print("TREE JS AGREEMENT PASSED: every branch-dual tree accepted and every planted defect refused "
          "for its own reason, by a checker in a second language")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
