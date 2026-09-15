r"""Complete, manifest-pinned replay of every paper-2 receipt.

External audit 2026-09-14 (R-01, R-03). A complete replay must establish that
EVERY receipt behind the published result was present AND passed, against the
instance the result is about. So this command:

  1. refuses a manifest whose sha256 is not the pinned one (a manifest shipped
     beside the data cannot vouch for itself);
  2. verifies the sha256 of every bound file -- persisted instances, integer
     weight vectors, per-shot records, receipt payloads;
  3. requires the receipts found in the payload files to be EXACTLY the
     manifest's: any missing, duplicated or unlisted receipt fails;
  4. re-derives each receipt's syndrome, candidate and proof hash, and its
     agreement with the per-shot record the manifest binds it to;
  5. checks every flat receipt with the production exact checker AND the
     pinned independently written reference checker, and every tree with the
     production tree checker AND the separately written tree checker B;
  6. reports COMPLETE only when every expected cohort count is met exactly.

It reads stored instances and weights; it rebuilds nothing, and imports no
producer or experiment code -- only the checkers. `--subset COHORT` checks one
cohort and says SUBSET, never COMPLETE.

    python tools/replay_receipts.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The manifest this release's claims rest on. Changing the corpus means
#: rebuilding the manifest AND changing this pin in the same, reviewed commit.
PINNED_MANIFEST_SHA256 = "23e336ca38bb403f4c4238e98ded481e9febd80fa037156cdd866774e5147aa4"
REFERENCE_CHECKER = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
TREE_CHECKER_B = ROOT / "tools" / "external" / "tree_checker_b.py"


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _deser_tree(node):
    if node[0] == "branch":
        return ("branch", int(node[1]), _deser_tree(node[2]), _deser_tree(node[3]))
    if node[0] == "bound":
        return ("bound", tuple((tuple(j) if isinstance(j, list) else int(j), tuple(F),
                                Fraction(int(n), int(d))) for (j, F, n, d) in node[1]))
    return ("infeasible", tuple(node[1]) if isinstance(node[1], list) else int(node[1]))


def _reference_doc(checks, n, syn, wvec, cand, facets):
    H = [[0] * n for _ in checks]
    for j, sup in enumerate(checks):
        for i in sup:
            H[j][i] = 1
    e = [0] * n
    for i in cand:
        e[i] = 1
    return {"H": H, "s": syn, "weights": [{"num": w, "den": 1} for w in wvec], "candidate": e,
            "facets": [{"rows": sorted(r) if isinstance(r, list) else [int(r)], "F": sorted(F),
                        "y": {"num": int(num), "den": int(den)}} for (r, F, num, den) in facets]}


def replay(manifest_path: pathlib.Path, *, pinned: str | None, subset: str | None) -> tuple[bool, list[str], dict]:
    fails: list[str] = []
    body = manifest_path.read_bytes()
    got = hashlib.sha256(body).hexdigest()
    if pinned is not None and got != pinned:
        return False, [f"manifest sha256 {got} is not the pinned {pinned}"], {}
    man = json.loads(body)

    for f, h in man["files"].items():
        p = ROOT / f
        if not p.exists():
            fails.append(f"bound file missing: {f}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != h:
            fails.append(f"bound file altered: {f}")
    if fails:
        return False, fails, {}

    from oneq.qldpc_check import QldpcCertificate, check_qldpc_bnb, check_qldpc_exact
    spec = importlib.util.spec_from_file_location("ref_checker", REFERENCE_CHECKER)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_checker"] = ref          # its dataclasses resolve their module by name
    spec.loader.exec_module(ref)

    instances = {k: json.loads((ROOT / v["file"]).read_text(encoding="utf-8"))
                 for k, v in man["instances"].items()}
    weight_cache: dict[str, list[int]] = {}
    record_cache: dict[str, dict[int, dict]] = {}

    def weights(f):
        if f not in weight_cache:
            weight_cache[f] = [int(x) for x in json.loads((ROOT / f).read_text(encoding="utf-8"))["vector"]]
        return weight_cache[f]

    def records(f):
        if f not in record_cache:
            record_cache[f] = {}
            for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r["shot_index"] in record_cache[f]:
                    fails.append(f"duplicate record {f}#{r['shot_index']}")
                record_cache[f][r["shot_index"]] = r
        return record_cache[f]

    # --- gather what the payload files actually contain
    found: dict[str, tuple] = {}

    def put(rid, payload):
        if rid in found:
            fails.append(f"duplicate receipt in payload: {rid}")
        found[rid] = payload

    flat = json.loads((ROOT / "evidence" / "qldpc_heldout_flat_certs.json").read_text(encoding="utf-8"))
    for c in flat["certificates"]:
        put(f"heldout_flat/{c['rung']}/{c['shot_index']}", ("flat", c["syndrome_support"], c["candidate"], c["facet_duals"]))
    add = json.loads((ROOT / "evidence" / "qldpc_heldout_addendum.json").read_text(encoding="utf-8"))
    for t in add["bd_trees"]:
        put(f"heldout_tree/{t['rung']}/{t['shot_index']}", ("tree", t["syndrome_support"], t["candidate"], t["tree"]))
    for t in add["dev_bd_trees"]:
        put(f"dev_tree/{t['rung']}/{t['shot_index']}", ("tree", t["syndrome_support"], t["candidate"], t["tree"]))
    for t in json.loads((ROOT / "evidence" / "qldpc_dev_refsched_trees.json").read_text(encoding="utf-8"))["trees"]:
        put(f"dev_refsched_tree/{t['rung']}/{t['shot_index']}", ("tree", t["syndrome_support"], t["candidate"], t["tree"]))

    listed = {r["id"]: r for r in man["receipts"]}
    for rid in sorted(set(found) - set(listed)):
        fails.append(f"receipt present but not in the manifest: {rid}")
    for rid in sorted(set(listed) - set(found)):
        fails.append(f"receipt in the manifest but missing from the payload: {rid}")

    passed: dict[str, int] = {k: 0 for k in man["expected_counts"]}
    for rid, m in listed.items():
        if subset and m["cohort"] != subset:
            continue
        if rid not in found:
            continue
        kind, syn_sup, cand, proof = found[rid]
        if _sha_json(sorted(int(i) for i in syn_sup)) != m["syndrome_sha256"]:
            fails.append(f"{rid}: syndrome hash mismatch")
            continue
        if _sha_json(sorted(int(i) for i in cand)) != m["candidate_sha256"]:
            fails.append(f"{rid}: candidate hash mismatch")
            continue
        if _sha_json(proof) != m["proof_sha256"]:
            fails.append(f"{rid}: proof hash mismatch")
            continue
        rec = records(m["record_file"]).get(m["shot_index"])
        if rec is None:
            fails.append(f"{rid}: bound record missing")
            continue
        ans = (rec.get("receipt") or {}).get("answer") or (rec.get("fallback") or {}).get("answer")
        if ans is None or sorted(int(i) for i in ans) != sorted(int(i) for i in cand):
            fails.append(f"{rid}: candidate disagrees with its per-shot record")
            continue
        if "syndrome" in m["record_binds"]:
            if sorted(rec["instance"]["syndrome_support"]) != sorted(syn_sup):
                fails.append(f"{rid}: syndrome disagrees with its per-shot record")
                continue
            if rec["instance"]["weights_sha256"] != m["weights_vector_sha256"]:
                fails.append(f"{rid}: record weights hash disagrees")
                continue

        inst = instances[m["instance"]]
        checks = [tuple(c) for c in inst["checks"]]
        n = inst["n_variables"]
        wvec = weights(m["weights_file"])
        if len(wvec) != n or hashlib.sha256(json.dumps(wvec).encode()).hexdigest() != m["weights_vector_sha256"]:
            fails.append(f"{rid}: weight vector does not match instance or hash")
            continue
        syn = [0] * len(checks)
        for j in syn_sup:
            syn[int(j)] = 1
        wx = {i: Fraction(w) for i, w in enumerate(wvec)}
        cand_t = tuple(int(i) for i in cand)

        if kind == "flat":
            fd = [(tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(a), int(b)))
                  for (r, F, a, b) in proof]
            a = check_qldpc_exact(checks=checks, weights=wx, syndrome=syn,
                                  cert=QldpcCertificate(error_support=cand_t, facet_duals=fd, box_duals={})).accepted
            b = bool(ref.check_certificate(_reference_doc(checks, n, syn, wvec, cand_t, proof))["objective_optimal"])
        else:
            a = check_qldpc_bnb(checks=checks, weights=wx, syndrome=syn, error_support=cand_t,
                                tree=_deser_tree(proof)).accepted
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump({"checks": [list(c) for c in checks], "weights": wvec, "syndrome": syn,
                           "candidate": list(cand_t), "tree": proof}, tf)
            b = subprocess.run([sys.executable, str(TREE_CHECKER_B), tf.name],
                               capture_output=True, text=True, timeout=600).returncode == 0
            pathlib.Path(tf.name).unlink(missing_ok=True)
        if a and b:
            passed[m["cohort"]] += 1
        else:
            fails.append(f"{rid}: production={a} independent={b}")

    for cohort, want in man["expected_counts"].items():
        if subset and cohort != subset:
            continue
        if passed[cohort] != want:
            fails.append(f"cohort {cohort}: {passed[cohort]} passed, {want} expected")
    return not fails, fails, passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "evidence" / "receipt_manifest.json"))
    ap.add_argument("--subset", default=None, help="check one cohort; reports SUBSET, never COMPLETE")
    a = ap.parse_args()
    ok, fails, passed = replay(pathlib.Path(a.manifest), pinned=PINNED_MANIFEST_SHA256, subset=a.subset)
    for f in fails[:40]:
        print("  FAIL", f)
    if len(fails) > 40:
        print(f"  ... and {len(fails) - 40} more")
    label = f"SUBSET ({a.subset})" if a.subset else "COMPLETE"
    if ok:
        total = sum(passed.values())
        print(f"{label} REPLAY PASSED: {total} receipts, each checked by two independently "
              f"written checkers against the pinned instance -- {passed}")
        return 0
    print(f"{label} REPLAY FAILED ({len(fails)} problem(s))")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
