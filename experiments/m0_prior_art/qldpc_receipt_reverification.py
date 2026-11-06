r"""Every stored paper-2 receipt re-verified by the checkers of the input-domain contract (2026-11-03).

    python experiments/m0_prior_art/qldpc_receipt_reverification.py

WHY. The input-domain contract (docs/QLDPC_RECEIPT_INPUT_DOMAIN.md; external audit 2 of release v2026.11.03, findings
F05-F07) changed three checkers: src/oneq/qldpc_check.py, tools/external/tree_checker_b.py and both JavaScript
checkers. Three files pinned the old bytes -- evidence/qldpc_heldout_addendum.json (tree_checker_b_sha256),
evidence/qldpc_tree_corpus.json (checker_b_sha256) and docs/evidence_annotations.json (qldpc_check.py) -- and a pin
moves only with a record that the corpus it vouches for still verifies under the new bytes. This is that record.

WHAT IT RUNS, and refuses to write unless every part passes:
  * tools/replay_receipts.replay against the pinned manifest: all 2,042 receipts, each flat receipt by the
    production exact checker AND the reference checker, each tree by the production tree checker AND tree checker B;
  * the two JavaScript agreement artifacts must name the current JavaScript bytes and the current manifest pin, and
    report every stored receipt accepted and every planted defect refused (re-run tools/qldpc_js_agreement.py and
    tools/qldpc_tree_js_agreement.py first);
  * the shared input-domain vectors must still be the ones the contract cites.
It records the sha256 of every checker, of the replay tool, of the manifest and of this script.

NOT COVERED, and said so in the record: the 20 paired-stress trees of evidence/qldpc_tree_corpus.json are not stored
(evidence/qldpc_paired_bd_closure.json keeps their syndromes and candidates, not the trees), so they are not
re-verified here; their checker-B counts in the corpus file were measured with the previous checker.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "evidence" / "qldpc_receipt_reverification_2026-11-03.json"   # the first re-pin's record
CHECKERS = ["src/oneq/qldpc_check.py", "tools/external/exact_lp_certificate_reference.py",
            "tools/external/tree_checker_b.py", "verifier-js/verify-qldpc-flat.mjs",
            "verifier-js/verify-qldpc-tree.mjs", "tools/replay_receipts.py"]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _trees_duals(t):
    if t[0] == "branch":
        yield from _trees_duals(t[2])
        yield from _trees_duals(t[3])
    elif t[0] == "bound":
        yield from t[1]


def _zero_multiplier_census() -> dict:
    """Facets with multiplier 0, per stored corpus (they are now validated like every other facet)."""
    import gzip
    ev = ROOT / "evidence"
    out: dict = {}

    def note(name, duals):
        c = out.setdefault(name, {"receipts": 0, "zero_multiplier_facets": 0, "receipts_with_one": 0})
        z = sum(1 for d in duals if int(d[2]) == 0)
        c["receipts"] += 1
        c["zero_multiplier_facets"] += z
        c["receipts_with_one"] += z > 0
    for c in json.loads((ev / "qldpc_heldout_flat_certs.json").read_text(encoding="utf-8"))["certificates"]:
        note("heldout_flat", c["facet_duals"])
    add = json.loads((ev / "qldpc_heldout_addendum.json").read_text(encoding="utf-8"))
    for k in ("bd_trees", "dev_bd_trees"):
        for t in add[k]:
            note(k, list(_trees_duals(t["tree"])))
    for t in json.loads((ev / "qldpc_dev_refsched_trees.json").read_text(encoding="utf-8"))["trees"]:
        note("dev_refsched_trees", list(_trees_duals(t["tree"])))
    for p in sorted((ev / "strong_baseline_v3" / "proofs").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        note(f"strong_baseline_{d['kind']}",
             d["facet_duals"] if d["kind"] == "flat" else list(_trees_duals(d["tree"])))
    for p in sorted((ev / "qldpc_scaling_prereg").glob("*.jsonl.gz")):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            R = json.loads(line)["receipt"]
            if R is not None:
                note(f"prereg_{R['kind']}", R["facet_duals"] if R["kind"] == "flat" else list(_trees_duals(R["tree"])))
    return out


def _reverify_prereg() -> dict:
    """Every receipt of the preregistered campaign, re-decided by both of its checkers under the current bytes."""
    import gzip
    import importlib.util
    import io
    from contextlib import redirect_stdout
    from fractions import Fraction
    from oneq.qldpc_check import QldpcCertificate, check_qldpc_bnb, check_qldpc_exact
    spec = importlib.util.spec_from_file_location(
        "ref_reverify", ROOT / "tools" / "external" / "exact_lp_certificate_reference.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_reverify"] = ref
    spec.loader.exec_module(ref)
    spec_b = importlib.util.spec_from_file_location("tree_b_reverify", ROOT / "tools" / "external" / "tree_checker_b.py")
    run = importlib.util.spec_from_file_location(
        "prereg_runner_reverify", ROOT / "experiments" / "m0_prior_art" / "qldpc_scaling_prereg_campaign.py")
    runner = importlib.util.module_from_spec(run)
    run.loader.exec_module(runner)
    out = {}
    import tempfile
    for name in runner.DATASETS:
        inst_name = runner.DATASETS[name][0]
        inst = json.loads((ROOT / "evidence" / "instances" / f"{inst_name}.json").read_text(encoding="utf-8"))
        checks = [tuple(c) for c in inst["checks"]]
        n = inst["n_variables"]
        D = runner.build(name, runner.SEED_DOMAIN)          # the integer weights, exactly as the campaign built them
        wint = [int(D["wint"][i]) for i in range(n)]
        wx = {i: Fraction(v) for i, v in enumerate(wint)}
        dense = None
        c = {"receipts": 0, "both_checkers_accept": 0}
        for line in gzip.open(ROOT / "evidence" / "qldpc_scaling_prereg" / f"{name}.jsonl.gz", "rt", encoding="utf-8"):
            r = json.loads(line)
            R = r["receipt"]
            if R is None:
                continue
            c["receipts"] += 1
            s = [0] * len(checks)
            for j in r["syndrome_support"]:
                s[j] = 1
            if R["kind"] == "flat":
                fd = tuple((tuple(a) if isinstance(a, list) else a, tuple(F), Fraction(p, q)) for (a, F, p, q)
                           in R["facet_duals"])
                ok_a = check_qldpc_exact(checks=checks, weights=wx, syndrome=s,
                                         cert=QldpcCertificate(error_support=tuple(R["answer"]),
                                                               facet_duals=fd)).accepted
                if dense is None:
                    dense = [[0] * n for _ in checks]
                    for j, cc in enumerate(checks):
                        for i in cc:
                            dense[j][i] = 1
                e = [0] * n
                for i in R["answer"]:
                    e[i] = 1
                doc = {"H": dense, "s": s, "weights": [{"num": v, "den": 1} for v in wint], "candidate": e,
                       "facets": [{"rows": sorted(a) if isinstance(a, list) else [a], "F": sorted(F),
                                   "y": {"num": p, "den": q}} for (a, F, p, q) in R["facet_duals"]]}
                try:
                    ok_b = bool(ref.check_certificate(doc)["objective_optimal"])
                except ref.CertificateError:
                    ok_b = False
            else:
                def deser(t):
                    if t[0] == "branch":
                        return ("branch", t[1], deser(t[2]), deser(t[3]))
                    if t[0] == "bound":
                        return ("bound", tuple((tuple(a) if isinstance(a, list) else a, tuple(F), Fraction(p, q))
                                               for (a, F, p, q) in t[1]))
                    return ("infeasible", tuple(t[1]) if isinstance(t[1], list) else t[1])
                ok_a = check_qldpc_bnb(checks=checks, weights=wx, syndrome=s, error_support=tuple(R["answer"]),
                                       tree=deser(R["tree"])).accepted
                with tempfile.TemporaryDirectory() as td:
                    pth = pathlib.Path(td) / "doc.json"
                    pth.write_text(json.dumps({"checks": [list(x) for x in checks], "weights": wint, "syndrome": s,
                                               "candidate": R["answer"], "tree": R["tree"]}), encoding="utf-8")
                    mod = importlib.util.module_from_spec(spec_b)
                    spec_b.loader.exec_module(mod)
                    try:
                        with redirect_stdout(io.StringIO()):
                            code = mod.main(str(pth))
                    except SystemExit as exc:
                        code = exc.code
                ok_b = code == 0
            c["both_checkers_accept"] += ok_a and ok_b
        if c["both_checkers_accept"] != c["receipts"]:
            raise SystemExit(f"prereg {name}: a stored receipt is refused under the current checkers: {c}")
        out[name] = c
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="one record per re-pin; never overwrite an earlier one")
    ap.add_argument("--why", default=None, help="what changed the checkers this time")
    a = ap.parse_args()
    out_path = pathlib.Path(a.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    import replay_receipts as R
    manifest = ROOT / "evidence" / "receipt_manifest.json"
    ok, fails, passed = R.replay(manifest, pinned=R.PINNED_MANIFEST_SHA256, subset=None)
    if not ok:
        raise SystemExit(f"replay failed, nothing written: {fails[:5]}")
    js = {}
    for name, key_stored, key_planted, key_refused in (
            ("qldpc_js_agreement.json", "flat_receipts_accepted", "planted_defects",
             "planted_defects_refused_for_their_reason"),
            ("qldpc_tree_js_agreement.json", "trees_accepted", "planted_defects",
             "planted_defects_refused_for_their_reason")):
        d = json.loads((ROOT / "evidence" / name).read_text(encoding="utf-8"))
        cur = sha(ROOT / d["checker"])
        if d["checker_sha256"] != cur:
            raise SystemExit(f"{name} names {d['checker_sha256']}, the checker is {cur}: re-run its harness")
        if d.get("manifest_sha256", R.PINNED_MANIFEST_SHA256) != R.PINNED_MANIFEST_SHA256:
            raise SystemExit(f"{name} was run against another manifest: re-run its harness")
        if d.get("disagreements") or d[key_planted] != d[key_refused]:
            raise SystemExit(f"{name}: a disagreement or an accepted planted defect")
        js[name] = {"checker": d["checker"], "checker_sha256": cur, "stored_accepted": d[key_stored],
                    "planted_defects_refused": d[key_refused]}
    # ZERO MULTIPLIERS (final referee, 2026-11-03): the checkers used to skip a facet whose multiplier is 0 before
    # validating it. Stored receipts DO list such facets, so the corpora that hold them are re-checked here too: the
    # strong baseline's 955 proofs (by its own replay tool) and every receipt of the preregistered campaign.
    import subprocess
    sb = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "replay_strong_baseline.py")], capture_output=True,
                        text=True, timeout=3600, cwd=ROOT)
    if sb.returncode != 0 or "STRONG-BASELINE REPLAY PASSED" not in sb.stdout:
        raise SystemExit(f"strong-baseline replay failed: {sb.stdout[-400:]} {sb.stderr[-400:]}")
    prereg = _reverify_prereg()
    zero = _zero_multiplier_census()
    vec = json.loads((ROOT / "conformance" / "qldpc_input_domain_vectors.json").read_text(encoding="utf-8"))
    if vec["contract"] != "docs/QLDPC_RECEIPT_INPUT_DOMAIN.md":
        raise SystemExit("the vector file no longer names the contract")
    me = pathlib.Path(__file__).resolve()
    out = {"schema": "oneq-qldpc-receipt-reverification/1", "date": "2026-11-03",
           "why": a.why or ("the input-domain contract changed the qLDPC checkers; the pins in "
                            "evidence/qldpc_heldout_addendum.json, evidence/qldpc_tree_corpus.json and "
                            "docs/evidence_annotations.json move to the new bytes on this record, with their "
                            "previous values kept beside them"),
           "replay": {"manifest": "evidence/receipt_manifest.json", "manifest_sha256": sha(manifest),
                      "pinned_in_replay_tool": R.PINNED_MANIFEST_SHA256, "complete": True,
                      "receipts_passed_by_cohort": passed, "receipts_passed": sum(passed.values()),
                      "each_checked_by": "flat: check_qldpc_exact and the reference checker; tree: check_qldpc_bnb "
                                         "and tree checker B"},
           "javascript_third_checkers": js,
           "strong_baseline_replay": sb.stdout.strip().splitlines()[-1],
           "prereg_campaign_receipts": prereg,
           "zero_multiplier_facets_in_stored_receipts": zero,
           "input_domain_vectors": {"file": "conformance/qldpc_input_domain_vectors.json",
                                    "vectors": len(vec["vectors"]),
                                    "sha256": sha(ROOT / "conformance" / "qldpc_input_domain_vectors.json"),
                                    "run_by": "tests/test_qldpc_input_domain.py"},
           "not_covered": ("the 20 paired-stress trees of evidence/qldpc_tree_corpus.json: not stored (only their "
                           "syndromes and candidates, evidence/qldpc_paired_bd_closure.json), so not re-verified here; "
                           "their checker-B counts in that file were measured with the previous checker"),
           "code_sha256": {me.relative_to(ROOT).as_posix(): sha(me), **{c: sha(ROOT / c) for c in CHECKERS}}}
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"REVERIFIED: {out['replay']['receipts_passed']} receipts by two checkers each, {passed}; "
          f"JavaScript: {', '.join(f'{k} {v['stored_accepted']}' for k, v in js.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
