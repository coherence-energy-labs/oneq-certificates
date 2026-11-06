r"""Write evidence/cert_conformance.json from a run, never by hand.

    python -B tools/write_cert_conformance.py            # runs every implementation on both vector sets, writes
    python -B tools/write_cert_conformance.py --check    # runs them and compares with the stored record; exit 1 if stale

Two vector sets, each decided by every implementation that decides its claim:

  conformance/cert_vectors.json     the EXACT certificate (`oneq.cert_format.recheck`, Python) against
                                    `verifier-js/verify-certificate.mjs --conformance`
  conformance/epsilon_vectors.json  the EPSILON claim: `check_epsilon` and the scaled-integer `verify_b`, both through
                                    the replay's strict loader (`tools/replay_epsilon_receipts.decide_receipt`), against
                                    `verifier-js/verify-epsilon.mjs --batch`

Until 2026-11-03 this record was typed once (2026-08-01) and said the two exact checkers were "written from the spec
in different languages, sharing no code". The repository's history says otherwise: the JavaScript exact checker's
parser, graph binding, T-join and T'-oddness tests were written from the specification (7a5a3cff, 2026-07-31), but its
acceptance step -- the exact lattice rule (888b7454) and then the dual repair with its spread-absorb and multi-start
variants (6b3a146d, e0767dc7, 900d6471) -- was ported from the Python checker and kept identical to it, and the two
shared a misreading of boundary edge keys until 56dce450 (2026-08-18). Agreement on a ported step is not independent
evidence. The record now carries, per implementation, what of it is independent, from those commits; the wording is
in INDEPENDENCE below and nowhere else.

The harness is not dark: the run also decides one deliberately bent copy of each set (a recorded verdict flipped) and
records that every implementation reports the mismatch.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "evidence" / "cert_conformance.json"
CERT_VECTORS = ROOT / "conformance" / "cert_vectors.json"
EPS_VECTORS = ROOT / "conformance" / "epsilon_vectors.json"
JS_EXACT = ROOT / "verifier-js" / "verify-certificate.mjs"
JS_EPS = ROOT / "verifier-js" / "verify-epsilon.mjs"
SUPERSEDES = "evidence/cert_conformance_2026-08-01.json"

INDEPENDENCE = {
    "exact_python": "the reference: src/oneq/matching_cert.py with the document reader src/oneq/cert_format.py",
    "exact_javascript": (
        "PARTLY independent. Written from the specification in JavaScript with node built-ins only (7a5a3cff, "
        "2026-07-31): its parser, graph binding, T-join test and T'-oddness test. NOT independent: its acceptance step, "
        "ported from the Python checker and kept identical to it -- the exact lattice rule (888b7454, 2026-08-04), then "
        "the dual repair and its spread-absorb and multi-start variants, written into both languages in the same "
        "commits (6b3a146d, e0767dc7, 900d6471, 2026-08-05). The two shared a misreading of boundary edge keys until "
        "56dce450 (2026-08-18). Agreement on the ported step is one implementation agreeing with itself."),
    "epsilon_check_epsilon": "the reference: src/oneq/matching_cert_epsilon.check_epsilon (exact rationals)",
    "epsilon_verify_b": (
        "a second verifier in the SAME language, by the same project, in scaled-integer arithmetic "
        "(tools/replay_epsilon_receipts.verify_b). It shares the receipt loader with check_epsilon, so it cannot "
        "expose a misreading in that loader; before 2026-11-03 both Python verifiers shared several "
        "(the pre-fix fuzz record, evidence/epsilon_checker_fuzz.json)."),
    "epsilon_javascript": (
        "an independent READING, not independent authorship. verifier-js/verify-epsilon.mjs was written on "
        "2026-11-03 by a fresh AI agent from paper 1's Secs. II-IV and a brief giving the wire format and the kinds "
        "of malformed input the differential fuzzer would send; the brief's author had read the Python checkers. "
        "The writing agent had file-system access: it was TOLD not to open the Python sources and REPORTED that it "
        "did not; the file list released with the brief (docs/reviews/JS_EPSILON_BRIEF_2026-11-03.md) is the "
        "coordinator's summary of that report, not a log. It is the same project's work."),
}


def _run_node(args, stdin=None):
    node = shutil.which("node")
    if node is None:
        raise SystemExit("node is not installed: the cross-language record cannot be written without it")
    p = subprocess.run([node, *args], input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=3600)
    return p, subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()


def _exact_set(path: pathlib.Path) -> dict:
    """Python's recheck against each recorded verdict, and the JavaScript harness's own count."""
    from oneq.cert_format import graph_from_json, recheck
    doc = json.loads(path.read_text(encoding="utf-8"))
    vecs = doc["vectors"]
    py_wrong = [x["name"] for x in vecs if recheck(x["document"], graph_from_json(x["graph"]))[0]
                != x["expected_accepted"]]
    p, node_v = _run_node([str(JS_EXACT), "--conformance", str(path)])
    m = re.search(r"conformance: (\d+)/(\d+) agree", p.stdout)
    js_agree, js_total = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return {"vectors": len(vecs), "expected_accept": sum(1 for x in vecs if x["expected_accepted"]),
            "expected_refuse": sum(1 for x in vecs if not x["expected_accepted"]),
            "python_agree": len(vecs) - len(py_wrong), "python_disagree": py_wrong,
            "javascript_line": m.group(0) if m else p.stdout.strip()[-200:],
            "javascript_agree": js_agree, "javascript_total": js_total, "javascript_exit": p.returncode,
            "node": node_v}


def _eps_set(vecs: list) -> dict:
    import fuzz_epsilon_checkers as fz

    def want(v):
        e = v["expected"]
        return (e["status"], e["epsilon"], e["lower_bound"])

    got_a, got_b = zip(*(fz.py_replay(v["case"]) for v in vecs)) if vecs else ((), ())
    got_js = fz.js_batch([v["case"] for v in vecs])
    return {"vectors": len(vecs), "classes": len({v["class"] for v in vecs}),
            "by_status": {s: sum(1 for v in vecs if v["expected"]["status"] == s)
                          for s in sorted({v["expected"]["status"] for v in vecs})},
            "check_epsilon_agree": sum(1 for v, g in zip(vecs, got_a) if tuple(g) == want(v)),
            "verify_b_agree": sum(1 for v, g in zip(vecs, got_b) if tuple(g) == want(v)),
            "javascript_agree": sum(1 for v, g in zip(vecs, got_js) if tuple(g) == want(v)),
            "disagreements": [v["name"] for v, a, b, j in zip(vecs, got_a, got_b, got_js)
                              if not (tuple(a) == tuple(b) == tuple(j) == want(v))]}


def _harness_can_fail() -> dict:
    """Flip one recorded verdict in each set; every implementation must report exactly one mismatch more."""
    cert = json.loads(CERT_VECTORS.read_text(encoding="utf-8"))
    bent = copy.deepcopy(cert)
    i = next(k for k, x in enumerate(bent["vectors"]) if x["expected_accepted"])
    bent["vectors"][i]["expected_accepted"] = False
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "bent_cert_vectors.json"
        p.write_text(json.dumps(bent), encoding="utf-8")
        exact = _exact_set(p)
    eps = json.loads(EPS_VECTORS.read_text(encoding="utf-8"))["vectors"]
    j = next(k for k, v in enumerate(eps) if v["expected"]["status"] == "EXACT_OPTIMAL")
    bent_eps = copy.deepcopy(eps)
    bent_eps[j]["expected"]["status"] = "NOT_PROVEN"
    e = _eps_set(bent_eps)
    return {"exact_flipped_vector": cert["vectors"][i]["name"],
            "exact_python_agree": exact["python_agree"], "exact_javascript_line": exact["javascript_line"],
            "exact_javascript_exit": exact["javascript_exit"],
            "epsilon_flipped_vector": eps[j]["name"],
            "epsilon_agree": [e["check_epsilon_agree"], e["verify_b_agree"], e["javascript_agree"]],
            "epsilon_vectors": e["vectors"]}


def build() -> dict:
    exact = _exact_set(CERT_VECTORS)
    eps_doc = json.loads(EPS_VECTORS.read_text(encoding="utf-8"))
    eps = _eps_set(eps_doc["vectors"])
    canfail = _harness_can_fail()
    n, ne = exact["vectors"], eps["vectors"]
    exact_ok = exact["python_agree"] == n and exact["javascript_agree"] == n and exact["javascript_exit"] == 0
    eps_ok = eps["check_epsilon_agree"] == eps["verify_b_agree"] == eps["javascript_agree"] == ne
    fail_ok = (canfail["exact_python_agree"] == n - 1 and canfail["exact_javascript_exit"] != 0
               and canfail["exact_javascript_line"] == f"conformance: {n - 1}/{n} agree"
               and canfail["epsilon_agree"] == [ne - 1] * 3)
    return {
        "schema": "oneq-certificate-conformance-result/2",
        "generated_by": "tools/write_cert_conformance.py",
        "supersedes": SUPERSEDES,
        # the three fields below keep their /1 meaning (the exact-checker pair) for readers of the old schema
        "vectors": n, "expected_accept": exact["expected_accept"], "expected_refuse": exact["expected_refuse"],
        "agree": f"conformance: {exact['javascript_agree']}/{n} agree",
        "javascript_exit": exact["javascript_exit"],
        "all_agree": exact_ok and eps_ok, "harness_can_fail": fail_ok,
        "exact_certificate": {**exact, "vectors_file": "conformance/cert_vectors.json"},
        "epsilon_claim": {**eps, "vectors_file": "conformance/epsilon_vectors.json",
                          "vectors_generated_by": eps_doc.get("generated_by")},
        "harness_bent_copies": canfail,
        "implementations": [
            {"claim": "exact", "lang": "python", "file": "src/oneq/matching_cert.py + src/oneq/cert_format.py",
             "runtime": platform.python_version(), "independence": INDEPENDENCE["exact_python"]},
            {"claim": "exact", "lang": "javascript", "file": "verifier-js/verify-certificate.mjs",
             "runtime": exact["node"], "dependencies": "none (node built-ins only)",
             "independence": INDEPENDENCE["exact_javascript"]},
            {"claim": "epsilon", "lang": "python", "file": "src/oneq/matching_cert_epsilon.py (check_epsilon)",
             "runtime": platform.python_version(), "independence": INDEPENDENCE["epsilon_check_epsilon"]},
            {"claim": "epsilon", "lang": "python", "file": "tools/replay_epsilon_receipts.py (verify_b)",
             "runtime": platform.python_version(), "independence": INDEPENDENCE["epsilon_verify_b"]},
            {"claim": "epsilon", "lang": "javascript", "file": "verifier-js/verify-epsilon.mjs",
             "runtime": exact["node"], "dependencies": "none (node built-ins only)",
             "independence": INDEPENDENCE["epsilon_javascript"]}],
        "what_agreement_means": (
            "every implementation returns the recorded verdict on every vector. Agreement is not correctness: it is "
            "the absence of a misreading that the implementations do NOT share. For the exact claim the JavaScript "
            "checker's acceptance step was ported from the Python one, so on that step agreement is not independent "
            "evidence; its parser, graph binding and T-join tests are. For the epsilon claim the two Python verifiers "
            "share a loader and a language, and the JavaScript one is an independent reading by the same project."),
        "divergence_this_found": (
            "exact (2026-07-31): Python deserialised a repeated dual member set into a dict (last value wins) while "
            "the JavaScript summed the array -- objective 2.0 and ACCEPT against 2.4 and refuse on identical bytes; "
            "both now refuse and the case is a vector. epsilon (2026-11-03): 20 classes of malformed input split the "
            "implementations, every one rooted in the Python readers (evidence/epsilon_checker_fuzz.json); fixed, "
            "every class is a vector (evidence/epsilon_checker_fuzz_after_fix.json)."),
        "why_mostly_refusals": ("a suite where everything verifies would not notice a checker that accepts "
                                "everything -- the failure that matters, since such a checker is worthless and looks "
                                "perfect"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare a fresh run with the stored record")
    a = ap.parse_args(argv)
    rec = build()
    if a.check:
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        same = {k: v for k, v in old.items() if k != "implementations"} == \
               {k: v for k, v in rec.items() if k != "implementations"}
        print("cert_conformance.json is", "current" if same else "STALE")
        return 0 if same and rec["all_agree"] and rec["harness_can_fail"] else 1
    if not (ROOT / SUPERSEDES).exists():
        raise SystemExit(f"{SUPERSEDES} (the 2026-08-01 record) must be kept before the record is rewritten")
    OUT.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(rec["agree"], "| epsilon:", rec["epsilon_claim"]["javascript_agree"], "/", rec["epsilon_claim"]["vectors"],
          "| all_agree", rec["all_agree"], "| harness_can_fail", rec["harness_can_fail"])
    return 0 if rec["all_agree"] and rec["harness_can_fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
