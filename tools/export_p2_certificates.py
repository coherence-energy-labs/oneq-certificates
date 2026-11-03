#!/usr/bin/env python3
"""Regenerate every exact P2 certificate and write it for the Lean checker.

WHY. P2's 1,851 exact certificates were checked by `check_qldpc_exact` and by an
independently written reference checker -- both Python. `lean/OneqCert` now carries a
checker whose soundness is a kernel-checked theorem (`OneqCert.Cert.accept_sound`).
Running it on the paper's certificates needs the certificates; the per-shot records
keep only their hashes. This regenerates them.

HOW, WITHOUT TOUCHING THE PAPER'S EVIDENCE. The producer is
`experiments/m0_prior_art/qldpc_exact_all_gate.py`. It is imported, not copied, and run
with its output root redirected to a temporary directory. The reference checker it
already calls on every accepted certificate is wrapped to capture that document before
delegating, so the certificates exported are exactly the ones the reference checked.

IDENTITY, CHECKED RATHER THAN ASSUMED. A regenerated certificate is the paper's only if
the run reproduced the paper. So every regenerated per-shot record must equal the
committed one in every field (instance, candidate, tier, verdict, answer, objective,
retrospective outcome) except the certificate hash, and each captured document must
agree with the record it pairs with: same answer, same objective. Anything else
refuses the export.

The one tolerated difference is MEASURED, not assumed: on 2026-09-23 the first rerun
reproduced uniform_p01 byte-for-byte except one record whose certificate hash moved
while its answer, objective and verdict did not -- the producer found a different dual
witness for the same claim. Such drift is listed shot by shot in the export summary.
It is not fatal, because the claim a certificate proves is "this answer is minimum
weight for this syndrome", and the exported (regenerated) witness is the one the Lean
checker then judges.

Output: one JSON line per certificate, in the format `oneqcert-check` reads.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import pathlib
import shutil
import sys
import tempfile
from fractions import Fraction
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
GATE = ROOT / "experiments" / "m0_prior_art" / "qldpc_exact_all_gate.py"
SUMMARY = ROOT / "evidence" / "p2_certificates_export.json"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathrel import rel  # noqa: E402
RUNGS = ("uniform_p01", "uniform_p02", "hetero_p02", "phenom_p01", "phenom_p02",
         "stress_p05", "xsector_p02")


def lean_line(cid: str, doc: dict) -> dict:
    """The reference checker's document, in the Lean checker's sparse form."""
    H = doc["H"]
    n = len(doc["weights"])
    ws = [Fraction(int(w["num"]), int(w["den"])) for w in doc["weights"]]
    D = 1
    for w in ws:
        D = D * w.denominator // math.gcd(D, w.denominator)
    return {
        "id": cid,
        "checks": [[i for i, b in enumerate(row) if b] for row in H],
        "syn": [int(b) & 1 for b in doc["s"]],
        "w": [[w.numerator, w.denominator] for w in ws],
        "e": [i for i in range(n) if doc["candidate"][i]],
        "facets": [{"rows": [int(r) for r in f["rows"]],
                    "F": [int(i) for i in f["F"]],
                    "y": [int(f["y"]["num"]), int(f["y"]["den"])]}
                   for f in doc["facets"]],
        "D": D,
    }


def regenerate(tmp: pathlib.Path) -> dict[str, list[dict]]:
    spec = importlib.util.spec_from_file_location("qldpc_exact_all_gate", GATE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the producer at {GATE}")
    gate: Any = importlib.util.module_from_spec(spec)
    sys.modules["qldpc_exact_all_gate"] = gate
    spec.loader.exec_module(gate)

    captured: dict[str, list[dict]] = {}
    current = {"rung": None}
    real_ref = gate._load_reference()

    class Capture:
        def check_certificate(self, doc):
            captured.setdefault(current["rung"], []).append(doc)
            return real_ref.check_certificate(doc)

    real_run_rung = gate.run_rung

    def run_rung(**kw):
        current["rung"] = kw["name"]
        return real_run_rung(**kw)

    (tmp / "evidence" / "per_shot").mkdir(parents=True)
    (tmp / "evidence" / "weights").mkdir(parents=True)
    gate.ROOT = tmp
    gate._load_reference = lambda: Capture()
    gate.run_rung = run_rung
    rc = gate.main()
    if rc not in (0, None):
        raise SystemExit(f"the producer exited {rc}")
    return captured


def _without_cert_hash(rec: dict) -> dict:
    r = json.loads(json.dumps(rec))
    r.get("receipt", {}).pop("certificate_hash", None)
    return r


def verify_identity(tmp: pathlib.Path, captured: dict[str, list[dict]],
                    drift: list[str]) -> list[str]:
    """Every regenerated record must equal the committed one in every field -- the
    instance, BP-OSD's candidate, the tier, the verdict, the answer and its objective,
    the retrospective logical outcome -- EXCEPT the certificate hash. A different hash
    with everything else equal is a different dual witness for the SAME claim (this
    answer is minimum weight for this syndrome), and any valid witness proves that
    claim. Hash drift is therefore counted and reported in `drift`, never hidden and
    never fatal; any other difference refuses the export."""
    bad = []
    for rung in RUNGS:
        mine = tmp / "evidence" / "per_shot" / f"{rung}.jsonl"
        theirs = ROOT / "evidence" / "per_shot" / f"{rung}.jsonl"
        recs = [json.loads(ln) for ln in theirs.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        regen = [json.loads(ln) for ln in mine.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        if len(regen) != len(recs):
            bad.append(f"{rung}: {len(regen)} regenerated records, {len(recs)} committed")
            continue
        for a, b in zip(regen, recs):
            if _without_cert_hash(a) != _without_cert_hash(b):
                bad.append(f"{rung} shot {b['shot_index']}: the regenerated record "
                           f"differs beyond its certificate hash -- not the paper's run")
            elif a.get("receipt", {}).get("certificate_hash") != \
                    b.get("receipt", {}).get("certificate_hash"):
                drift.append(f"{rung}:{b['shot_index']}")
        certified = [r for r in recs
                     if r.get("receipt", {}).get("exact_checker_verdict")
                     == "EXACT_CERTIFIED"]
        docs = captured.get(rung, [])
        if len(docs) != len(certified):
            bad.append(f"{rung}: {len(docs)} certificates captured, "
                       f"{len(certified)} records say EXACT_CERTIFIED")
            continue
        for r, doc in zip(certified, docs):
            ans = [i for i, b in enumerate(doc["candidate"]) if b]
            if sorted(r["receipt"]["answer"]) != ans:
                bad.append(f"{rung} shot {r['shot_index']}: captured answer {ans} "
                           f"!= record {r['receipt']['answer']}")
            obj = sum(Fraction(int(doc["weights"][i]["num"]),
                               int(doc["weights"][i]["den"])) for i in ans)
            if obj != r["receipt"]["answer_objective"]:
                bad.append(f"{rung} shot {r['shot_index']}: captured objective {obj} "
                           f"!= record {r['receipt']['answer_objective']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "evidence" / "p2_certificates.jsonl.gz")
    ap.add_argument("--cache", type=pathlib.Path, default=None,
                    help="directory to persist the regeneration in (reused if present)")
    a = ap.parse_args()
    drift: list[str] = []
    cache = a.cache
    if cache is not None and cache.is_dir():
        # a previous run's regeneration, reused: the identity check below re-runs
        # against it in full, so reuse saves time and trusts nothing
        captured = json.loads((cache / "captured.json").read_text(encoding="utf-8"))
        bad = verify_identity(cache, captured, drift)
    else:
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            captured = regenerate(tmp)
            if cache is not None:
                # persist BEFORE verifying: a regeneration costs ~20 minutes, and a
                # failure after it must not throw it away
                shutil.copytree(tmp / "evidence", cache / "evidence")
                (cache / "captured.json").write_text(json.dumps(captured),
                                                     encoding="utf-8")
            bad = verify_identity(tmp, captured, drift)
    if bad:
        print("EXPORT REFUSED")
        for b in bad[:30]:
            print("  " + b)
        return 1
    lines = []
    for rung in RUNGS:
        certified = [json.loads(ln) for ln in
                     (ROOT / "evidence" / "per_shot" / f"{rung}.jsonl")
                     .read_text(encoding="utf-8").splitlines() if ln.strip()]
        certified = [r for r in certified
                     if r.get("receipt", {}).get("exact_checker_verdict")
                     == "EXACT_CERTIFIED"]
        for r, doc in zip(certified, captured[rung]):
            lines.append(json.dumps(lean_line(f"{rung}:{r['shot_index']}", doc),
                                    separators=(",", ":")))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    # mtime=0: the archive's bytes are a function of its content alone
    with open(a.out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                 filename="") as gz:
        gz.write(payload)
    summary: dict[str, Any] = {
        "schema": "oneq.p2.certificate-export.v1",
        "certificates": len(lines),
        "per_rung": {r: len(captured.get(r, [])) for r in RUNGS},
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "records_identical_except_certificate_hash": True,
        "certificate_hash_drift": {
            "count": len(drift),
            "shots": drift,
            "meaning": "the rerun produced a DIFFERENT dual witness for the same "
                       "instance, answer and objective; the exported certificate is "
                       "the regenerated one, and it is what the Lean checker judges",
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8",
                       newline="\n")
    print(f"exported {len(lines)} certificates -> {rel(a.out, ROOT)} "
          f"(content sha256 {summary['content_sha256'][:16]}, "
          f"{len(payload)} bytes uncompressed); certificate-hash drift on "
          f"{len(drift)} shot(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
