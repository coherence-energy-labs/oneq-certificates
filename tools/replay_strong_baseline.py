#!/usr/bin/env python3
r"""Rebuild the strong-baseline tables from records and proofs alone -- no producer, no solver, no BP-OSD.

External audit 2026-09-15, finding 3: "a count of trees is not the trees themselves". This
command reads evidence/strong_baseline_v3/MANIFEST.json, verifies every file it pins, and then,
for every decision the experiment recorded:

  * re-checks the named proof against the persisted instance and integer weights with the
    production checkers (check_qldpc_exact for a flat certificate, check_qldpc_bnb for a tree),
    for the candidate that decision actually used -- a flat certificate is the shot's LP dual
    and certifies any candidate of the optimal weight, so the reference and every tier-1
    answer on a shot are each validated against it separately;
  * recomputes the retrospective logical outcome of every answer from the stored sampled
    error, using the persisted X checks for the frame test;
  * recomputes every number the summary artifact carries -- reference tiers, tier counts,
    optimum-hit counts, logical errors, the paired cells, McNemar -- and refuses if any
    disagrees with evidence/qldpc_strong_baseline_v3.json, which is what the paper typesets.

    python tools/replay_strong_baseline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "evidence" / "strong_baseline_v3" / "MANIFEST.json"
SUMMARY = ROOT / "evidence" / "qldpc_strong_baseline_v3.json"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _deser_tree(node):
    if node[0] == "branch":
        return ("branch", int(node[1]), _deser_tree(node[2]), _deser_tree(node[3]))
    if node[0] == "bound":
        return ("bound", tuple((tuple(j) if isinstance(j, list) else int(j), tuple(F), Fraction(int(a), int(b)))
                               for (j, F, a, b) in node[1]))
    return ("infeasible", tuple(node[1]) if isinstance(node[1], list) else int(node[1]))


def gf2_row_reduce(M):
    M = (np.array(M, dtype=np.uint8) % 2).copy()
    piv, r = [], 0
    for c in range(M.shape[1]):
        rows = np.nonzero(M[r:, c])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        M[[r, p]] = M[[p, r]]
        for q in np.nonzero(M[:, c])[0]:
            if q != r:
                M[q] ^= M[r]
        piv.append(c)
        r += 1
        if r == M.shape[0]:
            break
    return M[:r], piv


def in_rowspace(basis, piv, v):
    v = v.copy()
    for row, c in zip(basis, piv):
        if v[c]:
            v ^= row
    return not v.any()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    a = ap.parse_args()
    manifest_path = pathlib.Path(a.manifest).resolve()
    base = manifest_path.parent                      # records and proofs live beside the manifest they are bound by
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    fails: list[str] = []
    for rel, h in man["files"].items():
        p = ROOT / rel
        if not p.exists():
            fails.append(f"missing {rel}")
        elif _sha(p) != h:
            fails.append(f"altered {rel}")
    for key in ("instance_z", "instance_x"):
        if _sha(ROOT / man[key]["file"]) != man[key]["sha256"]:
            fails.append(f"altered {man[key]['file']}")
    if fails:
        print("REPLAY FAILED before any proof was read:", *fails[:10], sep="\n  ")
        return 1
    on_disk = {p.name[:-5] for p in (base / "proofs").glob("*.json")}
    listed = {pathlib.Path(rel).name[:-5] for rel in man["files"] if "/proofs/" in rel}
    if on_disk != listed:
        print(f"REPLAY FAILED: proofs on disk {len(on_disk)} vs listed {len(listed)}; "
              f"unlisted {sorted(on_disk - listed)[:5]} missing {sorted(listed - on_disk)[:5]}")
        return 1

    from oneq.qldpc_check import QldpcCertificate, check_qldpc_bnb, check_qldpc_exact
    inst_z = json.loads((ROOT / man["instance_z"]["file"]).read_text(encoding="utf-8"))
    inst_x = json.loads((ROOT / man["instance_x"]["file"]).read_text(encoding="utf-8"))
    n = inst_z["n_variables"]
    checks = [tuple(c) for c in inst_z["checks"]]
    HZ = np.zeros((len(checks), n), dtype=np.uint8)
    for j, c in enumerate(checks):
        HZ[j, list(c)] = 1
    HX = np.zeros((len(inst_x["checks"]), n), dtype=np.uint8)
    for j, c in enumerate(inst_x["checks"]):
        HX[j, list(c)] = 1
    xb, xp = gf2_row_reduce(HX)
    wx = {i: Fraction(1) for i in range(n)}

    def vec(support):
        v = np.zeros(n, dtype=np.uint8)
        v[list(support)] = 1
        return v

    def is_logical(fault):
        return bool(((HZ @ fault) % 2).any()) or not in_rowspace(xb, xp, fault)

    proofs: dict[str, dict] = {}

    def proof(pid):
        if pid not in proofs:
            proofs[pid] = json.loads((base / "proofs" / f"{pid}.json").read_text(encoding="utf-8"))
        return proofs[pid]

    checked: dict[tuple, bool] = {}

    def certify(pid, syn_sup, candidate):
        """The named proof, re-decided by the production checker for THIS candidate."""
        key = (pid, tuple(candidate))
        if key in checked:
            return checked[key]
        doc = proof(pid)
        if sorted(doc["syndrome_support"]) != sorted(syn_sup) or doc["instance"] != man["instance_z"]["file"]:
            checked[key] = False
            return False
        syn = [0] * len(checks)
        for j in syn_sup:
            syn[j] = 1
        if doc["kind"] == "flat":
            fd = [(tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(a), int(b)))
                  for (r, F, a, b) in doc["facet_duals"]]
            ok = check_qldpc_exact(checks=checks, weights=wx, syndrome=syn,
                                   cert=QldpcCertificate(error_support=tuple(candidate), facet_duals=fd,
                                                         box_duals={})).accepted
        else:
            ok = (list(candidate) == list(doc["candidate"])) and check_qldpc_bnb(
                checks=checks, weights=wx, syndrome=syn, error_support=tuple(candidate),
                tree=_deser_tree(doc["tree"])).accepted
        checked[key] = bool(ok)
        return checked[key]

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    mism: list[str] = []
    n_decisions = n_proof_checks = 0
    for lvl, L in summary["levels"].items():
        p = lvl.split("=")[1]
        shots = [json.loads(l) for l in (base / f"p{p}_shots.jsonl").read_text(encoding="utf-8").splitlines()]
        truth = {s["shot"]: vec(s["true_error_support"]) for s in shots}
        syn_of = {s["shot"]: s["syndrome_support"] for s in shots}
        for s in shots:                                   # the syndrome must be the sampled error's
            if sorted(int(j) for j in np.flatnonzero((HZ @ truth[s["shot"]]) % 2)) != sorted(s["syndrome_support"]):
                mism.append(f"{lvl} shot {s['shot']}: syndrome is not the sampled error's")
        ref_tiers = {t: 0 for t in ("tier2_lp_primal", "tierBD", "unresolved")}
        ref_err = 0
        ref_answer = {}
        for s in shots:
            r = s["reference"]
            ref_tiers[r["tier"]] += 1
            if r["support"] is not None:
                n_decisions += 1
                if not certify(r["proof"], s["syndrome_support"], r["support"]):
                    mism.append(f"{lvl} shot {s['shot']}: reference proof {r['proof']} refused")
                n_proof_checks += 1
                ref_answer[s["shot"]] = r["support"]
                ref_err += is_logical((vec(r["support"]) ^ truth[s["shot"]]) % 2)
        R = L["reference"]
        if (R["by_tier"] != ref_tiers or R["logical_errors_on_resolved"] != ref_err
                or R["resolved"] != len(ref_answer)):
            mism.append(f"{lvl}: reference summary disagrees (tiers {ref_tiers}, errors {ref_err})")
        for label, C in L["configs"].items():
            recs = [json.loads(l) for l in (base / f"p{p}_{label}.jsonl").read_text(encoding="utf-8").splitlines()]
            got = {"bp_osd_logical_errors": 0, "tier1": 0, "optimum_hit": 0, "tier2": 0, "tierBD": 0,
                   "unresolved": 0, "pipeline_logical_errors_on_resolved": 0,
                   "pipeline_vs_reference_different_optimum": 0, "pipeline_vs_reference_frame_differs": 0}
            n10 = n01 = 0
            for r in recs:
                k = r["shot"]
                bp = vec(r["bposd"]["support"])
                bp_valid = bool(np.array_equal((HZ @ bp) % 2, (HZ @ truth[k]) % 2))
                bp_err = is_logical((bp ^ truth[k]) % 2)
                got["bp_osd_logical_errors"] += bp_err
                ra = ref_answer.get(k)
                hit = bp_valid and ra is not None and len(r["bposd"]["support"]) == len(ra)
                got["optimum_hit"] += hit
                pipe = r["pipeline"]
                got[pipe["tier"] if pipe["tier"] != "unresolved" else "unresolved"] += 1
                if pipe["support"] is not None:
                    n_decisions += 1
                    if not certify(pipe["proof"], syn_of[k], pipe["support"]):
                        mism.append(f"{lvl} {label} shot {k}: pipeline proof {pipe['proof']} refused")
                    n_proof_checks += 1
                    if pipe["tier"] == "tier1" and list(pipe["support"]) != list(r["bposd"]["support"]):
                        mism.append(f"{lvl} {label} shot {k}: tier-1 answer is not BP-OSD's")
                    pe = is_logical((vec(pipe["support"]) ^ truth[k]) % 2)
                    got["pipeline_logical_errors_on_resolved"] += pe
                    if ra is not None:
                        got["pipeline_vs_reference_different_optimum"] += (list(pipe["support"]) != list(ra))
                        got["pipeline_vs_reference_frame_differs"] += is_logical((vec(pipe["support"]) ^ vec(ra)) % 2)
                    n10 += bp_err and not pe
                    n01 += (not bp_err) and pe
            nd, kmin = n10 + n01, min(n10, n01)
            pmc = (min(1.0, 2 * sum(math.comb(nd, i) for i in range(kmin + 1)) / 2 ** nd) if nd else 1.0)
            for key, val in got.items():
                if C[key] != val:
                    mism.append(f"{lvl} {label}: {key} summary {C[key]} vs replay {val}")
            if C["paired_on_resolved"] != {"bp_wrong_pipe_ok": n10, "bp_ok_pipe_wrong": n01} or abs(C["mcnemar_two_sided_p"] - pmc) > 1e-12:
                mism.append(f"{lvl} {label}: paired cells or McNemar disagree")
    verdict = {"schema": "oneq-strong-baseline-replay/1", "passed": not mism,
               "summary": str(SUMMARY.relative_to(ROOT) if SUMMARY.is_relative_to(ROOT) else SUMMARY).replace("\\", "/"),
               "summary_sha256": _sha(SUMMARY), "manifest_sha256": _sha(pathlib.Path(a.manifest)),
               "decisions": n_decisions, "proof_checks": len(checked), "proofs": len(proofs),
               "note": ("proof_checks counts distinct (proof, candidate) pairs decided by the production checker; "
                        "decisions that share a proof and a candidate are decided once and counted once here"),
               "proofs_by_kind": man["proofs_by_kind"], "disagreements": mism[:50],
               "command": "python tools/replay_strong_baseline.py",
               "reads": "records, proofs and the persisted instances only: no producer, no solver, no BP-OSD"}
    if manifest_path == MANIFEST.resolve():          # a replay of some other manifest is not this artifact's verdict
        (ROOT / "evidence" / "strong_baseline_v3_replay.json").write_text(
            json.dumps(verdict, indent=1) + "\n", encoding="utf-8", newline="\n")
    if mism:
        print(f"REPLAY FAILED ({len(mism)} disagreement(s)):", *mism[:20], sep="\n  ")
        return 1
    print(f"STRONG-BASELINE REPLAY PASSED: {n_decisions} decisions, {len(checked)} distinct proof checks over "
          f"{len(proofs)} stored proofs ({man['proofs_by_kind']}), every summary number in "
          f"{(SUMMARY.relative_to(ROOT) if SUMMARY.is_relative_to(ROOT) else SUMMARY)} recomputed from records and proofs alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
