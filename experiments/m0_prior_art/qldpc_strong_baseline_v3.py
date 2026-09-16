r"""The strong-baseline experiment, with every proof RETAINED (v3).

WHAT v2 STILL GOT WRONG (external audit 2026-09-15, finding 3): it checked every
receipt during the run and then threw the receipts away. `exact_flat` returned an
exact certificate that was used as a yes/no; `tree_receipt` verified a tree and
returned its node count. "The checker accepted a proof during our run" is not "the
reader has the proof and can check it", and the 2,042-receipt replay never read a
strong-baseline proof because none was stored.

WHAT THIS VERSION DOES, on the SAME seeds, syndromes, configurations and decision
rules as v2 (so v2's summary must reproduce, which tools/replay_strong_baseline.py
checks):

  * every accepted proof is written once, content-addressed, under
    evidence/strong_baseline_v3/proofs/<id>.json -- a flat certificate is the shot's
    exact LP dual (valid for any candidate of equal weight, so one file serves the
    reference and every tier-1 pipeline answer on that shot; each candidate is still
    validated separately at replay); a tree is the reference's branch-dual tree;
  * every decision -- reference, and pipeline per configuration -- names its proof;
  * the sampled error of every shot is kept (simulation truth, labelled as such) so
    the retrospective logical outcomes reconstruct too;
  * the instance is the persisted, hash-bound evidence/instances/bb144_z_code_capacity.json
    (Z checks) and bb144_x_code_capacity.json (X checks, for the logical-frame test);
    weights are the integer 1 on every variable;
  * a MANIFEST pins every file, and tools/replay_strong_baseline.py rebuilds Table II
    and the instrument table from the records and proofs alone -- no producer, no
    solver, no BP-OSD.

Two numbers per configuration, kept apart (audit finding 4): the TIER-1 RATE (how
often the flat-certification route proves BP-OSD's own answer optimal) and the
OPTIMUM-HIT RATE (how often BP-OSD's answer is feasible with weight equal to the
certified optimum, whether or not the flat route closed on it). Tier-1 refusal is
not evidence that BP-OSD missed the optimum; the second number is.

    python experiments/m0_prior_art/qldpc_strong_baseline_v3.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce  # noqa: E402
from qldpc_strong_baseline_gate import CONFIGS, NOISE, SHOTS  # noqa: E402

from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,  # noqa: E402
                             exact_milp, exactify_certificate, feldman_dual)
from oneq.qldpc_check import check_qldpc_bnb, check_qldpc_exact  # noqa: E402

OUT = ROOT / "evidence" / "qldpc_strong_baseline_v3.json"
DIR = ROOT / "evidence" / "strong_baseline_v3"
PROOFS = DIR / "proofs"
INSTANCE_Z = "evidence/instances/bb144_z_code_capacity.json"
INSTANCE_X = "evidence/instances/bb144_x_code_capacity.json"
BNB_MAX_NODES = 60000


def ser_facets(ec):
    return [[list(j) if isinstance(j, tuple) else int(j), sorted(int(i) for i in F),
             y.numerator, y.denominator] for (j, F, y) in ec.facet_duals]


def ser_tree(node):
    if node[0] == "branch":
        return ["branch", int(node[1]), ser_tree(node[2]), ser_tree(node[3])]
    if node[0] == "bound":
        return ["bound", [[list(j) if isinstance(j, tuple) else int(j), list(F),
                           y.numerator, y.denominator] for (j, F, y) in node[1]]]
    return ["infeasible", list(node[1]) if isinstance(node[1], tuple) else int(node[1])]


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


class ProofStore:
    def __init__(self):
        PROOFS.mkdir(parents=True, exist_ok=True)
        self.ids: dict[str, dict] = {}

    def put(self, doc: dict) -> str:
        pid = sha_json(doc)[:16]
        if pid not in self.ids:
            (PROOFS / f"{pid}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n",
                                                encoding="utf-8", newline="\n")
            self.ids[pid] = {"kind": doc["kind"], "bytes": (PROOFS / f"{pid}.json").stat().st_size,
                             "nodes": doc.get("nodes")}
        return pid


def exact_flat(checks, wf, wx, syn, support, cert):
    """The exact receipt for `support` from a float LP dual, or None."""
    fc = QldpcCertificate(error_support=tuple(support),
                          facet_duals=cert.facet_duals, box_duals=cert.box_duals)
    if not check_qldpc(checks=checks, weights=wf, syndrome=list(syn), cert=fc).accepted:
        return None
    ec = exactify_certificate(fc)
    r = check_qldpc_exact(checks=checks, weights=wx, syndrome=list(syn), cert=ec)
    return ec if r.accepted else None


def tree_receipt(checks, wf, wx, syn):
    """Solver candidate + branch-dual tree, kept ONLY if the tree checker accepts."""
    sup, wt, st = exact_milp(checks, wf, list(syn), return_status=True)
    if sup is None:
        return None, None, {"solver_status": st, "reason": "solver returned no candidate"}
    cand = tuple(int(i) for i in sup)
    U = int(round(float(wt)))
    tree, nodes = certified_bnb(checks, wf, list(syn), U, rpc_rounds=6, max_nodes=BNB_MAX_NODES)
    if tree is None:
        return None, None, {"solver_status": st, "reason": f"no tree within {BNB_MAX_NODES} nodes"}
    rb = check_qldpc_bnb(checks=checks, weights=wx, syndrome=list(syn), error_support=cand, tree=tree)
    if not rb.accepted:
        return None, None, {"solver_status": st, "reason": f"tree refused: {rb.reason[:80]}"}
    return cand, (tree, nodes), {"tree_nodes": nodes}


def main() -> int:
    inst_z = json.loads((ROOT / INSTANCE_Z).read_text(encoding="utf-8"))
    inst_x = json.loads((ROOT / INSTANCE_X).read_text(encoding="utf-8"))
    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    assert [list(c) for c in checks] == inst_z["checks"], "the persisted Z instance is not the code being run"
    assert [list(map(int, np.flatnonzero(HX[j]))) for j in range(m)] == inst_x["checks"], \
        "the persisted X instance is not the code being run"
    wf = {i: 1.0 for i in range(n)}
    wx = {i: Fraction(1) for i in range(n)}
    xb, xp = gf2_row_reduce(HX % 2)

    def is_logical(fault):
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    def vec(support):
        v = np.zeros(n, dtype=np.uint8)
        v[list(support)] = 1
        return v

    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix

    store = ProofStore()
    files: dict[str, str] = {}
    out = {"schema": "oneq-qldpc-strong-baseline/3",
           "supersedes": "evidence/qldpc_strong_baseline_v2.json (same seeds, syndromes, configurations and "
                         "decisions; v2 discarded every proof it checked)",
           "protocol": f"[[144,12,12]] Z-sector, uniform integer weights, {SHOTS} nontrivial syndromes per "
                       f"noise level from the SAME seeds as v1/v2, shared by every configuration",
           "instance_z": INSTANCE_Z, "instance_x": INSTANCE_X,
           "records": "evidence/strong_baseline_v3/ (per-level shots with the sampled error, per-configuration "
                      "records naming their proofs, proofs/<id>.json content-addressed, MANIFEST.json)",
           "replay": "python tools/replay_strong_baseline.py",
           "configs": CONFIGS, "levels": {}}

    for p in NOISE:
        rng = np.random.default_rng(int(10000 * p) + 31337)      # v1's seed rule
        shots = []
        while len(shots) < SHOTS:
            e = (rng.random(n) < p).astype(np.uint8)
            syn = (HZ @ e) % 2
            if syn.any():
                shots.append((syn, e))

        t0 = time.perf_counter()
        lp = [feldman_dual(checks, wf, list(s), return_primal=True, rpc_rounds=6) for s, _ in shots]
        # the shot's exact LP dual, once: it certifies ANY candidate of the optimal weight
        flat_id = []
        for (syn, _e), (cert, _b, primal) in zip(shots, lp):
            ec = exact_flat(checks, wf, wx, syn, primal, cert) if (cert is not None and primal is not None) else None
            flat_id.append(store.put({"kind": "flat", "instance": INSTANCE_Z, "weights": "uniform-1",
                                      "syndrome_support": [int(i) for i in np.flatnonzero(syn)],
                                      "facet_duals": ser_facets(ec)}) if ec is not None else None)

        ref, shot_rows = [], []
        for k, ((syn, e_true), (cert, _b, primal)) in enumerate(zip(shots, lp)):
            r = {"answer": None, "tier": "unresolved", "proof": None, "detail": {}}
            if flat_id[k] is not None:
                r.update(answer=tuple(int(i) for i in primal), tier="tier2_lp_primal", proof=flat_id[k])
            else:
                cand, tree_nodes, det = tree_receipt(checks, wf, wx, syn)
                r["detail"] = det
                if cand is not None:
                    tree, nodes = tree_nodes
                    pid = store.put({"kind": "tree", "instance": INSTANCE_Z, "weights": "uniform-1",
                                     "syndrome_support": [int(i) for i in np.flatnonzero(syn)],
                                     "candidate": list(cand), "nodes": int(nodes), "tree": ser_tree(tree)})
                    r.update(answer=cand, tier="tierBD", proof=pid)
            if r["answer"] is not None:
                r["logical_error"] = bool(is_logical((vec(r["answer"]) ^ e_true) % 2))
            ref.append(r)
            shot_rows.append({"shot": k, "syndrome_support": [int(i) for i in np.flatnonzero(syn)],
                              "true_error_support": [int(i) for i in np.flatnonzero(e_true)],
                              "reference": {"tier": r["tier"], "support": None if r["answer"] is None else list(r["answer"]),
                                            "proof": r["proof"], "detail": r["detail"],
                                            "logical_error": r.get("logical_error")}})
        rel = f"evidence/strong_baseline_v3/p{p}_shots.jsonl"
        with open(ROOT / rel, "w", encoding="utf-8", newline="\n") as f:
            for row in shot_rows:
                f.write(json.dumps(row) + "\n")
        files[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()

        tree_nodes = [store.ids[r["proof"]]["nodes"] for r in ref if r["tier"] == "tierBD"]
        tree_bytes = [store.ids[r["proof"]]["bytes"] for r in ref if r["tier"] == "tierBD"]
        level = {"reference": {
            "resolved": sum(r["answer"] is not None for r in ref),
            "unresolved": sum(r["answer"] is None for r in ref),
            "logical_errors_on_resolved": sum(bool(r.get("logical_error")) for r in ref),
            "by_tier": {t: sum(r["tier"] == t for r in ref) for t in ("tier2_lp_primal", "tierBD", "unresolved")},
            "trees": {"count": len(tree_nodes),
                      "nodes_median": (float(np.median(tree_nodes)) if tree_nodes else None),
                      "nodes_max": (max(tree_nodes) if tree_nodes else None),
                      "bytes_max": (max(tree_bytes) if tree_bytes else None)}},
            "configs": {}}

        for label, cfg in CONFIGS.items():
            dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, **cfg)
            recs = []
            for k, ((syn, e_true), (cert, _b, primal)) in enumerate(zip(shots, lp)):
                e_bp = np.asarray(dec.decode(syn.astype(np.uint8))).astype(np.uint8)
                bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
                bp_valid = bool(np.array_equal((HZ @ e_bp) % 2, syn))
                ra = ref[k]["answer"]
                U = None if ra is None else len(ra)                     # uniform weights: the reference is optimal
                rec = {"shot": k, "bposd": {"support": list(bp_sup), "syndrome_valid": bp_valid,
                                            "weight": len(bp_sup),
                                            "optimum_hit": (bool(bp_valid and U is not None and len(bp_sup) == U)),
                                            "logical_error": bool(is_logical((e_bp ^ e_true) % 2))}}
                answer, tier, proof, detail = None, "unresolved", None, {}
                if flat_id[k] is not None and bp_valid and exact_flat(checks, wf, wx, syn, bp_sup, cert) is not None:
                    answer, tier, proof = bp_sup, "tier1", flat_id[k]
                elif ref[k]["tier"] == "tier2_lp_primal":
                    answer, tier, proof = ra, "tier2", ref[k]["proof"]
                elif ref[k]["tier"] == "tierBD":
                    answer, tier, proof = ra, "tierBD", ref[k]["proof"]
                else:
                    detail = ref[k]["detail"]
                rec["pipeline"] = {"tier": tier, "support": None if answer is None else list(answer),
                                   "proof": proof, "detail": detail}
                if answer is not None:
                    rec["pipeline"]["logical_error"] = bool(is_logical((vec(answer) ^ e_true) % 2))
                    if ra is not None:
                        rec["pipeline"]["same_support_as_reference"] = (tuple(answer) == tuple(ra))
                        rec["pipeline"]["frame_differs_from_reference"] = bool(is_logical((vec(answer) ^ vec(ra)) % 2))
                recs.append(rec)

            resolved = [r for r in recs if r["pipeline"]["support"] is not None]
            pipe_err = sum(r["pipeline"]["logical_error"] for r in resolved)
            bp_err = sum(r["bposd"]["logical_error"] for r in recs)
            n10 = sum(r["bposd"]["logical_error"] and not r["pipeline"]["logical_error"] for r in resolved)
            n01 = sum((not r["bposd"]["logical_error"]) and r["pipeline"]["logical_error"] for r in resolved)
            nd, kmin = n10 + n01, min(n10, n01)
            pmc = (min(1.0, 2 * sum(math.comb(nd, i) for i in range(kmin + 1)) / 2 ** nd) if nd else 1.0)
            t1 = sum(r["pipeline"]["tier"] == "tier1" for r in recs)
            hit = sum(r["bposd"]["optimum_hit"] for r in recs)
            level["configs"][label] = {
                "bp_osd_logical_errors": bp_err,
                "bp_osd_syndrome_invalid": sum(not r["bposd"]["syndrome_valid"] for r in recs),
                "tier1": t1, "tier1_rate": round(100 * t1 / len(recs), 1),
                "optimum_hit": hit, "optimum_hit_rate": round(100 * hit / len(recs), 1),
                "optimum_hit_not_tier1": sum(r["bposd"]["optimum_hit"] and r["pipeline"]["tier"] != "tier1" for r in recs),
                "tier1_not_optimum_hit": sum(r["pipeline"]["tier"] == "tier1" and not r["bposd"]["optimum_hit"] for r in recs),
                "tier2": sum(r["pipeline"]["tier"] == "tier2" for r in recs),
                "tierBD": sum(r["pipeline"]["tier"] == "tierBD" for r in recs),
                "unresolved": sum(r["pipeline"]["tier"] == "unresolved" for r in recs),
                "pipeline_logical_errors_on_resolved": pipe_err,
                "pipeline_vs_reference_different_optimum": sum(
                    r["pipeline"].get("same_support_as_reference") is False for r in resolved),
                "pipeline_vs_reference_frame_differs": sum(
                    bool(r["pipeline"].get("frame_differs_from_reference")) for r in resolved),
                "paired_on_resolved": {"bp_wrong_pipe_ok": n10, "bp_ok_pipe_wrong": n01},
                "mcnemar_two_sided_p": pmc}
            rel = f"evidence/strong_baseline_v3/p{p}_{label}.jsonl"
            with open(ROOT / rel, "w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            files[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            c = level["configs"][label]
            print(f"  p={p} {label:26s} BP err {bp_err:3d} | pipeline err {pipe_err:3d} "
                  f"(t1 {c['tier1']} hit {c['optimum_hit']} t2 {c['tier2']} BD {c['tierBD']} unres {c['unresolved']}) "
                  f"| diff-optimum {c['pipeline_vs_reference_different_optimum']} "
                  f"frame-differs {c['pipeline_vs_reference_frame_differs']}", flush=True)
        level["seconds"] = round(time.perf_counter() - t0, 1)
        out["levels"][f"p={p}"] = level
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")

    for pid in store.ids:
        rel = f"evidence/strong_baseline_v3/proofs/{pid}.json"
        files[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    manifest = {"schema": "oneq-strong-baseline-manifest/1",
                "instance_z": {"file": INSTANCE_Z, "sha256": hashlib.sha256((ROOT / INSTANCE_Z).read_bytes()).hexdigest()},
                "instance_x": {"file": INSTANCE_X, "sha256": hashlib.sha256((ROOT / INSTANCE_X).read_bytes()).hexdigest()},
                "weights": "the integer 1 on every variable",
                "summary": "evidence/qldpc_strong_baseline_v3.json",
                "proofs": len(store.ids),
                "proofs_by_kind": {k: sum(1 for v in store.ids.values() if v["kind"] == k) for k in ("flat", "tree")},
                "files": dict(sorted(files.items()))}
    (DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    out["manifest"] = "evidence/strong_baseline_v3/MANIFEST.json"
    out["proofs"] = manifest["proofs_by_kind"]
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence -> {OUT}; {manifest['proofs']} proofs ({manifest['proofs_by_kind']}) under {PROOFS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
