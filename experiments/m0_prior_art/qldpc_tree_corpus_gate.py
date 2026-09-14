r"""THE COMPLETE TREE CORPUS: every branch-dual tree, both checkers, timed.

Audit round 7 items 2 and 4: the second-checker claim covered 36 trees
while the manuscript describes four tree-bearing corpora (held-out,
development non-circuit, development reference-schedule, paired-stress
replay), and the new tier shipped without operational cost numbers.

This gate assembles EVERY tree from EVERY corpus, regenerating each
deterministically from published seeds, and for each one records:

  producer time, node count, depth, serialized bytes,
  checker A (production, exact rational) verdict + time,
  checker B (separately implemented, from the schema) verdict + time

and emits the corpus manifest the auditor asked for, with checker
hashes. Any tree either checker refuses fails the gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code
from qldpc_bb_phenom import spacetime_system
from qldpc_bb_refsched_gate import build_refsched_circuit
from qldpc_bnb_residue_gate import find_residues_code_capacity
from qldpc_circuit_gate import dem_to_matrices
from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, check_qldpc,
                             exact_milp, feldman_dual, feldman_dual_lazy,
                             fixed_point_weights)
from oneq.qldpc_check import check_qldpc_bnb

CKB = ROOT / "tools" / "external" / "tree_checker_b.py"


def ser(node):
    if node[0] == "branch":
        return ["branch", int(node[1]), ser(node[2]), ser(node[3])]
    if node[0] == "bound":
        return ["bound",
                [[list(j) if isinstance(j, tuple) else int(j),
                  list(F), y.numerator, y.denominator]
                 for (j, F, y) in node[1]]]
    return ["infeasible",
            list(node[1]) if isinstance(node[1], tuple) else int(node[1])]


def depth(t):
    return 1 + max(depth(t[2]), depth(t[3])) if t[0] == "branch" else 1


def main() -> int:
    tmp = ROOT / "evidence" / "_corpus_tmp.json"

    def run_b(checks_list, wvec, synv, cand, tree_json):
        tmp.write_text(json.dumps(
            {"checks": checks_list,
             "weights": [int(v) for v in wvec],
             "syndrome": [int(b) for b in synv],
             "candidate": [int(i) for i in cand],
             "tree": tree_json}), encoding="utf-8")
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, str(CKB), str(tmp)],
                            capture_output=True, text=True, timeout=900)
        return rc.returncode == 0, time.perf_counter() - t0

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    cz = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    cz_list = [list(c) for c in cz]
    wu = {i: 1 for i in range(n)}
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix

    corpora = []          # (label, checks, wint, [(syn, cand, U, lazy)])

    # --- development non-circuit residues (uniform p02, hetero, stress)
    rw = np.random.default_rng(4242)
    p_h = 0.02 * rw.uniform(0.5, 2.0, n)
    wh, _ = fixed_point_weights(
        {i: float(np.log((1 - p_h[i]) / p_h[i])) for i in range(n)},
        scale=1000)
    dev = [("uniform_p02", 2107, np.full(n, 0.02), 470, wu, None),
           ("hetero_p02", 2101, p_h, 300, wh,
            [float(v) for v in p_h]),
           ("stress_p05", 2104, np.full(n, 0.05), 300, wu, None)]
    dev_items = []
    for tag, sd, parr, tgt, wint, chan in dev:
        dec = (BpOsdDecoder(csr_matrix(HZ), error_channel=chan,
                            max_iter=30, bp_method="ms",
                            schedule="parallel", osd_method="osd_cs",
                            osd_order=5) if chan else
               BpOsdDecoder(csr_matrix(HZ), error_rate=float(parr[0]),
                            max_iter=30, bp_method="ms",
                            schedule="parallel", osd_method="osd_cs",
                            osd_order=5))
        res = find_residues_code_capacity(HZ, cz, wint, dec, sd, parr, tgt)
        wf = {i: float(wint[i]) for i in range(n)}
        for (si, syn) in res:
            sup, wt, _st = exact_milp(cz, wf, list(syn), time_limit=120.0,
                                      return_status=True)
            dev_items.append((tag, si, list(syn), tuple(int(i)
                              for i in sup), int(round(float(wt))),
                              wint, cz, cz_list, False))
    corpora.append(("development non-circuit", dev_items))

    # --- paired-stress replay (seed 20260803) ---
    dec_p = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    rngp = np.random.default_rng(20260803)
    wfp = {i: 1.0 for i in range(n)}
    paired_items = []
    shots = 0
    while shots < 300:
        e = (rngp.random(n) < 0.05).astype(np.uint8)
        syn = (HZ @ e) % 2
        if not syn.any():
            continue
        shots += 1
        e_bp = np.asarray(dec_p.decode(syn.astype(np.uint8))) \
            .astype(np.uint8)
        cert, _b, primal = feldman_dual(cz, wfp, list(syn),
                                        return_primal=True, rpc_rounds=6)
        ok = False
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            if check_qldpc(checks=cz, weights=wfp, syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                ok = True
            elif primal is not None and check_qldpc(
                    checks=cz, weights=wfp, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                ok = True
        if not ok:
            sup, wt, _st = exact_milp(cz, wfp, list(syn),
                                      return_status=True)
            paired_items.append(("paired_stress", shots - 1, list(syn),
                                 tuple(int(i) for i in sup),
                                 int(round(float(wt))), wu, cz, cz_list,
                                 False))
    corpora.append(("paired-stress replay", paired_items))

    # --- reference-schedule development residues (seed 20260804) ---
    circ = build_refsched_circuit(3, 0.002)
    dem = circ.detector_error_model(decompose_errors=False)
    cc, wllr, _obs, nm = dem_to_matrices(dem)
    wint_c = {i: int(math.floor(1000 * float(wllr[i]) + 0.5))
              for i in range(nm)}
    wfc = {i: float(wint_c[i]) for i in range(nm)}
    cc_list = [list(c) for c in cc]
    from scipy.sparse import lil_matrix
    Hc = lil_matrix((len(cc), nm), dtype=np.uint8)
    for j, c0 in enumerate(cc):
        for i in c0:
            Hc[j, i] = 1
    dec_c = BpOsdDecoder(Hc.tocsr(), error_channel=[
        float(1 / (1 + np.exp(wllr[i]))) for i in range(nm)],
        max_iter=40, bp_method="ms", schedule="parallel",
        osd_method="osd_cs", osd_order=5)
    dets, _obs2, _ = dem.compile_sampler(seed=20260804).sample(
        shots=150, return_errors=False)
    ref_items = []
    idx = -1
    for s in range(dets.shape[0]):
        syn = dets[s].astype(np.uint8)
        if not syn.any():
            continue
        idx += 1
        e_bp = np.asarray(dec_c.decode(syn)).astype(np.uint8)
        cert, _b, primal = feldman_dual_lazy(cc, wfc, list(syn),
                                             return_primal=True)
        ok = False
        if cert is not None:
            bp_sup = tuple(int(i) for i in np.flatnonzero(e_bp))
            if check_qldpc(checks=cc, weights=wfc, syndrome=list(syn),
                           cert=QldpcCertificate(
                               error_support=bp_sup,
                               facet_duals=cert.facet_duals,
                               box_duals=cert.box_duals)).accepted:
                ok = True
            elif primal is not None and check_qldpc(
                    checks=cc, weights=wfc, syndrome=list(syn),
                    cert=QldpcCertificate(
                        error_support=tuple(primal),
                        facet_duals=cert.facet_duals,
                        box_duals=cert.box_duals)).accepted:
                ok = True
        if not ok:
            sup, wt, _st = exact_milp(cc, wfc, list(syn), time_limit=60.0,
                                      return_status=True)
            ref_items.append(("refsched_dev", idx, list(syn),
                              tuple(int(i) for i in sup),
                              int(sum(wint_c[int(i)] for i in sup)),
                              wint_c, cc, cc_list, True))
    corpora.append(("development reference-schedule", ref_items))

    # --- held-out corpus: trees already stored in the addendum ---
    add_p = ROOT / "evidence" / "qldpc_heldout_addendum.json"
    heldout_items = []
    if add_p.exists():
        add = json.loads(add_p.read_text(encoding="utf-8"))
        cst_h, wst_h = spacetime_system(HZ, 4, 0.02, 0.02)
        wph, _ = fixed_point_weights(wst_h, scale=1000)
        ho = json.loads((ROOT / "evidence" / "qldpc_heldout.json")
                        .read_text(encoding="utf-8"))
        chs = ho["rungs"]["hetero_p02"]["channel_seed"]
        rwh = np.random.default_rng(chs)
        pih = 0.02 * rwh.uniform(0.5, 2.0, n)
        whh, _ = fixed_point_weights(
            {i: float(np.log((1 - pih[i]) / pih[i])) for i in range(n)},
            scale=1000)
        for t in add.get("bd_trees", []):
            tag = t["rung"]
            if tag == "phenom_p02":
                ck, ckl, wi, lz = cst_h, [list(c) for c in cst_h], wph, False
            elif tag == "bb_refsched":
                ck, ckl, wi, lz = cc, cc_list, wint_c, True
            else:
                ck, ckl = cz, cz_list
                wi = whh if tag == "hetero_p02" else wu
                lz = False
            ss = set(t["syndrome_support"])
            synv = [1 if j in ss else 0 for j in range(len(ck))]
            heldout_items.append(
                (f"heldout_{tag}", t["shot_index"], synv,
                 tuple(int(i) for i in t["candidate"]),
                 t["answer_weight"], wi, ck, ckl, lz))
        corpora.insert(0, ("held-out (frozen campaign)", heldout_items))

    # --- run both checkers over everything, timed ---
    out = {"schema": "oneq-qldpc-tree-corpus/1",
           "checker_a": "src/oneq/qldpc_check.py:check_qldpc_bnb",
           "checker_b": "tools/external/tree_checker_b.py",
           "checker_b_sha256": hashlib.sha256(
               CKB.read_bytes()).hexdigest(),
           "corpora": {}, "trees": []}
    prod_t, ca_t, cb_t, nodes_l, depth_l, bytes_l = [], [], [], [], [], []
    for label, items in corpora:
        rec = {"trees": 0, "checker_a_accept": 0, "checker_b_accept": 0}
        for (tag, si, syn, cand, U, wint, checks, checks_list,
             lazy) in items:
            wf = {i: float(wint[i]) for i in range(len(wint))}
            wx = {i: Fraction(int(wint[i])) for i in range(len(wint))}
            t0 = time.perf_counter()
            tree, nodes = certified_bnb(checks, wf, syn, U,
                                        rpc_rounds=6, max_nodes=60000,
                                        use_lazy=lazy)
            tp = time.perf_counter() - t0
            assert tree is not None, f"{tag}/{si}: producer failed"
            t1 = time.perf_counter()
            ra = check_qldpc_bnb(checks=checks, weights=wx,
                                 syndrome=syn, error_support=cand,
                                 tree=tree)
            ta = time.perf_counter() - t1
            tj = ser(tree)
            okb, tb = run_b(checks_list,
                            [int(wint[i]) for i in range(len(wint))],
                            syn, list(cand), tj)
            rec["trees"] += 1
            rec["checker_a_accept"] += int(ra.accepted)
            rec["checker_b_accept"] += int(okb)
            nb = len(json.dumps(tj))
            prod_t.append(tp); ca_t.append(ta); cb_t.append(tb)
            nodes_l.append(nodes); depth_l.append(depth(tj))
            bytes_l.append(nb)
            out["trees"].append(
                {"corpus": label, "rung": tag, "shot_index": si,
                 "nodes": nodes, "depth": depth(tj), "bytes": nb,
                 "producer_s": round(tp, 3), "checker_a_s": round(ta, 4),
                 "checker_b_s": round(tb, 3),
                 "checker_a": bool(ra.accepted), "checker_b": bool(okb)})
        out["corpora"][label] = rec
        print(f"  {label:32s} {rec['trees']:3d} trees  A "
              f"{rec['checker_a_accept']}/{rec['trees']}  B "
              f"{rec['checker_b_accept']}/{rec['trees']}")
        assert rec["checker_a_accept"] == rec["trees"]
        assert rec["checker_b_accept"] == rec["trees"]

    def stats(v):
        v = sorted(v)
        return {"median": v[len(v) // 2],
                "p95": v[max(0, int(len(v) * 0.95) - 1)], "max": v[-1]}
    out["cost"] = {
        "producer_seconds": {k: round(x, 3)
                             for k, x in stats(prod_t).items()},
        "checker_a_seconds": {k: round(x, 4)
                              for k, x in stats(ca_t).items()},
        "checker_b_seconds": {k: round(x, 3)
                              for k, x in stats(cb_t).items()},
        "nodes": stats(nodes_l), "depth": stats(depth_l),
        "serialized_bytes": stats(bytes_l)}
    out["budget_declaration"] = (
        "every tree in every corpus closed within max_nodes=60000 and "
        "no wall-clock deadline was imposed; the 100% exact totals are "
        "therefore an OFFLINE certification result, while the flat "
        "receipt rate (99.15% held-out) is the fixed-cost coverage")
    tmp.unlink(missing_ok=True)
    total = sum(c["trees"] for c in out["corpora"].values())
    out["total_trees"] = total
    print(f"TOTAL: {total} trees, both checkers accept all")
    print(f"cost: {json.dumps(out['cost'])}")
    dest = ROOT / "evidence" / "qldpc_tree_corpus.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
