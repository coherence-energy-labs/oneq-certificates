r"""Re-check every stored certificate a stranger was handed.

Runs the checkers, never the producers: every stored branch-dual tree
goes through BOTH the production checker (exact rational) and the
separately implemented tree checker B, and their verdicts must agree
and be ACCEPT. Prints counts; exits nonzero on any disagreement or
refusal. No LP, no MIP, no network.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from oneq.qldpc_check import check_qldpc_bnb          # noqa: E402


def _deser(node):
    """JSON tree -> the production checker's tuple form."""
    if node[0] == "branch":
        return ("branch", int(node[1]), _deser(node[2]), _deser(node[3]))
    if node[0] == "bound":
        return ("bound",
                tuple((tuple(j) if isinstance(j, list) else int(j),
                       tuple(F), Fraction(int(n), int(d)))
                      for (j, F, n, d) in node[1]))
    return ("infeasible",
            tuple(node[1]) if isinstance(node[1], list) else int(node[1]))


def main() -> int:
    from qldpc_bb_gate import bb_gross_code
    from qldpc_bb_phenom import spacetime_system
    from oneq.qldpc_cert import fixed_point_weights

    add_p = ROOT / "evidence" / "qldpc_heldout_addendum.json"
    if not add_p.exists():
        print("no addendum artifact: nothing to re-check")
        return 1
    add = json.loads(add_p.read_text(encoding="utf-8"))
    ho = json.loads((ROOT / "evidence" / "qldpc_heldout.json")
                    .read_text(encoding="utf-8"))

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    cz = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    cz_list = [list(c) for c in cz]
    wu = [1] * n

    # instance rebuilds, from published seeds/specs only
    rw = np.random.default_rng(4242)
    p_dev = 0.02 * rw.uniform(0.5, 2.0, n)
    wh_dev, _ = fixed_point_weights(
        {i: float(np.log((1 - p_dev[i]) / p_dev[i])) for i in range(n)},
        scale=1000)
    ch = ho["rungs"]["hetero_p02"]["channel_seed"]
    rw2 = np.random.default_rng(ch)
    p_ho = 0.02 * rw2.uniform(0.5, 2.0, n)
    wh_ho, _ = fixed_point_weights(
        {i: float(np.log((1 - p_ho[i]) / p_ho[i])) for i in range(n)},
        scale=1000)
    cst, wst = spacetime_system(HZ, 4, 0.02, 0.02)
    wph, _ = fixed_point_weights(wst, scale=1000)

    ckb = ROOT / "tools" / "external" / "tree_checker_b.py"
    tmp = ROOT / "evidence" / "_recheck_tmp.json"
    both = agree = 0
    fails = []

    def check_one(label, checks_list, wvec, synv, cand, tree_json):
        nonlocal both, agree
        tree = _deser(tree_json)
        wx = {i: Fraction(int(wvec[i])) for i in range(len(wvec))}
        a = check_qldpc_bnb(checks=[tuple(c) for c in checks_list],
                            weights=wx, syndrome=synv,
                            error_support=tuple(int(i) for i in cand),
                            tree=tree)
        tmp.write_text(json.dumps(
            {"checks": checks_list, "weights": list(wvec),
             "syndrome": synv, "candidate": list(cand),
             "tree": tree_json}), encoding="utf-8")
        rc = subprocess.run([sys.executable, str(ckb), str(tmp)],
                            capture_output=True, text=True, timeout=600)
        b = (rc.returncode == 0)
        both += 1
        if a.accepted and b:
            agree += 1
        else:
            fails.append(f"{label}: production={a.accepted} B={b}")

    def records(tag, prefix):
        f = ROOT / "evidence" / "per_shot" / f"{prefix}{tag}.jsonl"
        return {json.loads(x)["shot_index"]: json.loads(x)
                for x in f.read_text(encoding="utf-8").splitlines()}

    # held-out trees -- each carries its own syndrome + candidate now
    refsched = None
    for t in add.get("bd_trees", []):
        tag = t["rung"]
        ss = set(t["syndrome_support"])
        if tag == "phenom_p02":
            cl = [list(c) for c in cst]
            wv = [wph[i] for i in range(len(wph))]
        elif tag == "bb_refsched":
            if refsched is None:
                from qldpc_bb_refsched_gate import build_refsched_circuit
                from qldpc_circuit_gate import dem_to_matrices
                import math as _m
                _c = build_refsched_circuit(3, 0.002)
                _d = _c.detector_error_model(decompose_errors=False)
                cc, wllr, _o, nm = dem_to_matrices(_d)
                refsched = ([list(c) for c in cc],
                            [int(_m.floor(1000 * float(wllr[i]) + 0.5))
                             for i in range(nm)])
            cl, wv = refsched
        else:
            cl = cz_list
            wv = ([wh_ho[i] for i in range(n)]
                  if tag == "hetero_p02" else wu)
        synv = [1 if j in ss else 0 for j in range(len(cl))]
        check_one(f"heldout/{tag}/{t['shot_index']}", cl, wv, synv,
                  t["candidate"], t["tree"])

    # development trees (carry their own syndrome support)
    for t in add.get("dev_bd_trees", []):
        wv = ([wh_dev[i] for i in range(n)]
              if t["rung"] == "hetero_p02" else wu)
        ss = set(t["syndrome_support"])
        synv = [1 if j in ss else 0 for j in range(len(cz_list))]
        check_one(f"dev/{t['rung']}/{t['shot_index']}", cz_list, wv,
                  synv, t["candidate"], t["tree"])

    tmp.unlink(missing_ok=True)
    print(f"branch-dual trees re-checked by BOTH checkers: "
          f"{agree}/{both} accepted by both")

    # ---- the FLAT certificates (2026-09-12): 99.15% of the campaign, and
    # until the flat-certs addendum they shipped as hashes only. Re-check
    # every stored one with BOTH the production exact checker and the
    # pinned external reference checker, from the stored payload -- no
    # producer runs here, which is the entire point of shipping them.
    flat_p = ROOT / "evidence" / "qldpc_heldout_flat_certs.json"
    fboth = fagree = 0
    if flat_p.exists():
        import hashlib
        import importlib.util
        from oneq.qldpc_check import QldpcCertificate, check_qldpc_exact
        from qldpc_exact_all_gate import _cert_doc, _dense_H
        flat = json.loads(flat_p.read_text(encoding="utf-8"))
        refp = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
        assert hashlib.sha256(refp.read_bytes()).hexdigest() == \
            flat["reference_checker_sha256"], "reference checker is not the pinned bytes"
        spec = importlib.util.spec_from_file_location("ref_checker", refp)
        ref = importlib.util.module_from_spec(spec)
        sys.modules["ref_checker"] = ref
        spec.loader.exec_module(ref)
        cx_list = [[int(i) for i in np.flatnonzero(HX[j])] for j in range(m)]
        cst1, wst1 = spacetime_system(HZ, 4, 0.01, 0.01)
        wph1, _ = fixed_point_weights(wst1, scale=1000)
        inst = {}

        def instance(tag):
            if tag in inst:
                return inst[tag]
            if tag in ("uniform_p01", "uniform_p02", "stress_p05"):
                v = (cz_list, wu)
            elif tag == "xsector_p02":
                v = (cx_list, wu)
            elif tag == "hetero_p02":
                v = (cz_list, [wh_ho[i] for i in range(n)])
            elif tag == "phenom_p01":
                v = ([list(c) for c in cst1], [wph1[i] for i in range(len(wph1))])
            elif tag == "phenom_p02":
                v = ([list(c) for c in cst], [wph[i] for i in range(len(wph))])
            elif tag == "bb_refsched":
                nonlocal refsched
                if refsched is None:
                    from qldpc_bb_refsched_gate import build_refsched_circuit
                    from qldpc_circuit_gate import dem_to_matrices
                    import math as _m
                    _d = build_refsched_circuit(3, 0.002).detector_error_model(
                        decompose_errors=False)
                    cc, wllr, _o, nm = dem_to_matrices(_d)
                    refsched = ([list(c) for c in cc],
                                [int(_m.floor(1000 * float(wllr[i]) + 0.5))
                                 for i in range(nm)])
                v = refsched
            else:
                raise KeyError(tag)
            cl, wv = v
            inst[tag] = (cl, wv, {i: Fraction(int(wv[i])) for i in range(len(wv))},
                         _dense_H([tuple(c) for c in cl], len(wv)),
                         {i: int(wv[i]) for i in range(len(wv))})
            return inst[tag]

        for c in flat["certificates"]:
            cl, wv, wx, dense, wint = instance(c["rung"])
            synv = [0] * len(cl)
            for j in c["syndrome_support"]:
                synv[j] = 1
            fd = [(tuple(r) if isinstance(r, list) else int(r), tuple(F),
                   Fraction(int(num), int(den))) for (r, F, num, den) in c["facet_duals"]]
            cand = tuple(int(i) for i in c["candidate"])
            ec = QldpcCertificate(error_support=cand, facet_duals=fd, box_duals={})
            a = check_qldpc_exact(checks=[tuple(x) for x in cl], weights=wx,
                                  syndrome=synv, cert=ec).accepted
            b = ref.check_certificate(_cert_doc(dense, synv, wint, cand, ec,
                                                len(wv)))["objective_optimal"]
            fboth += 1
            if a and b:
                fagree += 1
            else:
                fails.append(f"flat/{c['rung']}/{c['shot_index']}: production={a} reference={b}")
        print(f"flat certificates re-checked by BOTH checkers: "
              f"{fagree}/{fboth} accepted by both")
    else:
        print("flat certificates: addendum artifact absent -- NOT re-checked "
              "(a stranger holding this bundle cannot check flat receipts)")

    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print("RECHECK: every stored tree accepted by both checkers"
          + (f"; every one of {fagree} stored flat certificates accepted by both"
             if fboth else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
