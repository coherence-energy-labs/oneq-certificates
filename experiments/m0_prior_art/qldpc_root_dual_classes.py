r"""How many of paper 2's "branch-dual trees" are a single root dual, and do those duals certify BP-OSD's own answer?

    python experiments/m0_prior_art/qldpc_root_dual_classes.py

WHY (paper 2's second referee, 2026-11-02, finding M1). The producers store a FLAT proof only when the LP primal is
integral; on a shot whose primal is fractional but whose root dual already closes the lattice rule, they store a
one-node "tree" (["bound", duals]), and a BP-OSD answer of the optimal weight on that shot is filed tier BD, not
tier 1 -- although the paper's tier-1 rule (L > U - 1/D for BP-OSD's answer) is met. This counts, from the stored
proofs and the production exact checker (no LP, no solver):

  * the node histogram of every stored tree: strong baseline (per level), held-out, development;
  * for every configuration of the strong baseline (A / B / C and the post-hoc scaled A_s / B_s) and for the
    held-out stress rung: how many optimal BP-OSD answers fall on tree shots, how many of those trees are one
    node, and how many of those answers the root dual alone accepts under check_qldpc_exact.

The recorded tiers are not changed: this is the producer's classification, reported with its precondition.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq.qldpc_check import QldpcCertificate, check_qldpc_exact  # noqa: E402

V3 = ROOT / "evidence" / "strong_baseline_v3"
SC = ROOT / "evidence" / "strong_baseline_scaled"
OUT = ROOT / "evidence" / "qldpc_root_dual_classes.json"


def rows(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def nodes(t) -> int:
    return 1 + (nodes(t[2]) + nodes(t[3]) if t[0] == "branch" else 0)


def main() -> int:
    man = json.loads((V3 / "MANIFEST.json").read_text(encoding="utf-8"))
    inst = json.loads((ROOT / man["instance_z"]["file"]).read_text(encoding="utf-8"))
    checks = [tuple(c) for c in inst["checks"]]
    w = {i: Fraction(1) for i in range(inst["n_variables"])}

    def root_accepts(syn_sup, cand, duals) -> bool:
        syn = [0] * len(checks)
        for j in syn_sup:
            syn[j] = 1
        fd = [(tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(a), int(b))) for (r, F, a, b) in duals]
        return check_qldpc_exact(checks=checks, weights=w, syndrome=syn,
                                 cert=QldpcCertificate(error_support=tuple(cand), facet_duals=fd, box_duals={})).accepted

    out = {"schema": "oneq-qldpc-root-dual-classes/1", "strong_baseline": {}, "trees": {}}
    for p in ("0.01", "0.02", "0.03", "0.05", "0.07"):
        shots = {s["shot"]: s for s in rows(V3 / f"p{p}_shots.jsonl")}
        tree_proofs = {k: json.loads((V3 / "proofs" / f"{s['reference']['proof']}.json").read_text(encoding="utf-8"))
                       for k, s in shots.items() if s["reference"]["tier"] == "tierBD"}
        out["trees"][f"strong p={p}"] = dict(collections.Counter(str(nodes(d["tree"])) for d in tree_proofs.values()))
        lvl = {}
        cfg_rows = {lab: [(r["shot"], r["bposd"]["support"], r["bposd"]["optimum_hit"])
                          for r in rows(V3 / f"p{p}_{lab}.jsonl")]
                    for lab in ("baseline_30_osdcs5", "strong_100_osdcs10", "strongest_ps_100_osdcs10")}
        for lab in ("A_s", "B_s"):
            cfg_rows[lab] = [(r["shot"], r["support"], r["optimum_hit"]) for r in rows(SC / f"p{p}_{lab}.jsonl")]
        for lab, rs in cfg_rows.items():
            c = collections.Counter()
            for k, sup, hit in rs:
                if k not in tree_proofs or not hit:
                    continue
                c["optimal_on_tree_shot"] += 1
                t = tree_proofs[k]["tree"]
                if t[0] == "bound":
                    c["on_one_node_tree"] += 1
                    c["root_dual_accepts"] += root_accepts(shots[k]["syndrome_support"], sup, t[1])
            lvl[lab] = {k: c.get(k, 0) for k in ("optimal_on_tree_shot", "on_one_node_tree", "root_dual_accepts")}
        out["strong_baseline"][f"p={p}"] = lvl
    add = json.loads((ROOT / "evidence" / "qldpc_heldout_addendum.json").read_text(encoding="utf-8"))
    for key in ("bd_trees", "dev_bd_trees"):
        out["trees"][key] = dict(collections.Counter(str(nodes(x["tree"])) for x in add[key]))
    # held-out stress rung: uniform weights on the Z code-capacity instance, so the same check applies
    recs = {r["shot_index"]: r for r in rows(ROOT / "evidence" / "per_shot" / "heldout_stress_p05.jsonl")}
    c = collections.Counter()
    for x in add["bd_trees"]:
        if x["rung"] != "stress_p05":
            continue
        r = recs[x["shot_index"]]
        hit = r["bposd"]["syndrome_valid"] and r["bposd"]["objective"] == len(x["candidate"])
        c["trees"] += 1
        c["one_node"] += x["tree"][0] == "bound"
        if hit:
            c["bposd_optimal"] += 1
            if x["tree"][0] == "bound":
                c["bposd_optimal_on_one_node"] += 1
                c["root_dual_accepts_bposd"] += root_accepts(x["syndrome_support"], r["bposd"]["candidate"], x["tree"][1])
    out["heldout_stress"] = {k: c.get(k, 0) for k in ("trees", "one_node", "bposd_optimal", "bposd_optimal_on_one_node",
                                                      "root_dual_accepts_bposd")}
    # every held-out rung: on how many tree shots is BP-OSD's own answer at the proven optimum? A tree proves the
    # optimum's value, so a syndrome-valid answer of that objective is proven optimal by it, though not filed tier 1.
    # The objective scale (integer weights on the weighted rungs) is checked on every tier-1 record first.
    per = {}
    for x in add["bd_trees"]:
        rung = x["rung"]
        if rung not in per:
            rs = rows(ROOT / "evidence" / "per_shot" / f"heldout_{rung}.jsonl")
            for r in rs:
                if r["receipt"]["tier"] == "tier1" and r["bposd"]["objective"] != r["receipt"].get("answer_objective",
                                                                                                r["bposd"]["objective"]):
                    raise SystemExit(f"{rung} shot {r['shot_index']}: tier-1 objective differs from BP-OSD's")
            per[rung] = ({r["shot_index"]: r for r in rs}, collections.Counter())
        recs_r, cr = per[rung]
        r = recs_r[x["shot_index"]]
        if r["receipt"]["tier"] != "tierBD":
            raise SystemExit(f"{rung} shot {x['shot_index']}: a stored tree on a shot not filed tier BD")
        cr["trees"] += 1
        cr["one_node"] += x["tree"][0] == "bound"
        same = sorted(r["bposd"]["candidate"]) == sorted(x["candidate"])
        # the circuit records carry no syndrome_valid flag; there an answer identical to the tree's candidate, which
        # the tree checker verified against the syndrome, is valid, and any other is refused as undecidable here
        valid = r["bposd"].get("syndrome_valid", True if same else None)
        if valid is None:
            raise SystemExit(f"{rung} shot {x['shot_index']}: BP-OSD's syndrome validity is not recorded")
        opt = bool(valid) and r["bposd"]["objective"] == x["answer_weight"]
        cr["bposd_optimal"] += opt
        cr["bposd_is_tree_candidate"] += same
        if valid and r["bposd"]["objective"] < x["answer_weight"]:
            raise SystemExit(f"{rung} shot {x['shot_index']}: BP-OSD lighter than the proven optimum")
    out["heldout_trees_by_rung"] = {k: {m: c2.get(m, 0) for m in ("trees", "one_node", "bposd_optimal",
                                                                  "bposd_is_tree_candidate")}
                                    for k, (_, c2) in sorted(per.items())}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["trees"]), json.dumps(out["heldout_stress"]))
    for p, lvl in out["strong_baseline"].items():
        print(p, {k: (v["optimal_on_tree_shot"], v["on_one_node_tree"], v["root_dual_accepts"]) for k, v in lvl.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
