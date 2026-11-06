r"""Paper 2's tiers by WHAT THE RECEIPT PROVES, recounted from the stored proofs alone.

    python experiments/m0_prior_art/qldpc_tier_by_receipt.py

WHY (external audits of release v2026.11.03, adjudicated 2026-11-03, finding P2-02). Section III defines tier 1 as
BP-OSD's own answer proven optimal by a flat dual (L > U - 1/D); the producers filed tiers by their own route. They
decided tiers 1 and 2 with the floating-point screen, whose acceptance is EXACT EQUALITY L = U, and sent every shot
whose bound only crossed the lattice level to the branch-dual path, where a dual that closes at the root is stored as
a one-node "tree". The adjudication measured the consequence: all 2,002 stored flat receipts have L = U, and all 8
one-node held-out trees have U - 1 < L < U. So the producer's labels describe its path, not its proofs.

THE CLASSIFIER, over stored receipts only (no LP, no solver, no decoder). A SINGLE DUAL is a flat certificate or a
one-node tree (["bound", duals]); a BRANCHING proof is any tree with a branch node. On each shot:
  * tier 1     -- BP-OSD's own answer is accepted by the shot's single dual under the production exact checker
                  (check_qldpc_exact): proven optimal by ONE dual, attaining (L = U) or by the lattice rule;
  * tier 2     -- the shot's single dual proves a different answer optimal and BP-OSD's answer is heavier;
  * branching  -- the optimum's value is proven by a branching tree; split by whether BP-OSD's answer attains it.
Every single-dual decision also records whether the dual attains (L = U) or crosses by the lattice rule. Every
count is reconciled against the per-shot records and the paper's existing tables before anything is written.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from oneq.qldpc_check import QldpcCertificate, _exact_safe_dual_bound, check_qldpc_exact  # noqa: E402

EV = ROOT / "evidence"
OUT = EV / "qldpc_tier_by_receipt.json"
V3 = EV / "strong_baseline_v3"
SC = EV / "strong_baseline_scaled"
RUNG_INSTANCE = {"uniform_p01": "bb144_z_code_capacity", "uniform_p02": "bb144_z_code_capacity",
                 "hetero_p02": "bb144_z_code_capacity", "phenom_p01": "bb144_phenom_r4",
                 "phenom_p02": "bb144_phenom_r4", "xsector_p02": "bb144_x_code_capacity",
                 "stress_p05": "bb144_z_code_capacity", "bb_refsched": "bb144_refsched_r3_p002"}
RUNG_ORDER = list(RUNG_INSTANCE)
STRONG_CONFIGS = {"A": "baseline_30_osdcs5", "B": "strong_100_osdcs10", "C": "strongest_ps_100_osdcs10",
                  "A_s": None, "B_s": None}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def duals_of(proof) -> list:
    return [(tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(a), int(b))) for (r, F, a, b) in proof]


class Instance:
    def __init__(self, name: str, wvec: list[int]):
        d = json.loads((EV / "instances" / f"{name}.json").read_text(encoding="utf-8"))
        self.checks = [tuple(c) for c in d["checks"]]
        self.supports = [frozenset(c) for c in self.checks]
        self.n = d["n_variables"]
        self.w = {i: Fraction(v) for i, v in enumerate(wvec)}
        if len(wvec) != self.n:
            raise SystemExit(f"{name}: weight vector length {len(wvec)} != {self.n}")

    def syndrome(self, sup) -> list[int]:
        s = [0] * len(self.checks)
        for j in sup:
            s[int(j)] = 1
        return s

    def feasible(self, cand, syn) -> bool:
        e = set(int(i) for i in cand)
        return all(len(e & s) % 2 == b for s, b in zip(self.supports, syn))

    def weight(self, cand) -> Fraction:
        return sum((self.w[int(i)] for i in cand), Fraction(0))

    def single_dual(self, syn, cand, duals) -> tuple[bool, str | None]:
        """(accepted by the exact checker, 'attaining' | 'lattice' | None)."""
        r = check_qldpc_exact(checks=self.checks, weights=self.w, syndrome=syn,
                              cert=QldpcCertificate(error_support=tuple(int(i) for i in cand),
                                                    facet_duals=tuple(duals), box_duals=()))
        if not r.accepted:
            return False, None
        L, err = _exact_safe_dual_bound(self.supports, syn, self.w, duals)
        if err is not None:
            raise SystemExit(f"accepted dual whose bound the scan refuses: {err}")
        return True, ("attaining" if L == self.weight(cand) else "lattice")


def classify(inst: Instance, syn, bp_cand, proof_kind: str, proof, proven_answer) -> dict:
    """proof_kind 'flat' (facet duals) or 'tree' (serialized tree); proven_answer: the answer the proof certifies."""
    U = inst.weight(proven_answer)
    bp_feasible = inst.feasible(bp_cand, syn)
    bp_w = inst.weight(bp_cand) if bp_feasible else None
    if bp_feasible and bp_w < U:
        raise SystemExit("BP-OSD's answer is lighter than a proven optimum: a receipt is wrong")
    single = proof_kind == "flat" or proof[0] == "bound"
    if single:
        duals = duals_of(proof if proof_kind == "flat" else proof[1])
        ok_ans, how_ans = inst.single_dual(syn, proven_answer, duals)
        if not ok_ans:
            raise SystemExit("a stored single dual does not prove its own answer")
        ok_bp, how_bp = inst.single_dual(syn, bp_cand, duals) if bp_feasible else (False, None)
        if ok_bp:
            return {"tier": "tier1", "dual": how_bp, "single_dual": True}
        if bp_feasible and bp_w == U:
            raise SystemExit("BP-OSD attains the optimum but the single dual refuses it: the bound is answer-free")
        return {"tier": "tier2", "dual": how_ans, "single_dual": True}
    return {"tier": "branching_bp_optimal" if bp_feasible and bp_w == U else "branching_repair", "dual": None,
            "single_dual": False}


def heldout() -> dict:
    flat = json.loads((EV / "qldpc_heldout_flat_certs.json").read_text(encoding="utf-8"))
    add = json.loads((EV / "qldpc_heldout_addendum.json").read_text(encoding="utf-8"))
    proofs = {}
    for c in flat["certificates"]:
        proofs[(c["rung"], c["shot_index"])] = ("flat", c["facet_duals"], c["candidate"], c["syndrome_support"])
    for t in add["bd_trees"]:
        syn = t["syndrome_support"] if isinstance(t["syndrome_support"], list) else json.loads(t["syndrome_support"])
        proofs[(t["rung"], t["shot_index"])] = ("tree", t["tree"], t["candidate"], syn)
    out = {}
    for rung in RUNG_ORDER:
        wvec = [int(v) for v in json.loads((EV / "weights" / f"heldout_{rung}_weights.json")
                                           .read_text(encoding="utf-8"))["vector"]]
        inst = Instance(RUNG_INSTANCE[rung], wvec)
        recs = rows(EV / "per_shot" / f"heldout_{rung}.jsonl")
        c = collections.Counter()
        for r in recs:
            kind, proof, cand, syn_sup = proofs[(rung, r["shot_index"])]
            if sorted(int(i) for i in cand) != sorted(int(i) for i in r["receipt"]["answer"]):
                raise SystemExit(f"{rung}#{r['shot_index']}: stored proof's answer is not the record's answer")
            syn = inst.syndrome(syn_sup)
            got = classify(inst, syn, r["bposd"]["candidate"], kind, proof, cand)
            c[got["tier"]] += 1
            c[f"recorded_{r['receipt']['tier']}"] += 1
            if got["single_dual"]:
                c[f"single_dual_{got['dual']}"] += 1
            c[f"recorded_{r['receipt']['tier']}->{got['tier']}"] += 1
        c["N"] = len(recs)
        out[rung] = dict(sorted(c.items()))
    return out


def strong() -> dict:
    zw = [1] * 144
    inst = Instance("bb144_z_code_capacity", zw)
    out = {}
    for p in ("0.01", "0.02", "0.03", "0.05", "0.07"):
        shots = {s["shot"]: s for s in rows(V3 / f"p{p}_shots.jsonl")}
        proofs = {k: json.loads((V3 / "proofs" / f"{s['reference']['proof']}.json").read_text(encoding="utf-8"))
                  for k, s in shots.items()}
        lvl = {}
        for lab, fname in STRONG_CONFIGS.items():
            if fname is not None:
                rs = [(r["shot"], r["bposd"]["support"], r["pipeline"]["tier"] == "tier1")
                      for r in rows(V3 / f"p{p}_{fname}.jsonl")]
            else:
                rs = [(r["shot"], r["support"], bool(r["tier1"])) for r in rows(SC / f"p{p}_{lab}.jsonl")]
            c = collections.Counter()
            for k, sup, recorded_t1 in rs:
                pr = proofs[k]
                syn = inst.syndrome(shots[k]["syndrome_support"])
                ans = pr.get("candidate", shots[k]["reference"]["support"])   # the answer the proof proves
                got = classify(inst, syn, sup, pr["kind"], pr["facet_duals"] if pr["kind"] == "flat" else pr["tree"],
                               ans)
                c[got["tier"]] += 1
                c["recorded_tier1"] += recorded_t1
                if got["single_dual"]:
                    c[f"single_dual_{got['dual']}"] += 1
                if recorded_t1 and got["tier"] != "tier1":
                    raise SystemExit(f"p={p} {lab} shot {k}: recorded tier 1 is not tier 1 by its receipt")
            c["N"] = len(rs)
            c["optimum_hit"] = c["tier1"] + c["branching_bp_optimal"]
            lvl[lab] = dict(sorted(c.items()))
        out[f"p={p}"] = lvl
    return out


def main() -> int:
    out = {"schema": "oneq-qldpc-tier-by-receipt/1",
           "definition": ("tier1: BP-OSD's own answer accepted by the shot's single dual (a flat certificate or a "
                          "one-node tree) under check_qldpc_exact; tier2: the single dual proves another answer and "
                          "BP-OSD's is heavier; branching_bp_optimal / branching_repair: the optimum is proven by a "
                          "branching tree and BP-OSD's answer attains it / is heavier. single_dual_attaining: L = U; "
                          "single_dual_lattice: U - 1/D < L < U."),
           "heldout": heldout(), "strong_baseline": strong()}
    tot = collections.Counter()
    for v in out["heldout"].values():
        for k in ("N", "tier1", "tier2", "branching_bp_optimal", "branching_repair", "single_dual_attaining",
                  "single_dual_lattice"):
            tot[k] += v.get(k, 0)
    out["heldout_totals"] = dict(tot)
    if tot["tier1"] + tot["tier2"] + tot["branching_bp_optimal"] + tot["branching_repair"] != tot["N"]:
        raise SystemExit("held-out tiers do not partition the shots")
    me = pathlib.Path(__file__).resolve()
    out["code_sha256"] = {me.relative_to(ROOT).as_posix(): sha(me),
                          "src/oneq/qldpc_check.py": sha(ROOT / "src" / "oneq" / "qldpc_check.py")}
    inputs = [EV / "qldpc_heldout_flat_certs.json", EV / "qldpc_heldout_addendum.json", V3 / "MANIFEST.json"]
    inputs += [EV / "per_shot" / f"heldout_{r}.jsonl" for r in RUNG_ORDER]
    out["inputs_sha256"] = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["heldout_totals"]))
    for p, lvl in out["strong_baseline"].items():
        print(p, {k: (v.get("recorded_tier1", 0), v.get("tier1", 0), v["optimum_hit"]) for k, v in lvl.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
