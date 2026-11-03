r"""The strong baseline with SCALED min-sum: BP-OSD as practitioners configure it, on the v3 shots (POST HOC).

    python experiments/m0_prior_art/qldpc_strong_baseline_scaled.py            # decode, classify, write
    python experiments/m0_prior_art/qldpc_strong_baseline_scaled.py --replay   # recompute from records + proofs only

WHY (paper 2 referee, 2026-11-02). Configurations A and B of the strong baseline call ldpc 2.4.1's min-sum with its
DEFAULT scaling factor, 1.0, which the paper never stated. On this degenerate code unscaled min-sum is badly behaved,
and measurably so on the stored syndromes: at p = 0.05 configuration A fails on 108-114 of 200 shots whatever error
rate it is told, and configuration B jumps between about 10, 83 and 109 failures as the declared rate moves by 0.005;
with the scaling factor 0.625 both fail on 7 and 9. So the v3 columns for A and B measure an unscaled decoder, not
BP-OSD as it is normally run. This adds two scaled configurations on the SAME 200 shots per level:

  A_s = A with ms_scaling_factor 0.625 (min-sum, 30 iterations, OSD-CS 5)
  B_s = B with ms_scaling_factor 0.625 (min-sum, 100 iterations, OSD-CS 10)

and classifies their answers with the v3 proofs and the production checker, exactly as tools/replay_strong_baseline.py
decides v3's: an answer is OPTIMUM-HIT when it satisfies the syndrome with the certified optimum's weight, and TIER 1
when the shot's stored flat proof, re-decided by check_qldpc_exact for THIS candidate, accepts it. No LP and no solver
run here; the classification of every answer names its proof. The same rule is re-applied to v3's own A/B/C answers
and must reproduce v3's published tier-1 counts, or this file refuses to write.

This is a POST-HOC analysis: the scaling factor was chosen after the v3 results were known, and nothing here was
preregistered. The v3 records are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

V3 = ROOT / "evidence" / "strong_baseline_v3"
OUTDIR = ROOT / "evidence" / "strong_baseline_scaled"
SUMMARY = ROOT / "evidence" / "qldpc_strong_baseline_scaled.json"
ALPHA = 0.625
NOISE = ("0.01", "0.02", "0.03", "0.05", "0.07")


def configs():
    from qldpc_strong_baseline_gate import CONFIGS
    return {"A_s": dict(CONFIGS["baseline_30_osdcs5"], ms_scaling_factor=ALPHA),
            "B_s": dict(CONFIGS["strong_100_osdcs10"], ms_scaling_factor=ALPHA)}


def rows(p: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Classifier:
    """Decides optimum-hit and tier 1 for a candidate from the v3 records and proofs, as the v3 replay does."""

    def __init__(self):
        from oneq.qldpc_check import QldpcCertificate, check_qldpc_exact
        man = json.loads((V3 / "MANIFEST.json").read_text(encoding="utf-8"))
        inst_z = json.loads((ROOT / man["instance_z"]["file"]).read_text(encoding="utf-8"))
        inst_x = json.loads((ROOT / man["instance_x"]["file"]).read_text(encoding="utf-8"))
        self.instance = man["instance_z"]["file"]
        self.n = inst_z["n_variables"]
        self.checks = [tuple(c) for c in inst_z["checks"]]
        self.HZ = np.zeros((len(self.checks), self.n), dtype=np.uint8)
        for j, c in enumerate(self.checks):
            self.HZ[j, list(c)] = 1
        HX = np.zeros((len(inst_x["checks"]), self.n), dtype=np.uint8)
        for j, c in enumerate(inst_x["checks"]):
            HX[j, list(c)] = 1
        from qldpc_bb_gate import gf2_row_reduce
        self.xb, self.xp = gf2_row_reduce(HX % 2)
        self.wx = {i: Fraction(1) for i in range(self.n)}
        self._QC, self._check = QldpcCertificate, check_qldpc_exact
        self._cache: dict = {}

    def vec(self, support) -> np.ndarray:
        v = np.zeros(self.n, dtype=np.uint8)
        v[list(support)] = 1
        return v

    def logical(self, fault) -> bool:
        from qldpc_bb_gate import gf2_in_rowspace
        return bool(((self.HZ @ fault) % 2).any()) or not gf2_in_rowspace(self.xb, self.xp, fault, self.n)

    def flat_accepts(self, pid: str, syn_sup: list[int], candidate: list[int]) -> bool:
        key = (pid, tuple(candidate))
        if key not in self._cache:
            doc = json.loads((V3 / "proofs" / f"{pid}.json").read_text(encoding="utf-8"))
            ok = False
            if doc["kind"] == "flat" and sorted(doc["syndrome_support"]) == sorted(syn_sup) \
                    and doc["instance"] == self.instance:
                syn = [0] * len(self.checks)
                for j in syn_sup:
                    syn[j] = 1
                fd = [(tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(a), int(b)))
                      for (r, F, a, b) in doc["facet_duals"]]
                ok = self._check(checks=self.checks, weights=self.wx, syndrome=syn,
                                 cert=self._QC(error_support=tuple(candidate), facet_duals=fd, box_duals={})).accepted
            self._cache[key] = bool(ok)
        return self._cache[key]

    def classify(self, shot: dict, candidate: list[int]) -> dict:
        ref = shot["reference"]
        e_true = self.vec(shot["true_error_support"])
        cand = self.vec(candidate)
        valid = bool(np.array_equal((self.HZ @ cand) % 2, (self.HZ @ e_true) % 2))
        U = None if ref["support"] is None else len(ref["support"])
        hit = bool(valid and U is not None and len(candidate) == U)
        tier1 = bool(hit and ref["tier"] == "tier2_lp_primal"
                     and self.flat_accepts(ref["proof"], shot["syndrome_support"], candidate))
        return {"support": list(candidate), "syndrome_valid": valid, "weight": len(candidate),
                "optimum_hit": hit, "tier1": tier1, "tier1_proof": ref["proof"] if tier1 else None,
                "logical_error": self.logical((cand ^ e_true) % 2)}


def tally(recs: list[dict]) -> dict:
    n = len(recs)
    t1 = sum(r["tier1"] for r in recs)
    hit = sum(r["optimum_hit"] for r in recs)
    return {"shots": n, "bp_osd_logical_errors": sum(r["logical_error"] for r in recs),
            "bp_osd_syndrome_invalid": sum(not r["syndrome_valid"] for r in recs),
            "tier1": t1, "tier1_rate": round(100 * t1 / n, 1), "optimum_hit": hit,
            "optimum_hit_rate": round(100 * hit / n, 1),
            "optimum_hit_not_tier1": sum(r["optimum_hit"] and not r["tier1"] for r in recs)}


def published_rule_check(cl: Classifier) -> dict:
    """v3's A/B/C answers under THIS file's tier-1 rule must give v3's published tier-1 counts."""
    v3 = json.loads((ROOT / "evidence" / "qldpc_strong_baseline_v3.json").read_text(encoding="utf-8"))
    out = {}
    for p in NOISE:
        shots = {s["shot"]: s for s in rows(V3 / f"p{p}_shots.jsonl")}
        for label, pub in v3["levels"][f"p={p}"]["configs"].items():
            mine = sum(cl.classify(shots[r["shot"]], r["bposd"]["support"])["tier1"]
                       for r in rows(V3 / f"p{p}_{label}.jsonl"))
            out[f"p={p} {label}"] = {"published": pub["tier1"], "this_rule": mine}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="recompute the summary from the records and proofs only")
    a = ap.parse_args()
    cl = Classifier()
    check = published_rule_check(cl)
    bad = {k: v for k, v in check.items() if v["published"] != v["this_rule"]}
    if bad:
        print("REFUSED: the tier-1 rule here does not reproduce v3's published counts:", bad)
        return 1
    cfgs = configs()
    levels = {}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in NOISE:
        shots = rows(V3 / f"p{p}_shots.jsonl")
        levels[f"p={p}"] = {}
        for label, cfg in cfgs.items():
            rec_path = OUTDIR / f"p{p}_{label}.jsonl"
            if a.replay:
                recs = rows(rec_path)
                for s, r in zip(shots, recs):
                    if r["shot"] != s["shot"] or cl.classify(s, r["support"]) != {k: v for k, v in r.items()
                                                                                    if k != "shot"}:
                        print(f"REPLAY FAILED: p={p} {label} shot {r['shot']}")
                        return 1
            else:
                from ldpc import BpOsdDecoder
                from scipy.sparse import csr_matrix
                dec = BpOsdDecoder(csr_matrix(cl.HZ), error_rate=float(p), **cfg)
                recs = []
                for s in shots:
                    syn = np.zeros(len(cl.checks), dtype=np.uint8)
                    syn[s["syndrome_support"]] = 1
                    e = np.asarray(dec.decode(syn)).astype(np.uint8)
                    recs.append({"shot": s["shot"], **cl.classify(s, [int(i) for i in np.flatnonzero(e)])})
                with open(rec_path, "w", encoding="utf-8", newline="\n") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
            files[rec_path.relative_to(ROOT).as_posix() if rec_path.is_relative_to(ROOT) else str(rec_path)] = sha(rec_path)
            levels[f"p={p}"][label] = tally(recs)
            t = levels[f"p={p}"][label]
            print(f"p={p} {label}: BP-OSD errors {t['bp_osd_logical_errors']:3d} | tier1 {t['tier1']} "
                  f"| optimum hit {t['optimum_hit']} of {t['shots']}", flush=True)
    import ldpc
    out = {"schema": "oneq-qldpc-strong-baseline-scaled/1",
           "post_hoc": "the scaling factor was chosen after the v3 results were known; nothing here was preregistered",
           "shots": "the v3 shots (evidence/strong_baseline_v3/p*_shots.jsonl), 200 nontrivial per level",
           "configs": {k: {kk: vv for kk, vv in v.items()} for k, v in cfgs.items()},
           "ldpc": ldpc.__version__,
           "classification": ("optimum hit = syndrome-valid with the certified optimum's weight; tier 1 = the shot's "
                              "stored v3 flat proof accepts THIS candidate under check_qldpc_exact; the same rule "
                              "reproduces v3's published tier-1 counts for A/B/C"),
           "published_rule_check": check, "levels": levels, "files": files,
           "replay": "python experiments/m0_prior_art/qldpc_strong_baseline_scaled.py --replay"}
    if not a.replay:
        SUMMARY.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    else:
        old = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if old["levels"] != levels or old["files"] != files:
            print("REPLAY FAILED: the recomputed summary differs from the committed one")
            return 1
        print("REPLAY PASSED: every scaled-configuration record re-classified from records and proofs alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
