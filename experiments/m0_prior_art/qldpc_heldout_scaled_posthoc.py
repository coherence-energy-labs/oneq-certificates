r"""The preregistered held-out campaign's BP-OSD, re-run with SCALED min-sum on its own syndromes (POST HOC).

    python experiments/m0_prior_art/qldpc_heldout_scaled_posthoc.py

WHY (paper 2 referees, 2026-11-02). The held-out campaign decoded with ldpc 2.4.1's min-sum at its default scaling
factor 1 (configuration A: 30 iterations, OSD-CS 5); on the stress rung its answers missed the certified optimum on
178 of 300 syndromes. This asks what the same decoder does with ms_scaling_factor 0.625 and nothing else changed.

WHAT IS REPLAYED, AND HOW IT IS HELD. The campaign's seed rule (qldpc_heldout_gate.heldout_seed: SHA-256 of the rung
name and the freeze commit) and its code-capacity sampler regenerate every sampled fault, so for each rung:
  * the regenerated nontrivial syndromes must equal the stored ones, in order (evidence/per_shot/heldout_<rung>.jsonl);
  * CONTROL: the campaign's own decoder (scaling 1) must reproduce every stored BP-OSD candidate, and the campaign's
    logical test must reproduce every recorded retrospective flag, BP-OSD's and the receipt pipeline's;
  * then the scaled decoder: optimum hit (feasible with the certified optimum's weight -- uniform weights, so the
    receipt answer's length) and logical error against the regenerated fault, and the paired comparison of scaled
    BP-OSD with the receipt pipeline (exact two-sided McNemar).
The heterogeneous, phenomenological and circuit rungs use other matrices and weights and are not re-run.

POST HOC and NOT preregistered; the preregistered results stand as recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "qldpc_heldout_scaled_posthoc.json"
ALPHA = 0.625
# rung -> (matrix, declared and sampled error rate) -- qldpc_heldout_gate.py's own settings
RUNGS = {"uniform_p01": ("Z", 0.01), "uniform_p02": ("Z", 0.02), "xsector_p02": ("X", 0.02),
         "stress_p05": ("Z", 0.05)}
SETTINGS = dict(max_iter=30, bp_method="ms", schedule="parallel", osd_method="osd_cs", osd_order=5)


def mcnemar(n10: int, n01: int) -> float:
    nd, k = n10 + n01, min(n10, n01)
    return min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd) if nd else 1.0


def main() -> int:
    from ldpc import BpOsdDecoder
    import ldpc
    from scipy.sparse import csr_matrix
    from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
    from qldpc_heldout_gate import heldout_seed

    HX, HZ = bb_gross_code()
    n = HZ.shape[1]
    xb, xp = gf2_row_reduce(HX % 2)
    zb, zp = gf2_row_reduce(HZ % 2)
    freeze = json.loads((ROOT / "evidence" / "qldpc_heldout.json").read_text(encoding="utf-8"))["freeze_commit"]

    def logical(side: str, f) -> bool:              # qldpc_heldout_gate's logical_z_side / logical_x_side
        H, b, p = (HZ, xb, xp) if side == "Z" else (HX, zb, zp)
        return bool(((H @ f) % 2).any()) or not gf2_in_rowspace(b, p, f, n)

    def vec(support) -> np.ndarray:
        v = np.zeros(n, dtype=np.uint8)
        v[list(support)] = 1
        return v

    out = {"schema": "oneq-qldpc-heldout-scaled-posthoc/2",
           "post_hoc": "NOT preregistered; the scaling factor, 0.625, is a value in common use with ldpc (the default "
                       "of its Sinter wrapper) and was not tuned; it was first applied to the strong-baseline data, "
                       "after the held-out results were known, and is applied here unchanged",
           "decoder_campaign": dict(SETTINGS, ms_scaling_factor=1.0),
           "decoder_scaled": dict(SETTINGS, ms_scaling_factor=ALPHA), "ldpc": ldpc.__version__,
           "freeze_commit": freeze, "rungs": {}}
    for rung, (side, rate) in RUNGS.items():
        rec_path = ROOT / "evidence" / "per_shot" / f"heldout_{rung}.jsonl"
        recs = [json.loads(x) for x in rec_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        H = HZ if side == "Z" else HX
        rng = np.random.default_rng(heldout_seed(rung, freeze))
        faults = []
        while len(faults) < len(recs):
            e = (rng.random(n) < np.full(n, rate)).astype(np.uint8)
            if ((H @ e) % 2).any():
                faults.append(e)
        dec1 = BpOsdDecoder(csr_matrix(H), error_rate=rate, ms_scaling_factor=1.0, **SETTINGS)
        decs = BpOsdDecoder(csr_matrix(H), error_rate=rate, ms_scaling_factor=ALPHA, **SETTINGS)
        t = dict(syndrome_regenerated=0, control_candidate_reproduced=0, control_bposd_flag_reproduced=0,
                 control_pipeline_flag_reproduced=0, optimum_hit_recorded=0, optimum_hit_scaled=0,
                 bposd_logical_errors_recorded=0, bposd_logical_errors_scaled=0, pipeline_logical_errors=0,
                 paired_scaled_wrong_pipeline_ok=0, paired_scaled_ok_pipeline_wrong=0)
        for r, e in zip(recs, faults):
            syn = (H @ e) % 2
            t["syndrome_regenerated"] += sorted(int(j) for j in np.flatnonzero(syn)) == sorted(r["instance"]["syndrome_support"])
            c1 = np.asarray(dec1.decode(syn.astype(np.uint8))).astype(np.uint8)
            t["control_candidate_reproduced"] += sorted(int(i) for i in np.flatnonzero(c1)) == sorted(r["bposd"]["candidate"])
            bp_err = logical(side, (vec(r["bposd"]["candidate"]) ^ e) % 2)
            pipe_err = logical(side, (vec(r["receipt"]["answer"]) ^ e) % 2)
            t["control_bposd_flag_reproduced"] += bp_err == r["retrospective_simulation"]["bposd_logical_error"]
            t["control_pipeline_flag_reproduced"] += pipe_err == r["retrospective_simulation"]["pipeline_logical_error"]
            U = len(r["receipt"]["answer"])           # uniform weights: the certified optimum is the answer's length
            if "answer_objective" in r["receipt"] and r["receipt"]["answer_objective"] != U:
                raise SystemExit(f"{rung} shot {r['shot_index']}: recorded objective != answer length")
            t["optimum_hit_recorded"] += bool(r["bposd"]["syndrome_valid"]) and r["bposd"]["objective"] == U
            cs = np.asarray(decs.decode(syn.astype(np.uint8))).astype(np.uint8)
            ok = bool(np.array_equal((H @ cs) % 2, syn))
            t["optimum_hit_scaled"] += ok and int(cs.sum()) == U
            sc_err = logical(side, (cs ^ e) % 2)
            t["bposd_logical_errors_recorded"] += bp_err
            t["bposd_logical_errors_scaled"] += sc_err
            t["pipeline_logical_errors"] += pipe_err
            t["paired_scaled_wrong_pipeline_ok"] += sc_err and not pipe_err
            t["paired_scaled_ok_pipeline_wrong"] += (not sc_err) and pipe_err
        N = len(recs)
        held = all(t[k] == N for k in ("syndrome_regenerated", "control_candidate_reproduced",
                                       "control_bposd_flag_reproduced", "control_pipeline_flag_reproduced"))
        if not held:
            raise SystemExit(f"{rung}: the replay does not reproduce the campaign ({t}); nothing written")
        t["mcnemar_scaled_vs_pipeline_two_sided_p"] = mcnemar(t["paired_scaled_wrong_pipeline_ok"],
                                                              t["paired_scaled_ok_pipeline_wrong"])
        out["rungs"][rung] = {"shots": N, "declared_rate": rate, **t,
                              "records": (rec_path.relative_to(ROOT).as_posix()
                                          if rec_path.is_relative_to(ROOT) else str(rec_path)),
                              "records_sha256": hashlib.sha256(rec_path.read_bytes()).hexdigest()}
        print(f"{rung}: optimum hit {t['optimum_hit_recorded']} -> {t['optimum_hit_scaled']} of {N}; BP-OSD logical "
              f"errors {t['bposd_logical_errors_recorded']} -> {t['bposd_logical_errors_scaled']}; pipeline "
              f"{t['pipeline_logical_errors']}; controls reproduced", flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
