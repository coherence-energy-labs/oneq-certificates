r"""All EIGHT held-out datasets of paper 2, re-decoded with BP-OSD at ms_scaling_factor 0.625 (POST HOC).

    python experiments/m0_prior_art/qldpc_heldout_scaled_posthoc_all8.py

WHY (external audits of release v2026.11.03, adjudicated 2026-11-03, finding P2-01). The held-out campaign decoded
with ldpc 2.4.1's BpOsdDecoder at its default min-sum scaling factor, 1.0. `qldpc_heldout_scaled_posthoc.py` re-ran
the four uniform-weight code-capacity datasets with 0.625 and left the heterogeneous, phenomenological and circuit
datasets out, so paper 2 could not say whether the scaling explains BP-OSD's misses there. Every stored record holds
what that question needs: the syndrome (or, on the circuit dataset, a certified answer whose syndrome it is), the
certified optimum, and BP-OSD's stored answer. This decodes all eight and scores each answer against the receipts.

HELD BY CONTROLS, all of which must pass or nothing is written:
  * each persisted instance's check hash and each integer weight vector's hash match the files;
  * the decoder's channel is rebuilt exactly as the campaign built it -- the heterogeneous channel from the seed rule
    (and it must regenerate the stored integer weights), the phenomenological spacetime system (it must equal the
    persisted instance), the circuit's detector error model (it must equal the persisted instance and regenerate the
    stored weights);
  * at ms_scaling_factor 1.0 (the campaign's setting) the decoder must reproduce EVERY stored BP-OSD candidate;
  * on the circuit dataset, whose records carry no syndrome, the syndrome is H . answer, and a fresh Stim resample at
    the recorded seed must reproduce it on every shot.
Then, per dataset and configuration: optimum hits (syndrome-valid with the certified optimum's integer objective),
answers lighter than a receipt (any would contradict it; must be 0), logical errors against the regenerated faults
(code-capacity datasets) or observable agreement with the sampled truth (circuit), and the exact two-sided McNemar
comparison of optimum hits between the two configurations on the same syndromes. No pooled claim is made.

POST HOC and NOT preregistered: the preregistered results stand as recorded, and the preregistered comparison of the
two configurations on fresh seeds is docs/PREREG_qldpc_scaling_2026-11-03.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "qldpc_heldout_scaled_posthoc_all8.json"
ALPHAS = (1.0, 0.625)
COMMON = dict(bp_method="ms", schedule="parallel", osd_method="osd_cs", osd_order=5)
# dataset -> (persisted instance, channel kind, declared rate, BP iterations, logical side for code capacity)
RUNGS = {
    "uniform_p01": ("bb144_z_code_capacity", "rate", 0.01, 30, "Z"),
    "uniform_p02": ("bb144_z_code_capacity", "rate", 0.02, 30, "Z"),
    "hetero_p02": ("bb144_z_code_capacity", "hetero", 0.02, 30, "Z"),
    "phenom_p01": ("bb144_phenom_r4", "phenom", 0.01, 30, None),
    "phenom_p02": ("bb144_phenom_r4", "phenom", 0.02, 30, None),
    "xsector_p02": ("bb144_x_code_capacity", "rate", 0.02, 30, "X"),
    "stress_p05": ("bb144_z_code_capacity", "rate", 0.05, 30, "Z"),
    "bb_refsched": ("bb144_refsched_r3_p002", "circuit", 0.002, 40, None),
}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mcnemar(n10: int, n01: int) -> float:
    nd, k = n10 + n01, min(n10, n01)
    return min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd) if nd else 1.0


def main() -> int:
    import ldpc
    import scipy
    import stim
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix
    from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
    from qldpc_bb_phenom import spacetime_system
    from qldpc_bb_refsched_gate import build_refsched_circuit
    from qldpc_circuit_gate import dem_to_matrices
    from qldpc_heldout_gate import heldout_seed

    ho = json.loads((ROOT / "evidence" / "qldpc_heldout.json").read_text(encoding="utf-8"))
    freeze, ho_rungs = ho["freeze_commit"], ho["rungs"]
    HX, HZ = bb_gross_code()
    xb, xp = gf2_row_reduce(HX % 2)
    zb, zp = gf2_row_reduce(HZ % 2)

    def to_csr(checks, n):
        r = [j for j, c in enumerate(checks) for _ in c]
        c = [i for cc in checks for i in cc]
        return csr_matrix((np.ones(len(r), dtype=np.uint8), (r, c)), shape=(len(checks), n))

    inputs = {}
    out = {"schema": "oneq-qldpc-heldout-scaled-posthoc-all8/1",
           "post_hoc": ("NOT preregistered. The held-out syndromes and receipts were known before this was run; the "
                        "scaling factor 0.625 is the default of ldpc's SinterBpOsdDecoder and was not tuned. The "
                        "preregistered comparison on fresh seeds is docs/PREREG_qldpc_scaling_2026-11-03.md."),
           "question": "per dataset, how often does BP-OSD at each scaling factor return a certified optimum?",
           "decoder_common": COMMON, "alphas": list(ALPHAS), "freeze_commit": freeze,
           "environment": {"python": platform.python_version(), "ldpc": ldpc.__version__, "numpy": np.__version__,
                           "scipy": scipy.__version__, "stim": stim.__version__},
           "rungs": {}}
    t_all = time.perf_counter()
    for rung, (inst_name, kind, p, max_iter, side) in RUNGS.items():
        ip = ROOT / "evidence" / "instances" / f"{inst_name}.json"
        wp = ROOT / "evidence" / "weights" / f"heldout_{rung}_weights.json"
        rp = ROOT / "evidence" / "per_shot" / f"heldout_{rung}.jsonl"
        inputs.update({ip.relative_to(ROOT).as_posix(): sha(ip), wp.relative_to(ROOT).as_posix(): sha(wp),
                       rp.relative_to(ROOT).as_posix(): sha(rp)})
        inst = json.loads(ip.read_text(encoding="utf-8"))
        checks = [tuple(int(i) for i in c) for c in inst["checks"]]
        n = int(inst["n_variables"])
        wdoc = json.loads(wp.read_text(encoding="utf-8"))
        wv = [int(x) for x in wdoc["vector"]]
        ctl = {"instance_hash": hashlib.sha256(json.dumps([sorted(c) for c in checks], separators=(",", ":"))
                                               .encode()).hexdigest() == inst["checks_sha256"],
               "weights_hash": hashlib.sha256(json.dumps(wv).encode()).hexdigest() == wdoc["spec"]["weights_sha256"],
               "weights_length": len(wv) == n}
        H = to_csr(checks, n)
        recs = [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines() if x.strip()]
        p_i = None
        obs_truth = resampled = obs_cols = None
        if kind == "rate":
            chan = dict(error_rate=p)
        elif kind == "hetero":
            p_i = 0.02 * np.random.default_rng(heldout_seed("hetero_p02_channel", freeze)).uniform(0.5, 2.0, n)
            chan = dict(error_channel=[float(v) for v in p_i])
            ctl["channel_seed_recorded"] = heldout_seed("hetero_p02_channel", freeze) == ho_rungs[rung]["channel_seed"]
            ctl["channel_regenerates_weights"] = wv == [
                int(math.floor(1000 * float(np.log((1 - p_i[i]) / p_i[i])) + 0.5)) for i in range(n)]
        elif kind == "phenom":
            cst, _w = spacetime_system(HZ, 4, p, p)
            ctl["spacetime_system_equals_instance"] = [tuple(sorted(c)) for c in cst] == [tuple(sorted(c)) for c in checks]
            chan = dict(error_channel=[float(p)] * n)
        else:
            dem = build_refsched_circuit(3, 0.002).detector_error_model(decompose_errors=False)
            cks, wllr, obs_rows, n_mech = dem_to_matrices(dem)
            ctl["dem_equals_instance"] = [tuple(c) for c in cks] == [tuple(c) for c in checks] and n_mech == n
            ctl["dem_regenerates_weights"] = wv == [int(math.floor(1000 * float(wllr[i]) + 0.5)) for i in range(n)]
            chan = dict(error_channel=[float(1 / (1 + np.exp(wllr[i]))) for i in range(n)])
            obs_cols = set(int(i) for i in obs_rows.get(0, ()))
            seed = ho_rungs[rung]["seed"]
            ctl["seed_follows_rule"] = seed == heldout_seed("bb_refsched", freeze) % (2 ** 31 - 1)
            dets, obs, _ = dem.compile_sampler(seed=seed).sample(shots=150, return_errors=False)
            nz = [s for s in range(dets.shape[0]) if dets[s].any()]
            ctl["resample_nontrivial_count"] = len(nz) == len(recs)
            resampled = [dets[s].astype(np.uint8) for s in nz]
            obs_truth = [int(obs[s, 0]) for s in nz]
        faults = None
        if side is not None:
            Hm = HZ if side == "Z" else HX
            p_arr = p_i if kind == "hetero" else np.full(n, p)
            rng = np.random.default_rng(heldout_seed(rung, freeze))
            faults = []
            while len(faults) < len(recs):
                e = (rng.random(n) < p_arr).astype(np.uint8)
                if ((Hm @ e) % 2).any():
                    faults.append(e)

        def logical(f):
            Hm_, b_, pv_ = (HZ, xb, xp) if side == "Z" else (HX, zb, zp)
            return bool(((Hm_ @ f) % 2).any()) or not gf2_in_rowspace(b_, pv_, f, HZ.shape[1])

        decs = {a: BpOsdDecoder(H, max_iter=max_iter, ms_scaling_factor=a, **COMMON, **chan) for a in ALPHAS}
        per = {a: {"optimum_hit": 0, "syndrome_valid": 0, "lighter_than_receipt": 0, "reproduces_stored_candidate": 0,
                   "max_excess_over_optimum": 0} for a in ALPHAS}
        for a in ALPHAS:
            if faults is not None:
                per[a]["logical_errors"] = 0
            elif obs_truth is not None:
                per[a]["observable_agreement"] = 0
        hits = {a: [] for a in ALPHAS}
        syn_ok = resample_ok = U_ok = rec_hit = 0
        pipe_err = pipe_obs = 0
        for k, r in enumerate(recs):
            ans = sorted(int(i) for i in r["receipt"]["answer"])
            va = np.zeros(n, dtype=np.uint8)
            va[ans] = 1
            if "instance" in r:
                syn = np.zeros(len(checks), dtype=np.uint8)
                syn[r["instance"]["syndrome_support"]] = 1
            else:
                syn = ((H @ va) % 2).astype(np.uint8)
                resample_ok += bool(np.array_equal(syn, resampled[k]))
            syn_ok += bool(np.array_equal((H @ va) % 2, syn))
            U = sum(wv[i] for i in ans)
            U_ok += r["receipt"].get("answer_objective", U) == U
            vb = np.zeros(n, dtype=np.uint8)
            vb[[int(i) for i in r["bposd"]["candidate"]]] = 1
            rec_hit += bool(np.array_equal((H @ vb) % 2, syn)) and r["bposd"]["objective"] == U
            if faults is not None:
                pipe_err += logical((va ^ faults[k]) % 2)
            if obs_truth is not None:
                pipe_obs += int(sum(1 for i in ans if i in obs_cols) % 2 == obs_truth[k])
            for a in ALPHAS:
                c = np.asarray(decs[a].decode(syn)).astype(np.uint8)
                sup = sorted(int(i) for i in np.flatnonzero(c))
                feas = bool(np.array_equal((H @ c) % 2, syn))
                obj = sum(wv[i] for i in sup)
                hit = feas and obj == U
                hits[a].append(hit)
                d = per[a]
                d["syndrome_valid"] += feas
                d["optimum_hit"] += hit
                d["lighter_than_receipt"] += feas and obj < U
                d["reproduces_stored_candidate"] += sup == sorted(int(i) for i in r["bposd"]["candidate"])
                if feas:
                    d["max_excess_over_optimum"] = max(d["max_excess_over_optimum"], obj - U)
                if faults is not None:
                    d["logical_errors"] += logical((c ^ faults[k]) % 2)
                if obs_truth is not None:
                    d["observable_agreement"] += int(sum(1 for i in sup if i in obs_cols) % 2 == obs_truth[k])
        N = len(recs)
        ctl["every_receipt_answer_explains_its_syndrome"] = syn_ok == N
        ctl["certified_optimum_matches_recorded_objective"] = U_ok == N
        ctl["alpha_1_reproduces_every_stored_candidate"] = per[1.0]["reproduces_stored_candidate"] == N
        if resampled is not None:
            ctl["stim_resample_reproduces_every_syndrome"] = resample_ok == N
        if not all(ctl.values()):
            raise SystemExit(f"{rung}: a control failed, nothing written: {ctl}")
        if any(per[a]["lighter_than_receipt"] for a in ALPHAS):
            raise SystemExit(f"{rung}: a decoder answer is lighter than a receipt -- a receipt is wrong; STOP")
        h1, hs = hits[1.0], hits[0.625]
        n10 = sum(1 for x, y in zip(h1, hs) if x and not y)          # hit unscaled only
        n01 = sum(1 for x, y in zip(h1, hs) if y and not x)          # hit scaled only
        ctl["stored_answers_score_as_the_decoder_reproduces"] = rec_hit == per[1.0]["optimum_hit"]
        if not all(ctl.values()):
            raise SystemExit(f"{rung}: a control failed, nothing written: {ctl}")
        out["rungs"][rung] = {
            "shots": N, "instance": inst_name, "channel": kind, "declared_rate": p, "max_iter": max_iter,
            "controls": ctl, "recorded_optimum_hits": rec_hit,
            "alpha": {str(a): per[a] for a in ALPHAS},
            "paired_optimum_hit": {"hit_at_1.0_only": n10, "hit_at_0.625_only": n01,
                                   "exact_two_sided_mcnemar_p": mcnemar(n10, n01)},
            "pipeline": ({"logical_errors": pipe_err} if faults is not None else
                         {"observable_agreement": pipe_obs} if obs_truth is not None else {})}
        print(f"{rung}: hits {per[1.0]['optimum_hit']} -> {per[0.625]['optimum_hit']} of {N} "
              f"(1.0 only {n10}, 0.625 only {n01}, McNemar p {mcnemar(n10, n01):.3g}); controls held", flush=True)
    me = pathlib.Path(__file__).resolve()
    out["code_sha256"] = {me.relative_to(ROOT).as_posix(): sha(me), **{
        f"experiments/m0_prior_art/{m}": sha(ROOT / "experiments" / "m0_prior_art" / m)
        for m in ("qldpc_bb_gate.py", "qldpc_bb_phenom.py", "qldpc_bb_refsched_gate.py", "qldpc_circuit_gate.py",
                  "qldpc_heldout_gate.py")}}
    out["inputs_sha256"] = inputs
    out["totals_per_alpha"] = {str(a): sum(v["alpha"][str(a)]["optimum_hit"] for v in out["rungs"].values())
                               for a in ALPHAS}
    out["totals_note"] = "a sum over datasets for bookkeeping, not a pooled comparison"
    out["shots_total"] = sum(v["shots"] for v in out["rungs"].values())
    out["seconds"] = round(time.perf_counter() - t_all, 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence -> {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
