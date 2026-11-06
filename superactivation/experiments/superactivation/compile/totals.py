"""Deliverable 5: per-run totals and the INFEASIBILITY FACTOR, from the verified component counts.

Inputs (all produced by the scripts in this directory):
  verify_channels.json  per-use block counts (compiled, Choi-verified) for every controller draw
  verify_encoder*.json  encoder CZ (n=17 compiled + statevector-verified; n=24 from --n24 run)
  decoder_counts_n*.json per-branch decoder costs (component formula == compiled count, verified
                        at n=3..6 for every branch and mode in verify_decoder.json)
  full_run_n*.json      whole-shot circuits for sampled draws, logical and heavy-hex routed
Error model (as specified): first-order sum, eps = 0.0063 x (number of CZ); budget delta = margin /
(F - 1/4) of the code being run (here: the isometric encoder + fidelity-equivalent decoders, whose
float fidelity is recomputed in decoder_counts). Factor = eps / delta.

Usage: python totals.py
"""
import json
import math
import os

import numpy as np

EPS_CZ = 0.0063
p = math.sqrt(2) - 1
WP = [(1 - p) / 4] * 4 + [p / 2] * 2

vc = json.load(open("verify_channels.json"))
rows = {r["name"]: r for r in vc["rows"]}


def block_stats(name):
    cnt = rows[name]["compiled_counts"]
    cz = [c.get("cz", 0) + c.get("cond:cz", 0) for c in cnt]
    meas = [c.get("measure", 0) + c.get("cond:measure", 0) for c in cnt]
    ff = [c.get("feedforward_blocks", 0) for c in cnt]
    if len(cnt) == 12:   # randomized: order jP 0..5 x erased 0,1
        w = [WP[j] * 0.5 for j in range(6) for _ in (0, 1)]
        return {"cz_expected": float(np.dot(w, cz)), "cz_worst": max(cz), "meas_expected": float(np.dot(w, meas)),
                "ff_expected": float(np.dot(w, ff)), "per_draw_cz": cz}
    c = cnt[0]           # coherent: one dynamic circuit; unconditional + one of the two flag branches
    uncond = c.get("cz", 0)
    cond = c.get("cond:cz", 0)          # max over the if/else bodies = the z=0 body (z=1 body has none)
    return {"cz_expected": uncond + 0.5 * cond, "cz_worst": uncond + cond,
            "meas_expected": c.get("measure", 0) + 0.5 * c.get("cond:measure", 0),
            "ff_expected": c.get("feedforward_blocks", 0)}


blocks = {
    "R_opt": block_stats("block RANDOMIZED channels, D~=opt"),
    "R_coherentD": block_stats("block RANDOMIZED channels, D~=coherent"),
    "R_paper": block_stats("block RANDOMIZED channels, D~=paper (U on s2_first)"),
    "C_opt": block_stats("block COHERENT channels, D~=opt"),
    "C_paper": block_stats("block COHERENT channels, D~=paper (U on s2_first)"),
}

enc = {}
for fn in ("verify_encoder.json", "verify_encoder_n24.json"):
    if os.path.exists(fn):
        for r in json.load(open(fn))["partC"]:
            enc[r["n"]] = r
OUT = {"eps_per_cz": EPS_CZ, "blocks": blocks, "n": {}}
for n in (17, 24):
    dc = json.load(open(f"decoder_counts_n{n}.json"))["summary"]
    F = dc["F_isometric_encoder_restricted_decoders"]
    cf = f"certificate_compiled_n{n}.json"
    F_cert = json.load(open(cf))["certified_lower_bound_float"] if os.path.exists(cf) else None
    Fb = F_cert if F_cert is not None else F          # budget from the CERTIFIED bound when available
    delta = (Fb - 0.75) / (Fb - 0.25)
    if n in enc:
        enc_cz = enc[n]["cz_compiled"]
        enc_src = f"compiled + statevector-verified (err {enc[n]['statevector_err']:.1e})"
    else:
        import dicke as Dk
        enc_cz = 5 * n - 5 + Dk.dicke_unitary_cz(n)
        enc_src = "closed form (W 5n-5 + U 2(n-1)^2), statevector check pending"
    bell = 2   # Bell pair (R, input) + Bell measurement (R, out)
    res = {"F_code_float": F, "F_code_certified_lower": F_cert, "budget_delta": delta, "encoder_cz": enc_cz, "encoder_source": enc_src,
           "decoder_cz_expected": dc["decoder_cz_expected"], "decoder_cz_worst": dc["decoder_cz_worst"],
           "profiles": {}}
    for prof, bk in (("RANDOMIZED channels + measurement-based D~", "R_opt"),
                     ("RANDOMIZED channels + measurement-free D~", "R_coherentD"),
                     ("RANDOMIZED channels + paper D~ (coherent untwist)", "R_paper"),
                     ("COHERENT (Stinespring) channels + measurement-based D~", "C_opt"),
                     ("COHERENT (Stinespring) channels + paper D~", "C_paper")):
        b = blocks[bk]
        exp_cz = bell + enc_cz + n * b["cz_expected"] + dc["decoder_cz_expected"]
        worst_cz = bell + enc_cz + n * b["cz_worst"] + dc["decoder_cz_worst"]
        eps = EPS_CZ * exp_cz
        res["profiles"][prof] = {
            "cz_expected": exp_cz, "cz_worst": worst_cz,
            "share": {"encoder": enc_cz / exp_cz, "channel_uses": n * b["cz_expected"] / exp_cz,
                      "decoder": dc["decoder_cz_expected"] / exp_cz, "bell": bell / exp_cz},
            "eps_first_order": eps, "infeasibility_factor": eps / delta,
            "infeasibility_factor_worst": EPS_CZ * worst_cz / delta,
            "mid_circuit_measurements_expected": n * b["meas_expected"] + dc["decoder_measurements_expected"],
        }
    floor = (n - 1) * EPS_CZ / delta
    res["previous_floor_factor_(n-1)CZ"] = floor
    # routed (heavy-hex, ibm_kingston snapshot) from sampled whole shots: Sabre layout+routing of the
    # all-to-all circuit, and (_lnn) the nearest-neighbour encoder on a heavy-hex path + nearest-free ancillas
    for tag, fr in (("sampled_shots", f"full_run_n{n}.json"), ("sampled_shots_lnn", f"full_run_n{n}_lnn.json")):
        if not os.path.exists(fr):
            continue
        S = json.load(open(fr))["samples"]
        lg = np.array([s["logical"]["cz"] for s in S], float)
        rt = np.array([s["routed_heavyhex"]["cz"] for s in S], float) if "routed_heavyhex" in S[0] else None
        res[tag] = {"count": len(S), "logical_cz_mean": lg.mean(), "logical_cz_sd": lg.std(ddof=1),
                    "logical_2q_depth_mean": float(np.mean([s["depth_2q"] for s in S])),
                    "measurements_mean": float(np.mean([s["logical"]["measure"] for s in S]))}
        if rt is not None:
            ratio = rt / lg
            res[tag].update({
                "routed_cz_mean": rt.mean(), "routed_cz_sem": rt.std(ddof=1) / math.sqrt(len(rt)),
                "routed_over_logical_mean": ratio.mean(),
                "routed_depth_mean": float(np.mean([s["routed_heavyhex"]["depth"] for s in S])),
                "routed_2q_depth_mean": float(np.mean([s["routed_heavyhex"]["depth_2q"] for s in S])),
                "routed_infeasibility_factor": EPS_CZ * rt.mean() / delta})
    OUT["n"][n] = res

json.dump(OUT, open("totals.json", "w"), indent=1, default=str)
for n, r in OUT["n"].items():
    print(f"=== n = {n}: F = {r['F_code_float']:.10f}, budget delta = {r['budget_delta']:.3e}, encoder {r['encoder_cz']} CZ"
          f" ({r['encoder_source']}), decoder E {r['decoder_cz_expected']:.1f} / worst {r['decoder_cz_worst']}")
    for prof, x in r["profiles"].items():
        print(f"  {prof:58s} CZ E {x['cz_expected']:8.1f} worst {x['cz_worst']:6d}  eps {x['eps_first_order']:6.2f}"
              f"  factor {x['infeasibility_factor']:9.0f} (worst {x['infeasibility_factor_worst']:9.0f})"
              f"  meas E {x['mid_circuit_measurements_expected']:.1f}")
    print(f"  previous floor ((n-1) CZ, encoder only): factor {r['previous_floor_factor_(n-1)CZ']:.0f}")
    for tag in ("sampled_shots", "sampled_shots_lnn"):
        if tag in r:
            print(f"  {tag}:", json.dumps(r[tag]))
