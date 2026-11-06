"""Driver for docs/PREREG_superactivation_twirl_followup_2026-10-12.md: the depth arm of the first
hardware run with Pauli gate twirling ON. Reuses hw_flagged_channel's circuits and estimators.

  python hw_twirl_followup.py              validate (submits nothing)
  python hw_twirl_followup.py --submit --i-understand-this-spends-credits   (ONEQ_IBM_SUBMIT=1)
  python hw_twirl_followup.py --fetch
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("hw_flagged_channel", HERE.parent / "hw_flagged_channel.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

PREREG = ROOT / "docs" / "PREREG_superactivation_twirl_followup_2026-10-12.md"
PREREG_SHA256 = "6135a2ec550b1ab1169e5751b5a676409580970ece86e9e6ab1452c58cec16da"
NAMES = [f"pauli_I_{b}" for b in base.BASES] + [f"depth_{m}_{b}" for m in base.DEPTHS for b in base.BASES]
RANDOMIZATIONS = 32
UNTWIRLED_EPS = 0.046454902145618315          # job db6pkeqmb58s738c6h80, raw


def prereg_ok() -> bool:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


def analyse(counts: dict) -> dict:
    F = {0: base.bell_fidelity(counts["pauli_I_XX"], counts["pauli_I_YY"], counts["pauli_I_ZZ"])}
    for m in base.DEPTHS:
        F[m] = base.bell_fidelity(counts[f"depth_{m}_XX"], counts[f"depth_{m}_YY"], counts[f"depth_{m}_ZZ"])
    lam = {m: ((4 * f - 1) / 3, 4 * s / 3) for m, (f, s) in F.items()}
    ms = np.array(sorted(lam))
    slope = float(np.polyfit(ms, np.log([lam[m][0] for m in ms]), 1)[0])
    eps_pooled = 1 - math.exp(slope)
    per = {}
    l0, s0 = lam[0]
    for m in base.DEPTHS:
        lm, sm = lam[m]
        r = lm / l0
        e = 1 - r ** (1 / m)
        # delta method on e(r) = 1 - r^(1/m), r = lm/l0
        sr = r * math.hypot(sm / lm, s0 / l0)
        per[m] = (e, (r ** (1 / m - 1)) / m * sr)
    pairs = [(a, b) for i, a in enumerate(base.DEPTHS) for b in base.DEPTHS[i + 1:]]
    t1 = all(abs(per[a][0] - per[b][0]) <= 2 * math.hypot(per[a][1], per[b][1]) for a, b in pairs)
    return {
        "F": {str(m): {"F": f, "se": s} for m, (f, s) in F.items()},
        "eps_per_depth": {str(m): {"eps": e, "se": s} for m, (e, s) in per.items()},
        "eps_CZ_pooled": eps_pooled,
        "delta0": 1 - l0,
        "untwirled_eps_CZ": UNTWIRLED_EPS,
        "predictions": {
            "T1_per_depth_estimates_agree_2se": bool(t1),
            "T2_eps_in_[0.001,0.010]": bool(0.001 <= eps_pooled <= 0.010),
            "T3_eps_below_untwirled_over_3": bool(eps_pooled < UNTWIRLED_EPS / 3),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--i-understand-this-spends-credits", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        jobs = sorted(base.RECORDS.glob("twirl_*.job.json"))
        rec = json.loads(jobs[-1].read_text(encoding="utf-8"))
        svc, _ = base._service_backend()
        job = svc.job(rec["job_id"])
        if "DONE" not in str(job.status()):
            print(f"job {rec['job_id']}: {job.status()}")
            return 4
        result = job.result()
        counts = {name: result[i].data.c.get_counts() for i, name in enumerate(rec["circuit_names"])}
        spans = result.metadata.get("execution", {}).get("execution_spans")
        out = {"job": rec, "counts": counts, "analysis": analyse(counts),
               "execution_seconds": (spans.stop - spans.start).total_seconds() if spans else None}
        dest = base.RECORDS / f"twirl_{rec['job_id']}.result.json"
        dest.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
        print(json.dumps(out["analysis"], indent=2))
        print(f"-> {dest}")
        return 0
    svc, backend = base._service_backend()
    bad = base._layout_health(backend)
    u = svc.usage()
    remaining = (u.get("usage_limit_seconds") or 0) - (u.get("usage_consumed_seconds") or 0)
    print(f"prereg hash matches: {prereg_ok()} | layout: {'OK' if not bad else bad} | quota remaining {remaining} s")
    if not a.submit:
        return 0 if (prereg_ok() and not bad) else 3
    if not (a.i_understand_this_spends_credits and os.environ.get("ONEQ_IBM_SUBMIT") == "1"):
        print("REFUSED: needs --i-understand-this-spends-credits AND ONEQ_IBM_SUBMIT=1")
        return 3
    if not prereg_ok() or bad or remaining < base.MIN_QUOTA_S:
        print("REFUSED: prereg hash, layout health or quota")
        return 3
    from qiskit_ibm_runtime import SamplerV2
    names_all, isa_all = base._transpile(backend)
    isa = [isa_all[names_all.index(n)] for n in NAMES]
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = base.SHOTS
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = False
    sampler.options.twirling.num_randomizations = RANDOMIZATIONS
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run(isa)
    rec = {"prereg": PREREG.name, "prereg_sha256": PREREG_SHA256, "backend": backend.name,
           "layout": base.LAYOUT, "shots": base.SHOTS, "twirling": {"gates": True, "measure": False,
           "num_randomizations": RANDOMIZATIONS}, "circuit_names": NAMES, "job_id": job.job_id(),
           "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "quota_remaining_before_s": remaining}
    base.RECORDS.mkdir(parents=True, exist_ok=True)
    p = base.RECORDS / f"twirl_{job.job_id()}.job.json"
    p.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(f"submitted {len(isa)} twirled circuits: job {job.job_id()} -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
