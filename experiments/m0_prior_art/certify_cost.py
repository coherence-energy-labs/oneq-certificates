r"""What certification costs, measured on the configuration paper 1 names, with every condition recorded.

    python experiments/m0_prior_art/certify_cost.py --chunks 8          # 8 x 250 shots of sim_d5_r25

WHY (paper 1 referee, 2026-11-02, finding 1). The cost the abstract quoted ("about 6,610 times batch decoding") came
from evidence/decode_vs_certify.json, which records no noise level, shot count, commit or machine; the ledger says
it reproduces only at p = 0.002, while the paper attaches it to the distance-5, 25-round graph whose experiment ran
at p = 0.005. This measures, in ONE run on ONE process (single thread), on the SAME shots the epsilon rerun
certified (its seed plan, chunk for chunk, and its producer, CertifyingDecoder(epsilon_max=1e-6).certify):

  * PyMatching batch decoding (decode_batch over each chunk, divided by its shots) and per-shot decoding;
  * certification, per nontrivial shot, with the distribution (median, mean, p90, p99, max) and the split between
    accepted and refused shots, whose cost differs;
  * checking, per shot, by the epsilon checker the results use (matching_cert_epsilon.check_epsilon), so the claim
    "checking is cheap" is measured beside the producing cost it is compared with.

Every shot's verdict must equal the stored receipt's (data/epsilon_receipts/sim_d5_r25/chunk_*.jsonl.gz), or the run
refuses to write: the cost is measured on the very computation the paper reports. The machine, the package versions,
the commit and the processor load before and after are recorded; the workstation is shared, so absolute times are
this machine's under that load, and the ratios are measured in the same run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "certify_cost_d5_r25.json"
RECEIPTS = ROOT / "data" / "epsilon_receipts" / "sim_d5_r25"


def q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats_ms(xs: list[float]) -> dict:
    return {"n": len(xs), "median": statistics.median(xs), "mean": statistics.fmean(xs),
            "p90": q(xs, 0.90), "p99": q(xs, 0.99), "max": max(xs)}


def main() -> int:
    import numpy as np
    import psutil
    import pymatching
    import stim
    import epsilon_rerun as er
    from oneq.certifying_decoder import CertifyingDecoder
    from oneq.matching_cert import BOUNDARY, MatchingCertificate
    from oneq.matching_cert_epsilon import check_epsilon

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=8)
    a = ap.parse_args()
    cfg = er.CONFIGS["sim_d5_r25"]
    circ, m, edges = er._graph_sim(cfg)
    cd = CertifyingDecoder(edges, epsilon_max=er.EPS_MAX)
    load_before = psutil.cpu_percent(interval=2.0)
    decode_batch_us, decode_shot_us, cert_ms, check_ms = [], [], [], []
    acc_ms, ref_ms = [], []
    mism = []
    shots_total = nontrivial = 0
    for seed in range(cfg["seeds"][0], cfg["seeds"][0] + a.chunks):
        det, _obs = circ.compile_detector_sampler(seed=seed).sample(cfg["chunk"], separate_observables=True)
        stored = [json.loads(x) for x in gzip.decompress((RECEIPTS / f"chunk_{seed:06d}.jsonl.gz").read_bytes())
                  .decode().splitlines() if x.strip()]
        by_shot = {r["shot"]: r for r in stored}
        t0 = time.perf_counter()
        m.decode_batch(det)
        decode_batch_us.append(1e6 * (time.perf_counter() - t0) / len(det))
        for s in range(len(det)):
            shots_total += 1
            fired = [int(i) for i in np.flatnonzero(det[s])]
            if not fired:
                continue
            nontrivial += 1
            t0 = time.perf_counter()
            pairs = m.decode_to_edges_array(det[s])
            decode_shot_us.append(1e6 * (time.perf_counter() - t0))
            corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY)) for u, v in pairs)
            t0 = time.perf_counter()
            r = cd.certify(fired, corr)
            ms = 1e3 * (time.perf_counter() - t0)
            cert_ms.append(ms)
            duals = dict(r.cert.blossom_duals) if r.cert is not None else dict(r.best_dual or {})
            cert = MatchingCertificate(matched_edges=corr, node_potentials={}, blossom_duals=duals)
            t0 = time.perf_counter()
            v = check_epsilon(edges=edges, syndrome=fired, cert=cert, eps_max=er.EPS_MAX)
            check_ms.append(1e3 * (time.perf_counter() - t0))
            (acc_ms if v.accepted else ref_ms).append(ms)
            rec = by_shot.get(s)
            if rec is None or rec["status"] != v.status or rec["fired"] != fired:
                mism.append(f"seed {seed} shot {s}: stored {None if rec is None else rec['status']} vs {v.status}")
        print(f"chunk {seed}: {nontrivial} nontrivial so far, median certify {statistics.median(cert_ms):.1f} ms",
              flush=True)
    load_after = psutil.cpu_percent(interval=2.0)
    if mism:
        print("REFUSED: verdicts differ from the stored receipts:", *mism[:10], sep="\n  ")
        return 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL).stdout.strip()
    mean_batch = statistics.fmean(decode_batch_us)
    mean_shot = statistics.fmean(decode_shot_us)
    out = {"schema": "oneq-certify-cost/1",
           "config": "sim_d5_r25", "distance": cfg["d"], "rounds": cfg["rounds"], "p": cfg["p"],
           "detectors": m.num_detectors, "edges": len(edges),
           "seeds": [cfg["seeds"][0], cfg["seeds"][0] + a.chunks - 1], "shots": shots_total,
           "nontrivial": nontrivial, "same_shots_as": "data/epsilon_receipts/sim_d5_r25 (verdicts matched shot by shot)",
           "producer": "CertifyingDecoder(epsilon_max=1e-6).certify (the base ladder, as in the epsilon rerun)",
           "checker": "matching_cert_epsilon.check_epsilon",
           "decode_batch_us_per_shot": mean_batch, "decode_per_shot_us": stats_ms(decode_shot_us),
           "certify_ms": stats_ms(cert_ms), "certify_ms_accepted": stats_ms(acc_ms),
           "certify_ms_refused": stats_ms(ref_ms) if ref_ms else None,
           "refused_share_of_certify_time": (sum(ref_ms) / sum(cert_ms)) if cert_ms else None,
           "check_ms": stats_ms(check_ms),
           "ratio_mean_certify_over_mean_batch_decode": 1e3 * statistics.fmean(cert_ms) / mean_batch,
           "ratio_mean_certify_over_mean_per_shot_decode": 1e3 * statistics.fmean(cert_ms) / mean_shot,
           "ratio_median_check_over_median_certify": statistics.median(check_ms) / statistics.median(cert_ms),
           "machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                       "pymatching": pymatching.__version__, "stim": stim.__version__, "numpy": np.__version__,
                       "threads_used": 1, "cpu_percent_before": load_before, "cpu_percent_after": load_after,
                       "note": "shared workstation: other jobs ran during the measurement (cpu_percent above)"},
           "commit": commit,
           "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                             for p in ("experiments/m0_prior_art/certify_cost.py", "src/oneq/certifying_decoder.py",
                                       "src/oneq/matching_cert_epsilon.py")}}
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("nontrivial", "decode_batch_us_per_shot", "certify_ms", "check_ms",
                                          "ratio_mean_certify_over_mean_batch_decode")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
