r"""Why paper 1's two per-call PyMatching times differ: one run, the same calls, three call contexts.

    python -B experiments/m0_prior_art/decode_call_context.py --chunks 8      # the cost run's 8 x 250 shots

WHY (external audit 4, 2026-11-07, item 1.1). Sec. IX printed PyMatching's per-call decode on the simulation's graph
twice: a mean of 132 us in the cost paragraph (evidence/certify_cost_d5_r25.json) and a median of 17 us in the
decoder-dual paragraph (evidence/pymatching_native_dual_arms.json), and called the second "comparable with other
per-call times". The two runs differ in more than the statistic: the cost run calls Matching.decode_to_edges_array on
the dense detector vector of each shot right after certifying the previous shot, while the native-dual arms call the
compiled graph object's decode_to_edges_array with the fired detectors' indices right after decoding the same shot
with the dual export. This measures, in ONE process (single thread), on the shots of the cost run (its seed plan,
chunk for chunk, and its producer), every per-shot interval with time.perf_counter around exactly the call named:

  after_certification   Matching.decode_to_edges_array(dense row): the call the cost run times, made right after the
                        previous shot's certification and epsilon check, as there
  repeated              the same call on the same row, made again at once
  binding               Matching._matching_graph.decode_to_edges_array(fired indices, uint64), made right after: the
                        call the native-dual arms time (there with the patched build; here the installed one)
  certify               CertifyingDecoder(epsilon_max=1e-6).certify on the correction of `after_certification`
  batch                 decode_batch over each chunk, divided by its shots (as the cost run)

Each is reported as n, median, mean, p90, p99 and max (the cost run's statistics, certify_cost.stats_ms). Every shot's
verdict must equal the stored receipt's (data/epsilon_receipts/sim_d5_r25/chunk_*.jsonl.gz) and every correction of
the three decode calls must be the same edge set, or the run refuses to write. Machine, versions, load before and
after, the commit and the sha256 of every judged file are recorded.
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
OUT = ROOT / "evidence" / "decode_call_context_d5_r25.json"
RECEIPTS = ROOT / "data" / "epsilon_receipts" / "sim_d5_r25"
JUDGED = ("experiments/m0_prior_art/decode_call_context.py", "experiments/m0_prior_art/certify_cost.py",
          "experiments/m0_prior_art/epsilon_rerun.py", "src/oneq/certifying_decoder.py",
          "src/oneq/matching_cert_epsilon.py")
ARMS = ("after_certification", "repeated", "binding")


def edge_set(pairs) -> frozenset:
    """A decode's correction as a set of undirected edges (the boundary is -1)."""
    return frozenset((min(int(u), int(v)), max(int(u), int(v))) for u, v in pairs)


def summarise(us: dict[str, list[float]], certify_ms: list[float], batch_us: list[float]) -> dict:
    """The record's numbers from the raw intervals; every ratio is a mean over a mean of the same run."""
    from certify_cost import stats_ms
    out = {k: stats_ms(v) for k, v in us.items()}
    mean_cert_us = 1e3 * statistics.fmean(certify_ms)
    return {"decode_per_call_us": out, "certify_ms": stats_ms(certify_ms),
            "decode_batch_us_per_shot": statistics.fmean(batch_us),
            "ratio_mean_certify_over_mean_decode": {k: mean_cert_us / out[k]["mean"] for k in out},
            "ratio_mean_certify_over_mean_batch_decode": mean_cert_us / statistics.fmean(batch_us),
            "ratio_median_after_over_median_repeated": out["after_certification"]["median"] / out["repeated"]["median"]}


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
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    cfg = er.CONFIGS["sim_d5_r25"]
    circ, m, edges = er._graph_sim(cfg)
    cd = CertifyingDecoder(edges, epsilon_max=er.EPS_MAX)
    load_before = psutil.cpu_percent(interval=2.0)
    us = {k: [] for k in ARMS}
    certify_ms, batch_us, mism = [], [], []
    shots_total = nontrivial = 0
    for seed in range(cfg["seeds"][0], cfg["seeds"][0] + a.chunks):
        det, _obs = circ.compile_detector_sampler(seed=seed).sample(cfg["chunk"], separate_observables=True)
        stored = {r["shot"]: r for r in (json.loads(x) for x in gzip.decompress(
            (RECEIPTS / f"chunk_{seed:06d}.jsonl.gz").read_bytes()).decode().splitlines() if x.strip())}
        t0 = time.perf_counter()
        m.decode_batch(det)
        batch_us.append(1e6 * (time.perf_counter() - t0) / len(det))
        for s in range(len(det)):
            shots_total += 1
            fired_idx = np.flatnonzero(det[s])
            if fired_idx.size == 0:
                continue
            nontrivial += 1
            fired64 = fired_idx.astype(np.uint64)
            t0 = time.perf_counter()
            p1 = m.decode_to_edges_array(det[s])
            t1 = time.perf_counter()
            p2 = m.decode_to_edges_array(det[s])
            t2 = time.perf_counter()
            p3 = m._matching_graph.decode_to_edges_array(fired64)
            t3 = time.perf_counter()
            us["after_certification"].append(1e6 * (t1 - t0))
            us["repeated"].append(1e6 * (t2 - t1))
            us["binding"].append(1e6 * (t3 - t2))
            if not (edge_set(p1) == edge_set(p2) == edge_set(p3)):
                mism.append(f"seed {seed} shot {s}: the three decode calls returned different corrections")
            fired = [int(i) for i in fired_idx]
            corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY)) for u, v in p1)
            t4 = time.perf_counter()
            r = cd.certify(fired, corr)
            certify_ms.append(1e3 * (time.perf_counter() - t4))
            duals = dict(r.cert.blossom_duals) if r.cert is not None else dict(r.best_dual or {})
            v = check_epsilon(edges=edges, syndrome=fired, cert=MatchingCertificate(
                matched_edges=corr, node_potentials={}, blossom_duals=duals), eps_max=er.EPS_MAX)
            rec = stored.get(s)
            if rec is None or rec["status"] != v.status or rec["fired"] != fired:
                mism.append(f"seed {seed} shot {s}: stored {None if rec is None else rec['status']} vs {v.status}")
        print(f"chunk {seed}: {nontrivial} nontrivial so far", flush=True)
    load_after = psutil.cpu_percent(interval=2.0)
    if mism:
        print("REFUSED:", *mism[:10], sep="\n  ")
        return 1

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL).stdout.strip()
    doc = {"schema": "oneq-decode-call-context/1", "config": "sim_d5_r25", "distance": cfg["d"],
           "rounds": cfg["rounds"], "p": cfg["p"], "detectors": m.num_detectors, "edges": len(edges),
           "seeds": [cfg["seeds"][0], cfg["seeds"][0] + a.chunks - 1], "shots": shots_total, "nontrivial": nontrivial,
           "same_shots_as": "evidence/certify_cost_d5_r25.json (its seed plan) and data/epsilon_receipts/sim_d5_r25 "
                            "(verdicts matched shot by shot)",
           "arms": {"after_certification": "Matching.decode_to_edges_array(dense detector row), the first call on "
                                           "the shot, right after the previous shot's certification and check",
                    "repeated": "the same call on the same row, again at once",
                    "binding": "Matching._matching_graph.decode_to_edges_array(fired detector indices, uint64), "
                               "right after"},
           "corrections_identical_across_arms": nontrivial,
           **summarise(us, certify_ms, batch_us),
           "machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                       "pymatching": pymatching.__version__, "stim": stim.__version__, "numpy": np.__version__,
                       "threads_used": 1, "cpu_percent_before": load_before, "cpu_percent_after": load_after,
                       "note": "shared workstation: other jobs ran during the measurement (cpu_percent above)"},
           "head": {"commit": git("rev-parse", "HEAD"),
                    "uncommitted_changes_in_judged_files": bool(git("status", "--porcelain", "--", *JUDGED)),
                    "sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                               for p in JUDGED}}}
    a.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("nontrivial", "ratio_mean_certify_over_mean_decode")}, indent=1))
    for k in ARMS:
        print(k, doc["decode_per_call_us"][k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
