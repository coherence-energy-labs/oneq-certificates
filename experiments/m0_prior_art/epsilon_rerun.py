r"""Paper 1's scale runs, re-run with a SOUND epsilon checker and EVERY receipt retained.

WHY (external audit 2026-09-14, A-01 / B1 / B3 / M5). The published 99.76%
(10^6 simulated shots) and 99.69% (Willow) were acceptances by a checker that
compared a floating-point gap with 1e-6 and had no NaN guard; the witnesses
were not kept; 74,250 simulated shots were lost to memory kills, some of them
during syndrome-dependent allocation. This run replaces all of that:

  * the ORIGINAL seed plan (250-shot chunks, seeds 1..4000 for the 10^6 run),
    every chunk retried until it exists -- no lost shots;
  * each shot certified by CertifyingDecoder(epsilon_max=1e-6), whose
    acceptance IS matching_cert_epsilon.check_epsilon: exact rationals,
    epsilon proven and recorded per shot;
  * every receipt written to a gzipped JSONL file per chunk (published as a
    release asset; too large for git), its sha256 pinned in an index that IS
    committed;
  * the forgery battery run against the epsilon checker on every shot, plus
    a non-finite class (NaN / +inf in a dual) the historical battery lacked.

The decoding graph is persisted once as a hash-bound instance, so receipts
are replayable without rebuilding it.

    python experiments/m0_prior_art/epsilon_rerun.py sim_d5_r25 --workers 6
    python experiments/m0_prior_art/epsilon_rerun.py sim_d5_r25 --merge
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EPS_MAX = Fraction(1, 10**6)
INDEX_DIR = ROOT / "evidence" / "epsilon_rerun"
RECEIPT_DIR = ROOT / "data" / "epsilon_receipts"
INSTANCE_DIR = ROOT / "evidence" / "instances"

CONFIGS = {
    "sim_d5_r25": {"kind": "sim", "d": 5, "rounds": 25, "p": 0.005,
                   "chunk": 250, "seeds": (1, 4000), "method": "certify",
                   "reproduces": "evidence/million_shot_gate_d5_merged.json (seed plan of "
                                 "million_shot_gate.py: --shots 1000000 --chunk 250 --seed-base 1)"},
}
# Willow: every configuration at the generation the paper counts it at. Ten ran
# last at generation 4 (base ladder, first 10,000 shots); d=7 over 30 rounds ran
# last at generation 6 (full ladder, first 50,000 shots). Chunk keys are window
# indices 1..n; chunk k covers shots [(k-1)*chunk, k*chunk).
for _cfg in ("d3_at_q4_5_X_r13", "d3_at_q4_5_X_r30", "d3_at_q4_5_Z_r13", "d3_at_q4_5_Z_r30",
             "d5_at_q8_7_X_r13", "d5_at_q8_7_X_r30", "d5_at_q8_7_Z_r13", "d5_at_q8_7_Z_r30",
             "d7_at_q6_7_X_r13", "d7_at_q6_7_Z_r13", "d7_at_q6_7_X_r30", "d7_at_q6_7_Z_r30"):
    _final6 = _cfg.endswith("_r30") and _cfg.startswith("d7")
    CONFIGS[f"willow_{_cfg}"] = {
        "kind": "willow", "corpus_config": _cfg, "chunk": 250,
        "seeds": (1, 200 if _final6 else 40),
        "method": "certify_hard" if _final6 else "certify",
        "reproduces": (f"evidence/corpus_gen6_d7{_cfg[10]}30.json (generation 6, full ladder)" if _final6
                       else f"evidence/corpus_gen4/{_cfg}.json (generation 4, base ladder)")}

_W: dict = {}


def _graph_sim(cfg):
    import pymatching
    import stim
    from oneq.matching_cert import BOUNDARY
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=cfg["d"], rounds=cfg["rounds"],
        after_clifford_depolarization=cfg["p"], before_measure_flip_probability=cfg["p"],
        after_reset_flip_probability=cfg["p"], before_round_data_depolarization=cfg["p"])
    m = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0)))
             for e in m.edges()]
    return circ, m, edges


def _graph(cfg):
    """(sampler-or-shots, matching, edges) for either kind of configuration."""
    if cfg["kind"] == "sim":
        return _graph_sim(cfg)
    sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
    from google_corpus_gate import CORPUS, load_config
    m, edges, shots, _meta = load_config(CORPUS / cfg["corpus_config"])
    return shots, m, edges


def instance_doc(name, edges):
    body = [[u, v, float.hex(w)] for u, v, w in edges]
    return {"schema": "oneq-matching-instance/1", "name": name, "edges": body,
            "edges_sha256": hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()}


def _init(name):
    from oneq.certifying_decoder import CertifyingDecoder
    cfg = CONFIGS[name]
    circ, m, edges = _graph(cfg)
    _W.update(name=name, cfg=cfg, circ=circ, m=m, edges=edges,
              cd=CertifyingDecoder(edges, epsilon_max=EPS_MAX))


def run_chunk(job):
    """One chunk. Never raises: a failure returns a record naming the stage."""
    name, seed = job
    try:
        return _run_chunk(name, seed)
    except BaseException as exc:                         # noqa: BLE001
        return {"seed": seed, "failed": True, "stage": _W.get("stage", "startup"),
                "reason": f"{type(exc).__name__}: {exc}"[:300]}


def _run_chunk(name, seed):
    import numpy as np
    from oneq.forgeries import assert_all_false, forgeries
    from oneq.matching_cert import BOUNDARY, MatchingCertificate
    from oneq.matching_cert_epsilon import check_epsilon

    _W["stage"] = "startup"
    if _W.get("name") != name:
        _init(name)
    cfg, m, edges, cd = _W["cfg"], _W["m"], _W["edges"], _W["cd"]
    _W["stage"] = "sampling"
    if cfg["kind"] == "sim":
        det, _obs = _W["circ"].compile_detector_sampler(seed=seed).sample(cfg["chunk"], separate_observables=True)
    else:
        lo = (seed - 1) * cfg["chunk"]
        det = _W["circ"].window(lo, lo + cfg["chunk"])
        if len(det) != cfg["chunk"]:
            raise RuntimeError(f"corpus window {lo}.. holds {len(det)} shots, not {cfg['chunk']}")
    certify = cd.certify_hard if cfg["method"] == "certify_hard" else cd.certify
    status = Counter()
    forg = Counter()
    forg_acc = Counter()
    bogus = 0
    eps_max_seen = Fraction(0)
    lines = []
    t0 = time.perf_counter()
    _W["stage"] = "certifying"
    for s in range(cfg["chunk"]):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            status["TRIVIAL"] += 1
            continue
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY)) for a, b in m.decode_to_edges_array(det[s]))
        r = certify(fired, corr)
        duals =dict(r.cert.blossom_duals) if r.cert is not None else dict(r.best_dual or {})
        cert = MatchingCertificate(matched_edges=corr, node_potentials={}, blossom_duals=duals)
        v = check_epsilon(edges=edges, syndrome=fired, cert=cert, eps_max=EPS_MAX)
        status[v.status] += 1
        if v.accepted and v.epsilon > eps_max_seen:
            eps_max_seen = v.epsilon
        lines.append(json.dumps({
            "shot": s, "fired": fired, "correction": [list(e) for e in corr],
            "duals": [[sorted(int(x) for x in S), float.hex(float(z))] for S, z in duals.items() if z > 0],
            "status": v.status, "epsilon": None if v.epsilon is None else str(v.epsilon),
            "lower_bound": None if v.lower_bound is None else str(v.lower_bound)},
            separators=(",", ":")))
        if duals:
            battery = forgeries(cd.W, fired, corr, duals, r.primal)
            some = next(iter(duals))
            battery.append(("N_nan_dual", MatchingCertificate(
                matched_edges=corr, node_potentials={}, blossom_duals={**duals, some: math.nan}),
                "non-finite"))
            battery.append(("N_inf_dual", MatchingCertificate(
                matched_edges=corr, node_potentials={}, blossom_duals={**duals, some: math.inf}),
                "non-finite"))
            bogus += len(assert_all_false(cd.W, fired, r.primal,
                                          [b for b in battery if b[2] != "non-finite"]))
            for nm, fc, _why in battery:
                forg[nm] += 1
                if check_epsilon(edges=edges, syndrome=fired, cert=fc, eps_max=EPS_MAX).accepted:
                    forg_acc[nm] += 1
    _W["stage"] = "writing"
    body = ("\n".join(lines) + "\n").encode()
    RECEIPT_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
    rp = RECEIPT_DIR / name / f"chunk_{seed:06d}.jsonl.gz"
    tmp = rp.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(body, 6, mtime=0))
    os.replace(tmp, rp)
    return {"seed": seed, "shots": cfg["chunk"], "status": dict(status),
            "epsilon_max_accepted": str(eps_max_seen),
            "forgeries": dict(forg), "forgeries_accepted": dict(forg_acc), "battery_not_false": bogus,
            "receipts_file": rp.relative_to(ROOT).as_posix(),
            "receipts_sha256": hashlib.sha256(body).hexdigest(),
            "seconds": round(time.perf_counter() - t0, 2)}


def _index_path(name):
    return INDEX_DIR / name / "chunks.jsonl"


def _done(name):
    p = _index_path(name)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rp = ROOT / r["receipts_file"]
        if rp.exists() and hashlib.sha256(gzip.decompress(rp.read_bytes())).hexdigest() == r["receipts_sha256"]:
            out[r["seed"]] = r
    return out


def run(name, workers, limit=None, passes=6):
    import multiprocessing as mp
    cfg = CONFIGS[name]
    INDEX_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
    _c, _m, edges = _graph(cfg)
    inst = instance_doc(name, edges)
    ip = INSTANCE_DIR / f"matching_{name}.json"
    if ip.exists() and json.loads(ip.read_text(encoding="utf-8"))["edges_sha256"] != inst["edges_sha256"]:
        raise SystemExit(f"{ip} disagrees with the rebuilt graph; refusing to mix instances")
    ip.write_text(json.dumps(inst, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    lo, hi = cfg["seeds"]
    seeds = list(range(lo, hi + 1))[: limit or None]
    for attempt in range(passes):
        done = _done(name)
        todo = [s for s in seeds if s not in done]
        print(f"pass {attempt + 1}: {len(done)} chunks done, {len(todo)} to run on {workers} workers", flush=True)
        if not todo:
            break
        # the full ladder's exact tier is the memory-hungry one: a fresh process per chunk
        # base-ladder Willow workers measured at ~1.6 GB private each after tens of chunks on a
        # box with ~2 GB of commit to spare: recycle them every 4 chunks
        per_child = 1 if cfg["method"] == "certify_hard" else (4 if cfg["kind"] == "willow" else 40)
        with mp.Pool(workers, maxtasksperchild=per_child) as pool, \
                open(_index_path(name), "a", encoding="utf-8", newline="\n") as idx, \
                open(INDEX_DIR / name / "failures.jsonl", "a", encoding="utf-8", newline="\n") as fl:
            for k, r in enumerate(pool.imap_unordered(run_chunk, [(name, s) for s in todo]), 1):
                (fl if r.get("failed") else idx).write(json.dumps(r) + "\n")
                (fl if r.get("failed") else idx).flush()
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)}", flush=True)
    done = _done(name)
    missing = [s for s in seeds if s not in done]
    print(f"{len(done)}/{len(seeds)} chunks complete; missing {len(missing)}")
    return 0 if not missing else 1


def merge(name, limit=None):
    cfg = CONFIGS[name]
    lo, hi = cfg["seeds"]
    seeds = list(range(lo, hi + 1))[: limit or None]
    done = _done(name)
    missing = [s for s in seeds if s not in done]
    if missing:
        raise SystemExit(f"refusing to merge: {len(missing)} chunks missing (first {missing[:5]})")
    tot = Counter()
    forg, forg_acc = Counter(), Counter()
    eps = Fraction(0)
    bogus = 0
    for s in seeds:
        r = done[s]
        tot.update(r["status"])
        forg.update(r["forgeries"])
        forg_acc.update(r["forgeries_accepted"])
        eps = max(eps, Fraction(r["epsilon_max_accepted"]))
        bogus += r["battery_not_false"]
    nontrivial = sum(v for k, v in tot.items() if k != "TRIVIAL")
    accepted = tot["EXACT_OPTIMAL"] + tot["EPSILON_OPTIMAL"]
    fails = INDEX_DIR / name / "failures.jsonl"
    doc = {"schema": "oneq-epsilon-rerun/1", "config": name, "spec": cfg,
           "eps_max": str(EPS_MAX), "chunks": len(seeds), "shots": len(seeds) * cfg["chunk"],
           "nontrivial": nontrivial, "status": dict(tot), "accepted": accepted,
           "accepted_rate": accepted / nontrivial if nontrivial else None,
           "largest_accepted_epsilon": str(eps), "largest_accepted_epsilon_float": float(eps),
           "forgeries_attempted": sum(forg.values()), "forgeries_accepted": sum(forg_acc.values()),
           "forgeries_by_class": dict(forg), "forgeries_accepted_by_class": dict(forg_acc),
           "battery_attacks_not_false": bogus,
           "chunk_failures_retried": (len(fails.read_text(encoding="utf-8").splitlines())
                                      if fails.exists() else 0),
           "instance": f"evidence/instances/matching_{name}.json",
           "receipts": f"data/epsilon_receipts/{name}/ (per-chunk sha256 in chunks.jsonl)"}
    (INDEX_DIR / name / "summary.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8",
                                                     newline="\n")
    print(json.dumps({k: doc[k] for k in ("shots", "nontrivial", "status", "accepted_rate",
                                          "largest_accepted_epsilon_float", "forgeries_attempted",
                                          "forgeries_accepted", "battery_attacks_not_false")}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None, help="first N chunks only (pilot)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge(a.config, a.limit)
    return run(a.config, a.workers, a.limit)


if __name__ == "__main__":
    raise SystemExit(main())
