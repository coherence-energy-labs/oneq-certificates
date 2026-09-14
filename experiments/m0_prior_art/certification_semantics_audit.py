r"""WHAT DID PAPER 1'S "CERTIFIED OPTIMAL" ACTUALLY PROVE?

THE FINDING THAT FORCED THIS (2026-09-13). Re-running the gap-distribution
experiment at HEAD did not reproduce its own artifact: 253 of 400 d=7 shots
proven optimal, against the recorded 378. `git bisect` over a 60-shot probe
put the change at 86cbc2e (2026-08-04), the commit that closed external-audit
finding 4 -- the checker used to accept a dual whose floating-point gap was
at most 1e-6 with per-edge overload at most 1e-9, and the auditor showed any
positive tolerance certifies a correspondingly small suboptimality (1.0000005
accepted against a true optimum of 1.0). That commit said, in its own words,
that every certified-rate number was now UNVERIFIED until re-run. Every
certification artifact paper 1 quotes (million-shot, Willow generations 4-6,
stale calibration, seeded faults, gap distribution) was produced BEFORE it.

So there are two different claims, and paper 1 had been making the stronger
one with evidence for the weaker:

  * EXACT: w(J) = OPT over the IEEE-double weights, exactly.
  * EPSILON: w(J) - OPT <= eps, with eps proven.

THIS INSTRUMENT. For each headline configuration, on a seeded sample:

  1. EXPORT with the code that produced the artifact. The pinned commit's
     `src/` (and experiment) are extracted with `git archive` into a scratch
     directory and run in a SUBPROCESS whose `oneq` import is asserted to
     resolve there. Every certificate that code accepts is written out whole.
  2. JUDGE here, at HEAD, in exact rational arithmetic (fractions.Fraction;
     every double converts losslessly):
       - finiteness (the pinned checkers had no NaN guard -- audit finding,
         fixed in 10e2a75 -- so a non-finite dual would have been accepted),
       - admissibility: z >= 0, every set odd against T (boundary-aware),
         no nonzero node potentials, correction's odd set == syndrome,
       - the RIGOROUS slack: theta = min(1, min_e w_e / load_e) makes theta*z
         exactly feasible, so LB = theta * sum(z) is a true lower bound by
         weak duality and eps = w(J) - LB is PROVEN,
       - whether HEAD's exact checker accepts that same certificate,
       - whether HEAD's producer (base ladder, then escalation) certifies
         the same shot.
  3. OPTIONALLY (--oracle), P minus `exact_min_tjoin`'s weight. SUPPLEMENTARY
     ONLY: that oracle scores in Fractions but chooses with networkx, which
     breaks near-ties in float (commit 7c6d636), so it is not exact.

WHAT IT CANNOT SAY. The full-scale runs stored no certificates, so their
per-shot eps is unrecoverable; for those the ACCEPTANCE RULE is the
guarantee, and `rule_level_bound` records its ingredients per graph
(minimum edge weight, largest dual mass seen). The sample measures the
REALISED slack of the same code on the same distribution.

    python experiments/m0_prior_art/certification_semantics_audit.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import time
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "certification_semantics_audit.json"
BOUNDARY = -1

#: Each configuration, the commit whose code produced the artifact the paper
#: quotes, and how that run certified ("certify" or "certify_hard").
CONFIGS = [
    {"name": "million_shot_d5_r25", "artifact": "evidence/million_shot_gate_d5_merged.json",
     "commit": "d4a81ee", "kind": "sim", "d": 5, "rounds": 25, "p": 0.005,
     "seed": 20260913, "shots": 400, "method": "certify"},
    {"name": "gap_distribution_d3", "artifact": "evidence/gap_distribution.json",
     "commit": "cce1e91", "kind": "sim", "d": 3, "rounds": 3, "p": 0.005,
     "seed": 31337, "shots": 400, "method": "certify"},
    {"name": "gap_distribution_d5", "artifact": "evidence/gap_distribution.json",
     "commit": "cce1e91", "kind": "sim", "d": 5, "rounds": 5, "p": 0.005,
     "seed": 31337, "shots": 400, "method": "certify"},
    {"name": "gap_distribution_d7", "artifact": "evidence/gap_distribution.json",
     "commit": "cce1e91", "kind": "sim", "d": 7, "rounds": 7, "p": 0.005,
     "seed": 31337, "shots": 400, "method": "certify"},
    {"name": "willow_gen4_d5_X_r13", "artifact": "evidence/google_corpus_gate_gen4.json",
     "commit": "06344ae", "kind": "willow", "config": "d5_at_q8_7_X_r13",
     "lo": 0, "shots": 150, "method": "certify"},
    {"name": "willow_gen6_d7_X_r30", "artifact": "evidence/corpus_gen6_d7X30.json",
     "commit": "f57e87d", "kind": "willow", "config": "d7_at_q6_7_X_r30",
     "lo": 0, "shots": 40, "method": "certify_hard"},
]


# --------------------------------------------------------------------------
# EXPORT -- runs in a subprocess against the PINNED code. Imports nothing
# from HEAD's src/.
# --------------------------------------------------------------------------

def _export(cfg: dict, code_root: pathlib.Path) -> list[dict]:
    src = code_root / "src"
    sys.path.insert(0, str(src))
    import numpy as np
    import pymatching
    import stim

    import oneq
    if not pathlib.Path(oneq.__file__).resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"oneq resolved to {oneq.__file__}, not the pinned tree")
    from oneq.certifying_decoder import CertifyingDecoder

    if cfg["kind"] == "sim":
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=cfg["d"], rounds=cfg["rounds"],
            after_clifford_depolarization=cfg["p"],
            before_measure_flip_probability=cfg["p"],
            after_reset_flip_probability=cfg["p"],
            before_round_data_depolarization=cfg["p"])
        dem = circ.detector_error_model(decompose_errors=True)
        det = circ.compile_detector_sampler(seed=cfg["seed"]).sample(
            cfg["shots"], separate_observables=True)[0]
        rows = range(cfg["shots"])
    else:
        d = ROOT / "data" / "google_corpus" / cfg["config"]
        dem = stim.DetectorErrorModel.from_file(str(d / "error_model.dem"))
        nb = (dem.num_detectors + 7) // 8
        raw = np.frombuffer((d / "detection_events.b8").read_bytes(),
                            dtype=np.uint8).reshape(-1, nb)
        lo, hi = cfg["lo"], cfg["lo"] + cfg["shots"]
        det = np.unpackbits(raw[lo:hi], axis=1,
                            bitorder="little")[:, :dem.num_detectors].astype(bool)
        rows = range(hi - lo)
    m = pymatching.Matching.from_detector_error_model(dem)
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    cd = CertifyingDecoder(edges)
    certify = getattr(cd, cfg["method"])
    out = []
    for s in rows:
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in m.decode_to_edges_array(det[s]))
        r = certify(fired, corr)
        c = r.cert if r.accepted else None
        out.append({
            "shot": int(s), "fired": fired, "correction": [list(e) for e in corr],
            "accepted": bool(r.accepted), "primal_float": float(r.primal),
            "cert": None if c is None else {
                "node_potentials": {str(k): float.hex(float(v))
                                    for k, v in c.node_potentials.items()},
                "blossom_duals": [[sorted(int(x) for x in S), float.hex(float(z))]
                                  for S, z in c.blossom_duals.items()]}})
    return out


# --------------------------------------------------------------------------
# JUDGE -- HEAD, exact arithmetic.
# --------------------------------------------------------------------------

def _key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if (u, v) <= (v, u) else (v, u)


def exact_slack(W: dict, adj: dict, fired, correction, node_potentials,
                blossom_duals) -> dict:
    """The exact meaning of one certificate. Pure; no solver.

    W: canonical edge key -> Fraction weight (parallel edges collapsed to the
    minimum, as the checker does). adj: vertex -> set of incident keys.
    blossom_duals: iterable of (vertex set, value) with values as floats or
    hex strings. Returns a dict whose `eps` is a Fraction or None when the
    certificate is inadmissible (then `problems` says why).
    """
    problems: list[str] = []
    T = set(int(x) for x in fired)
    Tp = set(T) | ({BOUNDARY} if len(T) % 2 else set())

    def val(x):
        f = float.fromhex(x) if isinstance(x, str) else float(x)
        if not math.isfinite(f):
            problems.append(f"non-finite value {f!r}")
            return None
        return Fraction(f)

    for k, v in (node_potentials or {}).items():
        fv = val(v)
        if fv is not None and fv != 0:
            problems.append("nonzero node potential")
            break

    corr = [_key(int(a), int(b)) for a, b in correction]
    if any(k not in W for k in corr):
        problems.append("correction uses an edge absent from the graph")
        return {"eps": None, "problems": problems}
    P = sum((W[k] for k in corr), Fraction(0))
    deg: dict[int, int] = {}
    for a, b in corr:
        for x in (a, b):
            if x != BOUNDARY:
                deg[x] = deg.get(x, 0) + 1
    if {x for x, n in deg.items() if n % 2} != T:
        problems.append("correction odd-degree set != syndrome")

    load: dict = {}
    total = Fraction(0)
    for S, z in blossom_duals:
        S = set(int(x) for x in S)
        zf = val(z)
        if zf is None:
            continue
        if zf < 0:
            problems.append("negative dual")
        if len(S & Tp) % 2 == 0:
            problems.append("set not odd against T")
        total += zf
        seen = set()
        for x in S:
            for k in adj.get(x, ()):
                if k in seen:
                    continue
                seen.add(k)
                if (k[0] in S) != (k[1] in S):
                    load[k] = load.get(k, Fraction(0)) + zf
    if problems:
        return {"eps": None, "problems": sorted(set(problems))}
    theta = Fraction(1)
    worst_overload = Fraction(0)
    for k, L in load.items():
        if L > W[k]:
            theta = min(theta, W[k] / L)
            worst_overload = max(worst_overload, L - W[k])
    lb = theta * total
    return {"eps": P - lb, "primal": P, "dual_mass": total,
            "theta_is_one": theta == 1, "worst_overload": worst_overload,
            "problems": []}


def _graph(cfg: dict):
    import pymatching
    import stim
    if cfg["kind"] == "sim":
        circ = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=cfg["d"], rounds=cfg["rounds"],
            after_clifford_depolarization=cfg["p"],
            before_measure_flip_probability=cfg["p"],
            after_reset_flip_probability=cfg["p"],
            before_round_data_depolarization=cfg["p"])
        dem = circ.detector_error_model(decompose_errors=True)
    else:
        dem = stim.DetectorErrorModel.from_file(
            str(ROOT / "data" / "google_corpus" / cfg["config"] / "error_model.dem"))
    m = pymatching.Matching.from_detector_error_model(dem)
    return [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
             float(e[2].get("weight", 1.0))) for e in m.edges()]


def _judge(cfg: dict, exported: list[dict], oracle: bool) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from oneq.certifying_decoder import CertifyingDecoder
    from oneq.matching_cert import MatchingCertificate, check

    edges = _graph(cfg)
    W: dict = {}
    for u, v, w in edges:
        k = _key(u, v)
        if k not in W or Fraction(w) < W[k]:
            W[k] = Fraction(w)
    adj: dict = {}
    for k in W:
        for x in k:
            adj.setdefault(x, set()).add(k)
    cd = CertifyingDecoder(edges)

    acc = [e for e in exported if e["accepted"]]
    eps, problems, head_check, head_prod_base, head_prod_esc = [], {}, 0, 0, 0
    overload_seen, nonunit_theta, dual_mass_max = Fraction(0), 0, Fraction(0)
    oracle_deltas = []
    t0 = time.perf_counter()
    for e in acc:
        c = e["cert"]
        r = exact_slack(W, adj, e["fired"], e["correction"],
                        c["node_potentials"], c["blossom_duals"])
        if r["eps"] is None:
            for p in r["problems"]:
                problems[p] = problems.get(p, 0) + 1
            continue
        eps.append(r["eps"])
        overload_seen = max(overload_seen, r["worst_overload"])
        nonunit_theta += not r["theta_is_one"]
        dual_mass_max = max(dual_mass_max, r["dual_mass"])
        cert = MatchingCertificate(
            matched_edges=tuple(tuple(x) for x in e["correction"]),
            node_potentials={int(k): float.fromhex(v)
                             for k, v in c["node_potentials"].items()},
            blossom_duals={frozenset(S): float.fromhex(z)
                           for S, z in c["blossom_duals"]})
        head_check += bool(check(edges=edges, syndrome=e["fired"], cert=cert).accepted)
        corr = tuple(tuple(x) for x in e["correction"])
        rb = cd.certify(e["fired"], corr)
        head_prod_base += bool(rb.accepted)
        head_prod_esc += bool(rb.accepted or cd.escalated().certify(e["fired"], corr).accepted)
        if oracle:
            from oneq.exact_decode import exact_min_tjoin
            _c, ow, _why = exact_min_tjoin(edges, e["fired"])
            if ow is not None:
                oracle_deltas.append(r["primal"] - ow)
    secs = time.perf_counter() - t0

    w_min = min(W.values())
    fe = sorted(float(x) for x in eps)
    row = {
        "config": {k: v for k, v in cfg.items()},
        "nontrivial_sampled": len(exported),
        "accepted_by_pinned_code": len(acc),
        "pinned_rate": len(acc) / max(len(exported), 1),
        "inadmissible_certificates": problems,
        "admissible_certificates": len(eps),
        "eps_exact_max": str(max(eps)) if eps else None,
        "eps_max": fe[-1] if fe else None,
        "eps_median": fe[len(fe) // 2] if fe else None,
        "eps_exactly_zero": sum(1 for x in eps if x == 0),
        "eps_negative": sum(1 for x in eps if x < 0),
        "certificates_with_any_edge_overload": nonunit_theta,
        "largest_edge_overload": float(overload_seen),
        "head_exact_checker_accepts_same_certificate": head_check,
        "head_producer_base_ladder_certifies": head_prod_base,
        "head_producer_with_escalation_certifies": head_prod_esc,
        "rule_level_bound": {
            "gap_tolerance": 1e-6, "edge_tolerance": 1e-9,
            "min_edge_weight": float(w_min),
            "largest_dual_mass_in_sample": float(dual_mass_max),
            "note": ("for runs whose certificates were not stored, the "
                     "pinned rule implies w(J) - OPT <= 1e-6 + 1e-9 * "
                     "dual_mass / min_edge_weight, plus floating-point "
                     "summation error; with the graph's minimum weight and "
                     "the largest dual mass seen here that term is "
                     + f"{1e-9 * float(dual_mass_max) / float(w_min):.2e}")},
        "judge_seconds": secs,
    }
    if oracle:
        od = [float(x) for x in oracle_deltas]
        row["supplementary_oracle_not_exact"] = {
            "n": len(od), "exactly_equal": sum(1 for x in oracle_deltas if x == 0),
            "max_primal_minus_oracle": max(od) if od else None,
            "min_primal_minus_oracle": min(od) if od else None,
            "why_supplementary": ("exact_min_tjoin scores in Fractions but "
                                  "networkx chooses the matching with float "
                                  "tie-breaking, so it is not an exact oracle")}
    return row


def _archive(commit: str, dest: pathlib.Path) -> pathlib.Path:
    blob = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", commit, "src"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(dest, filter="data")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help=argparse.SUPPRESS)
    ap.add_argument("--code-root", help=argparse.SUPPRESS)
    ap.add_argument("--out-json", help=argparse.SUPPRESS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()

    if a.export:
        cfg = json.loads(a.export)
        res = _export(cfg, pathlib.Path(a.code_root))
        pathlib.Path(a.out_json).write_text(json.dumps(res), encoding="utf-8")
        return 0

    rows = []
    existing = {}
    if OUT.exists() and a.only:
        existing = {r["config"]["name"]: r
                    for r in json.loads(OUT.read_text(encoding="utf-8"))["rows"]}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for cfg in CONFIGS:
            if a.only and cfg["name"] not in a.only:
                if cfg["name"] in existing:
                    rows.append(existing[cfg["name"]])
                continue
            code = tmp / cfg["commit"]
            if not code.exists():
                _archive(cfg["commit"], code)
            dump = tmp / f"{cfg['name']}.json"
            t0 = time.perf_counter()
            subprocess.run([sys.executable, __file__, "--export", json.dumps(cfg),
                            "--code-root", str(code), "--out-json", str(dump)],
                           check=True, env={**os.environ, "PYTHONPATH": ""})
            exported = json.loads(dump.read_text(encoding="utf-8"))
            row = _judge(cfg, exported, a.oracle and cfg["kind"] == "sim")
            row["export_seconds"] = time.perf_counter() - t0 - row["judge_seconds"]
            rows.append(row)
            print(f"{cfg['name']:24s} pinned {row['accepted_by_pinned_code']}/"
                  f"{row['nontrivial_sampled']}  eps_max {row['eps_max']}  "
                  f"inadmissible {sum(row['inadmissible_certificates'].values())}  "
                  f"HEAD checker {row['head_exact_checker_accepts_same_certificate']}  "
                  f"HEAD producer {row['head_producer_base_ladder_certifies']}/"
                  f"{row['head_producer_with_escalation_certifies']}", flush=True)
    order = {c["name"]: i for i, c in enumerate(CONFIGS)}
    rows.sort(key=lambda r: order[r["config"]["name"]])
    doc = {"schema": "oneq-certification-semantics-audit/1",
           "how_produced": "experiments/m0_prior_art/certification_semantics_audit.py",
           "rows": rows,
           "claim_classes": {
               "exact": "w(J) = OPT over the IEEE-double weights, exactly",
               "epsilon": "w(J) - OPT <= eps with eps proven in exact arithmetic"},
           "pinned_acceptance_rule": ("float gap <= 1e-6 and per-edge overload <= 1e-9, "
                                      "no NaN guard; replaced 2026-08-04 (86cbc2e) "
                                      "after external-audit finding 4")}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"evidence -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
