r"""R177 reproduced on paper 1's own data: what PyMatching's OWN dual certifies, at what cost, and exact for what.

Sparse blossom (PyMatching 2) solves minimum-weight perfect matching by growing regions whose frozen radii form a
laminar family of T-odd sets on the detector graph: an Edmonds-Johnson cut packing, held in memory and discarded.
`third_party/pymatching_return_duals.patch` (R177, 2026-10-18) exports it (`decode_to_edges_array_with_duals`). R177
measured it on 60 simulated d=5, 25-round shots on Linux. Paper 1 may lean on it only once reproduced here, on its
own graphs, with the cost and the exactness scope stated. This driver runs the patched build in a CHILD process (it
never shadows the installed pymatching) and, per configuration and shot:

  RAW    the decoder is PyMatching on the graph's own binary64 weights, as in every run of paper 1. PyMatching solves
         an INTEGER problem (W_int = 2*round(w*c), c = (2^24 - 1)/max|w|), so its dual, divided by its normalising
         constant, is a packing that is optimal for the rounded weights and only approximately feasible for the raw
         ones; the receipt (its values rounded to binary64) is decided by check_epsilon on the RAW graph, where theta
         absorbs the overload and the proven slack is eps. The correction must equal plain PyMatching's.
  GRID   the graph's weights rounded onto a dyadic grid 2^-k and scaled to integers, so PyMatching's normalising
         constant is exactly 2 and the integer problem it solves IS the certified graph: the dual must prove the
         correction EXACTLY optimal on the rounded graph (eps_max = 0), and the same dual is then judged on the raw
         graph too.

Timing (median per shot): plain decode, decode-with-duals, region-to-cut conversion, check_epsilon.

    python experiments/m0_prior_art/pymatching_native_dual.py --patched DIR [--shots 2000]
      -> evidence/pymatching_native_dual.json
DIR holds a copy of the pymatching package whose _cpp_pymatching is the patched build.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "pymatching_native_dual.json"


def region_duals_to_cut_packing(regions):
    """Sparse blossom's frozen regions -> {vertex set: integer value} (internal units). As tools/r177_checker_bench.py.

    Each region R owns `shell` nodes reached at radii a_1 <= ... and has frozen growth r_R. For the breakpoints
    0 = t_0 < t_1 < ... < t_k < r_R the set S_j = area(descendants of R) U { n in shell(R) : arrival(n) <= t_j }
    carries z = t_{j+1} - t_j (the last one up to r_R).
    """
    children: dict[int, list[int]] = {}
    for i, (par, _src, _rad, _sn, _sa) in enumerate(regions):
        children.setdefault(par, []).append(i)
    area_cache: dict[int, frozenset] = {}

    def area(i):
        if i in area_cache:
            return area_cache[i]
        acc = set(regions[i][3])
        for c in children.get(i, ()):
            acc |= area(c)
        area_cache[i] = frozenset(acc)
        return area_cache[i]

    z: dict[frozenset, int] = {}
    for i, (_par, _src, rad, sn, sa) in enumerate(regions):
        if rad <= 0:
            continue
        inner = set()
        for c in children.get(i, ()):
            inner |= area(c)
        order = sorted(zip(sa, sn))
        cuts = sorted({t for t, _n in order if t < rad} | {0})
        for j, t in enumerate(cuts):
            t_next = cuts[j + 1] if j + 1 < len(cuts) else rad
            if t_next <= t:
                continue
            S = frozenset(inner | {n for (ta, n) in order if ta <= t})
            if S:
                z[S] = z.get(S, 0) + (t_next - t)
    return z


CHILD = r'''
import json, os, sys, time, pathlib, math
patched, root, shots, cfg_json, dll_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
if dll_dir:
    os.add_dll_directory(dll_dir)      # the MinGW C++ runtime the patched build links against
sys.path.insert(0, patched)
import pymatching
assert pathlib.Path(pymatching.__file__).resolve().is_relative_to(pathlib.Path(patched).resolve()), pymatching.__file__
sys.path.insert(1, root + "/src"); sys.path.insert(1, root + "/experiments/m0_prior_art")
import numpy as np
from fractions import Fraction
from oneq.matching_cert import MatchingCertificate
from oneq.matching_cert_epsilon import check_epsilon
from pymatching_native_dual import region_duals_to_cut_packing
B = -1
cfg = json.loads(cfg_json)
inst = json.loads(pathlib.Path(root, cfg["instance"]).read_text(encoding="utf-8"))
raw = [(int(u), int(v), float.fromhex(w)) for u, v, w in inst["edges"]]

def matching(edges):
    m = pymatching.Matching()
    for u, v, w in edges:
        if v == B:
            m.add_boundary_edge(u, weight=w, merge_strategy="disallow")
        elif u == B:
            m.add_boundary_edge(v, weight=w, merge_strategy="disallow")
        else:
            m.add_edge(u, v, weight=w, merge_strategy="disallow")
    return m

maxw = max(w for *_e, w in raw)
k = 24 - 2 - max(1, math.ceil(math.log2(maxw + 1)))
grid = 1 << k
grid_edges = [(u, v, float(max(1, round(w * grid)))) for u, v, w in raw]            # integers: PyMatching's c is 1
grid_scaled = [(u, v, w / grid) for u, v, w in grid_edges]                           # the rounded graph, weight units
arms = {"raw": (matching(raw), raw, Fraction(1)), "grid": (matching(grid_edges), grid_scaled, Fraction(grid))}
if cfg["kind"] == "sim":
    import stim
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=25,
        after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005, before_round_data_depolarization=0.005)
    det = np.concatenate([circ.compile_detector_sampler(seed=s).sample(250, separate_observables=True)[0]
                          for s in range(1, (shots + 249) // 250 + 1)])[:shots]
else:
    sys.path.insert(0, root + "/experiments/m0_prior_art")
    from google_corpus_gate import CORPUS, load_config
    _m, _e, packed, _meta = load_config(CORPUS / cfg["corpus"])
    det = packed.window(0, shots)
res = {a: {"n": 0, "same_correction": 0, "status": {}, "eps": [], "t_dec": [], "t_plain": [], "t_conv": [],
           "t_check": [], "sets": [], "raw_graph_status": {}, "raw_graph_eps": []} for a in arms}
for s in range(det.shape[0]):
    fired = np.flatnonzero(det[s]).astype(np.uint64)
    if fired.size == 0:
        continue
    F_list = [int(x) for x in fired]
    for a, (m, judge_edges, scale) in arms.items():
        g = m._matching_graph
        t0 = time.perf_counter(); e_arr, regions, nc = g.decode_to_edges_array_with_duals(fired); t1 = time.perf_counter()
        plain = g.decode_to_edges_array(fired); t2 = time.perf_counter()
        z = region_duals_to_cut_packing(regions)
        Z = {S: float(Fraction(v) / Fraction(nc) / scale) for S, v in z.items()}             # binary64 receipt values
        t3 = time.perf_counter()
        corr = tuple((int(x), int(y)) for x, y in e_arr)
        r = res[a]; r["n"] += 1
        r["same_correction"] += sorted(map(tuple, e_arr.tolist())) == sorted(map(tuple, plain.tolist()))
        cert = MatchingCertificate(matched_edges=corr, node_potentials={}, blossom_duals=Z)
        t4 = time.perf_counter()
        v = check_epsilon(edges=judge_edges, syndrome=F_list, cert=cert, eps_max=Fraction(0) if a == "grid" else Fraction(1, 10**6))
        t5 = time.perf_counter()
        r["status"][v.status] = r["status"].get(v.status, 0) + 1
        if v.epsilon is not None:
            r["eps"].append(float(v.epsilon))
        r["t_dec"].append(t1 - t0); r["t_plain"].append(t2 - t1); r["t_conv"].append(t3 - t2); r["t_check"].append(t5 - t4)
        r["sets"].append(len(Z))
        if a == "grid":                                   # the same dual, judged on the RAW graph
            w2 = check_epsilon(edges=raw, syndrome=F_list, cert=cert, eps_max=Fraction(1, 10**6))
            r["raw_graph_status"][w2.status] = r["raw_graph_status"].get(w2.status, 0) + 1
            if w2.epsilon is not None:
                r["raw_graph_eps"].append(float(w2.epsilon))
out = {}
for a, r in res.items():
    med = lambda xs: float(np.median(xs)) if xs else None
    out[a] = {"nontrivial": r["n"], "correction_equals_plain_pymatching": r["same_correction"], "status": r["status"],
              "eps_max_seen": max(r["eps"]) if r["eps"] else None, "eps_median": med(r["eps"]),
              "decode_us_median": 1e6 * med(r["t_plain"]), "decode_with_duals_us_median": 1e6 * med(r["t_dec"]),
              "convert_us_median": 1e6 * med(r["t_conv"]), "check_ms_median": 1e3 * med(r["t_check"]),
              "sets_per_receipt_median": med(r["sets"])}
    if a == "grid":
        out[a]["grid_bits"] = k
        out[a]["same_dual_on_raw_graph"] = {"status": r["raw_graph_status"],
                                            "eps_max_seen": max(r["raw_graph_eps"]) if r["raw_graph_eps"] else None,
                                            "eps_median": med(r["raw_graph_eps"])}
print(json.dumps(out))
'''

CONFIGS = {
    "sim_d5_r25": {"kind": "sim", "instance": "evidence/instances/matching_sim_d5_r25.json"},
    "willow_d5_at_q8_7_X_r30": {"kind": "willow", "corpus": "d5_at_q8_7_X_r30",
                                "instance": "evidence/instances/matching_willow_d5_at_q8_7_X_r30.json"},
    "willow_d7_at_q6_7_X_r30": {"kind": "willow", "corpus": "d7_at_q6_7_X_r30",
                                "instance": "evidence/instances/matching_willow_d7_at_q6_7_X_r30.json"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", required=True)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--dll-dir", default="", help="directory of the C++ runtime DLLs the patched build needs")
    a = ap.parse_args()
    rows = {}
    for name in a.configs:
        p = subprocess.run([sys.executable, "-B", "-c", CHILD, a.patched, str(ROOT), str(a.shots),
                            json.dumps(CONFIGS[name]), a.dll_dir], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=7200)
        if p.returncode != 0:
            raise SystemExit(f"{name}: child failed\n{p.stderr[-3000:]}")
        rows[name] = json.loads(p.stdout.strip().splitlines()[-1])
        print(name, json.dumps(rows[name]), flush=True)
    patch = (ROOT / "third_party" / "pymatching_return_duals.patch").read_bytes()
    doc = {"schema": "oneq-pymatching-native-dual/1", "configs": rows, "shots_per_config": a.shots,
           "patch_sha256": __import__("hashlib").sha256(patch.replace(b"\r\n", b"\n")).hexdigest(),
           "build": "pymatching-2.4.0 sdist (sha256 0c30dafa...2326) + the patch, MinGW-w64 g++ 15 (UCRT), "
                    "-include cstdint, against CPython " + sys.version.split()[0],
           "platform": platform.platform(),
           "sim_shots": "the 10^6-shot run's own seed plan, chunks 1.. (250 shots each)",
           "willow_shots": "the first shots of the development configuration"}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
