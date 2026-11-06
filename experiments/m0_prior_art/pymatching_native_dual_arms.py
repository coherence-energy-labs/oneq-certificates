r"""The two measurements pymatching_native_dual.py did not make (paper 1, final referee, MAJOR 1, 2026-11-03).

    python -B experiments/m0_prior_art/pymatching_native_dual_arms.py --patched DIR --dll-dir DIR [--shots 2000]
      -> evidence/pymatching_native_dual_arms.json      (evidence/pymatching_native_dual.json is left as it is)

pymatching_native_dual.py ran two arms: RAW (PyMatching on the published binary64 weights, its dual divided by its
normalising constant and judged on those weights) and GRID (weights re-rounded by us onto a dyadic grid, dual judged
exactly on the re-rounded integers). Paper 1 credited the RAW arm's numbers to the GRID dual and said the RAW dual
was optimal "for PyMatching's internal rounding" without checking it. This driver measures, per shot:

  1. the RAW dual checked EXACTLY (eps_max = 0, rational arithmetic) against PyMatching's OWN integer weights,
     W_int = 2 * round(w * c), c = (2^24 - 1) / max|w| (user_graph.h iter_discretized_edges, C round: half away from
     zero), in its own units -- with the normalising constant the decoder reports checked against that formula;
  2. the GRID and RAW corrections compared edge set for edge set, and both compared with the correction the paper
     certifies for the same shot (the released receipt of the epsilon rerun), with the weight difference of every
     GRID correction that differs from RAW, exactly, on the published weights and on PyMatching's integer weights;
  3. the cost per call: decode-with-duals plus the region-to-cut conversion against a plain decode of the same
     shot, as medians over the same calls (comparable with other per-call figures, not with a mean over a batch).

The patched build runs in a CHILD process and never shadows the installed pymatching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "pymatching_native_dual_arms.json"

CHILD = r'''
import gzip, json, math, os, pathlib, sys, time
patched, root, shots, cfg_json, dll_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
if dll_dir:
    os.add_dll_directory(dll_dir)
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

def key(u, v):
    return (u, v) if u <= v else (v, u)

def c_round(x):
    """C's round() for a non-negative double: half away from zero, exactly (x - floor(x) is exact below 2**52)."""
    f = math.floor(x)
    return f + 1 if x - f >= 0.5 else f

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

maxw = max(abs(w) for *_e, w in raw)
c = float((1 << 24) - 1) / maxw                                       # double arithmetic, as user_graph.cc
W_int = {key(u, v): 2 * c_round(w * c) for u, v, w in raw}            # PyMatching's own integer weights
int_edges = [(u, v, float(W_int[key(u, v)])) for u, v, _w in raw]
W_raw = {key(u, v): Fraction(w) for u, v, w in raw}
k = 24 - 2 - max(1, math.ceil(math.log2(maxw + 1)))                   # the GRID arm of pymatching_native_dual.py
grid = 1 << k
grid_edges = [(u, v, float(max(1, round(w * grid)))) for u, v, w in raw]
m_raw, m_grid = matching(raw), matching(grid_edges)
if cfg["kind"] == "sim":
    import stim
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=25,
        after_clifford_depolarization=0.005, before_measure_flip_probability=0.005,
        after_reset_flip_probability=0.005, before_round_data_depolarization=0.005)
    det = np.concatenate([circ.compile_detector_sampler(seed=s).sample(250, separate_observables=True)[0]
                          for s in range(1, (shots + 249) // 250 + 1)])[:shots]
else:
    from google_corpus_gate import CORPUS, load_config
    _m, _e, packed, _meta = load_config(CORPUS / cfg["corpus"])
    det = packed.window(0, shots)
released = {}                                                          # global shot -> certified correction
rdir = pathlib.Path(root, "data", "epsilon_receipts", cfg["receipts"])
for s in range(1, (shots + 249) // 250 + 1):
    for line in gzip.decompress((rdir / f"chunk_{s:06d}.jsonl.gz").read_bytes()).decode().splitlines():
        r = json.loads(line)
        released[(s - 1) * 250 + r["shot"]] = (frozenset(key(int(a), int(b)) for a, b in r["correction"]), r["status"])
n = 0
nc_ok = 0
st_int = {}
eq = {"grid_eq_raw": 0, "raw_eq_released": 0, "grid_eq_released": 0, "released_compared": 0,
      "control_reoriented_still_equal": 0, "control_dropped_edge_caught": 0}
diff = {"n": 0, "heavier_on_published": 0, "lighter_on_published": 0, "equal_on_published": 0,
        "max_heavier_on_published": "0", "max_lighter_on_published": "0", "integer_weight_never_below_raw": True,
        "released_status_of_those_shots": {}}
t_plain, t_dual, t_conv, ratio = [], [], [], []
for s in range(det.shape[0]):
    fired = np.flatnonzero(det[s]).astype(np.uint64)
    if fired.size == 0:
        continue
    F = [int(x) for x in fired]
    n += 1
    g = m_raw._matching_graph
    t0 = time.perf_counter(); e_arr, regions, nc = g.decode_to_edges_array_with_duals(fired); t1 = time.perf_counter()
    plain = g.decode_to_edges_array(fired); t2 = time.perf_counter()
    z = region_duals_to_cut_packing(regions); t3 = time.perf_counter()
    t_plain.append(t2 - t1); t_dual.append(t1 - t0); t_conv.append(t3 - t2); ratio.append((t1 - t0 + t3 - t2) / (t2 - t1))
    nc_ok += nc == 2 * c
    corr = tuple(key(int(a), int(b)) for a, b in e_arr)
    v = check_epsilon(edges=int_edges, syndrome=F, eps_max=Fraction(0),
                      cert=MatchingCertificate(matched_edges=corr, node_potentials={},
                                               blossom_duals={S: float(val) for S, val in z.items()}))
    st_int[v.status] = st_int.get(v.status, 0) + 1
    raw_set = frozenset(corr)
    grid_set = frozenset(key(int(a), int(b)) for a, b in m_grid._matching_graph.decode_to_edges_array(fired))
    eq["grid_eq_raw"] += grid_set == raw_set
    if s in released:
        eq["released_compared"] += 1
        eq["raw_eq_released"] += raw_set == released[s][0]
        eq["grid_eq_released"] += grid_set == released[s][0]
        # POSITIVE CONTROLS of the comparison (final referee, 2026-11-03): the released correction with every edge
        # written the other way round must still compare equal, and with one edge dropped it must not
        flipped = frozenset(key(b, a) for a, b in released[s][0])
        eq["control_reoriented_still_equal"] += flipped == raw_set
        e0 = min(released[s][0])
        eq["control_dropped_edge_caught"] += (released[s][0] - {e0}) != raw_set and (released[s][0] - {e0}) != grid_set
    if grid_set != raw_set:
        diff["n"] += 1
        d_pub = sum((W_raw[e] for e in grid_set), Fraction(0)) - sum((W_raw[e] for e in raw_set), Fraction(0))
        d_int = sum(W_int[e] for e in grid_set) - sum(W_int[e] for e in raw_set)
        diff["integer_weight_never_below_raw"] &= d_int >= 0
        if d_pub > 0:
            diff["heavier_on_published"] += 1
            diff["max_heavier_on_published"] = str(max(Fraction(diff["max_heavier_on_published"]), d_pub))
        elif d_pub < 0:
            diff["lighter_on_published"] += 1
            diff["max_lighter_on_published"] = str(max(Fraction(diff["max_lighter_on_published"]), -d_pub))
        else:
            diff["equal_on_published"] += 1
        stt = released[s][1] if s in released else "no receipt"
        diff["released_status_of_those_shots"][stt] = diff["released_status_of_those_shots"].get(stt, 0) + 1
med = lambda xs: float(np.median(xs))
print(json.dumps({
    "nontrivial": n, "pymatching_c": c, "normalising_constant_equals_2c": nc_ok,
    "raw_dual_on_pymatching_integer_weights": st_int, "corrections": eq, "grid_differs_from_raw": diff,
    "cost_per_call_us": {"plain_decode_median": 1e6 * med(t_plain), "decode_with_duals_median": 1e6 * med(t_dual),
                         "region_to_cut_conversion_median": 1e6 * med(t_conv),
                         "with_duals_plus_conversion_median": 1e6 * med([a + b for a, b in zip(t_dual, t_conv)]),
                         "ratio_of_medians": med([a + b for a, b in zip(t_dual, t_conv)]) / med(t_plain),
                         "median_of_per_call_ratios": med(ratio)},
    "grid_bits": k}))
'''

CONFIGS = {
    "sim_d5_r25": {"kind": "sim", "instance": "evidence/instances/matching_sim_d5_r25.json", "receipts": "sim_d5_r25"},
    "willow_d5_at_q8_7_X_r30": {"kind": "willow", "corpus": "d5_at_q8_7_X_r30", "receipts": "willow_d5_at_q8_7_X_r30",
                                "instance": "evidence/instances/matching_willow_d5_at_q8_7_X_r30.json"},
    "willow_d7_at_q6_7_X_r30": {"kind": "willow", "corpus": "d7_at_q6_7_X_r30", "receipts": "willow_d7_at_q6_7_X_r30",
                                "instance": "evidence/instances/matching_willow_d7_at_q6_7_X_r30.json"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", required=True)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--dll-dir", default="")
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
    me = pathlib.Path(__file__).read_bytes()
    doc = {"schema": "oneq-pymatching-native-dual-arms/1", "configs": rows, "shots_per_config": a.shots,
           "companion": "evidence/pymatching_native_dual.json (the RAW and GRID arms' certification, left as recorded)",
           "pymatching_integer_weights": "W_int = 2*round(w*c), c = (2^24 - 1)/max|w|, C round (half away from zero)",
           "released_receipts": "data/epsilon_receipts/<config>/chunk_000001.. (250 shots each): the corrections the "
                                "paper certifies for the same shots",
           "patch_sha256": hashlib.sha256(patch.replace(b"\r\n", b"\n")).hexdigest(),
           "script_sha256": hashlib.sha256(me.replace(b"\r\n", b"\n")).hexdigest(),
           "platform": platform.platform(), "python": sys.version.split()[0]}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
