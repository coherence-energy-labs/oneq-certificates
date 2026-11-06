r"""How wrong were paper 1's published Sec. VIII numbers? Re-run both published gap artifacts, score both ways.

The published Table V came from `evidence/gap_distribution.json` (2026-07-31, producer at cce1e91) and Fig. 3 from
`evidence/gap_distribution_exact_rule.json` (2026-11-02). Both used the scorer that charged an edge once per selected
BP-OSD component (external audit 2, F01). Neither stored the selections, so the overcount cannot be read back from
them. This re-runs each experiment as it was run -- seed 31337, the released BP-OSD and belief-matching settings, the
producer that decided each artifact's reference subset (cce1e91 extracted with `git archive`, or this tree's) --
stores every selection, and FIRST requires the old scoring to reproduce the stored artifact exactly (every summary
statistic equal, and for the 2026-11-02 artifact every stored per-shot gap bit for bit). Only then does it report how
many shots were overcounted and what the scores become once the selection is projected onto edges mod 2, in exact
rationals.

    python experiments/m0_prior_art/gap_distribution_original_rescore.py
      -> evidence/gap_distribution_v4/original_rescored.json
"""
from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "gap_distribution_v4" / "original_rescored.json"
#: artifact -> (file, the commit whose src/oneq decided its reference subset, the commit whose script scored it)
ARTIFACTS = {"table_v_2026-07-31": ("evidence/gap_distribution.json", "cce1e91", "cce1e91"),
             "fig3_2026-11-02": ("evidence/gap_distribution_exact_rule.json", "624d41f2", "624d41f2")}


def summarise(v):
    """Verbatim the published script's summary (gap_distribution.py at cce1e91 and at 624d41f2)."""
    if not v:
        return None
    arr = np.array(v)
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)), "max": float(arr.max()),
            "frac_zero": float((np.abs(arr) < 1e-9).mean()), "frac_below_zero": float((arr < -1e-9).mean())}


CHILD = r'''
import json, sys, pathlib, importlib.util
src, scorer = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
import numpy as np, pymatching, stim
from collections import Counter
from fractions import Fraction
from scipy.sparse import csc_matrix
from ldpc import BpOsdDecoder
import oneq.certifying_decoder as ocd
assert pathlib.Path(ocd.__file__).resolve().is_relative_to(pathlib.Path(src).resolve())
spec = importlib.util.spec_from_file_location("g", scorer)
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
assert g.CertifyingDecoder is ocd.CertifyingDecoder
B = -1
out = []
for d in (3, 5, 7):
    p = 5e-3
    circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    dem = circ.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    edges = [(int(e[0]), B if e[1] is None else int(e[1]), float(e[2].get("weight", 1.0))) for e in m.edges()]
    cd = ocd.CertifyingDecoder(edges)
    mechs = g.dem_mechanisms(dem, cd.W, cd.key)
    rr, cc = [], []
    for j, ((u, v), _w, _p) in enumerate(mechs):
        for x in (u, v):
            if x != B:
                rr.append(x); cc.append(j)
    H = csc_matrix((np.ones(len(rr), np.uint8), (rr, cc)), shape=(dem.num_detectors, len(mechs)))
    dec = BpOsdDecoder(H, error_channel=[mm[2] for mm in mechs], max_iter=30, bp_method="ms",
                       schedule="parallel", osd_method="osd_cs", osd_order=5)
    det, _ = circ.compile_detector_sampler(seed=31337).sample(400, separate_observables=True)
    Wf = {k: Fraction(w) for k, w in cd.W.items()}
    for s in range(400):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        mw = tuple(cd.key((u, v if v >= 0 else B)) for u, v in m.decode_to_edges_array(det[s]))
        r = cd.certify(fired, mw)
        sel = np.asarray(dec.decode(det[s].astype(np.uint8))).astype(bool).ravel()
        odd, w_raw = g.odd_and_weight(mechs, sel)
        cnt = Counter(mechs[j][0] for j in np.flatnonzero(sel))
        xor = sorted(e for e, c in cnt.items() if c % 2)
        out.append({"d": d, "shot": s, "accepted": bool(r.accepted), "primal_hex": float(r.primal).hex(),
                    "bound_hex": float(r.bound).hex(), "primal_exact": str(sum((Wf[k] for k in mw), Fraction())),
                    "consistent": odd == set(fired), "raw_hex": float(w_raw).hex(),
                    "xor_exact": str(sum((Wf[e] for e in xor), Fraction())),
                    "max_mult": max(cnt.values()) if cnt else 0, "cancelled": any(c > 1 for c in cnt.values()),
                    "bp_converged": bool(dec.converge)})
print(json.dumps(out))
'''


def run_child(src: pathlib.Path, scorer: pathlib.Path):
    p = subprocess.run([sys.executable, "-B", "-c", CHILD, str(src), str(scorer)], capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=3600)
    if p.returncode != 0:
        raise SystemExit(f"child failed: {p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def analyse(recs, stored):
    out = {}
    srows = {r["distance"]: r for r in stored["rows"]}
    for d in (3, 5, 7):
        rs = [r for r in recs if r["d"] == d]
        st = srows[d]["decoders"]["bp_osd"]
        sub = [r for r in rs if r["accepted"] and r["consistent"]]
        raw = [float.fromhex(r["raw_hex"]) - float.fromhex(r["primal_hex"]) for r in sub]
        allb = [float.fromhex(r["raw_hex"]) - float.fromhex(r["bound_hex"]) for r in rs if r["consistent"]]
        reproduced = (summarise(raw) == st["gap_vs_proven_optimum"]
                      and summarise(allb) == st["gap_vs_certified_bound"]
                      and ("per_shot_gap_vs_proven_optimum" not in st
                           or [x.hex() for x in raw] == [float(x).hex() for x in st["per_shot_gap_vs_proven_optimum"]]))
        if not reproduced:
            raise SystemExit(f"d={d}: the old scoring does not reproduce the stored artifact; refusing to rescore it")
        g = [Fraction(r["xor_exact"]) - Fraction(r["primal_exact"]) for r in sub]
        over = [r for r in sub if r["cancelled"]]
        overcount = [float(Fraction(float.fromhex(r["raw_hex"])) - Fraction(r["xor_exact"])) for r in over]
        out[d] = {"reference_shots": len(sub), "reproduces_stored_artifact": True,
                  "overcounted_shots": len(over),
                  "overcounted_true_gap_exactly_zero": sum(1 for r in over
                                                           if Fraction(r["xor_exact"]) == Fraction(r["primal_exact"])),
                  "overcounted_bp_converged": sum(1 for r in over if r["bp_converged"]),
                  "overcount_min": min(overcount) if overcount else None,
                  "overcount_max": max(overcount) if overcount else None,
                  "max_copies_of_one_edge": max((r["max_mult"] for r in over), default=0),
                  "published": {k: st["gap_vs_proven_optimum"][k] for k in ("frac_zero", "median", "p99", "max")},
                  "rescored": {"exact_zero": sum(1 for x in g if x == 0),
                               "exact_zero_rate": sum(1 for x in g if x == 0) / len(g),
                               "tolerance_zero": int(sum(1 for x in g if abs(x) < Fraction(1, 10**9))),
                               "tiny_nonzero_below_1e9": sum(1 for x in g if x != 0 and abs(x) < Fraction(1, 10**9)),
                               "negative_exact": sum(1 for x in g if x < 0),
                               **{k: v for k, v in summarise([float(x) for x in g]).items()
                                  if k in ("median", "p99", "max")}}}
    return out


def main() -> int:
    res = {}
    for name, (art, commit, script_commit) in ARTIFACTS.items():
        stored = json.loads((ROOT / art).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            blob = subprocess.run(["git", "archive", commit, "src/oneq"], cwd=ROOT, capture_output=True,
                                  check=True).stdout
            tarfile.open(fileobj=io.BytesIO(blob)).extractall(td, filter="data")
            scorer = pathlib.Path(td) / "published_gap_distribution.py"
            scorer.write_bytes(subprocess.run(["git", "show", f"{script_commit}:experiments/m0_prior_art/"
                                               "gap_distribution.py"], cwd=ROOT, capture_output=True,
                                              check=True).stdout)
            recs = run_child(pathlib.Path(td) / "src", scorer)
        res[name] = {"artifact": art, "producer_commit": commit, "scorer_commit": script_commit,
                     "rows": analyse(recs, stored)}
        print(name, json.dumps({d: (r["reference_shots"], r["overcounted_shots"]) for d, r in res[name]["rows"].items()}),
              flush=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "oneq-gap-original-rescored/1", "commit": head, "results": res},
                              indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
