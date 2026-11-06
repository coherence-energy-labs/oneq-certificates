"""Shot-level bootstrap of the coding gain F(n=2) - F(n=1) for the v1 and v2 hardware runs.

The analytic standard error the drivers print leaves out the (1,1) branch (its estimator is a
nonlinear function of R's conditional Bloch vectors). Here every circuit's counts are resampled
multinomially, the full analysis is re-run, and the percentile interval of the gain is reported.
Writes evidence/superactivation/coding_gain_bootstrap.json.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("hw_coding_gain", HERE.parent / "hw_coding_gain.py")
hw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw)
EV = ROOT / "evidence" / "superactivation"


def resample(counts: dict, rng) -> dict:
    out = {}
    for name, c in counts.items():
        keys = list(c)
        p = np.array([c[k] for k in keys], dtype=float)
        n = int(p.sum())
        draw = rng.multinomial(n, p / p.sum())
        out[name] = {k: int(v) for k, v in zip(keys, draw) if v}
    return out


def boot(path: pathlib.Path, B: int, seed: int) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    counts = rec["counts"]
    rng = np.random.default_rng(seed)
    gains, f1s, f2s = [], [], []
    for _ in range(B):
        a = hw.analyse(resample(counts, rng))
        gains.append(a["gain"])
        f1s.append(a["n1"]["F"])
        f2s.append(a["n2"]["F"])
    point = hw.analyse(counts)
    g = np.array(gains)
    return {"job": rec["job"]["job_id"], "layout": rec["job"]["layout"], "B": B,
            "gain": point["gain"], "gain_boot_se": float(g.std(ddof=1)),
            "gain_95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
            "F1": point["n1"]["F"], "F1_boot_se": float(np.std(f1s, ddof=1)),
            "F2": point["n2"]["F"], "F2_boot_se": float(np.std(f2s, ddof=1)),
            "z": point["gain"] / float(g.std(ddof=1))}


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = {
        "v1": boot(sorted(EV.glob("coding_gain_d*.result.json"))[-1], B, 1),
        "v2": boot(sorted(EV.glob("coding_gain_v2_*.result.json"))[-1], B, 2),
    }
    (EV / "coding_gain_bootstrap.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
