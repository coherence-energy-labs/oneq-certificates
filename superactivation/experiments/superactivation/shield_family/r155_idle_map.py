"""R155 family A -> a per-qubit IDLE MAP (white rate G, quasi-static sigma), and the feed-forward stall L at B's sites.
  echo  (total E):  -ln r_e  = G E                      (a static detuning refocuses; white decay does not)
  Ramsey at W, 2W:  -ln r_k  = G t_k + sigma^2 t_k^2 / 2
Weighted least squares over the three, G, sigma^2 >= 0. Then a far qubit f that waits an extra L for a feed-forward
loses contrast by exp(-[G L + sigma^2 ((t+L)^2 - t^2) / 2]) relative to the read without it, t its own wait (W/4).

A qubit with a ratio outside (0, 1.2] is NOT mapped: a Ramsey that has gone negative is a coherent FIXED detuning
(cos(Delta t) -- q25, q45, q149 on 2026-10-17), which neither G nor sigma represents.

  python r155_idle_map.py REPO_ROOT     reads R155's committed record and result; writes idle_map_<job>.json beside them
"""
import json
import math
import pathlib
import sys

import numpy as np
from scipy.optimize import brentq, nnls

JOB = "db9dsqpgv2cc73e5c8fg"


def fit_qubit(v: dict, W: float, E: float) -> dict | None:
    """G and sigma for one qubit from its echo and its Ramsey at W and 2W (R155's per-qubit A row); None when fewer
    than three ratios are usable"""
    rows, y, w = [], [], []
    for t, r, sd in ((E, v.get("ratio_echo"), v.get("sd_ratio_echo")), (W, v["ratio_w1"], v["sd_ratio_w1"]),
                     (2 * W, v["ratio_w2"], v["sd_ratio_w1"] * math.sqrt(2))):
        if r is None or not (0 < r <= 1.2):
            continue
        rr = min(r, 1.0)
        rows.append([t, (t * t / 2) if t != E else 0.0])
        y.append(-math.log(max(rr, 1e-6)))
        w.append(1 / max(sd / max(rr, 1e-6), 1e-4))
    if len(rows) < 3:
        return None
    A = np.array(rows) * np.array(w)[:, None]
    sol, _ = nnls(A, np.array(y) * np.array(w))
    G, s2 = float(sol[0]), float(sol[1])
    return dict(white_rate_per_s=G, white_T2_s=1 / G if G > 0 else float("inf"), sigma_rad_per_s=math.sqrt(s2),
                t2_vendor_s=v["t2_vendor_s"])


def stall(ratio: float, G: float, sigma: float, t0: float) -> float:
    """the extra wait L that takes a far qubit's contrast from its read-only value down by `ratio`; nan when the
    ratio is not a loss or no L under 1 ms explains it"""
    if not (0 < ratio < 1):
        return float("nan")

    def loss(L):
        return G * L + sigma ** 2 * ((t0 + L) ** 2 - t0 ** 2) / 2 + math.log(ratio)
    return brentq(loss, 0, 1e-3) if loss(1e-3) > 0 else float("nan")


def idle_map(rec: dict, res: dict) -> dict:
    dt = rec["snapshot"]["dt"]
    W, E = rec["w_dt"] * dt, rec["echo_dt"] * dt
    out = {q: f for q, v in res["A"]["per_qubit"].items() if (f := fit_qubit(v, W, E)) is not None}
    stalls = []
    for k, st in enumerate(rec["b_sites"]):
        f = str(st["f"])
        raw = res["B"][str(k)]["raw"]
        far = list(raw["read"])[-1]
        ratio = raw["read_ff"][far][0] / raw["read"][far][0]
        L = (stall(ratio, out[f]["white_rate_per_s"], out[f]["sigma_rad_per_s"], W / 4) if f in out
             else float("nan"))
        stalls.append(dict(site=k, far=int(f), ratio=ratio, stall_s=L))
    return dict(schema="oneq.r155.idle_map/1", model="-ln r = G t + sigma^2 t^2 / 2 (Ramsey), G t (echo)", W_s=W,
                echo_s=E, per_qubit=out, stalls=stalls)


def main(root: pathlib.Path) -> int:
    d = root / "evidence" / "superactivation" / "r155_diag"
    rec = json.loads((d / "frozen_20261017T023024Z.json").read_text(encoding="utf-8"))
    res = json.loads((d / f"result_{JOB}.json").read_text(encoding="utf-8"))
    m = idle_map(rec, res)
    out = m["per_qubit"]
    G = np.array([x["white_rate_per_s"] for x in out.values()])
    S = np.array([x["sigma_rad_per_s"] for x in out.values()])
    V = np.array([x["t2_vendor_s"] for x in out.values()])
    print(f"idle map over {len(out)} qubits:")
    print(f"  white T2 / vendor T2: median {np.median((1 / np.maximum(G, 1e-12)) / V):.2f}")
    print(f"  quasi-static sigma: median {np.median(S) / (2 * math.pi) / 1e3:.1f} kHz; 90th pct "
          f"{np.percentile(S, 90) / (2 * math.pi) / 1e3:.1f} kHz")
    worst = sorted(out.items(), key=lambda kv: -kv[1]["sigma_rad_per_s"])[:5]
    print("  largest sigma:", [(q, round(x["sigma_rad_per_s"] / (2 * math.pi) / 1e3, 1)) for q, x in worst], "kHz")
    for s in m["stalls"]:
        print(f"  B site {s['site']}: far q{s['far']} contrast ratio {s['ratio']:.4f} -> feed-forward stall "
              f"L = {s['stall_s'] * 1e6:.2f} us")
    # the committed map's bytes are per_qubit + the windows (the stalls are printed, as they always were)
    keep = {k: m[k] for k in ("schema", "model", "W_s", "echo_s", "per_qubit")}
    (d / f"idle_map_{JOB}.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote idle_map_{JOB}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(pathlib.Path(sys.argv[1])))
