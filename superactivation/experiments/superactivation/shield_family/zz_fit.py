"""Always-on coupling between neighbouring qubits, measured from a calibration block's xs family (R153).

THE MODEL. IBM calibrates each qubit's frequency with its neighbours in |0>. Static coupling then acts, relative to
that frame, as a CONDITIONAL PHASE on every coupled pair: CPhase(zeta t) = diag(1, 1, 1, e^{i zeta t}). Pauli-twirled,
it is the channel p(ZI) = p(IZ) = p(ZZ) = sin^2(zeta t / 2) / 4, which reproduces the coherent <X_i> and <X_i X_j> of
the xs family exactly (checked: tests/test_zz_crosstalk.py).

THE IDENTITY THAT MAKES IT MEASURABLE. In the xs family every qubit idles in |+> for the same wait. For a coupled pair
    R_ij = <X_i X_j> / (<X_i> <X_j>) = 1 / (1 - sin^2(zeta_ij t / 2)),
EXACTLY, whatever each qubit's own dephasing and whatever its other neighbours do (each factor appears once in the
numerator and once in the denominator). So every edge's zeta is read from the block's own counts, edge by edge, and the
qubit's intrinsic T2 follows from its contrast with its neighbours' factors divided out.
THE CONTROL: uncoupled pairs at graph distance 2. On heavy-hex every such pair shares a neighbour, which correlates them
at SECOND order (R - 1 ~ s_ij s_jk ~ 1e-4 at the couplings measured), far below the statistical sd (~1e-2): at this
precision they must give R = 1, and their z-scores calibrate the error bars. When those z-scores are over-dispersed
(rms > 1), every edge's sd is inflated by that rms, so the error bars the data supports are the ones reported. The w0 circuit (no wait) carries the same SPAM and H-gate correlations without the
idle, so the estimator is R(w) / R(w0).

Inputs are the calibration block's counts ONLY (xs_w0, xs_w and the readout fit): nothing here reads a protocol
circuit, so a model built from it predicts R146's p0 and R147's units held out.

  python zz_fit.py RECORD_DIR      (e.g. evidence/superactivation/campaign_r152/20261017T003152Z)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_diag_run as D146  # noqa: E402
from oneq.hw import campaign as C  # noqa: E402
from oneq.hw import stats as T  # noqa: E402

BOOT = 200


def coupled_pairs(snapshot: dict, qubits: list) -> list:
    """every physically coupled pair of the block's qubits (an edge with a CZ calibration), as layout INDICES"""
    idx = {q: i for i, q in enumerate(qubits)}
    out = []
    for k in snapshot["cz"]:
        a, b = (int(x) for x in k.split("_"))
        if a in idx and b in idx:
            out.append((idx[a], idx[b]))
    return sorted(out)


def _bits(counts: dict, n: int):
    keys = list(counts)
    B = np.array([T.bits_of(k, n) for k in keys], dtype=np.int8)
    w = np.array([int(counts[k]) for k in keys], dtype=float)
    return B, w


def _moments(B, w, readout_rows):
    """readout-corrected <X_i> for every qubit and the bit-level pair moments, from weighted bitstrings"""
    x = 1 - 2 * B.astype(float)                       # +-1 per qubit (|+> -> H -> |0> reads +1)
    tot = w.sum()
    m1 = (w @ x) / tot
    c1 = np.array([D146._readout_inverse(m1[i], readout_rows[i]) for i in range(len(m1))])
    return x, tot, m1, c1


def _pair_c(x, w, tot, i, j, m1, c1, ri, rj):
    mij = float(w @ (x[:, i] * x[:, j])) / tot
    return D146._pair_inverse(mij, c1[i], c1[j], ri, rj)


def ratio_table(counts_w0: dict, counts_w: dict, n: int, readout_rows: list, pairs: list, weights=None):
    out = {}
    stats = {}
    for tag, cnt in (("w0", counts_w0), ("w", counts_w)):
        B, w = _bits(cnt, n)
        if weights is not None:
            w = weights[tag]
        x, tot, m1, c1 = _moments(B, w, readout_rows)
        stats[tag] = (x, w, tot, m1, c1)
    for (i, j) in pairs:
        r = {}
        for tag in ("w0", "w"):
            x, w, tot, m1, c1 = stats[tag]
            cij = _pair_c(x, w, tot, i, j, m1, c1, readout_rows[i], readout_rows[j])
            r[tag] = cij / (c1[i] * c1[j])
        out[(i, j)] = r["w"] / r["w0"]
    contrast = {tag: stats[tag][4] for tag in stats}
    return out, contrast


def fit_zz(rec: dict, counts: dict, readout: dict, boot: int = BOOT, seed: int = 152) -> dict:
    """per coupled edge: s = 1 - 1/R (= sin^2(zeta t / 2)), zeta (rad/s) and bootstrap sds; per qubit: the intrinsic
    T2 with every neighbour's coupling divided out; and the CONTROL: R on uncoupled pairs at graph distance 2"""
    qubits = rec["layout"]
    n = len(qubits)
    t = rec["wait_dt"] * rec["snapshot"]["dt"]
    rows = [readout[q] for q in qubits]
    pairs = coupled_pairs(rec["snapshot"], qubits)
    nb = {i: set() for i in range(n)}
    for i, j in pairs:
        nb[i].add(j)
        nb[j].add(i)
    far = sorted({(min(i, k), max(i, k)) for i in range(n) for j in nb[i] for k in nb[j]
                  if k != i and k not in nb[i]})
    R, contrast = ratio_table(counts["xs_w0"], counts["xs_w"], n, rows, pairs + far)
    # bootstrap: multinomial resampling of each circuit's counts
    rng = np.random.default_rng(seed)
    Bw0, ww0 = _bits(counts["xs_w0"], n)
    Bw, ww = _bits(counts["xs_w"], n)
    boots = {p: [] for p in pairs + far}
    for _ in range(boot):
        rw = {"w0": rng.multinomial(int(ww0.sum()), ww0 / ww0.sum()).astype(float),
              "w": rng.multinomial(int(ww.sum()), ww / ww.sum()).astype(float)}
        Rb, _ = ratio_table(counts["xs_w0"], counts["xs_w"], n, rows, pairs + far, weights=rw)
        for p in boots:
            boots[p].append(Rb[p])
    edges = {}
    for (i, j) in pairs:
        r = R[(i, j)]
        s = max(0.0, 1 - 1 / r)
        zeta = 2 * math.asin(min(1.0, math.sqrt(s))) / t
        bs = np.array(boots[(i, j)])
        edges[f"{qubits[min(i, j)]}_{qubits[max(i, j)]}"] = dict(
            R=r, R_sd=float(bs.std(ddof=1)), s=s, zeta_rad_s=zeta,
            zeta_sd=float(np.std([2 * math.asin(min(1.0, math.sqrt(max(0.0, 1 - 1 / b)))) / t for b in bs], ddof=1)),
            zeta_over_2pi_khz=zeta / (2 * math.pi) / 1e3)
    control = {f"{qubits[i]}_{qubits[k]}": dict(R=R[(i, k)], R_sd=float(np.std(boots[(i, k)], ddof=1)))
               for (i, k) in far}
    t2 = {}
    for i, q in enumerate(qubits):
        ratio = contrast["w"][i] / contrast["w0"][i] if contrast["w0"][i] > 0 else float("nan")
        coup = math.prod(1 - edges[f"{qubits[min(i, j)]}_{qubits[max(i, j)]}"]["s"] for j in nb[i])
        intrinsic = ratio / coup if coup > 0 else float("nan")
        t2[q] = dict(ratio=ratio, coupling_factor=coup, intrinsic_ratio=intrinsic,
                     t2_eff_s=-t / math.log(ratio) if 0 < ratio < 1 else float("inf"),
                     t2_intrinsic_s=-t / math.log(intrinsic) if 0 < intrinsic < 1 else float("inf"),
                     t2_vendor_s=rec["snapshot"]["t2"][q])
    zc = np.array([(c["R"] - 1) / c["R_sd"] for c in control.values() if c["R_sd"] > 0])
    inflate = max(1.0, float(np.sqrt(np.mean(zc ** 2)))) if len(zc) else 1.0
    for e in edges.values():
        e["R_sd_bootstrap"], e["zeta_sd_bootstrap"] = e["R_sd"], e["zeta_sd"]
        e["R_sd"], e["zeta_sd"] = e["R_sd"] * inflate, e["zeta_sd"] * inflate
    ze = np.array([(e["R"] - 1) / e["R_sd"] for e in edges.values() if e["R_sd"] > 0])
    return dict(wait_s=t, edges=edges, control_uncoupled=control, t2=t2,
                summary=dict(n_edges=len(edges), n_control=len(control),
                             edge_z_median=float(np.median(ze)), control_z_median=float(np.median(zc)),
                             control_z_rms=float(np.sqrt(np.mean(zc ** 2))), sd_inflation=inflate,
                             zeta_khz_median=float(np.median([e["zeta_over_2pi_khz"] for e in edges.values()])),
                             t2_intrinsic_over_vendor_median=float(np.median(
                                 [v["t2_intrinsic_s"] / v["t2_vendor_s"] for v in t2.values()
                                  if math.isfinite(v["t2_intrinsic_s"])])),
                             t2_eff_over_vendor_median=float(np.median(
                                 [v["t2_eff_s"] / v["t2_vendor_s"] for v in t2.values() if math.isfinite(v["t2_eff_s"])]))))


def fit_block(record_dir: pathlib.Path, part: str) -> dict:
    loaded_rec = json.loads((record_dir / "composite.json").read_text(encoding="utf-8"))
    counts = json.loads(next(record_dir.glob("counts_*.json")).read_text(encoding="utf-8"))
    per = C.split(loaded_rec, counts)[part]
    rec = json.loads((record_dir / f"{part}.json").read_text(encoding="utf-8"))
    fit = D146.fit_device(rec, per)
    return dict(part=part, zz=fit_zz(rec, per, fit["readout"]), readout=fit["readout"])


if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1]).resolve()
    out = {}
    for part in ("r146", "blk147"):
        r = fit_block(d, part)
        out[part] = r["zz"]
        s = r["zz"]["summary"]
        print(f"{part}: {s['n_edges']} coupled edges, zeta/2pi median {s['zeta_khz_median']:.1f} kHz, edge z median "
              f"{s['edge_z_median']:.1f}; CONTROL {s['n_control']} uncoupled pairs z median {s['control_z_median']:+.2f}"
              f" (rms {s['control_z_rms']:.2f}); T2 eff/vendor {s['t2_eff_over_vendor_median']:.2f} -> intrinsic/vendor "
              f"{s['t2_intrinsic_over_vendor_median']:.2f}", flush=True)
    dest = d / "zz_fit.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    dest = dest.resolve()
    print("->", dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest))
