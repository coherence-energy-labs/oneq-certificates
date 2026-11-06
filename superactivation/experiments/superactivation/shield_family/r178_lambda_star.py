"""R178 (c): lambda*(m) -- the noise scale at which the two-use protocol's predicted F_e reaches 3/4, m = 3..8, on
heavy-hex (FakeKingston layout) and all-to-all, and whether a cheaper shield compile moves it.

MODEL (oneq.hw.engine), every term labelled:
  - device: FakeKingston's own calibration (heavy-hex; layout by oneq.hw.layout over representatives spanning the
    family, hw_predict.choose_layout), or an ALL-TO-ALL device with FakeKingston's MEDIAN per-operation numbers
    (CZ error and duration, sx error, readout p01/p10, T1, T2, readout duration) -- the same per-gate quality without
    routing, so the two lambda* are comparable;
  - CZ: cz_factor 2.01 (R145's in-circuit ladder, edge 47-48, MEASURED);
  - twirl: strategy "active", dressing_factor 1.54 (R165): "active" dresses no idle qubit, so the dressing term is
    INERT -- this is the R165 correction (the runtime default "active-accum" would add it; the positive control below
    shows the knob is live there);
  - R167b's global feed-forward sync acts only on DYNAMIC circuits (mid-circuit read + if_else). This lane predicts
    the STATIC lean design (hw_two_use.py, R145's circuits), which has no feed-forward, so the sync costs it nothing;
    R167b's lesson (every superposed qubit pays the stall) is why the static design is the one predicted;
  - schedule ALAP (R146 V5, MEASURED on ibm_kingston).
  lambda multiplies EVERY error source (NoiseParams.scaled): CZ, sx, idle, readout.

ESTIMATOR: hw_predict's stratified control variate (exact stratum weights; each unit's ideal score exact; the per-unit
loss sampled), units drawn from their own stream so every lambda is scored on the SAME transpiled circuits (paired).
lambda* is read off the curve F_e(lambda) by power-law interpolation of the loss between bracketing points
(`lambda_star`, pure, tested). m = 3, 4: the two-use closed-form code's ideal F_e is below 3/4 (0.696, 0.735), and
R142's best two-use codes cross only at d >= 20 > 16, so lambda* does not exist there: no device suffices.

COMPILE LEVER: the CZ count per shield (`cz_per_shield`, pure: least-squares slope of the layout's mean CZ against m)
for the baseline compile and for hw_two_use.Compile(fanout=True) (P's CZ(key, s_t) for odd t from the key's Z-copy:
the parity-ladder fan-out R143(b) names, at one copy), and lambda* under each.

CONTROLS: (negative) twirl "active" vs "off" must give the SAME prediction on the same circuits and draws (no idle
dressing under either); (positive) "active-accum" with dressing 1.54 must LOWER it; (anchor) lambda -> 0 must return
the exact ideal F_e (loss 0).

  python r178_lambda_star.py --device hh|a2a --m 5,6,7,8 --units U --traj T [--lams 0.12,0.3] [--compile fanout]
writes evidence/superactivation/r178_lambda_star/<device>_<compile>_m<M>.json per m (a checkpoint) and prints.
  python r178_lambda_star.py summarize       -> evidence/superactivation/r178_lambda_star.json
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def _rel(p):
    """p relative to ROOT when it lies inside it, else p itself: relative_to raises on a path outside the
    root, and a scratch output path would crash the run after its work was done"""
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

OUTD = ROOT / "evidence" / "superactivation" / "r178_lambda_star"
SUMMARY = ROOT / "evidence" / "superactivation" / "r178_lambda_star.json"
CZ_FACTOR = 2.01
DRESSING = 1.54
BAR = 0.75


# ------------------------------------------------------------------------------------------------ pure pieces
def lambda_star(F0: float, lams: list, Fs: list, bar: float = BAR) -> dict:
    """lambda at which F(lambda) = bar, from the exact ideal F0 = F(0) and predictions F(lambda_i) (lambda_i > 0).
    Loss L = F0 - F is interpolated as a power law L = c lambda^a between the two points that bracket the crossing
    (a from those two points; with one point below the bar only, a = 1 from the origin -- linear, which OVERSTATES
    lambda* when the loss is concave in lambda (L(lam) >= (lam/l2) L2 on [0, l2], so the bar is reached earlier: the
    OPTIMISTIC direction, a noisier device read as good enough) and understates it when convex; corrected 2026-10-18,
    the first wording had the direction reversed). Returns {'lambda_star', 'kind'} with kind one of
      'none'      F0 <= bar: no device suffices
      'bracket'   bar lies between two predicted points
      'below'     every predicted point is already below the bar: interpolated from the origin (linear)
      'above'     every predicted point is above the bar: extrapolated from the last two points (flagged)"""
    if F0 <= bar:
        return {"lambda_star": None, "kind": "none"}
    pts = sorted(zip(lams, Fs, strict=True))
    need = F0 - bar
    L = [(lam, F0 - f) for lam, f in pts]
    above = [(lam, l_) for lam, l_ in L if l_ < need]
    below = [(lam, l_) for lam, l_ in L if l_ >= need]
    if above and below:
        (l1, L1), (l2, L2) = above[-1], below[0]
        if L1 <= 0:
            return {"lambda_star": l1 + (l2 - l1) * (need - L1) / (L2 - L1), "kind": "bracket"}
        a = math.log(L2 / L1) / math.log(l2 / l1)
        return {"lambda_star": l1 * (need / L1) ** (1 / a), "kind": "bracket", "exponent": a}
    if below:
        l2, L2 = below[0]
        return {"lambda_star": l2 * need / L2, "kind": "below"}
    if len(above) >= 2 and above[-2][1] > 0:
        (l1, L1), (l2, L2) = above[-2], above[-1]
        a = math.log(L2 / L1) / math.log(l2 / l1)
        return {"lambda_star": l2 * (need / L2) ** (1 / a), "kind": "above", "exponent": a}
    l2, L2 = above[-1]
    return {"lambda_star": l2 * need / L2 if L2 > 0 else None, "kind": "above"}


def cz_per_shield(ms: list, czs: list) -> float:
    """least-squares slope d(mean CZ)/dm"""
    x, y = np.asarray(ms, float), np.asarray(czs, float)
    if len(x) < 2:
        raise ValueError("need two m")
    return float(np.polyfit(x, y, 1)[0])


def argmin_lambda(rows: dict) -> tuple:
    """the m whose lambda* is LARGEST (the least device improvement 1/lambda* needed); rows: m -> lambda* or None"""
    ok = {m: v for m, v in rows.items() if v is not None}
    if not ok:
        return None, None
    m = max(ok, key=lambda k: ok[k])
    return m, 1.0 / ok[m]


# ------------------------------------------------------------------------------------------------ devices
def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def heavy_hex():
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    from oneq.hw import snapshot as S
    be = FakeKingston()
    return be, S.from_backend(be)


def all_to_all(n: int, ref):
    """full connectivity with the reference snapshot's MEDIAN per-operation numbers"""
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    from oneq.hw import snapshot as S
    med = lambda xs: float(np.median([x for x in xs if x is not None and math.isfinite(x)]))  # noqa: E731
    e2 = med([v[0] for v in ref.cz.values()])
    tcz = med([v[1] for v in ref.cz.values()])
    be = GenericBackendV2(num_qubits=n, basis_gates=["cz", "sx", "x", "rz", "id"],
                          coupling_map=CouplingMap.from_full(n), seed=1)
    snap = S.DeviceSnapshot(name=f"all_to_all_{n}_kingston_median", num_qubits=n, calibrated_utc=None, dt=ref.dt,
                            cz={(a, b): (e2, tcz) for a in range(n) for b in range(a + 1, n)},
                            t1=(med(ref.t1),) * n, t2=(med(ref.t2),) * n, p01=(med(ref.p01),) * n,
                            p10=(med(ref.p10),) * n, sx_error=(med(ref.sx_error),) * n,
                            sx_duration=(med(ref.sx_duration),) * n, meas_duration=(med(ref.meas_duration),) * n)
    return be, snap


# ------------------------------------------------------------------------------------------------ the scan
def base_params(twirl="active"):
    from oneq.hw import noise as N
    return N.NoiseParams(cz_factor=CZ_FACTOR, twirl=twirl, dressing_factor=DRESSING)


def scan(m: int, device: str, units: int, traj: int, lams: list, compile_: str, seed: int = 7, controls: bool = False):
    import hw_predict as HP
    import hw_two_use as H
    from qiskit import transpile

    from oneq.hw import engine as E
    from oneq.hw import noise as N
    E.MAX_ACTIVE = 25
    t0 = time.time()
    cmp = H.Compile(fanout=compile_ == "fanout")
    be_hh, snap_hh = heavy_hex()
    if device == "hh":
        be, snap = be_hh, snap_hh
        choice = HP.choose_layout(be, m, snap, N.NoiseParams(), cmp)
        layout, cmap = choice.initial_layout, choice.coupling_map
    else:
        be, snap = all_to_all(4 + 2 * m, snap_hh)
        layout, cmap = list(range(4 + 2 * m)), None

    def tr(qc, s):
        return transpile(qc, be, initial_layout=layout, optimization_level=3, seed_transpiler=s, coupling_map=cmap)

    unit_rng = np.random.default_rng(seed)
    circs = []
    i = 0
    for key, W in H.strata(m):
        for _ in range(units):
            u = H.sample_in_stratum(m, key, unit_rng)
            tqc = tr(H.circuit(m, u, cmp), 1000 + i)
            i += 1
            circs.append(("|".join(map(str, key)), W, u, tqc, H.ideal_expectation(m, u, cmp)))
    ee = []
    for w, u in H.erased_erased_units(m):
        ee.append((w, u, tr(H.circuit(m, u, cmp), 1), H.ideal_expectation(m, u, cmp)))
    czs = [c[3].count_ops().get("cz", 0) for c in circs]
    t_tr = time.time() - t0
    print(f"[{device} {compile_} m={m}] {len(circs)} kept circuits, mean CZ {np.mean(czs):.1f}, "
          f"active qubits <= {max(len({c[3].find_bit(q).index for ins in c[3].data for q in ins.qubits}) for c in circs)}"
          f" ({t_tr:.0f}s transpile)", flush=True)

    def predict(params, rng_seed):
        rng = np.random.default_rng(rng_seed)
        rows = {}
        for name, W, u, tqc, ideal in circs:
            r = E.expectation(tqc, H.score(m, u, cmp), snap, params, traj, rng)
            rows.setdefault(name, {"weight": W, "loss": [], "se": []})
            rows[name]["loss"].append(ideal - r.mean)
            rows[name]["se"].append(r.se)
        loss_ee, w_ee = 0.0, 0.0
        for w, u, tqc, ideal in ee:
            r = E.expectation(tqc, H.score(m, u, cmp), snap, params, traj, rng)
            loss_ee += w * (ideal - r.mean)
            w_ee += w
        out = {}
        for name, rr in rows.items():
            lv = np.array(rr["loss"])
            out[name] = dict(weight=rr["weight"], units=len(lv), mean_loss=float(lv.mean()),
                             se_loss=float(lv.std(ddof=1) / math.sqrt(len(lv))) if len(lv) > 1 else 0.0,
                             traj_se=float(math.sqrt(sum(s * s for s in rr["se"])) / len(lv)), mean_cz=0.0, max_cz=0)
        out["erased_erased"] = dict(weight=H.BRANCH_WEIGHT["erased_erased"], units=24, mean_loss=loss_ee / w_ee,
                                    se_loss=0.0, traj_se=0.0)
        c = HP.combine(m, out)
        return c["F_e_pred"], c["F_e_se"], out

    curve = []
    for lam in lams:
        t1 = time.time()
        F, se, _ = predict(base_params().scaled(lam), seed + 1)
        curve.append({"lambda": lam, "F_e": F, "se": se, "seconds": time.time() - t1})
        print(f"   lambda {lam:g}: F_e {F:.4f} +- {se:.4f}  ({time.time() - t1:.0f}s)", flush=True)
    F0 = H.f_e_closed(m)
    ls = lambda_star(F0, [c["lambda"] for c in curve], [c["F_e"] for c in curve])
    ctl = {}
    if controls:
        lam = lams[0]
        Fa, _, _ = predict(base_params("active").scaled(lam), seed + 1)
        Fo, _, _ = predict(base_params("off").scaled(lam), seed + 1)
        Fc, _, _ = predict(base_params("active-accum").scaled(lam), seed + 1)
        Fz, _, _ = predict(base_params().scaled(1e-9), seed + 1)
        ctl = {"lambda": lam, "active": Fa, "off": Fo, "active_accum_dressing_1.54": Fc,
               "negative_active_equals_off": abs(Fa - Fo) < 1e-12, "positive_accum_lowers": Fc < Fa - 1e-4,
               "anchor_lambda_0": Fz, "anchor_ideal": F0, "anchor_ok": abs(Fz - F0) < 1e-6}
        print(f"   controls: {ctl}", flush=True)
    rec = dict(m=m, d=2 ** m, device=device, snapshot=snap.name, snapshot_sha256=snap.sha256(), compile=cmp.name,
               layout=layout, units_per_stratum=units, trajectories=traj, mean_cz=float(np.mean(czs)),
               cz_per_circuit=czs, F_e_ideal=F0, curve=curve, lambda_star=ls, controls=ctl,
               noise=base_params().describe(),
               inputs={str(_rel(p)): sha(p) for p in (HERE / "r178_lambda_star.py", HERE / "hw_two_use.py",
                                                                  HERE / "hw_predict.py",
                                                                  ROOT / "src" / "oneq" / "hw" / "engine.py",
                                                                  ROOT / "src" / "oneq" / "hw" / "noise.py")},
               nonreproducible={"seconds": time.time() - t0, "transpile_seconds": t_tr,
                                "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / f"{device}_{cmp.name}_m{m}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"LAMBDA* {device} {cmp.name} m={m}: ideal {F0:.4f}; {ls}  ({time.time() - t0:.0f}s)", flush=True)
    return rec


def summarize() -> dict:
    import hw_two_use as H
    out = {"bar": BAR, "m3_m4": {m: {"F_e_ideal": H.f_e_closed(m), "lambda_star": None,
                                     "why": "ideal F_e of the two-use closed-form code is below 3/4; R142's best two-use "
                                            "codes cross only at d >= 20 > 2^4"} for m in (3, 4)},
           "runs": {}, "inputs": {}}
    for p in sorted(OUTD.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out["runs"][p.stem] = {k: r[k] for k in ("m", "device", "compile", "mean_cz", "F_e_ideal", "curve",
                                                 "lambda_star", "controls", "units_per_stratum", "trajectories")}
        out["inputs"][str(_rel(p))] = sha(p)
    for dev in ("hh", "a2a"):
        for cmpn in ("baseline", "fanout"):
            rows = {r["m"]: r["lambda_star"]["lambda_star"] for r in out["runs"].values()
                    if r["device"] == dev and r["compile"] == cmpn}
            if not rows:
                continue
            rows.update({3: None, 4: None})
            best_m, improvement = argmin_lambda(rows)
            czr = {r["m"]: r["mean_cz"] for r in out["runs"].values() if r["device"] == dev and r["compile"] == cmpn}
            out[f"{dev}_{cmpn}"] = {"lambda_star_by_m": dict(sorted(rows.items())), "best_m": best_m,
                                    "device_improvement_x": improvement, "mean_cz_by_m": dict(sorted(czr.items())),
                                    "cz_per_shield": cz_per_shield(list(czr), list(czr.values())) if len(czr) > 1
                                    else None}
    SUMMARY.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs", "inputs")}, indent=1, default=str))
    return out


def main(argv) -> int:
    if argv and argv[0] == "summarize":
        summarize()
        return 0
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("hh", "a2a"), required=True)
    ap.add_argument("--m", default="5,6,7,8")
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--traj", type=int, default=24)
    ap.add_argument("--lams", default="0.12,0.3")
    ap.add_argument("--compile", default="baseline", choices=("baseline", "fanout"))
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args(argv)
    for m in [int(x) for x in a.m.split(",")]:
        scan(m, a.device, a.units, a.traj, [float(x) for x in a.lams.split(",")], a.compile, controls=a.controls)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
