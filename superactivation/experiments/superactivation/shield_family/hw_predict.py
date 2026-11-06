"""Predicted RAW F_e of the hardware two-use design (hw_two_use.py) on a device snapshot, by oneq.hw.engine.

Estimator (stratified, with a control variate of KNOWN mean): the kept branches are split into the 12 strata of
hw_two_use.strata() (branch x P's sector x beta x the erased use's flip), each with its exact weight; the exact ideal
F_avg of the design is known in closed form and every sampled unit's ideal score is computed exactly, so
    F_avg_pred = F_avg_ideal - sum_b W_b mean_{u in b} [ideal_u - noisy_u]
The per-unit LOSS varies far less within a stratum than the per-unit fidelity does across the design (beta either
matches the sector-determined copy bit -- score near 2 -- or never does). Both-erased units are enumerated exactly.
The layout is chosen once under the full calibrated model whatever scenario is predicted, and the units come from
their own random stream, so scenarios (no readout, no idle, a scaled device) are scored on the SAME units: paired.

  python hw_predict.py M UNITS_PER_STRATUM TRAJ [--live BACKEND] [--snapshot JSON] [--no-readout] [--no-idle]
                       [--cz-factor X] [--scale S]
writes evidence/superactivation/shield_family/hw_predict_m{M}_{backend}_{scenario}.json and prints one line.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import hw_two_use as H  # noqa: E402
from oneq.hw import engine as E  # noqa: E402
from oneq.hw import layout as L  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "shield_family"


def fake_kingston():
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    return FakeKingston()


def representatives(m: int, seed: int = 1, cmp: H.Compile = H.BASELINE) -> list:
    """circuits that span the family's interactions. A single random unit does NOT: in P's s0 sector everything
    between the two key-copy CNOTs is diagonal on the key, so the transpiler cancels both CNOTs exactly and copy0 is
    left free to land anywhere -- and the s1 sector, whose X on the key keeps them, then pays ~50 extra CZs of SWAPs
    on that layout (measured: 33 CZ for the representative, 82 for an s1 unit)."""
    rng = np.random.default_rng(seed)
    units = [H.Unit("kept_kept", ("s1", (0,) * m), 0, ("Z", 1), (0, 0)),
             H.Unit("kept_erased", ("s1", (1,) * m), 1, ("Y", -1), (0, 1)),
             H.Unit("kept_kept", ("s0", ("I",) * m), 1, ("X", 1), (0, 0)),
             H.sample_unit(m, "kept_kept", rng), H.sample_unit(m, "kept_erased", rng)]
    return [H.circuit(m, u, cmp) for u in units]


def choose_layout(be, m: int, snap: S.DeviceSnapshot, params: N.NoiseParams, cmp: H.Compile = H.BASELINE):
    """ONE initial layout for every unit, chosen by oneq.hw.layout on the snapshot over the representatives"""
    return L.choose_layout(representatives(m, cmp=cmp), be, snap, params)


def isa(be, qc, layout, seed, coupling_map=None):
    from qiskit import transpile
    return transpile(qc, be, initial_layout=layout, optimization_level=3, seed_transpiler=seed,
                     coupling_map=coupling_map)


def predict(m: int, units: int, traj: int, params: N.NoiseParams, be, snap: S.DeviceSnapshot, seed: int = 7,
            cmp: H.Compile = H.BASELINE) -> dict:
    # TWO streams: the units drawn depend only on the seed, never on the noise model, so every scenario (no
    # readout, no idle, a scaled device) is scored on the SAME units and scenario differences are paired. One shared
    # stream let each scenario's different trajectory draws pick different units.
    unit_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    # the layout is chosen under the FULL calibrated model whatever scenario is predicted: a scenario that switched
    # off idles chose a different layout with MORE CZs and read a larger loss -- comparing layouts, not noise
    choice = choose_layout(be, m, snap, N.NoiseParams(), cmp)
    layout = choice.initial_layout
    rows = {}
    i = 0
    for key, W in H.strata(m):
        losses, czs, dur = [], [], []
        for _ in range(units):
            u = H.sample_in_stratum(m, key, unit_rng)
            tqc = isa(be, H.circuit(m, u, cmp), layout, seed=1000 + i, coupling_map=choice.coupling_map)
            i += 1
            ideal = H.ideal_expectation(m, u, cmp)
            r = E.expectation(tqc, H.score(m, u, cmp), snap, params, traj, rng)
            losses.append((ideal - r.mean, r.se))
            czs.append(tqc.count_ops().get("cz", 0))
            dur.append(N.duration_s(tqc, snap))
        lv = np.array([x for x, _ in losses])
        name = "|".join(map(str, key))
        rows[name] = dict(weight=W, units=units, mean_loss=float(lv.mean()),
                          se_loss=float(lv.std(ddof=1) / math.sqrt(units)),
                          traj_se=float(math.sqrt(sum(s * s for _, s in losses)) / units),
                          mean_cz=float(np.mean(czs)), max_cz=int(max(czs)), mean_duration_s=float(np.mean(dur)))
        print(f"  {name:28s} W={W:.4f} loss {rows[name]['mean_loss']:+.4f}+-{rows[name]['se_loss']:.4f} "
              f"cz {rows[name]['mean_cz']:.1f}", flush=True)
    loss_ee, w_ee, dur_ee = 0.0, 0.0, []
    for w, u in H.erased_erased_units(m):
        tqc = isa(be, H.circuit(m, u, cmp), layout, seed=1, coupling_map=choice.coupling_map)
        r = E.expectation(tqc, H.score(m, u, cmp), snap, params, traj, rng)
        loss_ee += w * (H.ideal_expectation(m, u, cmp) - r.mean)
        w_ee += w
        dur_ee.append(N.duration_s(tqc, snap))
    rows["erased_erased"] = dict(weight=H.BRANCH_WEIGHT["erased_erased"], units=24, mean_loss=loss_ee / w_ee,
                                 se_loss=0.0, traj_se=0.0, mean_cz=0.0, max_cz=0,
                                 mean_duration_s=float(np.mean(dur_ee)))
    return dict(m=m, d=2 ** m, backend=snap.name, snapshot_sha256=snap.sha256(), calibrated_utc=snap.calibrated_utc,
                noise=params.describe(), layout=layout, layout_proxy=choice.best.proxy,
                layout_mean_cz=choice.best.mean_cz, layout_candidates=choice.n_candidates,
                units_per_stratum=units, trajectories=traj, **combine(m, rows), strata=rows)


def combine(m: int, rows: dict) -> dict:
    """the prediction from per-stratum losses -- pure arithmetic over recorded numbers:
        F_avg = F_avg_ideal - sum_b W_b loss_b,   se^2 = sum_b W_b^2 (se_loss_b^2 + traj_se_b^2),
        F_e = (3 F_avg - 1) / 2 (the 6 cardinal states are a 2-design), so F_e's se is 1.5 x F_avg's.
    The strata must carry the design's whole weight."""
    total = sum(r["weight"] for r in rows.values())
    if abs(total - 1) > 1e-12:
        raise ValueError(f"strata weights sum to {total}, not 1: the prediction would not be the design's")
    f_e_ideal = H.f_e_closed(m)
    f_avg_ideal = (2 * f_e_ideal + 1) / 3
    loss = sum(r["weight"] * r["mean_loss"] for r in rows.values())
    se = math.sqrt(sum(r["weight"] ** 2 * (r["se_loss"] ** 2 + r["traj_se"] ** 2) for r in rows.values()))
    f_avg = f_avg_ideal - loss
    return dict(F_e_ideal=f_e_ideal, F_avg_ideal=f_avg_ideal, F_avg_pred=f_avg, F_avg_se=se,
                F_e_pred=(3 * f_avg - 1) / 2, F_e_se=1.5 * se, margin_pred=(3 * f_avg - 1) / 2 - 0.75)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("m", type=int)
    ap.add_argument("units", type=int, help="units per stratum (12 kept strata + the 24 enumerated both-erased)")
    ap.add_argument("traj", type=int)
    ap.add_argument("--cz-factor", type=float, default=N.CZ_FACTOR_R133)
    ap.add_argument("--no-readout", action="store_true")
    ap.add_argument("--no-idle", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="every error source x this (a what-if device)")
    ap.add_argument("--seed", type=int, default=7, help="the unit stream's seed (an independent sample: change it)")
    ap.add_argument("--tag", default="", help="suffix for the output file, so a second sample never overwrites")
    ap.add_argument("--compile", default="baseline", help="baseline | fanout | fanin | fanout+fanin (hw_two_use.Compile)")
    ap.add_argument("--snapshot", help="DeviceSnapshot JSON (default: the backend's own calibration)")
    ap.add_argument("--live", help="transpile against this LIVE backend's target (read-only; spends nothing)")
    a = ap.parse_args()
    params = N.NoiseParams(cz_factor=a.cz_factor, readout=not a.no_readout, idle=not a.no_idle).scaled(a.scale)
    if a.live:
        from qiskit_ibm_runtime import QiskitRuntimeService
        be = QiskitRuntimeService().backend(a.live)
    else:
        be = fake_kingston()
    snap = (S.from_backend(be) if not a.snapshot
            else S.DeviceSnapshot.from_json(json.loads(pathlib.Path(a.snapshot).read_text(encoding="utf-8"))))
    t = time.time()
    cmp = H.Compile(fanout="fanout" in a.compile, fanin="fanin" in a.compile)
    rec = predict(a.m, a.units, a.traj, params, be, snap, seed=a.seed, cmp=cmp)
    rec["compile"] = cmp.name
    rec["seconds"] = time.time() - t
    rec["seed"] = a.seed
    OUT.mkdir(parents=True, exist_ok=True)
    tag = (f"cz{a.cz_factor:.2f}" + ("_noro" if a.no_readout else "") + ("_noidle" if a.no_idle else "")
           + (f"_x{a.scale:g}" if a.scale != 1.0 else "") + (f"_{a.tag}" if a.tag else "")
           + (f"_{cmp.name}" if cmp != H.BASELINE else ""))
    (OUT / f"hw_predict_m{a.m}_{snap.name}_{tag}.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"m={a.m} d={2 ** a.m} {snap.name} {tag}: F_e ideal {rec['F_e_ideal']:.4f} -> predicted RAW "
          f"{rec['F_e_pred']:.4f} +- {rec['F_e_se']:.4f} (margin {rec['margin_pred']:+.4f}); layout CZ "
          f"{rec['layout_mean_cz']:.1f} ({rec['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
