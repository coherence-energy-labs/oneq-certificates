"""R161's coherent detunings HELD OUT on R152's two protocols (EXPLORATORY: outside every preregistered verdict).

R161 measured q149's "3-5 us T2" to be a coherent +49.7 kHz frame offset with contrast 0.984. R152 itself already
holds that qubit's offset in ITS window: the block's Ramsey ratio 0.353 at W = 3.264 us is cos(Delta W) at contrast
~0.98, |Delta|/2pi ~ 58.6 kHz -- R161 supplies the sign (+). q25, on R146's layout, reads ~0.73 at 2.24 us and ~0.56
at 3.264 us: a ~50 kHz offset at both waits (sign unmeasured: both are scored). Each variant takes the best physical
base so far (V3c: R152's repair, vendor sx, R155's white T2 + sigma, CZ at the ladder rate x 1), gives each detuned
qubit its echo-level T2 (the offset is no longer disguised as dephasing), and adds DeviceSnapshot.detuning. NOTHING
is fitted to R146 or R147's outcomes.

  python r161_detuning_heldout.py REPO_ROOT [--traj N]    -> evidence/superactivation/r161_spectator_cz/detuning_heldout_r152.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import r152_posthoc as P152  # noqa: E402
import r155_heldout_r152 as HO  # noqa: E402

from oneq.hw import snapshot as S  # noqa: E402

TWO_PI = 2 * math.pi


def offset_from_ratio(ratio: float, wait_s: float, contrast: float = 0.98) -> float:
    """|Delta| in rad/s from a Ramsey ratio read as contrast x cos(Delta W) (the smallest alias)"""
    return math.acos(max(-1.0, min(1.0, ratio / contrast))) / wait_s


def detuned(model: S.DeviceSnapshot, offsets: dict, vendor: S.DeviceSnapshot) -> S.DeviceSnapshot:
    """the model with coherent offsets on the named qubits, each with the vendor's (echo-level) T2"""
    t2 = list(model.t2)
    for q in offsets:
        t2[q] = min(vendor.t2[q], 2 * model.t1[q])
    return S.DeviceSnapshot(**{**model.__dict__, "t2": tuple(t2), "detuning": dict(offsets)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    fit146, fitblk = art["result"]["r146"]["fit"]["xs"], art["result"]["block"]["fit"]["xs"]
    w146 = art["parts"]["r146"]["wait_dt"] * art["parts"]["r146"]["snapshot"]["dt"]
    wblk = art["parts"]["blk147"]["wait_dt"] * art["parts"]["blk147"]["snapshot"]["dt"]
    d149 = offset_from_ratio(fitblk["149"]["ratio"], wblk)
    d25 = 0.5 * (offset_from_ratio(fit146["25"]["ratio"], w146) + offset_from_ratio(fitblk["25"]["ratio"], wblk))
    base6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    base7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    arms = [("V3c (base)", {}), ("+q149 +58.6k", {149: +d149}), ("+q149 -58.6k (sign check)", {149: -d149}),
            ("+q149 +, q25 +", {149: +d149, 25: +d25}), ("+q149 +, q25 -", {149: +d149, 25: -d25})]
    out: dict = dict(schema="oneq.r161.detuning_heldout/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, measured=dict(r146=P["meas146"], r147=P["meas147"]),
                     offsets_khz=dict(q149=d149 / TWO_PI / 1e3, q25=d25 / TWO_PI / 1e3,
                                      q149_from="R152 block ratio %.3f at %.3g s; sign from R161" % (
                                          fitblk["149"]["ratio"], wblk),
                                      q25_from="R152 ratios %.3f at %.3g s and %.3f at %.3g s; sign unmeasured" % (
                                          fit146["25"]["ratio"], w146, fitblk["25"]["ratio"], wblk)),
                     variants={})
    print(f"offsets: q149 {d149 / TWO_PI / 1e3:.1f} kHz, q25 {d25 / TWO_PI / 1e3:.1f} kHz", flush=True)
    for name, offs in arms:
        m6 = detuned(base6, {q: v for q, v in offs.items() if q < base6.num_qubits}, P["vendor"]) if offs else base6
        m7 = detuned(base7, offs, P["vendor"]) if offs else base7
        f6 = P["fe146"](m6, quasistatic="coherent")
        f7, k7 = P["fe147"](m7, quasistatic="coherent")
        out["variants"][name] = dict(offsets_khz={str(q): v / TWO_PI / 1e3 for q, v in offs.items()}, r146=f6,
                                     r146_miss=P["meas146"] - f6, r147=f7, r147_miss=P["meas147"] - f7, residual=k7)
        print(f"{name:28s} R146 {f6:.4f} (miss {P['meas146'] - f6:+.4f})  R147 {f7:.4f} (miss {P['meas147'] - f7:+.4f})"
              f"  residual {json.dumps({k: round(v, 4) for k, v in k7.items()})}", flush=True)
    dest = root / "evidence" / "superactivation" / "r161_spectator_cz" / "detuning_heldout_r152.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
