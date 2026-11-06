"""R172: the design levers priced on the measured model -- the same two protocols R152 ran, under each twirling choice
(EXPLORATORY: outside every preregistered verdict; items 5 and 6 of the owner's 25-item list).

R166/R173 MEASURED what the server plays:
- twirling ON: every single-qubit run is a two-sx template, every CZ layer takes 132 ns, and under active-accum the
  idle touched qubits get two sx per layer;
- twirling OFF: the compiled gates, at 68 ns per layer.
Held out on R152, the twirl-on model (template + dressing x2) closes R146 (-0.006 +- 0.008) and leaves R147's
dynamic units at -0.024 +- 0.002.

This prices, on the SAME circuits and the same R152 calibration (V3c), what each choice would have given:
  as run   active-accum: template + dressing x2 (the model that closes R146)
  active   "active": the template, no idle dressing
  off      twirling off: the compiled pulses, no template, no dressing. The CZ is still the engine's Pauli channel at
           the twirled ladder rate. Untwirled, its coherent part is NOT randomized: a coherent error of the same
           infidelity can do better or worse in a protocol than its Pauli twirl, so this arm is a PROJECTION until
           a coherent-CZ model is measured.
Each arm reports the model's F_e. "bias-corrected" adds the as-run arm's measured miss -- a model gap carried over
unchanged, which is an assumption: the dynamic remainder is not yet explained.

  python r172_design_levers.py REPO_ROOT [--traj N] -> evidence/superactivation/r172_design/levers_r152.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

ARMS = (("as run (active-accum: template + dressing x2)",
         dict(twirl_template=True, twirl="active-accum", dressing_factor=2.0)),
        ("active (template, no idle dressing)", dict(twirl_template=True)),
        ("off (compiled pulses; the CZ still a Pauli channel: a projection)", dict()))


def with_bias(variants: dict, reference: str) -> dict:
    """each arm's F_e plus the reference arm's measured miss (pure): the model gap carried over unchanged"""
    if reference not in variants:
        raise KeyError(f"no reference arm {reference!r}")
    ref = variants[reference]
    out = {}
    for name, v in variants.items():
        out[name] = dict(r146=v["r146"], r147=v["r147"],
                         r146_bias_corrected=v["r146"] + ref["r146_miss"],
                         r147_bias_corrected=v["r147"] + ref["r147_miss"],
                         gain_vs_reference=dict(r146=v["r146"] - ref["r146"], r147=v["r147"] - ref["r147"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    import r163_twirl_dressing_heldout as R163
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    variants = R163.score_arms(ARMS, P["fe146"], P["fe147"], m6, m7, P["meas146"], P["meas147"])
    R163.report(variants, width=66)
    out = dict(schema="oneq.r172.design_levers/1", label="EXPLORATORY -- outside every preregistered verdict",
               r152_job=art["job"], traj=a.traj, measured=dict(r146=P["meas146"], r147=P["meas147"]),
               bar=0.75, variants=variants, bias_corrected=with_bias(variants, ARMS[0][0]))
    for name, v in out["bias_corrected"].items():
        print(f"{name:66s} bias-corrected R146 {v['r146_bias_corrected']:.4f}  R147 {v['r147_bias_corrected']:.4f}"
              f"  (gain vs as-run: R146 {v['gain_vs_reference']['r146']:+.4f}, R147 {v['gain_vs_reference']['r147']:+.4f})")
    d = root / "evidence" / "superactivation" / "r172_design"
    d.mkdir(parents=True, exist_ok=True)
    (d / "levers_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                        newline="\n")
    print("wrote levers_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
