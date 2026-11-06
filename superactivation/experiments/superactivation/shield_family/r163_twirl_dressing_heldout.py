"""R163: the server's TWIRL DRESSING held out on R152's two protocols (EXPLORATORY: outside every preregistered verdict).

R152 ran with gate twirling on and qiskit-ibm-runtime's default strategy, "active-accum": "the union of instruction
qubits in the circuit up to the current twirled layer will be twirled in each individual twirled layer" (the
runtime's TwirlingOptions docstring, 0.50.0). A qubit twirled while IDLE in a layer gets a random Pauli before and
after it: two physical pulses whenever that Pauli is X or Y, so one pulse's error on average. A CZ ladder cannot see
it (every ladder qubit is in every layer), nor can the readout, Ramsey or sx families (no two-qubit gate to twirl).
R152's per-unit attribution put its excess at single-qubit locations, ~4x the vendor's sx error, on busy qubits.

Each variant takes the best physical base so far (V3c: R152's repair, vendor sx, R155's white T2 + sigma, CZ at the
ladder rate x 1) and adds NoiseParams.twirl (oneq.hw.noise._with_twirl_dressing, one pulse = 1.5 x the qubit's sx
error, scaled by sx_factor). NOTHING is fitted to R146 or R147's outcomes.

  python r163_twirl_dressing_heldout.py REPO_ROOT [--traj N]   -> evidence/superactivation/r163_twirl/heldout_r152.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import r152_posthoc as P152  # noqa: E402
import r155_heldout_r152 as HO  # noqa: E402

ARMS = (("V3c (twirl off)", dict()), ("+ dressing, active-accum", dict(twirl="active-accum")),
        ("+ dressing, active-circuit", dict(twirl="active-circuit")), ("+ dressing, all", dict(twirl="all")),
        ("+ active-accum, pulses x2", dict(twirl="active-accum", sx_factor=2.0)))


def score_arms(arms, fe146, fe147, m6, m7, meas146: float, meas147: float) -> dict:
    """per arm, both protocols predicted and missed (measured - predicted). PURE given the two predictors: each arm's
    params go to both, with the quasi-static spread on (V3c), and nothing else is added"""
    out = {}
    for name, kw in arms:
        if name in out:
            raise ValueError(f"arm {name!r} named twice")
        f6 = fe146(m6, quasistatic="coherent", **kw)
        f7, k7 = fe147(m7, quasistatic="coherent", **kw)
        out[name] = dict(params=dict(kw), r146=f6, r146_miss=meas146 - f6, r147=f7, r147_miss=meas147 - f7,
                         residual=dict(k7))
    return out


def report(variants: dict, width: int = 28) -> None:
    for name, v in variants.items():
        print(f"{name:{width}s} R146 {v['r146']:.4f} (miss {v['r146_miss']:+.4f})  R147 {v['r147']:.4f} "
              f"(miss {v['r147_miss']:+.4f})  residual {json.dumps({k: round(x, 4) for k, x in v['residual'].items()})}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    out: dict = dict(schema="oneq.r163.twirl_dressing_heldout/1",
                     label="EXPLORATORY -- outside every preregistered verdict", r152_job=art["job"], traj=a.traj,
                     measured=dict(r146=P["meas146"], r147=P["meas147"]),
                     variants=score_arms(ARMS, P["fe146"], P["fe147"], m6, m7, P["meas146"], P["meas147"]))
    report(out["variants"])
    d = root / "evidence" / "superactivation" / "r163_twirl"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                         newline="\n")
    print("wrote heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
