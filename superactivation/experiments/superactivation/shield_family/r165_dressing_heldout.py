"""R165 held out on R152 (EXPLORATORY: outside every preregistered verdict): the twirl dressing at its MEASURED size.

R163 charged the dressing of idle qubits at one pulse per layer and closed 40% of R146's held-out miss. R165 measured
it on healthy idle qubits (hw_r165_dressing_bench.py): a touched |0> qubit flips 2.65x the one-pulse estimate per
layer. The engine already charges a touched idle qubit's T1 as a Pauli-twirled channel -- exact under the random
frame active-accum gives it -- so of that total the DRESSING TERM owns the pulse part only: 1.54 pulse units if the
frame inverts the qubit for the whole 132 ns layer, 1.75 if for the 100 ns between pulse centres, 2 if the server
plays a fixed two-sx dressing (R166 will read which). Each arm sets NoiseParams.dressing_factor, which scales the
dressing ALONE (R163's "x2" arm used sx_factor, which also doubled every sx gate of the protocol: a confound).

Base: V3c (R152's repair, vendor sx, R155's white T2 + sigma, CZ at the ladder rate x 1). NOTHING is fitted to R146
or R147's outcomes; the factors come from R165's own shots.

  python r165_dressing_heldout.py REPO_ROOT [--traj N]   -> evidence/superactivation/r165_dressing_bench/heldout_r152.json
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
import r163_twirl_dressing_heldout as R163  # noqa: E402  -- the same scoring, one implementation

ARMS = (("V3c (twirl off)", dict()),
        ("+ dressing x1 (R163)", dict(twirl="active-accum")),
        ("+ dressing x1.54 (R165, 132 ns inverted)", dict(twirl="active-accum", dressing_factor=1.54)),
        ("+ dressing x1.75 (R165, 100 ns inverted)", dict(twirl="active-accum", dressing_factor=1.75)),
        ("+ dressing x2 (a fixed two-sx dressing)", dict(twirl="active-accum", dressing_factor=2.0)))


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
    out: dict = dict(schema="oneq.r165.dressing_heldout/1",
                     label="EXPLORATORY -- outside every preregistered verdict", r152_job=art["job"], traj=a.traj,
                     measured=dict(r146=P["meas146"], r147=P["meas147"]),
                     variants=R163.score_arms(ARMS, P["fe146"], P["fe147"], m6, m7, P["meas146"], P["meas147"]))
    R163.report(out["variants"], width=44)
    d = root / "evidence" / "superactivation" / "r165_dressing_bench"
    d.mkdir(parents=True, exist_ok=True)
    (d / "heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                         newline="\n")
    print("wrote heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
