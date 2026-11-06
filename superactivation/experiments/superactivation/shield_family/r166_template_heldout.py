"""R166 held out on R152 (EXPLORATORY: outside every preregistered verdict): the twirled circuits as the server PLAYS
them.

R166 read the executed schedule: with twirling on, every single-qubit run is played as one two-sx template (an H
compiled with one sx plays two), every twirled CZ layer gets a 64 ns slot (period 132 ns, not 68), and idle touched
qubits under active-accum get two sx per layer. NoiseParams.twirl_template puts the first two into the engine
(noise.template_resynthesis); the dressing at factor 2 is the third, at its scheduled count (R165's shots said
1.54-1.75 in pulse units). R152's attribution put the protocols' excess at single-qubit-gate locations, about 4x the
vendor sx: this is the test of whether the template is that excess. NOTHING is fitted to R146 or R147.

Base: V3c (R152's repair, vendor sx, R155's white T2 + sigma, CZ at the ladder rate x 1). R147 on its frozen
stand-ins (the hold as frozen; the stand-ins' zero-duration virtual edges are not twirled layers).

  python r166_template_heldout.py REPO_ROOT [--traj N] -> evidence/superactivation/r166_executed_schedule/template_heldout_r152.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

ARMS = (("V3c + dressing x1.54 (R165, the best before R166)", dict(twirl="active-accum", dressing_factor=1.54)),
        ("V3c + template, no dressing", dict(twirl_template=True)),
        ("V3c + template + dressing x2 (two sx per layer, as scheduled)",
         dict(twirl_template=True, twirl="active-accum", dressing_factor=2.0)),
        ("V3c + template + dressing x1.54", dict(twirl_template=True, twirl="active-accum", dressing_factor=1.54)))


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
    out: dict = dict(schema="oneq.r166.template_heldout/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, measured=dict(r146=P["meas146"], r147=P["meas147"]),
                     variants=R163.score_arms(ARMS, P["fe146"], P["fe147"], m6, m7, P["meas146"], P["meas147"]))
    R163.report(out["variants"], width=62)
    d = root / "evidence" / "superactivation" / "r166_executed_schedule"
    (d / "template_heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                                  newline="\n")
    print("wrote template_heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
