#!/usr/bin/env python3
"""The device model's held-out regression record: every past protocol job re-predicted with NOTHING fitted, under
the model of the mode it was SUBMITTED in (oneq.hw.twirl_mode), bound to the sources that produced it.

R152 (job db9c50oc77vs73817rog) ran R146's p0 circuits and R147 under the runtime's default twirling
(server-active-accum). Two models of that mode are scored on the V3c base:
  v1  the template + the dressing as first-order error ops at its scheduled count (twirl_mode's preset today)
  v2  the template + the dressing as PULSES (template_resynthesis dress_idle: a slot per layer, neighbouring gates
      absorbed), which tests/test_model_regression_schedules.py checks pulse for pulse against R166's schedule
tests/test_model_regression.py refuses the record when a source changes, and checks its bounds: every number must
stay where it was recorded, so a model change that worsens the held-out fit cannot land silently.

  python tools/model_regression.py [--traj N]   -> evidence/hw/model_regression.json   (~15 min)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
sys.path.insert(0, str(ROOT / "src"))
SOURCES = ("src/oneq/hw/noise.py", "src/oneq/hw/engine.py", "tools/model_regression.py",
           "experiments/superactivation/shield_family/r152_posthoc.py",
           "experiments/superactivation/shield_family/r155_heldout_r152.py")
ARMS = (("v1: template + dressing x2 as error ops", dict(twirl_template=True, twirl="active-accum",
                                                          dressing_factor=2.0)),
        ("v2: template + dressing as pulses", dict(twirl_template=True, twirl_template_dress=True)))


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    import r152_posthoc as P152
    import r155_heldout_r152 as HO
    import r163_twirl_dressing_heldout as R163
    art = P152.load(ROOT / HO.R152_DIR)
    idle = json.loads((ROOT / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    m6 = HO.with_vendor_sx(HO.with_idle_map(P["m146"], idle, True), P["vendor"])
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    variants = R163.score_arms(ARMS, P["fe146"], P["fe147"], m6, m7, P["meas146"], P["meas147"])
    R163.report(variants, width=44)
    rec = dict(schema="oneq.hw.model_regression/1", job="db9c50oc77vs73817rog", submitted_mode="server-active-accum",
               traj=a.traj, measured=dict(r146=P["meas146"], r147=P["meas147"]),
               sd=dict(r146=0.0084, r147_main=0.0022, r147_inline=0.0084, r147_static=0.0105), variants=variants,
               sources_sha256={s: sha(s) for s in SOURCES})
    out = ROOT / "evidence" / "hw" / "model_regression.json"
    out.write_text(json.dumps(rec, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
