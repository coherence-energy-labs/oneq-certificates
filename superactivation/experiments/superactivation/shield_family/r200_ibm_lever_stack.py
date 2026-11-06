"""R200 -- does ANY lever stack put IBM Kingston, on its real heavy-hex, over 3/4 on the channel-faithful two-use design?

R186 part 2 priced the design per device and put IBM at NO-GO, component by component (heavy-hex m = 6: loss 0.151,
of which CZ 0.062, readout 0.040, idle 0.017, sx 0.015). Since then:
  - the idle cost is removable (R196's echo);
  - a routing-free embedding could plausibly halve the CZ count, which heavy-hex triples against all-to-all
    (81 vs 28);
  - readout correction is allowed only as a labelled SECONDARY estimate: raw readout carries the primary claim
    (owner's rule, docs/OPEN_WORK.md, DECIDED 2026-10-18).
This evaluates the levers JOINTLY, interactions included, with R186's exact evaluator: multipliers on the base model
(ro = readout probability, idle, cz, sx), on one shared sample of design units.

  python r200_ibm_lever_stack.py [UNITS] [M] [OUT_JSON]
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import replace

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import r186_faithful_requirements as R  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "r200_ion_robustness" / "ibm_lever_stack.json"
BAR = 0.75


def configs() -> list:
    F = R.FULL
    return [R.IDEAL, F,
            replace(F, ro01=0.0, ro10=0.0, label="ro0"),
            replace(F, ro01=0.0, ro10=0.0, idle=0.0, label="ro0_idle0"),
            replace(F, ro01=0.0, ro10=0.0, idle=0.0, cz=0.5, label="ro0_idle0_cz.5"),
            replace(F, ro01=0.0, ro10=0.0, idle=0.0, cz=0.5, sx=0.5, label="ro0_idle0_cz.5_sx.5"),
            replace(F, ro01=0.0, ro10=0.0, idle=0.0, cz=0.25, sx=0.5, label="ro0_idle0_cz.25_sx.5"),
            # the primary claim's rule: readout RAW, every other lever
            replace(F, idle=0.0, cz=0.5, sx=0.5, label="RAWro_idle0_cz.5_sx.5"),
            replace(F, idle=0.0, cz=0.25, sx=0.25, label="RAWro_idle0_cz.25_sx.25")]


def run(units: int = 64, m: int = 6, seed: int = 7) -> dict:
    t0 = time.time()
    cfgs = configs()
    dev = R.make_device("kingston_hh", m, "mix")
    (res,) = R.run_design(m, "mix", [dev], cfgs, units, seed=seed)
    comb = R.combine_design(res)
    ix = R.index_of(cfgs)
    rows = {c.label: dict(F_e=float(comb["F_e"][ix[c.label]]), se=float(comb["se"][ix[c.label]]),
                          margin=float(comb["F_e"][ix[c.label]]) - BAR) for c in cfgs}
    raw = {k: v for k, v in rows.items() if k.startswith("RAWro")}
    return dict(schema="oneq.r200.ibm_lever_stack/1", device="kingston_hh", m=m, decoder="mix", units=units,
                seed=seed, mean_cz=float(sum(res["cz"]) / len(res["cz"])), bar=BAR, configs=rows,
                best_under_raw_readout=max(raw.items(), key=lambda kv: kv[1]["F_e"])[0],
                seconds=time.time() - t0, sources_sha256=R.sources_sha(),
                evidence_class="PROJECTED (R186's exact evaluator on the validated Kingston model)")


def main() -> int:
    units = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    dest = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else OUT
    res = run(units, m)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in res["configs"].items():
        print(f"{k:28s} F_e {v['F_e']:.4f} +- {v['se']:.4f}  margin {v['margin']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
