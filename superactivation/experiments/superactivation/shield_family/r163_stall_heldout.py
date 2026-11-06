"""R163 (dynamic): R155's MEASURED feed-forward stall in R147's stand-ins, on R163's best model, held out on R152
(EXPLORATORY: outside every preregistered verdict).

R147's stand-ins hold the key and the read shields for tau = the read + an ASSUMED 0.65 us feed-forward stall.
R155 measured the stall at three sites from each far qubit's own idle map: L = 0.39, 0.85 and 2.19 us (B2). With the
twirl dressing (R163, active-accum at one pulse) the static circuits close; R147's dynamic units keep ~-0.038. This
re-predicts R147 with every hold at the read + each measured L (r155_heldout_r152.with_hold, which rewrites exactly
the holds and refuses any other delay), on the V3c base with the dressing. NOTHING is fitted to R147's outcomes.

  python r163_stall_heldout.py REPO_ROOT [--traj N]    -> evidence/superactivation/r163_twirl/stall_heldout_r152.json
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

STALLS_US = (0.65, 0.39, 0.85, 2.19)              # R147's assumption, then R155 B2's three measured sites


def stall_variants(stalls_us, stand_ins, rec147: dict, fe147, m7, meas147: float, with_hold=HO.with_hold) -> dict:
    """per stall L: every stand-in hold rewritten to the read + L, and R147 re-predicted on it with the dressing
    (active-accum) -- PURE given the predictor and the hold rewriter"""
    read_s = rec147["tau_s"] - rec147["tau_ff_assumed_s"]
    if read_s <= 0:
        raise ValueError(f"the record's read time is {read_s} s: tau_s must exceed the assumed stall")
    out = {}
    for L in stalls_us:
        if not L > 0:
            raise ValueError(f"a stall must be positive: {L} us")
        hold = read_s + L * 1e-6
        stand, changed = with_hold(stand_ins, rec147, hold)
        f7, k7 = fe147(m7, stand_ins=stand, quasistatic="coherent", twirl="active-accum")
        out[f"L={L} us"] = dict(hold_s=hold, holds_changed=sum(changed.values()), r147=f7, r147_miss=meas147 - f7,
                                residual=dict(k7))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--traj", type=int, default=24)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    art = P152.load(root / HO.R152_DIR)
    idle = json.loads((root / HO.R155_DIR / HO.IDLE_MAP).read_text(encoding="utf-8"))["per_qubit"]
    P = P152.predictors(art, a.traj)
    rec147 = P["record147"]
    read_s = rec147["tau_s"] - rec147["tau_ff_assumed_s"]
    m7 = HO.with_vendor_sx(HO.with_idle_map(P["m147"], idle, True), P["vendor"])
    out: dict = dict(schema="oneq.r163.stall_heldout/1", label="EXPLORATORY -- outside every preregistered verdict",
                     r152_job=art["job"], traj=a.traj, measured_r147=P["meas147"], read_s=read_s,
                     model="V3c + twirl dressing active-accum (R163)",
                     variants=stall_variants(STALLS_US, P["stand_ins"], rec147, P["fe147"], m7, P["meas147"]))
    for name, v in out["variants"].items():
        print(f"{name} (hold {v['hold_s'] * 1e6:.2f} us)  R147 {v['r147']:.4f} (miss {v['r147_miss']:+.4f})"
              f"  residual {json.dumps({k: round(x, 4) for k, x in v['residual'].items()})}", flush=True)
    d = root / "evidence" / "superactivation" / "r163_twirl"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stall_heldout_r152.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8",
                                               newline="\n")
    print("wrote stall_heldout_r152.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
