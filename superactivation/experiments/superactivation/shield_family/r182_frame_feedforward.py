"""R182 -- R147's key waited 2.572 us at every conditional, and a Pauli frame removes the wait exactly (OFFLINE,
2026-10-18).

R171's syndromes: on hardware the "early" key is exposed like the inline one, and only a long key hold closes R147's
main residual. The structure says why. R152's R147 circuits carry NO barrier (R153: the early and inline twins were
the same DAG), so the key finishes its preparation before the shield reads start and then IDLES at its conditional Z
until the feed-forward arrives, 2.704 us after the read starts (R166). `oneq.hw.dynamic_timing.conditional_waits`
puts that at 2.572 us on every unit, early and inline alike.

The correction is a Pauli and everything after it on the qubits it reaches is Clifford, so it need not run on the
device: `oneq.hw.pauli_frame.track` moves it into the post-processing, exactly. Per unit this records:
  - the key's structural wait before and after tracking (s);
  - the frame (which final bit XORs which mid-circuit bits) and the conditionals removed;
  - the exact check: the frame-corrected outcome law against the original's, for EVERY input on the key wire.
And the projection (a HYPOTHESIS until a run): the F_e gain of removing the wait, from R171's key-hold arms (score
residual at holds 0 / 0.52 / 2.7 us, interpolated at the structural wait), times R147's own estimator sensitivity
(control_variate, measured here by shifting the main units' means), two ways -- the score arms alone, and scaled by
how much of the modelled wait damage the copy-bit syndrome actually shows.

  python r182_frame_feedforward.py -> evidence/superactivation/r182_frame/frame_r147.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
R171 = ROOT / "evidence" / "superactivation" / "r171_residual" / "syndromes_r152.json"
OUT = ROOT / "evidence" / "superactivation" / "r182_frame"
EXACT = 1e-12
COPY_MEASURED = 0.0210                       # R171: layout-0 copy bit on hardware
COPY_PREDICTED = {0.0: 0.0100, 0.52e-6: 0.0134, 2.7e-6: 0.0282}   # R171's arms (syndromes_r152.json: ell)


class FrameLaneRefused(ValueError):
    """a unit the frame cannot move exactly"""


def unit_row(qc) -> dict:
    """one unit: the key's wait before and after tracking, the frame, and the exact check on every key input"""
    from oneq.hw import contract_synth as CS
    from oneq.hw import dynamic_timing as DT
    from oneq.hw import pauli_frame as PF
    (key,) = CS.input_wires(qc)
    before = [w.get(key, 0.0) for w in DT.conditional_waits(qc, twirled=True)]
    new, frame = PF.track(qc)
    dp = PF.check(qc, new, frame, wire=key)
    if dp > EXACT:
        raise FrameLaneRefused(f"{qc.name}: the frame-corrected law differs by {dp:.3e}")
    after = [w.get(key, 0.0) for w in DT.conditional_waits(new, twirled=True)]
    return dict(key_wire=key, key_wait_s=max(before, default=0.0), key_wait_after_s=max(after, default=0.0),
                conditionals_removed=frame.removed, frame=frame.to_json()["flips"], max_abs_dp_any_input=dp)


def interp(points: dict, x: float) -> float:
    """piecewise-linear through {x: y}, clamped at the ends"""
    xs = sorted(points)
    if x <= xs[0]:
        return points[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if x <= b:
            return points[a] + (x - a) / (b - a) * (points[b] - points[a])
    return points[xs[-1]]


def projection(residual_by_hold: dict, wait_s: float, sensitivity: float, copy_measured: float,
               copy_predicted: dict) -> dict:
    """the projected hardware F_e gain of removing a wait of `wait_s` (pure):
      score: removing the wait moves the model from its hold-`wait_s` prediction to its hold-0 one, and the hardware
             with it if the model charges the wait right: d(score) = residual(wait) - residual(0) (residual =
             measured - predicted), times dF_e/d(score);
      copy-scaled: the same, times (measured - predicted at hold 0) / (predicted at the wait - predicted at hold 0)
             of the copy bit -- the fraction of the modelled wait damage the copy-bit syndrome actually shows."""
    d_score = interp(residual_by_hold, wait_s) - interp(residual_by_hold, 0.0)
    gain = sensitivity * d_score
    c0, cw = interp(copy_predicted, 0.0), interp(copy_predicted, wait_s)
    frac = (copy_measured - c0) / (cw - c0) if cw != c0 else float("nan")
    return dict(wait_s=wait_s, d_score=d_score, sensitivity=sensitivity, gain_score=gain,
                copy_fraction_shown=frac, gain_copy_scaled=gain * frac)


def sensitivity(rec: dict, counts: dict) -> float:
    """dF_e / d(main-unit mean score) of R147's estimator, by shifting every main unit's measured mean"""
    import hw_dynamic_run as R147
    a = R147.analyse(rec, counts)
    means = {n: v["measured"] for n, v in a["per_unit"].items()}
    main = [u for u in rec["units"] if u["kind"] == "main"]
    base = R147.control_variate(means, main)["F_e"]
    d = 0.01
    shifted = {n: (m + d if a["per_unit"][n]["kind"] == "main" else m) for n, m in means.items()}
    return (R147.control_variate(shifted, main)["F_e"] - base) / d


def main() -> int:
    import r152_posthoc as P152

    from oneq.hw import freeze as F
    art = P152.load(R152_DIR)
    rec = art["parts"]["r147"]
    circs = F.load(rec, (R152_DIR / "r147.qpy").read_bytes(), check_sources=False)
    rows = {u["name"]: dict(unit_row(circs[u["name"]]), kind=u["kind"]) for u in rec["units"]}
    fed = [r for r in rows.values() if r["conditionals_removed"]]
    waits = sorted({round(r["key_wait_s"], 12) for r in fed})
    kh = json.loads(R171.read_text(encoding="utf-8"))["key_hold"]
    residual = {float(k.split()[0]) * 1e-6: v["main_mean_residual"] for k, v in kh.items()}
    sens = sensitivity(rec, {n: art["counts"]["r147__" + n] for n in rec["circuit_order"]})
    wait = float(np.median([r["key_wait_s"] for r in fed]))
    out = dict(schema="oneq.r182.frame/1", label="EXPLORATORY -- an exact rewrite and a projected gain; no run",
               r152_job=art["job"], units=len(rows), units_with_feed_forward=len(fed),
               conditionals_removed=sum(r["conditionals_removed"] for r in rows.values()),
               key_waits_s=waits, key_wait_after_s=max(r["key_wait_after_s"] for r in rows.values()),
               worst_max_abs_dp_any_input=max(r["max_abs_dp_any_input"] for r in rows.values()),
               projection=projection(residual, wait, sens, COPY_MEASURED, COPY_PREDICTED),
               measured_r147=art["result"]["r147"]["measured"]["F_e"], per_unit=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "frame_r147.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    pj = out["projection"]
    print(f"{out['units']} units, {out['units_with_feed_forward']} with feed-forward, {out['conditionals_removed']} "
          f"conditionals moved to software; key waits {waits} s -> {out['key_wait_after_s']} s; worst "
          f"|dp| {out['worst_max_abs_dp_any_input']:.2e}")
    print(f"projected F_e gain: {pj['gain_score']:+.4f} (score arms) / {pj['gain_copy_scaled']:+.4f} (copy-scaled, "
          f"{pj['copy_fraction_shown']:.2f} of the modelled damage); dF_e/dscore {pj['sensitivity']:.3f}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
