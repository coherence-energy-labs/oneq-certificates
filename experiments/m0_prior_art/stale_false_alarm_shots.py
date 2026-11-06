r"""Paper 1 Sec. VII B: are the stale-calibration false alarms at different drift factors the SAME base shots?

The stale-calibration experiment decodes the same 800 base shots per distance at every drift factor (sampler seed
20260731), so its pooled false-alarm count is not a sum of independent trials (external audit 2, Sec. 8). The
published artifact (`evidence/detection_outcomes/d<d>.json`) records only counts. This re-runs
`detection_outcomes.stale_calibration` unchanged for each distance, recording WHICH base shot each outcome belongs to,
and must reproduce the published counts exactly before it reports which shots the false alarms fell on.

    python experiments/m0_prior_art/stale_false_alarm_shots.py   -> evidence/stale_false_alarm_shots.json
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
sys.path.insert(0, str(ROOT / "src"))
import detection_outcomes as do  # noqa: E402

OUT = ROOT / "evidence" / "stale_false_alarm_shots.json"


def false_alarm_base_shots(outs: list[dict], base: list[int]) -> list[int]:
    """The base shots on which the certificate raised a false alarm: outcome i belongs to base shot base[i], and a
    false alarm is a correction that was NOT defective (within eps_max of the reference optimum) and NOT certified."""
    if len(outs) != len(base):
        raise SystemExit(f"{len(outs)} outcomes for {len(base)} nontrivial base shots: positions are not indices")
    return [base[i] for i, o in enumerate(outs) if (not o["defective"]) and o["state"] != "CERTIFIED"]


def swept_summary(out: dict) -> tuple[int, int]:
    """(false alarms over the swept drift factors, distinct (distance, base shot) pairs among them); 1.0 is the
    control and is not swept."""
    swept = [s for _d, rows in out.items() for f, r in rows.items() if f != "1.0" for s in r["false_alarm_base_shots"]]
    distinct = {(d, s) for d, rows in out.items() for f, r in rows.items() if f != "1.0"
                for s in r["false_alarm_base_shots"]}
    return len(swept), len(distinct)


def main() -> int:
    import numpy as np
    captured = []
    real_tally = do._tally

    def tally(rows):
        captured.append(list(rows))
        return real_tally(rows)
    do._tally = tally
    out = {}
    for d in (3, 5, 7):
        captured.clear()
        p = 0.005
        res = do.stale_calibration(d, 800, p)
        published = json.loads((ROOT / "evidence" / "detection_outcomes" / f"d{d}.json").read_text(encoding="utf-8"))
        if [r["outcomes"] for r in res["rows"]] != [r["outcomes"] for r in published["stale_calibration"]["rows"]]:
            raise SystemExit(f"d={d}: the re-run does not reproduce the published stale-calibration counts")
        circ = do._circuit(d, d, p)
        det, _ = circ.compile_detector_sampler(seed=20260731).sample(800, separate_observables=True)
        base = [s for s in range(800) if np.flatnonzero(det[s]).size]          # outcome i belongs to base shot base[i]
        rows = {}
        for r, outs in zip(res["rows"], captured):
            if any(r["skipped"].values()):
                raise SystemExit(f"d={d}: shots were skipped, so outcome positions are not base-shot indices")
            rows[str(r["drift_factor"])] = {"false_alarm_base_shots": false_alarm_base_shots(outs, base),
                                            "n": len(outs)}
        out[str(d)] = rows
        print(d, json.dumps(rows), flush=True)
    swept, distinct = swept_summary(out)
    doc = {"schema": "oneq-stale-false-alarm-shots/1", "rows": out,
           "swept_false_alarms": swept, "distinct_base_shots_among_them": distinct,
           "reproduces_published_counts": True}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("swept_false_alarms", "distinct_base_shots_among_them")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
