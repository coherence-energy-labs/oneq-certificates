r"""What would certifying held-out distance-7 Willow shots cost? Measured on DEVELOPMENT shots only.

For docs/PREREG_heldout_d7_2026-11-03.md. The preregistered held-out d=7 sample must be sized before any of its shots
is read, so the per-shot cost is measured here on shots the producer was developed on (d7_at_q6_7, 30 rounds, both
bases), with exactly the per-shot work the held-out run will do (`epsilon_rerun._run_chunk` for a `certify_hard`
configuration): the full ladder `CertifyingDecoder(edges, epsilon_max=1e-6).certify_hard`, `check_epsilon` on the
resulting packing, and the forgery battery (`oneq.forgeries.forgeries` plus the NaN and +inf duals), every attack
judged by `check_epsilon`.

Sample rule (fixed, not chosen by looking): in each basis, shots 17 + 5000*i for i = 0..9 -- ten per basis, twenty in
all, spread over the 50,000. Every one of them was certified in development (generation 6 and the epsilon rerun).
Reports per-shot CPU seconds for each stage, which ladder tier closed the shot, and the process's peak working set.

    python experiments/m0_prior_art/heldout_d7_cost_calibration.py   -> evidence/heldout_d7_cost_calibration.json
"""
from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from oneq.certifying_decoder import CertifyingDecoder           # noqa: E402
from oneq.forgeries import assert_all_false, forgeries          # noqa: E402
from oneq.matching_cert import BOUNDARY, MatchingCertificate    # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon            # noqa: E402
from google_corpus_gate import CORPUS, load_config              # noqa: E402

OUT = ROOT / "evidence" / "heldout_d7_cost_calibration.json"
EPS_MAX = 1e-6
SHOTS = [17 + 5000 * i for i in range(10)]


def _peak_mb():
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)


def main() -> int:
    rows = []
    for basis in ("X", "Z"):
        cfg = f"d7_at_q6_7_{basis}_r30"
        m, edges, shots, meta = load_config(CORPUS / cfg)
        cd = CertifyingDecoder(edges, epsilon_max=EPS_MAX)
        for s in SHOTS:
            det = shots.window(s, s + 1)[0]
            fired = [int(i) for i in np.flatnonzero(det)]
            corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY)) for a, b in m.decode_to_edges_array(det))
            t0 = time.process_time()
            r = cd.certify_hard(fired, corr)
            t1 = time.process_time()
            duals = dict(r.cert.blossom_duals) if r.cert is not None else dict(r.best_dual or {})
            v = check_epsilon(edges=edges, syndrome=fired, eps_max=EPS_MAX,
                              cert=MatchingCertificate(matched_edges=corr, node_potentials={}, blossom_duals=duals))
            t2 = time.process_time()
            n_forg = bogus = acc = 0
            if duals:
                battery = forgeries(cd.W, fired, corr, duals, r.primal)
                some = next(iter(duals))
                battery += [("N_nan_dual", MatchingCertificate(matched_edges=corr, node_potentials={},
                                                               blossom_duals={**duals, some: math.nan}), "non-finite"),
                            ("N_inf_dual", MatchingCertificate(matched_edges=corr, node_potentials={},
                                                               blossom_duals={**duals, some: math.inf}), "non-finite")]
                bogus = len(assert_all_false(cd.W, fired, r.primal, [b for b in battery if b[2] != "non-finite"]))
                for _nm, fc, _w in battery:
                    n_forg += 1
                    acc += check_epsilon(edges=edges, syndrome=fired, cert=fc, eps_max=EPS_MAX).accepted
            t3 = time.process_time()
            row = {"config": cfg, "shot": s, "detectors_fired": len(fired), "status": v.status,
                   "rung": r.rung, "stats": {k: r.stats[k] for k in ("rungs_tried", "tier") if k in r.stats},
                   "cpu_s_certify_hard": round(t1 - t0, 3), "cpu_s_check": round(t2 - t1, 3),
                   "cpu_s_battery": round(t3 - t2, 3), "cpu_s_total": round(t3 - t0, 3),
                   "forgeries": n_forg, "forgeries_accepted": acc, "battery_not_false": bogus,
                   "peak_working_set_mb": _peak_mb()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    tot = [r["cpu_s_total"] for r in rows]
    doc = {"schema": "oneq-heldout-d7-cost-calibration/1", "sample": "development shots 17 + 5000*i, i = 0..9, "
                                                                      "d7_at_q6_7 X and Z at 30 rounds",
           "producer": "CertifyingDecoder(edges, epsilon_max=1e-6).certify_hard + check_epsilon + forgery battery",
           "platform": platform.platform(), "python": sys.version.split()[0], "rows": rows,
           "summary": {"shots": len(rows), "cpu_s_per_shot_mean": float(np.mean(tot)),
                       "cpu_s_per_shot_median": float(np.median(tot)), "cpu_s_per_shot_max": float(max(tot)),
                       "cpu_s_certify_mean": float(np.mean([r["cpu_s_certify_hard"] for r in rows])),
                       "cpu_s_battery_mean": float(np.mean([r["cpu_s_battery"] for r in rows])),
                       "peak_working_set_mb": max(r["peak_working_set_mb"] for r in rows),
                       "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}}}}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
