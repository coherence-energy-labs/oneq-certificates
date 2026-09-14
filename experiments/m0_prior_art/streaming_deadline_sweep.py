r"""THE DEFERRED TIER, exercised at last -- and its invariants tested.

`streaming.py` names three tiers and stakes its honesty on the third:

    HONEST  a shot whose certificate misses the lag budget is emitted as a
            DEFERRED receipt -- recorded, never skipped (a skip is a dark
            gate) -- and its certificate, when it lands, goes to a side
            channel named `late`, where it cannot be mistaken for having
            met the clock.

Every published streaming artifact in this repository reports

    "tiers": {"certified": 1479, "degraded": 21, "deferred": 0}

with `lag_budget_s=None`, which DISABLES deferral entirely. So the
mechanism the module leads with has never fired in any evidence we ship.
That is a dark tier: "the deferral path works" and "the deferral path has
never run" print identically from the outside, and the honest-lag claim
rests entirely on code nobody has watched execute.

This gate runs a ladder of deadlines from generous to impossible and
asserts what must hold at every rung:

  1. COVERAGE OF THE TIER. Some rung must actually produce deferrals, and
     some rung must produce none. A sweep that only ever certifies proves
     nothing about deferral; a sweep that only ever defers proves nothing
     about the budget being respected when it is met.
  2. NOTHING IS LOST. certified + degraded + deferred == shots, at every
     rung. A deferral is a RECORD, not a discard, and the difference is
     the entire claim.
  3. EVERY DEFERRAL RESOLVES. Each deferred shot's certificate must turn
     up on the `late` channel by the end of the drain -- deferred means
     "not yet", never "never". |late| == deferred.
  4. LATE IS NOT LAUNDERED. A late receipt must never be counted as
     certified-on-time: the tier counters and the late channel are
     disjoint accounts of the same shot.
  5. ORDERING SURVIVES. Receipts are emitted in shot order at every rung,
     deadline pressure included -- the property that makes the watermark
     mean anything.
  6. MONOTONICITY. Tightening the deadline may not REDUCE deferrals. Not
     a threshold, a direction: it is the only statement about the budget
     that does not require inventing a millisecond target.

No fabricated target anywhere. The ladder is derived from the measured
p50 of an unconstrained run on this machine, so it is a shape claim about
the mechanism, not a performance claim about the hardware.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from oneq.streaming import StreamingCertifier          # noqa: E402


def _corpus(config, n):
    """The streaming gate's corpus, built the same way it builds it.

    Reuses `load_config` rather than re-deriving a decoding graph: two
    definitions of "the corpus" is how the streaming gate and this gate
    would end up measuring different systems and both look green.
    """
    import numpy as np
    import streaming_gate as sg
    from oneq.matching_cert import BOUNDARY
    from oneq.certifying_decoder import CertifyingDecoder

    m, det, edges = sg.load_config(config, n)
    cd = CertifyingDecoder(edges)
    shots = []
    for s in range(det.shape[0]):
        f = [int(i) for i in np.flatnonzero(det[s])]
        if not f:
            continue
        corr = tuple(cd.key((u, v if v >= 0 else BOUNDARY))
                     for u, v in m.decode_to_edges_array(det[s]))
        shots.append((f, corr))
    return edges, shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="d5_at_q8_7_X_r30")
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    edges, shots = _corpus(a.config, a.shots)

    # ---- rung 0: no budget, to MEASURE the ladder rather than guess it
    sc = StreamingCertifier(edges, workers=a.workers,
                            max_pending=4 * a.workers)
    lat, order = [], []
    t0 = {}

    def timed(seq):
        for i, item in enumerate(seq):
            t0[i] = time.perf_counter()
            yield item

    for r in sc.run(timed(shots)):
        lat.append(time.perf_counter() - t0[r.shot_id])
        order.append(r.shot_id)
    base = np.array(lat)
    p50 = float(np.percentile(base, 50))
    print(f"unconstrained: {len(shots)} shots, p50 {1e3*p50:.0f} ms, "
          f"p99 {1e3*float(np.percentile(base, 99)):.0f} ms")
    assert order == sorted(order), "ordering broken with no budget at all"

    # a ladder ANCHORED to the measured p50: generous -> impossible
    ladder = [8.0 * p50, 2.0 * p50, 1.0 * p50, 0.25 * p50, 0.02 * p50]
    rungs = []
    for budget in ladder:
        s = StreamingCertifier(edges, workers=a.workers,
                               max_pending=4 * a.workers,
                               lag_budget_s=budget)
        tt = {}

        def timed2(seq, tt=tt):
            for i, item in enumerate(seq):
                tt[i] = time.perf_counter()
                yield item

        tiers = {"certified": 0, "degraded": 0, "deferred": 0}
        seen, deferred_ids = [], []
        for r in s.run(timed2(shots)):
            tiers[r.tier] += 1
            seen.append(r.shot_id)
            if r.tier == "deferred":
                deferred_ids.append(r.shot_id)
        late_ids = [sid for (sid, _rc) in s.stats.late]

        rung = {
            "lag_budget_s": round(budget, 6),
            "budget_over_p50": round(budget / p50, 3),
            "tiers": dict(tiers),
            "emitted": len(seen),
            "ordered": seen == sorted(seen),
            "late_resolved": len(late_ids),
            "deferred_ids_all_resolved":
                sorted(deferred_ids) == sorted(late_ids),
            "accounts_disjoint":
                tiers["certified"] + tiers["degraded"] + tiers["deferred"]
                == len(shots),
            # THE MODULE'S OWN COUNTER must agree with the receipts it
            # emitted. Found by sensitivity: breaking `st.deferred += 1`
            # left this gate green, because it tallies tiers from the
            # RECEIPTS and never consulted the published stats. A
            # consumer reads StreamStats, not my tally, so a counter that
            # disagrees with the stream is a lie to every such consumer.
            "stats_agree": (s.stats.deferred == tiers["deferred"]),
        }
        rungs.append(rung)
        print(f"  budget {budget*1e3:7.1f} ms ({budget/p50:5.2f}x p50): "
              f"cert {tiers['certified']:4d} degr {tiers['degraded']:3d} "
              f"defer {tiers['deferred']:4d}  late {len(late_ids):4d}  "
              f"ordered={rung['ordered']}")

    # ---- the assertions, each named
    fail = []
    if not any(r["tiers"]["deferred"] > 0 for r in rungs):
        fail.append("NO RUNG DEFERRED: the deferred tier is still dark -- "
                    "this sweep proves nothing about it")
    if not any(r["tiers"]["deferred"] == 0 for r in rungs):
        fail.append("EVERY RUNG DEFERRED: no rung shows the budget being "
                    "MET, so 'respects the deadline' is untested")
    for r in rungs:
        if not r["accounts_disjoint"]:
            fail.append(f"SHOTS LOST at budget {r['lag_budget_s']}: tiers "
                        f"sum to {sum(r['tiers'].values())} != {len(shots)}")
        if not r["ordered"]:
            fail.append(f"ORDERING BROKEN at budget {r['lag_budget_s']} -- "
                        f"the watermark means nothing if receipts reorder")
        if not r["stats_agree"]:
            fail.append(f"STATS DISAGREE at budget {r['lag_budget_s']}: "
                        f"StreamStats.deferred != the deferred receipts "
                        f"actually emitted -- consumers read the stats")
        if not r["deferred_ids_all_resolved"]:
            fail.append(f"DEFERRAL UNRESOLVED at budget {r['lag_budget_s']}: "
                        f"{r['tiers']['deferred']} deferred but "
                        f"{r['late_resolved']} landed on `late` -- deferred "
                        f"must mean 'not yet', never 'never'")
    defs = [r["tiers"]["deferred"] for r in rungs]
    if any(b < a_ for a_, b in zip(defs, defs[1:])):
        fail.append(f"NON-MONOTONE: tightening the deadline REDUCED "
                    f"deferrals somewhere in {defs} -- the budget is not "
                    f"the thing deciding the tier")

    out = {"schema": "oneq-streaming-deadline-sweep/1",
           "config": a.config, "shots": len(shots), "workers": a.workers,
           "unconstrained_p50_s": p50,
           "unconstrained_p99_s": float(np.percentile(base, 99)),
           "ladder_anchored_to": "measured p50 of an unconstrained run on "
                                 "this machine -- no fabricated target",
           "rungs": rungs,
           "deferred_tier_exercised": any(r["tiers"]["deferred"] > 0
                                          for r in rungs),
           "budget_met_somewhere": any(r["tiers"]["deferred"] == 0
                                       for r in rungs),
           "claim": ("the DEFERRED tier fires under deadline pressure, "
                     "every deferral resolves on the `late` channel, "
                     "nothing is lost or reordered, and tightening the "
                     "budget only ever increases deferrals")}
    dest = pathlib.Path(a.out) if a.out else (
        ROOT / "evidence" / "streaming_deadline_sweep.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest}")

    if fail:
        print(f"\nDEADLINE SWEEP FAILED ({len(fail)}):")
        for f in fail:
            print(f"  {f}")
        return 1
    print("\nDEFERRED TIER PROVEN LIVE: it fires, it resolves, it never "
          "loses or reorders a shot, and it responds monotonically to the "
          "budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
