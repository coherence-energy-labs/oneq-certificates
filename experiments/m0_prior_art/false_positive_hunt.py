r"""Adversarial hunt: can ANY forged certificate make the checker say yes?

A false positive is program-ending, not a tuning issue. The gate for milestone
0.2 is ZERO of them, so the honest way to earn that number is to attack the
checker rather than to watch it pass.

THE ATTACKS LIVE IN `oneq.forgeries` AND ARE NOT RESTATED HERE. This file used
to carry its own copy of the battery, which is the twin-copy trap: a battery
weakened in one copy still reports "0 accepted" from the other, and the scale
claim is only ever as strong as its weakest copy. One battery, three consumers
(this hunt, the million-shot gate, the hardware-corpus gate).

THE MEASUREMENT THAT MATTERS is not "the checker rejected these". It is that
the checker rejected these AND still accepts genuine certificates. A checker
that refuses everything is trivially free of false positives and useless, so
both numbers are reported and the gate needs both.

AND THE BATTERY ITSELF IS AUDITED. Every attack's claim is independently
re-derived as FALSE before any acceptance is scored against it. That guard
exists because this project has twice produced "attacks" that were accepted
because their claims were TRUE -- equal-weight alternative optima, then a
construction that degenerated into a valid certificate. Both looked exactly
like a program-ending false positive. Without the audit, a run can report a
clean sheet while attacking nothing at all.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from oneq.certifying_decoder import CertifyingDecoder          # noqa: E402
from oneq.forgeries import assert_all_false, forgeries         # noqa: E402
from oneq.matching_cert import BOUNDARY, check                 # noqa: E402


def main() -> int:
    import pymatching
    import stim
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else d
    p = 5e-3
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)
    m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    edges = [(int(e[0]), BOUNDARY if e[1] is None else int(e[1]),
              float(e[2].get("weight", 1.0))) for e in m.edges()]
    cd = CertifyingDecoder(edges)
    det, _ = circ.compile_detector_sampler(seed=99).sample(
        shots, separate_observables=True)

    tried, accepted = Counter(), Counter()
    reasons: dict[str, str] = {}
    genuine_ok = 0
    nontrivial = 0
    bogus: list = []
    for s in range(shots):
        fired = [int(i) for i in np.flatnonzero(det[s])]
        if not fired:
            continue
        nontrivial += 1
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY))
                     for a, b in m.decode_to_edges_array(det[s]))
        r = cd.certify(fired, corr)
        if r.accepted:
            genuine_ok += 1
        if not r.best_dual:
            continue
        battery = forgeries(cd.W, fired, corr, r.best_dual, r.primal)
        bogus += assert_all_false(cd.W, fired, r.primal, battery)
        for name, forged, _why in battery:
            tried[name] += 1
            v = check(edges=edges, syndrome=fired, cert=forged)
            if v.accepted:
                accepted[name] += 1
            else:
                reasons.setdefault(name, v.reason)

    total, acc = sum(tried.values()), sum(accepted.values())
    print(f"d={d} rounds={rounds}  {nontrivial} nontrivial shots")
    print(f"  genuine certificates accepted : {genuine_ok}/{nontrivial}")
    print(f"  forgeries attempted           : {total}")
    print(f"  forgeries ACCEPTED            : {acc}")
    print(f"  battery attacks that were NOT actually false: {len(bogus)}")
    for name in sorted(tried):
        flag = "  <-- FALSE POSITIVE" if accepted[name] else ""
        print(f"      {name:38s} {accepted[name]:4d}/{tried[name]:<5d} "
              f"{reasons.get(name,'-')[:44]}{flag}")

    out = {"schema": "oneq-false-positive-hunt/2", "distance": d,
           "rounds": rounds, "shots": shots, "nontrivial": nontrivial,
           "genuine_accepted": genuine_ok, "forgeries_attempted": total,
           "forgeries_accepted": acc, "per_attack": dict(tried),
           "per_attack_accepted": dict(accepted),
           "refusal_reasons": reasons,
           "battery_attacks_that_were_not_actually_false": len(bogus),
           "gate": ("ZERO forgeries accepted AND genuine certificates still "
                    "accepted AND the battery independently confirmed to be "
                    "attacking. A checker that refuses everything passes the "
                    "first clause and fails the point; a battery that attacks "
                    "nothing passes it vacuously.")}
    # BYTES, NOT TEXT: `write_text` translates a newline to
    # os.linesep on Windows, so re-running this would leave a
    # tracked LF artifact CRLF -- `git diff` empty, `git status`
    # modified. `tools/dem_cert_run.py` already uses the same
    # idiom via newline="".
    (ROOT / "evidence" / f"false_positive_hunt_d{d}.json").write_bytes(
        json.dumps(out, indent=2).encode("utf-8"))
    if bogus:
        print("VOID: the battery emitted attacks whose claims were true",
              file=sys.stderr)
        return 2
    return 0 if acc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
