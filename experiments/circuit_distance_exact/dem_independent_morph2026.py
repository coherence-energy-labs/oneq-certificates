r"""R198 part 17's circuits, their fault set derived WITHOUT stim's error analysis, and compared to stim's.

    python dem_independent_morph2026.py > ../../evidence/circuit_distance_exact/records/dem_independent_morph2026.jsonl

WHY. `dem_independent.py` removed stim's error analysis from the trusted base for the 66 models of the IBM,
three-colour and Shaw-Terhal 2024 claims, but its `--circuit` list predates `morph2026.py`, so the 14 circuits of
Table II (and the four models the Lean kernel checks) still rested on stim alone (paper 3's referee, 2026-10-29).
This runs the SAME comparison -- `dem_independent.compare`, unchanged: the backward Pauli-sensitivity pass against
the mechanism list `export_model.mechanisms` takes from stim, exact equality of mechanisms, detector coordinates
and observable count -- on every (target, basis, rounds) the campaign record ran (a window row's pair (T, T+1)
contributes both models), read from `records/morph2026_campaign.jsonl` so no model the claims rest on is skipped. `dem_independent.py` itself is pinned
by committed evidence, so it is imported, not edited.

Exit status 1 if any model differs: an unequal row is a finding, never a skip.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dem_independent as di  # noqa: E402
import morph2026 as m26  # noqa: E402

CAMPAIGN = HERE.parents[1] / "evidence" / "circuit_distance_exact" / "records" / "morph2026_campaign.jsonl"


def campaign_models(path: pathlib.Path = CAMPAIGN) -> list[tuple[str, str, int]]:
    """Every (target, basis, rounds) the campaign record ran, in a fixed order."""
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        name, basis = r.get("code"), r.get("basis")
        if name not in m26.TARGETS or basis not in ("Z", "X"):
            continue
        if r.get("kind") == "window" and isinstance(r.get("T"), int):
            seen.update({(name, basis, r["T"]), (name, basis, r["T"] + 1)})  # a window row compares T with T+1
        elif isinstance(r.get("rounds"), int):
            seen.add((name, basis, r["rounds"]))
    if not seen:
        raise SystemExit(f"no (target, basis, rounds) rows in {path}")
    order = list(m26.TARGETS)
    return sorted(seen, key=lambda k: (order.index(k[0]), k[1], k[2]))


def main() -> int:
    unequal = 0
    for name, basis, rounds in campaign_models():
        row = di.compare(m26.build(m26.spec(name), basis, rounds))
        unequal += not row["equal"]
        print(json.dumps(dict(circuit="morph2026", target=name, basis=basis, rounds=rounds, **row)), flush=True)
    return 1 if unequal else 0


if __name__ == "__main__":
    sys.exit(main())
