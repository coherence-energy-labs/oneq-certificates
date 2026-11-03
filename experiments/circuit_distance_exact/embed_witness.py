r"""R198: does a witness found in a T0-round model exist, unchanged, in the T-round model? (the upper bound for every T)

    python embed_witness.py --circuit 3c --basis Z --t0 2 --tmax 12 --ids 1 4 58 ...

A projected witness is a set of mechanisms identified by (detector coordinates (site, type, round), observables).
For each T it looks up the mechanisms with exactly those coordinates in the T-round projected model, checks that
the set is undetectable and flips a logical there, and lifts it to the T-round FULL DEM (meet in the middle) and
verifies that. A start-boundary witness (rounds 0..r) should embed for every T > r; an end-boundary one must be
shifted instead -- this script only embeds by identical coordinates and reports per T, so it can fail.
JSON lines on stdout; writes nothing under evidence/.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402
import tools_cde as tc  # noqa: E402


def keyed(model: pathlib.Path):
    mech, det, T = tc.read_model(model)
    keys = [(tuple(sorted(det[d] for d in dets)), obs) for dets, obs in mech]
    return mech, det, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", choices=bc.CIRCUITS, required=True)
    ap.add_argument("--l", type=int, default=12)
    ap.add_argument("--basis", choices=("Z", "X"), required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="r198_embed_"))
    src = work / "t0.txt"
    em.export(a.circuit, a.basis, a.t0, src, l=a.l)
    _, _, keys0 = keyed(src)
    want = [keys0[j] for j in a.ids]
    for T in range(a.t0, a.tmax + 1):
        m = work / f"t{T}.txt"
        em.export(a.circuit, a.basis, T, m, l=a.l)
        mech, det, keys = keyed(m)
        index = {k: j for j, k in enumerate(keys)}
        ids = [index.get(k) for k in want]
        rec = dict(circuit=a.circuit, basis=a.basis, t0=a.t0, T=T, weight=len(want), embedded=None not in ids)
        if rec["embedded"]:
            syn, obs = set(), 0
            for j in ids:
                syn ^= set(mech[j][0])
                obs ^= mech[j][1]
            rec["projected_undetectable_logical"] = (not syn) and obs != 0
            full = tc.lift(a.circuit, a.basis, T, ids, a.l) if rec["projected_undetectable_logical"] else None
            rec["full_lift"] = full
            rec["full_verified"] = bool(full) and tc.verify_full(a.circuit, a.basis, T, full, a.l)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
