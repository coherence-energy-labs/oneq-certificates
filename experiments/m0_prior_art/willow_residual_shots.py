r"""The two Willow d=7x30 shots the full ladder left uncertified: decoder or producer?

Paper 1 reports that after the exact tier one shot per basis on Willow d=7,
30 rounds stayed uncertified, at gaps 0.803199 (X) and 0.111032 (Z). The exact
tier's CONVERGED bound equals the minimum T-join weight, so the two readings
are opposite findings (external audit 2026-09-14, B4):

  * the separation converged  -> PyMatching's correction is heavier than the
                                 optimum on real hardware syndromes;
  * it stalled or ran out     -> our producer stopped short; nothing about
                                 the decoder follows.

STEP 1 (pinned code): the gen-6 run is replayed on the one 50-shot chunk that
contained each residual, with the code of the artifact's commit, to identify
the shot whose gap reproduces exactly.
STEP 2 (HEAD): for that shot, (a) an independent minimum-weight T-join --
exact_min_tjoin, which scores in exact fractions (its networkx matching breaks
float ties, immaterial against a 0.1-0.8 gap); (b) PyMatching's correction
weight in exact fractions; (c) today's exact tier with its recorded exit
reason.

    python experiments/m0_prior_art/willow_residual_shots.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence" / "willow_residual_shots.json"
COMMIT = "f57e87d"
TARGETS = {"X": ("d7_at_q6_7_X_r30", 26700, 26750, 0.803199),
           "Z": ("d7_at_q6_7_Z_r30", 5300, 5350, 0.111032)}


def _find(code_root: str, config: str, lo: int, hi: int, gap: float) -> dict:
    """Runs INSIDE a subprocess against the pinned tree."""
    sys.path.insert(0, str(pathlib.Path(code_root) / "src"))
    sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
    import numpy as np

    import oneq
    assert pathlib.Path(oneq.__file__).resolve().is_relative_to(pathlib.Path(code_root).resolve())
    from google_corpus_gate import load_config
    from oneq.certifying_decoder import CertifyingDecoder
    from oneq.matching_cert import BOUNDARY

    m, edges, det, _meta = load_config(ROOT / "data" / "google_corpus" / config)
    cd = CertifyingDecoder(edges)
    window = det.window(lo, hi)
    found = []
    for s in range(lo, hi):
        fired = [int(i) for i in np.flatnonzero(window[s - lo])]
        if not fired:
            continue
        corr = tuple(cd.key((a, b if b >= 0 else BOUNDARY)) for a, b in m.decode_to_edges_array(window[s - lo]))
        r = cd.certify_hard(fired, corr)
        if not r.accepted:
            found.append({"shot": s, "gap": round(r.gap, 6), "fired": fired,
                          "correction": [list(e) for e in corr]})
    return {"found": found, "gap_expected": gap}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--find":
        args = json.loads(sys.argv[2])
        res = _find(**args)
        pathlib.Path(sys.argv[3]).write_text(json.dumps(res), encoding="utf-8")
        return 0

    sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
    from certification_semantics_audit import _archive

    out = {"schema": "oneq-willow-residual-shots/1", "pinned_commit": COMMIT, "shots": {}}
    with tempfile.TemporaryDirectory() as tmp:
        code = pathlib.Path(tmp) / COMMIT
        _archive(COMMIT, code)
        for basis, (config, lo, hi, gap) in TARGETS.items():
            dump = pathlib.Path(tmp) / f"{basis}.json"
            subprocess.run([sys.executable, __file__, "--find",
                            json.dumps({"code_root": str(code), "config": config, "lo": lo,
                                        "hi": hi, "gap": gap}), str(dump)],
                           check=True, env={**os.environ, "PYTHONPATH": ""})
            found = json.loads(dump.read_text(encoding="utf-8"))["found"]
            match = [f for f in found if abs(f["gap"] - gap) < 1e-6]
            row = {"config": config, "chunk": [lo, hi], "pinned_uncertified": len(found),
                   "gap_reproduced": len(match) == 1}
            if len(match) != 1:
                row["note"] = f"expected exactly one shot at gap {gap}; found {[f['gap'] for f in found]}"
                out["shots"][basis] = row
                continue
            shot = match[0]
            row.update(shot=shot["shot"], gap=shot["gap"], detectors_fired=len(shot["fired"]))

            sys.path.insert(0, str(ROOT / "src"))
            import oneq.tjoin_exact as tj
            from google_corpus_gate import load_config
            from oneq.certifying_decoder import CertifyingDecoder
            from oneq.exact_decode import exact_min_tjoin

            m, edges, _det, _meta = load_config(ROOT / "data" / "google_corpus" / config)
            cd = CertifyingDecoder(edges)
            corr = tuple(tuple(e) for e in shot["correction"])
            primal = sum((Fraction(cd.W[cd.key(e)]) for e in corr), Fraction(0))
            _oc, ow, why = exact_min_tjoin(edges, shot["fired"])
            row["pymatching_weight"] = float(primal)
            row["exact_min_tjoin_weight"] = None if ow is None else float(ow)
            row["pymatching_minus_optimum"] = None if ow is None else float(primal - ow)
            row["oracle_note"] = why
            r3 = cd.certify_exact(shot["fired"], corr)
            row["head_exact_tier"] = {"accepted": bool(r3.accepted), "bound": float(r3.bound),
                                      "gap": float(r3.gap), "exit": dict(tj.LAST_EXIT)}
            out["shots"][basis] = row
            print(f"{basis}: shot {row['shot']}  PyMatching {row['pymatching_weight']:.6f}  "
                  f"optimum {row['exact_min_tjoin_weight']}  excess {row['pymatching_minus_optimum']}  "
                  f"exact tier now: accepted={r3.accepted} exit={tj.LAST_EXIT}", flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"evidence -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
