r"""Is every model of the IBM re-certification (evidence/circuit_distance_exact/ibm_recert_20261026/) the export of
IBM's depth-8 cycle? Re-export each one and compare the bytes.

    python ibm_recert_models.py > ../../evidence/circuit_distance_exact/records/ibm_recert_reexport.jsonl

WHY. The 22 shipped CDP5 proofs re-check against the 22 shipped model files (tests/test_r198_ibm_recert.py), but a
certificate about a model file says something about IBM's cycle only if that file IS the cycle's exported model.
The manifest records each model's SHA-256; nothing compared it with a fresh export (paper 3's referee, 2026-10-29).
This rebuilds every (basis, rounds) cell with `export_model.export(..., time_order=True)` -- the call `run_lane.py
certify --maxround 1` made -- into a temporary file and requires byte equality with the shipped model.

Exit status 1 if any cell differs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_model as em  # noqa: E402

PKG = HERE.parents[1] / "evidence" / "circuit_distance_exact" / "ibm_recert_20261026"


def reexport(o: dict, workdir: pathlib.Path) -> dict:
    """One manifest observation, re-exported and compared."""
    out = workdir / f"ibm_l12_{o['basis']}{o['rounds']}.txt"
    info = em.export("ibm", o["basis"], o["rounds"], out, l=12, time_order=True)
    got = hashlib.sha256(out.read_bytes()).hexdigest()
    shipped = hashlib.sha256((PKG / o["model"]).read_bytes()).hexdigest()
    return dict(basis=o["basis"], rounds=o["rounds"], model=o["model"], manifest_sha256=o["model_sha256"],
                shipped_sha256=shipped, reexport_sha256=got, mechanisms=info["mechanisms"],
                equal=got == shipped == o["model_sha256"])


def main() -> int:
    obs = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))["observations"]
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for o in sorted(obs, key=lambda x: (x["basis"], x["rounds"])):
            row = reexport(o, pathlib.Path(td))
            bad += not row["equal"]
            print(json.dumps(row), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
