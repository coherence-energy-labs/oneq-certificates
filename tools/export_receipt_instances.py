r"""Persist every normative decoding instance the paper-2 receipts are proofs about.

WHY (external audit 2026-09-14, R-03). Replay used to REBUILD the check
matrices and even the integer weights (through float logarithms and
rounding) and then check receipts against whatever it rebuilt. A valid proof
about a rebuilt instance is a proof about that instance; nothing bound it to
the instance the published experiment used. Now the instance itself is an
artifact: check supports as integer lists, with a sha256 over a canonical
encoding, written once here. Replay (tools/replay_receipts.py) reads these
files and the shipped integer weight vectors, never a generator.

Rebuilding is kept, but as a separate reproducibility check: `--verify`
regenerates every instance from its construction and requires exact
equality with the stored file.

    python tools/export_receipt_instances.py            # write
    python tools/export_receipt_instances.py --verify   # rebuild must equal stored
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "instances"


def canonical_checks_sha256(checks) -> str:
    """sha256 of the check supports as sorted integer lists, compact JSON."""
    body = json.dumps([sorted(int(i) for i in c) for c in checks], separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def legacy_h_hash(checks) -> str:
    """The 16-hex H_hash the per-shot records carry (qldpc_exact_all_gate._hash_checks)."""
    return hashlib.sha256(json.dumps([list(c) for c in checks]).encode()).hexdigest()[:16]


def build() -> dict[str, dict]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
    from qldpc_bb_gate import bb_gross_code
    from qldpc_bb_phenom import spacetime_system
    from qldpc_bb_refsched_gate import build_refsched_circuit
    from qldpc_circuit_gate import dem_to_matrices

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    cz = [tuple(int(i) for i in np.flatnonzero(HZ[j])) for j in range(m)]
    cx = [tuple(int(i) for i in np.flatnonzero(HX[j])) for j in range(m)]
    st1, _ = spacetime_system(HZ, 4, 0.01, 0.01)
    st2, _ = spacetime_system(HZ, 4, 0.02, 0.02)
    if [list(c) for c in st1] != [list(c) for c in st2]:
        raise SystemExit("phenomenological check structure depends on p; store both")
    dem = build_refsched_circuit(3, 0.002).detector_error_model(decompose_errors=False)
    cc, _wllr, _obs, nmech = dem_to_matrices(dem)

    spec = {
        "bb144_z_code_capacity": (cz, n, "[[144,12,12]] H_Z rows (Z checks detect X errors), "
                                        "bb_gross_code()"),
        "bb144_x_code_capacity": (cx, n, "[[144,12,12]] H_X rows (X-sector decoding), bb_gross_code()"),
        "bb144_phenom_r4": ([tuple(c) for c in st1], None,
                            "spacetime_system(H_Z, rounds=4): detectors telescope to new-error "
                            "mechanisms; check structure independent of p"),
        "bb144_refsched_r3_p002": ([tuple(int(i) for i in c) for c in cc], nmech,
                                   "detector error model of build_refsched_circuit(3, 0.002), "
                                   "decompose_errors=False, via dem_to_matrices"),
    }
    out = {}
    for name, (checks, nvar, definition) in spec.items():
        if nvar is None:
            nvar = 1 + max(i for c in checks for i in c)
        out[name] = {"schema": "oneq-decoding-instance/1", "name": name,
                     "definition": definition, "n_variables": int(nvar),
                     "checks": [sorted(int(i) for i in c) for c in checks],
                     "checks_sha256": canonical_checks_sha256(checks),
                     "legacy_H_hash": legacy_h_hash(checks)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    built = build()
    if a.verify:
        bad = []
        for name, doc in built.items():
            p = OUT / f"{name}.json"
            if not p.exists() or json.loads(p.read_text(encoding="utf-8")) != doc:
                bad.append(name)
        print("REBUILD " + ("MATCHES every stored instance" if not bad else f"DIFFERS: {bad}"))
        return 1 if bad else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc in built.items():
        (OUT / f"{name}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n",
                                          encoding="utf-8", newline="\n")
        print(f"wrote {name}: {len(doc['checks'])} checks, n={doc['n_variables']}, "
              f"sha256 {doc['checks_sha256'][:16]}, legacy H_hash {doc['legacy_H_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
