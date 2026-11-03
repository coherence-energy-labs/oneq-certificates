r"""wu_bind.py -- bind the window formula to the exact files the lane's certificates and searches ran on.

export_model.export (the lane's exporter, imported read-only) writes a projected model into OUR output directory;
its sha256 must equal the record's, and its mechanism set (read back through the detector coordinates the file
lists) must equal the model this package computes. Records are read from evidence/circuit_distance_exact/records.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

LANE = pathlib.Path(__file__).resolve().parent
RECORDS = LANE.parents[1] / "evidence" / "circuit_distance_exact" / "records"
RESULTS = LANE.parents[1] / "evidence" / "circuit_distance_exact" / "results.json"
sys.path.insert(0, str(LANE))

import export_model as em  # noqa: E402


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_masks(path: pathlib.Path, n2: int = 72) -> set:
    """An exported model file as a set of (detector mask, observable mask) in this package's indexing."""
    with open(path) as f:
        nd, nm, no, norb, gl, gm = map(int, f.readline().split())
        mech = []
        for _ in range(nm):
            p = list(map(int, f.readline().split()))
            mech.append((p[1:1 + p[0]], p[1 + p[0]]))
        f.readline()
        det = [tuple(map(int, f.readline().split())) for _ in range(nd)]
    W = 2 * n2
    idx = [r * W + t * n2 + k for k, t, r in det]
    out = set()
    for dets, obs in mech:
        dm = 0
        for d in dets:
            dm ^= 1 << idx[d]
        out.add((dm, obs))
    return out


def recorded_models() -> list:
    """[(circuit, basis, T, time_order, sha256, mechanisms or None, detectors or None, source)] from the lane's
    certificate and sweep records."""
    out = []
    for line in (RECORDS / "ibm_proofs_w9.jsonl").read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            out.append(("ibm", r["basis"], r["T"], False, r["dem_sha256"], None, None, "ibm_proofs_w9"))
    for name in ("3c_T2_certificates_w10.jsonl", "3c_sweep_shift_w10.jsonl"):
        for line in (RECORDS / name).read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                out.append(("3c", r["basis"], r["rounds"], bool(r.get("time_order")), r["model_sha256"],
                            r.get("mechanisms"), r.get("detectors"), name.split(".")[0]))
    return out


def recorded_sizes() -> list:
    """[(circuit, basis, T, mechanisms, detectors, source)]: the IBM sweep's 'ND .. NM ..' heads and the 3-colour
    sweep's fields."""
    import re
    out = []
    for line in (RECORDS / "ibm_sweep_w9.jsonl").read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            m = re.search(r"ND (\d+) NM (\d+)", r["head"])
            out.append(("ibm", r["basis"], r["T"], int(m.group(2)), int(m.group(1)), "ibm_sweep_w9"))
    for line in (RECORDS / "3c_sweep_shift_w10.jsonl").read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            out.append(("3c", r["basis"], r["rounds"], r["mechanisms"], r["detectors"], "3c_sweep_shift_w10"))
    return out


def export_and_hash(kind: str, basis: str, T: int, time_order: bool, outdir: pathlib.Path):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{kind}_{basis}{T}{'_to' if time_order else ''}.txt"
    em.export(kind, basis, T, p, project=True, time_order=time_order)
    return p, sha(p)


def witnesses() -> list:
    """[(circuit, basis, rounds, full ids)] from results.json (IBM and 3-colour weight-10 / 11 witnesses)."""
    r = json.loads(RESULTS.read_text())
    out = []
    for key, kind in (("ibm_gross", "ibm"), ("three_colour", "3c")):
        for w in r[key]["witnesses"]:
            out.append((kind, w["basis"], w["rounds"], list(w["full"]), w["weight"]))
    return out
