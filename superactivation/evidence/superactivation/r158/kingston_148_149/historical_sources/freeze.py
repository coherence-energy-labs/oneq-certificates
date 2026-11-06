"""Freeze a run: the ISA circuits themselves are the registered object, and the hardware run LOADS them.

Generalizes experiments/tw_teleport/tw2/freeze.py. `freeze(named, meta)` returns (record, qpy_bytes):
  qpy_bytes  QPY of every circuit in the registered order
  record     the order, the SHA-256 of each circuit's own QPY and of the whole file, library versions, the SHA-256
             of every source file named in meta["sources"] (line endings normalized, so a CRLF checkout hashes like
             the committed bytes), and the caller's meta: snapshot hash and calibration time, layout choice, noise
             parameters, the prediction, the preregistration's path and hash.
`load(record, qpy_bytes)` verifies the file hash and the circuit names and order before returning anything, and
refuses on any mismatch. The run never re-transpiles: library patch versions change transpiler output (observed,
qiskit 2.5.0 vs 2.5.2), so the circuits that were predicted must be the circuits that run.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import pathlib
from importlib import metadata

PACKAGES = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy")


class FrozenMismatch(ValueError):
    """the frozen bytes, names or sources do not match their record"""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def source_sha256(path) -> str:
    return sha256(pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n"))


def versions() -> dict:
    out: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _qpy(circuits) -> bytes:
    from qiskit import qpy
    buf = io.BytesIO()
    qpy.dump(list(circuits), buf)
    return buf.getvalue()


def freeze(named: list, meta: dict) -> tuple[dict, bytes]:
    """named: [(name, ISA QuantumCircuit)] in the registered order"""
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        raise FrozenMismatch("duplicate circuit names: a result could be looked up by the wrong circuit")
    circuits = []
    for n, qc in named:
        c = qc.copy()
        c.name = n
        circuits.append(c)
    blob = _qpy(circuits)
    sources = {str(p): source_sha256(p) for p in meta.get("sources", ())}
    record = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        circuit_order=names, circuit_sha256={c.name: sha256(_qpy([c])) for c in circuits}, qpy_sha256=sha256(blob),
        n_circuits=len(circuits), versions=versions(), source_sha256=sources,
        **{k: v for k, v in meta.items() if k != "sources"})
    return record, blob


def load(record: dict, blob: bytes, check_sources: bool = True) -> dict:
    """name -> circuit, after every check; raises FrozenMismatch"""
    from qiskit import qpy
    if sha256(blob) != record["qpy_sha256"]:
        raise FrozenMismatch("the QPY bytes do not hash to the frozen record")
    circuits = qpy.load(io.BytesIO(blob))
    names = [c.name for c in circuits]
    if names != record["circuit_order"]:
        raise FrozenMismatch("circuit names or order differ from the frozen record")
    if check_sources:
        changed = [p for p, h in record.get("source_sha256", {}).items()
                   if not pathlib.Path(p).exists() or source_sha256(p) != h]
        if changed:
            raise FrozenMismatch(f"sources changed since the freeze: {changed} (re-freeze and re-predict)")
    return dict(zip(names, circuits))
