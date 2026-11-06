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

A source pin names a file IN THE CHECKOUT THAT RUNS. Every freeze before 2026-10-18 keyed its pins by the absolute
path it was made at, and `load` read that path: a submission from a worktree checked the main checkout's files, and
R159's never-run Fez receiver, frozen in the ONE_Q-R155-check worktree, passed from main by reading THAT tree's
superseded relay compiler. `freeze` now writes repo-relative POSIX keys (`pin_key`), and `load` maps every key,
legacy absolute ones included, onto this checkout (`resolve_pin`).
"""
from __future__ import annotations

import datetime
import hashlib
import io
import pathlib
from importlib import metadata

PACKAGES = ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy")
ROOT = pathlib.Path(__file__).resolve().parents[3]          # src/oneq/hw/freeze.py -> the checkout
TOP_LEVEL = ("src", "experiments", "tools", "tests", "docs", "evidence")


class FrozenMismatch(ValueError):
    """the frozen bytes, names or sources do not match their record"""


def pin_key(path) -> str:
    """the key a source is pinned under: its POSIX path inside this checkout, so one pin names the same file in every
    clone and worktree; a file outside the checkout (a scratch plan, say) keeps its absolute path"""
    p = pathlib.Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def resolve_pin(key: str) -> pathlib.Path | None:
    """the file a pin names, read from THIS checkout whenever it is a file of the repository; None when it cannot be.
      - a relative key is read under ROOT ('..' refused: a pin may not climb out);
      - an absolute key -- a Windows checkout's, a Linux home's, another worktree's -- is read at its
        repo-relative tail, from the first top-level directory of this repository at which the remainder is a file
        HERE; the old prefix is never read, so no other tree on the machine can answer for this one;
      - an absolute key into ANOTHER checkout (a .git at the prefix before a top-level directory) whose file this
        checkout lacks resolves to None: the code that runs here does not have it;
      - any other absolute key is an external file (a scratch plan outside every checkout) and is read where it is."""
    k = key.replace("\\", "/")
    parts = [x for x in k.split("/") if x]
    if ".." in parts or not parts:
        return None
    drive = len(parts[0]) == 2 and parts[0][1] == ":"
    if not (k.startswith("/") or drive):
        return ROOT.joinpath(*parts)
    tops = [i for i, part in enumerate(parts) if part in TOP_LEVEL]
    for i in tops:
        if ROOT.joinpath(*parts[i:]).is_file():
            return ROOT.joinpath(*parts[i:])
    anchor = pathlib.Path(parts[0] + "/") if drive else pathlib.Path("/")
    for i in tops:
        if anchor.joinpath(*parts[1 if drive else 0:i], ".git").exists():
            return None
    return pathlib.Path(key)


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
    sources = {pin_key(p): source_sha256(p) for p in meta.get("sources", ())}
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
        changed = []
        for key, h in record.get("source_sha256", {}).items():
            here = resolve_pin(key)
            if here is None or not here.is_file() or source_sha256(here) != h:
                changed.append(key)
        if changed:
            raise FrozenMismatch(f"sources changed since the freeze: {changed} (re-freeze and re-predict)")
    return dict(zip(names, circuits))
