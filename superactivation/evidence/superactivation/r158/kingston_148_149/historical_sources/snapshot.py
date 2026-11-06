"""A device calibration snapshot: everything a prediction or a layout choice reads, frozen and hashable.

Generalizes experiments/tw_teleport/tw2/layout.DeviceSnapshot (which carried one symmetric readout error) with what
the hardware record says matters:
  - ASYMMETRIC readout. On ibm_kingston P(read 0 | prep 1) is 2-10x P(read 1 | prep 0), from T1 decay during the
    readout (ledger R81, R138). Bob's frame bits that should read 1 flip more often.
  - Every per-qubit duration that idle relaxation is computed from.
  - The calibration timestamp, so a record can show which window it used. The same layout lost 1.4 points between
    two windows on one day (R138).
  - A content hash (sha256 of canonical JSON), so a frozen design names the exact calibration it was scored on.
`from_backend` only READS a BackendV2 (fake or live): no job, no credits.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

NAN = float("nan")


class SnapshotError(ValueError):
    """the snapshot is incomplete or inconsistent for the qubits a caller asked about"""


@dataclass(frozen=True)
class DeviceSnapshot:
    name: str
    num_qubits: int
    calibrated_utc: str | None          # backend.properties().last_update_date, when the backend reports one
    dt: float                           # seconds per sample
    cz: dict                            # (a, b), a < b -> (error, duration_s); the worse of the two directions
    t1: tuple                           # seconds; nan if unknown
    t2: tuple
    p01: tuple                          # P(read 1 | prep 0); nan if unknown
    p10: tuple                          # P(read 0 | prep 1)
    sx_error: tuple
    sx_duration: tuple                  # seconds
    meas_duration: tuple                # seconds
    # (a, b), a < b -> 15 fractions, in noise.PAULIS2 order (IX, IY, IZ, XI, ..., ZZ; qubit a first), of that edge's CZ
    # error probability: its full Pauli channel's SHAPE, the rate staying in `cz`. MEASURED in-circuit (R146,
    # oneq.hw.cz_channel); a vendor calibration has none (uniform), and an empty map is left out of the JSON, so every
    # earlier snapshot keeps its hash.
    cz_structure: dict = field(default_factory=dict)
    # (a, b), a < b -> zeta in rad/s: the always-on CONDITIONAL PHASE between coupled qubits, relative to a frame
    # calibrated with the neighbours in |0> -- CPhase(zeta t) over a shared idle (R153; MEASURED from a calibration
    # block's xs family by experiments/superactivation/shield_family/zz_fit.py). Empty for a vendor calibration, and
    # an empty map is left out of the JSON, so every earlier snapshot keeps its hash and its predictions.
    zz: dict = field(default_factory=dict)
    # q -> sigma in rad/s: the spread of qubit q's QUASI-STATIC detuning -- a frequency offset fixed within a shot and
    # random from shot to shot (R155: a Ramsey at two waits against an echo, experiments/superactivation/shield_family/
    # r155_idle_map.py). Over a window of tau it dephases as exp(-sigma^2 tau^2 / 2), a Gaussian a single exponential T2
    # cannot represent, and an echo cancels it (noise.compile_ops). Where sigma is given, t2 must be the WHITE (echo)
    # T2, or the static part is counted twice. Empty for a vendor calibration, and an empty map is left out of the JSON,
    # so every earlier snapshot keeps its hash and its predictions.
    sigma: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for q, s in self.sigma.items():
            if not (isinstance(q, int) and 0 <= q < self.num_qubits):
                raise SnapshotError(f"quasi-static sigma for qubit {q!r}, which is not a qubit of {self.name}")
            if not (math.isfinite(s) and s >= 0.0):
                raise SnapshotError(f"quasi-static sigma for qubit {q} must be a finite spread >= 0 rad/s: {s}")
        for key, z in self.zz.items():
            if key not in self.cz or key[0] >= key[1]:
                raise SnapshotError(f"crosstalk for {key}, which is not a calibrated edge stored a < b")
            if not (math.isfinite(z) and z >= 0.0):
                raise SnapshotError(f"crosstalk zeta for {key} must be a finite rate >= 0 rad/s: {z}")
        for key, frac in self.cz_structure.items():
            if key not in self.cz:
                raise SnapshotError(f"CZ structure for edge {key}, which has no CZ calibration")
            if len(frac) != 15 or min(frac) < 0.0 or abs(sum(frac) - 1.0) > 1e-9:
                raise SnapshotError(f"CZ structure for edge {key} is not 15 non-negative fractions summing to 1: {frac}")

    # ---------------------------------------------------------------- derived
    def readout_error(self, q: int) -> float:
        return 0.5 * (self.p01[q] + self.p10[q])

    def edge(self, a: int, b: int) -> tuple:
        key = (a, b) if a < b else (b, a)
        if key not in self.cz:
            raise SnapshotError(f"no CZ calibration for edge {key} on {self.name}")
        return self.cz[key]

    def edge_structure(self, a: int, b: int) -> tuple | None:
        """the edge's 15 measured fractions in (a, b) qubit order -- or None, a uniform channel. Stored a < b; asked
        the other way round, the two qubits' letters swap."""
        if a < b:
            return self.cz_structure.get((a, b))
        frac = self.cz_structure.get((b, a))
        if frac is None:
            return None
        from .noise import PAULIS2
        return tuple(frac[PAULIS2.index((q, p))] for p, q in PAULIS2)

    def complete(self, q: int) -> bool:
        """every number a prediction reads for qubit q is finite and physical"""
        vals = (self.t1[q], self.t2[q], self.p01[q], self.p10[q], self.sx_error[q], self.sx_duration[q],
                self.meas_duration[q])
        return all(math.isfinite(v) for v in vals) and self.t1[q] > 0 and self.t2[q] > 0

    # ---------------------------------------------------------------- serialization
    def to_json(self) -> dict:
        d = asdict(self)
        d["cz"] = {f"{a}_{b}": list(v) for (a, b), v in sorted(self.cz.items())}
        d.pop("cz_structure")
        d.pop("zz")
        d.pop("sigma")
        if self.sigma:
            d["sigma"] = {str(q): float(v) for q, v in sorted(self.sigma.items())}
        if self.zz:
            d["zz"] = {f"{a}_{b}": float(v) for (a, b), v in sorted(self.zz.items())}
        if self.cz_structure:
            d["cz_structure"] = {f"{a}_{b}": list(v) for (a, b), v in sorted(self.cz_structure.items())}
        for k in ("t1", "t2", "p01", "p10", "sx_error", "sx_duration", "meas_duration"):
            d[k] = [None if not math.isfinite(x) else x for x in d[k]]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "DeviceSnapshot":
        cz = {tuple(int(x) for x in k.split("_")): (float(v[0]), float(v[1])) for k, v in d["cz"].items()}
        tup = {k: tuple(NAN if x is None else float(x) for x in d[k])
               for k in ("t1", "t2", "p01", "p10", "sx_error", "sx_duration", "meas_duration")}
        st = {tuple(int(x) for x in k.split("_")): tuple(float(x) for x in v)
              for k, v in d.get("cz_structure", {}).items()}
        zz = {tuple(int(x) for x in k.split("_")): float(v) for k, v in d.get("zz", {}).items()}
        sigma = {int(k): float(v) for k, v in d.get("sigma", {}).items()}
        return cls(name=d["name"], num_qubits=int(d["num_qubits"]), calibrated_utc=d["calibrated_utc"],
                   dt=float(d["dt"]), cz=cz, cz_structure=st, zz=zz, sigma=sigma, **tup)

    def sha256(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _f(x) -> float:
    return NAN if x is None else float(x)


def from_backend(backend) -> DeviceSnapshot:
    """read-only extraction from a BackendV2 target (+ properties() for asymmetric readout when it exists)"""
    tgt = backend.target
    nq = tgt.num_qubits
    cz: dict = {}
    for qargs, props in tgt["cz"].items():
        if props is None or props.error is None:
            continue
        a, b = sorted(qargs)
        err, dur = float(props.error), _f(props.duration)
        if (a, b) in cz:
            err = max(err, cz[(a, b)][0])
        cz[(a, b)] = (err, dur)
    qp = tgt.qubit_properties

    def qprop(q, attr):
        return _f(getattr(qp[q], attr, None)) if qp is not None and qp[q] is not None else NAN

    def per_q(op, attr):
        out = []
        for q in range(nq):
            p = tgt[op].get((q,)) if op in tgt.operation_names else None
            out.append(_f(getattr(p, attr, None)) if p is not None else NAN)
        return tuple(out)

    ro = per_q("measure", "error")
    p01, p10 = list(ro), list(ro)                       # fallback: symmetric
    calibrated = None
    props = backend.properties() if hasattr(backend, "properties") else None
    if props is not None:
        calibrated = str(getattr(props, "last_update_date", None))
        for q in range(nq):
            try:
                p01[q] = float(props.qubit_property(q, "prob_meas1_prep0")[0])
                p10[q] = float(props.qubit_property(q, "prob_meas0_prep1")[0])
            except Exception:                           # noqa: BLE001 -- a qubit without the split keeps the symmetric value
                pass
    return DeviceSnapshot(
        name=getattr(backend, "name", "backend"), num_qubits=nq, calibrated_utc=calibrated, dt=float(backend.dt),
        cz=cz, t1=tuple(qprop(q, "t1") for q in range(nq)), t2=tuple(qprop(q, "t2") for q in range(nq)),
        p01=tuple(p01), p10=tuple(p10), sx_error=per_q("sx", "error"), sx_duration=per_q("sx", "duration"),
        meas_duration=per_q("measure", "duration"))
