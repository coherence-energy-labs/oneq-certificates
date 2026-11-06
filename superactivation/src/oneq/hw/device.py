"""The device posterior: what experiments have established about a device, in the only coordinates experiments CAN
establish, each with its uncertainty and where it came from.

Coordinates (all independent Gaussian here; correlations are a named simplification):
  readout   ("p01", q), ("p10", q)                    probabilities
  sx        ("sx", q)                                 the engine's total sx error probability
  coherence ("t2", q), ("t1", q)                      seconds, effective in-circuit values
  CZ        ("cz", a, b, key)                         the 9 LEARNABLE quantities of the edge's twirled CZ
                                                      (oneq.hw.cz_channel: a CZ-fixed Pauli's f^2, or a pair's product)
and the device's schedule. Everything not measured falls back to the vendor snapshot it was built over, and says so.

THE GAUGE LAW: a CZ coordinate must be one of the 9 learnable keys. The other 6 directions of a twirled CZ's noise
cannot be seen by any experiment, so a posterior holding one would be holding a number no measurement could ever
check; it is refused at construction. A snapshot for the engine is produced at the posterior's mean (or a draw) in
the symmetric gauge.

Versioned and content-addressed: sha256 over canonical JSON names the exact posterior a prediction was made from."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from oneq.hw import cz_channel as CC
from oneq.hw import snapshot as S

_KEYS = {str(k): k for k in {CC.learnable_key(p) for p in CC.NONI}}
LEARNABLE = tuple(sorted(_KEYS))                         # the 9, as the posterior's string keys


class GaugeError(ValueError):
    """a coordinate no experiment can determine"""


@dataclass(frozen=True)
class Coord:
    mean: float
    sd: float
    source: str                      # "measured" | "vendor"
    estimator: str                   # the fit that produced it, e.g. "hw_diag_run.fit_cz_channels@<sha>"
    epoch: str                       # calibration time or job id it describes

    def to_json(self) -> list:
        return [self.mean, self.sd, self.source, self.estimator, self.epoch]


def _key_str(key: tuple) -> str:
    return "|".join(str(x) for x in key)


def _key_from(s: str) -> tuple:
    parts = s.split("|")
    if parts[0] == "cz":
        return ("cz", int(parts[1]), int(parts[2]), parts[3])
    return (parts[0], int(parts[1]))


@dataclass
class DevicePosterior:
    base: S.DeviceSnapshot                               # the vendor calibration every unmeasured number comes from
    coords: dict = field(default_factory=dict)           # key -> Coord
    schedule: str = "alap"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in self.coords:
            if key[0] == "cz":
                _, a, b, lk = key
                if lk not in _KEYS:
                    raise GaugeError(f"{key}: not a learnable quantity of a twirled CZ -- the 9 are {sorted(_KEYS)}")
                if (a, b) not in self.base.cz:
                    raise S.SnapshotError(f"{key}: the base snapshot has no edge {(a, b)}")
            elif key[0] not in ("p01", "p10", "sx", "t1", "t2"):
                raise ValueError(f"unknown coordinate {key}")
        for c in self.coords.values():
            if not (math.isfinite(c.mean) and math.isfinite(c.sd) and c.sd >= 0):
                raise ValueError(f"coordinate is not a finite mean with a non-negative sd: {c}")

    # ------------------------------------------------------------ the engine's view
    def _snapshot_from(self, value) -> S.DeviceSnapshot:
        d = self.base.to_json()
        edges: dict = {}
        for key, c in self.coords.items():
            v = value(key, c)
            if key[0] == "cz":
                edges.setdefault((key[1], key[2]), {})[_KEYS[key[3]]] = v
            elif key[0] == "sx":
                d["sx_error"][key[1]] = max(v, 0.0) / 1.5
            elif key[0] in ("p01", "p10"):
                d[key[0]][key[1]] = min(max(v, 0.0), 0.5)
            else:
                d[key[0]][key[1]] = max(v, 1e-9)
        st = dict(d.get("cz_structure", {}))
        for (a, b), g in edges.items():
            if len(g) != 9:
                raise ValueError(f"edge {(a, b)} has {len(g)} of its 9 learnable quantities: a partial edge cannot be "
                                 f"turned into a channel")
            # the fit's own regularization, exactly: invert as measured (shot noise may put a quantity above 1), then
            # clip negative probabilities -- clamping g at 1 first would give the engine a different device
            ch = CC.project_physical(CC.symmetric_channel({k: max(v, 0.0) for k, v in g.items()}))
            p = 1 - ch["II"]
            d["cz"][f"{a}_{b}"] = [p / 1.25, d["cz"][f"{a}_{b}"][1]]
            st[f"{a}_{b}"] = [ch[q] / p for q in CC.NONI] if p > 0 else [1 / 15] * 15
        d["cz_structure"] = st
        return S.DeviceSnapshot.from_json(d)

    def snapshot(self) -> S.DeviceSnapshot:
        """the engine's snapshot at the posterior mean (symmetric gauge), to run at cz_factor 1"""
        return self._snapshot_from(lambda key, c: c.mean)

    def draw(self, rng: np.random.Generator) -> S.DeviceSnapshot:
        """one draw of the device, every coordinate independently from its posterior"""
        return self._snapshot_from(lambda key, c: c.mean + rng.normal(0.0, c.sd) if c.sd > 0 else c.mean)

    # ------------------------------------------------------------ learning
    def update(self, other: "DevicePosterior") -> "DevicePosterior":
        """combine with an independent measurement of the same device: precision-weighted per coordinate (a coordinate
        only one side measured is carried as it is, and an exact one -- sd 0, infinite precision -- outweighs any noisy
        one). Refuses a different base calibration -- that is drift, not more data, and belongs to a time-indexed model,
        not a merge -- and two exact values that disagree: no weighting reconciles them, and keeping either one would
        silently discard the other."""
        if other.base.sha256() != self.base.sha256():
            raise ValueError("a different base calibration: combine across epochs with a drift model, not update()")
        out = dict(self.coords)
        for key, c in other.coords.items():
            if key in out and out[key].sd == 0 and c.sd == 0 and out[key].mean != c.mean:
                raise ValueError(f"{key}: two exact values disagree ({out[key].mean} vs {c.mean})")
            if key not in out or out[key].sd == 0 or c.sd == 0:
                out[key] = c if key not in out or c.sd == 0 else out[key]
                continue
            a, b = out[key], c
            wa, wb = 1 / a.sd ** 2, 1 / b.sd ** 2
            out[key] = Coord((wa * a.mean + wb * b.mean) / (wa + wb), math.sqrt(1 / (wa + wb)), "measured",
                             f"{a.estimator}+{b.estimator}", f"{a.epoch}+{b.epoch}")
        return DevicePosterior(self.base, out, other.schedule if other.schedule != "alap" else self.schedule,
                               {**self.provenance, **other.provenance})

    # ------------------------------------------------------------ identity
    def to_json(self) -> dict:
        return dict(schema="oneq.device_posterior/v1", base=self.base.to_json(), base_sha256=self.base.sha256(),
                    schedule=self.schedule, provenance=self.provenance,
                    coords={_key_str(k): c.to_json() for k, c in sorted(self.coords.items(), key=lambda kv: str(kv[0]))})

    @classmethod
    def from_json(cls, d: dict) -> "DevicePosterior":
        if d.get("schema") != "oneq.device_posterior/v1":
            raise ValueError(f"not a device posterior: {d.get('schema')!r}")
        base = S.DeviceSnapshot.from_json(d["base"])
        if base.sha256() != d["base_sha256"]:
            raise ValueError("the base snapshot does not hash to its recorded sha256")
        coords = {_key_from(k): Coord(*v) for k, v in d["coords"].items()}
        return cls(base, coords, d["schedule"], d.get("provenance", {}))

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def from_diag_fit(rec: dict, fit: dict, estimator: str, epoch: str) -> DevicePosterior:
    """R146's fit (hw_diag_run.fit_in_circuit) as a posterior over the base snapshot it was frozen against"""
    base = S.DeviceSnapshot.from_json(rec["snapshot"])
    coords: dict = {}
    for q, r in fit["readout"].items():
        q = int(q)
        coords[("p01", q)] = Coord(r["p01"], math.sqrt(max(r["p01"] * (1 - r["p01"]), 1e-6) / r["shots0"]),
                                   "measured", estimator, epoch)
        coords[("p10", q)] = Coord(r["p10"], math.sqrt(max(r["p10"] * (1 - r["p10"]), 1e-6) / r["shots1"]),
                                   "measured", estimator, epoch)
    for q, x in fit["sx"].items():
        coords[("sx", int(q))] = Coord(x["p_sx"], x["p_sx_sd"], "measured", estimator, epoch)
    for q, x in fit["xs"].items():
        t2 = x["t2_eff_s"]
        if math.isfinite(t2) and t2 > 0:
            if "t2_sd" not in x:
                # an uncertainty nobody measured is not a coordinate's sd: refuse rather than invent one
                raise ValueError(f"qubit {q}: the fit gives T2 without its sd -- re-run hw_diag_run.fit_in_circuit")
            if math.isfinite(x["t2_sd"]):
                coords[("t2", int(q))] = Coord(t2, x["t2_sd"], "measured", estimator, epoch)
    for k, lad in fit["ladders"].items():
        a, b = (int(x) for x in k.split("_"))
        sds = lad.get("learnable_sd", {})
        for lk, g in lad["learnable"].items():
            coords[("cz", a, b, lk)] = Coord(g, sds.get(lk, 0.0), "measured", estimator, epoch)
    sched = fit["timing"]["schedule"]
    return DevicePosterior(base, coords, "alap" if sched == "undetermined" else sched,
                           {"estimator": estimator, "epoch": epoch})
