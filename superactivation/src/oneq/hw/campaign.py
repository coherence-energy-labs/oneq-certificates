"""One job, many preregistered experiments: compose their frozen records into ONE submission, split the results back,
and keep everything the job returns.

Why one job (R152). Experiments that share a device and a calibration window share its state: a calibration block run
in the same job measures the device the protocols ran on, not the device of another hour, and every job pays its
fixed overhead once. One SamplerV2 job has ONE options set, so parts whose Sampler options differ cannot share one,
and compose() refuses them.

compose(parts, meta) -> (record, blob)
    parts: [Part(tag, record, blob, shots, ...)] -- each experiment's OWN frozen record, untouched and re-verified
    (oneq.hw.freeze.load checks its QPY hash and order). The composite registers every part's circuits as
    '<tag>__<name>', the parts' orders concatenated in the order given, and pins each part: the sha256 of its record's
    canonical JSON, its QPY hash, every circuit's own sha256, its shots and its preregistration. Its own QPY is a
    fresh freeze of the renamed circuits; the run never re-transpiles.
split(record, counts) -> {tag: {name: counts}}: the composite's results under each part's own names, by the
    composite's registered names only (a name nobody registered is refused, and so is a missing one).
flight_record(result, record) -> (index, arrays): EVERYTHING a SamplerV2 result carries, for analyses no one has
    thought of yet -- every shot of every classical register as the raw packed bytes (BitArray.array: shot order kept,
    or, for a kerneled job, every shot's complex IQ point per clbit -- oneq.hw.iq reads them back,
    so correlations across registers and drift within a circuit stay readable), each PUB's metadata, the execution
    spans (start, stop, which PUBs and how many of their shots each covered: the job's own clock, per circuit), and
    the job's metadata. arrays go to an .npz; index is JSON.
calibration_drift(before, after, qubits, edges): what the vendor calibration says moved between freeze and fetch."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import freeze as F

SEP = "__"


class CampaignRefused(ValueError):
    """a part, a name or a result does not fit the composite it is offered to"""


@dataclass
class Part:
    tag: str                       # namespace of this part's circuits inside the job: '<tag>__<name>'
    record: dict                   # the part's own frozen record (oneq.hw.freeze)
    blob: bytes                    # its QPY
    shots: dict                    # name -> shots, exactly the part's registered circuits
    randomizations: int | None     # the Sampler gate-twirl randomizations the part was designed and predicted for
    prereg_path: str = ""
    prereg_sha256: str = ""
    records_dir: str = ""          # where the part's own runner looks for its job records (its ONE RUN guard)
    run_tag: str = ""              # the job-record prefix the part's own runner uses
    extra: dict = field(default_factory=dict)


def record_sha256(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def compose(parts: list, meta: dict, interleave: bool = False) -> tuple[dict, bytes]:
    """one frozen job of several parts. interleave=True orders the circuits UNIT BY UNIT: for each circuit name of the
    first part, every part's circuit of that name (in part order), then whatever the other parts hold beyond it. A
    paired A/B needs this: concatenated, one arm runs before the other and an execution-span drift (R168) reads as
    the arms' difference (found by the R183 adversary pass, 2026-10-18). Default False keeps every earlier job's order."""
    if not parts:
        raise CampaignRefused("a campaign of no parts")
    tags = [p.tag for p in parts]
    if len(set(tags)) != len(tags):
        raise CampaignRefused(f"two parts share a tag: {tags}")
    for t in tags:
        if not t or SEP in t or not t.replace("_", "").isalnum():
            raise CampaignRefused(f"tag {t!r}: alphanumerics and single underscores only, no {SEP!r}")
    twirl = {p.randomizations for p in parts}
    if len(twirl) != 1:
        raise CampaignRefused(f"one job has one options set, and these parts need different twirling: "
                              f"{ {p.tag: p.randomizations for p in parts} }")
    named = []
    shots: dict = {}
    pinned = []
    for p in parts:
        circuits = F.load(p.record, p.blob, check_sources=False)       # refuses a tampered or reordered part
        order = list(p.record["circuit_order"])
        if set(p.shots) != set(order):
            raise CampaignRefused(f"{p.tag}: its shots name {len(p.shots)} circuits, its record registers "
                                  f"{len(order)} -- not the same set")
        for n in order:
            if not isinstance(p.shots[n], int) or p.shots[n] < 1:
                raise CampaignRefused(f"{p.tag}/{n}: shots must be a positive integer, not {p.shots[n]!r}")
            named.append((f"{p.tag}{SEP}{n}", circuits[n]))
            shots[f"{p.tag}{SEP}{n}"] = p.shots[n]
        pinned.append(dict(tag=p.tag, record_sha256=record_sha256(p.record), qpy_sha256=p.record["qpy_sha256"],
                           circuit_order=order, circuit_sha256=p.record.get("circuit_sha256", {}),
                           shots=dict(p.shots), prereg=p.prereg_path, prereg_sha256=p.prereg_sha256,
                           records_dir=p.records_dir, run_tag=p.run_tag, extra=p.extra))
    if interleave:
        by_name = {full: qc for full, qc in named}
        inner = [n.split(SEP, 1)[1] for n, _ in named if n.startswith(f"{parts[0].tag}{SEP}")]
        ordered: list = []
        for n in inner:
            for p in parts:
                full = f"{p.tag}{SEP}{n}"
                if full in by_name:
                    ordered.append(full)
        ordered += [full for full, _ in named if full not in set(ordered)]
        named = [(full, by_name[full]) for full in ordered]
    rec, blob = F.freeze(named, dict(meta, parts=pinned, shots=shots, randomizations=parts[0].randomizations,
                                     interleaved=bool(interleave)))
    return rec, blob


def verify_parts(record: dict, parts: list) -> None:
    """the parts handed back at analysis are the ones the composite pinned, byte for byte"""
    pinned = {p["tag"]: p for p in record["parts"]}
    if set(pinned) != {p.tag for p in parts}:
        raise CampaignRefused(f"parts {sorted(p.tag for p in parts)} are not the pinned {sorted(pinned)}")
    for p in parts:
        if record_sha256(p.record) != pinned[p.tag]["record_sha256"]:
            raise CampaignRefused(f"{p.tag}: its record is not the one the composite pinned")
        if p.record["qpy_sha256"] != pinned[p.tag]["qpy_sha256"]:
            raise CampaignRefused(f"{p.tag}: its QPY is not the one the composite pinned")


def split(record: dict, counts: dict) -> dict:
    expected = set(record["circuit_order"])
    if set(counts) != expected:
        extra, missing = sorted(set(counts) - expected), sorted(expected - set(counts))
        raise CampaignRefused(f"results for unregistered circuits {extra[:3]} / missing {missing[:3]}")
    out: dict = {p["tag"]: {} for p in record["parts"]}
    for name, c in counts.items():
        tag, _, inner = name.partition(SEP)
        out[tag][inner] = c
    for p in record["parts"]:
        if set(out[p["tag"]]) != set(p["circuit_order"]):
            raise CampaignRefused(f"{p['tag']}: the split does not return exactly its registered circuits")
    return out


# ---------------------------------------------------------------- the flight record
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (datetime.datetime, datetime.date)):
        return x.isoformat()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def _slice_json(x):
    if isinstance(x, slice):
        return [x.start, x.stop, x.step]
    if isinstance(x, (tuple, list)):
        return [_slice_json(v) for v in x]
    return _jsonable(x)


def _spans(execution) -> list:
    """every span's clock, the PUBs it ran with how many of their shots, and -- where the runtime gives it -- the
    exact SLICE of each PUB's data it covered. For a twirled job (TwirledSliceSpanV2) a PUB's data is shaped
    (randomizations, shots per randomization) and flattened randomization-major, so the slice ties every shot to its
    randomization and its time window (found 2026-10-17 on R152: without it, shot order could not be read as either)"""
    spans = (execution or {}).get("execution_spans")
    if spans is None:
        return []
    out = []
    for s in spans:
        row = dict(start=s.start.isoformat(), stop=s.stop.isoformat(),
                   seconds=(s.stop - s.start).total_seconds(), kind=type(s).__name__,
                   pubs={int(i): int(np.asarray(s.mask(i)).sum()) for i in s.pub_idxs})
        slices = getattr(s, "_data_slices", None)
        if slices:
            row["slices"] = {int(i): _slice_json(v) for i, v in slices.items()}
        # TwirledSliceSpanV2 keeps the shots REQUESTED apart from `_data_slices` (which holds only the 4-tuple): when
        # fewer than R x n, the last frame is truncated, and without this the record could not say so (R183, 250 of
        # 8 x 32; oneq.hw.randomization.shape reads it)
        pub_shots = getattr(s, "_pub_shots", None)
        if pub_shots:
            row["pub_shots"] = {int(i): int(v) for i, v in pub_shots.items()}
        out.append(row)
    return out


def flight_record(result, record: dict) -> tuple[dict, dict]:
    """(index, arrays) for a SamplerV2 PrimitiveResult of the composite, PUB i = the i-th registered circuit"""
    order = list(record["circuit_order"])
    if len(result) != len(order):
        raise CampaignRefused(f"{len(result)} results for {len(order)} registered circuits")
    arrays: dict = {}
    pubs = []
    for i, (name, pub) in enumerate(zip(order, result)):
        data = pub.data
        regs = list(data.keys()) if hasattr(data, "keys") else [k for k in vars(data)]
        entry = dict(index=i, name=name, registers={}, metadata=_jsonable(getattr(pub, "metadata", {})))
        for r in regs:
            ba = getattr(data, r)
            key = f"{i:04d}{SEP}{r}"
            if hasattr(ba, "num_bits"):                  # classified: a BitArray, kept as its packed bytes
                arrays[key] = np.asarray(ba.array)
                entry["registers"][r] = dict(key=key, num_bits=int(ba.num_bits), num_shots=int(ba.num_shots),
                                             shape=list(np.asarray(ba.array).shape))
                continue
            # kerneled (meas_type="kerneled"): per-shot complex IQ, (..., shots, clbits). Flattened over every leading
            # axis in C order -- randomization-major, the BitArray convention -- and stored as float32 (re, im):
            # amplitudes of ~1e7-1e8 keep 7 significant digits, far below any cluster's width
            iq = np.asarray(ba)
            if not np.iscomplexobj(iq) or iq.ndim < 2:
                raise CampaignRefused(f"{name}: register {r} is neither a BitArray nor complex per-shot IQ "
                                      f"({iq.dtype}, shape {iq.shape})")
            flat = iq.reshape(-1, iq.shape[-1])
            arrays[key] = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32)
            entry["registers"][r] = dict(key=key, kind="iq", num_bits=int(iq.shape[-1]), num_shots=int(flat.shape[0]),
                                         shape=list(iq.shape), stored="float32 (shots, clbits, re/im)")
        pubs.append(entry)
    md = dict(getattr(result, "metadata", {}) or {})
    execution = md.get("execution")
    spans = _spans(execution)
    md_rest = {k: v for k, v in md.items() if k != "execution"}
    shots_by_pub = {i: sum(e["registers"][r]["num_shots"] for r in list(e["registers"])[:1])
                    for i, e in enumerate(pubs)}
    covered: dict = {}
    for s in spans:
        for i, n in s["pubs"].items():
            covered[i] = covered.get(i, 0) + n
    index = dict(schema="oneq.hw.campaign.flight_record/1", composite_qpy_sha256=record["qpy_sha256"],
                 n_pubs=len(pubs), pubs=pubs, execution_spans=spans,
                 spans_cover_every_shot=bool(spans) and all(covered.get(i, 0) == n for i, n in shots_by_pub.items()),
                 result_metadata=_jsonable(md_rest))
    index["frame_homogeneity"] = frame_homogeneity(index, arrays)
    return index, arrays


def frame_homogeneity(index: dict, arrays: dict) -> dict:
    """do each twirled circuit's frames agree? oneq.hw.randomization.homogeneity_map over every BIT register (exact
    chi2 tests, Bonferroni over the whole job): R176/R177 found that a frame spread is where shot-only error bars fail
    (coherent error a twirl samples) and where a frame POSITION effect hides. On R152 it flags q149, q45 and q43 --
    R169's coherent qubits -- and nothing else in 5,851 tests. Kerneled (IQ) registers are listed for a later map on
    their classified bits. A failure here is RECORDED, never raised: the flight record's job is to keep the shots."""
    from oneq.hw import randomization as RZ
    bit = [p["name"] for p in index["pubs"] if all(r.get("kind") != "iq" for r in p["registers"].values())]
    kerneled = [p["name"] for p in index["pubs"] if p["name"] not in set(bit)]
    try:
        out = RZ.homogeneity_map(RZ.from_flight(index, arrays, names=bit)) if bit else dict(tests=0, flagged=0)
    except Exception as e:                     # capture must never lose the shots to an analysis
        out = dict(refused=f"{type(e).__name__}: {e}")
    out["kerneled_not_mapped"] = kerneled
    return out


def bitstrings(index: dict, arrays: dict, name: str, register: str | None = None) -> list:
    """one circuit's shots, in shot order, as qiskit bitstrings (clbit 0 rightmost), from a flight record"""
    from qiskit.primitives.containers import BitArray
    pub = next(p for p in index["pubs"] if p["name"] == name)
    reg = register or next(iter(pub["registers"]))
    r = pub["registers"][reg]
    return BitArray(arrays[r["key"]], r["num_bits"]).get_bitstrings()


def calibration_drift(before, after, qubits: list, edges: list) -> dict:
    """the vendor calibration's own change between two snapshots (oneq.hw.snapshot.DeviceSnapshot), on the qubits and
    edges a campaign used: readout, T1, T2, sx and CZ error, as (before, after, ratio)"""
    def row(a, b):
        return dict(before=a, after=b, ratio=(b / a) if a else None)
    out = dict(calibrated_before=before.calibrated_utc, calibrated_after=after.calibrated_utc, qubits={}, edges={})
    for q in qubits:
        out["qubits"][str(q)] = {k: row(getattr(before, k)[q], getattr(after, k)[q])
                                 for k in ("p01", "p10", "t1", "t2", "sx_error")}
    for a, b in edges:
        out["edges"][f"{a}_{b}"] = row(before.edge(a, b)[0], after.edge(a, b)[0])
    return out


__all__ = ["Part", "compose", "verify_parts", "split", "flight_record", "bitstrings", "calibration_drift",
           "record_sha256", "CampaignRefused", "SEP"]
