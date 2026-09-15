r"""A decoding model built from the DEVICE'S OWN published calibration.

WHY THIS EXISTS, stated against the preregistration rather than around it.

`docs/PREREGISTRATION_PHASE6.md` section 2 pins "Model N: the device's own
DEM, raw IEEE doubles, verbatim". The lane that actually ran
(`experiments/ibm_live/rep_code_cert.py::decode_graph`) gives EVERY edge
weight 1.0 and labels itself "uniform, declared". So the run certified
minimum-weight against a FLAT model, not against N, and the headline claim
as preregistered -- "the correction is minimum-weight w.r.t. the pinned,
published model N" -- was not the claim tested. That deviation is recorded
in `docs/PHASE6_DEVIATIONS.md`, not repaired by redefining N.

AND THE CLAUSE CANNOT BE SATISFIED LITERALLY ON THIS HARDWARE. A DEM is an
artifact the experimenter publishes alongside a dataset; Google ships one
with the Willow archive, which is why the phrase was written that way. IBM
publishes CALIBRATION -- per-qubit readout error, per-pair two-qubit gate
error, T1/T2, gate durations -- and no DEM for an arbitrary circuit. "The
device's own DEM" therefore names something that does not exist for this
lane. The nearest well-defined object, and what this module builds, is:

    a DEM DERIVED from the device's published calibration, for the SPECIFIC
    physical qubits the job was transpiled onto, with the calibration
    snapshot pinned alongside it so the derivation is re-runnable.

That is a different object from N as frozen, and every artifact it feeds
must say so. Anything computed here is POST-HOC and EXPLORATORY with
respect to the frozen preregistration: the model was chosen after the data
existed, which is exactly the freedom preregistration exists to remove. It
is reported as a separate, labelled analysis and it never upgrades a
preregistered claim.

WHY IT IS STILL WORTH BUILDING. Under uniform weights every edge costs the
same, so minimum-weight corrections are massively DEGENERATE: ties are the
norm, `p_ambiguity` is dominated by an artifact of the model rather than by
the device, and the logical gap Delta_L carries almost no information. The
calibration says the flat model is badly wrong -- on `ibm_marrakesh` the
readout error spans 0.00049 to 0.50659 across qubits and the CZ error spans
0.00101 to 1.0. Weights that reflect that turn a degenerate objective into
a discriminating one, which is what makes F6 measurable at all.

EVERY WEIGHT TRACES TO A PUBLISHED NUMBER. The mechanism composition below
is an explicit, auditable approximation -- named as such, with each term
attributed -- rather than an opaque black box. An approximation whose parts
are visible can be argued with; that is the point.
"""

from __future__ import annotations

import math
from fractions import Fraction

BOUNDARY = -1


def _parse_edge_key(key: str) -> tuple[int, int]:
    """`"u-v"` -> `(u, v)`, where EITHER endpoint may be the sink `-1`.

    The separator and the sign are the same character, so this is written
    once, here, and made total. The version it replaces reached
    `rest.split("-", 1)[1]` unguarded: a key of `"-"` raised IndexError
    from inside the rebuild, several frames from anything that names the
    artifact. A malformed key is a malformed artifact and should say so.
    """
    neg = key.startswith("-")
    head, sep, tail = (key[1:] if neg else key).partition("-")
    if not sep or not head or not tail:
        raise ValueError(
            f"edge key {key!r} is not 'u-v': the pinned provenance names an "
            f"edge that has no two endpoints")
    return int("-" + head if neg else head), int(tail)


def edges_from_pinned(prov: dict):
    """Rebuild the decoding graph from PINNED weights -- no transcendentals.

    The verifier's entry point. `rep_code_edges` computes weights with
    `math.log` / `math.exp`, which are libm calls and therefore not
    bit-identical across platforms; recomputing them on another machine can
    land an ulp away and disagree with a correct certificate. This reads the
    exact rationals the producer emitted instead, so re-derivation is
    byte-stable anywhere Python runs.

    Takes the provenance dict returned by `rep_code_edges` and returns the
    same edge list with `Fraction` weights.
    """
    out = []
    for key, rec in (prov.get("edges") or {}).items():
        u, v = _parse_edge_key(key)
        w = rec.get("weight_exact")
        if w is None:
            return None                     # pre-pin artifact: refuse
        out.append((u, v, Fraction(w)))
    return out


def snapshot_calibration(backend, physical_qubits, *, pairs=None) -> dict:
    """Pin the device's published calibration for the qubits actually used.

    Returns a JSON-safe dict. Nothing is averaged away: per-qubit and
    per-pair numbers are kept as published, because the whole reason this
    module exists is that they are NOT homogeneous.
    """
    t = backend.target
    q = [int(x) for x in physical_qubits]
    out: dict = {
        "backend": backend.name,
        "physical_qubits": q,
        "dt_seconds": float(getattr(backend, "dt", 0.0) or 0.0),
        "readout_error": {}, "t1_seconds": {}, "t2_seconds": {},
        "sx_error": {}, "two_qubit_error": {},
        "measure_duration_seconds": {},
    }
    for i in q:
        try:
            props = backend.qubit_properties(i)
            out["t1_seconds"][str(i)] = float(props.t1)
            out["t2_seconds"][str(i)] = float(props.t2)
        except Exception:
            pass
        if "measure" in t and (i,) in t["measure"]:
            e = t["measure"][(i,)]
            if e is not None:
                if e.error is not None:
                    out["readout_error"][str(i)] = float(e.error)
                if getattr(e, "duration", None) is not None:
                    out["measure_duration_seconds"][str(i)] = float(e.duration)
        if "sx" in t and (i,) in t["sx"]:
            e = t["sx"][(i,)]
            if e is not None and e.error is not None:
                out["sx_error"][str(i)] = float(e.error)
    names = [n for n in ("cz", "ecr", "cx") if n in t]
    for name in names:
        for key, e in t[name].items():
            if key is None or e is None or e.error is None:
                continue
            if all(k in q for k in key):
                out["two_qubit_error"][f"{name}:{key[0]}-{key[1]}"] = \
                    float(e.error)
    if pairs:
        out["requested_pairs"] = [[int(a), int(b)] for a, b in pairs]
    return out


def _two_qubit_error(cal: dict, a: int, b: int) -> float | None:
    """Published two-qubit error for a pair, either orientation."""
    for name in ("cz", "ecr", "cx"):
        for k in (f"{name}:{a}-{b}", f"{name}:{b}-{a}"):
            if k in cal["two_qubit_error"]:
                return cal["two_qubit_error"][k]
    return None


def _idle_flip(cal: dict, qubit: int, seconds: float) -> float:
    """Bit-flip probability from amplitude damping over `seconds`.

    (1 - exp(-t/T1)) / 2 -- the standard half-of-relaxation approximation
    for a symmetrized flip channel. Returns 0 when T1 is unpublished rather
    than inventing one.
    """
    t1 = cal["t1_seconds"].get(str(qubit))
    if not t1 or seconds <= 0:
        return 0.0
    return (1.0 - math.exp(-seconds / t1)) / 2.0


def _combine(*ps: float) -> float:
    """Odds of an ODD number of independent flips -- the parity that a
    detector actually sees. Not a sum: two flips cancel."""
    odd = 0.0
    for p in ps:
        p = min(max(p, 0.0), 0.5)
        odd = odd * (1 - p) + (1 - odd) * p
    return odd


def _weight(p: float, *, floor: float = 1e-9) -> float:
    """The matching weight log((1-p)/p). A mechanism that cannot happen
    would be an infinite weight, so p is floored and the floor is declared
    rather than hidden."""
    p = min(max(p, floor), 0.5 - 1e-12)
    return math.log((1.0 - p) / p)


def rep_code_edges(cal: dict, d: int, r: int, data_q, anc_q,
                   *, round_seconds: float | None = None):
    """Weighted space-time matching graph for the repetition-code memory.

    Node id = t * (d-1) + i, matching `rep_code_cert.decode_graph` exactly,
    so the two models are interchangeable and directly comparable.

    THE MECHANISMS, each attributed:
      time-like  (i,t)->(i,t+1)  ancilla i is misreported this round --
                                 readout error of anc_q[i], plus its idle
                                 flip over one round
      space-like (i,t)->(i+1,t)  data qubit i+1 flips; it is shared by
                                 ancillas i and i+1, so a flip lights both
      boundary   at columns 0    data qubit 0 (resp. d-1) flips; it touches
                 and d-2         only one ancilla, so it lights one detector
    A data qubit's per-round flip probability composes its two-qubit gate
    errors for the round with its idle flip.

    Returns (edges, provenance) where provenance maps each edge to the
    published numbers it was built from.
    """
    data_q = [int(x) for x in data_q]
    anc_q = [int(x) for x in anc_q]
    if len(data_q) != d or len(anc_q) != d - 1:
        raise ValueError(f"layout mismatch: {len(data_q)} data, "
                         f"{len(anc_q)} ancilla for d={d}")

    if round_seconds is None:
        durs = [v for v in cal["measure_duration_seconds"].values() if v]
        round_seconds = (max(durs) if durs else 0.0) * 2.0

    def data_flip(j: int) -> tuple[float, dict]:
        parts, why = [], {}
        for a in anc_q:
            e = _two_qubit_error(cal, data_q[j], a)
            if e is not None:
                parts.append(e)
                why[f"2q:{data_q[j]}-{a}"] = e
        idle = _idle_flip(cal, data_q[j], round_seconds)
        if idle:
            parts.append(idle)
            why[f"idle:{data_q[j]}"] = idle
        return _combine(*parts), why

    def anc_flip(i: int) -> tuple[float, dict]:
        ro = cal["readout_error"].get(str(anc_q[i]), 0.0)
        idle = _idle_flip(cal, anc_q[i], round_seconds)
        why = {f"readout:{anc_q[i]}": ro}
        if idle:
            why[f"idle:{anc_q[i]}"] = idle
        return _combine(ro, idle), why

    def nid(i, t):
        return t * (d - 1) + i

    edges, prov = [], {}

    def add(u, v, p, why):
        # THE WEIGHT IS PINNED, NOT RE-DERIVED BY THE VERIFIER.
        #
        # `_weight` is `log((1-p)/p)` and `_idle_flip` is `exp(-t/T1)`: both
        # are libm calls, and libm is not bit-identical across platforms or
        # versions. A stranger recomputing this model on another machine can
        # legitimately land an ulp away, decode to a different correction,
        # and disagree with a certificate that is perfectly correct -- which
        # is indistinguishable, from their side, from a forged one.
        #
        # So the derivation is a PRODUCER step and its output is DATA. Every
        # weight is emitted here as an exact rational (a float IS a binary
        # rational, so Fraction(w) is lossless) and the artifact carries it.
        # A verifier reads the pinned value verbatim and never calls a
        # transcendental. That is the same posture the preregistration takes
        # toward model N -- "raw IEEE doubles, verbatim" -- and the same
        # producer/checker split the rest of the lane uses.
        w = _weight(p)
        edges.append((u, v, w))
        prov[f"{u}-{v}"] = {"p": p, "weight": w,
                            "weight_exact": str(Fraction(w)),
                            "from": why}

    for t in range(r + 1):
        for i in range(d - 1):
            if t + 1 <= r:
                p, why = anc_flip(i)
                add(nid(i, t), nid(i, t + 1), p, why)
            if i + 1 < d - 1:
                p, why = data_flip(i + 1)
                add(nid(i, t), nid(i + 1, t), p, why)
        p0, why0 = data_flip(0)
        add(nid(0, t), BOUNDARY, p0, why0)
        pl, whyl = data_flip(d - 1)
        add(nid(d - 2, t), BOUNDARY, pl, whyl)

    return edges, {"round_seconds": round_seconds, "edges": prov}
