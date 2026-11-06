"""R168: the TIME STRUCTURE of five recorded ibm_kingston jobs -- shot order, shot-to-shot memory, correlated bursts
across uncoupled qubits, telegraph switching, and drift across jobs (the owner's list, items 9, 10, 11 and 16).
OFFLINE: it reads flight records and snapshots; no IBM access.

Jobs (every shot kept in stored order by oneq.hw.campaign.flight_record; every PUB twirled with 8 randomizations):
R152 db9c50oc77vs73817rog (bits, 594 PUBs), R155 db9dsqpgv2cc73e5c8fg (bits, 62), R161+R158 db9q3bs24mrc7384ecn0
(raw IQ, 163), R165 active db9s4p424mrc7384gskg and active-accum db9s4r8c77vs7381ug60 (raw IQ, 7 each).

THE CLOCK -- what an analysis may call "before":
  PROVEN by the metadata (span_layout): every PUB's shots ran inside ONE execution span (its slice covers all 8
    randomizations and every shot), spans are time-ordered and disjoint, and the data is randomization-major, so
    randomization r of a PUB is stored rows [r n, (r + 1) n).
  NOT in the metadata: any order inside a span (the runtime's window guarantees only that the data ran inside it).
  OBSERVED in IBM's documentation (guides/repetition-rate-execution, qiskit-ibm-runtime ~0.47): a batch runs
    "column-wise over a matrix of circuits and shots" -- for each shot index, every circuit once -- with an implicit
    reset at each circuit's start, and no order guaranteed across PUBs. Read with one bound circuit per randomization
    and one batch per span (H_I, interleaved sweeps), stored shots k and k+1 of one block are one SWEEP apart (every
    circuit of the span runs in between: ~9 ms to ~0.2 s), and a shot's physical predecessor is ANOTHER circuit's
    shot. The alternative, H_C, is circuit-major blocks: stored order is time inside a block.
  TESTED both ways (order_tests): a slow change is a k-profile SHARED by the blocks of a span under H_I and a
    between-block difference under H_C; a reset failure is an immediate-predecessor effect, which under pub-major
    sweeps makes a PUB's randomization 0 (preceded by ANOTHER PUB) differ from its others (reset_regression).

ANALYSES -- pure functions, each against its own permutation null; tests/test_r168_time_structure.py plants every
effect in synthetic data and recovers it at its planted size:
  9  memory_test         P(1 now | 1 before) - P(1 now | 0 before) along stored order inside each (PUB,
                         randomization) block, against 1000 within-block shuffles; per qubit, per prepared state,
                         pooled. cross_memory_test asks it qubit x qubit. reset_regression: the true predecessor.
  10 burst_test          per-shot error multiplicity across qubits against (a) the Poisson-binomial of the block's
                         own per-qubit rates and (b) independent column shuffles -- or windowed shuffles, which keep
                         slow drift common to the qubits and remove only shot-level coincidence. cluster_test: do
                         high-multiplicity shots bunch in time, under either clock.
  11 telegraph_test      window dispersion and a two-state (epidemic) change-point likelihood ratio, on the sweep
                         series of a span (H_I) and inside blocks (H_C); centre_drift_test: the IQ cluster centre.
  16 structure_function  the vendor calibration's change against elapsed time over every recorded window;
     binomial_dispersion the vendor readout against its own 4096-shot binomial; in-circuit readout and Ramsey
                         ratios across jobs; stability_table: a per-qubit ranking for layout.

  python r168_time_structure.py [--perm N]      -> evidence/superactivation/r168_time_structure/*.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "oneq.r168.time_structure/1"
SEED = 168
N_PERM = 1000
RANDOMIZATIONS = 8
VENDOR_RO_SHOTS = 4096          # every vendor p01/p10 on record is an integer count out of 4096 (checked at load)
EV = ROOT / "evidence" / "superactivation"
OUTDIR = EV / "r168_time_structure"
JOBS = (
    dict(tag="r152", job="db9c50oc77vs73817rog", dir="campaign_r152/20261017T003152Z", record="composite.json",
         spans="spans_db9c50oc77vs73817rog.json"),
    dict(tag="r155", job="db9dsqpgv2cc73e5c8fg", dir="r155_diag", record="frozen_20261017T023024Z.json"),
    dict(tag="r161", job="db9q3bs24mrc7384ecn0", dir="r161_spectator_cz/job_20261017T124413Z", record="job.json",
         iq_calibration=("r161__ro_all0", "r161__ro_all1")),
    dict(tag="r165_active", job="db9s4p424mrc7384gskg", dir="r165_dressing_bench",
         record="frozen_20261017T174831Z.json", iq_calibration=("ro_all0", "ro_all1")),
    dict(tag="r165_accum", job="db9s4r8c77vs7381ug60", dir="r165_dressing_bench",
         record="frozen_20261017T174831Z.json", iq_calibration=("ro_all0", "ro_all1")),
)
VENDOR_DOC = dict(
    source="https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution (qiskit-ibm-runtime~=0.47.0)",
    read="2026-10-17",
    quotes=["Circuits you submit in a primitive job are batched together for execution on the QPU.",
            "These circuits are executed by iterating over the circuits for each shot requested;",
            "the execution is column-wise over a matrix of circuits and shots",
            "there is no guarantee on the order the circuits from PUBs are executed.",
            "There is also no guarantee that circuits from the same job would run as a single batch on the QPU.",
            "an implicit reset is typically inserted at the beginning of the circuit",
            "imperfections in the reset process can introduce state-preparation errors."],
    silent_on="twirling randomizations (whether each is its own circuit in the sweep) and the sweep's circuit order")
R165_EDGE = (148, 149)


class R168Refused(ValueError):
    """data an analysis cannot read faithfully"""


# ================================================================== layout (pure)
def unpack_bits(packed, num_bits: int) -> np.ndarray:
    """BitArray bytes (shots, n_bytes) -- big-endian, clbit 0 the LOWEST bit of the LAST byte -> (shots, num_bits) bool
    with clbit c in column c"""
    a = np.asarray(packed)
    if a.dtype != np.uint8 or a.ndim != 2:
        raise R168Refused(f"packed bits must be uint8 (shots, bytes), not {a.dtype} {a.shape}")
    if not 8 * (a.shape[1] - 1) < num_bits <= 8 * a.shape[1]:
        raise R168Refused(f"{num_bits} bits are not what {a.shape[1]} bytes store")
    return np.unpackbits(a, axis=1)[:, ::-1][:, :num_bits].astype(bool)


def split_blocks(x, randomizations: int = RANDOMIZATIONS) -> np.ndarray:
    """(shots, ...) stored randomization-major -> (randomizations, shots per randomization, ...)"""
    x = np.asarray(x)
    if x.ndim < 1 or x.shape[0] == 0 or x.shape[0] % randomizations:
        raise R168Refused(f"{x.shape[0] if x.ndim else 0} shots do not split into {randomizations} randomizations")
    return x.reshape(randomizations, x.shape[0] // randomizations, *x.shape[1:])


def _iso(s: str) -> datetime.datetime:
    t = datetime.datetime.fromisoformat(str(s))
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def _slice_ok(s, length: int) -> bool:
    return list(s[:2]) == [0, length] and (len(s) < 3 or s[2] in (None, 1))


def span_layout(spans: list, shots_by_pub: dict, randomizations: int = RANDOMIZATIONS) -> dict:
    """what the execution metadata PROVES about time: per PUB the one span holding all its shots, its position in the
    span's slice listing (an ORDER the runtime does not promise), shots per randomization; per span its window, size
    and per-shot time; spans in time order, overlaps and gaps. A PUB covered partly, twice or never is a problem."""
    pubs, rows, problems = {}, [], []
    for si, s in enumerate(spans):
        sl = s.get("slices")
        if not sl:
            raise R168Refused(f"span {si} carries no data slices: its shots cannot be tied to randomizations")
        per, shots = set(), 0
        for pos, (key, v) in enumerate(sl.items()):
            i = int(key)
            R, n = int(v[0][0]), int(v[0][1])
            whole = (R == randomizations and _slice_ok(v[2], R) and _slice_ok(v[3], n)
                     and R * n == shots_by_pub.get(i, -1))
            if not whole:
                problems.append(f"span {si}: PUB {i} is not covered whole (shape {v[0]}, slices {v[2]} {v[3]})")
            if i in pubs:
                problems.append(f"PUB {i} is in spans {pubs[i]['span']} and {si}")
            listed = (s.get("pubs") or {}).get(str(i), (s.get("pubs") or {}).get(i))
            if listed is not None and int(listed) != R * n:
                problems.append(f"span {si}: PUB {i} counts {listed} shots but its slice holds {R * n}")
            pubs[i] = dict(span=si, position=pos, per_rand=n)
            per.add(n)
            shots += R * n
        sec = float(s["seconds"])
        rows.append(dict(span=si, start=str(s["start"]), stop=str(s["stop"]), seconds=sec, n_pubs=len(sl),
                         n_circuits=len(sl) * randomizations, shots=shots, per_rand=sorted(per),
                         us_per_shot=1e6 * sec / shots, sweep_ms_if_interleaved=1e3 * sec / max(per)))
    missing = sorted(set(shots_by_pub) - set(pubs))
    if missing:
        problems.append(f"{len(missing)} PUBs are in no span: {missing[:5]}")
    t = [(_iso(r["start"]), _iso(r["stop"])) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: t[i][0])
    overlaps = [[a, b] for a, b in zip(order, order[1:]) if t[a][1] > t[b][0]]
    gaps = [(t[b][0] - t[a][1]).total_seconds() for a, b in zip(order, order[1:])]
    return dict(pubs=pubs, spans=rows, time_order=order, overlaps=overlaps, gaps_s=gaps, problems=problems,
                every_pub_whole_in_one_span=not problems)


def ideal(qc, qubits=()) -> dict | None:
    """the IDEAL outcome of every clbit of a terminal-readout Clifford circuit -- 0, 1, or -1 when random -- and the
    ideal P(1) before readout of every measured qubit and of `qubits`, from the stabilizer tableau of the circuit
    without its measurements (barriers and delays dropped: ideal idles). None for a dynamic or non-Clifford circuit, or
    one that acts on a qubit after measuring it."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, StabilizerState
    body = QuantumCircuit(qc.num_qubits)
    meas, measured, touched = {}, set(), set()
    for inst in qc.data:
        op = inst.operation
        qs = [qc.find_bit(b).index for b in inst.qubits]
        if op.name == "measure":
            c = qc.find_bit(inst.clbits[0]).index
            if c in meas or qs[0] in measured:          # a clbit written twice, or a qubit read twice (mid-circuit)
                return None
            meas[c] = qs[0]
            measured.add(qs[0])
            continue
        if op.name in ("barrier", "delay"):
            continue
        if inst.clbits or op.name in ("reset", "if_else", "while_loop", "for_loop", "switch_case", "box"):
            return None
        if measured.intersection(qs):
            return None
        body.append(op, qs)
        touched.update(qs)
    if sorted(meas) != list(range(qc.num_clbits)):
        return None
    try:
        st = StabilizerState(Clifford(body))
    except Exception:  # noqa: BLE001 -- qiskit raises QiskitError for any non-Clifford instruction
        return None
    p1 = {q: (float(st.probabilities([q])[1]) if q in touched else 0.0) for q in set(meas.values()) | set(qubits)}
    exp = [0 if p1[meas[c]] < 1e-9 else (1 if p1[meas[c]] > 1 - 1e-9 else -1) for c in range(qc.num_clbits)]
    return dict(expected=exp, final_p1=p1, touched=sorted(touched))


# ================================================================== permutation helpers (pure)
def _perm_index(n: int, m: int, rng, window: int | None = None) -> np.ndarray:
    """m independent permutations of range(n), one per row -- each only inside consecutive windows of `window`
    positions when given (the last partial window on its own): a null that keeps any change slower than the window"""
    if not window or window >= n:
        return rng.permuted(np.tile(np.arange(n), (m, 1)), axis=1)
    k = n // window
    head = rng.permuted(np.broadcast_to(np.arange(k * window).reshape(k, window), (m, k, window)).copy(), axis=2)
    out = [head.reshape(m, k * window)]
    if k * window < n:
        out.append(rng.permuted(np.tile(np.arange(k * window, n), (m, 1)), axis=1))
    return np.concatenate(out, axis=1)


MIN_DISTINCT = 10       # a null with fewer distinct values than this is too discrete for a z (its p still stands)
IQ_PERM = 400           # the IQ centre's null (float arrays over every block: memory-bound, so fewer replicates)


def _null_summary(obs, null, tail: str = "two") -> dict:
    """observed value against its permutation replicates: mean, sd, excess, z and the permutation p (+1 corrected).
    z is withheld (None) when the null takes fewer than MIN_DISTINCT distinct values: a column with one error per
    block has an almost constant null, and (obs - mean) / sd would read a 1e-4 difference as z ~ 1e3"""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if obs is None or not np.isfinite(obs) or len(null) < 10:
        return dict(value=None if obs is None or not np.isfinite(obs) else float(obs), null_mean=None, null_sd=None,
                    excess=None, z=None, p=None, n_null=int(len(null)))
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    if tail == "upper":
        p = (1 + int(np.sum(null >= obs))) / (1 + len(null))
    else:
        p = (1 + int(np.sum(np.abs(null - mu) >= abs(obs - mu)))) / (1 + len(null))
    distinct = int(len(np.unique(np.round(null, 12))))
    return dict(value=float(obs), null_mean=mu, null_sd=sd, excess=float(obs - mu),
                z=float((obs - mu) / sd) if sd > 0 and distinct >= MIN_DISTINCT else None, p=p,
                n_null=int(len(null)), null_distinct=distinct)


# ================================================================== 9: shot-to-shot memory (pure)
def lag_counts(E) -> np.ndarray:
    """(n, Q) bool in time order -> (4, Q): pairs whose earlier shot is 1; of those, the later is 1; pairs whose earlier
    shot is 0; of those, the later is 1"""
    E = np.asarray(E, dtype=bool)
    if E.ndim != 2 or E.shape[0] < 2:
        raise R168Refused(f"a lag needs (n >= 2, Q) shots, got {E.shape}")
    prev, now = E[:-1], E[1:]
    return np.stack([prev.sum(0), (prev & now).sum(0), (~prev).sum(0), (~prev & now).sum(0)]).astype(float)


def delta(counts) -> np.ndarray:
    """P(1 | 1 before) - P(1 | 0 before) from lag counts (..., 4, Q); nan where a condition never occurred"""
    c = np.asarray(counts, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 1, :] / c[..., 0, :] - c[..., 3, :] / c[..., 2, :]


def memory_test(blocks: list, n_perm: int = N_PERM, rng=None, chunk: int = 250, window: int | None = None) -> dict:
    """blocks: [(E (n, Q_b) bool in stored order, keys (Q_b,))] -- one (PUB, randomization) block's indicators (bit !=
    ideal) of deterministic clbits, each column named by a key (its physical qubit). Per key and pooled over keys:
    Delta = P(1 | 1 before) - P(1 | 0 before) over every adjacent pair INSIDE a block, against `n_perm` replicates that
    permute each block's shots (rows, jointly over its columns: within-shot correlations kept, order destroyed). The
    null's mean is not zero (a block of n shots with t ones has Delta = -1/(n-1) on average, and pooling blocks of
    different rates biases it up), so the effect is the EXCESS over the null mean. With `window`, shots are permuted
    only inside consecutive windows of that many: a slow (telegraph) change survives in the null, so what remains is
    memory at the lag itself."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    keys = sorted({k for _, ks in blocks for k in ks}, key=lambda k: (str(type(k)), k))
    col = {k: i for i, k in enumerate(keys)}
    obs = np.zeros((4, len(keys)))
    null = np.zeros((n_perm, 4, len(keys)))
    used = 0
    for E, ks in blocks:
        E = np.asarray(E, dtype=bool)
        if E.ndim != 2 or E.shape[0] < 2 or E.shape[1] != len(ks):
            continue
        g = np.array([col[k] for k in ks])
        if len(set(g.tolist())) != len(g):
            raise R168Refused("a block names one key twice")
        obs[:, g] += lag_counts(E)
        n = E.shape[0]
        for s in range(0, n_perm, chunk):
            m = min(chunk, n_perm - s)
            P = E[_perm_index(n, m, rng, window)]
            prev, now = P[:, :-1], P[:, 1:]
            null[s:s + m, 0, g] += prev.sum(1)
            null[s:s + m, 1, g] += (prev & now).sum(1)
            null[s:s + m, 2, g] += (~prev).sum(1)
            null[s:s + m, 3, g] += (~prev & now).sum(1)
        used += 1
    if not used:
        raise R168Refused("no block of two or more shots")
    d_obs, d_null = delta(obs), delta(null)
    per = {}
    for i, k in enumerate(keys):
        r = _null_summary(d_obs[i], d_null[:, i])
        r.update(pairs_after_1=int(obs[0, i]), pairs_after_0=int(obs[2, i]),
                 p1_after_1=float(obs[1, i] / obs[0, i]) if obs[0, i] else None,
                 p1_after_0=float(obs[3, i] / obs[2, i]) if obs[2, i] else None)
        per[str(k)] = r
    pooled = _null_summary(delta(obs.sum(1, keepdims=True))[0], delta(null.sum(2, keepdims=True))[:, 0])
    pooled.update(pairs_after_1=int(obs[0].sum()), pairs_after_0=int(obs[2].sum()))
    # the keys' standardized excesses summed (Stouffer), calibrated by the SAME replicates (each standardized alike):
    # every key weighs alike, where the pooled Delta is ruled by the keys with the most errors
    ok = np.array([per[str(k)]["z"] is not None for k in keys])
    if ok.any():
        mu, sd = np.nanmean(d_null[:, ok], 0), np.nanstd(d_null[:, ok], 0, ddof=1)
        s_obs = float(np.sum((d_obs[ok] - mu) / sd))
        s_null = np.nansum((d_null[:, ok] - mu) / sd, 1)
        stouffer = _null_summary(s_obs, s_null)
        stouffer.update(keys=int(ok.sum()))
    else:
        stouffer = dict(value=None, keys=0)
    zs = [v["z"] for v in per.values() if v["z"] is not None]
    return dict(per_key=per, pooled=pooled, stouffer=stouffer, n_blocks=used, n_keys=len(keys), n_perm=n_perm,
                window=window, max_abs_z=max((abs(z) for z in zs), default=None),
                n_abs_z_over_3=int(sum(abs(z) > 3 for z in zs)))


def cross_memory_test(blocks: list, n_perm: int = N_PERM, rng=None, chunk: int = 100) -> dict:
    """the same memory qubit x qubit: Delta[a, b] = P(b now | a before) - P(b now | not a before), over adjacent pairs
    inside blocks that hold both keys, against the same within-block row shuffles. Off the diagonal it is the
    'other qubits' memory; its family-wise reading is the largest |z| against the null distribution of the largest
    |z| (each replicate standardized by the null's own mean and sd)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    keys = sorted({k for _, ks in blocks for k in ks}, key=lambda k: (str(type(k)), k))
    col = {k: i for i, k in enumerate(keys)}
    Q = len(keys)
    obs = np.zeros((4, Q, Q))
    null = np.zeros((n_perm, 4, Q, Q))

    def counts(P):                                  # P (..., n, Qb) float -> (..., 4, Qb, Qb)
        prev, now = P[..., :-1, :], P[..., 1:, :]
        n11 = np.einsum("...ka,...kb->...ab", prev, now)
        n01 = np.einsum("...ka,...kb->...ab", 1 - prev, now)
        n1 = np.broadcast_to(prev.sum(-2)[..., :, None], n11.shape)
        n0 = (prev.shape[-2] - n1)
        return np.stack([n1, n11, n0, n01], axis=-3)
    used = 0
    for E, ks in blocks:
        E = np.asarray(E, dtype=float)
        if E.ndim != 2 or E.shape[0] < 2 or E.shape[1] != len(ks):
            continue
        g = np.array([col[k] for k in ks])
        ix = np.ix_(g, g)
        c = counts(E)
        for j in range(4):
            obs[j][ix] += c[j]
        n = E.shape[0]
        for s in range(0, n_perm, chunk):
            m = min(chunk, n_perm - s)
            cp = counts(E[_perm_index(n, m, rng)])
            for j in range(4):
                null[s:s + m, j][(slice(None),) + ix] += cp[:, j]
        used += 1
    if not used:
        raise R168Refused("no block of two or more shots")
    with np.errstate(invalid="ignore", divide="ignore"):
        d_obs = obs[1] / obs[0] - obs[3] / obs[2]
        d_null = null[:, 1] / null[:, 0] - null[:, 3] / null[:, 2]
        mu, sd = np.nanmean(d_null, 0), np.nanstd(d_null, 0, ddof=1)
        z = (d_obs - mu) / sd
        zn = (d_null - mu) / sd
    distinct = np.array([[len(np.unique(np.round(v[np.isfinite(v)], 12))) for v in row]
                         for row in np.moveaxis(d_null, 0, -1)])
    off = ~np.eye(Q, dtype=bool) & np.isfinite(z) & (sd > 0) & (distinct >= MIN_DISTINCT)
    if not off.any():
        return dict(keys=[str(k) for k in keys], n_pairs=0, n_blocks=used, n_perm=n_perm)
    zmax = float(np.max(np.abs(z[off])))
    zmax_null = np.array([np.nanmax(np.abs(r[off])) for r in zn])
    i, j = np.argwhere(off & (np.abs(np.nan_to_num(z)) == zmax))[0]
    pooled_obs = obs[:, off].sum(-1)
    pooled_null = null[:, :, off].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        po = pooled_obs[1] / pooled_obs[0] - pooled_obs[3] / pooled_obs[2]
        pn = pooled_null[:, 1] / pooled_null[:, 0] - pooled_null[:, 3] / pooled_null[:, 2]
    n3 = int(np.sum(np.abs(z[off]) > 3))
    n3_null = float(np.mean([np.sum(np.abs(r[off]) > 3) for r in zn]))
    return dict(keys=[str(k) for k in keys], n_pairs=int(off.sum()), n_blocks=used, n_perm=n_perm,
                max_abs_z=zmax, max_pair=[str(keys[i]), str(keys[j])], max_pair_delta=float(d_obs[i, j]),
                familywise_p=(1 + int(np.sum(zmax_null >= zmax))) / (1 + len(zmax_null)),
                n_abs_z_over_3=n3, n_abs_z_over_3_null_mean=n3_null,
                pooled_off_diagonal=_null_summary(po, pn))


def reset_regression(rates: list, s_pred: list, n_perm: int = N_PERM, rng=None) -> dict:
    """the IMMEDIATE-predecessor (reset) effect. rates[P] (R, Q_P): PUB P's per-randomization error rates on its
    deterministic clbits; s_pred[P] (R, Q_P): the predecessor's final P(1) on each clbit's qubit (nan if unknown).
    Model inside each (PUB, clbit): rate = a + eps * s_pred, eps by least squares on the deviations from the
    (PUB, clbit) means (a PUB's own level never enters). Null: the randomization labels of each PUB permuted (jointly
    over its clbits): every block keeps its rate, its tie to its predecessor is broken. eps > 0 is a reset that
    leaves a |1> behind as a |1>."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    X, Y = [], []
    for r, s in zip(rates, s_pred):
        r, s = np.asarray(r, float), np.asarray(s, float)
        if r.shape != s.shape or r.ndim != 2 or r.shape[0] < 2:
            raise R168Refused(f"rates {r.shape} and predecessor states {s.shape} must be one (R >= 2, Q) shape")
        keep = np.all(np.isfinite(s), axis=0) & np.all(np.isfinite(r), axis=0)
        if not keep.any():
            continue
        X.append(s[:, keep] - s[:, keep].mean(0))
        Y.append(r[:, keep])
    sxx = float(sum((x ** 2).sum() for x in X))
    if not X or sxx <= 0:
        return dict(eps=None, n_pubs=len(X), sxx=sxx, note="no PUB whose predecessor state changes across its "
                                                          "randomizations: the effect is not identifiable")

    def est(Ys):
        return sum(float((x * (y - y.mean(0))).sum()) for x, y in zip(X, Ys)) / sxx
    e_obs = est(Y)
    e_null = np.array([est([y[rng.permutation(y.shape[0])] for y in Y]) for _ in range(n_perm)])
    out = _null_summary(e_obs, e_null)
    out.update(eps=e_obs, n_pubs=len(X), n_clbit_series=int(sum(x.shape[1] for x in X)),
               n_contrasts=int(sum((np.abs(x).sum(0) > 0).sum() for x in X)), sxx=sxx)
    return out


# ================================================================== the order tests (pure)
def k_profile_test(blocks: list, n_bins: int = 10, n_perm: int = N_PERM, rng=None, chunk: int = 100) -> dict:
    """blocks: [(E (n, Q_b) bool, keys)] of ONE span, every block with the same n. Under interleaved sweeps (H_I) the
    stored index k is time for every block alike, so a slow change is a profile in k SHARED by the blocks. Each
    block's bin counts are standardized by their exact within-block permutation (hypergeometric) variance, summed
    over blocks per (bin, key); T = sum of squares / blocks contributing (~ n_bins - 1 per key with no profile).
    Null: each block's shots permuted. Also the first stored shot against the rest, pooled (a start-of-batch or
    block-boundary effect)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    ns = {np.asarray(E).shape[0] for E, _ in blocks}
    if len(ns) != 1:
        raise R168Refused(f"a k-profile needs one n across the span's blocks, got {sorted(ns)}")
    n = ns.pop()
    if n < 2 * n_bins:
        raise R168Refused(f"{n} shots per block are too few for {n_bins} bins")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    size = np.diff(edges)
    keys = sorted({k for _, ks in blocks for k in ks}, key=lambda k: (str(type(k)), k))
    col = {k: i for i, k in enumerate(keys)}
    S_obs = np.zeros((n_bins, len(keys)))
    S_null = np.zeros((n_perm, n_bins, len(keys)))
    B = np.zeros(len(keys))
    first, first_exp, first_var = 0.0, 0.0, 0.0
    for E, ks in blocks:
        E = np.asarray(E, dtype=bool)
        g = np.array([col[k] for k in ks])
        t = E.sum(0).astype(float)
        live = (t > 0) & (t < n)
        if not live.any():
            continue
        p = t / n
        var = size[:, None] * p * (1 - p) * (n - size[:, None]) / (n - 1)
        sd = np.where(live, np.sqrt(np.maximum(var, 1e-300)), np.inf)
        e = size[:, None] * p

        def bins(X):                                # (..., n, Qb) -> (..., n_bins, Qb)
            cs = np.concatenate([np.zeros(X.shape[:-2] + (1, X.shape[-1])), np.cumsum(X, axis=-2)], axis=-2)
            return np.take(cs, edges[1:], axis=-2) - np.take(cs, edges[:-1], axis=-2)
        S_obs[:, g] += (bins(E.astype(float)) - e) / sd
        for s in range(0, n_perm, chunk):
            m = min(chunk, n_perm - s)
            S_null[s:s + m][:, :, g] += (bins(E[_perm_index(n, m, rng)].astype(float)) - e) / sd
        B[g] += live
        first += float(E[0].sum())
        first_exp += float(p.sum())
        first_var += float((p * (1 - p)).sum())
    ok = B > 0
    if not ok.any():
        raise R168Refused("no block with a varying column")
    T_obs = float((S_obs[:, ok] ** 2 / B[ok]).sum())
    T_null = (S_null[:, :, ok] ** 2 / B[ok]).sum((1, 2))
    out = _null_summary(T_obs, T_null, tail="upper")
    out.update(statistic="sum over bins, keys of (sum_b standardized bin deviation)^2 / blocks",
               n_bins=n_bins, n=n, n_keys=int(ok.sum()), excess_factor=T_obs / float(T_null.mean()),
               shared_profile=(S_obs[:, ok].sum(1) / math.sqrt(B[ok].sum())).tolist(),
               first_shot=dict(observed=first, expected=first_exp,
                               z=(first - first_exp) / math.sqrt(first_var) if first_var > 0 else None))
    return out


def block_profile_test(rates, n: int, n_perm: int = N_PERM, rng=None) -> dict:
    """rates (P, R, Q): the per-randomization error rates of P IDENTICAL-across-randomizations PUBs of one span (no
    two-qubit gate, so no twirl difference) on the same Q clbits, n shots per block. Under circuit-major blocks the
    randomization index is a time proxy: rand-major execution puts a slow change in a randomization profile SHARED by
    the PUBs (prof); pub-major execution makes neighbouring randomizations of one PUB alike (lag). Null: the
    randomization labels permuted inside each PUB (jointly over clbits)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    r = np.asarray(rates, float)
    if r.ndim != 3 or r.shape[0] < 1 or r.shape[1] < 3:
        raise R168Refused(f"rates must be (PUBs, randomizations >= 3, clbits), got {r.shape}")
    pbar = r.mean(1, keepdims=True)
    X = (r - pbar) / np.sqrt(np.maximum(pbar * (1 - pbar), 1.0 / n) / n)

    def stats(Z):
        return float((Z.sum(0) ** 2).sum() / Z.shape[0]), float((Z[:, 1:] * Z[:, :-1]).sum())
    o = stats(X)
    nl = np.array([stats(np.stack([x[rng.permutation(x.shape[0])] for x in X])) for _ in range(n_perm)])
    return dict(rand_profile=_null_summary(o[0], nl[:, 0], tail="upper"),
                within_pub_lag1=_null_summary(o[1], nl[:, 1], tail="upper"), n_pubs=int(r.shape[0]),
                n_clbits=int(r.shape[2]), shared_rand_profile=(X.sum(0).mean(1) / math.sqrt(r.shape[0])).tolist())


# ================================================================== 10: bursts (pure)
def poisson_binomial(p) -> np.ndarray:
    """pmf of the number of successes of independent Bernoulli(p_i): exact, by convolution"""
    p = np.asarray(p, float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise R168Refused("Poisson-binomial rates must be a 1-D array in [0, 1]")
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for q in p:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= 1 - q
    return pmf


def _tail_counts(M, m_max: int) -> np.ndarray:
    """M (..., n) ints -> (..., m_max + 1): the number of shots with multiplicity >= m, m = 0..m_max"""
    M = np.minimum(np.asarray(M, int), m_max)
    lead = M.shape[:-1]
    flat = M.reshape(-1, M.shape[-1])
    off = flat + (m_max + 1) * np.arange(flat.shape[0])[:, None]
    h = np.bincount(off.ravel(), minlength=flat.shape[0] * (m_max + 1)).reshape(flat.shape[0], m_max + 1)
    return np.cumsum(h[:, ::-1], axis=1)[:, ::-1].reshape(lead + (m_max + 1,))


def _shuffle_columns(E, m: int, rng, window: int | None) -> np.ndarray:
    """m replicates of E (n, Q), every column permuted independently -- inside consecutive windows of `window` shots
    when given (the last partial window on its own)"""
    X = np.broadcast_to(E, (m,) + E.shape).copy()
    if not window or window >= E.shape[0]:
        return rng.permuted(X, axis=1)
    n_full = (E.shape[0] // window) * window
    head = X[:, :n_full].reshape(m, n_full // window, window, E.shape[1])
    X[:, :n_full] = rng.permuted(head, axis=2).reshape(m, n_full, E.shape[1])
    if n_full < E.shape[0]:
        X[:, n_full:] = rng.permuted(X[:, n_full:], axis=1)
    return X


def burst_test(blocks: list, n_perm: int = N_PERM, rng=None, window: int | None = None, chunk: int = 100) -> dict:
    """blocks: [E (n, Q_b) bool] -- per shot, which qubits erred (or were IQ outliers). Multiplicity M = per-shot
    count. The observed number of shots with M >= m against (a) the Poisson-binomial of each block's own per-qubit
    rates, summed over blocks (independent qubits, rates exact per block), and (b) `n_perm` replicates permuting every
    column independently inside its block -- inside windows of `window` shots when given, which keeps slow drift common
    to the qubits and removes only shot-level coincidence. Also the pairwise coincidence count sum M (M - 1) / 2 and
    the largest multiplicity seen."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    blocks = [np.asarray(E, dtype=bool) for E in blocks if np.asarray(E).size]
    if not blocks:
        raise R168Refused("no shots")
    m_max = max(E.shape[1] for E in blocks)
    obs = np.zeros(m_max + 1)
    pb = np.zeros(m_max + 1)
    pb_var = np.zeros(m_max + 1)
    pairs_obs = pairs_pb = 0.0
    null = np.zeros((n_perm, m_max + 1))
    pairs_null = np.zeros(n_perm)
    maxm_obs, maxm_null = 0, np.zeros(n_perm)
    shots = 0
    for E in blocks:
        n, Q = E.shape
        M = E.sum(1)
        obs += np.pad(_tail_counts(M, Q), (0, m_max - Q))
        pairs_obs += float((M * (M - 1)).sum() / 2)
        maxm_obs = max(maxm_obs, int(M.max()))
        p = E.mean(0)
        t = np.cumsum(poisson_binomial(p)[::-1])[::-1]
        pb += np.pad(n * t, (0, m_max - Q))
        pb_var += np.pad(n * t * (1 - t), (0, m_max - Q))
        pairs_pb += n * float((p.sum() ** 2 - (p ** 2).sum()) / 2)
        for s in range(0, n_perm, chunk):
            m = min(chunk, n_perm - s)
            Mn = _shuffle_columns(E, m, rng, window).sum(2)
            null[s:s + m] += np.pad(_tail_counts(Mn, Q), ((0, 0), (0, m_max - Q)))
            pairs_null[s:s + m] += (Mn * (Mn - 1)).sum(1) / 2
            maxm_null[s:s + m] = np.maximum(maxm_null[s:s + m], Mn.max(1))
        shots += n
    top = int(max(maxm_obs, np.max(np.nonzero(pb > 1e-3)[0], initial=0))) + 1
    rows = []
    for m in range(1, min(top, m_max) + 1):
        s = _null_summary(obs[m], null[:, m], tail="upper")
        rows.append(dict(m=m, observed=int(obs[m]), poisson_binomial=float(pb[m]),
                         pb_sd=float(math.sqrt(pb_var[m])), ratio_to_pb=float(obs[m] / pb[m]) if pb[m] > 0 else None,
                         perm_mean=s["null_mean"], perm_sd=s["null_sd"], z_perm=s["z"], p_perm=s["p"]))
    return dict(shots=shots, n_blocks=len(blocks), max_qubits=m_max, window=window, n_perm=n_perm, tails=rows,
                pairs=dict(observed=pairs_obs, poisson_binomial=pairs_pb,
                           **{k: v for k, v in _null_summary(pairs_obs, pairs_null, "upper").items() if k != "value"}),
                max_multiplicity=dict(observed=maxm_obs,
                                      p_perm=(1 + int(np.sum(maxm_null >= maxm_obs))) / (1 + n_perm)))


def high_threshold(p, alpha: float = 0.01) -> int:
    """the smallest multiplicity whose Poisson-binomial upper tail (independent qubits at rates p) is <= alpha"""
    t = np.cumsum(poisson_binomial(p)[::-1])[::-1]
    ok = np.nonzero(t <= alpha)[0]
    return int(ok[0]) if len(ok) else len(t)


def cluster_test(groups: list, n_perm: int = N_PERM, rng=None, lags: int = 3, chunk: int = 200) -> dict:
    """groups: [H (B, n) bool] -- per span, which stored shots of its B blocks are HIGH (e.g. multiplicity in the
    block's upper 1% Poisson-binomial tail). Two clocks: A = high pairs at stored lag 1..lags inside a block (H_C:
    consecutive shots; H_I: consecutive sweeps of one circuit), and S = high pairs sharing a stored index k across
    blocks of a span (H_I: one sweep, ms apart). Null: every block's shots permuted."""
    rng = rng if rng is not None else np.random.default_rng(SEED)

    def stats(H):                                       # (..., B, n) -> (A, S)
        A = sum((H[..., :, d:] & H[..., :, :-d]).sum((-1, -2)) for d in range(1, lags + 1))
        c = H.sum(-2)
        return A, (c * (c - 1) // 2).sum(-1)
    A_obs = S_obs = 0
    A_null, S_null = np.zeros(n_perm), np.zeros(n_perm)
    high = 0
    for H in groups:
        H = np.asarray(H, dtype=bool)
        if H.ndim != 2 or H.shape[1] <= lags:
            continue
        a, s_ = stats(H)
        A_obs += int(a)
        S_obs += int(s_)
        high += int(H.sum())
        B, n = H.shape
        for s in range(0, n_perm, chunk):
            m = min(chunk, n_perm - s)
            Hp = np.broadcast_to(H, (m, B, n)).copy()
            Hp = rng.permuted(Hp, axis=2)
            a_, s2 = stats(Hp)
            A_null[s:s + m] += a_
            S_null[s:s + m] += s2
    return dict(high_shots=high, lags=lags, n_perm=n_perm,
                within_block_adjacent=_null_summary(A_obs, A_null, "upper"),
                same_sweep_across_blocks=_null_summary(S_obs, S_null, "upper"))


# ================================================================== 11: telegraph switching (pure)
def window_sums(X, W: int) -> np.ndarray:
    """(..., n) -> (..., n // W): sums over consecutive windows of W (a remainder shorter than W is dropped)"""
    X = np.asarray(X)
    k = X.shape[-1] // W
    if k < 1:
        raise R168Refused(f"a window of {W} is longer than the series ({X.shape[-1]})")
    return X[..., :k * W].reshape(X.shape[:-1] + (k, W)).sum(-1)


def _xlogx_bern(y, n):
    """y log(y / n) + (n - y) log(1 - y / n), with 0 log 0 = 0"""
    y, n = np.asarray(y, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, y / np.maximum(n, 1e-300), 0.0)
        a = np.where(y > 0, y * np.log(np.where(p > 0, p, 1.0)), 0.0)
        b = np.where(n - y > 0, (n - y) * np.log(np.where(p < 1, 1 - p, 1.0)), 0.0)
    return a + b


def epidemic_lr(Y, N) -> tuple:
    """the largest two-state binomial likelihood ratio over segments [a, b) of windows (not the whole series): rate
    inside vs outside the segment against one rate -- a telegraph excursion, or a switch when the segment touches an
    end. Y (..., M) counts, N (M,) trials per window. Returns (LR (...), a (...), b (...))."""
    Y = np.asarray(Y, float)
    N = np.broadcast_to(np.asarray(N, float), Y.shape[-1:])
    M = Y.shape[-1]
    cy = np.concatenate([np.zeros(Y.shape[:-1] + (1,)), np.cumsum(Y, -1)], -1)
    cn = np.concatenate([[0.0], np.cumsum(N)])
    a, b = np.triu_indices(M + 1, 1)
    keep = ~((a == 0) & (b == M))
    a, b = a[keep], b[keep]
    yin = cy[..., b] - cy[..., a]
    nin = cn[b] - cn[a]
    ytot = cy[..., -1:]
    LR = 2 * (_xlogx_bern(yin, nin) + _xlogx_bern(ytot - yin, cn[-1] - nin) - _xlogx_bern(ytot, cn[-1]))
    best = np.argmax(LR, -1)
    return np.take_along_axis(LR, best[..., None], -1)[..., 0], a[best], b[best]


def _corr_last(a, b):
    """Pearson correlation along the last axis (0 where either side is constant)"""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), 0.0)


def telegraph_test(E, W: int, n_perm: int = N_PERM, rng=None, chunk: int = 100, half=None, half_rand=None) -> dict:
    """E (B, n) bool: one qubit's error indicators in the B blocks of one span (its deterministic clbits). Under H_I
    the sweep series y[k] = sum_b E[b, k] is its error count at time k:
      disp_sweep  sum over windows of W sweeps of (Y_w - mean)^2 -- reported as a variance ratio to the null mean;
      lr_sweep    the largest two-state (segment) binomial likelihood ratio on the windows (epidemic_lr);
      split_half  (with `half`, a (B,) bool splitting the blocks into two disjoint sets of circuits, e.g. even and
                  odd PUBs) the correlation of the two halves' window profiles: a change in TIME appears in both at
                  the same windows, a change that belongs to some circuits does not;
      split_rand  (with `half_rand`) the same for even against odd RANDOMIZATIONS of the same circuits -- the
                  sharpest clock test: under circuit-major blocks two randomizations of one PUB never run at once,
                  and an event that only some circuits can see (a frequency jump seen by a Ramsey, not by a readout)
                  is still in both halves;
    under H_C, each block's own series: disp_block = sum_b sum_w (Y_bw - mean_b)^2.
    Null: every block's shots permuted independently -- each block keeps its own rate, so a circuit that errs more is
    never read as a moment that errs more. Returns each statistic against the null, the segment and its two rates."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    E = np.asarray(E, dtype=bool)
    if E.ndim != 2 or E.shape[1] < 2 * W:
        raise R168Refused(f"a telegraph test needs (blocks, n >= 2W) shots, got {E.shape} with W = {W}")
    B, n = E.shape
    N = np.full(n // W, float(B * W))
    masks = []
    for mk in (half, half_rand):
        h_ = None if mk is None else np.asarray(mk, dtype=bool)
        if h_ is not None and (h_.shape != (B,) or h_.all() or not h_.any()):
            raise R168Refused("a split half needs a (blocks,) mask with blocks on both sides")
        masks.append(h_)
    h, hr = masks

    def split(X, hm, like):
        if hm is None:
            return np.zeros_like(like)
        return _corr_last(window_sums(X[..., hm, :].sum(-2), W), window_sums(X[..., ~hm, :].sum(-2), W))

    def stats(X):                                       # (..., B, n) -> disp_sweep, lr, a, b, disp_block, splits
        Ys = window_sums(X.sum(-2), W)
        ds = ((Ys - Ys.mean(-1, keepdims=True)) ** 2).sum(-1)
        lr, a, b = epidemic_lr(Ys, N)
        Yb = window_sums(X, W)
        db = ((Yb - Yb.mean(-1, keepdims=True)) ** 2).sum((-1, -2))
        return ds, lr, a, b, db, split(X, h, ds), split(X, hr, ds)
    ds, lr, a, b, db, sp, sr = stats(E.astype(float))
    nd, nl, nb, ns, nr = (np.zeros(n_perm) for _ in range(5))
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        Xp = rng.permuted(np.broadcast_to(E, (m, B, n)).astype(float), axis=2)
        d_, l_, _, _, b_, s_, r_ = stats(Xp)
        nd[s:s + m], nl[s:s + m], nb[s:s + m], ns[s:s + m], nr[s:s + m] = d_, l_, b_, s_, r_
    Ys = window_sums(E.sum(0).astype(float), W)
    a, b = int(a), int(b)
    inside = Ys[a:b].sum() / (B * W * (b - a))
    outside = (Ys.sum() - Ys[a:b].sum()) / (B * W * (len(Ys) - (b - a))) if (len(Ys) - (b - a)) else None
    out = dict(blocks=B, n=n, window=W, rate=float(E.mean()),
               disp_sweep=dict(**_null_summary(float(ds), nd, "upper"), ratio=float(ds) / float(nd.mean())
                               if nd.mean() > 0 else None),
               lr_sweep=dict(**_null_summary(float(lr), nl, "upper"), segment_sweeps=[a * W, b * W],
                             rate_inside=float(inside), rate_outside=None if outside is None else float(outside)),
               disp_block=dict(**_null_summary(float(db), nb, "upper"), ratio=float(db) / float(nb.mean())
                               if nb.mean() > 0 else None),
               profile=(Ys / (B * W)).tolist())
    if h is not None:
        out["split_half"] = dict(**_null_summary(float(sp), ns, "upper"), blocks=[int(h.sum()), int((~h).sum())])
    if hr is not None:
        out["split_rand"] = dict(**_null_summary(float(sr), nr, "upper"), blocks=[int(hr.sum()), int((~hr).sum())])
    return out


def centre_drift_test(V, W: int, n_perm: int = N_PERM, rng=None, chunk: int = 50) -> dict:
    """V (B, n) float, nan where a shot is not used (read outside the prepared state's cluster): a qubit's projected
    IQ in units of its cluster sd, in the B blocks of one span. Window means over sweeps (H_I): the dispersion
    sum_w c_w (m_w - m)^2 and the linear slope of m_w per 1000 sweeps; null: every block's shots permuted."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    V = np.asarray(V, float)
    if V.ndim != 2 or V.shape[1] < 2 * W:
        raise R168Refused(f"centre drift needs (blocks, n >= 2W), got {V.shape}")
    B, n = V.shape

    def stats(X):
        fin = np.isfinite(X)
        s = window_sums(np.where(fin, X, 0.0).sum(-2), W)
        c = window_sums(fin.sum(-2).astype(float), W)
        mw = s / np.maximum(c, 1)
        m = s.sum(-1, keepdims=True) / np.maximum(c.sum(-1, keepdims=True), 1)
        disp = (c * (mw - m) ** 2).sum(-1)
        x = (np.arange(s.shape[-1]) + 0.5) * W
        xm = (c * x).sum(-1, keepdims=True) / np.maximum(c.sum(-1, keepdims=True), 1)
        slope = (c * (x - xm) * (mw - m)).sum(-1) / np.maximum((c * (x - xm) ** 2).sum(-1), 1e-300)
        return disp, 1000 * slope
    d, sl = stats(V)
    nd, ns = np.zeros(n_perm), np.zeros(n_perm)
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        d_, s_ = stats(rng.permuted(np.broadcast_to(V, (m, B, n)).copy(), axis=2))
        nd[s:s + m], ns[s:s + m] = d_, s_
    return dict(blocks=B, n=n, window=W, used=int(np.isfinite(V).sum()),
                dispersion=dict(**_null_summary(float(d), nd, "upper"),
                                ratio=float(d) / float(nd.mean()) if nd.mean() > 0 else None),
                slope_sd_per_1000_sweeps=_null_summary(float(sl), ns))


def between_groups(means, sems) -> dict:
    """chi2 of group means against their weighted mean, with each group's own sem: are the groups one value?"""
    from scipy.stats import chi2
    m, s = np.asarray(means, float), np.asarray(sems, float)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    if ok.sum() < 2:
        return dict(chi2=None, dof=int(max(ok.sum() - 1, 0)), p=None)
    w = 1 / s[ok] ** 2
    mu = float((w * m[ok]).sum() / w.sum())
    c = float((w * (m[ok] - mu) ** 2).sum())
    return dict(chi2=c, dof=int(ok.sum() - 1), p=float(chi2.sf(c, ok.sum() - 1)), weighted_mean=mu,
                range=float(m[ok].max() - m[ok].min()))


# ================================================================== 16: drift across jobs (pure)
def structure_function(times_h, X) -> dict:
    """X (T, Q) positive values at T calibration times (hours): for every pair of calibrations, the mean over qubits of
    (log X_j - log X_i)^2 against the elapsed time. A quantity that only re-measures with noise has a FLAT structure
    function (its intercept is twice the noise variance); one that drifts as a random walk grows linearly in dt.
    Fitted D(dt) = a + b dt by least squares over the pairs; returns the pairs, a, b."""
    t = np.asarray(times_h, float)
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] != len(t) or len(t) < 3:
        raise R168Refused(f"a structure function needs (T >= 3, Q) values at T times, got {X.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(X > 0, np.log(X), np.nan)
    i, j = np.triu_indices(len(t), 1)
    D = (L[j] - L[i]) ** 2                              # (pairs, Q)
    dt = np.abs(t[j] - t[i])
    keep = dt > 0
    D, dt = D[keep], dt[keep]
    Dm = np.nanmean(D, 1)
    A = np.stack([np.ones_like(dt), dt], 1)
    (a, b), *_ = np.linalg.lstsq(A, Dm, rcond=None)
    return dict(dt_h=dt.tolist(), mean_sq_dlog=Dm.tolist(), median_abs_dlog=np.nanmedian(np.sqrt(D), 1).tolist(),
                intercept=float(a), slope_per_h=float(b), per_pair_per_qubit=D)


def diffusion_fit(sf: dict, n_boot: int = 1000, rng=None) -> dict:
    """the structure function's (intercept, slope) with a bootstrap over QUBITS (pairs of one qubit share its noise)"""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    D, dt = np.asarray(sf["per_pair_per_qubit"]), np.asarray(sf["dt_h"])
    A = np.stack([np.ones_like(dt), dt], 1)
    Q = D.shape[1]
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, Q, Q)
        Dm = np.nanmean(D[:, idx], 1)
        (a, b), *_ = np.linalg.lstsq(A, Dm, rcond=None)
        reps.append((a, b))
    reps = np.array(reps)
    a, b = sf["intercept"], sf["slope_per_h"]
    return dict(intercept=a, intercept_sd=float(reps[:, 0].std(ddof=1)), slope_per_h=b,
                slope_sd=float(reps[:, 1].std(ddof=1)),
                slope_z=float(b / reps[:, 1].std(ddof=1)) if reps[:, 1].std(ddof=1) > 0 else None,
                white_rel_sd=float(math.sqrt(max(a, 0) / 2)),
                walk_rel_sd_per_sqrt_h=float(math.sqrt(max(b, 0))))


def refresh_epochs(times_h, X) -> dict:
    """consecutive calibration windows whose values are IDENTICAL carry ONE measurement (the vendor stamps a new
    window when ANY property updates; on record it re-measures T1, T2 and sx once a day and readout every few
    hours). Returns the epochs: the time of the first window that carried each, its values, and the window indices."""
    t = np.asarray(times_h, float)
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] != len(t) or len(t) < 1:
        raise R168Refused(f"epochs need (T, Q) values at T times, got {X.shape}")
    keep = [0]
    for i in range(1, len(t)):
        if not np.array_equal(np.nan_to_num(X[keep[-1]], nan=-1.0), np.nan_to_num(X[i], nan=-1.0)):
            keep.append(i)
    return dict(times_h=t[keep], X=X[keep], index=keep, n_epochs=len(keep), n_windows=len(t))


def epoch_changes(ep: dict) -> list:
    """between consecutive epochs: elapsed hours, the median and 90th-percentile |log change| over qubits, and the
    fraction of qubits that changed by more than a factor 2"""
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(ep["X"] > 0, np.log(ep["X"]), np.nan)
    for i in range(1, ep["n_epochs"]):
        d = np.abs(L[i] - L[i - 1])
        d = d[np.isfinite(d)]
        out.append(dict(dt_h=float(ep["times_h"][i] - ep["times_h"][i - 1]),
                        median_abs_dlog=float(np.median(d)) if len(d) else None,
                        p90_abs_dlog=float(np.percentile(d, 90)) if len(d) else None,
                        frac_over_2x=float(np.mean(d > math.log(2))) if len(d) else None, qubits=int(len(d))))
    return out


def binomial_dispersion(P, N: int = VENDOR_RO_SHOTS) -> dict:
    """P (T, Q): a proportion re-measured T times from N shots each. Per qubit, chi2 of the counts against one common
    rate (T - 1 dof): beyond binomial sampling is real change. Also the relative sd beyond sampling."""
    from scipy.stats import chi2
    P = np.asarray(P, float)
    k = P * N
    pbar = np.nanmean(P, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.nansum((k - N * pbar) ** 2, 0) / (N * pbar * (1 - pbar))
    dof = np.sum(np.isfinite(P), 0) - 1
    p = np.where((dof > 0) & np.isfinite(c), chi2.sf(np.nan_to_num(c), np.maximum(dof, 1)), np.nan)
    var_obs = np.nanvar(P, 0, ddof=1)
    var_bin = pbar * (1 - pbar) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(np.maximum(var_obs - var_bin, 0)) / pbar
    return dict(chi2=c, dof=dof, p=p, excess_rel_sd=rel, mean=pbar)


def two_wait_detuning(r1: float, r2: float, W: float) -> dict | None:
    """a coherent offset |Delta| and contrast C from the Ramsey ratios at W and 2W, read as r(t) = C cos(Delta t)
    (no decay inside 2W; the smallest alias): cos 2x / cos x = r2 / r1 gives cos x = (rho + sqrt(rho^2 + 8)) / 4"""
    if not (math.isfinite(r1) and math.isfinite(r2)) or abs(r1) < 1e-6 or W <= 0:
        return None
    rho = r2 / r1
    c = (rho + math.sqrt(rho * rho + 8)) / 4
    if not -1 <= c <= 1:
        return None
    x = math.acos(c)
    return dict(abs_hz=x / (2 * math.pi * W), contrast=r1 / c if c else None, cos_x=c)


def two_wait_sd(r1: float, s1: float, r2: float, s2: float, W: float, n: int = 4000, rng=None) -> dict | None:
    """two_wait_detuning with its shot error: the ratios resampled from their binomial sds"""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    est = two_wait_detuning(r1, r2, W)
    if est is None:
        return None
    hz = [e["abs_hz"] for e in (two_wait_detuning(a, b, W) for a, b in
                                zip(rng.normal(r1, s1, n), rng.normal(r2, s2, n))) if e is not None]
    return dict(est, abs_hz_sd=float(np.std(hz, ddof=1)) if len(hz) > 10 else None, draws_ok=len(hz))


def stability_table(metrics: dict, veto: dict | None = None) -> list:
    """metrics: name -> (Q,) array, larger = LESS stable (nan = not measured). Per qubit: each metric's percentile
    among the qubits that have it (0 = most stable), and the composite = their mean. veto: name -> (Q,) bool, a
    condition that disqualifies a qubit for layout whatever its average (a readout that jumped tenfold between jobs
    must not be averaged away by eight quiet metrics). Rows sorted: unvetoed most stable first, then vetoed, then
    unmeasured."""
    veto = veto or {}
    names = sorted(metrics)
    Q = len(next(iter(metrics.values())))
    pct = {}
    for k in names:
        v = np.asarray(metrics[k], float)
        ok = np.isfinite(v)
        r = np.full(Q, np.nan)
        if ok.sum() > 1:
            order = v[ok].argsort().argsort()
            r[ok] = order / (ok.sum() - 1)
        pct[k] = r
    rows = []
    for q in range(Q):
        vals = [pct[k][q] for k in names if np.isfinite(pct[k][q])]
        rows.append(dict(qubit=q, composite=float(np.mean(vals)) if vals else None, n_metrics=len(vals),
                         veto=sorted(k for k, v in veto.items() if bool(np.asarray(v)[q])),
                         **{k: (None if not np.isfinite(metrics[k][q]) else float(metrics[k][q])) for k in names}))
    rows.sort(key=lambda r: (r["composite"] is None, bool(r["veto"]),
                             r["composite"] if r["composite"] is not None else 0))
    for i, r in enumerate(rows):
        r["rank"] = i + 1 if r["composite"] is not None else None
    return rows


VETO_CHI2_PER_DOF = 25.0      # an in-circuit change between jobs at ~5 sd (1 dof) disqualifies
VETO_READOUT = 0.05           # a vendor readout error above 5% is a broken readout
VETO_TELEGRAPH_Z = 5.0        # a flagged switch (telegraph_flags' threshold)


# ================================================================== loading (reads files; no network)
def _discriminators(iq_by_name, qubits_by_name, cal):
    from oneq.hw import iq as IQ
    return IQ.calibrate(iq_by_name, qubits_by_name, *cal)


def load_job(spec: dict) -> dict:
    """one job: its frozen record and circuits, flight record, spans (with data slices), snapshots before and after,
    and per PUB: physical qubits per clbit, bits (vendor or the job's own IQ discriminators), the IQ projection in
    units of the cluster sd and the outlier flags (kerneled jobs), the ideal outcomes, and whether it has 2q gates"""
    from oneq.hw import freeze as F
    from oneq.hw import iq as IQ
    d = EV / spec["dir"]
    rp = d / spec["record"]
    rec = json.loads(rp.read_text(encoding="utf-8"))
    circs = F.load(rec, rp.with_suffix(".qpy").read_bytes(), check_sources=False)
    idx = json.loads((d / f"flight_{spec['job']}.json").read_text(encoding="utf-8"))
    if idx["composite_qpy_sha256"] != rec["qpy_sha256"]:
        raise R168Refused(f"{spec['tag']}: the flight record is not of this frozen record")
    arr = np.load(d / f"flight_{spec['job']}.npz")
    spans = idx["execution_spans"]
    if not all(s.get("slices") for s in spans):
        alt = json.loads((d / spec["spans"]).read_text(encoding="utf-8"))["spans"]
        same = len(alt) == len(spans) and all(a["start"] == s["start"] and a["stop"] == s["stop"] and
                                              {int(k): v for k, v in a["pubs"].items()} ==
                                              {int(k): v for k, v in s["pubs"].items()} for a, s in zip(alt, spans))
        if not same:
            raise R168Refused(f"{spec['tag']}: the spans file does not match the flight record's spans")
        spans = alt
    shots = {p["index"]: next(iter(p["registers"].values()))["num_shots"] for p in idx["pubs"]}
    layout = span_layout(spans, shots)
    qubits_by_name = {n: IQ.measured_qubits(qc) for n, qc in circs.items()}
    kerneled = "iq_calibration" in spec
    iq = {}
    discs = None
    if kerneled:
        iq = {p["name"]: IQ.iq_columns(idx, arr, p["name"]) for p in idx["pubs"]}
        discs = _discriminators(iq, qubits_by_name, spec["iq_calibration"])
    interest = sorted({q for qs in qubits_by_name.values() for q in qs})
    pubs = []
    for p in idx["pubs"]:
        name = p["name"]
        if len(p["registers"]) != 1:
            raise R168Refused(f"{spec['tag']}/{name}: {len(p['registers'])} registers; one is expected")
        reg = next(iter(p["registers"].values()))
        qs = qubits_by_name[name]
        if kerneled:
            z = iq[name]
            bits = np.stack([IQ.classify(discs[q], z[:, j]) for j, q in enumerate(qs)], 1)
            out = np.stack([IQ.outliers(discs[q], z[:, j]) for j, q in enumerate(qs)], 1)
            proj = np.empty(z.shape, float)
            for j, q in enumerate(qs):
                dd = discs[q]
                w = np.array(dd.w)
                sd = float(np.sqrt(w @ np.array(dd.cov).reshape(2, 2) @ w))
                proj[:, j] = (np.stack([z[:, j].real, z[:, j].imag], -1) @ w - dd.threshold) / sd
        else:
            bits = unpack_bits(arr[reg["key"]], reg["num_bits"])
            out = proj = None
        if bits.shape[1] != len(qs):
            raise R168Refused(f"{spec['tag']}/{name}: {bits.shape[1]} clbits but {len(qs)} measured qubits")
        qc = circs[name]
        pubs.append(dict(index=p["index"], name=name, qubits=qs, bits=bits, proj=proj, outlier=out,
                         ideal=ideal(qc, interest),
                         two_q=any(len(i.qubits) == 2 and i.operation.name not in ("barrier",) for i in qc.data),
                         **layout["pubs"][p["index"]]))
    before = rec.get("snapshot")
    after = json.loads((d / f"after_{spec['job']}.snapshot.json").read_text(encoding="utf-8"))
    return dict(tag=spec["tag"], job=spec["job"], kerneled=kerneled, layout=layout, pubs=pubs, circuits=circs,
                discriminators=({str(q): dd.to_json() for q, dd in discs.items()} if discs else None),
                timestamps=(idx.get("job_metrics") or {}).get("timestamps"), before=before, after=after,
                record_dir=d.relative_to(ROOT).as_posix() if d.is_relative_to(ROOT) else str(d))


# ================================================================== per-job assembly (reads loaded jobs)
def det_blocks(pub: dict, channel: str = "error", select=None):
    """the PUB's deterministic clbits as 8 blocks: [(E (n, Qd) bool, physical qubits, expected values)]"""
    ide = pub["ideal"]
    if ide is None:
        return []
    cols = [j for j, e in enumerate(ide["expected"]) if e >= 0 and (select is None or pub["qubits"][j] in select)]
    if not cols:
        return []
    exp = np.array([ide["expected"][j] for j in cols], bool)
    if channel == "error":
        X = pub["bits"][:, cols] != exp
    elif channel == "outlier":
        if pub["outlier"] is None:
            return []
        X = pub["outlier"][:, cols]
    else:
        raise R168Refused(f"unknown channel {channel!r}")
    qs = [pub["qubits"][j] for j in cols]
    return [(b, qs, exp) for b in split_blocks(X)]


def is_readout(name: str) -> bool:
    tail = name.split("__")[-1]
    return tail.startswith("ro_") or "_ro_" in tail


# ================================================================== the reports (main)
def order_report(job: dict, n_perm: int, rng) -> dict:
    lay = job["layout"]
    out = dict(metadata=dict(every_pub_whole_in_one_span=lay["every_pub_whole_in_one_span"], problems=lay["problems"],
                             spans=lay["spans"], time_order=lay["time_order"], overlaps=lay["overlaps"],
                             gaps_s=lay["gaps_s"]), k_profile=[], block_profile=[], reset=[])
    by_span: dict = {}
    for p in job["pubs"]:
        by_span.setdefault(p["span"], []).append(p)
    rates_dict, s_dict, rates_idx, s_idx = [], [], [], []
    per_qubit: dict = {}
    for si, ps in sorted(by_span.items()):
        blocks = [(E, qs) for p in ps for E, qs, _ in det_blocks(p)]
        ns = {E.shape[0] for E, _ in blocks}
        if blocks and len(ns) == 1 and ns.pop() >= 20:
            r = k_profile_test(blocks, 10, n_perm, rng)
            r.update(span=si, n_blocks=len(blocks))
            out["k_profile"].append(r)
        groups: dict = {}
        for p in ps:
            ide = p["ideal"]
            if ide is None or p["two_q"] or min(ide["expected"]) < 0:
                continue
            groups.setdefault(tuple(p["qubits"]), []).append(p)
        for qs, gp in groups.items():
            if len(gp) < 2:
                continue
            exp = [np.array(g["ideal"]["expected"], bool) for g in gp]
            rates = np.stack([split_blocks(g["bits"] != e).mean(1) for g, e in zip(gp, exp)])
            r = block_profile_test(rates, gp[0]["per_rand"], n_perm, rng)
            r.update(span=si, pubs=[g["name"] for g in gp], n_qubits=len(qs))
            out["block_profile"].append(r)
        # the reset regression under pub-major sweeps, two candidate PUB orders: the slice listing and the PUB index
        for order_name, seq, R_out, S_out in (("slice_listing", sorted(ps, key=lambda p: p["position"]), rates_dict,
                                               s_dict),
                                              ("pub_index", sorted(ps, key=lambda p: p["index"]), rates_idx, s_idx)):
            for pos, p in enumerate(seq):
                ide = p["ideal"]
                if ide is None or len(seq) < 2:
                    continue
                prev = seq[pos - 1]
                cols = [j for j, e in enumerate(ide["expected"]) if e >= 0]
                if not cols:
                    continue
                exp = np.array([ide["expected"][j] for j in cols], bool)
                rate = split_blocks(p["bits"][:, cols] != exp).mean(1)            # (R, Qd)
                s_self = np.array([ide["final_p1"][p["qubits"][j]] for j in cols])
                s_prev = np.array([_final_p1(prev, p["qubits"][j]) for j in cols])
                s = np.tile(s_self, (RANDOMIZATIONS, 1))
                s[0] = s_prev
                R_out.append(rate)
                S_out.append(s)
                if order_name == "slice_listing":
                    for k, j in enumerate(cols):
                        per_qubit.setdefault(p["qubits"][j], ([], []))
                        per_qubit[p["qubits"][j]][0].append(rate[:, k:k + 1])
                        per_qubit[p["qubits"][j]][1].append(s[:, k:k + 1])
    for name, R_, S_ in (("slice_listing", rates_dict, s_dict), ("pub_index", rates_idx, s_idx)):
        r = reset_regression(R_, S_, n_perm, rng)
        r.update(order=name)
        out["reset"].append(r)
    rows = []
    for q, (R_, S_) in sorted(per_qubit.items()):             # the same regression, one qubit at a time
        r = reset_regression(R_, S_, n_perm, rng)
        if r.get("eps") is not None:
            rows.append(dict(qubit=q, eps=r["eps"], sd=r["null_sd"], z=r["z"], p=r["p"],
                             contrasts=r["n_contrasts"]))
    rows.sort(key=lambda r: -(r["z"] if r["z"] is not None else -np.inf))
    out["reset_per_qubit"] = dict(order="slice_listing", rows=rows, n_qubits=len(rows),
                                  n_p_below_001=sum(1 for r in rows if r["p"] is not None and r["p"] <= 0.01),
                                  expected_under_null=0.01 * len(rows))
    return out


def _final_p1(pub: dict, q: int) -> float:
    """the predecessor PUB's final P(1) on qubit q: measured if it measured q, else ideal (Clifford), else unknown"""
    if q in pub["qubits"]:
        return float(pub["bits"][:, pub["qubits"].index(q)].mean())
    ide = pub["ideal"]
    if ide is not None and q in ide["final_p1"]:
        return ide["final_p1"][q]
    return float("nan")


def memory_report(jobs: list, n_perm: int, rng) -> dict:
    out = {}
    for job in jobs:
        sets = {"readout": [p for p in job["pubs"] if is_readout(p["name"])],
                "all_deterministic": [p for p in job["pubs"] if p["ideal"] is not None]}
        if job["tag"].startswith("r165"):
            z = set(json.loads((EV / "r165_dressing_bench/frozen_20261017T174831Z.json").read_text(encoding="utf-8"))
                    ["design"]["groups"]["z"])
            sets["z_group"] = [p for p in job["pubs"] if p["name"].startswith("bench_")]
            sel = {"z_group": z}
        else:
            sel = {}
        jr = {}
        for sname, ps in sets.items():
            row = {}
            for state in (0, 1):
                blocks = []
                for p in ps:
                    for E, qs, exp in det_blocks(p, select=sel.get(sname)):
                        m = exp == bool(state)
                        if m.any():
                            blocks.append((E[:, m], [q for q, k in zip(qs, m) if k]))
                if blocks:
                    row[f"prepared_{state}"] = memory_test(blocks, n_perm, rng)
                    row[f"prepared_{state}_windowed10"] = memory_test(blocks, n_perm, rng, window=10)
            if sname == "readout" and ps and max(len(p["qubits"]) for p in ps) <= 30:   # R155's 146: too wide
                cb = [(E, qs) for p in ps for E, qs, _ in det_blocks(p)]
                if cb:
                    row["cross_qubit"] = cross_memory_test(cb, n_perm, rng)
            jr[sname] = row
        out[job["tag"]] = jr
    return out


def _uncoupled(qubits: list, edges: set) -> list:
    """a greedy maximal set of the qubits with no CZ edge between any two (lowest index first)"""
    nb: dict = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    chosen, blocked = [], set()
    for q in sorted(qubits):
        if q in blocked:
            continue
        chosen.append(q)
        blocked |= {q} | nb.get(q, set())
    return chosen


def burst_report(jobs: list, n_perm: int, rng) -> dict:
    """per job, sets of shots where independent qubits are the null's premise: R165's 21 idle qubits (pairwise
    uncoupled by design, never in a two-qubit gate); R155's whole-chip readout and idle PUBs (146 qubits, and the
    greedy uncoupled subset); R152's and R161's PUBs without a two-qubit gate (all their qubits, and the uncoupled
    subset). PUBs WITH two-qubit gates correlate their qubits' errors by propagation (a CZ copies X into Z) and are
    kept as a labelled reference only."""
    out = {}
    for job in jobs:
        edges = {tuple(int(x) for x in k.split("_")) for k in (job["before"] or {}).get("cz", {})}
        jr = {}
        if job["tag"].startswith("r165"):
            sets = {"idle21": dict(select=None, exclude=set(R165_EDGE), pubs=job["pubs"])}
        elif job["tag"] == "r155":
            chip = [p for p in job["pubs"] if p["ideal"] is not None and len(p["qubits"]) > 100]
            allq = sorted(set(chip[0]["qubits"])) if chip else []
            sets = {"chip146": dict(select=None, exclude=set(), pubs=chip),
                    "chip_uncoupled": dict(select=set(_uncoupled(allq, edges)), exclude=set(), pubs=chip)}
        else:
            no2q = [p for p in job["pubs"] if p["ideal"] is not None and not p["two_q"]]
            allq = sorted({q for p in no2q for q in p["qubits"]})
            sets = {"no2q": dict(select=None, exclude=set(), pubs=no2q),
                    "no2q_uncoupled": dict(select=set(_uncoupled(allq, edges)), exclude=set(), pubs=no2q),
                    "reference_with_2q_gates": dict(select=None, exclude=set(),
                                                    pubs=[p for p in job["pubs"] if p["two_q"]])}
        for sname, sd in sets.items():
            for channel in (("error", "outlier") if job["kerneled"] else ("error",)):
                blocks, spans_H, top = [], {}, []
                for p in sd["pubs"]:
                    sel = sd["select"]
                    if sd["exclude"]:
                        sel = set(p["qubits"]) - sd["exclude"] if sel is None else sel - sd["exclude"]
                    for r, (E, qs, _) in enumerate(det_blocks(p, channel, sel)):
                        if E.shape[1] < 2:
                            continue
                        blocks.append(E)
                        M = E.sum(1)
                        thr = high_threshold(E.mean(0))
                        spans_H.setdefault((p["span"], E.shape[0]), []).append(M >= thr)
                        k = int(np.argmax(M))
                        top.append((int(M[k]), p["name"], r, k, int(p["span"]), len(qs), float(E.mean())))
                if not blocks:
                    continue
                res = dict(qubits_per_shot=sorted({E.shape[1] for E in blocks}),
                           independent=burst_test(blocks, n_perm, rng),
                           windowed10=burst_test(blocks, n_perm, rng, window=10),
                           clustering=cluster_test([np.stack(v) for v in spans_H.values()], n_perm, rng))
                top.sort(reverse=True)
                res["top_shots"] = [dict(multiplicity=t[0], pub=t[1], rand=t[2], k=t[3], span=t[4], qubits=t[5],
                                         block_rate=t[6]) for t in top[:8]]
                jr[f"{sname}/{channel}"] = res
        out[job["tag"]] = jr
    return out


def telegraph_report(jobs: list, n_perm: int, rng) -> dict:
    out = {}
    for job in jobs:
        per: dict = {}
        for p in job["pubs"]:
            for rnd, (E, qs, _) in enumerate(det_blocks(p)):
                for j, q in enumerate(qs):
                    per.setdefault((p["span"], q, E.shape[0]), []).append((p["index"], rnd, E[:, j]))
        rows = []
        for (si, q, n), cols in sorted(per.items()):
            E = np.stack([c for _, _, c in cols])
            if E.sum() < 5 or n < 40:
                continue
            pubs_here = sorted({i for i, _, _ in cols})
            half = None
            if len(pubs_here) >= 2:                     # even / odd position among the span's PUBs: disjoint circuits
                rank = {i: k for k, i in enumerate(pubs_here)}
                half = np.array([rank[i] % 2 == 0 for i, _, _ in cols])
            half_rand = np.array([rnd % 2 == 0 for _, rnd, _ in cols])
            W = max(5, n // 25)
            r = telegraph_test(E, W, n_perm, rng, half=half, half_rand=half_rand)
            r.update(span=si, qubit=q, n_pubs=len(pubs_here))
            rows.append(r)
        out[job["tag"]] = dict(tests=rows, flags=telegraph_flags(rows, n_perm))
        if job["tag"].startswith("r165"):
            out[job["tag"]]["q109"] = q109_report(job, n_perm, rng)
        if job["kerneled"]:
            drift = []
            spans_c: dict = {}
            for p in job["pubs"]:
                ide = p["ideal"]
                if ide is None:
                    continue
                for j, q in enumerate(p["qubits"]):
                    if ide["expected"][j] != 0:
                        continue
                    v = np.where(p["bits"][:, j], np.nan, p["proj"][:, j])
                    per_key = (p["span"], q, p["per_rand"])
                    spans_c.setdefault(per_key, []).extend(split_blocks(v))
            for (si, q, n), cols in sorted(spans_c.items()):
                V = np.stack(cols)
                if n < 40:
                    continue
                r = centre_drift_test(V, max(5, n // 25), min(n_perm, IQ_PERM), rng)
                r.update(span=si, qubit=q, mean=float(np.nanmean(V)), sd=float(np.nanstd(V)))
                drift.append(r)
            across = {}
            for q in sorted({r["qubit"] for r in drift}):
                rs = [r for r in drift if r["qubit"] == q]
                across[str(q)] = dict(spans=[r["span"] for r in rs], means=[r["mean"] for r in rs],
                                      **between_groups([r["mean"] for r in rs],
                                                       [r["sd"] / math.sqrt(max(r["used"], 1)) for r in rs]))
            out[job["tag"]]["iq_centre"] = dict(within_span=drift, across_spans=across)
    return out


TELEGRAPH_STATS = ("disp_sweep", "lr_sweep", "disp_block", "split_half")


def telegraph_flags(rows: list, n_perm: int, z_flag: float = 5.0) -> dict:
    """a (span, qubit) is SWITCHING when its sweep dispersion or segment likelihood ratio beats every one of the
    n_perm replicates (p at its floor) with z > z_flag; a flag replicated in the split halves (z > 3) is a change in
    TIME. Expected false flags under the null: at most (tests x 2) / (n_perm + 1)."""
    floor = 1 / (n_perm + 1)
    flagged = []
    for r in rows:
        hit = [k for k in ("disp_sweep", "lr_sweep") if r[k].get("p") is not None and r[k]["p"] <= floor + 1e-12
               and (r[k].get("z") or 0) > z_flag]
        if hit:
            sh = r.get("split_half") or {}
            sr = r.get("split_rand") or {}
            flagged.append(dict(span=r["span"], qubit=r["qubit"], stats=hit, rate=r["rate"],
                                disp_sweep_ratio=r["disp_sweep"]["ratio"], disp_block_ratio=r["disp_block"]["ratio"],
                                z_disp=r["disp_sweep"]["z"], z_lr=r["lr_sweep"]["z"],
                                segment_sweeps=r["lr_sweep"]["segment_sweeps"],
                                rate_inside=r["lr_sweep"]["rate_inside"], rate_outside=r["lr_sweep"]["rate_outside"],
                                split_half_r=sh.get("value"), split_half_z=sh.get("z"),
                                split_rand_r=sr.get("value"), split_rand_z=sr.get("z"),
                                replicated_in_time=bool((sh.get("z") or 0) > 3 or (sr.get("z") or 0) > 3)))
    return dict(n_tests=len(rows), flagged=flagged, expected_false_at_most=2 * len(rows) * floor, z_flag=z_flag)


def q109_report(job: dict, n_perm: int, rng) -> dict:
    """R165's suspect: q109's error per circuit and randomization (binomial sd) beside the Z group's other qubits,
    and its own telegraph test in each circuit (8 blocks) -- a switch shows as a time segment, a steady excess as a
    level"""
    out = {}
    for p in job["pubs"]:
        if not p["name"].startswith("bench_") and p["name"] not in ("ro_all0", "ro_all1"):
            continue
        blocks = det_blocks(p)
        if not blocks:
            continue
        qs = blocks[0][1]
        if 109 not in qs:
            continue
        j = qs.index(109)
        E = np.stack([b[0][:, j] for b in blocks])
        n = E.shape[1]
        per_rand = E.mean(1)
        others = {q: float(np.mean([b[0][:, qs.index(q)].mean() for b in blocks]))
                  for q in (88, 21, 17, 15, 10, 71) if q in qs}
        row = dict(rate=float(E.mean()), sd=math.sqrt(max(E.mean() * (1 - E.mean()), 1 / E.size) / E.size),
                   per_rand=per_rand.tolist(), z_group_others=others,
                   between_rand_chi2=between_groups(per_rand, np.sqrt(np.maximum(per_rand * (1 - per_rand),
                                                                                 1 / n) / n)))
        if E.sum() >= 5:
            row["telegraph"] = telegraph_test(E, max(5, n // 25), n_perm, rng)
        out[p["name"]] = row
    return out


def _snap_times(snaps: list) -> tuple:
    t0 = min(_iso(s["calibrated_utc"].replace(" ", "T")) for s in snaps)
    return [(_iso(s["calibrated_utc"].replace(" ", "T")) - t0).total_seconds() / 3600 for s in snaps], t0


def drift_report(jobs: list, rng) -> dict:
    """vendor calibration across the jobs' windows (and every window on record), in-circuit readout and Ramsey ratios
    across jobs, q149's coherent offset, and the per-qubit stability table"""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibration_window as CW
    own = []
    for job in jobs:
        for when, s in (("before", job["before"]), ("after", job["after"])):
            own.append((f"{job['tag']}/{when}", s))
    uniq: dict = {}
    for label, s in own:
        uniq.setdefault(s["calibrated_utc"], (label, s))
    same_ts_differs = [lab for lab, s in own if uniq[s["calibrated_utc"]][1] != s and
                       {k: v for k, v in s.items() if k != "calibrated_utc"} !=
                       {k: v for k, v in uniq[s["calibrated_utc"]][1].items() if k != "calibrated_utc"}]
    hist = [(path, snap.to_json()) for path, snap in CW.history("ibm_kingston")]
    allw: dict = {s["calibrated_utc"]: (p, s) for p, s in hist}
    for ts, (lab, s) in uniq.items():
        allw.setdefault(ts, (lab, s))
    out = dict(windows_of_these_jobs=[dict(label=lab, calibrated=ts) for ts, (lab, _) in sorted(
        uniq.items(), key=lambda kv: _iso(kv[0].replace(" ", "T")))],
               n_windows_on_record=len(allw), same_timestamp_different_numbers=same_ts_differs)
    for scope, wins in (("these_jobs", [s for _, s in uniq.values()]), ("all_on_record", [s for _, s in allw.values()])):
        wins = sorted(wins, key=lambda s: _iso(s["calibrated_utc"].replace(" ", "T")))
        th, t0 = _snap_times(wins)
        sc = dict(n_windows=len(wins), span_h=max(th), first=str(t0),
                  windows=[s["calibrated_utc"] for s in wins], _X={}, _E={})
        cz_keys = sorted(set.intersection(*[set(s["cz"]) for s in wins]))
        series = {k: np.array([[np.nan if v is None else v for v in s[k]] for s in wins], float)
                  for k in ("t1", "t2", "p01", "p10", "sx_error")}
        series["cz"] = np.array([[s["cz"][e][0] for e in cz_keys] for s in wins], float)
        for k, X in series.items():
            ep = refresh_epochs(th, X)
            row = dict(n_epochs=ep["n_epochs"], epoch_times_h=ep["times_h"].tolist(),
                       epoch_windows=[wins[i]["calibrated_utc"] for i in ep["index"]], changes=epoch_changes(ep))
            if ep["n_epochs"] >= 3:
                sf = structure_function(ep["times_h"], ep["X"])
                row["fit"] = diffusion_fit(sf, 500, rng)
            if k in ("p01", "p10"):
                ints = np.nanmax(np.abs(ep["X"] * VENDOR_RO_SHOTS - np.round(ep["X"] * VENDOR_RO_SHOTS)))
                bd = binomial_dispersion(ep["X"], VENDOR_RO_SHOTS)
                ok = np.isfinite(bd["p"])
                row["binomial"] = dict(counts_are_integer_of_4096=bool(ints < 1e-6), max_noninteger=float(ints),
                                       qubits=int(ok.sum()), frac_p_below_1e3=float(np.mean(bd["p"][ok] < 1e-3)),
                                       median_excess_rel_sd=float(np.nanmedian(bd["excess_rel_sd"])),
                                       epochs=ep["n_epochs"])
                row["_bd"] = bd
            if k == "cz":
                row["n_edges"] = len(cz_keys)
            sc[k] = row
            sc["_X"][k] = X
            sc["_E"][k] = ep
        out[scope] = sc
    out["per_job_window"] = []
    for job in jobs:
        b, a = job["before"], job["after"]
        out["per_job_window"].append(dict(job=job["tag"], before=b["calibrated_utc"], after=a["calibrated_utc"],
                                          remeasured={k: bool(b[k] != a[k]) for k in
                                                      ("t1", "t2", "p01", "p10", "sx_error", "cz")}))
    out["in_circuit"] = in_circuit_report(jobs)
    out["_vendor"] = vendor_metrics(out["all_on_record"])
    return out


def in_circuit_report(jobs: list) -> dict:
    """per job and qubit: the readout assignment errors from its own readout PUBs (start and, where run, end), and the
    Ramsey ratios r(W) = <Z>(W) / <Z>(0) of its xs family at each wait, with binomial sds"""
    ro, ram = {}, {}
    for job in jobs:
        t = job["timestamps"] or {}
        for p in job["pubs"]:
            tail = p["name"].split("__")[-1]
            ide = p["ideal"]
            if ide is None:
                continue
            if tail in ("ro_all0", "ro_all1", "ro_end0", "ro_end1") or tail.endswith(("_ro_all0", "_ro_all1")):
                prep = int(tail[-1])
                where = "end" if ("end" in tail or p["name"].startswith("info__end")) else "start"
                for j, q in enumerate(p["qubits"]):
                    e = float(np.mean(p["bits"][:, j] != bool(prep)))
                    n = p["bits"].shape[0]
                    ro.setdefault(str(q), []).append(dict(job=job["tag"], when=where, prep=prep, err=e, n=n,
                                                          sd=math.sqrt(max(e * (1 - e), 1 / n) / n),
                                                          vendor_before=job["before"]["p01" if prep == 0 else "p10"][q],
                                                          vendor_after=job["after"]["p01" if prep == 0 else "p10"][q],
                                                          time=t.get("running")))
        xs = [p for p in job["pubs"] if "xs_" in p["name"].split("__")[-1]]
        groups: dict = {}
        for p in xs:
            groups.setdefault(p["name"].split("__")[0] if "__" in p["name"] else "", []).append(p)
        for g, ps in groups.items():
            w0 = next((p for p in ps if p["name"].endswith("xs_w0")), None)
            if w0 is None:
                continue
            for p in ps:
                if p is w0:
                    continue
                qc = job["circuits"][p["name"]]
                wait = _wait_dt(qc)
                for j, q in enumerate(p["qubits"]):
                    if q not in w0["qubits"]:
                        continue
                    z0 = 1 - 2 * float(w0["bits"][:, w0["qubits"].index(q)].mean())
                    zw = 1 - 2 * float(p["bits"][:, j].mean())
                    n0, nw = w0["bits"].shape[0], p["bits"].shape[0]
                    r = zw / z0 if abs(z0) > 1e-9 else float("nan")
                    sd = abs(r) * math.sqrt((1 - zw ** 2) / nw / max(zw ** 2, 1e-12) + (1 - z0 ** 2) / n0 / z0 ** 2)
                    ram.setdefault(str(q), []).append(dict(job=job["tag"], circuit=p["name"], wait_dt=wait.get(q),
                                                           ratio=r, sd=sd, time=t.get("running")))
    changes = []
    for q, rows in ro.items():
        for prep in (0, 1):
            st = [r for r in rows if r["prep"] == prep and r["when"] == "start"]
            jobs_seen = sorted({r["job"] for r in st})
            if len(jobs_seen) < 2:
                continue
            byj = {jt: [r for r in st if r["job"] == jt] for jt in jobs_seen}
            m = [np.mean([r["err"] for r in byj[jt]]) for jt in jobs_seen]
            s = [math.sqrt(sum(r["sd"] ** 2 for r in byj[jt])) / len(byj[jt]) for jt in jobs_seen]
            changes.append(dict(qubit=int(q), prep=prep, jobs=jobs_seen, err=m, sd=s, **between_groups(m, s)))
    ram_changes = []
    for q, rows in ram.items():
        for wait in sorted({r["wait_dt"] for r in rows if r["wait_dt"]}):
            rs = [r for r in rows if r["wait_dt"] == wait and not r["circuit"].endswith("_y")
                  and "echo" not in r["circuit"]]
            if len({r["job"] for r in rs}) >= 2:
                ram_changes.append(dict(qubit=int(q), wait_dt=wait, jobs=[r["job"] for r in rs],
                                        ratio=[r["ratio"] for r in rs], sd=[r["sd"] for r in rs],
                                        **between_groups([r["ratio"] for r in rs], [r["sd"] for r in rs])))
    q149 = {}
    dt = float(jobs[0]["before"]["dt"])
    if any(float(j["before"]["dt"]) != dt for j in jobs):
        raise R168Refused("the jobs' snapshots disagree on dt: a wait in samples is not one duration")
    for jt in sorted({r["job"] for r in ram.get("149", [])}):
        rs = {r["wait_dt"]: r for r in ram["149"] if r["job"] == jt and "echo" not in r["circuit"]
              and not r["circuit"].endswith("_y")}
        if 816 in rs and 1632 in rs:
            q149[jt] = dict(r_W=rs[816]["ratio"], r_W_sd=rs[816]["sd"], r_2W=rs[1632]["ratio"], r_2W_sd=rs[1632]["sd"],
                            two_wait=two_wait_sd(rs[816]["ratio"], rs[816]["sd"], rs[1632]["ratio"], rs[1632]["sd"],
                                                 816 * dt),
                            note="r(t) = C cos(Delta t) with no decay inside 2W; the smallest alias")
        elif 816 in rs:
            r1 = rs[816]["ratio"]
            q149[jt] = dict(r_W=r1, r_W_sd=rs[816]["sd"], note="one wait: contrast and offset are not separable",
                            if_contrast_0984_hz=math.acos(max(-1, min(1, r1 / 0.984))) / (2 * math.pi * 816 * dt),
                            if_offset_49_7khz_contrast=r1 / math.cos(2 * math.pi * 49.7e3 * 816 * dt))
    return dict(readout=ro, readout_changes=changes, ramsey=ram, ramsey_changes=ram_changes, q149_offset=q149)


def _wait_dt(qc) -> dict:
    """per qubit, the total delay (dt) the circuit puts on it"""
    w: dict = {}
    for inst in qc.data:
        if inst.operation.name == "delay":
            for b in inst.qubits:
                q = qc.find_bit(b).index
                w[q] = w.get(q, 0) + int(inst.operation.duration)
    return w


def vendor_metrics(scope: dict) -> dict:
    """per qubit (all of the chip): the vendor's variability over the windows of one scope -- log-sd of T1, T2 and sx
    error, readout dispersion beyond its 4096-shot binomial -- and two levels (1/T1, readout error). Larger = worse."""
    E = scope["_E"]
    m = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in ("t1", "t2", "sx_error"):                 # over the EPOCHS: a repeated window is not a measurement
            m[f"vendor_logsd_{k}"] = np.nanstd(np.log(E[k]["X"]), 0, ddof=1)
        m["vendor_readout_excess_rel_sd"] = np.nanmean(np.stack([scope[k]["_bd"]["excess_rel_sd"]
                                                                 for k in ("p01", "p10")]), 0)
        m["vendor_level_inverse_t1_per_us"] = 1e-6 / np.nanmedian(E["t1"]["X"], 0)
        m["vendor_level_readout_error"] = (np.nanmedian(E["p01"]["X"], 0) + np.nanmedian(E["p10"]["X"], 0)) / 2
    return m


def stability_report(vendor: dict, telegraph: dict | None, in_circuit: dict | None, n_windows: int) -> dict:
    """the per-qubit stability table: the vendor metrics for every qubit and, where these five jobs saw the qubit,
    the in-circuit flags (the largest telegraph z over jobs and spans; the readout and Ramsey change between jobs as
    chi2 per dof). Each metric is a percentile among the qubits that have it; composite = their mean."""
    Q = len(next(iter(vendor.values())))
    m = dict(vendor)
    if telegraph:
        tz = np.full(Q, np.nan)
        for jr in telegraph.values():
            for r in jr.get("tests", []):
                zs = [r[k]["z"] for k in ("disp_sweep", "lr_sweep", "disp_block") if r[k].get("z") is not None]
                if zs:
                    q = int(r["qubit"])
                    tz[q] = max(np.nan_to_num(tz[q], nan=-np.inf), max(zs))
        m["incircuit_telegraph_max_z"] = tz
    if in_circuit:
        for key, src in (("incircuit_readout_change_chi2_per_dof", in_circuit.get("readout_changes", [])),
                         ("incircuit_ramsey_change_chi2_per_dof", in_circuit.get("ramsey_changes", []))):
            v = np.full(Q, np.nan)
            for r in src:
                if r.get("chi2") is not None and r.get("dof"):
                    q = int(r["qubit"])
                    v[q] = max(np.nan_to_num(v[q], nan=-np.inf), r["chi2"] / r["dof"])
            m[key] = v
    with np.errstate(invalid="ignore"):
        veto = dict(vendor_readout_broken=np.nan_to_num(m["vendor_level_readout_error"], nan=1.0) > VETO_READOUT,
                    vendor_t1_unmeasured=~np.isfinite(m["vendor_level_inverse_t1_per_us"]))
        for key, thr in (("incircuit_readout_change_chi2_per_dof", VETO_CHI2_PER_DOF),
                         ("incircuit_ramsey_change_chi2_per_dof", VETO_CHI2_PER_DOF),
                         ("incircuit_telegraph_max_z", VETO_TELEGRAPH_Z)):
            if key in m:
                veto[key.replace("incircuit_", "") + f"_over_{thr:g}"] = np.nan_to_num(m[key], nan=-np.inf) > thr
    rows = stability_table(m, veto)
    return dict(metrics=sorted(m), windows=n_windows, table=rows,
                vetoed=[r["qubit"] for r in rows if r["veto"]],
                clean_order=[r["qubit"] for r in rows if not r["veto"] and r["composite"] is not None],
                note="percentile per metric among the qubits that have it (0 = most stable); composite = their "
                     "mean; a VETO (broken or unmeasured vendor readout/T1, an in-circuit readout or Ramsey change "
                     "between jobs at chi2/dof > 25, a flagged telegraph switch) sorts a qubit after every clean one. "
                     "Vendor metrics cover every qubit; in-circuit ones only the qubits these five jobs measured, so "
                     "a qubit no job saw cannot be vetoed by them")


def pooled_eps(rows: list) -> dict:
    """inverse-variance mean of independent eps estimates (each with its permutation-null sd)"""
    ok = [r for r in rows if r.get("eps") is not None and r.get("null_sd")]
    if not ok:
        return dict(eps=None, sd=None, z=None, n=0)
    w = np.array([1 / r["null_sd"] ** 2 for r in ok])
    e = np.array([r["eps"] for r in ok])
    m, sd = float((w * e).sum() / w.sum()), float(1 / math.sqrt(w.sum()))
    return dict(eps=m, sd=sd, z=m / sd, n=len(ok))


def summary_report(res: dict, n_perm: int = N_PERM) -> dict:
    """the headline numbers of every report, each with its null (cleaned JSON results in, a flat dict out)"""
    out: dict = {}
    n_iq_perm = min(n_perm, IQ_PERM)
    if "order" in res:
        o = res["order"]
        out["order_metadata"] = {t: dict(every_pub_whole_in_one_span=r["metadata"]["every_pub_whole_in_one_span"],
                                         spans=len(r["metadata"]["spans"]), overlaps=r["metadata"]["overlaps"])
                                 for t, r in o.items()}
        out["k_profile"] = [dict(job=t, span=k["span"], blocks=k["n_blocks"], z=k["z"], p=k["p"],
                                 excess_factor=k["excess_factor"], first_shot_z=k["first_shot"]["z"])
                            for t, r in o.items() for k in r["k_profile"]]
        out["reset_eps"] = {order: dict(per_job={t: dict(eps=next(x for x in r["reset"] if x["order"] == order)
                                                         .get("eps"),
                                                         sd=next(x for x in r["reset"] if x["order"] == order)
                                                         .get("null_sd"),
                                                         z=next(x for x in r["reset"] if x["order"] == order)
                                                         .get("z")) for t, r in o.items()},
                                        pooled=pooled_eps([next(x for x in r["reset"] if x["order"] == order)
                                                           for r in o.values()]))
                            for order in ("slice_listing", "pub_index")}
        out["reset_per_qubit"] = {t: dict(top=r["reset_per_qubit"]["rows"][:3],
                                          n_qubits=r["reset_per_qubit"]["n_qubits"],
                                          n_p_below_001=r["reset_per_qubit"]["n_p_below_001"],
                                          expected_under_null=r["reset_per_qubit"]["expected_under_null"])
                                  for t, r in o.items() if "reset_per_qubit" in r}
    if "telegraph" in res:
        t = res["telegraph"]
        flags = [dict(job=j, **f) for j, r in t.items() for f in r["flags"]["flagged"]]
        out["telegraph"] = dict(n_tests=sum(r["flags"]["n_tests"] for r in t.values()), flagged=flags,
                                expected_false_at_most=sum(r["flags"]["expected_false_at_most"] for r in t.values()),
                                replicated_in_time=sum(f["replicated_in_time"] for f in flags))
        sweep_vs_block = [(f["disp_sweep_ratio"], f["disp_block_ratio"]) for f in flags]
        out["telegraph"]["clock_signature"] = dict(
            note="under H_I (k is time) a change raises the sweep dispersion far above the per-block one; under H_C "
                 "(blocks contiguous in time) the sweep series cannot carry it",
            pairs_sweep_block=sweep_vs_block,
            sweep_ratio_exceeds_block_ratio=sum(1 for a, b in sweep_vs_block if a is not None and b is not None
                                                and a > b))
        floor = 1 / (n_iq_perm + 1)
        iq = []
        for j, r in t.items():
            for x in (r.get("iq_centre") or {}).get("within_span", []):
                if x["dispersion"].get("z") and x["dispersion"]["z"] > 5 and x["dispersion"]["p"] <= floor + 1e-12:
                    iq.append(dict(job=j, span=x["span"], qubit=x["qubit"], dispersion_ratio=x["dispersion"]["ratio"],
                                   z=x["dispersion"]["z"], slope_sd_per_1000_sweeps=x["slope_sd_per_1000_sweeps"]
                                   ["value"], slope_z=x["slope_sd_per_1000_sweeps"]["z"]))
        out["iq_centre_flags"] = iq
        out["iq_centre_across_spans_p_below_1e6"] = [
            dict(job=j, qubit=q, chi2=v["chi2"], dof=v["dof"], range_sd=v["range"])
            for j, r in t.items() for q, v in (r.get("iq_centre") or {}).get("across_spans", {}).items()
            if v.get("p") is not None and v["p"] < 1e-6]
        out["q109"] = {j: {c: dict(rate=v["rate"], sd=v["sd"], others=v["z_group_others"],
                                   telegraph_lr_z=(v.get("telegraph") or {}).get("lr_sweep", {}).get("z"),
                                   telegraph_disp_z=(v.get("telegraph") or {}).get("disp_sweep", {}).get("z"))
                           for c, v in r.get("q109", {}).items()} for j, r in t.items() if "q109" in r}
    if "memory" in res:
        mem = {}
        for j, jr in res["memory"].items():
            for s, row in jr.items():
                for k, r in row.items():
                    if k.startswith("prepared"):
                        mem[f"{j}/{s}/{k}"] = dict(pooled_excess=r["pooled"]["excess"], pooled_sd=r["pooled"]["null_sd"],
                                                   pooled_z=r["pooled"]["z"], stouffer_z=r["stouffer"].get("value") and
                                                   r["stouffer"]["value"] / max(r["stouffer"]["keys"], 1) ** 0.5,
                                                   stouffer_perm_z=r["stouffer"].get("z"),
                                                   keys_abs_z_over_3=r["n_abs_z_over_3"], keys=r["n_keys"],
                                                   max_abs_z=r["max_abs_z"])
                    else:
                        mem[f"{j}/{s}/cross"] = dict(max_abs_z=r.get("max_abs_z"), pair=r.get("max_pair"),
                                                     familywise_p=r.get("familywise_p"),
                                                     n_abs_z_over_3=r.get("n_abs_z_over_3"),
                                                     null_mean_n_abs_z_over_3=r.get("n_abs_z_over_3_null_mean"),
                                                     pooled_z=(r.get("pooled_off_diagonal") or {}).get("z"))
        out["memory"] = mem
    if "bursts" in res:
        b = {}
        for j, jr in res["bursts"].items():
            for s, r in jr.items():
                tails = {row["m"]: row for row in r["independent"]["tails"]}
                b[f"{j}/{s}"] = dict(shots=r["independent"]["shots"], qubits=r["qubits_per_shot"],
                                     **{f"M>={m}": dict(obs=tails[m]["observed"], pb=tails[m]["poisson_binomial"],
                                                        z_perm=tails[m]["z_perm"])
                                        for m in (2, 3, 5) if m in tails},
                                     pairs_z=r["independent"]["pairs"]["z"], pairs_windowed_z=r["windowed10"]["pairs"]["z"],
                                     pairs_ratio=(r["independent"]["pairs"]["observed"] /
                                                  r["independent"]["pairs"]["null_mean"]
                                                  if r["independent"]["pairs"].get("null_mean") else None),
                                     max_multiplicity=r["independent"]["max_multiplicity"],
                                     clustering_within_block_z=r["clustering"]["within_block_adjacent"]["z"],
                                     clustering_same_sweep_z=r["clustering"]["same_sweep_across_blocks"]["z"])
        out["bursts"] = b
    if "drift" in res:
        d = res["drift"]
        out["vendor_refresh"] = {k: dict(epochs=d["all_on_record"][k]["n_epochs"],
                                         windows=d["all_on_record"]["n_windows"],
                                         changes=d["all_on_record"][k]["changes"])
                                 for k in ("t1", "t2", "sx_error", "p01", "p10", "cz")}
        out["vendor_readout_binomial"] = {k: d["all_on_record"][k]["binomial"] for k in ("p01", "p10")}
        out["per_job_window"] = d["per_job_window"]
        ic = d["in_circuit"]
        out["incircuit_readout_changes_p_below_1e3"] = sorted(
            [dict(qubit=r["qubit"], prep=r["prep"], jobs=r["jobs"], err=r["err"], chi2=r["chi2"], dof=r["dof"])
             for r in ic["readout_changes"] if r.get("p") is not None and r["p"] < 1e-3],
            key=lambda r: -r["chi2"] / max(r["dof"], 1))
        out["incircuit_ramsey_changes_p_below_1e3"] = sorted(
            [dict(qubit=r["qubit"], wait_dt=r["wait_dt"], jobs=r["jobs"], ratio=r["ratio"], sd=r["sd"], chi2=r["chi2"],
                  dof=r["dof"]) for r in ic["ramsey_changes"] if r.get("p") is not None and r["p"] < 1e-3],
            key=lambda r: -r["chi2"] / max(r["dof"], 1))
        out["q149_offset"] = ic["q149_offset"]
    return out


# ================================================================== main
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write(name: str, obj: dict) -> pathlib.Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    p = OUTDIR / name
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_clean(obj), indent=1) + "\n")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--only", default="", help="comma list of reports: order,memory,bursts,telegraph,drift")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    t0 = time.time()
    jobs = [load_job(s) for s in JOBS]
    print(f"loaded {len(jobs)} jobs in {time.time() - t0:.0f} s", flush=True)
    from oneq.hw import freeze as F
    head = dict(schema=SCHEMA, jobs=[dict(tag=j["tag"], job=j["job"], dir=j["record_dir"], kerneled=j["kerneled"],
                                          timestamps=j["timestamps"]) for j in jobs], n_perm=a.perm, seed=SEED,
                source_sha256={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)): F.source_sha256(p)
                               for p in (pathlib.Path(__file__).resolve(),
                                         ROOT / "tests" / "test_r168_time_structure.py")})
    reports = (("order", lambda r: {j["tag"]: order_report(j, a.perm, r) for j in jobs}),
               ("memory", lambda r: memory_report(jobs, a.perm, r)),
               ("bursts", lambda r: burst_report(jobs, a.perm, r)),
               ("telegraph", lambda r: telegraph_report(jobs, a.perm, r)),
               ("drift", lambda r: drift_report(jobs, r)))
    done = {}
    for name, fn in reports:
        if only and name not in only:
            continue
        t = time.time()
        res = fn(np.random.default_rng(SEED))
        done[name] = res
        extra = dict(vendor_doc=VENDOR_DOC) if name == "order" else {}
        p = _write(f"{name}.json", dict(head, report=name, **extra, result=res))
        print(f"{name}: {time.time() - t:.0f} s -> {_shown(p)}", flush=True)
    cleaned = {k: _clean(v) for k, v in done.items()}
    for name, _ in reports:                          # a report not run now is read back from its last run
        if name not in cleaned and (OUTDIR / f"{name}.json").is_file():
            cleaned[name] = json.loads((OUTDIR / f"{name}.json").read_text(encoding="utf-8"))["result"]
    if "drift" in done:
        st = stability_report(done["drift"]["_vendor"], cleaned.get("telegraph"), done["drift"]["in_circuit"],
                              done["drift"]["all_on_record"]["n_windows"])
        p = _write("stability.json", dict(head, report="stability", result=st))
        print(f"stability -> {_shown(p)}", flush=True)
    p = _write("summary.json", dict(head, report="summary", result=summary_report(cleaned, a.perm)))
    print(f"summary -> {_shown(p)}", flush=True)
    return 0


def _shown(p: pathlib.Path) -> str:
    """a path for display: relative to the repository when it is inside it (relative_to raises otherwise)"""
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


if __name__ == "__main__":
    raise SystemExit(main())
