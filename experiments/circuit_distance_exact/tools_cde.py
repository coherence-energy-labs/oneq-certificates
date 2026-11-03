r"""R198 shared helpers: compile the three C programs, run them, lift and verify witnesses, check the
time-window lemma's preconditions. Imported by the lane's scripts and by tests/test_r198_circuit_distance_exact.py.

*** THE TIME-WINDOW LEMMA. *** Give each mechanism the round of its first detector. If every mechanism touches
detectors in at most two consecutive rounds (P1), interior rounds are exact time-translates of one another (P2),
and the start and end windows do not depend on the number of rounds (P3), then "no undetectable logical error of
weight <= w" for every T = 1..w+2 implies it for every T. Proof: in a minimum-weight solution the set of
mechanism rounds has no gap (if round r were empty between occupied rounds, the parts before and after touch
disjoint detector layers, so each is undetectable on its own and one still flips a logical, with smaller weight);
w mechanisms occupy at most w distinct rounds, so a gap-free set spans at most w rounds, and such a block embeds
unchanged into the (w+2)-round model. `window_preconditions` machine-checks P1-P3 on two consecutive exported
models. What it can NOT check is every T: that models beyond the last checked pair keep the same windows rests on
the builders' time-periodicity (every round is the same loop body), a structural argument -- so "every T" is
MACHINE-CHECKED up to the largest T run and DERIVED beyond it.

*** TIME-SHIFT ANCHORING (rule S). *** With orbits numbered in round order, a solution whose first round is
r0 >= 2 is a time-translate (P2, and P3 for the end window) of a solution starting in round 1 of the
(T - r0 + 1)-round model. So "no solution starting in round 0 or 1" for every T = 1..w+2 covers every T, and each
model needs anchors in two rounds only (`cdist ... MAXROUND=1`; it refuses orbits that are not time-ordered).
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402

C_PROGRAMS = ("cdist", "cdist_prove", "ccheck")


def build_c(out_dir: pathlib.Path, *, liar: bool = False) -> dict[str, pathlib.Path]:
    """gcc -O3 (with OpenMP when available) each program into out_dir; {} when no gcc on PATH."""
    gcc = shutil.which("gcc")
    if gcc is None:
        return {}
    out = {}
    for name in C_PROGRAMS:
        exe = out_dir / (name + ".exe")
        base = [gcc, "-O3", "-o", str(exe), str(HERE / f"{name}.c")]
        r = subprocess.run(base[:2] + ["-fopenmp"] + base[2:], capture_output=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            subprocess.run(base, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        out[name] = exe
    if liar:
        exe = out_dir / "cdist_prove_liar.exe"
        subprocess.run([gcc, "-O2", "-DLIAR", "-o", str(exe), str(HERE / "cdist_prove.c")], check=True,
                       capture_output=True, stdin=subprocess.DEVNULL)
        out["liar"] = exe
    return out


def run(exe: pathlib.Path, *args, threads: int = 4) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    return subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)


def verify_full(circuit_kind: str, basis: str, rounds: int, ids: list[int], l: int = 12) -> bool:
    """Is `ids` (indices of the FULL DEM's error instructions) undetectable and logical?"""
    _, mech, _ = em.mechanisms(bc.build(circuit_kind, basis, rounds, l))
    return _undetectable_logical(mech, ids)


def _undetectable_logical(mech, ids) -> bool:
    det, obs = defaultdict(int), defaultdict(int)
    for j in ids:
        for d in mech[j][0]:
            det[d] ^= 1
        for o in mech[j][1]:
            obs[o] ^= 1
    return not any(det.values()) and any(obs.values())


def _measurement_counts(circuit) -> tuple[int, int]:
    """(measurements, outcomes that are random given the earlier ones) on the noiseless circuit, by stim's tableau."""
    import stim
    sim = stim.TableauSimulator()
    n = rnd = 0
    for inst in circuit.without_noise().flattened():
        if inst.name in ("M", "MX"):
            for t in inst.targets_copy():
                q = t.value
                if inst.name == "MX":
                    sim.h(q)
                rnd += sim.peek_z(q) == 0
                sim.measure(q)
                if inst.name == "MX":
                    sim.h(q)
                n += 1
        elif inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "QUBIT_COORDS"):
            sim.do(stim.Circuit(str(inst)))
    return n, rnd


def _reduce(v: int, basis: dict) -> int:
    while v:
        h = v.bit_length() - 1
        if h not in basis:
            return v
        v ^= basis[h]
    return 0


def deterministic_gap(circuit) -> dict:
    """The deterministic measurement parities that the circuit's detectors and observables do NOT span.

    The noiseless outcomes are uniform on an affine space whose dimension is the number of random outcomes (stim's
    tableau, exact). Shots minus a reference shot span its direction space; 64 shots beyond the dimension span it
    except with probability < 2^-64, and the rank is ASSERTED equal to the tableau's count, so an unlucky draw fails
    loudly instead of passing. A parity is deterministic iff it vanishes on every direction: the null space. Returns
    a completion basis of that null space modulo span(detectors, observables), as record-index lists."""
    n, rnd = _measurement_counts(circuit)
    shots = circuit.without_noise().compile_sampler(seed=0).sample(rnd + 64)
    rows = [int("".join("1" if x else "0" for x in reversed(s ^ shots[0])) or "0", 2) for s in shots[1:]]
    piv: dict[int, int] = {}  # reduced row echelon form keyed by lowest set bit
    for v in rows:
        for c, r in piv.items():
            if v >> c & 1:
                v ^= r
        if v:
            c = (v & -v).bit_length() - 1
            for k in piv:
                if piv[k] >> c & 1:
                    piv[k] ^= v
            piv[c] = v
    assert len(piv) == rnd, f"sampled directions have rank {len(piv)}, the tableau says {rnd}"
    null = [(1 << f) | sum(1 << c for c, r in piv.items() if r >> f & 1) for f in range(n) if f not in piv]
    span: dict[int, int] = {}
    nm, obs = 0, {}
    for inst in circuit.flattened():
        if inst.name in ("M", "MX"):
            nm += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            v = _reduce(sum(1 << (nm + t.value) for t in inst.targets_copy()), span)
            if v:
                span[v.bit_length() - 1] = v
        elif inst.name == "OBSERVABLE_INCLUDE":
            k = int(inst.gate_args_copy()[0])
            obs[k] = obs.get(k, 0) ^ sum(1 << (nm + t.value) for t in inst.targets_copy())
    for v in obs.values():
        v = _reduce(v, span)
        if v:
            span[v.bit_length() - 1] = v
    spanned, gap = len(span), []
    for v in null:
        w = _reduce(v, span)
        if w:
            span[w.bit_length() - 1] = w
            gap.append([k for k in range(n) if v >> k & 1])
    assert len(span) == n - rnd, "detectors or observables are not deterministic parities"
    return dict(measurements=n, random=rnd, deterministic=n - rnd, spanned=spanned, gap=gap)


def witness_survives_gap(c, ids: list[int]) -> dict:
    """Is a full-DEM witness (indices into circuit c's DEM) still undetectable when the deterministic parities the
    detectors omit are added as detectors? Each DEM mechanism merges faults with equal symptoms; with the extra
    detectors they may split. The witness survives iff one fault can be chosen per mechanism so the extra
    detectors cancel (an XOR over a few bits, solved exactly over every reachable value). It can fail in general
    (the toy control), but NOT for this lane's CSS memory experiments: every mechanism there has a pure-X
    (memory-Z) / pure-Z (memory-X) component, which flips no record of the other type, so a meta-check-free choice
    always exists (adversarial pass 2026-10-21) -- a pass here adds no evidence beyond the full-DEM check."""
    import stim
    gap = deterministic_gap(c)["gap"]
    _, old, _ = em.mechanisms(c)
    nd, nm = c.num_detectors, c.num_measurements
    c2 = c.copy()
    for g in gap:
        c2.append("DETECTOR", [stim.target_rec(r - nm) for r in g])
    _, new, _ = em.mechanisms(c2)
    extra = defaultdict(set)
    for dets, obs in new:
        extra[(tuple(d for d in dets if d < nd), obs)].add(sum(1 << (d - nd) for d in dets if d >= nd))
    reach = {0}
    for j in ids:
        reach = {r ^ p for r in reach for p in extra[old[j]]}
    return dict(gap=len(gap), survives=_undetectable_logical(old, ids) and 0 in reach)


def lift(circuit_kind: str, basis: str, rounds: int, projected_ids: list[int], l: int = 12) -> list[int] | None:
    """A projected witness -> a FULL-DEM witness of the same weight, or None if no choice is undetectable.

    Each projected mechanism has a set of full preimages (same kept detectors and observables, different dropped
    detectors). Choose one per projected mechanism so that the FULL syndrome vanishes -- an XOR constraint, solved by
    meet in the middle: tabulate the full syndromes of every choice for the first half of the groups, then look up
    each choice for the second half (product of a half instead of the whole product). The result is re-verified."""
    import itertools
    _, mech, coords = em.mechanisms(bc.build(circuit_kind, basis, rounds, l))
    keep_type = 1 if basis == "Z" else 0
    keep = sorted(d for d, v in coords.items() if int(v[1]) == keep_type)
    new_id = {d: i for i, d in enumerate(keep)}
    groups, order = {}, []
    for j, (dets, obs) in enumerate(mech):
        pd = tuple(sorted(new_id[d] for d in dets if d in new_id))
        if not pd and not obs:
            continue
        key = (pd, obs)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(j)
    cand = [groups[order[r]] for r in projected_ids]
    syn = {j: frozenset(mech[j][0]) for g in cand for j in g}
    half = len(cand) // 2

    def sums(gs):
        for choice in itertools.product(*gs):
            s = frozenset()
            for j in choice:
                s = s ^ syn[j]
            yield s, choice
    table: dict[frozenset, list] = {}
    for s, choice in sums(cand[:half]):
        table.setdefault(s, []).append(choice)
    for s, right in sums(cand[half:]):
        for left in table.get(s, ()):
            choice = left + right
            if len(set(choice)) == len(choice) and _undetectable_logical(mech, choice):
                return list(choice)
    return None


def read_model(path: pathlib.Path, *, dims: bool = False):
    with open(path) as f:
        nd, nm, no, norb, gl, gm = map(int, f.readline().split())
        mech = []
        for _ in range(nm):
            p = list(map(int, f.readline().split()))
            mech.append((p[1:1 + p[0]], p[1 + p[0]]))
        f.readline()
        det = [tuple(map(int, f.readline().split())) for _ in range(nd)]
    out = (mech, det, max(r for _, _, r in det))
    return out + ((gl, gm),) if dims else out


def _shapes(mech, det, drift=(0, 0), dims=(1, 1)):
    """Mechanisms grouped by start round r0, each written relative to its start: detector (s, type, r) becomes
    (s - r0 * drift, type, r - r0). drift is the spatial translation that carries one round's gates onto the next
    round's -- (0, 0) when every round is the same, (1, 0) for a morphing circuit whose two contraction rounds
    differ by the translation x -- so "time-translate" means the circuit's own space-time symmetry."""
    gl, gm = dims

    def back(s, r0):
        i, j = divmod(s, gm)
        return ((i - r0 * drift[0]) % gl) * gm + (j - r0 * drift[1]) % gm

    by_start, span_ok = defaultdict(set), True
    for dets, obs in mech:
        rs = [det[d][2] for d in dets]
        r0 = min(rs)
        span_ok &= max(rs) - r0 <= 1
        by_start[r0].add((tuple(sorted((back(det[d][0], r0), det[d][1], det[d][2] - r0) for d in dets)), obs))
    return by_start, span_ok


def _affine_consistent(rows) -> bool:
    """rows of (input bits, output bits): is there a GF(2) map with output = F(input) for every row? Equivalently,
    no combination of rows has zero input and nonzero output."""
    basis: dict[int, tuple[int, int]] = {}
    for x, y in rows:
        while x:
            h = x.bit_length() - 1
            if h not in basis:
                basis[h] = (x, y)
                break
            x, y = x ^ basis[h][0], y ^ basis[h][1]
        else:
            if y:
                return False
    return True


def _shift_obs_affine(ma, da, mb, db, drift, dims) -> bool:
    """With a nonzero drift the space-time translation tau carries the logical representatives to OTHER
    representatives, so observable masks are not equal across windows -- they must be AFFINE: pair every mechanism
    of the (T+1)-round model starting at r0 >= 2 with its translate in the T-round model (start r0 - 1, equal
    detector shape) and require obs_T(tau m) = M obs_T+1(m) + K syn_T+1(m) consistent over all of them, and the
    same in reverse. Then an undetectable set is logical iff its translate is (tau maps detectors bijectively on
    those rounds), which is what the time-shift lemma uses."""
    gl, gm = dims
    nd_b = len(db)

    def key(dets, det, r0):
        i_j = [divmod(det[d][0], gm) for d in dets]
        return tuple(sorted((((i - r0 * drift[0]) % gl) * gm + (j - r0 * drift[1]) % gm, det[d][1], det[d][2] - r0)
                            for (i, j), d in zip(i_j, dets)))

    in_a = {}
    for dets, obs in ma:
        r0 = min(da[d][2] for d in dets)
        if r0 >= 1:
            in_a[(r0, key(dets, da, r0))] = (dets, obs)
    fwd, rev = [], []
    for dets, obs in mb:
        r0 = min(db[d][2] for d in dets)
        if r0 < 2:
            continue
        hit = in_a.get((r0 - 1, key(dets, db, r0)))
        if hit is None:
            return False
        syn_b = sum(1 << d for d in dets)
        syn_a = sum(1 << d for d in hit[0])
        fwd.append((obs | syn_b << 32, hit[1]))
        rev.append((hit[1] | syn_a << 32, obs))
    assert nd_b < 1 << 20
    return _affine_consistent(fwd) and _affine_consistent(rev)


def window_preconditions(model_t: pathlib.Path, model_t1: pathlib.Path, drift=(0, 0)) -> dict:
    """P1-P3 on the T-round and (T+1)-round models, time translation composed with the spatial `drift`. With a
    nonzero drift the windows are compared on their detector structure and the observables are required to
    transform affinely (`_shift_obs_affine`); with none, windows must match observables and all (unchanged)."""
    ma, da, Ta, dims = read_model(model_t, dims=True)
    mb, db, Tb = read_model(model_t1)
    if Tb != Ta + 1:
        raise ValueError("pass consecutive round counts")
    if drift != (0, 0):
        obs_ok = _shift_obs_affine(ma, da, mb, db, drift, dims)
        ma, mb = [(d, 0) for d, _ in ma], [(d, 0) for d, _ in mb]
    else:
        obs_ok = True
    sa, oka = _shapes(ma, da, drift, dims)
    sb, okb = _shapes(mb, db, drift, dims)
    interior = list(range(1, Tb - 1))
    ref = sb[interior[0]] if interior else set()
    p2 = all(sb[j] == ref for j in interior) and all(sa[j] == ref for j in range(1, Ta - 1))
    p3_start, p3_final, p3_last = sa[0] == sb[0], sa[Ta] == sb[Tb], sa[Ta - 1] == sb[Tb - 1]
    # A shift into the 1-round model moves only solutions that live entirely in the final layer (r0 = T), so for
    # the pair (1, 2) time-shift anchoring needs the final-layer window alone: in a 1-round model the last round IS
    # the start round, and its window legitimately differs from a longer model's last (interior-like) round.
    shift_ok = bool(oka and okb and p2 and p3_final and (Ta == 1 or p3_last) and obs_ok)
    out = dict(T=Ta, P1=bool(oka and okb), P2=bool(p2), P3=bool(p3_start and p3_final and p3_last),
               P3_start=bool(p3_start), P3_final=bool(p3_final), P3_last=bool(p3_last),
               interior_shapes=len(ref), holds=bool(oka and okb and p2 and p3_start and p3_final and p3_last and obs_ok),
               time_shift_ok=shift_ok)
    if drift != (0, 0):
        out.update(drift=list(drift), obs_affine=bool(obs_ok))
    return out


def sat_none(model: pathlib.Path, w: int, seconds: float = 60.0):
    """Independent cross-check with a different program and solver (CDCL via pysat): True = UNSAT at <= w
    (no anchored solution), False = SAT (witness re-verified), None = undecided in time."""
    import threading
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    from pysat.solvers import Solver
    with open(model) as f:
        nd, nm, no, norb, _, _ = map(int, f.readline().split())
        mech = []
        for _ in range(nm):
            p = list(map(int, f.readline().split()))
            mech.append((p[1:1 + p[0]], [o for o in range(no) if p[1 + p[0]] >> o & 1]))
        anchors = list(map(int, f.readline().split()))
    pool = IDPool()
    e = [pool.id(("e", j)) for j in range(nm)]
    cls = []

    def xor_of(lits):
        acc = lits[0]
        for x in lits[1:]:
            a = pool.id(("x", pool.top + 1))
            cls.extend([[-a, acc, x], [-a, -acc, -x], [a, -acc, x], [a, acc, -x]])
            acc = a
        return acc
    det_inc, obs_inc = [[] for _ in range(nd)], [[] for _ in range(no)]
    for j, (dets, obs) in enumerate(mech):
        for d in dets:
            det_inc[d].append(e[j])
        for o in obs:
            obs_inc[o].append(e[j])
    for d in range(nd):
        if det_inc[d]:
            cls.append([-xor_of(det_inc[d])])
    cls.append([xor_of(obs_inc[o]) for o in range(no) if obs_inc[o]])
    cls.append([e[j] for j in anchors])
    cls.extend(CardEnc.atmost(lits=e, bound=w, vpool=pool, encoding=EncType.seqcounter).clauses)
    with Solver(name="g4", bootstrap_with=cls) as s:
        timer = threading.Timer(seconds, s.interrupt)
        timer.start()
        res = s.solve_limited(expect_interrupt=True)
        timer.cancel()
        if res is None:
            return None
        if res:
            model_set = set(v for v in s.get_model() if v > 0)
            chosen = [j for j in range(nm) if e[j] in model_set]
            syn, lg = np.zeros(nd, dtype=np.uint8), np.zeros(no, dtype=np.uint8)
            for j in chosen:
                for d in mech[j][0]:
                    syn[d] ^= 1
                for o in mech[j][1]:
                    lg[o] ^= 1
            assert not syn.any() and lg.any(), "SAT witness failed re-verification"
            return False
        return True
