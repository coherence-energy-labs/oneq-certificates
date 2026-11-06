r"""Replay the retained epsilon receipts of paper 1's rerun, completely, two ways.

For every configuration under evidence/epsilon_rerun/ this tool

  1. binds the run: summary.json names the instance, whose edge list must hash
     to its recorded edges_sha256; chunks.jsonl must list every chunk of the
     declared plan exactly once -- a missing, duplicated or extra chunk FAILS;
  2. binds every chunk: the receipt file must exist and its decompressed bytes
     must hash to the index; the receipts it holds must be the chunk's
     nontrivial shots, each once, in order, with the chunk's status histogram;
  3. re-decides every receipt with TWO checkers that share no code:
       * oneq.matching_cert_epsilon.check_epsilon (Fraction arithmetic, set-centric
         load accumulation), and
       * the verifier below, written separately from Lemma 1: every double is an
         integer times a power of two, so the whole instance is scaled to
         integers and theta is chosen by integer cross-multiplication, walking
         EDGES rather than sets;
     both must reproduce the recorded status, epsilon and lower bound exactly;
  4. reproduces summary.json's totals and largest accepted epsilon from the
     receipts themselves.

With --battery it also attacks accepted receipts at the acceptance boundary:
one correction edge's weight is raised to the smallest double that pushes the
proven slack past eps_max, which both checkers must refuse, and to a value
that keeps it inside, which both must accept; and each receipt is replayed
against a different configuration's graph, which must not be accepted.

    python tools/replay_epsilon_receipts.py --jobs 8               # everything
    python tools/replay_epsilon_receipts.py --config sim_d5_r25 --sample 20
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pathlib
import random
import re
import sys
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.matching_cert import MatchingCertificate          # noqa: E402
from oneq.matching_cert_epsilon import INVALID_INPUT, EpsilonVerdict, check_epsilon   # noqa: E402

INDEX = ROOT / "evidence" / "epsilon_rerun"
B = -1
ACCEPT = ("EXACT_OPTIMAL", "EPSILON_OPTIMAL")


class ReplayError(Exception):
    pass


# --------------------------------------------------------------- verifier B
def _dyadic(x: float) -> tuple[int, int]:
    """x = m * 2**e with integer m (finite doubles only)."""
    if not math.isfinite(x):
        raise ValueError("non-finite")
    m, e = math.frexp(x)
    m_int = int(m * (1 << 53))
    return m_int, e - 53


def _is_vertex(x) -> bool:
    """An integer vertex id: never a bool (True is not detector 1), never a float, never a string."""
    return type(x) is int and (x >= 0 or x == B)


def verify_b(edges, fired, correction, duals, eps_max: Fraction):
    """(status, epsilon, lower_bound) by integer arithmetic. Independent of check_epsilon.

    THE INPUT CONTRACT (one policy for every implementation; 2026-11-03, after the JavaScript fuzzer showed this
    verifier accepting a self-loop and `true` as a detector and raising on a non-finite weight): vertices are
    integers (the boundary is -1); the graph is simple -- no self-loop, no edge listed twice in either orientation;
    weights and packing values are finite and non-negative; every packing set and every fired detector is a vertex of
    the graph; anything else is INVALID_INPUT, never coerced and never an exception.
    """
    raw = []
    seen = set()
    V = {B}
    for e in edges:
        if len(e) != 3:
            return "INVALID_INPUT", None, None
        u, v, w = e
        if not (_is_vertex(u) and _is_vertex(v)) or u == v:
            return "INVALID_INPUT", None, None
        if type(w) not in (int, float) or not math.isfinite(w) or w < 0:
            return "INVALID_INPUT", None, None
        k = (min(u, v), max(u, v))
        if k in seen:
            return "INVALID_INPUT", None, None
        seen.add(k)
        V.update(k)
        raw.append((k[0], k[1], float(w)))
    pairs = []
    for S, z in duals:
        if not all(_is_vertex(x) and x in V for x in S):
            return "INVALID_INPUT", None, None
        if type(z) not in (int, float) or not math.isfinite(z) or z < 0:
            return "INVALID_INPUT", None, None
        pairs.append((frozenset(S), float(z)))
    if len({S for S, _ in pairs}) != len(pairs):
        return "INVALID_INPUT", None, None
    if not all(_is_vertex(x) and x != B for x in fired):
        return "INVALID_INPUT", None, None
    if not all(len(e) == 2 and _is_vertex(e[0]) and _is_vertex(e[1]) for e in correction):
        return "INVALID_INPUT", None, None
    # one common power of two for every weight and every dual
    exps = [_dyadic(w)[1] for *_uv, w in raw if w] + [_dyadic(z)[1] for _S, z in pairs if z]
    shift = -min(exps) if exps and min(exps) < 0 else 0

    def scaled(x):
        m, e = _dyadic(x)
        return m << (e + shift) if x else 0

    W = {}
    for u, v, w in raw:
        s = scaled(w)
        W[(u, v)] = min(W.get((u, v), s), s)
    T = set(fired)
    if len(T) != len(fired):
        return "INVALID_INPUT", None, None
    parity = Counter()
    P = 0
    used = set()
    for u, v in correction:
        k = (min(u, v), max(u, v))
        if k not in W or k in used:
            return "INVALID_INPUT", None, None
        used.add(k)
        P += W[k]
        parity[u] ^= 1
        parity[v] ^= 1
    if {x for x, bit in parity.items() if bit and x != B} != T:
        return "INVALID_INPUT", None, None
    Tp = T | ({B} if len(T) % 2 else set())
    Z = 0
    live = []
    for S, z in pairs:
        if len(S & Tp) % 2 == 0:
            return "INVALID_INPUT", None, None
        zi = scaled(z)
        if zi:
            Z += zi
            live.append((S, zi))
    # theta = min(1, min w/L): num/den, compared by cross-multiplication
    member = {}
    for i, (S, _zi) in enumerate(live):
        for x in S:
            member.setdefault(x, set()).add(i)
    tn, td = 1, 1
    for (u, v), w in W.items():
        if u not in member and v not in member:
            continue
        L = sum(live[i][1] for i in member.get(u, set()) ^ member.get(v, set()))
        if L > w and w * td < tn * L:
            tn, td = w, L
    scale = 1 << shift
    lb = Fraction(tn * Z, td * scale)
    eps = Fraction(P, scale) - lb
    if eps < 0:
        return "VERIFIER_FAILURE", eps, lb
    if eps == 0:
        return "EXACT_OPTIMAL", eps, lb
    return ("EPSILON_OPTIMAL" if eps <= eps_max else "NOT_PROVEN"), eps, lb


# --------------------------------------------------------------- replay
_EDGES: dict = {}


class ReceiptFormatError(ValueError):
    """A receipt or instance field outside the declared domain. Refused, never coerced."""


#: A hexadecimal floating literal as float.hex() writes it, or an exact non-canonical spelling of one.
_HEX = re.compile(r"([+-]?)0[xX]([0-9a-fA-F]+)(?:\.([0-9a-fA-F]*))?[pP]([+-]?[0-9]{1,5})")


def strict_hex(s, what="value") -> float:
    """The binary64 value a hex literal names EXACTLY, or ReceiptFormatError.

    LOSSLESS, after the JavaScript fuzzer (2026-11-03) showed `float.fromhex` reading '0.5' as hexadecimal 0x0.5 =
    0.3125 and silently rounding a literal finer than binary64: both Python verifiers then decided a different number
    from the one the receipt names. The literal must carry 0x and an exponent, and its exact value must equal the
    double it rounds to; non-finite values are not literals of this form and are refused here.
    """
    m = _HEX.fullmatch(s) if isinstance(s, str) else None
    if m is None:
        raise ReceiptFormatError(f"{what}: {s!r} is not a hexadecimal floating literal")
    sign, ip, fp, ex = m.group(1), m.group(2), m.group(3) or "", int(m.group(4))
    exact = Fraction(int(ip + fp, 16), 16 ** len(fp)) * (Fraction(2) ** ex)
    if sign == "-":
        exact = -exact
    try:
        x = float.fromhex(s)
    except OverflowError:
        raise ReceiptFormatError(f"{what}: {s!r} overflows binary64") from None
    if not math.isfinite(x) or Fraction(x) != exact:
        raise ReceiptFormatError(f"{what}: {s!r} is not exactly a binary64 value")
    return x


def strict_vertex(x, what="vertex"):
    if not _is_vertex(x):
        raise ReceiptFormatError(f"{what}: {x!r} is not an integer vertex id")
    return x


def _instance(name):
    p = ROOT / "evidence" / "instances" / f"matching_{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    body = d["edges"]
    if hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest() != d["edges_sha256"]:
        raise ReplayError(f"{p.name}: edge list does not hash to edges_sha256")
    try:
        return [(strict_vertex(u), strict_vertex(v), strict_hex(w, "weight")) for u, v, w in body], d["edges_sha256"]
    except ReceiptFormatError as exc:
        raise ReplayError(f"{p.name}: {exc}") from None


def _edges(name):
    if name not in _EDGES:
        _EDGES[name] = _instance(name)[0]
    return _EDGES[name]


def substitute_for(name, merged):
    """The configuration whose graph the substitution attack replays `name`'s receipts against: the first other one
    whose instance is a DIFFERENT graph. Two configurations can share a graph -- fresh_sim_d5_r25 re-samples
    sim_d5_r25's circuit -- and a receipt replayed against its own graph is no attack: when fresh_sim_d5_r25 was
    merged (2026-11-02) it became the first name in sorted order, and the 2026-11-03 strict replay of sim_d5_r25
    "accepted" 4,000 substituted graphs that were the same graph. None when no other graph exists."""
    own = _instance(name)[1]
    return next((m for m in merged if m != name and _instance(m)[1] != own), None)


def _receipt_duals(rec):
    """[(members, value)] read losslessly from a receipt; ReceiptFormatError on anything outside the domain."""
    out, seen = [], set()
    for item in rec["duals"]:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], list):
            raise ReceiptFormatError(f"dual entry {item!r} is not [members, value]")
        members = tuple(strict_vertex(x, "dual set member") for x in item[0])
        # A packing is a MAP from sets to values; a list encodes one only while its keys are distinct, so a repeated
        # set is refused here, before check_epsilon (which receives a dict) could see only one of its values.
        key = frozenset(members)
        if key in seen:
            raise ReceiptFormatError(f"dual set {sorted(key)} listed twice")
        seen.add(key)
        out.append((members, strict_hex(item[1], "dual value")))
    return out


def _receipt_correction(rec):
    corr = []
    for e in rec["correction"]:
        if not isinstance(e, list) or len(e) != 2:
            raise ReceiptFormatError(f"correction entry {e!r} is not an edge [u, v]")
        corr.append((strict_vertex(e[0], "correction endpoint"), strict_vertex(e[1], "correction endpoint")))
    return corr


def _receipt_fired(rec):
    if not isinstance(rec["fired"], list):
        raise ReceiptFormatError("fired is not a list")
    return [strict_vertex(x, "fired detector") for x in rec["fired"]]


def _cert(rec, duals):
    return MatchingCertificate(matched_edges=tuple(_receipt_correction(rec)), node_potentials={},
                               blossom_duals={frozenset(S): z for S, z in duals})


def _invalid_pair(reason, eps_max):
    return EpsilonVerdict(INVALID_INPUT, reason, Fraction(eps_max)), (INVALID_INPUT, None, None)


def decide_receipt(edges, rec, eps_max):
    """Both verdicts on one receipt as the replay reads it: a field outside the domain is INVALID_INPUT for both."""
    try:
        duals = _receipt_duals(rec)
    except ReceiptFormatError as exc:
        return _invalid_pair(str(exc), eps_max)
    return _decide_both(edges, rec, duals, eps_max)


def _decide_both(edges, rec, duals, eps_max, corr=None):
    try:
        fired = _receipt_fired(rec)
        corr = corr or _receipt_correction(rec)
        cert = _cert(rec, duals)
    except ReceiptFormatError as exc:
        return _invalid_pair(str(exc), eps_max)
    a = check_epsilon(edges=edges, syndrome=fired, cert=cert, eps_max=eps_max)
    b = verify_b(edges, fired, corr, duals, eps_max)
    return a, b


_BLIND: dict = {}


def parity_blind_verifiers():
    """(check_epsilon, verify_b) with ONLY their T-oddness test removed: what a checker that forgot the parity rule
    would decide. Built from the real sources by deleting exactly one statement each (asserted to occur once), so they
    cannot drift from the real verifiers in anything else. Used only to show that an attack is REAL -- that it would be
    accepted but for the parity test -- never to decide a receipt.

    Why they exist (paper 1, final referee, 2026-11-03): the battery used to call a parity forgery "real" when its
    parity-blind bound was at least the heavier correction's weight. The adversary's linear program maximises sum z
    with no cap, so that bound overshot the weight by hundreds to thousands of units, and a parity-blind checker
    returns VERIFIER_FAILURE on such a document, not acceptance: the counter had no upper end and counted forgeries
    that would fool nothing."""
    if not _BLIND:
        import inspect
        import types
        src = (ROOT / "src" / "oneq" / "matching_cert_epsilon.py").read_text(encoding="utf-8")
        needle = 'raise ValueError("dual set is not odd against the syndrome")'
        if src.count(needle) != 1:
            raise ReplayError("check_epsilon's parity test is not the one statement the parity-blind copy removes")
        mod = types.ModuleType("matching_cert_epsilon_parity_blind")
        sys.modules[mod.__name__] = mod          # dataclasses resolve their module by name
        exec(compile(src.replace(needle, "pass  # PARITY TEST REMOVED: a parity-blind copy"),
                     "matching_cert_epsilon_parity_blind", "exec"), mod.__dict__)
        vb = inspect.getsource(verify_b)
        needle_b = ("        if len(S & Tp) % 2 == 0:\n"
                    "            return \"INVALID_INPUT\", None, None\n")
        if vb.count(needle_b) != 1:
            raise ReplayError("verify_b's parity test is not the one statement the parity-blind copy removes")
        ns = dict(globals())
        exec(compile(vb.replace(needle_b, ""), "verify_b_parity_blind", "exec"), ns)
        _BLIND["check_epsilon"], _BLIND["verify_b"] = mod.check_epsilon, ns["verify_b"]
    return _BLIND["check_epsilon"], _BLIND["verify_b"]


def calibrated_parity_forgery(edges, fired, cert, heavy_w, eps_max, rounds=8):
    """Scale the parity adversary's packing so a parity-blind checker ACCEPTS it, or None if no scaling does.

    oneq.forgeries.parity_forgery maximises sum z under the edge constraints and nothing else, so its parity-blind
    bound overshoots the heavier correction's weight. A bound in [w(J') - eps_max, w(J')] is what a parity-blind
    checker accepts as eps-optimal; shrinking every value by one factor keeps the packing feasible and moves the bound
    down proportionally, so the values are scaled by w(J')/bound (1 - 2^-40) until the parity-blind copy of
    check_epsilon, RUN on the document, accepts it."""
    from oneq.forgeries import _blind_bound, _key
    blind_check, _vb = parity_blind_verifiers()
    hw = Fraction(heavy_w)
    W = {_key((u, v)): w for u, v, w in edges}
    z = dict(cert.blossom_duals)
    for _ in range(rounds):
        c = MatchingCertificate(cert.matched_edges, {}, z)
        verdict = blind_check(edges=edges, syndrome=fired, cert=c, eps_max=eps_max)
        if verdict.status in ACCEPT:
            return c
        b = _blind_bound(W, z)
        if b <= 0 or b < hw - eps_max:
            return None                  # the packing cannot reach the heavier weight: scaling up would overload it
        z = {S: val * (float(hw / b) * (1 - 2 ** -40)) for S, val in z.items()}
    return None


def _forgery_battery(edges, rec, duals, eps_max, tally):
    """The audited forgery battery (oneq.forgeries, 2026-11-03) on one accepted receipt, decided by BOTH verifiers.

    Records, per class, which verdict refused it -- NOT_PROVEN means the attack reached the weak-duality bound,
    INVALID_INPUT that the input domain refused it -- so a battery whose attacks never reach the bound is visible.
    """
    from oneq.forgeries import STRICTLY_SUBOPTIMAL, _key, assert_all_false, forgeries, parity_forgery
    W = {}
    for u, v, w in edges:
        W[_key((u, v))] = w
    corr = tuple(_key(e) for e in _receipt_correction(rec))
    fired = _receipt_fired(rec)
    z = {frozenset(S): v for S, v in duals}
    primal = sum(W[k] for k in corr)
    battery = forgeries(W, fired, corr, z, primal)
    some = next(iter(z))
    battery += [("N_nan_dual", MatchingCertificate(corr, {}, {**z, some: math.nan}), "non-finite"),
                ("N_inf_dual", MatchingCertificate(corr, {}, {**z, some: math.inf}), "non-finite")]
    # The T'-oddness adversary (oneq.forgeries.parity_forgery, the linear-programming search that found the R177
    # defect), on every receipt the battery attacks (2026-11-03; before, only tests/test_forgery_battery_edge_sets.py
    # ran it, on 4 receipts per configuration). Its claim -- a strictly heavier valid correction -- is audited false
    # with the rest. Its packing is scaled until a parity-blind copy of check_epsilon accepts it, and whether it is
    # real is decided by RUNNING the parity-blind copies of both verifiers on it, not by comparing two numbers.
    pf = parity_forgery(W, fired, corr, z)
    if pf is None:
        tally["diag:no_disjoint_cycle"] += 1
    else:
        pcert, heavy_w, _uncapped_bound = pf
        tally["diag:attempted"] += 1
        cal = calibrated_parity_forgery(edges, fired, pcert, heavy_w, eps_max)
        if cal is None:
            tally["diag:no_scaling_fools_a_parity_blind_checker"] += 1
        else:
            battery.append(("P_parity_not_T_prime_odd", cal, STRICTLY_SUBOPTIMAL))
            blind_check, blind_b = parity_blind_verifiers()
            ab = blind_check(edges=edges, syndrome=fired, cert=cal, eps_max=eps_max)
            bb = blind_b(edges, fired, [tuple(e) for e in cal.matched_edges],
                         [(tuple(sorted(S)), val) for S, val in cal.blossom_duals.items()], eps_max)
            tally["diag:accepted_by_parity_blind_check_epsilon"] += ab.status in ACCEPT
            tally["diag:accepted_by_parity_blind_verify_b"] += bb[0] in ACCEPT
            tally["diag:accepted_by_both_parity_blind_copies"] += ab.status in ACCEPT and bb[0] in ACCEPT
    if assert_all_false(W, fired, primal, [b for b in battery if b[2] != "non-finite"]):
        raise ReplayError(f"shot {rec['shot']}: the battery produced an attack whose claim is not false")
    for nm, fc, _why in battery:
        a = check_epsilon(edges=edges, syndrome=fired, cert=fc, eps_max=eps_max)
        listed = [(tuple(sorted(S)), val) for S, val in fc.blossom_duals.items()]
        b = verify_b(edges, fired, [tuple(e) for e in fc.matched_edges], listed, eps_max)
        if a.status in ACCEPT or b[0] in ACCEPT:
            raise ReplayError(f"shot {rec['shot']}: forgery {nm} ACCEPTED (check_epsilon {a.status}, B {b[0]})")
        if a.status != b[0]:
            raise ReplayError(f"shot {rec['shot']}: the verifiers refuse forgery {nm} differently "
                              f"({a.status} vs {b[0]})")
        tally[f"{nm}:{a.status}"] += 1


def verify_chunk(job):
    """Re-decide one chunk's receipts. job = (name, seed, index_row, eps_max, battery, other_name[, forgeries])."""
    name, seed, r, eps_max_s, battery, other_name, *rest = job
    n_forg = rest[0] if rest else 0
    eps_max = Fraction(eps_max_s)
    edges = _edges(name)
    other = _edges(other_name) if other_name else None
    body = gzip.decompress((ROOT / r["receipts_file"]).read_bytes())
    if hashlib.sha256(body).hexdigest() != r["receipts_sha256"]:
        raise ReplayError(f"{name} chunk {seed}: receipt file changed during replay")
    recs = [json.loads(x) for x in body.decode().splitlines() if x]
    shots = [x["shot"] for x in recs]
    if shots != sorted(set(shots)):
        raise ReplayError(f"{name} chunk {seed}: shots duplicated or out of order")
    hist = Counter(x["status"] for x in recs)
    want = {k: v for k, v in r["status"].items() if k != "TRIVIAL"}
    if dict(hist) != want:
        raise ReplayError(f"{name} chunk {seed}: receipts {dict(hist)} != index {want}")
    if len(recs) + r["status"].get("TRIVIAL", 0) != r["shots"]:
        raise ReplayError(f"{name} chunk {seed}: receipts + trivial != chunk size")
    chunk_largest = Fraction(0)
    attacked = forged = 0
    bat = Counter()
    forg = Counter()
    for rec in recs:
        a, b = decide_receipt(edges, rec, eps_max)
        want_eps = None if rec["epsilon"] is None else Fraction(rec["epsilon"])
        want_lb = None if rec["lower_bound"] is None else Fraction(rec["lower_bound"])
        if not (a.status == b[0] == rec["status"]):
            raise ReplayError(f"{name} chunk {seed} shot {rec['shot']}: recorded {rec['status']}, "
                              f"check_epsilon {a.status}, verifier B {b[0]}")
        if rec["status"] != "INVALID_INPUT" and not (a.epsilon == b[1] == want_eps and a.lower_bound == b[2] == want_lb):
            raise ReplayError(f"{name} chunk {seed} shot {rec['shot']}: epsilon or bound does not reproduce")
        if rec["status"] in ACCEPT:
            chunk_largest = max(chunk_largest, want_eps)
            if attacked < battery:
                _attack(name, edges, rec, _receipt_duals(rec), eps_max, bat, other)
                attacked += 1
            if forged < n_forg and rec["duals"]:
                _forgery_battery(edges, rec, _receipt_duals(rec), eps_max, forg)
                forged += 1
    if chunk_largest != Fraction(r["epsilon_max_accepted"]):
        raise ReplayError(f"{name} chunk {seed}: largest accepted epsilon does not reproduce")
    return {"seed": seed, "checked": len(recs), "battery": dict(bat), "forgeries": dict(forg)}


def replay_config(name, *, sample=None, battery=0, other_name=None, rng=None, jobs=1, forgeries=0,
                  partial=False, eps_max=None):
    """Re-decide a configuration's receipts. `partial=True` is the PARTIAL-PLAN mode (2026-11-05): a sample whose run
    was stopped by its preregistered budget has no summary.json (the merge refuses a missing chunk), so the plan is
    the chunks the index holds, every one of them is replayed, and the totals are reported from the index instead of
    reconciled with a summary; eps_max must then be given. The result says `partial: true` and lists the seeds."""
    if partial:
        if eps_max is None:
            raise ReplayError(f"{name}: a partial-plan replay needs eps_max")
        summ, eps_max = None, Fraction(eps_max)
    else:
        summ = json.loads((INDEX / name / "summary.json").read_text(encoding="utf-8"))
        eps_max = Fraction(summ["eps_max"])
    _edges_list, sha = _instance(name)
    rows = [json.loads(x) for x in (INDEX / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    # an index row whose receipt file was later rewritten is superseded, not a duplicate:
    # keep only the row whose hash the file on disk carries, and demand exactly one
    by_seed = {}
    for r in rows:
        rp = ROOT / r["receipts_file"]
        if not rp.exists():
            continue
        if hashlib.sha256(gzip.decompress(rp.read_bytes())).hexdigest() == r["receipts_sha256"]:
            if r["seed"] in by_seed:
                raise ReplayError(f"{name}: chunk {r['seed']} indexed twice with the same receipt hash")
            by_seed[r["seed"]] = r
    if partial:
        # the plan is what the stopped run completed: every verified index line
        plan = set(by_seed)
        complete_plan = False
        if not plan:
            raise ReplayError(f"{name}: no verified chunk in the index")
    else:
        # the plan is what summary.json merged; a pilot merge (--limit) covers a prefix
        lo, hi = summ["spec"]["seeds"]
        plan = set(range(lo, lo + summ["chunks"]))
        complete_plan = summ["chunks"] == hi - lo + 1
        missing = sorted(plan - set(by_seed))
        extra = sorted(s for s in by_seed if not lo <= s <= hi)
        if missing or extra:
            raise ReplayError(f"{name}: chunks missing {missing[:5]} ({len(missing)}), extra {extra[:5]}")

    tot = Counter()
    largest = Fraction(0)
    seeds = sorted(plan)
    for seed in seeds:
        tot.update(by_seed[seed]["status"])
        largest = max(largest, Fraction(by_seed[seed]["epsilon_max_accepted"]))
    chosen = seeds if sample is None else sorted((rng or random).sample(seeds, min(sample, len(seeds))))
    work = [(name, s, by_seed[s], str(eps_max), battery, other_name, forgeries) for s in chosen]
    checked = 0
    bat = Counter()
    forg = Counter()
    worst = 0.0

    def absorb(res):
        nonlocal checked, worst
        checked += res["checked"]
        b = dict(res["battery"])
        worst = max(worst, b.pop("boundary_excess_over_eps_max_max", 0.0))
        bat.update(b)
        forg.update(res.get("forgeries", {}))

    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            for res in pool.imap_unordered(verify_chunk, work, chunksize=4):
                absorb(res)
    else:
        for job in work:
            absorb(verify_chunk(job))
    nontrivial = sum(v for k, v in tot.items() if k != "TRIVIAL")
    if not partial and (dict(tot) != summ["status"] or nontrivial != summ["nontrivial"] or
                        tot["EXACT_OPTIMAL"] + tot["EPSILON_OPTIMAL"] != summ["accepted"] or
                        largest != Fraction(summ["largest_accepted_epsilon"])):
        raise ReplayError(f"{name}: totals do not reproduce summary.json")
    battery_out = dict(bat)
    if worst:
        battery_out["boundary_excess_over_eps_max_max"] = worst
    out = {"config": name, "chunks": len(plan), "complete_plan": complete_plan,
           "nontrivial": nontrivial, "receipts_rechecked": checked,
           "full": sample is None, "instance_sha256": sha, "battery": battery_out}
    if partial:
        out.update({"partial": True, "eps_max": str(eps_max), "seeds": sorted(plan), "status": dict(tot),
                    "accepted": tot["EXACT_OPTIMAL"] + tot["EPSILON_OPTIMAL"], "largest_accepted_epsilon": str(largest)})
    if battery and other_name:
        out["substituted_graph"] = {"config": other_name, "instance_sha256": _instance(other_name)[1]}
    if forgeries:
        out["forgery_battery"] = {
            "receipts_per_chunk": forgeries,
            "refusals_by_class_and_verdict": {k: v for k, v in forg.items() if not k.startswith("diag:")},
            "parity_adversary": {k[5:]: v for k, v in forg.items() if k.startswith("diag:")}}
    return out


def _attack(name, edges, rec, duals, eps_max, bat, other_graph):
    """Boundary and substitution attacks on one accepted receipt."""
    _a0, b0 = _decide_both(edges, rec, duals, eps_max)
    eps0 = b0[1]
    corr = [(min(u, v), max(u, v)) for u, v in rec["correction"]]
    if not corr:
        return
    W = {}
    for u, v, w in edges:
        k = (min(u, v), max(u, v))
        W[k] = min(W.get(k, w), w)
    k = corr[0]
    # (i) just past the boundary: the smallest double weight on edge k that makes
    # w(J) exceed the old bound by more than eps_max. Raising one correction edge
    # can only raise theta, so recompute rather than assume; walk up until refused.
    target = W[k] + float(eps_max - eps0)
    w_up = math.nextafter(target, math.inf)
    for _ in range(64):
        e2 = [(u, v, (w_up if (min(u, v), max(u, v)) == k else w)) for u, v, w in edges]
        a, b = _decide_both(e2, rec, duals, eps_max)
        if a.status != b[0] or (b[1] is not None and a.epsilon != b[1]):
            raise ReplayError(f"{name} shot {rec['shot']}: checkers disagree at the boundary")
        if b[1] is not None and b[1] > eps_max:
            bat["boundary_attempts"] += 1
            if a.status in ACCEPT:
                raise ReplayError(f"{name} shot {rec['shot']}: accepted with proven slack above eps_max")
            bat["boundary_just_outside_refused"] += 1
            bat["boundary_excess_over_eps_max_max"] = max(
                bat.get("boundary_excess_over_eps_max_max", 0.0), float(b[1] - eps_max))
            break
        w_up = math.nextafter(w_up, math.inf)
    # (ii) just inside: half the remaining room, a true claim both must accept
    w_in = W[k] + float(eps_max - eps0) / 2
    e3 = [(u, v, (w_in if (min(u, v), max(u, v)) == k else w)) for u, v, w in edges]
    a, b = _decide_both(e3, rec, duals, eps_max)
    if a.status != b[0]:
        raise ReplayError(f"{name} shot {rec['shot']}: checkers disagree just inside the boundary")
    if b[1] is not None and b[1] <= eps_max:
        if a.status not in ACCEPT:
            raise ReplayError(f"{name} shot {rec['shot']}: refused a true claim just inside the boundary")
        bat["boundary_just_inside_accepted"] += 1
    # (iii) graph substitution: the same receipt against another configuration's graph
    if other_graph is not None:
        a, b = _decide_both(other_graph, rec, duals, eps_max)
        if a.status != b[0]:
            raise ReplayError(f"{name} shot {rec['shot']}: checkers disagree on a substituted graph")
        bat["substituted_graph_accepted" if a.status in ACCEPT else "substituted_graph_refused"] += 1


#: What decided the verdicts a record holds: this tool (verifier B lives in it), the epsilon checker, the certificate
#: type, and the forgery battery. Pinned in every record's head (2026-11-05); the paper's generator holds them to the
#: tree it typesets from, as it does every other evidence writer's.
JUDGED_FILES = ["tools/replay_epsilon_receipts.py", "src/oneq/matching_cert_epsilon.py", "src/oneq/matching_cert.py",
                "src/oneq/forgeries.py"]


#: The repository this tool is part of -- the judged bytes live here, whatever data root a caller points ROOT at.
_REPO = pathlib.Path(__file__).resolve().parents[1]


def record_head(a) -> dict:
    import datetime as _dt
    import subprocess

    def git(*args):
        """git's answer, or None where there is no repository (a reader's extracted bundle): the bytes are still
        pinned by sha256 below; only the commit comparison is then impossible, and the record says so."""
        try:
            r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)
        except OSError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *JUDGED_FILES)
    return {"tool": "tools/replay_epsilon_receipts.py",
            "commit": commit,
            "repository": "present" if commit else "absent: commit comparison not possible, bytes pinned by sha256",
            "uncommitted_changes_in_judged_files": bool(dirty) if commit else None,
            "sha256": {p: hashlib.sha256((_REPO / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                       for p in JUDGED_FILES},
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "jobs": a.jobs, "sample": a.sample, "battery": a.battery,
            "forgeries": a.forgeries, "partial": a.partial, "seed": a.seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None, help="re-check only N random chunks per config")
    ap.add_argument("--battery", type=int, default=0, help="boundary attacks per chunk re-checked")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--forgeries", type=int, default=0,
                    help="run the audited forgery battery on this many accepted receipts per chunk, both verifiers")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--json", default=None, help="write the per-configuration results here")
    ap.add_argument("--partial", action="store_true",
                    help="partial-plan mode: the named configurations have no summary.json (a run stopped by its "
                         "budget); replay every chunk their index holds")
    ap.add_argument("--eps-max", default="1/1000000", help="eps_max for --partial (a summary carries it otherwise)")
    a = ap.parse_args()
    merged = sorted(p.parent.name for p in INDEX.glob("*/summary.json"))
    if a.partial and not a.config:
        print("--partial needs the configurations named with --config", file=sys.stderr)
        return 2
    names = a.config or merged
    if not names:
        print("no merged epsilon-rerun configurations found", file=sys.stderr)
        return 2
    rng = random.Random(a.seed)
    out = []
    try:
        for n in names:
            other = substitute_for(n, merged) if a.battery else None
            res = replay_config(n, sample=a.sample, battery=a.battery, other_name=other, rng=rng, jobs=a.jobs,
                                forgeries=a.forgeries, partial=a.partial, eps_max=a.eps_max if a.partial else None)
            out.append(res)
            print(json.dumps(res), flush=True)
    except ReplayError as exc:
        print(f"EPSILON REPLAY FAILED: {exc}", file=sys.stderr)
        return 1
    kind = ("PARTIAL-PLAN (every chunk a stopped run completed)" if a.partial
            else "COMPLETE" if a.sample is None and all(r["complete_plan"] for r in out)
            else "PARTIAL" if a.sample is None else f"SAMPLED ({a.sample} chunks/config)")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"kind": kind, "battery_per_chunk": a.battery,
                                                    "head": record_head(a), "results": out}, indent=1) + "\n",
                                        encoding="utf-8", newline="\n")
    print(f"{kind} EPSILON REPLAY PASSED: {sum(r['receipts_rechecked'] for r in out)} receipts in "
          f"{len(out)} configurations, each decided identically by two checkers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
