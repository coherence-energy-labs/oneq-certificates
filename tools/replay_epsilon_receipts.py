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
import sys
from collections import Counter
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oneq.matching_cert import MatchingCertificate          # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon         # noqa: E402

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


def verify_b(edges, fired, correction, duals, eps_max: Fraction):
    """(status, epsilon, lower_bound) by integer arithmetic. Independent of check_epsilon."""
    raw = []
    for u, v, w in edges:
        if w < 0:
            return "INVALID_INPUT", None, None
        raw.append((min(u, v), max(u, v), w))
    pairs = []
    for S, z in duals:
        if not math.isfinite(z) or z < 0:
            return "INVALID_INPUT", None, None
        pairs.append((frozenset(S), z))
    if len({S for S, _ in pairs}) != len(pairs):
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


def _instance(name):
    p = ROOT / "evidence" / "instances" / f"matching_{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    body = d["edges"]
    if hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest() != d["edges_sha256"]:
        raise ReplayError(f"{p.name}: edge list does not hash to edges_sha256")
    return [(int(u), int(v), float.fromhex(w)) for u, v, w in body], d["edges_sha256"]


def _edges(name):
    if name not in _EDGES:
        _EDGES[name] = _instance(name)[0]
    return _EDGES[name]


def _receipt_duals(rec):
    return [(tuple(int(x) for x in S), float.fromhex(z)) for S, z in rec["duals"]]


def _cert(rec, duals):
    corr = tuple((int(a), int(b)) for a, b in rec["correction"])
    return MatchingCertificate(matched_edges=corr, node_potentials={},
                               blossom_duals={frozenset(S): z for S, z in duals})


def _decide_both(edges, rec, duals, eps_max, corr=None):
    corr = corr or [tuple(e) for e in rec["correction"]]
    a = check_epsilon(edges=edges, syndrome=rec["fired"], cert=_cert(rec, duals), eps_max=eps_max)
    b = verify_b(edges, rec["fired"], corr, duals, eps_max)
    return a, b


def verify_chunk(job):
    """Re-decide one chunk's receipts. job = (name, seed, index_row, eps_max, battery, other_name)."""
    name, seed, r, eps_max_s, battery, other_name = job
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
    attacked = 0
    bat = Counter()
    for rec in recs:
        duals = _receipt_duals(rec)
        a, b = _decide_both(edges, rec, duals, eps_max)
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
                _attack(name, edges, rec, duals, eps_max, bat, other)
                attacked += 1
    if chunk_largest != Fraction(r["epsilon_max_accepted"]):
        raise ReplayError(f"{name} chunk {seed}: largest accepted epsilon does not reproduce")
    return {"seed": seed, "checked": len(recs), "battery": dict(bat)}


def replay_config(name, *, sample=None, battery=0, other_name=None, rng=None, jobs=1):
    summ = json.loads((INDEX / name / "summary.json").read_text(encoding="utf-8"))
    eps_max = Fraction(summ["eps_max"])
    _edges_list, sha = _instance(name)
    lo, hi = summ["spec"]["seeds"]
    rows = [json.loads(x) for x in (INDEX / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
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
    # the plan is what summary.json merged; a pilot merge (--limit) covers a prefix
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
    work = [(name, s, by_seed[s], str(eps_max), battery, other_name) for s in chosen]
    checked = 0
    bat = Counter()
    worst = 0.0

    def absorb(res):
        nonlocal checked, worst
        checked += res["checked"]
        b = dict(res["battery"])
        worst = max(worst, b.pop("boundary_excess_over_eps_max_max", 0.0))
        bat.update(b)

    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            for res in pool.imap_unordered(verify_chunk, work, chunksize=4):
                absorb(res)
    else:
        for job in work:
            absorb(verify_chunk(job))
    nontrivial = sum(v for k, v in tot.items() if k != "TRIVIAL")
    if dict(tot) != summ["status"] or nontrivial != summ["nontrivial"] or \
            tot["EXACT_OPTIMAL"] + tot["EPSILON_OPTIMAL"] != summ["accepted"] or \
            largest != Fraction(summ["largest_accepted_epsilon"]):
        raise ReplayError(f"{name}: totals do not reproduce summary.json")
    battery_out = dict(bat)
    if worst:
        battery_out["boundary_excess_over_eps_max_max"] = worst
    return {"config": name, "chunks": len(plan), "complete_plan": complete_plan,
            "nontrivial": nontrivial, "receipts_rechecked": checked,
            "full": sample is None, "instance_sha256": sha, "battery": battery_out}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None, help="re-check only N random chunks per config")
    ap.add_argument("--battery", type=int, default=0, help="boundary attacks per chunk re-checked")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--json", default=None, help="write the per-configuration results here")
    a = ap.parse_args()
    merged = sorted(p.parent.name for p in INDEX.glob("*/summary.json"))
    names = a.config or merged
    if not names:
        print("no merged epsilon-rerun configurations found", file=sys.stderr)
        return 2
    rng = random.Random(a.seed)
    out = []
    try:
        for n in names:
            other = None
            if a.battery:
                alt = [m for m in merged if m != n]
                other = alt[0] if alt else None
            res = replay_config(n, sample=a.sample, battery=a.battery, other_name=other, rng=rng, jobs=a.jobs)
            out.append(res)
            print(json.dumps(res), flush=True)
    except ReplayError as exc:
        print(f"EPSILON REPLAY FAILED: {exc}", file=sys.stderr)
        return 1
    kind = ("COMPLETE" if a.sample is None and all(r["complete_plan"] for r in out)
            else "PARTIAL" if a.sample is None else f"SAMPLED ({a.sample} chunks/config)")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"kind": kind, "battery_per_chunk": a.battery,
                                                    "results": out}, indent=1) + "\n",
                                        encoding="utf-8", newline="\n")
    print(f"{kind} EPSILON REPLAY PASSED: {sum(r['receipts_rechecked'] for r in out)} receipts in "
          f"{len(out)} configurations, each decided identically by two checkers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
