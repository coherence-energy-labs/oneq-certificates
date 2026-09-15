"""The epsilon-receipt replay must FAIL on every planted defect, and its second
verifier must agree with check_epsilon and be sound against brute force."""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
import random
import sys
from fractions import Fraction

import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import replay_epsilon_receipts as R  # noqa: E402
from oneq.matching_cert import MatchingCertificate  # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon  # noqa: E402

EDGES = [(0, 1, 0.1), (1, -1, 0.1), (0, -1, 0.15)]
EPS = Fraction(1, 10**6)


def _rec(shot, z):
    fired, corr = [0, 1], [[0, 1]]
    duals = [[[0], z], [[1], z]]
    cert = MatchingCertificate(matched_edges=((0, 1),), node_potentials={},
                               blossom_duals={frozenset([0]): z, frozenset([1]): z})
    v = check_epsilon(edges=EDGES, syndrome=fired, cert=cert, eps_max=EPS)
    return {"shot": shot, "fired": fired, "correction": corr,
            "duals": [[S, float.hex(x)] for S, x in duals], "status": v.status,
            "epsilon": str(v.epsilon), "lower_bound": str(v.lower_bound)}


def _build(tmp_path, monkeypatch, *, chunks=2):
    root = tmp_path
    (root / "evidence" / "instances").mkdir(parents=True)
    body = [[u, v, float.hex(w)] for u, v, w in EDGES]
    (root / "evidence" / "instances" / "matching_toy.json").write_text(json.dumps(
        {"edges": body, "edges_sha256": hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()}))
    idx = root / "evidence" / "epsilon_rerun" / "toy"
    idx.mkdir(parents=True)
    rows, tot = [], {}
    for seed in range(1, chunks + 1):
        recs = [_rec(0, 0.05), _rec(1, 0.0499999999), _rec(2, 0.04)]
        status = {}
        for r in recs:
            status[r["status"]] = status.get(r["status"], 0) + 1
        status["TRIVIAL"] = 1
        payload = ("\n".join(json.dumps(r) for r in recs) + "\n").encode()
        rp = root / "data" / f"chunk_{seed}.jsonl.gz"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_bytes(gzip.compress(payload))
        acc = [Fraction(r["epsilon"]) for r in recs if r["status"] in R.ACCEPT]
        rows.append({"seed": seed, "shots": 4, "status": status, "epsilon_max_accepted": str(max(acc)),
                     "receipts_file": rp.relative_to(root).as_posix(),
                     "receipts_sha256": hashlib.sha256(payload).hexdigest()})
        for k, v in status.items():
            tot[k] = tot.get(k, 0) + v
    (idx / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    nontrivial = sum(v for k, v in tot.items() if k != "TRIVIAL")
    summ = {"eps_max": str(EPS), "spec": {"seeds": [1, chunks]}, "chunks": chunks, "status": tot,
            "nontrivial": nontrivial, "accepted": tot.get("EXACT_OPTIMAL", 0) + tot.get("EPSILON_OPTIMAL", 0),
            "largest_accepted_epsilon": max(r["epsilon_max_accepted"] for r in rows)}
    (idx / "summary.json").write_text(json.dumps(summ))
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(R, "INDEX", root / "evidence" / "epsilon_rerun")
    monkeypatch.setattr(R, "_EDGES", {})
    other = [[5, 6, float.hex(1.0)]]
    (root / "evidence" / "instances" / "matching_other.json").write_text(json.dumps(
        {"edges": other, "edges_sha256": hashlib.sha256(json.dumps(other, separators=(",", ":")).encode()).hexdigest()}))
    return root, idx, rows


def test_toy_corpus_has_all_three_states_and_replays(tmp_path, monkeypatch):
    _root, _idx, rows = _build(tmp_path, monkeypatch)
    assert set(rows[0]["status"]) == {"EXACT_OPTIMAL", "EPSILON_OPTIMAL", "NOT_PROVEN", "TRIVIAL"}
    res = R.replay_config("toy", battery=2, other_name="other")
    assert res["complete_plan"] and res["receipts_rechecked"] == 6
    assert res["battery"]["boundary_just_outside_refused"] >= 2
    assert res["battery"]["substituted_graph_refused"] >= 2


def test_missing_chunk_file_fails(tmp_path, monkeypatch):
    root, _idx, rows = _build(tmp_path, monkeypatch)
    (root / rows[1]["receipts_file"]).unlink()
    with pytest.raises(R.ReplayError, match="missing"):
        R.replay_config("toy")


def test_tampered_receipt_without_rehash_fails(tmp_path, monkeypatch):
    root, _idx, rows = _build(tmp_path, monkeypatch)
    rp = root / rows[0]["receipts_file"]
    rp.write_bytes(gzip.compress(gzip.decompress(rp.read_bytes()).replace(b"NOT_PROVEN", b"EXACT_OPTIMAL")))
    with pytest.raises(R.ReplayError, match="missing"):
        R.replay_config("toy")


def test_tampered_receipt_with_consistent_index_fails(tmp_path, monkeypatch):
    root, idx, rows = _build(tmp_path, monkeypatch)
    rp = root / rows[0]["receipts_file"]
    recs = [json.loads(x) for x in gzip.decompress(rp.read_bytes()).decode().splitlines()]
    recs[2]["status"] = "EPSILON_OPTIMAL"          # a refusal relabelled as an acceptance
    payload = ("\n".join(json.dumps(r) for r in recs) + "\n").encode()
    rp.write_bytes(gzip.compress(payload))
    rows[0]["receipts_sha256"] = hashlib.sha256(payload).hexdigest()
    (idx / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(R.ReplayError):
        R.replay_config("toy")


def test_extra_chunk_outside_the_plan_fails(tmp_path, monkeypatch):
    root, idx, rows = _build(tmp_path, monkeypatch)
    rows.append(dict(rows[0], seed=99))
    (idx / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(R.ReplayError, match="extra"):
        R.replay_config("toy")


def test_duplicated_chunk_row_fails(tmp_path, monkeypatch):
    root, idx, rows = _build(tmp_path, monkeypatch)
    rows.append(rows[0])
    (idx / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(R.ReplayError, match="twice"):
        R.replay_config("toy")


def test_summary_totals_that_do_not_reproduce_fail(tmp_path, monkeypatch):
    _root, idx, _rows = _build(tmp_path, monkeypatch)
    s = json.loads((idx / "summary.json").read_text())
    s["accepted"] += 1
    (idx / "summary.json").write_text(json.dumps(s))
    with pytest.raises(R.ReplayError, match="totals"):
        R.replay_config("toy")


def test_altered_instance_fails(tmp_path, monkeypatch):
    root, _idx, _rows = _build(tmp_path, monkeypatch)
    p = root / "evidence" / "instances" / "matching_toy.json"
    d = json.loads(p.read_text())
    d["edges"][0][2] = float.hex(0.2)
    p.write_text(json.dumps(d))
    with pytest.raises(R.ReplayError, match="hash"):
        R.replay_config("toy")


def test_verifier_b_refuses_malformed_packings():
    fired, corr = [0, 1], [(0, 1)]
    assert R.verify_b(EDGES, fired, corr, [((0,), math.nan)], EPS)[0] == "INVALID_INPUT"
    assert R.verify_b(EDGES, fired, corr, [((0,), -0.1)], EPS)[0] == "INVALID_INPUT"
    assert R.verify_b(EDGES, fired, corr, [((0, 1), 0.1)], EPS)[0] == "INVALID_INPUT"   # even set
    assert R.verify_b(EDGES, fired, corr, [((0,), 0.05), ((0,), 0.05)], EPS)[0] == "INVALID_INPUT"
    assert R.verify_b(EDGES, fired, [(0, -1)], [((0,), 0.05)], EPS)[0] == "INVALID_INPUT"


def _brute_opt(edges, fired):
    W = {}
    for u, v, w in edges:
        k = (min(u, v), max(u, v))
        W[k] = min(W.get(k, Fraction(w)), Fraction(w))
    keys = list(W)
    best = None
    for bits in itertools.product((0, 1), repeat=len(keys)):
        par = {}
        for b, (u, v) in zip(bits, keys):
            if b:
                for x in (u, v):
                    par[x] = par.get(x, 0) ^ 1
        if {x for x, p in par.items() if p and x != -1} == set(fired):
            w = sum((W[k] for b, k in zip(bits, keys) if b), Fraction(0))
            best = w if best is None else min(best, w)
    return best, W


def test_verifier_b_agrees_with_check_epsilon_and_is_sound_on_random_instances():
    rng = random.Random(11)
    seen = set()
    for _ in range(300):
        n = rng.randint(2, 5)
        verts = list(range(n)) + [-1]
        edges = [(u, v, rng.choice([0.0, 0.1, 0.3, 1.0, 2.5, rng.random()]))
                 for u, v in itertools.combinations(verts, 2) if rng.random() < 0.7]
        if len(edges) > 10:
            edges = edges[:10]
        fired = sorted(rng.sample(range(n), rng.randint(1, n)))
        opt, W = _brute_opt(edges, fired)
        if opt is None:
            continue
        # a random correction: any feasible edge subset found by brute force order
        keys = list(W)
        corr = None
        for bits in itertools.product((0, 1), repeat=len(keys)):
            par = {}
            for b, (u, v) in zip(bits, keys):
                if b:
                    for x in (u, v):
                        par[x] = par.get(x, 0) ^ 1
            if {x for x, p in par.items() if p and x != -1} == set(fired) and rng.random() < 0.3:
                corr = [k for b, k in zip(bits, keys) if b]
                break
        if corr is None:
            continue
        Tp = set(fired) | ({-1} if len(fired) % 2 else set())
        pool = [S for r in (1, 2, 3) for S in itertools.combinations(verts, r) if len(set(S) & Tp) % 2]
        duals = [(S, rng.choice([0.0, 0.05, 0.1, 0.5, rng.random()])) for S in rng.sample(pool, min(len(pool), rng.randint(1, 4)))]
        cert = MatchingCertificate(matched_edges=tuple(corr), node_potentials={},
                                   blossom_duals={frozenset(S): z for S, z in duals})
        a = check_epsilon(edges=edges, syndrome=fired, cert=cert, eps_max=EPS)
        b = R.verify_b(edges, fired, corr, duals, EPS)
        assert a.status == b[0], (edges, fired, corr, duals, a, b)
        if a.status != "INVALID_INPUT":
            assert a.epsilon == b[1] and a.lower_bound == b[2]
            assert b[2] <= opt                                      # Lemma 1, against brute force
            w = sum((W[k] for k in corr), Fraction(0))
            assert w - opt <= b[1]
        seen.add(a.status)
    assert {"EXACT_OPTIMAL", "NOT_PROVEN"} <= seen
