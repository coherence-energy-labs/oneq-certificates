r"""Differential fuzzing of the three epsilon-checkers on random and adversarial receipts.

THE THREE IMPLEMENTATIONS
  py_a   `oneq.matching_cert_epsilon.check_epsilon`, reached the way the replay reaches it
         (`tools/replay_epsilon_receipts.py`: `_receipt_duals`, `_cert`, `_decide_both`), so the comparison
         includes the loader's conversions, which is where a malformed receipt actually meets the checker;
  py_b   `verify_b`, the scaled-integer second verifier of the same replay (same language, same author);
  js     `verifier-js/verify-epsilon.mjs --batch`, written 2026-11-03 by a fresh agent from the manuscript's
         Sections II-IV and the brief in `docs/reviews/JS_EPSILON_BRIEF_2026-11-03.md`, without the Python source.
  py_raw `check_epsilon` called directly on the raw values (no loader conversions), reported alongside so a
         disagreement can be put at the loader or at the checker.

THE CASES. Small random graphs (2-6 detectors plus the boundary), random syndromes, the minimum-weight T-join by
brute force and the optimal T'-odd packing by linear programming, then perturbed so every outcome occurs:
exactly optimal, epsilon-optimal, not proven, overloaded (theta < 1), and eps_max placed exactly at, one unit
below and one unit above the receipt's own epsilon. Each valid case is also MUTATED, one defect at a time, into
the malformed domain the review named: T'-oddness traps (an even set; the boundary-side set {B, t} that is odd
against T but even against T + B), repeated sets, repeated correction edges, non-finite and negative values,
parallel edges, self-loops, fractional / boolean / string vertex ids, a detector fired twice, the boundary in the
syndrome, non-canonical and non-binary64 hex literals, unknown vertices.

THE VERDICT. A case AGREES when all three return the same status and, for every status but INVALID_INPUT, the same
exact epsilon and lower bound. Every disagreement is reported with its mutation class; a class is a CONTRACT
DISAGREEMENT (the manuscript does not decide it) or a DEFECT (one reading accepts what the definition refuses).

    python tools/fuzz_epsilon_checkers.py --cases 2000 --seed 20261103 --out evidence/epsilon_checker_fuzz.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from oneq.matching_cert import MatchingCertificate                     # noqa: E402
from oneq.matching_cert_epsilon import check_epsilon                   # noqa: E402
import replay_epsilon_receipts as rep                                  # noqa: E402

B = -1
JS = ROOT / "verifier-js" / "verify-epsilon.mjs"


# ---------------------------------------------------------------- valid instances

def _key(u, v):
    if u == B:
        return (v, u)
    return (u, v) if v == B or u <= v else (v, u)


def _odd_of(edge_set):
    deg = Counter()
    for u, v in edge_set:
        for x in (u, v):
            if x != B:
                deg[x] += 1
    return {x for x, c in deg.items() if c % 2}


def _weight(rng):
    """A random binary64 weight: dyadic small values, or a random double (an LLR-like magnitude)."""
    r = rng.random()
    if r < 0.35:
        return rng.randint(1, 16) / 4.0
    if r < 0.45:
        return 0.0
    return rng.uniform(0.05, 9.0)


def random_instance(rng):
    """(edges, fired, correction, duals) with correction a minimum-weight T-join and duals an optimal packing."""
    import numpy as np
    from scipy.optimize import linprog
    n = rng.randint(2, 6)
    V = list(range(n))
    E = {}
    for u in V:                                   # a connected spine, then extra edges and boundary edges
        if u > 0:
            E[_key(rng.randrange(u), u)] = _weight(rng)
    for _ in range(rng.randint(0, n + 2)):
        u, v = rng.sample(V + [B], 2)
        E.setdefault(_key(u, v), _weight(rng))
    for u in V:
        if rng.random() < 0.5:
            E.setdefault(_key(u, B), _weight(rng))
    if not any(B in k for k in E):
        E[_key(rng.choice(V), B)] = _weight(rng)
    keys = sorted(E)
    if len(keys) > 13:
        keys = keys[:13]
        E = {k: E[k] for k in keys}
    fired = sorted(rng.sample(V, rng.randint(1, n)))
    T = set(fired)
    best = None
    for r in range(len(keys) + 1):              # brute-force minimum-weight T-join, exact
        for comb in itertools.combinations(keys, r):
            if _odd_of(comb) == T:
                w = sum((Fraction(E[k]) for k in comb), Fraction(0))
                if best is None or w < best[0]:
                    best = (w, comb)
    if best is None:
        return None
    Tp = T | ({B} if len(T) % 2 else set())
    verts = V + [B]
    family = [frozenset(S) for r in range(1, len(verts) + 1) for S in itertools.combinations(verts, r)
              if len(set(S) & Tp) % 2 == 1]
    A = np.zeros((len(keys), len(family)))
    for j, S in enumerate(family):
        for i, (u, v) in enumerate(keys):
            if (u in S) != (v in S):
                A[i, j] = 1.0
    res = linprog(-np.ones(len(family)), A_ub=A, b_ub=np.array([E[k] for k in keys]),
                  bounds=[(0, None)] * len(family), method="highs")
    duals = [(sorted(S), float(x)) for S, x in zip(family, res.x) if x > 1e-12] if res.status == 0 else []
    return [(u, v, E[(u, v)]) for (u, v) in keys], fired, [list(e) for e in best[1]], duals


def wire(edges, fired, correction, duals, eps_max="1/1000000", **extra):
    return {"edges": [[u, v, w if isinstance(w, str) else float(w).hex()] for u, v, w in edges],
            "fired": fired, "correction": correction,
            "duals": [[S, z if isinstance(z, str) else float(z).hex()] for S, z in duals],
            "eps_max": eps_max, **extra}


# ---------------------------------------------------------------- mutations (one defect each)

def mutations(case, rng):
    """[(class, mutated case)] -- each changes exactly one thing."""
    out = []
    e, f, c, d = case["edges"], case["fired"], case["correction"], case["duals"]
    verts = sorted({x for u, v, _w in e for x in (u, v)})
    Tp = set(f) | ({B} if len(f) % 2 else set())

    def m(cls, **kw):
        out.append((cls, {**case, **kw}))
    # perturbations that stay in the valid domain (the outcome changes, the input stays well-formed)
    if d:
        m("valid:packing_scaled_up_overload", duals=[[S, float.hex(float.fromhex(z) * 1.25)] for S, z in d])
        m("valid:packing_scaled_down", duals=[[S, float.hex(float.fromhex(z) * 0.5)] for S, z in d])
        m("valid:packing_tiny_shortfall",
          duals=[[S, float.hex(float.fromhex(z) * (1 - 2.0 ** -40))] for S, z in d])
        if Tp and sorted(Tp)[:1] not in [S for S, _z in d]:
            m("valid:zero_valued_set", duals=d + [[sorted(Tp)[:1], "0x0.0p+0"]])
        m("valid:negative_zero_value", duals=[[d[0][0], "-0x0.0p+0"]] + d[1:])
    m("valid:reversed_edge_orientation_in_correction", correction=[[v, u] for u, v in c])
    m("valid:empty_packing", duals=[])
    # T'-oddness traps
    even = [x for x in verts if x not in Tp]
    if even:
        m("trap:set_with_no_T_vertex", duals=d + [[[even[0]], "0x1.0p+0"]])
    if len(f) % 2:
        m("trap:boundary_side_parity_{B,t}", duals=d + [[sorted([B, f[0]]), "0x1.0p-2"]])
    if len(f) >= 2:
        m("trap:pair_of_fired", duals=d + [[sorted(f[:2]), "0x1.0p-2"]])
    m("trap:empty_set", duals=d + [[[], "0x1.0p-2"]])
    # repeats
    if d:
        m("repeat:set_listed_twice_same_value", duals=d + [d[0]])
        m("repeat:set_listed_twice_other_value", duals=d + [[d[0][0], "0x1.0p-3"]])
        m("repeat:set_listed_twice_permuted", duals=d + [[list(reversed(d[0][0])), d[0][1]]])
    if c:
        m("repeat:correction_edge_twice", correction=c + [c[0]])
        m("repeat:correction_edge_twice_reversed", correction=c + [[c[0][1], c[0][0]]])
    if len(f) >= 1:
        m("syndrome:detector_fired_twice", fired=f + [f[0]])
        m("syndrome:boundary_in_syndrome", fired=f + [B])
        if 2 in verts and 2 not in f:
            m("syndrome:entry_2_not_fired", fired=sorted(f + [2]))
    # non-finite and negative values
    for bad in ("inf", "-inf", "nan"):
        if d:
            m(f"nonfinite:dual_{bad}", duals=[[d[0][0], bad]] + d[1:])
        m(f"nonfinite:weight_{bad}", edges=[[e[0][0], e[0][1], bad]] + e[1:])
    if d:
        m("negative:dual", duals=[[d[0][0], "-0x1.0p-3"]] + d[1:])
    m("negative:weight", edges=[[e[0][0], e[0][1], "-0x1.0p+0"]] + e[1:])
    # graph shape
    m("graph:parallel_edge_heavier", edges=e + [[e[0][0], e[0][1], float.hex(float.fromhex(e[0][2]) + 1.0)]])
    m("graph:parallel_edge_lighter", edges=e + [[e[0][1], e[0][0], float.hex(float.fromhex(e[0][2]) / 2)]])
    m("graph:self_loop", edges=e + [[verts[0], verts[0], "0x1.0p+0"]])
    m("graph:boundary_first_edge", edges=[[v, u, w] if v == B else [u, v, w] for u, v, w in e])
    # vertex-id domain
    if c:
        u0, v0 = c[0]
        m("id:fractional_in_correction", correction=[[u0 + 0.9 if u0 >= 0 else u0, v0]] + c[1:])
        m("id:integral_float_in_correction", correction=[[float(u0), v0]] + c[1:])
        m("id:boolean_in_correction", correction=[[True, v0]] + c[1:])
        m("id:string_in_correction", correction=[[str(u0), v0]] + c[1:])
    if d:
        S0 = d[0][0]
        m("id:fractional_in_set", duals=[[[x + 0.9 if x >= 0 else x for x in S0], d[0][1]]] + d[1:])
        m("id:boolean_in_set", duals=[[[True if x == 1 else x for x in S0], d[0][1]]] + d[1:])
        m("id:unknown_vertex_in_set", duals=[[sorted(S0 + [max(verts) + 7]), d[0][1]]] + d[1:])
    m("id:fractional_in_syndrome", fired=[f[0] + 0.5] + f[1:])
    m("id:boolean_in_syndrome", fired=[True if x == 1 else x for x in f])
    m("id:unknown_fired_detector", fired=sorted(f + [max(verts) + 7]))
    # hex literal domain
    if d:
        z0 = float.fromhex(d[0][1])
        m("hex:non_canonical_spelling", duals=[[d[0][0], _noncanonical(z0)]] + d[1:])
        m("hex:not_binary64_extra_digits", duals=[[d[0][0], _extra_bit(z0)]] + d[1:])
        m("hex:decimal_not_hex", duals=[[d[0][0], repr(z0)]] + d[1:])
        m("hex:overflow", duals=[[d[0][0], "0x1.0p+2000"]] + d[1:])
        m("hex:json_number_not_string", duals=[[d[0][0], z0]] + d[1:])
    m("hex:subnormal_weight", edges=[[e[0][0], e[0][1], "0x0.0000000000001p-1022"]] + e[1:])
    return out


def _noncanonical(z):
    """The same exact value, spelled as an integer mantissa: 1.5 -> '0x3p-1'."""
    if z == 0:
        return "0x0p+0"
    q = Fraction(z)
    j = q.denominator.bit_length() - 1             # denominator is a power of two
    return f"0x{q.numerator:x}p-{j}"


def _extra_bit(z):
    """A hex literal one 56th-bit away from z: not a binary64 value (Python's fromhex rounds it back to z)."""
    mant, ex = float.hex(z).split("p")
    head, _, frac = mant.partition(".")
    return f"{head}.{frac.ljust(13, '0')}1p{ex}"


# ---------------------------------------------------------------- the implementations

def py_replay(case):
    """(py_a, py_b) through the replay's own loader and decision path (`decide_receipt`, 2026-11-03 strict)."""
    inv = ("INVALID_INPUT", None, None)
    try:
        edges = [(rep.strict_vertex(u), rep.strict_vertex(v), rep.strict_hex(w, "weight")) for u, v, w in case["edges"]]
    except (rep.ReceiptFormatError, TypeError, ValueError):
        return inv, inv                          # the instance itself is outside the domain: nothing to decide
    try:
        rec = {"fired": case["fired"], "correction": case["correction"], "duals": case["duals"]}
        a, b = rep.decide_receipt(edges, rec, Fraction(case["eps_max"]))
        va = (a.status, None if a.epsilon is None or a.status == "INVALID_INPUT" else str(a.epsilon),
              None if a.lower_bound is None or a.status == "INVALID_INPUT" else str(a.lower_bound))
        vb = (b[0], None if b[1] is None or b[0] == "INVALID_INPUT" else str(b[1]),
              None if b[2] is None or b[0] == "INVALID_INPUT" else str(b[2]))
        return va, vb
    except Exception as exc:                                   # noqa: BLE001
        crash = ("HARNESS_RAISED", type(exc).__name__, None)
        return crash, crash


def py_raw(case):
    try:
        edges = [(u, v, float.fromhex(w) if isinstance(w, str) else w) for u, v, w in case["edges"]]
        duals = {frozenset(S): (float.fromhex(z) if isinstance(z, str) else z) for S, z in case["duals"]}
        cert = MatchingCertificate(matched_edges=tuple(tuple(e) for e in case["correction"]), node_potentials={},
                                   blossom_duals=duals)
        a = check_epsilon(edges=edges, syndrome=case["fired"], cert=cert, eps_max=Fraction(case["eps_max"]))
        return (a.status, None if a.status == "INVALID_INPUT" else str(a.epsilon),
                None if a.status == "INVALID_INPUT" else str(a.lower_bound))
    except Exception as exc:                                   # noqa: BLE001
        return ("HARNESS_RAISED", type(exc).__name__, None)


def js_batch(cases):
    node = shutil.which("node")
    if node is None or not JS.is_file():
        raise SystemExit("node or verifier-js/verify-epsilon.mjs is missing")
    text = "\n".join(json.dumps({"id": str(i), **c}) for i, c in enumerate(cases)) + "\n"
    p = subprocess.run([node, str(JS), "--batch"], input=text, capture_output=True, text=True,
                       encoding="utf-8", timeout=3600)
    if p.returncode != 0:
        raise SystemExit(f"the JavaScript checker exited {p.returncode}: {p.stderr[-2000:]}")
    out = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
    if len(out) != len(cases) or [o["id"] for o in out] != [str(i) for i in range(len(cases))]:
        raise SystemExit("the JavaScript checker did not return one line per case, in order")
    return [(o["status"], o["epsilon"] if o["status"] != "INVALID_INPUT" else None,
             o["lower_bound"] if o["status"] != "INVALID_INPUT" else None) for o in out]


# ---------------------------------------------------------------- run

def build_cases(n_base, seed):
    rng = random.Random(seed)
    cases = []
    while sum(1 for c in cases if c[0] == "base") < n_base:
        inst = random_instance(rng)
        if inst is None:
            continue
        edges, fired, corr, duals = inst
        base = wire(edges, fired, corr, duals)
        cases.append(("base", base))
        cases.extend(mutations(base, rng))
    return cases


def add_eps_boundary(cases):
    """For each base case, three copies with eps_max at, just below and just above its own exact epsilon."""
    extra = []
    for cls, c in cases:
        if cls != "base":
            continue
        va, _vb = py_replay(c)
        if va[0] in ("EPSILON_OPTIMAL", "NOT_PROVEN"):
            eps = Fraction(va[1])
            unit = Fraction(1, 2**1100)
            extra.append(("valid:eps_max_equals_epsilon", {**c, "eps_max": str(eps)}))
            extra.append(("valid:eps_max_one_unit_below", {**c, "eps_max": str(eps - unit)}))
            extra.append(("valid:eps_max_one_unit_above", {**c, "eps_max": str(eps + unit)}))
    return cases + extra


def replay_real(configs, chunks_per_config):
    """The JavaScript checker's verdict on real receipts, against the verdict both Python verifiers recorded."""
    import gzip
    node = shutil.which("node")
    out = {}
    for name in configs:
        inst = ROOT / "evidence" / "instances" / f"matching_{name}.json"
        rdir = ROOT / "data" / "epsilon_receipts" / name
        chunks = sorted(rdir.glob("chunk_*.jsonl.gz"))[:chunks_per_config] if rdir.is_dir() else []
        if not inst.is_file() or not chunks:
            out[name] = {"status": "ABSENT"}
            continue
        n = agree = 0
        mismatches = []
        for ch in chunks:
            recorded = [json.loads(x) for x in gzip.decompress(ch.read_bytes()).decode().splitlines() if x.strip()]
            p = subprocess.run([node, str(JS), "--instance", str(inst), "--receipts", str(ch)],
                               capture_output=True, text=True, encoding="utf-8", timeout=3600)
            if p.returncode != 0:
                raise SystemExit(f"{name} {ch.name}: JavaScript replay exited {p.returncode}: {p.stderr[-500:]}")
            got = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
            if len(got) != len(recorded):
                raise SystemExit(f"{name} {ch.name}: {len(got)} verdicts for {len(recorded)} receipts")
            for r, g in zip(recorded, got):
                n += 1
                same = (g["status"], g["epsilon"], g["lower_bound"]) == (r["status"], r["epsilon"], r["lower_bound"])
                agree += same
                if not same and len(mismatches) < 5:
                    mismatches.append({"chunk": ch.name, "shot": r["shot"], "recorded": [r["status"], r["epsilon"]],
                                       "js": [g["status"], g["epsilon"], g.get("reason")]})
        out[name] = {"chunks": len(chunks), "receipts": n, "identical_status_epsilon_bound": agree,
                     "mismatches": mismatches}
    return out


def write_vectors(path, seed, per_class):
    """Standing vectors: every fuzz class (valid and malformed), each case decided identically by all three."""
    cases = add_eps_boundary(build_cases(40, seed))
    picked, count = [], Counter()
    for cls, c in cases:
        if count[cls] < per_class:
            picked.append((cls, c))
            count[cls] += 1
    js = js_batch([c for _cls, c in picked])
    vectors, k = [], Counter()
    for (cls, c), vj in zip(picked, js):
        va, vb = py_replay(c)
        if not va == vb == vj:
            raise SystemExit(f"{cls}: the implementations disagree ({va}, {vb}, {vj}); fix that before writing vectors")
        vectors.append({"name": f"{cls}#{k[cls]}", "class": cls, "case": c,
                        "expected": {"status": va[0], "epsilon": va[1], "lower_bound": va[2]}})
        k[cls] += 1
    doc = {"schema": "oneq-epsilon-conformance-vectors/1", "generated_by": "tools/fuzz_epsilon_checkers.py "
           f"--write-vectors (seed {seed}, {per_class} per class)",
           "contract": "vertices are integers (boundary -1), never bools or floats; the graph is simple (no self-loop, "
                       "no edge twice in either orientation); weights and packing values are exact binary64 hex "
                       "literals, finite and non-negative; every set member and endpoint is a vertex of the graph; a "
                       "set or a correction edge listed twice is refused; anything else is INVALID_INPUT, never "
                       "coerced",
           "classes": sorted(count), "vectors": vectors}
    pathlib.Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(vectors)} vectors in {len(count)} classes -> {path}")
    return 0


REPLAY_CONFIGS = ("sim_d5_r25", "fresh_sim_d5_r25", "willow_d3_at_q4_5_X_r13", "willow_d5_at_q8_7_Z_r30",
                  "willow_d7_at_q6_7_X_r30", "willow_d7_at_q6_7_Z_r30", "heldout_d3_at_q4_5_Z_r50",
                  "heldout_d5_at_q8_7_X_r50")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2000, help="random valid base cases (mutants are added)")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--replay-chunks", type=int, default=0,
                    help="also replay this many real receipt chunks per configuration through the JavaScript checker")
    ap.add_argument("--write-vectors", type=str, default=None,
                    help="write the first N cases of every class, with the verdict all three implementations agree "
                         "on, as standing conformance vectors (refuses if any disagree)")
    ap.add_argument("--per-class", type=int, default=3)
    a = ap.parse_args(argv)
    if a.write_vectors:
        return write_vectors(a.write_vectors, a.seed, a.per_class)
    cases = add_eps_boundary(build_cases(a.cases, a.seed))
    js = js_batch([c for _cls, c in cases])
    by_class = defaultdict(Counter)
    examples = defaultdict(list)
    for (cls, c), vj in zip(cases, js):
        va, vb = py_replay(c)
        vr = py_raw(c)
        agree = va == vb == vj
        key = "agree" if agree else "disagree"
        by_class[cls][key] += 1
        by_class[cls]["status:" + "|".join(str(x[0]) for x in (va, vb, vj))] += 1
        if not agree and len(examples[cls]) < 2:
            examples[cls].append({"case": c, "py_a_replay": va, "py_b_replay": vb, "js": vj, "py_a_raw": vr})
    valid = [k for k in by_class if k == "base" or k.startswith("valid:")]
    report = {
        "schema": "oneq-epsilon-checker-fuzz/1", "seed": a.seed, "base_cases": a.cases, "cases": len(cases),
        "implementations": {"py_a": "check_epsilon via tools/replay_epsilon_receipts.py",
                            "py_b": "verify_b (tools/replay_epsilon_receipts.py)",
                            "js": "verifier-js/verify-epsilon.mjs (written from docs/reviews/"
                                  "JS_EPSILON_BRIEF_2026-11-03.md)"},
        "valid_domain_cases": sum(sum(v for k2, v in by_class[k].items() if k2 in ("agree", "disagree"))
                                  for k in valid),
        "valid_domain_disagreements": sum(by_class[k]["disagree"] for k in valid),
        "per_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "disagreeing_classes": sorted(k for k, v in by_class.items() if v["disagree"]),
        "examples": dict(examples)}
    if a.replay_chunks:
        report["real_receipts_js_vs_recorded"] = replay_real(REPLAY_CONFIGS, a.replay_chunks)
    text = json.dumps(report, indent=1, default=str)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "valid_domain_cases", "valid_domain_disagreements",
                                             "disagreeing_classes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
