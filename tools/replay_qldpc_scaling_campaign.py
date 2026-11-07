#!/usr/bin/env python3
r"""Replay the preregistered scaling campaign of paper 2 from its archived inputs and its records -- no producer.

    python -B tools/replay_qldpc_scaling_campaign.py
    python -B tools/replay_qldpc_scaling_campaign.py --out evidence/qldpc_scaling_prereg/costs/checking.json
    python -B tools/replay_qldpc_scaling_campaign.py --dataset uniform_p01        # one dataset: says SUBSET

External audit 3 of release v2026.11.05 (paper A = our paper 2): "a verifier should not need to call the
producer's model-building code to recover the precise instance whose optimum it is checking". This command reads
evidence/qldpc_scaling_prereg/inputs/MANIFEST.json, whose sha256 is pinned below, and:

  1. verifies the sha256 of every bound file -- the eight archived inputs, the eight record files, the persisted
     instances they reference, summary.json -- and refuses an input that names another dataset or seed;
  2. for every one of the 6,500 records: re-derives the syndrome from the archived sampled fault (NumPy datasets),
     recomputes each answer's feasibility and integer objective from the archived checks and weights, and the
     receipt answer's objective, against the record;
  3. re-checks every receipt with BOTH checkers the campaign used -- the production exact checker
     (src/oneq/qldpc_check.py, check_qldpc_exact for a flat receipt, check_qldpc_bnb for a tree) and the second
     checker the record's `second_checker` flag names (the reference flat checker
     tools/external/exact_lp_certificate_reference.py, or tree checker B tools/external/tree_checker_b.py) --
     in exact rational arithmetic, against the archived instance;
  4. re-scores both answers against the receipt exactly as the campaign defines it (tier 1 by the shot's single
     dual, an optimum on a branching tree, a heavier miss, not a correction), and requires a hit to be exactly
     "feasible with objective equal to the certified optimum";
  5. recomputes each answer's logical outcome from the archived logical data: the GF(2) row space of the archived
     stabilizer rows and the archived sampled fault (code capacity), or the archived observable columns and sampled
     observable bit (circuit);
  6. recomputes every number of the preregistered analysis -- receipts, kinds, optimum hits and their split,
     misses, logical errors, observable agreement, the paired cells and their exact McNemar p -- and refuses unless
     each equals summary.json, which is what the paper typesets.
It imports no experiment code, no producer module, no solver, no BP-OSD, no sampler and no NumPy: the Python
standard library and the three checker modules only -- the production checker and the two second checkers
(tests/test_replay_qldpc_scaling_campaign.py checks sys.modules). With --out it also writes the per-receipt checking latency of each checker, measured here.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import pathlib
import platform
import sys
import tempfile
import time
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The manifest this replay's verdict rests on. Changing the archive means re-running the writer
#: (experiments/m0_prior_art/qldpc_scaling_prereg_export_inputs.py) AND changing this pin in the same commit.
PINNED_MANIFEST_SHA256 = "0c6a5fa5c6658a7b7a0e1a92ab161dc5018d0cd976b5eafefd66146e953579fe"
MANIFEST = ROOT / "evidence" / "qldpc_scaling_prereg" / "inputs" / "MANIFEST.json"
REFERENCE_CHECKER = ROOT / "tools" / "external" / "exact_lp_certificate_reference.py"
TREE_CHECKER_B = ROOT / "tools" / "external" / "tree_checker_b.py"
PRODUCTION_CHECKER = ROOT / "src" / "oneq" / "qldpc_check.py"
INPUT_SCHEMA = "oneq-qldpc-scaling-prereg-input/1"
MANIFEST_SCHEMA = "oneq-qldpc-scaling-prereg-inputs-manifest/1"
REPORT_SCHEMA = "oneq-qldpc-scaling-prereg-replay/1"
ALPHAS = ("1.0", "0.625")
#: imported by this replay, any of these would be a producer-side dependency; so would any module under
#: experiments/ and any oneq module but the production checker
FORBIDDEN_MODULES = ("numpy", "scipy", "ldpc", "stim", "highspy", "pymatching", "sinter")
ALLOWED_ONEQ = ("oneq", "oneq.qldpc_check")


def forbidden_loaded(names) -> list[str]:
    """The producer-side modules among `names` (keys of sys.modules)."""
    exp = (ROOT / "experiments").as_posix().lower()
    bad = []
    for k in sorted(names):
        mod = sys.modules.get(k)
        f = str(getattr(mod, "__file__", "") or "").replace(os.sep, "/").lower()
        if (any(k == x or k.startswith(x + ".") for x in FORBIDDEN_MODULES) or f.startswith(exp)
                or (k.split(".")[0] == "oneq" and k not in ALLOWED_ONEQ)):
            bad.append(k)
    return bad


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def seed_rule(domain: str, name: str) -> int:
    """The preregistered seed rule (docs/PREREG_qldpc_scaling_2026-11-03.md), restated from its text."""
    s = int(hashlib.sha256(f"{domain}||{name}".encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)
    return s % (2 ** 31 - 1) if name == "bb_refsched" else s


def mcnemar_exact(n10: int, n01: int) -> float:
    """Exact two-sided McNemar p, computed as a rational and rounded once."""
    nd, k = n10 + n01, min(n10, n01)
    if nd == 0:
        return 1.0
    return float(min(Fraction(1), Fraction(2 * sum(math.comb(nd, i) for i in range(k + 1)), 2 ** nd)))


def _mask(support) -> int:
    m = 0
    for i in support:
        m |= 1 << int(i)
    return m


class XorBasis:
    """GF(2) row space by highest-bit elimination over Python ints."""

    def __init__(self, rows):
        self.piv: dict[int, int] = {}
        for v in rows:
            self.add(v)

    def add(self, v: int) -> None:
        while v:
            p = v.bit_length() - 1
            if p not in self.piv:
                self.piv[p] = v
                return
            v ^= self.piv[p]

    def contains(self, v: int) -> bool:
        while v:
            p = v.bit_length() - 1
            if p not in self.piv:
                return False
            v ^= self.piv[p]
        return True


def _deser_tree(node):
    if node[0] == "branch":
        return ("branch", int(node[1]), _deser_tree(node[2]), _deser_tree(node[3]))
    if node[0] == "bound":
        return ("bound", tuple((tuple(j) if isinstance(j, list) else int(j), tuple(F), Fraction(int(a), int(b)))
                               for (j, F, a, b) in node[1]))
    return ("infeasible", tuple(node[1]) if isinstance(node[1], list) else int(node[1]))


def _flat_duals(facets):
    return tuple((tuple(r) if isinstance(r, list) else int(r), tuple(F), Fraction(int(p), int(q)))
                 for (r, F, p, q) in facets)


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                   # its dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


class Checkers:
    """The two checkers the campaign used for each receipt kind, and nothing else."""

    def __init__(self):
        from oneq.qldpc_check import QldpcCertificate, check_qldpc_bnb, check_qldpc_exact
        self.Cert, self.exact, self.bnb = QldpcCertificate, check_qldpc_exact, check_qldpc_bnb
        self.ref = _load_by_path("replay_scaling_reference_checker", REFERENCE_CHECKER)
        self.tree_b = _load_by_path("replay_scaling_tree_checker_b", TREE_CHECKER_B)
        self.tmp = tempfile.mkdtemp(prefix="replay_scaling_")

    def write_tree_doc(self, doc: dict) -> pathlib.Path:
        """Tree checker B reads an instance document from a file, as the campaign ran it."""
        p = pathlib.Path(self.tmp) / "tree_doc.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def tree_b_accepts(self, p: pathlib.Path) -> bool:
        code = None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                code = self.tree_b.main(str(p))
            except SystemExit as exc:
                code = exc.code
            except Exception:                 # malformed input: the checker's own __main__ reads it as exit 1
                code = 1
        return code == 0


def _fail(fails: list, tag: str, msg: str) -> None:
    fails.append(f"[{tag}] {msg}")


def replay_dataset(name: str, man_entry: dict, summ_entry: dict, domain: str, root: pathlib.Path, ck: Checkers,
                   fails: list, timings: list | None) -> dict:
    """Every record of one dataset, re-checked and re-scored from the archive; returns the recomputed analysis."""
    inp = json.loads((root / man_entry["input"]["file"]).read_text(encoding="utf-8"))
    if inp.get("schema") != INPUT_SCHEMA or inp.get("dataset") != name:
        _fail(fails, "identity", f"{name}: the bound input file is for dataset {inp.get('dataset')!r} "
                                 f"(schema {inp.get('schema')!r})")
        return {}
    if inp.get("seed_domain") != domain or inp.get("seed") != seed_rule(domain, name) or \
            summ_entry.get("seed") != inp.get("seed"):
        _fail(fails, "identity", f"{name}: seed {inp.get('seed')} is not the preregistered rule's "
                                 f"{seed_rule(domain, name)} (summary {summ_entry.get('seed')})")
        return {}
    cref = inp["checks"]
    if man_entry["instances"].get(cref["file"]) != cref["sha256"]:
        _fail(fails, "identity", f"{name}: the input's instance reference is not the one the manifest binds")
        return {}
    inst = json.loads((root / cref["file"]).read_text(encoding="utf-8"))
    checks = [tuple(int(i) for i in c) for c in inst["checks"]]
    n = int(inst["n_variables"])
    m = len(checks)
    if n != inp["n_variables"] or n != cref["n_variables"] or m != cref["rows"]:
        _fail(fails, "identity", f"{name}: the archived instance has {m} x {n}, the input says otherwise")
        return {}
    w = inp["weights"]
    if (len(w) != n or any(type(x) is not int or x < 0 for x in w)
            or _sha(json.dumps(w).encode()) != inp["weights_sha256"]):
        _fail(fails, "weights", f"{name}: the weight vector is not {n} non-negative integers matching its sha256")
        return {}
    cmask = [_mask(c) for c in checks]
    wx = {i: Fraction(x) for i, x in enumerate(w)}

    def syndrome_of(support) -> list[int]:
        e = _mask(support)
        return [j for j in range(m) if (cmask[j] & e).bit_count() & 1]

    def objective(support) -> int:
        return sum(w[int(i)] for i in support)

    def proper(support, bound: int) -> bool:
        return (isinstance(support, list) and all(type(i) is int and 0 <= i < bound for i in support)
                and support == sorted(set(support)))

    logical = inp["logical"]
    stab = obs = None
    if logical["kind"] == "css_residual":
        sref = logical["stabilizer_rows"]
        if man_entry["instances"].get(sref["file"]) != sref["sha256"]:
            _fail(fails, "identity", f"{name}: the stabilizer reference is not the one the manifest binds")
            return {}
        sinst = json.loads((root / sref["file"]).read_text(encoding="utf-8"))
        if int(sinst["n_variables"]) != n:
            _fail(fails, "identity", f"{name}: the stabilizer rows have another width")
            return {}
        stab = XorBasis(_mask(c) for c in sinst["checks"])
    elif logical["kind"] == "declared_observable":
        obs = _mask(logical["observable_columns"])
    elif logical["kind"] != "none":
        _fail(fails, "identity", f"{name}: unknown logical test {logical['kind']!r}")
        return {}

    with gzip.open(root / man_entry["records"]["file"], "rt", encoding="utf-8") as f:
        recs = [json.loads(x) for x in f]
    shots = inp["samples"]["shots"]
    if [r["shot_index"] for r in recs] != list(range(len(recs))) or len(recs) != man_entry["nontrivial"] \
            or len(recs) != summ_entry["nontrivial"] or len(shots) != len(recs):
        _fail(fails, "records", f"{name}: {len(recs)} records, {len(shots)} archived samples, manifest "
                                f"{man_entry['nontrivial']}, summary {summ_entry['nontrivial']}")
        return {}
    dense = None
    tally = {a: {"tier1": 0, "optimum_branching": 0, "miss": 0, "not_a_correction": 0} for a in ALPHAS}
    hits = {a: [] for a in ALPHAS}
    lerr = {a: [] for a in ALPHAS}
    agree = {a: 0 for a in ALPHAS}
    receipt_lerr = 0
    kinds = {"flat": 0, "tree": 0}
    unresolved = 0
    for r, sh in zip(recs, shots):
        k = r["shot_index"]
        if sh["shot_index"] != k or sh["sample_index"] != r["sample_index"]:
            _fail(fails, "samples", f"{name} shot {k}: archived sample is shot {sh['shot_index']} / draw "
                                    f"{sh['sample_index']}, the record says draw {r['sample_index']}")
            continue
        syn_sup = r["syndrome_support"]
        if not proper(syn_sup, m) or not syn_sup:
            _fail(fails, "records", f"{name} shot {k}: malformed or trivial syndrome")
            continue
        fault = sh.get("fault")
        if logical["kind"] != "declared_observable":
            if not proper(fault, n) or syndrome_of(fault) != syn_sup:
                _fail(fails, "samples", f"{name} shot {k}: the archived fault does not produce the recorded syndrome")
                continue
        elif sh.get("observable") not in (0, 1):
            _fail(fails, "samples", f"{name} shot {k}: no archived observable bit")
            continue
        syn = [0] * m
        for j in syn_sup:
            syn[j] = 1
        ans = {}
        for a in ALPHAS:
            A = r["answers"][a]
            sup = A["support"]
            if not proper(sup, n):
                _fail(fails, "records", f"{name} shot {k}: malformed answer at {a}")
                break
            ok = syndrome_of(sup) == syn_sup
            if ok != A["syndrome_valid"] or objective(sup) != A["objective"]:
                _fail(fails, "objective", f"{name} shot {k} at {a}: recomputed feasible={ok}, objective "
                                          f"{objective(sup)}; the record says {A['syndrome_valid']}, {A['objective']}")
                break
            ans[a] = (sup, ok, objective(sup))
        else:
            R = r["receipt"]
            if R is None:
                unresolved += 1
                if not r.get("unresolved_reason"):
                    _fail(fails, "records", f"{name} shot {k}: no receipt and no reason")
                continue
            cand = R["answer"]
            if not proper(cand, n) or syndrome_of(cand) != syn_sup or objective(cand) != R["optimum"]:
                _fail(fails, "objective", f"{name} shot {k}: the receipt's answer is infeasible or its objective "
                                          f"{objective(cand) if proper(cand, n) else '?'} is not the recorded "
                                          f"optimum {R['optimum']}")
                continue
            if R.get("second_checker") is not True or r.get("unresolved_reason") is not None:
                _fail(fails, "records", f"{name} shot {k}: the record does not say both checkers accepted")
                continue
            U = R["optimum"]
            if R["kind"] == "flat":
                fd = _flat_duals(R["facet_duals"])
                t0 = time.perf_counter()
                a_ok = ck.exact(checks=checks, weights=wx, syndrome=syn,
                                cert=ck.Cert(error_support=tuple(cand), facet_duals=fd)).accepted
                t1 = time.perf_counter()
                if dense is None:
                    dense = [[0] * n for _ in checks]
                    for j, c in enumerate(checks):
                        for i in c:
                            dense[j][i] = 1
                e = [0] * n
                for i in cand:
                    e[i] = 1
                doc = {"H": dense, "s": syn, "weights": [{"num": x, "den": 1} for x in w], "candidate": e,
                       "facets": [{"rows": sorted(rr) if isinstance(rr, list) else [rr], "F": sorted(F),
                                   "y": {"num": p, "den": q}} for (rr, F, p, q) in R["facet_duals"]]}
                t2 = time.perf_counter()
                try:
                    b_ok = bool(ck.ref.check_certificate(doc)["objective_optimal"])
                except ck.ref.CertificateError:
                    b_ok = False
                t3 = time.perf_counter()
                second = "reference_flat"
            elif R["kind"] == "tree":
                t0 = time.perf_counter()
                a_ok = ck.bnb(checks=checks, weights=wx, syndrome=syn, error_support=tuple(cand),
                              tree=_deser_tree(R["tree"])).accepted
                t1 = time.perf_counter()
                path = ck.write_tree_doc({"checks": [list(c) for c in checks], "weights": list(w), "syndrome": syn,
                                          "candidate": list(cand), "tree": R["tree"]})
                t2 = time.perf_counter()
                b_ok = ck.tree_b_accepts(path)
                t3 = time.perf_counter()
                second = "tree_checker_b"
            else:
                _fail(fails, "receipt", f"{name} shot {k}: unknown receipt kind {R['kind']!r}")
                continue
            if not (a_ok and b_ok):
                _fail(fails, "receipt", f"{name} shot {k}: {R['kind']} receipt refused (production checker "
                                        f"{a_ok}, {second} {b_ok})")
                continue
            kinds[R["kind"]] += 1
            if timings is not None:
                timings.append({"dataset": name, "shot_index": k, "kind": R["kind"],
                                "tree_nodes": R.get("nodes") if R["kind"] == "tree" else None,
                                "production_checker_s": t1 - t0, "second_checker_s": t3 - t2,
                                "second_checker": second})
            single = R["kind"] == "flat" or R["tree"][0] == "bound"
            duals = None if not single else (fd if R["kind"] == "flat" else _flat_duals(R["tree"][1]))
            for a in ALPHAS:
                sup, ok, obj = ans[a]
                if not ok:
                    want = "not_a_correction"
                elif obj < U:
                    _fail(fails, "optimum", f"{name} shot {k} at {a}: an answer lighter than the certified optimum")
                    want = None
                elif single:
                    t1ok = ck.exact(checks=checks, weights=wx, syndrome=syn,
                                    cert=ck.Cert(error_support=tuple(sup), facet_duals=duals)).accepted
                    if t1ok != (obj == U):
                        _fail(fails, "score", f"{name} shot {k} at {a}: the single dual's verdict {t1ok} is not "
                                              f"'objective equals the optimum' ({obj} vs {U})")
                    want = "tier1" if t1ok else "miss"
                else:
                    want = "optimum_branching" if obj == U else "miss"
                if want is not None and want != r["answers"][a]["score"]:
                    _fail(fails, "score", f"{name} shot {k} at {a}: recomputed score {want}, the record says "
                                          f"{r['answers'][a]['score']}")
                if want is not None:
                    tally[a][want] += 1
                    hits[a].append(want in ("tier1", "optimum_branching"))
            # the logical outcome of every answer, from the archived logical data alone
            if stab is not None:
                fm = _mask(fault)

                def is_logical(sup):
                    res = _mask(sup) ^ fm
                    return any((c & res).bit_count() & 1 for c in cmask) or not stab.contains(res)
                got = {a: is_logical(ans[a][0]) for a in ALPHAS}
                got["receipt"] = is_logical(cand)
                if r.get("logical_error") != got:
                    _fail(fails, "logical", f"{name} shot {k}: recomputed logical errors {got}, the record says "
                                            f"{r.get('logical_error')}")
                for a in ALPHAS:
                    lerr[a].append(got[a])
                receipt_lerr += got["receipt"]
            elif obs is not None:
                truth = sh["observable"]
                got = {a: int(((_mask(ans[a][0]) & obs).bit_count() & 1) == truth) for a in ALPHAS}
                got["receipt"] = int(((_mask(cand) & obs).bit_count() & 1) == truth)
                if r.get("observable_agrees") != got:
                    _fail(fails, "logical", f"{name} shot {k}: recomputed observable agreement {got}, the record "
                                            f"says {r.get('observable_agrees')}")
                for a in ALPHAS:
                    agree[a] += got[a]
            elif "logical_error" in r or "observable_agrees" in r:
                _fail(fails, "logical", f"{name} shot {k}: a logical outcome where the dataset has no logical test")
    # ---- the preregistered analysis, recomputed and compared with summary.json field by field
    sampled = recs[-1]["sample_index"] + 1 if recs else 0
    got = {"dataset": name, "seed": inp["seed"], "sampled": sampled, "trivial": sampled - len(recs),
           "nontrivial": len(recs), "receipts": kinds["flat"] + kinds["tree"], "unresolved": unresolved,
           "receipt_kinds": dict(kinds), "per_configuration": {}}
    if inp["samples"]["sampled"] != sampled:
        _fail(fails, "samples", f"{name}: the archive says {inp['samples']['sampled']} draws, the records {sampled}")
    for a in ALPHAS:
        t = tally[a]
        d = {"optimum_hit": t["tier1"] + t["optimum_branching"], "tier1_single_dual": t["tier1"],
             "optimum_on_branching_tree": t["optimum_branching"], "miss_heavier": t["miss"],
             "not_a_correction": t["not_a_correction"]}
        if stab is not None:
            d["logical_errors"] = sum(lerr[a])
        if obs is not None:
            d["observable_agreement"] = agree[a]
        got["per_configuration"][a] = d
    n10 = sum(1 for x, y in zip(hits["1.0"], hits["0.625"]) if x and not y)
    n01 = sum(1 for x, y in zip(hits["1.0"], hits["0.625"]) if y and not x)
    got["paired_optimum_hit"] = {"hit_at_1.0_only": n10, "hit_at_0.625_only": n01,
                                 "exact_two_sided_mcnemar_p": mcnemar_exact(n10, n01)}
    if stab is not None:
        m10 = sum(1 for x, y in zip(lerr["1.0"], lerr["0.625"]) if x and not y)
        m01 = sum(1 for x, y in zip(lerr["1.0"], lerr["0.625"]) if y and not x)
        got["paired_logical_error_secondary"] = {"error_at_1.0_only": m10, "error_at_0.625_only": m01,
                                                 "exact_two_sided_mcnemar_p": mcnemar_exact(m10, m01)}
        got["receipt_answer_logical_errors"] = receipt_lerr
    stored = {k: v for k, v in summ_entry.items() if k not in ("seconds", "records_sha256", "code_sha256", "deviation")}
    for key in sorted(set(stored) | set(got)):
        if stored.get(key) != got.get(key):
            _fail(fails, "analysis", f"{name}: {key} recomputed {got.get(key)!r}, summary.json says {stored.get(key)!r}")
    return got


def replay(manifest_path: pathlib.Path = MANIFEST, *, pinned: str | None = PINNED_MANIFEST_SHA256,
           root: pathlib.Path = ROOT, datasets: list[str] | None = None,
           timings: list | None = None) -> tuple[bool, list[str], dict]:
    fails: list[str] = []
    before = set(sys.modules)
    body = pathlib.Path(manifest_path).read_bytes()
    if pinned is not None and _sha(body) != pinned:
        return False, [f"[pin] manifest sha256 {_sha(body)} is not the pinned {pinned}"], {}
    man = json.loads(body)
    if man.get("schema") != MANIFEST_SCHEMA:
        return False, [f"[pin] manifest schema {man.get('schema')!r}"], {}
    bound = {man["summary"]["file"]: man["summary"]["sha256"]}
    for name, d in man["datasets"].items():
        bound[d["input"]["file"]] = d["input"]["sha256"]
        bound[d["records"]["file"]] = d["records"]["sha256"]
        bound.update(d["instances"])
    for f, h in sorted(bound.items()):
        p = root / f
        if not p.is_file():
            _fail(fails, "altered", f"bound file missing: {f}")
        elif _sha(p.read_bytes()) != h:
            _fail(fails, "altered", f"bound file altered: {f}")
    if fails:
        return False, fails, {}
    summ = json.loads((root / man["summary"]["file"]).read_text(encoding="utf-8"))
    if summ.get("smoke") is not False or set(summ["datasets"]) != set(man["datasets"]):
        return False, ["[identity] summary.json is a smoke run or names other datasets than the manifest"], {}
    for name, d in man["datasets"].items():
        if summ["datasets"][name]["records_sha256"] != d["records"]["sha256"]:
            _fail(fails, "identity", f"{name}: the bound record file is not the one summary.json records")
    if fails:
        return False, fails, {}
    names = list(summ["datasets"]) if datasets is None else list(datasets)      # the preregistered order
    unknown = [x for x in names if x not in man["datasets"]]
    if unknown:
        return False, [f"[identity] unknown datasets {unknown}"], {}
    ck = Checkers()
    out = {}
    for name in names:
        out[name] = replay_dataset(name, man["datasets"][name], summ["datasets"][name], summ["seed_domain"], root,
                                   ck, fails, timings)
    loaded = forbidden_loaded(set(sys.modules) - before)
    if loaded:
        _fail(fails, "import", f"producer-side modules loaded during the replay: {loaded}")
    if datasets is None:
        receipts = sum(v.get("receipts", 0) for v in out.values())
        if receipts != man["expected"]["nontrivial"] or len(out) != man["expected"]["datasets"]:
            _fail(fails, "analysis", f"{receipts} receipts over {len(out)} datasets; the manifest expects "
                                     f"{man['expected']['nontrivial']} over {man['expected']['datasets']}")
    return not fails, fails, out


def nearest_rank(values: list[float]) -> dict:
    """Nearest-rank quantiles: the ceil(q N)-th smallest value; the median of an even count is the lower middle."""
    v = sorted(values)
    out = {k: v[max(1, -(-int(q * 1000) * len(v) // 1000)) - 1] for k, q in (("median", 0.5), ("p90", 0.9),
                                                                                 ("p99", 0.99))}
    out["max"], out["n"] = v[-1], len(v)
    return out


def latency_report(timings: list[dict]) -> dict:
    """Per dataset, columnar series (data, not claims) and nearest-rank quantiles (the scalars a text may quote)."""
    by: dict[str, list[dict]] = {}
    for t in timings:
        by.setdefault(t["dataset"], []).append(t)
    series, summary = {}, {}
    for name, ts in by.items():
        both = [t["production_checker_s"] + t["second_checker_s"] for t in ts]
        series[name] = {"shot_index": [t["shot_index"] for t in ts], "kind": [t["kind"] for t in ts],
                        "production_checker_s": [t["production_checker_s"] for t in ts],
                        "second_checker_s": [t["second_checker_s"] for t in ts], "both_s": both}
        summary[name] = {"both": nearest_rank(both),
                         "production_checker": nearest_rank([t["production_checker_s"] for t in ts]),
                         "second_checker": nearest_rank([t["second_checker_s"] for t in ts])}
    allb = [t["production_checker_s"] + t["second_checker_s"] for t in timings]
    summary["all"] = {"both": nearest_rank(allb)} if allb else {}
    trees = [{"dataset": t["dataset"], "shot_index": t["shot_index"], "nodes": t["tree_nodes"],
              "production_checker_s": t["production_checker_s"], "second_checker_s": t["second_checker_s"]}
             for t in timings if t["kind"] == "tree"]
    return {"series": series, "summary_s": summary, "tree_receipts": trees}


def _environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "threads": "one Python thread; no solver, no native library"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--dataset", action="append", default=None, help="replay one dataset (repeatable): SUBSET")
    ap.add_argument("--out", default=None, help="write the recomputed analysis and per-receipt checking latency")
    a = ap.parse_args()
    timings: list = []
    t0 = time.perf_counter()
    ok, fails, out = replay(pathlib.Path(a.manifest), datasets=a.dataset, timings=timings)
    wall = time.perf_counter() - t0
    whole = forbidden_loaded(set(sys.modules))     # a fresh interpreter: nothing producer-side may be loaded at all
    if whole:
        ok = False
        fails.append(f"[import] producer-side modules in this interpreter: {whole}")
    for f in fails[:40]:
        print("  FAIL", f)
    if len(fails) > 40:
        print(f"  ... and {len(fails) - 40} more")
    label = "SUBSET" if a.dataset else "COMPLETE"
    if a.out and ok:
        man_sha = _sha(pathlib.Path(a.manifest).read_bytes())
        rep = {"schema": REPORT_SCHEMA, "verdict": f"{label} REPLAY PASSED", "manifest_sha256": man_sha,
               "tool_sha256": _sha(pathlib.Path(__file__).read_bytes()),
               "checkers_sha256": {(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):
                                   _sha(p.read_bytes()) for p in (PRODUCTION_CHECKER, REFERENCE_CHECKER, TREE_CHECKER_B)},
               "datasets": out, "receipts_checked": len(timings), "wall_seconds": wall,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - wall)),
               "timing": {"what": "wall clock (time.perf_counter) of each checker's own call on each receipt; the "
                                  "second tree checker is called in process on a document file this tool writes "
                                  "first, and its time includes reading and parsing that file",
                          "quantile_rule": "nearest rank: the ceil(q N)-th smallest; median = q 0.5",
                          "environment": _environment()},
               **latency_report(timings)}
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(rep, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if ok:
        n = sum(v.get("receipts", 0) for v in out.values())
        print(f"{label} REPLAY PASSED: {n} receipts of {len(out)} datasets re-checked by both of their checkers, "
              f"re-scored and their analysis recomputed from the archived inputs, no producer ({wall:.1f} s)")
        return 0
    print(f"{label} REPLAY FAILED ({len(fails)} problem(s))")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
