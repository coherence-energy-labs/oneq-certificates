"""ONE input domain, seven qLDPC checkers: docs/QLDPC_RECEIPT_INPUT_DOMAIN.md, enforced and pinned.

External audit 2 (2026-11-03, F05/F06) and its adjudication found the seven implementations reading the same
malformed input seven ways: bare int() in the production checkers (0.9 -> 0, True -> 1, "1" -> 1, out-of-range check
indices accepted), `& 1` on the syndrome in tree checker B (2 -> 0, -1 -> 1), and no way for the JavaScript checkers
to tell 1.0 from 1. Every vector of conformance/qldpc_input_domain_vectors.json runs here through all seven, and every
applicable implementation must reach the vector's single expected verdict -- so a future divergence fails.

The same file pins the two other contract clauses: the root guard (a root bound above a verified-feasible candidate's
weight is an instrument failure in every tree checker) and bound-only mode's LOWER_BOUND_PROVEN result.

FAIL-BEFORE. Set ONEQ_QLDPC_IMPL_ROOT to a directory holding older copies of src/oneq/qldpc_check.py,
tools/external/{exact_lp_certificate_reference,tree_checker_b}.py and verifier-js/verify-qldpc-{flat,tree}.mjs and the
same tests run against those copies instead (the release v2026.11.03 copies fail them).
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = pathlib.Path(os.environ.get("ONEQ_QLDPC_IMPL_ROOT", str(ROOT))).resolve()
VECTORS = json.loads((ROOT / "conformance" / "qldpc_input_domain_vectors.json").read_text(encoding="utf-8"))
BASES = VECTORS["bases"]
NODE = shutil.which("node")


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if IMPL == ROOT:
    import oneq.qldpc_check as P            # the module the mutation sandbox swaps in
else:
    P = _load("qldpc_check_under_test", IMPL / "src" / "oneq" / "qldpc_check.py")
R = _load("qldpc_reference_under_test", IMPL / "tools" / "external" / "exact_lp_certificate_reference.py")
TREE_B = IMPL / "tools" / "external" / "tree_checker_b.py"
JS = {"JS-flat": IMPL / "verifier-js" / "verify-qldpc-flat.mjs", "JS-tree": IMPL / "verifier-js" / "verify-qldpc-tree.mjs"}


# ----------------------------------------------------------------------------- instances
def instance(vec: dict) -> dict:
    inst = copy.deepcopy(BASES[vec["base"]])
    path = vec["path"]
    if path is not None:
        obj = inst
        for k in path[:-1]:
            obj = obj[k]
        obj[path[-1]] = copy.deepcopy(vec["value"])
    return inst


def tree_of(inst: dict):
    return inst["tree"] if "tree" in inst else ["bound", inst["facets"]]


def _plain(x) -> bool:
    return type(x) is int


# ----------------------------------------------------------------------------- adapters (raw values untouched)
def _p_facet(f):
    r, F, num, den = f
    if _plain(num) and _plain(den) and den != 0:
        y = Fraction(num, den)
    elif _plain(den) and den == 1:
        y = num                              # the Python API takes y itself: probe the raw value there
    else:
        raise LookupError("no slot")
    return (tuple(r) if isinstance(r, list) else r, tuple(F), y)


def _p_tree(t):
    if not isinstance(t, list) or not t:
        return t
    if t[0] == "branch" and len(t) == 4:
        return ("branch", t[1], _p_tree(t[2]), _p_tree(t[3]))
    if t[0] == "bound" and len(t) == 2:
        return ("bound", tuple(_p_facet(f) for f in t[1]))
    if t[0] == "infeasible" and len(t) == 2:
        return ("infeasible", tuple(t[1]) if isinstance(t[1], list) else t[1])
    return tuple(t)


def run_python(impl: str, inst: dict) -> bool:
    checks = [tuple(c) for c in inst["checks"]]
    weights = {i: v for i, v in enumerate(inst["w"])}
    if impl == "P-tree":
        r = P.check_qldpc_bnb(checks=checks, weights=weights, syndrome=list(inst["s"]),
                              error_support=tuple(inst["cand"]), tree=_p_tree(tree_of(inst)))
        return r.accepted
    cert = P.QldpcCertificate(error_support=tuple(inst["cand"]),
                              facet_duals=tuple(_p_facet(f) for f in inst["facets"]), box_duals=())
    fn = P.check_qldpc if impl == "P-screen" else P.check_qldpc_exact
    return fn(checks=checks, weights=weights, syndrome=list(inst["s"]), cert=cert).accepted


def run_reference(vec: dict, inst: dict) -> bool:
    """The reference takes a dense document: H and the candidate as bit matrices. A probed check entry or candidate
    entry becomes the probed BIT at the base's position -- the one place this format holds that slot."""
    base = BASES[vec["base"]]
    n = base["n"]
    H = [[0] * n for _ in base["checks"]]
    for j, c in enumerate(base["checks"]):
        for i in c:
            H[j][i] = 1
    e = [0] * n
    for i in base["cand"]:
        e[i] = 1
    doc = {"H": H, "s": copy.deepcopy(inst["s"]), "weights": [{"num": v, "den": 1} for v in inst["w"]],
           "candidate": e,
           "facets": [{"rows": (sorted(f[0]) if isinstance(f[0], list) and all(_plain(x) for x in f[0])
                                else f[0] if isinstance(f[0], list) else [f[0]]),
                       "F": (sorted(f[1]) if all(_plain(x) for x in f[1]) else list(f[1])),
                       "y": {"num": f[2], "den": f[3]}} for f in inst["facets"]]}
    path = vec["path"]
    if path and path[0] == "checks":
        doc["H"][path[1]][base["checks"][path[1]][path[2]]] = vec["value"]
    if path and path[0] == "cand" and len(path) == 2:
        doc["candidate"][base["cand"][path[1]]] = vec["value"]
    try:
        return bool(R.check_certificate(doc)["objective_optimal"])
    except R.CertificateError:
        return False


def run_tree_b(inst: dict, tmp: pathlib.Path) -> bool:
    doc = {"checks": inst["checks"], "weights": inst["w"], "syndrome": inst["s"], "candidate": inst["cand"],
           "tree": tree_of(inst)}
    p = tmp / "doc.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    mod = _load("tree_checker_b_under_test", TREE_B)
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            code = mod.main(str(p))
    except SystemExit as exc:
        code = exc.code
    except Exception:                          # the script's __main__ maps any exception to MALFORMED, exit 1
        code = 1
    return code == 0


def js_support(vec: dict, inst: dict) -> list:
    """JavaScript batches carry the syndrome as a SUPPORT list, so a probed syndrome entry becomes the probed INDEX
    at that position of the support -- out of the domain in that format exactly when it is out of it here."""
    base = BASES[vec["base"]]
    sup = [j for j, b in enumerate(base["s"]) if b == 1]
    path = vec["path"]
    if path and path[0] == "s":
        if path[1] in sup:
            return [vec["value"] if j == path[1] else j for j in sup]
        return sup + [vec["value"]]
    return sup


def js_batch(impl: str, vecs: list) -> dict:
    """One node run per implementation for every applicable vector: id -> accepted."""
    receipts, insts, weights = [], {}, {}
    for v in vecs:
        inst = instance(v)
        rec = {"id": v["id"], "instance": v["id"], "weights": v["id"], "syndrome_support": js_support(v, inst),
               "candidate": inst["cand"], "expect": "accept"}
        if impl == "JS-tree":
            rec["tree"] = tree_of(inst)
        else:
            rec["facet_duals"] = inst["facets"]
        insts[v["id"]] = {"n": inst["n"], "checks": inst["checks"]}
        weights[v["id"]] = inst["w"]
        receipts.append(rec)
    return _node(impl, {"instances": insts, "weights": weights, "receipts": receipts})


def _node(impl: str, batch: dict, text: str | None = None) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "batch.json"
        p.write_text(text if text is not None else json.dumps(batch), encoding="utf-8")
        r = subprocess.run([NODE, str(JS[impl]), str(p)], capture_output=True, text=True, timeout=120)
    got = {}
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] != "SUMMARY":
            got[parts[0]] = parts[1] == "ACCEPT"
    return got


# ----------------------------------------------------------------------------- applicability and expectation
def applies(impl: str, vec: dict) -> bool:
    if impl in vec.get("not_expressible", {}):
        return False
    is_tree = "tree" in BASES[vec["base"]]
    if is_tree and impl in ("P-flat", "P-screen", "R-flat", "JS-flat"):
        return False
    return True


def expected(impl: str, vec: dict) -> bool:
    if impl == "P-screen" and "expect_screen" in vec:
        return vec["expect_screen"] == "accept"
    return vec["expect"] == "accept"


@pytest.fixture(scope="module")
def js_verdicts():
    if NODE is None:
        pytest.skip("node is not installed: the two JavaScript checkers cannot be run here")
    out = {}
    for impl in JS:
        vecs = [v for v in VECTORS["vectors"] if applies(impl, v)]
        got = js_batch(impl, vecs)
        # A checker that dies on one malformed instance drops the whole batch (the pre-contract JavaScript did, at
        # instance validation). Re-run what is missing one vector at a time: a run that still prints no verdict
        # ended without accepting, and is read as a refusal -- never as an acceptance, never as a skip.
        for v in [v for v in vecs if v["id"] not in got]:
            got[v["id"]] = js_batch(impl, [v]).get(v["id"], False)
        controls = [v["id"] for v in vecs if v["field"] == "control"]
        assert all(got[c] for c in controls), f"{impl} refused a control: the harness, not the vectors, is broken"
        out[impl] = got
    return out


def verdict(impl: str, vec: dict, tmp: pathlib.Path, js: dict) -> bool:
    inst = instance(vec)
    if impl.startswith("P-"):
        return run_python(impl, inst)
    if impl == "R-flat":
        return run_reference(vec, inst)
    if impl == "R-tree":
        return run_tree_b(inst, tmp)
    return js[impl][vec["id"]]


def test_the_vector_file_is_the_contracts():
    assert VECTORS["schema"] == "oneq-qldpc-input-domain-vectors/1"
    assert VECTORS["implementations"] == ["P-flat", "P-tree", "P-screen", "R-flat", "R-tree", "JS-flat", "JS-tree"]
    ids = [v["id"] for v in VECTORS["vectors"]]
    assert len(ids) == len(set(ids)) == 109
    # the expectations are not all one verdict: a checker that refuses everything must fail this file
    assert sum(v["expect"] == "accept" for v in VECTORS["vectors"]) == 15
    assert sum(v["field"] == "control" for v in VECTORS["vectors"]) == len(BASES)


@pytest.mark.parametrize("vec", VECTORS["vectors"], ids=lambda v: f"{v['id']}-{v['field']}-{v['value']!r}")
def test_every_implementation_reaches_the_one_expected_verdict(vec, tmp_path, js_verdicts):
    wrong = []
    for impl in VECTORS["implementations"]:
        if not applies(impl, vec):
            continue
        got = verdict(impl, vec, tmp_path, js_verdicts)
        if got != expected(impl, vec):
            wrong.append(f"{impl}: {'ACCEPT' if got else 'REFUSE'}")
    assert not wrong, f"{vec['field']} {vec['value']!r} on {vec['base']} (expect {vec['expect']}): {wrong}"


@pytest.mark.parametrize("lexeme", ["1.0", "1e0", "1E0", "10e-1"])
def test_javascript_reads_the_token_and_refuses_a_non_integer_literal(lexeme, js_verdicts):
    # Python's json turns every one of these into a float, which the Python checkers refuse by type; JavaScript
    # cannot see the difference in the Number, so it must read the token. B1's facet numerator is the probed slot.
    for impl in JS:
        base = BASES["B1"]
        rec = {"id": "x", "instance": "I", "weights": "W", "syndrome_support": [0], "candidate": [1],
               "expect": "accept"}
        if impl == "JS-tree":
            rec["tree"] = ["bound", [[0, [], 7777, 1]]]
        else:
            rec["facet_duals"] = [[0, [], 7777, 1]]
        batch = {"instances": {"I": {"n": base["n"], "checks": base["checks"]}}, "weights": {"W": base["w"]},
                 "receipts": [rec]}
        good = _node(impl, batch, json.dumps(batch).replace("7777", "1"))
        bad = _node(impl, batch, json.dumps(batch).replace("7777", lexeme))
        assert good == {"x": True}, (impl, good)
        assert bad == {"x": False}, (impl, lexeme, bad)


# ----------------------------------------------------------------------------- the root guard
ONE_NODE = dict(checks=[(0, 1)], weights={0: Fraction(1), 1: Fraction(1)}, syndrome=[1], error_support=(1,),
                tree=("bound", ((0, (), Fraction(1)),)))


def test_root_guard_production_tree_refuses_an_inflated_root_bound(monkeypatch):
    # L <= OPT <= U holds for any valid dual, so the guard can fire only on a broken bound: inject one.
    assert P.check_qldpc_bnb(**ONE_NODE).accepted, "control: the genuine one-node tree is accepted"
    real = P._exact_safe_dual_bound

    def inflated(*a, **k):
        L, err = real(*a, **k)
        return (None, err) if L is None else (L + 10, err)
    monkeypatch.setattr(P, "_exact_safe_dual_bound", inflated)
    r = P.check_qldpc_bnb(**ONE_NODE)
    assert not r.accepted
    assert "IMPOSSIBLE" in r.reason and "root bound" in r.reason, r.reason


def test_root_guard_tree_checker_b_refuses_an_inflated_root_bound(tmp_path):
    src = TREE_B.read_text(encoding="utf-8")
    anchor = "        return by - z\n"
    assert src.count(anchor) == 1, "the bound's return line moved: update this fault injection"
    broken = tmp_path / "tree_checker_b_inflated.py"
    broken.write_text(src.replace(anchor, "        return by - z + 10\n"), encoding="utf-8")
    doc = tmp_path / "one_node.json"
    doc.write_text(json.dumps({"checks": [[0, 1]], "weights": [1, 1], "syndrome": [1], "candidate": [1],
                               "tree": ["bound", [[0, [], 1, 1]]]}), encoding="utf-8")
    ok = subprocess.run([sys.executable, "-B", str(TREE_B), str(doc)], capture_output=True, text=True, timeout=60)
    assert ok.returncode == 0, ("control", ok.stdout)
    bad = subprocess.run([sys.executable, "-B", str(broken), str(doc)], capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "root bound" in bad.stdout, bad.stdout


# ----------------------------------------------------------------------------- bound-only mode
def test_bound_only_on_an_optimum_2_instance_proves_a_bound_and_claims_no_optimum():
    # x0 = 1 is the only solution, weight 2; the tree proves every solution weighs more than T = 0
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(),
                          tree=("bound", ((0, (), 2),)), bound_only_threshold=0)
    assert r.accepted, r.reason
    assert r.checks["status"] == "LOWER_BOUND_PROVEN"
    assert r.primal_weight is None
    assert "EXACT-OPTIMAL" not in r.reason and "objective" not in r.checks
    assert r.checks["threshold"] == {"num": 0, "den": 1}
    assert r.checks["lower_bound"] == {"num": 1, "den": 1}     # (floor(0*1) + 1)/1, below the true optimum 2


def test_bound_only_on_an_infeasible_instance_reports_no_finite_objective():
    r = P.check_qldpc_bnb(checks=[()], weights={0: 2}, syndrome=[1], error_support=(),
                          tree=("infeasible", 0), bound_only_threshold=0)
    assert r.accepted, r.reason
    assert r.checks["status"] == "LOWER_BOUND_PROVEN"
    assert r.primal_weight is None and r.dual_bound is None
    assert r.checks["feasibility"] == "not established"


def test_bound_only_lattice_rounds_a_threshold_off_the_lattice():
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(),
                          tree=("bound", ((0, (), 2),)), bound_only_threshold=Fraction(1, 2))
    assert r.accepted and r.checks["lower_bound"] == {"num": 1, "den": 1}, r.checks


def test_bound_only_refuses_a_candidate_and_a_false_threshold():
    kw = dict(checks=[(0,)], weights={0: 2}, syndrome=[1], tree=("bound", ((0, (), 2),)))
    assert not P.check_qldpc_bnb(**kw, error_support=(0,), bound_only_threshold=0).accepted
    r = P.check_qldpc_bnb(**kw, error_support=(), bound_only_threshold=2)     # min > 2 is false
    assert not r.accepted and r.primal_weight is None, r


def test_bound_only_reports_its_fields_exactly_and_rounds_on_a_fractional_lattice():
    # weights in (1/2)Z: D = 2, and "every solution weighs more than 1" means "at least 3/2", not 1 + 1 nor 1/2
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: Fraction(3, 2)}, syndrome=[1], error_support=(),
                          tree=("bound", ((0, (), Fraction(3, 2)),)), bound_only_threshold=1)
    assert r.accepted, r.reason
    assert r.checks["exact"] is True and r.checks["bnb"] is True and r.checks["lattice_denominator"] == 2
    assert r.checks["threshold"] == {"num": 1, "den": 1}
    assert r.checks["lower_bound"] == {"num": 3, "den": 2}, r.checks


# ----------------------------------------------------------------------------- the validator's own shape rules
# Each case is malformed in exactly the way one rule of _domain_refusal guards: the rule must REFUSE with its own
# words, where without it the checker either raises (the domain check is outside the exact path's try) or refuses
# later for another reason.
FLAT1 = dict(checks=[(0,)], weights={0: Fraction(1)}, syndrome=[1])


def test_a_candidate_that_is_a_string_is_not_a_list():
    r = P.check_qldpc_exact(**FLAT1, cert=P.QldpcCertificate(error_support="0", facet_duals=((0, (), 1),)))
    assert not r.accepted and "candidate is not a list" in r.reason, r.reason


def test_a_facet_with_four_fields_is_refused_not_unpacked():
    r = P.check_qldpc_exact(**FLAT1, cert=P.QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1, 1),)))
    assert not r.accepted and "is not (check, F, y)" in r.reason, r.reason


def test_a_malformed_node_does_not_stop_the_tree_scan():
    # the malformed child is scanned first (stack order); the bad row in the other child must still be found
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 1}, syndrome=[1], error_support=(0,),
                          tree=("branch", 0, ("bound", ((0.9, (), 1),)), ()))
    assert not r.accepted and r.reason.startswith("INPUT DOMAIN") and "0.9" in r.reason, r.reason


def test_a_three_field_branch_is_left_to_the_walk_to_name():
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 1}, syndrome=[1], error_support=(0,),
                          tree=("branch", 0, ("bound", ())))
    assert not r.accepted and "malformed branch" in r.reason, r.reason


def test_a_three_field_infeasible_leaf_is_left_to_the_walk_to_name():
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 1}, syndrome=[1], error_support=(0,),
                          tree=("branch", 0, ("infeasible", 0.9, 1), ("bound", ())))
    assert not r.accepted and "malformed infeasible leaf" in r.reason, r.reason


def test_a_tree_multiplier_type_is_the_exact_paths_rule_alone():
    # the exact path refuses a non-rational y by TYPE in _exact_rational; the domain check must not restate it
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 1}, syndrome=[1], error_support=(0,),
                          tree=("bound", ((0, (), "1"),)))
    assert not r.accepted and r.reason.startswith("REJECTED MALFORMED INPUT"), r.reason


@pytest.mark.parametrize("box, needle", [(((0, 1.0, 2),), "is not (i, mu)"),
                                         (((0, "x"),), "box dual mu 'x' is not a real number")])
def test_the_screen_refuses_a_malformed_box_dual(box, needle):
    r = P.check_qldpc(checks=[(0,)], weights={0: 1.0}, syndrome=[1],
                      cert=P.QldpcCertificate(error_support=(0,), facet_duals=((0, (), 1.0),), box_duals=box))
    assert not r.accepted and needle in r.reason, r.reason


def test_candidate_mode_still_says_exact_optimal():
    r = P.check_qldpc_bnb(checks=[(0,)], weights={0: 2}, syndrome=[1], error_support=(0,),
                          tree=("bound", ((0, (), 2),)))
    assert r.accepted and r.checks["status"] == "EXACT_OPTIMAL" and r.checks["objective"] == {"num": 2, "den": 1}
