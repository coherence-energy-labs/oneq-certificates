r"""Operator mutation score for one module: how much of its behaviour a test set pins.

The mutation gate (tools/mutation_gate.py) holds hand-written mutants, each aimed
at a named test: it shows the tests catch the defects someone thought of. This
tool answers the other question -- what fraction of MECHANICALLY generated
defects a fixed test set catches -- by applying standard operators at every
eligible site of one module and running the tests against each mutant in a
sandbox copy of the tree.

Operators, one site per mutant:
  ROR  relational      <  <=  >  >=  ==  !=  in / not in  is / is not
  LCR  logical         and <-> or
  UOD  unary           remove a `not`
  AOR  arithmetic      + <-> -   * <-> /   % -> //   // -> *   ^ & | rotated
  CRP  constant        0 <-> 1, other ints n -> n+1, True <-> False
  NEG  condition       negate the test of an `if` or `while`
  MMS  min <-> max
  BRK  continue <-> break

A mutant is spliced into the ORIGINAL bytes at the node's exact span (UTF-8
column offsets), so every other byte -- comments included -- is unchanged, and it
is kept only if the spliced file parses to exactly the intended tree. Sites in
f-strings, annotations and decorators are skipped, as are string and float
constants.

Every run must be able to fail: the unmutated baseline must pass in every
sandbox, the target module the tests import must be the sandbox's copy, and a
score with nothing killed is refused as dark.

    python tools/mutation_score.py src/oneq/qldpc_check.py \
        --tests tests/test_qldpc_exact.py tests/test_qldpc_bnb.py \
        --copy evidence/qldpc_heldout_flat_certs.json --workers 4
"""
from __future__ import annotations

import argparse
import ast
import copy
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

BASE_COPY = ("conftest.py", "pyproject.toml", "src", "tools", "conformance", "experiments/m0_prior_art")

ROR = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
       ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
AOR = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.Mod: ast.FloorDiv,
       ast.FloorDiv: ast.Mult, ast.BitXor: ast.BitOr, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd}


@dataclasses.dataclass(frozen=True)
class Site:
    operator: str
    kind: str            # node class name
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    detail: str          # e.g. "Lt->LtE" or "0->1"
    index: int = 0       # which op of a chained comparison


def _parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node          # noqa: SLF001


def _skipped(node) -> bool:
    """Inside an f-string, an annotation or a decorator."""
    child, parent = node, getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.JoinedStr):
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child in parent.decorator_list:
            return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is parent.returns:
            return True
        if isinstance(parent, ast.arg) and child is parent.annotation:
            return True
        if isinstance(parent, ast.AnnAssign) and child is parent.annotation:
            return True
        child, parent = parent, getattr(parent, "_parent", None)
    return False


def _pos(node):
    return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset


def sites(source: str) -> list[Site]:
    tree = ast.parse(source)
    _parents(tree)
    out: list[Site] = []
    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or _skipped(node):
            continue
        p = _pos(node)
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in ROR:
                    out.append(Site("ROR", "Compare", *p, f"{type(op).__name__}->{ROR[type(op)].__name__}", k))
        elif isinstance(node, ast.BoolOp):
            out.append(Site("LCR", "BoolOp", *p, "And->Or" if isinstance(node.op, ast.And) else "Or->And"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append(Site("UOD", "UnaryOp", *p, "remove not"))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in AOR:
            out.append(Site("AOR", type(node).__name__, *p, f"{type(node.op).__name__}->{AOR[type(node.op)].__name__}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
            v = node.value
            new = (not v) if isinstance(v, bool) else (1 if v == 0 else 0 if v == 1 else v + 1)
            out.append(Site("CRP", "Constant", *p, f"{v!r}->{new!r}"))
        elif isinstance(node, (ast.If, ast.While)):
            out.append(Site("NEG", type(node).__name__, *_pos(node.test), "negate condition"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            out.append(Site("MMS", "Name", *_pos(node.func), "min->max" if node.func.id == "min" else "max->min"))
        elif isinstance(node, (ast.Continue, ast.Break)):
            out.append(Site("BRK", type(node).__name__, *p, "Continue->Break" if isinstance(node, ast.Continue) else "Break->Continue"))
    return sorted(set(out), key=lambda s: (s.lineno, s.col, s.operator, s.index, s.detail))


class _Apply(ast.NodeTransformer):
    """The intended tree: the site's node replaced, everything else untouched."""

    def __init__(self, site: Site):
        self.site = site
        self.hits = 0
        self.replacement = None

    def _match(self, node, kind):
        s = self.site
        return (type(node).__name__ == kind and hasattr(node, "lineno")
                and _pos(node) == (s.lineno, s.col, s.end_lineno, s.end_col))

    def generic_visit(self, node):
        node = super().generic_visit(node)
        s = self.site
        if s.operator in ("NEG",) and isinstance(node, (ast.If, ast.While)) and self._match(node.test, type(node.test).__name__):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.replacement = node.test
            self.hits += 1
            return node
        if s.operator == "MMS" and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and self._match(node.func, "Name"):
            node.func = ast.Name(id="max" if node.func.id == "min" else "min", ctx=ast.Load())
            self.replacement = node.func
            self.hits += 1
            return node
        if not self._match(node, s.kind) or s.operator in ("NEG", "MMS"):
            return node
        self.hits += 1
        if s.operator == "ROR":
            node.ops = list(node.ops)
            node.ops[s.index] = ROR[type(node.ops[s.index])]()
        elif s.operator == "LCR":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif s.operator == "UOD":
            node = node.operand
        elif s.operator == "AOR":
            node.op = AOR[type(node.op)]()
        elif s.operator == "CRP":
            v = node.value
            node = ast.Constant(value=(not v) if isinstance(v, bool) else (1 if v == 0 else 0 if v == 1 else v + 1))
        elif s.operator == "BRK":
            node = ast.Break() if isinstance(node, ast.Continue) else ast.Continue()
        self.replacement = node
        return node


def _line_offsets(data: bytes) -> list[int]:
    offs = [0]
    for i, b in enumerate(data):
        if b == 10:
            offs.append(i + 1)
    return offs


def mutate(source: str, site: Site) -> str | None:
    """The mutated source, or None when the splice cannot reproduce the intended tree."""
    tree = ast.parse(source)
    t = _Apply(site)
    want = t.visit(copy.deepcopy(tree))
    if t.hits != 1 or t.replacement is None:
        return None
    node = t.replacement
    if isinstance(node, ast.stmt):
        text = ast.unparse(node)
    else:
        text = "(" + ast.unparse(node) + ")"
    data = source.encode("utf-8")
    offs = _line_offsets(data)
    a = offs[site.lineno - 1] + site.col
    b = offs[site.end_lineno - 1] + site.end_col
    out = (data[:a] + text.encode("utf-8") + data[b:]).decode("utf-8")
    try:
        got = ast.parse(out)
    except SyntaxError:
        return None
    if ast.dump(got) != ast.dump(ast.fix_missing_locations(want)):
        return None
    return out


# ------------------------------------------------------------------ running
def _sandbox(base: pathlib.Path, copies: list[str], tests: list[str]) -> pathlib.Path:
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for rel in list(BASE_COPY) + copies + tests:
        src = ROOT / rel
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        elif src.exists():
            shutil.copy2(src, dst)
        else:
            raise SystemExit(f"--copy path does not exist: {rel}")
    return base


def _guard_test(target_rel: str) -> str:
    mod = target_rel[len("src/"):-3].replace("/", ".")
    return (
        "import importlib, pathlib\n\n"
        "def test_the_target_is_the_sandbox_copy():\n"
        f"    m = importlib.import_module({mod!r})\n"
        "    here = pathlib.Path(__file__).resolve().parents[1]\n"
        "    assert pathlib.Path(m.__file__).resolve().is_relative_to(here), m.__file__\n")


def _pytest(box: pathlib.Path, tests: list[str], timeout: float):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}
    cmd = [sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "--no-header", *tests]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=box, env=env, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return "timeout", time.perf_counter() - t0, ""
    return r.returncode, time.perf_counter() - t0, (r.stdout[-1500:] + r.stderr[-800:])


def score(target_rel: str, tests: list[str], copies: list[str], workers: int, out: pathlib.Path | None,
          limit: int | None = None, equivalents: dict | None = None) -> dict:
    source = (ROOT / target_rel).read_text(encoding="utf-8")
    all_sites = sites(source)
    mutants, unsplicable = [], []
    for s in all_sites:
        m = mutate(source, s)
        (mutants if m is not None else unsplicable).append((s, m))
    if limit:
        mutants = mutants[:limit]
    tmp = pathlib.Path(tempfile.gettempdir()) / "oneq_mutation_score"
    boxes = []
    guard = "tests/test__mutation_score_guard.py"
    for w in range(workers):
        box = _sandbox(tmp / f"{pathlib.Path(target_rel).stem}_w{w}", copies, tests)
        (box / guard).write_text(_guard_test(target_rel), encoding="utf-8")
        boxes.append(box)
    run_tests = [guard] + tests
    base_times = []
    for box in boxes:
        code, secs, tail = _pytest(box, run_tests, timeout=900)
        if code != 0:
            raise SystemExit(f"baseline does not pass in {box} (exit {code}):\n{tail}")
        base_times.append(secs)
    timeout = max(60.0, 10 * max(base_times))

    import queue
    free = queue.Queue()
    for box in boxes:
        free.put(box)

    def run_one(item):
        site, text = item
        box = free.get()
        try:
            path = box / target_rel
            path.write_text(text, encoding="utf-8", newline="")
            code, secs, tail = _pytest(box, run_tests, timeout)
            path.write_text(source, encoding="utf-8", newline="")
        finally:
            free.put(box)
        if code == 5 or code in (2, 3, 4):
            return site, "error", secs, tail
        return site, ("timeout" if code == "timeout" else "killed" if code != 0 else "survived"), secs, tail

    results = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(run_one, mutants), 1):
            results.append(res)
            if k % 25 == 0:
                print(f"  {k}/{len(mutants)}", flush=True)
    errors = [r for r in results if r[1] == "error"]
    if errors:
        s, _st, _secs, tail = errors[0]
        raise SystemExit(f"{len(errors)} mutant runs ended in a pytest error, first at line {s.lineno}:\n{tail}")
    counts = {k: sum(1 for r in results if r[1] == k) for k in ("killed", "survived", "timeout")}
    if counts["killed"] == 0:
        raise SystemExit("no mutant was killed: the run is dark")
    by_op = {}
    for s, status, _secs, _tail in results:
        d = by_op.setdefault(s.operator, {"killed": 0, "survived": 0, "timeout": 0})
        d[status] += 1
    lines = source.splitlines()
    eq = equivalents or {}

    def key(s):
        return f"{s.operator}|{s.detail}|{lines[s.lineno - 1].strip()}"

    # a declaration is a claim: one that a test kills, or that matches no mutant, fails the run
    killed_eq = [key(s) for s, status, _secs, _tail in results if status != "survived" and key(s) in eq]
    if killed_eq:
        raise SystemExit(f"declared equivalent but killed, so not equivalent: {killed_eq[:3]}")
    stale = sorted(set(eq) - {key(s) for s, *_rest in results})
    if stale and limit is None:
        raise SystemExit(f"declared equivalents that match no mutant (stale): {stale[:3]}")
    survivors = []
    for s, status, _secs, _tail in results:
        if status == "survived":
            mut = next(m for ss, m in mutants if ss == s).splitlines()
            survivors.append({**dataclasses.asdict(s), "original": lines[s.lineno - 1].strip(),
                              "mutant": mut[s.lineno - 1].strip() if s.lineno - 1 < len(mut) else "",
                              "equivalent": eq.get(key(s))})
    n_eq = sum(1 for x in survivors if x["equivalent"])

    def lf_sha256(p: pathlib.Path) -> str:
        # CRLF normalised to LF: the bytes git stores for a text file, so the pin verifies in
        # every clone. 2026-09-15: two killing tests were pinned by their CRLF working-tree bytes,
        # and a fresh clone would have refused to typeset paper 2's score.
        return hashlib.sha256(p.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()

    doc = {
        "schema": "oneq-mutation-score/1",
        "target": target_rel,
        "digest_domain": "sha256-lf",
        "target_sha256": lf_sha256(ROOT / target_rel),
        "tests": {t: lf_sha256(ROOT / t) for t in tests},
        "copied": list(BASE_COPY) + copies,
        "operators": sorted(by_op),
        "sites": len(all_sites),
        "mutants_run": len(mutants),
        "unsplicable": len(unsplicable),
        **counts,
        "score_killed_over_run": counts["killed"] / len(mutants),
        "equivalent_declared": n_eq,
        "score_excluding_equivalents": counts["killed"] / (len(mutants) - n_eq),
        "by_operator": by_op,
        "survivors": survivors,
        "baseline_seconds": [round(x, 2) for x in base_times],
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "command": "python tools/mutation_score.py " + " ".join(sys.argv[1:]),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    for box in boxes:
        shutil.rmtree(box, ignore_errors=True)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="module path relative to the repository, under src/")
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--copy", nargs="*", default=[], help="extra files or directories the tests read")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N mutants only (smoke run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--equivalents", default=None,
                    help="JSON object: 'OPERATOR|detail|original line' -> why that mutant is equivalent")
    a = ap.parse_args()
    target = a.target.replace("\\", "/")
    if not target.startswith("src/") or not target.endswith(".py"):
        raise SystemExit("target must be a module under src/")
    out = pathlib.Path(a.out) if a.out else (ROOT / "evidence" / "mutation_score" / f"{pathlib.Path(target).stem}.json")
    eqs = json.loads(pathlib.Path(a.equivalents).read_text(encoding="utf-8")) if a.equivalents else None
    doc = score(target, a.tests, a.copy, a.workers, None if a.limit else out, a.limit, eqs)
    print(json.dumps({k: doc[k] for k in ("target", "sites", "mutants_run", "unsplicable", "killed",
                                          "survived", "timeout", "score_killed_over_run",
                                          "equivalent_declared", "score_excluding_equivalents", "wall_seconds")}))
    for s in doc["survivors"]:
        tag = "EQUIVALENT" if s["equivalent"] else "SURVIVED"
        print(f"  {tag} {s['operator']} line {s['lineno']}: {s['original']}  =>  {s['mutant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
