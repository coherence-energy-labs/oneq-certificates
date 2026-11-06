"""R189: the ion run as ONE parameterized-shot program -- the same 2,000 frozen units, one per shot, in one Helios job.

THE PROBLEM IT SOLVES. R188 priced the run at ~10,200 HQC, and 98.6% of that is Quantinuum's fixed 5 HQC per program:
2,000 distinct single-shot circuits are 2,000 programs. The certificate needs distinct circuits (R186 part 3: a
distribution-free bound is distinct-circuit-bound; clustering shots on one circuit was measured dead), so the obvious
levers -- more shots per circuit, fewer circuits -- are closed.

THE MOVE. "Distinct circuit" is a property of a SHOT, not of a PROGRAM. Helios exposes the shot index at run time
(guppylang.std.qsystem.utils.get_current_shot; Quantinuum's "Parameterized Shots" guide), so one program can carry the
whole frozen design as a compile-time table and run row `shot` on shot `shot`. Every shot is still one fresh draw of a
unit fixed at freeze time plus one shot -- the error-indicator certificate's premise, untouched (each group's indicator
is exactly Bernoulli; r186_certification.binomial) -- but the fixed charge is paid once:

    HQC = 5 + (C / 5000) (Max N1q + 10 Max N2q + 5 Max N_M)                    (Helios costing page, 2026-10-19)

with C = 2,000 shots: about 5 + 2,000 x 0.1, i.e. ~200 HQC instead of ~10,200. N_M counts measurements AND qubit
allocations; Rz is not counted (virtual).

WHAT IS GENERATED. `emit(design)` returns Guppy SOURCE TEXT (no guppylang import here, so the repo's own environment
can generate and test it): straight-line code with literal qubit indices that mirrors hw_faithful.circuit (decoder
'mix', erase='none') gate for gate, with every per-unit difference a classical branch on the shot's table row.
  * A gate that a unit omits is a branch NOT TAKEN, never a zero-angle rotation: a zero-angle Rxy is still a physical
    pulse (noise and HQC).
  * p(phi) on the key is emitted as rz(phi): equal up to a global phase, which is unobservable within a shot.
  * Each shot outputs its shot index and its unit's row id, so results map to units whatever order the provider
    returns them in, and a short or re-ordered job is caught by the analysis rather than mis-scored.

WHAT OWNS CORRECTNESS. Not this file. `tools/verify_ion_param_shots.py` (run in an environment with guppylang and
selene-sim) compiles the source, runs every shot in selene, extracts the native instruction trace of each shot
(CircuitExtractor; the program has no measurement-dependent control flow, so the trace does not depend on the
simulator), rebuilds it as a circuit and checks it EXACTLY against that unit's frozen QASM (an MPS overlap of
U_frozen^dag U_executed on |0...0>, plus the measured-qubit order); it also counts the native operations the HQC
formula prices. Its receipt is what a preregistration may cite.

  python ion_param_shots.py emit DESIGN_DIR      write DESIGN_DIR/param_shots_program.py and its table
  python ion_param_shots.py analyse DESIGN_DIR RESULTS_JSON
                                                 the provider's per-shot results -> the exact Clopper-Pearson verdict
                                                 (every row once, on its own shot, or refused)
"""
from __future__ import annotations

import hashlib
import json
import numbers
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]

BRANCH = {"kept_kept": 0, "kept_erased": 1, "erased_erased": 2, "erased_kept": 3}
BASIS = {"Z": 0, "X": 1, "Y": 2}
G_S0 = {"I": 0, "X": 1, "Y": 2, "Z": 3}
G_S, G_SDG = 4, 5
HEAD = ("uid", "branch", "eneg", "basis", "beta")


class ParamShotsRefused(ValueError):
    pass


def layout(m: int) -> dict:
    """hw_faithful.Q: q0 input, q1 copy, q2/q3 key copies, s(u,t) = 4 + 2um + t, a(u,t) = 4 + (2u+1)m + t"""
    return dict(n=4 + 4 * m, s=lambda u, t: 4 + 2 * u * m + t, a=lambda u, t: 4 + (2 * u + 1) * m + t)


def use_fields(choice: list, m: int) -> list:
    """[kind (0 s0 / 1 s1), zkey, phase_k (phase = k pi/4)] + g[0..m-1] + cz[0..m-1], from hw_two_use._apply_p"""
    kind, lab = choice[0], list(choice[1])
    if len(lab) != m:
        raise ParamShotsRefused(f"a label of length {len(lab)} for m = {m}")
    if kind == "s0":
        if any(g not in G_S0 for g in lab):
            raise ParamShotsRefused(f"an s0 label outside IXYZ: {lab}")
        return [0, lab.count("Y") % 2, 0] + [G_S0[g] for g in lab] + [1 if g in "IZ" else 0 for g in lab]
    if kind == "s1":
        if any(c not in (0, 1) for c in lab):
            raise ParamShotsRefused(f"an s1 label outside {{0, 1}}: {lab}")
        return [1, 0, sum(1 if c == 0 else -1 for c in lab)] + [G_S if c == 0 else G_SDG for c in lab] + [1] * m
    raise ParamShotsRefused(f"unknown label kind {kind!r}")


def table(design: dict) -> list:
    """one row of ints per frozen unit, in the design's order; row[0] is the unit's index in design['units']"""
    m = design["m"]
    if design.get("decoder") != "mix":
        raise ParamShotsRefused(f"the emitter mirrors decoder 'mix' only, not {design.get('decoder')!r}")
    rows = []
    for i, u in enumerate(design["units"]):
        basis, e = u["inp"][0], int(u["inp"][1])
        if basis not in BASIS or e not in (1, -1):
            raise ParamShotsRefused(f"unit {i}: input {u['inp']}")
        if u["branch"] not in BRANCH or u["beta"] not in (0, 1) or len(u["choices"]) != 2:
            raise ParamShotsRefused(f"unit {i}: branch/beta/choices malformed")
        rows.append([i, BRANCH[u["branch"]], 1 if e < 0 else 0, BASIS[basis], int(u["beta"])]
                    + use_fields(u["choices"][0], m) + use_fields(u["choices"][1], m))
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise ParamShotsRefused("rows of unequal width")
    return rows


def _col(m: int) -> dict:
    """field name -> column index in a row"""
    col = {k: i for i, k in enumerate(HEAD)}
    for u in range(2):
        base = len(HEAD) + u * (3 + 2 * m)
        col[f"k{u}"], col[f"z{u}"], col[f"p{u}"] = base, base + 1, base + 2
        for t in range(m):
            col[f"g{u}_{t}"] = base + 3 + t
            col[f"c{u}_{t}"] = base + 3 + m + t
    return col


def emit(design: dict) -> str:
    """the Guppy program's source text (deterministic: same design, same bytes)"""
    m = design["m"]
    L = layout(m)
    col = _col(m)
    n = L["n"]
    out = []
    w = out.append

    def r(name):
        return f"r[{col[name]}]"

    w('"""GENERATED by experiments/superactivation/shield_family/ion_param_shots.py -- do not edit by hand.')
    w(f"R189 parameterized-shot program: m = {m}, {len(design['units'])} frozen units, one per shot (row = shot).")
    w(f'design qpy_sha256 {design["qpy_sha256"]}"""')
    w("import json")
    w("import pathlib")
    w("")
    w("from guppylang import guppy")
    w("from guppylang.std.angles import pi")
    w("from guppylang.std.builtins import comptime, result")
    w("from guppylang.std.qsystem.utils import get_current_shot")
    w("from guppylang.std.quantum import ch, cx, cz, discard, h, measure, qubit, rz, ry, s, sdg, x, y, z")
    w("")
    w('TABLE = json.loads((pathlib.Path(__file__).resolve().parent / "param_shots_table.json").read_text('
      'encoding="utf-8"))')
    w("")
    w("")
    w("@guppy")
    w("def main() -> None:")
    w("    tab = comptime(TABLE)")
    w("    shot = get_current_shot()")
    w("    r = tab[shot]")
    w('    result("shot", shot)')
    w(f'    result("uid", {r("uid")})')
    for q in range(n):
        w(f"    q{q} = qubit()")
    # ---- Alice: input
    w(f"    if {r('eneg')} == 1:")
    w("        x(q0)")
    w(f"    if {r('basis')} >= 1:")
    w("        h(q0)")
    w(f"    if {r('basis')} == 2:")
    w("        s(q0)")
    w("    cx(q0, q1)")
    w("    h(q0)")
    w("    h(q1)")
    for u in range(2):
        for t in range(m):
            w(f"    h(q{L['a'](u, t)})")
            w(f"    cx(q{L['a'](u, t)}, q{L['s'](u, t)})")
        w(f"    cx(q{u}, q{2 + u})")
    # ---- the channels: P_d on both uses (hw_two_use._apply_p, key = q_u, shields = s(u, t))
    for u in range(2):
        key = f"q{u}"
        w(f"    if {r(f'k{u}')} == 0:")
        for t in range(m):
            sh = f"q{L['s'](u, t)}"
            g = r(f"g{u}_{t}")
            w(f"        if {g} == 1:")
            w(f"            x({sh})")
            w(f"        elif {g} == 2:")
            w(f"            y({sh})")
            w(f"        elif {g} == 3:")
            w(f"            z({sh})")
        w(f"        if {r(f'z{u}')} == 1:")
        w(f"            z({key})")
        for t in range(m):
            w(f"        if {r(f'c{u}_{t}')} == 1:")
            w(f"            cz({key}, q{L['s'](u, t)})")
        w("    else:")
        for t in range(m):
            sh = f"q{L['s'](u, t)}"
            w(f"        if {r(f'g{u}_{t}')} == {G_S}:")
            w(f"            s({sh})")
            w("        else:")
            w(f"            sdg({sh})")
            w(f"        cz({key}, {sh})")
        w(f"        rz({key}, pi * (0.25 * {r(f'p{u}')}))")
        w(f"        x({key})")
    # ---- Bob
    w(f"    if {r('branch')} == 2:")
    w("        b0 = measure(q0)")
    w("        b1 = measure(q1)")
    w('        result("c0", b0.read())')
    w('        result("c1", b1.read())')
    for q in range(2, n):
        w(f"        discard(q{q})")

    def bob_kept(dec: int, splits: list) -> None:
        """Bob on the kept use `dec` (hw_faithful.circuit): its copy CNOT, the mix pair decoder, the basis rotation
        of its key, then one measurement block per (condition, order)"""
        w(f"        cx(q{dec}, q{2 + dec})")
        for t in range(m):
            a, sh = f"q{L['a'](dec, t)}", f"q{L['s'](dec, t)}"
            w(f"        if {r('beta')} == 1:")                   # mix: beta 1 -> cz1 (beta = 1 branch)
            w(f"            rz({a}, pi)")
            w(f"            ry({sh}, pi * 1.5)")
            w(f"            rz({sh}, pi)")
            w(f"            cz({a}, {sh})")
            w(f"            ry({a}, pi * 0.25)")
            w("        else:")                                     # mix: beta 0 -> cz2 (no ry: beta = 0)
            w(f"            cx({a}, {sh})")
            w(f"            ch({sh}, {a})")
        w(f"        if {r('basis')} == 0:")                       # input Z -> Bob measures X
        w(f"            h(q{dec})")
        w(f"        elif {r('basis')} == 2:")                     # input Y -> Bob measures Y
        w(f"            sdg(q{dec})")
        w(f"            h(q{dec})")
        for head, order in splits:
            ind = "            " if head else "        "
            if head:
                w(head)
            for j, q in enumerate(order):
                w(f"{ind}m{j} = measure(q{q})")
            for j in range(len(order)):
                w(f'{ind}result("c{j}", m{j}.read())')
            for q in range(n):
                if q not in order:
                    w(f"{ind}discard(q{q})")

    if any(u["branch"] == "erased_kept" for u in design["units"]):
        # R224: use 0 erased, use 1 kept -- emitted only for a design that has such units, so a folded design's program
        # keeps its bytes
        w(f"    elif {r('branch')} == {BRANCH['erased_kept']}:")
        bob_kept(1, [(None, [3] + [L["a"](1, t) for t in range(m)] + [0, 1])])
    w("    else:")
    pair = [L["a"](0, t) for t in range(m)]
    measured_kk = [2] + pair + [3, 0]
    measured_ke = [2] + pair + [1, 0]
    # the outer branches already took erased_erased (and erased_kept), so this split is exactly kept_kept (0) /
    # kept_erased (1)
    bob_kept(0, [(f"        if {r('branch')} == 0:", measured_kk), ("        else:", measured_ke)])
    w("")
    return "\n".join(out) + "\n"


def table_json(rows: list) -> str:
    """one row per line, row index = shot index (deterministic bytes)"""
    return "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n]\n"


def measured_order(design: dict, i: int) -> list:
    """the frozen circuit's measured qubits in clbit order (hw_faithful.circuit), for the verifier"""
    m = design["m"]
    L = layout(m)
    u = design["units"][i]
    if u["branch"] == "erased_erased":
        return [0, 1]
    if u["branch"] == "erased_kept":
        return [3] + [L["a"](1, t) for t in range(m)] + [0, 1]
    return [2] + [L["a"](0, t) for t in range(m)] + [3 if u["branch"] == "kept_kept" else 1, 0]


def counts_from_shots(design: dict, records: list) -> dict:
    """the provider's per-shot records ({"shot": i, "uid": j, "c0": b, ...}) -> ion_run_package.analyse's counts
    {unit name: {bitstring (clbit 0 rightmost): 1}}. Refused unless every row ran exactly once, on the shot that
    names it, with exactly its unit's number of bits: a short, duplicated or re-ordered job is caught here, never
    scored"""
    units = design["units"]
    seen: dict = {}

    def booly(v) -> bool:
        return isinstance(v, bool) or getattr(getattr(v, "dtype", None), "kind", "") == "b"

    def integer(v, what):
        # strict (R203 adversary pass): int(0.7) is 0, so a coerced float scored as a clean bit; refuse instead.
        # numpy integers are integers (numbers.Integral); bools and floats are not shot indices
        if booly(v) or not isinstance(v, numbers.Integral):
            raise ParamShotsRefused(f"{what} {v!r} is not an integer")
        return int(v)

    def is_bit(v) -> bool:
        return booly(v) or (isinstance(v, numbers.Integral) and int(v) in (0, 1))

    for rec in records:
        if "shot" not in rec or "uid" not in rec:
            raise ParamShotsRefused(f"a shot record without its shot index or unit id: {sorted(rec)[:6]}")
        i, uid = integer(rec["shot"], "shot index"), integer(rec["uid"], "unit id")
        if not 0 <= i < len(units):
            raise ParamShotsRefused(f"shot index {i} outside the {len(units)}-row table")
        if uid != i:
            raise ParamShotsRefused(f"shot {i} ran row {uid}: the table is indexed by shot")
        if i in seen:
            raise ParamShotsRefused(f"shot {i} reported twice")
        k = units[i]["n_clbits"]
        bits = [k_ for k_ in rec if k_.startswith("c") and k_[1:].isdigit()]
        if sorted(int(b[1:]) for b in bits) != list(range(k)):
            raise ParamShotsRefused(f"shot {i}: bits {sorted(bits)}, its unit measures {k}")
        foreign = sorted(set(rec) - {"shot", "uid"} - set(bits))
        if foreign:                     # the program writes only shot, uid and c0..c(k-1): anything else is not ours
            raise ParamShotsRefused(f"shot {i}: tags the program never writes: {foreign[:5]}")
        vals = [rec[f"c{j}"] for j in reversed(range(k))]
        if not all(is_bit(v) for v in vals):
            raise ParamShotsRefused(f"shot {i}: a bit that is not 0 or 1: {[v for v in vals if not is_bit(v)][:3]}")
        seen[i] = "".join(str(int(v)) for v in vals)
    missing = [i for i in range(len(units)) if i not in seen]
    if missing:
        raise ParamShotsRefused(f"{len(missing)} rows never ran, e.g. shots {missing[:5]}")
    return {units[i]["name"]: {b: 1} for i, b in seen.items()}


def hqc(c_shots: int, max_n1q: int, max_n2q: int, max_nm: int) -> float:
    """Helios costing page (docs.quantinuum.com, read 2026-10-19): one program, C shots"""
    return 5 + c_shots * (max_n1q + 10 * max_n2q + 5 * max_nm) / 5000


def normalise_records(raw) -> list:
    """the provider's per-shot results -> [{tag: value}]: a list of dicts, or a list of shots that are each a list
    of [tag, value] pairs (Guppy/selene `QsysShot.entries`), optionally under a top-level "shots" key. A tag seen
    twice in one shot is refused: its value would be ambiguous"""
    shots = raw.get("shots") if isinstance(raw, dict) else raw
    if not isinstance(shots, list):
        raise ParamShotsRefused("the results are not a list of shots")
    out = []
    for k, s in enumerate(shots):
        if isinstance(s, dict):
            out.append(dict(s))
            continue
        if not isinstance(s, list) or any(not (isinstance(p, list) and len(p) == 2) for p in s):
            raise ParamShotsRefused(f"shot record {k} is neither a dict nor a list of [tag, value] pairs")
        tags = [p[0] for p in s]
        if len(set(tags)) != len(tags):
            raise ParamShotsRefused(f"shot record {k} repeats a tag: {sorted(tags)}")
        out.append({p[0]: p[1] for p in s})
    return out


def analyse_shots(design: dict, raw) -> dict:
    """the provider's results file -> ion_run_package.analyse's verdict, through counts_from_shots' refusals"""
    import ion_run_package as P
    return P.analyse(design, counts_from_shots(design, normalise_records(raw)))


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "analyse":
        sys.path.insert(0, str(HERE))
        sys.path.insert(0, str(ROOT / "src"))
        design = json.loads((pathlib.Path(sys.argv[2]) / "design.json").read_text(encoding="utf-8"))
        res = analyse_shots(design, json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8")))
        print(json.dumps(res, indent=1))
        return 0
    if len(sys.argv) != 3 or sys.argv[1] != "emit":
        print(__doc__)
        return 2
    d = pathlib.Path(sys.argv[2])
    design = json.loads((d / "design.json").read_text(encoding="utf-8"))
    rows = table(design)
    src = emit(design)
    tab = table_json(rows)
    (d / "param_shots_program.py").write_text(src, encoding="utf-8", newline="\n")
    (d / "param_shots_table.json").write_text(tab, encoding="utf-8", newline="\n")
    print(f"program sha256 {hashlib.sha256(src.encode()).hexdigest()}  table sha256 "
          f"{hashlib.sha256(tab.encode()).hexdigest()}  rows {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
