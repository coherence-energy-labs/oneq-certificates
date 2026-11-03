r"""R198: the certificate calculus's soundness, as a Lean theorem -- a receipt producer and a Lean-free gate.

    python lean_receipt.py --build C:/lq/CertR198      # PRODUCER: copy lean/CertR198 to a SHORT path, lake build,
                                                      # print every theorem's axioms, write the receipt
    python lean_receipt.py                             # GATE (default): needs no Lean

WHAT IS PROVED (lean/CertR198, see its SPEC.md): branching on any firing detector loses no completion (rule C);
the independent-detector and fractional-cover bounds, in the checker's exact integer arithmetic, are lower bounds
(rules D/G); under a CDP4/CDP5 header (R205/R206) the packing over the co-touch of ALLOWED mechanisms and the
exact-integer dual raise (rules D'/G': every raise keeps the dual feasible, and feasible weights bound every
completion), and CDP5's weight-3 terminal as a valid tree; the radius-2 leaves; the search-tree predicate's
soundness; anchoring with an AFFINE observable map (rules A + I, without needing a free action); the main theorem
`cert_sound` -- a valid certificate for weight w means every undetectable logical error has weight > w -- and its
scoped form for rule S, which only excludes IRREDUCIBLE solutions (the checker's verdict says exactly that since
2026-10-21). A concrete 6-mechanism model runs the whole chain (d = 3); a 10-detector one shows a prune only the
dual raise makes (ccheck.c agrees on it).

WHAT IS NOT PROVED: that ccheck.c implements the Lean predicate (SPEC.md argues it line by line; the 2026-10-21
findings F1/F5 were exactly where that argument failed, now closed by the loaders' refusals; the brute-force
oracle test holds the C's own bound and terminal procedures against exhaustive search), the trace decoding, the
leaf index, the GF(2) elimination, parsing and integer widths, the compiler, rule S across models, and everything
upstream of the DEM. The gate cannot see those; it pins what Lean checked.

The receipt (evidence/circuit_distance_exact/lean_r198_receipt.json) carries the SHA-256 of every Lean source and
the axioms Lean printed for every theorem AxiomCheck.lean names. The gate REFUSES when a source changed since the
receipt, when the theorem lists differ, when any theorem rests on an axiom beyond propext / Classical.choice /
Quot.sound (sorryAx named separately: a hole, not a choice), or when the build did not exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEAN = ROOT / "lean" / "CertR198"
RECEIPT = ROOT / "evidence" / "circuit_distance_exact" / "lean_r198_receipt.json"
STANDARD = {"propext", "Classical.choice", "Quot.sound"}
# An already-built mathlib (v4.34.0) to link instead of fetching one; opt-in, so a fresh machine just runs lake.
SHARED_PACKAGES = (pathlib.Path(os.environ["ONEQ_MATHLIB_PACKAGES"]) if os.environ.get("ONEQ_MATHLIB_PACKAGES")
                   else None)
sys.path.insert(0, str(ROOT / "tools"))
from roster_digest import roster_digest  # noqa: E402  (the repo's one checkout-invariant source digest)


def sources(root: pathlib.Path = LEAN) -> dict[str, str]:
    """rel path -> SHA-256 of its text with CRLF read as LF: what Lean checked does not depend on line endings, and a
    hash of working-copy bytes would differ between a Windows checkout (core.autocrlf) and the stored blob."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and not rel.startswith(".lake/") and (p.suffix == ".lean" or p.name in
                                                              ("lakefile.toml", "lean-toolchain", "lake-manifest.json")):
            out[rel] = roster_digest(p)
    return out


def declared(root: pathlib.Path = LEAN) -> list[str]:
    return re.findall(r"^#print axioms (\S+)", (root / "CertR198" / "AxiomCheck.lean").read_text(encoding="utf-8"), re.M)


def parse_axioms(text: str) -> dict[str, list[str]]:
    """Lean prints "'X' depends on axioms: [a, b]" or "'X' does not depend on any axioms" per #print axioms."""
    out = {}
    for m in re.finditer(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text):
        out[m.group(1)] = sorted(a.strip() for a in m.group(2).replace("\n", " ").split(",") if a.strip())
    for m in re.finditer(r"'([^']+)' does not depend on any axioms", text):
        out[m.group(1)] = []
    return out


def check(receipt_path: pathlib.Path = RECEIPT, root: pathlib.Path = LEAN) -> list[str]:
    """[] when the receipt certifies the current sources; otherwise every reason it does not."""
    if not receipt_path.exists():
        return ["no receipt: run --build"]
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    bad = []
    if r.get("build_exit") != 0:
        bad.append(f"the build exited {r.get('build_exit')}")
    now = sources(root)
    for rel in sorted(set(now) | set(r.get("sources", {}))):
        if now.get(rel) != r["sources"].get(rel):
            bad.append(f"STALE: {rel} differs from the proved source")
    want = declared(root)
    if sorted(want) != sorted(r.get("theorems", {})):
        bad.append("the receipt's theorem list differs from AxiomCheck.lean's")
    for name, axioms in r.get("theorems", {}).items():
        if "sorryAx" in axioms:
            bad.append(f"{name} rests on sorryAx (a hole in the proof)")
        extra = sorted(set(axioms) - STANDARD - {"sorryAx"})
        if extra:
            bad.append(f"{name} rests on non-standard axioms {extra}")
    return bad


def build(where: pathlib.Path) -> int:
    if where.exists():
        shutil.rmtree(where)
    shutil.copytree(LEAN, where, ignore=shutil.ignore_patterns(".lake"))
    if SHARED_PACKAGES is not None and SHARED_PACKAGES.exists():   # reuse a built mathlib; else lake fetches one
        (where / ".lake").mkdir()
        subprocess.run(["cmd", "/c", "mklink", "/J", str(where / ".lake" / "packages"), str(SHARED_PACKAGES)],
                       check=True, capture_output=True, stdin=subprocess.DEVNULL)
    b = subprocess.run(["lake", "build"], cwd=where, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    print(b.stdout[-600:], b.stderr[-600:])
    a = subprocess.run(["lake", "env", "lean", "CertR198/AxiomCheck.lean"], cwd=where, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    theorems = parse_axioms(a.stdout + a.stderr)
    receipt = dict(project="lean/CertR198", build_exit=b.returncode or a.returncode, sources=sources(where),
                   theorems=theorems, lean=(where / "lean-toolchain").read_text().strip(),
                   axiom_check_lines=len(theorems))
    RECEIPT.write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"receipt: {len(theorems)} theorems, build exit {receipt['build_exit']} -> {RECEIPT}")
    return 0 if not check() else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=pathlib.Path)
    a = ap.parse_args()
    if a.build:
        sys.exit(build(a.build))
    bad = check()
    for b in bad:
        print("REFUSED:", b)
    if bad:
        sys.exit(1)
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    print(f"OK: {len(r['theorems'])} theorems of lean/CertR198, standard axioms only, sources as proved")


if __name__ == "__main__":
    main()
