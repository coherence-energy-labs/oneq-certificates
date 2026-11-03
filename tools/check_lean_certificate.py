#!/usr/bin/env python3
"""P2's certificates, judged by a checker whose soundness is a Lean theorem.

WHAT IS PROVED.  `lean/OneqCert` proves `safe_bound_le_weight` (the bound the exact
checker's `_exact_facet_scan` returns never exceeds the weight of any correction that
meets every facet's parity, for any rational weights and any y >= 0) and
`Cert.accept_sound`: when the EXECUTABLE Lean checker `accept` returns true, the
candidate is a minimum-weight correction.  The compiled `oneqcert-check` runs that
function on every P2 exact certificate (evidence/p2_certificates.jsonl.gz) and on
forgeries built from each of them in six declared ways; every certificate must be
accepted and every forgery refused.

THREE MODES, BECAUSE THE PROVER IS HEAVY AND THE WALL MUST NOT NEED IT.

  --build DIR   the PRODUCER.  Copies the repo's Lean sources into DIR (a lake
                project on a SHORT path: mathlib's package tree exceeds Windows'
                260-character limit under a deep checkout), runs `lake build`, runs
                `AxiomCheck.lean`, and writes evidence/lean_safe_bound.json carrying
                the SHA-256 of every source it proved.

  --mutants DIR the DRILL, after --build DIR.  Applies each MUTANTS edit -- a
                weakening of accept_sound that Lean accepts, keeping its name and
                axioms -- and records that each fails to build AT the pinned statement
                (evidence/lean_safe_bound_mutants.json).

  --check       the GATE (default).  Needs no Lean.  Refuses when
                  * any source's hash differs from the receipt -- a proof of an
                    older file is not a proof of this one;
                  * the receipt's theorem list differs from AxiomCheck.lean's, or it
                    lacks a HEADLINE theorem;
                  * any theorem rests on an axiom outside the standard three
                    (`sorryAx` is named separately: it means a hole, not a choice);
                  * the build did not exit 0;
                  * OneqCert/Statements.lean, which restates accept_sound's full type and
                    is elaborated from it, is not the REVIEWED statement
                    (STATEMENTS_SHA256).  Sources and axioms alone pass a WEAKENED
                    headline once the receipt is rebuilt;
                  * the mutant drill is missing or stale, or a mutant built or failed
                    anywhere but the pinned statement;
                  * the checker source cites an `OneqCert.<name>` the receipt does
                    not certify -- a comment claiming a Lean theorem is a claim;
                  * the judged certificates are not the committed ones, any was
                    refused or unparsed, or any forgery was accepted.

WHAT IS NOT PROVED: the JSON reader in Main.lean, and the Lean compiler; and that the
PYTHON checker computes the Lean `safeBound` (by reading and adversarial tests). The
paper's certificates no longer depend on that last link: the Lean checker judges them
directly. See lean/OneqCert/README.md.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "OneqCert"
OUT = ROOT / "evidence" / "lean_safe_bound.json"
CHECKER_SRC = ROOT / "src" / "oneq" / "qldpc_check.py"
CERTS = ROOT / "evidence" / "p2_certificates.jsonl.gz"
EXPORT = ROOT / "evidence" / "p2_certificates_export.json"
SCHEMA = "oneq.lean.safe-bound.v1"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathrel import rel  # noqa: E402

ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
STATEMENTS = "OneqCert/Statements.lean"
SOURCES = ("lakefile.toml", "lean-toolchain", "lake-manifest.json", "OneqCert.lean",
           "OneqCert/SafeBound.lean", "OneqCert/WeakDuality.lean",
           "OneqCert/Checker.lean", STATEMENTS, "OneqCert/AxiomCheck.lean", "Main.lean")
#: The theorem the P2 claim rests on, and its pinned restatement; AxiomCheck.lean must print both.
HEADLINE = ("OneqCert.Cert.accept_sound", "OneqCert.Statements.accept_sound_pinned")
#: content_hash of OneqCert/Statements.lean as REVIEWED (2026-10-22). It restates accept_sound's full type and
#: is elaborated from it, so a weakened accept_sound fails to build there. The source and axiom checks alone
#: pass a weakened headline once the receipt is rebuilt (an adversary showed that on this repo's R211 gate). A
#: weakening therefore has to edit that file AND this constant; change it only after reading the new statement.
STATEMENTS_SHA256 = "d9e4448398dc1330d33224f4c0286a8035813e6a3eb34e80701909334f922102"
MUTANT_OUT = ROOT / "evidence" / "lean_safe_bound_mutants.json"

_PRINT = re.compile(r"^#print axioms\s+(\S+)", re.M)
_DEPENDS = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]", re.S)
_NO_AXIOMS = re.compile(r"'([^']+)' does not depend on any axioms")
_CITED = re.compile(r"OneqCert\.([A-Za-z_][A-Za-z0-9_']*)")


def content_hash(data: bytes) -> str:
    """SHA-256 of the text with CRLF folded to LF.

    A receipt that hashes working-copy bytes verifies only in the checkout that
    wrote it: with core.autocrlf the same commit is LF in one tree and CRLF in the
    next, and every hash would read STALE. Lean parses both identically, so the
    line ending is not part of what was proved."""
    return hashlib.sha256(data.replace(bytes([13, 10]), bytes([10]))).hexdigest()


def source_hashes(lean_dir: pathlib.Path = LEAN) -> dict[str, str]:
    out = {}
    for src in SOURCES:
        p = lean_dir / src
        if not p.is_file():
            raise FileNotFoundError(f"Lean source missing: {p}")
        out[src] = content_hash(p.read_bytes())
    return out


def expected_theorems(axiom_check_text: str) -> list[str]:
    return _PRINT.findall(axiom_check_text)


def parse_axioms(output: str) -> dict[str, list[str]]:
    """Every `#print axioms` answer in a Lean run, as {theorem: [axiom, ...]}."""
    got: dict[str, list[str]] = {}
    for name, body in _DEPENDS.findall(output):
        got[name] = sorted(a.strip() for a in body.replace("\n", " ").split(",")
                           if a.strip())
    for name in _NO_AXIOMS.findall(output):
        got[name] = []
    return got


def cited_theorems(checker_text: str) -> set[str]:
    return set(_CITED.findall(checker_text))


def verdict(receipt: dict, hashes: dict[str, str], expected: list[str],
            cited: set[str]) -> list[str]:
    """Every reason the receipt does not certify the current sources. Empty = pass."""
    bad: list[str] = []
    if receipt.get("schema") != SCHEMA:
        bad.append(f"schema {receipt.get('schema')!r} is not {SCHEMA!r}")
    if receipt.get("build_exit") != 0:
        bad.append(f"lake build exited {receipt.get('build_exit')!r}, not 0")
    rec_src = receipt.get("sources", {})
    for src in sorted(set(hashes) | set(rec_src)):
        if rec_src.get(src) != hashes.get(src):
            bad.append(f"STALE: {src} is not the file that was proved "
                       f"(receipt {str(rec_src.get(src))[:12]}, "
                       f"tree {str(hashes.get(src))[:12]})")
    thms = receipt.get("theorems", {})
    exp = {f"OneqCert.{t}" if not t.startswith("OneqCert.") else t for t in expected}
    if not exp:
        bad.append("AxiomCheck.lean prints no theorems: the check would be vacuous")
    for t in sorted(exp - set(thms)):
        bad.append(f"theorem {t} is in AxiomCheck.lean but not in the receipt")
    for t in sorted(set(thms) - exp):
        bad.append(f"receipt certifies {t}, which AxiomCheck.lean does not print")
    for t, axioms in sorted(thms.items()):
        if "sorryAx" in axioms:
            bad.append(f"{t} depends on sorryAx: it has a hole, it is not proved")
        extra = set(axioms) - ALLOWED_AXIOMS - {"sorryAx"}
        if extra:
            bad.append(f"{t} rests on non-standard axioms {sorted(extra)}")
    for c in sorted(cited):
        if f"OneqCert.{c}" not in thms:
            bad.append(f"the checker cites OneqCert.{c}, which the receipt does not "
                       f"certify")
    for t in HEADLINE:
        if t not in exp:
            bad.append(f"AxiomCheck.lean does not print the headline {t}")
    if hashes.get(STATEMENTS) != STATEMENTS_SHA256:
        bad.append(f"PINNED STATEMENT changed: {STATEMENTS} is not the reviewed statement of "
                   f"accept_sound (read the new statement, then update STATEMENTS_SHA256)")
    return bad


def _accept_sound_span(text: str) -> tuple[int, int]:
    a = text.find("theorem accept_sound (h : c.accept = true) :")
    if a < 0 or text.count("theorem accept_sound") != 1:
        raise ValueError("accept_sound's declaration is not unique in Checker.lean")
    b = text.find("\nend Cert", a)
    if b < 0:
        raise ValueError("accept_sound is no longer the last declaration of its section")
    return a, b + 1


def _replace_headline(declaration: str):
    def edit(text: str) -> str:
        a, b = _accept_sound_span(text)
        return text[:a] + declaration + "\n" + text[b:]
    return edit


#: Edits Lean ACCEPTS in Checker.lean that weaken the headline while keeping its name and axioms. Each must fail
#: to build AT the pinned statement, or the pin is not what refuses it.
MUTANTS = {
    "headline_trivialized": ("accept_sound's conclusion replaced by True",
                             _replace_headline("theorem accept_sound (h : c.accept = true) : True := trivial\n")),
    "minimality_made_vacuous": (
        "accept_sound's minimality kept as text but disjoined with True, so it says nothing",
        _replace_headline("theorem accept_sound (h : c.accept = true) :\n"
                          "    (∀ e' ⊆ Finset.range c.w.length, c.Consistent e' →\n"
                          "      c.weight c.e.toFinset ≤ c.weight e') ∨ True :=\n"
                          "  Or.inr trivial\n")),
}


def mutant_verdict(record: dict | None, base_checker: str, hashes: dict[str, str]) -> list[str]:
    """Every reason the mutant drill does not show the pinned statement refusing each MUTANTS edit."""
    if record is None:
        return ["no mutant drill: run --mutants DIR after --build DIR"]
    bad = []
    if record.get("base_sources") != hashes:
        bad.append("the mutant drill ran on other sources: re-run --mutants")
    if sorted(record.get("mutants", {})) != sorted(MUTANTS):
        bad.append(f"the drill's mutants {sorted(record.get('mutants', {}))} are not {sorted(MUTANTS)}")
    for name, (_what, edit) in MUTANTS.items():
        got = record.get("mutants", {}).get(name)
        if not got:
            continue
        if got.get("checker_lean") != edit(base_checker):
            bad.append(f"mutant {name}: the drill built a different Checker.lean than this edit gives")
        if got.get("build_exit") == 0:
            bad.append(f"mutant {name} BUILT: the pinned statement did not refuse it")
        elif not str(got.get("first_error", "")).startswith(STATEMENTS + ":"):
            bad.append(f"mutant {name} failed outside the pinned statement ({got.get('first_error')!r}): it must "
                       f"compile in Checker.lean and be refused by {STATEMENTS}")
    return bad


def certs_hash(path: pathlib.Path = CERTS) -> str | None:
    """SHA-256 of the certificate file's CONTENT (the decompressed JSON lines)."""
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def corpus_verdict(corpus: dict | None, content_sha: str | None,
                   export: dict | None) -> list[str]:
    """Every reason the corpus run does not certify the committed certificates."""
    if corpus is None:
        return ["the receipt carries no corpus run: the P2 certificates were never "
                "put through the proved checker"]
    bad: list[str] = []
    if content_sha is None:
        bad.append(f"{rel(CERTS, ROOT)} is missing")
    elif corpus.get("certificates_sha256") != content_sha:
        bad.append("STALE: the certificates the Lean checker judged are not the "
                   "committed ones")
    if export is None:
        bad.append(f"{rel(EXPORT, ROOT)} is missing")
    else:
        if export.get("content_sha256") != content_sha:
            bad.append("the export summary describes different certificates")
        if corpus.get("accepted") != export.get("certificates"):
            bad.append(f"the Lean checker accepted {corpus.get('accepted')} of "
                       f"{export.get('certificates')} exported certificates")
    if corpus.get("exe_exit") != 0:
        bad.append(f"oneqcert-check exited {corpus.get('exe_exit')!r}")
    if corpus.get("refused", 1) != 0:
        bad.append(f"the Lean checker REFUSED {corpus.get('refused')} P2 certificate(s):"
                   f" {corpus.get('refused_ids', [])[:10]}")
    if corpus.get("malformed", 1) != 0:
        bad.append(f"{corpus.get('malformed')} certificate line(s) did not parse")
    if not corpus.get("accepted"):
        bad.append("the corpus run accepted nothing: the check would be vacuous")
    fg = corpus.get("forgeries")
    if not fg:
        bad.append("no forgery run: nothing shows the checker can REFUSE on this corpus")
    else:
        if fg.get("accepted", 1) != 0:
            bad.append(f"the proved checker ACCEPTED {fg.get('accepted')} forged "
                       f"certificate(s): {fg.get('accepted_ids', [])[:10]}")
        if fg.get("refused") != fg.get("total") or not fg.get("total"):
            bad.append(f"forgeries refused {fg.get('refused')} of {fg.get('total')}")
        missing = [k for k in FORGERY_KINDS if not fg.get("per_kind", {}).get(k)]
        if missing:
            bad.append(f"forgery kinds never exercised: {missing}")
    return bad


FORGERY_KINDS = ("syndrome_flip", "right_parity_facet", "negative_dual",
                 "row_out_of_range", "duplicate_in_candidate", "all_duals_zero")


def forge(line: dict, kind: str) -> dict | None:
    """One certificate made INVALID in a declared way, or None if this certificate
    cannot carry that forgery. Every forgery is false by construction, so the proved
    checker must refuse all of them; one acceptance means the corpus run proves nothing.
    """
    f = json.loads(json.dumps(line))
    f["id"] = f"{line['id']}#{kind}"
    if kind == "syndrome_flip":
        # the candidate no longer satisfies the syndrome it claims to correct
        f["syn"][0] ^= 1
    elif kind == "right_parity_facet":
        # toggle one variable of the facet's support in or out of F: parity flips to
        # the RIGHT one, whose inequality is false
        for fac in f["facets"]:
            sup: set[int] = set()
            for r in fac["rows"]:
                sup ^= set(f["checks"][r])
            if sup:
                v = min(sup)
                fac["F"] = sorted(set(fac["F"]) ^ {v})
                return f
        return None
    elif kind == "negative_dual":
        pos = [fac for fac in f["facets"] if fac["y"][0] > 0]
        if not pos:
            return None
        pos[0]["y"][0] = -pos[0]["y"][0]
    elif kind == "row_out_of_range":
        if not f["facets"]:
            return None
        f["facets"][0]["rows"] = [len(f["checks"])]
    elif kind == "duplicate_in_candidate":
        if not f["e"]:
            return None
        f["e"] = f["e"] + [f["e"][0]]
    elif kind == "all_duals_zero":
        # with no dual mass the bound is at most 0, so it certifies only a
        # weight-zero answer; skip the certificates where that is TRUE
        U = sum(Fraction(*f["w"][i]) for i in f["e"])
        if U < Fraction(1, f["D"]):
            return None
        for fac in f["facets"]:
            fac["y"] = [0, 1]
    return f


def run_exe(exe: pathlib.Path, plain: pathlib.Path) -> tuple[int, dict, list[str]]:
    r = subprocess.run([str(exe), str(plain)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                       timeout=7200)
    out = r.stdout.splitlines()
    summary = next((ln for ln in reversed(out) if ln.startswith("SUMMARY ")), "")
    counts = {k: int(v) for k, v in (kv.split("=") for kv in summary.split()[1:])}
    return r.returncode, counts, out


def run_corpus(build_dir: pathlib.Path) -> dict:
    exe_build = _lake(build_dir, "build", "oneqcert-check")
    print(exe_build.stdout[-1500:], exe_build.stderr[-1500:], sep="\n")
    exe = build_dir / ".lake" / "build" / "bin" / (
        "oneqcert-check.exe" if sys.platform == "win32" else "oneqcert-check")
    plain = build_dir / "p2_certificates.jsonl"
    with gzip.open(CERTS, "rb") as f:
        raw = f.read()
    plain.write_bytes(raw)
    rc, counts, out = run_exe(exe, plain)
    # the negative control, on the REAL distribution: every certificate, forged in
    # every declared way it can carry, must be refused
    lines = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    forged = [g for ln in lines for k in FORGERY_KINDS
              if (g := forge(ln, k)) is not None]
    fplain = build_dir / "p2_forgeries.jsonl"
    fplain.write_text("\n".join(json.dumps(g, separators=(",", ":")) for g in forged)
                      + "\n", encoding="utf-8", newline="\n")
    frc, fcounts, fout = run_exe(exe, fplain)
    per_kind = {k: sum(1 for g in forged if g["id"].endswith("#" + k))
                for k in FORGERY_KINDS}
    return {
        "certificates_sha256": certs_hash(),
        "exe_exit": (rc if frc == 0 else frc) if exe_build.returncode == 0
        else exe_build.returncode,
        "accepted": counts.get("accepted", -1),
        "refused": counts.get("refused", -1),
        "malformed": counts.get("malformed", -1),
        "refused_ids": [ln.split()[0] for ln in out if ln.endswith(" REFUSE")],
        "forgeries": {
            "per_kind": per_kind,
            "total": len(forged),
            "refused": fcounts.get("refused", -1),
            "accepted": fcounts.get("accepted", -1),
            "malformed": fcounts.get("malformed", -1),
            "accepted_ids": [ln.split()[0] for ln in fout if ln.endswith(" ACCEPT")],
        },
    }


def _lake(build_dir: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    lake = shutil.which("lake") or str(pathlib.Path.home() / ".elan" / "bin" / "lake")
    return subprocess.run([lake, *args], cwd=build_dir, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, timeout=3600)


def build(build_dir: pathlib.Path) -> int:
    if len(str(build_dir.resolve())) > 60:
        print(f"refusing {build_dir}: put the lake project on a short path "
              f"(mathlib's tree exceeds MAX_PATH under a deep one)")
        return 2
    if not (build_dir / ".lake" / "packages" / "mathlib").is_dir():
        print(f"{build_dir} has no mathlib checkout: run `lake exe cache get` there "
              f"first (this tool will not download or compile mathlib for you)")
        return 2
    hashes = source_hashes()
    for src in SOURCES:
        dst = build_dir / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEAN / src, dst)
    b = _lake(build_dir, "build", "OneqCert")
    print(b.stdout[-3000:], b.stderr[-2000:], sep="\n")
    ax = _lake(build_dir, "env", "lean", "OneqCert/AxiomCheck.lean")
    print(ax.stdout, ax.stderr, sep="\n")
    ver = _lake(build_dir, "env", "lean", "--version").stdout.strip()
    manifest = json.loads((LEAN / "lake-manifest.json").read_text(encoding="utf-8"))
    mathlib = next((p.get("rev") for p in manifest.get("packages", [])
                    if p.get("name") == "mathlib"), None)
    receipt = {
        "schema": SCHEMA,
        "claim": "OneqCert.Cert.accept_sound: an accepted certificate proves its "
                 "candidate is a minimum-weight correction; every P2 exact certificate "
                 "was run through that proved `accept`",
        "build_exit": b.returncode if ax.returncode == 0 else ax.returncode,
        "lean_version": ver,
        "mathlib_rev": mathlib,
        "sources": hashes,
        "theorems": parse_axioms(ax.stdout + ax.stderr),
        "corpus": run_corpus(build_dir) if CERTS.is_file() else None,
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"wrote {rel(OUT, ROOT)}")
    return check()


def _lf_text(path: pathlib.Path) -> str:
    return path.read_bytes().replace(bytes([13, 10]), bytes([10])).decode("utf-8")


def mutants(build_dir: pathlib.Path) -> int:
    """Each MUTANTS edit applied to the BUILT copy's Checker.lean: it must compile there and the pinned statement must
    refuse it. The copy is restored and rebuilt afterwards. Run after --build DIR."""
    checker = build_dir / "OneqCert" / "Checker.lean"
    base = _lf_text(LEAN / "OneqCert" / "Checker.lean")
    if source_hashes(build_dir) != source_hashes():
        print(f"{build_dir} is not a build of the current sources: run --build {build_dir} first")
        return 2
    record: dict = {"base_sources": source_hashes(), "mutants": {}}
    try:
        for name, (what, edit) in MUTANTS.items():
            text = edit(base)
            checker.write_text(text, encoding="utf-8", newline="\n")
            b = _lake(build_dir, "build", "OneqCert")
            err = re.search(r"error: (\S+?\.lean):(\d+):(\d+): ([^\n]*)", b.stdout + b.stderr)
            record["mutants"][name] = {
                "what": what, "checker_lean": text, "build_exit": b.returncode,
                "first_error": (f"{err.group(1)}:{err.group(2)}:{err.group(3)}: {err.group(4)}" if err else "")}
            print(f"{name}: exit {b.returncode}; {record['mutants'][name]['first_error'] or 'NO ERROR'}")
    finally:
        checker.write_text(base, encoding="utf-8", newline="\n")
        print(f"restored: lake build exit {_lake(build_dir, 'build', 'OneqCert').returncode}")
    MUTANT_OUT.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8", newline="\n")
    print(f"wrote {rel(MUTANT_OUT, ROOT)}")
    return check()


def check() -> int:
    if not OUT.is_file():
        print(f"no receipt at {rel(OUT, ROOT)}: run --build first")
        return 1
    receipt = json.loads(OUT.read_text(encoding="utf-8"))
    bad = verdict(
        receipt, source_hashes(),
        expected_theorems((LEAN / "OneqCert" / "AxiomCheck.lean")
                          .read_text(encoding="utf-8")),
        cited_theorems(CHECKER_SRC.read_text(encoding="utf-8")))
    export = (json.loads(EXPORT.read_text(encoding="utf-8"))
              if EXPORT.is_file() else None)
    bad += corpus_verdict(receipt.get("corpus"), certs_hash(), export)
    drill = json.loads(MUTANT_OUT.read_text(encoding="utf-8")) if MUTANT_OUT.is_file() else None
    bad += mutant_verdict(drill, _lf_text(LEAN / "OneqCert" / "Checker.lean"), source_hashes())
    if bad:
        print("LEAN CERTIFICATE REFUSED")
        for b in bad:
            print("  " + b)
        return 1
    n = len(receipt["theorems"])
    c = receipt["corpus"]
    print(f"LEAN CERTIFICATE OK -- {n} theorems kernel-checked on "
          f"{receipt['lean_version']}, standard axioms only, receipt matches every "
          f"source byte, accept_sound's statement is the pinned one and Lean refused "
          f"{len(MUTANTS)} weakenings of it there; the proved checker accepted {c['accepted']} of "
          f"{c['accepted']} P2 exact certificates, 0 refused, and refused all "
          f"{c['forgeries']['total']} forgeries built from them")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", type=pathlib.Path, metavar="DIR")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mutants", type=pathlib.Path, metavar="DIR",
                    help="after --build DIR: Lean must refuse every MUTANTS edit at the pinned statement")
    a = ap.parse_args()
    if a.build:
        return build(a.build)
    if a.mutants:
        return mutants(a.mutants)
    return check()


if __name__ == "__main__":
    # lake prints check marks; a cp1252 console cannot encode them, and setting
    # PYTHONIOENCODING here would be too late -- stdout is already open.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
