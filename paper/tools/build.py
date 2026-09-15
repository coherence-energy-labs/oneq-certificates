r"""Build a manuscript, and fail on anything a referee would see as careless.

    python paper/tools/build.py certifying-decoders          # draft
    python paper/tools/build.py certifying-decoders --release

A LaTeX run that "succeeds" can still ship `[?]` for a reference, an empty
citation, or a line running into the margin. Each of those is a red here:

  * undefined references or citations, "labels may have changed" after the
    final pass, any BibTeX warning;
  * an overfull box wider than OVERFULL_PT;
  * the arXiv abstract (extracted from the abstract environment and
    de-TeXed) longer than arXiv's 1920-character limit.

--release additionally refuses while the author block still holds the
placeholder in `owner.tex`: the name and affiliation on a paper are the
owner's to state, and a build that would put a placeholder on arXiv must
not produce a file that looks finished.

Outputs go to paper/<name>/build/ (git-ignored); `arxiv_metadata.txt` there
holds the title and the plain-text abstract for the submission form.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
OVERFULL_PT = 1.0
ARXIV_ABSTRACT_LIMIT = 1920
PLACEHOLDER = "AUTHOR-NAME-REQUIRED"
PLACEHOLDERS = (PLACEHOLDER, "AUTHOR-AFFILIATION-REQUIRED")


def owner_placeholders(owner_tex: str) -> list[str]:
    """Placeholders still present in the author block; empty means set."""
    return [p for p in PLACEHOLDERS if p in owner_tex]


def load_macros(*files: pathlib.Path) -> dict[str, str]:
    """Zero-argument macros defined by \\newcommand in `files`."""
    macros: dict[str, str] = {}
    for f in files:
        if f.exists():
            for name, body in re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$",
                                         f.read_text(encoding="utf-8"), re.M):
                macros[name] = body
    return macros


def detex_abstract(tex: str, macros: dict[str, str] | None = None) -> str:
    """The abstract as arXiv's form will count it: macros expanded, markup
    removed. Counting the unexpanded source would measure macro NAMES, not
    the numbers a reader sees."""
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("no abstract environment")
    s = m.group(1)
    for name in sorted(macros or {}, key=len, reverse=True):
        body = macros[name]
        s = re.sub(r"\\" + name + r"(?![A-Za-z])(\\ )?",
                   lambda mm, body=body: body + (" " if mm.group(1) else ""), s)
    s = re.sub(r"\\ensuremath\{(.*?)\}(?=[\s.,;:)]|$)", r"\1", s)
    s = re.sub(r"\\times\s*10\^\{(-?\d+)\}", lambda mm: "\u00d710^" + mm.group(1), s)
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(?:emph|textit|textbf|textsc|mbox|text)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\cite[pt]?\{[^}]*\}", "", s)
    s = s.replace("\\%", "%").replace("~", " ").replace("---", "\u2014")
    s = s.replace("--", "\u2013").replace("{,}", ",").replace("\\,", " ")
    s = s.replace("$\\times$", "\u00d7").replace("\\times", "\u00d7")
    s = s.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def log_problems(log: str, blg: str) -> list[str]:
    probs: list[str] = []
    for pat, what in [
        (r"LaTeX Warning: Reference `[^']+' on page \d+ undefined", "undefined reference"),
        (r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", "undefined citation"),
        (r"There were undefined references", "undefined references"),
        (r"Label\(s\) may have changed", "labels may have changed"),
        (r"Package natbib Warning: Citation", "undefined citation (natbib)"),
        (r"LaTeX Error:", "LaTeX error"),
        (r"Float too large for page", "float too large"),
    ]:
        for m in re.finditer(pat, log):
            probs.append(f"{what}: {m.group(0)[:120]}")
    for m in re.finditer(r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too (?:wide|high)\)[^\n]*", log):
        if float(m.group(1)) > OVERFULL_PT:
            probs.append(f"overfull box: {m.group(0)[:120]}")
    for m in re.finditer(r"^Warning--.*$", blg, re.M):
        # apsrev4-2.bst prints this control-entry notice on EVERY run, with
        # any bibliography; it is the one BibTeX line exempted, by exact text.
        if m.group(0).strip() == "Warning--jnrlst (dependency: not reversed) set 1":
            continue
        probs.append(f"bibtex: {m.group(0)[:120]}")
    return probs


def float_pileup(aux: str) -> list[str]:
    """Floats typeset after the last section begins.

    REVTeX's `floatfix` prints "A float is stuck" whenever it rescues a
    congested float queue, including when the float then lands one column
    later beside its reference -- so that warning cannot be the test. The
    defect it warns of is floats deferred to the end of the paper, and that
    is measurable: every figure and table label's page, from the .aux,
    against the page on which the final section starts."""
    labels = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"\\newlabel\{([^}]*)\}\{\{[^}]*\}\{(\d+)\}", aux)}
    sections = [p for k, p in labels.items() if k.startswith("sec:")]
    if not sections:
        return ["no section labels in the .aux; float placement not checkable"]
    last = max(sections)
    return [f"float {k} placed on page {p}, after the final section begins on page {last}"
            for k, p in labels.items()
            if k.startswith(("fig:", "tab:")) and p > last]


def build(name: str, release: bool) -> int:
    src = PAPER / name
    main = src / "main.tex"
    tex = main.read_text(encoding="utf-8")
    owner = (PAPER / "owner.tex").read_text(encoding="utf-8")
    held = owner_placeholders(owner)
    if release and held:
        print(f"REFUSED: paper/owner.tex still holds {', '.join(held)}. The author "
              f"name and affiliation are the owner's to set; a release build "
              f"will not produce a finished-looking file without them.")
        return 2
    numbers = (src / "generated" / "numbers.tex").read_text(encoding="utf-8")
    if release and "[pending]" in numbers:
        pending = sorted(set(re.findall(r"\\newcommand\{\\(\w+)\}\{\\textbf\{\[pending\]\}\}", numbers)))
        print(f"REFUSED: {len(pending)} numbers are still pending an experiment "
              f"({', '.join(pending[:6])}{', ...' if len(pending) > 6 else ''}); a release build "
              f"will not typeset a result that does not exist yet.")
        return 2
    abstract = detex_abstract(tex, load_macros(src / "generated" / "numbers.tex"))
    out = src / "build"
    shutil.rmtree(out, ignore_errors=True)
    if out.exists():
        # ignore_errors hides WHY: on Windows a shell or viewer holding the
        # directory open is the usual cause. A stale build must never be
        # mistaken for this one, so refuse rather than build into it.
        print(f"REFUSED: could not clear {out} (is a shell or PDF viewer holding it open?)")
        return 3
    out.mkdir()
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           "-file-line-error", f"-outdir={out}", "main.tex"]
    # BibTeX runs inside the output directory, where a relative path to the
    # shared bibliography does not resolve; name the directory instead.
    env = {**os.environ, "BIBINPUTS": str(PAPER) + os.pathsep}
    r = subprocess.run(cmd, cwd=src, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    log = (out / "main.log").read_text(encoding="utf-8", errors="replace") \
        if (out / "main.log").exists() else ""
    blg = (out / "main.blg").read_text(encoding="utf-8", errors="replace") \
        if (out / "main.blg").exists() else ""
    probs = log_problems(log, blg)
    if (out / "main.aux").exists():
        probs += float_pileup((out / "main.aux").read_text(encoding="utf-8", errors="replace"))
    if r.returncode != 0:
        probs.insert(0, f"latexmk exited {r.returncode}")
    if len(abstract) > ARXIV_ABSTRACT_LIMIT:
        probs.append(f"arXiv abstract is {len(abstract)} characters "
                     f"(limit {ARXIV_ABSTRACT_LIMIT})")
    title = re.search(r"\\title\{([^}]*)\}", tex)
    (out / "arxiv_metadata.txt").write_text(
        f"TITLE\n{title.group(1) if title else '?'}\n\nABSTRACT "
        f"({len(abstract)} of {ARXIV_ABSTRACT_LIMIT} characters)\n{abstract}\n",
        encoding="utf-8")
    pdf = out / "main.pdf"
    pages = re.search(r"Output written on .*?\((\d+) pages?", log, re.S)
    print(f"BUILD {name}: {'draft' if not release else 'release'}  "
          f"pdf={'yes' if pdf.exists() else 'NO'}  "
          f"pages={pages.group(1) if pages else '?'}  abstract={len(abstract)} chars")
    for p in probs:
        print("  FAIL", p)
    if r.returncode != 0 and not probs[1:]:
        print(r.stdout[-3000:])
    return 1 if probs else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--release", action="store_true")
    a = ap.parse_args()
    return build(a.name, a.release)


if __name__ == "__main__":
    raise SystemExit(main())
