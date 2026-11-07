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


# Top-level directories of the released tree; a \texttt path starting with one of them is a claim about that tree.
RELEASED_TOPS = ("src", "tools", "experiments", "evidence", "docs", "tests", "paper", "verifier-js", "conformance",
                 "lean", "schemas", "rtl", "data")
BUNDLE_ENV = "ONEQ_RELEASE_BUNDLE"
#: Which of the release's bundles carries each manuscript; the rest are in the decoder bundle. Since 2026-11-03 paper 4
#: travels in its own signed bundle in the same tag (tests/test_build_release_tree_gate.py holds this to the builder).
MANUSCRIPT_BUNDLE = {"superactivation": "oneq-superactivation"}
DEFAULT_BUNDLE = "oneq-certifying-decoder"


def bundle_dir(name: str = "", env: dict | None = None, paper: pathlib.Path = PAPER) -> pathlib.Path:
    """The minted bundle `--release` checks manuscript `name`'s named paths against: the bundle that carries it.

    * $ONEQ_RELEASE_BUNDLE names a directory HOLDING the bundles (a trial mint's .build/mint_trial), so the next
      release is rehearsed against its own bundles without writing under release/, which holds the published ones;
    * run from inside a minted bundle (SHA256SUMS beside paper/), the tree itself is the bundle -- a reader rebuilding
      the paper from the release has no release/ copy of it;
    * otherwise release/<bundle>.
    The bundle checked still has to name the tag release.tex names."""
    sub = MANUSCRIPT_BUNDLE.get(name, DEFAULT_BUNDLE)
    v = (os.environ if env is None else env).get(BUNDLE_ENV, "").strip()
    if v:
        return pathlib.Path(v).resolve() / sub
    if (paper.parent / "SHA256SUMS").is_file():
        return paper.parent
    return paper.parent / "release" / sub


BUNDLE = bundle_dir()
FILE_EXTS = "py|json|jsonl|md|mjs|js|c|h|lean|tex|gz|txt|csv|ps1|sh"
# Files a manuscript names that are someone else's and are not claimed to be in our release -- each with its reason.
EXTERNAL_NAMES = {
    "decoder_setup.py": "IBM's own qLDPC repository file, named by P3 as the source of IBM's circuit model",
}


def named_paths(tex: str) -> set[str]:
    r"""Every repository path a manuscript names in \texttt or \nolinkurl, normalised (\allowbreak and \_ removed)."""
    out = set()
    for m in re.finditer(r"\\(?:texttt|nolinkurl)\{((?:[^{}]|\{[^{}]*\})*)\}", tex):
        p = re.sub(r"\\allowbreak(?:\{\})?\s*|\\(?=_)|\s+", "", m.group(1))
        if re.fullmatch(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-*]+)+/?", p) and p.split("/", 1)[0] in RELEASED_TOPS:
            out.add(p)
        elif re.fullmatch(r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*\.(?:" + FILE_EXTS + ")", p):
            out.add(p)                 # a file named by its bare name: some file of that name must ship
    return out


def release_tag(release_tex: str) -> str | None:
    m = re.search(r"\\newcommand\{\\ReleaseTag\}\{([^}]*)\}", release_tex)
    return m.group(1) if m else None


def release_tree_misses(tex: str, tag: str, bundle: pathlib.Path = BUNDLE) -> list[str]:
    r"""Why the minted bundle cannot be the release the manuscript cites, or the paths it names that it lacks.

    *** WHY THIS IS HERE (2026-11-03). *** The revised manuscripts named a preregistration, its per-shot records and
    an input contract "in the release", while `\ReleaseTag` still pointed at the tag the external audits had reviewed,
    which contains none of them; the release build passed, because it refused only the {UNSET} placeholder. A
    referee found it by fetching the paths at the tag. The bundle ships its own `paper/release.tex`, so its tag and
    its `SHA256SUMS` together say exactly what that release carries.
    """
    sums, rel = bundle / "SHA256SUMS", bundle / "paper" / "release.tex"
    if not (sums.is_file() and rel.is_file()):
        return [f"no minted bundle at {bundle} (SHA256SUMS and paper/release.tex) to check the named paths against"]
    if release_tag(rel.read_text(encoding="utf-8")) != tag:
        return [f"the minted bundle is release {release_tag(rel.read_text(encoding='utf-8'))}, not {tag}"]
    files = {ln.split("  ", 1)[1].strip() for ln in sums.read_text(encoding="utf-8").splitlines() if "  " in ln}
    missing = []
    for p in sorted(named_paths(tex) - set(EXTERNAL_NAMES)):
        if "*" in p:
            rx = re.compile(re.escape(p).replace(r"\*", "[^/]*"))
            hit = any(rx.fullmatch(f) or rx.match(f) for f in files)
        elif p.endswith("/"):
            hit = any(f.startswith(p) for f in files)
        elif "/" not in p:
            hit = any(f == p or f.endswith("/" + p) for f in files)
        else:
            hit = p in files or any(f.startswith(p + "/") for f in files)
        if not hit and p in release_asset_names(bundle, files):
            hit = True
        if not hit:
            missing.append(p)
    return missing


#: Shipped records that list files attached to the release as ASSETS rather than carried in the tree (2026-11-06:
#: paper 3 names the two archives of its large proofs; external audit 3). A name counts as carried only when the record
#: itself ships in the bundle, so a manuscript cannot name an asset the release does not describe.
RELEASE_ASSET_RECORDS = ("evidence/circuit_distance_exact/recert_20261102/release_assets.json",)


def release_asset_names(bundle: pathlib.Path, files: set[str]) -> set[str]:
    import json
    out = set()
    for rec in RELEASE_ASSET_RECORDS:
        if rec in files and (bundle / rec).is_file():
            out |= {a["name"] for a in json.loads((bundle / rec).read_text(encoding="utf-8")).get("archives", [])}
    return out


def cites_release(tex: str, numbers: str) -> bool:
    r"""Does the manuscript cite the release, directly (\ReleaseTag) or through a generated macro whose body is it?

    Paper 4 names its release as \LaneRelease, which its generator derives as \ReleaseTag when the builder ships the
    lane. Testing the manuscript alone for the literal \ReleaseTag skipped the release-tree check for paper 4, and its
    release build passed against a bundle that named another tag (found 2026-11-03 preparing the next mint)."""
    if "\\ReleaseTag" in tex:
        return True
    via = [n for n, body in re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", numbers, re.M) if "\\ReleaseTag" in body]
    return any(re.search(r"\\" + n + r"(?![A-Za-z])", tex) for n in via)


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


#: Words with an fi/fl/ff ligature. A PDF whose fonts lack a Unicode map drops these letters from its text layer
#: ("certificate" extracts as "certicate"), so search and copy fail although the page looks right (external audit 4,
#: X1, 2026-11-07: every PDF of v2026.11.05 used 12-13 bitmap Type 3 fonts from T1 fontenc without lmodern).
LIGATURE = re.compile(r"\b[A-Za-z]*(?:ffi|ffl|fi|fl|ff)[A-Za-z]*\b")


def type3_fonts(pdffonts_out: str) -> list[str]:
    """The names of the Type 3 (bitmap) fonts in `pdffonts` output (columns: name, type ..., the header ruled off)."""
    lines = pdffonts_out.splitlines()
    rule = next((i for i, ln in enumerate(lines) if ln.startswith("----")), None)
    if rule is None:
        return []
    return [ln.split()[0] for ln in lines[rule + 1:] if " Type 3 " in f" {ln} "]


def lost_ligatures(abstract: str, pdf_text: str, k: int = 12) -> list[str]:
    """Ligature words of the abstract that the PDF's extracted text does not contain (up to `k` distinct words)."""
    flat = re.sub(r"-\s*\n\s*", "", pdf_text)                        # undo hyphenation at line ends
    flat = re.sub(r"\s+", " ", flat)
    words = []
    for w in LIGATURE.findall(abstract):
        if w.lower() not in (x.lower() for x in words):
            words.append(w)
    return [w for w in words[:k] if re.search(r"\b" + re.escape(w) + r"\b", flat, re.I) is None]


def text_layer_problems(pdf: pathlib.Path, abstract: str) -> tuple[list[str], str | None]:
    """(problems, note): Type 3 fonts, and abstract words whose ligatures the text layer lost. The note says when
    the poppler tools are absent, so a skipped check is announced, never silent."""
    fonts, totext = shutil.which("pdffonts"), shutil.which("pdftotext")
    if not (fonts and totext and pdf.exists()):
        return [], "text-layer check NOT run (pdffonts/pdftotext not on PATH)"
    probs = []
    f = subprocess.run([fonts, str(pdf)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    t3 = type3_fonts(f.stdout)
    if t3:
        probs.append(f"{len(t3)} Type 3 (bitmap) font(s) without a reliable text layer: {', '.join(t3[:4])}")
    t = subprocess.run([totext, "-enc", "UTF-8", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    lost = lost_ligatures(abstract, t.stdout)
    if lost:
        probs.append(f"the PDF's text layer loses ligatures: {', '.join(lost[:6])} not found by pdftotext")
    return probs, None


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
    rel = PAPER / "release.tex"
    numbers = (src / "generated" / "numbers.tex").read_text(encoding="utf-8")
    cites = cites_release(tex, numbers)
    if release and cites and (not rel.is_file() or "{UNSET}" in rel.read_text(encoding="utf-8")):
        print("REFUSED: the manuscript cites \\ReleaseTag and paper/release.tex does not name a minted release; "
              "set it when the release exists.")
        return 2
    if release and cites:
        misses = release_tree_misses(tex, release_tag(rel.read_text(encoding="utf-8")) or "", bundle_dir(name))
        if misses:
            print(f"REFUSED: release {release_tag(rel.read_text(encoding='utf-8'))} does not carry "
                  f"{len(misses)} path(s) the manuscript names as in it: {', '.join(misses[:8])}"
                  f"{', ...' if len(misses) > 8 else ''}")
            return 2
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
    layer, note = text_layer_problems(pdf, abstract)
    probs += layer
    if note:
        print("  NOTE", note)
    # TeX wraps its log at 79 characters, so on a long build path the count can start the next line (the release
    # rehearsal printed pages=? for paper 3 in C:\lq\rehearsal2\..., 2026-11-03)
    pages = re.search(r"Output written on .*?\(\s*(\d+)\s+pages?", log, re.S)
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
