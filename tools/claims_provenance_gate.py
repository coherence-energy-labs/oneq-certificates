r"""Every material number in the docs must trace to an evidence artifact.

THE PATTERN THIS GATE ENDS: the doc-claims gate checked numbers someone
REGISTERED, so every unregistered number could rot freely -- and did (stale
abstract totals twice; 737,773 in prose where the artifact says 751,060; a
bundle whose paper described artifacts it did not carry). This gate inverts
the burden of proof: it sweeps the technical docs for material numbers
(comma-grouped counts, two-decimal rates) and demands each one exist in the
evidence tree -- including `derived_claims.json`, so a sum is only ever as
fresh as its constituents. A number that traces to nothing is a red, with
its file and line, and the fix is to regenerate the artifact or correct the
prose, never to whitelist reflexively.

Second sweep, same inversion: every `src/oneq/*.py` module and
`evidence/*.json` artifact the PAPER names must be carried by the release
bundle manifest -- a paper that describes what the bundle does not contain
is a claim traveling without its receipt.

The whitelist (`tools/claims_whitelist.json`) is for numbers that are not
claims: DOIs, dollar figures, years, arXiv ids. Every entry carries a
reason, and the gate prints the whitelist size so growth is visible.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ["docs/PAPER_DRAFT.md", "docs/QLDPC_CERT_DESIGN.md",
        "docs/REALTIME_EMISSION_DESIGN.md", "docs/PAPER2_QLDPC_DRAFT.md"]
# THE TYPESET MANUSCRIPTS AND THEIR GENERATED NUMBERS. Measured values reach
# the LaTeX through macros that paper/tools/make_numbers.py writes from
# artifacts, but a number can still be typed straight into prose, and the
# generated file itself must trace -- so both are swept, after LaTeX's
# thousands separator `{,}` is read as the comma it prints.
TEX_GLOBS = ["paper/*/main.tex", "paper/*/generated/*.tex"]


def _swept_docs() -> list[str]:
    tex = sorted(str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).replace("\\", "/")
                 for g in TEX_GLOBS for p in ROOT.glob(g))
    return DOCS + tex


def _prose(text: str) -> str:
    """What a reader sees, for number matching: LaTeX `1{,}000` is 1,000;
    `\\%` is a percent sign. Markdown passes through unchanged."""
    return text.replace("{,}", ",").replace("\\%", "%")


WHITELIST = ROOT / "tools" / "claims_whitelist.json"

#: numeric lists longer than this are data series and do not feed the truth set
SERIES_MIN = 64

COMMA_NUM = re.compile(r"(?<![\d.$])(\d{1,3}(?:,\d{3})+)(?![\d%])")
RATE = re.compile(r"(?<![\d.])(\d{1,3}\.\d{2})(?=%|\b)")


def truth_set(use_cache: bool = True) -> set[str]:
    """Every number derivable from the evidence tree, in doc formats -- restricted to the strings a swept token can
    EQUAL. A token is a COMMA_NUM group (\\d{1,3}(,\\d{3})+) or a RATE group (\\d{1,3}\\.\\d{2}), so a rendering of any
    other shape can never be asked for, and leaving it out changes no membership answer. That is 4.6 million strings
    instead of 67.3 million (2026-10-19), and it is what took this gate from ~70 s per push to a fraction of it.
      * an int's `str` never has that shape, and its `f"{v:,}"` has it exactly when v >= 1,000;
      * a float's `.2f` rendering is `-?digits.dd` or nan/inf, and has the RATE shape exactly when it is 4 to 6
        characters with no sign;
      * the same rounding and int() calls run in the same order as before, so a value that raised (int(nan),
        int(inf)) still ends its file's walk at the same point, keeping what that file had added -- the old
        behaviour, exactly.
    A file's contribution is a pure function of its BYTES and of `_file_contribution`'s code, so it is cached under
    the repository's git directory, keyed by both (`_cache_dir`): a warm push reads ~4,600 small files instead of
    walking 368 MB. A fresh clone -- what the wall runs on -- has no cache and computes everything. Pass
    use_cache=False (or set ONEQ_CLAIMS_NO_CACHE=1) to compute every file regardless."""
    import os
    cache = None if (not use_cache or os.environ.get("ONEQ_CLAIMS_NO_CACHE") == "1") else _cache_dir()
    vals: set[str] = set()
    for p in (ROOT / "evidence").rglob("*.json"):
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        key = hashlib.sha256(raw).hexdigest()
        hit = cache / f"{key}.txt" if cache is not None else None
        if hit is not None and hit.is_file():
            text = hit.read_text(encoding="utf-8")
            if text:
                vals.update(text.split("\n"))
            continue
        got = _file_contribution(raw)
        vals |= got
        if hit is not None:
            tmp = hit.with_name(f"{key}.{os.getpid()}.tmp")
            try:
                tmp.write_text("\n".join(sorted(got)), encoding="utf-8", newline="\n")
                os.replace(tmp, hit)                     # atomic: a reader never sees a half-written entry
            except OSError:
                tmp.unlink(missing_ok=True)
    return vals


def _file_contribution(raw: bytes) -> set[str]:
    """ONE evidence file's token-shaped renderings, exactly as the pre-cache loop computed them: decoded as UTF-8 with
    universal newlines (what `read_text` did), parsed, walked; a decode or parse error contributes nothing, and an
    exception during the walk keeps what was added before it"""
    vals: set[str] = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            if v >= 1000:
                vals.add(f"{v:,}")
        elif isinstance(v, float):
            for x in (v, v * 100):
                s = f"{x:.2f}"
                if 4 <= len(s) <= 6 and s[0] != "-":
                    vals.add(s)
                r = round(x, 2)
                if r == int(r) and int(r) >= 1000:
                    vals.add(f"{int(r):,}")
            # seconds-valued artifacts are quoted as milliseconds in prose
            ms = int(round(v * 1000))
            if ms >= 1000:
                vals.add(f"{ms:,}")

    def walk(o):
        if isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            # A LONG NUMERIC LIST IS A DATA SERIES, NOT A CLAIM. Storing
            # per-shot gaps (2026-09-13, ~2,600 floats) put so many
            # two-decimal renderings into this set that a planted "43.21%"
            # traced by coincidence and the gate's own orphan test went
            # green-blind. A number a document quotes is a summary, and
            # summaries live in scalar fields; series are skipped.
            if len(o) > SERIES_MIN and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in o):
                return
            for x in o:
                walk(x)
        else:
            add(o)

    try:
        walk(json.loads(raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")))
    except Exception:
        pass
    return vals


def _cache_dir():
    """<git common dir>/oneq-cache/claims-truth/<code key>, or None outside a git work tree. The code key hashes
    `_file_contribution`'s source and SERIES_MIN, so any change to what a file contributes starts a fresh cache"""
    import inspect
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True,
                           timeout=20, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    code_key = hashlib.sha256((inspect.getsource(_file_contribution) + f"|SERIES_MIN={SERIES_MIN}")
                              .encode("utf-8")).hexdigest()[:16]
    d = (ROOT / r.stdout.strip()).resolve() / "oneq-cache" / "claims-truth" / code_key
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=None,
                    help="override the swept docs (ABSOLUTE paths allowed) "
                         "-- exists so the suite can plant an orphan number "
                         "in a scratch doc and PROVE this gate still fires; "
                         "a gate nobody can watch go red is decorative")
    ap.add_argument("--no-bundle-check", action="store_true")
    args = ap.parse_args()
    docs = args.docs if args.docs is not None else _swept_docs()
    wl = json.loads(WHITELIST.read_text(encoding="utf-8")) \
        if WHITELIST.exists() else {}
    truth = truth_set()
    orphans = []
    for rel in docs:
        p = pathlib.Path(rel)
        if not p.is_absolute():
            p = ROOT / rel
        if not p.exists():
            continue
        for ln, line in enumerate(_prose(p.read_text(encoding="utf-8"))
                                  .splitlines(), 1):
            for m in COMMA_NUM.finditer(line):
                tok = m.group(1)
                if tok in truth or tok in wl:
                    continue
                orphans.append((rel, ln, tok, line.strip()[:90]))
            for m in RATE.finditer(line):
                tok = m.group(1)
                if tok in truth or tok in wl:
                    continue
                orphans.append((rel, ln, tok, line.strip()[:90]))

    # BUNDLE COMPLETENESS: what the paper names, the bundle must carry
    bundle_gaps = []
    if args.no_bundle_check:
        return _report(truth, wl, orphans, bundle_gaps)
    paper = "\n".join((ROOT / d).read_text(encoding="utf-8")
                      for d in DOCS if (ROOT / d).exists())
    sys.path.insert(0, str(ROOT / "tools"))
    import build_certificate_bundle as bcb
    carried = {src for (src, _why) in bcb.CONTENTS} \
        | set(bcb.EVIDENCE) | set(bcb.OPTIONAL)
    for pat in (r"src/oneq/\w+\.py", r"evidence/[\w.]+\.json",
                r"experiments/m0_prior_art/\w+\.py", r"tools/\w+\.py"):
        for ref in set(re.findall(pat, paper)):
            if ref not in carried and (ROOT / ref).exists():
                bundle_gaps.append(ref)

    return _report(truth, wl, orphans, bundle_gaps)


def _report(truth, wl, orphans, bundle_gaps) -> int:
    print(f"CLAIMS PROVENANCE -- {len(truth):,} token-shaped renderings of the "
          f"evidence tree's numbers, whitelist {len(wl)}")
    if orphans:
        print(f"\n{len(orphans)} ORPHAN NUMBER(S) -- in the docs, in NO "
              f"artifact:")
        for rel, ln, tok, ctx in orphans[:20]:
            print(f"  {rel}:{ln}  {tok}  | {ctx}")
    if bundle_gaps:
        print(f"\n{len(bundle_gaps)} file(s) the paper names but the bundle "
              f"does not carry:")
        for r in sorted(bundle_gaps):
            print(f"  {r}")
    if orphans or bundle_gaps:
        return 1
    print("every material number traces; every named file is carried")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
