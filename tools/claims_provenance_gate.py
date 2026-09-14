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
    tex = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
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


def truth_set() -> set[str]:
    """Every number derivable from the evidence tree, in doc formats."""
    vals: set[str] = set()

    def add(v):
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            vals.add(f"{v:,}")
            vals.add(str(v))
        elif isinstance(v, float):
            for x in (v, v * 100):
                vals.add(f"{x:.2f}")
                r = round(x, 2)
                if r == int(r):
                    vals.add(f"{int(r):,}")
            # seconds-valued artifacts are quoted as milliseconds in prose
            vals.add(f"{int(round(v * 1000)):,}")

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

    for p in (ROOT / "evidence").rglob("*.json"):
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return vals


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
    print(f"CLAIMS PROVENANCE -- {len(truth):,} numbers in the evidence "
          f"tree, whitelist {len(wl)}")
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
