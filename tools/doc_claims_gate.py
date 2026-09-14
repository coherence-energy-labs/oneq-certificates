#!/usr/bin/env python3
r"""Doc-claims gate: prose may not state a number the evidence does not support.

WHY THIS EXISTS, AND WHOSE MISTAKE IT CATCHES. For several sessions the status
reports said "6/6 PROMOTABLE". The verifier said 5/6 and had said 5/6 all along.
The number in the prose was authored by hand from memory, conflating RE-VERIFIED
(6/6: rebuild, witness, arithmetic, replica agreement) with PROMOTABLE (5/6,
which additionally demands versioned schema, recomputed hashes, a valid
signature, and provably DISTINCT replica information sets).

Every automated gate in this repo was green while that claim was false, because
no gate read the prose. The engine was disciplined and the sentence about the
engine was not -- and the sentence is what a reader receives.

THE PRINCIPLE, ALREADY APPLIED ONE LEVEL DOWN. The passport was made honest by
DERIVING each claim's wording from its evidence level instead of authoring it.
This is that same rule lifted to human-facing documents: a headline count is
recomputed from the artifacts, and a document asserting a different one fails.

WHAT IT DELIBERATELY DOES NOT DO. It does not check English, judge tone, or
verify claims that are not numeric. It checks the small set of countable
headlines that have actually been misreported. A gate that tried to audit all
prose would be unfalsifiable and would be switched off within a week.

    python tools/doc_claims_gate.py
    python tools/doc_claims_gate.py --json evidence/doc_claims_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pathrel import rel  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


#: Directories whose .md files are NOT this repository's live prose.
#: `release` and `.build` hold frozen and staging copies of a published
#: bundle, whose numbers are historical by construction.
_NOT_LIVE_PROSE = (".git", "release", ".build", "node_modules", ".venv")


def doc_files(root=None):
    r"""The documents this gate judges -- ONE list, for every caller.

    *** THIS LIST WAS WRITTEN TWICE AND THE COPIES DREW APART. ***
    `tests/test_end_to_end.py` re-implemented the same comprehension, so
    adding `.build` here on 2026-09-06 fixed the gate and left the test
    reading a different set of documents -- the test then failed on a
    staging copy the gate had correctly stopped scanning.

    The test directly below that one in the suite is
    `test_the_headline_and_the_doc_gate_cannot_disagree`, whose docstring
    says "one scorer, two readers -- the root fix for a drifting
    denominator". The denominator drifted anyway, because only the SCORER
    was shared and the CORPUS was not. It is shared now.
    """
    root = ROOT if root is None else root
    return [p for p in root.rglob("*.md")
            if not any(x in p.parts for x in _NOT_LIVE_PROSE)]


def ground_truth() -> dict:
    """Recompute every countable headline FROM THE ARTIFACTS.

    THIS IMPORTS THE VERIFIER'S SCORER RATHER THAN RE-DERIVING IT, and the
    reason is the whole point of this file. A first version reimplemented the
    counting and disagreed with the headline three separate ways: it read result
    keys that do not exist (reporting 0 closures), it scoped closures by
    `improves_published` (0 again -- these closures CONFIRM the published value
    and prove it exact rather than raising it), and it counted 7 records against
    a headline of 6. Each was a re-derivation of a rule that already existed.

    A gate against drifting numbers must not be a second source of drift, so the
    scope lives in verify_closures.score_records and both readers call it.
    """
    from verify_closures import (_catalogue, _records, catalogue_is_pinned,
                                 score_records)
    cat = _catalogue()
    # THIS CONDITION WAS DEAD, TWICE OVER.
    #
    # catalogue_is_pinned takes ROWS and returns (bool, digest).
    # `_catalogue()` returns a dict keyed by code_id, so the call raised
    # TypeError and the `except` set pinned = True. Even had it not
    # raised, bool() on a non-empty tuple is always True. Both routes
    # reported PINNED regardless of the catalogue.
    #
    # score_records treats this as load-bearing -- "promotion is
    # withheld when the INPUT cannot be identified" -- so a condition
    # that cannot be False silently promotes against a catalogue nobody
    # identified. verify_closures.py has always unpacked it correctly;
    # the two readers the docstring says "cannot disagree" did.
    #
    # No fallback: if the catalogue cannot be read, that is a RED, not
    # a pass. Defaulting to True is how the check died in the first
    # place.
    pinned, _cat_digest = catalogue_is_pinned(list(cat.values()))
    sc = score_records(list(_records()), cat, pinned)
    out = {"closures_total": sc["total"],
           "closures_reverified": sc["reverified"],
           "closures_promotable": sc["promotable"],
           "tautological_replica_sets": sc["tautological_replica_sets"],
           "scope": "from verify_closures.score_records -- the SAME function "
                    "that prints the headline, so the two cannot disagree"}
    # SELF-CHECK: these cannot legitimately be zero while records exist. Without
    # it, a key-name drift inside the verifier silently converts this gate into
    # a generator of false accusations against correct prose -- and a gate that
    # cries wolf is switched off, taking the real checks with it.
    if sc["total"] == 0 or sc["reverified"] == 0:
        raise SystemExit(
            "doc_claims_gate: scored 0 closures from a non-empty record set. "
            "REFUSING to judge prose against a number it cannot itself trust.")
    mg = ROOT / "evidence" / "mutation_gate.json"
    if mg.exists():
        d = json.loads(mg.read_text(encoding="utf-8"))
        res = d.get("results") or d.get("mutants") or []
        out["mutants_total"] = d.get("total", len(res))
        out["mutants_killed"] = d.get("killed", sum(
            1 for r in res if r.get("killed")))
    out.update(_suite_size())
    out.update(_wall_size())
    out.update(_roadmap_position())
    return out


def _roadmap_position() -> dict:
    """How far along the roadmap says the program is.

    *** DERIVED, AND HELD BY NOTHING. *** `evidence/roadmap.json` is
    written by a standing gate on every wall run, and the README's front
    page cited it as **"14 of 33 roadmap gates closed"** while it said
    **22 of 34**. The sentence carrying the number promises it is
    *"derived from artifacts on disk and held there by a standing
    gate"*: the deriving was real, the holding was not, because the
    claim is spelled "N of M" and every ratio pattern here expected
    "N/M".

    Absent or unreadable is UNCHECKABLE, never zero -- a missing
    artifact must not license whatever is typed.
    """
    p = ROOT / "evidence" / "roadmap.json"
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if "closed" not in d or "gates" not in d:
        return {}
    # *** AND IT MUST BE AS FRESH AS THE LEDGER IT DESCRIBES. ***
    #
    # MEASURED 2026-08-26: closing `ML-GAP` moved the position to 23 of
    # 34 while this artifact still said 22, and README's "22 of 34" was
    # checked against it AND PASSED. A check reading a number exactly as
    # old as the claim it validates cannot catch a drift between them.
    #
    # `wall_census.json` had solved this with `roster_source_sha256` and
    # nothing had ported it here. Same rule, same reason: a position
    # derived by a DIFFERENT roadmap_gate.py is not wrong, it is
    # UNCHECKABLE, and one expected red after editing the ledger is the
    # price of refusing to guess.
    #
    # AFTER the well-formedness guard, not before it: an artifact with
    # its counts missing is UNCHECKABLE whichever ledger wrote it, and
    # running this first reported "predates the stamp" for a malformed
    # file. The existing test said so within the minute.
    import hashlib
    # THE TOOL IS FOUND BESIDE THIS MODULE, not under ROOT. A test that
    # patches ROOT is redirecting where EVIDENCE is read from; the tools
    # do not move with it, and resolving them under the patched root made
    # this check silently unreachable in every fixture.
    tool = pathlib.Path(__file__).resolve().parent / "roadmap_gate.py"
    if not tool.is_file():
        return {}
    live = hashlib.sha256(tool.read_bytes()).hexdigest()
    stamped = d.get("source_sha256")
    if stamped != live:
        return {"roadmap_note":
                ("roadmap.json predates the freshness stamp"
                 if stamped is None else
                 "roadmap.json came from a DIFFERENT "
                 "tools/roadmap_gate.py") +
                "; regenerate it with `python tools/roadmap_gate.py "
                "--json evidence/roadmap.json`. If you just edited the "
                "ledger, this is the one expected red."}
    return {"roadmap_closed": d["closed"], "roadmap_gates": d["gates"]}


# *** THE HOLE THIS GATE HAD FROM THE DAY IT WAS WRITTEN. ***
#
# It checked closure counts and mutant counts, and nothing else, while
# README.md's headline table said "139 files, 1,296 tests" and
# "all 29 gates". Measured: 1,426 selected of 1,429 collected, across 33
# verdict rows. Off by 130 tests and 4 gates, green the whole time --
# because the gate that exists to stop prose drifting from the artifacts
# was only pointed at two of the numbers in the prose.
#
# A gate aimed one claim away from the defect cannot see it, and this is
# that, inside the very tool written against it.

def _suite_size() -> dict:
    """The suite's size, COLLECTED LIVE. No artifact, so never stale.

    Collection is the honest source: it applies the same `-m` selection
    and `--strict-markers` the wall runs under, so `tests_total` is the
    number a reader reproduces by running the documented command rather
    than a number some earlier run wrote down. It costs ~4 s against a
    ~200 s wall.
    """
    r = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "pytest", "tests/", "-q",
         "--collect-only", "--strict-markers", "-m", "not slow_exhaustive"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+tests collected(?:\s*\((\d+)\s+"
                  r"deselected\))?", r.stdout)
    if not m:                      # no deselection: "N tests collected"
        m2 = re.search(r"(\d+)\s+tests collected", r.stdout)
        if not m2:
            raise SystemExit(
                "doc_claims_gate: could not read a test count from "
                "collection. REFUSING to judge prose about the suite's "
                "size against a number it failed to obtain -- a gate that "
                "guesses here would license any figure.\n"
                + r.stdout[-800:] + r.stderr[-800:])
        sel = tot = int(m2.group(1))
        desel = 0
    else:
        sel, tot = int(m.group(1)), int(m.group(2))
        desel = int(m.group(3) or 0)
    files = len(list((ROOT / "tests").glob("test_*.py")))
    return {"tests_total": sel, "tests_collected": tot,
            "tests_deselected": desel, "test_files": files,
            **_tree_size()}


def _tree_size() -> dict:
    r"""Modules, tools and experiment lanes, counted from the tree.

    *** THE GATE WAS AIMED ONE CLAIM AWAY, AGAIN. *** The README's own
    status table opens "every number below is recomputed by a gate", and
    two of its rows were not: `113 modules in src/oneq, 74 tools` had
    drifted to 122 and 87 -- an 8% and an 18% error sitting one line
    above the paragraph that tells the story of the last time a headline
    drifted. `tests` and `gates` were checked because they were the two
    that had embarrassed us before, which is the definition of a gate
    aimed at the last defect instead of at the class.

    `__init__.py` is excluded: it re-exports and is not a module a
    reader would count. A dotfile or a `.pyi` would be, so the glob is
    `*.py` rather than anything cleverer.

    *** AND THE LANE COUNT WAS UNSTABLE BY CONSTRUCTION. *** It counted
    every directory under `experiments/`, which includes `__pycache__` --
    a build artifact, gitignored, present in a worktree that has run
    anything and absent from a fresh clone. So the claim it checks had a
    different correct value depending on whether the reader had run the
    suite, and no value of the README could be right in both. A gate
    whose expected answer depends on build output is the same defect as a
    hash pin that reads the working tree instead of the object database
    (`git_pin.py`), and it is fixed the same way: count only what a fresh
    clone holds.
    """
    mods = [p for p in (ROOT / "src" / "oneq").glob("*.py")
            if p.name != "__init__.py"]
    tools = list((ROOT / "tools").glob("*.py"))
    # *** AND EXCLUDING `__pycache__` DID NOT FINISH THE JOB. *** The
    # rule above is right -- count only what a fresh clone holds -- and
    # `iterdir()` cannot implement it: git does not track empty
    # directories, so `experiments/arr` and `experiments/qsep`, both
    # empty, were counted here and absent from every clone. Measured:
    # the clone holds 20 lanes and this worktree reported 22, so the
    # README could not be right in both places, which is the same defect
    # one layer in.
    #
    # AN EMPTY DIRECTORY IS NOT A LANE, and that rule needs no git --
    # which matters, because the mutation sandbox copies the tree
    # WITHOUT `.git` and a git-based count would disagree with itself
    # there. "Holds any file" is the predicate, not "holds a `.py`":
    # `c0_admissibility_audit` and `framework_tests` are real lanes
    # carrying data and documents, and a code-only rule dropped them,
    # giving 18 against the clone's 20. Verified against
    # `git ls-files experiments` -- the two sets are equal, symmetric
    # difference empty.
    lanes = [p for p in (ROOT / "experiments").iterdir()
             if p.is_dir() and not p.name.startswith(("__", "."))
             and any(f.is_file() for f in p.rglob("*"))]
    # The layout map's other three counts, by the same rule -- what a
    # fresh clone holds: `verifier-js/*.mjs` (all tracked), the top-level
    # directories under `release/` (each holds tracked files; a tarball
    # beside its directory is the same bundle, not a tenth), and every
    # `.md` under `docs/`. The evidence count was NOT given a truth: a
    # gitignored smoke artifact (`evidence/gtct_wall_probe.json`) sits at
    # the top level of any worktree that has run the wall, so no single
    # number is right in both a clone and a worktree -- the README's
    # `evidence/` line states no count instead.
    verifiers = list((ROOT / "verifier-js").glob("*.mjs"))
    release = ROOT / "release"
    bundles = ([p for p in release.iterdir()
                if p.is_dir() and not p.name.startswith(("__", "."))]
               if release.is_dir() else [])
    docs = [p for p in (ROOT / "docs").rglob("*.md")
            if "__pycache__" not in p.parts]
    return {"modules_total": len(mods), "tools_total": len(tools),
            "lanes_total": len(lanes), "verifiers_total": len(verifiers),
            "release_total": len(bundles), "docs_total": len(docs)}


def _wall_size() -> dict:
    """How many gates the wall runs -- from the wall's OWN LAST RUN.

    Only a full run knows: `--fast`, a missing node and a missing
    iverilog each legitimately change the count, so no static reading
    can answer it. `tools/gate_all.py` writes `evidence/wall_census.json`
    at the end of every run, carrying the sha256 of its own source.

    *** STALENESS IS A RED, NOT A SKIP. *** If that digest no longer
    matches, the wall has been edited since the census was taken and the
    count in prose CANNOT be checked. Returning nothing here would make
    the claim silently unchecked, which is the dark-gate shape this
    repository keeps finding; instead the truth dict carries an
    explicitly poisoned value that no prose can match, so the gate goes
    red and says why.
    """
    def unavailable(why: str) -> dict:
        # EVERY key this function can supply must be PRESENT and None on
        # every failure path. An early `return` that omits `skips_total`
        # does not make the claim unchecked-and-loud, it makes it
        # invisible: `scan` skips a key that is not in the truth dict at
        # all. That is how "0 skips" in the README's headline row sat
        # unexamined while the same claim in the quick start was caught
        # -- one line apart, one absent key.
        return {"gates_total": None, "gates_note": why,
                "skips_total": None, "skips_note": why}

    cen = ROOT / "evidence" / "wall_census.json"
    if not cen.exists():
        return unavailable("no evidence/wall_census.json -- run "
                           "`python tools/gate_all.py` once")
    d = json.loads(cen.read_text(encoding="utf-8"))
    live = hashlib.sha256(
        (ROOT / "tools" / "gate_all.py").read_bytes()).hexdigest()
    if d.get("roster_source_sha256") != live:
        # EXPECTED EXACTLY ONCE after the wall itself is edited, and the
        # message has to say so or it reads as a defect. The census is
        # written at the END of a run and this gate runs partway through,
        # so a run that CHANGED tools/gate_all.py necessarily judges the
        # previous run's census. The census that run writes makes the
        # next one green. That one red is the price of refusing to guess,
        # and guessing here would mean agreeing with whatever the README
        # says about a wall nobody has run.
        return unavailable("wall_census.json came from a DIFFERENT "
                           "tools/gate_all.py. If you just edited the "
                           "wall, this is the one expected red: the "
                           "census written at the END of this run makes "
                           "the next one green. Otherwise, run "
                           "`python tools/gate_all.py`")
    if d.get("fast_mode"):
        return unavailable("the census came from a --fast run, which "
                           "deliberately runs fewer gates and no suite "
                           "than the counts prose refers to")
    out = {"gates_total": d["gates_ran"],
           "gates_declared": d["gates_declared"]}
    # "0 skips" is a claim, and in a repository whose doctrine is that a
    # skip is a dark gate it is a load-bearing one. It cannot come from
    # collection -- only a RUN knows -- so it comes from the suite's own
    # junit report, and its absence is UNCHECKABLE rather than zero.
    s = d.get("suite") or {}
    out["skips_total"] = s.get("tests_skipped")
    if out["skips_total"] is None:
        out["skips_note"] = s.get("suite_report", "no suite census")
    return out


# Each pattern captures (numerator, denominator) and names the truth it must
# match. Anchored on the shapes that were actually misreported.
_PATTERNS = (
    (re.compile(r"(\d+)\s*/\s*(\d+)\s+(?:closures?\s+)?PROMOTABLE", re.I),
     "closures_promotable", "closures_total"),
    (re.compile(r"(\d+)\s*/\s*(\d+)\s+(?:mutants?\s+)?killed", re.I),
     "mutants_killed", "mutants_total"),
    # *** THE POSITION ON THE ROADMAP IS A HEADLINE TOO. ***
    # MEASURED 2026-08-24: the README front page said **"14 of 33
    # roadmap gates closed"** while `evidence/roadmap.json` -- written by
    # a standing gate, in the same repository, on every wall run -- said
    # 22 of 34. Eight gates and one whole row of drift, on the line that
    # tells a reader how far the program has got, and no pattern here
    # could see it because it is spelled "N of M" rather than "N/M".
    #
    # The sentence containing it promises the number is "derived from
    # artifacts on disk and held there by a standing gate". It was
    # derived; nothing held it.
    # The `\*{0,2}` is load-bearing and was found by the pattern failing
    # to fire on the very line it was written for: the README bolds the
    # figure, so the text is `**22 of 34** roadmap gates` and the
    # asterisks sit between the second number and the noun. A pattern
    # that silently matches nothing is the same dark check as a
    # `--select` that selects nothing.
    (re.compile(r"\b(\d+)\s+of\s+(\d+)\*{0,2}\s+roadmap\s+gates\b", re.I),
     "roadmap_closed", "roadmap_gates"),
)


#: SINGLE-NUMBER headlines. Each capture is one count, matched against
#: one recomputed quantity.
#:
#: These are the two the README got wrong -- "1,296 tests" against 1,426
#: and "29 gates" against 33 -- and they were wrong for weeks because
#: every pattern above this line is a RATIO. A drift check that only
#: understands `n/m` cannot see a bare count, which is most of what a
#: headline table contains.
#:
#: Thousands separators are part of the shape: the README writes 1,296
#: and a pattern without `[\d,]` silently matches the "296" and compares
#: THAT, which passes or fails for reasons unrelated to the claim.
_SINGLE = (
    (re.compile(r"\b(\d[\d,]*)\s+tests\b(?!\s+collected)", re.I),
     "tests_total"),
    # `the 199-mutant registry` -- a bare total wearing a HYPHEN.
    # MEASURED 2026-08-22: the front page said "the 178-mutant registry"
    # while the artifact recorded 199, a drift of 21 that this gate could
    # not see because every pattern here expected a SPACE before the
    # noun. It sat in the section headed "A gate that cannot fail is not
    # a gate", which is the joke the previous fix already made once: the
    # gate was aimed one shape away from its own front page.
    #
    # Scoped by construction -- singles are only checked in
    # _HEADLINE_DOCS -- so the "14-mutant" and "26-mutant" figures in the
    # blueprint documents, which are about specific modules rather than
    # this registry, are not swept in.
    (re.compile(r"\b(\d[\d,]*)-mutants?\b", re.I), "mutants_total"),
    (re.compile(r"\b(\d[\d,]*)\s+(?:test\s+)?files\b", re.I), "test_files"),
    (re.compile(r"\b(?:all\s+)?(\d[\d,]*)\s+gates\b", re.I), "gates_total"),
    (re.compile(r"\b(\d[\d,]*)\s+skips?\b", re.I), "skips_total"),
    # `113 modules in src/oneq, 74 tools, 22 experiment lanes` -- the
    # row that drifted while the line above it promised a gate was
    # recomputing it. "modules" is required to name `src/oneq` so a
    # sentence about somebody else's module count is not swept in.
    (re.compile(r"\b(\d[\d,]*)\s+modules?\s+in\s+`?src[\\/]oneq`?", re.I),
     "modules_total"),
    (re.compile(r"\b(\d[\d,]*)\s+tools\b", re.I), "tools_total"),
    (re.compile(r"\b(\d[\d,]*)\s+experiment\s+lanes\b", re.I),
     "lanes_total"),
)

#: *** THE SCOPE, AND WHY IT IS NOT "EVERY DOCUMENT". ***
#:
#: A bare count has no subject. Run the patterns above over the whole
#: tree and they flag "177 tests" in a table about coherence_lang, "13
#: tests" meaning the thirteen in one file, "5 gates" meaning one
#: experiment's five -- nineteen findings, seventeen of them wrong. A
#: gate that cries wolf is switched off within a week, taking the real
#: checks with it, and this file says so at the top.
#:
#: A ratio (`5/6 PROMOTABLE`) carries its own subject and can be matched
#: anywhere. A total cannot, so the total is checked only where the
#: document's job is to state THIS repository's totals.
#: *** REPRODUCE.md ADDED 2026-09-13, AND IT SHOULD HAVE BEEN HERE FIRST. ***
#: The fence-blindness this file already fixed once for the README ("it sat
#: inside a fence, so only the mutant-count comment patterns ever looked at
#: it, and it drifted 130 tests") was never extended to the document a
#: STRANGER actually runs. Measured today: REPRODUCE.md told a third-party
#: reproducer to expect `~1,240 tests` against 4,177 and `16 gates` against
#: 101 -- drifts of 3.4x and 6.3x, both inside fences, both invisible here.
#: Git says the figures were true when written (2026-08-19 and 2026-08-04,
#: the latter in a commit titled "I cloned the repo and followed my own
#: instructions"), which is exactly how a reproduction package rots: it is
#: correct on the day it is written and nothing watches it afterwards.
#: A stranger comparing their terminal to these lines is the first
#: credibility test either manuscript faces, so this document states THIS
#: repository's totals as much as the front page does.
_HEADLINE_DOCS = ("README.md", "REPRODUCE.md")

#: THE LAYOUT MAP IS A HEADLINE IN A FENCE. The README's `## Repository
#: layout` block is a fenced table of counts, and on 2026-09-04 five of
#: them were wrong at once -- "113 modules" (151), "75 gates" (98 tools),
#: "29 standing gates" (78), "168 files" (225), "6 checkers" (7), "45
#: documents" (54) -- one screen below a status table this gate keeps
#: exact. The transcript exemption skipped the block because it is
#: fenced; but nothing PRINTED it. Inside these sections of a headline
#: doc, fenced lines are scanned with _SINGLE plus the shapes below,
#: which name their subject so they cannot match anything else.
_LAYOUT_SECTIONS = ("Repository layout",)
_LAYOUT = (
    (re.compile(r"\b(\d[\d,]*)\s+zero-dependency\s+second-language\s+"
                r"checkers\b", re.I), "verifiers_total"),
    (re.compile(r"\b(\d[\d,]*)\s+signed,\s+stranger-verifiable\s+bundles\b",
                re.I), "release_total"),
    (re.compile(r"\b(\d[\d,]*)\s+documents\b", re.I), "docs_total"),
)

#: What keeps that allowlist from becoming the loophole: a line ANYWHERE
#: that states a count while invoking the repo-wide command is a headline
#: living in an unlisted file, and is reported as one.
_REPO_WIDE = re.compile(r"pytest\s+tests/|tools[\\/]gate_all\.py")


# A fenced line that INVOKES something is an instruction, so its
# trailing comment asserts rather than records.
_COMMAND_RE = re.compile(r"^\s*(?:\$\s*)?(?:python|pytest|node|bash|sh|"
                         r"npm|make|git)\b")

# Bare counts in such comments, mapped to the artifact quantity.
_COMMENT_COUNTS = (
    (re.compile(r"#.*?(\d+)\s+mutants?\b", re.I), "mutants_total"),
)


def scan(truth: dict, docs) -> tuple[list, list]:
    bad, scoped = [], []
    for path in docs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # QUOTED OUTPUT IS EVIDENCE, NOT AN ASSERTION.
        #
        # A fenced block holding a verifier transcript records what the tool
        # PRINTED at the time. One such block in the red-team audit shows a
        # pre-fix "6/6 PROMOTABLE" -- the exact behaviour that audit exists to
        # document. Editing it to match today's number would falsify the record
        # to satisfy a gate, which is the failure mode this gate is against.
        fenced, lines = False, text.splitlines()
        in_headline_doc = rel(path, ROOT).replace("\\", "/") in _HEADLINE_DOCS
        section = ""
        for line_no, line in enumerate(lines, 1):
            if line.startswith("#") and not fenced:
                section = line.lstrip("#").strip()
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            # A LAYOUT MAP IS A TABLE OF CLAIMS, NOT A TRANSCRIPT. Nothing
            # printed the README's `## Repository layout` block; it is
            # prose that happens to be monospaced, and the transcript
            # exemption below let five of its counts rot (see _LAYOUT).
            in_layout = (fenced and in_headline_doc
                         and section in _LAYOUT_SECTIONS)
            if fenced:
                # ...BUT A COMMAND'S TRAILING COMMENT IS AN ASSERTION,
                # NOT A TRANSCRIPT. `python tools/mutation_gate.py
                # # 58 mutants, all must be killed` is a claim about
                # what the tool WILL do; nothing printed it. That
                # distinction is why "58 mutants" survived in AUDIT.md
                # and REPRODUCE.md while the artifact said 90 -- the
                # gate held the right number and skipped the line.
                #
                # Transcripts stay exempt: they are lines the tool
                # PRINTED, and rewriting one to satisfy a gate would
                # falsify the record this gate exists to protect.
                stripped = line.strip()
                if not _COMMAND_RE.match(stripped) and not in_layout:
                    continue
                # THE QUICK START IS A HEADLINE TOO. `python -m pytest
                # tests/ -q   # 1,296 tests, 0 skips expected` is the
                # first command a reader runs and the first number they
                # compare against their own terminal. It sat inside a
                # fence, so only the mutant-count comment patterns ever
                # looked at it, and it drifted 130 tests.
                counts = _COMMENT_COUNTS + (
                    _SINGLE if in_headline_doc else ()) + (
                    _LAYOUT if in_layout else ())
                for pat, key in counts:
                    for m in pat.finditer(stripped):
                        got = int(m.group(1).replace(",", ""))
                        want = truth.get(key)
                        # An UNCHECKABLE truth is not a pass here either
                        # -- the fenced quick start is where a reader
                        # first compares a number to their own terminal.
                        if want is None and key in {
                                k for _p, k in _SINGLE + _LAYOUT}:
                            bad.append({
                                "file": rel(path, ROOT), "line": line_no,
                                "wrote": m.group(0),
                                "evidence_says": "UNCHECKABLE: " + str(
                                    truth.get(key.split("_")[0] + "_note",
                                              "no recomputed value")),
                                "claim": key, "text": stripped[:150]})
                        elif want is not None and got != want:
                            # SAME RECORD SHAPE as the pattern branch
                            # below -- my first version appended a
                            # tuple, so the gate detected the drift and
                            # then crashed formatting its own report.
                            # A finding that cannot be printed is a
                            # finding nobody acts on.
                            bad.append({
                                "file": rel(path, ROOT),
                                "line": line_no,
                                "wrote": m.group(0),
                                "evidence_says": str(want),
                                "claim": key,
                                "text": stripped[:150]})
                continue
            for pat, num_key, den_key in _PATTERNS:
                for m in pat.finditer(line):
                    if num_key not in truth:
                        continue
                    n, d = int(m.group(1)), int(m.group(2))
                    tn, td = truth[num_key], truth.get(den_key)
                    if n == tn and (td is None or d == td):
                        continue
                    # A CORRECTED HISTORICAL REFERENCE IS NOT AN OVERCLAIM,
                    # and the exemption is self-limiting: a line may cite a
                    # superseded figure ONLY if it also states the true one.
                    # There is no way to abuse this without printing the truth,
                    # which is the outcome the gate wants anyway. A qualifier
                    # whitelist ("as of", "previously") would have been a
                    # loophole -- three words and any claim goes quiet.
                    # *** THE EXEMPTION MUST SPEAK THE CLAIM'S OWN
                    # SPELLING. *** It looked only for `tn/td`, while the
                    # roadmap-position pattern matches `N of M` -- so for
                    # that family the exemption was UNREACHABLE, and a
                    # paragraph correctly quoting the superseded figure
                    # beside the true one was flagged anyway. Measured
                    # the moment the first such paragraph was written:
                    # the restamping pass then rewrote the quotation to
                    # today's number, falsifying the record to satisfy
                    # the gate -- the exact failure this gate exists
                    # against, caused by the gate.
                    #
                    # Still self-limiting: a line may cite a superseded
                    # figure ONLY if it also states the true one, in
                    # either spelling.
                    if (f"{tn}/{td}" in line
                            or re.search(rf"\b{tn}\s+of\s+{td}\b", line)):
                        scoped.append({"file": rel(path, ROOT),
                                       "line": line_no, "cites": m.group(0),
                                       "alongside_truth": f"{tn}/{td}"})
                        continue
                    bad.append({
                            "file": rel(path, ROOT),
                            "line": line_no, "wrote": m.group(0),
                            "evidence_says": f"{tn}/{td}",
                            "claim": num_key,
                            "text": line.strip()[:150]})
            for pat, key in _SINGLE:
                if key not in truth:
                    continue
                if not in_headline_doc:
                    # An unlisted document may still be stating a repo
                    # total -- if it does, the allowlist is hiding drift
                    # rather than avoiding noise, and it must be told.
                    if _REPO_WIDE.search(line) and pat.search(line):
                        bad.append({
                            "file": rel(path, ROOT), "line": line_no,
                            "wrote": pat.search(line).group(0),
                            "evidence_says": (
                                f"{truth[key]} -- and this line states a "
                                f"repo-wide count while invoking the "
                                f"repo-wide command, so it is a headline "
                                f"in a document not listed in "
                                f"_HEADLINE_DOCS"),
                            "claim": key, "text": line.strip()[:150]})
                    continue
                for m in pat.finditer(line):
                    want = truth[key]
                    got = int(m.group(1).replace(",", ""))
                    # *** A TRUTH OF None IS NOT A PASS. *** It means the
                    # count could not be recomputed -- a stale census, a
                    # --fast run. Skipping the line there would leave the
                    # claim silently unchecked, which is the dark gate
                    # this file exists against, so it reds and prints the
                    # reason the value is unavailable.
                    if want is None:
                        bad.append({
                            "file": rel(path, ROOT), "line": line_no,
                            "wrote": m.group(0),
                            "evidence_says": "UNCHECKABLE: " + str(
                                truth.get(key.split("_")[0] + "_note",
                                          truth.get("gates_note", "?"))),
                            "claim": key, "text": line.strip()[:150]})
                        continue
                    if got == want:
                        continue
                    # Same self-limiting exemption as the ratio branch: a
                    # line may cite a superseded figure only if it prints
                    # the true one beside it.
                    if re.search(rf"\b{want:,}\b|\b{want}\b",
                                 line.replace(m.group(0), "", 1)):
                        scoped.append({"file": rel(path, ROOT),
                                       "line": line_no, "cites": m.group(0),
                                       "alongside_truth": str(want)})
                        continue
                    bad.append({
                        "file": rel(path, ROOT), "line": line_no,
                        "wrote": m.group(0), "evidence_says": str(want),
                        "claim": key, "text": line.strip()[:150]})
    return bad, scoped


def main() -> int:
    # The wall sets PYTHONIOENCODING=utf-8; a reader running this gate
    # by hand on Windows gets cp1252 and a UnicodeEncodeError the moment
    # a finding quotes a line containing an arrow. A gate that crashes
    # while PRINTING a real finding reports nothing, which is the worst
    # of both outcomes -- it found the drift and told no one.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):    # pragma: no cover
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    truth = ground_truth()
    # `.build` is TRANSIENT STAGING, and it was missing from `_NOT_LIVE_PROSE`
    # while two sibling gates already skipped it -- `hash_pin_audit.SKIP_DIRS`
    # and `prior_art_gate.SKIP`, the latter naming exactly this pair,
    # "release" and ".build". The consequence was live: a candidate bundle
    # staged on 2026-09-05 still carried "458 mutants" after the registry
    # reached 466, so a leftover COPY of two documents reddened a gate about
    # the repository's own prose.
    docs = doc_files()
    bad, scoped = scan(truth, docs)

    print("DOC-CLAIMS GATE -- headlines recomputed from the artifacts\n")
    for k, v in truth.items():
        if k not in ("tautological_replica_sets", "scope"):
            print(f"  {k:28s} {v}")
    for t in truth["tautological_replica_sets"]:
        print(f"  ! {t['code_id']}: {t['certificates']} certificates but "
              f"{t['distinct_runs']} distinct run(s) -- 'replicas agree' is a "
              f"TAUTOLOGY here, not evidence")
    print(f"\n  scanned {len(docs)} documents")
    if bad:
        print(f"\n  {len(bad)} PROSE CLAIM(S) THE EVIDENCE DOES NOT SUPPORT:\n")
        for b in bad:
            print(f"    {b['file']}:{b['line']}  wrote '{b['wrote']}'  "
                  f"evidence says {b['evidence_says']}")
            print(f"        {b['text']}")
    else:
        print("\n  every countable headline matches the artifacts")
    if a.json:
        pathlib.Path(a.json).write_text(
            json.dumps({"ground_truth": truth, "violations": bad,
                        "corrected_historical_references": scoped}, indent=2),
            encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
