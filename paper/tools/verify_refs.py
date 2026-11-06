r"""Check every bibliography entry against the record its identifier resolves to.

WHY. Two citations in paper 1 were wrong in ways no proofreading catches,
because the wrong part was never compared with anything: an author list
invented for a preprint whose search snippet showed none, and a venue ("CACM
2011") taken from a reference list tagged "from prior knowledge, not
fetched". A citation is a claim about other people's work. This tool makes
each one a checked claim.

WHAT IT COMPARES, per entry:
  * doi     -> Crossref (DataCite for Zenodo DOIs): title, author family
               names in order, year, volume, issue, first page;
  * eprint  -> the arXiv abstract page's citation metadata: author family
               names in order, and the title when the entry has no DOI.
A collaboration author (double braces) must match the source's first
author or creator name. A trailing "others" matches any continuation.

MODES.
  --fetch   query the sources, write refs_verified.json, report mismatches.
  (default) OFFLINE: compare refs.bib against the stored snapshot. This is
            what the test suite runs; a hand edit to refs.bib that
            disagrees with what the source said is a red without network.

    python paper/tools/verify_refs.py --fetch
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import pathlib
import re
import sys
import time
import unicodedata
import urllib.request

PAPER = pathlib.Path(__file__).resolve().parents[1]
BIB = PAPER / "refs.bib"
SNAPSHOT = PAPER / "refs_verified.json"
UA = {"User-Agent": "ONE_Q-refcheck/1.0 (mailto:Josh@coherenceenergylabs.com)"}

# ------------------------------------------------------------------ parsing

_ENTRY = re.compile(r"@(\w+)\{([^,\s]+),\s*\n(.*?)\n\}", re.S)
_FIELD = re.compile(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$")

_TEX_ACCENT = re.compile(r"\{?\\[\"'`^~=.uvHckr]\{?([A-Za-z])\}?\}?")


def parse_bib(text: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for kind, key, body in _ENTRY.findall(text):
        fields = {"_type": kind.lower()}
        for line in body.splitlines():
            m = _FIELD.match(line)
            if m:
                fields[m.group(1).lower()] = m.group(2)
        if key in entries:
            raise ValueError(f"duplicate bibliography key {key}")
        entries[key] = fields
    return entries


def _detex(s: str) -> str:
    s = _TEX_ACCENT.sub(r"\1", s)
    return s.replace("{", "").replace("}", "")


def norm_text(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    # A SPACING accent (category Sk, e.g. U+00B8 in Crossref's "Badzia" U+00B8 "g") decomposes under NFKD into a
    # SPACE plus a combining mark, so the name read "badzia g" and the correct "Badzi{\k{a}}g" was refused
    # (2026-11-03). Non-ASCII spacing accents are dropped first; ASCII ^ and ` are left as they were.
    s = "".join(c for c in s if not (ord(c) > 127 and unicodedata.category(c) == "Sk"))
    s = unicodedata.normalize("NFKD", _detex(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def family(name: str) -> str:
    """The family name's last word, normalised.

    'Given Family' -> last word; 'Family, Given' (BibTeX's form for a
    multi-word family name such as 'Le Gall, Francois') -> last word before
    the comma. Both give 'gall' for Le Gall."""
    part = name.split(",")[0] if "," in name else name
    words = norm_text(part).split()
    return words[-1] if words else ""


_INITIAL = re.compile(r"[A-Z]\.(-?[A-Z]\.)*")
_PARTICLES = {"de", "van", "von", "der", "den", "da", "di", "la", "le", "du", "del"}


def ambiguous_names(field: str) -> list[str]:
    """Names BibTeX would split by guessing: 'Given Family' form with two or more FULL-WORD given tokens.

    BibTeX takes the last word as the family name, so 'Laleh Aghababaie Beni' typeset as 'L. A. Beni' while the
    author's own recent records give the family name 'Aghababaie Beni' (2026-11-02, paper 3's referee). The
    sources cannot settle it either: arXiv's metadata splits on the last word too. Such a name must be written
    'Family, Given', so that a person has decided the split and BibTeX no longer guesses."""
    raw = field.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return []
    out = []
    for name in (p.strip() for p in raw.split(" and ")):
        if name == "others" or "," in name or name.startswith("{"):
            continue
        given = name.split()[:-1]
        full = [g for g in given if not _INITIAL.fullmatch(g) and g.lower() not in _PARTICLES]
        if len(full) >= 2:
            out.append(name)
    return out


def bib_authors(field: str) -> tuple[list[str], bool, str | None]:
    """(family names, open-ended with 'others', collaboration or None)."""
    raw = field.strip()
    # the field regex has already removed the outer braces, so a
    # collaboration written {{Name}} arrives here as {Name}
    if raw.startswith("{") and raw.endswith("}"):
        return [], False, norm_text(raw)
    parts = [p.strip() for p in raw.split(" and ")]
    others = parts[-1] == "others"
    if others:
        parts = parts[:-1]
    return [family(p) for p in parts], others, None


# ------------------------------------------------------------------ fetching

def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=40) as r:
                return r.read()
        except Exception:                                   # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_doi(doi: str) -> dict:
    if doi.lower().startswith("10.5281/zenodo"):
        a = json.loads(_get("https://api.datacite.org/dois/" + doi))["data"]["attributes"]
        return {"source": "datacite", "title": a["titles"][0]["title"],
                "authors": [c["name"] for c in a["creators"]],
                "year": int(a["publicationYear"])}
    m = json.loads(_get("https://api.crossref.org/works/" + doi))["message"]
    dp = (m.get("published-print") or m.get("published-online")
          or m.get("issued"))["date-parts"][0]
    authors = []
    for a in m.get("author", []):
        authors.append(a["family"] if "family" in a else a.get("name", ""))
    return {"source": "crossref", "title": m["title"][0] if m.get("title") else "",
            "authors": authors, "year": int(dp[0]),
            "volume": m.get("volume"), "issue": m.get("issue"),
            "page": m.get("page") or m.get("article-number")}


GITHUB_RELEASE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/releases/tag/([\w.\-]+)")


def fetch_github(url: str) -> dict:
    """The release's publication year and the README's words (the authors' own statement of authorship)."""
    owner, repo, tag = GITHUB_RELEASE.fullmatch(url).groups()
    rel = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"))
    readme = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/readme"))
    import base64
    text = base64.b64decode(readme["content"]).decode("utf-8", "replace")
    return {"source": "github", "year": int(rel["published_at"][:4]), "release": rel.get("name", tag),
            "readme_words": " ".join(sorted(set(norm_text(text).split())))}


GITHUB_COMMIT_FILE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/blob/([0-9a-f]{40})/([\w./\-]+)")


def fetch_github_commit_file(url: str) -> dict:
    """A file of a public repository at a pinned commit that no release carries (2026-11-03, paper 3: Pan's research
    records E-037 and E-043 are on his main branch after release v2.0). The file must exist at that commit; the
    entry's year is the commit's; the repository's README AT THAT COMMIT must name every cited author."""
    owner, repo, sha, path = GITHUB_COMMIT_FILE.fullmatch(url).groups()
    commit = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"))
    blob = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={sha}"))
    readme = json.loads(_get(f"https://api.github.com/repos/{owner}/{repo}/readme?ref={sha}"))
    import base64
    text = base64.b64decode(readme["content"]).decode("utf-8", "replace")
    body = base64.b64decode(blob.get("content", "")).decode("utf-8", "replace")
    heading = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    date = commit["commit"]["committer"]["date"]
    return {"source": "github-commit", "commit": commit["sha"], "date": date, "year": int(date[:4]),
            "path": path, "blob": blob.get("sha"), "heading": heading,
            "readme_words": " ".join(sorted(set(norm_text(text).split())))}


# A record's own identifier in a title ("E-037"): the token an entry's (possibly translated) title must share with the
# file's first line, since a translated title cannot be compared with its source word for word.
RECORD_ID = re.compile(r"\b[A-Z]{1,3}-\d{2,4}\b")


PYPI_PROJECT = re.compile(r"https://pypi\.org/project/([\w.-]+)/?")


def fetch_pypi(url: str) -> dict:
    """A package's own requested citations: every BibTeX block in its PyPI project description (the README).

    Software released on PyPI with no DOI, arXiv id or GitHub release (2026-11-03: ldpc, whose repository has no
    releases) is held to the authors' own statement of how to cite it, field by field."""
    name = PYPI_PROJECT.fullmatch(url).group(1)
    info = json.loads(_get(f"https://pypi.org/pypi/{name}/json"))["info"]
    blocks = parse_bib((info.get("description") or "").replace("\r\n", "\n"))
    return {"source": "pypi", "project": name, "latest_version": info.get("version"),
            "attributions": [{k: v for k, v in b.items() if k in ("_type", "title", "author", "year", "url")}
                             for b in blocks.values()]}


def fetch_arxiv(eprint: str) -> dict:
    page = _get("https://arxiv.org/abs/" + eprint).decode("utf-8", "replace")

    def meta(name):
        return [html.unescape(x) for x in
                re.findall(r'<meta name="citation_' + name + r'" content="([^"]*)"', page)]
    title = meta("title")
    if not title:
        raise ValueError(f"arXiv page for {eprint} carried no citation metadata")
    authors = [a.split(",")[0].strip() for a in meta("author")]
    date = meta("date")
    return {"source": "arxiv", "title": title[0], "authors": authors,
            "year": int(date[0][:4]) if date else None}


# ------------------------------------------------------------------ comparing

def compare(key: str, e: dict, snap: dict) -> list[str]:
    errs: list[str] = []
    fams, others, collab = bib_authors(e.get("author", ""))

    def authors_match(src: list[str], what: str):
        if collab is not None:
            if not src or norm_text(src[0]) != collab:
                errs.append(f"{key}: collaboration {collab!r} != {what} first "
                            f"author {src[:1]}")
            return
        # Sources give FAMILY names (Crossref's `family`, the part of arXiv's
        # "Family, Given" before the comma). A bib name matches when its last
        # word is one of that family's words: 'van Tilborg' and 'Le Gall'
        # match, and so does Crossref's 'Taghavi N.' for Taghavi.
        sf = [set(norm_text(a).split()) for a in src]
        n = len(fams)
        if (len(sf) < n) or (not others and len(sf) != n) or any(
                f not in words for f, words in zip(fams, sf)):
            shown = [norm_text(a) for a in src[:n + 1]]
            errs.append(f"{key}: authors {fams}{'+others' if others else ''} "
                        f"!= {what} {shown}{'...' if len(src) > n + 1 else ''}")

    if "doi" in e:
        s = snap.get("doi")
        if not s:
            return [f"{key}: no fetched record for doi {e['doi']}"]
        if norm_text(e.get("title", "")) != norm_text(s["title"]):
            errs.append(f"{key}: title {e.get('title')!r} != {s['source']} {s['title']!r}")
        authors_match(s["authors"], s["source"])
        if int(e["year"]) != s["year"]:
            errs.append(f"{key}: year {e['year']} != {s['source']} {s['year']}")
        for bf, sf in (("volume", "volume"), ("number", "issue")):
            if bf in e and s.get(sf) and norm_text(e[bf]) != norm_text(s[sf]):
                errs.append(f"{key}: {bf} {e[bf]} != {s['source']} {s[sf]}")
            if s.get(sf) and bf not in e and s["source"] == "crossref" and bf == "volume":
                errs.append(f"{key}: source has volume {s[sf]}; entry omits it")
        if "pages" in e and s.get("page"):
            if e["pages"].split("-")[0] != str(s["page"]).split("-")[0]:
                errs.append(f"{key}: first page {e['pages']} != {s['source']} {s['page']}")
    if "eprint" in e:
        s = snap.get("arxiv")
        if not s:
            return errs + [f"{key}: no fetched record for eprint {e['eprint']}"]
        if collab is None:
            # arXiv lists a collaboration's members individually, so a
            # collaboration entry is held to its DOI record instead.
            authors_match(s["authors"], "arXiv")
        elif "doi" not in e:
            errs.append(f"{key}: a collaboration author needs a DOI record to check")
        if "doi" not in e:
            if norm_text(e.get("title", "")) != norm_text(s["title"]):
                errs.append(f"{key}: title {e.get('title')!r} != arXiv {s['title']!r}")
            if int(e["year"]) != s["year"]:
                errs.append(f"{key}: year {e['year']} != arXiv {s['year']}")
    if "doi" not in e and "eprint" not in e and GITHUB_RELEASE.fullmatch(e.get("url", "")):
        # A work released only as a GitHub repository (no DOI, no arXiv id yet). The release must exist with the
        # entry's year, and the repository's own README must name every cited author: GitHub accounts need not carry
        # a person's name, so the README -- the authors' statement of authorship -- is the source checked.
        s = snap.get("github")
        if not s:
            return errs + [f"{key}: no fetched record for {e['url']}"]
        if int(e["year"]) != s["year"]:
            errs.append(f"{key}: year {e['year']} != GitHub release {s['year']}")
        words = set(s["readme_words"].split())
        missing = [f for f in fams if f not in words]
        if collab is not None or missing or not fams:
            errs.append(f"{key}: author(s) {missing or fams} not named in the repository README")
        return errs
    if "doi" not in e and "eprint" not in e and GITHUB_COMMIT_FILE.fullmatch(e.get("url", "")):
        # A file at a pinned commit: it existed there, the year is the commit's, and the README at that commit names
        # every cited author (the same authorship rule as a release)
        s = snap.get("github_commit")
        if not s:
            return errs + [f"{key}: no fetched record for {e['url']}"]
        _, _, owner_repo_sha, url_path = GITHUB_COMMIT_FILE.fullmatch(e["url"]).groups()
        if s.get("commit") != owner_repo_sha or not s.get("blob"):
            errs.append(f"{key}: the file is not recorded at commit {owner_repo_sha}")
        if s.get("path") != url_path:
            errs.append(f"{key}: url path {url_path!r} != the fetched file {s.get('path')!r}")
        # the title may be a translation (it must then say so); it must carry the record's own identifier, as the
        # file's first line does, and nothing else of that form
        ids_title = set(RECORD_ID.findall(e.get("title", "").replace("{", "").replace("}", "")))
        ids_head = set(RECORD_ID.findall(s.get("heading", "")))
        if not ids_head or ids_title != ids_head:
            errs.append(f"{key}: title identifiers {sorted(ids_title)} != the file heading's {sorted(ids_head)}")
        if any(ord(ch) >= 0x2E80 for ch in s.get("heading", "")) and "translat" not in e.get("note", "").lower():
            errs.append(f"{key}: the file's heading is not in English; the entry's note must say the title is translated")
        if int(e["year"]) != s["year"]:
            errs.append(f"{key}: year {e['year']} != commit {s['year']}")
        words = set(s["readme_words"].split())
        missing = [f for f in fams if f not in words]
        if collab is not None or missing or not fams:
            errs.append(f"{key}: author(s) {missing or fams} not named in the repository README at that commit")
        return errs
    if "doi" not in e and "eprint" not in e and PYPI_PROJECT.fullmatch(e.get("url", "")):
        # Software on PyPI: the entry must BE one of the citations the package's README asks for -- same url, title,
        # year and authors -- since the README is the authors' statement of how the work is to be cited.
        s = snap.get("pypi")
        if not s:
            return errs + [f"{key}: no fetched record for {e['url']}"]
        same_url = [a for a in s["attributions"] if a.get("url", "").rstrip("/") == e["url"].rstrip("/")]
        if not same_url:
            return errs + [f"{key}: the {s['project']} README asks to be cited with no entry at {e['url']}"]
        a = same_url[0]
        if norm_text(e.get("title", "")) != norm_text(a.get("title", "")):
            errs.append(f"{key}: title {e.get('title')!r} != the README's {a.get('title')!r}")
        if str(e.get("year")) != str(a.get("year")):
            errs.append(f"{key}: year {e.get('year')} != the README's {a.get('year')}")
        authors_match(bib_authors(a.get("author", ""))[0], "the README's citation")
        return errs
    if "doi" not in e and "eprint" not in e:
        errs.append(f"{key}: no doi, eprint, GitHub release or PyPI project -- nothing to verify it against")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fetch-new", action="store_true",
                    help="fetch only entries the snapshot lacks, leaving every other record as it was fetched (refs.bib "
                         "is shared by several manuscripts; a full re-fetch rewrites their records too)")
    a = ap.parse_args()
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    if a.fetch or a.fetch_new:
        if a.fetch:
            snapshot = {"fetched": _dt.date.today().isoformat(), "entries": {}}
            todo = entries
        else:
            snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
            todo = {k: e for k, e in entries.items() if k not in snapshot["entries"]}
            snapshot.setdefault("fetched_later", {}).update({k: _dt.date.today().isoformat() for k in todo})
        for key, e in todo.items():
            rec = {}
            if "doi" in e:
                rec["doi"] = fetch_doi(e["doi"])
            if "eprint" in e:
                rec["arxiv"] = fetch_arxiv(e["eprint"])
            if "doi" not in e and "eprint" not in e and GITHUB_RELEASE.fullmatch(e.get("url", "")):
                rec["github"] = fetch_github(e["url"])
            if "doi" not in e and "eprint" not in e and GITHUB_COMMIT_FILE.fullmatch(e.get("url", "")):
                rec["github_commit"] = fetch_github_commit_file(e["url"])
            if "doi" not in e and "eprint" not in e and PYPI_PROJECT.fullmatch(e.get("url", "")):
                rec["pypi"] = fetch_pypi(e["url"])
            snapshot["entries"][key] = rec
            print(f"  fetched {key}", flush=True)
        # newline="\n": the repository pins *.json to LF, and Windows text
        # mode would otherwise write CRLF into a tracked artifact
        SNAPSHOT.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8", newline="\n")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    errs = []
    for key, e in entries.items():
        if key not in snapshot["entries"]:
            errs.append(f"{key}: not in the snapshot -- run with --fetch")
            continue
        errs += compare(key, e, snapshot["entries"][key])
        errs += [f"{key}: author {n!r} has several full given names, so BibTeX guesses the family name; "
                 f"write it 'Family, Given'" for n in ambiguous_names(e.get("author", ""))]
    stale = set(snapshot["entries"]) - set(entries)
    errs += [f"{k}: in the snapshot but not in refs.bib" for k in sorted(stale)]
    print(f"REFERENCES -- {len(entries)} entries checked against sources fetched "
          f"{snapshot['fetched']}")
    for x in errs:
        print("  MISMATCH", x)
    if not errs:
        print("  every entry agrees with its source")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
