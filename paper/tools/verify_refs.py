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
    if "doi" not in e and "eprint" not in e:
        errs.append(f"{key}: no doi or eprint -- nothing to verify it against")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    if a.fetch:
        snapshot = {"fetched": _dt.date.today().isoformat(), "entries": {}}
        for key, e in entries.items():
            rec = {}
            if "doi" in e:
                rec["doi"] = fetch_doi(e["doi"])
            if "eprint" in e:
                rec["arxiv"] = fetch_arxiv(e["eprint"])
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
