r"""Build the arXiv upload for one manuscript, and prove it compiles on its own.

    python paper/tools/arxiv_package.py circuit-distance        # after `build.py circuit-distance --release`

WHAT arXiv NEEDS. A flat source tree: arXiv compiles the upload in its own root, so `\input{../owner.tex}` cannot
reach outside it, and arXiv does not run BibTeX on an upload that carries a .bbl. The package therefore holds main.tex
with its two shared inputs rewritten to sit beside it, owner.tex and release.tex, every generated table and figure the
manuscript reads, the .bbl the release build produced, and, for paper 3, its supplementary proofs under anc/ (arXiv
publishes anc/ as ancillary files).

WHY THE COMPILE CHECK. A tarball is only an upload if it builds from nothing else. The package is extracted into an
empty temporary directory and compiled there twice with pdflatex, no BibTeX, no access to the repository; the page
count must equal the release build's, and the log must carry no undefined reference or citation. Otherwise the tool
refuses and writes nothing to paper/<name>/build/arxiv/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ANC = {"circuit-distance": ["experiments/circuit_distance_exact/WINDOW_PROOF.md",
                            "experiments/circuit_distance_exact/FOLD_PROOF.md"]}


def pages(pdf: pathlib.Path) -> int:
    raw = pdf.read_bytes()
    counts = [int(x) for x in re.findall(rb"/Type\s*/Pages\b.*?/Count\s+(\d+)", raw, re.S)]
    return max(counts) if counts else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args()
    src = PAPER / a.name
    build = src / "build"
    bbl = build / "main.bbl"
    if not (build / "main.pdf").is_file() or not bbl.is_file():
        raise SystemExit(f"REFUSED: run `python paper/tools/build.py {a.name} --release` first")
    if "{UNSET}" in (PAPER / "release.tex").read_text(encoding="utf-8"):
        raise SystemExit("REFUSED: paper/release.tex names no minted release")
    tex = (src / "main.tex").read_text(encoding="utf-8")
    flat = tex.replace(r"\input{../owner.tex}", r"\input{owner.tex}").replace(r"\input{../release.tex}",
                                                                             r"\input{release.tex}")
    if "../" in flat:
        raise SystemExit("REFUSED: main.tex still reaches outside its directory after flattening")
    files: dict[str, bytes] = {"main.tex": flat.encode("utf-8"), "main.bbl": bbl.read_bytes(),
                               "owner.tex": (PAPER / "owner.tex").read_bytes(),
                               "release.tex": (PAPER / "release.tex").read_bytes()}
    for sub in ("generated", "figures"):
        for p in sorted((src / sub).glob("*")) if (src / sub).is_dir() else []:
            if p.is_file() and p.suffix in (".tex", ".pdf", ".png"):
                files[f"{sub}/{p.name}"] = p.read_bytes()
    for rel in ANC.get(a.name, []):
        files[f"anc/{pathlib.Path(rel).name}"] = (ROOT / rel).read_bytes()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(files[name]), 0, 0o644
            tf.addfile(info, io.BytesIO(files[name]))
    blob = buf.getvalue()

    with tempfile.TemporaryDirectory(prefix="arxiv_") as td:
        t = pathlib.Path(td)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(t, filter="data")
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=t,
                               capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL)
        log = (t / "main.log").read_text(encoding="utf-8", errors="replace") if (t / "main.log").exists() else ""
        if r.returncode != 0 or not (t / "main.pdf").exists():
            raise SystemExit(f"REFUSED: the package does not compile on its own:\n{r.stdout[-1500:]}")
        bad = [ln for ln in log.splitlines() if re.search(r"undefined|Citation .* undefined|Reference .* undefined",
                                                           ln, re.I) and "Font" not in ln]
        if bad:
            raise SystemExit("REFUSED: undefined references or citations in the standalone build:\n  "
                             + "\n  ".join(bad[:10]))
        n_pkg, n_rel = pages(t / "main.pdf"), pages(build / "main.pdf")
        if n_pkg != n_rel:
            raise SystemExit(f"REFUSED: the standalone build has {n_pkg} pages, the release build {n_rel}")

    out = build / "arxiv"                       # inside build/, which git ignores: the package is derived
    out.mkdir(exist_ok=True)
    tgz = out / f"{a.name}_arxiv.tar.gz"
    tgz.write_bytes(blob)
    meta = (build / "arxiv_metadata.txt").read_text(encoding="utf-8") if (build / "arxiv_metadata.txt").exists() else ""
    (out / "PACKAGE.json").write_text(json.dumps({
        "manuscript": a.name, "tarball": tgz.name, "sha256": hashlib.sha256(blob).hexdigest(), "pages": n_pkg,
        "files": sorted(files), "standalone_compile": "pdflatex x2 in an empty directory, no BibTeX: OK",
        "ancillary": [f"anc/{pathlib.Path(r).name}" for r in ANC.get(a.name, [])]}, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    shown = tgz.relative_to(ROOT) if tgz.is_relative_to(ROOT) else tgz
    print(f"ARXIV PACKAGE {a.name}: {len(files)} files, {n_pkg} pages, compiles standalone -> {shown}")
    if meta:
        print(meta[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
