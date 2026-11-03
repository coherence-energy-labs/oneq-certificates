r"""Package the paper-3 re-certification campaign (recert_campaign.py) as committed evidence.

    python recert_package.py --run C:/lq/recert/records.jsonl C:/lq/recert --run C:/lq/recert_z/records.jsonl C:/lq/recert_z \
        --run C:/lq/recert_x/records.jsonl C:/lq/recert_x      # writes evidence/circuit_distance_exact/recert_20261102/
    python recert_package.py ... --check                      # report only, write nothing

REFUSES unless the campaign is complete and clean: every cell recert_campaign.cells() derives from the records appears
exactly once, every one VERIFIED by the committed checker, every re-exported model byte-identical to the one its record
names, and every row's prover and checker sources are the committed ones (sha256 of the bytes git stores, LF).

WHY SOME PROOFS DIFFER FROM THEIR RECORDS, AND WHY THAT IS SHOWN, NOT ASSUMED. On 2026-10-21 (R205, commit ce5e5b02)
the proof language advanced to CDP5, which omits trace bits for subtrees the checker decides itself, so a record written
by an earlier prover cannot match its regenerated CDP5 proof byte for byte. A record FILE's date does not say which
prover wrote a ROW (the first version of this tool used it, and the three-colour file -- committed after CDP5, written
by a campaign that ran across the change from another checkout -- refuted it). So every cell whose proof differs must
have been REPRODUCED: recert_old_prover.py compiles the prover of a named commit, runs it on the identical model, and
must get the record's proof sha256 bit for bit (records/recert_old_prover.jsonl). This tool REFUSES a differing cell
without such a reproduction. The proof hashes in the manifest are the regenerated CDP5 ones throughout, so they bind
proofs anyone can regenerate.

WHAT SHIPS. Every model, and the proofs of the families that fit in git (morph72, morph2026), each re-hashed against its
row; the larger proofs (three-colour, morph144) ship as hashes and the command that regenerates them. DETERMINISM is
taken from records/recert_determinism.jsonl (recert_determinism.py): proofs with trace bits regenerated at another
thread count must be byte-identical, and at least one such run must exist, or the check would be dark.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import recert_campaign as rc  # noqa: E402

OUT = ROOT / "evidence" / "circuit_distance_exact" / "recert_20261102"
SHIP = ("morph72", "morph2026")
CDP5_COMMIT = "ce5e5b02"
OLD = ROOT / "evidence" / "circuit_distance_exact" / "records" / "recert_old_prover.jsonl"
DET = ROOT / "evidence" / "circuit_distance_exact" / "records" / "recert_determinism.jsonl"


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_lf_sha(rel: str) -> str:
    """sha256 of the file as git stores it (HEAD): what every clone receives, whatever the checkout's line endings."""
    return hashlib.sha256(subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True,
                                         check=True).stdout).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    # one --run per campaign process (records file, its work directory); processes ran on disjoint cells
    ap.add_argument("--run", nargs=2, action="append", required=True, metavar=("RECORDS", "WORK"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    errors = []

    def need(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    # a cell two processes both ran (they overlap at the end) must agree byte for byte: the second row is then more
    # determinism evidence, not a conflict
    rows, work_of, repeats, aborted = [], {}, [], []
    for records, work in a.run:
        if not pathlib.Path(records).is_file():          # a run that has not finished a cell yet
            print(f"note: {records} does not exist; that run contributes nothing")
            continue
        for x in pathlib.Path(records).read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            k = rc.key(r)
            if r["proof_sha256"] is None and not r["prover"] and not r["checker"] and not r["verified"]:
                # a prover that wrote no proof and printed nothing was STOPPED (2026-11-03: a campaign process killed
                # at a cell boundary had already started the next cell), not refuted: it carries no verdict either
                # way. It is listed in the manifest, and the cell must still be completed by a run that finished
                # (the coverage check below refuses otherwise)
                aborted.append({"run": pathlib.Path(records).parent.name, "cell": f"{r['family']} {r['basis']}"
                                f"{r['rounds']} w{r['w']}", "prover_s": r["prover_s"]})
                continue
            if k in work_of:
                prev = next(p for p in rows if rc.key(p) == k)
                need(r["verified"] and r["proof_sha256"] == prev["proof_sha256"]
                     and r["model_sha256"] == prev["model_sha256"], f"{k}: two runs disagree")
                repeats.append(r)
                continue
            rows.append(r)
            work_of[k] = pathlib.Path(work)

    want = {rc.key(c): c for c in rc.cells(rc.FAMILIES)}
    got = {rc.key(r) for r in rows}
    need(got == set(want), f"cells missing: {len(set(want) - got)}; unexpected: {len(got - set(want))}")
    prover_sha = git_lf_sha("experiments/circuit_distance_exact/cdist_prove.c")
    checker_sha = git_lf_sha("experiments/circuit_distance_exact/ccheck.c")
    cdp5_date = git("show", "-s", "--format=%cI", CDP5_COMMIT)
    # every reproduction of a recorded proof by an earlier prover (recert_old_prover.py), by cell
    old = collections.defaultdict(list)
    if OLD.is_file():
        for x in OLD.read_text(encoding="utf-8").splitlines():
            if x.strip():
                o = json.loads(x)
                old[(o["family"], o["kind"], o["circuit"], o["basis"], o["rounds"], o["w"], o["maxround"])].append(o)
    reproduced_by = {}
    for r in rows:
        tag = f"{r['family']} {r['kind']} {r['circuit']} {r['basis']}{r['rounds']} w{r['w']}"
        need(r["verified"], f"{tag}: not verified")
        need(r["model_equal"], f"{tag}: model differs from its record")
        need(r["prover_source_sha256"] == prover_sha and r["checker_source_sha256"] == checker_sha,
             f"{tag}: not the committed prover and checker")
        if not r["proof_equal"]:
            hits = [o for o in old[rc.key(r)] if o["reproduces_record"]
                    and o["recorded_proof_sha256"] == r["proof_sha256_record"]]
            need(bool(hits), f"{tag}: the proof differs from its record and no earlier prover has reproduced the record")
            if hits:
                # the earlier prover must ship as a file: a reader of the release has no git history to `git show`
                src_file = HERE / "old_provers" / f"cdist_prove_{hits[0]['commit']}.c"
                need(src_file.is_file() and sha(src_file) == hits[0]["prover_source_sha256_git"],
                     f"{tag}: the reproducing prover's source is not shipped as old_provers/{src_file.name}")
                reproduced_by[rc.key(r)] = {"commit": hits[0]["commit"], "magic": hits[0]["proof_magic"],
                                            "prover_source_sha256_git": hits[0]["prover_source_sha256_git"],
                                            "prover_source_file": (src_file.relative_to(ROOT).as_posix()
                                                                   if src_file.is_relative_to(ROOT) else str(src_file))}
    # DETERMINISM (second referee of 2026-11-02, M1): only a re-run at ANOTHER thread count, of a proof that carries
    # trace bits, can show that scheduling does not change the bytes. records/recert_determinism.jsonl holds those runs;
    # a proof with no trace bits is recorded there but cannot count. Cells two campaign processes both ran (same thread
    # count) must also agree, and are reported separately.
    first = {rc.key(r): r for r in rows}
    det = [json.loads(x) for x in DET.read_text(encoding="utf-8").splitlines() if x.strip()] if DET.is_file() else []
    for d in det:
        k = (d["family"], d["kind"], d["circuit"], d["basis"], d["rounds"], d["w"], d["maxround"])
        need(k in first and d["campaign_proof_sha256"] == first[k]["proof_sha256"],
             f"determinism run: {k} is not a cell of this campaign, or compared against another proof")
        need(d["identical"], f"determinism run: {k} regenerated a different proof at {d['threads']} threads")
        need(d["threads"] != d["campaign_threads"], f"determinism run: {k} used the campaign's own thread count")
    need(sum(d["can_test_determinism"] for d in det) >= 1,
         "no determinism run on a proof with trace bits: the check would be dark")
    for r in repeats:
        need(r["verified"] and r["proof_sha256"] == first[rc.key(r)]["proof_sha256"],
             f"{rc.key(r)}: two campaign processes regenerated different proofs")
    if errors:
        print("REFUSED:\n  " + "\n  ".join(errors[:20]) + (f"\n  ... {len(errors) - 20} more" if len(errors) > 20 else ""))
        return 1
    if a.check:
        print(f"complete and clean: {len(rows)} cells; would ship {sum(r['family'] in SHIP for r in rows)}")
        return 0

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    (OUT / "proofs").mkdir()
    (OUT / ".gitattributes").write_text("*.bin binary\n*.txt binary\n", encoding="utf-8", newline="\n")
    cells = []
    for r in sorted(rows, key=lambda r: [str(x) for x in rc.key(r)]):
        src = work_of[rc.key(r)] / r["family"] / r["kind"]
        pf = src / r["proof_file"]
        # the format header is read from the file only once the file is shown to be the recorded proof
        if not pf.is_file() or sha(pf) != r["proof_sha256"]:
            raise SystemExit(f"{pf}: missing, or not the proof the campaign recorded")
        proof_bytes = pf.read_bytes()[:8]
        cell = {k: r[k] for k in ("family", "kind", "circuit", "basis", "rounds", "w", "maxround", "model_sha256",
                                  "proof_sha256", "proof_bytes", "verified", "model_equal", "proof_equal",
                                  "prover_s", "checker_s", "checker")}
        bits = re.search(r"\((\d+) trace bits, (\d+) tasks", r["prover"])
        if not bits:
            raise SystemExit(f"{pf}: the prover's line does not report its trace bits")
        cell["trace_bits"], cell["tasks"] = int(bits.group(1)), int(bits.group(2))
        cell["recorded_proof_sha256"] = r["proof_sha256_record"]
        cell["proof_magic"] = proof_bytes[:4].decode("ascii", "replace")
        cell["shipped"] = r["family"] in SHIP
        cell["record_reproduced_by"] = reproduced_by.get(rc.key(r))
        sub = f"{r['family']}/{r['kind']}"
        # every model ships (they are small); proofs ship for the families that fit in git
        ship = [("models", r["model_file"], r["model_sha256"])]
        if cell["shipped"]:
            ship.append(("proofs", r["proof_file"], r["proof_sha256"]))
        for kind, name, h in ship:
            dst = OUT / kind / sub / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, dst)
            if hashlib.sha256(dst.read_bytes()).hexdigest() != h:
                raise SystemExit(f"{dst}: copied bytes are not the recorded ones")
        cell["model"] = f"models/{sub}/{r['model_file']}"
        cell["proof"] = f"proofs/{sub}/{r['proof_file']}" if cell["shipped"] else None
        if cell["proof_magic"] != "CDP5":
            raise SystemExit(f"{r['proof_file']}: not a CDP5 proof ({proof_bytes!r})")
        cells.append(cell)
    fam = collections.defaultdict(lambda: collections.Counter())
    for c in cells:
        f = fam[c["family"]]
        f["cells"] += 1
        f["proof_equal"] += bool(c["proof_equal"])
        f["proof_bytes"] += c["proof_bytes"]
        f["prover_s"] += c["prover_s"] or 0
        f["checker_s"] += c["checker_s"] or 0
    manifest = {
        "schema": "oneq.r198.recert.v1", "date": "2026-11-02",
        "why": "paper 3 review (docs/reviews/PAPER3_REFEREE_2026-11-02.md): every certificate the results rest on, "
               "other than IBM's (ibm_recert_20261026), re-proved and re-checked by the COMMITTED prover and the "
               "Lean-modelled checker",
        "prover": {"source": "experiments/circuit_distance_exact/cdist_prove.c", "sha256_git": prover_sha},
        "checker": {"source": "experiments/circuit_distance_exact/ccheck.c", "sha256_git": checker_sha},
        "hash_note": "sha256_git is over the bytes git stores (LF line endings)",
        "compiler": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "command": "python experiments/circuit_distance_exact/recert_campaign.py --work DIR --out DIR/records.jsonl "
                   "--threads 8",
        "cdp5": {"commit": CDP5_COMMIT, "date": cdp5_date,
                 "rule": "a cell whose regenerated proof differs from its record is accepted only when an earlier "
                         "prover, compiled from the named commit, reproduces the recorded proof bit for bit "
                         "(records/recert_old_prover.jsonl); see cells[].record_reproduced_by"},
        "shipped_families": list(SHIP),
        "regenerate": "the prover is deterministic; recert_campaign.py --families 3c 3c_t2 morph144 regenerates those "
                      "proofs, whose sha256 is listed per cell, bit for bit",
        "totals": {k: dict(v) for k, v in sorted(fam.items())},
        "campaign_threads": 8,
        "determinism": {"record": "records/recert_determinism.jsonl",
                        "cells_with_trace_bits": sum(d["can_test_determinism"] for d in det),
                        "identical": sum(d["identical"] and d["can_test_determinism"] for d in det),
                        "threads": sorted({d["threads"] for d in det}),
                        "max_proof_bytes": max((d["proof_bytes"] for d in det if d["can_test_determinism"]), default=0),
                        "repeats_across_processes": len(repeats)},
        "aborted_attempts": {"rule": "a run whose prover wrote no proof and printed nothing was stopped, not "
                                     "refuted; it decides nothing, and its cell was completed by another run",
                             "attempts": aborted},
        "cells": cells,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    shown = OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT
    print(f"wrote {shown}: {len(cells)} cells, {sum(c['shipped'] for c in cells)} shipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
