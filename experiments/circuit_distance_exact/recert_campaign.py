r"""Re-certify every recorded lower-bound certificate of paper 3 with the COMMITTED prover and checker.

    python recert_campaign.py --work C:/lq/recert --out C:/lq/recert/records.jsonl [--threads 8] [--families ...]

WHY (paper 3 referee, 2026-11-02). Only IBM's 22 certificates were re-certified by the committed, Lean-modelled checker
(evidence/circuit_distance_exact/ibm_recert_20261026). The three-colour, morphing and Table II certificates were checked
by the checker version of their day, several before the fixes the Lean formalization prompted, and their proofs are not
in the repository. This re-runs every recorded cell -- the same circuit, basis, rounds, weight bound and anchoring rule
as its record -- through the committed `cdist_prove` and `ccheck`, and records for each:
  * VERIFIED or not by the committed checker (sources hashed into every row);
  * whether the exported model is byte-identical to the one the record names (model_sha256);
  * whether the regenerated proof is byte-identical to the recorded one (the prover is deterministic: the first probe
    reproduced three-colour T = 1 memory-X bit for bit), so a recorded proof hash binds a proof anyone can regenerate.

Cells come from the records themselves, never from a hand-written list: records/morph72_shift_certificates_w5.jsonl,
records/morph2026_campaign.jsonl (kinds cert and shift_cert), records/morph144_certificates_w11.jsonl,
records/3c_shift_certificates_w10.jsonl, records/3c_T2_certificates_w10.jsonl. Resumable: a cell already in --out with a finished verdict is skipped.
Models and proofs stay under --work; nothing here writes into the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402
import morph2026 as m26  # noqa: E402
import tools_cde as tc  # noqa: E402

REC = HERE.parents[1] / "evidence" / "circuit_distance_exact" / "records"
FAMILIES = ("morph72", "morph2026", "morph144", "3c", "3c_t2")


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lf_sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rows(name: str) -> list[dict]:
    return [json.loads(x) for x in (REC / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def cells(families) -> list[dict]:
    out = []
    for fam in families:
        if fam == "morph72":
            src = [dict(r, kind="shift_cert", circuit="morph72") for r in rows("morph72_shift_certificates_w5.jsonl")]
        elif fam == "morph144":
            src = [dict(r, kind="cert", circuit="morph144") for r in rows("morph144_certificates_w11.jsonl")]
        elif fam == "3c":
            src = [dict(r, kind="shift_cert", circuit="3c") for r in rows("3c_shift_certificates_w10.jsonl")]
        elif fam == "3c_t2":
            # the three-colour search's own certificates at T = 2 (no time shift), 2026-10-20; added 2026-11-03 so
            # that "every certificate" includes them
            src = [dict(r, kind="cert", circuit="3c") for r in rows("3c_T2_certificates_w10.jsonl")]
        elif fam == "morph2026":
            src = [dict(r, circuit=r["code"]) for r in rows("morph2026_campaign.jsonl")
                   if r.get("kind") in ("cert", "shift_cert")]
        else:
            raise SystemExit(f"unknown family {fam}")
        for r in src:
            out.append(dict(family=fam, kind=r["kind"], circuit=r["circuit"], basis=r["basis"], rounds=r["rounds"],
                            w=r["w"], maxround=r.get("maxround", -1), model_sha256_record=r["model_sha256"],
                            proof_sha256_record=r.get("proof_sha256")))
    return out


def key(c: dict) -> tuple:
    return (c["family"], c["kind"], c["circuit"], c["basis"], c["rounds"], c["w"], c["maxround"])


def claim(claims: pathlib.Path, c: dict) -> bool:
    claims.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(claims / ("_".join(str(x) for x in key(c)) + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def run_cell(c: dict, exe: dict, work: pathlib.Path, threads: int) -> dict:
    d = work / c["family"] / c["kind"]
    d.mkdir(parents=True, exist_ok=True)
    shift = [c["maxround"]] if c["maxround"] >= 0 else []
    if c["family"] == "morph2026":
        r = m26.certify(m26.spec(c["circuit"]), c["basis"], c["rounds"], c["w"], exe, d, threads, c["maxround"])
        model = d / f"{c['circuit']}_{c['basis']}{c['rounds']}{'_t' if c['maxround'] >= 0 else ''}.txt"
        proof = d / f"proof_{c['circuit']}_{c['basis']}{c['rounds']}_w{c['w']}.bin"
        prover, checker = r.get("prover", ""), r.get("checker", "")
        prover_s, checker_s = r.get("prover_s"), r.get("checker_s")
        ok_exit = r.get("prover_exit") == 0 and r.get("checker_exit") == 0
    else:
        model = d / f"{c['circuit']}_{c['basis']}{c['rounds']}.txt"
        em.export(c["circuit"], c["basis"], c["rounds"], model, time_order=c["maxround"] >= 0)
        proof = d / f"proof_{c['circuit']}_{c['basis']}{c['rounds']}_w{c['w']}.bin"
        t0 = time.time()
        p = tc.run(exe["cdist_prove"], model, c["w"], proof, *shift, threads=threads)
        prover, prover_s = p.stdout.strip(), round(time.time() - t0, 1)
        checker, checker_s, ok_exit = "", None, False
        if p.returncode == 0:
            t1 = time.time()
            k = tc.run(exe["ccheck"], model, proof, bc.group_kind(c["circuit"]), threads=threads)
            checker, checker_s, ok_exit = k.stdout.strip(), round(time.time() - t1, 1), k.returncode == 0
    m_sha = sha(model) if model.is_file() else None
    p_sha = sha(proof) if proof.is_file() else None
    return dict(c, model_sha256=m_sha, model_equal=m_sha == c["model_sha256_record"],
                proof_sha256=p_sha, proof_bytes=proof.stat().st_size if proof.is_file() else None,
                proof_equal=(p_sha == c["proof_sha256_record"]) if c["proof_sha256_record"] else None,
                prover=prover.replace(str(d), "<work>"), checker=checker, prover_s=prover_s, checker_s=checker_s,
                verified=bool(ok_exit and checker.startswith("VERIFIED")),
                prover_source_sha256=lf_sha(HERE / "cdist_prove.c"), checker_source_sha256=lf_sha(HERE / "ccheck.c"),
                model_file=model.name, proof_file=proof.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    # a second process on disjoint cells, into its own --out, halves the wall clock; recert_package.py merges them
    ap.add_argument("--basis", choices=("X", "Z"), default=None)
    ap.add_argument("--reverse", action="store_true")
    # processes sharing a --claims directory never run the same cell: a cell goes to the first to create its claim file
    ap.add_argument("--claims", default=None)
    a = ap.parse_args()
    work, out = pathlib.Path(a.work), pathlib.Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(key(r))
    exe = tc.build_c(work)
    todo = [c for c in cells(a.families) if key(c) not in done and a.basis in (None, c["basis"])]
    if a.reverse:
        todo.reverse()
    print(f"{len(todo)} cells to run ({len(done)} already recorded)", flush=True)
    bad = 0
    for i, c in enumerate(todo, 1):
        if a.claims and not claim(pathlib.Path(a.claims), c):
            print(f"[{i}/{len(todo)}] {c['family']} {c['basis']}{c['rounds']}: claimed by another process", flush=True)
            continue
        r = run_cell(c, exe, work, a.threads)
        bad += not r["verified"]
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[{i}/{len(todo)}] {c['family']} {c['kind']} {c['circuit']} {c['basis']}{c['rounds']} w{c['w']}: "
              f"{'VERIFIED' if r['verified'] else 'NOT VERIFIED'} model_equal={r['model_equal']} "
              f"proof_equal={r['proof_equal']} ({r['prover_s']} s + {r['checker_s']} s)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
