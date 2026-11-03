r"""R198 lane driver: exact circuit distances with checkable certificates.

    python run_lane.py search  --circuit ibm|3c [--l 12] --basis Z --rounds 2 --wmin 1 --wmax 10
    python run_lane.py sweep   --circuit ibm --basis Z --tmax 11 --w 9          # one search per round count
    python run_lane.py certify --circuit 3c  --basis Z --rounds 2 --w 10        # prove + independent check
    python run_lane.py lift    --circuit ibm --basis Z --rounds 2 --ids 702 1513 ...
    python run_lane.py window  --circuit ibm --basis Z --t 10                   # P1-P3 on models T, T+1

Every subcommand prints JSON on stdout and writes nothing under evidence/ (`collect_results.py` does that).
Models are the same-type projections (a LOWER bound on circuit distance); FOUND witnesses are lifted to the
full DEM before they count as upper bounds. Compiled programs go to --work (default: a temp directory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import export_model as em  # noqa: E402
import tools_cde as tc  # noqa: E402


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_search(text: str) -> dict:
    none = [int(m.group(1)) for m in re.finditer(r"^w<=(\d+): NONE", text, re.M)]
    found = re.search(r"^w<=(\d+): FOUND weight (\d+) witness:([\d ]+)", text, re.M)
    return dict(none_up_to=max(none) if none else 0,
                found_weight=int(found.group(2)) if found else None,
                witness=[int(x) for x in found.group(3).split()] if found else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("search", "sweep", "certify", "lift", "window"))
    ap.add_argument("--circuit", choices=bc.CIRCUITS, required=True)
    ap.add_argument("--l", type=int, default=12)
    ap.add_argument("--basis", choices=("Z", "X"), required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--wmin", type=int, default=1)
    ap.add_argument("--wmax", type=int, default=10)
    ap.add_argument("--w", type=int, default=9)
    ap.add_argument("--tmax", type=int, default=11)
    ap.add_argument("--tmin", type=int, default=1)
    ap.add_argument("--maxround", type=int, default=-1,
                    help="time-shift anchoring (rule S): anchors in rounds <= MAXROUND of a time-ordered model")
    ap.add_argument("--t", type=int, default=10)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--work")
    a = ap.parse_args()
    work = pathlib.Path(a.work) if a.work else pathlib.Path(tempfile.mkdtemp(prefix="r198_"))
    work.mkdir(parents=True, exist_ok=True)
    group = bc.group_kind(a.circuit)

    if a.cmd == "lift":
        full = tc.lift(a.circuit, a.basis, a.rounds, a.ids, a.l)
        print(json.dumps(dict(projected=a.ids, full=full, weight=len(full) if full else None,
                              verified=bool(full) and tc.verify_full(a.circuit, a.basis, a.rounds, full, a.l),
                              minus_one_fails=bool(full) and not tc.verify_full(a.circuit, a.basis, a.rounds, full[1:], a.l))))
        return
    if a.cmd == "window":
        m0, m1 = work / f"w_{a.t}.txt", work / f"w_{a.t + 1}.txt"
        em.export(a.circuit, a.basis, a.t, m0, l=a.l)
        em.export(a.circuit, a.basis, a.t + 1, m1, l=a.l)
        print(json.dumps(dict(circuit=a.circuit, basis=a.basis,
                              **tc.window_preconditions(m0, m1, bc.time_drift(a.circuit)))))
        return
    exe = tc.build_c(work)
    if not exe:
        sys.exit("no gcc on PATH")
    rounds_list = range(a.tmin, a.tmax + 1) if a.cmd == "sweep" else [a.rounds]
    shift = [a.maxround] if a.maxround >= 0 else []
    for T in rounds_list:
        model = work / f"{a.circuit}_l{a.l}_{a.basis}{T}.txt"
        info = em.export(a.circuit, a.basis, T, model, l=a.l, time_order=a.maxround >= 0)
        rec = dict(cmd=a.cmd, **info, maxround=a.maxround, model_sha256=sha(model))
        t0 = time.time()
        if a.cmd in ("search", "sweep"):
            lo, hi = (a.w, a.w) if a.cmd == "sweep" else (a.wmin, a.wmax)
            r = tc.run(exe["cdist"], model, lo, hi, 20, *shift, threads=a.threads)
            # a weight-w pass explores every set of <= w mechanisms, so NONE at w covers every smaller weight too
            rec.update(parse_search(r.stdout), weights=[lo, hi], seconds=round(time.time() - t0, 1))
        else:
            proof = work / f"proof_{a.circuit}_{a.basis}{T}_w{a.w}.bin"
            p = tc.run(exe["cdist_prove"], model, a.w, proof, *shift, threads=a.threads)
            rec.update(w=a.w, prover=p.stdout.strip(), prover_exit=p.returncode, prover_s=round(time.time() - t0, 1))
            if p.returncode == 0:
                t1 = time.time()
                c = tc.run(exe["ccheck"], model, proof, group, threads=a.threads)
                rec.update(checker=c.stdout.strip(), checker_exit=c.returncode, checker_s=round(time.time() - t1, 1),
                           proof_sha256=sha(proof), proof_bytes=proof.stat().st_size)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
