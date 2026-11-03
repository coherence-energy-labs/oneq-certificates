r"""Is the producer that certified paper 1's held-out sample the one that certified its development shots?

    python experiments/m0_prior_art/producer_identity_willow.py [--workers 3]

WHY (paper 1, second referee, 2026-11-02, finding M2). The held-out paragraph compares the held-out rate with the
development rate of the same placements "with the same base ladder". The development receipts were written by the
2026-09-14 epsilon rerun; producer modules changed after that (2026-10-18, 2026-10-21), and the held-out artifacts
record no commit. A rate comparison across two codes is not a comparison of two samples.

THE TEST IS EXACT. A receipt file's body is the certification of each shot with no timing in it, so a deterministic
producer regenerates it byte for byte. This re-runs EVERY chunk of the eight development configurations of the held-out
placements (d3_at_q4_5 and d5_at_q8_7, X and Z, 13 and 30 rounds) and of the four held-out configurations, through
epsilon_rerun's own unmodified chunk runner with the receipts redirected to a scratch directory, and compares each
chunk's body sha256 with the one its committed index records. It records the commit and the bytes (as git stores them)
of every module the producer and checker import. Nothing under evidence/ is touched but this tool's own output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
OUT = ROOT / "evidence" / "producer_identity_willow.json"
DEV = [f"willow_{p}_{b}_r{r}" for p in ("d3_at_q4_5", "d5_at_q8_7") for b in "XZ" for r in (13, 30)]
HELD = [f"heldout_{p}_{b}_r50" for p in ("d3_at_q4_5", "d5_at_q8_7") for b in "XZ"]
FRESH = ["fresh_sim_d5_r25"]
SAMPLE = {**{n: "development" for n in DEV}, **{n: "held-out" for n in HELD}, **{n: "fresh-seed" for n in FRESH}}


def _worker_init(scratch: str) -> None:
    import epsilon_rerun as er
    import fresh_seed_sim  # noqa: F401  (registers fresh_sim_d5_r25)
    import heldout_willow  # noqa: F401  (registers the heldout_ configurations)
    er.RECEIPT_DIR = pathlib.Path(scratch)


def _loaded() -> list[str]:
    """Every module of THIS repository the worker has loaded: the producer's real import closure, not a list."""
    out = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and pathlib.Path(f).resolve().is_relative_to(ROOT) and f.endswith(".py"):
            out.append(pathlib.Path(f).resolve().relative_to(ROOT).as_posix())
    return sorted(set(out))


def _job(job):
    import epsilon_rerun as er
    r = er.run_chunk(job)
    return job, r, _loaded()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import epsilon_rerun as er

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    # the src/oneq and experiment trees must be clean for the run to be attributed to a commit; the modules actually
    # loaded are pinned afterwards, from the workers themselves
    dirty = git("status", "--porcelain", "--", "src/oneq", "experiments/m0_prior_art")
    if dirty:
        raise SystemExit(f"producer sources have uncommitted changes; refusing to attribute a result to a commit:\n{dirty}")
    commit = git("rev-parse", "HEAD")
    index = {}
    for name in DEV + HELD + FRESH:
        rows = [json.loads(x) for x in (er.INDEX_DIR / name / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        index[name] = {r["seed"]: r["receipts_sha256"] for r in rows}
    jobs = [(name, seed) for name in DEV + HELD + FRESH for seed in sorted(index[name])]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(scratch,)) as ex:
            got = list(ex.map(_job, jobs, chunksize=1))
    res = [(job, r) for job, r, _mods in got]
    modules = sorted({m for _j, _r, mods in got for m in mods})
    if git("status", "--porcelain", "--", *modules) or git("rev-parse", "HEAD") != commit:
        raise SystemExit("a loaded module changed, or HEAD moved, during the run")
    head = {"commit": commit, "modules": "every repository module the workers loaded",
            "sha256": {m: hashlib.sha256((ROOT / m).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for m in modules}}
    out = {"schema": "oneq-producer-identity-willow/2", "head": head, "configs": {}}
    bad = 0
    for name in DEV + HELD + FRESH:
        rs = [(seed, r) for (n, seed), r in res if n == name]
        failed = [seed for seed, r in rs if r.get("failed")]
        differ = [seed for seed, r in rs if not r.get("failed") and r["receipts_sha256"] != index[name][seed]]
        out["configs"][name] = {"sample": SAMPLE[name], "chunks": len(rs),
                                "identical": len(rs) - len(failed) - len(differ), "differ": differ, "failed": failed,
                                "nontrivial": sum(sum(v for k, v in r["status"].items() if k != "TRIVIAL")
                                                  for _s, r in rs if not r.get("failed"))}
        bad += len(differ) + len(failed)
        c = out["configs"][name]
        print(f"{name:28s} {c['identical']}/{c['chunks']} chunks byte-identical; differ {differ} failed {failed}",
              flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("ALL IDENTICAL" if not bad else f"{bad} chunk(s) not reproduced")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
