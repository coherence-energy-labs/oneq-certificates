"""R203 rehearsal -- the Helios run's FIXED analysis command, end to end, on the committed package, before any spend.

`python ion_sequential.py analyse PKG RESULTS_DIR` is what decides the claim. Its pieces are unit-tested on small
copies; this drives the whole command on the real 8,000-unit R203 package, with synthetic provider output in the
exact shapes the device run produces:
  - batch_<k>.json      per-shot records {shot, uid, c0..c(k-1)}, every unit's bits chosen so Bob's own score rule
                        reads its ideal outcome, or an error with the scenario's per-group probability;
  - batch_<k>.job.json  {provider_job_id, hugr_sha256}, the digest the batch's receipt verified;
  - jobs.json           the account's full job list.
Scenarios (seeded; the error rates are R195's executed Helios projection per (group, input basis) x lambda -- since
R225 the package is read by rule 'betting_ad', whose witness scores errors by basis, so the rehearsal's device errs by
basis as the projection does; the hopeless scenario moved from 3x to 4x, because the mixture survives 3x ~21% of the
time):
  at_spec        lambda 1, every batch available -> the claim must HOLD, at an early look;
  four_x         lambda 4, every batch           -> it must FAIL (nothing at m = 8 survives 4x), stopped for futility;
  stopped_early  lambda 2.5, only batches 1-4 ran -> NOT DECIDED, never a pass (at 2x the mixture already clears by
                 look 2,000, inside the batches that ran);
  rerun          at_spec's results, but jobs.json shows batch 1 submitted twice -> REFUSED (non-zero exit).
Each scenario runs the CLI as a subprocess and records its exit code, verdict, stopping look and the HQC spent.

  python r203_rehearsal.py      -> evidence/superactivation/r203_betting/rehearsal.json
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import ion_run_package as P  # noqa: E402
import r225_witness_choice as C  # noqa: E402

PKG = ROOT / "evidence" / "superactivation" / "ion_run_m8_r224"   # R224 re-freeze; R203's package was folded (R213)
OUT = ROOT / "evidence" / "superactivation" / "r203_betting" / "rehearsal.json"
SCENARIOS = {"at_spec": (1.0, None), "four_x": (4.0, None), "stopped_early": (2.5, 4), "rerun": (1.0, None)}


def ideal_and_error_bits(u: dict, design: dict) -> tuple:
    """(bits Bob scores as the unit's ideal, bits he scores as an error), each a list over clbits"""
    import hw_faithful as HF
    sc = HF.score(design["m"], P._unit(u), design["decoder"])
    n = u["n_clbits"]
    ideal = None
    for x in range(1 << min(n, 12)):                   # the ideal is found within the low bits for these units
        b = [(x >> j) & 1 for j in range(n)] if n <= 12 else [(x >> j) & 1 if j < 12 else 0 for j in range(n)]
        if abs(sc(b) - u["ideal_score"]) < 1e-9:
            ideal = b
            break
    if ideal is None and u["group"].startswith("kept"):
        for o in (0, 1):
            b = [0] * n
            b[0], b[n - 1] = u["beta"], o
            if abs(sc(b) - u["ideal_score"]) < 1e-9:
                ideal = b
                break
    if ideal is None:
        raise ValueError(f"{u['name']}: no bitstring reaches the ideal score")
    for j in range(n):
        e = list(ideal)
        e[j] ^= 1
        if abs(sc(e) - u["ideal_score"]) > 1e-9:
            return ideal, e
    raise ValueError(f"{u['name']}: no single flip changes the score")


def write_results(res: pathlib.Path, design: dict, seq: dict, bits: list, lam: float, batches_run, rng,
                  rates: dict) -> dict:
    res.mkdir(parents=True, exist_ok=True)
    jobs = []
    errs = 0
    for k, (a, b) in enumerate(seq["batches"], 1):
        if batches_run is not None and k > batches_run:
            break
        recs = []
        for i, u in enumerate(design["units"][a:b]):
            ideal, err = bits[a + i]
            bad = rng.random() < min(1.0, lam * rates[f"{u['group']}|{u['inp'][0]}"])
            errs += bad
            v = err if bad else ideal
            recs.append(dict({"shot": i, "uid": i}, **{f"c{j}": int(v[j]) for j in range(len(v))}))
        (res / f"batch_{k}.json").write_text(json.dumps(recs), encoding="utf-8")
        rc = json.loads((PKG / f"batch_{k}" / "param_shots_receipt.json").read_text(encoding="utf-8"))
        job = dict(provider_job_id=f"rehearsal-{k}", hugr_sha256=rc["hugr_sha256"])
        (res / f"batch_{k}.job.json").write_text(json.dumps(job), encoding="utf-8")
        jobs.append(job)
    (res / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    return dict(errors_injected=errs, jobs=jobs)


def analyse(res: pathlib.Path) -> dict:
    p = subprocess.run([sys.executable, str(HERE / "ion_sequential.py"), "analyse", str(PKG), str(res)],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1800, cwd=str(HERE))
    out = dict(exit=p.returncode)
    if p.returncode == 0:
        v = json.loads(p.stdout)
        out.update(verdict=v["verdict"], stopped_at=v["stopped_at"], looks_reached=len(v["looks"]),
                   log_wealth=[round(r["log_wealth"], 3) for r in v["looks"] if "log_wealth" in r],
                   log_bar=v["looks"][0].get("log_bar"), claim_scope=v["claim_scope"])
    else:
        out["refusal"] = (p.stderr.strip().splitlines() or [""])[-1][:300]
    return out


def run() -> dict:
    design = json.loads((PKG / "design.json").read_text(encoding="utf-8"))
    seq = json.loads((PKG / "sequential.json").read_text(encoding="utf-8"))
    costs = [json.loads((PKG / f"batch_{k}" / "param_shots_receipt.json").read_text(encoding="utf-8"))["hqc"]["total"]
             for k in range(1, len(seq["batches"]) + 1)]
    rates = {k: v for k, v in C.device_rates().items() if "|" in k}
    bits = [ideal_and_error_bits(u, design) for u in design["units"]]
    package = PKG.relative_to(ROOT).as_posix() if PKG.is_relative_to(ROOT) else str(PKG)
    out = dict(schema="oneq.r203_rehearsal/2", package=package, rule=seq["rule"], rates=rates,
               rates_source="R195 executed projection, Helios, per (group, input basis) x lambda (SIMULATED)",
               scenarios={})
    with tempfile.TemporaryDirectory() as td:
        for i, (name, (lam, batches_run)) in enumerate(SCENARIOS.items()):
            res = pathlib.Path(td) / name
            rng = np.random.default_rng([203, i])
            meta = write_results(res, design, seq, bits, lam, batches_run, rng, rates)
            if name == "rerun":
                jobs = meta["jobs"] + [dict(meta["jobs"][0], provider_job_id="rehearsal-1-again")]
                (res / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
            r = dict(lam=lam, batches_run=batches_run or len(seq["batches"]), errors_injected=meta["errors_injected"],
                     **analyse(res))
            if r.get("stopped_at"):
                # the batches that RAN up to the stopping look (a NOT REACHED look's batch never ran)
                r["hqc_spent"] = round(sum(c for k, (c, (a, b)) in enumerate(zip(costs, seq["batches"]), 1)
                                           if b <= r["stopped_at"] and k <= r["batches_run"]), 1)
            out["scenarios"][name] = r
    return out


def main() -> int:
    out = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k, v in out["scenarios"].items():
        print(f"{k:14s} exit {v['exit']}  {v.get('verdict', v.get('refusal', ''))[:70]}  stopped {v.get('stopped_at')}"
              f"  HQC {v.get('hqc_spent')}")
    print("->", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
