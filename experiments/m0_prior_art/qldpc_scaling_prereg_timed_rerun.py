r"""THE PREREGISTERED SCALING CAMPAIGN, RE-RUN WITH PER-SHOT TIMING -- its production cost, stage by stage, and a
byte-for-byte reproduction check of its records.

    python -B experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py --dataset stress_p05
    python -B experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py --dataset all

Why (external audit 3 of release v2026.11.05, paper A = our paper 2, "costs need to follow the new headline
experiment"): the run of record wrote one wall time per dataset and no per-shot timing, so the campaign had no
production-latency distribution. A total divided by a shot count is not one. This script re-runs the campaign's own
`run_dataset()` on the PREREGISTERED seeds, with timers around every stage of every shot, and writes the timings.

THE BYTES. Each dataset is re-run under the exact bytes it ran under: the runner, the production checker
(src/oneq/qldpc_check.py), the producer module (src/oneq/qldpc_cert.py) and both second checkers, each read from the
working tree or from git history by the sha256 that evidence/qldpc_scaling_prereg/summary.json records for that
dataset (the four datasets run before DEVIATION 1 ran under the committed runner, the four after it under the
changed one). Nothing is re-implemented here: the loop, the receipt rule, the scoring are the runner's own.

THE REPRODUCTION CHECK. The producer is deterministic except where a wall-clock limit binds (the MILP's 30 s time
limit). The re-run's records are serialised exactly as the runner serialises them (one sorted-key JSON line per
record, gzip with mtime 0) and compared with the run of record: the sha256 of the whole file, and record by record.
A mismatch is RECORDED, never repaired, and the artifact says which shots differ and in which fields.

THE STAGES (wall clock, time.perf_counter; CPU, time.process_time of this process, all its threads):
  sample         the dataset's sampler (NumPy or Stim), one call per sampled shot, trivial ones included;
  decode_1.0, decode_0.625   BpOsdDecoder.decode at each setting;
  lp             feldman_dual / feldman_dual_lazy, the floating-point LP of the shot;
  exactify       rationalising its dual;
  prod_check     the production exact checker's acceptance calls made while the receipt is being fixed
                 (check_qldpc_exact on each tried candidate; check_qldpc_bnb on a tree);
  milp, bnb      the MILP candidate and the branch-dual tree, when no single dual closes;
  second_checker the reference flat checker (in process) or tree checker B (a subprocess, wall clock only);
  scoring        the production checker's tier-1 calls on the two answers, after the receipt is fixed;
  other          the rest of the shot: the runner's own Python between those calls.
A call nested inside another timed call (the LP inside the branch-dual tree) is part of the outer stage only.
A shot's total runs from the end of its sampling to the start of the next sample (or of the analysis).

Writes evidence/qldpc_scaling_prereg/costs/production_<dataset>.json and nothing else under evidence/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import io
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RUNNER_REL = "experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py"
CHECK_REL = "src/oneq/qldpc_check.py"
CERT_REL = "src/oneq/qldpc_cert.py"
REF_REL = "tools/external/exact_lp_certificate_reference.py"
TREE_B_REL = "tools/external/tree_checker_b.py"
PREREG_DIR = ROOT / "evidence" / "qldpc_scaling_prereg"
SUMMARY = PREREG_DIR / "summary.json"
OUT_DIR = PREREG_DIR / "costs"
SCHEMA = "oneq-qldpc-scaling-prereg-production-cost/1"
#: limits the runner of record imposes, each verified against the bytes that ran (see _limits)
LIMIT_MARKERS = {"bnb_max_nodes": "max_nodes=60000", "tree_checker_b_timeout_s": "timeout=600"}


#: the stages that fix a receipt: the producer's LP and its rationalisation, the production checker's acceptance of a
#: candidate, and the MILP and branch-dual tree when no single dual closes
PRODUCTION_STAGES = ("lp", "exactify", "prod_check", "milp", "bnb")
STAGES = ("decode_1.0", "decode_0.625") + PRODUCTION_STAGES + ("second_checker", "scoring", "other")
QUANTILES = (("median", 0.5), ("p90", 0.9), ("p99", 0.99))


def nearest_rank(values: list[float]) -> dict:
    """Nearest-rank quantiles: the ceil(q N)-th smallest value; the median of an even count is the lower middle."""
    v = sorted(values)
    out = {k: v[max(1, -(-int(q * 1000) * len(v) // 1000)) - 1] for k, q in QUANTILES}
    out["max"], out["n"] = v[-1], len(v)
    return out


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cpu_name() -> str | None:
    """The processor's marketing name (Windows registry; /proc/cpuinfo elsewhere), or None if neither answers."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split())
    except Exception:
        pass
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return " ".join(line.split(":", 1)[1].split())
    except Exception:
        pass
    return None


def bytes_by_sha(rel: str, want: str) -> tuple[bytes, str]:
    """The bytes of `rel` whose sha256 is `want`: the working tree if it matches, else the newest git revision whose
    blob does. Read-only git; refuses if no revision holds those bytes."""
    p = ROOT / rel
    if p.is_file() and sha(p.read_bytes()) == want:
        return p.read_bytes(), "working tree"
    revs = subprocess.run(["git", "log", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.split()
    for c in revs:
        r = subprocess.run(["git", "show", f"{c}:{rel}"], cwd=ROOT, capture_output=True)
        if r.returncode == 0 and sha(r.stdout) == want:
            return r.stdout, c
    raise SystemExit(f"no revision of {rel} has sha256 {want}: the bytes of record cannot be recovered")


def load_module(name: str, src: bytes, filename: pathlib.Path, package: str | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(filename)
    if package is not None:
        mod.__package__ = package
    sys.modules[name] = mod                      # dataclasses resolve their module by name
    exec(compile(src, str(filename), "exec"), mod.__dict__)
    return mod


class Clock:
    """Outermost-call timers. Every event is (stage, t0, t1, cpu_seconds, note)."""

    def __init__(self):
        self.events: list[tuple] = []
        self.depth = 0

    def wrap(self, stage, fn, observe=None):
        def timed(*a, **k):
            if self.depth:
                return fn(*a, **k)
            self.depth += 1
            c0, t0 = time.process_time(), time.perf_counter()
            out, note = None, None
            try:
                out = fn(*a, **k)
                return out
            finally:
                t1, c1 = time.perf_counter(), time.process_time()
                self.depth -= 1
                if observe is not None and out is not None:
                    note = observe(out)
                self.events.append((stage, t0, t1, c1 - c0, note))
        return timed

    def mark(self, stage):
        t = time.perf_counter()
        self.events.append((stage, t, t, 0.0, None))


def _limits(runner_src: bytes, cert_mod) -> dict:
    text = runner_src.decode("utf-8")
    out = {}
    for k, marker in LIMIT_MARKERS.items():
        if marker not in text:
            raise SystemExit(f"the runner of record no longer contains {marker!r}; restate the limits")
        out[k] = int(marker.split("=")[1])
    out["milp_time_limit_s"] = float(inspect.signature(cert_mod.exact_milp).parameters["time_limit"].default)
    out["bnb_max_depth"] = int(inspect.signature(cert_mod.certified_bnb).parameters["max_depth"].default)
    out["memory_cap"] = None
    out["dataset_wall_cap"] = None
    return out


def _serialise(recs: list) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        for r in recs:
            g.write((json.dumps(r, sort_keys=True) + "\n").encode())
    return buf.getvalue()


def _shots(events: list, recs: list) -> tuple[list, dict]:
    """Split the event stream at sample calls into per-shot stage times; map nontrivial shots to their records."""
    events = sorted(events, key=lambda e: e[1])
    samples = [i for i, e in enumerate(events) if e[0] == "sample"]
    loop_end = next(e[1] for e in events if e[0] == "loop_end")
    shots, ordinal = [], -1
    instrument = 0.0
    for si, idx in enumerate(samples):
        ordinal += 1
        start = events[idx][2]
        end = events[samples[si + 1]][1] if si + 1 < len(samples) else loop_end
        inner = [e for e in events[idx + 1:(samples[si + 1] if si + 1 < len(samples) else len(events))]
                 if e[0] not in ("loop_end",)]
        inst = sum(e[2] - e[1] for e in inner if e[0] == "instrument")
        instrument += inst
        inner = [e for e in inner if e[0] != "instrument"]
        if not inner:
            continue                                   # a trivial syndrome: sampled, counted, not used
        wall: dict[str, float] = {}
        cpu: dict[str, float] = {}
        seen_second = False
        notes = {}
        for st, t0, t1, c, note in inner:
            if st == "second_checker":
                seen_second = True
            if st == "exact_check":
                st = "scoring" if seen_second else "prod_check"
            if st == "bnb_check":
                st = "prod_check"
            wall[st] = wall.get(st, 0.0) + (t1 - t0)
            cpu[st] = cpu.get(st, 0.0) + c
            if note is not None:
                notes.setdefault(st, []).append(note)
        total = end - start - inst
        wall["other"] = total - sum(wall.values())
        shots.append({"sample_ordinal": ordinal, "wall": wall, "cpu": cpu, "total_wall": total, "notes": notes,
                      "sample_wall": events[idx][2] - events[idx][1]})
    if len(shots) != len(recs):
        raise SystemExit(f"{len(shots)} timed nontrivial shots against {len(recs)} records")
    for k, (s, r) in enumerate(zip(shots, recs)):
        if s["sample_ordinal"] != r["sample_index"] or r["shot_index"] != k:
            raise SystemExit(f"timed shot {k} is sample {s['sample_ordinal']}, its record says {r['sample_index']}")
    return shots, {"instrument_seconds_excluded": instrument, "samples": len(samples)}


def run_one(name: str, scratch: pathlib.Path) -> dict:
    import numpy as np
    import psutil
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    entry = summary["datasets"][name]
    code = entry.get("code_sha256", summary["code_sha256"])
    got = {rel: bytes_by_sha(rel, h) for rel, h in code.items()}
    # the production checker, then the producer that imports it, under the bytes of record
    import oneq
    chk = load_module("oneq.qldpc_check", got[CHECK_REL][0], ROOT / CHECK_REL, package="oneq")
    oneq.qldpc_check = chk
    cert = load_module("oneq.qldpc_cert", got[CERT_REL][0], ROOT / CERT_REL, package="oneq")
    oneq.qldpc_cert = cert
    runner = load_module("prereg_runner_of_record", got[RUNNER_REL][0], ROOT / RUNNER_REL)
    ref = load_module("ref_checker_prereg", got[REF_REL][0], ROOT / REF_REL)
    tree_b = scratch / "tree_checker_b.py"
    tree_b.write_bytes(got[TREE_B_REL][0])
    if sha(tree_b.read_bytes()) != code[TREE_B_REL]:
        raise SystemExit("tree checker B was not materialised byte for byte")
    limits = _limits(got[RUNNER_REL][0], cert)

    clock = Clock()
    proc = psutil.Process()
    threads_seen: list[int] = []
    busy0 = psutil.cpu_times()
    counter = {"n": 0}

    for fname, stage in (("feldman_dual", "lp"), ("feldman_dual_lazy", "lp"), ("exactify_certificate", "exactify"),
                         ("exact_milp", "milp"), ("certified_bnb", "bnb")):
        obs = None
        if fname == "exact_milp":
            def obs(out):
                st = out[2] if isinstance(out, tuple) and len(out) == 3 and isinstance(out[2], dict) else {}
                return {"found": out[0] is not None, "status_code": st.get("status_code"),
                        "message": st.get("message"), "tier": st.get("tier")}
        if fname == "certified_bnb":
            def obs(out):
                return {"tree": out[0] is not None, "nodes": int(out[1]) if out[1] is not None else None}
        setattr(cert, fname, clock.wrap(stage, getattr(cert, fname), obs))
    chk.check_qldpc_exact = clock.wrap("exact_check", chk.check_qldpc_exact)
    chk.check_qldpc_bnb = clock.wrap("bnb_check", chk.check_qldpc_bnb)

    import ldpc
    real_dec = ldpc.BpOsdDecoder

    class TimedDecoder:
        def __init__(self, *a, **k):
            self._d = real_dec(*a, **k)
            self.decode = clock.wrap(f"decode_{k['ms_scaling_factor']}", self._d.decode)

    ldpc.BpOsdDecoder = TimedDecoder
    ref_proxy = types.SimpleNamespace(check_certificate=clock.wrap("second_checker", ref.check_certificate),
                                      CertificateError=ref.CertificateError)
    runner.subprocess = types.SimpleNamespace(run=clock.wrap("second_checker", subprocess.run))
    real_build, real_analyse = runner.build, runner.analyse

    def build(n, d):
        D = real_build(n, d)
        timed_sample = clock.wrap("sample", D["sample"])

        def sample():
            counter["n"] += 1
            if counter["n"] % 25 == 1:                 # thread census, timed and excluded from the shot it lands in
                t0 = time.perf_counter()
                threads_seen.append(proc.num_threads())
                clock.events.append(("instrument", t0, time.perf_counter(), 0.0, None))
            return timed_sample()
        D["sample"] = sample
        return D

    def analyse(*a, **k):
        clock.mark("loop_end")
        return real_analyse(*a, **k)

    runner.build, runner.analyse = build, analyse
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_start = time.perf_counter()
    cpu_start = time.process_time()
    s, recs = runner.run_dataset(name, runner.SEED_DOMAIN, runner.DATASETS[name][5], ref_proxy, tree_b)
    wall_total, cpu_total = time.perf_counter() - t_start, time.process_time() - cpu_start
    busy1 = psutil.cpu_times()
    threads_seen.append(proc.num_threads())

    # ---- reproduction: the whole file, then record by record, then the analysis
    blob = _serialise(recs)
    stored = (PREREG_DIR / f"{name}.jsonl.gz").read_bytes()
    stored_recs = [json.loads(x) for x in gzip.decompress(stored).decode("utf-8").splitlines()]
    differing = []
    for k in range(max(len(recs), len(stored_recs))):
        a = recs[k] if k < len(recs) else None
        b = stored_recs[k] if k < len(stored_recs) else None
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            keys = sorted(x for x in set(a or {}) | set(b or {})
                          if json.dumps((a or {}).get(x), sort_keys=True) != json.dumps((b or {}).get(x), sort_keys=True))
            differing.append({"shot_index": k, "fields": keys})
    keep = {"seconds", "records_sha256", "code_sha256", "deviation"}
    ana_re = {k: v for k, v in s.items() if k not in keep}
    ana_st = {k: v for k, v in entry.items() if k not in keep}
    shots, meta = _shots(clock.events, recs)
    # COLUMNAR, so that a per-shot series is a data series and never a claim (tools/claims_provenance_gate.py skips
    # long numeric lists); the scalars the paper may quote are the summary's nearest-rank quantiles
    unknown = {st for sh in shots for st in sh["wall"]} - set(STAGES)
    if unknown:
        raise SystemExit(f"untimed stage names {unknown}")
    cols = {"shot_index": [r["shot_index"] for r in recs], "sample_index": [r["sample_index"] for r in recs],
            "receipt_kind": [None if r["receipt"] is None else r["receipt"]["kind"] for r in recs],
            "total_wall_s": [sh["total_wall"] for sh in shots],
            "production_wall_s": [sum(sh["wall"].get(st, 0.0) for st in PRODUCTION_STAGES) for sh in shots],
            "sample_wall_s": [sh["sample_wall"] for sh in shots],
            "wall_s": {st: [sh["wall"].get(st, 0.0) for sh in shots] for st in STAGES},
            "cpu_s": {st: [sh["cpu"].get(st, 0.0) for sh in shots] for st in STAGES if st != "other"}}
    trees = [{"shot_index": r["shot_index"], "nodes": r["receipt"]["nodes"],
              "production_wall_s": cols["production_wall_s"][k]}
             for k, r in enumerate(recs) if r["receipt"] is not None and r["receipt"]["kind"] == "tree"]
    solver_calls = [{"shot_index": r["shot_index"], "stage": st, **note}
                    for sh, r in zip(shots, recs) for st, notes in sh["notes"].items() for note in notes]
    summary = {"production": nearest_rank(cols["production_wall_s"]), "total": nearest_rank(cols["total_wall_s"]),
               "decode_1.0": nearest_rank(cols["wall_s"]["decode_1.0"]),
               "decode_0.625": nearest_rank(cols["wall_s"]["decode_0.625"]),
               "second_checker": nearest_rank(cols["wall_s"]["second_checker"]),
               "scoring": nearest_rank(cols["wall_s"]["scoring"]),
               "milp_calls": sum(1 for c in solver_calls if c["stage"] == "milp"),
               "milp_not_optimal": sum(1 for c in solver_calls if c["stage"] == "milp"
                                       and c.get("tier") != "3A_solver_reported_optimum"),
               "bnb_calls": sum(1 for c in solver_calls if c["stage"] == "bnb"),
               "bnb_without_tree": sum(1 for c in solver_calls if c["stage"] == "bnb" and not c.get("tree"))}
    tot = busy1.user + busy1.system + busy1.idle - (busy0.user + busy0.system + busy0.idle)
    box = (busy1.user + busy1.system - busy0.user - busy0.system) / tot if tot > 0 else None
    import scipy
    import stim
    out = {
        "schema": SCHEMA, "dataset": name,
        "what": "the preregistered campaign's own run_dataset() re-run on its preregistered seeds under the bytes "
                "of record, with per-stage timers; production cost and a byte-for-byte reproduction check",
        "code_of_record": {rel: {"sha256": code[rel], "from": got[rel][1]} for rel in code},
        "timing_script_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "reproduction": {
            "records_sha256_of_record": entry["records_sha256"],
            "records_sha256_rerun": sha(blob),
            "records_file_sha256": sha(stored),
            "records_byte_identical": blob == stored and sha(blob) == entry["records_sha256"],
            "records": len(recs), "differing_records": differing,
            "analysis_identical": ana_re == ana_st,
            "unresolved": sum(r["receipt"] is None for r in recs),
        },
        "limits": limits,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "ldpc": ldpc.__version__, "stim": stim.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_name": cpu_name(), "logical_cpus": os.cpu_count(),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "process_threads_min": min(threads_seen), "process_threads_max": max(threads_seen),
            "peak_working_set_bytes": int(getattr(proc.memory_info(), "peak_wset", 0) or 0),
            "box_cpu_utilisation_during_run": box,
            "concurrent_jobs_note": "the workstation was shared with other work; box_cpu_utilisation_during_run is "
                                    "the whole machine's busy fraction over this run",
        },
        "run": {"wall_seconds": wall_total, "cpu_seconds_this_process": cpu_total,
                "cpu_over_wall": cpu_total / wall_total if wall_total > 0 else None,
                "seconds_of_record": entry["seconds"], "started_utc": started,
                "clock_resolution_s": {c: time.get_clock_info(c).resolution for c in ("perf_counter", "process_time")},
                **meta},
        "quantile_rule": "nearest rank: the ceil(q N)-th smallest of the N nontrivial shots; median = q 0.5",
        "production_stages": list(PRODUCTION_STAGES),
        "summary_s": summary,
        "tree_receipts": trees,
        "solver_calls": solver_calls,
        "shots": cols,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="one dataset name, or all (in the preregistered order)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    names = list(summary["datasets"]) if a.dataset == "all" else [a.dataset]
    out_dir = pathlib.Path(a.out_dir).resolve()
    if out_dir.is_relative_to(ROOT / "evidence") and out_dir != OUT_DIR.resolve():
        raise SystemExit("this script writes under evidence/ only to evidence/qldpc_scaling_prereg/costs/")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in summary["datasets"]:
            raise SystemExit(f"unknown dataset {name}")
        # one dataset per process: each loads its own bytes of record into a clean interpreter
        if len(names) > 1:
            r = subprocess.run([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "--dataset", name,
                                "--out-dir", str(out_dir)])
            if r.returncode:
                return r.returncode
            continue
        with tempfile.TemporaryDirectory() as td:
            res = run_one(name, pathlib.Path(td))
        p = out_dir / f"production_{name}.json"
        p.write_text(json.dumps(res, indent=None, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        rp = res["reproduction"]
        print(f"{name}: {rp['records']} records, byte-identical {rp['records_byte_identical']}, "
              f"{len(rp['differing_records'])} differ, analysis identical {rp['analysis_identical']}, "
              f"{res['run']['wall_seconds']:.1f} s wall (of record {res['run']['seconds_of_record']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
