r"""ARCHIVE THE NORMATIVE INPUTS of the preregistered scaling campaign, so its receipts replay without the producer.

    python -B experiments/m0_prior_art/qldpc_scaling_prereg_export_inputs.py           # write
    python -B experiments/m0_prior_art/qldpc_scaling_prereg_export_inputs.py --check   # compare, write nothing

Why (external audit 3 of release v2026.11.05, paper A = our paper 2): the campaign's runner builds each dataset's
integer weights, decoder channel and logical test at run time -- the heterogeneous weights from a seeded RNG drawn
inside `build()`, the phenomenological weights through `fixed_point_weights(spacetime_system(...))`, the circuit
model through `build_refsched_circuit` -- so the records under evidence/qldpc_scaling_prereg/ could only be replayed
by calling that model-building code. "A verifier should not need to call the producer's model-building code to
recover the precise instance whose optimum it is checking."

This writer is the one place that calls it. For each of the eight datasets it exports what `build()` fixes:
  * the checks -- a reference, by path and sha256, to the persisted evidence/instances/*.json they equal (it refuses
    if they do not; for the code-capacity datasets it also requires the sampler's own matrix to equal them);
  * the integer weight vector, the normative objective, in full, with its sha256;
  * the decoder's input: its parity-check matrix (the checks), every BpOsdDecoder parameter, the two scaling factors,
    and its channel (`error_rate`, or the per-mechanism `error_channel` exactly as the runner passed it);
  * the logical test's data: for the code-capacity datasets the stabilizer rows whose GF(2) row space decides a
    residual (a reference to the persisted instance they equal) and the rule; for the circuit dataset the columns of
    its one declared observable; the phenomenological datasets have none (the campaign records no logical outcome);
  * the sampled data the logical outcomes are computed from, which the records do not carry: every nontrivial shot's
    sampled fault (NumPy datasets), or its sampled observable bit (the circuit dataset). They are obtained by
    re-drawing the dataset's own seeded sampler in the campaign's order, and each is accepted only if it reproduces
    that shot's recorded syndrome and sample index exactly; the archive, not a re-draw, is what a replay reads.
Then a manifest binds each dataset to the sha256 of its input file and of its record file. The records and
summary.json are the preregistered result and are not touched.

Writes evidence/qldpc_scaling_prereg/inputs/<dataset>.json and evidence/qldpc_scaling_prereg/inputs/MANIFEST.json.
The replay is tools/replay_qldpc_scaling_campaign.py, which imports none of this.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
RUNNER = ROOT / "experiments" / "m0_prior_art" / "qldpc_scaling_prereg_campaign.py"
PREREG_DIR = ROOT / "evidence" / "qldpc_scaling_prereg"
OUT_DIR = PREREG_DIR / "inputs"
INPUT_SCHEMA = "oneq-qldpc-scaling-prereg-input/1"
MANIFEST_SCHEMA = "oneq-qldpc-scaling-prereg-inputs-manifest/1"
#: the model-building code build() runs; the manifest records the bytes the export was made under
BUILD_CODE = ("experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py", "experiments/m0_prior_art/qldpc_bb_gate.py",
              "experiments/m0_prior_art/qldpc_bb_phenom.py", "experiments/m0_prior_art/qldpc_bb_refsched_gate.py",
              "experiments/m0_prior_art/qldpc_circuit_gate.py", "src/oneq/qldpc_cert.py")
#: the sector instance whose rows decide a code-capacity residual: Z-side datasets test against H_X, X-side against H_Z
STABILIZER_INSTANCE = {"Z": "bb144_x_code_capacity", "X": "bb144_z_code_capacity"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rel(p: pathlib.Path) -> str:
    """A repository path for the manifest; a path outside the repository would bind a file no reader has."""
    q = p.resolve()
    if not q.is_relative_to(ROOT):
        raise SystemExit(f"{q} is outside the repository; the archive names repository files only")
    return q.relative_to(ROOT).as_posix()


def dumps(obj) -> bytes:
    """The archive's one serialisation: sorted keys, no whitespace, LF, UTF-8."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def weights_sha256(w: list[int]) -> str:
    """The convention of fixed_point_weights and of the receipt manifest: sha256 of json.dumps of the int list."""
    return sha(json.dumps([int(x) for x in w]).encode())


def load_runner():
    spec = importlib.util.spec_from_file_location("prereg_runner_export", RUNNER)
    run = importlib.util.module_from_spec(spec)
    sys.modules["prereg_runner_export"] = run
    spec.loader.exec_module(run)
    return run


def records(name: str) -> list[dict]:
    with gzip.open(PREREG_DIR / f"{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def _instance_ref(inst_name: str, rows: list[list[int]], what: str, *, exact: bool) -> dict:
    """A reference to the persisted instance the rows equal: list for list when exact, else row for row as sets."""
    p = ROOT / "evidence" / "instances" / f"{inst_name}.json"
    inst = json.loads(p.read_text(encoding="utf-8"))
    persisted = [list(map(int, c)) for c in inst["checks"]]
    same = (persisted == rows) if exact else ([sorted(c) for c in persisted] == [sorted(r) for r in rows])
    if not same:
        raise SystemExit(f"{what} differ from {rel(p)}: the export would have to carry them inline; it refuses to "
                         f"guess which is normative")
    return {"file": rel(p), "sha256": sha(p.read_bytes()), "checks_sha256": inst["checks_sha256"],
            "n_variables": int(inst["n_variables"]), "rows": len(rows)}


def export(name: str, run=None, samples: bool = True) -> dict:
    """The exact normative input of one dataset, as build() makes it, with its sampled data verified (samples=False:
    the instance, decoder input and logical test alone, without re-drawing the sampler)."""
    import numpy as np
    run = run or load_runner()
    inst_name, kind, p, max_iter, side, target = run.DATASETS[name]
    D = run.build(name, run.SEED_DOMAIN)
    n = int(D["n"])
    checks = [[int(i) for i in c] for c in D["checks"]]
    doc = {"schema": INPUT_SCHEMA, "dataset": name, "family": kind, "error_rate": float(p),
           "seed_domain": run.SEED_DOMAIN, "seed": int(D["seed"]),
           "seed_rule": "int(sha256(domain + '||' + dataset).hexdigest()[:16], 16) & (2**63 - 1), reduced modulo "
                        "2**31 - 1 for bb_refsched",
           "n_variables": n,
           "checks": _instance_ref(inst_name, checks, f"{name}: the checks build() returns", exact=True)}
    if doc["checks"]["n_variables"] != n:
        raise SystemExit(f"{name}: build() has {n} columns, the persisted instance {doc['checks']['n_variables']}")
    if kind == "hetero":
        doc["channel_seed"] = int(run.seed_for(run.SEED_DOMAIN, "hetero_p02_channel"))
    w = [int(D["wint"][i]) for i in range(n)]
    if sorted(D["wint"]) != list(range(n)) or any(x < 0 for x in w):
        raise SystemExit(f"{name}: the weight vector is not a non-negative integer on every column")
    doc["weights"] = w
    doc["weights_sha256"] = weights_sha256(w)
    chan = {k: (float(v) if not isinstance(v, list) else [float(x) for x in v]) for k, v in D["chan"].items()}
    doc["decoder"] = {"class": "ldpc.BpOsdDecoder", "parity_check_matrix": "the checks, as a 0/1 matrix, row j = check j",
                      "max_iter": int(D["max_iter"]), **{k: v for k, v in run.COMMON.items()},
                      "ms_scaling_factor": [float(a) for a in run.ALPHAS], "channel": chan}
    recs = records(name)
    if len(recs) != target or [r["shot_index"] for r in recs] != list(range(target)):
        raise SystemExit(f"{name}: the record file does not hold shots 0..{target - 1} in order")
    if kind in ("rate", "hetero"):
        from qldpc_bb_gate import bb_gross_code
        HX, HZ = bb_gross_code()
        Hm = HZ if side == "Z" else HX
        sampler_rows = [sorted(int(i) for i in np.flatnonzero(row)) for row in (Hm % 2)]
        if sampler_rows != [sorted(c) for c in checks]:
            raise SystemExit(f"{name}: the sampler's syndrome matrix is not the dataset's checks")
        stab = HX if side == "Z" else HZ
        stab_rows = [sorted(int(i) for i in np.flatnonzero(row)) for row in (stab % 2)]
        doc["logical"] = {
            "kind": "css_residual", "side": side,
            "stabilizer_rows": _instance_ref(STABILIZER_INSTANCE[side], stab_rows, f"{name}: the stabilizer rows",
                                             exact=False),
            "rule": "r = answer XOR sampled fault (both as 0/1 vectors over the n columns); a logical error iff some "
                    "check of the dataset has odd overlap with r, or r is not in the GF(2) row space of "
                    "stabilizer_rows"}
    elif kind == "circuit":
        doc["logical"] = {"kind": "declared_observable", "observable_columns": sorted(int(i) for i in D["obs_cols"]),
                          "rule": "an answer agrees iff the parity of its overlap with observable_columns equals the "
                                  "shot's sampled observable bit"}
    else:
        doc["logical"] = {"kind": "none", "why": "the campaign records no logical outcome for the phenomenological "
                                                 "datasets, as in the held-out campaign"}
    if not samples:
        return doc
    # the sampled data, re-drawn in the campaign's order and accepted only where it reproduces the record
    shots, sampled, k = [], 0, 0
    while k < target:
        syn, fault, ob = D["sample"]()
        sampled += 1
        if not np.asarray(syn).any():
            continue
        r = recs[k]
        sup = [int(j) for j in np.flatnonzero(np.asarray(syn))]
        if sup != r["syndrome_support"] or sampled - 1 != r["sample_index"]:
            raise SystemExit(f"{name} shot {k}: the re-drawn sample does not reproduce the record (sample "
                             f"{sampled - 1} vs {r['sample_index']}); the archive cannot be produced on this host")
        if fault is not None:
            shots.append({"shot_index": k, "sample_index": sampled - 1,
                          "fault": [int(i) for i in np.flatnonzero(np.asarray(fault))]})
        else:
            shots.append({"shot_index": k, "sample_index": sampled - 1, "observable": int(ob)})
        k += 1
    summ = json.loads((PREREG_DIR / "summary.json").read_text(encoding="utf-8"))["datasets"][name]
    if sampled != summ["sampled"] or sampled - target != summ["trivial"]:
        raise SystemExit(f"{name}: {sampled} draws for {target} nontrivial shots; the summary says {summ['sampled']}")
    doc["samples"] = {
        "what": ("the sampled fault of every nontrivial shot, as a support" if kind != "circuit" else
                 "the sampled value of the declared observable on every nontrivial shot"),
        "sampled": sampled, "trivial": sampled - target, "shots": shots,
        "sampler": ("numpy.random.default_rng(seed); fault = rng.random(n) < p_i per draw" if kind != "circuit" else
                    "the detector error model's stim compile_sampler(seed=seed), sampled in batches of 1,000"),
        "note": "a replay reads these values; it never re-samples (a seeded sampler need not reproduce across "
                "machines, library versions or sampling procedures)"}
    return doc


def write_all(check: bool = False) -> dict:
    run = load_runner()
    import ldpc
    import numpy
    import stim
    files, datasets = {}, {}
    for name in run.DATASETS:
        body = dumps(export(name, run))
        files[OUT_DIR / f"{name}.json"] = body
        recp = PREREG_DIR / f"{name}.jsonl.gz"
        doc = json.loads(body)
        inst = {doc["checks"]["file"]: doc["checks"]["sha256"]}
        if doc["logical"]["kind"] == "css_residual":
            inst[doc["logical"]["stabilizer_rows"]["file"]] = doc["logical"]["stabilizer_rows"]["sha256"]
        datasets[name] = {"input": {"file": rel(OUT_DIR / f"{name}.json"), "sha256": sha(body)},
                          "records": {"file": rel(recp), "sha256": sha(recp.read_bytes())},
                          "instances": inst, "nontrivial": len(doc["samples"]["shots"])}
    summ_p = PREREG_DIR / "summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8"))
    for name, d in datasets.items():
        if d["records"]["sha256"] != summ["datasets"][name]["records_sha256"]:
            raise SystemExit(f"{name}: the record file is not the one summary.json records")
    me = pathlib.Path(__file__).resolve()
    manifest = {"schema": MANIFEST_SCHEMA,
                "what": "binds each dataset of the preregistered scaling campaign to its archived normative input "
                        "and to its record file; the replay (tools/replay_qldpc_scaling_campaign.py) pins this "
                        "file's sha256",
                "summary": {"file": rel(summ_p), "sha256": sha(summ_p.read_bytes())},
                "datasets": datasets,
                "expected": {"datasets": len(datasets), "nontrivial": sum(d["nontrivial"] for d in datasets.values())},
                "written_by": {"file": rel(me), "sha256": sha(me.read_bytes())},
                "built_from": {r: sha((ROOT / r).read_bytes()) for r in BUILD_CODE},
                "libraries": {"numpy": numpy.__version__, "stim": stim.__version__, "ldpc": ldpc.__version__}}
    files[OUT_DIR / "MANIFEST.json"] = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    stale = [p for p, b in files.items() if not p.is_file() or p.read_bytes() != b]
    if check:
        return {"stale": [str(p) for p in stale], "files": len(files)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, b in files.items():
        p.write_bytes(b)
    return {"written": [str(p) for p in files], "changed": [str(p) for p in stale]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare the archive with a fresh export; write nothing")
    a = ap.parse_args()
    res = write_all(check=a.check)
    if a.check:
        print("archive current" if not res["stale"] else f"STALE: {res['stale']}")
        return 1 if res["stale"] else 0
    print(f"wrote {len(res['written'])} files ({len(res['changed'])} changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
