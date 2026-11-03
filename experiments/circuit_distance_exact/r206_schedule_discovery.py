r"""R206: proof-gated search over fixed-depth IBM bivariate-bicycle extraction schedules.

This is a bounded, reproducible experiment, NOT global schedule optimization.
The search proposes 240 shuffled X-term orders, assigns one feasible Z-term
order to each, refuses data-qubit conflicts and nondeterministic stabilizers,
then proves the distance of every admitted same-type projected DEM.

Each case includes a full-DEM logical witness and an independently rederived
full detector-error mechanism set. A proposed improvement without all gates is
NOT DECIDED, never a discovery. The sampled schedule space and fixed seven-slot
hardware architecture limit any conclusion. AlphaSyndrome (IBM, ASPLOS 2026)
already provides sophisticated MCTS schedule search; we do NOT claim novelty
from schedule enumeration. The possible differentiated component is the
independent proof gate on every reported candidate, including negative results.

Usage: python experiments/circuit_distance_exact/r206_schedule_discovery.py --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import random
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import bb_circuits as bc  # noqa: E402
import dem_independent as independent  # noqa: E402
import export_model as exporter  # noqa: E402
import tools_cde as tools  # noqa: E402

SEED = 20662
SAMPLED_X_ORDERS = 240
L, ROUNDS, UPPER_W, THREADS = 6, 2, 6, 4
BASE_X, BASE_Z = tuple(bc.SX), tuple(bc.SZ)
EXPECTED_DETECTORS, EXPECTED_LOGICALS = 144, 12
FOUND_RE = re.compile(r"^w<=(\d+): FOUND weight (\d+) witness:([\d ]+)", re.M)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proposals(seed: int = SEED, sample: int = SAMPLED_X_ORDERS):
    """Seeded, explicitly bounded search. Each term is used once per check.

    At any paired layer X and Z must touch opposite code-data halves;
    otherwise the same data qubit would participate in two CNOTs. Busy-set
    assertions in bc.ibm_cycle are a second, independent rejection.
    """
    rng = random.Random(seed)
    xorders = list(itertools.permutations(range(6)))
    rng.shuffle(xorders)
    for ix, xt in enumerate(xorders[:sample]):
        sx = ("idle", *xt)
        available = [set(range(3)), set(range(3, 6))]
        sz = [None] * 7
        sz[6] = "idle"
        order = list(range(1, 6))
        rng.shuffle(order)
        for slot in order:
            other_half = 0 if sx[slot] >= 3 else 1
            choices = available[other_half]
            if not choices:
                break
            term = rng.choice(sorted(choices))
            choices.remove(term)
            sz[slot] = term
        else:
            leftover = sorted(available[0] | available[1])
            if len(leftover) == 1:
                sz[0] = leftover[0]
                yield ix, sx, tuple(sz)


def circuit_is_valid(sx: tuple, sz: tuple) -> bool:
    """Reject nondeterministic detectors in both bases before optimization."""
    saved = bc.SX, bc.SZ
    try:
        bc.SX, bc.SZ = sx, sz
        for basis in ("X", "Z"):
            try:
                dem = bc.ibm_cycle(basis, ROUNDS, l=L).detector_error_model(decompose_errors=False)
            except (ValueError, AssertionError):
                return False
            if (dem.num_detectors, dem.num_observables) != (EXPECTED_DETECTORS, EXPECTED_LOGICALS):
                return False
        return True
    finally:
        bc.SX, bc.SZ = saved


def find_and_verify(kind: str, sx: tuple, sz: tuple, binpaths: dict, output: pathlib.Path) -> dict:
    saved = bc.SX, bc.SZ
    base = output / "cases" / kind
    base.mkdir(parents=True, exist_ok=True)
    record: dict = {"name": kind, "sx": list(sx), "sz": list(sz), "basis": []}
    try:
        bc.SX, bc.SZ = sx, sz
        for basis in ("X", "Z"):
            model = base / f"{basis}_model.txt"
            info = exporter.export("ibm", basis, ROUNDS, model, l=L)
            c = bc.ibm_cycle(basis, ROUNDS, l=L)
            check_faults = independent.compare(c)
            if not check_faults["equal"]:
                raise RuntimeError(f"{kind} {basis}: independently extracted DEM differs")
            run = tools.run(binpaths["cdist"], model, 1, UPPER_W, 20, threads=THREADS)
            if run.returncode != 0:
                raise RuntimeError(f"{kind} {basis}: search error: {run.stderr}")
            match = FOUND_RE.search(run.stdout)
            if match is None:
                # The claimed upper bound has not been established. Do not
                # label the schedule better or distance-preserving.
                raise RuntimeError(f"{kind} {basis}: no witness <= {UPPER_W}; NOT DECIDED")
            search_w, witness_w = int(match.group(1)), int(match.group(2))
            if search_w != witness_w:
                raise RuntimeError("search witness weight inconsistent")
            witness = [int(i) for i in match.group(3).split()]
            if len(witness) != witness_w or len(witness) != len(set(witness)):
                raise RuntimeError("witness contains repeated or missing mechanism indices")
            lifted = tools.lift("ibm", basis, ROUNDS, witness, l=L)
            if (not lifted or len(lifted) != witness_w
                    or not tools.verify_full("ibm", basis, ROUNDS, lifted, l=L)):
                raise RuntimeError(f"{kind} {basis}: projected witness FAILS full-model verification")
            proof = base / f"{basis}_lower_{witness_w - 1}.cdp5"
            pv = tools.run(binpaths["cdist_prove"], model, witness_w - 1, proof, threads=THREADS)
            if pv.returncode != 0:
                raise RuntimeError(f"{kind} {basis}: lower-bound prover did not close ({pv.stdout})")
            cv = tools.run(binpaths["ccheck"], model, proof, "full", threads=THREADS)
            if cv.returncode != 0:
                raise RuntimeError(f"{kind} {basis}: stranger checker REJECTED ({cv.stdout})")
            assert proof.read_bytes().startswith(b"CDP5")
            record["basis"].append({
                "basis": basis,
                "status": "PROJECTED_EXACT_AND_FULL_UPPER_WITNESS",
                "fault_weight": witness_w,
                "model_path": str(model.relative_to(output)).replace("\\", "/"),
                "model_sha256": digest(model.read_bytes()),
                "proof_path": str(proof.relative_to(output)).replace("\\", "/"),
                "proof_sha256": digest(proof.read_bytes()),
                "proof_bytes": proof.stat().st_size,
                "projected_witness": witness,
                "full_witness": lifted,
                "full_witness_verified": True,
                "dem_independent_equal": True,
                "detectors": info["detectors"],
                "mechanisms": info["mechanisms"],
                "search_provenance": run.stdout.splitlines()[0],
            })
    finally:
        bc.SX, bc.SZ = saved
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    output = a.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_dir = output / "_temporary_build"
    build_dir.mkdir(exist_ok=True)
    binaries = tools.build_c(build_dir)
    if set(binaries) != {"cdist", "cdist_prove", "ccheck"}:
        raise RuntimeError("GCC/OpenMP binaries not available; REFUSED")
    admissible = []
    attempted = 0
    for ix, sx, sz in proposals():
        attempted += 1
        if circuit_is_valid(sx, sz):
            admissible.append((ix, sx, sz))
    print(f"R206 validity: sampled {SAMPLED_X_ORDERS}, feasible side-matched {attempted}, "
          f"two-basis detector-valid {len(admissible)}", flush=True)
    records = [find_and_verify("baseline", BASE_X, BASE_Z, binaries, output)]
    for ix, sx, sz in admissible:
        case = find_and_verify(f"proposal_{ix:03d}", sx, sz, binaries, output)
        case["sample_index"] = ix
        records.append(case)
        print("CASE", case["name"], [(b["basis"], b["fault_weight"]) for b in case["basis"]], flush=True)
    baseline = {r["basis"]: r["fault_weight"] for r in records[0]["basis"]}
    for row in records:
        distances = {b["basis"]: b["fault_weight"] for b in row["basis"]}
        row["verdict"] = (
            "BASELINE" if row["name"] == "baseline"
            else "REJECTED_DISTANCE_REGRESSION" if any(distances[k] < baseline[k] for k in baseline)
            else "SAME_PROJECTED_DISTANCE__LER_NOT_MEASURED" if distances == baseline
            else "NOT_DECIDED"
        )
    # Prove that every accepted statement resolves to real bytes. The
    # registry hash deliberately excludes the registry itself.
    registry = {
        "schema": "oneq.r206.proof_gated_schedule_search.v1",
        "scope": "seeded sample, fixed hardware depth, same-type projected distance, full-model upper witness",
        "sample_seed": SEED,
        "sample_size": SAMPLED_X_ORDERS,
        "side_matched_proposals": attempted,
        "valid_two_basis_schedules": len(admissible),
        "candidate_count_including_baseline": len(records),
        "input_model": "[[72,12,6]] IBM-style BB, two memory rounds, seven CNOT slots",
        "source_hashes": {name: digest((HERE / name).read_bytes()) for name in
                          ("bb_circuits.py", "export_model.py", "dem_independent.py",
                           "cdist.c", "cdist_prove.c", "ccheck.c",
                           "r206_schedule_discovery.py")},
        "not_claimed": ["globally optimal scheduling", "physical-hardware improvement",
                        "logical-error-rate superiority", "novel schedule synthesis theorem"],
        "prior_art": "AlphaSyndrome, ASPLOS 2026, arXiv:2601.12509; R198 circuit-distance work",
        "records": records,
    }
    for item in records:
        for entry in item["basis"]:
            for suffix in ("model", "proof"):
                got = digest((output / entry[suffix + "_path"]).read_bytes())
                if got != entry[suffix + "_sha256"]:
                    raise RuntimeError("artifact mutated in the middle of the run")
    # Write raw LF bytes, not platform-translated text. Published registry
    # bytes must be identical on Windows, Linux and clean Git checkouts.
    (output / "registry.json").write_bytes((json.dumps(registry, indent=2) + "\n").encode("utf-8"))
    print("R206 RESULT", json.dumps({
        "generated": len(records),
        "rejected": sum(r["verdict"] == "REJECTED_DISTANCE_REGRESSION" for r in records),
        "same_distance": sum(r["verdict"] == "SAME_PROJECTED_DISTANCE__LER_NOT_MEASURED" for r in records),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "registry": str(output / "registry.json"),
    }), flush=True)
    # Temporary executables are build products; the registry only binds
    # committed source and proof artifacts, not a compiler's binary output.
    import shutil
    shutil.rmtree(build_dir)


if __name__ == "__main__":
    main()
