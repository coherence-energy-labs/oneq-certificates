r"""THE FROZEN HELD-OUT CONFIRMATORY CAMPAIGN -- runner of record.

Committed BEFORE the freeze per docs/HELD_OUT_PREDECLARATION.md. After
the freeze commit (tag: held-out-freeze) this script is the ONLY thing
that runs, and nothing in the freeze set may change until the results
section is written. Every design decision lives in the predeclaration;
this file only executes it.

SEEDS: seed_rung = SHA256("oneq-heldout-2026" || rung_name ||
freeze_commit_sha)[:16] as int, masked to 63 bits -- derived at runtime
from the tag, so seeds provably postdate the freeze and predate nothing.

RUNGS (same protocols as canonical, fresh seeds): uniform p01 (300
nontrivial), uniform p02 (470), hetero p02 fixed-point (300; fresh
channel from the same seed rule), phenom r=4 p01/p02 (150 each),
xsector p02 (200), stress p05 (300), refsched circuit r=3 p=0.002
(150 shots, via subprocess with --prefix heldout_ --bd-close).

PIPELINE per shot: tier 0 / 1 / 2 flat exact / tier BD branch-dual
closure; every exact certificate through the pinned reference checker;
per-shot records at evidence/per_shot/heldout_*.jsonl. Results are
reported AS OBSERVED -- no re-runs, no exclusions; discordance with the
canonical rates is a finding.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
from qldpc_bb_phenom import spacetime_system
from qldpc_exact_all_gate import _load_reference, run_rung
from oneq.qldpc_cert import fixed_point_weights


def heldout_seed(rung: str, freeze_sha: str) -> int:
    h = hashlib.sha256(
        f"oneq-heldout-2026||{rung}||{freeze_sha}".encode()).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def main() -> int:
    freeze_sha = subprocess.run(
        ["git", "rev-parse", "held-out-freeze"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()
    assert len(freeze_sha) == 40, \
        "tag held-out-freeze absent: freeze first, run second"
    print(f"freeze commit: {freeze_sha}")

    ref = _load_reference()
    from ldpc import BpOsdDecoder
    from scipy.sparse import csr_matrix, lil_matrix

    HX, HZ = bb_gross_code()
    m, n = HZ.shape
    checks_z = [tuple(int(i) for i in np.flatnonzero(HZ[j]))
                for j in range(m)]
    checks_x = [tuple(int(i) for i in np.flatnonzero(HX[j]))
                for j in range(m)]
    xb, xp = gf2_row_reduce(HX % 2)
    zb, zp = gf2_row_reduce(HZ % 2)

    def logical_z_side(fault):
        if ((HZ @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(xb, xp, fault, n)

    def logical_x_side(fault):
        if ((HX @ fault) % 2).any():
            return True
        return not gf2_in_rowspace(zb, zp, fault, n)

    wint_u = {i: 1 for i in range(n)}
    spec_u = {"scale": 1, "rounding": "uniform-unit",
              "definition": "w_i = 1 (Hamming weight)",
              "weights_sha256": hashlib.sha256(
                  json.dumps([1] * n).encode()).hexdigest()}

    out = {"schema": "oneq-qldpc-heldout/1",
           "freeze_commit": freeze_sha,
           "seed_rule": "SHA256('oneq-heldout-2026'||rung||freeze_sha)"
                        "[:16] hex as int, 63-bit mask",
           "rungs": {}}

    def sampler_cc(H, p_arr, seed):
        rng = np.random.default_rng(seed)

        def s():
            e = (rng.random(n) < p_arr).astype(np.uint8)
            return (H @ e) % 2, e
        return s

    # --- uniform p01 / p02 ---
    for tag, p, tgt in (("uniform_p01", 0.01, 300),
                        ("uniform_p02", 0.02, 470)):
        sd = heldout_seed(tag, freeze_sha)
        dec = BpOsdDecoder(csr_matrix(HZ), error_rate=p, max_iter=30,
                           bp_method="ms", schedule="parallel",
                           osd_method="osd_cs", osd_order=5)
        out["rungs"][tag] = run_rung(
            name=tag, checks=checks_z, wint=wint_u, wspec=spec_u,
            sample_syndrome=sampler_cc(HZ, np.full(n, p), sd),
            decoder=dec, target=tgt, ref=ref, is_logical=logical_z_side,
            bd_close=True, record_prefix="heldout_")
        out["rungs"][tag]["seed"] = sd

    # --- hetero p02: fresh channel from the seed rule too ---
    sd_ch = heldout_seed("hetero_p02_channel", freeze_sha)
    sd_sh = heldout_seed("hetero_p02", freeze_sha)
    rngw = np.random.default_rng(sd_ch)
    p_i = 0.02 * rngw.uniform(0.5, 2.0, n)
    llr = {i: float(np.log((1 - p_i[i]) / p_i[i])) for i in range(n)}
    wint_h, spec_h = fixed_point_weights(llr, scale=1000)
    dec_h = BpOsdDecoder(csr_matrix(HZ),
                         error_channel=[float(v) for v in p_i],
                         max_iter=30, bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    out["rungs"]["hetero_p02"] = run_rung(
        name="hetero_p02", checks=checks_z, wint=wint_h, wspec=spec_h,
        sample_syndrome=sampler_cc(HZ, p_i, sd_sh), decoder=dec_h,
        target=300, ref=ref, is_logical=logical_z_side, bd_close=True,
        record_prefix="heldout_")
    out["rungs"]["hetero_p02"]["seed"] = sd_sh
    out["rungs"]["hetero_p02"]["channel_seed"] = sd_ch

    # --- phenom p01 / p02 ---
    for p, tag in ((0.01, "phenom_p01"), (0.02, "phenom_p02")):
        sd = heldout_seed(tag, freeze_sha)
        checks_st, w_st = spacetime_system(HZ, 4, p, p)
        nn = len(w_st)
        wint_p, spec_p = fixed_point_weights(w_st, scale=1000)
        A = lil_matrix((len(checks_st), nn))
        for j, sup in enumerate(checks_st):
            for i in sup:
                A[j, i] = 1
        dec_p = BpOsdDecoder(A.tocsr().astype(np.uint8),
                             error_channel=[float(p)] * nn,
                             max_iter=30, bp_method="ms",
                             schedule="parallel", osd_method="osd_cs",
                             osd_order=5)
        rng_p = np.random.default_rng(sd)

        def sample_p(rng=rng_p, nn=nn, checks_st=checks_st, p=p):
            e = (rng.random(nn) < p).astype(np.uint8)
            syn = np.array([sum(e[i] for i in sup) % 2
                            for sup in checks_st], dtype=np.uint8)
            return syn, e
        out["rungs"][tag] = run_rung(
            name=tag, checks=checks_st, wint=wint_p, wspec=spec_p,
            sample_syndrome=sample_p, decoder=dec_p, target=150, ref=ref,
            bd_close=True, record_prefix="heldout_")
        out["rungs"][tag]["seed"] = sd

    # --- xsector p02 ---
    sd = heldout_seed("xsector_p02", freeze_sha)
    dec_x = BpOsdDecoder(csr_matrix(HX), error_rate=0.02, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    out["rungs"]["xsector_p02"] = run_rung(
        name="xsector_p02", checks=checks_x, wint=wint_u, wspec=spec_u,
        sample_syndrome=sampler_cc(HX, np.full(n, 0.02), sd),
        decoder=dec_x, target=200, ref=ref, is_logical=logical_x_side,
        bd_close=True, record_prefix="heldout_")
    out["rungs"]["xsector_p02"]["seed"] = sd

    # --- stress p05 ---
    sd = heldout_seed("stress_p05", freeze_sha)
    dec_s = BpOsdDecoder(csr_matrix(HZ), error_rate=0.05, max_iter=30,
                         bp_method="ms", schedule="parallel",
                         osd_method="osd_cs", osd_order=5)
    out["rungs"]["stress_p05"] = run_rung(
        name="stress_p05", checks=checks_z, wint=wint_u, wspec=spec_u,
        sample_syndrome=sampler_cc(HZ, np.full(n, 0.05), sd),
        decoder=dec_s, target=300, ref=ref, is_logical=logical_z_side,
        bd_close=True, record_prefix="heldout_")
    out["rungs"]["stress_p05"]["seed"] = sd

    # --- refsched circuit (subprocess with its own validation) ---
    sd = heldout_seed("bb_refsched", freeze_sha) % (2**31 - 1)
    r = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "m0_prior_art" /
             "qldpc_bb_refsched_gate.py"),
         "--rounds", "3", "--p", "0.002", "--shots", "150",
         "--seed", str(sd), "--prefix", "heldout_", "--bd-close"],
        capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-600:])
    assert r.returncode == 0, r.stderr[-800:]
    out["rungs"]["bb_refsched"] = json.loads(
        (ROOT / "evidence" / "heldout_qldpc_bb_refsched_gate.json")
        .read_text(encoding="utf-8"))
    out["rungs"]["bb_refsched"]["seed"] = sd

    # --- summary table, generated from the run results only ---
    print("\n=== HELD-OUT RESULTS (as observed) ===")
    for tag, run in out["rungs"].items():
        nn = run["nontrivial"]
        ec = run["exact_certified"]
        bd = run.get("tierBD", 0)
        rest = run.get("tier3A", 0) + run.get("tier3B", 0) \
            + run.get("residue_unsolved", 0)
        print(f"  {tag:14s} {ec}/{nn} exact "
              f"(t1 {run.get('tier1', run.get('tier1', 0))}, "
              f"t2 {run.get('tier2', 0)}, BD {bd}, open {rest})")

    dest = ROOT / "evidence" / "qldpc_heldout.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # Extend the release manifest with the exact held-out record AND weight
    # subjects.  The original merge named only the JSONL files, leaving eight
    # held-out weight vectors present but undeclared; a verifier could then
    # skip the entire weight directory or accept an arbitrary subset.
    pdir = ROOT / "evidence" / "per_shot"
    man_path = pdir / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    held_manifest = {
        "freeze_commit": freeze_sha,
        "seed_rule": out["seed_rule"],
        "rungs": {},
    }
    for tag, run in out["rungs"].items():
        records = pdir / f"heldout_{tag}.jsonl"
        weights = ROOT / "evidence" / "weights" / \
            f"heldout_{tag}_weights.json"
        if not records.is_file() or not weights.is_file():
            raise RuntimeError(f"held-out release subject missing for {tag}")
        weights_doc = json.loads(weights.read_text(encoding="utf-8"))
        held_manifest["rungs"][tag] = {
            "records_file": f"evidence/per_shot/heldout_{tag}.jsonl",
            "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "record_count": run["nontrivial"],
            "weights_file": f"evidence/weights/heldout_{tag}_weights.json",
            "weights_sha256": weights_doc["spec"]["weights_sha256"],
            "exact_certified": run["exact_certified"],
            "nontrivial": run["nontrivial"],
            "seed": run["seed"],
        }
    man["schema"] = "oneq-per-shot-manifest/3"
    man["heldout"] = held_manifest
    man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"evidence -> {dest} + held-out records/weights manifest closure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
