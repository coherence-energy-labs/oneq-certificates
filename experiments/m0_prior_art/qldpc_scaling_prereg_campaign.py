r"""THE PREREGISTERED SCALING CAMPAIGN -- runner of record for docs/PREREG_qldpc_scaling_2026-11-03.md.

    python experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py --dataset all
    python experiments/m0_prior_art/qldpc_scaling_prereg_campaign.py --smoke 5 --out-dir <scratch>   # a dry run

On FRESH syndromes of the eight paper-2 dataset families, BP-OSD (ldpc 2.4.1 BpOsdDecoder) decodes every syndrome
twice, with ms_scaling_factor 1.0 (the class default) and 0.625 (the default of ldpc's SinterBpOsdDecoder), every
other parameter equal. Each syndrome receives ONE exact receipt for its optimum, independent of both answers, and
both answers are scored against it. The preregistration fixes the seeds, the datasets, the shot counts, this code
and the analysis; the conclusion is whatever the data say.

PER SHOT:
  1. sample a fault from the dataset's own model with its preregistered seed; trivial syndromes are counted, not used;
  2. decode with both configurations;
  3. the receipt: the dataset's LP producer (feldman_dual with 6 redundant-parity rounds; feldman_dual_lazy at
     circuit level), rationalised once; the first of (answer at 1.0, answer at 0.625, the LP primal) that the
     production exact checker accepts with that single dual fixes the optimum U. If none does, the MILP proposes a
     candidate and certified_bnb a branch-dual tree, accepted by check_qldpc_bnb. Every receipt is then checked by
     the second checker too (the reference flat checker, or tree checker B); a receipt either refuses is UNRESOLVED;
  4. each configuration's answer is scored: tier 1 (accepted by the shot's single dual), optimum hit on a branching
     tree (feasible with objective U), or a miss (feasible and heavier, or not a correction);
  5. logical outcome against the sampled fault (code-capacity datasets), or the declared observable against the
     sampled truth (circuit dataset); the phenomenological datasets carry no logical outcome, as in the campaign.

OUTPUT (real run): evidence/qldpc_scaling_prereg/<dataset>.jsonl.gz (deterministic gzip, one record per nontrivial
shot, receipts included) and evidence/qldpc_scaling_prereg/summary.json, which carries the preregistered analysis.
A smoke run uses a DIFFERENT seed domain, a few shots, and refuses to write under evidence/.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import platform
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
PREREG = ROOT / "docs" / "PREREG_qldpc_scaling_2026-11-03.md"
OUT_DIR = ROOT / "evidence" / "qldpc_scaling_prereg"
SEED_DOMAIN = "oneq-qldpc-scaling-prereg-2026-11-03"
SMOKE_DOMAIN = "oneq-qldpc-scaling-SMOKE-never-the-preregistered-seeds"
ALPHAS = (1.0, 0.625)
COMMON = dict(bp_method="ms", schedule="parallel", osd_method="osd_cs", osd_order=5)
# dataset -> (persisted instance, family, error rate, BP iterations, logical side, nontrivial target)
DATASETS = {
    "uniform_p01": ("bb144_z_code_capacity", "rate", 0.01, 30, "Z", 1000),
    "uniform_p02": ("bb144_z_code_capacity", "rate", 0.02, 30, "Z", 1000),
    "hetero_p02": ("bb144_z_code_capacity", "hetero", 0.02, 30, "Z", 1000),
    "phenom_p01": ("bb144_phenom_r4", "phenom", 0.01, 30, None, 500),
    "phenom_p02": ("bb144_phenom_r4", "phenom", 0.02, 30, None, 500),
    "xsector_p02": ("bb144_x_code_capacity", "rate", 0.02, 30, "X", 1000),
    "stress_p05": ("bb144_z_code_capacity", "rate", 0.05, 30, "Z", 1000),
    "bb_refsched": ("bb144_refsched_r3_p002", "circuit", 0.002, 40, None, 500),
}
SEED_NAMES = list(DATASETS) + ["hetero_p02_channel"]


def seed_for(domain: str, name: str) -> int:
    s = int(hashlib.sha256(f"{domain}||{name}".encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)
    return s % (2 ** 31 - 1) if name == "bb_refsched" else s


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mcnemar(n10: int, n01: int) -> float:
    nd, k = n10 + n01, min(n10, n01)
    return min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd) if nd else 1.0


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(name: str, domain: str):
    """Everything the dataset fixes: checks, integer weights, decoder channel, sampler, logical test."""
    from scipy.sparse import csr_matrix
    from oneq.qldpc_cert import fixed_point_weights
    from qldpc_bb_gate import bb_gross_code, gf2_in_rowspace, gf2_row_reduce
    from qldpc_bb_phenom import spacetime_system
    inst_name, kind, p, max_iter, side, target = DATASETS[name]
    inst = json.loads((ROOT / "evidence" / "instances" / f"{inst_name}.json").read_text(encoding="utf-8"))
    checks = [tuple(int(i) for i in c) for c in inst["checks"]]
    n = int(inst["n_variables"])
    HX, HZ = bb_gross_code()
    seed = seed_for(domain, name)
    obs_cols = None
    if kind in ("rate", "hetero"):
        Hm = HZ if side == "Z" else HX
        if kind == "hetero":
            p_arr = 0.02 * np.random.default_rng(seed_for(domain, "hetero_p02_channel")).uniform(0.5, 2.0, n)
            wint, _spec = fixed_point_weights({i: float(np.log((1 - p_arr[i]) / p_arr[i])) for i in range(n)},
                                              scale=1000)
            chan = dict(error_channel=[float(v) for v in p_arr])
        else:
            p_arr = np.full(n, p)
            wint = {i: 1 for i in range(n)}
            chan = dict(error_rate=p)
        rng = np.random.default_rng(seed)

        def sample():
            e = (rng.random(n) < p_arr).astype(np.uint8)
            return ((Hm @ e) % 2).astype(np.uint8), e, None
        rb, rp = gf2_row_reduce((HX if side == "Z" else HZ) % 2)

        def logical(f):
            return bool(((Hm @ f) % 2).any()) or not gf2_in_rowspace(rb, rp, f, n)
    elif kind == "phenom":
        cst, w_st = spacetime_system(HZ, 4, p, p)
        if [tuple(sorted(c)) for c in cst] != [tuple(sorted(c)) for c in checks]:
            raise SystemExit("phenomenological system differs from the persisted instance")
        wint, _spec = fixed_point_weights(w_st, scale=1000)
        chan = dict(error_channel=[float(p)] * n)
        rng = np.random.default_rng(seed)

        def sample():
            e = (rng.random(n) < p).astype(np.uint8)
            syn = np.array([int(sum(int(e[i]) for i in c) % 2) for c in checks], dtype=np.uint8)
            return syn, e, None
        logical = None
    else:
        from qldpc_bb_refsched_gate import build_refsched_circuit
        from qldpc_circuit_gate import dem_to_matrices
        dem = build_refsched_circuit(3, 0.002).detector_error_model(decompose_errors=False)
        cks, wllr, obs_rows, n_mech = dem_to_matrices(dem)
        if [tuple(c) for c in cks] != checks or n_mech != n:
            raise SystemExit("circuit detector error model differs from the persisted instance")
        wint = {i: int(math.floor(1000 * float(wllr[i]) + 0.5)) for i in range(n)}
        chan = dict(error_channel=[float(1 / (1 + np.exp(wllr[i]))) for i in range(n)])
        obs_cols = set(int(i) for i in obs_rows.get(0, ()))
        sampler = dem.compile_sampler(seed=seed)
        buf: list = []

        def sample():
            if not buf:
                d, o, _ = sampler.sample(shots=1000, return_errors=False)
                buf.extend((d[s].astype(np.uint8), int(o[s, 0])) for s in range(d.shape[0]))
            syn, ob = buf.pop(0)
            return syn, None, ob
        logical = None
    rows = [j for j, c in enumerate(checks) for _ in c]
    cols = [i for c in checks for i in c]
    H = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(checks), n))
    return dict(checks=checks, n=n, wint=wint, chan=chan, sample=sample, logical=logical, obs_cols=obs_cols,
                H=H, seed=seed, kind=kind, max_iter=max_iter, target=target)


def run_dataset(name: str, domain: str, target: int, ref, tree_b: pathlib.Path) -> tuple[dict, list]:
    from ldpc import BpOsdDecoder
    from oneq.qldpc_cert import (QldpcCertificate, certified_bnb, exact_milp, exactify_certificate, feldman_dual,
                                 feldman_dual_lazy)
    from oneq.qldpc_check import check_qldpc_bnb, check_qldpc_exact
    D = build(name, domain)
    checks, n, wint, H = D["checks"], D["n"], D["wint"], D["H"]
    wf = {i: float(wint[i]) for i in range(n)}
    wx = {i: Fraction(int(wint[i])) for i in range(n)}
    decs = {a: BpOsdDecoder(H, max_iter=D["max_iter"], ms_scaling_factor=a, **COMMON, **D["chan"]) for a in ALPHAS}
    dense = None
    recs, sampled, trivial = [], 0, 0

    def obj(sup):
        return int(sum(wint[int(i)] for i in sup))

    def feasible(sup, syn):
        e = set(int(i) for i in sup)
        return all(len(e & set(c)) % 2 == int(b) for c, b in zip(checks, syn))

    def ref_flat(syn, cand, fd) -> bool:
        nonlocal dense
        if dense is None:
            dense = [[0] * n for _ in checks]
            for j, c in enumerate(checks):
                for i in c:
                    dense[j][i] = 1
        e = [0] * n
        for i in cand:
            e[int(i)] = 1
        doc = {"H": dense, "s": [int(b) for b in syn], "weights": [{"num": int(wint[i]), "den": 1} for i in range(n)],
               "candidate": e, "facets": [{"rows": sorted(r) if isinstance(r, list) else [r], "F": sorted(F),
                                           "y": {"num": a, "den": b}} for (r, F, a, b) in fd]}
        try:
            return bool(ref.check_certificate(doc)["objective_optimal"])
        except ref.CertificateError:
            return False

    def ser_tree(t):
        if t[0] == "branch":
            return ["branch", int(t[1]), ser_tree(t[2]), ser_tree(t[3])]
        if t[0] == "bound":
            return ["bound", [[list(j) if isinstance(j, (tuple, list, frozenset)) else int(j), sorted(int(i) for i in F),
                               Fraction(y).numerator, Fraction(y).denominator] for (j, F, y) in t[1]]]
        return ["infeasible", list(t[1]) if isinstance(t[1], (tuple, list)) else int(t[1])]

    def tree_b_accepts(syn, cand, tree) -> bool:
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "doc.json"
            p.write_text(json.dumps({"checks": [list(c) for c in checks], "weights": [int(wint[i]) for i in range(n)],
                                     "syndrome": [int(b) for b in syn], "candidate": [int(i) for i in cand],
                                     "tree": tree}), encoding="utf-8")
            r = subprocess.run([sys.executable, "-B", str(tree_b), str(p)], capture_output=True, text=True, timeout=600)
        return r.returncode == 0

    t0 = time.perf_counter()
    while len(recs) < target:
        sampled += 1
        syn, fault, ob = D["sample"]()
        if not syn.any():
            trivial += 1
            continue
        s = [int(b) for b in syn]
        ans = {a: sorted(int(i) for i in np.flatnonzero(np.asarray(decs[a].decode(syn)))) for a in ALPHAS}
        rec = {"shot_index": len(recs), "sample_index": sampled - 1, "syndrome_support": [j for j, b in enumerate(s) if b],
               "answers": {str(a): {"support": ans[a], "syndrome_valid": feasible(ans[a], s), "objective": obj(ans[a])}
                           for a in ALPHAS}}
        # DEVIATION 1 (2026-11-03, docs/PREREG_qldpc_scaling_2026-11-03_DEVIATIONS.md): the phenomenological
        # datasets use the lazy-separation LP, as the circuit dataset always did. Under the committed runner the
        # 6-round redundant-parity LP on the 360 x 864 spacetime system took 18.6 s, 42.9 s and 206.5 s on three of
        # the first 81 phenom_p02 shots, and the dataset did not finish in 3,000 s. The receipt's FORM changes
        # (fewer flat receipts, more trees); the certified optimum of a resolved shot, and so every score, does not.
        lp = (feldman_dual_lazy(checks, wf, s, return_primal=True) if D["kind"] in ("circuit", "phenom")
              else feldman_dual(checks, wf, s, return_primal=True, rpc_rounds=6))
        cert, _bound, primal = lp
        receipt = None
        if cert is not None:
            ec = exactify_certificate(QldpcCertificate(error_support=(), facet_duals=cert.facet_duals,
                                                       box_duals=cert.box_duals))
            fd = [[sorted(int(r) for r in j) if isinstance(j, (tuple, list, frozenset)) else int(j),
                   sorted(int(i) for i in F), y.numerator, y.denominator] for (j, F, y) in ec.facet_duals if y != 0]
            tries = [ans[1.0], ans[0.625]] + ([sorted(int(i) for i in primal)] if primal is not None else [])
            for cand in tries:
                if feasible(cand, s) and check_qldpc_exact(
                        checks=checks, weights=wx, syndrome=s,
                        cert=QldpcCertificate(error_support=tuple(cand), facet_duals=ec.facet_duals)).accepted:
                    receipt = {"kind": "flat", "answer": cand, "optimum": obj(cand), "facet_duals": fd,
                               "second_checker": ref_flat(s, cand, fd)}
                    break
        if receipt is None:
            ms, _w, st = exact_milp(checks, wf, s, return_status=True)
            if ms is not None:
                ms = sorted(int(i) for i in ms)
                tree, nodes = certified_bnb(checks, wf, s, obj(ms), rpc_rounds=6, max_nodes=60000, use_lazy=True)
                if tree is not None and check_qldpc_bnb(checks=checks, weights=wx, syndrome=s,
                                                        error_support=tuple(ms), tree=tree).accepted:
                    stree = ser_tree(tree)
                    receipt = {"kind": "tree", "answer": ms, "optimum": obj(ms), "tree": stree, "nodes": nodes,
                               "second_checker": tree_b_accepts(s, ms, stree)}
        rec["receipt"] = receipt if (receipt is not None and receipt["second_checker"]) else None
        rec["unresolved_reason"] = (None if rec["receipt"] is not None else
                                    "no receipt closed" if receipt is None else "the second checker refused")
        if rec["receipt"] is not None:
            R = rec["receipt"]
            for a in ALPHAS:
                A = rec["answers"][str(a)]
                if not A["syndrome_valid"]:
                    A["score"] = "not_a_correction"
                    continue
                if A["objective"] < R["optimum"]:
                    raise SystemExit(f"{name} shot {rec['shot_index']}: an answer lighter than its receipt; STOP")
                single = R["kind"] == "flat" or R["tree"][0] == "bound"
                if single:
                    duals = R["facet_duals"] if R["kind"] == "flat" else R["tree"][1]
                    fdq = tuple((tuple(r) if isinstance(r, list) else r, tuple(F), Fraction(p_, q_))
                                for (r, F, p_, q_) in duals)
                    t1 = check_qldpc_exact(checks=checks, weights=wx, syndrome=s,
                                           cert=QldpcCertificate(error_support=tuple(A["support"]),
                                                                 facet_duals=fdq)).accepted
                    A["score"] = "tier1" if t1 else "miss"
                else:
                    A["score"] = "optimum_branching" if A["objective"] == R["optimum"] else "miss"
                if A["score"] == "miss" and A["objective"] == R["optimum"]:
                    raise SystemExit(f"{name}: an answer at the optimum refused by an answer-free bound; STOP")
        if D["logical"] is not None:
            rec["logical_error"] = {str(a): D["logical"]((np.isin(np.arange(n), ans[a]).astype(np.uint8) ^ fault) % 2)
                                    for a in ALPHAS}
            if rec["receipt"] is not None:
                rec["logical_error"]["receipt"] = D["logical"](
                    (np.isin(np.arange(n), rec["receipt"]["answer"]).astype(np.uint8) ^ fault) % 2)
        if D["obs_cols"] is not None:
            oc = D["obs_cols"]
            rec["observable_agrees"] = {str(a): int(sum(1 for i in ans[a] if i in oc) % 2 == ob) for a in ALPHAS}
            if rec["receipt"] is not None:
                rec["observable_agrees"]["receipt"] = int(sum(1 for i in rec["receipt"]["answer"] if i in oc) % 2 == ob)
        recs.append(rec)
    summary = analyse(name, recs, sampled, trivial, D)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    return summary, recs


def analyse(name: str, recs: list, sampled: int, trivial: int, D: dict) -> dict:
    """The preregistered analysis, per dataset; no pooled claim is computed."""
    res = {"dataset": name, "seed": D["seed"], "sampled": sampled, "trivial": trivial, "nontrivial": len(recs),
           "receipts": sum(r["receipt"] is not None for r in recs),
           "unresolved": sum(r["receipt"] is None for r in recs),
           "receipt_kinds": {k: sum(1 for r in recs if r["receipt"] and r["receipt"]["kind"] == k) for k in ("flat", "tree")},
           "per_configuration": {}}
    resolved = [r for r in recs if r["receipt"] is not None]
    hit = {}
    for a in ALPHAS:
        sc = [r["answers"][str(a)]["score"] for r in resolved]
        hit[a] = [s_ in ("tier1", "optimum_branching") for s_ in sc]
        d = {"optimum_hit": sum(hit[a]), "tier1_single_dual": sc.count("tier1"),
             "optimum_on_branching_tree": sc.count("optimum_branching"), "miss_heavier": sc.count("miss"),
             "not_a_correction": sc.count("not_a_correction")}
        if D["logical"] is not None:
            d["logical_errors"] = sum(r["logical_error"][str(a)] for r in recs)
        if D["obs_cols"] is not None:
            d["observable_agreement"] = sum(r["observable_agrees"][str(a)] for r in recs)
        res["per_configuration"][str(a)] = d
    n10 = sum(1 for x, y in zip(hit[1.0], hit[0.625]) if x and not y)
    n01 = sum(1 for x, y in zip(hit[1.0], hit[0.625]) if y and not x)
    res["paired_optimum_hit"] = {"hit_at_1.0_only": n10, "hit_at_0.625_only": n01,
                                 "exact_two_sided_mcnemar_p": mcnemar(n10, n01)}
    if D["logical"] is not None:
        e1 = [r["logical_error"]["1.0"] for r in recs]
        e2 = [r["logical_error"]["0.625"] for r in recs]
        m10 = sum(1 for x, y in zip(e1, e2) if x and not y)
        m01 = sum(1 for x, y in zip(e1, e2) if y and not x)
        res["paired_logical_error_secondary"] = {"error_at_1.0_only": m10, "error_at_0.625_only": m01,
                                                 "exact_two_sided_mcnemar_p": mcnemar(m10, m01)}
        res["receipt_answer_logical_errors"] = sum(r["logical_error"].get("receipt", 0) for r in recs)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="one dataset name, or all")
    ap.add_argument("--smoke", type=int, default=None, help="dry run: this many nontrivial shots on SMOKE seeds")
    ap.add_argument("--out-dir", default=None, help="smoke runs only: where to write")
    a = ap.parse_args()
    import ldpc
    names = list(DATASETS) if a.dataset == "all" else [a.dataset]
    if any(x not in DATASETS for x in names):
        raise SystemExit(f"unknown dataset in {names}")
    if a.smoke is not None:
        domain = SMOKE_DOMAIN
        if not a.out_dir:
            raise SystemExit("a smoke run needs --out-dir")
        out_dir = pathlib.Path(a.out_dir).resolve()
        if out_dir.is_relative_to(ROOT / "evidence"):
            raise SystemExit("a smoke run never writes under evidence/")
    else:
        domain = SEED_DOMAIN
        text = PREREG.read_text(encoding="utf-8")
        missing = [s for s in SEED_NAMES if str(seed_for(domain, s)) not in text]
        if missing:
            raise SystemExit(f"the preregistration does not state the seeds this runner derives: {missing}")
        out_dir = OUT_DIR
        if any((out_dir / f"{x}.jsonl.gz").exists() for x in names):
            raise SystemExit("output exists: the preregistered run is not re-run over its own records")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _load("ref_checker_prereg", ROOT / "tools" / "external" / "exact_lp_certificate_reference.py")
    tree_b = ROOT / "tools" / "external" / "tree_checker_b.py"
    summ_path = out_dir / "summary.json"
    summary = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {
        "schema": "oneq-qldpc-scaling-prereg/1", "preregistration": "docs/PREREG_qldpc_scaling_2026-11-03.md",
        "seed_domain": domain, "smoke": a.smoke is not None, "configurations": {str(x): dict(
            COMMON, ms_scaling_factor=x) for x in ALPHAS}, "datasets": {}}
    me = pathlib.Path(__file__).resolve()
    for name in names:
        target = a.smoke if a.smoke is not None else DATASETS[name][5]
        s, recs = run_dataset(name, domain, target, ref, tree_b)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
            for r in recs:
                g.write((json.dumps(r, sort_keys=True) + "\n").encode())
        (out_dir / f"{name}.jsonl.gz").write_bytes(buf.getvalue())
        s["records_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
        here = {me.relative_to(ROOT).as_posix(): sha(me), **{
            rel: sha(ROOT / rel) for rel in ("src/oneq/qldpc_check.py", "src/oneq/qldpc_cert.py",
                                             "tools/external/exact_lp_certificate_reference.py",
                                             "tools/external/tree_checker_b.py")}}
        # every dataset records the bytes it ran under; the summary-level code_sha256 stays what the FIRST dataset
        # ran under (the committed runner), so a dataset run after a deviation cannot relabel the ones before it
        s["code_sha256"] = here
        if a.smoke is None and here != summary.get("code_sha256", here):
            s["deviation"] = ("run after DEVIATION 1 of docs/PREREG_qldpc_scaling_2026-11-03_DEVIATIONS.md: the "
                              "phenomenological datasets use the lazy-separation LP")
        summary["datasets"][name] = s
        summary["environment"] = {"python": platform.python_version(), "ldpc": ldpc.__version__,
                                  "numpy": np.__version__}
        summary.setdefault("code_sha256", here)
        summ_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
        pc = s["per_configuration"]
        print(f"{name}: {s['nontrivial']} nontrivial, {s['receipts']} receipts, {s['unresolved']} unresolved; "
              f"optimum hits 1.0 {pc['1.0']['optimum_hit']} / 0.625 {pc['0.625']['optimum_hit']}; "
              f"paired {s['paired_optimum_hit']} ({s['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
