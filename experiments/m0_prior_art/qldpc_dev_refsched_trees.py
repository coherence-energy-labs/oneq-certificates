r"""Ship the four development-circuit branch-dual trees that were never stored.

External audit 2026-09-14 (R-04 / M8): paper 2 said "every one of the 23 shots
the flat relaxation left open was closed by a tree" and "the 36 stored trees",
but only 19 development trees were stored; the 4 on the reference-schedule
circuit were "re-derivable" -- counted in evidence/qldpc_bnb_residue.json and
never shipped, so no reader could check them.

This regenerates them, bound to the development record rather than to
today's solver: the syndrome is re-sampled from the development seed exactly
as qldpc_bnb_residue_gate does; the CANDIDATE is the answer the development
per-shot record (evidence/per_shot/bb_refsched.jsonl) already stores for that
shot; the instance and weights are the persisted artifacts
(evidence/instances/bb144_refsched_r3_p002.json,
evidence/weights/bb_refsched_weights.json). A tree is produced for the
candidate's own weight and kept only if BOTH tree checkers accept it.

Produced 2026-09-14, after the fact, for the development data; the held-out
campaign is not touched.

    python experiments/m0_prior_art/qldpc_dev_refsched_trees.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

from oneq.qldpc_cert import certified_bnb  # noqa: E402
from oneq.qldpc_check import check_qldpc_bnb  # noqa: E402

OUT = ROOT / "evidence" / "qldpc_dev_refsched_trees.json"
SEED, SHOTS = 20260804, 150                     # qldpc_bnb_residue_gate
EXPECTED_SHOTS = (30, 31, 97, 143)              # qldpc_bnb_residue.json


def _ser(node):
    if node[0] == "branch":
        return ["branch", int(node[1]), _ser(node[2]), _ser(node[3])]
    if node[0] == "bound":
        return ["bound", [[list(j) if isinstance(j, tuple) else int(j), list(F),
                           y.numerator, y.denominator] for (j, F, y) in node[1]]]
    return ["infeasible", list(node[1]) if isinstance(node[1], tuple) else int(node[1])]


def main() -> int:
    from qldpc_bb_refsched_gate import build_refsched_circuit

    inst = json.loads((ROOT / "evidence" / "instances" / "bb144_refsched_r3_p002.json")
                      .read_text(encoding="utf-8"))
    wdoc = json.loads((ROOT / "evidence" / "weights" / "bb_refsched_weights.json")
                      .read_text(encoding="utf-8"))
    wvec = [int(x) for x in wdoc["vector"]]
    if hashlib.sha256(json.dumps(wvec).encode()).hexdigest() != wdoc["spec"]["weights_sha256"]:
        raise SystemExit("weight vector does not match its recorded hash")
    checks = [tuple(c) for c in inst["checks"]]
    wf = {i: float(v) for i, v in enumerate(wvec)}
    wx = {i: Fraction(v) for i, v in enumerate(wvec)}
    recs = {json.loads(x)["shot_index"]: json.loads(x)
            for x in (ROOT / "evidence" / "per_shot" / "bb_refsched.jsonl")
            .read_text(encoding="utf-8").splitlines()}

    dem = build_refsched_circuit(3, 0.002).detector_error_model(decompose_errors=False)
    dets, _obs, _ = dem.compile_sampler(seed=SEED).sample(shots=SHOTS, return_errors=False)
    nontrivial = [dets[s].astype(np.uint8) for s in range(dets.shape[0]) if dets[s].any()]

    trees, fails = [], []
    for k in EXPECTED_SHOTS:
        syn = nontrivial[k]
        rec = recs[k]
        cand = tuple(int(i) for i in (rec.get("fallback") or {}).get("answer") or ())
        if not cand:
            fails.append(f"shot {k}: development record has no fallback answer")
            continue
        U = sum(wvec[i] for i in cand)
        tree, nodes = certified_bnb(checks, wf, list(syn), U, rpc_rounds=6,
                                    max_nodes=60000, max_depth=80, use_lazy=True)
        if tree is None:
            fails.append(f"shot {k}: no tree within budget")
            continue
        ra = check_qldpc_bnb(checks=checks, weights=wx, syndrome=[int(b) for b in syn],
                             error_support=cand, tree=tree)
        doc = {"checks": [list(c) for c in checks], "weights": wvec,
               "syndrome": [int(b) for b in syn], "candidate": list(cand), "tree": _ser(tree)}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(doc, tf)
        rb = subprocess.run([sys.executable, str(ROOT / "tools" / "external" / "tree_checker_b.py"),
                             tf.name], capture_output=True, text=True, timeout=600)
        pathlib.Path(tf.name).unlink(missing_ok=True)
        ok = bool(ra.accepted) and rb.returncode == 0
        print(f"  dev bb_refsched shot {k}: U={U} nodes={nodes} production={ra.accepted} "
              f"B={rb.returncode == 0}")
        if not ok:
            fails.append(f"shot {k}: production={ra.accepted} B={rb.returncode == 0}")
            continue
        trees.append({"rung": "bb_refsched", "shot_index": k, "answer_weight": U,
                      "bnb_nodes": nodes,
                      "syndrome_support": [int(i) for i in np.flatnonzero(syn)],
                      "candidate": list(cand), "candidate_source": "development per-shot record",
                      "tree": _ser(tree)})
    OUT.write_text(json.dumps({
        "schema": "oneq-qldpc-dev-refsched-trees/1",
        "produced": "2026-09-14, after the fact, for the development reference-schedule records",
        "instance": "evidence/instances/bb144_refsched_r3_p002.json",
        "weights": "evidence/weights/bb_refsched_weights.json",
        "trees": trees, "failures": fails}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(trees)}/{len(EXPECTED_SHOTS)} trees stored, both checkers accepting -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
